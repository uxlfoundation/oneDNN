/*******************************************************************************
* Copyright 2026 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <cassert>
#include <vector>
#include <unordered_map>

#include "cpu/x64/ir/emitter/backend_avx2.hpp"
#include "cpu/x64/ir/emitter/emitter.hpp"
#include "cpu/x64/utils/jit_regops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

template <typename backend_t>
void emit(backend_t &be, const ir_t &ir, const reg_alloc_result_t &alloc,
        const reg_config_t &rc, data_section_t &data) {

    // The backend holds the generator.
    jit_generator_t &gen = be.gen();

    // One label per loop, keyed by the `loop_begin` instruction index. A
    // `loop_end` jumps back to `labels[op.match]`. A map, so only loops get an
    // entry.
    std::unordered_map<int, Xbyak::Label> labels;
    // One label per `label` id, for `label`/`jmp`/`jz`.
    std::vector<Xbyak::Label> label_id_to_label(ir.n_labels());

    // if vreg needs to be spilled
    auto spilled = [&](int vr) { return alloc.assignments[vr].spilled; };
    // get a physical register from a virtual one
    auto phys = [&](int vr) { return alloc.assignments[vr].phys; };
    // get a stack slot for a virtual register
    auto slot = [&](int vr) {
        return gen.ptr[gen.rsp + (int)alloc.assignments[vr].slot];
    };
    // stack slot byte offset for a vec spill
    auto slot_off = [&](int vr) { return (int)alloc.assignments[vr].slot; };
    // Data type of a vec vreg.
    auto dt_of = [&](int vr) { return ir.vreg_info()[vr].dt; };

    // Reserve scratch registers for the spills.
    const Xbyak::Reg64 gpr_scratch0(rc.gpr_scratch[0]);
    const Xbyak::Reg64 gpr_scratch1(rc.gpr_scratch[1]);
    const int vec_scratch0 = rc.vec_scratch[0];
    const int vec_scratch1 = rc.vec_scratch[1];
    const int vec_scratch2 = rc.vec_scratch[2];

    // Move a spilled vec value between its stack slot and a scratch register,
    // as a vector load/store against the stack frame (rsp).
    const int rsp_idx = gen.rsp.getIdx();
    auto spill_reload
            = [&](int vr, int p) { be.vload(p, rsp_idx, slot_off(vr)); };
    auto spill_store
            = [&](int vr, int p) { be.vstore(rsp_idx, slot_off(vr), p); };

    // Resolve a virtual register that an instruction READS (use) to a
    // concrete physical register, hiding whether the allocator spilled it:
    //   - not spilled: the value is already in a physical register, so just
    //     return that register (no extra instruction).
    //   - spilled: the value lives on the stack slot, so emit a reload into the
    //     caller-provided scratch register `scr` and return it.
    // The caller picks `scr` (gpr_scratch0/1 for gpr, vec_scratch0/1/2 for
    // vec) so that an instruction with several spilled operands reloads each
    // into a different scratch and they do not clobber one another. The
    // returned register is valid only until the next reload into the same
    // scratch, so use it right away. These helpers handle only reads. Writing
    // a spilled result back is done by the defining instruction (compute into
    // scratch, then store to the slot).
    //
    // TODO: introduce loop-depth spill weights to optimize spills. Currently,
    // the spilling strategy is naive and is only good for low pressure kernels
    // (e.g. GEMV).
    //
    // gpr reloads are ISA-neutral (a plain `mov`), so `gpr_use` emits them
    // directly. A spilled vec source is reloaded through the backend, since the
    // reload instruction is ISA-specific. The `vec_use` returns a physical
    // index rather than a typed register.
    auto gpr_use = [&](int vr, const Xbyak::Reg64 &scr) -> Xbyak::Reg64 {
        if (!spilled(vr)) return Xbyak::Reg64(phys(vr));
        // reload the spilled gpr from its stack slot
        gen.mov(scr, slot(vr));
        return scr;
    };

    auto vec_use = [&](int vr, int scr_idx) -> int {
        if (!spilled(vr)) return phys(vr);
        // reload the spilled vector register from its stack slot
        spill_reload(vr, scr_idx);
        return scr_idx;
    };

    // Lower each IR instruction. Spilled operands are handled as follows:
    //
    // - Inputs that an instruction reads are accessed through gpr_use/vec_use.
    //   These return the register directly, or reload the value from its spill
    //   slot into a scratch register if needed.
    //
    // - The output that an instruction writes is handled separately inside each
    //   case. If the destination is spilled, we reload it first (for
    //   read-modify-write operations), perform the operation, and then store
    //   the result back to its spill slot.
    //
    // Scratch register usage:
    // - gpr_scratch0/vec_scratch0: scratch register for destinations
    // - gpr_scratch1/vec_scratch1/vec_scratch2: scratch registers for sources
    //
    // This separation ensures that spilled source and destination values never
    // use the same scratch register.
    for (int i = 0; i < ir.n_ops(); i++) {
        const op_t &op = ir.ops()[i];
        switch (op.kind) {
            // General-purpose register ops. ISA-neutral, emitted directly.
            case op_kind_t::mov_imm: {
                if (!spilled(op.dst)) {
                    gen.mov(Xbyak::Reg64(phys(op.dst)), op.imm);
                } else {
                    gen.mov(gpr_scratch0, op.imm);
                    gen.mov(slot(op.dst), gpr_scratch0);
                }
                break;
            }
            case op_kind_t::mov_reg: {
                Xbyak::Reg64 s = gpr_use(op.s0, gpr_scratch1);
                if (!spilled(op.dst))
                    gen.mov(Xbyak::Reg64(phys(op.dst)), s);
                else
                    gen.mov(slot(op.dst), s);
                break;
            }
            case op_kind_t::add_imm: {
                if (!spilled(op.dst)) {
                    gen.add(Xbyak::Reg64(phys(op.dst)), op.imm);
                } else {
                    gen.mov(gpr_scratch0, slot(op.dst));
                    gen.add(gpr_scratch0, op.imm);
                    gen.mov(slot(op.dst), gpr_scratch0);
                }
                break;
            }
            case op_kind_t::add_reg: {
                Xbyak::Reg64 s = gpr_use(op.s0, gpr_scratch1);
                if (!spilled(op.dst)) {
                    gen.add(Xbyak::Reg64(phys(op.dst)), s);
                } else {
                    gen.mov(gpr_scratch0, slot(op.dst));
                    gen.add(gpr_scratch0, s);
                    gen.mov(slot(op.dst), gpr_scratch0);
                }
                break;
            }
            case op_kind_t::load: {
                Xbyak::Reg64 base = op.mem.is_param
                        ? Xbyak::Reg64(rc.param_reg)
                        : gpr_use(op.mem.base, gpr_scratch1);
                Xbyak::Reg64 d = spilled(op.dst) ? gpr_scratch0
                                                 : Xbyak::Reg64(phys(op.dst));
                gen.mov(d, gen.ptr[base + (int)op.mem.disp]);
                if (spilled(op.dst)) gen.mov(slot(op.dst), d);
                break;
            }

            // Vector ops. Emitting the instruction is the backend's job. A
            // spilled dst is always stored back to its slot after the op.
            // A read-modify-write (`rmw`) op also reloads a spilled dst before
            // the op. An op that overwrites dst does not.
            case op_kind_t::vzero: { // overwrites dst
                int d = spilled(op.dst) ? vec_scratch0 : phys(op.dst);
                be.vzero(d);
                if (spilled(op.dst)) spill_store(op.dst, d);
                break;
            }
            case op_kind_t::vload: { // overwrites dst
                int base = gpr_use(op.mem.base, gpr_scratch0).getIdx();
                int d = spilled(op.dst) ? vec_scratch0 : phys(op.dst);
                be.vload(d, base, op.mem.disp);
                if (spilled(op.dst)) spill_store(op.dst, d);
                break;
            }
            case op_kind_t::vfma: { // rmw: reads and writes dst
                int d = spilled(op.dst) ? vec_scratch0 : phys(op.dst);
                if (spilled(op.dst)) spill_reload(op.dst, d);
                int a = vec_use(op.s0, vec_scratch1);
                int b = vec_use(op.s1, vec_scratch2);
                be.vfma(d, a, b, dt_of(op.s0));
                if (spilled(op.dst)) spill_store(op.dst, d);
                break;
            }
            case op_kind_t::vhreduce: { // reads and writes dst
                int d = spilled(op.dst) ? vec_scratch0 : phys(op.dst);
                if (spilled(op.dst)) spill_reload(op.dst, d);
                int ws = vec_use(op.s0, vec_scratch1);
                be.vhreduce(d, ws, dt_of(op.dst));
                if (spilled(op.dst)) spill_store(op.dst, d);
                break;
            }

            // Mask ops. Emitting the instruction is the backend's job.
            // Spilling is not supported for mask vreg for now so we assert
            // `no spills`.
            case op_kind_t::set_mask_imm: {
                assert(!spilled(op.dst) && "set_mask_imm: mask spilled");
                be.set_mask_imm(phys(op.dst), (int)op.imm, data);
                break;
            }
            case op_kind_t::vload_masked: { // overwrites dst
                int base = gpr_use(op.mem.base, gpr_scratch0).getIdx();
                int d = spilled(op.dst) ? vec_scratch0 : phys(op.dst);
                assert((op.s1 < 0 || !spilled(op.s1))
                        && "vload_masked: mask spilled");
                int mask = (op.s1 >= 0) ? phys(op.s1) : -1;
                be.vload_masked(d, base, op.mem.disp, mask, (int)op.imm,
                        dt_of(op.dst), data);
                if (spilled(op.dst)) spill_store(op.dst, d);
                break;
            }
            case op_kind_t::vstore_masked: {
                int base = gpr_use(op.mem.base, gpr_scratch0).getIdx();
                int s = vec_use(op.s0, vec_scratch0);
                assert((op.s1 < 0 || !spilled(op.s1))
                        && "vstore_masked: mask spilled");
                int mask = (op.s1 >= 0) ? phys(op.s1) : -1;
                be.vstore_masked(base, op.mem.disp, s, mask, (int)op.imm,
                        dt_of(op.s0), data);
                break;
            }

            // Control flow. ISA-neutral, emitted directly.
            case op_kind_t::loop_begin: {
                Xbyak::Reg64 c = spilled(op.dst) ? gpr_scratch0
                                                 : Xbyak::Reg64(phys(op.dst));
                if (op.init_is_reg) {
                    Xbyak::Reg64 iv = gpr_use(op.s0, gpr_scratch1);
                    gen.mov(c, iv);
                } else {
                    gen.mov(c, op.imm);
                }
                if (spilled(op.dst)) gen.mov(slot(op.dst), c);
                gen.L(labels[i]); // body start
                break;
            }
            case op_kind_t::loop_end: {
                // dec sets ZF, so the back-edge is a plain jnz with no cmp. The
                // counter starts >= 1 and lands on exactly 0, so jnz matches jg.
                if (!spilled(op.dst)) {
                    Xbyak::Reg64 c(phys(op.dst));
                    gen.dec(c);
                } else {
                    gen.mov(gpr_scratch0, slot(op.dst));
                    gen.dec(gpr_scratch0);
                    gen.mov(slot(op.dst), gpr_scratch0);
                }
                gen.jnz(labels[op.match]); // back-edge to the matching loop_begin
                break;
            }
            case op_kind_t::label: {
                gen.L(label_id_to_label[(int)op.label_id]);
                break;
            }
            case op_kind_t::jmp: {
                gen.jmp(label_id_to_label[(int)op.label_id],
                        Xbyak::CodeGenerator::T_NEAR);
                break;
            }
            case op_kind_t::jz: {
                Xbyak::Reg64 c = gpr_use(op.s0, gpr_scratch0);
                gen.cmp(c, 0);
                gen.jz(label_id_to_label[(int)op.label_id],
                        Xbyak::CodeGenerator::T_NEAR);
                break;
            }
        }
    }
}

void emit(jit_generator_t &gen, const ir_t &ir, const reg_alloc_result_t &alloc,
        const reg_config_t &reg_cfg, data_section_t &data) {
    const cpu_isa_t isa = gen.max_cpu_isa();
    if (is_superset(isa, avx512_core)) {
        assert(!"avx512 emitter is not supported");
    } else {
        avx2_backend_t be(gen, isa);
        emit(be, ir, alloc, reg_cfg, data);
    }
}

void emit_data_section(jit_generator_t &gen, data_section_t &data) {
    for (auto &c : data.constants) {
        gen.align(data_section_t::alignment);
        gen.L(c.second);
        for (unsigned char byte : c.first)
            gen.db(byte);
    }
}

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
