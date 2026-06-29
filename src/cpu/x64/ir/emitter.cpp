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

#include "cpu/x64/ir/emitter.hpp"
#include "cpu/x64/utils/jit_regops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

void emit(jit_generator_t &host, const ir_t &ir,
        const reg_alloc_result_t &alloc, const reg_config_t &reg_cfg,
        cpu_isa_t isa, data_section_t &data,
        const inject_postops_fn_t &inject) {
    if (is_superset(isa, avx512_core)) {
        // TODO: emit_avx512(host, ir, alloc, reg_cfg, isa, data, inject);
        assert(!"avx512 IR emitter is not implemented yet");
    } else {
        emit_avx2(host, ir, alloc, reg_cfg, isa, data, inject);
    }
}

// TODO: consider moving ISA-specific emitters to separate files.
void emit_avx2(jit_generator_t &host, const ir_t &ir,
        const reg_alloc_result_t &alloc, const reg_config_t &reg_cfg,
        cpu_isa_t isa, data_section_t &data,
        const inject_postops_fn_t &inject) {
    // `isa` is forwarded for future extension-specific instruction selection
    // (e.g. avx2_vnni_2). The current f32 path does not need it yet.
    MAYBE_UNUSED(isa);

    // One label per loop, keyed by the loop_begin instruction index (loop_end
    // jumps back to labels[in.match]). A map, so only loops get an entry.
    std::unordered_map<int, Xbyak::Label> labels;

    // Label IDs for all labels. (TODO: better name?)
    std::vector<Xbyak::Label> label_ids(ir.n_labels);

    // f32 is the only data type supported today, so a vector holds 8 elements.
    // When a new data type lands this becomes vlen / data_type_size(dt),
    // taken from each vec vreg's data type.
    // TODO: do we need it at all except for a couple of f32 specific spots?
    const int simd_w = 8;

    // Data type of a vec vreg.
    auto dt_of = [&](int v) { return ir.vreg_info[v].dt; };
    auto is_f32 = [&](int v) { return dt_of(v) == data_type::f32; };

    // Reserve scratch registers for the spills.
    const Xbyak::Reg64 gpr_scratch0(reg_cfg.gpr_scratch[0]);
    const Xbyak::Reg64 gpr_scratch1(reg_cfg.gpr_scratch[1]);
    const Xbyak::Ymm vec_scratch0(reg_cfg.vec_scratch[0]);
    const Xbyak::Ymm vec_scratch1(reg_cfg.vec_scratch[1]);
    const Xbyak::Ymm vec_scratch2(reg_cfg.vec_scratch[2]);

    // if vreg needs to be spilled
    auto spilled = [&](int v) { return alloc.assignments[v].spilled; };
    // get a physical register from a virtual one
    auto phys = [&](int v) { return alloc.assignments[v].phys; };
    // get a stack slot for a virtual register
    auto slot = [&](int v) {
        return host.ptr[host.rsp + (int)alloc.assignments[v].slot];
    };

    // Resolve a virtual register that an instruction READS ("use") to a
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
    // a spilled result back is done by the defining op (compute into scratch,
    // then store to the slot).
    //
    // TODO: introduce loop-depth spill weights to optimize spills. Currently,
    // the spilling strategy is naive and is only good for low pressure kernels
    // (e.g. GEMV).
    auto gpr_use = [&](int v, const Xbyak::Reg64 &scr) -> Xbyak::Reg64 {
        if (!spilled(v)) return Xbyak::Reg64(phys(v));
        // reload the spilled gpr from its stack slot
        host.mov(scr, slot(v));
        return scr;
    };

    auto vec_use = [&](int v, const Xbyak::Ymm &scr) -> Xbyak::Ymm {
        if (!spilled(v)) return Xbyak::Ymm(phys(v));
        // reload the spilled vector register from its stack slot
        host.vmovups(scr, slot(v));
        return scr;
    };

    // Lower each IR instruction to avx2. Spilled operands are handled as
    // follows:
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
    for (int i = 0; i < ir.n_instrs(); i++) {
        const instr_t &in = ir.instrs[i];
        switch (in.op) {
            case op_kind_t::mov_imm: {
                if (!spilled(in.dst)) {
                    host.mov(Xbyak::Reg64(phys(in.dst)), in.imm);
                } else {
                    host.mov(gpr_scratch0, in.imm);
                    host.mov(slot(in.dst), gpr_scratch0);
                }
                break;
            }
            case op_kind_t::mov_reg: {
                Xbyak::Reg64 s = gpr_use(in.s0, gpr_scratch1);
                if (!spilled(in.dst))
                    host.mov(Xbyak::Reg64(phys(in.dst)), s);
                else
                    host.mov(slot(in.dst), s);
                break;
            }
            case op_kind_t::add_imm: {
                if (!spilled(in.dst)) {
                    host.add(Xbyak::Reg64(phys(in.dst)), in.imm);
                } else {
                    host.mov(gpr_scratch0, slot(in.dst));
                    host.add(gpr_scratch0, in.imm);
                    host.mov(slot(in.dst), gpr_scratch0);
                }
                break;
            }
            case op_kind_t::add_reg: {
                Xbyak::Reg64 s = gpr_use(in.s0, gpr_scratch1);
                if (!spilled(in.dst)) {
                    host.add(Xbyak::Reg64(phys(in.dst)), s);
                } else {
                    host.mov(gpr_scratch0, slot(in.dst));
                    host.add(gpr_scratch0, s);
                    host.mov(slot(in.dst), gpr_scratch0);
                }
                break;
            }
            case op_kind_t::load: {
                Xbyak::Reg64 base = in.mem.is_param
                        ? Xbyak::Reg64(reg_cfg.param_reg)
                        : gpr_use(in.mem.base, gpr_scratch1);
                Xbyak::Reg64 d = spilled(in.dst) ? gpr_scratch0
                                                 : Xbyak::Reg64(phys(in.dst));
                host.mov(d, host.ptr[base + (int)in.mem.disp]);
                if (spilled(in.dst)) host.mov(slot(in.dst), d);
                break;
            }
            case op_kind_t::vzero: {
                Xbyak::Ymm d = spilled(in.dst) ? vec_scratch0
                                               : Xbyak::Ymm(phys(in.dst));
                host.vxorps(d, d, d);
                if (spilled(in.dst)) host.vmovups(slot(in.dst), d);
                break;
            }
            case op_kind_t::vload: {
                Xbyak::Reg64 base = gpr_use(in.mem.base, gpr_scratch0);
                Xbyak::Ymm d = spilled(in.dst) ? vec_scratch0
                                               : Xbyak::Ymm(phys(in.dst));
                host.vmovups(d, host.ptr[base + (int)in.mem.disp]);
                if (spilled(in.dst)) host.vmovups(slot(in.dst), d);
                break;
            }
            case op_kind_t::vfma: {
                Xbyak::Ymm d = spilled(in.dst) ? vec_scratch0
                                               : Xbyak::Ymm(phys(in.dst));
                if (spilled(in.dst)) host.vmovups(d, slot(in.dst));
                Xbyak::Ymm a = vec_use(in.s0, vec_scratch1);
                Xbyak::Ymm b = vec_use(in.s1, vec_scratch2);
                if (is_f32(in.dst) && is_f32(in.s0) && is_f32(in.s1))
                    host.vfmadd231ps(d, a, b);
                else
                    assert(!"vfma: dtype not implemented");
                if (spilled(in.dst)) host.vmovups(slot(in.dst), d);
                break;
            }
            case op_kind_t::vadd: {
                Xbyak::Ymm d = spilled(in.dst) ? vec_scratch0
                                               : Xbyak::Ymm(phys(in.dst));
                if (spilled(in.dst)) host.vmovups(d, slot(in.dst));
                Xbyak::Ymm s = vec_use(in.s0, vec_scratch1);
                if (is_f32(in.dst) && is_f32(in.s0))
                    host.vaddps(d, d, s);
                else
                    assert(!"vadd: dtype not implemented");
                if (spilled(in.dst)) host.vmovups(slot(in.dst), d);
                break;
            }
            case op_kind_t::vhreduce: {
                Xbyak::Ymm d = spilled(in.dst) ? vec_scratch0
                                               : Xbyak::Ymm(phys(in.dst));
                if (spilled(in.dst)) host.vmovups(d, slot(in.dst));
                Xbyak::Ymm w = vec_use(in.s0, vec_scratch1);
                if (is_f32(in.dst))
                    regops::horizontal_add_ps(&host, d, w);
                else
                    assert(!"vhreduce: dtype not implemented");
                if (spilled(in.dst)) host.vmovups(slot(in.dst), d);
                break;
            }
            case op_kind_t::inject_postops: {
                // The `inject_postops` instruction lowers to a JIT-based
                // external injector. The physical argument registers must
                // be live in registers, not spilled to the stack. We then pass
                // them to the callback received from the builder, which emits
                // the injector code.
                const auto &args = ir.inject_postops_args[(int)in.imm];
                std::vector<int> acc_phys;
                acc_phys.reserve(args.acc.size());
                for (int v : args.acc) {
                    assert(!spilled(v)
                            && "inject_postops: accumulator spilled");
                    acc_phys.push_back(phys(v));
                }
                assert(!spilled(args.base_ptr)
                        && "inject_postops: base pointer spilled");
                assert(inject && "inject_postops: missing injector callback");
                inject(acc_phys, phys(args.base_ptr));
                break;
            }
            case op_kind_t::set_mask_imm: {
                // Create a vector mask where the first `imm` elements are
                // active (all bits set), and the rest are zero.
                //
                // The data for the  mask is stored in the constant data section
                // and loaded once. The builder uses `set_mask_imm` only once so
                // the mask can be reused across all relevant masked operations.

                // Used only for f32 data type.
                constexpr int elem_bytes = sizeof(float);
                constexpr unsigned char active_byte = 0xff;

                std::vector<unsigned char> bytes(simd_w * elem_bytes, 0);
                const int active_bytes = (int)in.imm * elem_bytes;

                for (int b = 0; b < active_bytes; b++)
                    bytes[b] = active_byte;

                // This label represents the memory location of the mask in the
                // data section.
                data.constants.emplace_back(std::move(bytes), Xbyak::Label());
                Xbyak::Label &lbl = data.constants.back().second;

                Xbyak::Ymm d = spilled(in.dst) ? vec_scratch0
                                               : Xbyak::Ymm(phys(in.dst));
                // Load the mask data.
                host.vmovups(d, host.ptr[host.rip + lbl]);

                // TODO: rematerialize instead of spilling. This mask is a
                // constant in the data section, so we can reload it directly
                // using `[rip + lbl]` whenever needed.
                // Right now, the generic spill path copies it to the stack
                // first, then reloads it from there. This adds an extra store
                // and creates an unnecessary duplicate of the same constant.
                //
                // If we treat the data-section address as the mask's read-only
                // `spill slot` then we can avoid the extra spill.
                if (spilled(in.dst)) host.vmovups(slot(in.dst), d);
                break;
            }
            case op_kind_t::vload_masked: {
                // The `in.imm` specifies the number of active elements.
                //   - 1 active element: use `vmovss` (no mask register needed)
                //   - Full vector: use `vmovups`, (no mask register needed)
                //   - Partial vector: use `vmaskmovps` with the mask stored in
                //     s1
                Xbyak::Reg64 base = gpr_use(in.mem.base, gpr_scratch0);
                const auto addr = host.ptr[base + (int)in.mem.disp];
                Xbyak::Ymm d = spilled(in.dst) ? vec_scratch0
                                               : Xbyak::Ymm(phys(in.dst));
                if (is_f32(in.dst)) {
                    if (in.imm == 1)
                        host.vmovss(Xbyak::Xmm(d.getIdx()), addr);
                    else if (in.imm >= simd_w)
                        host.vmovups(d, addr);
                    else {
                        Xbyak::Ymm m = vec_use(in.s1, vec_scratch1);
                        host.vmaskmovps(d, m, addr);
                    }
                } else {
                    // `vmaskmov` is applied only to f32 other precisions need a
                    // different mechanism here.
                    assert(!"vload_masked: dtype not implemented");
                }
                if (spilled(in.dst)) host.vmovups(slot(in.dst), d);
                break;
            }
            case op_kind_t::vstore_masked: {
                // The `in.imm` specifies the number of active elements.
                // Choose the simplest instruction for each case:
                //   - 1 active element: use `vmovss` (no mask register needed)
                //   - Full vector: use `vmovups` (no mask register needed)
                //   - Partial vector: use `vmaskmovps` with the mask stored in
                //     s1
                Xbyak::Reg64 base = gpr_use(in.mem.base, gpr_scratch0);
                Xbyak::Ymm s = vec_use(in.s0, vec_scratch0);
                const auto addr = host.ptr[base + (int)in.mem.disp];

                if (is_f32(in.s0)) {
                    if (in.imm == 1)
                        host.vmovss(addr, Xbyak::Xmm(s.getIdx()));
                    else if (in.imm >= simd_w)
                        host.vmovups(addr, s);
                    else {
                        Xbyak::Ymm m = vec_use(in.s1, vec_scratch1);
                        host.vmaskmovps(addr, m, s);
                    }
                } else
                    assert(!"vstore_masked: dtype not implemented");
                break;
            }
            case op_kind_t::loop_begin: {
                Xbyak::Reg64 c = spilled(in.dst) ? gpr_scratch0
                                                 : Xbyak::Reg64(phys(in.dst));
                if (in.init_is_reg) {
                    Xbyak::Reg64 iv = gpr_use(in.s0, gpr_scratch1);
                    host.mov(c, iv);
                } else {
                    host.mov(c, in.imm);
                }
                if (spilled(in.dst)) host.mov(slot(in.dst), c);
                host.L(labels[i]); // body start
                break;
            }
            case op_kind_t::loop_end: {
                if (!spilled(in.dst)) {
                    Xbyak::Reg64 c(phys(in.dst));
                    host.dec(c);
                    host.cmp(c, 0);
                } else {
                    host.mov(gpr_scratch0, slot(in.dst));
                    host.dec(gpr_scratch0);
                    host.mov(slot(in.dst), gpr_scratch0);
                    host.cmp(gpr_scratch0, 0);
                }
                host.jg(labels[in.match]);
                break;
            }
            case op_kind_t::label: {
                host.L(label_ids[(int)in.imm]);
                break;
            }
            case op_kind_t::jmp: {
                host.jmp(label_ids[(int)in.imm], Xbyak::CodeGenerator::T_NEAR);
                break;
            }
            case op_kind_t::jz: {
                Xbyak::Reg64 c = gpr_use(in.s0, gpr_scratch0);
                host.cmp(c, 0);
                host.jz(label_ids[(int)in.imm], Xbyak::CodeGenerator::T_NEAR);
                break;
            }
        }
    }
}

void emit_data_section(jit_generator_t &host, data_section_t &data) {
    for (auto &c : data.constants) {
        if (data.align) host.align(data.align);
        host.L(c.second);
        for (unsigned char byte : c.first)
            host.db(byte);
    }
}

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
