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

#ifndef CPU_X64_IR_IR_HPP
#define CPU_X64_IR_IR_HPP

// Minimal linear IR for JIT kernels.
//
// An `ir_t` is a flat list of instructions. Each instruction reads and writes
// virtual registers: integer-named placeholders for the values a kernel works
// with, used instead of physical registers until the allocator assigns them.
// Every virtual register belongs to one of three kinds (gpr/vec/mask), which
// defines the kind of physical register it can later land in. Operations are
// target-neutral. The concrete instructions are chosen later by the emitter.
//
// A vec vreg also carries the element data type of the values it holds. The
// operations stay data-type generic. The builder tags each vec vreg with a
// dtype and the emitter lowers each op to the instruction for that dtype.
//
// The virtual registers are mutable. A virtual register is a named value that
// may be written more than once. In this IR we deliberately allow reassignment,
// e.g. a pointer register is loaded and then advanced with `add` each loop
// iteration, reusing the same name. Each virtual register occupies one physical
// location (register or spill slot) for the whole of its live range, so a write
// overwrites that location in place. This keeps the builder and the register
// allocator simple. The allocator relies only on liveness.
//
// Memory operands are "base register + displacement", where the displacement is
// a single constant offset known at build time and encoded directly into the
// instruction (`[reg + disp]`). There is no second index register and no
// scale. Any distance that is only known at run time is not a displacement, the
// builder instead folds it into the base pointer with an explicit `add` (a
// running pointer that is advanced each iteration instead of recomputing an
// index). Restricting addresses to one register keeps every memory op reading
// at most one pointer, which lowers register pressure and simplifies spilling.
//
// Loops are explicit structured nodes with a runtime counter register
// (loop_begin/loop_end). Compile-time loops are unrolled by the builder.

#include <vector>

#include "common/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

// gpr  - General-purpose register
// vec  - Vector register
// mask - Mask register (emitter decides whether lower it to k or vector
//        register)

enum class reg_kind_t { gpr, vec, mask };

enum class op_kind_t {
    // General-purpose register operations

    // dst = immediate value
    mov_imm,
    // dst = source register (s0)
    mov_reg,
    // dst += immediate value
    add_imm,
    // dst += source register (s0)
    add_reg,
    // dst = [base + disp], base is the parameter register if is_param
    load,

    // Vector operations
    //
    // dst = 0 (clear vector)
    vzero,
    // dst = [base + disp] (load full vector)
    vload,
    // dst += s0 * s1 (fused multiply-add, e.g., vfmadd231ps)
    vfma,
    // dst += s0 (vector add, e.g., vaddps)
    vadd,
    // horizontal reduction of dst (uses s0 as workspace), result in element 0
    vhreduce,

    // Mask operations
    //
    // set_mask_imm creates a mask for `imm` active elements.
    set_mask_imm,

    // Masked vector load/store
    //
    // loads `imm` elements. Mask vreg = s1 or -1.
    vload_masked,
    // stores `imm` elements. Mmask vreg = s1 or -1.
    vstore_masked,

    // Control flow
    //
    // initialize loop counter (dst = imm or s0 if init_is_reg)
    loop_begin,
    // decrement counter and jump to matching loop_begin if not zero
    loop_end,
    // bind label id `imm`
    label,
    // jump to label id `imm`
    jmp,
    // if s0 == 0 jump to label id `imm`
    jz,
};

// A memory address of the form "base register + displacement".
//   base     - virtual gpr holding the pointer. Runtime offsets are folded into
//              it by the builder (incrementing the pointer), so the only
//              non-register part of an address is a build-time constant.
//   disp     - that constant byte offset, encoded directly in the instruction
//              There is no index register and no scale.
//   is_param - when true the base is not `base` but the kernel's argument
//              pointer (a fixed register set by the ABI). Used to read fields
//              of the kernel-parameter struct. The register allocator does not
//              manage it, so it is not counted as a virtual-register use.
struct mem_t {
    // virtual gpr holding the pointer (ignored when is_param)
    int base = -1;
    // constant byte offset
    dim_t disp = 0;
    // base is the kernel-argument pointer
    bool is_param = false;
};

// A single IR instruction uses one fixed struct that all operations share.
// It works like a tagged union, where the op field decides what kind of
// instruction it is. Each operation only uses the fields it needs and ignores
// the rest, which stay at default values.
//
// The meaning of each field for a specific operation is explained in the
// `op_kind_t` comments. Whether dst, s0, and s1 are used for reading or writing
// is clearly defined in ir_t::def_use().
//
// op   - which operation this instruction is. Determines how other fields are
//        used.
// dst  - virtual register that is written to, or -1 if none is written.
//        Some ops (e.g. vfma/vadd) both read from and write to dst.
// s0,s1 - source virtual registers (inputs), or -1 if not used.
// imm   - immediate value whose meaning depends on the op:
//         * mov_imm        -> literal constant
//         * loop_begin     -> loop trip count
//         * set_mask_imm   -> active element count
//         * vload_masked / vstore_masked -> active element count
// mem   - memory address used only by load/store operations.
// match - for loop_end, index of matching loop_begin instruction.
// init_is_reg - for loop_begin, if true, initialize loop counter from s0
//                (runtime value) instead of imm.
struct instr_t {
    op_kind_t op;
    int dst = -1;
    int s0 = -1, s1 = -1;
    dim_t imm = 0;
    mem_t mem;
    int match = -1;
    bool init_is_reg = false;
};

// Metadata for virtual registers that contains the register kind and the
// data type of the values it holds. `dt` is `undef` for gpr and mask.
struct vreg_info_t {
    reg_kind_t kind;
    data_type_t dt = data_type::undef;
};

// An `ir_t` is the instruction list plus, for each virtual register, its info
// (kind and for a vec its element data type) so the allocator knows which
// physical registers it can use and the emitter knows which dtype-specific
// instruction to lower each op to.
//
// * The builder fills it through the helpers below
// * The allocator and emitter read it
struct ir_t {
    // Info for each vreg, indexed by its id
    std::vector<vreg_info_t> vreg_info;
    // All instructions, in order
    std::vector<instr_t> instrs;

    // Label counter.
    int n_labels = 0;

    // Add a vreg of kind `k` and, for a vec, element data type `dt`.
    // Return its id.
    int new_vreg(reg_kind_t k, data_type_t dt = data_type::undef);

    // Convenience wrappers around `new_vreg` for the specific kinds.
    int new_gpr() { return new_vreg(reg_kind_t::gpr); }
    int new_vec(data_type_t dt) { return new_vreg(reg_kind_t::vec, dt); }
    int new_mask() { return new_vreg(reg_kind_t::mask); }

    // Add a new label and return its id
    int new_label() { return n_labels++; }
    int n_vregs() const { return (int)vreg_info.size(); }
    int n_instrs() const { return (int)instrs.size(); }

    // Each helper appends one instruction and names registers by `id`. The ones
    // returning `int` return the new instruction's index.
    //
    // Refer to documentation for `op_kind_t` for each helper.

    // gpr
    int mov_imm(int dst, dim_t imm);
    void mov_reg(int dst, int src);
    void add_imm(int dst, dim_t imm);
    void add_reg(int dst, int src);
    void load_param(int dst, dim_t disp);
    void load(int dst, int base, dim_t disp);

    // vec
    void vzero(int dst);
    void vload(int dst, int base, dim_t disp);
    void vfma(int dst, int a, int b);
    void vadd(int dst, int src);
    void vhreduce(int dst, int workspace);

    // vec (masked)
    void vload_masked(int dst, int base, dim_t disp, int mask, int elems);
    void vstore_masked(int base, dim_t disp, int src, int mask, int elems);

    // mask
    void set_mask_imm(int mask, int n_elems);

    // control flow
    int loop_begin_imm(int counter, dim_t count);
    int loop_begin_reg(int counter, int init);
    void loop_end(int counter, int begin_idx);
    void label(int label_id);
    void jmp(int label_id);
    void jz(int cond, int label_id);

    // Fill defs/uses with the vregs this instruction writes/reads. Liveness and
    // the allocator depend on it, so it must match what the emitter emits.
    void def_use(const instr_t &in, std::vector<int> &defs,
            std::vector<int> &uses) const;
};

// Loop helpers shared by IR builders.
//
// If the loop count is known at build time and is 1, don't generate a loop.
// Just emit `body` once.
//
// These helpers either:
// - generate a runtime loop (creating the loop counter internally), or
// - inline `body` once when only one iteration is needed.
//
// This avoids repeating:
//   if (count > 1) { loop_begin; ...; loop_end; } else { ... }
// in every builder.

// Emit a runtime loop with `count_imm` as the iteration count.
// If `count_imm` is 0 or 1, emit `body` once without generating a loop.
template <typename body_t>
void emit_loop_imm(ir_t &ir, dim_t count_imm, body_t body) {
    if (count_imm > 1) {
        const int counter = ir.new_gpr();
        const int begin = ir.loop_begin_imm(counter, count_imm);
        body();
        ir.loop_end(counter, begin);
    } else {
        body();
    }
}

// Like the helper above, but `step` runs after `body` on each iteration of a
// real loop. When `body` is inlined (0 or 1 iterations), `step` is skipped.
// Use this for per-iteration pointer updates that aren't needed for a single
// iteration.
template <typename body_t, typename step_t>
void emit_loop_imm(ir_t &ir, dim_t count_imm, body_t body, step_t step) {
    if (count_imm > 1) {
        const int counter = ir.new_gpr();
        const int begin = ir.loop_begin_imm(counter, count_imm);
        body();
        step();
        ir.loop_end(counter, begin);
    } else {
        body();
    }
}

// Loop with a runtime iteration count held in `count_reg`. Use this when the
// iteration count is not known at IR generation time.
template <typename body_t>
void emit_loop_reg(ir_t &ir, int count_reg, body_t body) {
    const int counter = ir.new_gpr();
    const int begin = ir.loop_begin_reg(counter, count_reg);
    body();
    ir.loop_end(counter, begin);
}

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
