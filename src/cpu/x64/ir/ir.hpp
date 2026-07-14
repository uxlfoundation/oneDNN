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
// An `ir_t` is a flat list of operations. Each operation reads and writes
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
// Memory operands are base register + displacement, where the displacement is
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

// vreg_t and label_t are strong aliases for `int`. They are distinct id types
// that do not implicitly convert to or from `int`, or to each other, so a
// vreg, a label, and a raw int cannot be mixed up by accident. `none` marks an
// unset id. Cast to `int` where a raw index is needed.

// A virtual register id. A placeholder the allocator later maps to a physical
// register or a spill slot. Its kind (gpr/vec/mask) and, for a vec, element
// data type live separately in vreg_info_.
enum class vreg_t : int { none = -1 };

// A label id names a jump target for label/jmp/jz.
enum class label_t : int { none = -1 };

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
    // dst += sum_{i=0}^{N-1} (s0[i] * s1[i]), where N is the dot length
    vdot,
    // horizontal reduction of dst; result in element 0. s0 is scratch
    // (overwritten).
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
    // bind label id `label_id`
    label,
    // jump to label id `label_id`
    jmp,
    // if s0 == 0 jump to label id `label_id`
    jz,
};

// A memory address of the form base register + displacement.
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
    vreg_t base = vreg_t::none;
    // constant byte offset
    dim_t disp = 0;
    // base is the kernel-argument pointer
    bool is_param = false;
};

// An `op_t` is a single IR operation. One fixed struct represents every
// operation and works like a tagged union, where the `kind` field decides which
// operation it is. Each operation only uses the fields it needs and ignores the
// rest, which stay at default values.
//
// The meaning of each field for a specific operation is explained in the
// `op_kind_t` comments. Whether dst, s0, and s1 are used for reading or writing
// is clearly defined in ir_t::def_use().
//
// kind - which operation this is. Determines how other fields are used.
// dst  - virtual register that is written to, or `none` if none is written.
//        Some ops (e.g. vdot/vadd) both read from and write to dst.
// s0,s1 - source virtual registers (inputs), or `none` if not used.
// imm   - immediate value whose meaning depends on the kind:
//         * mov_imm        -> literal constant
//         * loop_begin     -> loop trip count
//         * set_mask_imm   -> active element count
//         * vload_masked / vstore_masked -> active element count
// mem   - memory address used only by load/store operations.
// match - for loop_end, index of matching loop_begin operation.
// label_id - target label id for label/jmp/jz. When unused it's `none`.
// init_is_reg - for loop_begin, if true, initialize loop counter from s0
//                (runtime value) instead of imm.
struct op_t {
    op_kind_t kind;
    vreg_t dst = vreg_t::none;
    vreg_t s0 = vreg_t::none, s1 = vreg_t::none;
    dim_t imm = 0;
    mem_t mem;
    int match = -1;
    label_t label_id = label_t::none;
    bool init_is_reg = false;
};

// Metadata for virtual registers that contains the register kind and the
// data type of the values it holds. `dt` is `undef` for gpr and mask.
struct vreg_info_t {
    reg_kind_t kind;
    data_type_t dt = data_type::undef;
};

// An `ir_t` is the operation list plus, for each virtual register, its info
// (kind and for a vec its element data type) so the allocator knows which
// physical registers it can use and the emitter knows which dtype-specific
// instruction to lower each op to.
//
// * The builder fills it through the helpers below
// * The allocator and emitter read it
//
// Export for testing.
#ifdef _MSC_VER
// ir_t is exported whole so that new member functions are covered automatically
// without annotating each one. It has std::vector members, which triggers
// C4251. The library and the tests are build by the same compiler, so no
// mismatch is possible. Silence it.
#pragma warning(push)
#pragma warning(disable : 4251)
#endif
struct DNNL_API ir_t {
    const std::vector<vreg_info_t> &vreg_info() const { return vreg_info_; }
    const std::vector<op_t> &ops() const { return ops_; }
    int n_labels() const { return n_labels_; }

    // Convenience wrappers around `new_vreg` for the specific kinds.
    vreg_t new_gpr() { return new_vreg(reg_kind_t::gpr); }
    vreg_t new_vec(data_type_t dt) { return new_vreg(reg_kind_t::vec, dt); }
    vreg_t new_mask() { return new_vreg(reg_kind_t::mask); }

    // Add a new label and return its id
    label_t new_label() { return static_cast<label_t>(n_labels_++); }
    int n_vregs() const { return (int)vreg_info_.size(); }
    int n_ops() const { return (int)ops_.size(); }

    // Each interface appends one operation and names registers by `id`. The
    // ones returning `int` return the new operation's index.
    //
    // Refer to documentation for `op_kind_t` for each interface. The notes here
    // add only the parameter-level details that are not obvious from the
    // signature.

    // gpr
    int mov_imm(vreg_t dst, dim_t imm);
    void mov_reg(vreg_t dst, vreg_t src);
    void add_imm(vreg_t dst, dim_t imm);
    void add_reg(vreg_t dst, vreg_t src);
    // Like `load`, but reads a field of the kernel-argument struct through the
    // ABI parameter pointer instead of a virtual base register.
    void load_param(vreg_t dst, dim_t disp);
    void load(vreg_t dst, vreg_t base, dim_t disp);

    // vec
    void vzero(vreg_t dst);
    void vload(vreg_t dst, vreg_t base, dim_t disp);
    void vdot(vreg_t dst, vreg_t a, vreg_t b);
    // `workspace` is scratch. It is overwritten by this call, so pass a vreg
    // whose value is not needed afterwards.
    void vhreduce(vreg_t dst, vreg_t workspace);

    // vec (masked)
    // `elems` is the number of active elements. `mask` is the mask register
    // holding that pattern (from `set_mask_imm`), or `vreg_t::none` for a
    // single element or a full vector, where no mask register is needed.
    void vload_masked(
            vreg_t dst, vreg_t base, dim_t disp, vreg_t mask, int elems);
    // Same shape as `vload_masked`, but stores `src` to [base + disp] instead
    // of loading.
    void vstore_masked(
            vreg_t base, dim_t disp, vreg_t src, vreg_t mask, int elems);

    // mask
    void set_mask_imm(vreg_t mask, int n_elems);

    // control flow
    // Pass the returned op index to `loop_end` to close the loop.
    int loop_begin_imm(vreg_t counter, dim_t count);
    // Like `loop_begin_imm`, but the iteration count is the runtime value in
    // `init` rather than the compile-time `count`.
    int loop_begin_reg(vreg_t counter, vreg_t init);
    // Close the loop opened by `loop_begin_imm` or `loop_begin_reg`, passing
    // the op index it returned as `begin_idx`.
    void loop_end(vreg_t counter, int begin_idx);
    void label(label_t label_id);
    void jmp(label_t label_id);
    void jz(vreg_t cond, label_t label_id);

    // Fill defs/uses with the vregs this operation writes/reads. Liveness and
    // the allocator depend on it, so it must match what the emitter emits.
    void def_use(const op_t &op, std::vector<int> &defs,
            std::vector<int> &uses) const;

private:
    // Add a vreg of kind `k` and, for a vec, element data type `dt`.
    // Return its id.
    vreg_t new_vreg(reg_kind_t k, data_type_t dt = data_type::undef);

    // Info for each vreg, indexed by its id
    std::vector<vreg_info_t> vreg_info_;
    // All operations, in order
    std::vector<op_t> ops_;

    // Label counter.
    int n_labels_ = 0;
};
#ifdef _MSC_VER
#pragma warning(pop)
#endif

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
// count_imm == 0 emits nothing, == 1 inlines `body` once, > 1 builds a loop.
template <typename body_t>
void emit_loop_imm(ir_t &ir, dim_t count_imm, body_t body) {
    if (count_imm == 0) return;
    if (count_imm == 1) {
        body();
        return;
    }
    const vreg_t counter = ir.new_gpr();
    const int begin = ir.loop_begin_imm(counter, count_imm);
    body();
    ir.loop_end(counter, begin);
}

// Like the helper above, but `step` runs after `body` on each loop iteration.
// A single inlined iteration (count_imm == 1) skips `step`. Use this for
// per-iteration pointer updates not needed for a single iteration.
template <typename body_t, typename step_t>
void emit_loop_imm(ir_t &ir, dim_t count_imm, body_t body, step_t step) {
    if (count_imm == 0) return;
    if (count_imm == 1) {
        body();
        return;
    }
    const vreg_t counter = ir.new_gpr();
    const int begin = ir.loop_begin_imm(counter, count_imm);
    body();
    step();
    ir.loop_end(counter, begin);
}

// Loop with a runtime iteration count held in `count_reg`. Use this when the
// iteration count is not known at IR generation time. A count of 0 skips the
// loop. A negative count never reaches 0 and gets stuck, so the caller must
// ensure count_reg >= 0.
template <typename body_t>
void emit_loop_reg(ir_t &ir, vreg_t count_reg, body_t body) {
    const vreg_t counter = ir.new_gpr();
    const label_t skip = ir.new_label();
    ir.jz(count_reg, skip); // zero-trip guard. A negative count gets stuck.
    const int begin = ir.loop_begin_reg(counter, count_reg);
    body();
    ir.loop_end(counter, begin);
    ir.label(skip);
}

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
