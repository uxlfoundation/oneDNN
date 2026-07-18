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

#ifndef CPU_X64_IR_POSTOPS_INJECTOR_HPP
#define CPU_X64_IR_POSTOPS_INJECTOR_HPP

// Driver for the JIT-based post-ops injector. It lowers the `inject_postops` IR
// operation (see `emitter.hpp` and `inject_postops_fn_t`).
//
// The IR operation and the emitter callback that drives this
// `postops_injector_t` are builder-independent. An IR-based kernel sets it up
// in `generate()`:
//   1. Create this object from the post-ops chain and the destination
//      descriptor.
//   2. Wrap it in a callback passed to the emitter that calls `apply()`.
//   3. Call `maybe_prepare_table()` once after the postamble.
//
// ISA handling:
// - The only ISA-specific detail is the vector width (Ymm or Zmm), chosen from
//   the `isa`. The width-specific injector is stored type-erased, so IR-based
//   kernels do not need to be templated.
//
// Register handling:
// - The injector saves and restores every register it uses, so its registers
//   need not be reserved in the IR allocator. The save/restore cost is paid
//   once per call.

#include <memory>
#include <vector>

#include "common/c_types_map.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

// Interfaces marked with DNNL_API are exported for testing purposes only.
struct postops_injector_t {
    // gen            - generator the injected post-op code is emitted into
    // isa            - kernel ISA
    // post_ops       - attribute post-ops chain to apply
    // dst_md         - destination memory descriptor (used by binary post-ops)
    // param_reg      - gpr holding the kernel-parameter-struct pointer (used by
    //                  binary post-ops to reach their right-hand-side arguments)
    // rhs_arg_offset - byte offset in the parameter struct of the binary
    //                  right-hand-side argument pointer array
    // dst_orig_off   - byte offset in the parameter struct of the destination
    //                  origin pointer, used to turn an accumulator address into
    //                  its position in the destination tensor
    // tail_elems     - right-hand-side elements a partial (tail) load reads
    DNNL_API postops_injector_t(jit_generator_t &gen, cpu_isa_t isa,
            const post_ops_t &post_ops, const memory_desc_t &dst_md,
            const Xbyak::Reg64 &param_reg, size_t rhs_arg_offset,
            size_t dst_orig_off, int tail_elems);

    postops_injector_t(const postops_injector_t &) = delete;
    postops_injector_t &operator=(const postops_injector_t &) = delete;

    // Apply the post-ops to the accumulators in `acc_phys` (physical vec
    // register indices). For binary post-ops, `base_phys` and `out_byte_off`
    // give each accumulator's output address. `elems` is the active element
    // count, `-1` for a full vector, and limits how many right-hand-side
    // elements are read. `mask_phys` is the tail opmask (a K register), used
    // only by the avx512 emitter, which is not enabled yet.
    void DNNL_API apply(const std::vector<int> &acc_phys, int base_phys,
            const std::vector<dim_t> &out_byte_off, int mask_phys, int elems);

    // Emit the post-ops constant table. Call once, after the postamble. Only
    // eltwise post-ops have a table, so this is a no-op without one.
    void DNNL_API maybe_prepare_table();

private:
    // Type-erased `jit_uni_postops_injector_base_t<Vmm>`. Cast to the target
    // type when used, based on `is_zmm_`.
    std::shared_ptr<void> injector_;
    bool is_zmm_ = false;
    // Whether the chain has an eltwise post-op (the only kind with a table).
    bool with_eltwise_ = false;
    // Whether the chain has a binary post-op (the only kind needing rhs args).
    bool with_binary_ = false;
    // Number of right-hand-side elements a tail load reads.
    int tail_elems_ = 0;
};

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
