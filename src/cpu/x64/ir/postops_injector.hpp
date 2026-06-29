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

// Reusable driver for the JIT-based post-ops injector. It implements the
// lowering logic for `inject_postops` IR instruction (see `emitter.hpp` and
// `inject_postops_fn_t`).
//
// - The IR instruction and the emitter callback that uses this
//    `postops_injector_t` are builder-independent.
// - The IR-based kernel does the following in `generate()`:
//   - Creates this object from post-ops + destination descriptor.
//   - Wraps it inside a callback that is passed to the emitter and calls `apply()`.
//   - Calls `prepare_table()` once after the postamble.
//
// ISA handling:
// - The only ISA-specific detail is vector width (Ymm/Zmm).
// - The IR-based kernels do not need to be templated. The width is chosen
//   based on the `isa`.
// - The width-specific injector is stored in a type-erased manner.
//
// Register handling:
// - The injector saves and restores all registers it uses.
// - No need to reserve registers in the IR allocator.
// - Save/restore cost is paid once per call, after the computation is done.

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

struct postops_injector_t {
    // host      - generator buffer the injected post-op code is emitted into
    // isa       - kernel isa
    // post_ops  - attribute post-ops chain to apply
    // dst_md    - destination memory descriptor (used by binary post-ops)
    // param_reg - gpr holding the kernel-parameter-struct pointer (used by
    //             binary post-ops to reach their right-hand-side arguments)
    postops_injector_t(jit_generator_t *host, cpu_isa_t isa,
            const post_ops_t &post_ops, const memory_desc_t &dst_md,
            const Xbyak::Reg64 &param_reg);

    postops_injector_t(const postops_injector_t &) = delete;
    postops_injector_t &operator=(const postops_injector_t &) = delete;

    // Apply the post-ops to the accumulators in `acc_phys` (physical
    // vec-register indices).
    void apply(const std::vector<int> &acc_phys);

    // Emit the post-ops constant table. Call once, after the postamble. Only
    // eltwise post-ops have a table, so this is a no-op without one.
    void maybe_prepare_table();

private:
    // Type-erased `jit_uni_postops_injector_base_t<Vmm>`. It is cast to the
    // appropriate target type when used, based on `is_zmm_`.
    std::shared_ptr<void> injector_;
    bool is_zmm_ = false;
    // Whether the chain has an eltwise post-op (the only kind with a table).
    bool with_eltwise_ = false;
};

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
