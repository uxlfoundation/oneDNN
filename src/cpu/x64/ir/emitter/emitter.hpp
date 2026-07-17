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

#ifndef CPU_X64_IR_EMITTER_EMITTER_HPP
#define CPU_X64_IR_EMITTER_EMITTER_HPP

// Lowers an IR along with its register allocation into target machine code,
// using a `jit_generator`.

#include <deque>
#include <functional>
#include <utility>
#include <vector>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/ir/ir.hpp"
#include "cpu/x64/ir/reg_alloc.hpp"
#include "cpu/x64/ir/reg_config.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

// Static data the emitter returns to be used after the kernel code
// (after postamble). For example: AVX2 mask tables.
//
// A `std::deque` keeps element addresses stable. A label is referenced while
// the emitter goes through the IR and is only bound later by
// `emit_data_section`, so the addresses must stay valid as more constants are
// appended.
struct data_section_t {
    std::deque<std::pair<std::vector<unsigned char>, Xbyak::Label>> constants;
    static constexpr int alignment = 32;
};

// Lowering/code emission pass. Walks the IR once and turns each abstract
// operation into target-specific instructions, using the physical registers
// chosen by the allocator.
//
// Values that cannot stay in a register are spilled on the stack. This is
// handled by loading spilled inputs into reserved scratch registers before use,
// and storing results back on stack after use. These scratch registers are
// excluded from allocation so they are always available for spill handling.
//
// Supporting another target ISA mainly requires providing a similar backend
// (see `emitter/backend_avx2.hpp`) or extending an existing one, then adding a
// dispatch branch in `emit()`.
//
// `data` collects static constants the lowering needs (e.g. mask tables). The
// emitter appends to it and references each entry rip-relative. The caller
// then emits the bytes with `emit_data_section()` after the `postamble()`.
//
// This is the main entry point for the emitter. It dispatches by ISA family to
// the matching backend.
//
// `inject` lowers the `inject_postops` operation to the JIT post-ops injector.
// It is the interoperability layer between the IR and the non-IR injector. When
// the IR has no `inject_postops` op it is unused and may be left empty.
//
//   acc_phys  - physical vec register indices of the accumulators, in the order
//               passed to `ir_t::inject_postops()`
//   base_phys - physical gpr index of the base (output) pointer
using inject_postops_fn_t
        = std::function<void(const std::vector<int> &acc_phys, int base_phys)>;

// Export for testing.
void DNNL_API emit(jit_generator_t &gen, const ir_t &ir,
        const reg_alloc_result_t &alloc, const reg_config_t &reg_cfg,
        data_section_t &data, const inject_postops_fn_t &inject = {});

// Emit the accumulated static data after the kernel's postamble. It aligns,
// binds each label and then write the data bytes with `db`.
// Must be called once, after the emitter and the postamble.
void DNNL_API emit_data_section(jit_generator_t &gen, data_section_t &data);

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
