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

#ifndef CPU_X64_IR_REG_CONFIG_HPP
#define CPU_X64_IR_REG_CONFIG_HPP

#include <vector>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/ir/reg_alloc.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

// A few physical registers are selected before the emitter runs and passed in.
// The register allocator avoids using them, so they are always available for
// these specific purposes:
//  - param_reg: stores a pointer to the kernel's input arguments.
//  - gpr_scratch / vec_scratch: temporary registers that the emitter can use
//    whenever needed. If a value could not stay in a register and was spilled,
//    the emitter loads it into one of these registers, performs the required
//    work, and then stores it back.
//
// `pools` contains the registers that are available for allocation for each
// register kind, along with the stack space needed when a value for that
// kind is spilled.
//
// TODO: Consider adding a second pass. The first pass could determine how many
// registers need to be spilled, giving a more accurate estimate of the register
// scratch size. Now we always book a constant size during the `generate()`
// call.
struct reg_config_t {
    reg_pools_t pools;
    std::vector<int> gpr_scratch; // >= 2 entries
    std::vector<int> vec_scratch; // >= 3 entries
    int param_reg = 0;
};

// Creates the register config for the given ISA.
//
// The number of vector registers and their spill-slot size are inferred
// directly from the ISA. For example, AVX2 uses 16 vector registers with
// 32-byte spill slots, while AVX-512 uses 32 vector registers with 64-byte
// spill slots.
//
// Some registers are reserved and are not included in the allocatable pools:
// - `rsp_reg` (stack pointer)
// - `param_reg` (parameter pointer)
// - `gpr_scratch` registers
// - `vec_scratch` registers
//
// The emitter uses these scratch registers when loading and storing spilled
// values.
//
// Export for testing.
reg_config_t DNNL_API make_reg_config(cpu_isa_t isa, int param_reg, int rsp_reg,
        const std::vector<int> &gpr_scratch,
        const std::vector<int> &vec_scratch);

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
