/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#ifndef GPU_INTEL_CONV_JIT_V2_KERNEL_HPP
#define GPU_INTEL_CONV_JIT_V2_KERNEL_HPP

#include "gpu/intel/conv/jit/v2/builder.hpp"
#include "gpu/intel/conv/jit/v2/kernel_desc.hpp"
#include "gpu/intel/jit/codegen/kernel.hpp"
#include "gpu/intel/jit/ir/ir.hpp"
#include "gpu/intel/jit/ir/v2/builder.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace conv {
namespace jit {

using namespace intel::jit;

namespace v2 {

using namespace intel::jit::v2;

class kernel_t : public ir_kernel_t {
public:
    kernel_t(const kernel_desc_base_t &_desc, const intel::engine_t *engine)
        : ir_kernel_t(_desc, engine, {GENERATOR_NAME, GENERATOR_LINE}) {

        auto &desc = static_cast<const kernel_desc_t &>(_desc);

        // Build IR for the kernel.
        var_manager_t var_mgr(kernel_iface());
        stmt_t body = build_ir(options(), desc, var_mgr);
        generate_from_ir(body);
    }
};

} // namespace v2
} // namespace jit
} // namespace conv
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
