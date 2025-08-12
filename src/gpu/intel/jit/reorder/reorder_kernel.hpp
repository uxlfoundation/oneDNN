/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_REORDER_REORDER_KERNEL_HPP
#define GPU_INTEL_JIT_REORDER_REORDER_KERNEL_HPP

#include "gpu/intel/jit/codegen/kernel.hpp"
#include "gpu/intel/jit/codegen/ngen_helpers.hpp"
#include "gpu/intel/jit/codegen/register_scope.hpp"
#include "gpu/intel/jit/ir/ir_builder.hpp"
#include "gpu/intel/jit/ir/message.hpp"
#include "gpu/intel/jit/ir/reorder.hpp"
#include "gpu/intel/jit/ir/tensor.hpp"
#include "gpu/intel/jit/reorder/ir_builder.hpp"
#include "gpu/intel/jit/utils/ngen_type_bridge.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

class reorder_kernel_t : public ir_kernel_t {
public:
    reorder_kernel_t(const reorder_config_t &cfg,
            const std::string &kernel_name, const kernel_info_t &kernel_info,
            bool require_dpas, const primitive_desc_t *pd = nullptr)
        : ir_kernel_t(kernel_info.iface(kernel_name), cfg.exec_cfg(),
                kernel_info.nd_range().local_range(), require_dpas,
                {GENERATOR_NAME, GENERATOR_LINE}) {
        const primitive_attr_t *attr = (pd) ? pd->attr() : nullptr;
        const memory_desc_t *dst_md = (pd) ? pd->dst_md() : nullptr;
        reorder_ir_builder_t builder(cfg, kernel_info, attr, dst_md);
        const stmt_t &body = builder.stmt();
        generate_from_ir(body);
    }
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
