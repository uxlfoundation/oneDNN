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

#ifndef GPU_INTEL_ZE_KERNEL_HPP
#define GPU_INTEL_ZE_KERNEL_HPP

#include "gpu/intel/compute/kernel.hpp"

#include "xpu/ze/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ze {

class kernel_t : public compute::kernel_impl_t {
public:
    static status_t make(compute::kernel_t &compute_kernel,
            const std::shared_ptr<xpu::ze::wrapper_t<ze_module_handle_t>>
                    &amodule,
            ze_kernel_handle_t akernel, const std::string &kernel_name);
    ~kernel_t() override = default;

    status_t check_alignment(
            const compute::kernel_arg_list_t &arg_list) const override;

    status_t set_arg(
            int arg_index, size_t arg_size, const void *arg_value) const;

    status_t parallel_for(impl::stream_t &stream,
            const compute::nd_range_t &range,
            const compute::kernel_arg_list_t &arg_list,
            const xpu::event_t &deps, xpu::event_t &out_dep) override;

    status_t get_kernel_binary(xpu::binary_t &binary) const override;

    std::string name() const override { return kernel_name_; }

    status_t dump() const override;

private:
    // See description in the class implementation.
    friend class kernel_compat_t;

    kernel_t() = delete;
    kernel_t(const std::shared_ptr<xpu::ze::wrapper_t<ze_module_handle_t>>
                     &amodule,
            ze_kernel_handle_t akernel, const std::string &kernel_name);

    // Note: `shared_ptr` is mandatory to cover cases when a single module
    // is used to compile several kernels. In that case, this abstraction
    // shouldn't destroy the module and `shared_ptr` takes care of destroying
    // the last one through ref_counting.
    //
    // Additionally, it's important to initialize `module` first to ensure the
    // order of destruction is `kernel_` first and then `module_`.
    std::shared_ptr<xpu::ze::wrapper_t<ze_module_handle_t>> module_;
    xpu::ze::wrapper_t<ze_kernel_handle_t> kernel_;
    std::string kernel_name_;

    DNNL_DISALLOW_COPY_AND_ASSIGN(kernel_t);
};

} // namespace ze
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_ZE_KERNEL_HPP
