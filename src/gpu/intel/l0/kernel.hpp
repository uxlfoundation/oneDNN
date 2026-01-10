/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef GPU_INTEL_L0_KERNEL_HPP
#define GPU_INTEL_L0_KERNEL_HPP

#include <thread>

#include "common/rw_mutex.hpp"
#include "gpu/intel/compute/kernel.hpp"
#include "gpu/intel/l0/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace l0 {

class kernel_t : public compute::kernel_impl_t {
public:
    static status_t make(compute::kernel_t &compute_kernel,
            const std::shared_ptr<module_wrapper_t> module_ptr,
            const ze_kernel_handle_t kernel_ptr,
            const std::string &kernel_name);
    ~kernel_t() override;

    status_t check_alignment(
            const compute::kernel_arg_list_t &arg_list) const override;
    status_t set_arg(
            int arg_index, size_t arg_size, const void *arg_value) const;
    status_t parallel_for(impl::stream_t &stream,
            const compute::nd_range_t &range,
            const compute::kernel_arg_list_t &arg_list,
            const xpu::event_t &deps, xpu::event_t &out_dep) override;

    status_t get_kernel_binary(xpu::binary_t &binary) const override;
    std::string name() const override;
    status_t dump() const override;

private:
    friend class kernel_compat_t;
    kernel_t(const std::shared_ptr<module_wrapper_t> module_ptr,
            const ze_kernel_handle_t kernel_ptr,
            const std::string &kernel_name);

    std::shared_ptr<module_wrapper_t> module_;
    ze_kernel_handle_t kernel_;
    std::string kernel_name_;

    std::shared_ptr<ze_event_pool_handle_t> event_pool_;
    std::shared_ptr<event_wrapper_t> event_;

    kernel_t() = delete;
    DNNL_DISALLOW_COPY_AND_ASSIGN(kernel_t);
};

} // namespace l0
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_L0_KERNEL_HPP
