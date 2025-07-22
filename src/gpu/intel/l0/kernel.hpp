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

class module_wrapper_t {
public:
    module_wrapper_t(ze_module_handle_t module);
    ~module_wrapper_t();
    operator ze_module_handle_t() const;

private:
    ze_module_handle_t module_;

    module_wrapper_t() = delete;
    DNNL_DISALLOW_COPY_AND_ASSIGN(module_wrapper_t);
};

class kernel_wrapper_t {
public:
    kernel_wrapper_t(
            const char *kernel_name, const ze_module_handle_t module_ptr);
    ~kernel_wrapper_t();

    operator ze_kernel_handle_t() const;

    status_t set_arg(int arg_index, size_t arg_size, const void *arg_value);
    ze_event_handle_t create_out_event(
            const ze_context_handle_t context_ptr, const bool profiling);

private:
    ze_kernel_handle_t kernel_;
    ze_event_pool_handle_t event_pool_;
    ze_event_handle_t event_;

    kernel_wrapper_t() = delete;
    DNNL_DISALLOW_COPY_AND_ASSIGN(kernel_wrapper_t);
};

class kernel_t : public compute::kernel_impl_t {
public:
    static status_t make(compute::kernel_t &compute_kernel,
            const std::shared_ptr<module_wrapper_t> module_ptr,
            const ze_kernel_handle_t kernel_ptr);
    ~kernel_t() override;

    status_t check_alignment(
            const compute::kernel_arg_list_t &arg_list) const override;
    kernel_wrapper_t *get();
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
            const ze_kernel_handle_t kernel_ptr);

    std::shared_ptr<module_wrapper_t> module_;
    ze_kernel_handle_t main_kernel_;
    std::string kernel_name_;
    xpu::binary_t kernel_binary_;

    utils::rw_mutex_t mutex_;
    std::unordered_map<std::thread::id, std::unique_ptr<kernel_wrapper_t>>
            kernels_;

    kernel_t() = delete;
    DNNL_DISALLOW_COPY_AND_ASSIGN(kernel_t);
};

} // namespace l0
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_L0_KERNEL_HPP
