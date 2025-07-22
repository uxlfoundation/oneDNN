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

class kernel_wrapper_t {
public:
    kernel_wrapper_t(ze_kernel_handle_t kernel = nullptr);
    operator ze_kernel_handle_t() const;
    status_t set_arg(int arg_index, size_t arg_size, const void *arg_value);

private:
    ze_kernel_handle_t kernel_;
};

class kernel_cache_t {
public:
    kernel_cache_t(ze_module_handle_t module, ze_kernel_handle_t main_kernel);
    ~kernel_cache_t();

    status_t get(kernel_wrapper_t **kernel);
    status_t clone_kernel(ze_kernel_handle_t *cloned_kernel);

private:
    ze_module_handle_t module_;
    ze_kernel_handle_t main_kernel_;
    std::unordered_map<std::thread::id, kernel_wrapper_t> kernels_;
    utils::rw_mutex_t mutex_;
};

class kernel_t : public compute::kernel_impl_t {
public:
    static status_t make(compute::kernel_t &compute_kernel,
            ze_module_handle_t module, ze_kernel_handle_t kernel,
            const compute::program_src_t &src);
    ~kernel_t() override;

    status_t parallel_for(impl::stream_t &stream,
            const compute::nd_range_t &range,
            const compute::kernel_arg_list_t &arg_list,
            const xpu::event_t &deps, xpu::event_t &out_dep) override;
    status_t get_kernel_binary(xpu::binary_t &binary) const override;
    std::string name() const override;
    status_t dump() const override;
    status_t check_alignment(
            const compute::kernel_arg_list_t &arg_list) const override;
    void save_output_events() override;

private:
    friend class kernel_compat_t;
    kernel_t(const ze_module_handle_t module, const ze_kernel_handle_t kernel,
            const gpu::intel::compute::program_src_t &src);

    ze_module_handle_t module_;
    ze_kernel_handle_t kernel_;
    compute::program_src_t src_;
    std::shared_ptr<kernel_cache_t> cache_;
    bool save_events_;
};

} // namespace l0
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_L0_KERNEL_HPP
