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

#ifndef GPU_INTEL_ZE_UTILS_HPP
#define GPU_INTEL_ZE_UTILS_HPP

#include "gpu/intel/compute/device_info.hpp"

#include "xpu/ze/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ze {

status_t init_gpu_hw_info(impl::engine_t *engine, ze_device_handle_t device,
        ze_context_handle_t context, uint32_t &ip_version,
        compute::gpu_arch_t &gpu_arch, compute::gpu_product_t &product,
        uint64_t &native_extensions, bool &mayiuse_systolic,
        bool &mayiuse_ngen_kernels);

status_t get_module_binary(
        ze_module_handle_t module_handle, xpu::binary_t &binary);

status_t get_kernel_binary(ze_kernel_handle_t kernel, xpu::binary_t &binary);

bool mayiuse_microkernels(ze_device_handle_t device,
        ze_context_handle_t context, const std::string &code);

status_t compile_ocl_module_to_binary(ze_device_handle_t device,
        ze_context_handle_t context, const std::string &code,
        const std::string &options, xpu::binary_t &binary);

status_t create_kernels(ze_device_handle_t device, ze_context_handle_t context,
        const std::vector<const char *> &kernel_names,
        const xpu::binary_t &binary, ze_module_handle_t *module_ptr,
        std::vector<ze_kernel_handle_t> &kernels);

} // namespace ze
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
