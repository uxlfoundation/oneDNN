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

#ifndef GPU_INTEL_DEVICELESS_KERNEL_HPP
#define GPU_INTEL_DEVICELESS_KERNEL_HPP

#include "gpu/intel/compute/kernel.hpp"

#include "xpu/utils.hpp"

#include <string>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace deviceless {

// Holds only the binary, which is what a primitive exports to its cache blob.
// Without a device, the kernel can't run.
class kernel_t : public compute::kernel_impl_t {
public:
    kernel_t(const xpu::binary_t &binary, const char *kernel_name);
    ~kernel_t() override = default;

    status_t get_binary(
            const impl::engine_t *engine, xpu::binary_t &binary) const override;
    status_t get_kernel_binary(xpu::binary_t &binary) const override;
    status_t get_binary_size(
            const impl::engine_t *engine, size_t *binary_size) const override;

    status_t dump() const override;
    std::string name() const override;
    status_t check_alignment(
            const compute::kernel_arg_list_t &arg_list) const override;

    static status_t make(compute::kernel_t &compute_kernel,
            const xpu::binary_t &binary, const char *kernel_name);

private:
    xpu::binary_t binary_;
    std::string name_;
};

} // namespace deviceless
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_DEVICELESS_KERNEL_HPP
