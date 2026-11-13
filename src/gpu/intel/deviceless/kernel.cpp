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

#include "gpu/intel/deviceless/kernel.hpp"

#include "gpu/intel/utils.hpp"

#include <memory>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace deviceless {

kernel_t::kernel_t(const xpu::binary_t &binary, const char *kernel_name)
    : binary_(binary), name_(kernel_name) {}

status_t kernel_t::get_binary(
        const impl::engine_t *engine, xpu::binary_t &binary) const {
    binary = binary_;
    return status::success;
}

status_t kernel_t::get_kernel_binary(xpu::binary_t &binary) const {
    binary = binary_;
    return status::success;
}

status_t kernel_t::get_binary_size(
        const impl::engine_t *engine, size_t *binary_size) const {
    *binary_size = binary_.size();
    return status::success;
}

status_t kernel_t::dump() const {
    return gpu_utils::dump_kernel_binary(binary_, name_);
}

std::string kernel_t::name() const {
    return name_;
}

status_t kernel_t::check_alignment(
        const compute::kernel_arg_list_t &arg_list) const {
    return status::unimplemented;
}

status_t kernel_t::make(compute::kernel_t &compute_kernel,
        const xpu::binary_t &binary, const char *kernel_name) {
    if (!kernel_name) return status::invalid_arguments;
    compute_kernel = compute::kernel_t(
            std::make_shared<kernel_t>(binary, kernel_name));
    return status::success;
}

} // namespace deviceless
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
