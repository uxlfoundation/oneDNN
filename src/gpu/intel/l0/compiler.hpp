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

#ifndef GPU_INTEL_L0_COMPILER_HPP
#define GPU_INTEL_L0_COMPILER_HPP

#include "gpu/intel/l0/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace l0 {

status_t ocloc_get_extensions(std::string &extensions);
bool ocloc_mayiuse_microkernels(const std::string &kernel_code);
status_t ocloc_build_kernels(const std::string &kernel_code,
        const std::string &options, const std::string &ip_version,
        xpu::binary_t &binary);

} // namespace l0
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_L0_COMPILER_HPP