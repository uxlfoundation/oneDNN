/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
* Copyright 2020 Codeplay Software Limited
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

#include "gpu/nvidia/cudnn_sum.hpp"
#include "gpu/gpu_impl_list.hpp"
#include "gpu/nvidia/sycl_cuda_engine.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

namespace {

// clang-format off
constexpr impl_list_item_t cuda_sum_impl_list[] = {
        GPU_SUM_INSTANCE_NVIDIA(gpu::nvidia::cudnn_ref_sum_t)
        nullptr
};
// clang-format on

} // namespace

void check_kernel_size(const std::string &kernel_name, size_t kernel_size,
        size_t icache_size) {
    if (kernel_size > icache_size) {
        ir_warning() << kernel_name
                     << " larger than icache, kernel: " << kernel_size
                     << " bytes, icache: " << icache_size << " bytes\n";
    }
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
