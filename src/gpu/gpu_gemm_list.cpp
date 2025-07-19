/*******************************************************************************
* Copyright 2021-2025 Intel Corporation
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

#include "common/compiler_workarounds.hpp"

#include "gpu/gpu_impl_list.hpp"

#if DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
#include "gpu/intel/jit/binary_format.hpp"

#include "gpu/intel/gemm/jit.hpp"
#include "gpu/intel/gemm/jit_xe_hp_systolic.hpp"
#include "gpu/intel/gemm/ref.hpp"
#include "gpu/intel/gemm/with_post_ops.hpp"

#ifdef DNNL_DEV_MODE
#include "gpu/intel/gemm/conv.hpp"
#endif

#endif

namespace dnnl {
namespace impl {
namespace gpu {

namespace {

// clang-format off
constexpr impl_list_item_t impl_list[] = {
        GPU_INSTANCE_INTEL_DEVMODE(intel::gemm::conv_gemm_t)
        GPU_INSTANCE_INTEL(intel::gemm::xe_hp_systolic_gemm_t)
        GPU_INSTANCE_INTEL(intel::gemm::gemm_with_post_ops_t)
        GPU_INSTANCE_INTEL(intel::gemm::gen_gemm_t)
        GPU_INSTANCE_INTEL_REF(intel::gemm::ref_gemm_t)
        nullptr,
};
// clang-format on
} // namespace

const impl_list_item_t *get_gemm_impl_list(const gemm_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace gpu
} // namespace impl
} // namespace dnnl
