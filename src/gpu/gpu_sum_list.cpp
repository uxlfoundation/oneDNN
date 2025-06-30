/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#include "gpu/gpu_impl_list.hpp"

#include "common/utils.hpp"
#include "gpu/gpu_sum_pd.hpp"

#include "gpu/generic/ref_sum.hpp"

#if DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
#include "gpu/intel/many_inputs_sum.hpp"
#include "gpu/intel/multi_po_reorder_sum.hpp"
#include "gpu/intel/simple_sum.hpp"
#include "gpu/intel/xe_sum.hpp"
#endif

#if DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
#include "gpu/nvidia/cudnn_sum.hpp"
#endif

#ifdef GENERIC_SYCL_KERNELS_ENABLED
#include "gpu/generic/sycl/ref_sum.hpp"
#include "gpu/generic/sycl/ref_sum_many_inputs.hpp"
#endif

namespace dnnl {
namespace impl {
namespace gpu {

namespace {

// clang-format off
constexpr impl_list_item_t impl_list[] = REG_SUM_P({
        GPU_SUM_INSTANCE_INTEL(intel::multi_po_reorder_sum_t)
        GPU_SUM_INSTANCE_INTEL(intel::xe_sum_t)
        GPU_SUM_INSTANCE_INTEL(intel::many_inputs_sum_t)
        GPU_SUM_INSTANCE_INTEL(intel::simple_sum_t<data_type::f32>)
        GPU_SUM_INSTANCE_NVIDIA(nvidia::cudnn_ref_sum_t)
        GPU_SUM_INSTANCE_GENERIC_SYCL(generic::sycl::ref_sum_t)
        GPU_SUM_INSTANCE_GENERIC_SYCL(generic::sycl::ref_sum_many_inputs_t)
        GPU_SUM_INSTANCE_GENERIC(generic::ref_sum_t)
        nullptr,
});
// clang-format on

} // namespace

const impl_list_item_t *get_sum_impl_list() {
    return impl_list;
}

} // namespace gpu
} // namespace impl
} // namespace dnnl
