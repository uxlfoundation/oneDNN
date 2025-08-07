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

#ifndef GPU_INTEL_RESAMPLING_CONFIG_HPP
#define GPU_INTEL_RESAMPLING_CONFIG_HPP

#include "gpu/gpu_resampling_pd.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace resampling {

using fwd_pd_t = gpu_resampling_fwd_pd_t;
using bwd_pd_t = gpu_resampling_bwd_pd_t;
using desc_t = resampling_desc_t;

// Resampling
struct conf_t {
    dim_idx_t ndims;
    offsets_t off;
    dim_t MB, C;
    dim_t ID, IH, IW;
    dim_t OD, OH, OW;
    float FD, FH, FW;
    int vect_size;
    dims_t padded_strides;
    compute::range_t gws = compute::range_t::empty();
    compute::range_t lws = compute::range_t::empty();
    int sub_group_size;
    dim_t padded_c;
    attr_info_t attr_info;
    compute::dispatch_t dispatch;
};

} // namespace resampling
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
