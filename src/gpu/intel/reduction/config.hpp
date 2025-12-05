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

#ifndef GPU_INTEL_REDUCTION_CONFIG_HPP
#define GPU_INTEL_REDUCTION_CONFIG_HPP

#include "gpu/gpu_reduction_pd.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace reduction {

using pd_t = gpu_reduction_pd_t;

struct conf_t {
    // Used by reference implementation
    alg_kind_t alg;
    int ndims, div;
    float eps, power;
    dim_t src_dims[MAX_NDIMS], reduce_dims[MAX_NDIMS], dst_dims[MAX_NDIMS];
    bool is_reduction_dim[MAX_NDIMS];
    int hwd_reduction_size, hwd_size;
    data_type_t src_type, dst_type;
    memory_desc_info_t src_md_info, dst_md_info;
    compute::dispatch_t dispatch;
    offsets_t off;
    attr_info_t attr_info;
    bool require_stateless_addressing;
};

} // namespace reduction
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
