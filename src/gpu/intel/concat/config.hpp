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

#ifndef GPU_INTEL_CONCAT_CONFIG_HPP
#define GPU_INTEL_CONCAT_CONFIG_HPP

#include "gpu/gpu_concat_pd.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace concat {

using pd_t = gpu_concat_pd_t;

// Concat
struct conf_t {
    dim_t dst_extern_dim_size;
    dim_t src_extern_dim_sizes[64];
    dim_t offset[64];
    dim_t padded_offset[64];
    dim_t n_blocks;
    dim_t blocks[6];
    dim_t strides[6];
    dim_t inner_axis;
    dim_t dst_concat_axis;
    dim_t dst_padded_concat_axis;
    dim_t dst_offset0;
    dim_t read_block;
    dim_t write_block;
    dim_t gws0_block;
    dim_t read_overlap;
    int n;
    int simd;
    int data_type_size;
    compute::range_t gws_d = compute::range_t::one();
    compute::range_t lws_d;

    data_type_t src_type, dst_type;
    compute::dispatch_t dispatch;
    int ndims;
    memory_desc_info_t src_md_infos[16]; // simple concat does not use this
    memory_desc_info_t dst_md_info;
    int concat_axis;
    int sub_group_size;
    int iter_dim_idx, iter_dim_chunk;
    scales_query_t scale_src[64];
    uint64_t scales_mask;
    bool use_large_index = true;
    bool require_stateless_addressing = true;
};

} // namespace concat
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_CONCAT_CONFIG_HPP
