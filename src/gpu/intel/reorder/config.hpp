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

#include "gpu/gpu_reorder_pd.hpp"
#include "gpu/intel/primitive_conf.hpp"
#include "gpu/intel/utils.hpp"

#ifndef GPU_INTEL_REORDER_CONFIG_HPP
#define GPU_INTEL_REORDER_CONFIG_HPP

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace reorder {

using pd_t = gpu_reorder_pd_t;

// Reorder
enum class custom_kernel_t {
    none,
    dense_vector,
    unroll_16b,
    unroll_16b16c,
    unroll_16a16b,
    plain_to_ABcd84a42b,
    vectorize_last_dim,
    plain_to_ABxx8ayb,
    plain_xFxE_to_abcdef,
    transpose8x8,
    transpose16x16,
    nchw,
    unaligned_sizes,
    alt,
    vectorize_groups,
    pad_innermost,
    xb_to_xab_xba
};

struct block_desc_t {
    dim_idx_t dim_idx;
    int blk_size;
    int step_size;
};

#define LOOP_NEST_LEVEL 4
struct vectorize_last_dim_t {
    dim_idx_t vector_dim;
    int rescale_coeff;
    // composition of data within 16-item packet
    block_desc_t src_vct[LOOP_NEST_LEVEL];
    block_desc_t dst_vct[LOOP_NEST_LEVEL];
    // dimensions to loop over when accessing packets defined above
    block_desc_t src_blk[LOOP_NEST_LEVEL];
    block_desc_t dst_blk[LOOP_NEST_LEVEL];
    int src_blk_limits[MAX_NDIMS];
    int dst_blk_limits[MAX_NDIMS];
    int src_vect_limit;
    int dst_vect_limit;
};

struct vectorize_group_t {
    dim_idx_t vector_dim;
    dim_idx_t src_loop_dim;
    dim_idx_t dst_loop_dim;
    int group_size;
    int innermost_size;
};

struct xb_to_xab_xba_t {
    int vd;
    dim_t blk_size;
    dim_idx_t src_blk_dim;
    dim_t src_blk_coeff;
    dim_idx_t dst_blk_dim;
    dim_t dst_blk_coeff;
};

union implementation_t {
    vectorize_group_t vg;
    xb_to_xab_xba_t ab;
    vectorize_last_dim_t vld;
};

struct conf_t {
    bool has_padding;

    quantization_t src_quant, dst_quant;
    sum_quantization_t sum_quant;

    custom_kernel_t implementation;
    int ndims;
    size_t nelems;
    bool subbyte_pack = false;

    compute::dispatch_t dispatch;

    int sub_group_size;
    memory_desc_info_t src_md_info;
    memory_desc_info_t dst_md_info;

    implementation_t aux_data;
};

} // namespace reorder
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
