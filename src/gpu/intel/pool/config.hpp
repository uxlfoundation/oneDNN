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

#ifndef GPU_INTEL_POOL_CONFIG_HPP
#define GPU_INTEL_POOL_CONFIG_HPP

#include "gpu/gpu_pooling_pd.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace pool {

using pd_t = pooling_pd_t;
using fwd_pd_t = gpu_pooling_fwd_pd_t;
using bwd_pd_t = gpu_pooling_bwd_pd_t;
using desc_t = pooling_desc_t;

struct conf_t {
    int ndims;
    dim_t mb, c;
    dim_t mb_padded;
    dim_t c_padded;
    dim_t id, ih, iw, od, oh, ow;
    dim_t stride_d, stride_h, stride_w;
    dim_t kd, kh, kw;
    dim_t dd, dh, dw;
    dim_t f_pad, t_pad, l_pad;
    data_type_t src_dt;
    data_type_t dst_dt;
    alg_kind_t alg;
    bool is_plain;
    bool is_training, is_backward;
    bool use_mb_c_block, use_only_c_block;
    int unroll_mb_count = 1;
    bool vectorize = true;
    bool require_stateless_addressing;
    int chunks_per_c_block, chunks_per_mb_block;
    int vect_dt_n;
    int nvect;
    compute::dispatch_t dispatch;
    int sub_group_size;
    dim_t global_spatial_chunk;
    dim_t num_batches = 1;
    int mb_block_size = 16;

    attr_info_t attr_info;
    memory_desc_info_t src_md_info;
    memory_desc_info_t dst_md_info;
};

void set_default_conf(conf_t &conf, const desc_t &desc,
        const memory_desc_t &src_md, const memory_desc_t &dst_md,
        const primitive_attr_t &attr);

} // namespace pool
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
