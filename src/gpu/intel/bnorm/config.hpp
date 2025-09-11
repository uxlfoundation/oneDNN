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

#ifndef GPU_INTEL_BNORM_CONFIG_HPP
#define GPU_INTEL_BNORM_CONFIG_HPP

#include "common/batch_normalization_pd.hpp"
#include "common/c_types_map.hpp"
#include "common/opdesc.hpp"
#include "gpu/gpu_batch_normalization_pd.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace bnorm {

using desc_t = batch_normalization_desc_t;
using pd_t = batch_normalization_pd_t;
using fwd_pd_t = gpu_batch_normalization_fwd_pd_t;
using bwd_pd_t = gpu_batch_normalization_bwd_pd_t;

enum class impl_t {
    unknown = 0,
    ref,
    simple,
    reusable,
    xe,
    nhwc_opt,
    nhwc_reusable
};

struct conf_t {
    data_type_t data_type;
    size_t elsz;
    dim_idx_t ndims;
    dim_t mb, ic, id, ih, iw;
    int mb_block;
    dim_idx_t reduce_dim_idx;
    dim_t reduce_dim;
    dim_t nn, sp, sp_tail;
    int vect_size;
    dim_t stat_sp_nblocks, stat_sp_tail;
    dim_t update_sp_nblocks, update_sp_tail;
    dim_t reduce_stat_nblocks;
    bool with_relu;
    dim_t stat_ic;
    bool is_forward, is_backward;
    bool use_scale, use_shift, save_stats, is_training;
    bool calculate_stats, calculate_diff_stats;
    bool fuse_norm_relu, fuse_norm_add_relu;
    bool diff_scale, diff_shift;
    float relu_negative_slope, eps;
    int sub_group_size;
    bool skip_reduce_stat;
    bool use_stats_one_pass;
    dim_t calc_stat_ic;
    int max_ic_block;
    impl_t impl = impl_t::unknown;
};

} // namespace bnorm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
