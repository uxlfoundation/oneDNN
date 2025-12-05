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

#ifndef GPU_INTEL_CONV_CONFIG_HPP
#define GPU_INTEL_CONV_CONFIG_HPP

#include "gpu/gpu_convolution_pd.hpp"
#include "gpu/gpu_deconvolution_pd.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace conv {

using pd_t = convolution_pd_t;
using fwd_pd_t = gpu_convolution_fwd_pd_t;
using bwd_data_pd_t = gpu_convolution_bwd_data_pd_t;
using bwd_weights_pd_t = gpu_convolution_bwd_weights_pd_t;
using desc_t = convolution_desc_t;

enum version_t {
    ver_unused,
    ver_1stconv,
    ver_16mb16c,
    ver_32mb16c,
    ver_32mb32c,
    ver_32c,
    ver_8ow16c,
    ver_nhwc,
    ver_nchw,
    ver_mb_block,
    ver_ow_block,

    // Xe_HP-specific versions.
    ver_v1,
    ver_v2
};

struct conf_t {
    prop_kind_t prop_kind;

    int ndims;
    dim_t mb;
    dim_t ngroups, ic, oc;
    dim_t ngroups_without_padding, oc_without_padding, ic_without_padding;
    dim_t id, ih, iw, od, oh, ow;
    dim_t f_pad, l_pad, t_pad;
    dim_t back_pad, r_pad, b_pad;
    dim_t kd, kh, kw, kwb;
    dim_t stride_d, stride_h, stride_w;
    dim_t dilate_d, dilate_h, dilate_w;

    int oh_block, ow_block;
    int oc_block, ic_block;
    dim_t ocb;
    int mb_block;
    int iw_tail;
    size_t wei_slm_size, src_slm_size, dst_slm_size;
    int sub_group_size;

    compute::range_t gws_d = compute::range_t::empty();
    compute::range_t lws_d = compute::range_t::empty();
    compute::dispatch_t dispatch;

    bool with_bias, with_groups;

    attr_info_t attr_info;

    bool is_depthwise;
    bool is_nhwc;
    bool reorder_wei = false;
    bool reorder_bias = false;
    bool stochastic_round = false;
    int ver;
    format_tag_t src_tag, dst_tag, wei_tag;
    bool is_nchw;
    bool is_src_nchw, is_src_nhwc;
    bool is_dst_nhwc;
    bool require_stateless_addressing;

    int tile_size;
    int wino_m;
    int wino_r;
    dim_t wino_ih, wino_oh;
    dim_t wino_iw, wino_ow;
    dim_t wino_ic;
    dim_t wino_oc;
    int wino_ic_block;
    int wino_oc_block;
    int vect_size;
    compute::range_t U_gws_d = compute::range_t::empty();
    compute::range_t U_lws_d = compute::range_t::empty();
    compute::range_t V_gws_d = compute::range_t::empty();
    compute::range_t V_lws_d = compute::range_t::empty();
    compute::range_t M_gws_d = compute::range_t::empty();
    compute::range_t M_lws_d = compute::range_t::empty();
    bool is_fused;

    data_type_t src_data_type;
    data_type_t weights_data_type;
    data_type_t bias_data_type;
    data_type_t dst_data_type;
    data_type_t acc_data_type;

    memory_desc_info_t src_md_info;
    memory_desc_info_t wei_md_info;
    memory_desc_info_t dst_md_info;
};

void set_default_conf(conf_t &conf, const desc_t &cd,
        const memory_desc_t &src_md, const memory_desc_t &weights_md,
        const memory_desc_t &dst_md, const memory_desc_t &bias_md,
        const primitive_attr_t &attr);

} // namespace conv

namespace deconv {

using pd_t = deconvolution_pd_t;
using bwd_weights_pd_t = gpu_deconvolution_bwd_weights_pd_t;
using desc_t = deconvolution_desc_t;

} // namespace deconv
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
