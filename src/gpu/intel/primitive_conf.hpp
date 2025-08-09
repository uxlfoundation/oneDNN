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

#ifndef GPU_INTEL_PRIMITIVE_CONF_HPP
#define GPU_INTEL_PRIMITIVE_CONF_HPP

#include <stdint.h>

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"

#include "gpu/gpu_utils.hpp"

#include "gpu/intel/block_structure.hpp"
#include "gpu/intel/compute/dispatch.hpp"
#include "gpu/intel/compute/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

bool memory_desc_ndims_ok(const memory_desc_t *md);

template <typename T, typename... Rest>
bool memory_desc_ndims_ok(const T *first, const Rest *...rest) {
    return memory_desc_ndims_ok(first) || memory_desc_ndims_ok(rest...);
}

struct memory_desc_info_t {
    // Max levels of blocking
    static const int max_nlevels = 3;

    int ndims;
    data_type_t data_type;

    size_t size;
    dim_t offset0;
    dim_t dims[MAX_NDIMS];
    dim_t padded_dims[MAX_NDIMS];

    int nlevels;
    dim_t blocks[MAX_NDIMS][max_nlevels + 1];
    dim_t strides[MAX_NDIMS][max_nlevels + 1];

    static memory_desc_info_t create(const memory_desc_wrapper &mdw);
};

struct attr_info_t {
    static attr_info_t create(const primitive_attr_t *attr);

    bool initialized = false;

    bool with_binary;
    bool with_eltwise;
    int eltwise_idx;
    int binary_idx;
    alg_kind_t eltwise_alg;
    float eltwise_scale;
    float eltwise_alpha;
    float eltwise_beta;

    bool with_sum;
    int sum_idx;
    float sum_scale;
    data_type_t sum_data_type;

    bool with_src0_scale;
    bool with_src1_scale;
    bool with_src_scales;
    bool with_wei_scales;
    bool with_dst_scales;
    int wei_scales_mask;
    int dst_scales_mask;
    data_type_t src_scales_data_type;
    data_type_t wei_scales_data_type;
    data_type_t dst_scales_data_type;

    bool with_src_zpoints;
    bool with_wei_zpoints;
    bool with_dst_zpoints;
    bool with_per_ic_src_zpoints;
    bool with_per_oc_dst_zpoints;
    data_type_t src_zpoints_data_type;
    data_type_t wei_zpoints_data_type;
    data_type_t dst_zpoints_data_type;
    bool with_dst_sround;
};

template <size_t ndims>
using strides_t = std::array<dim_t, ndims>;
template <>
struct compute::scalar_type_traits_t<strides_t<2>> {
    static const auto type = scalar_type_t::_int64x3_t;
};
template <>
struct compute::scalar_type_traits_t<strides_t<3>> {
    static const auto type = scalar_type_t::_int64x3_t;
};
template <>
struct compute::scalar_type_traits_t<strides_t<4>> {
    static const auto type = scalar_type_t::_int64x4_t;
};
template <>
struct compute::scalar_type_traits_t<strides_t<5>> {
    static const auto type = scalar_type_t::_int64x5_t;
};
template <>
struct compute::scalar_type_traits_t<strides_t<6>> {
    static const auto type = scalar_type_t::_int64x5_t;
};

struct offsets_t {
    dim_t src_off[4][MAX_NDIMS];
    dim_t wei_off[4][MAX_NDIMS];
    dim_t dst_off[4][MAX_NDIMS];
};

// Convolution
enum conv_version_t {
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

struct conv_conf_t {
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

struct quantization_t : public gpu::quantization_t {
public:
    using gpu::quantization_t::quantization_t;

    void define_macros(
            compute::kernel_ctx_t &kernel_ctx, const std::string &name) const;
};

struct sum_quantization_t : public gpu::sum_quantization_t {
public:
    using gpu::sum_quantization_t::sum_quantization_t;

    void define_macros(
            compute::kernel_ctx_t &kernel_ctx, const std::string &name) const;
};

void set_default_conf(conv_conf_t &conf, const convolution_desc_t &cd,
        const memory_desc_t &src_md, const memory_desc_t &weights_md,
        const memory_desc_t &dst_md, const memory_desc_t &bias_md,
        const primitive_attr_t &attr);

void set_offsets(compute::kernel_ctx_t &kernel_ctx,
        const memory_desc_wrapper &md, const char *str);

void set_offsets(const memory_desc_wrapper &md, dim_t offs[4][MAX_NDIMS]);

struct outer_strides_getter_t {
    template <size_t ndims>
    operator strides_t<ndims>() const {
        strides_t<ndims> ret;
        gpu_assert(into<dim_t>(ndims) >= md.ndims());
        for (int d = ndims - 1; d >= 0; d--) {
            // Assumes size 1 dimensions are dense w.r.t. the neighboring dims
            // so they can be used for size calculations in some layouts.
            ret[d] = [&]() {
                if (d >= md.ndims())
                    return static_cast<dim_t>(0);
                else if (md.padded_dims()[d] > 1)
                    return md.strides()[d];
                else if (d == md.ndims() - 1)
                    return static_cast<dim_t>(1);
                else
                    return ret[d + 1] * md.padded_dims()[d + 1];
            }();
        }
        return ret;
    }

    const memory_desc_wrapper &md;
};

outer_strides_getter_t get_outer_strides(const memory_desc_wrapper &md);

block_layout_t get_inner_layout(const memory_desc_wrapper &md);

void def_offsets(const dim_t offs[4][MAX_NDIMS],
        compute::kernel_ctx_t &kernel_ctx, const char *str,
        const dim_idx_t ndims);

void def_block_offsets(const block_layout_t &layout,
        compute::kernel_ctx_t &kernel_ctx, const char *str);

void def_data_type(compute::kernel_ctx_t &kernel_ctx, data_type_t dt,
        const char *str, bool with_punning = true);
void def_data_type(compute::kernel_ctx_t &kernel_ctx, data_type_t dt,
        const std::string &str, bool with_punning = true);

void def_memory_desc_info(compute::kernel_ctx_t &kernel_ctx,
        const memory_desc_info_t &md_info, const char *prefix,
        bool with_punning = true);

void def_binary_alg_kinds(compute::kernel_ctx_t &kernel_ctx);

void def_eltwise_alg_kinds(compute::kernel_ctx_t &kernel_ctx);

bool post_ops_with_binary_ok(const primitive_attr_t *attr,
        const memory_desc_t &dst_md, const int max_ndims_supported = 2);

constexpr int prelu_max_ndims = 5;
status_t get_prelu_md(int prelu_mask, const dim_t *dst_dims,
        memory_desc_t &weight_mem_desc, int weight_ndims);

status_t def_post_ops_cfg(compute::kernel_ctx_t &kernel_ctx,
        const post_ops_t &post_ops, const memory_desc_t &dst_md);

int append_post_ops_to_arg_list_base(const exec_args_t &args,
        compute::kernel_arg_list_t &arg_list, int post_op_idx,
        const post_ops_t &post_ops, memory_desc_wrapper dst_mdw);
int append_post_ops_to_arg_list_gemm(const exec_args_t &args,
        compute::kernel_arg_list_t &arg_list, int post_op_idx,
        const post_ops_t &post_ops, memory_desc_wrapper dst_mdw);
int append_post_ops_to_arg_list(const exec_ctx_t &ctx,
        compute::kernel_arg_list_t &arg_list, int post_op_idx,
        const post_ops_t &post_ops, memory_desc_wrapper dst_mdw);

bool post_ops_preserves_zeroes(
        const exec_ctx_t &ctx, const post_ops_t &post_ops);

status_t def_attr_info_impl(compute::kernel_ctx_t &kernel_ctx,
        const attr_info_t &attr_info, const post_ops_t &post_ops,
        const memory_desc_t &dst_md, bool with_punning = true);

status_t def_attr_info(compute::kernel_ctx_t &kernel_ctx,
        const attr_info_t &attr_info, const post_ops_t &post_ops,
        const memory_desc_t &dst_md, bool with_punning = true);

void def_dispatch(
        compute::kernel_ctx_t &kernel_ctx, const compute::dispatch_t &dispatch);

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
