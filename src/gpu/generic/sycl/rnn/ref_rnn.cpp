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

// General architecture
//
// to propagate to the previous iteration and 1 states to propagate
// to the previous layer
// index 0 is dh for cell(t-1, l) to consume
// index 1 is dc for cell(t-1, l) to consume
// index 2 is dh for cell(t, l-1) to consume
// this indexing enables to have the same indexing for states in elemwise
// function
// only the cell execution function should be impacted

#include "gpu/generic/sycl/rnn/ref_rnn.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc.hpp"

#include "common/matmul_pd.hpp"
#include "common/stream.hpp"
#include "common/type_helpers.hpp"
#include "gpu/gpu_stream.hpp"
#include "xpu/sycl/types.hpp"

#include <memory>

#define DPRINT(fmt, ...) \
    do { \
        if (get_verbose_dev_mode(verbose_t::debuginfo) >= 2) { \
            printf(fmt, __VA_ARGS__); \
            fflush(nullptr); \
        } \
    } while (0)
namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::math;
using namespace prop_kind;
using namespace alg_kind;
using namespace rnn_utils;
using namespace dnnl::impl::memory_tracking::names;

status_t ref_rnn_fwd_t::pd_t::set_default_params() {
    using namespace format_tag;
    if (src_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(src_layer_md_, tnc));
    if (dst_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(dst_layer_md_, tnc));

    // Optional parameters
    if ((!types::is_zero_md(&src_iter_md_))
            && (src_iter_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(src_iter_md_, ldnc));
    if ((!types::is_zero_md(&bias_md_))
            && (bias_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(bias_md_, ldgo));
    if ((!types::is_zero_md(&dst_iter_md_))
            && (dst_iter_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(dst_iter_md_, ldnc));

    return status::success;
}

status_t ref_rnn_bwd_t::pd_t::set_default_params() {
    using namespace format_tag;
    if (src_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(src_layer_md_, tnc));
    if (weights_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(weights_layer_md_, ldgoi));

    if (dst_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(dst_layer_md_, tnc));

    if (weights_iter_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(weights_iter_md_, ldgoi));

    if (diff_src_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(diff_src_layer_md_, tnc));
    if (diff_weights_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(diff_weights_layer_md_, ldigo));

    if (diff_weights_iter_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(diff_weights_iter_md_, ldigo));

    if (diff_dst_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(diff_dst_layer_md_, tnc));

    // Optional parameters
    if ((!types::is_zero_md(&src_iter_md_))
            && (src_iter_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(src_iter_md_, ldnc));
    if ((!types::is_zero_md(&src_iter_c_md_))
            && (src_iter_c_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(src_iter_c_md_, ldnc));
    if ((!types::is_zero_md(&bias_md_))
            && (bias_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(bias_md_, ldgo));
    if ((!types::is_zero_md(&dst_iter_md_))
            && (dst_iter_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(dst_iter_md_, ldnc));
    if ((!types::is_zero_md(&dst_iter_c_md_))
            && (dst_iter_c_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(dst_iter_c_md_, ldnc));

    if ((!types::is_zero_md(&diff_src_iter_md_))
            && (diff_src_iter_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(diff_src_iter_md_, ldnc));
    if ((!types::is_zero_md(&diff_src_iter_c_md_))
            && (diff_src_iter_c_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(diff_src_iter_c_md_, ldnc));
    if ((!types::is_zero_md(&diff_bias_md_))
            && (diff_bias_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(diff_bias_md_, ldgo));
    if ((!types::is_zero_md(&diff_dst_iter_md_))
            && (diff_dst_iter_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(diff_dst_iter_md_, ldnc));
    if ((!types::is_zero_md(&diff_dst_iter_c_md_))
            && (diff_dst_iter_c_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(diff_dst_iter_c_md_, ldnc));

    return status::success;
}

// The inputs of create_matmul_pd describe a matmul in column major.
// Below, we have to transpose the a and b descriptor to describe
// the matmul as a row major problem.
status_t create_matmul_pd(impl::engine_t *engine,
        std::shared_ptr<primitive_desc_t> &matmul_pd, dim_t m, dim_t n, dim_t k,
        std::pair<dim_t, dim_t> a_strides, std::pair<dim_t, dim_t> b_strides,
        std::pair<dim_t, dim_t> c_strides, data_type_t a_dt, data_type_t b_dt,
        data_type_t c_dt, float beta, fpmath_mode_t fpmath_mode,
        bool deterministic) {
    memory_desc_t a_md, b_md, c_md, bias_md;

    dims_t a_dims = {n, k}, b_dims = {k, m}, c_dims = {n, m};

    dims_t b_strides_md = {b_strides.first, b_strides.second};
    dims_t a_strides_md = {a_strides.first, a_strides.second};
    dims_t c_strides_md = {c_strides.first, c_strides.second};

    CHECK(memory_desc_init_by_strides(b_md, 2, b_dims, a_dt, b_strides_md));
    CHECK(memory_desc_init_by_strides(a_md, 2, a_dims, b_dt, a_strides_md));
    CHECK(memory_desc_init_by_strides(c_md, 2, c_dims, c_dt, c_strides_md));

    primitive_attr_t attr;
    if (beta != 0) { CHECK(attr.post_ops_.append_sum(beta)); }
    CHECK(attr.set_fpmath_mode(fpmath_mode));
    attr.deterministic_ = deterministic;

    matmul_desc_t matmul_desc;
    dnnl::impl::matmul_desc_init(&matmul_desc, &a_md, &b_md, &bias_md, &c_md);

    primitive_desc_iterator_t it(engine,
            reinterpret_cast<op_desc_t *>(&matmul_desc), &attr, nullptr);

    while (++it != it.end()) {
        if (*it) {
            matmul_pd = *it;
            return status::success;
            break;
        }
    }
    return status::unimplemented;
};

status_t ref_rnn_fwd_t::pd_t::init(impl::engine_t *engine) {
    using namespace prop_kind;
    using namespace utils;
    using namespace rnn_utils;
    using namespace format_tag;

    assert(engine->kind() == engine_kind::gpu);

    const alg_kind_t cell_kind = this->desc()->cell_kind;

    data_type_t src_layer_dt = this->desc()->src_layer_desc.data_type;
    data_type_t weights_iter_dt = this->desc()->weights_iter_desc.data_type;
    data_type_t weights_layer_dt = this->desc()->weights_layer_desc.data_type;
    data_type_t bias_dt = this->desc()->bias_desc.data_type;

    acc_data_t = data_type::f32;

    src_type = src_layer_dt;
    weights_type = weights_layer_dt;

    VDISPATCH_RNN(
            one_of(cell_kind, alg_kind::vanilla_rnn), VERBOSE_BAD_ALGORITHM);
    VDISPATCH_RNN(weights_iter_dt == weights_layer_dt, VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_RNN_SC(this->set_default_params(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_RNN(this->with_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);
    VDISPATCH_RNN(IMPLICATION(this->desc()->prop_kind != forward_inference,
                          bias_dt == dnnl_f32),
            VERBOSE_UNSUPPORTED_BIAS_CFG);

    init_rnn_conf(rnn_conf, this, acc_data_t);

    // Check that only supported attr have been passed.
    primitive_attr_t::skip_mask_t attr_mask
            = primitive_attr_t::skip_mask_t::rnn_tparams;
    if (weights_layer_dt == data_type::s8) {
        attr_mask = attr_mask | primitive_attr_t::skip_mask_t::rnn_data_qparams
                | primitive_attr_t::skip_mask_t::rnn_weights_qparams
                | primitive_attr_t::skip_mask_t::fpmath_mode;
    }
    VDISPATCH_RNN(this->attr()->has_default_values(attr_mask),
            VERBOSE_UNSUPPORTED_ATTR);

    // Set weights descriptors to desired format
    VDISPATCH_RNN_SC(set_weights_desc(this->weights_layer_md_, rnn_conf),
            "unsupported weights layer memory descriptor");
    VDISPATCH_RNN_SC(set_weights_desc(this->weights_iter_md_, rnn_conf),
            "unsupported weights iter memory descriptor");

    // Check dimensions consistency
    VDISPATCH_RNN((this->SIC() == this->DHC() || (this->T() == 1)),
            VERBOSE_INCONSISTENT_DIM, "SIC", (int)this->SIC(), "DHC",
            (int)this->DHC());

    set_rnn_conf(rnn_conf, *this->desc());

    dim_t workspace_size = get_workspace_size(rnn_conf);

    // initialize the workspace_pd if needed
    if (rnn_conf.use_workspace) {
        dims_t ws_dims = {workspace_size};
        VDISPATCH_RNN_SC(memory_desc_init_by_tag(
                                 this->ws_md_, 1, ws_dims, data_type::u8, x),
                "memory_desc_init_by_tag()");
    }

    memory_desc_t state_md;
    dims_t state_dims = {rnn_conf.n_layer, rnn_conf.n_dir, rnn_conf.n_iter + 1,
            rnn_conf.mb, rnn_conf.states_ws_ld};

    CHECK(memory_desc_init_by_tag(state_md, 5, state_dims,
            rnn_conf.aux_data_type, format_tag::abcde));

    // using is_l2r/r2l to account for bidirectional as well
    // if both l2r and r2l are true, case is bidirectional concat
    // is_sum changes behaviour from concat to sum for bidirectional case

    bool is_l2r = !(this->desc()->direction == dnnl_unidirectional_right2left);
    bool is_r2l = !(this->desc()->direction == dnnl_unidirectional_left2right);
    bool is_sum = this->desc()->direction == dnnl_bidirectional_sum;

    xpu::sycl::md_t copy_src_md = xpu::sycl::md_t(this->src_md(0));
    xpu::sycl::md_t copy_dst_md = xpu::sycl::md_t(&state_md);

    copy_init_layer_conf_ = sycl_rnn_copy_conf_t {copy_src_md, copy_dst_md,
            rnn_conf.is_fwd ? rnn_conf.slc : rnn_conf.dhc, rnn_conf.n_dir,
            rnn_conf.n_layer, rnn_conf.n_iter, rnn_conf.mb,
            rnn_conf.is_fwd ? rnn_conf.states_ws_ld
                            : rnn_conf.scratch_diff_states_ld,
            true, true, is_l2r, is_r2l, is_sum};

    copy_src_md = this->src_md(1)->data_type == data_type::undef
            ? xpu::sycl::md_t()
            : xpu::sycl::md_t(this->src_md(1));

    copy_init_iter_conf_ = sycl_rnn_copy_conf_t {copy_src_md, copy_dst_md,
            rnn_conf.is_fwd ? rnn_conf.sic : rnn_conf.dhc, rnn_conf.n_dir,
            rnn_conf.n_layer, rnn_conf.n_iter, rnn_conf.mb,
            rnn_conf.is_fwd ? rnn_conf.states_ws_ld
                            : rnn_conf.scratch_diff_states_ld,
            false, true, is_l2r, is_r2l, is_sum};

    copy_src_md = xpu::sycl::md_t(&state_md);
    copy_dst_md = xpu::sycl::md_t(this->dst_md(0));

    copy_res_layer_conf_ = sycl_rnn_copy_conf_t {copy_src_md, copy_dst_md,
            rnn_conf.is_fwd ? rnn_conf.dhc : rnn_conf.slc, rnn_conf.n_dir,
            rnn_conf.n_layer, rnn_conf.n_iter, rnn_conf.mb,
            rnn_conf.is_fwd ? rnn_conf.states_ws_ld
                            : rnn_conf.scratch_diff_states_ld,
            true, false, is_l2r, is_r2l, is_sum};

    copy_src_md = xpu::sycl::md_t(&state_md);
    copy_dst_md = this->dst_md(1)->data_type == data_type::undef
            ? xpu::sycl::md_t()
            : xpu::sycl::md_t(this->dst_md(1));

    copy_res_iter_conf_ = sycl_rnn_copy_conf_t {copy_src_md, copy_dst_md,
            rnn_conf.is_fwd ? rnn_conf.dhc : rnn_conf.sic, rnn_conf.n_dir,
            rnn_conf.n_layer, rnn_conf.n_iter, rnn_conf.mb,
            rnn_conf.is_fwd ? rnn_conf.states_ws_ld
                            : rnn_conf.scratch_diff_states_ld,
            false, false, is_l2r, is_r2l, is_sum};

    sycl_rnn_bias_fwd_conf_t_ = sycl_rnn_bias_fwd_conf_t();
    sycl_rnn_bias_fwd_conf_t_.gates_type = rnn_conf.ws_data_type;
    sycl_rnn_bias_fwd_conf_t_.states_data_type
            = xpu::sycl::md_t(&state_md).data_type();
    sycl_rnn_bias_fwd_conf_t_.bias_type = bias_dt;
    sycl_rnn_bias_fwd_conf_t_.batch = rnn_conf.mb;
    sycl_rnn_bias_fwd_conf_t_.dhc = rnn_conf.dhc;
    sycl_rnn_bias_fwd_conf_t_.gates_ws_ld = rnn_conf.gates_ws_ld;
    sycl_rnn_bias_fwd_conf_t_.states_ws_ld = rnn_conf.states_ws_ld;
    sycl_rnn_bias_fwd_conf_t_.activation_kind = this->activation_kind();
    sycl_rnn_bias_fwd_conf_t_.alpha = this->desc()->alpha;

    auto fpmath_mode = this->attr()->fpmath_.mode_;
    bool deterministic = this->attr()->deterministic_;

    float matmul_iter_fwd_beta = this->is_lbr() ? 0.0f : 1.0f;

    // Setup matmul PDs
    dim_t batch = rnn_conf.mb;
    dim_t n_gates = rnn_conf.n_gates;
    dim_t slc = rnn_conf.slc;
    dim_t sic = rnn_conf.sic;
    dim_t dhc = rnn_conf.dhc;

    dims_t wei_layer_strides;
    get_outer_strides(this->weights_md(0), wei_layer_strides);
    dims_t wei_iter_strides;
    get_outer_strides(this->weights_md(1), wei_iter_strides);

    VDISPATCH_RNN_SC(
            create_matmul_pd(engine, matmul_layer_fwd_pd_, n_gates * dhc, batch,
                    slc, {rnn_conf.states_ws_ld, 1},
                    {wei_layer_strides[2], wei_layer_strides[4]},
                    {rnn_conf.scratch_gates_ld, 1}, weights_type,
                    rnn_conf.aux_data_type, rnn_conf.aux_data_type, 0.0,
                    fpmath_mode, deterministic),
            "create_matmul_pd(matmul_layer_fwd_pd_)");

    VDISPATCH_RNN_SC(
            create_matmul_pd(engine, matmul_iter_fwd_pd_, n_gates * dhc, batch,
                    sic, {rnn_conf.states_ws_ld, 1},
                    {wei_iter_strides[2], wei_iter_strides[4]},
                    {rnn_conf.gates_ws_ld, 1}, weights_type,
                    rnn_conf.aux_data_type, rnn_conf.aux_data_type,
                    matmul_iter_fwd_beta, fpmath_mode, deterministic),
            "create_matmul_pd(matmul_iter_fwd_pd_)");

    init_scratchpad(rnn_conf.use_workspace ? 0 : workspace_size);
    return status::success;
}

status_t ref_rnn_bwd_t::pd_t::init(impl::engine_t *engine) {
    using namespace prop_kind;
    using namespace utils;
    using namespace rnn_utils;
    using namespace format_tag;

    assert(engine->kind() == engine_kind::gpu);

    const alg_kind_t cell_kind = this->desc()->cell_kind;

    data_type_t src_layer_dt = this->desc()->src_layer_desc.data_type;
    data_type_t weights_iter_dt = this->desc()->weights_iter_desc.data_type;
    data_type_t weights_layer_dt = this->desc()->weights_layer_desc.data_type;
    data_type_t bias_dt = this->desc()->bias_desc.data_type;

    acc_data_t = data_type::f32;

    src_type = src_layer_dt;
    weights_type = weights_layer_dt;

    VDISPATCH_RNN(
            one_of(cell_kind, alg_kind::vanilla_rnn), VERBOSE_BAD_ALGORITHM);
    VDISPATCH_RNN(weights_iter_dt == weights_layer_dt, VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_RNN_SC(this->set_default_params(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_RNN(this->with_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);
    VDISPATCH_RNN(IMPLICATION(this->desc()->prop_kind != forward_inference,
                          utils::one_of(bias_dt, dnnl_f32)),
            VERBOSE_UNSUPPORTED_BIAS_CFG);

    init_rnn_conf(rnn_conf, this, acc_data_t);

    // Check that only supported attr have been passed.
    primitive_attr_t::skip_mask_t attr_mask
            = primitive_attr_t::skip_mask_t::rnn_tparams;
    if (weights_layer_dt == data_type::s8) {
        attr_mask = attr_mask | primitive_attr_t::skip_mask_t::rnn_data_qparams
                | primitive_attr_t::skip_mask_t::rnn_weights_qparams
                | primitive_attr_t::skip_mask_t::fpmath_mode;
    }
    VDISPATCH_RNN(this->attr()->has_default_values(attr_mask),
            VERBOSE_UNSUPPORTED_ATTR);

    // Set weights descriptors to desired format
    VDISPATCH_RNN_SC(set_weights_desc(this->weights_layer_md_, rnn_conf),
            "unsupported weights layer memory descriptor");
    VDISPATCH_RNN_SC(set_weights_desc(this->weights_iter_md_, rnn_conf),
            "unsupported weights iter memory descriptor");

    // Check dimensions consistency
    VDISPATCH_RNN((this->SIC() == this->DHC() || (this->T() == 1)),
            VERBOSE_INCONSISTENT_DIM, "SIC", (int)this->SIC(), "DHC",
            (int)this->DHC());

    set_rnn_conf(rnn_conf, *this->desc());

    dim_t workspace_size = get_workspace_size(rnn_conf);

    // initialize the workspace_pd if needed
    if (rnn_conf.use_workspace) {
        dims_t ws_dims = {workspace_size};
        VDISPATCH_RNN_SC(memory_desc_init_by_tag(
                                 this->ws_md_, 1, ws_dims, data_type::u8, x),
                "memory_desc_init_by_tag()");
    }

    memory_desc_t state_md;

    dims_t state_dims = {rnn_conf.n_layer, rnn_conf.n_dir, rnn_conf.n_iter + 1,
            rnn_conf.n_states + 1, rnn_conf.mb,
            rnn_conf.scratch_diff_states_ld};

    CHECK(memory_desc_init_by_tag(state_md, 6, state_dims,
            rnn_conf.aux_data_type, format_tag::abcdef));

    // using is_l2r/r2l to account for bidirectional as well
    // if both l2r and r2l are true, case is bidirectional concat
    // is_sum changes behaviour from concat to sum for bidirectional case

    bool is_l2r = !(this->desc()->direction == dnnl_unidirectional_right2left);
    bool is_r2l = !(this->desc()->direction == dnnl_unidirectional_left2right);
    bool is_sum = this->desc()->direction == dnnl_bidirectional_sum;

    xpu::sycl::md_t copy_src_md = xpu::sycl::md_t(this->diff_dst_md(0));
    xpu::sycl::md_t copy_dst_md = xpu::sycl::md_t(&state_md);

    copy_init_layer_conf_ = sycl_rnn_copy_conf_t {copy_src_md, copy_dst_md,
            rnn_conf.dhc, rnn_conf.n_dir, rnn_conf.n_layer, rnn_conf.n_iter,
            rnn_conf.mb, rnn_conf.scratch_diff_states_ld, true, true, is_l2r,
            is_r2l, is_sum};

    copy_src_md = this->diff_dst_md(1);

    copy_init_iter_conf_ = sycl_rnn_copy_conf_t {copy_src_md, copy_dst_md,
            rnn_conf.dhc, rnn_conf.n_dir, rnn_conf.n_layer, rnn_conf.n_iter,
            rnn_conf.mb, rnn_conf.scratch_diff_states_ld, false, true, is_l2r,
            is_r2l, is_sum};

    copy_src_md = xpu::sycl::md_t(&state_md);
    copy_dst_md = xpu::sycl::md_t(this->diff_src_md(0));

    copy_res_layer_conf_ = sycl_rnn_copy_conf_t {copy_src_md, copy_dst_md,
            rnn_conf.slc, rnn_conf.n_dir, rnn_conf.n_layer, rnn_conf.n_iter,
            rnn_conf.mb, rnn_conf.scratch_diff_states_ld, true, false, is_l2r,
            is_r2l, is_sum};

    copy_src_md = xpu::sycl::md_t(&state_md);
    copy_dst_md = this->diff_src_md(1);

    copy_res_iter_conf_ = sycl_rnn_copy_conf_t {copy_src_md, copy_dst_md,
            rnn_conf.sic, rnn_conf.n_dir, rnn_conf.n_layer, rnn_conf.n_iter,
            rnn_conf.mb, rnn_conf.scratch_diff_states_ld, false, false, is_l2r,
            is_r2l, is_sum};

    sycl_rnn_bias_bwd_conf_t_ = sycl_rnn_bias_bwd_conf_t();
    sycl_rnn_bias_bwd_conf_t_.diff_states_type = rnn_conf.diff_data_type;
    sycl_rnn_bias_bwd_conf_t_.diff_gates_type = rnn_conf.ws_data_type;
    sycl_rnn_bias_bwd_conf_t_.gates_type = rnn_conf.ws_data_type;
    sycl_rnn_bias_bwd_conf_t_.batch = rnn_conf.mb;
    sycl_rnn_bias_bwd_conf_t_.dhc = rnn_conf.dhc;
    sycl_rnn_bias_bwd_conf_t_.gates_ws_ld = rnn_conf.scratch_diff_gates_ld;
    sycl_rnn_bias_bwd_conf_t_.states_ws_ld = rnn_conf.states_ws_ld;
    sycl_rnn_bias_bwd_conf_t_.scratch_diff_states_ld
            = rnn_conf.scratch_diff_states_ld;
    sycl_rnn_bias_bwd_conf_t_.activation_kind = this->activation_kind();
    sycl_rnn_bias_bwd_conf_t_.alpha = this->desc()->alpha;

    auto fpmath_mode = this->attr()->fpmath_.mode_;
    bool deterministic = this->attr()->deterministic_;

    float matmul_iter_bwd_beta = this->is_lbr() ? 1.0f : 0.0f;

    // Setup matmul PDs
    auto diff_type = this->diff_src_md(0)->data_type;

    dim_t batch = rnn_conf.mb;
    dim_t n_gates = rnn_conf.n_gates;
    dim_t slc = rnn_conf.slc;
    dim_t sic = rnn_conf.sic;
    dim_t dhc = rnn_conf.dhc;

    dims_t wei_layer_strides;
    get_outer_strides(this->weights_md(0), wei_layer_strides);
    dims_t wei_iter_strides;
    get_outer_strides(this->weights_md(1), wei_iter_strides);
    dims_t diff_weights_layer;
    get_outer_strides(this->diff_weights_md(0), diff_weights_layer);
    dims_t diff_weights_iter;
    get_outer_strides(this->diff_weights_md(1), diff_weights_iter);

    VDISPATCH_RNN_SC(create_matmul_pd(engine, matmul_iter_bwd_pd_, sic, batch,
                             n_gates * dhc, {rnn_conf.scratch_diff_gates_ld, 1},
                             {wei_iter_strides[4], wei_iter_strides[2]},
                             {rnn_conf.scratch_diff_states_ld, 1}, weights_type,
                             diff_type, diff_type, matmul_iter_bwd_beta,
                             fpmath_mode, deterministic),
            "create_matmul_pd(matmul_iter_bwd_pd_)");
    VDISPATCH_RNN_SC(
            create_matmul_pd(engine, matmul_diff_wei_iter_pd_, n_gates * dhc,
                    sic, batch, {1, rnn_conf.states_ws_ld},
                    {rnn_conf.scratch_diff_gates_ld, 1},
                    {diff_weights_iter[2], diff_weights_iter[4]}, diff_type,
                    diff_type, diff_type, 1.0f, fpmath_mode, deterministic),
            "create_matmul_pd(matmul_diff_wei_iter_pd_)");
    VDISPATCH_RNN_SC(
            create_matmul_pd(engine, matmul_layer_bwd_pd_, slc, batch,
                    n_gates * dhc, {rnn_conf.scratch_diff_gates_ld, 1},
                    {wei_layer_strides[4], wei_layer_strides[2]},
                    {rnn_conf.scratch_diff_states_ld, 1}, weights_type,
                    diff_type, diff_type, 0.0f, fpmath_mode, deterministic),
            "create_matmul_pd(matmul_layer_bwd_pd_)");

    VDISPATCH_RNN_SC(
            create_matmul_pd(engine, matmul_diff_wei_layer_pd_, n_gates * dhc,
                    slc, batch, {1, rnn_conf.states_ws_ld},
                    {rnn_conf.scratch_diff_gates_ld, 1},
                    {diff_weights_layer[2], diff_weights_layer[4]}, diff_type,
                    diff_type, diff_type, 1.0f, fpmath_mode, deterministic),
            "create_matmul_pd(matmul_diff_wei_layer_pd_)");

    init_scratchpad(rnn_conf.use_workspace ? 0 : workspace_size);
    return status::success;
}

bool ref_rnn_common_base_t::create_nested_matmul(impl::engine_t *engine,
        const std::shared_ptr<primitive_desc_t> &prim_desc,
        std::shared_ptr<impl::primitive_t> &prim) {
    std::pair<std::shared_ptr<impl::primitive_t>, cache_state_t> pair;
    bool gemm_ok = prim_desc->create_primitive_nested(pair, engine)
            == status::success;
    prim = pair.first;
    return gemm_ok;
};

status_t ref_rnn_fwd_t::init_(impl::engine_t *engine) {
    using namespace rnn_utils;

    switch (pd()->cell_kind()) {
        case dnnl_vanilla_rnn:
            cell_func = [this](const cell_ctx_t &cell_struct) -> status_t {
                return this->cell_execution(cell_struct);
            };
            break;
        default: break;
    }
    grid_func = [this](const grid_ctx_t &grid_struct) -> status_t {
        return this->linear_execution(grid_struct);
    };

    const conf_t &rnn = pd()->rnn_conf;
    rnn_utils::set_workspace_offsets(rnn, ws_gates_offset_, ws_states_offset_);

    // IMPORTANT SYCL STUFF
    const auto copy_fwd_kid = ::sycl::get_kernel_id<ref_rnn_copy_fwd_t>();
    this->create_kernel(engine, copy_fwd_kid, &copy_fwd_kernel_);
    const auto bias_fwd_kid = ::sycl::get_kernel_id<ref_rnn_bias_fwd>();
    this->create_kernel(engine, bias_fwd_kid, &bias_fwd_kernel_);

    bool matmul_ok = true;

    matmul_ok = matmul_ok
            && create_nested_matmul(
                    engine, pd()->matmul_layer_fwd_pd_, matmul_layer_fwd_);
    matmul_ok = matmul_ok
            && create_nested_matmul(
                    engine, pd()->matmul_iter_fwd_pd_, matmul_iter_fwd_);

    if (!matmul_ok) return status::runtime_error;

    return status::success;
}

status_t ref_rnn_bwd_t::init_(impl::engine_t *engine) {
    using namespace rnn_utils;

    switch (pd()->cell_kind()) {
        case dnnl_vanilla_rnn:
            cell_func = [this](const cell_ctx_t &cell_struct) -> status_t {
                return this->cell_execution(cell_struct);
            };
            break;
        default: break;
    }
    grid_func = [this](const grid_ctx_t &grid_struct) -> status_t {
        return this->linear_execution(grid_struct);
    };

    const conf_t &rnn = pd()->rnn_conf;
    rnn_utils::set_workspace_offsets(rnn, ws_gates_offset_, ws_states_offset_);

    // IMPORTANT SYCL STUFF
    const auto copy_bwd_kid = ::sycl::get_kernel_id<ref_rnn_copy_bwd_t>();
    this->create_kernel(engine, copy_bwd_kid, &copy_bwd_kernel_);
    const auto bias_bwd_kid = ::sycl::get_kernel_id<ref_rnn_bias_bwd>();
    this->create_kernel(engine, bias_bwd_kid, &bias_bwd_kernel_);

    bool matmul_ok = true;

    matmul_ok = matmul_ok
            && create_nested_matmul(
                    engine, pd()->matmul_layer_bwd_pd_, matmul_layer_bwd_);
    matmul_ok = matmul_ok
            && create_nested_matmul(
                    engine, pd()->matmul_iter_bwd_pd_, matmul_iter_bwd_);
    matmul_ok = matmul_ok
            && create_nested_matmul(engine, pd()->matmul_diff_wei_layer_pd_,
                    matmul_diff_wei_layer_);
    matmul_ok = matmul_ok
            && create_nested_matmul(engine, pd()->matmul_diff_wei_iter_pd_,
                    matmul_diff_wei_iter_);

    if (!matmul_ok) return status::runtime_error;

    return status::success;
}

status_t ref_rnn_fwd_t::matmul_primitive(impl::engine_t *engine,
        const exec_ctx_t &ctx, std::unique_ptr<memory_storage_t> &a,
        std::unique_ptr<memory_storage_t> &b,
        std::unique_ptr<memory_storage_t> &c, matmul_kind_t matmul_kind) const {
    std::shared_ptr<impl::primitive_desc_t> matmul_pd;

    switch (matmul_kind) {
        case matmul_iter_fwd: matmul_pd = pd()->matmul_iter_fwd_pd_; break;
        case matmul_layer_fwd: matmul_pd = pd()->matmul_layer_fwd_pd_; break;
        default: assert(!"unknown matmul_kind"); return status::runtime_error;
    }

    std::unique_ptr<memory_t, memory_deleter_t> arg1, arg2, arg3;
    exec_args_t matmul_args;
    CHECK(safe_ptr_assign(arg2,
            new memory_t(
                    ctx.stream()->engine(), matmul_pd->src_md(0), a->clone())));
    CHECK(safe_ptr_assign(arg1,
            new memory_t(ctx.stream()->engine(), matmul_pd->weights_md(0),
                    b->clone())));
    CHECK(safe_ptr_assign(arg3,
            new memory_t(
                    ctx.stream()->engine(), matmul_pd->dst_md(0), c->clone())));

    matmul_args[DNNL_ARG_SRC] = memory_arg_t {arg1.get(), true};
    matmul_args[DNNL_ARG_WEIGHTS] = memory_arg_t {arg2.get(), true};
    matmul_args[DNNL_ARG_DST] = memory_arg_t {arg3.get(), false};

    exec_ctx_t matmul_ctx(ctx, std::move(matmul_args));
    const auto init_matmul_nested_scratchpad
            = [&](const std::shared_ptr<impl::primitive_t> &matmul, int key) {
                  auto *nested_grantor
                          = create_nested_grantor(ctx.get_scratchpad_grantor(),
                                  key, matmul->pd()->scratchpad_registry());
                  matmul_ctx.set_scratchpad_grantor(nested_grantor);
              };

    switch (matmul_kind) {
        case matmul_iter_fwd:
            init_matmul_nested_scratchpad(
                    matmul_iter_fwd_, rnn_utils::scratch_t::key_gemm_iter_fwd);
            CHECK(matmul_iter_fwd_->execute(matmul_ctx));
            break;
        case matmul_layer_fwd:
            init_matmul_nested_scratchpad(matmul_layer_fwd_,
                    rnn_utils::scratch_t::key_gemm_layer_fwd);
            CHECK(matmul_layer_fwd_->execute(matmul_ctx));
            break;
        default: assert(!"unknown matmul_kind"); return status::runtime_error;
    }

    return status::success;
}

status_t ref_rnn_bwd_t::matmul_primitive(impl::engine_t *engine,
        const exec_ctx_t &ctx, std::unique_ptr<memory_storage_t> &a,
        std::unique_ptr<memory_storage_t> &b,
        std::unique_ptr<memory_storage_t> &c, matmul_kind_t matmul_kind) const {
    std::shared_ptr<impl::primitive_desc_t> matmul_pd;

    switch (matmul_kind) {
        case matmul_iter_bwd: matmul_pd = pd()->matmul_iter_bwd_pd_; break;
        case matmul_layer_bwd: matmul_pd = pd()->matmul_layer_bwd_pd_; break;
        case matmul_diff_wei_iter:
            matmul_pd = pd()->matmul_diff_wei_iter_pd_;
            break;
        case matmul_diff_wei_layer:
            matmul_pd = pd()->matmul_diff_wei_layer_pd_;
            break;
        default: assert(!"unknown matmul_kind"); return status::runtime_error;
    }

    std::unique_ptr<memory_t, memory_deleter_t> arg1, arg2, arg3;
    exec_args_t matmul_args;

    CHECK(safe_ptr_assign(arg2,
            new memory_t(
                    ctx.stream()->engine(), matmul_pd->src_md(0), a->clone())));
    CHECK(safe_ptr_assign(arg1,
            new memory_t(ctx.stream()->engine(), matmul_pd->weights_md(0),
                    b->clone())));
    CHECK(safe_ptr_assign(arg3,
            new memory_t(
                    ctx.stream()->engine(), matmul_pd->dst_md(0), c->clone())));

    matmul_args[DNNL_ARG_SRC] = memory_arg_t {arg1.get(), true};
    matmul_args[DNNL_ARG_WEIGHTS] = memory_arg_t {arg2.get(), true};
    matmul_args[DNNL_ARG_DST] = memory_arg_t {arg3.get(), false};

    exec_ctx_t matmul_ctx(ctx, std::move(matmul_args));

    const auto init_matmul_nested_scratchpad
            = [&](const std::shared_ptr<impl::primitive_t> &matmul, int key) {
                  auto *nested_grantor
                          = create_nested_grantor(ctx.get_scratchpad_grantor(),
                                  key, matmul->pd()->scratchpad_registry());
                  matmul_ctx.set_scratchpad_grantor(nested_grantor);
              };

    switch (matmul_kind) {
        case matmul_iter_bwd:
            init_matmul_nested_scratchpad(
                    matmul_iter_bwd_, rnn_utils::scratch_t::key_gemm_iter_bwd);
            CHECK(matmul_iter_bwd_->execute(matmul_ctx));
            break;
        case matmul_layer_bwd:
            init_matmul_nested_scratchpad(matmul_layer_bwd_,
                    rnn_utils::scratch_t::key_gemm_layer_bwd);
            CHECK(matmul_layer_bwd_->execute(matmul_ctx));
            break;
        case matmul_diff_wei_iter:
            init_matmul_nested_scratchpad(matmul_diff_wei_iter_,
                    rnn_utils::scratch_t::key_gemm_diff_wei_iter);
            CHECK(matmul_diff_wei_iter_->execute(matmul_ctx));
            break;
        case matmul_diff_wei_layer:
            init_matmul_nested_scratchpad(matmul_diff_wei_layer_,
                    rnn_utils::scratch_t::key_gemm_diff_wei_layer);
            CHECK(matmul_diff_wei_layer_->execute(matmul_ctx));
            break;

        default: assert(!"unknown matmul_kind"); return status::runtime_error;
    }

    return status::success;
}

//*************** Grid computations strategy: linear ***************//

status_t ref_rnn_common_base_t::execution_loop(const grid_ctx_t &grid_struct) {
    dim_t n_layer = grid_struct.rnn.n_layer;
    dim_t n_dir = grid_struct.rnn.n_dir;
    dim_t n_iter = grid_struct.rnn.n_iter;

    for (dim_t dir = 0; dir < n_dir; dir++) {
        for (dim_t j = 0; j < n_layer; j++) {
            dim_t lay = j;
            for (dim_t i = 0; i < n_iter; i += grid_struct.rnn.iter_loop) {
                dim_t iter = i;
                const cell_ctx_t c_struct
                        = {grid_struct.engine, grid_struct.ctx, dir, lay, iter,
                                grid_struct.user_data, grid_struct.workspace,
                                grid_struct.scratch, grid_struct.rnn};
                CHECK(cell_func(c_struct));
            }
        }
    }
    return status::success;
}

status_t ref_rnn_fwd_t::linear_execution(const grid_ctx_t &grid_struct) {

    CHECK(execution_loop(grid_struct));

    return status::success;
}

status_t ref_rnn_bwd_t::linear_execution(const grid_ctx_t &grid_struct) {

    if (pd()->diff_weights_overwrite()) {
        gpu::stream_t *stream
                = utils::downcast<gpu::stream_t *>(grid_struct.ctx.stream());
        auto zero = [&](const memory_storage_t *data, int arg_id) {
            auto mdw = memory_desc_wrapper(pd()->arg_md(arg_id));
            return stream->fill(*data, 0, mdw.size(), stream->ctx().get_deps(),
                    stream->ctx().get_deps());
        };

        CHECK(zero(grid_struct.user_data.diff_bias(), DNNL_ARG_DIFF_BIAS));
        CHECK(zero(grid_struct.user_data.diff_wei_layer(),
                DNNL_ARG_DIFF_WEIGHTS_LAYER));
        CHECK(zero(grid_struct.user_data.diff_wei_iter(),
                DNNL_ARG_DIFF_WEIGHTS_ITER));
    }

    CHECK(execution_loop(grid_struct));

    return status::success;
}

//********* GRID computations strategy: utility functions **********//

status_t ref_rnn_common_base_t::launch_copy(bool fwd, const exec_ctx_t &ctx,
        const kernel_t &cpy_kernel, const sycl_rnn_copy_conf_t &copy_conf,
        ::sycl::range<3> global_range, ::sycl::range<3> local_range,
        const memory_storage_t &input, const memory_storage_t &output) const {
    parallel_for(ctx, cpy_kernel, [&](::sycl::handler &cgh) {
        auto src_mem_arg = input
                ? utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        &input)
                          ->get_in_memory_arg(ctx.stream(), cgh)
                : xpu::sycl::memory_storage_base_t::empty_in_memory_arg(
                        ctx.stream(), cgh);
        auto dst_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        &output)
                          ->get_out_memory_arg(ctx.stream(), cgh);
        if (fwd) {
            ref_rnn_copy_fwd_t copy_kernel_fwd(
                    copy_conf, src_mem_arg, dst_mem_arg);
            cgh.parallel_for(::sycl::nd_range<3>(global_range, local_range),
                    copy_kernel_fwd);
        } else {
            ref_rnn_copy_bwd_t copy_kernel_bwd(
                    copy_conf, src_mem_arg, dst_mem_arg);
            cgh.parallel_for(::sycl::nd_range<3>(global_range, local_range),
                    copy_kernel_bwd);
        }
    });

    return status::success;
}

status_t ref_rnn_common_base_t::do_copy(bool fwd, const exec_ctx_t &ctx,
        size_t batch_range, size_t lay_iter_range, size_t channel_range,
        const sycl_rnn_copy_conf_t &copy_conf, const kernel_t &cpy_kernel,
        const memory_storage_t &input, const memory_storage_t &output) const {
    auto max_wg_size_per_dim = rnn_utils::calc_local_range(ctx);
    size_t local_batch = max_wg_size_per_dim;
    size_t local_channel = max_wg_size_per_dim;
    size_t local_lay_dir = max_wg_size_per_dim;
    size_t global_batch = rnn_utils::calc_global_range(
            static_cast<size_t>(max_wg_size_per_dim),
            static_cast<size_t>(batch_range));
    size_t global_channels = rnn_utils::calc_global_range(
            static_cast<size_t>(max_wg_size_per_dim), channel_range);
    size_t global_lay_dir = rnn_utils::calc_global_range(
            static_cast<size_t>(max_wg_size_per_dim),
            static_cast<size_t>(lay_iter_range));

    return launch_copy(fwd, ctx, cpy_kernel, copy_conf,
            {global_lay_dir, global_batch, global_channels},
            {local_lay_dir, local_batch, local_channel}, input, output);
}

status_t ref_rnn_fwd_t::copy_init_layer(const exec_ctx_t &ctx, dim_t batch,
        dim_t dhc, dim_t sic, dim_t slc, dim_t n_iter, dim_t n_layer,
        dim_t n_dir, dim_t states_ws_ld, const memory_storage_t &input,
        const memory_storage_t &output) const {
    return do_copy(true, ctx, batch, n_iter, slc, pd()->copy_init_layer_conf_,
            copy_fwd_kernel_, input, output);
}

status_t ref_rnn_bwd_t::copy_init_layer(const exec_ctx_t &ctx, dim_t batch,
        dim_t dhc, dim_t sic, dim_t slc, dim_t n_iter, dim_t n_layer,
        dim_t n_dir, dim_t states_ws_ld, const memory_storage_t &input,
        const memory_storage_t &output) const {
    return do_copy(false, ctx, batch, n_iter, dhc, pd()->copy_init_layer_conf_,
            copy_bwd_kernel_, input, output);
}

status_t ref_rnn_fwd_t::copy_init_iter(const exec_ctx_t &ctx, dim_t batch,
        dim_t dhc, dim_t sic, dim_t slc, dim_t n_iter, dim_t n_layer,
        dim_t n_dir, dim_t states_ws_ld, const memory_storage_t &input,
        const memory_storage_t &output) const {
    return do_copy(true, ctx, batch, n_layer * n_dir,
            std::max(static_cast<size_t>(sic), static_cast<size_t>(dhc)),
            pd()->copy_init_iter_conf_, copy_fwd_kernel_, input, output);
}

status_t ref_rnn_bwd_t::copy_init_iter(const exec_ctx_t &ctx, dim_t batch,
        dim_t dhc, dim_t sic, dim_t slc, dim_t n_iter, dim_t n_layer,
        dim_t n_dir, dim_t states_ws_ld, const memory_storage_t &input,
        const memory_storage_t &output) const {
    return do_copy(false, ctx, batch, n_layer * n_dir, dhc,
            pd()->copy_init_iter_conf_, copy_bwd_kernel_, input, output);
}

status_t ref_rnn_fwd_t::copy_res_layer(const exec_ctx_t &ctx, dim_t batch,
        dim_t dhc, dim_t sic, dim_t slc, dim_t n_iter, dim_t n_layer,
        dim_t n_dir, dim_t states_ws_ld, const memory_storage_t &input,
        const memory_storage_t &output) const {
    return do_copy(true, ctx, batch, n_iter, dhc, pd()->copy_res_layer_conf_,
            copy_fwd_kernel_, input, output);
}

status_t ref_rnn_bwd_t::copy_res_layer(const exec_ctx_t &ctx, dim_t batch,
        dim_t dhc, dim_t sic, dim_t slc, dim_t n_iter, dim_t n_layer,
        dim_t n_dir, dim_t states_ws_ld, const memory_storage_t &input,
        const memory_storage_t &output) const {
    return do_copy(false, ctx, batch, n_iter, slc, pd()->copy_res_layer_conf_,
            copy_bwd_kernel_, input, output);
}

status_t ref_rnn_fwd_t::copy_res_iter(const exec_ctx_t &ctx, dim_t batch,
        dim_t dhc, dim_t sic, dim_t slc, dim_t n_iter, dim_t n_layer,
        dim_t n_dir, dim_t states_ws_ld, const memory_storage_t &input,
        const memory_storage_t &output) const {
    return do_copy(true, ctx, batch, n_layer * n_dir, dhc,
            pd()->copy_res_iter_conf_, copy_fwd_kernel_, input, output);
}

status_t ref_rnn_bwd_t::copy_res_iter(const exec_ctx_t &ctx, dim_t batch,
        dim_t dhc, dim_t sic, dim_t slc, dim_t n_iter, dim_t n_layer,
        dim_t n_dir, dim_t states_ws_ld, const memory_storage_t &input,
        const memory_storage_t &output) const {
    return do_copy(false, ctx, batch, n_layer * n_dir,
            std::max(static_cast<size_t>(sic), static_cast<size_t>(dhc)),
            pd()->copy_res_iter_conf_, copy_bwd_kernel_, input, output);
}

status_t ref_rnn_fwd_t::rnn_bias(const exec_ctx_t &ctx, dim_t batch, dim_t dhc,
        dim_t iter, dim_t lay, dim_t dir, const rnn_utils::workspace_t &ws,
        const rnn_utils::scratch_t &scratch,
        const rnn_utils ::user_data_t &user_data) const {

    auto max_wg_size_per_dim = calc_local_range(ctx);

    parallel_for(ctx, bias_fwd_kernel_, [&](::sycl::handler &cgh) {
        auto src_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        scratch.gates(0).get())
                          ->get_inout_memory_arg(ctx.stream(), cgh);
        auto bias_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        user_data.bias(lay, dir).get())
                          ->get_in_memory_arg(ctx.stream(), cgh);

        auto dst_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        ws.states(lay + 1, dir, iter).get())
                          ->get_out_memory_arg(ctx.stream(), cgh);
        ref_rnn_bias_fwd bias_kernel(pd()->sycl_rnn_bias_fwd_conf_t_,
                src_mem_arg, bias_mem_arg, dst_mem_arg);

        size_t local_batch = max_wg_size_per_dim;
        size_t local_channel = max_wg_size_per_dim;
        size_t global_batch
                = calc_global_range(static_cast<size_t>(max_wg_size_per_dim),
                        static_cast<size_t>(batch));
        size_t global_channels
                = calc_global_range(static_cast<size_t>(max_wg_size_per_dim),
                        static_cast<size_t>(dhc));
        cgh.parallel_for(
                ::sycl::nd_range<3>(
                        ::sycl::range<3>(global_channels, global_batch, 1),
                        ::sycl::range<3>(local_channel, local_batch, 1)),
                bias_kernel);
    });

    return status::success;
}

status_t ref_rnn_bwd_t::rnn_bias(const exec_ctx_t &ctx, dim_t batch, dim_t dhc,
        dim_t iter, dim_t lay, dim_t dir, dim_t n_layer,
        const std::unique_ptr<dnnl::impl::memory_storage_t> &diff_states_layer,
        const std::unique_ptr<dnnl::impl::memory_storage_t> &diff_cell_iter,
        const rnn_utils ::user_data_t &user_data,
        const std::unique_ptr<dnnl::impl::memory_storage_t> &scratch_gate,
        const std::unique_ptr<dnnl::impl::memory_storage_t> &diff_gates) const {

    auto max_wg_size_per_dim = calc_local_range(ctx);

    parallel_for(ctx, bias_bwd_kernel_, [&](::sycl::handler &cgh) {
        auto gates_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        scratch_gate.get())
                          ->get_in_memory_arg(ctx.stream(), cgh);
        auto diff_lay_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        diff_states_layer.get())
                          ->get_in_memory_arg(ctx.stream(), cgh);
        auto diff_iter_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        diff_cell_iter.get())
                          ->get_in_memory_arg(ctx.stream(), cgh);
        auto scratch_diff_gates_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        diff_gates.get())
                          ->get_out_memory_arg(ctx.stream(), cgh);
        auto diff_bias_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        user_data.diff_bias(n_layer - lay - 1, dir).get())
                          ->get_inout_memory_arg(ctx.stream(), cgh);

        ref_rnn_bias_bwd bias_kernel(pd()->sycl_rnn_bias_bwd_conf_t_,
                gates_mem_arg, diff_lay_mem_arg, diff_iter_mem_arg,
                scratch_diff_gates_mem_arg, diff_bias_mem_arg);

        size_t local_batch = max_wg_size_per_dim;
        size_t local_channel = max_wg_size_per_dim;
        size_t global_batch
                = calc_global_range(static_cast<size_t>(max_wg_size_per_dim),
                        static_cast<size_t>(batch));
        size_t global_channels
                = calc_global_range(static_cast<size_t>(max_wg_size_per_dim),
                        static_cast<size_t>(dhc));
        cgh.parallel_for(
                ::sycl::nd_range<3>(
                        ::sycl::range<3>(global_channels, global_batch, 1),
                        ::sycl::range<3>(local_channel, local_batch, 1)),
                bias_kernel);
    });

    return status::success;
}

// //********************* Execution function *********************//

void ref_rnn_common_base_t::debug_print(const rnn_utils::conf_t &rnn, dim_t slc,
        dim_t sic, bool with_bias, bool with_dst_iter) const {
    DPRINT("\n%s\n", "+++++++++++++++");
    DPRINT("%s\n", "+++++++++++++++");
    DPRINT("  n_layer         = %lld\n", static_cast<long long>(rnn.n_layer));
    DPRINT("  n_dir           = %lld\n", static_cast<long long>(rnn.n_dir));
    DPRINT("  n_iter          = %lld\n", static_cast<long long>(rnn.n_iter));
    DPRINT("  n_gates         = %lld\n", static_cast<long long>(rnn.n_gates));
    DPRINT("  n_bias          = %lld\n", static_cast<long long>(rnn.n_bias));
    DPRINT("  n_weights_layer = %lld\n", static_cast<long long>(slc));
    DPRINT("  n_weights_iter  = %lld\n", static_cast<long long>(sic));
    DPRINT("  batch           = %lld\n", static_cast<long long>(rnn.mb));
    DPRINT("  slc             = %lld\n", static_cast<long long>(rnn.slc));
    DPRINT("  sic             = %lld\n", static_cast<long long>(rnn.sic));
    DPRINT("  dhc             = %lld\n", static_cast<long long>(rnn.dhc));
    DPRINT("  dlc             = %lld\n", static_cast<long long>(rnn.dlc));
    DPRINT("%s\n", "+++++++++++++++");
    DPRINT("  use_workspace   = %s\n", rnn.use_workspace ? "yes" : "no");
    DPRINT("%s\n", "+++++++++++++++");
    DPRINT("  with_bias       = %s\n", with_bias ? "yes" : "no");
    DPRINT("  with_dst_iter   = %s\n", with_dst_iter ? "yes" : "no");
    DPRINT("%s\n", "+++++++++++++++");
}

void ref_rnn_common_base_t::get_user_data(const exec_ctx_t &ctx,
        rnn_utils::user_data_t &user_data, cpy_ctx_t &cpy_ctx, bool is_fwd,
        const rnn_pd_t *pd) const {
    auto &src_layer_native_ = CTX_IN_STORAGE(DNNL_ARG_SRC_LAYER);
    auto &src_iter_native_ = CTX_IN_STORAGE(DNNL_ARG_SRC_ITER);
    auto &wei_layer_native_ = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS_LAYER);
    auto &wei_iter_native_ = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS_ITER);
    auto &bias_native_ = CTX_IN_STORAGE(DNNL_ARG_BIAS);

    auto &dst_last_layer_native_ = is_fwd ? CTX_OUT_STORAGE(DNNL_ARG_DST_LAYER)
                                          : CTX_IN_STORAGE(DNNL_ARG_DST_LAYER);
    auto &dst_last_iter_native_ = is_fwd ? CTX_OUT_STORAGE(DNNL_ARG_DST_ITER)
                                         : CTX_IN_STORAGE(DNNL_ARG_DST_ITER);

    auto &diff_dst_layer_native_ = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST_LAYER);
    auto &diff_dst_iter_native_ = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST_ITER);

    auto &diff_src_layer_native_ = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC_LAYER);
    auto &diff_src_iter_native_ = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC_ITER);

    auto &diff_weights_layer_native_
            = CTX_OUT_STORAGE(DNNL_ARG_DIFF_WEIGHTS_LAYER);
    auto &diff_weights_iter_native_
            = CTX_OUT_STORAGE(DNNL_ARG_DIFF_WEIGHTS_ITER);
    auto &diff_bias_native_ = CTX_OUT_STORAGE(DNNL_ARG_DIFF_BIAS);
    user_data = rnn_utils::user_data_t(wei_layer_native_, pd->weights_md(0),
            diff_weights_layer_native_, pd->diff_weights_md(0),
            wei_iter_native_, pd->weights_md(1), diff_weights_iter_native_,
            pd->diff_weights_md(1), bias_native_, pd->weights_md(2),
            diff_bias_native_, pd->diff_weights_md(2));

    if (is_fwd) {
        cpy_ctx.cpy_in_lay = &src_layer_native_;
        cpy_ctx.cpy_in_iter = &src_iter_native_;
        cpy_ctx.cpy_out_lay = &dst_last_layer_native_;
        cpy_ctx.cpy_out_iter = &dst_last_iter_native_;
    } else {
        cpy_ctx.cpy_in_lay = &diff_dst_layer_native_;
        cpy_ctx.cpy_in_iter = &diff_dst_iter_native_;
        cpy_ctx.cpy_out_lay = &diff_src_layer_native_;
        cpy_ctx.cpy_out_iter = &diff_src_iter_native_;
    }
}

status_t ref_rnn_fwd_t::execute_(const exec_ctx_t &ctx) const {

    impl::engine_t *engine = ctx.stream()->engine();

    const conf_t &rnn = this->pd()->rnn_conf;

    if (get_verbose_dev_mode(verbose_t::debuginfo) >= 2) {
        debug_print(rnn, this->pd()->SLC(), this->pd()->SIC(),
                this->pd()->with_bias(), this->pd()->with_dst_iter());
    }

    auto scratch_workspace
            = ctx.get_scratchpad_grantor().get_memory_storage(key_rnn_space);
    auto &workspace_ = rnn.is_training ? CTX_OUT_STORAGE(DNNL_ARG_WORKSPACE)
                                       : *scratch_workspace;
    const auto &workspace = rnn_utils::workspace_t(workspace_, rnn);

    const auto scratch
            = rnn_utils::scratch_t(rnn, ctx.get_scratchpad_grantor());

    rnn_utils::user_data_t user_data;
    cpy_ctx_t cpy_ctx;
    get_user_data(ctx, user_data, cpy_ctx, true, this->pd());

    CHECK(copy_init_layer(ctx, rnn.mb, rnn.dhc, rnn.sic, rnn.slc, rnn.n_iter,
            rnn.n_layer, rnn.n_dir, rnn.states_ws_ld, *cpy_ctx.cpy_in_lay,
            workspace.states()));
    CHECK(copy_init_iter(ctx, rnn.mb, rnn.dhc, rnn.sic, rnn.slc, rnn.n_iter,
            rnn.n_layer, rnn.n_dir, rnn.states_ws_ld, *cpy_ctx.cpy_in_iter,
            workspace.states()));

    // run the execution on the grid
    const grid_ctx_t &grid_struct {
            engine, ctx, user_data, workspace, scratch, pd()->rnn_conf};

    CHECK(this->grid_func(grid_struct));

    // Finally we copy the results to the result buffers
    CHECK(copy_res_layer(ctx, rnn.mb, rnn.dhc, rnn.sic, rnn.slc, rnn.n_iter,
            rnn.n_layer, rnn.n_dir, rnn.states_ws_ld, workspace.states(),
            *cpy_ctx.cpy_out_lay));
    CHECK(copy_res_iter(ctx, rnn.mb, rnn.dhc, rnn.sic, rnn.slc, rnn.n_iter,
            rnn.n_layer, rnn.n_dir, rnn.states_ws_ld, workspace.states(),
            *cpy_ctx.cpy_out_iter));

    return status::success;
};

status_t ref_rnn_bwd_t::execute_(const exec_ctx_t &ctx) const {

    impl::engine_t *engine = ctx.stream()->engine();
    const conf_t &rnn = this->pd()->rnn_conf;

    if (get_verbose_dev_mode(verbose_t::debuginfo) >= 2) {
        debug_print(rnn, this->pd()->SLC(), this->pd()->SIC(),
                this->pd()->with_bias(), this->pd()->with_dst_iter());
    }

    auto scratch_workspace
            = ctx.get_scratchpad_grantor().get_memory_storage(key_rnn_space);
    auto &workspace_ = CTX_IN_STORAGE(DNNL_ARG_WORKSPACE);
    const auto &workspace = rnn_utils::workspace_t(workspace_, rnn);

    const auto scratch
            = rnn_utils::scratch_t(rnn, ctx.get_scratchpad_grantor());

    rnn_utils::user_data_t user_data;
    cpy_ctx_t cpy_ctx;
    get_user_data(ctx, user_data, cpy_ctx, false, this->pd());

    CHECK(copy_init_layer(ctx, rnn.mb, rnn.dhc, rnn.sic, rnn.slc, rnn.n_iter,
            rnn.n_layer, rnn.n_dir, rnn.scratch_diff_states_ld,
            *cpy_ctx.cpy_in_lay, scratch.diff_states()));
    CHECK(copy_init_iter(ctx, rnn.mb, rnn.dhc, rnn.sic, rnn.slc, rnn.n_iter,
            rnn.n_layer, rnn.n_dir, rnn.scratch_diff_states_ld,
            *cpy_ctx.cpy_in_iter, scratch.diff_states()));

    // run the execution on the grid
    const grid_ctx_t &grid_struct {
            engine, ctx, user_data, workspace, scratch, pd()->rnn_conf};

    CHECK(this->grid_func(grid_struct));

    // Finally we copy the results to the result buffers
    CHECK(copy_res_layer(ctx, rnn.mb, rnn.dhc, rnn.sic, rnn.slc, rnn.n_iter,
            rnn.n_layer, rnn.n_dir, rnn.scratch_diff_states_ld,
            scratch.diff_states(), *cpy_ctx.cpy_out_lay));
    CHECK(copy_res_iter(ctx, rnn.mb, rnn.dhc, rnn.sic, rnn.slc, rnn.n_iter,
            rnn.n_layer, rnn.n_dir, rnn.scratch_diff_states_ld,
            scratch.diff_states(), *cpy_ctx.cpy_out_iter));

    return status::success;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
