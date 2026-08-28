/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include <assert.h>

#include "deconvolution_pd.hpp"
#include "oneapi/dnnl/dnnl.h"
#include "opdesc.hpp"
#include "primitive_desc_iface.hpp"
#include "primitive_desc_iterator.hpp"

#include "c_types_map.hpp"
#include "memory_desc.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::alg_kind;
using namespace dnnl::impl::types;

#define VCHECK_DECONV(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, deconvolution, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__)

#define VCHECK_DECONV_UNIMPL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, deconvolution, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__)
namespace {
status_t deconv_desc_init(deconvolution_desc_t *deconv_desc,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *dst_desc,
        const dims_t strides, const dims_t dilates, const dims_t padding_l,
        const dims_t padding_r) {
    VCHECK_DECONV(!any_null(deconv_desc, src_desc, weights_desc, dst_desc,
                          strides, padding_l),
            VERBOSE_NULL_ARG);
    VCHECK_DECONV(
            one_of(alg_kind, deconvolution_direct, deconvolution_winograd),
            VERBOSE_BAD_ALGORITHM);

    VCHECK_DECONV(!any_memory_desc_host_scalar(
                          src_desc, weights_desc, bias_desc, dst_desc),
            VERBOSE_UNSUPPORTED_FORMAT_KIND);

    if (padding_r == nullptr) padding_r = padding_l;

    auto dd = deconvolution_desc_t();
    dd.primitive_kind = primitive_kind::deconvolution;
    dd.prop_kind = prop_kind;
    dd.alg_kind = alg_kind;
    dd.use_inversion = false; // Must be always `false` for deconv.

    dd.diff_src_desc = dd.src_desc = zero_md();
    dd.diff_dst_desc = dd.dst_desc = zero_md();
    dd.diff_weights_desc = dd.weights_desc = zero_md();
    dd.diff_bias_desc = dd.bias_desc = zero_md();

    const bool is_fwd = one_of(prop_kind, forward_training, forward_inference);
    const bool with_bias
            = bias_desc && bias_desc->format_kind != format_kind::undef;
    const bool with_groups = weights_desc->ndims == src_desc->ndims + 1;

    bool runtime_dims_or_strides
            = memory_desc_wrapper(src_desc).has_runtime_dims_or_strides()
            || memory_desc_wrapper(weights_desc).has_runtime_dims_or_strides()
            || memory_desc_wrapper(dst_desc).has_runtime_dims_or_strides();
    if (with_bias)
        runtime_dims_or_strides = runtime_dims_or_strides
                || memory_desc_wrapper(bias_desc).has_runtime_dims_or_strides();
    VCHECK_DECONV_UNIMPL(
            !runtime_dims_or_strides, VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    (prop_kind == backward_data ? dd.diff_src_desc : dd.src_desc) = *src_desc;
    (is_fwd ? dd.dst_desc : dd.diff_dst_desc) = *dst_desc;
    (prop_kind == backward_weights ? dd.diff_weights_desc : dd.weights_desc)
            = *weights_desc;
    if (with_bias)
        (prop_kind == backward_weights ? dd.diff_bias_desc : dd.bias_desc)
                = *bias_desc;

    int sp_dims = src_desc->ndims - 2;
    utils::array_copy(dd.strides, strides, sp_dims);
    utils::array_copy(dd.padding[0], padding_l, sp_dims);
    utils::array_copy(dd.padding[1], padding_r, sp_dims);
    if (dilates)
        utils::array_copy(dd.dilates, dilates, sp_dims);
    else
        utils::array_set(dd.dilates, 0, sp_dims);

    dd.accum_data_type = types::default_accum_data_type(src_desc->data_type,
            weights_desc->data_type, dst_desc->data_type, prop_kind);
    VCHECK_DECONV(dd.accum_data_type != data_type::undef,
            VERBOSE_INVALID_DATATYPE, "accumulation");

    const dim_t g = with_groups ? weights_desc->dims[0] : 1;
    VCHECK_DECONV(src_desc->ndims == dst_desc->ndims,
            VERBOSE_INCONSISTENT_NDIMS_WITH_VALS, "src", "dst", src_desc->ndims,
            dst_desc->ndims);
    VCHECK_DECONV(utils::one_of(src_desc->ndims, 3, 4, 5), VERBOSE_BAD_NDIMS,
            "src", src_desc->ndims);
    VCHECK_DECONV(utils::one_of(weights_desc->ndims, src_desc->ndims,
                          src_desc->ndims + 1),
            VERBOSE_INCONSISTENT_NDIMS_WITH_VALS, "src", "weights",
            weights_desc->ndims, src_desc->ndims);
    VCHECK_DECONV(IMPLICATION(with_bias, bias_desc->ndims == 1),
            VERBOSE_BAD_NDIMS, "bias", bias_desc->ndims);
    VCHECK_DECONV(
            IMPLICATION(with_bias, bias_desc->dims[0] == dst_desc->dims[1]),
            VERBOSE_INCONSISTENT_DIM, "bias", 0, "dst", 1);
    VCHECK_DECONV(src_desc->dims[0] == dst_desc->dims[0],
            VERBOSE_INCONSISTENT_DIM, "src", 0, "dst", 0);
    VCHECK_DECONV(src_desc->dims[1] == g * weights_desc->dims[with_groups + 1],
            VERBOSE_INCONSISTENT_DIM, "src", 1, "weights", with_groups + 1);
    VCHECK_DECONV(dst_desc->dims[1] == g * weights_desc->dims[with_groups + 0],
            VERBOSE_INCONSISTENT_DIM, "dst", 1, "weights", with_groups + 0);
    for (int i = 2; i < src_desc->ndims; ++i) {
        dim_t src = src_desc->dims[i];
        dim_t ker = weights_desc->dims[with_groups + i];
        dim_t dil = dd.dilates[i - 2];
        dim_t pad_l = padding_l[i - 2];
        dim_t pad_r = padding_r[i - 2];
        dim_t str = strides[i - 2];
        dim_t dst = dst_desc->dims[i];
        dim_t ker_range = 1 + (ker - 1) * (dil + 1);

        VCHECK_DECONV(str > 0, VERBOSE_BAD_DIM, "strides", i - 2);
        VCHECK_DECONV(dil >= 0, "%s: dilation (%d) must be non-negative",
                VERBOSE_INCONSISTENT_PRB, static_cast<int>(dil));
        VCHECK_DECONV(pad_l >= 0,
                "%s: left padding value (%d) must be non-negative",
                VERBOSE_INCONSISTENT_PRB, static_cast<int>(pad_l));
        VCHECK_DECONV(pad_r + str > 0,
                "%s: right padding (%d) and stride (%d) must sum up to a "
                "positive value",
                VERBOSE_INCONSISTENT_PRB, static_cast<int>(pad_r),
                static_cast<int>(str));
        VCHECK_DECONV((dst - ker_range + pad_l + pad_r) / str + 1 == src,
                "%s: mismatch between actual and computed src dims, src (%d) "
                "!= (dst(%d) - ker(%d) + pad_l(%d) + pad_r(%d))/ str(%d) + 1",
                VERBOSE_INCONSISTENT_PRB, static_cast<int>(src),
                static_cast<int>(dst), static_cast<int>(ker_range),
                static_cast<int>(pad_l), static_cast<int>(pad_r),
                static_cast<int>(str));
    }

    *deconv_desc = dd;
    return success;
}

status_t deconv_attr_check(const deconvolution_desc_t &desc,
        const engine_t *engine, const primitive_attr_t *attr) {
    using smask_t = primitive_attr_t::skip_mask_t;

    if (attr == nullptr) return status::success;
    if (attr->has_default_values()) return status::success;

    // Check attributes
    if (utils::one_of(desc.prop_kind, prop_kind::forward_inference,
                prop_kind::forward_training)) {
        const data_type_t src_dt = desc.src_desc.data_type;
        const data_type_t dst_dt = desc.dst_desc.data_type;

        auto fwd_attr_mask
                = smask_t::post_ops | smask_t::sum_dt | smask_t::fpmath_mode;

        bool is_int8 = utils::one_of(src_dt, data_type::s8, data_type::u8);
        if (engine->kind() == engine_kind::gpu)
            is_int8 = is_int8
                    || utils::one_of(dst_dt, data_type::s8, data_type::u8,
                            data_type::s32);
        if (is_int8) fwd_attr_mask |= smask_t::scales | smask_t::zero_points;

        VCHECK_DECONV_UNIMPL(attr->has_default_values(fwd_attr_mask, dst_dt),
                VERBOSE_UNSUPPORTED_ATTR);

        // Check scales
        if (!attr->scales_.has_default_values()) {
            const auto &sc = attr->scales_;
            const bool with_groups
                    = desc.src_desc.ndims != desc.weights_desc.ndims;
            VCHECK_DECONV_UNIMPL(
                    IMPLICATION(!sc.has_default_values(DNNL_ARG_SRC),
                            sc.get_mask(DNNL_ARG_SRC) == 0),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
            VCHECK_DECONV_UNIMPL(
                    IMPLICATION(!sc.has_default_values(DNNL_ARG_WEIGHTS),
                            utils::one_of(sc.get_mask(DNNL_ARG_WEIGHTS), 0,
                                    with_groups ? 3 : 1)),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
            VCHECK_DECONV_UNIMPL(
                    IMPLICATION(!sc.has_default_values(DNNL_ARG_DST),
                            sc.get_mask(DNNL_ARG_DST) == 0),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
        }

        // Check zero points
        if (!attr->zero_points_.has_default_values()) {
            const auto &zp = attr->zero_points_;

            VCHECK_DECONV_UNIMPL(
                    IMPLICATION(!zp.has_default_values(DNNL_ARG_SRC),
                            utils::one_of(
                                    zp.get_mask(DNNL_ARG_SRC), 0, 1 << 1)),
                    VERBOSE_UNSUPPORTED_ZP_CFG);
            VCHECK_DECONV_UNIMPL(zp.has_default_values(DNNL_ARG_WEIGHTS),
                    VERBOSE_UNSUPPORTED_ZP_CFG);
            VCHECK_DECONV_UNIMPL(
                    IMPLICATION(!zp.has_default_values(DNNL_ARG_DST),
                            utils::one_of(
                                    zp.get_mask(DNNL_ARG_DST), 0, 1 << 1)),
                    VERBOSE_UNSUPPORTED_ZP_CFG);
        }

        // Check post-ops
        if (!attr->post_ops_.has_default_values()) {
            const auto &po = attr->post_ops_;
            using namespace primitive_kind;
            VCHECK_DECONV_UNIMPL(
                    po.has_default_values({binary, eltwise, prelu, sum}),
                    VERBOSE_UNSUPPORTED_POSTOP);

            // Check sum
            VCHECK_DECONV_UNIMPL(
                    po.check_sum_consistency(dst_dt, is_int8, true),
                    VERBOSE_UNSUPPORTED_POSTOP);

            // Note: verbose support is inside the call.
            CHECK(po.validate_binary(engine->kind(), &desc.dst_desc));
        }
    } else {
        auto bwd_attr_mask = smask_t::fpmath_mode;
        VCHECK_DECONV_UNIMPL(attr->has_default_values(bwd_attr_mask),
                VERBOSE_UNSUPPORTED_ATTR);
    }

    return status::success;
}

status_t fwd_conv_descr_create(convolution_desc_t &cd,
        const deconvolution_desc_t &dd, const memory_desc_t *bias_md,
        data_type_t dst_dt) {
    VDISPATCH_DECONVOLUTION_IC(
            utils::one_of(dd.prop_kind, prop_kind::forward_training,
                    prop_kind::forward_inference),
            VERBOSE_BAD_PROPKIND);
    VDISPATCH_DECONVOLUTION_IC(dd.alg_kind == alg_kind::deconvolution_direct,
            VERBOSE_BAD_ALGORITHM);

    // create a fwd convolution descriptor with padding adjusted
    // to the perspective of backward propagation, namely:
    // - left padding replaced by left overflow
    // - right padding replaced by right overflow
    const int ndims_spatial = dd.dst_desc.ndims - 2;
    dims_t overflow_l;
    dims_t overflow_r;
    dim_t ks = 1;
    for (int i = 0; i < ndims_spatial; i++) {
        VDISPATCH_DECONVOLUTION_IC(dd.strides[i] == 1,
                VERBOSE_UNSUPPORTED_FEATURE,
                "only unit strides are allowed for bwd-to-fwd conversion");

        const dim_t K
                = dd.weights_desc
                          .dims[dd.weights_desc.ndims - ndims_spatial + i];
        ks *= K;
        const dim_t D = dd.dilates[i];
        const dim_t PL = dd.padding[0][i]; // left padding
        const dim_t PR = dd.padding[1][i]; // right padding
        constexpr dim_t S = 1;
        // the following relations hold for unit stride only
        overflow_l[i] = ((K - 1) * (D + 1) - PL) / S;
        overflow_r[i] = ((K - 1) * (D + 1) - PR) / S;
        VDISPATCH_DECONVOLUTION_IC(overflow_l[i] >= 0 && overflow_r[i] >= 0,
                VERBOSE_UNSUPPORTED_FEATURE,
                "Unsupported padding was provided");
    }

    assert(dst_dt != data_type::undef);
    memory_desc_t dst_md_patched;
    CHECK(memory_desc_init_by_md_and_dt(dst_md_patched, dd.dst_desc, dst_dt));

    CHECK(conv_desc_init(&cd, prop_kind::forward_training,
            alg_kind::convolution_direct, &dd.src_desc, &dd.weights_desc,
            bias_md, &dst_md_patched, dd.strides, dd.dilates, overflow_l,
            overflow_r));

    // Keep this internal descriptor distinct and tell an opted-in forward
    // implementation to reverse its spatial weight indices.
    if (ks > 1) {
        cd.diff_src_desc = cd.src_desc;
        cd.diff_dst_desc = cd.dst_desc;
    }
    // Note: internal field to hint this conv is created from deconv.
    cd.use_inversion = true;
    return status::success;
}

status_t conv_descr_create(convolution_desc_t &cd,
        const deconvolution_desc_t &dd, const memory_desc_t *bias_md,
        data_type_t src_dt) {
    using namespace prop_kind;
    alg_kind_t alg_kind = dd.alg_kind == alg_kind::deconvolution_direct
            ? alg_kind::convolution_direct
            : alg_kind::convolution_winograd;

    const memory_desc_t *src_md, *dst_md, *d_weights_d;
    memory_desc_t src_md_patched;
    prop_kind_t prop_kind;

    // The forward deconvolution is implemented as a backward-by-data
    // convolution: only in that case the created convolution descriptor carries
    // the backward memory descriptors that forward-only implementations cannot
    // read, and only that case relies on the spatial weights inversion.
    const bool is_fwd_deconv
            = utils::one_of(dd.prop_kind, forward_training, forward_inference);

    if (is_fwd_deconv) {
        prop_kind = backward_data;
        assert(src_dt != data_type::undef);
        CHECK(memory_desc_init_by_md_and_dt(
                src_md_patched, dd.dst_desc, src_dt));
        src_md = &src_md_patched;
        dst_md = &dd.src_desc;
        d_weights_d = &dd.weights_desc;
    } else if (dd.prop_kind == backward_data) {
        assert(src_dt == data_type::undef);
        prop_kind = forward_training;
        src_md = &dd.diff_dst_desc;
        dst_md = &dd.diff_src_desc;
        d_weights_d = &dd.weights_desc;
    } else {
        assert(src_dt == data_type::undef);
        prop_kind = dd.prop_kind;
        src_md = &dd.diff_dst_desc;
        dst_md = &dd.src_desc;
        d_weights_d = &dd.diff_weights_desc;
    }

    /* create weights desc for convolution by swapping OC and IC axes */
    memory_desc_t c_weights_d;
    const bool with_groups = d_weights_d->ndims == src_md->ndims + 1;
    int perm[DNNL_MAX_NDIMS] {}; // deconv to conv weight permutation
    for (int d = 0; d < DNNL_MAX_NDIMS; ++d)
        perm[d] = d;
    nstl::swap(perm[0 + with_groups], perm[1 + with_groups]);
    CHECK(memory_desc_permute_axes(c_weights_d, *d_weights_d, perm));

    CHECK(conv_desc_init(&cd, prop_kind, alg_kind, src_md, &c_weights_d,
            bias_md, dst_md, dd.strides, dd.dilates, dd.padding[0],
            dd.padding[1]));

    if (is_fwd_deconv) {
        // Manually update forward descriptors since certain implementations
        // might not handle backward ones.
        cd.src_desc = cd.diff_src_desc;
        cd.dst_desc = cd.diff_dst_desc;

        // Note: internal field to indicate the conv opdesc is created from
        // deconv.
        cd.use_inversion = true;
    }

    return status::success;
}

} // namespace

namespace dnnl {
namespace impl {

status_t create_conv_pd(std::shared_ptr<primitive_desc_t> &conv_pd,
        const engine_t *engine, const deconvolution_pd_t *deconv_pd,
        data_type_t src_dt, bool force_empty_bias, bool allow_wei_compensation,
        bool copy_attr, bool (*filter)(const primitive_desc_t *conv_pd)) {
    // By default the nested convolution is created with default attributes:
    // forward deconvolution applies post-ops and/or bias afterwards, and
    // backward deconvolution does not support attributes. When `copy_attr` is
    // set, the deconvolution attributes are forwarded to the convolution, e.g.
    // when the nested convolution is expected to apply post-ops itself.
    primitive_attr_t conv_attr = copy_attr
            ? primitive_attr_t(*deconv_pd->attr())
            : primitive_attr_t();
    if (!conv_attr.is_initialized()) return status::out_of_memory;

    const memory_desc_t *bias_md = nullptr;
    if (!force_empty_bias && deconv_pd->with_bias())
        bias_md = deconv_pd->invariant_bia_md();

    convolution_desc_t cd;
    while (true) {
        auto fwd_desc_st = fwd_conv_descr_create(
                cd, *deconv_pd->desc(), bias_md, src_dt);
        if (fwd_desc_st != status::success) break;

        primitive_desc_iterator_t it(
                engine, (op_desc_t *)&cd, &conv_attr, nullptr);
        if (!it.is_initialized()) break;

        while (++it != it.end()) {
            conv_pd = *it;
            // The forward-convolution-with-inversion descriptor produces a
            // correct deconvolution result only with implementations that honor
            // the `use_inversion` hint (i.e. reverse their spatial weight
            // indices). Common code cannot identify such implementations, hence
            // the caller must supply a `filter` selecting one. Without a filter,
            // skip the forward path entirely and fall back to the backward-data
            // descriptor below, which is correct for any implementation. See a
            // longer comment for a `filter` below.
            if (!filter || !filter(conv_pd.get())) continue;
            return status::success;
        }
        break;
    }

    CHECK(conv_descr_create(cd, *deconv_pd->desc(), bias_md, src_dt));

    primitive_desc_iterator_t it(engine, (op_desc_t *)&cd, &conv_attr, nullptr);
    if (!it.is_initialized()) return status::out_of_memory;

    while (++it != it.end()) {
        conv_pd = *it;
        // The nested convolution is expected to produce plain weights: the
        // deconvolution reorders them itself and does not expect any weights
        // compensation to be applied. Skip implementations that request extra
        // weights handling unless the caller explicitly allows it.
        if (!allow_wei_compensation
                && conv_pd->invariant_wei_md()->extra.flags != 0)
            continue;
        // `filter` is the mechanism to fetch a desired implementation from the
        // iterator while traversing the whole list: the iterator yields every
        // accepted convolution implementation in turn, and the caller-provided
        // predicate keeps skipping candidates until one it recognizes (e.g. a
        // specific implementation type via a `dynamic_cast`) is found. This
        // lets the common code own the iteration while the caller, which is the
        // only one that knows the concrete implementation types, decides which
        // convolution pd is acceptable.
        if (filter && !filter(conv_pd.get())) continue;
        return status::success;
    }
    return status::unimplemented;
}

} // namespace impl
} // namespace dnnl

status_t dnnl_deconvolution_forward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *dst_desc,
        const dims_t strides, const dims_t dilates, const dims_t padding_l,
        const dims_t padding_r, const primitive_attr_t *attr) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;

    auto deconv_desc = deconvolution_desc_t();
    CHECK(deconv_desc_init(&deconv_desc, prop_kind, alg_kind, src_desc,
            weights_desc, bias_desc, dst_desc, strides, dilates, padding_l,
            padding_r));
    CHECK(deconv_attr_check(deconv_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&deconv_desc, nullptr, attr);
}

status_t dnnl_deconvolution_backward_data_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        alg_kind_t alg_kind, const memory_desc_t *diff_src_desc,
        const memory_desc_t *weights_desc, const memory_desc_t *diff_dst_desc,
        const dims_t strides, const dims_t dilates, const dims_t padding_l,
        const dims_t padding_r, const primitive_desc_iface_t *hint_fwd_pd,
        const primitive_attr_t *attr) {

    auto deconv_desc = deconvolution_desc_t();
    CHECK(deconv_desc_init(&deconv_desc, backward_data, alg_kind, diff_src_desc,
            weights_desc, nullptr, diff_dst_desc, strides, dilates, padding_l,
            padding_r));
    CHECK(deconv_attr_check(deconv_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&deconv_desc, hint_fwd_pd, attr);
}

status_t dnnl_deconvolution_backward_weights_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        alg_kind_t alg_kind, const memory_desc_t *src_desc,
        const memory_desc_t *diff_weights_desc,
        const memory_desc_t *diff_bias_desc, const memory_desc_t *diff_dst_desc,
        const dims_t strides, const dims_t dilates, const dims_t padding_l,
        const dims_t padding_r, const primitive_desc_iface_t *hint_fwd_pd,
        const primitive_attr_t *attr) {

    auto deconv_desc = deconvolution_desc_t();
    CHECK(deconv_desc_init(&deconv_desc, backward_weights, alg_kind, src_desc,
            diff_weights_desc, diff_bias_desc, diff_dst_desc, strides, dilates,
            padding_l, padding_r));
    CHECK(deconv_attr_check(deconv_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&deconv_desc, hint_fwd_pd, attr);
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
