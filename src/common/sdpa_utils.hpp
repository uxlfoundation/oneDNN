/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#ifndef COMMON_SDPA_UTILS_HPP
#define COMMON_SDPA_UTILS_HPP

#include "oneapi/dnnl/dnnl.h"

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/sdpa_types.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {

#define VCHECK_SDPA(f, msg, ...) \
    VCHECK(primitive, create, check, sdpa, (f), msg, ##__VA_ARGS__);

#define VCHECK_SDPA_COND(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, sdpa, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__);

#define VCHECK_SDPA_ATTR_TYPE( \
        variable_check, variable, attribute_member_name, expected_types) \
    VCONDCHECK(primitive, create, check, sdpa, (variable_check), \
            status::invalid_arguments, VERBOSE_INVALID_DATATYPE, \
            format_verbose_string(#variable attribute_member_name \
                    "(%s). must be " expected_types, \
                    attr2str(variable).c_str()) \
                    .c_str())

#define VCHECK_SDPA_UNIMPL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, sdpa, (cond), status::unimplemented, \
            msg, ##__VA_ARGS__);

static inline status_t sdpa_desc_check(const memory_desc_t *q_desc,
        const memory_desc_t *k_desc, const memory_desc_t *v_desc,
        const memory_desc_t *dst_desc, const memory_desc_t *attn_mask_md,
        const engine_t *engine, const primitive_attr_t *attr,
        const primitive_attr_t *kq_attr, const primitive_attr_t *vs_attr) {
    int ndims = dst_desc->ndims;
    int r = ndims - 2, c = ndims - 1;
    VCHECK_SDPA_COND(utils::everyone_is(ndims, q_desc->ndims, k_desc->ndims,
                             v_desc->ndims),
            "number of dimensions have to match. expected: %d q: %d k: %d v: "
            "%d",
            ndims, q_desc->ndims, k_desc->ndims, v_desc->ndims);

    VCHECK_SDPA_COND(q_desc->dims[c] == k_desc->dims[r],
            "q_desc->dims[%d](%s) must match k_desc->dims[%d](%s)", c,
            md2dim_str(q_desc).c_str(), r, md2dim_str(k_desc).c_str());
    VCHECK_SDPA_COND(k_desc->dims[c] == v_desc->dims[r],
            "k_desc->dims[%d](%s) must match v_desc->dims[%d](%s)", c,
            md2dim_str(k_desc).c_str(), r, md2dim_str(v_desc).c_str());
    VCHECK_SDPA_COND(dst_desc->dims[r] == q_desc->dims[r],
            "dst_desc->dims[%d](%s) == q_desc->dims[%d](%s)", r,
            md2dim_str(dst_desc).c_str(), r, md2dim_str(q_desc).c_str());
    VCHECK_SDPA_COND(dst_desc->dims[c] == v_desc->dims[c],
            "dst_desc->dims[%d](%s) == v_desc->dims[%d](%s)", c,
            md2dim_str(dst_desc).c_str(), c, md2dim_str(v_desc).c_str());

    VCHECK_SDPA_COND(!any_memory_desc_host_scalar(
                             q_desc, k_desc, v_desc, dst_desc, attn_mask_md),
            VERBOSE_UNSUPPORTED_FORMAT_KIND);

    return status::success;
}

static inline status_t sdpa_attr_check(const memory_desc_t *q_desc,
        const memory_desc_t *k_desc, const memory_desc_t *v_desc,
        const engine_t *engine, const primitive_attr_t *attr,
        const primitive_attr_t *kq_attr, const primitive_attr_t *vs_attr) {
    using smask_t = primitive_attr_t::skip_mask_t;

    if (utils::everyone_is(nullptr, attr, kq_attr, vs_attr))
        return status::success;
    if (attr && attr->has_default_values() && kq_attr
            && kq_attr->has_default_values() && vs_attr
            && vs_attr->has_default_values()) {
        return status::success;
    }

    using namespace dnnl::impl::data_type;
    if (kq_attr && !kq_attr->has_default_values()) {
        const auto &sc = kq_attr->scales_;
        const auto &zp = kq_attr->zero_points_;
        if (!sc.has_default_values()) {
            const auto &scale_dt = sc.get_data_type(DNNL_ARG_WEIGHTS);
            VCHECK_SDPA_ATTR_TYPE(utils::one_of(scale_dt, f16, bf16, f32),
                    kq_attr, "scales", "f16, bf16, or f32");
        }
        if (!zp.has_default_values()) {
            const auto &zp_dt = zp.get_data_type(DNNL_ARG_WEIGHTS);
            VCHECK_SDPA_ATTR_TYPE(utils::one_of(zp_dt, s4, u4, u8, s8, s32),
                    kq_attr, "zero_points", "u4, s4, u8, s8, or s32");
        }
    }

    if (vs_attr && !vs_attr->has_default_values()) {
        const auto &sc = vs_attr->scales_;
        const auto &zp = vs_attr->zero_points_;

        if (!sc.has_default_values()) {
            const auto &scale_dt = sc.get_data_type(DNNL_ARG_WEIGHTS);
            VCHECK_SDPA_ATTR_TYPE(utils::one_of(scale_dt, f16, bf16, f32),
                    vs_attr, "scales", "f16, bf16, or f32");

            // By default, host scalar scales are not supported for GPU
            // as the value should be accessed differently in the kernel
            VCHECK_SDPA_UNIMPL(IMPLICATION(engine->kind() == engine_kind::gpu,
                                       !sc.has_host_scalars()),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
        }
        if (!zp.has_default_values()) {
            const auto &zp_dt = zp.get_data_type(DNNL_ARG_WEIGHTS);
            VCHECK_SDPA_ATTR_TYPE(utils::one_of(zp_dt, s4, u4, u8, s8, s32),
                    vs_attr, "zero_points", "u4, s4, u8, s8, or s32");

            // By default, host scalar zero points are not supported for GPU
            // as the value should be accessed differently in the kernel
            VCHECK_SDPA_UNIMPL(IMPLICATION(engine->kind() == engine_kind::gpu,
                                       !zp.has_host_scalars()),
                    VERBOSE_UNSUPPORTED_ZP_CFG);
        }
    }

    if (attr) {
        smask_t attr_mask = smask_t::none;
        VCHECK_SDPA_UNIMPL(
                attr->has_default_values(attr_mask), VERBOSE_UNSUPPORTED_ATTR);
    }

    return status::success;
}

static inline sdpa_desc_t create_sdpa_desc(const memory_desc_t *q_md,
        const memory_desc_t *k_md, const memory_desc_t *v_md,
        const memory_desc_t *dst_md, const memory_desc_t *attn_mask_md,
        data_type_t scale_dt, bool invert_scale, dim_t kv_head_number,
        attn_mask_type_t attn_mask_type, alg_kind_t softmax_alg,
        const primitive_attr_t *kq_attr, const primitive_attr_t *vs_attr,
        prop_kind_t prop) {
    auto sdpa_desc = sdpa_desc_t();
    sdpa_desc.primitive_kind = primitive_kind::sdpa;
    sdpa_desc.q_desc = *q_md;
    sdpa_desc.k_desc = *k_md;
    sdpa_desc.kq_acc_dt = data_type::f32;
    sdpa_desc.vs_acc_dt = data_type::f32;
    if (kq_attr) {
        sdpa_desc.kq_scales = kq_attr->scales_.get(DNNL_ARG_WEIGHTS);
        sdpa_desc.kq_zero_points = kq_attr->zero_points_.get(DNNL_ARG_WEIGHTS);
        if (kq_attr->acc_mode_ == accumulation_mode::f16) {
            sdpa_desc.kq_acc_dt = data_type::f16;
        }
    }
    if (vs_attr) {
        sdpa_desc.vs_scales = vs_attr->scales_.get(DNNL_ARG_WEIGHTS);
        sdpa_desc.vs_zero_points = vs_attr->zero_points_.get(DNNL_ARG_WEIGHTS);
        if (vs_attr->acc_mode_ == accumulation_mode::f16) {
            sdpa_desc.vs_acc_dt = data_type::f16;
        }
    }
    sdpa_desc.v_desc = *v_md;
    sdpa_desc.dst_desc = *dst_md;
    if (attn_mask_md) sdpa_desc.attn_mask_desc = *attn_mask_md;
    sdpa_desc.scale_dt = scale_dt;
    sdpa_desc.invert_scale = invert_scale;
    sdpa_desc.kv_head_number = kv_head_number;
    sdpa_desc.mask_type = attn_mask_type;
    sdpa_desc.softmax_alg = softmax_alg;
    sdpa_desc.prop_kind = prop;
    return sdpa_desc;
}

static inline sdpa_desc_t create_sdpa_desc(const memory_desc_t *q_md,
        const memory_desc_t *k_md, const memory_desc_t *v_md,
        const memory_desc_t *dst_md,
        const memory_desc_t *diff_q_md,
        const memory_desc_t *diff_k_md, const memory_desc_t *diff_v_md,
        const memory_desc_t *diff_dst_md,
        const memory_desc_t *attn_mask_md,
        data_type_t scale_dt, bool invert_scale, dim_t kv_head_number,
        attn_mask_type_t attn_mask_type, alg_kind_t softmax_alg,
        const primitive_attr_t *kq_attr, const primitive_attr_t *vs_attr) {
    auto sdpa_desc = sdpa_desc_t();
    sdpa_desc.primitive_kind = primitive_kind::sdpa;
    sdpa_desc.q_desc = *q_md;
    sdpa_desc.k_desc = *k_md;
    sdpa_desc.v_desc = *v_md;

    sdpa_desc.dst_desc = *dst_md;
    sdpa_desc.diff_dst_desc = *diff_dst_md;

    sdpa_desc.diff_q_desc = *diff_q_md;
    sdpa_desc.diff_k_desc = *diff_k_md;
    sdpa_desc.diff_v_desc = *diff_v_md;
    if (attn_mask_md) sdpa_desc.attn_mask_desc = *attn_mask_md;
    sdpa_desc.scale_dt = scale_dt;
    sdpa_desc.invert_scale = invert_scale;
    sdpa_desc.kv_head_number = kv_head_number;
    sdpa_desc.mask_type = attn_mask_type;
    sdpa_desc.softmax_alg = softmax_alg;
    sdpa_desc.prop_kind = prop_kind::backward;
    return sdpa_desc;
}

static inline status_t create_sdpa_pd(
        std::shared_ptr<primitive_desc_t> &sdpa_pd_, engine_t *engine,
        const memory_desc_t *q_md, const memory_desc_t *k_md,
        const memory_desc_t *v_md, const memory_desc_t *dst_md,
        const memory_desc_t *attn_mask_md, data_type_t scale_dt,
        bool invert_scale, dim_t kv_head_number,
        attn_mask_type_t attn_mask_type, alg_kind_t softmax_alg,
        const primitive_attr_t *attr, const primitive_attr_t *kq_attr = nullptr,
        const primitive_attr_t *vs_attr = nullptr, prop_kind_t prop = prop_kind::forward_inference) {
    CHECK(sdpa_attr_check(q_md, k_md, v_md, engine, attr, kq_attr, vs_attr));
    CHECK(sdpa_desc_check(q_md, k_md, v_md, dst_md, attn_mask_md, engine, attr,
            kq_attr, vs_attr));

    auto sdpa_desc = create_sdpa_desc(q_md, k_md, v_md, dst_md, attn_mask_md,
            scale_dt, invert_scale, kv_head_number, attn_mask_type, softmax_alg,
            kq_attr, vs_attr, prop);

    primitive_attr_t sdpa_attr = attr ? *attr : default_attr();

    primitive_desc_iterator_t it(
            engine, (op_desc_t *)&sdpa_desc, &sdpa_attr, nullptr);

    sdpa_pd_ = *(++it);
    VCHECK_SDPA_COND(sdpa_pd_, "failed to create the SDPA primitive");

    return status::success;
}

} // namespace impl
} // namespace dnnl

#endif
