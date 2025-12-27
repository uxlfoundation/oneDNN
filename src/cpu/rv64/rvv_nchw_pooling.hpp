/******************************************************************************
* Copyright 2024 Intel Corporation
* Copyright 2023 KNS Group LLC (YADRO)
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

#ifndef CPU_RV64_RVV_NCHW_POOLING_HPP
#define CPU_RV64_RVV_NCHW_POOLING_HPP

#include "common/primitive.hpp"
#include "cpu/cpu_pooling_pd.hpp"
#include "cpu/rv64/rvv_postops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct riscv_nchw_pooling_fwd_t : public primitive_t {
    struct pd_t : public cpu_pooling_fwd_pd_t {
        using cpu_pooling_fwd_pd_t::cpu_pooling_fwd_pd_t;

        DECLARE_COMMON_PD_T_("RISCV64GCV", riscv_nchw_pooling_fwd_t)

        status_t init(engine_t *engine) {
            UNUSED(engine);

            const format_tag_t desired_fmt_tag = utils::pick(ndims() - 3,
                    format_tag::ncw, format_tag::nchw, format_tag::ncdhw);

            VDISPATCH_POOLING(desc_.prop_kind == prop_kind::forward_inference,
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_POOLING(
                    utils::one_of(desc()->alg_kind, alg_kind::pooling_max,
                            alg_kind::pooling_avg_include_padding,
                            alg_kind::pooling_avg_exclude_padding),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_POOLING(set_default_params() == status::success,
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_POOLING(memory_desc_wrapper(dst_md()).is_dense(false),
                    VERBOSE_UNSUPPORTED_SPARSE_CFG);
            const bool is_f16 = src_md()->data_type == data_type::f16;
            VDISPATCH_POOLING(utils::one_of(src_md()->data_type, data_type::f32,
                                      data_type::f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_POOLING(src_md()->data_type == dst_md()->data_type,
                    VERBOSE_UNSUPPORTED_DT);
            if (is_f16) {
                VDISPATCH_POOLING(DNNL_RISCV_USE_ZVFH_INTRINSICS,
                        VERBOSE_UNSUPPORTED_ISA);
                VDISPATCH_POOLING(desc()->accum_data_type == data_type::f32,
                        VERBOSE_UNSUPPORTED_DT);
                // Fallback to reference if post-ops are requested for f16
                if (!attr()->post_ops_.has_default_values())
                    return status::unimplemented;
            }
            VDISPATCH_POOLING(
                    platform::has_data_type_support(src_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_POOLING(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_POOLING(!is_dilated(), VERBOSE_UNSUPPORTED_FEATURE,
                    "does not support dilations");
            using sm = primitive_attr_t::skip_mask_t;
            VDISPATCH_POOLING(attr()->has_default_values(sm::post_ops),
                    VERBOSE_UNSUPPORTED_ATTR);

            if (!attr()->post_ops_.has_default_values()) {
                const auto &po = attr()->post_ops_;
                const bool ok = (po.len() == 1)
                        && (po.entry_[0].is_binary()
                                || po.entry_[0].is_eltwise());
                VDISPATCH_POOLING(ok, VERBOSE_UNSUPPORTED_POSTOP);
            }
            VDISPATCH_POOLING(
                    memory_desc_matches_tag(*src_md(), desired_fmt_tag),
                    VERBOSE_UNSUPPORTED_TAG_S, "src");
            VDISPATCH_POOLING(
                    memory_desc_matches_tag(*dst_md(), desired_fmt_tag),
                    VERBOSE_UNSUPPORTED_TAG_S, "dst");
            VDISPATCH_POOLING(
                    attr_.set_default_formats(dst_md(0)) == status::success,
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_POOLING(KW() < max_kernel_width,
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "kernel width exceeds maximum");

            if (!attr()->post_ops_.has_default_values()) {
                const auto &po = attr()->post_ops_;
                if (po.len() == 1
                        && (po.entry_[0].is_binary()
                                || po.entry_[0].is_eltwise())) {
                    CHECK(postops_.init(engine, po, *dst_md()));
                }
            }
            return status::success;
        }

        rvv_postops_t postops_;
    };

    riscv_nchw_pooling_fwd_t(const pd_t *apd);

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

    constexpr static int max_kernel_width = 32;

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
