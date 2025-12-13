/******************************************************************************
* Copyright 2023 Intel Corporation
* Copyright 2025 ISCAS
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

#ifndef CPU_RV64_JIT_RVV_NCHW_POOLING_HPP
#define CPU_RV64_JIT_RVV_NCHW_POOLING_HPP

#include "common/primitive.hpp"
#include "cpu/cpu_pooling_pd.hpp"
#include "cpu/rv64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct jit_rvv_nchw_pool_kernel_t;

struct jit_rvv_nchw_pooling_fwd_t : public primitive_t {
    struct pd_t : public cpu_pooling_fwd_pd_t {
        using cpu_pooling_fwd_pd_t::cpu_pooling_fwd_pd_t;

        DECLARE_COMMON_PD_T_("jit:rvv", jit_rvv_nchw_pooling_fwd_t)

        status_t init(engine_t *engine) {
            UNUSED(engine);

            const format_tag_t desired_fmt_tag = utils::pick(ndims() - 3,
                    format_tag::ncw, format_tag::nchw, format_tag::ncdhw);

            VDISPATCH_POOLING(desc_.prop_kind == prop_kind::forward_inference,
                    VERBOSE_BAD_PROPKIND);

            VDISPATCH_POOLING(
                    utils::one_of(desc()->alg_kind, alg_kind::pooling_max,
                            alg_kind::pooling_avg_include_padding),
                    VERBOSE_BAD_ALGORITHM);

            VDISPATCH_POOLING(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_POOLING(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            const bool is_fp32 = src_md()->data_type == data_type::f32;
            const bool is_fp16 = src_md()->data_type == data_type::f16;

            VDISPATCH_POOLING(is_fp32 || is_fp16, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_POOLING(src_md()->data_type == dst_md()->data_type,
                    VERBOSE_UNSUPPORTED_DT);

            if (is_fp32 && !mayiuse(v)) return status::unimplemented;
            if (is_fp16 && !mayiuse(zvfh)) return status::unimplemented;

            VDISPATCH_POOLING(
                    memory_desc_matches_tag(*src_md(), desired_fmt_tag),
                    VERBOSE_UNSUPPORTED_TAG_S, "src");
            VDISPATCH_POOLING(
                    memory_desc_matches_tag(*dst_md(), desired_fmt_tag),
                    VERBOSE_UNSUPPORTED_TAG_S, "dst");

            return status::success;
        }
    };

    jit_rvv_nchw_pooling_fwd_t(const pd_t *apd);
    ~jit_rvv_nchw_pooling_fwd_t();

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_rvv_nchw_pool_kernel_t> kernel_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
