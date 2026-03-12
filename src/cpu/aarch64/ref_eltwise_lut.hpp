/*******************************************************************************
* Copyright 2026 Arm Ltd. and affiliates
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed tos in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef CPU_AARCH64_REF_ELTWISE_LUT_HPP
#define CPU_AARCH64_REF_ELTWISE_LUT_HPP

#include <cstdint>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_eltwise_pd.hpp"
#include "cpu/platform.hpp"

#include "cpu/primitive_attr_postops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <::dnnl::impl::data_type_t data_type>
struct ref_eltwise_lut_fwd_t : public primitive_t {
    using data_t = typename ::dnnl::impl::prec_traits_t<data_type>::type;

    struct pd_t : public cpu_eltwise_fwd_pd_t {
        using cpu_eltwise_fwd_pd_t::cpu_eltwise_fwd_pd_t;
        DECLARE_COMMON_PD_T("ref_eltwise_lut", ref_eltwise_lut_fwd_t);

        status_t init(engine_t *engine) {
            using namespace ::dnnl::impl;
            using namespace ::dnnl::impl::utils;

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());

            VDISPATCH_ELTWISE(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_ELTWISE(everyone_is(data_type, src_md()->data_type,
                                      dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(platform::has_data_type_support(data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_ELTWISE(src_d.is_dense(true), VERBOSE_NONTRIVIAL_STRIDE);
            VDISPATCH_ELTWISE(
                    src_d == dst_d, VERBOSE_INCONSISTENT_MDS, "src", "dst");
            VDISPATCH_ELTWISE(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            VDISPATCH_ELTWISE(data_type == ::dnnl::impl::data_type::bf16,
                    VERBOSE_UNSUPPORTED_DT);

            const auto *spec = get_bf16_fwd_lut_spec_(desc()->alg_kind);
            VDISPATCH_ELTWISE(spec != nullptr, VERBOSE_BAD_ALGORITHM);

            const float alpha = spec->ignore_alpha_beta ? 0.f : desc()->alpha;
            const float beta = spec->ignore_alpha_beta ? 0.f : desc()->beta;
            bf16_lut_.resize(1u << 16);
            for (uint32_t raw = 0; raw < (1u << 16); ++raw) {
                const bfloat16_t x_bf16(

                        static_cast<uint16_t>(raw), /*ignored=*/true);
                const float x = static_cast<float>(x_bf16);
                const float y = compute_eltwise_scalar_fwd(
                        desc()->alg_kind, x, alpha, beta);
                bf16_lut_[raw] = bfloat16_t(y);
            }

            return status::success;
        }

        std::vector<bfloat16_t> bf16_lut_;

    private:
        struct bf16_fwd_lut_spec_t {
            alg_kind_t alg_kind;
            bool ignore_alpha_beta;
        };

        static const bf16_fwd_lut_spec_t *get_bf16_fwd_lut_spec_(
                alg_kind_t alg_kind) {
            // Add new LUT eltwise algos
            static const bf16_fwd_lut_spec_t specs[] = {
                    {alg_kind::eltwise_gelu_erf, /*ignore_alpha_beta*/ true},
                    // SiLU is swish with alpha = 1.
                    {alg_kind::eltwise_swish, /*ignore_alpha_beta*/ false},
                    {alg_kind::eltwise_gelu_tanh, /*ignore_alpha_beta*/ true},
                    {alg_kind::eltwise_tanh, /*ignore_alpha_beta*/ true},
                    {alg_kind::eltwise_logistic, /*ignore_alpha_beta*/ true},
                    {alg_kind::eltwise_exp, /*ignore_alpha_beta*/ true},
                    {alg_kind::eltwise_log, /*ignore_alpha_beta*/ true},
                    {alg_kind::eltwise_sqrt, /*ignore_alpha_beta*/ true},
            };

            for (const auto &s : specs) {
                if (s.alg_kind == alg_kind) return &s;
            }
            return nullptr;
        }
    };

    ref_eltwise_lut_fwd_t(const pd_t *apd) : primitive_t(apd) {}
    status_t init(engine_t * /*engine*/) override { return status::success; }

    status_t execute(const exec_ctx_t & /*ctx*/) const override {
        return status::unimplemented;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_REF_ELTWISE_LUT_HPP
