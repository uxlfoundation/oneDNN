/*******************************************************************************
* Copyright 2025 Arm Ltd. and affiliates
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

#include <mutex>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/bfloat16.hpp"
#include "common/dnnl_thread.hpp"

#include "cpu/platform.hpp"
#include "cpu/primitive_attr_postops.hpp"
#include "cpu/cpu_eltwise_pd.hpp"

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
            VDISPATCH_ELTWISE(set_default_formats_common(),
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_ELTWISE(src_d == dst_d, VERBOSE_INCONSISTENT_MDS, "src", "dst");
            VDISPATCH_ELTWISE(
                    attr_.set_default_formats(dst_md(0)) == status::success,
                    VERBOSE_UNSUPPORTED_POSTOP);

            // Only bf16 + GELU(erf) is implemented here.
            if (desc()->alg_kind != ::dnnl::impl::alg_kind::eltwise_gelu_erf)
                return status::unimplemented;
            if (data_type != ::dnnl::impl::data_type::bf16)
                return status::unimplemented;

            return status::success;
        }
    };

    ref_eltwise_lut_fwd_t(const pd_t *apd) : primitive_t(apd) {}
    status_t init(engine_t * /*engine*/) override { return status::success; }

    status_t execute(const exec_ctx_t & /*ctx*/) const override {
        return status::unimplemented;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    void maybe_build_gelu_bf16_lut_() const {
        std::call_once(gelu_bf16_once_, [&]() {
            constexpr float inv_sqrt2 = 0.70710678118654752440f; // 1/sqrt(2)
            // gelu_bf16_lut_.resize(1u << 16);
            for (uint32_t i = 0; i < (1u << 16); ++i) {
                // Expand bf16 payload to f32
                const uint32_t expanded = (i << 16);
                float x;
                std::memcpy(&x, &expanded, sizeof(float));
                const float y = x * 0.5f * (1.0f + std::erf(x * inv_sqrt2));
                gelu_bf16_lut_[i] = data_t(y);
            }
        });
    }

    // Per-primitive-instance storage.
    mutable std::once_flag gelu_bf16_once_;
    // mutable std::vector<data_t> gelu_bf16_lut_;
   mutable  bfloat16_t gelu_bf16_lut_[1u << 16];
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_REF_ELTWISE_LUT_HPP