/*******************************************************************************
* Copyright 2026 SpacemiT Corporation
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

#ifndef CPU_RV64_RVV_DWCONV_HPP
#define CPU_RV64_RVV_DWCONV_HPP

#include "common/primitive.hpp"
#include "common/utils.hpp"
#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct rvv_dwconv_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T("dw_k3s1:rvv", rvv_dwconv_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace format_tag;

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper wei_d(weights_md(0));
            const memory_desc_wrapper bia_d(weights_md(1));
            const memory_desc_wrapper dst_d(dst_md());
            const auto *conv_desc = this->desc();
            const int gr_shift = with_groups() ? 1 : 0;
            const int nd = src_d.ndims();
            const int64_t i = wei_d.dims()[gr_shift + 1];
            const int64_t kh = wei_d.dims()[gr_shift + 2];
            const int64_t kw = wei_d.dims()[gr_shift + 3];
            const int64_t stride_h = conv_desc->strides[0];
            const int64_t stride_w = conv_desc->strides[1];
            const int64_t dilate_h = conv_desc->dilates[0];
            const int64_t dilate_w = conv_desc->dilates[1];

            VDISPATCH_CONV(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(with_groups() && IC() == G(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(platform::has_data_type_support(src_d.data_type()),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(src_d.data_type() == f16, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(wei_d.data_type() == src_d.data_type(),
                    VERBOSE_INCONSISTENT_DT, "src", "weights");
            VDISPATCH_CONV(dst_d.data_type() == src_d.data_type(),
                    VERBOSE_INCONSISTENT_DT, "src", "dst");
            VDISPATCH_CONV(IMPLICATION(with_bias(),
                                   utils::one_of(bia_d.data_type(), f16, f32)),
                    VERBOSE_UNSUPPORTED_BIAS_CFG);
            VDISPATCH_CONV(i == 1, VERBOSE_BAD_DIM, "weights", gr_shift + 1);
            VDISPATCH_CONV(kh == 3 && kw == 3, VERBOSE_BAD_DIM, "weights",
                    gr_shift + 2);
            VDISPATCH_CONV(stride_h == 1 && stride_w == 1, VERBOSE_BAD_DIM);
            VDISPATCH_CONV(dilate_h == 0 && dilate_w == 0, VERBOSE_BAD_DIM);

            VDISPATCH_CONV(memory_desc_matches_tag(*src_md(0), nhwc),
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_CONV(memory_desc_matches_tag(*dst_md(0), nhwc),
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_CONV(memory_desc_matches_tag(*weights_md(0), goihw),
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_CONV(IMPLICATION(with_bias(),
                                   memory_desc_matches_tag(*weights_md(1), x)),
                    VERBOSE_UNSUPPORTED_TAG);

            return status::success;
        }
    };

    rvv_dwconv_fwd_t(const pd_t *apd) : primitive_t(apd) {}
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const {
        return static_cast<const pd_t *>(primitive_t::pd().get());
    }
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_RVV_DWCONV_HPP
