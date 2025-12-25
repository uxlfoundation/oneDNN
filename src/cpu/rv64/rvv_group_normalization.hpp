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
#ifndef CPU_RV64_RVV_GROUP_NORMALIZATION_HPP
#define CPU_RV64_RVV_GROUP_NORMALIZATION_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "cpu/cpu_group_normalization_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct rvv_group_normalization_fwd_t : public primitive_t {
    struct pd_t : public cpu_group_normalization_fwd_pd_t {
        using cpu_group_normalization_fwd_pd_t::
                cpu_group_normalization_fwd_pd_t;

        DECLARE_COMMON_PD_T("RISCV64GCV", rvv_group_normalization_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace format_tag;

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());

            VDISPATCH_GNORM(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_GNORM(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_GNORM((src_md()->data_type == f32
                                    && platform::has_data_type_support(
                                            src_md()->data_type)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_GNORM((dst_md()->data_type == f32
                                    && platform::has_data_type_support(
                                            dst_md()->data_type)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_GNORM((check_scale_shift_data_type()),
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "unsupported scale or shift data type");
            VDISPATCH_GNORM(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_GNORM(memory_desc_matches_one_of_tag(
                                    *src_md(), ncdhw, nchw, ncw, nc),
                    VERBOSE_UNSUPPORTED_TAG_S, "src");
            VDISPATCH_GNORM(memory_desc_matches_one_of_tag(
                                    *dst_md(), ncdhw, nchw, ncw, nc),
                    VERBOSE_UNSUPPORTED_TAG_S, "dst");
            VDISPATCH_GNORM(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            return status::success;
        }
    };

    rvv_group_normalization_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
