/*******************************************************************************
* Copyright 2026 Arm Ltd. and affiliates
* Copyright 2026 Intel Corporation
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

#ifndef CPU_AARCH64_ELTWISE_LUT_HPP
#define CPU_AARCH64_ELTWISE_LUT_HPP

#include <vector>

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_eltwise_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

// This implementation precomputes eltwise results for all possible bf16 inputs
// at primitive creation time. This makes execution a table lookup, but increases
// initialization time and stores one fixed-size 128 KiB LUT per primitive.
struct eltwise_lut_fwd_t : public primitive_t {
    using data_t = ::dnnl::impl::bfloat16_t;

    struct pd_t : public cpu_eltwise_fwd_pd_t {
        using cpu_eltwise_fwd_pd_t::cpu_eltwise_fwd_pd_t;
        DECLARE_COMMON_PD_T("lut", eltwise_lut_fwd_t);

        status_t init(const engine_t *engine) {
            using namespace ::dnnl::impl;
            using namespace ::dnnl::impl::utils;

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());

            VDISPATCH_ELTWISE(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_ELTWISE(everyone_is(data_type::bf16, src_md()->data_type,
                                      dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_ELTWISE(src_d.is_dense(false), VERBOSE_NONTRIVIAL_STRIDE);
            VDISPATCH_ELTWISE(
                    src_d == dst_d, VERBOSE_INCONSISTENT_MDS, "src", "dst");
            VDISPATCH_ELTWISE(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            VDISPATCH_ELTWISE(check_alg_kind(), VERBOSE_BAD_ALGORITHM);

            return status::success;
        }

    private:
        bool check_alg_kind() const {
            using namespace ::dnnl::impl::alg_kind;
            return utils::one_of(desc()->alg_kind, eltwise_gelu_erf,
                    eltwise_swish, eltwise_gelu_tanh, eltwise_tanh,
                    eltwise_logistic, eltwise_exp, eltwise_log, eltwise_sqrt);
        }
    };

    eltwise_lut_fwd_t(const pd_t *apd) : primitive_t(apd) {}
    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::vector<data_t> lut_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ELTWISE_LUT_HPP
