/*******************************************************************************
* Copyright 2026 Advanced Micro Devices, Inc.
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

#ifndef CPU_REF_DYNAMIC_QUANTIZE_HPP
#define CPU_REF_DYNAMIC_QUANTIZE_HPP

#include "common/primitive.hpp"
#include "common/type_helpers.hpp"

#include "cpu/cpu_reduction_pd.hpp"
#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct ref_dynamic_quantize_reduction_t : public primitive_t {
    struct pd_t : public cpu_reduction_pd_t {
        using cpu_reduction_pd_t::cpu_reduction_pd_t;

        DECLARE_COMMON_PD_T(
                "ref:dynamic_quantize", ref_dynamic_quantize_reduction_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            VDISPATCH_REDUCTION(is_dynamic_quantize(), VERBOSE_BAD_ALGORITHM);
            VDISPATCH_REDUCTION(
                    platform::has_data_type_support(src_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_REDUCTION(
                    platform::has_data_type_support(scale_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_REDUCTION(set_default_params() == status::success,
                    VERBOSE_UNSUPPORTED_TAG);

            return status::success;
        }
    };

    ref_dynamic_quantize_reduction_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_ref(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_ref(const exec_ctx_t &ctx) const;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
