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

#ifndef CPU_X64_DYNAMIC_QUANTIZE_HPP
#define CPU_X64_DYNAMIC_QUANTIZE_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"

#include "cpu/cpu_reduction_pd.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

// Optimized x64 implementation of reduction_dynamic_quantize. Wraps the
// AVX-512 fused per-token and per-group INT8 kernels ported from ZenDNN.
// Non-matching configurations (other granularities, compute-only, non-2D,
// non-dense layouts) return unimplemented so the portable reference handles
// them.
struct dynamic_quantize_reduction_t : public primitive_t {
    struct pd_t : public cpu_reduction_pd_t {
        using cpu_reduction_pd_t::cpu_reduction_pd_t;

        DECLARE_COMMON_PD_T(
                "avx512_core:dynamic_quantize", dynamic_quantize_reduction_t);

        status_t init(engine_t *engine);

        // Granularity handled by the ported kernels.
        enum class granularity_t { per_token, per_group };

        granularity_t gran_ = granularity_t::per_token;
        dim_t M_ = 0;
        dim_t N_ = 0;
        dim_t G_ = 1; // number of column groups for per-group mode
        cpu_isa_t isa_ = isa_undef;
    };

    dynamic_quantize_reduction_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const {
        return static_cast<const pd_t *>(primitive_t::pd().get());
    }
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
