/*******************************************************************************
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
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

#ifndef CPU_RV64_JIT_RVV_BATCH_NORMALIZATION_KERNEL_HPP
#define CPU_RV64_JIT_RVV_BATCH_NORMALIZATION_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "cpu/rv64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct jit_rvv_batch_normalization_fwd_kernel_t : public jit_generator_t {
    struct call_params_t {
        const float *src;
        float *dst;
        dim_t len;
        const float *mean;
        const float *scale_mul;
        const float *scale_add;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_batch_normalization_fwd_kernel_t)

    jit_rvv_batch_normalization_fwd_kernel_t(
            bool per_elem_params, bool with_relu);

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    bool per_elem_params_;
    bool with_relu_;
};

void jit_rvv_batch_normalization_apply_f32(const float *src, float *dst,
        dim_t len, const float *mean, const float *scale_mul,
        const float *scale_add, bool per_elem_params, bool with_relu);

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_JIT_RVV_BATCH_NORMALIZATION_KERNEL_HPP
