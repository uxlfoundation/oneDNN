/*******************************************************************************
* Copyright 2026 ZTE Corporation
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

#ifndef CPU_RV64_JIT_RVV_POOLING_KERNEL_HPP
#define CPU_RV64_JIT_RVV_POOLING_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "cpu/rv64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct jit_rvv_pooling_fwd_kernel_t : public jit_generator_t {
    struct call_params_t {
        const float *src;
        float *dst;
        dim_t channels;
        dim_t id_start, ih_start, iw_start;
        dim_t id_end, ih_end, iw_end;
        dim_t inW_stride; // IW * C (elements)
        dim_t inD_stride; // IH * IW * C (elements)
        float init_val; // -FLT_MAX for max, 0.0 for avg
        float scale_val; // 1.0/count for avg, unused for max
        float relu_alpha;
        bool with_relu;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_pooling_fwd_kernel_t)

    enum class alg_t : int { max_pool, avg_include, avg_exclude };

    jit_rvv_pooling_fwd_kernel_t(alg_t alg);

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    alg_t alg_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_JIT_RVV_POOLING_KERNEL_HPP
