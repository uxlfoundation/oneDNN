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

#ifndef CPU_RV64_JIT_RVV_GEMM_CONVOLUTION_POST_KERNEL_HPP
#define CPU_RV64_JIT_RVV_GEMM_CONVOLUTION_POST_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "cpu/rv64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct jit_rvv_gemm_convolution_post_kernel_t : public jit_generator_t {
    struct call_params_t {
        float *dst;
        const float *bias;
        dim_t len;
        float scalar_bias;
        float relu_alpha;
        float scale;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_gemm_convolution_post_kernel_t)

    jit_rvv_gemm_convolution_post_kernel_t(bool vector_bias, bool with_relu,
            bool relu_alpha_zero, bool with_scale);

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    bool vector_bias_;
    bool with_relu_;
    bool relu_alpha_zero_;
    bool with_scale_;
};

void jit_rvv_gemm_convolution_apply_bias(
        float *dst, const float *bias, dim_t len);

void jit_rvv_gemm_convolution_apply_scalar_bias(
        float *dst, dim_t len, float bias);

void jit_rvv_gemm_convolution_apply_scalar_bias_relu(
        float *dst, dim_t len, float bias, float relu_alpha, float scale);

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_JIT_RVV_GEMM_CONVOLUTION_POST_KERNEL_HPP
