/*******************************************************************************
* Copyright 2025 ZTE Corporation
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

#ifndef CPU_RV64_GEMM_JIT_RVV_GEMM_KERNEL_HPP
#define CPU_RV64_GEMM_JIT_RVV_GEMM_KERNEL_HPP

#include "cpu/rv64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace gemm_utils {

// RVV JIT micro-kernel for f32 GEMM on RV64, implementing the same tile shape
// as rvv_gemm_f32::kernel_mxn for the most important case:
//   - isTransA = false
//   - isTransB = false
//
// It computes a single 8x4 block (m = 8 rows, n = 4 columns) of:
//
//   C[0:8, 0:4] = alpha * A[0:8, 0:K] * B[0:K, 0:4] + beta * C[0:8, 0:4]
//
// using RVV vectorization over the M dimension and a 4-way unrolled K-loop
// with a software-pipelined load/FMA schedule to better hide vector/FMA
// latency.
struct jit_rvv_gemm_kernel_t : public jit_generator_t {
    struct call_params_t {
        const float *A;
        const float *B;
        float *C;
        dim_t lda;
        dim_t ldb;
        dim_t ldc;
        dim_t K;
        float alpha;
        float beta;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_gemm_kernel_t)

    jit_rvv_gemm_kernel_t();

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;
};

} // namespace gemm_utils
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
