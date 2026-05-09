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

// RVV JIT micro-kernel for f32 GEMM on RV64.
//
// Computes an m x n_cols tile of:
//   C[0:m, 0:n_cols] = alpha * A[0:m, 0:K] * B[0:K, 0:n_cols]
//                    + beta * C[0:m, 0:n_cols]
//
// Design choices:
//   - LMUL is fixed to m4 (4 vector registers per group)
//   - n_cols is fixed at JIT compile time (1..7), determining the number
//     of accumulator register groups emitted
//   - m (tile height) is a runtime parameter; the JIT code uses vsetvli
//     to set VL accordingly, so any m <= VLEN/32*4 is supported
//   - isTransA/isTransB determine A/B memory access patterns
//
// Vector register layout (LMUL=m4, 8 groups of 4 regs):
//   v0..v3   : accumulator c0 (column 0)
//   v4..v7   : accumulator c1 (column 1)
//   v8..v11  : accumulator c2 (column 2)
//   v12..v15 : accumulator c3 (column 3)
//   v16..v19 : accumulator c4 (column 4)
//   v20..v23 : accumulator c5 (column 5)
//   v24..v27 : accumulator c6 (column 6)
//   v28..v31 : temporary for A loads and C update
//
// When n_cols < 7, only the first n_cols accumulator groups are used.
struct jit_rvv_gemm_kernel_t : public jit_generator_t {
    struct call_params_t {
        const float *A;
        const float *B;
        float *C;
        dim_t lda;
        dim_t ldb;
        dim_t ldc;
        dim_t K;
        dim_t m;
        float alpha;
        float beta;
        const float *bias;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_gemm_kernel_t)

    // Construct a JIT kernel for a specific n_cols (1..7), transpose modes,
    // and optional fused-bias support.
    jit_rvv_gemm_kernel_t(
            dim_t n_cols, bool isTransA, bool isTransB, bool has_bias);

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    dim_t n_cols_;
    bool isTransA_;
    bool isTransB_;
    bool has_bias_;
};

} // namespace gemm_utils
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
