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

#ifndef CPU_RV64_GEMM_JIT_RVV_GEMM_S8_KERNEL_HPP
#define CPU_RV64_GEMM_JIT_RVV_GEMM_S8_KERNEL_HPP

#include "cpu/rv64/jit_generator.hpp"

#include <array>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace gemm_utils {

// RVV JIT micro-kernel for int8 GEMM on RV64.
//
// Computes an m x n_cols tile of:
//   C[0:m, 0:n_cols] = sum_k A[0:m, 0:K] * B[0:K, 0:n_cols]
//                    + (has_bias ? bias[0:m] : 0)
//
// Vector register layout:
//   v0..v3    accumulator c0 (e32, LMUL=m4)
//   v4..v7    accumulator c1
//   v8..v11   accumulator c2
//   v12..v15  accumulator c3
//   v16..v19  accumulator c4
//   v20..v23  accumulator c5
//   v24       A row buffer in e8 (LMUL=m1) and e16 (LMUL=m2, overlaps v24-v25)
//   v26       scratch / C-load temporary
struct jit_rvv_gemm_s8_kernel_t : public jit_generator_t {
    struct call_params_t {
        const void *A; // weights, s8 (M x K GEMM A axis)
        const void *B; // src, s8 or u8 (K x N GEMM B axis)
        void *C; // dst, s32 or f32
        dim_t lda;
        dim_t ldb;
        dim_t ldc;
        dim_t K;
        dim_t m;
        float alpha; // applied only when dst_is_f32
        float beta; // applied only when dst_is_f32
        const float *bias; // optional, f32; broadcast per-row
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_gemm_s8_kernel_t)

    jit_rvv_gemm_s8_kernel_t(dim_t n_cols, bool isTransA, bool isTransB,
            bool b_signed, bool dst_is_f32, bool has_bias);

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    dim_t n_cols_;
    bool isTransA_;
    bool isTransB_;
    bool b_signed_; // true: B is s8; false: B is u8
    bool dst_is_f32_; // true: dst is f32; false: dst is s32
    bool has_bias_;
};

struct jit_rvv_gemm_s8_kernel_table_t {
    std::array<const jit_rvv_gemm_s8_kernel_t *, 8> nb {};
    std::array<const jit_rvv_gemm_s8_kernel_t *, 8> b {};
};

const jit_rvv_gemm_s8_kernel_table_t &get_jit_rvv_gemm_s8_kernel_table(
        bool isTransA, bool isTransB, bool b_signed, bool dst_is_f32);

} // namespace gemm_utils
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
