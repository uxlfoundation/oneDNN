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

#ifndef CPU_RV64_GEMM_JIT_RVV_GEMM_F16_KERNEL_HPP
#define CPU_RV64_GEMM_JIT_RVV_GEMM_F16_KERNEL_HPP

#include "common/c_types_map.hpp"

#include "cpu/rv64/jit_generator.hpp"

#include <array>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace gemm_utils {

// RVV JIT micro-kernel for half-precision GEMM (f16 or bf16 inputs, dst of the
// same type) on RV64.
//
// Computes an m x n_cols tile of:
//   C[0:m, 0:n_cols] = A[0:m, 0:K] * B[0:K, 0:n_cols]
// with f32 accumulation; the accumulators are narrowed to the input data type
// once at the C store (overwrite-only, matching the epilogue contract of the
// s8 GEMM kernel for narrow dst types — the driver rejects beta != 0).
//
// Design mirrors jit_rvv_gemm_kernel_t (f32): outer-product micro-kernel with
// LMUL=m4 f32 accumulators and n_cols (1..6) broadcast B scalars; A rows are
// loaded at SEW=e16 LMUL=m2 (vfwmacc.vf for f16 / vfwmaccbf16.vf for bf16,
// gated by the caller), double-buffered to overlap the A load with FMAs.
//
// Vector register layout:
//   v0..v23  : six f32 accumulator groups (LMUL=m4, one per column)
//   v24..v25 : A double-buffer 0 (e16/m2; also reused as epilogue temporary)
//   v28..v29 : A double-buffer 1 (e16/m2)
struct jit_rvv_gemm_f16_kernel_t : public jit_generator_t {
    struct call_params_t {
        const void *A; // f16/bf16 elements
        const void *B; // f16/bf16 elements
        void *C; // same data type as A/B
        dim_t lda;
        dim_t ldb;
        dim_t ldc;
        dim_t K;
        dim_t m;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_gemm_f16_kernel_t)

    // Construct a JIT kernel for a specific n_cols (1..6) and A access pattern.
    // in_dt selects f16 (Zvfh) vs bf16 (Zvfbfwma); the caller gates the ISA.
    jit_rvv_gemm_f16_kernel_t(dim_t n_cols, bool isTransA, data_type_t in_dt);

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    dim_t n_cols_;
    bool isTransA_;
    bool is_bf16_;
};

struct jit_rvv_gemm_f16_kernel_table_t {
    std::array<const jit_rvv_gemm_f16_kernel_t *, 8> nb {};
};

const jit_rvv_gemm_f16_kernel_table_t &get_jit_rvv_gemm_f16_kernel_table(
        bool isTransA, data_type_t in_dt);

} // namespace gemm_utils
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
