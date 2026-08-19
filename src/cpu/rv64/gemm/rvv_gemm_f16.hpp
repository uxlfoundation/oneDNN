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

#ifndef CPU_RV64_GEMM_RVV_GEMM_F16_HPP
#define CPU_RV64_GEMM_RVV_GEMM_F16_HPP

#include "common/c_types_map.hpp"

#include "cpu/rv64/gemm/rvv_gemm_utils_f32.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

// rvv_gemm_f16 computes C = A * B with f32 accumulation, where A, B and C all
// hold `dt` elements (f16 or bf16); the accumulators are narrowed to `dt` once
// at the store. The JIT epilogue is overwrite-only, so alpha must be 1 and
// beta must be 0 (unimplemented is returned otherwise), and K is never split
// across threads or blocks — the kernel accumulates the full reduction before
// the single store.
//
// transb must be 'N'/'n' (the matmul mapping never transposes the src side);
// transa may be 'N' (row-major weights) or 'T' (col-major weights).
//
// `part` optionally supplies the thread partition computed at primitive
// initialization (see gemm_utils::gemm_partition_t); nthr_k is forced to 1.
// `ws_buffers` holds the per-thread A-copy workspace (char elements) booked
// in the caller's scratchpad; unimplemented is returned when the A copy is
// active for the computed partition and no workspace is provided.
status_t rvv_gemm_f16(const char *transa, const char *transb, const dim_t *M,
        const dim_t *N, const dim_t *K, const float *alpha, const void *A,
        const dim_t *lda, const void *B, const dim_t *ldb, const float *beta,
        void *C, const dim_t *ldc, data_type_t dt, char *ws_buffers = nullptr,
        const gemm_utils::gemm_partition_t *part = nullptr);

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_GEMM_RVV_GEMM_F16_HPP
