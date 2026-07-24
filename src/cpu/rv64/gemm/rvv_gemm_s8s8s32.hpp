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

#ifndef CPU_RV64_GEMM_RVV_GEMM_S8S8S32_HPP
#define CPU_RV64_GEMM_RVV_GEMM_S8S8S32_HPP

#include "common/c_types_map.hpp"

#include "cpu/rv64/gemm/rvv_gemm_utils_f32.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

// RVV int8 GEMM driver. Mirrors rvv_gemm_f32() in structure but accepts s8
// weights, s8/u8 src, and a wider set of dst types: s32, f32, s8, u8, f16,
// bf16. The dst element width is no longer uniform (1, 2 or 4 bytes), so
// `dst_dt` selects both the type and the element size; `dst_dt_sz` is derived
// internally.
//
// Per-destination alpha/beta contract enforced by the driver:
//   - s32   : alpha must be 1; beta in {0, 1}. K-split (nthr_k > 1) is
//             supported because the JIT kernel implements the
//             read-modify-write epilogue for s32.
//   - f32   : any alpha, any beta. K-split supported.
//   - s8    : alpha must be 1; beta must be 0. K-split is forced off
//             (nthr_k = 1) because the JIT s8 epilogue is overwrite-only.
//   - u8    : same as s8.
//   - f16   : any alpha; beta must be 0. K-split forced off. Requires
//             Zvfh; caller (matmul primitive) gates on mayiuse(zvfh).
//   - bf16  : any alpha; beta must be 0. K-split forced off. Requires
//             Zvfbfwma; caller gates on mayiuse(zvfbfwma).
//
// b_signed selects between s8 and u8 on the B (src) axis; A (weights) is
// always s8.
//
// `bias` is an optional f32 vector of length M, broadcast across the N axis.
// When non-null it is fused into the JIT kernel's C-update phase, matching the
// f32 GEMM kernel convention. `bias_is_scalar` must be set when `bias` is a
// single value that broadcasts over M (bia_mask=0 / last dim == 1): the kernel
// then splats one float instead of reading a full vector.
//
// `part` optionally supplies the thread partition computed at primitive
// initialization (see gemm_utils::gemm_partition_t). When provided, the driver
// reuses it instead of recomputing from dnnl_get_current_num_threads(), so the
// per-thread workspace offsets stay consistent with the scratchpad capacity
// booked at init. Pass nullptr to recompute and malloc (inner_product / conv).
//
// Scratchpad contract mirrors rvv_gemm_f32(). c_buffers carries whatever the
// kernel's C-update epilogue writes, which is dst_dt_sz bytes/element:
//   - dst_dt == s32  : 4 bytes/element raw s32 acc
//   - dst_dt == f32  : 4 bytes/element f32
//   - dst_dt == f16  : 2 bytes/element f16
//   - dst_dt == bf16 : 2 bytes/element bf16
//   - dst_dt == s8   : 1 byte/element  s8 (beta == 0 only)
//   - dst_dt == u8   : 1 byte/element  u8 (beta == 0 only)
// ws_buffers holds int8 elements (one per-thread A-copy cache). Pass nullptr
// for either to fall back to malloc/free inside the function.
//
// `dst_dt` must be one of: data_type::s32, f32, s8, u8, f16, bf16. Callers
// (matmul primitive) gate f16 on mayiuse(zvfh) and bf16 on mayiuse(zvfbfwma)
// before passing them through.
status_t rvv_gemm_s8s8s32(const char *transa, const char *transb,
        const dim_t *M, const dim_t *N, const dim_t *K, const float *alpha,
        const int8_t *A, const dim_t *lda, const void *B, const dim_t *ldb,
        const float *beta, void *C, const dim_t *ldc, const float *bias,
        bool b_signed, data_type_t dst_dt, int32_t *c_buffers = nullptr,
        int8_t *ws_buffers = nullptr, bool bias_is_scalar = false,
        const gemm_utils::gemm_partition_t *part = nullptr);

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_GEMM_RVV_GEMM_S8S8S32_HPP
