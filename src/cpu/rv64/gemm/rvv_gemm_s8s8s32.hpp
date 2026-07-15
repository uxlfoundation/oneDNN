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
// weights, s8/u8 src, and s32/f32 dst. The dst element width is 4 bytes either
// way; pass `dst_is_f32 = true` to request an f32 epilogue (alpha/beta applied
// after fcvt of the s32 accumulator) or `false` to write the raw s32
// accumulator. The s32 path only implements alpha == 1 and beta in {0, 1}
// (beta == 1 is the read-modify-write vsadd that lets K-tile accumulation in
// the wrapper work); anything else is rejected as unimplemented rather than
// silently mishandled.
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
// kernel's C-update epilogue writes, which is 4 bytes/element in either case:
//   - dst_is_f32 == false: raw s32 accumulators
//   - dst_is_f32 == true : f32 values, already fcvt-converted and alpha-scaled
//     in-kernel (so the K-split reduction below must sum them as float, not
//     reinterpret the bits as int32)
// ws_buffers holds int8 elements (one per-thread A-copy cache). Pass nullptr
// for either to fall back to malloc/free inside the function.
status_t rvv_gemm_s8s8s32(const char *transa, const char *transb,
        const dim_t *M, const dim_t *N, const dim_t *K, const float *alpha,
        const int8_t *A, const dim_t *lda, const void *B, const dim_t *ldb,
        const float *beta, void *C, const dim_t *ldc, const float *bias,
        bool b_signed, bool dst_is_f32, int32_t *c_buffers = nullptr,
        int8_t *ws_buffers = nullptr, bool bias_is_scalar = false,
        const gemm_utils::gemm_partition_t *part = nullptr);

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_GEMM_RVV_GEMM_S8S8S32_HPP
