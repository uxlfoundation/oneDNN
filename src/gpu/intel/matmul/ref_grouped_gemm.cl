/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "gpu/intel/include/types.h"
#if WITH_POST_OP
#include "grouped_post_ops.h"
#endif

// Grouped GEMM OCL reference kernel
//
// Per group g: C[g] = A[g] * B[g] (+ bias / scales for the 2Dx3D pattern)
//
// 2D grouped src (variable M) x 3D dense wei -> 2D grouped dst (variable M)
// partner_offsets buffer is dst_offsets
//
// 2D grouped src (variable K) x 2D grouped wei (variable M) -> dense 3D dst
// partner_offsets buffer is wei_offsets
//
// Both patterns are unified via runtime strides + per-group bases:
//   idx = base + i * stride_i + j * stride_j
//   base = group_start * group_stride
//
// get_global_id(0): group index
// get_global_id(1): output row (m)
// get_global_id(2): output column (n)
//
// Supported:
//  Data types: f32, f16, bf16
//  Row-wise (per-token) src scales; column-wise weight scales
//  Bias (shape [group_count, N])
//  Post-ops (2Dx3D only): eltwise (swish), grouped/dense/nvfp4 binary scale
__kernel void ref_grouped_gemm_matmul(__global const SRC_DATA_T *src,
        __global const int *src_offsets, __global const WEI_DATA_T *wei,
        __global const int *partner_offsets, __global DST_DATA_T *dst,
        const int group_count, const int is_2dby2d, const long m_fixed,
        const long k_fixed, const long N, const long src_stride_m,
        const long src_stride_k, const long wei_stride_k,
        const long wei_stride_n, const long dst_stride_m,
        const long dst_stride_n, const long src_group_stride,
        const long wei_group_stride, const long dst_group_stride
#if WITH_BIAS
        ,
        __global const BIA_DATA_T *bias // Bias [group_count, N]
#endif
#if WITH_SRC_SCALES
        ,
        __global const float *src_scales
#endif
#if WITH_WEI_SCALES
        ,
        __global const float *wei_scales
#endif
#if WITH_POST_OP
        ,
        __global const BINARY_SCALE_GROUPED_DATA_T *binary_grouped_scale,
        __global const BINARY_SCALE_DENSE_DATA_T *binary_dense_scale,
        __global const float *binary_nvfp4_scale
#endif
) {
    const int group_id = get_global_id(0);
    const int m = get_global_id(1);
    const int n = get_global_id(2);

    if (group_id >= group_count) return;
    if (n >= N) return;

    const int src_start = (group_id == 0) ? 0 : src_offsets[group_id - 1];
    const int src_end = src_offsets[group_id];
    const int var_extent
            = src_end - src_start; // M_g or K_g depending on the pattern

    int M_g, K_g;
    int src_group_start, wei_group_start, dst_group_start;
    if (is_2dby2d) {
        // partner_offsets == wei_offsets
        if (partner_offsets[group_id] != src_end
                || (group_id > 0 && partner_offsets[group_id - 1] != src_start))
            return;
        M_g = m_fixed;
        K_g = var_extent;
        src_group_start = src_start; // along total_K
        wei_group_start = src_start; // along total_K
        dst_group_start = group_id; // dense [G, M, N]
    } else {
        // partner_offsets == dst_offsets
        const int dst_start
                = (group_id == 0) ? 0 : partner_offsets[group_id - 1];
        const int dst_end = partner_offsets[group_id];
        if (dst_end - dst_start != var_extent) return;
        M_g = var_extent;
        K_g = k_fixed;
        src_group_start = src_start; // along total_M
        wei_group_start = group_id; // dense [G, K, N]
        dst_group_start = dst_start; // along total_M
    }

    // Note, that K_g == 0 must still write zeros
    if (M_g == 0 || m >= M_g) return;

    const long src_base = (long)src_group_start * src_group_stride;
    const long wei_base = (long)wei_group_start * wei_group_stride;
    const long dst_base = (long)dst_group_start * dst_group_stride;

    ACC_DATA_T acc = (ACC_DATA_T)0;
    for (int k = 0; k < K_g; k++) {
        const long src_idx
                = src_base + (long)m * src_stride_m + (long)k * src_stride_k;
        const long wei_idx
                = wei_base + (long)k * wei_stride_k + (long)n * wei_stride_n;
        acc += SRC_TO_REF(src[src_idx]) * WEI_TO_REF(wei[wei_idx]);
    }

#if WITH_SRC_SCALES
    // Row-wise (per global token) src scale; global token = src_group_start + m.
    acc *= (ACC_DATA_T)src_scales[src_group_start + m];
#endif
#if WITH_WEI_SCALES
    // Column-wise weight scale, shape [group_count, N].
    acc *= (ACC_DATA_T)wei_scales[(long)group_id * N + n];
#endif
#if WITH_BIAS
    acc += BIA_TO_REF(bias[(long)group_id * N + n]);
#endif

#if WITH_POST_OP
    // Post-ops apply to the 2Dx3D pattern only; there partner_offsets is
    // dst_offsets (cumulative dst token starts per group).
    acc = apply_post_ops_chain(acc, m, n, group_id, N, partner_offsets,
            binary_grouped_scale, binary_dense_scale, binary_nvfp4_scale);
#endif

    const long dst_idx
            = dst_base + (long)m * dst_stride_m + (long)n * dst_stride_n;
    dst[dst_idx] = REF_TO_DST(acc);
}
