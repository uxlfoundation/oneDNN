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

// Grouped GEMM OCL reference kernel
//
// For each group g:
//  C[g] = A[g] * B[g] + bias[g],
//  where A[g] is M_g x K, B[g] is K x N, C[g] is M_g x N
//
// Memory layout:
//  A/src: [total_tokens, K] with grouped encoding (values in buffer 0, offsets in buffer 1)
//  B/wei: [num_experts, K, N] dense 3D tensor
//  C/dst: [total_tokens, N] with grouped encoding (values in buffer 0, offsets in buffer 1)
//
// offsets is an array of size group_count, with cumulative values [M_0, M_0 + M_1, M_0 + M_1 + M_2, ...]
// Note, that M_g can be zero for some groups
//
// get_global_id(0): group index
// get_global_id(1): output row (M dimension within group)
// get_global_id(2): output column (N dimension)

// Supported below:
//  Data types: f32, f16, bf16
//  Row-wise (per-token) src scales
//  Column-wise weight scales
//  Bias addition (shape [group_count, N])
__kernel void ref_grouped_gemm_matmul(
        __global const SRC_DATA_T *src, // Buffer 0: concatenated values
        __global const int
                *src_offsets, // Buffer 1: group offsets [group_count]
        __global const WEI_DATA_T *wei, // Dense weights [group_count, K, N]
        __global DST_DATA_T *dst, // Buffer 0: concatenated output values
        __global const int
                *dst_offsets, // Buffer 1: output group offsets [group_count]
        const int group_count // Number of groups
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
) {

    const int group_id = get_global_id(0);
    const int m = get_global_id(1);
    const int n = get_global_id(2);

    const int src_start = (group_id == 0) ? 0 : src_offsets[group_id - 1];
    const int src_end = src_offsets[group_id];
    const int M = src_end - src_start;

    if (group_id >= group_count) return;
    if (n >= N) return;
    if (K <= 0) return;
    if (m >= M) return; // upper bound is currently very large (total_tokens)
    if (M <= 0) return; // skip empty or invalid groups

    const int dst_start = (group_id == 0) ? 0 : dst_offsets[group_id - 1];
    const int dst_end = dst_offsets[group_id];
    const int dst_M = dst_end - dst_start;

    if (dst_M != M)
        return; // src and dst must have same token count per group for now

    const long src_offset = (long)src_start * K;
    const long wei_offset = (long)group_id * K * N;
    const long dst_offset = (long)dst_start * N;

    __global const SRC_DATA_T *src_group = src + src_offset;
    __global const WEI_DATA_T *wei_group = wei + wei_offset;
    __global DST_DATA_T *dst_group = dst + dst_offset;

    ACC_DATA_T acc = (ACC_DATA_T)0;
    for (int k = 0; k < K; k++) {
        const long src_idx = (long)m * K + k;
#if WEI_TRANSPOSED
        const long wei_idx = (long)n * K + k;
#else
        const long wei_idx = (long)k * N + n;
#endif
        ACC_DATA_T src_val = SRC_TO_REF(src_group[src_idx]);
        ACC_DATA_T wei_val = WEI_TO_REF(wei_group[wei_idx]);
        acc += src_val * wei_val;
    }

    // Apply row-wise src scale
#if WITH_SRC_SCALES
    const int token_idx = src_start + m;
    const float src_scale = src_scales[token_idx];
    acc *= (ACC_DATA_T)src_scale;
#endif

    // Apply column-wise weight scale
#if WITH_WEI_SCALES
    const long wei_scale_idx = (long)group_id * N + n;
    const float wei_scale = wei_scales[wei_scale_idx];
    acc *= (ACC_DATA_T)wei_scale;
#endif

#if WITH_BIAS
    const long bias_idx = (long)group_id * N + n;
    acc += BIA_TO_REF(bias[bias_idx]);
#endif

    const long out_idx = (long)m * N + n;

    dst_group[out_idx] = REF_TO_DST(acc);
}
