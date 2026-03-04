/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include "gpu/intel/bnorm/xe.h"
#include "gpu/intel/bnorm/xe_reduce.h"
#include "gpu/intel/include/io.h"

// BWD kernels that support both blocked and NHWC layouts (USE_NHWC definition).
// These kernels perform IC tail processing for NHWC and for ic % 8 == 0
// cases only.
// Two types of reduction are implemented:
// 1) Reduction over scratchpad (reduce_temp) with SLM use, implemented by
//    xe_reduce_stats kernel.
// 2) Atomics-based reduction with SLM use (FUSED_ATOMICS_REDUCTION definition),
//    implemented as part of calc kernels, see xe_*_fused_reduction()
//    functions in xe.h. This reduction implementation requires zeroing and
//    finalization steps, see xe_fused_reduce_* kernels in xe_reduce.cl

#define LOAD_Nx16_USING_LOOP(n, dest, src) \
    { \
        for (int k = 0; k < n; ++k) { \
            dest[k] = block_load(dest[k], &src[k * IC_BLOCK_STRIDE]); \
        } \
    }

#define LOAD_8x16_USING_LAYOUT(dest, src) \
    { \
        if (USE_NHWC) { \
            LOAD_Nx16_USING_LOOP(8, dest, src); \
        } else { \
            dest = block_load(dest, src); \
        } \
    }

#define LOAD_Nx16_USING_LOOP_HALF(n, dest, src) \
    { \
        for (int k = 0; k < n; k += 2) { \
            dest[k] = block_load(dest[k], &src[k * IC_BLOCK_STRIDE]); \
        } \
    }

NAMED_KERNEL_ATTR(CALC)
__kernel void xe_calculate_stats(__global DATA_T *src, __global float *mean,
        __global DATA_T *diff_dst, __global char *ws,
        __global float *temp_reduce, volatile __global atomic_float *diff_scale,
        volatile __global atomic_float *diff_shift) {

    const int mb = GWS_GET_STAT_MB();
    const int c = GWS_GET_STAT_IC();
    const int sp_block_idx = GWS_GET_STAT_SP();
    const int mb_sp_idx = mb * STAT_SP_NBLOCKS + sp_block_idx;
    const int group_c_offset = REDUCE_STAT_NBLOCKS * 16 * (int)(c / 16);
    const int simd_id = get_sub_group_local_id();
#if HAS_IC_TAIL
    const bool is_last_ic_block = c + 16 > IC;
    const bool is_last_sp_block = (sp_block_idx == STAT_SP_NBLOCKS - 1);
#endif

    temp_reduce += group_c_offset;

#if USE_NHWC
    const int offset = c + sp_block_idx * STAT_SP_BLOCK * IC;
#else
    const int offset = (c & 15) + sp_block_idx * STAT_SP_BLOCK * 16
            + (c & ~15) * SP + mb * SP * IC;
#endif
    src += offset;
    diff_dst += offset;
    ws += offset;

    float v_mean;
    MAYBE_LAST_IC_BLOCK_LOAD_1x16(v_mean, mean, c);

    float8 diff_gamma = 0.0f;
    float8 diff_beta = 0.0f;

#if HAS_STAT_SP_TAIL
    int sp;
    if (sp_block_idx == STAT_SP_TAIL) {
        sp = SP - STAT_SP_TAIL * STAT_SP_BLOCK;
    } else {
        sp = STAT_SP_BLOCK;
    }
#else
    int sp = STAT_SP_BLOCK;
#endif

    const int C_PARALLEL_FACTOR = 8;

    for (; sp > C_PARALLEL_FACTOR - 1; sp -= C_PARALLEL_FACTOR) {
        float8 src_data;
        float8 dd_data;

#if FUSE_BN_RELU == 1
        char8 ws_data;
        LOAD_8x16_USING_LAYOUT(ws_data, ws);
#endif // #if FUSE_BN_RELU == 1

#if IS_IC_EQ_8
        LOAD_Nx16_USING_LOOP_HALF(8, src_data, src);
        LOAD_Nx16_USING_LOOP_HALF(8, dd_data, diff_dst);
        float8 t_src = intel_sub_group_shuffle_down(src_data, src_data, 8);
        float8 t_dd = intel_sub_group_shuffle_down(dd_data, dd_data, 8);
        for (int k = 0; k < 7; k += 2) {
            dd_data[k + 1] = t_dd[k];
            src_data[k + 1] = t_src[k];
        }
#elif HAS_IC_TAIL
        const bool is_last_sp = sp - C_PARALLEL_FACTOR <= C_PARALLEL_FACTOR - 1;
        if (is_last_sp && is_last_ic_block && is_last_sp_block) {
            LOAD_Nx16_USING_LOOP(7, src_data, src);
            LOAD_Nx16_USING_LOOP(7, dd_data, diff_dst);
            dd_data[7] = simd_id < 8
                    ? into_float(diff_dst[7 * IC_BLOCK_STRIDE + simd_id])
                    : 0.0f;
            src_data[7] = simd_id < 8
                    ? into_float(src[7 * IC_BLOCK_STRIDE + simd_id])
                    : 0.0f;
        } else {
            LOAD_Nx16_USING_LOOP(8, src_data, src);
            LOAD_Nx16_USING_LOOP(8, dd_data, diff_dst);
        }
#else
        LOAD_8x16_USING_LAYOUT(src_data, src);
        LOAD_8x16_USING_LAYOUT(dd_data, diff_dst);
#endif

        src += C_PARALLEL_FACTOR * IC_BLOCK_STRIDE;
        diff_dst += C_PARALLEL_FACTOR * IC_BLOCK_STRIDE;
#if FUSE_BN_RELU == 1
        ws += C_PARALLEL_FACTOR * IC_BLOCK_STRIDE;
#endif // #if FUSE_BN_RELU == 1

#if FUSE_BN_RELU == 1
        const float8 C_ZERO = 0.0f;
        dd_data = select(C_ZERO, dd_data, convert_int8(ws_data));
#endif // #if FUSE_BN_RELU == 1

        const float8 v0 = src_data - v_mean;
        diff_gamma = fma(v0, dd_data, diff_gamma);
        diff_beta += dd_data;
    }

#if HAS_STAT_SP_TAIL
    if (sp_block_idx == STAT_SP_TAIL) {
        sp = (SP - STAT_SP_TAIL * STAT_SP_BLOCK) % C_PARALLEL_FACTOR;
        while (sp-- >= 1) {
#if FUSE_BN_RELU == 1
            char ws_data = block_load(ws_data, ws);
#else
            const char ws_data = 1;
#endif // #if FUSE_BN_RELU == 1

#if HAS_IC_TAIL
            float src_data, dd_data;
            if (sp == 0 && is_last_ic_block) {
                src_data = simd_id < 8 ? into_float(src[simd_id]) : 0.0f;
                dd_data = simd_id < 8 ? into_float(diff_dst[simd_id]) : 0.0f;
            } else {
                src_data = block_load(src_data, src);
                dd_data = block_load(dd_data, diff_dst);
            }
#else
            float src_data, dd_data;
            src_data = block_load(src_data, src);
            dd_data = block_load(dd_data, diff_dst);
#endif

            src += IC_BLOCK_STRIDE;
            diff_dst += IC_BLOCK_STRIDE;
#if FUSE_BN_RELU == 1
            ws += IC_BLOCK_STRIDE;
#endif // #if FUSE_BN_RELU == 1

            if (ws_data != 0) {
                const float v0 = src_data - v_mean;
                const float diff_gamma_tmp = fma(v0, dd_data, diff_gamma[0]);
                diff_gamma[0] = diff_gamma_tmp;
                diff_beta[0] += dd_data;
            }
        }
    }
#endif // #if HAS_STAT_SP_TAIL

    float sum_diff_gamma = 0;
    float sum_diff_beta = 0;
    for (int i = 0; i < 8; i++) {
        sum_diff_gamma += diff_gamma[i];
        sum_diff_beta += diff_beta[i];
    }

#if FUSED_ATOMICS_REDUCTION
    __local float local_gamma[2 * CALC_SLM_SIZE];
    __local float *local_beta = local_gamma + CALC_SLM_SIZE;
    diff_gamma[0] = sum_diff_gamma;
    diff_beta[0] = sum_diff_beta;
    xe_calc_fused_reduction(diff_scale, diff_shift, c, diff_gamma, diff_beta,
            NULL, NULL, local_gamma, local_beta);
#else
    // scratchpad layout:
    // PADDED_IC - diff_gamma reduction, wrote by xe_reduce_stats kernel
    // REDUCE_STAT_NBLOCKS * PADDED_IC - diff_gamma stats calculated by this kernel
    // PADDED_IC - diff_beta reduction, wrote by xe_reduce_stats kernel
    // REDUCE_STAT_NBLOCKS * PADDED_IC - diff_beta stats calculated by this kernel
    block_write(&temp_reduce[PADDED_IC + mb_sp_idx * 16], sum_diff_gamma);
    block_write(&temp_reduce[2 * PADDED_IC + REDUCE_STAT_NBLOCKS * PADDED_IC
                        + mb_sp_idx * 16],
            sum_diff_beta);
#endif // FUSED_ATOMICS_REDUCTION
}

inline void write_8x16_block(__global DATA_T *ptr, int c, float8 val) {
#if USE_NHWC
#if HAS_IC_TAIL
    const int simd_id = get_sub_group_local_id();
    const bool is_last_ic_block = c + 16 > IC;
    if (is_last_ic_block) {
        if (simd_id < 8) {
            for (int k = 0; k < 8; ++k)
                write(ptr + k * IC_BLOCK_STRIDE + simd_id, val[k]);
        }
    } else
#endif // HAS_IC_TAIL
        for (int k = 0; k < 8; ++k)
            block_write(&ptr[k * IC_BLOCK_STRIDE], val[k]);
#else
    block_write(ptr, &val);
#endif // #if USE_NHWC
}
inline void write_1x16_block(__global DATA_T *ptr, int c, float val) {
#if HAS_IC_TAIL
    const int simd_id = get_sub_group_local_id();
    const bool is_last_ic_block = c + 16 > IC;
    if (!is_last_ic_block) {
        block_write(ptr, val);
    } else {
        if (simd_id < 8) { write(ptr + simd_id, val); }
    }
#else
    block_write(ptr, val);
#endif
}

KERNEL_ATTR
__kernel void xe_bnorm_bwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *diff_dst,
        __global float *scaleshift, __global char *ws,
        __global DATA_T *diff_src, __global float *diff_scale,
        __global float *diff_shift, float eps, __global DATA_T *diff_src_add) {

    const int c = GWS_GET_IC();
    const int simd_id = get_sub_group_local_id();
#if HAS_IC_TAIL
    const bool is_last_ic_block = c + 16 > IC;
#endif

    float v_variance;
    MAYBE_LAST_IC_BLOCK_LOAD_1x16(v_variance, variance, c);
#if CALCULATE_DIFF_STATS == 1
    float v_mean;
    MAYBE_LAST_IC_BLOCK_LOAD_1x16(v_mean, mean, c);
    float diff_gamma;
    MAYBE_LAST_IC_BLOCK_LOAD_1x16(diff_gamma, diff_scale, c);
#if DIFF_SHIFT == 1
    float diff_beta;
    MAYBE_LAST_IC_BLOCK_LOAD_1x16(diff_beta, diff_shift, c);
#else
    float diff_beta;
    MAYBE_LAST_IC_BLOCK_LOAD_1x16(diff_beta, diff_shift,
            PADDED_IC + REDUCE_STAT_NBLOCKS * PADDED_IC + c);
#endif // #if DIFF_SHIFT == 1
#endif // #if CALCULATE_DIFF_STATS == 1

#if USE_SCALE == 1
    float gamma;
    MAYBE_LAST_IC_BLOCK_LOAD_1x16(gamma, scaleshift, c);
#else
    const float gamma = 1;
#endif // #if USE_SCALE == 1

    const int sp_block_idx = GWS_GET_SP();
#if USE_NHWC
    const int offset = c + sp_block_idx * VECT_SIZE * IC;
#else
    const int mb = GWS_GET_MB();
    const int offset = (c & 15) + sp_block_idx * VECT_SIZE * 16 + (c & ~15) * SP
            + mb * SP * IC;
#endif

#if HAS_IC_TAIL
    const bool is_last_sp_block = sp_block_idx == SP / VECT_SIZE - 1;
#endif

    src += offset;
    diff_dst += offset;
    ws += offset;
    diff_src += offset;
#if FUSE_BN_ADD_RELU
    diff_src_add += offset;
#endif

#if HAS_SP_TAIL
    int sp;
    if (sp_block_idx == SP_TAIL / VECT_SIZE) {
        sp = SP - SP_TAIL;
    } else {
        sp = VECT_SIZE;
    }
#else
    int sp = VECT_SIZE;
#endif

    const float sqrt_variance = 1.0f / sqrt(v_variance + eps);

    const int C_PARALLEL_FACTOR = 8;
    for (; sp > C_PARALLEL_FACTOR - 1; sp -= C_PARALLEL_FACTOR) {
        float8 src_data;
        float8 dd_data;

#if FUSE_BN_RELU == 1
        char8 ws_data;
        LOAD_8x16_USING_LAYOUT(ws_data, ws);
#endif // #if FUSE_BN_RELU == 1

#if IS_IC_EQ_8
        LOAD_Nx16_USING_LOOP_HALF(8, src_data, src);
        LOAD_Nx16_USING_LOOP_HALF(8, dd_data, diff_dst);
        float8 t_dd = intel_sub_group_shuffle_down(dd_data, dd_data, 8);
        float8 t_src = intel_sub_group_shuffle_down(src_data, src_data, 8);
        for (int k = 0; k < 7; k += 2) {
            dd_data[k + 1] = t_dd[k];
            src_data[k + 1] = t_src[k];
        }
#elif HAS_IC_TAIL && !HAS_SP_TAIL
        const bool is_last_sp = sp - C_PARALLEL_FACTOR <= C_PARALLEL_FACTOR - 1;
        if (is_last_sp && is_last_ic_block && is_last_sp_block) {
            LOAD_Nx16_USING_LOOP(7, src_data, src);
            LOAD_Nx16_USING_LOOP(7, dd_data, diff_dst);
            dd_data[7] = simd_id < 8
                    ? into_float(diff_dst[7 * IC_BLOCK_STRIDE + simd_id])
                    : 0.0f;
            src_data[7] = simd_id < 8
                    ? into_float(src[7 * IC_BLOCK_STRIDE + simd_id])
                    : 0.0f;
        } else {
            LOAD_Nx16_USING_LOOP(8, src_data, src);
            LOAD_Nx16_USING_LOOP(8, dd_data, diff_dst);
        }
#else
        LOAD_8x16_USING_LAYOUT(dd_data, diff_dst);
        LOAD_8x16_USING_LAYOUT(src_data, src);
#endif // IS_IC_EQ_8

        src += C_PARALLEL_FACTOR * IC_BLOCK_STRIDE;
        diff_dst += C_PARALLEL_FACTOR * IC_BLOCK_STRIDE;
#if FUSE_BN_RELU == 1
        ws += C_PARALLEL_FACTOR * IC_BLOCK_STRIDE;
#endif // #if FUSE_BN_RELU == 1

#if FUSE_BN_RELU == 1
        const float8 C_ZERO = 0.0f;
        dd_data = select(C_ZERO, dd_data, convert_int8(ws_data));
#if FUSE_BN_ADD_RELU
        write_8x16_block(diff_src_add, c, dd_data);
#endif
#endif // #if FUSE_BN_RELU == 1

#if CALCULATE_DIFF_STATS == 1
        dd_data -= (diff_beta
                           + (src_data - v_mean) * diff_gamma * sqrt_variance)
                / (MB * ID * IH * IW);
#endif // #if CALCULATE_DIFF_STATS == 1

        dd_data *= gamma * sqrt_variance;
        write_8x16_block(diff_src, c, dd_data);

        diff_src += C_PARALLEL_FACTOR * IC_BLOCK_STRIDE;
#if FUSE_BN_ADD_RELU
        diff_src_add += C_PARALLEL_FACTOR * IC_BLOCK_STRIDE;
#endif
    }

#if HAS_SP_TAIL
    if (sp_block_idx == SP_TAIL / VECT_SIZE) {
        sp = (SP - SP_TAIL) % C_PARALLEL_FACTOR;
        while (sp-- >= 1) {
#if FUSE_BN_RELU == 1
            char ws_data = block_load(ws_data, ws);
#endif // #if FUSE_BN_RELU == 1

#if HAS_IC_TAIL
            float dd_data;
            if (sp == 0 && is_last_ic_block)
                dd_data = simd_id < 8 ? into_float(diff_dst[simd_id]) : 0.0f;
            else
                dd_data = block_load(dd_data, diff_dst);
#if CALCULATE_DIFF_STATS == 1
            float src_data;
            if (sp == 0 && is_last_ic_block)
                src_data = simd_id < 8 ? into_float(src[simd_id]) : 0.0f;
            else
                src_data = block_load(src_data, src);
#endif // #if CALCULATE_DIFF_STATS == 1
#else
            float dd_data = block_load(dd_data, diff_dst);
#if CALCULATE_DIFF_STATS == 1
            float src_data = block_load(src_data, src);
#endif // #if CALCULATE_DIFF_STATS == 1
#endif // HAS_IC_TAIL

            src += IC_BLOCK_STRIDE;
            diff_dst += IC_BLOCK_STRIDE;
#if FUSE_BN_RELU == 1
            ws += IC_BLOCK_STRIDE;
#endif // #if FUSE_BN_RELU == 1

#if FUSE_BN_RELU == 1
            if (ws_data == 0) dd_data = 0;
#if FUSE_BN_ADD_RELU
            write_1x16_block(diff_src_add, c, dd_data);
#endif
#endif // #if FUSE_BN_RELU == 1

#if CALCULATE_DIFF_STATS == 1
            dd_data -= (diff_beta
                               + (src_data - v_mean) * diff_gamma
                                       * sqrt_variance)
                    / (MB * ID * IH * IW);
#endif // #if CALCULATE_DIFF_STATS == 1

            dd_data *= gamma * sqrt_variance;
            write_1x16_block(diff_src, c, dd_data);

            diff_src += IC_BLOCK_STRIDE;
#if FUSE_BN_ADD_RELU
            diff_src_add += IC_BLOCK_STRIDE;
#endif
        }
    }
#endif // #if HAS_SP_TAIL
}
