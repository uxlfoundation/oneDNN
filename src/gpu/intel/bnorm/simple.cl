/*******************************************************************************simple_simple_reduce_index
* Copyright 2019 Intel Corporation
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

#include "gpu/intel/include/dispatch.h"
#include "gpu/intel/include/io.h"
#include "gpu/intel/include/types.h"

int simple_reduce_index(int x[5]) {
    int dim[5] = {MB, IC, ID, IH, IW};
    dim[REDUCE_DIM_IDX] = 1;
    return x[0] * (dim[2] * dim[3] * dim[4]) + x[2] * (dim[3] * dim[4])
            + x[3] * dim[4] + x[4];
}

NAMED_KERNEL_ATTR(CALC)
__kernel void simple_calculate_mean_variance(
        __global DATA_T *src, __global float *mean, __global float *variance) {
    int x[5];
    x[0] = GWS_GET_STAT_MB();
    x[1] = GWS_GET_STAT_IC();
    x[2] = GWS_GET_STAT_ID();
    x[3] = GWS_GET_STAT_IH();
    x[4] = GWS_GET_STAT_IW();

    // sum of all src elements from given C
    float src_sum[VECT_DT_N];
    // sum of all src^2 elements from given C
    float src_pow_sum[VECT_DT_N];
    for (int i = 0; i < VECT_DT_N; i++) {
        src_sum[i] = 0;
        src_pow_sum[i] = 0;
    }

    for (int i = 0; i < REDUCE_DIM; i += SUB_GROUP_SIZE * VECT_DT_N) {
        x[REDUCE_DIM_IDX] = i;
        int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4]);
        float src_arr[VECT_DT_N];
        block_load(src_arr, src + src_off, VECT_DT_N);
        for (int j = 0; j < VECT_DT_N; j++) {
            src_sum[j] += src_arr[j];
            src_pow_sum[j] += src_arr[j] * src_arr[j];
        }
    }
    float sum = 0;
    float pow_sum = 0;
    for (int i = 0; i < VECT_DT_N; i++) {
        sum += src_sum[i];
        pow_sum += src_pow_sum[i];
    }

    x[REDUCE_DIM_IDX] = 0;
    int reduce_idx = simple_reduce_index(x);

    float total_sum = sub_group_reduce_add(sum);
    float total_pow_sum = sub_group_reduce_add(pow_sum);
    int local_id = get_sub_group_local_id();
    if (local_id == 0) {
        float calc_mean = total_sum / (MB * ID * IH * IW);
        float calc_variance
                = total_pow_sum / (MB * ID * IH * IW) - calc_mean * calc_mean;
        mean[x[1]] = calc_mean;
        variance[x[1]] = calc_variance < 0 ? 0 : calc_variance;
    }
}

KERNEL_ATTR
__kernel void simple_bnorm_fwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *dst, __global float *scale,
        __global float *shift, __global char *ws, float eps,
        __global DATA_T *src_add, float relu_alpha) {
    const int n = GWS_GET_MB();
    const int c = GWS_GET_IC();
    const int d = GWS_GET_ID();
    const int h = GWS_GET_IH();
    const int w = GWS_GET_IW();
#if USE_SCALE == 1
    float sm = scale[c];
#else
    float sm = 1;
#endif
#if USE_SHIFT == 1
    float sv = shift[c];
#else
    float sv = 0;
#endif

#if SAVE_STATS == 0
    variance += IC;
#endif
    float v_mean = mean[c];
    float v_variance = variance[c];
    const int off = SRC_OFF(n, c, d, h, w);
    float v0 = load(v0, src + off);
    float sqrt_variance = 1.0f / sqrt(v_variance + eps);
    float bn_res = sm * (v0 - v_mean) * sqrt_variance + sv;
#if FUSE_BN_ADD_RELU == 1
    bn_res += load(bn_res, src_add + off);
#endif

#if FUSE_BN_RELU == 1
    if (bn_res <= 0) {
        bn_res = 0;
#if IS_TRAINING == 1
        ws[off] = 0;
    } else {
        ws[off] = -1;
#endif
    }
#endif

#if WITH_RELU
#if WITH_LEAKY_RELU
    if (bn_res < 0) { bn_res *= relu_alpha; }
#else
    bn_res = max(bn_res, 0.0f);
#endif //WITH_LEAKY_RELU
#endif //WITH_RELU

    write(dst + off, bn_res);
}

#if IS_BWD

#if MB_BLOCK == 16
#define MB16
#endif

NAMED_KERNEL_ATTR(CALC)
__kernel void simple_calculate_stats(__global DATA_T *src, __global float *mean,
        __global DATA_T *diff_dst, __global char *ws,
        __global float *reduce_temp) {
    const int mb = GWS_GET_STAT_MB();
    const int stat_mb_block_idx = mb / MB_BLOCK;

    const int c = GWS_GET_STAT_IC();

    const int sp_beg = GWS_GET_STAT_SP();
    const int stat_sp_block = GWS_GET_STAT_SP_BLOCK();
    const int stat_sp_nblocks = ID * IH * IW / stat_sp_block;
    const int stat_sp_block_idx = sp_beg / stat_sp_block;

    const int mb_sp_idx
            = stat_mb_block_idx * stat_sp_nblocks + stat_sp_block_idx;

    const int s_off = c * ID * IH * IW * MB_BLOCK + mb * IC * ID * IH * IW
            + sp_beg * MB_BLOCK * IC_BLOCK;
    src += s_off;
    diff_dst += s_off;
#if FUSE_BN_RELU == 1
    ws += s_off;
#endif
    float diff_gamma0[VECT_DT_N], diff_beta0[VECT_DT_N];
    float diff_gamma1[VECT_DT_N], diff_beta1[VECT_DT_N];
    for (int i = 0; i < VECT_DT_N; i++) {
        diff_gamma0[i] = 0.0f;
        diff_beta0[i] = 0.0f;
        diff_gamma1[i] = 0.0f;
        diff_beta1[i] = 0.0f;
    }
    float v_mean = as_float(
            intel_sub_group_block_read((const __global uint *)&mean[c]));

    for (int sp = sp_beg; sp < sp_beg + stat_sp_block; sp++) {
        float dd0[VECT_DT_N];
        block_load(dd0, diff_dst, VECT_DT_N);
        float ss0[VECT_DT_N];
        block_load(ss0, src, VECT_DT_N);
#ifdef MB16
        float dd1[VECT_DT_N];
        block_load(dd1, diff_dst + 8 * 16, VECT_DT_N);
        float ss1[VECT_DT_N];
        block_load(ss1, src + 8 * 16, VECT_DT_N);
#endif
#if FUSE_BN_RELU == 1
        int ws0_v[VECT_DT_N];
        block_load(ws0_v, ws, VECT_DT_N);
        for (int i = 0; i < VECT_DT_N; i++) {
            if (!ws0_v[i]) dd0[i] = 0.0f;
        }
#ifdef MB16
        int ws1_v[VECT_DT_N];
        block_load(ws1_v, ws + 8 * 16, VECT_DT_N);
        for (int i = 0; i < VECT_DT_N; i++) {
            if (!ws1_v[i]) dd1[i] = 0.0f;
        }
#endif
        ws += MB_BLOCK * IC_BLOCK;
#endif
        for (int i = 0; i < VECT_DT_N; i++) {
            diff_gamma0[i] = fma(ss0[i] - v_mean, dd0[i], diff_gamma0[i]);
            diff_beta0[i] += dd0[i];
        }
#ifdef MB16
        for (int i = 0; i < VECT_DT_N; i++) {
            diff_gamma1[i] = fma(ss1[i] - v_mean, dd1[i], diff_gamma1[i]);
            diff_beta1[i] += dd1[i];
        }
#endif

        src += MB_BLOCK * IC_BLOCK;
        diff_dst += MB_BLOCK * IC_BLOCK;
    }
    float v_diff_gamma = 0.0f, v_diff_beta = 0.0f;
    for (int i = 0; i < VECT_DT_N; i++) {
        v_diff_gamma += diff_gamma0[i];
        v_diff_beta += diff_beta0[i];
    }
#ifdef MB16
    for (int i = 0; i < VECT_DT_N; i++) {
        v_diff_gamma += diff_gamma1[i];
        v_diff_beta += diff_beta1[i];
    }
#endif
    intel_sub_group_block_write(
            (__global uint *)&reduce_temp[mb_sp_idx * IC + c],
            as_uint(v_diff_gamma));
    intel_sub_group_block_write(
            (__global uint *)&reduce_temp[REDUCE_STAT_NBLOCKS * IC
                    + mb_sp_idx * IC + c],
            as_uint(v_diff_beta));
}

NAMED_KERNEL_ATTR(REDUCE)
__kernel void simple_reduce_stats(__global float *reduce_temp,
        __global float *diff_scale, __global float *diff_shift,
        __global float *variance, float eps) {
    const int c = GWS_GET_REDUCE_STAT_IC();
    reduce_temp += c;
    float diff_gamma = 0.0f, diff_beta = 0.0f;
    for (int i = 0; i < REDUCE_STAT_NBLOCKS; i++) {
        diff_gamma += reduce_temp[i * IC];
        diff_beta += reduce_temp[REDUCE_STAT_NBLOCKS * IC + i * IC];
    }

    float sqrt_variance = 1.0f / sqrt(variance[c] + eps);

    diff_scale[c] = diff_gamma * sqrt_variance;
#if USE_SHIFT == 1
    diff_shift[c] = diff_beta;
#else
    // When USE_SHIFT == 0, `diff_shift` is a second part of reduce_temp
    diff_shift[REDUCE_STAT_NBLOCKS * IC + c] = diff_beta;
#endif
}

KERNEL_ATTR
__kernel void simple_bnorm_bwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *diff_dst,
        __global float *scale, __global char *ws, __global DATA_T *diff_src,
        __global float *diff_scale, __global float *diff_shift, float eps,
        __global DATA_T *diff_src_add) {
    const int n = GWS_GET_MB();
    const int c = GWS_GET_IC();
    const int d = GWS_GET_ID();
    const int h = GWS_GET_IH();
    const int w = GWS_GET_IW();

#if USE_SCALE == 1
    float gamma = as_float(
            intel_sub_group_block_read((const __global uint *)&scale[c]));
#else
    float gamma = 1.0f;
#endif

    float v_variance = as_float(
            intel_sub_group_block_read((const __global uint *)&variance[c]));
    float sqrt_variance = 1.0f / sqrt(v_variance + eps);

#if CALCULATE_STATS == 1
    float v_mean = as_float(
            intel_sub_group_block_read((const __global uint *)&mean[c]));
    float diff_gamma = as_float(
            intel_sub_group_block_read((const __global uint *)&diff_scale[c]));
#if USE_SHIFT == 1
    float diff_beta = as_float(
            intel_sub_group_block_read((const __global uint *)&diff_shift[c]));
#else
    float diff_beta = as_float(intel_sub_group_block_read(
            (const __global uint *)&diff_shift[REDUCE_STAT_NBLOCKS * IC + c]));
#endif // #if USE_SHIFT == 1
#endif

    const uint d_off = SRC_OFF(n, c, d, h, w);
    diff_src += d_off;
#if FUSE_BN_ADD_RELU == 1
    diff_src_add += d_off;
#endif
    diff_dst += d_off;
    src += d_off;

    VECT_FLOAT_T blockD0;
    blockD0 = block_load(blockD0, diff_dst);
#ifdef MB16
    VECT_FLOAT_T blockD1;
    blockD1 = block_load(blockD1, diff_dst + 8 * IC_BLOCK);
#endif
#if FUSE_BN_RELU == 1
    ws += d_off;
    VECT_CHAR_T blockWS0;
    blockWS0 = block_load(blockWS0, ws);
    blockD0 = select((VECT_FLOAT_T)0.0f, blockD0, CONVERT_VECT_INT_T(blockWS0));
#if FUSE_BN_ADD_RELU == 1
    block_write(diff_src_add, &blockD0);
#endif
#ifdef MB16
    VECT_CHAR_T blockWS1;
    blockWS1 = block_load(blockWS1, ws + 8 * IC_BLOCK);
    blockD1 = select((VECT_FLOAT_T)0.0f, blockD1, CONVERT_VECT_INT_T(blockWS1));
#if FUSE_BN_ADD_RELU == 1
    block_write(diff_src_add + 8 * 16, &blockD1);
#endif
#endif
#endif

    gamma *= sqrt_variance;

#if CALCULATE_STATS == 1
    diff_gamma *= sqrt_variance;
    diff_gamma /= (MB * ID * IH * IW);
    diff_beta /= (MB * ID * IH * IW);

    VECT_FLOAT_T blockS0;
    blockS0 = block_load(blockS0, src);
    blockD0 -= fma((VECT_FLOAT_T)diff_gamma, blockS0 - (VECT_FLOAT_T)v_mean,
            (VECT_FLOAT_T)diff_beta);
#ifdef MB16
    VECT_FLOAT_T blockS1;
    blockS1 = block_load(blockS1, src + 8 * IC_BLOCK);
    blockD1 -= fma((VECT_FLOAT_T)diff_gamma, blockS1 - (VECT_FLOAT_T)v_mean,
            (VECT_FLOAT_T)diff_beta);
#endif
#endif
    blockD0 *= gamma;
    block_write(diff_src, &blockD0);
#ifdef MB16
    blockD1 *= gamma;
    block_write(diff_src + 8 * 16, &blockD1);
#endif
}

#endif
