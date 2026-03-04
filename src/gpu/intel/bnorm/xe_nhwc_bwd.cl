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

// BWD kernels that are that are specially optimized for NHWC layout
// (NHWC_OPTIMIZED definition).
// These kernels not supported IC tail processing.
// Two types of reduction are implemented:
// 1) Reduction over scratchpad (reduce_temp) with SLM use, implemented by
//    xe_reduce_stats_nhwc kernel.
// 2) Atomics-based reduction with SLM use (FUSED_ATOMICS_REDUCTION definition),
//    implemented as part of calc kernels, see xe_*_fused_reduction()
//    functions in xe.h. This reduction implementation requires zeroing and
//    finalization steps, see xe_fused_reduce_* kernels in xe_reduce.cl

NAMED_KERNEL_ATTR(CALC)
__kernel void xe_calculate_stats_nhwc(__global DATA_T *src,
        __global float *mean, __global DATA_T *diff_dst, __global char *ws,
        __global float *temp_reduce, volatile __global atomic_float *diff_scale,
        volatile __global atomic_float *diff_shift) {

    const int mb = GWS_GET_STAT_MB();
    const int c = GWS_GET_STAT_IC();
    const int sp_block_idx = GWS_GET_STAT_SP();
    const int ic_block_offset = (c / SG_SIZE) * IC_BLOCK;
    const int offset = ic_block_offset + sp_block_idx * STAT_SP_BLOCK * IC;

    mean += ic_block_offset;
    src += offset;
    diff_dst += offset;
    ws += offset;

    float v_mean[IC_BLOCK_SGROUPS];
    for (int sg = 0; sg < IC_BLOCK_SGROUPS; ++sg) {
        v_mean[sg] = block_load(v_mean[sg], &mean[sg * SG_SIZE]);
    }

    VECT_FLOAT_T diff_gamma[IC_BLOCK_SGROUPS / VECT_SIZE] = {0.0f};
    VECT_FLOAT_T diff_beta[IC_BLOCK_SGROUPS / VECT_SIZE] = {0.0f};
#if HAS_IC_VECT_TAIL
    float diff_gamma_tail[IC_TAIL_SGROUPS] = {0.0f};
    float diff_beta_tail[IC_TAIL_SGROUPS] = {0.0f};
#else
    float *diff_gamma_tail = NULL;
    float *diff_beta_tail = NULL;
#endif

#if USE_WORKAROUND
    for (int sp = 0; sp < STAT_SP_BLOCK; ++sp) {
        if (sp_block_idx * STAT_SP_BLOCK + sp >= SP) break;
#else // issue
#if HAS_STAT_SP_BLOCK_TAIL
    for (int sp = 0; sp < min(STAT_SP_BLOCK, SP - sp_block_idx * STAT_SP_BLOCK);
            ++sp) {
#else
    for (int sp = 0; sp < STAT_SP_BLOCK; ++sp) {
#endif
#endif // USE_WORKAROUND
        for (int sg = 0; sg < IC_BLOCK_SGROUPS / VECT_SIZE; ++sg) {
            const int sg_idx = sg * SG_SIZE * VECT_SIZE;
#if FUSE_BN_RELU
            VECT_CHAR_T ws_vect;
            ws_vect = block_load(ws_vect, &ws[sg_idx]);
#endif
            VECT_FLOAT_T src_vect;
            src_vect = block_load(src_vect, &src[sg_idx]);
            VECT_FLOAT_T dd_vect;
            dd_vect = block_load(dd_vect, &diff_dst[sg_idx]);
            VECT_FLOAT_T v0;
            for (int vect = 0; vect < VECT_SIZE; ++vect) {
                int sg_idx = sg * VECT_SIZE + vect;
#if VECT_SIZE > 1
                v0[vect] = src_vect[vect] - v_mean[sg_idx];
#else
                v0 = src_vect - v_mean[sg_idx];
#endif
            }

#if FUSE_BN_RELU
            dd_vect = select(
                    (VECT_FLOAT_T)0.0f, dd_vect, CONVERT_VECT_INT_T(ws_vect));
#endif

            diff_gamma[sg] = fma(v0, dd_vect, diff_gamma[sg]);
            diff_beta[sg] += dd_vect;
        }
#if HAS_IC_VECT_TAIL
        for (int sg = 0; sg < IC_TAIL_SGROUPS; ++sg) {
            const int sg_idx = IC_VECT_SGROUPS + sg;
#if FUSE_BN_RELU
            char ws_tail;
            ws_tail = block_load(ws_tail, &ws[sg_idx * SG_SIZE]);
#endif
            float src_tail;
            src_tail = block_load(src_tail, &src[sg_idx * SG_SIZE]);
            float dd_tail;
            dd_tail = block_load(dd_tail, &diff_dst[sg_idx * SG_SIZE]);
            float v0 = src_tail - v_mean[sg_idx];
#if FUSE_BN_RELU
            dd_tail = select(0.0f, dd_tail, convert_int(ws_tail));
#endif
            diff_gamma_tail[sg] = fma(v0, dd_tail, diff_gamma_tail[sg]);
            diff_beta_tail[sg] += dd_tail;
        }
#endif
        src += IC;
        diff_dst += IC;
#if FUSE_BN_RELU
        ws += IC;
#endif
    }

#if FUSED_ATOMICS_REDUCTION
    __local float local_gamma[2 * CALC_SLM_SIZE];
    __local float *local_beta = local_gamma + CALC_SLM_SIZE;
    xe_calc_fused_reduction(diff_scale, diff_shift, ic_block_offset, diff_gamma,
            diff_beta, diff_gamma_tail, diff_beta_tail, local_gamma,
            local_beta);
#else
    // scratchpad layout:
    // PADDED_IC - diff_gamma reduction, wrote by xe_reduce_stats kernel
    // REDUCE_STAT_NBLOCKS * PADDED_IC - diff_gamma stats calculated by this kernel
    // PADDED_IC - diff_beta reduction, wrote by xe_reduce_stats kernel
    // REDUCE_STAT_NBLOCKS * PADDED_IC - diff_beta stats calculated by this kernel

    for (int sg = 0; sg < IC_BLOCK_SGROUPS; ++sg) {
        const int reduce_off = sp_block_idx * SG_SIZE
                + REDUCE_STAT_NBLOCKS * SG_SIZE
                        * (sg + (int)(c / SG_SIZE) * (IC_BLOCK / SG_SIZE));

        const int diff_gamma_offset = PADDED_IC + reduce_off;
        const int diff_beta_offset
                = 2 * PADDED_IC + REDUCE_STAT_NBLOCKS * PADDED_IC + reduce_off;

#if HAS_IC_VECT_TAIL
        if (sg >= IC_VECT_SGROUPS) {
            block_write(&temp_reduce[diff_gamma_offset],
                    &diff_gamma_tail[sg - IC_VECT_SGROUPS], 1);
            block_write(&temp_reduce[diff_beta_offset],
                    &diff_beta_tail[sg - IC_VECT_SGROUPS], 1);
        } else
#endif
        {
#if VECT_SIZE > 1
            block_write(&temp_reduce[diff_gamma_offset],
                    &diff_gamma[sg / VECT_SIZE][sg % VECT_SIZE], 1);
            block_write(&temp_reduce[diff_beta_offset],
                    &diff_beta[sg / VECT_SIZE][sg % VECT_SIZE], 1);
#else
            block_write(&temp_reduce[diff_gamma_offset], &diff_gamma[sg], 1);
            block_write(&temp_reduce[diff_beta_offset], &diff_beta[sg], 1);
#endif
        }
    }
#endif // FUSED_ATOMICS_REDUCTION
}

KERNEL_ATTR
__kernel void xe_bnorm_bwd_nhwc(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *diff_dst,
        __global float *scaleshift, __global char *ws,
        __global DATA_T *diff_src, __global float *diff_scale,
        __global float *diff_shift, float eps, __global DATA_T *diff_src_add) {

    const int c = GWS_GET_IC();
    const int ic_block_offset = (c / SG_SIZE) * IC_BLOCK;

    variance += ic_block_offset;
    mean += ic_block_offset;
    diff_scale += ic_block_offset;
    diff_shift += ic_block_offset;
    scaleshift += ic_block_offset;

    VECT_FLOAT_T v_variance[IC_BLOCK_SGROUPS / VECT_SIZE],
            v_mean[IC_BLOCK_SGROUPS / VECT_SIZE],
            diff_gamma[IC_BLOCK_SGROUPS / VECT_SIZE],
            diff_beta[IC_BLOCK_SGROUPS / VECT_SIZE],
            sqrt_variance[IC_BLOCK_SGROUPS / VECT_SIZE],
            gamma[IC_BLOCK_SGROUPS / VECT_SIZE];
    for (int sg = 0; sg < IC_BLOCK_SGROUPS / VECT_SIZE; ++sg) {
        const int sg_idx = sg * SG_SIZE * VECT_SIZE;
        v_variance[sg] = block_load(v_variance[sg], &variance[sg_idx]);
#if CALCULATE_DIFF_STATS == 1
        v_mean[sg] = block_load(v_mean[sg], &mean[sg_idx]);
        diff_gamma[sg] = block_load(diff_gamma[sg], &diff_scale[sg_idx]);
#if DIFF_SHIFT == 1
        diff_beta[sg] = block_load(diff_beta[sg], &diff_shift[sg_idx]);
#else
        diff_beta[sg] = block_load(diff_beta[sg], &diff_shift[IC + REDUCE_STAT_NBLOCKS * IC + sg_idx]);
#endif // #if DIFF_SHIFT == 1
#endif // #if CALCULATE_DIFF_STATS == 1
#if USE_SCALE == 1
        gamma[sg] = block_load(gamma[sg], &scaleshift[sg_idx]);
#else
        gamma[sg] = (VECT_FLOAT_T)1.0f;
#endif
        sqrt_variance[sg]
                = (VECT_FLOAT_T)1.0f / sqrt(v_variance[sg] + (VECT_FLOAT_T)eps);
    }
#if HAS_IC_VECT_TAIL
    float v_variance_tail[IC_TAIL_SGROUPS], v_mean_tail[IC_TAIL_SGROUPS],
            diff_gamma_tail[IC_TAIL_SGROUPS], diff_beta_tail[IC_TAIL_SGROUPS],
            sqrt_variance_tail[IC_TAIL_SGROUPS], gamma_tail[IC_TAIL_SGROUPS];
    for (int sg = 0; sg < IC_TAIL_SGROUPS; ++sg) {
        const int sg_idx = (IC_VECT_SGROUPS + sg) * SG_SIZE;
        v_variance_tail[sg] = block_load(v_variance_tail[sg], &variance[sg_idx]);

#if CALCULATE_DIFF_STATS == 1
        v_mean_tail[sg] = block_load(v_mean_tail[sg], &mean[sg_idx]);
        diff_gamma_tail[sg] = block_load(diff_gamma_tail[sg], &diff_scale[sg_idx]);
#if DIFF_SHIFT == 1
        diff_beta_tail[sg] = block_load(diff_beta_tail[sg], &diff_shift[sg_idx]);
#else
        diff_beta_tail[sg] = block_load(diff_beta_tail[sg], &diff_shift[IC + REDUCE_STAT_NBLOCKS * IC + sg_idx]);
#endif // #if DIFF_SHIFT == 1
#endif // #if CALCULATE_DIFF_STATS == 1
#if USE_SCALE == 1
        gamma_tail[sg] = block_load(gamma_tail[sg], &scaleshift[sg_idx]);
#else
        gamma_tail[sg] = 1.0f;
#endif
        sqrt_variance_tail[sg] = 1.0f / sqrt(v_variance_tail[sg] + eps);
    }
#endif

    const int sp_block_idx = GWS_GET_SP();
    const int offset = ic_block_offset + sp_block_idx * UPDATE_SP_BLOCK * IC;

    src += offset;
    diff_dst += offset;
    ws += offset;
    diff_src += offset;
#if FUSE_BN_ADD_RELU
    diff_src_add += offset;
#endif

#if HAS_UPDATE_SP_BLOCK_TAIL
    for (int sp = 0;
            sp < min(UPDATE_SP_BLOCK, SP - sp_block_idx * UPDATE_SP_BLOCK);
            sp += UPDATE_SP_UNROLL) {
#else
    for (int sp = 0; sp < UPDATE_SP_BLOCK; sp += UPDATE_SP_UNROLL) {
#endif
        for (int sg = 0; sg < IC_BLOCK_SGROUPS / VECT_SIZE; ++sg) {
            const int sg_idx = sg * SG_SIZE * VECT_SIZE;

            VECT_FLOAT_T src_vect[UPDATE_SP_UNROLL];
            unroll_for(int i = 0; i < UPDATE_SP_UNROLL; i++) {
                src_vect[i] = block_load(src_vect[i], &src[sg_idx + IC * i]);
            }
            VECT_FLOAT_T dd_vect[UPDATE_SP_UNROLL];
            unroll_for(int i = 0; i < UPDATE_SP_UNROLL; i++) {
                dd_vect[i] = block_load(dd_vect[i], &diff_dst[sg_idx + IC * i]);
            }

#if FUSE_BN_RELU
            VECT_CHAR_T ws_vect[UPDATE_SP_UNROLL];
            unroll_for(int i = 0; i < UPDATE_SP_UNROLL; i++) {
                ws_vect[i] = block_load(ws_vect[i], &ws[sg_idx + IC * i]);
                dd_vect[i] = select((VECT_FLOAT_T)0.0f, dd_vect[i],
                        CONVERT_VECT_INT_T(ws_vect[i]));
            }
#if FUSE_BN_ADD_RELU
            unroll_for(int i = 0; i < UPDATE_SP_UNROLL; i++) {
                block_write(&diff_src_add[sg_idx + IC * i], &dd_vect[i]);
            }
#endif
#endif

#if CALCULATE_DIFF_STATS == 1
            unroll_for(int i = 0; i < UPDATE_SP_UNROLL; i++) {
                dd_vect[i]
                        -= (diff_beta[sg]
                                   + (src_vect[i] - v_mean[sg]) * diff_gamma[sg]
                                           * sqrt_variance[sg])
                        / (MB * ID * IH * IW);
            }
#endif // #if CALCULATE_DIFF_STATS == 1
            unroll_for(int i = 0; i < UPDATE_SP_UNROLL; i++) {
                dd_vect[i] *= gamma[sg] * sqrt_variance[sg];
            }
            unroll_for(int i = 0; i < UPDATE_SP_UNROLL; i++) {
                block_write(&diff_src[sg_idx + IC * i], &dd_vect[i]);
            }
        }
#if HAS_IC_VECT_TAIL
        for (int sg = 0; sg < IC_TAIL_SGROUPS; ++sg) {
            const int sg_idx = (IC_VECT_SGROUPS + sg) * SG_SIZE;
            float src_tail[UPDATE_SP_UNROLL];
            unroll_for(int i = 0; i < UPDATE_SP_UNROLL; i++) {
                src_tail[i] = block_load(src_tail[i], &src[sg_idx + IC * i]);
            }
            float dd_tail[UPDATE_SP_UNROLL];
            unroll_for(int i = 0; i < UPDATE_SP_UNROLL; i++) {
                dd_tail[i] = block_load(dd_tail[i], &diff_dst[sg_idx + IC * i]);
            }
#if FUSE_BN_RELU
            char ws_tail[UPDATE_SP_UNROLL];
            unroll_for(int i = 0; i < UPDATE_SP_UNROLL; i++) {
                ws_tail[i] = block_load(ws_tail[i], &ws[sg_idx + IC * i]);
                dd_tail[i] = select(0.0f, dd_tail[i], convert_int(ws_tail[i]));
            }
#if FUSE_BN_ADD_RELU
            unroll_for(int i = 0; i < UPDATE_SP_UNROLL; i++) {
                block_write(&diff_src_add[sg_idx + IC * i], &dd_tail[i], 1);
            }
#endif
#endif
#if CALCULATE_DIFF_STATS == 1
            unroll_for(int i = 0; i < UPDATE_SP_UNROLL; i++) {
                dd_tail[i] -= (diff_beta_tail[sg]
                                      + (src_tail[i] - v_mean_tail[sg])
                                              * diff_gamma_tail[sg]
                                              * sqrt_variance_tail[sg])
                        / (MB * ID * IH * IW);
            }
#endif // #if CALCULATE_DIFF_STATS == 1
            unroll_for(int i = 0; i < UPDATE_SP_UNROLL; i++) {
                dd_tail[i] *= gamma_tail[sg] * sqrt_variance_tail[sg];
                block_write(&diff_src[sg_idx + IC * i], &dd_tail[i], 1);
            }
        }
#endif
        src += IC * UPDATE_SP_UNROLL;
        diff_dst += IC * UPDATE_SP_UNROLL;
        diff_src += IC * UPDATE_SP_UNROLL;
#if FUSE_BN_RELU
#if FUSE_BN_ADD_RELU
        diff_src_add += IC * UPDATE_SP_UNROLL;
#endif
        ws += IC * UPDATE_SP_UNROLL;
#endif
    }
}
