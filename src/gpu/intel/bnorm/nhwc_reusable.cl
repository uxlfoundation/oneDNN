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
#include "gpu/intel/bnorm/nhwc_reusable.h"
#include "gpu/intel/bnorm/xe_reduce.h"

// Two sets of nhwc-optimized reusable kernels which are implemented with and
// without use of private memory buffers.
// These two ways require different layouts of a scratchpadd and/or SLM buffers.
// The names of kernels and relative functions distinguish by suffix "buff".

// Atomic-based reduction for 1pass algorithm, for no private buffers kernels.
void nhwc_reusable_1pass_fused_reduction(volatile __global atomic_float *mean,
        volatile __global atomic_float *variance, off_t dst_offset,
        SUM_DATA_T *sum, SUM_DATA_T *sum_sq, __local SUM_DATA_T *local_sum,
        __local SUM_DATA_T *local_sum_sq, off_t vect_size) {
    const int local_id = get_local_id(1);
    const int simd_id = get_sub_group_local_id();
    const int row_size = vect_size * SUB_GROUP_SIZE;
    const int group_size = get_local_size(1);
    if (local_id > 0) {
        unroll_4_for(int v_idx = 0; v_idx < vect_size; v_idx++) {
            const int slm_offset
                    = local_id * row_size + v_idx * SUB_GROUP_SIZE + simd_id;
            local_sum[slm_offset] = sum[v_idx];
            local_sum_sq[slm_offset] = sum_sq[v_idx];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0) {
        unroll_16_for(int l_id = 1; l_id < group_size; l_id++) {
            for (int v_idx = 0; v_idx < vect_size; v_idx++) {
                const int off
                        = l_id * row_size + v_idx * SUB_GROUP_SIZE + simd_id;
                SUM_DATA_T tmp = local_sum[off];
                SUM_DATA_T tmp_sq = local_sum_sq[off];
                sum[v_idx] = summation(tmp.s1, sum[v_idx]);
                sum_sq[v_idx] = summation(tmp_sq.s1, sum_sq[v_idx]);
                sum[v_idx] = summation(tmp.s0, sum[v_idx]);
                sum_sq[v_idx] = summation(tmp_sq.s0, sum_sq[v_idx]);
            }
        }
        unroll_4_for(int v_idx = 0; v_idx < vect_size; v_idx++) {
            const int off = v_idx * SUB_GROUP_SIZE + simd_id;
            atomic_add_global(&mean[dst_offset + off], sum[v_idx].s0);
            atomic_add_global(&variance[dst_offset + off], sum_sq[v_idx].s0);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return;
}

// Atomic-based reduction for 1pass algorithm, for kernels with private buffers.
void nhwc_reusable_1pass_fused_reduction_buff(
        volatile __global atomic_float *mean,
        volatile __global atomic_float *variance, off_t dst_offset,
        SUM_DATA_T *sum, SUM_DATA_T *sum_sq, __local SUM_DATA_T *local_sum,
        __local SUM_DATA_T *local_sum_sq, off_t ic_block) {
    const int local_id = get_local_id(1);
    const int simd_id = get_sub_group_local_id();
    const int row_size = ic_block;
    const int group_size = get_local_size(1);
    const int ic_block_sgroups = ic_block / SUB_GROUP_SIZE;

    if (local_id > 0) {
        unroll_16_for(int sg = 0; sg < ic_block_sgroups; sg++) {
            const int slm_offset
                    = local_id * row_size + sg * SUB_GROUP_SIZE + simd_id;
            local_sum[slm_offset] = sum[sg];
            local_sum_sq[slm_offset] = sum_sq[sg];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0) {
        unroll_16_for(int l_id = 1; l_id < group_size; l_id++) {
            unroll_4_for(int sg = 0; sg < ic_block_sgroups; sg++) {
                const int off = l_id * row_size + sg * SUB_GROUP_SIZE + simd_id;
                SUM_DATA_T tmp = local_sum[off];
                SUM_DATA_T tmp_sq = local_sum_sq[off];
                sum[sg] = summation(tmp.s1, sum[sg]);
                sum_sq[sg] = summation(tmp_sq.s1, sum_sq[sg]);
                sum[sg] = summation(tmp.s0, sum[sg]);
                sum_sq[sg] = summation(tmp_sq.s0, sum_sq[sg]);
            }
        }
        unroll_4_for(int sg = 0; sg < ic_block_sgroups; sg++) {
            const int off = sg * SUB_GROUP_SIZE + simd_id;
            atomic_add_global(&mean[dst_offset + off], sum[sg].s0);
            atomic_add_global(&variance[dst_offset + off], sum_sq[sg].s0);
        }
    }
    return;
}

// Atomic-based reduction for regular algorithm, for no private buffers kernels.
void nhwc_reusable_reg_fused_reduction(volatile __global atomic_float *dst,
        off_t dst_offset, float *sum, __local float *local_sum,
        off_t vect_size) {
    const int local_id = get_local_id(1);
    const int simd_id = get_sub_group_local_id();
    const int row_size = vect_size * SUB_GROUP_SIZE;
    const int group_size = get_local_size(1);
    if (local_id > 0) {
        unroll_4_for(int v_idx = 0; v_idx < vect_size; v_idx++) {
            const int slm_offset
                    = local_id * row_size + v_idx * SUB_GROUP_SIZE + simd_id;
            local_sum[slm_offset] = sum[v_idx];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0) {
        unroll_16_for(int l_id = 1; l_id < group_size; l_id++) {
            for (int v_idx = 0; v_idx < vect_size; v_idx++) {
                const int off
                        = l_id * row_size + v_idx * SUB_GROUP_SIZE + simd_id;
                sum[v_idx] += local_sum[off];
            }
        }
        unroll_4_for(int v_idx = 0; v_idx < vect_size; v_idx++) {
            const int off = v_idx * SUB_GROUP_SIZE + simd_id;
            atomic_add_global(&dst[dst_offset + off], sum[v_idx]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return;
}

// Atomic-based reduction for regular algorithm,
// for kernels with private buffers.
void nhwc_reusable_reg_fused_reduction_buff(volatile __global atomic_float *dst,
        off_t dst_offset, float *sum, __local float *local_sum,
        off_t ic_block) {

    const int local_id = get_local_id(1);
    const int simd_id = get_sub_group_local_id();
    const int group_size = get_local_size(1);
    const int row_size = ic_block;
    const int ic_block_sgroups = ic_block / SUB_GROUP_SIZE;

    if (local_id > 0) {
        unroll_16_for(int sg = 0; sg < ic_block_sgroups; sg++) {
            const int slm_offset
                    = local_id * row_size + sg * SUB_GROUP_SIZE + simd_id;
            local_sum[slm_offset] = sum[sg];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0) {
        unroll_16_for(int l_id = 1; l_id < group_size; l_id++) {
            unroll_4_for(int sg = 0; sg < ic_block_sgroups; sg++) {
                const int off = l_id * row_size + sg * SUB_GROUP_SIZE + simd_id;
                sum[sg] += local_sum[off];
            }
        }
        unroll_4_for(int sg = 0; sg < ic_block_sgroups; sg++) {
            const int off = sg * SUB_GROUP_SIZE + simd_id;
            atomic_add_global(&dst[dst_offset + off], sum[sg]);
        }
    }
    return;
}

// Calculate mean, regular algorithm, no private memory buffers used.
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) __kernel void
nhwc_reusable_calc_mean(__global DATA_T *src, __global float *reduce_temp,
        volatile __global atomic_float *mean, off_t ic_size, off_t ic_block,
        off_t sp_size, off_t stat_sp_block, off_t reduce_stat_nblocks,
        int use_fused_atomics_reduction, __local float *local_sum) {
    const int c = get_global_id(0);
    const int sp_block_idx = get_global_id(1);

    const int ic_block_offset = (c / SUB_GROUP_SIZE) * ic_block;
    const int src_off
            = ic_block_offset + sp_block_idx * stat_sp_block * ic_size;

    // reduce_temp layout: reduce_stat_nblocks rows x ic columns
    const int reduce_off = ic_block_offset + sp_block_idx * ic_size;

    src += src_off;
    reduce_temp += reduce_off;

    const int sp_idx_bnd = sp_size % stat_sp_block
            ? min(stat_sp_block, sp_size - sp_block_idx * stat_sp_block)
            : stat_sp_block;
    const int ic_block_sgroups = ic_block / SUB_GROUP_SIZE;
    const int ic_tail_sgroups = ic_block_sgroups % VECT_SIZE;
    const int ic_vect_sgroups = ic_block_sgroups - ic_tail_sgroups;

    // vectorized part
    for (int sg = 0; sg < ic_block_sgroups / VECT_SIZE; ++sg) {
        VECT_FLOAT_T v_mean = 0.0f;
        // reduce
        for (int sp = 0; sp < sp_idx_bnd; ++sp) {
            VECT_FLOAT_T tmp = block_load(
                    tmp, &src[sg * SUB_GROUP_SIZE * VECT_SIZE + sp * ic_size]);
            v_mean += tmp;
        }
        // store res
        if (use_fused_atomics_reduction) {
            const int dst_off
                    = ic_block_offset + sg * VECT_SIZE * SUB_GROUP_SIZE;
            nhwc_reusable_reg_fused_reduction(
                    mean, dst_off, (float *)(&v_mean), local_sum, VECT_SIZE);
        } else {
            const int sg_off = sg * VECT_SIZE * SUB_GROUP_SIZE;
            for (int v_idx = 0; v_idx < VECT_SIZE; v_idx++) {
                float tmp_store =
#if VECT_SIZE > 1
                        v_mean[v_idx];
#else
                        v_mean;
#endif
                block_write(&reduce_temp[sg_off + v_idx * SUB_GROUP_SIZE],
                        tmp_store);
            }
        }
    }
    // tails
    for (int sg = 0; sg < ic_tail_sgroups; ++sg) {
        float v_mean = 0.0f;
        // reduce
        for (int sp = 0; sp < sp_idx_bnd; ++sp) {
            float tmp = block_load(tmp,
                    &src[(ic_vect_sgroups + sg) * SUB_GROUP_SIZE
                            + sp * ic_size]);
            v_mean += tmp;
        }
        // store res
        if (use_fused_atomics_reduction) {
            const int dst_off
                    = ic_block_offset + (ic_vect_sgroups + sg) * SUB_GROUP_SIZE;
            nhwc_reusable_reg_fused_reduction(
                    mean, dst_off, &v_mean, local_sum, 1);
        } else {
            const int sg_off = (ic_vect_sgroups + sg) * SUB_GROUP_SIZE;
            block_write(&reduce_temp[sg_off], v_mean);
        }
    }
}

// Calculate mean, regular algorithm, private memory buffers used.
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) __kernel void
nhwc_reusable_calc_mean_buff(__global DATA_T *src, __global float *reduce_temp,
        volatile __global atomic_float *mean, off_t ic_size, off_t ic_block,
        off_t sp_size, off_t stat_sp_block, off_t reduce_stat_nblocks,
        int use_fused_atomics_reduction, __local float *local_sum) {

    const int c = get_global_id(0);
    const int sp_block_idx = get_global_id(1);

    const int ic_block_offset = (c / SUB_GROUP_SIZE) * ic_block;
    const int src_off
            = ic_block_offset + sp_block_idx * stat_sp_block * ic_size;

    // reduce_temp layout: reduce_stat_nblocks rows x ic columns
    const int reduce_off = ic_block_offset + sp_block_idx * ic_size;

    src += src_off;
    reduce_temp += reduce_off;

    const int sp_idx_bnd = sp_size % stat_sp_block
            ? min(stat_sp_block, sp_size - sp_block_idx * stat_sp_block)
            : stat_sp_block;
    const int ic_block_sgroups
            = min(ic_size - ic_block_offset, ic_block) / SUB_GROUP_SIZE;
    const int ic_vect_sgroups = ic_block_sgroups / VECT_SIZE;
    const int ic_tail_sgroups = ic_block_sgroups % VECT_SIZE;

    float v_mean[MAX_IC_BLOCK_SGROUPS] = {0.0f};
    for (int sp = 0; sp < sp_idx_bnd; ++sp) {
        // vectorized part
        for (int sg = 0; sg < ic_vect_sgroups; ++sg) {
            float s_vect[VECT_SIZE];
            block_load(
                    s_vect, &src[sg * SUB_GROUP_SIZE * VECT_SIZE], VECT_SIZE);
            for (int vect = 0; vect < VECT_SIZE; ++vect) {
                v_mean[sg * VECT_SIZE + vect] += s_vect[vect];
            }
        }
#if MAY_HAVE_IC_TAIL
        // tails
        for (int sg = 0; sg < ic_tail_sgroups; ++sg) {
            const int sg_idx = ic_vect_sgroups * VECT_SIZE + sg;
            float tmp = block_load(tmp, &src[sg_idx * SUB_GROUP_SIZE]);
            v_mean[sg_idx] += tmp;
        }
#endif // HAS_IC_VECT_TAIL
        src += ic_size;
    } // sp_loop

    // store res
    if (use_fused_atomics_reduction) {
        nhwc_reusable_reg_fused_reduction_buff(
                mean, ic_block_offset, (float *)(&v_mean), local_sum, ic_block);
    } else {
        for (int sg = 0; sg < ic_block_sgroups; ++sg) {
            const int sg_off = sg * SUB_GROUP_SIZE;
            block_write(&reduce_temp[sg_off], v_mean[sg]);
        }
    }
}

// Calculate variance, regular algorithm, no private memory buffers used.
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) __kernel void
nhwc_reusable_calc_var(__global DATA_T *src, __global float *mean,
        __global float *reduce_temp, volatile __global atomic_float *variance,
        off_t ic_size, off_t ic_block, off_t sp_size, off_t stat_sp_block,
        off_t reduce_stat_nblocks, int use_fused_atomics_reduction,
        __local float *local_sum) {

    const int c = get_global_id(0);
    const int sp_block_idx = get_global_id(1);

    const int ic_block_offset = (c / SUB_GROUP_SIZE) * ic_block;
    const int src_off
            = ic_block_offset + sp_block_idx * stat_sp_block * ic_size;

    // reduce_temp layout: reduce_stat_nblocks rows x ic columns
    const int reduce_off = ic_block_offset + sp_block_idx * ic_size;

    src += src_off;
    reduce_temp += reduce_off + reduce_stat_nblocks * ic_size;
    mean += ic_block_offset;

    const int sp_idx_bnd = sp_size % stat_sp_block
            ? min(stat_sp_block, sp_size - sp_block_idx * stat_sp_block)
            : stat_sp_block;
    const int ic_block_sgroups = ic_block / SUB_GROUP_SIZE;
    const int ic_tail_sgroups = ic_block_sgroups % VECT_SIZE;
    const int ic_vect_sgroups = ic_block_sgroups - ic_tail_sgroups;

    // vectorized part
    for (int sg = 0; sg < ic_block_sgroups / VECT_SIZE; ++sg) {
        VECT_FLOAT_T v_var = 0.0f;
        VECT_FLOAT_T v_mean
                = block_load(v_mean, &mean[sg * SUB_GROUP_SIZE * VECT_SIZE]);
        // reduce
        for (int sp = 0; sp < sp_idx_bnd; ++sp) {
            VECT_FLOAT_T tmp = block_load(
                    tmp, &src[sg * SUB_GROUP_SIZE * VECT_SIZE + sp * ic_size]);
            const VECT_FLOAT_T v0 = tmp - v_mean;
            v_var = fma(v0, v0, v_var);
        }
        // store res
        if (use_fused_atomics_reduction) {
            const int dst_off
                    = ic_block_offset + sg * VECT_SIZE * SUB_GROUP_SIZE;
            nhwc_reusable_reg_fused_reduction(
                    variance, dst_off, (float *)(&v_var), local_sum, VECT_SIZE);
        } else {
            const int sg_off = sg * VECT_SIZE * SUB_GROUP_SIZE;
            for (int v_idx = 0; v_idx < VECT_SIZE; v_idx++) {
                float tmp_store =
#if VECT_SIZE > 1
                        v_var[v_idx];
#else
                        v_var;
#endif
                block_write(&reduce_temp[sg_off + v_idx * SUB_GROUP_SIZE],
                        tmp_store);
            }
        }
    }
    // tails
    for (int sg = 0; sg < ic_tail_sgroups; ++sg) {
        float v_var = 0.0f;
        float v_mean = block_load(
                v_mean, &mean[(ic_vect_sgroups + sg) * SUB_GROUP_SIZE]);
        // reduce
        for (int sp = 0; sp < sp_idx_bnd; ++sp) {
            float tmp = block_load(tmp,
                    &src[(ic_vect_sgroups + sg) * SUB_GROUP_SIZE
                            + sp * ic_size]);
            const float v0 = tmp - v_mean;
            v_var = fma(v0, v0, v_var);
        }
        // store res
        if (use_fused_atomics_reduction) {
            const int dst_off
                    = ic_block_offset + (ic_vect_sgroups + sg) * SUB_GROUP_SIZE;
            nhwc_reusable_reg_fused_reduction(
                    variance, dst_off, &v_var, local_sum, 1);
        } else {
            const int sg_off = (ic_vect_sgroups + sg) * SUB_GROUP_SIZE;
            block_write(&reduce_temp[sg_off], v_var);
        }
    }
}

// Calculate variance, regular algorithm, private memory buffers used.
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) __kernel void
nhwc_reusable_calc_var_buff(__global DATA_T *src, __global float *mean,
        __global float *reduce_temp, volatile __global atomic_float *variance,
        off_t ic_size, off_t ic_block, off_t sp_size, off_t stat_sp_block,
        off_t reduce_stat_nblocks, int use_fused_atomics_reduction,
        __local float *local_sum) {

    const int c = get_global_id(0);
    const int sp_block_idx = get_global_id(1);

    const int ic_block_offset = (c / SUB_GROUP_SIZE) * ic_block;
    const int src_off
            = ic_block_offset + sp_block_idx * stat_sp_block * ic_size;

    // reduce_temp layout: reduce_stat_nblocks rows x ic columns
    const int reduce_off = ic_block_offset + sp_block_idx * ic_size;

    src += src_off;
    reduce_temp += reduce_off + reduce_stat_nblocks * ic_size;
    mean += ic_block_offset;

    const int sp_idx_bnd = sp_size % stat_sp_block
            ? min(stat_sp_block, sp_size - sp_block_idx * stat_sp_block)
            : stat_sp_block;
    const int ic_block_sgroups
            = min(ic_size - ic_block_offset, ic_block) / SUB_GROUP_SIZE;
    const int ic_vect_sgroups = ic_block_sgroups / VECT_SIZE;
    const int ic_tail_sgroups = ic_block_sgroups % VECT_SIZE;

    float v_mean[MAX_IC_BLOCK_SGROUPS] = {0.0f};
    for (int sg = 0; sg < ic_block_sgroups; ++sg) {
        v_mean[sg] = block_load(v_mean[sg], &mean[(sg * SUB_GROUP_SIZE)]);
    }

    float v_var[MAX_IC_BLOCK_SGROUPS] = {0.0f};
    float v0[MAX_IC_BLOCK_SGROUPS] = {0.0f};

    for (int sp = 0; sp < sp_idx_bnd; ++sp) {
        // vectorized part
        for (int sg = 0; sg < ic_vect_sgroups; ++sg) {
            float s_vect[VECT_SIZE];
            block_load(
                    s_vect, &src[sg * SUB_GROUP_SIZE * VECT_SIZE], VECT_SIZE);
            for (int vect = 0; vect < VECT_SIZE; ++vect) {
                int sg_idx = sg * VECT_SIZE + vect;
                v0[sg_idx] = s_vect[vect] - v_mean[sg_idx];
                v_var[sg_idx] = fma(v0[sg_idx], v0[sg_idx], v_var[sg_idx]);
            }
        }

#if MAY_HAVE_IC_TAIL
        // tails
        for (int sg = 0; sg < ic_tail_sgroups; ++sg) {
            const int sg_idx = ic_vect_sgroups * VECT_SIZE + sg;
            float s_tail = block_load(s_tail, &src[sg_idx * SUB_GROUP_SIZE]);
            v0[sg_idx] = s_tail - v_mean[sg_idx];
            v_var[sg_idx] = fma(v0[sg_idx], v0[sg_idx], v_var[sg_idx]);
        }
#endif // HAS_IC_VECT_TAIL
        src += ic_size;
    } // sp_loop

    // store res
    if (use_fused_atomics_reduction) {
        nhwc_reusable_reg_fused_reduction_buff(variance, ic_block_offset,
                (float *)(&v_var), local_sum, ic_block);
    } else {
        for (int sg = 0; sg < ic_block_sgroups; ++sg) {
            const int sg_off = sg * SUB_GROUP_SIZE;
            block_write(&reduce_temp[sg_off], v_var[sg]);
        }
    }
}

// Calculate mean and variance at once, 1pass algorithm
// no private memory buffers used.
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) __kernel void
nhwc_reusable_calc_mean_var(__global DATA_T *src, __global float *reduce_temp,
        volatile __global atomic_float *mean,
        volatile __global atomic_float *variance, off_t ic_size, off_t ic_block,
        off_t sp_size, off_t stat_sp_block, off_t reduce_stat_nblocks,
        int use_fused_atomics_reduction, __local SUM_DATA_T *local_sum,
        __local SUM_DATA_T *local_sum_sq) {
    const int c = get_global_id(0);
    const int sp_block_idx = get_global_id(1);
    const int simd_id = get_sub_group_local_id();

    const int ic_block_offset = (c / SUB_GROUP_SIZE) * ic_block;
    const int src_off
            = ic_block_offset + sp_block_idx * stat_sp_block * ic_size;

    // reduce_temp layout: reduce_stat_nblocks rows x ic columns
    const int reduce_off = ic_block_offset + sp_block_idx * ic_size;

    const int variance_off = reduce_stat_nblocks * ic_size;

    src += src_off;
    reduce_temp += reduce_off;

    const int sp_idx_bnd = sp_size % stat_sp_block
            ? min(stat_sp_block, sp_size - sp_block_idx * stat_sp_block)
            : stat_sp_block;
    const int ic_block_sgroups = ic_block / SUB_GROUP_SIZE;
    const int ic_tail_sgroups = ic_block_sgroups % VECT_SIZE;
    const int ic_vect_sgroups = ic_block_sgroups - ic_tail_sgroups;

    // vectorized part
    for (int sg = 0; sg < ic_block_sgroups / VECT_SIZE; ++sg) {
        SUM_DATA_T sum[VECT_SIZE] = {0.0f};
        SUM_DATA_T sum_sq[VECT_SIZE] = {0.0f};
        // reduce
        for (int sp = 0; sp < sp_idx_bnd; ++sp) {
            VECT_FLOAT_T s_vect = block_load(s_vect,
                    &src[sg * SUB_GROUP_SIZE * VECT_SIZE + sp * ic_size]);
            for (int v_idx = 0; v_idx < VECT_SIZE; ++v_idx) {
#if VECT_SIZE > 1
#define S_VECT s_vect[v_idx]
#else
#define S_VECT s_vect
#endif
                sum[v_idx] = summation(S_VECT, sum[v_idx]);
                sum_sq[v_idx] = summation(S_VECT * S_VECT, sum_sq[v_idx]);
            }
        }
        // store res
        if (use_fused_atomics_reduction) {
            const int dst_off
                    = ic_block_offset + sg * VECT_SIZE * SUB_GROUP_SIZE;
            nhwc_reusable_1pass_fused_reduction(mean, variance, dst_off, sum,
                    sum_sq, local_sum, local_sum_sq, VECT_SIZE);
        } else {

            const int sg_off = sg * VECT_SIZE * SUB_GROUP_SIZE;
            for (int v_idx = 0; v_idx < VECT_SIZE; v_idx++) {
                const int reduce_off = sg_off + v_idx * SUB_GROUP_SIZE;
                block_write(&reduce_temp[reduce_off], sum[v_idx].s0);
                block_write(&reduce_temp[variance_off + reduce_off],
                        sum_sq[v_idx].s0);
            }
        }
    }
    // tails
    for (int sg = 0; sg < ic_tail_sgroups; ++sg) {
        SUM_DATA_T sum = 0.0f;
        SUM_DATA_T sum_sq = 0.0f;
        for (int sp = 0; sp < sp_idx_bnd; ++sp) {
            float src_v = block_load(src_v,
                    &src[(ic_vect_sgroups + sg) * SUB_GROUP_SIZE
                            + sp * ic_size]);
            sum = summation(src_v, sum);
            sum_sq = summation(src_v * src_v, sum_sq);
        }
        // store res
        if (use_fused_atomics_reduction) {
            const int dst_off
                    = ic_block_offset + (ic_vect_sgroups + sg) * SUB_GROUP_SIZE;
            nhwc_reusable_1pass_fused_reduction(mean, variance, dst_off, &sum,
                    &sum_sq, local_sum, local_sum_sq, 1);
        } else {
            const int sg_off = (ic_vect_sgroups + sg) * SUB_GROUP_SIZE;
            block_write(&reduce_temp[sg_off], sum.s0);
            block_write(&reduce_temp[variance_off + sg_off], sum_sq.s0);
        }
    }
}

// Calculate mean and variance at once, 1pass algorithm,
// private memory buffers used.
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) __kernel void
nhwc_reusable_calc_mean_var_buff(__global DATA_T *src,
        __global float *reduce_temp, volatile __global atomic_float *mean,
        volatile __global atomic_float *variance, off_t ic_size, off_t ic_block,
        off_t sp_size, off_t stat_sp_block, off_t reduce_stat_nblocks,
        int use_fused_atomics_reduction, __local SUM_DATA_T *local_sum,
        __local SUM_DATA_T *local_sum_sq) {

    const int c = get_global_id(0);
    const int sp_block_idx = get_global_id(1);
    const int simd_id = get_sub_group_local_id();

    const int ic_block_offset = (c / SUB_GROUP_SIZE) * ic_block;
    const int src_off
            = ic_block_offset + sp_block_idx * stat_sp_block * ic_size;

    // reduce_temp layout: reduce_stat_nblocks rows x ic columns
    const int reduce_off = ic_block_offset + sp_block_idx * ic_size;

    const int variance_off = reduce_stat_nblocks * ic_size;

    src += src_off;
    reduce_temp += reduce_off;

    const int sp_idx_bnd = sp_size % stat_sp_block
            ? min(stat_sp_block, sp_size - sp_block_idx * stat_sp_block)
            : stat_sp_block;
    const int ic_block_sgroups
            = min(ic_size - ic_block_offset, ic_block) / SUB_GROUP_SIZE;
    const int ic_vect_sgroups = ic_block_sgroups / VECT_SIZE;
    const int ic_tail_sgroups = ic_block_sgroups % VECT_SIZE;

    SUM_DATA_T sum[MAX_IC_BLOCK_SGROUPS] = {0.0f};
    SUM_DATA_T sum_sq[MAX_IC_BLOCK_SGROUPS] = {0.0f};

    for (int sp = 0; sp < sp_idx_bnd; ++sp) {
        // vectorized part
        for (int sg = 0; sg < ic_vect_sgroups; ++sg) {
            float s_vect[VECT_SIZE];
            block_load(
                    s_vect, &src[sg * SUB_GROUP_SIZE * VECT_SIZE], VECT_SIZE);
            for (int vect = 0; vect < VECT_SIZE; ++vect) {
                const int sum_idx = sg * VECT_SIZE + vect;
                sum[sum_idx] = summation(s_vect[vect], sum[sum_idx]);
                sum_sq[sum_idx] = summation(
                        s_vect[vect] * s_vect[vect], sum_sq[sum_idx]);
            }
        }
#if MAY_HAVE_IC_TAIL
        // tails
        for (int sg = 0; sg < ic_tail_sgroups; ++sg) {
            const int sg_idx = ic_vect_sgroups * VECT_SIZE + sg;
            float s_tail = block_load(s_tail, &src[sg_idx * SUB_GROUP_SIZE]);
            sum[sg_idx] = summation(s_tail, sum[sg_idx]);
            sum_sq[sg_idx] = summation(s_tail * s_tail, sum_sq[sg_idx]);
        }
#endif
        src += ic_size;
    }
    // store res
    if (use_fused_atomics_reduction) {
        nhwc_reusable_1pass_fused_reduction_buff(mean, variance,
                ic_block_offset, sum, sum_sq, local_sum, local_sum_sq,
                ic_block);
    } else {
        for (int sg = 0; sg < ic_block_sgroups; ++sg) {
            const int reduce_off = sg * SUB_GROUP_SIZE;
            block_write(&reduce_temp[reduce_off], sum[sg].s0);
            block_write(&reduce_temp[variance_off + reduce_off], sum_sq[sg].s0);
        }
    }
}

// Main FWD kernel, common for regular and 1pass algorithms
// no private memory buffers used.
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) __kernel void
nhwc_reusable_norm_fwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *dst,
        __global float *scaleshift, __global float *shift, __global char *ws,
        float eps, __global DATA_T *src_add, float relu_alpha, off_t ic_size,
        off_t ic_block, off_t sp_size, off_t update_sp_block) {
    const int c = get_global_id(0);
    const int sp = get_global_id(1) * update_sp_block;

    const int ic_block_offset = (c / SUB_GROUP_SIZE) * ic_block;
    mean += ic_block_offset;
    variance += ic_block_offset;
    shift += ic_block_offset;
    scaleshift += ic_block_offset;
    const uint d_off = sp * ic_size + ic_block_offset;

    src += d_off;
#if FUSE_BN_ADD_RELU
    src_add += d_off;
#endif
    dst += d_off;
#if FUSE_BN_RELU && IS_TRAINING
    ws += d_off;
#endif

    const bool has_sp_block_tail = sp_size % update_sp_block;
    const int sp_idx_bnd = has_sp_block_tail
            ? min(update_sp_block, sp_size - sp)
            : update_sp_block;
    const int ic_block_sgroups = ic_block / SUB_GROUP_SIZE;
    for (int sp_idx = 0; sp_idx < sp_idx_bnd; sp_idx++) {
        // vectorized part
        for (int sg = 0; sg < ic_block_sgroups / VECT_SIZE; ++sg) {
            const int sg_idx = sg * SUB_GROUP_SIZE * VECT_SIZE;
            VECT_FLOAT_T sm;
            if (USE_SCALE) {
                sm = block_load(sm, &scaleshift[sg_idx]);
            } else {
                sm = (VECT_FLOAT_T)1.0f;
            }
            VECT_FLOAT_T sv;
            if (USE_SHIFT) {
                sv = block_load(sv, &shift[sg_idx]);
            } else {
                sv = (VECT_FLOAT_T)0.0f;
            }
            VECT_FLOAT_T s_vect = block_load(s_vect, &src[sg_idx]);
            VECT_FLOAT_T v_mean = block_load(v_mean, &mean[sg_idx]);
            VECT_FLOAT_T v_variance = block_load(v_variance, &variance[sg_idx]);
            const VECT_FLOAT_T sqrt_variance
                    = sm / sqrt(v_variance + (VECT_FLOAT_T)eps);
            VECT_FLOAT_T d_vect = fma(s_vect - v_mean, sqrt_variance, sv);

#if FUSE_BN_RELU
#if FUSE_BN_ADD_RELU
            {
                VECT_FLOAT_T tmp_add = block_load(tmp_add, &src_add[sg_idx]);
                d_vect += tmp_add;
            }
#endif
            const VECT_N(int) ws_vect = ISGREATER(d_vect, (VECT_FLOAT_T)0.0f);
            d_vect = select((VECT_FLOAT_T)0.0f, d_vect, ws_vect);
#if IS_TRAINING
            {
                VECT_N(char) tmp_ws = VECT_N(convert_char)(ws_vect);
                block_write(&ws[sg_idx], &tmp_ws);
            }
#endif // IS_TRAINING
#endif // FUSE_BN_RELU

#if WITH_RELU && WITH_LEAKY_RELU
            VECT_N(int) l_vect;
#endif //WITH_RELU && WITH_LEAKY_RELU
#if WITH_RELU
#if WITH_LEAKY_RELU
            l_vect = isless(d_vect, 0.0f);
            d_vect = select(d_vect, d_vect * relu_alpha, l_vect);
#else
            d_vect = max(d_vect, (VECT_FLOAT_T)0.0f);
#endif //WITH_LEAKY_RELU
#endif //WITH_RELU

            block_write(&dst[sg_idx], &d_vect);
        } // sg loop

        const int ic_tail_sgroups = (ic_block / SUB_GROUP_SIZE) % VECT_SIZE;
        const int ic_vect_sgroups = ic_block_sgroups - ic_tail_sgroups;
        const bool has_ic_vect_tail = ic_tail_sgroups > 0;
        if (has_ic_vect_tail) {
            // tails
            for (int sg = 0; sg < ic_tail_sgroups; ++sg) {
                const int sg_idx = (ic_vect_sgroups + sg) * SUB_GROUP_SIZE;

                float sm_tail;
                if (USE_SCALE) {
                    sm_tail = block_load(sm_tail, &scaleshift[sg_idx]);
                } else {
                    sm_tail = 1.0f;
                }
                float sv_tail;
                if (USE_SHIFT) {
                    sv_tail = block_load(sv_tail, &shift[sg_idx]);
                } else {
                    sv_tail = 0.0f;
                }
                float v_mean_tail = block_load(v_mean_tail, &mean[sg_idx]);
                float v_variance_tail
                        = block_load(v_variance_tail, &variance[sg_idx]);
                const float sqrt_variance_tail
                        = sm_tail / sqrt(v_variance_tail + eps);
                float s_tail = block_load(s_tail, &src[sg_idx]);
                float d_tail = fma(
                        s_tail - v_mean_tail, sqrt_variance_tail, sv_tail);

                if (FUSE_BN_ADD_RELU) {
                    float tmp_add = block_load(tmp_add, &src_add[sg_idx]);
                    d_tail += tmp_add;
                }
#if FUSE_BN_RELU
                if (d_tail <= 0) d_tail = 0.0f;
#if IS_TRAINING
                const int ws_tail = d_tail > 0.0f ? -1 : 0;
                {
                    char tmp_ws = convert_char(ws_tail);
                    block_write(&ws[sg_idx], tmp_ws);
                }
#endif // IS_TRAINING
#endif // FUSE_BN_RELU
#if WITH_RELU
#if WITH_LEAKY_RELU
                if (d_tail < 0) d_tail *= relu_alpha;
#else
                d_tail = max(d_tail, 0.0f);
#endif //WITH_LEAKY_RELU
#endif //WITH_RELU
                block_write(&dst[sg_idx], d_tail);
            }
        } // has_ic_vect_tail

        src += ic_size;
#if FUSE_BN_ADD_RELU
        src_add += ic_size;
#endif
        dst += ic_size;
#if FUSE_BN_RELU && IS_TRAINING
        ws += ic_size;
#endif
    } // sp loop
}

// Main FWD kernel, common for regular and 1pass algorithms,
// private memory buffers used.
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) __kernel void
nhwc_reusable_norm_fwd_buff(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *dst,
        __global float *scaleshift, __global float *shift, __global char *ws,
        float eps, __global DATA_T *src_add, float relu_alpha, off_t ic_size,
        off_t ic_block, off_t sp_size, off_t update_sp_block) {

    const int c = get_global_id(0);
    const int sp = get_global_id(1) * update_sp_block;

    const int ic_block_offset = (c / SUB_GROUP_SIZE) * ic_block;

    mean += ic_block_offset;
    variance += ic_block_offset;
    shift += ic_block_offset;
    scaleshift += ic_block_offset;
    const uint d_off = sp * ic_size + ic_block_offset;

    src += d_off;
#if FUSE_BN_ADD_RELU
    src_add += d_off;
#endif
    dst += d_off;
#if FUSE_BN_RELU && IS_TRAINING
    ws += d_off;
#endif

    VECT_FLOAT_T sm_v[MAX_IC_VECT_SGROUPS], sv_v[MAX_IC_VECT_SGROUPS],
            v_mean_v[MAX_IC_VECT_SGROUPS], v_variance_v[MAX_IC_VECT_SGROUPS],
            sqrt_variance_v[MAX_IC_VECT_SGROUPS];
#if MAY_HAVE_IC_TAIL
    float sm_t[MAX_IC_TAIL_SGROUPS], sv_t[MAX_IC_TAIL_SGROUPS],
            v_mean_t[MAX_IC_TAIL_SGROUPS], v_variance_t[MAX_IC_TAIL_SGROUPS],
            sqrt_variance_t[MAX_IC_TAIL_SGROUPS];
#endif

    const bool has_sp_block_tail = sp_size % update_sp_block;
    const int sp_idx_bnd = has_sp_block_tail
            ? min(update_sp_block, sp_size - sp)
            : update_sp_block;

    const int ic_block_sgroups
            = min(ic_size - ic_block_offset, ic_block) / SUB_GROUP_SIZE;
    const int ic_vect_sgroups = ic_block_sgroups / VECT_SIZE;
    const int ic_tail_sgroups = ic_block_sgroups % VECT_SIZE;
    const bool has_ic_vect_tail = ic_tail_sgroups > 0;

    for (int sg = 0; sg < ic_vect_sgroups; ++sg) {
        const int sg_idx = sg * SUB_GROUP_SIZE * VECT_SIZE;

        if (USE_SCALE) {
            sm_v[sg] = block_load(sm_v[sg], &scaleshift[sg_idx]);
        } else {
            sm_v[sg] = (VECT_FLOAT_T)1.0f;
        }
        if (USE_SHIFT) {
            sv_v[sg] = block_load(sv_v[sg], &shift[sg_idx]);
        } else {
            sv_v[sg] = (VECT_FLOAT_T)0.0f;
        }
        v_mean_v[sg] = block_load(v_mean_v[sg], &mean[sg_idx]);
        v_variance_v[sg] = block_load(v_variance_v[sg], &variance[sg_idx]);
        sqrt_variance_v[sg]
                = sm_v[sg] / sqrt(v_variance_v[sg] + (VECT_FLOAT_T)eps);
    }

#if MAY_HAVE_IC_TAIL
    for (int sg = 0; sg < ic_tail_sgroups; ++sg) {
        const int sg_idx = (ic_vect_sgroups * VECT_SIZE + sg) * SUB_GROUP_SIZE;
        if (USE_SCALE) {
            sm_t[sg] = block_load(sm_t[sg], &scaleshift[sg_idx]);
        } else {
            sm_t[sg] = 1.0f;
        }
        if (USE_SHIFT) {
            sv_t[sg] = block_load(sv_t[sg], &shift[sg_idx]);
        } else {
            sv_t[sg] = 0.0f;
        }
        v_mean_t[sg] = block_load(v_mean_t[sg], &mean[sg_idx]);
        v_variance_t[sg] = block_load(v_variance_t[sg], &variance[sg_idx]);
        sqrt_variance_t[sg] = sm_t[sg] / sqrt(v_variance_t[sg] + eps);
    }
#endif //MAY_HAVE_IC_TAIL

    for (int sp_idx = 0; sp_idx < sp_idx_bnd; sp_idx++) {
        // vectorized part
        for (int sg = 0; sg < ic_vect_sgroups; ++sg) {
            const int sg_idx = sg * SUB_GROUP_SIZE * VECT_SIZE;

            VECT_FLOAT_T d_vect;
            VECT_FLOAT_T s_vect = block_load(s_vect, &src[sg_idx]);
            d_vect = fma(s_vect - v_mean_v[sg], sqrt_variance_v[sg], sv_v[sg]);

#if FUSE_BN_RELU
#if FUSE_BN_ADD_RELU
            {
                VECT_FLOAT_T tmp_add = block_load(tmp_add, &src_add[sg_idx]);
                d_vect += tmp_add;
            }
#endif
            const VECT_N(int) ws_vect = ISGREATER(d_vect, (VECT_FLOAT_T)0.0f);
            d_vect = select((VECT_FLOAT_T)0.0f, d_vect, ws_vect);
#if IS_TRAINING
            {
                VECT_N(char) tmp_ws = VECT_N(convert_char)(ws_vect);
                block_write(&ws[sg_idx], &tmp_ws);
            }
#endif // IS_TRAINING
#endif // FUSE_BN_RELU

#if WITH_RELU && WITH_LEAKY_RELU
            VECT_N(int) l_vect;
#endif //WITH_RELU && WITH_LEAKY_RELU
#if WITH_RELU
#if WITH_LEAKY_RELU
            l_vect = isless(d_vect, 0.0f);
            d_vect = select(d_vect, d_vect * relu_alpha, l_vect);
#else
            d_vect = max(d_vect, (VECT_FLOAT_T)0.0f);
#endif //WITH_LEAKY_RELU
#endif //WITH_RELU
            block_write(&dst[sg_idx], &d_vect);
        } // sg loop

#if MAY_HAVE_IC_TAIL
        // tails
        for (int sg = 0; sg < ic_tail_sgroups; ++sg) {
            const int sg_idx
                    = (ic_vect_sgroups * VECT_SIZE + sg) * SUB_GROUP_SIZE;
            float d_tail;
            float s_tail = block_load(s_tail, &src[sg_idx]);
            d_tail = fma(s_tail - v_mean_t[sg], sqrt_variance_t[sg], sv_t[sg]);
            if (FUSE_BN_ADD_RELU) {
                float tmp_add = block_load(tmp_add, &src_add[sg_idx]);
                d_tail += tmp_add;
            }
#if FUSE_BN_RELU
            if (d_tail <= 0) d_tail = 0.0f;
#if IS_TRAINING
            const int ws_tail = d_tail > 0.0f ? -1 : 0;
            {
                char tmp_ws = convert_char(ws_tail);
                block_write(&ws[sg_idx], tmp_ws);
            }
#endif // IS_TRAINING
#endif // FUSE_BN_RELU
#if WITH_RELU
#if WITH_LEAKY_RELU
            if (d_tail < 0) d_tail *= relu_alpha;
#else
            d_tail = max(d_tail, 0.0f);
#endif //WITH_LEAKY_RELU
#endif //WITH_RELU
            block_write(&dst[sg_idx], d_tail);
        }
#endif //MAY_HAVE_IC_TAIL
        src += ic_size;
#if FUSE_BN_ADD_RELU
        src_add += ic_size;
#endif
        dst += ic_size;
#if FUSE_BN_RELU && IS_TRAINING
        ws += ic_size;
#endif
    } // sp loop
}

// Atomic-based reduction, BWD pass, for no private buffers kernels.
void nhwc_reusable_bwd_fused_reduction(
        volatile __global atomic_float *diff_scale,
        volatile __global atomic_float *diff_shift, off_t dst_offset,
        float *diff_gamma, float *diff_beta, __local float *local_sums,
        off_t vect_size, off_t calc_slm_size) {
    const int local_id = get_local_id(1);
    const int simd_id = get_sub_group_local_id();
    const int row_size = vect_size * SUB_GROUP_SIZE;
    const int group_size = get_local_size(1);

    __local float *local_gamma = local_sums;
    __local float *local_beta = local_sums + calc_slm_size / sizeof(float);

    if (local_id > 0) {
        unroll_4_for(int v_idx = 0; v_idx < vect_size; v_idx++) {
            const int slm_offset
                    = local_id * row_size + v_idx * SUB_GROUP_SIZE + simd_id;
            local_gamma[slm_offset] = diff_gamma[v_idx];
            local_beta[slm_offset] = diff_beta[v_idx];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0) {
        unroll_16_for(int l_id = 1; l_id < group_size; l_id++) {
            for (int v_idx = 0; v_idx < vect_size; v_idx++) {
                const int off
                        = l_id * row_size + v_idx * SUB_GROUP_SIZE + simd_id;
                diff_gamma[v_idx] += local_gamma[off];
                diff_beta[v_idx] += local_beta[off];
            }
        }
        unroll_4_for(int v_idx = 0; v_idx < vect_size; v_idx++) {
            const int off = v_idx * SUB_GROUP_SIZE + simd_id;
            atomic_add_global(&diff_scale[dst_offset + off], diff_gamma[v_idx]);
            atomic_add_global(&diff_shift[dst_offset + off], diff_beta[v_idx]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return;
}

// Atomic-based reduction, BWD pass, for kernel with private buffers.
void nhwc_reusable_bwd_fused_reduction_buff(
        volatile __global atomic_float *diff_scale,
        volatile __global atomic_float *diff_shift, off_t dst_offset,
        VECT_FLOAT_T *diff_gamma_v, VECT_FLOAT_T *diff_beta_v,
        float *diff_gamma_t, float *diff_beta_t, __local float *local_sums,
        off_t ic_block, off_t calc_slm_size) {
    const int local_id = get_local_id(1);
    const int simd_id = get_sub_group_local_id();
    const int row_size = ic_block;
    const int group_size = get_local_size(1);

    const int ic_block_sgroups = ic_block / SUB_GROUP_SIZE;
    const int ic_vect_sgroups = ic_block_sgroups / VECT_SIZE;
    const int ic_tail_sgroups = ic_block_sgroups % VECT_SIZE;
    __local float *local_gamma = local_sums;
    __local float *local_beta = local_sums + calc_slm_size / sizeof(float);

    if (local_id > 0) {
        for (int sg = 0; sg < ic_vect_sgroups; sg++) {
            unroll_16_for(int e = 0; e < VECT_SIZE; e++) {
                const int flat_sg = sg * VECT_SIZE + e;
                const int slm_offset = local_id * row_size
                        + flat_sg * SUB_GROUP_SIZE + simd_id;
                local_gamma[slm_offset] = VFLOAT(diff_gamma_v[sg], e);
                local_beta[slm_offset] = VFLOAT(diff_beta_v[sg], e);
            }
        }
#if MAY_HAVE_IC_TAIL
        for (int sg = 0; sg < ic_tail_sgroups; sg++) {
            const int flat_sg = ic_vect_sgroups * VECT_SIZE + sg;
            const int slm_offset
                    = local_id * row_size + flat_sg * SUB_GROUP_SIZE + simd_id;
            local_gamma[slm_offset] = diff_gamma_t[sg];
            local_beta[slm_offset] = diff_beta_t[sg];
        }
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0) {
        unroll_16_for(int l_id = 1; l_id < group_size; l_id++) {
            for (int sg = 0; sg < ic_vect_sgroups; sg++) {
                unroll_4_for(int e = 0; e < VECT_SIZE; e++) {
                    const int flat_sg = sg * VECT_SIZE + e;
                    const int off = l_id * row_size + flat_sg * SUB_GROUP_SIZE
                            + simd_id;
                    VFLOAT(diff_gamma_v[sg], e) += local_gamma[off];
                    VFLOAT(diff_beta_v[sg], e) += local_beta[off];
                }
            }
#if MAY_HAVE_IC_TAIL
            for (int sg = 0; sg < ic_tail_sgroups; sg++) {
                const int flat_sg = ic_vect_sgroups * VECT_SIZE + sg;
                const int off
                        = l_id * row_size + flat_sg * SUB_GROUP_SIZE + simd_id;
                diff_gamma_t[sg] += local_gamma[off];
                diff_beta_t[sg] += local_beta[off];
            }
#endif
        }
        for (int sg = 0; sg < ic_vect_sgroups; sg++) {
            unroll_4_for(int e = 0; e < VECT_SIZE; e++) {
                const int flat_sg = sg * VECT_SIZE + e;
                const int off = flat_sg * SUB_GROUP_SIZE + simd_id;
                atomic_add_global(&diff_scale[dst_offset + off],
                        VFLOAT(diff_gamma_v[sg], e));
                atomic_add_global(&diff_shift[dst_offset + off],
                        VFLOAT(diff_beta_v[sg], e));
            }
        }
#if MAY_HAVE_IC_TAIL
        for (int sg = 0; sg < ic_tail_sgroups; sg++) {
            const int flat_sg = ic_vect_sgroups * VECT_SIZE + sg;
            const int off = flat_sg * SUB_GROUP_SIZE + simd_id;
            atomic_add_global(&diff_scale[dst_offset + off], diff_gamma_t[sg]);
            atomic_add_global(&diff_shift[dst_offset + off], diff_beta_t[sg]);
        }
#endif
    }
    return;
}

// Calculate stats for BWD pass
// no private memory buffers used.
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) __kernel void
nhwc_reusable_calc_stat(__global DATA_T *src, __global float *mean,
        __global DATA_T *diff_dst, __global char *ws,
        __global float *temp_reduce, __global float *temp_reduce_shift,
        volatile __global atomic_float *diff_scale,
        volatile __global atomic_float *diff_shift, off_t ic_size,
        off_t ic_block, off_t sp_size, off_t stat_sp_block,
        off_t reduce_stat_nblocks, int use_fused_atomics_reduction,
        __local float *local_sums, off_t calc_slm_size) {
    const int c = get_global_id(0);
    const int sp_block_idx = get_global_id(1);
    const int ic_block_offset = (c / SUB_GROUP_SIZE) * ic_block;
    const int offset = ic_block_offset + sp_block_idx * stat_sp_block * ic_size;

    mean += ic_block_offset;
    src += offset;
    diff_dst += offset;
    ws += offset;

    // scratchpad layout: (reduce_stat_nblocks + 1) rows x ic columns
    const int reduce_off = ic_block_offset + (sp_block_idx + 1) * ic_size;

    temp_reduce += reduce_off;
    temp_reduce_shift += reduce_off;

    const bool has_sp_block_tail = sp_size % stat_sp_block;
    const int sp_idx_bnd = has_sp_block_tail
            ? min(stat_sp_block, sp_size - sp_block_idx * stat_sp_block)
            : stat_sp_block;
    const int ic_block_sgroups = ic_block / SUB_GROUP_SIZE;
    const int ic_tail_sgroups = ic_block_sgroups % VECT_SIZE;
    const int ic_vect_sgroups = ic_block_sgroups - ic_tail_sgroups;

    // vectorized part

    for (int sg = 0; sg < ic_block_sgroups / VECT_SIZE; ++sg) {
        const int sg_idx = sg * SUB_GROUP_SIZE * VECT_SIZE;
        VECT_FLOAT_T diff_gamma = 0.0f;
        VECT_FLOAT_T diff_beta = 0.0f;
        VECT_FLOAT_T v_mean = block_load(v_mean, &mean[(sg_idx)]);

        // reduce
        for (int sp = 0; sp < sp_idx_bnd; ++sp) {
            const int tn_idx = sg_idx + sp * ic_size;
#if FUSE_BN_RELU
            VECT_N(char) ws_vect = block_load(ws_vect, &ws[tn_idx]);
#endif
            VECT_FLOAT_T src_vect = block_load(src_vect, &src[tn_idx]);
            VECT_FLOAT_T dd_vect = block_load(dd_vect, &diff_dst[tn_idx]);
            const VECT_FLOAT_T v0 = src_vect - v_mean;
#if FUSE_BN_RELU
            dd_vect = select(
                    (VECT_FLOAT_T)0.0f, dd_vect, VECT_N(convert_int)(ws_vect));
#endif
            diff_gamma = fma(v0, dd_vect, diff_gamma);
            diff_beta += dd_vect;
        } // sp loop

        // store results
        if (use_fused_atomics_reduction) {
            const int dst_off
                    = ic_block_offset + sg * VECT_SIZE * SUB_GROUP_SIZE;
            nhwc_reusable_bwd_fused_reduction(diff_scale, diff_shift, dst_off,
                    (float *)(&diff_gamma), (float *)(&diff_beta), local_sums,
                    VECT_SIZE, calc_slm_size);
        } else {
            // Two different scratchpads: for diff_gamma and diff_beta
            // scratchpad layout (elements):
            // ic_size - final reduced data,
            //           wrote by nhwc_reusable_reduce_stats kernel
            // reduce_stat_nblocks * ic_size - initialy reduced data,
            //           calculated by this kernel

            const int sg_off = sg * VECT_SIZE * SUB_GROUP_SIZE;
            for (int v_idx = 0; v_idx < VECT_SIZE; v_idx++) {
                float tmp_dg =
#if VECT_SIZE > 1
                        diff_gamma[v_idx];
#else
                        diff_gamma;
#endif
                block_write(
                        &temp_reduce[sg_off + v_idx * SUB_GROUP_SIZE], tmp_dg);
                float tmp_db =
#if VECT_SIZE > 1
                        diff_beta[v_idx];
#else
                        diff_beta;
#endif
                block_write(&temp_reduce_shift[sg_off + v_idx * SUB_GROUP_SIZE],
                        tmp_db);
            }
        }
    } // sg loop

    // tails
    for (int sg = 0; sg < ic_tail_sgroups; ++sg) {
        const int sg_idx = (ic_vect_sgroups + sg) * SUB_GROUP_SIZE;
        float diff_gamma = 0.0f;
        float diff_beta = 0.0f;
        float v_mean = block_load(v_mean, &mean[(sg_idx)]);

        // reduce
        for (int sp = 0; sp < sp_idx_bnd; ++sp) {
            const int tn_idx = sg_idx + sp * ic_size;
#if FUSE_BN_RELU
            char ws_vect = block_load(ws_vect, &ws[tn_idx]);
#endif
            float src_vect = block_load(src_vect, &src[tn_idx]);
            float dd_vect = block_load(dd_vect, &diff_dst[tn_idx]);
            const float v0 = src_vect - v_mean;
#if FUSE_BN_RELU
            dd_vect = select(0.0f, dd_vect, convert_int(ws_vect));
#endif
            diff_gamma = fma(v0, dd_vect, diff_gamma);
            diff_beta += dd_vect;
        } // sp loop

        // store results
        if (use_fused_atomics_reduction) {
            const int dst_off
                    = ic_block_offset + (ic_vect_sgroups + sg) * SUB_GROUP_SIZE;
            nhwc_reusable_bwd_fused_reduction(diff_scale, diff_shift, dst_off,
                    (float *)(&diff_gamma), (float *)(&diff_beta), local_sums,
                    1, calc_slm_size);
        } else {
            const int sg_off = (ic_vect_sgroups + sg) * SUB_GROUP_SIZE;
            block_write(&temp_reduce[sg_off], diff_gamma);
            block_write(&temp_reduce_shift[sg_off], diff_beta);
        }
    } // sg loop
}

// Calculate stats for BWD pass, private memory buffers used.
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) __kernel void
nhwc_reusable_calc_stat_buff(__global DATA_T *src, __global float *mean,
        __global DATA_T *diff_dst, __global char *ws,
        __global float *temp_reduce, __global float *temp_reduce_shift,
        volatile __global atomic_float *diff_scale,
        volatile __global atomic_float *diff_shift, off_t ic_size,
        off_t ic_block, off_t sp_size, off_t stat_sp_block,
        off_t reduce_stat_nblocks, int use_fused_atomics_reduction,
        __local float *local_sums, off_t calc_slm_size) {

    const int c = get_global_id(0);
    const int sp_block_idx = get_global_id(1);
    const int ic_block_offset = (c / SUB_GROUP_SIZE) * ic_block;
    const int offset = ic_block_offset + sp_block_idx * stat_sp_block * ic_size;

    mean += ic_block_offset;
    src += offset;
    diff_dst += offset;
    ws += offset;

    // scratchpad layout: (reduce_stat_nblocks + 1) rows x ic columns
    const int reduce_off = ic_block_offset + (sp_block_idx + 1) * ic_size;

    temp_reduce += reduce_off;
    temp_reduce_shift += reduce_off;

    const bool has_sp_block_tail = sp_size % stat_sp_block;
    const int sp_idx_bnd = has_sp_block_tail
            ? min(stat_sp_block, sp_size - sp_block_idx * stat_sp_block)
            : stat_sp_block;
    const int ic_block_sgroups
            = min(ic_size - ic_block_offset, ic_block) / SUB_GROUP_SIZE;
    const int ic_vect_sgroups = ic_block_sgroups / VECT_SIZE;
    const int ic_tail_sgroups = ic_block_sgroups % VECT_SIZE;

    VECT_FLOAT_T v_mean_v[MAX_IC_VECT_SGROUPS];
    VECT_FLOAT_T diff_gamma_v[MAX_IC_VECT_SGROUPS];
    VECT_FLOAT_T diff_beta_v[MAX_IC_VECT_SGROUPS];
    for (int sg = 0; sg < ic_vect_sgroups; ++sg) {
        const int sg_idx = sg * SUB_GROUP_SIZE * VECT_SIZE;
        v_mean_v[sg] = block_load(v_mean_v[sg], &mean[sg_idx]);
        diff_gamma_v[sg] = (VECT_FLOAT_T)0.0f;
        diff_beta_v[sg] = (VECT_FLOAT_T)0.0f;
    }
#if MAY_HAVE_IC_TAIL
    float v_mean_t[MAX_IC_TAIL_SGROUPS];
    float diff_gamma_t[MAX_IC_TAIL_SGROUPS];
    float diff_beta_t[MAX_IC_TAIL_SGROUPS];
    for (int sg = 0; sg < ic_tail_sgroups; ++sg) {
        const int sg_idx = (ic_vect_sgroups * VECT_SIZE + sg) * SUB_GROUP_SIZE;
        v_mean_t[sg] = block_load(v_mean_t[sg], &mean[sg_idx]);
        diff_gamma_t[sg] = 0.0f;
        diff_beta_t[sg] = 0.0f;
    }
#endif

    for (int sp = 0; sp < sp_idx_bnd; ++sp) {
        // vector part
        for (int sg = 0; sg < ic_vect_sgroups; ++sg) {
            const int sg_idx = sg * SUB_GROUP_SIZE * VECT_SIZE;
#if FUSE_BN_RELU
            VECT_N(char) ws_vect = block_load(ws_vect, &ws[sg_idx]);
#endif

            VECT_FLOAT_T src_vect = block_load(src_vect, &src[sg_idx]);
            VECT_FLOAT_T dd_vect = block_load(dd_vect, &diff_dst[sg_idx]);
            VECT_FLOAT_T v0_vect = src_vect - v_mean_v[sg];
#if FUSE_BN_RELU
            dd_vect = select(
                    (VECT_FLOAT_T)0.0f, dd_vect, VECT_N(convert_int)(ws_vect));
#endif
            diff_gamma_v[sg] = fma(v0_vect, dd_vect, diff_gamma_v[sg]);
            diff_beta_v[sg] += dd_vect;
        }

#if MAY_HAVE_IC_TAIL
        // tails
        for (int sg = 0; sg < ic_tail_sgroups; ++sg) {
            const int sg_idx
                    = (ic_vect_sgroups * VECT_SIZE + sg) * SUB_GROUP_SIZE;
#if FUSE_BN_RELU
            char ws_tail = block_load(ws_tail, &ws[sg_idx]);
#endif
            float src_tail = block_load(src_tail, &src[sg_idx]);
            float dd_tail = block_load(dd_tail, &diff_dst[sg_idx]);
            float v0 = src_tail - v_mean_t[sg];
#if FUSE_BN_RELU
            dd_tail = select(0.0f, dd_tail, convert_int(ws_tail));
#endif

            diff_gamma_t[sg] = fma(v0, dd_tail, diff_gamma_t[sg]);
            diff_beta_t[sg] += dd_tail;
        }
#endif
        src += ic_size;
        diff_dst += ic_size;
#if FUSE_BN_RELU
        ws += ic_size;
#endif
    } // sp loop

    // store results
    if (use_fused_atomics_reduction) {
        nhwc_reusable_bwd_fused_reduction_buff(diff_scale, diff_shift,
                ic_block_offset, diff_gamma_v, diff_beta_v,
#if MAY_HAVE_IC_TAIL
                diff_gamma_t, diff_beta_t,
#else
                NULL, NULL,
#endif
                local_sums, ic_block, calc_slm_size);

    } else {
        for (int sg = 0; sg < ic_vect_sgroups; ++sg) {
            const int sg_off = sg * SUB_GROUP_SIZE * VECT_SIZE;
            block_write(&temp_reduce[sg_off], &diff_gamma_v[sg]);
            block_write(&temp_reduce_shift[sg_off], &diff_beta_v[sg]);
        }
#if MAY_HAVE_IC_TAIL
        for (int sg = 0; sg < ic_tail_sgroups; ++sg) {
            const int sg_off
                    = (ic_vect_sgroups * VECT_SIZE + sg) * SUB_GROUP_SIZE;
            block_write(&temp_reduce[sg_off], diff_gamma_t[sg]);
            block_write(&temp_reduce_shift[sg_off], diff_beta_t[sg]);
        }
#endif
    }
}

// Main BWD pass kernel
// no private memory buffers used.
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) __kernel void
nhwc_reusable_norm_bwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *diff_dst,
        __global float *scaleshift, __global char *ws,
        __global DATA_T *diff_src, __global float *diff_scale,
        __global float *diff_shift, float eps, __global DATA_T *diff_src_add,
        off_t ic_size, off_t ic_block, off_t sp_size, off_t update_sp_block) {
    const int c = get_global_id(0);
    const int ic_block_offset = (c / SUB_GROUP_SIZE) * ic_block;

    variance += ic_block_offset;
    mean += ic_block_offset;
    diff_scale += ic_block_offset;
    diff_shift += ic_block_offset;
    scaleshift += ic_block_offset;

    const int sp_block_idx = get_global_id(1);
    const int offset
            = ic_block_offset + sp_block_idx * update_sp_block * ic_size;

    src += offset;
    diff_dst += offset;
    ws += offset;
    diff_src += offset;
#if FUSE_BN_ADD_RELU
    diff_src_add += offset;
#endif

    const bool has_sp_block_tail = sp_size % update_sp_block;
    const int sp_idx_bnd = has_sp_block_tail
            ? min(update_sp_block, sp_size - sp_block_idx * update_sp_block)
            : update_sp_block;
    const int ic_block_sgroups = ic_block / SUB_GROUP_SIZE;

    for (int sp = 0; sp < sp_idx_bnd; ++sp) {
        // vectorized part
        for (int sg = 0; sg < ic_block_sgroups / VECT_SIZE; ++sg) {
            const int sg_idx = sg * SUB_GROUP_SIZE * VECT_SIZE;

            VECT_FLOAT_T v_variance = block_load(v_variance, &variance[sg_idx]);
            const VECT_FLOAT_T sqrt_variance
                    = (VECT_FLOAT_T)1.0f / sqrt(v_variance + (VECT_FLOAT_T)eps);
            VECT_FLOAT_T gamma;
            if (USE_SCALE) {
                gamma = block_load(gamma, &scaleshift[sg_idx]);
            } else {
                gamma = (VECT_FLOAT_T)1.0f;
            }
            VECT_FLOAT_T src_vect = block_load(src_vect, &src[sg_idx]);
            VECT_FLOAT_T dd_vect = block_load(dd_vect, &diff_dst[sg_idx]);

#if FUSE_BN_RELU
            VECT_N(char) ws_vect = block_load(ws_vect, &ws[sg_idx]);
            dd_vect = select(
                    (VECT_FLOAT_T)0.0f, dd_vect, VECT_N(convert_int)(ws_vect));
#if FUSE_BN_ADD_RELU
            block_write(&diff_src_add[sg_idx], &dd_vect);
#endif
#endif
#if CALCULATE_STATS == 1
            VECT_FLOAT_T v_mean = block_load(v_mean, &mean[sg_idx]);
            VECT_FLOAT_T diff_gamma
                    = block_load(diff_gamma, &diff_scale[sg_idx]);
            VECT_FLOAT_T diff_beta = block_load(diff_beta, &diff_shift[sg_idx]);
            dd_vect -= (diff_beta
                               + (src_vect - v_mean) * diff_gamma
                                       * sqrt_variance)
                    / sp_size;
#endif
            dd_vect *= gamma * sqrt_variance;
            block_write(&diff_src[sg_idx], &dd_vect);

        } // sg loop

        const int ic_tail_sgroups = (ic_block / SUB_GROUP_SIZE) % VECT_SIZE;
        const int ic_vect_sgroups = ic_block_sgroups - ic_tail_sgroups;

        // tails
        for (int sg = 0; sg < ic_tail_sgroups; ++sg) {
            const int sg_idx = (ic_vect_sgroups + sg) * SUB_GROUP_SIZE;

            float v_variance = block_load(v_variance, &variance[sg_idx]);
            const float sqrt_variance
                    = (float)1.0f / sqrt(v_variance + (float)eps);
            float gamma;
            if (USE_SCALE) {
                gamma = block_load(gamma, &scaleshift[sg_idx]);
            } else {
                gamma = 1.0f;
            }
            float src_vect = block_load(src_vect, &src[sg_idx]);
            float dd_vect = block_load(dd_vect, &diff_dst[sg_idx]);

#if FUSE_BN_RELU
            char ws_vect = block_load(ws_vect, &ws[sg_idx]);
            dd_vect = select(0.0f, dd_vect, convert_int(ws_vect));
#if FUSE_BN_ADD_RELU
            block_write(&diff_src_add[sg_idx], dd_vect);
#endif
#endif

#if CALCULATE_STATS == 1
            float v_mean = block_load(v_mean, &mean[sg_idx]);
            float diff_gamma = block_load(diff_gamma, &diff_scale[sg_idx]);
            float diff_beta = block_load(diff_beta, &diff_shift[sg_idx]);
            dd_vect -= (diff_beta
                               + (src_vect - v_mean) * diff_gamma
                                       * sqrt_variance)
                    / sp_size;
#endif
            dd_vect *= gamma * sqrt_variance;
            block_write(&diff_src[sg_idx], dd_vect);
        }

        src += ic_size;
        diff_dst += ic_size;
        diff_src += ic_size;
#if FUSE_BN_RELU
#if FUSE_BN_ADD_RELU
        diff_src_add += ic_size;
#endif
        ws += ic_size;
#endif
    } // sp loop
}

// Main BWD pass kernel, private memory buffers used.
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) __kernel void
nhwc_reusable_norm_bwd_buff(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *diff_dst,
        __global float *scaleshift, __global char *ws,
        __global DATA_T *diff_src, __global float *diff_scale,
        __global float *diff_shift, float eps, __global DATA_T *diff_src_add,
        off_t ic_size, off_t ic_block, off_t sp_size, off_t update_sp_block) {

    const int c = get_global_id(0);
    const int ic_block_offset = (c / SUB_GROUP_SIZE) * ic_block;

    variance += ic_block_offset;
    mean += ic_block_offset;
    diff_scale += ic_block_offset;
    diff_shift += ic_block_offset;
    scaleshift += ic_block_offset;

    const int sp_block_idx = get_global_id(1);
    const int offset
            = ic_block_offset + sp_block_idx * update_sp_block * ic_size;

    src += offset;
    diff_dst += offset;
    ws += offset;
    diff_src += offset;
#if FUSE_BN_ADD_RELU
    diff_src_add += offset;
#endif

    const bool has_sp_block_tail = sp_size % update_sp_block;
    const int sp_idx_bnd = has_sp_block_tail
            ? min(update_sp_block, sp_size - sp_block_idx * update_sp_block)
            : update_sp_block;
    const int ic_block_sgroups
            = min(ic_size - ic_block_offset, ic_block) / SUB_GROUP_SIZE;
    const int ic_vect_sgroups = ic_block_sgroups / VECT_SIZE;
    const int ic_tail_sgroups = ic_block_sgroups % VECT_SIZE;

    VECT_FLOAT_T v_variance_v[MAX_IC_VECT_SGROUPS],
            v_mean_v[MAX_IC_VECT_SGROUPS], diff_gamma_v[MAX_IC_VECT_SGROUPS],
            diff_beta_v[MAX_IC_VECT_SGROUPS],
            sqrt_variance_v[MAX_IC_VECT_SGROUPS], gamma_v[MAX_IC_VECT_SGROUPS];
#if MAY_HAVE_IC_TAIL
    float v_variance_t[MAX_IC_TAIL_SGROUPS], v_mean_t[MAX_IC_TAIL_SGROUPS],
            diff_gamma_t[MAX_IC_TAIL_SGROUPS], diff_beta_t[MAX_IC_TAIL_SGROUPS],
            sqrt_variance_t[MAX_IC_TAIL_SGROUPS], gamma_t[MAX_IC_TAIL_SGROUPS];
#endif

    for (int sg = 0; sg < ic_vect_sgroups; ++sg) {
        const int sg_idx = sg * SUB_GROUP_SIZE * VECT_SIZE;

        v_variance_v[sg] = block_load(v_variance_v[sg], &variance[sg_idx]);
#if CALCULATE_STATS == 1
        v_mean_v[sg] = block_load(v_mean_v[sg], &mean[sg_idx]);
        diff_gamma_v[sg] = block_load(diff_gamma_v[sg], &diff_scale[sg_idx]);
        diff_beta_v[sg] = block_load(diff_beta_v[sg], &diff_shift[sg_idx]);
#endif // #if CALCULATE_DIFF_STATS == 1
        if (USE_SCALE) {
            gamma_v[sg] = block_load(gamma_v[sg], &scaleshift[sg_idx]);
        } else {
            gamma_v[sg] = (VECT_FLOAT_T)1.0f;
        }
        sqrt_variance_v[sg] = (VECT_FLOAT_T)1.0f
                / sqrt(v_variance_v[sg] + (VECT_FLOAT_T)eps);
    }

#if MAY_HAVE_IC_TAIL
    for (int sg = 0; sg < ic_tail_sgroups; ++sg) {
        const int sg_idx = (ic_vect_sgroups * VECT_SIZE + sg) * SUB_GROUP_SIZE;
        v_variance_t[sg] = block_load(v_variance_t[sg], &variance[sg_idx]);
#if CALCULATE_STATS == 1
        v_mean_t[sg] = block_load(v_mean_t[sg], &mean[sg_idx]);
        diff_gamma_t[sg] = block_load(diff_gamma_t[sg], &diff_scale[sg_idx]);
        diff_beta_t[sg] = block_load(diff_beta_t[sg], &diff_shift[sg_idx]);
#endif // #if CALCULATE_DIFF_STATS == 1
        if (USE_SCALE) {
            gamma_t[sg] = block_load(gamma_t[sg], &scaleshift[sg_idx]);
        } else {
            gamma_t[sg] = 1.0f;
        }
        sqrt_variance_t[sg] = 1.0f / sqrt(v_variance_t[sg] + eps);
    }
#endif
    for (int sp = 0; sp < sp_idx_bnd; ++sp) {
        // vector part
        for (int sg = 0; sg < ic_vect_sgroups; ++sg) {
            const int sg_idx = sg * SUB_GROUP_SIZE * VECT_SIZE;

            VECT_FLOAT_T src_vect = block_load(src_vect, &src[sg_idx]);
            VECT_FLOAT_T dd_vect = block_load(dd_vect, &diff_dst[sg_idx]);
#if FUSE_BN_RELU
            VECT_N(char) ws_vect = block_load(ws_vect, &ws[sg_idx]);
            dd_vect = select(
                    (VECT_FLOAT_T)0.0f, dd_vect, VECT_N(convert_int)(ws_vect));
#if FUSE_BN_ADD_RELU
            block_write(&diff_src_add[sg_idx], &dd_vect);
#endif
#endif
#if CALCULATE_STATS == 1
            dd_vect -= (diff_beta_v[sg]
                               + (src_vect - v_mean_v[sg]) * diff_gamma_v[sg]
                                       * sqrt_variance_v[sg])
                    / sp_size;
#endif
            dd_vect *= gamma_v[sg] * sqrt_variance_v[sg];
            block_write(&diff_src[sg_idx], &dd_vect);
        } // vector sg loop

#if MAY_HAVE_IC_TAIL
        // tails
        for (int sg = 0; sg < ic_tail_sgroups; ++sg) {
            const int sg_idx
                    = (ic_vect_sgroups * VECT_SIZE + sg) * SUB_GROUP_SIZE;
            float src_tail = block_load(src_tail, &src[sg_idx]);
            float dd_tail = block_load(dd_tail, &diff_dst[sg_idx]);
#if FUSE_BN_RELU
            char ws_tail = block_load(ws_tail, &ws[sg_idx]);
            dd_tail = select(0.0f, dd_tail, convert_int(ws_tail));
#if FUSE_BN_ADD_RELU
            block_write(&diff_src_add[sg_idx], dd_tail);
#endif
#endif
#if CALCULATE_STATS == 1
            dd_tail -= (diff_beta_t[sg]
                               + (src_tail - v_mean_t[sg]) * diff_gamma_t[sg]
                                       * sqrt_variance_t[sg])
                    / sp_size;
#endif
            dd_tail *= gamma_t[sg] * sqrt_variance_t[sg];
            block_write(&diff_src[sg_idx], dd_tail);
        } // tail sg loop
#endif
        src += ic_size;
        diff_dst += ic_size;
        diff_src += ic_size;
#if FUSE_BN_RELU
#if FUSE_BN_ADD_RELU
        diff_src_add += ic_size;
#endif
        ws += ic_size;
#endif
    } // sp loop
}

// Aux kernel performs initial zero-padding or finalization of stat vectors
// if atomic-based reduction is used
__kernel void nhwc_reusable_reduce_aux(__global float *ptr1,
        __global float *ptr2, float eps, off_t sp_size, int use_stats_one_pass,
        int init_stage, int is_fwd) {
    const int c = get_global_id(0);
    if (init_stage) {
        // initialization
        ptr1[c] = 0.0f;
        ptr2[c] = 0.0f;
    } else {
        // finalization
        if (is_fwd) {
            if (use_stats_one_pass) {
                ptr1[c] /= sp_size;
                float tmp_var
                        = max(0.0f, (ptr2[c] / sp_size) - (ptr1[c] * ptr1[c]));
                ptr2[c] = tmp_var;
            } else {
                ptr1[c] /= sp_size;
            }
        } else {
            ptr1[c] *= 1.0f / sqrt(ptr2[c] + eps);
        }
    }
}

// Reduction thru scratchpad, FWD pass, regular algorithm
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) __kernel void
nhwc_reusable_reduce_fwd_reg(__global float *reduce_scratchpad,
        off_t scratchpad_off, __global float *dst, off_t ic_size,
        off_t reduce_ic_sub_groups, off_t reduce_stat_nblocks, off_t sp_size,
        __local float *local_sum) {
    const int ic_sub_group = get_global_id(0) / SUB_GROUP_SIZE;
    const int group_c = get_global_id(1);
    const int simd_id = get_sub_group_local_id();
    const int c = group_c * SUB_GROUP_SIZE + simd_id;
    float sum = 0.0f;

    const int reduce_chunk = reduce_stat_nblocks / reduce_ic_sub_groups;
    const int reduce_scratchpad_off
            = scratchpad_off + c + ic_sub_group * reduce_chunk * ic_size;
    reduce_scratchpad += reduce_scratchpad_off;

    unroll_16_for(int i = 0; i < reduce_chunk; i++) {
        sum += reduce_scratchpad[i * ic_size];
    }

    if (ic_sub_group > 0) {
        local_sum[ic_sub_group * SUB_GROUP_SIZE + simd_id] = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ic_sub_group == 0) {
        unroll_16_for(int i = 1; i < reduce_ic_sub_groups; i++) {
            sum += local_sum[i * SUB_GROUP_SIZE + simd_id];
        }
        dst[c] = sum / sp_size;
    }
}

// Reduction thru scratchpad, FWD pass, 1pass algorithm
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) __kernel void
nhwc_reusable_reduce_fwd_1pass(__global float *reduce_temp,
        __global float *mean, __global float *variance, off_t ic_size,
        off_t reduce_ic_sub_groups, off_t reduce_stat_nblocks, off_t sp_size,
        __local SUM_DATA_T *local_sum, __local SUM_DATA_T *local_sum_sq) {
    const int ic_sub_group = get_global_id(0) / SUB_GROUP_SIZE;
    const int group_c = get_global_id(1);
    const int simd_id = get_sub_group_local_id();
    const int c = group_c * SUB_GROUP_SIZE + simd_id;
    SUM_DATA_T sum;
    SUM_DATA_T sum_sq;
    sum.s0 = 0;
    sum.s1 = 0;
    sum_sq.s0 = 0;
    sum_sq.s1 = 0;

    const int offs_sq = reduce_stat_nblocks * ic_size;
    const int reduce_chunk = reduce_stat_nblocks / reduce_ic_sub_groups;
    const int offs = c + ic_sub_group * reduce_chunk * ic_size;

    unroll_16_for(int i = 0; i < reduce_chunk; i++) {
        float tmp = reduce_temp[offs + i * ic_size];
        sum = summation(tmp, sum);
    }
    unroll_16_for(int i = 0; i < reduce_chunk; i++) {
        float tmp = reduce_temp[offs_sq + offs + i * ic_size];
        sum_sq = summation(tmp, sum_sq);
    }
    if (ic_sub_group > 0) {
        local_sum[ic_sub_group * SUB_GROUP_SIZE + simd_id] = sum;
        local_sum_sq[ic_sub_group * SUB_GROUP_SIZE + simd_id] = sum_sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ic_sub_group == 0) {
        unroll_16_for(int i = 1; i < reduce_ic_sub_groups; i++) {
            SUM_DATA_T tmp = local_sum[i * SUB_GROUP_SIZE + simd_id];
            SUM_DATA_T tmp_sq = local_sum_sq[i * SUB_GROUP_SIZE + simd_id];
            sum = summation(tmp.s1, sum);
            sum_sq = summation(tmp_sq.s1, sum_sq);
            sum = summation(tmp.s0, sum);
            sum_sq = summation(tmp_sq.s0, sum_sq);
        }
        float tmp_mean = sum.s0 / sp_size;
        mean[c] = tmp_mean;
        float tmp_var
                = max(0.0f, (sum_sq.s0 / sp_size) - (tmp_mean * tmp_mean));
        variance[c] = tmp_var;
    }
}

// Reduction thru scratchpad, BWD pass
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE))) __kernel void
nhwc_reusable_reduce_stat(__global float *temp_reduce,
        __global float *temp_reduce_shift, __global float *diff_scale,
        __global float *diff_shift, __global float *variance, float eps,
        off_t ic_size, off_t reduce_ic_sub_groups, off_t reduce_stat_nblocks,
        __local float *local_gamma, __local float *local_beta) {
    const int ic_sub_group = get_global_id(0) / SUB_GROUP_SIZE;
    const int group_c = get_global_id(1);
    const int simd_id = get_sub_group_local_id();
    const int c = group_c * SUB_GROUP_SIZE + simd_id;

    float diff_gamma = 0.0f;
    float diff_beta = 0.0f;

    const int reduce_chunk = reduce_stat_nblocks / reduce_ic_sub_groups;
    const int scratchpad_off
            = ic_size + c + ic_sub_group * reduce_chunk * ic_size;

    temp_reduce += scratchpad_off;
    temp_reduce_shift += scratchpad_off;

    unroll_16_for(int i = 0; i < reduce_chunk; i++) {
        diff_gamma += temp_reduce[i * ic_size];
        diff_beta += temp_reduce_shift[i * ic_size];
    }
    if (ic_sub_group > 0) {
        local_gamma[ic_sub_group * SUB_GROUP_SIZE + simd_id] = diff_gamma;
        local_beta[ic_sub_group * SUB_GROUP_SIZE + simd_id] = diff_beta;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ic_sub_group == 0) {
        unroll_16_for(int i = 1; i < reduce_ic_sub_groups; i++) {
            diff_gamma += local_gamma[i * SUB_GROUP_SIZE + simd_id];
            diff_beta += local_beta[i * SUB_GROUP_SIZE + simd_id];
        }
        float sqrt_variance = 1.0f / sqrt(variance[c] + eps);

        diff_scale[c] = diff_gamma * sqrt_variance;
        diff_shift[c] = diff_beta;
    }
}
