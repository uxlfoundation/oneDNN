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

#include "gpu/intel/include/eltwise.h"
#include "gpu/intel/include/io.h"
#include "gpu/intel/include/post_ops.h"

__attribute__((intel_reqd_sub_group_size(SIMD))) __kernel void xe_eltwise_fwd(
        __global DATA_T *src, __global DATA_T *dst, dim_t nelems, float alpha,
        float beta) {
    const dim_t grsize = get_local_size(0);
    const dim_t grid = get_group_id(0);
    const dim_t sgid = get_sub_group_id();
    const dim_t lid = get_sub_group_local_id();

    const dim_t gid = get_global_id(0);

    dim_t offset
            = (grid * grsize + sgid * get_max_sub_group_size()) * VECT_DT_N;

    VECT_N(POST_OP_DATA_T) val;
    const int nel_per_read = SIMD * VECT_DT_N;

    // READ
    if (!NELEMS_OVERFLOW || offset + nel_per_read < nelems) {
        val = block_load(val, src + offset);

    } else {
        // read data in the same access pattern block_reads would
        dim_t pos = offset + lid;
#if VECT_DT_N > 1
        for (int i = 0; i < VECT_DT_N && pos < nelems; ++i) {
            val[i] = load(val[i], src, pos);
            pos += SIMD;
        }
#else
        if (pos < nelems) val = load(val, src, pos);
#endif
    }

    // COMPUTE
#if VECT_DT_N > 1
    for (int i = 0; i < VECT_DT_N; ++i) {
        val[i] = fwd_eltwise(val[i], alpha, beta, 1.0f);
    }
#else
    val = fwd_eltwise(val, alpha, beta, 1.0f);
#endif

    // WRITE
    if (!NELEMS_OVERFLOW || offset + nel_per_read < nelems) {
        block_write(dst + offset, val);

    } else {
        dim_t pos = offset + lid;
#if VECT_DT_N > 1
        for (int i = 0; i < VECT_DT_N && pos < nelems; ++i) {
            write(dst + pos, val[i]);
            pos += SIMD;
        }
#else
        if (pos < nelems) write(dst + pos, val);
#endif
    }
}

__attribute__((intel_reqd_sub_group_size(SIMD))) __kernel void xe_eltwise_bwd(
        __global DATA_T *src, __global DATA_T *diff_src,
        __global DATA_T *diff_dst, dim_t nelems, float alpha, float beta) {
    const dim_t grsize = get_local_size(0);
    const dim_t grid = get_group_id(0);
    const dim_t sgid = get_sub_group_id();
    const dim_t lid = get_sub_group_local_id();

    dim_t offset = (grid * grsize + sgid * SIMD) * VECT_DT_N;
    //TODO: It should be implemented two distinct offsets
    //The one for src and the second for diff_src

    VECT_N(POST_OP_DATA_T) val_dd;
    VECT_N(POST_OP_DATA_T) val_src;
    const int nel_per_read = SIMD * VECT_DT_N;

    // READ
    if (!NELEMS_OVERFLOW || offset + nel_per_read < nelems) {
        val_src = block_load(val_src, src + offset);
        val_dd = block_load(val_dd, diff_dst + offset);

    } else {
        // read data in the same access pattern block_reads would
        dim_t pos = offset + lid;
#if VECT_DT_N > 1
        for (int i = 0; i < VECT_DT_N && pos < nelems; ++i) {
            val_dd[i] = load(val_dd[i], diff_dst, pos);
            val_src[i] = load(val_src[i], src, pos);
            pos += SIMD;
        }
#else
        if (pos < nelems) {
            val_dd = load(val_dd, diff_dst, pos);
            val_src = load(val_src, src, pos);
        }
#endif
    }

    // COMPUTE
#if VECT_DT_N > 1
    for (int i = 0; i < VECT_DT_N; ++i) {
        val_dd[i] = bwd_eltwise(val_dd[i], val_src[i], alpha, beta);
    }
#else
    val_dd = bwd_eltwise(val_dd, val_src, alpha, beta);
#endif

    // WRITE
    if (!NELEMS_OVERFLOW || offset + nel_per_read < nelems) {
        block_write(diff_src + offset, val_dd);

    } else {
        // write data in the same access pattern block_writes would
        dim_t pos = offset + lid;
#if VECT_DT_N > 1
        for (int i = 0; i < VECT_DT_N && pos < nelems; ++i) {
            write(diff_src + pos, val_dd[i]);
            pos += SIMD;
        }
#else
        if (pos < nelems) write(diff_src + pos, val_dd);
#endif
    }
}
