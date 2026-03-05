/*******************************************************************************
* Copyright 2021 Intel Corporation
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
#include "gpu/intel/include/types_interop.h"
#include "gpu/intel/reduction/common.h"

#if defined(IS_MAX)
#define ATOMIC_ACCUMULATE(atomic_p, data) atomic_max_global(atomic_p, data)
#elif defined(IS_MIN)
#define ATOMIC_ACCUMULATE(atomic_p, data) atomic_min_global(atomic_p, data)
#elif defined(IS_MEAN) || defined(IS_SUM)
#define ATOMIC_ACCUMULATE(atomic_p, data) atomic_add_global(atomic_p, data)
#endif

// Define accumulation functions
#define DEF_atomic_accumulate(dt) \
    dt atomic_accumulate(int alg, __global ATOMIC(dt) * atomic_p, dt data) { \
        switch (alg) { \
            case (REDUCTION_MAX): return atomic_max_global(atomic_p, data); \
            case (REDUCTION_MIN): return atomic_min_global(atomic_p, data); \
            case (REDUCTION_MEAN): \
            case (REDUCTION_SUM): return atomic_add_global(atomic_p, data); \
        } \
        printf("Atomic accumulate on unexpected algorithm\n"); \
        return 0; \
    }

#if ATOMIC_REDUCTION_SIZE > 1
#define MAYBE_ATOMIC(x) ATOMIC(x)
DEF_atomic_accumulate(float);
#else
#define MAYBE_ATOMIC(x) x
#endif

#if VECT_DT_N == 1
#define GET_ELEM(x, idx) x
#else
#define GET_ELEM(x, idx) x[idx]
#endif

#define REDUCTION_WI_COUNT (ATOMIC_REDUCTION_SIZE * LOCAL_SIZE)

KERNEL_ATTR
__kernel void atomic_reduce(__global SRC_DATA_T *src,
        __global MAYBE_ATOMIC(DST_DATA_T) * dst, int inner_size, off_t div,
        float power, float eps, off_t num_reductions,
        dispatch_gws_rt_params_t gws_params) {
    ASSUME(inner_size > 0);
    ASSUME(num_reductions > 0);
    const int local_idx = get_sub_group_id();
    const int sglid = get_sub_group_local_id();
    const int subgroup_size = get_max_sub_group_size();

    off_t atomic_idx = GWS_GET_OFF(ATOMIC, gws_params);

    // src inner dim is split into the subgroup, vectorization, and inner_group blocks
    // subgroup and inner_group blocks are dispatched, and the vectorization block
    // is handled by a single work item. Since the vectorization size is compiled in,
    // update offsets using it: the inner_group stride is subgroup_size * vec_size
    off_t SRC_OFF = GWS_GET_OFF(SRC, gws_params) - sglid;
    off_t src_ig_idx = SRC_OFF / subgroup_size;
    SRC_OFF += src_ig_idx * (VECT_DT_N - 1) * subgroup_size;
    src += SRC_OFF;

    // Do the same for dst
    off_t DST_OFF = GWS_GET_OFF(DST, gws_params);
    off_t dst_ig_idx = DST_OFF / subgroup_size;
    DST_OFF += dst_ig_idx * (VECT_DT_N - 1) * subgroup_size;
    dst += DST_OFF;

    const int beg = local_idx + atomic_idx * LOCAL_SIZE;
    ASSUME(beg < REDUCTION_WI_COUNT);
    const int tail_count = num_reductions % REDUCTION_WI_COUNT;
    VECT_DEF_ACC_DATA_T acc;
    init_acc(REDUCTION_ALG, &acc);
    // XXX: To match static kernel performance, both regular and tail cases
    // need optimized unrolling. We first detect which case we're in, and dispatch
    // to the appropriately-unrolled loop.
    const int iters = div_up(num_reductions - beg, REDUCTION_WI_COUNT);
    if (beg < tail_count) {
        unroll_for_by(FULL_UNROLL_FACTOR)(off_t i = 0; i < iters; i++) {
            const off_t src_off = (beg + i * REDUCTION_WI_COUNT) * inner_size;
            DEF_ACC_DATA_T src_vals[VECT_DT_N];
            block_load(src_vals, src + src_off, VECT_DT_N);
            unroll_for(uint i = 0; i < VECT_DT_N; i++) {
                GET_ELEM(acc, i) = reduce(REDUCTION_ALG, GET_ELEM(acc, i),
                        src_vals[i], power);
            }
        }
    } else {
        unroll_for_by(TAIL_UNROLL_FACTOR)(off_t i = 0; i < iters; i++) {
            const off_t src_off = (beg + i * REDUCTION_WI_COUNT) * inner_size;
            DEF_ACC_DATA_T src_vals[VECT_DT_N];
            block_load(src_vals, src + src_off, VECT_DT_N);
            unroll_for(uint i = 0; i < VECT_DT_N; i++) {
                GET_ELEM(acc, i) = reduce(REDUCTION_ALG, GET_ELEM(acc, i),
                        src_vals[i], power);
            }
        }
    }

    // Store results to SLM
    __local DEF_ACC_DATA_T
            local_acc_buf[LOCAL_SIZE][GWS_SGS_DEFAULT * VECT_DT_N];
    unroll_for(int i = 0; i < VECT_DT_N; i++) {
        local_acc_buf[local_idx][sglid + i * subgroup_size] = GET_ELEM(acc, i);
    }

    // Wait for all subgroups to finish
    barrier(CLK_LOCAL_MEM_FENCE);

    // In the first subgroup of each work group:
    if (local_idx == 0) {
        // Perform the SLM reduction
        VECT_DEF_ACC_DATA_T local_acc;
        init_acc(SECONDARY_REDUCTION_ALG, &local_acc);

        unroll_for(int slm_off = 0; slm_off < LOCAL_SIZE; slm_off++) {
            unroll_for(int v = 0; v < VECT_DT_N; v++) {
                DEF_ACC_DATA_T slm_data
                        = local_acc_buf[slm_off][sglid + v * subgroup_size];
                GET_ELEM(local_acc, v) = reduce(SECONDARY_REDUCTION_ALG,
                        GET_ELEM(local_acc, v), slm_data, power);
            }
        }

        // Finalize data, then (atomically) accumulate into to dst
        unroll_for(int v = 0; v < VECT_DT_N; v++) {
            float f = finalize(REDUCTION_ALG,
                    convert_float(GET_ELEM(local_acc, v)), div, power, eps);
#if ATOMIC_REDUCTION_SIZE > 1
            DST_DATA_T dst_data = TO_DST(f);
            DST_DATA_T old_val = atomic_accumulate(
                    REDUCTION_ALG, &dst[v * subgroup_size], dst_data);
#else
            write(dst + v * subgroup_size, f);
#endif
        }
    }
}
