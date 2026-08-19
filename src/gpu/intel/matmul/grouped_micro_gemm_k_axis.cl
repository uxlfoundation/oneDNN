/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "gpu/intel/include/conversion.h"
#include "gpu/intel/include/tile_ops.h"
#include "gpu/intel/include/types_interop.h"
#include "gpu/intel/include/utils.h"

#include "gemm_grouped.h"

#ifdef DST_DT_BF16
#define DST_TILE_DATA_T ushort
#define CONVERT_TILE_DATA_T(v) (into_bf16(convert_float(v)).data)
#else
#define DST_TILE_DATA_T DST_DATA_T
#define CONVERT_TILE_DATA_T CONVERT_DATA_T
#endif

#ifndef DST_DT_F32
DECLARE_2D_TILE(c_tile_type_dst, DST_TILE_DATA_T, SUBGROUP_SIZE,
        ugemm_grouped_c_type_block0, ugemm_grouped_c_type_block1,
        ugemm_grouped_c_type_nblock0, ugemm_grouped_c_type_nblock1)
#endif

#if defined(A_DT_BF16)
#define AS_A_TILE_PTR(p) ((const global ushort *)(p))
#else
#define AS_A_TILE_PTR(p) (p)
#endif

#if defined(B_DT_BF16)
#define AS_B_TILE_PTR(p) ((const global ushort *)(p))
#else
#define AS_B_TILE_PTR(p) (p)
#endif

void store_results(ugemm_grouped_c_type *tile, global DST_DATA_T *ptr, int m,
        int n, int ldc, int sg_i0, int sg_j0) {
#if DST_DT_F32
    tile_store(*tile, ptr, m, n, ldc, sg_i0, sg_j0);
#else
    c_tile_type_dst tile_dst;
    tile_convert((*tile), tile_dst, CONVERT_TILE_DATA_T);
    tile_store(
            tile_dst, (global DST_TILE_DATA_T *)ptr, m, n, ldc, sg_i0, sg_j0);
#endif
}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__attribute__((reqd_work_group_size(ugemm_grouped_sg_per_wg_m * SUBGROUP_SIZE,
        ugemm_grouped_sg_per_wg_n, ugemm_grouped_sg_per_wg_k))) kernel void
grouped_micro_gemm_k_axis(const global A_DATA_T *a, long lda,
        const global B_DATA_T *b, long ldb, global DST_DATA_T *dst, long ldc,
        const global int *offsets, const long m, const long n) {
#if WITH_SLM
    local char slm[ugemm_grouped_slm_size];
#else
    local char *slm = NULL;
#endif

    off_t sg_i = sub_group_broadcast(get_local_id(0) / SUBGROUP_SIZE, 0);
    off_t sg_j = sub_group_broadcast(get_local_id(1), 0);

    off_t wg_i0 = get_group_id(0) * ugemm_grouped_wg_tile_m;
    off_t wg_j0 = get_group_id(1) * ugemm_grouped_wg_tile_n;
    off_t sg_i0 = wg_i0 + sg_i * ugemm_grouped_sg_tile_m;
    off_t sg_j0 = wg_j0 + sg_j * ugemm_grouped_sg_tile_n;

    off_t batch = sub_group_broadcast(get_group_id(2), 0);

    off_t k_offset, kg;
    if (get_num_groups(2) == 1) {
        k_offset = 0;
        kg = *offsets;
    } else {
        int2 k_range
                = *(global int2 *)(offsets + (batch > 0 ? batch - 1 : batch));
        k_offset = batch > 0 ? k_range.x : 0;
        kg = batch > 0 ? (k_range.y - k_range.x) : k_range.x;
    }
    dst += batch * m * n;

    ugemm_grouped_c_type c_tile;
    if (kg == 0) {
        tile_fill(c_tile, 0.0f);
    } else {
        a += k_offset * lda / A_ELEMS_PER_BYTE;
        b += k_offset * ldb / B_ELEMS_PER_BYTE;

        c_tile = ugemm_grouped(AS_A_TILE_PTR(a), lda, AS_B_TILE_PTR(b), ldb, m,
                n, kg, wg_i0, wg_j0, 0, sg_i, sg_j, slm);
    }

    store_results(&c_tile, dst, m, n, ldc, sg_i0, sg_j0);
}
