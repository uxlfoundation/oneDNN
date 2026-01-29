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
#include "gpu/intel/sdpa/utils.h"

#include "gemm_grouped.h"

#if WITH_BIAS
DECLARE_2D_TILE(bias_tile_type, float, SUBGROUP_SIZE,
        ugemm_grouped_c_type_block0, ugemm_grouped_c_type_block1,
        ugemm_grouped_c_type_nblock0, ugemm_grouped_c_type_nblock1)

#ifndef BIA_DT_F32
DECLARE_2D_TILE(bias_in_tile_type, BIA_DATA_T, SUBGROUP_SIZE,
        ugemm_grouped_c_type_block0, ugemm_grouped_c_type_block1,
        ugemm_grouped_c_type_nblock0, ugemm_grouped_c_type_nblock1)
DECLARE_2D_TILE_COPY_REBLOCK(bias_in_tile_type, SUBGROUP_SIZE,
        ugemm_grouped_c_type_block0, ugemm_grouped_c_type_block1,
        ugemm_grouped_c_type_nblock0, ugemm_grouped_c_type_nblock1,
        bias_tile_type, SUBGROUP_SIZE, ugemm_grouped_c_type_block0,
        ugemm_grouped_c_type_block1, ugemm_grouped_c_type_nblock0,
        ugemm_grouped_c_type_nblock1, CONVERT_FLOAT_T)
#endif
#endif

#ifndef DST_DT_F32
DECLARE_2D_TILE(c_tile_type_dst, DST_DATA_T, SUBGROUP_SIZE,
        ugemm_grouped_c_type_block0, ugemm_grouped_c_type_block1,
        ugemm_grouped_c_type_nblock0, ugemm_grouped_c_type_nblock1)

DECLARE_2D_TILE_COPY_REBLOCK(ugemm_grouped_c_type, SUBGROUP_SIZE,
        ugemm_grouped_c_type_block0, ugemm_grouped_c_type_block1,
        ugemm_grouped_c_type_nblock0, ugemm_grouped_c_type_nblock1,
        c_tile_type_dst, SUBGROUP_SIZE, ugemm_grouped_c_type_block0,
        ugemm_grouped_c_type_block1, ugemm_grouped_c_type_nblock0,
        ugemm_grouped_c_type_nblock1, CONVERT_DATA_T)

DECLARE_2D_TILE_COPY_REBLOCK(c_tile_type_dst, SUBGROUP_SIZE,
        ugemm_grouped_c_type_block0, ugemm_grouped_c_type_block1,
        ugemm_grouped_c_type_nblock0, ugemm_grouped_c_type_nblock1,
        ugemm_grouped_c_type, SUBGROUP_SIZE, ugemm_grouped_c_type_block0,
        ugemm_grouped_c_type_block1, ugemm_grouped_c_type_nblock0,
        ugemm_grouped_c_type_nblock1, CONVERT_DATA_T)

#endif

#if WITH_SRC_ATTR_SCALES
#define SRC_ATTR_SCALE_ARGS , src_attr_scales
#else
#define SRC_ATTR_SCALE_ARGS
#endif
#if WITH_SRC_ATTR_ZP
#define SRC_ATTR_ZP_ARGS , src_attr_zp
#else
#define SRC_ATTR_ZP_ARGS
#endif
#if WITH_SRC_ATTR_SCALES || WITH_SRC_ATTR_ZP
#define SRC_ATTR_LD_ARGS , ldsrcq
#else
#define SRC_ATTR_LD_ARGS
#endif
#if WITH_WEI_ATTR_SCALES
#define WEI_ATTR_SCALE_ARGS , wei_attr_scales
#else
#define WEI_ATTR_SCALE_ARGS
#endif
#if WITH_WEI_ATTR_ZP
#define WEI_ATTR_ZP_ARGS , wei_attr_zp
#else
#define WEI_ATTR_ZP_ARGS
#endif
#if WITH_WEI_ATTR_SCALES || WITH_WEI_ATTR_ZP
#define WEI_ATTR_LD_ARGS , ldweiq
#else
#define WEI_ATTR_LD_ARGS
#endif

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) kernel void
grouped_micro_gemm(const global SRC_DATA_T *src, int ldsrc,
        const global WEI_DATA_T *wei, long4 wei_strides, global DST_DATA_T *dst,
        int lddst, const global int *src_offsets, const global int *dst_offsets,
        const global SRC_ATTR_SCALES_DATA_T *src_attr_scales,
        const global SRC_ATTR_ZP_DATA_T *src_attr_zp, const int ldsrcq,
        const global WEI_ATTR_SCALES_DATA_T *wei_attr_scales,
        const global WEI_ATTR_ZP_DATA_T *wei_attr_zp, const int ldweiq, int n,
        int k, const global BIA_DATA_T *bias) {

#if ugemm_grouped_slm_size > 0
    local char slm[ugemm_grouped_slm_size];
#else
    local char slm[1];
#endif

    unsigned long batch = get_group_id(2);
    int2 src_offset
            = *(global int2 *)(src_offsets + (batch > 0 ? batch - 1 : batch));

    int sg_i = sub_group_broadcast(get_local_id(0) / SUBGROUP_SIZE, 0);
    int sg_j = sub_group_broadcast(get_local_id(1), 0);

    unsigned long wg_i0 = get_group_id(0) * ugemm_grouped_wg_tile_m;
    unsigned long wg_j0 = get_group_id(1) * ugemm_grouped_wg_tile_n;
    unsigned long sg_i0 = wg_i0 + sg_i * ugemm_grouped_sg_tile_m;
    unsigned long sg_j0 = wg_j0 + sg_j * ugemm_grouped_sg_tile_n;

    int m = batch > 0 ? (src_offset.y - src_offset.x) : src_offset.x;
    if (wg_j0 >= m) return; /* early exit if outside batch */

    src_offset.x = batch > 0 ? src_offset.x : 0;

    src += src_offset.x * ldsrc / SRC_ELEMS_PER_BYTE;
    wei += batch * wei_strides[0] / WEI_ELEMS_PER_BYTE;
    dst += src_offset.x * lddst;

    int ldwei = wei_strides[2] == 1 ? wei_strides[1] : wei_strides[2];
#if WITH_SRC_ATTR_SCALES
    src_attr_scales += src_offset.x;
#endif
#if WITH_SRC_ATTR_ZP
    src_attr_zp += src_offset.x;
#endif
#if WITH_WEI_ATTR_SCALES
    wei_attr_scales += batch * n * (k / WEI_GROUP_SIZE);
#endif
#if WITH_WEI_ATTR_ZP
    wei_attr_zp += batch * n * (k / WEI_GROUP_SIZE);
#endif

    ugemm_grouped_c_type c_tile = ugemm_grouped(wei, ldwei, src, ldsrc, n, m, k,
            wg_i0, wg_j0, 0, sg_i, sg_j,
            slm WEI_ATTR_SCALE_ARGS WEI_ATTR_ZP_ARGS WEI_ATTR_LD_ARGS
                    SRC_ATTR_SCALE_ARGS SRC_ATTR_ZP_ARGS SRC_ATTR_LD_ARGS);
#if WITH_BIAS
#define binary_add(x, y) ((x) + (y))
    bias += batch * n;
    bias_tile_type bias_tile;
#if BIA_DT_F32
    tile_load(&bias_tile, bias, n, m, 0, sg_i0, sg_j0);
#else
    {
        bias_in_tile_type bias_in_tile;
        tile_load(&bias_in_tile, bias, n, m, 0, sg_i0, sg_j0);
        tile_copy_reblock(bias_in_tile, &bias_tile);
    }
#endif
    tile_binary(c_tile, bias_tile, binary_add);
#endif

#if DST_DT_F32
    tile_store(c_tile, dst, n, m, lddst, sg_i0, sg_j0);
    //tile_store_t_block2d(c_tile, dst, n, m, lddst, sg_j0, sg_i0);
#else
    c_tile_type_dst c_tile_dst;
    tile_copy_reblock(c_tile, &c_tile_dst);
    tile_store(c_tile_dst, dst, n, m, lddst, sg_i0, sg_j0);
    //tile_store_block2d(c_tile_dst, dst, n, m, lddst, sg_j0, sg_i0);
#endif
}
