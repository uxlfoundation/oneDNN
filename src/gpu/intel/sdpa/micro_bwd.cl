/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#include "gpu/intel/include/tile_ops.h"
#include "gpu/intel/include/types_interop.h"
#include "gpu/intel/sdpa/utils.h"

/* Microkernel headers -- generated at runtime */
#include "gemm_kq.h"
#include "gemm_ktq.h"
#include "gemm_qdSt.h"
#include "gemm_vs.h"
#include "gemm_vtdA.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define DIV_UP(x, y) (((x) + (y)-1) / (y))

#define sg_per_wg (ugemm_kq_sg_per_wg_m * ugemm_kq_sg_per_wg_n)
#define q_tile_sg_n DIV_UP(ugemm_kq_wg_tile_n, sg_per_wg)
//tmp?
#define q_tile_sg_m DIV_UP(ugemm_kq_wg_tile_m, sg_per_wg)

/* Instantiate tile types and operations */
typedef ugemm_kq_c_type s_tile_type;
typedef ugemm_vs_c_type a_tile_type;
//typedef ugemm_vtdA_c_type p_tile_type;
typedef ugemm_vs_c_type p_tile_type;

// Tile debugging example for s_tile
//
// example: declare print tile function macro for S_tile
// DECLARE_2D_TILE_PRINT(s_tile_type, float, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
//                       ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
//                       ugemm_kq_c_type_nblock1)
//
// example: Prints the entire S_tile in the (0, 1, 0) work group
// print_tile(S_tile, "%7.2f", 0, 1, 0, ugemm_kq_sg_per_wg_m, ugemm_kq_sg_per_wg_n);
DECLARE_2D_TILE_PRINT(s_tile_type, float, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1)

#ifdef QRY_DT_F32
#define FMA_TYPE float
#elif QRY_DT_F16
#define VEC_TYPE2 half2
#define FMA_TYPE half
#elif defined(QRY_DT_BF16)
#define VEC_TYPE2 ushort2
#define FMA_TYPE ushort
#else
#error "Data type not supported for VEC_TYPE2"
#endif

#ifdef SCALE_DT_BF16
#define SCALES_TO_FLOAT cvt_bf16_to_f32
#else
#define SCALES_TO_FLOAT convert_float
#endif

#ifdef VAL_ATTR_SCALES_DT_BF16
#define VAL_SCALES_TO_FLOAT cvt_bf16_to_f32
#else
#define VAL_SCALES_TO_FLOAT convert_float
#endif

#if KEY_ATTR_SCALES_DT_BF16
#define KEY_SCALES_TO_FLOAT cvt_bf16_to_f32
#else
#define KEY_SCALES_TO_FLOAT convert_float
#endif

/*
won't work, since br*nbr < SG_SZ
DECLARE_2D_TILE(a_t_tile_type, FMA_TYPE, SUBGROUP_SIZE,
                               ugemm_vs_c_type_block1, ugemm_vs_c_type_block0,
                               1, ugemm_vs_c_type_nblock1 * ugemm_vs_c_type_nblock0)
DECLARE_2D_TILE_BLOCK_OPS(a_t_tile_type, FMA_TYPE, SUBGROUP_SIZE,
                               ugemm_vs_c_type_block1, ugemm_vs_c_type_block0,
                               1, ugemm_vs_c_type_nblock1 * ugemm_vs_c_type_nblock0)
*/

#if USE_SYSTOLIC_UKERNEL
DECLARE_2D_TILE(q_tile_type, uint, SUBGROUP_SIZE, D_MAX / 2, 1, 1, q_tile_sg_n)
#else
DECLARE_2D_TILE(q_tile_type, FMA_TYPE, SUBGROUP_SIZE, D_MAX, 1, 1, q_tile_sg_n)
#endif

#if BLOCK_Q

#if USE_SYSTOLIC_UKERNEL
DECLARE_2D_TILE_BLOCK_OPS(
        q_tile_type, uint, SUBGROUP_SIZE, D_MAX / 2, 1, 1, q_tile_sg_n)
#else
DECLARE_2D_TILE_BLOCK_OPS(
        q_tile_type, FMA_TYPE, SUBGROUP_SIZE, D_MAX, 1, 1, q_tile_sg_n)
#endif

#elif Q_ALIGN < 4

#if USE_SYSTOLIC_UKERNEL
DECLARE_2D_TILE_LOAD_PACKED_VEC(q_tile_type, QRY_DATA_T, VEC_TYPE2,
        SUBGROUP_SIZE, D_MAX / 2, 1, 1, q_tile_sg_n)
#endif

#endif

#if BLOCK_A
DECLARE_2D_TILE(a_tile_type_dst, DST_DATA_T, SUBGROUP_SIZE, ugemm_vs_sg_tile_m,
        1, 1, ugemm_vs_sg_tile_n)
#else
DECLARE_2D_TILE(a_tile_type_dst, DST_DATA_T, SUBGROUP_SIZE, ugemm_vs_sg_tile_m,
        8, 1, ugemm_vs_sg_tile_n / 8)
#endif

DECLARE_2D_TILE(s_tile_type_packed, uint, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1 / 2, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1)
DECLARE_2D_TILE(s_tile_type_packed_t, uint, SUBGROUP_SIZE,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_block0 / 2,
        ugemm_kq_c_type_nblock1, ugemm_kq_c_type_nblock0)

DECLARE_2D_TILE(s_tile_type_reblock, FMA_TYPE, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_n, 1, ugemm_kq_sg_tile_n / ugemm_vs_sg_tile_n,
        ugemm_kq_sg_tile_m)
DECLARE_2D_TILE_BLOCK_OPS(s_tile_type_reblock, FMA_TYPE, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_n, 1, ugemm_kq_sg_tile_n / ugemm_vs_sg_tile_n,
        ugemm_kq_sg_tile_m)

DECLARE_2D_TILE(p_tile_type_reblock, FMA_TYPE, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_n, 1, ugemm_kq_sg_tile_n / ugemm_vs_sg_tile_n,
        ugemm_kq_sg_tile_m)
DECLARE_2D_TILE_BLOCK_OPS(p_tile_type_reblock, FMA_TYPE, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_n, 1, ugemm_kq_sg_tile_n / ugemm_vs_sg_tile_n,
        ugemm_kq_sg_tile_m)

DECLARE_2D_TILE(
        s_sum_tile_type, float, SUBGROUP_SIZE, ugemm_kq_sg_tile_n, 1, 1, 1)
DECLARE_2D_TILE_PRINT(
        s_sum_tile_type, float, SUBGROUP_SIZE, ugemm_kq_sg_tile_n, 1, 1, 1)

DECLARE_2D_TILE(
        a_scale_tile_type, float, SUBGROUP_SIZE, ugemm_vs_sg_tile_n, 1, 1, 1)

#if BROADCAST_MASK_Q
#define mask_br ugemm_kq_sg_tile_m
#define mask_bc 1
#define mask_nbr 1
#define mask_nbc 1
#else
#define mask_br ugemm_kq_c_type_block0
#define mask_bc ugemm_kq_c_type_block1
#define mask_nbr ugemm_kq_c_type_nblock0
#define mask_nbc ugemm_kq_c_type_nblock1
#endif

DECLARE_2D_TILE(kmask_tile_type_float, float, SUBGROUP_SIZE, ugemm_kq_sg_tile_m,
        1, 1, 1)

#if WITH_ATTN_MASK
DECLARE_2D_TILE(mask_tile_type, MSK_DATA_T, SUBGROUP_SIZE, mask_br, mask_bc,
        mask_nbr, mask_nbc)

#if BROADCAST_MASK_Q
DECLARE_2D_TILE_BLOCK_OPS(mask_tile_type, MSK_DATA_T, SUBGROUP_SIZE, mask_br,
        mask_bc, mask_nbr, mask_nbc)
#endif
DECLARE_2D_TILE(mask_tile_type_float, float, SUBGROUP_SIZE, mask_br, mask_bc,
        mask_nbr, mask_nbc)
DECLARE_2D_TILE_COPY_REBLOCK(mask_tile_type, SUBGROUP_SIZE, mask_br, mask_bc,
        mask_nbr, mask_nbc, mask_tile_type_float, SUBGROUP_SIZE, mask_br,
        mask_bc, mask_nbr, mask_nbc, CONVERT_FLOAT_T)
#endif

#if BLOCK_A
DECLARE_2D_TILE_BLOCK_OPS(a_tile_type_dst, DST_DATA_T, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_m, 1, 1, ugemm_vs_sg_tile_n)
#endif
#if BLOCK_2D_A
DECLARE_2D_TILE_BLOCK2D_OPS(a_tile_type_dst, DST_DATA_T, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_m, 8, 1, ugemm_vs_sg_tile_n / 8)
#endif

#if BLOCK_A
DECLARE_2D_TILE_COPY_REBLOCK(a_tile_type, SUBGROUP_SIZE, ugemm_vs_c_type_block0,
        ugemm_vs_c_type_block1, ugemm_vs_c_type_nblock0,
        ugemm_vs_c_type_nblock1, a_tile_type_dst, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_m, 1, 1, ugemm_vs_sg_tile_n, CONVERT_DATA_T)
#else
DECLARE_2D_TILE_COPY_REBLOCK(a_tile_type, SUBGROUP_SIZE, ugemm_vs_c_type_block0,
        ugemm_vs_c_type_block1, ugemm_vs_c_type_nblock0,
        ugemm_vs_c_type_nblock1, a_tile_type_dst, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_m, 8, 1, ugemm_vs_sg_tile_n / 8, CONVERT_DATA_T)
#endif

DECLARE_2D_TILE_COPY_REBLOCK(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, s_tile_type_reblock, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_n, 1, ugemm_kq_sg_tile_n / ugemm_vs_sg_tile_n,
        ugemm_kq_sg_tile_m, CONVERT_DATA_T)
DECLARE_2D_TILE_COPY_REBLOCK(p_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, p_tile_type_reblock, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_n, 1, ugemm_kq_sg_tile_n / ugemm_vs_sg_tile_n,
        ugemm_kq_sg_tile_m, CONVERT_DATA_T)

DECLARE_2D_TILE_VREDUCE(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, s_sum_tile_type, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_n, 1, 1, 1)

//DECLARE_2D_TILE_VREDUCE(p_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
//ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
//ugemm_kq_c_type_nblock1, s_sum_tile_type, SUBGROUP_SIZE,
//ugemm_kq_sg_tile_n, 1, 1, 1)

DECLARE_2D_TILE_HREDUCE(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, s_sum_tile_type, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_n, 1, 1, 1)
DECLARE_2D_TILE_HREDUCE(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, kmask_tile_type_float, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_m, 1, 1, 1)
#if WITH_ATTN_MASK
DECLARE_2D_TILE_HREDUCE(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, mask_tile_type_float, SUBGROUP_SIZE, mask_br,
        mask_bc, mask_nbr, mask_nbc)
#endif

DECLARE_2D_TILE_HREDUCE(a_tile_type, SUBGROUP_SIZE, ugemm_vs_c_type_block0,
        ugemm_vs_c_type_block1, ugemm_vs_c_type_nblock0,
        ugemm_vs_c_type_nblock1, a_scale_tile_type, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_n, 1, 1, 1)

#if ugemm_kq_wg_tile_n == ugemm_vs_wg_tile_n \
        && (ugemm_kq_sg_tile_n % ugemm_vs_sg_tile_n) == 0
DECLARE_2D_TILE_RSELECT(a_scale_tile_type, SUBGROUP_SIZE, ugemm_vs_sg_tile_n, 1,
        1, 1, s_sum_tile_type, SUBGROUP_SIZE, ugemm_kq_sg_tile_n, 1, 1, 1)
#endif

#if PREFETCH_REMAINDER
#define cooperative_prefetch_2d_maybe_rem cooperative_prefetch_2d_rem
#else
#define cooperative_prefetch_2d_maybe_rem( \
        ptr, r, c, rmax, cmax, ld, sg_id, n_sg, sg_size, caching) \
    cooperative_prefetch_2d(ptr, rmax, cmax, ld, sg_id, n_sg, sg_size, caching)
#endif

#if TRANSPOSE_K
#define cooperative_prefetch_2d_k( \
        ptr, r, c, rmax, cmax, ld, sg_id, n_sg, sg_size, caching) \
    cooperative_prefetch_2d_maybe_rem( \
            ptr, c, r, cmax, rmax, ld, sg_id, n_sg, sg_size, caching)
#else
#define cooperative_prefetch_2d_k cooperative_prefetch_2d_maybe_rem
#endif

#define tile_load_block_rem_q(t, ptr, n, ld, off_r, off_c, load_rem) \
    if (load_rem) { \
        tile_load_block(t, ptr, n, ld, off_r, off_c); \
    } else { \
        tile_load_block(t, ptr, ld, off_r, off_c); \
    }

#define tile_store_block_rem_q(t, ptr, n, ld, off_r, off_c, store_rem) \
    if (store_rem) { \
        tile_store_block(t, ptr, n, ld, off_r, off_c); \
    } else { \
        tile_store_block(t, ptr, ld, off_r, off_c); \
    }

#define binary_add(x, y) ((x) + (y))

/* As of 03/19/2025, the OpenCL compiler errors out at runtime when
   ukernels return values that go unused:

     Error during the build of OpenCL program. Build log:
     error: parsing vISA inline assembly failed:
     near line 833: null: undefined variable
     error: backend compiler failed build.

   Maneuver around the issue (e.g. while debugging) by writing data to
   volatile local memory:

     A_tile1 = ugemm_vs(...); // A_tile1 (result of microkernel) unused

     volatile local float f;  // avoid error by copying to local memory
     for (int i = 0; i < 8; i++)
         f = A_tile1.x[i][0];
*/

inline void tile_load_src0(q_tile_type *Q_tile, const global QRY_DATA_T *Q,
        int m, int n, int ldq, int offset_r, int offset_c, int load_rem) {

#if USE_SYSTOLIC_UKERNEL

#if BLOCK_Q
    tile_load_block_rem_q(Q_tile, (global uint *)Q, n, ldq >> 1, offset_r,
            offset_c, load_rem);
#elif Q_ALIGN >= 4
    tile_load(Q_tile, (global uint *)Q, (m + 1) >> 1, n, ldq >> 1, offset_r,
            offset_c);
#else
    tile_load_packed_vec2(Q_tile, Q, m, n, ldq, offset_r, offset_c);
#endif

#else // FMA

#if BLOCK_Q
    tile_load_block_rem_q(Q_tile, Q, n, ldq, offset_r, offset_c, load_rem);
#else
    tile_load(Q_tile, Q, m, n, ldq, offset_r, offset_c);
#endif

#endif
}

inline void tile_load_src1(q_tile_type *Q_tile, const global QRY_DATA_T *Q,
        int m, int n, int ldq, int offset_r, int offset_c, int load_rem) {

#if USE_SYSTOLIC_UKERNEL

#if BLOCK_Q
    tile_load_block_rem_q(Q_tile, (global uint *)Q, n, ldq >> 1, offset_r,
            offset_c, load_rem);
#elif Q_ALIGN >= 4
    tile_load(Q_tile, (global uint *)Q, (m + 1) >> 1, n, ldq >> 1, offset_r,
            offset_c);
#else
    tile_load_packed_vec2(Q_tile, Q, m, n, ldq, offset_r, offset_c);
#endif

#else // FMA

#if BLOCK_Q
    tile_load_block_rem_q(Q_tile, Q, n, ldq, offset_r, offset_c, load_rem);
#else
    tile_load(Q_tile, Q, m, n, ldq, offset_r, offset_c);
#endif

#endif
}

inline void tile_store_t_slm_src1(q_tile_type *Q_tile, local QRY_DATA_T *Q_slm,
        int panel, int ld, int offset_r, int offset_c) {
#if USE_SYSTOLIC_UKERNEL
    tile_store_t_sys_src1(
            *Q_tile, (local uint *)&Q_slm[0], ld / 2, offset_r, offset_c);
#else // FMA
    tile_store_t_packed_src1(*Q_tile, Q_slm, panel, ld, offset_r, offset_c);
#endif
}

#define DO_MM 1

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) kernel void
micro_sdpa_bwd(const global KEY_DATA_T *K, const global QRY_DATA_T *Q,
        const global VAL_DATA_T *V, const global float *ws,
        //TODO: calculate Di AoT and pass in (separate kernel?) or inline?
        const global VAL_DATA_T *A, const global VAL_DATA_T *dA,
        global DST_DATA_T *S2_test, global DST_DATA_T *dK,
        global DST_DATA_T *dQ, global DST_DATA_T *dV,
        const global SCALE_DATA_T *scale_ptr, int d, int k, int q,
        const int attn_mask_type
#if WITH_ATTN_MASK
        ,
        const global MSK_DATA_T *msk
#endif
        ,
        KEY_OFFSETS, QRY_OFFSETS, VAL_OFFSETS, DST_OFFSETS
#if WITH_ATTN_MASK
        ,
        MSK_OFFSETS
#endif
        ,
        const int remainder_k, const int remainder_q) {

    //TODO: store logsumexp instead of sums and maxs separately
    const global float *ws_maxes = ws; //TODO: batch TODO: logsumexp
    const global float *ws_sums = ws + q; //TODO: batch
    //if(get_global_id(0) == 0 && get_global_id(1) == 0)
    //printf("logsumexp%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", ws[0], ws[1], ws[2], ws[3], ws[4], ws[5], ws[6], ws[7],  ws[8], ws[9], ws[10], ws[11], ws[12], ws[13], ws[14], ws[15]);
    //return;

    //TODO: batch Di
    /*

    if(get_global_id(0) == 0) {
        for(int i=0; i<64; ++i) {
            if(i == 32) printf("\n");
            printf("ws%f\n", (float)ws[i]);
        }
    }
    return;

    */

    uint n_wg_k = DIV_UP(k, ugemm_kq_wg_tile_m);
    uint wg_k = get_group_id(0) % n_wg_k;
    uint wg_q = get_group_id(0) / n_wg_k;

    uint sg_ij = sub_group_broadcast(get_local_id(1), 0);
    uint b1 = get_group_id(2);

    uint b0, b0_kv;
    uint wg_i0 = wg_k * ugemm_kq_wg_tile_m;
    uint wg_j0 = wg_q * ugemm_kq_wg_tile_n;

    uint q_group_size;
    // tmp avoid batched grouping logic for GQA
    b0 = get_group_id(1);
    b0_kv = b0 / KV_GROUP_SIZE;
    q_group_size = q;

    /* Calculate the number of keys to process */
    int k0end = k;
    int q0end = q;
    //TODO: apply masking to keys

    /* Leading dimension for matrices */
    uint ldk = TRANSPOSE_K ? KEY_S3 : KEY_S2;
    uint ldkt = TRANSPOSE_K ? KEY_S2 : KEY_S3;
    uint ldq = QRY_S2;
    uint ldv = VAL_S2;
    uint lda = DST_S2;

    /* Subgroup IDs for each GEMM */
    uint sg_i_kq = sg_ij % ugemm_kq_sg_per_wg_m;
    uint sg_j_kq = sg_ij / ugemm_kq_sg_per_wg_m;
    //single sg w/size=16
    //printf(" sg_per_wg%d (ugemm_kq_sg_per_wg_m%d * ugemm_kq_sg_per_wg_n%d) q_tile_sg_n%d DIV_UP(ugemm_kq_wg_tile_n%d, sg_per_wg%d) sg_i_kq%d = sg_ij%d %% ugemm_kq_sg_per_wg_m %d; sg_j_kq %d = sg_ij %d / ugemm_kq_sg_per_wg_m%d  ",
    //sg_per_wg, ugemm_kq_sg_per_wg_m, ugemm_kq_sg_per_wg_n,
    //q_tile_sg_n, ugemm_kq_wg_tile_n, sg_per_wg,
    //sg_i_kq, sg_ij, ugemm_kq_sg_per_wg_m, sg_j_kq , sg_ij , ugemm_kq_sg_per_wg_m);

    // sg_per_wg1 (ugemm_kq_sg_per_wg_m1 * ugemm_kq_sg_per_wg_n1) q_tile_sg_n16 DIV_UP(ugemm_kq_wg_tile_n16, sg_per_wg1) sg_i_kq0 = sg_ij0 % ugemm_kq_sg_per_wg_m 1; sg_j_kq 0 = sg_ij 0 / ugemm_kq_sg_per_wg_m1

    uint sg_i_vs = sg_ij % ugemm_vs_sg_per_wg_m;
    uint sg_j_vs = sg_ij / ugemm_vs_sg_per_wg_m;

    /* SLM allocations -- place in one array to work around compiler bug */
#define Q_slm_size (D_MAX * ugemm_kq_wg_tile_n * sizeof(QRY_DATA_T))
#define dQ_slm_size (D_MAX * ugemm_kq_wg_tile_n * sizeof(float))
#define dK_slm_size (ugemm_kq_wg_tile_m * D_MAX * sizeof(float))
#define dV_slm_size \
    (D_MAX * ugemm_kq_wg_tile_n \
            * sizeof(float)) // TODO: is this D_MAX x  ugemm_kq_n, or m?
#define S_slm_size \
    (ugemm_kq_wg_tile_m * ugemm_kq_wg_tile_n * sizeof(QRY_DATA_T))
#define ugemm_slm_size MAX(ugemm_kq_slm_size, ugemm_vs_slm_size)

    local char slm[Q_slm_size + dQ_slm_size + dK_slm_size + dV_slm_size
            + S_slm_size + ugemm_slm_size + Q_slm_size];

    local QRY_DATA_T *Q_slm = (local QRY_DATA_T *)&slm[0];
    local float *dQ_slm = (local float *)&slm[Q_slm_size];
    local float *dK_slm = (local float *)&slm[Q_slm_size + dQ_slm_size];
    local float *dV_slm
            = (local float *)&slm[Q_slm_size + dQ_slm_size + dK_slm_size];
    local QRY_DATA_T *S_slm = (local QRY_DATA_T *)&slm[Q_slm_size + dQ_slm_size
            + dK_slm_size + dV_slm_size];
    local uint *ugemm_slm = (local uint *)&slm[Q_slm_size + dQ_slm_size
            + dK_slm_size + dV_slm_size + S_slm_size];
    local float *dS_slm = (local float *)&slm[Q_slm_size + dQ_slm_size
            + dK_slm_size + dV_slm_size + S_slm_size + ugemm_slm_size];

    const bool need_sum_barrier = (ugemm_vs_barrier_count == 0);

    const size_t k_offset = KEY_BATCH(
            b1, b0_kv); //TODO: b0_kv needed? no groups if non-quantized
    const size_t v_offset = VAL_BATCH(b1, b0_kv);
    const size_t q_offset = QRY_BATCH(b1, b0);
    const size_t a_offset = DST_BATCH(b1, b0);

    /* Locate K/Q/V/A matrices within batch */
    K += k_offset;
    Q += q_offset;
    V += v_offset;
    A += a_offset;

    dK += k_offset;
    dQ += q_offset;
    dV += v_offset;
    dA += a_offset;

#if WITH_ATTN_MASK
    msk += MSK_BATCH(b1 % MSK_D0, b0 % MSK_D1);
#if BLOCK_MSK == false
    int mask_aligned = (((size_t)msk) % 4) == 0;
#endif
#endif

    //ldq16 ldk32
    //ldq16 off_c0 remq0
    //printf("ldq%d ldk%d ldkt%d\n", ldq, ldk, ldkt);

    //printf("k0end %d : %d", k0end, ugemm_kq_wg_tile_m);
    if (k0end > 0) {
        //if (q0end > 0) {
        /* Load Q tile, destined for SLM */
        q_tile_type Q_tile;
        uint q0_copy = q_tile_sg_n * sg_ij;

        //printf("ldq%d off_c%d remq%d\n", ldq, wg_j0 + q0_copy, remainder_q);
        tile_load_src1(&Q_tile, Q, d, q_group_size, ldq, 0, wg_j0 + q0_copy,
                remainder_q);

        /*
        if(get_group_id(0) == 0) {
        for(int i=0; i<16; ++i) {
            if(get_sub_group_local_id() == i) {
                printf("[ %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f, ]\n ",
                    as_half2(Q_tile.x[0] ).s0, as_half2(Q_tile.x[0] ).s1,
                    as_half2(Q_tile.x[1] ).s0, as_half2(Q_tile.x[1] ).s1,
                    as_half2(Q_tile.x[2] ).s0, as_half2(Q_tile.x[2] ).s1,
                    as_half2(Q_tile.x[3] ).s0, as_half2(Q_tile.x[3] ).s1,
                    as_half2(Q_tile.x[4] ).s0, as_half2(Q_tile.x[4] ).s1,
                    as_half2(Q_tile.x[5] ).s0, as_half2(Q_tile.x[5] ).s1,
                    as_half2(Q_tile.x[6] ).s0, as_half2(Q_tile.x[6] ).s1,
                    as_half2(Q_tile.x[7] ).s0, as_half2(Q_tile.x[7] ).s1,
                    as_half2(Q_tile.x[8] ).s0, as_half2(Q_tile.x[8] ).s1,
                    as_half2(Q_tile.x[9] ).s0, as_half2(Q_tile.x[9] ).s1,
                    as_half2(Q_tile.x[10]).s0, as_half2(Q_tile.x[10]).s1,
                    as_half2(Q_tile.x[11]).s0, as_half2(Q_tile.x[11]).s1,
                    as_half2(Q_tile.x[12]).s0, as_half2(Q_tile.x[12]).s1,
                    as_half2(Q_tile.x[13]).s0, as_half2(Q_tile.x[13]).s1,
                    as_half2(Q_tile.x[14]).s0, as_half2(Q_tile.x[14]).s1,
                    as_half2(Q_tile.x[15]).s0, as_half2(Q_tile.x[15]).s1);
            }
        }
        }
        return
        */

        barrier(CLK_LOCAL_MEM_FENCE);

        ///* Store Q tile to SLM */
        tile_store_t_slm_src1(
                &Q_tile, Q_slm, ugemm_kq_sg_tile_n, D_MAX, q0_copy, 0);

        //#if Q_ARRIVE_AWAIT_BARRIER
        barrier(CLK_LOCAL_MEM_FENCE);
        //intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);
        //#endif

        /*
        if(get_group_id(0) == 0 && get_sub_group_local_id() == 0) {
            for(int i=0; i<ugemm_kq_wg_tile_n; ++i) {
                for(int j=0; j<D_MAX; ++j) {
                        printf("%6.1f ", Q_slm[i * D_MAX + j]);
                }
                printf("qslm\n");
            }

        }
        */
    }

    // printf("%p %p %p %p scale\n", dK, dQ, dV, dA); return;
    /* Load scale */
    float scale = 1.f;
    float iscale = 1.f;
    if (k0end > 0) {
        //if (q0end > 0) {
#if WITH_ATTN_SCALE
#if INVERT_SCALE
        iscale = SCALES_TO_FLOAT(*scale_ptr);
        scale = native_recip(iscale);
#else
        scale = SCALES_TO_FLOAT(*scale_ptr);
        iscale = native_recip(scale);
#endif
#endif
        //scale *= 1.442695f; // log2(e)
    }

    //if (k0end > 0) {
    if (q0end > 0) {
        /* Initialize dQ, dK to zero */
        const uint n_col_sg
                = DIV_UP(ugemm_kq_wg_tile_n, SUBGROUP_SIZE * sg_per_wg);
        const float zero = 0.f;
#pragma unroll
        for (int q = 0; q < n_col_sg; q++) {
            intel_sub_group_block_write(
                    (local uint *)&dQ_slm[(q + sg_ij * n_col_sg)
                            * SUBGROUP_SIZE],
                    as_uint(zero));
            intel_sub_group_block_write(
                    (local uint *)&dK_slm[(q + sg_ij * n_col_sg)
                            * SUBGROUP_SIZE],
                    as_uint(zero));
        }
    }

    uint sg_i0_kq = sg_i_kq * ugemm_kq_sg_tile_m; // *16
    uint sg_j0_kq = sg_j_kq * ugemm_kq_sg_tile_n; // *16

    //s_sum_tile_type S_sum_tile, S_max_tile;
    s_sum_tile_type S_logsumexp_tile;
    //TODO: tile load (non-full) w/bounds check
    tile_load_full(&S_logsumexp_tile, ws_maxes, ugemm_kq_wg_tile_n,
            sg_j0_kq + wg_j0, 0);
    //tile_load_full(
    //&S_max_tile, ws_maxes, ugemm_kq_wg_tile_n, sg_j0_kq + wg_j0, 0);
    //tile_load_full(
    //&S_sum_tile, ws_sums, ugemm_kq_wg_tile_n, sg_j0_kq + wg_j0, 0);

    //#if SOFTMAX_INF_AS_ZERO //TODO: needed? would be set in forward pass so just read should suffice? can differ in training and forward?? ask Yixin
    //#define set_zeros(v) vselect(-FLT_MAX, v, visfinite(v))
    //tile_elementwise(S_max_tile, set_zeros);

    /* Rescale by 1 / (column sums) */
    //#define set_zeros2(v) (vselect(native_vrecip(v), 1.f, v == 0))
    //tile_elementwise(S_sum_tile, set_zeros2);
    //#else
    //tile_elementwise(S_sum_tile, native_vrecip);
    //#endif

    // print_tile(S_max_tile, "%7.3f", 0, 0 ,0, ugemm_kq_sg_per_wg_m, ugemm_kq_sg_per_wg_n);
    // print_tile(S_sum_tile, "%7.3f", 0, 0 ,0, ugemm_kq_sg_per_wg_m, ugemm_kq_sg_per_wg_n);

    a_tile_type dV_tile;

    if (k0end > 0) {
        //if (q0end > 0) {
        /* Clear accumulator */
        tile_fill(dV_tile, 0.0f);

        //#if Q_ARRIVE_AWAIT_BARRIER
        //intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
        //#else
        barrier(CLK_LOCAL_MEM_FENCE);
        //#endif
    }

    /* Main loop over k blocks */
    //for (int k0 = 0; k0 < k0end; k0 += ugemm_kq_wg_tile_m) {
    //for (int q0 = 0; q0 < q0end; q0 += ugemm_kq_wg_tile_n) {

    int k0 = wg_i0;

    //bool first = (q0 == 0);
    //int knext = q0 + ugemm_kq_wg_tile_n;
    //bool last = (qnext >= q0end);

    bool first = (k0 == 0);
    int knext = k0 + ugemm_kq_wg_tile_m;
    bool last = (knext >= k0end);
    //
    //        // TODO: load attn mask
    //
    //        // TODO: need k mask for in/oob ?
    // ldk = 32, D_MAX=32>hs16, k0end:total#k, ugemm_kq_wg_tile_n < queries, d=16
    // ldq16 ldk32

    /* Calculate S = (K^T) * Q */
#if DO_MM
    s_tile_type S_tile
            = ugemm_kq(K, ldk, Q_slm, D_MAX, k0end, ugemm_kq_wg_tile_n, d, k0,
                    0, 0, sg_i_kq, sg_j_kq, (local char *)ugemm_slm);
#else
    s_tile_type S_tile;
#endif

    /* TODO: Apply attention mask */

    /* TODO: Apply k mask */
    //if (remainder_k) { tile_hbroadcast_min(&S_tile, k_mask); }
    /* TODO: causal masking fns???  */

    /* Before softmax, we will need to scale columns by maximum values to avoid overflow. */
    //tile_vbroadcast_sub(&S_tile, S_max_tile);
#define mulscale(x) (x * scale)
    tile_elementwise(S_tile, mulscale);
    tile_vbroadcast_sub(&S_tile, S_logsumexp_tile);

/* Scale + exponentiate */
#define scaled_exp(x) native_vexp2(x * 1.442695f)
    tile_elementwise(S_tile, scaled_exp);

    //print_tile(S_sum_tile, "%7.3f", 0, 0 ,0, ugemm_kq_sg_per_wg_m, ugemm_kq_sg_per_wg_n);
    //tile_vbroadcast_mul(&S_tile, S_sum_tile);

    uint sg_i0_s2 = sg_i_kq * ugemm_kq_sg_tile_m + k0;
    uint sg_j0_s2 = sg_j_kq * ugemm_kq_sg_tile_n + wg_j0;
    s_tile_type_reblock S_tile_reblock;
    tile_copy_reblock(S_tile, &S_tile_reblock);
    //tile_store(S_tile, S2_test, 32, 32, 32, sg_j0_s2, sg_i0_s2);
    //return;

    //uint sg_i0_ds = sg_i_kq * ugemm_kq_sg_tile_m;
    //uint sg_j0_ds = sg_j_kq * ugemm_kq_sg_tile_n;
    //tile_store_full(S_tile, dS_slm, ugemm_kq_wg_tile_m, sg_i0_ds, sg_j0_ds);

#if USE_SYSTOLIC_UKERNEL
    tile_store_sys_src2(S_tile_reblock, (local FMA_TYPE *)S_slm,
            ugemm_vs_sg_tile_n, ugemm_kq_wg_tile_n, sg_i0_kq, sg_j0_kq);
#else
    tile_store_t_packed_src1(S_tile_reblock, S_slm, ugemm_vs_sg_tile_n,
            ugemm_kq_wg_tile_m, sg_j0_kq, sg_i0_kq);
#endif

    //   a_tile_type ttile;
    //   tile_fill(ttile, get_sub_group_local_id());
    //   //tile_store_full(ttile, S_slm, D_MAX, sg_i_vs * ugemm_vs_sg_tile_m, sg_j_vs * ugemm_vs_sg_tile_n);
    //   tile_store_t_full(ttile, S_slm, D_MAX, sg_i_vs * ugemm_vs_sg_tile_m, sg_j_vs * ugemm_vs_sg_tile_n);
    //   barrier(CLK_LOCAL_MEM_FENCE);
    //     if(get_group_id(0) == 0 && get_sub_group_local_id() == 0) {
    //         for(int i=0; i<ugemm_kq_wg_tile_n; ++i) {
    //             for(int j=0; j<D_MAX; ++j) {
    //                     printf("%6.1f ", S_slm[i * D_MAX + j]);
    //             }
    //             printf("sslm\n");
    //         }
    //
    //   }
    //   return;

    barrier(CLK_LOCAL_MEM_FENCE);

    int k_chunk = min(k0end - k0, ugemm_kq_wg_tile_m);
#if DO_MM
    a_tile_type dV_tile1 = ugemm_vs(dA + ldv * wg_j0, lda, S_slm,
            ugemm_kq_wg_tile_m, d, ugemm_kq_wg_tile_n, k_chunk, 0, 0, 0,
            sg_i_vs, sg_j_vs, (local char *)ugemm_slm);
#else
    a_tile_type dV_tile1;
#endif

    tile_binary(dV_tile, dV_tile1, binary_add);
    // tile_store(dV_tile, S2_test, 32, 32, 32, sg_j0_s2, sg_i0_s2);
    //
    // update dV
    a_tile_type_dst dV_tile_dst;
    /* Convert to half precision and store */
    tile_copy_reblock(dV_tile, &dV_tile_dst);
    //tile_copy_reblock(dV_tile2, &dV_tile_dst);

    uint sg_i0_vs = sg_i_vs * ugemm_vs_sg_tile_m;
    uint sg_j0_vs = sg_j_vs * ugemm_vs_sg_tile_n;

    //if ((wg_i0==16 && wg_j0==0) || (wg_i0==0 && wg_j0==0)) {
    //tile_store(dV_tile_dst, dV, d, q_group_size, ldv, sg_i0_vs, sg_j0_vs);
    //}

    tile_atomic_add(dV_tile_dst, dV + k0 * ldv, d, k, ldv, sg_i0_vs, sg_j0_vs);

    // /update dV
    // return;

    s_sum_tile_type D_i;
    tile_fill(D_i, 0.0f);
    if (k0end > 0) {
        /* Load Q tile, destined for SLM */
        s_tile_type dA_tile1, A_tile; // for D_i calculation
        q_tile_type
                dA_tile; //TODO: convert dA_tile -> dA_tile1 : s_tile_type instead of double read
        uint q0_copy = q_tile_sg_n * sg_ij;

        tile_load_src1(&dA_tile, dA, d, q_group_size, lda, 0, wg_j0 + q0_copy,
                remainder_q);
        tile_store_t_slm_src1(
                &dA_tile, Q_slm, ugemm_kq_sg_tile_n, D_MAX, q0_copy, 0);
        //tile_store_packed_src1(
        //dA_tile, Q_slm, ugemm_kq_sg_tile_n, D_MAX, q0_copy, 0);

        tile_load(&dA_tile1, (global FMA_TYPE *)dA, d, q, lda, 0,
                wg_j0 + q0_copy);
        tile_load(&A_tile, (global FMA_TYPE *)A, d, q, lda, 0, wg_j0 + q0_copy);
#define binary_mul(x, y) ((x) * (y))
        tile_binary(A_tile, dA_tile1, binary_mul);
        //h reduce this shit, maybe in SLM, this should be separate kernel
        //tile_vreduce_add(A_tile, &D_i); // TODO: may be transposed?
        //tile_hreduce_add(A_tile, &D_i); // TODO: may be transposed?
        //tile_store(A_tile, S2_test, 32, 32, 32, sg_i0_s2, sg_j0_s2);   // layout.C = N
        tile_store_t_full(
                A_tile, S_slm, ugemm_kq_wg_tile_m, sg_i0_kq, sg_j0_kq);
        for (int j = 0; j < ugemm_kq_wg_tile_n; j++) {
            for (int i = get_sub_group_local_id(); i < ugemm_kq_wg_tile_m;
                    i += SUBGROUP_SIZE) {
                if (j != 0) S_slm[i] += S_slm[j * ugemm_kq_wg_tile_m + i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        tile_load_full(&D_i, S_slm, ugemm_kq_wg_tile_m, sg_i0_kq, sg_j0_kq);

        //tile_load_full(&A_tile, S_slm, ugemm_kq_wg_tile_m, sg_i0_kq, sg_j0_kq);
        //tile_store(A_tile, S2_test, 32, 32, 32, sg_i0_s2, sg_j0_s2);   // layout.C = N
        //return;

        barrier(CLK_LOCAL_MEM_FENCE);

        //             if(get_group_id(0) == 0 && get_sub_group_local_id() == 0) {
        //                for(int i=0; i<ugemm_kq_wg_tile_n; ++i) {
        //                    for(int j=0; j<D_MAX; ++j) {
        //                            printf("%6.1f ", Q_slm[i * D_MAX + j]);
        //                    }
        //                    printf("daslm\n");
        //                }

        //             }
        //             return;
    }

    // tile_fill(dA_tile1, get_group_id(0));

#if DO_MM
    p_tile_type P_tile = ugemm_vtdA(V + k0 * ldv, ldv, Q_slm, D_MAX,
            ugemm_kq_wg_tile_n, k0end, d, 0, 0, 0, sg_i_kq, sg_j_kq,
            (local char *)ugemm_slm);
#else
    p_tile_type P_tile;
#endif
    //if(get_sub_group_id() == 0 && get_group_id(0) == 0)
    //printf("D_i %f ", D_i.x[0].s0);
    // printf("Stile %f ", S_tile.x[0].s0);
    tile_store(P_tile, S2_test, 32, 32, 32, sg_j0_s2, sg_i0_s2); // layout.C = T
    //tile_store(P_tile, S2_test, 32, 32, 32, sg_i0_s2, sg_j0_s2);   // layout.C = N
    //return; //seems right (transposed?)

    //tile_hbroadcast_sub(&P_tile, D_i);
    tile_vbroadcast_sub(&P_tile, D_i);

    //tile_store(P_tile, S2_test, 32, 32, 32, sg_i0_s2, sg_j0_s2);
    //tile_store(P_tile, S2_test, 32, 32, 32, sg_j0_s2, sg_i0_s2);
    //return;

    //tile_fill(S_tile, 1.f);

    // reload since too many registers used
    //tile_load_full(&S_tile, dS_slm, ugemm_kq_wg_tile_m, sg_i_kq * ugemm_kq_sg_tile_m, sg_j_kq * ugemm_kq_sg_tile_n);
    //barrier(CLK_LOCAL_MEM_FENCE);
    tile_binary(P_tile, S_tile, binary_mul); // is this right?

#define scale_dS(x) (x * scale)
    //tile_elementwise(P_tile, scale_dS); // TODO: scaled by log2? scale needed at all?? rolled into dP?

    p_tile_type_reblock P_tile_reblock;
    tile_copy_reblock(P_tile, &P_tile_reblock);
    tile_store(P_tile_reblock, S2_test, 32, 32, 32, sg_j0_s2, sg_i0_s2);
    //tile_store(S_tile, S2_test, 32, 32, 32, sg_i0_s2, sg_j0_s2);
    //return;

    // SLM for dK = dS^t * Q
    tile_store_t_packed_src1(P_tile_reblock, S_slm, ugemm_vs_sg_tile_n,
            ugemm_kq_wg_tile_m, sg_j0_kq, sg_i0_kq);

    // SLM for dQ = dS * K, panel = sg_tile_n, ld=wg_tile_m
    tile_store_block_packed(P_tile_reblock, Q_slm, ugemm_vs_sg_tile_n,
            ugemm_kq_wg_tile_m, sg_j0_kq, sg_i0_kq);
    //tile_store_t_packed_src1(P_tile_reblock, Q_slm, ugemm_vs_sg_tile_n, ugemm_kq_wg_tile_m, sg_j0_kq, sg_i0_kq);

    barrier(CLK_LOCAL_MEM_FENCE);

    // dK = dS^t * Q
#if DO_MM
    a_tile_type dK_tile1
            //= ugemm_vs(Q + ldq * wg_j0, ldq, S_slm, ugemm_kq_wg_tile_m,
            = ugemm_qdSt(Q + ldq * wg_j0, ldq, S_slm, ugemm_kq_wg_tile_m, d,
                    ugemm_kq_wg_tile_n, ugemm_kq_wg_tile_m, 0, 0, 0, sg_i_vs,
                    sg_j_vs, (local char *)ugemm_slm);
#else
    a_tile_type dK_tile1;
#endif
    // tile_store(dK_tile1, S2_test, 32, 32, 32, sg_j0_s2, sg_i0_s2);

    // update dK
    a_tile_type_dst dK_tile_dst;
    /* Convert to half precision and store */
    tile_copy_reblock(dK_tile1, &dK_tile_dst);

    uint sg_i0_dk = sg_i_kq * ugemm_kq_sg_tile_m;
    uint sg_j0_dk = sg_j_kq * ugemm_kq_sg_tile_n;
    tile_atomic_add(dK_tile_dst, dK + k0, k, d, ldk, sg_i0_dk, sg_j0_dk);

    ////tile_fill(dK_tile1, get_group_id(0) + 1);

    /*
    //tile_store_t_full(dK_tile1, S_slm, 32, sg_i_vs * ugemm_vs_sg_tile_m, sg_j_vs * ugemm_vs_sg_tile_n);
    //barrier(CLK_LOCAL_MEM_FENCE);

    uint sg_i0_dk = sg_i_kq * ugemm_kq_sg_tile_m;
    uint sg_j0_dk = sg_j_kq * ugemm_kq_sg_tile_n + k0;

    // write transposed slm to global
    __global KEY_DATA_T* dK_write = dK + ldk * (sg_i0_dk) + sg_j0_dk;
    _Pragma("unroll") for (int j = 0; j < ugemm_vs_wg_tile_n; j++, dK_write += ldk) {
        _Pragma("unroll") for (int i0 = 0; i0 < ugemm_vs_wg_tile_m; i0 += SUBGROUP_SIZE) {
            // TODO: bounds checking?
            int i = i0 + get_sub_group_local_id();
            global_atomic_add(dK_write + i, S_slm[j * D_MAX + i]);
        }
    }
    */

    // /update dK

    barrier(CLK_LOCAL_MEM_FENCE);

    // figure out dS00, dS01 * K0, K1 order
    // dS00*K0 + dS01*K1 -> correct?
    // K0'*dS00' + K1'*dS01'

    // dQ = dS * K
#if DO_MM
    a_tile_type dQ_tile = ugemm_ktq(K + k0, ldk, Q_slm, ugemm_kq_wg_tile_m, d,
            ugemm_kq_wg_tile_n, ugemm_kq_wg_tile_m, 0, 0, 0, sg_i_vs, sg_j_vs,
            (local char *)ugemm_slm);
#else
    a_tile_type dQ_tile;
#endif

    // update dQ
    a_tile_type_dst dQ_tile_dst;
    /* Convert to half precision and store */
    tile_copy_reblock(dQ_tile, &dQ_tile_dst);

    //tile_atomic_add(dQ_tile_dst,
    //    dQ + wg_j0, k, d, ldk, sg_j0_vs, sg_i0_vs);

    //tile_fill(dQ_tile_dst, wg_i0 + 1); // 1,2,3,4

    // tile_store(dQ_tile_dst, S2_test, 32, 32, 32, sg_j0_s2, sg_i0_s2);

    uint sg_i0_dq = sg_i_kq * ugemm_kq_sg_tile_m;
    uint sg_j0_dq = sg_j_kq * ugemm_kq_sg_tile_n;
    // sum accross rows (i?)
    tile_atomic_add(
            dQ_tile_dst, dQ + wg_j0 * ldq, k, d, ldq, sg_i0_dq, sg_j0_dq);

    // sum accross cols (j?)
    //tile_atomic_add(dQ_tile_dst,
    //dQ + k0 * ldq, k, d, ldq, sg_i0_dq, sg_j0_dq);
}
