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
#define DIV_UP(x, y) (((x) + (y) - 1) / (y))

#define sg_per_wg (ugemm_kq_sg_per_wg_m * ugemm_kq_sg_per_wg_n)
#define q_tile_sg_n DIV_UP(ugemm_kq_wg_tile_n, sg_per_wg)

//tmp?
#define sg_per_wg_Bc (ugemm_qdSt_sg_per_wg_m * ugemm_qdSt_sg_per_wg_n)
#define sg_per_wg_Br (ugemm_ktq_sg_per_wg_m * ugemm_ktq_sg_per_wg_n)
//#define q_tile_sg_m DIV_UP(ugemm_kq_wg_tile_m, sg_per_wg)

#define dmax_tile_sg_n DIV_UP(D_MAX, sg_per_wg)

/* Instantiate tile types and operations */
typedef ugemm_kq_c_type s_tile_type; // Bc*Br tile
typedef ugemm_qdSt_c_type a_tile_type; // D*Bc tile
typedef ugemm_vtdA_c_type p_tile_type; // Bc*Br tile
typedef ugemm_vs_c_type dv_tile_type; // D*Bc tile
typedef ugemm_ktq_c_type ktq_tile_type; // D*Br tile

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

#if USE_SYSTOLIC_UKERNEL
DECLARE_2D_TILE(k_tile_type, uint, SUBGROUP_SIZE, ugemm_kq_wg_tile_m, 1, 1,
        dmax_tile_sg_n)
#else
DECLARE_2D_TILE(k_tile_type, FMA_TYPE, SUBGROUP_SIZE, ugemm_kq_wg_tile_m, 1, 1,
        dmax_tile_sg_n)
#endif

#if BLOCK_Q

#if USE_SYSTOLIC_UKERNEL
DECLARE_2D_TILE_BLOCK_OPS(k_tile_type, uint, SUBGROUP_SIZE,
        ugemm_kq_wg_tile_m / 2, 1, 1, dmax_tile_sg_n)
#else
DECLARE_2D_TILE_BLOCK_OPS(k_tile_type, FMA_TYPE, SUBGROUP_SIZE,
        ugemm_kq_wg_tile_m, 1, 1, dmax_tile_sg_n)
#endif

#elif Q_ALIGN < 4

#if USE_SYSTOLIC_UKERNEL
DECLARE_2D_TILE_LOAD_PACKED_VEC(
        k_tile_type, QRY_DATA_T, VEC_TYPE2, SUBGROUP_SIZE, D_MAX / 2)
#endif

#endif

#if BLOCK_A
DECLARE_2D_TILE(a_tile_type_dst, DST_DATA_T, SUBGROUP_SIZE, ugemm_kq_sg_tile_m,
        1, 1, ugemm_kq_sg_tile_n)
#else
DECLARE_2D_TILE(a_tile_type_dst, DST_DATA_T, SUBGROUP_SIZE, ugemm_kq_sg_tile_m,
        8, 1, ugemm_kq_sg_tile_n / 8)
#endif

DECLARE_2D_TILE(s_tile_type_packed, uint, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1 / 2, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1)
DECLARE_2D_TILE(s_tile_type_packed_t, uint, SUBGROUP_SIZE,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_block0 / 2,
        ugemm_kq_c_type_nblock1, ugemm_kq_c_type_nblock0)

DECLARE_2D_TILE(s_tile_type_reblock, FMA_TYPE, SUBGROUP_SIZE,
        ugemm_kq_c_type_block0, 1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_block1 *ugemm_kq_c_type_nblock1)
DECLARE_2D_TILE_BLOCK_OPS(s_tile_type_reblock, FMA_TYPE, SUBGROUP_SIZE,
        ugemm_kq_c_type_block0, 1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_block1 *ugemm_kq_c_type_nblock1)

//TODO: this reblock !matching p_tile_type, use nblock as well? 2d loads require bc=1
DECLARE_2D_TILE(p_tile_type_reblock, FMA_TYPE, SUBGROUP_SIZE,
        ugemm_vtdA_c_type_block0, 1, ugemm_vtdA_c_type_nblock0,
        ugemm_vtdA_c_type_block1 *ugemm_vtdA_c_type_nblock1)
DECLARE_2D_TILE_BLOCK_OPS(p_tile_type_reblock, FMA_TYPE, SUBGROUP_SIZE,
        ugemm_vtdA_c_type_block0, 1, ugemm_vtdA_c_type_nblock0,
        ugemm_vtdA_c_type_block1 *ugemm_vtdA_c_type_nblock1)

DECLARE_2D_TILE(
        s_sum_tile_type, float, SUBGROUP_SIZE, ugemm_kq_sg_tile_n, 1, 1, 1)
DECLARE_2D_TILE(
        p_sum_tile_type, float, SUBGROUP_SIZE, ugemm_vtdA_sg_tile_n, 1, 1, 1)

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
        ugemm_kq_sg_tile_m, 1, 1, ugemm_kq_sg_tile_n)
#endif
#if BLOCK_2D_A
DECLARE_2D_TILE_BLOCK2D_OPS(a_tile_type_dst, DST_DATA_T, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_m, 8, 1, ugemm_kq_sg_tile_n / 8)
#endif

#if BLOCK_A
DECLARE_2D_TILE_COPY_REBLOCK(a_tile_type, SUBGROUP_SIZE, ugemm_vs_c_type_block0,
        ugemm_vs_c_type_block1, ugemm_vs_c_type_nblock0,
        ugemm_vs_c_type_nblock1, a_tile_type_dst, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_m, 1, 1, ugemm_kq_sg_tile_n, CONVERT_DATA_T)
#else
DECLARE_2D_TILE_COPY_REBLOCK(a_tile_type, SUBGROUP_SIZE, ugemm_vs_c_type_block0,
        ugemm_vs_c_type_block1, ugemm_vs_c_type_nblock0,
        ugemm_vs_c_type_nblock1, a_tile_type_dst, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_m, 8, 1, ugemm_kq_sg_tile_n / 8, CONVERT_DATA_T)
#endif

DECLARE_2D_TILE_COPY_REBLOCK(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, s_tile_type_reblock, SUBGROUP_SIZE,
        ugemm_kq_c_type_block0, 1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_block1 *ugemm_kq_c_type_nblock1, CONVERT_DATA_T)
DECLARE_2D_TILE_COPY_REBLOCK(p_tile_type, SUBGROUP_SIZE,
        ugemm_vtdA_c_type_block0, ugemm_vtdA_c_type_block1,
        ugemm_vtdA_c_type_nblock0, ugemm_vtdA_c_type_nblock1,
        p_tile_type_reblock, SUBGROUP_SIZE, ugemm_vtdA_c_type_block0, 1,
        ugemm_vtdA_c_type_nblock0,
        ugemm_vtdA_c_type_block1 *ugemm_vtdA_c_type_nblock1, CONVERT_DATA_T)

DECLARE_2D_TILE_VREDUCE(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, s_sum_tile_type, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_n, 1, 1, 1)

DECLARE_2D_TILE_VREDUCE(p_tile_type, SUBGROUP_SIZE, ugemm_vtdA_c_type_block0,
        ugemm_vtdA_c_type_block1, ugemm_vtdA_c_type_nblock0,
        ugemm_vtdA_c_type_nblock1, p_sum_tile_type, SUBGROUP_SIZE,
        ugemm_vtdA_sg_tile_n, 1, 1, 1)

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

inline void tile_load_k(k_tile_type *K_tile, const global KEY_DATA_T *K, int m,
        int n, int ldq, int offset_r, int offset_c, int load_rem) {

#if USE_SYSTOLIC_UKERNEL

#if BLOCK_Q
    tile_load_block_rem_q(K_tile, (global uint *)K, n, ldq >> 1, offset_r,
            offset_c, load_rem);
#elif Q_ALIGN >= 4
    tile_load(K_tile, (global uint *)K, (m + 1) >> 1, n, ldq >> 1, offset_r,
            offset_c);
#else
    tile_load_packed_vec2(K_tile, K, m, n, ldq, offset_r, offset_c);
#endif

#else // FMA

#if BLOCK_Q
    tile_load_block_rem_q(K_tile, K, n, ldq, offset_r, offset_c, load_rem);
#else
    tile_load(K_tile, K, m, n, ldq, offset_r, offset_c);
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

inline void tile_store_t_slm_k(k_tile_type *K_tile, local QRY_DATA_T *K_slm,
        int panel, int ld, int offset_r, int offset_c) {
#if USE_SYSTOLIC_UKERNEL
    tile_store_t_sys_src1(
            *K_tile, (local uint *)&K_slm[0], ld / 2, offset_r, offset_c);
#else // FMA
    tile_store_packed_src1(*K_tile, K_slm, panel, ld, offset_r, offset_c);
#endif
}

#define DO_MM 1

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) kernel void
micro_sdpa_bwd(const global KEY_DATA_T *K, const global QRY_DATA_T *Q,
        const global VAL_DATA_T *V, global float *ws,
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

    uint wg_k = get_group_id(0);

    uint sg_ij = sub_group_broadcast(get_local_id(1), 0);

    uint b0, b1;
    b0 = get_group_id(1);
    b1 = get_group_id(2);

    uint wg_i0 = wg_k * ugemm_kq_wg_tile_m;

    const uint preprocess_batch = b1 * (DST_D1 * 2 * q) + b0 * (2 * q);
    const global float *ws_logsumexp = ws + preprocess_batch;
    const global float *ws_Di = ws + preprocess_batch + q;

    /* Calculate the number of keys to process */
    int k0end = k;
    int q0end = q;
    //TODO: apply masking to keys

    /* Leading dimension for matrices */
    uint ldk = TRANSPOSE_K ? KEY_S3 : KEY_S2;
    uint ldkt = TRANSPOSE_K ? KEY_S2 : KEY_S3; //todo: remove
    uint ldq = QRY_S2;
    uint ldv = VAL_S2;
    uint lda = DST_S2;

    /* Subgroup IDs for each GEMM */
    uint sg_i_kq = sg_ij % ugemm_kq_sg_per_wg_m;
    uint sg_j_kq = sg_ij / ugemm_kq_sg_per_wg_m;

    uint sg_i_vs = sg_ij % ugemm_vs_sg_per_wg_m;
    uint sg_j_vs = sg_ij / ugemm_vs_sg_per_wg_m;

    uint sg_i_qdSt = sg_ij % ugemm_qdSt_sg_per_wg_m;
    uint sg_j_qdSt = sg_ij / ugemm_qdSt_sg_per_wg_m;

    uint sg_i_ktq = sg_ij % ugemm_ktq_sg_per_wg_m;
    uint sg_j_ktq = sg_ij / ugemm_ktq_sg_per_wg_m;

    /* SLM allocations -- place in one array to work around compiler bug */
#define K_slm_size (ugemm_kq_wg_tile_m * D_MAX * sizeof(KEY_DATA_T))
#define Q_slm_size (D_MAX * ugemm_kq_wg_tile_n * sizeof(QRY_DATA_T))
#define S_slm_size (ugemm_kq_wg_tile_n * ugemm_kq_wg_tile_m * sizeof(float))
//#define dS_slm_size (ugemm_kq_wg_tile_m * ugemm_kq_wg_tile_n * sizeof(float))

#define dQ_slm_size \
    (D_MAX * ugemm_kq_wg_tile_n \
            * sizeof(float)) // not used yet, needed w/more register pressure?
#define dK_slm_size (ugemm_kq_wg_tile_m * D_MAX * sizeof(float))
#define dV_slm_size (ugemm_kq_wg_tile_m * D_MAX * sizeof(float))

#define ugemm_slm_size \
    MAX(MAX(MAX(MAX(ugemm_kq_slm_size, ugemm_vs_slm_size), \
                    ugemm_vtdA_slm_size), \
                ugemm_qdSt_slm_size), \
            ugemm_ktq_slm_size)

    local char slm[K_slm_size + Q_slm_size + S_slm_size
       //+ dS_slm_size
            + ugemm_slm_size];
            //+ ugemm_slm_size + dK_slm_size + dV_slm_size];

    // + dQ_slm_size + dK_slm_size + dV_slm_size // TODO: reintroduce?

    local KEY_DATA_T *K_slm = (local KEY_DATA_T *)&slm[0];

    // used for caching various A,B gemm tiles
    // allocating max data type width (sizeof(float) >= sizeof(QRY_DATA_T, VAL_DATA_T))
    local float *Q_slm = (local float *)&slm[K_slm_size];
    local float *S_slm = (local float *)&slm[K_slm_size + Q_slm_size];

    // used to store intermediate score between multiple gemms since registers get clobbered
    //local float *dS_slm
            //= (local float *)&slm[K_slm_size + Q_slm_size + S_slm_size];

    // ugemm scratch space
    local uint *ugemm_slm = (local uint
                    *)&slm[K_slm_size + Q_slm_size + S_slm_size];
                    //*)&slm[K_slm_size + Q_slm_size + S_slm_size + dS_slm_size];

    // used for accumulation of dV, dK across q-loop
    /*
    */
    local float *dK_slm = (local float *)&slm[K_slm_size + Q_slm_size
            //+ S_slm_size + dS_slm_size + ugemm_slm_size];
            + S_slm_size + ugemm_slm_size];
    local float *dV_slm = (local float *)&slm[K_slm_size + Q_slm_size
            //+ S_slm_size + dS_slm_size + ugemm_slm_size + dK_slm_size];
            + S_slm_size + ugemm_slm_size + dK_slm_size];

    const size_t k_offset = KEY_BATCH(b1, b0);
    const size_t v_offset = VAL_BATCH(b1, b0);
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

    if (q0end > 0) {
        /* Load K tile, destined for SLM */

        k_tile_type K_tile;

        uint k0_copy = dmax_tile_sg_n
                * sg_ij; //each sg will be responsible for dmax_tile_sg_n columns
        tile_load_k(&K_tile, K, k, d, ldk, wg_i0, k0_copy, remainder_k);
        ///* Store K tile to SLM */

        // QQQ what is panel even for?? is it just panel_size=64? w/Pr
        tile_store_t_slm_k(&K_tile, K_slm, ugemm_kq_wg_tile_m,
                ugemm_kq_wg_tile_m, 0, k0_copy);

    }

    /* Load scale */
    float scale = 1.f;
    float iscale = 1.f;
    if (q0end > 0) {
#if WITH_ATTN_SCALE
#if INVERT_SCALE
        iscale = SCALES_TO_FLOAT(*scale_ptr);
        scale = native_recip(iscale);
#else
        scale = SCALES_TO_FLOAT(*scale_ptr);
        iscale = native_recip(scale);
#endif
#endif
    }

    // not used currently, needed w/more reg pressure?
    if (q0end > 0) {
        /* Initialize dQ, dK to zero */
        const uint n_col_sg
                = DIV_UP(ugemm_kq_wg_tile_n, SUBGROUP_SIZE * sg_per_wg);
        const float zero = 0.f;
#pragma unroll
        for (int q = 0; q < n_col_sg; q++) {
            intel_sub_group_block_write(
                    (local uint *)&dK_slm[(q + sg_ij * n_col_sg)
                            * SUBGROUP_SIZE],
                    as_uint(zero));
            intel_sub_group_block_write(
                    (local uint *)&dV_slm[(q + sg_ij * n_col_sg)
                            * SUBGROUP_SIZE],
                    as_uint(zero));
        }
    }

    uint sg_i0_kq = sg_i_kq * ugemm_kq_sg_tile_m; // *16
    uint sg_j0_kq = sg_j_kq * ugemm_kq_sg_tile_n; // *16

    a_tile_type dK_tile;
    dv_tile_type dV_tile;
    tile_fill(dK_tile, 0.0f);
    tile_fill(dV_tile, 0.0f);

    if (q0end > 0) {
        /* Clear accumulator */
        //TODO: split barrier for k in slm
        //#if Q_ARRIVE_AWAIT_BARRIER
        //intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
        //#else
        barrier(CLK_LOCAL_MEM_FENCE);
        //#endif
    }

    const int k0 = wg_i0;

    /* Main loop over k blocks */
    for (int q0 = 0; q0 < q0end; q0 += ugemm_kq_wg_tile_n) {

        bool first = (q0 == 0);
        int qnext = q0 + ugemm_kq_wg_tile_n;
        bool last = (qnext >= q0end);

        //        // TODO: load attn mask
        //
        //        // TODO: need k mask for in/oob ?

        /* Calculate S = (K^T) * Q */
#if DO_MM
        s_tile_type S_tile = ugemm_kq(K_slm, ugemm_kq_wg_tile_m, Q + q0 * ldq,
                ldq, ugemm_kq_wg_tile_m, ugemm_kq_wg_tile_n, d, 0, 0, 0,
                sg_i_kq, sg_j_kq, (local char *)ugemm_slm);
#else
        s_tile_type S_tile;
#endif

        uint sg_i0_s2 = sg_i_kq * ugemm_kq_sg_tile_m + k0;
        uint sg_j0_s2 = sg_j_kq * ugemm_kq_sg_tile_n + q0;
        /* TODO: Apply attention mask */

        /* TODO: Apply k mask */
        //if (remainder_k) { tile_hbroadcast_min(&S_tile, k_mask); }
        /* TODO: causal masking fns???  */

        //TODO: global -> slm -> tile?
        s_sum_tile_type S_logsumexp_tile;
        //TODO: tile load (non-full) w/bounds check
        tile_load(&S_logsumexp_tile, ws_logsumexp, q, 1, ugemm_kq_wg_tile_n,
                sg_j0_kq + q0, 0);
#define mulscale(x) (x * scale)
        tile_elementwise(S_tile, mulscale);
        tile_vbroadcast_sub(&S_tile, S_logsumexp_tile);

/* Scale + exponentiate */
#define scaled_exp(x) native_vexp2(x * 1.44269504089f)
        tile_elementwise(S_tile, scaled_exp);

        s_tile_type_reblock S_tile_reblock;
        tile_copy_reblock(S_tile, &S_tile_reblock);
        uint sg_i0_ds = sg_i_kq * ugemm_kq_sg_tile_m;
        uint sg_j0_ds = sg_j_kq * ugemm_kq_sg_tile_n;

        //tile_store_full(S_tile, dS_slm, ugemm_kq_wg_tile_n, sg_j0_ds,
                //sg_i0_ds); //seems right, s2 16x32


#if USE_SYSTOLIC_UKERNEL
        tile_store_sys_src2(S_tile_reblock, (local FMA_TYPE *)S_slm,
                ugemm_vs_sg_tile_n, ugemm_kq_wg_tile_n, sg_i0_kq, sg_j0_kq);
#else

        tile_store_t_packed_src1(S_tile_reblock, S_slm, ugemm_kq_wg_tile_m,
                ugemm_kq_wg_tile_m, sg_i0_kq, sg_j0_kq); //which?

#endif

        if (q0end > 0) {
            // TODO: if() needed?
            q_tile_type
                    dA_tile; //TODO: convert dA_tile -> dA_tile1 : s_tile_type instead of double read
            uint q0_copy = q_tile_sg_n * sg_ij;

            tile_load_src1(&dA_tile, dA, d, q, lda, 0, q0 + q0_copy,
                    remainder_q);

            //tile_store_block_packed(dA_tile, Q_slm, ugemm_kq_sg_tile_n, ugemm_kq_wg_tile_n, 0, q0_copy);
            tile_store_block_packed(dA_tile, Q_slm, D_MAX, D_MAX, 0, q0_copy);
            //should this be verbatim? not T

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        //int k_chunk = min(k0end - k0, ugemm_kq_wg_tile_m); apply k chunk to tile load instead of ugemm_vs limit (pad w/0)
#if DO_MM
        // Q_slm -> D_MAX x wg_n;  S_slm wg_n x wg_m
        dv_tile_type dV_tile1 = ugemm_vs(Q_slm, D_MAX, S_slm,
                ugemm_kq_wg_tile_m, d, ugemm_kq_wg_tile_m, ugemm_kq_wg_tile_n,
                0, 0, 0, sg_i_vs, sg_j_vs, (local char *)ugemm_slm);
#else
        dv_tile_type dV_tile1;
#endif
        uint sg_i0_vs = sg_i_vs * ugemm_vs_sg_tile_m;
        uint sg_j0_vs = sg_j_vs * ugemm_vs_sg_tile_n;

        tile_binary(dV_tile, dV_tile1, binary_add);

        //slm dv tile
        //dv_tile_type dV_tile_slm;
        //tile_load_full(
            //&dV_tile_slm, dV_slm, ugemm_kq_wg_tile_m, sg_i0_vs, sg_j0_vs);
        //barrier(CLK_LOCAL_MEM_FENCE);
        //tile_binary(dV_tile_slm, dV_tile1, binary_add);
        //barrier(CLK_LOCAL_MEM_FENCE);
        //tile_store_full(dV_tile_slm, dV_slm, ugemm_kq_wg_tile_m, sg_i0_vs, sg_j0_vs);

        // Calculate D_i tile, TODO: this should be a separate kernel and replace calculation w/DRAM read
        p_sum_tile_type D_i;
        tile_fill(D_i, 0.0f);
        if (q0end > 0) {
            uint q0_copy = q_tile_sg_n * sg_ij;

            tile_load(&D_i, ws_Di, q0end, 1, q0end, q0 + sg_j0_kq,
                    0); // for vbroadcast
        }

#if DO_MM
        p_tile_type dP_tile = ugemm_vtdA(V + k0 * ldv, ldv, Q_slm, D_MAX,
                ugemm_kq_wg_tile_m, ugemm_kq_wg_tile_n, d, 0, 0, 0, sg_i_kq,
                sg_j_kq, (local char *)ugemm_slm);
#else
        p_tile_type dP_tile;
#endif
        tile_vbroadcast_sub(&dP_tile,
                D_i); // needs output to be transposed from vtdA layout.C = N

        // reload since too many registers used
        p_tile_type S2_tile;
        //tile_load_full(&S2_tile, dS_slm, ugemm_kq_wg_tile_n,
                //sg_j_kq * ugemm_vtdA_sg_tile_n, sg_i_kq * ugemm_vtdA_sg_tile_m);

        tile_load_t_full(&S2_tile, S_slm, ugemm_kq_wg_tile_m,
                sg_j_kq * ugemm_vtdA_sg_tile_n, sg_i_kq * ugemm_vtdA_sg_tile_m);
        barrier(CLK_LOCAL_MEM_FENCE);
#define binary_mul(x, y) ((x) * (y))
        tile_binary(dP_tile, S2_tile, binary_mul); // is this right?

        p_tile_type_reblock P_tile_reblock;
        tile_copy_reblock(dP_tile, &P_tile_reblock);

        // SLM for dK = dS^t * Q
        uint q0_copy = q_tile_sg_n * sg_ij;

        tile_store_block_packed(P_tile_reblock, Q_slm, ugemm_kq_wg_tile_m,
                ugemm_kq_wg_tile_m, sg_j0_kq, sg_i0_kq);
        barrier(CLK_LOCAL_MEM_FENCE);

        // dK = dS^t * Q
#if DO_MM
        // TODO: cache Q tile instead of re-reading from glm
        a_tile_type dK_tile1 = ugemm_qdSt(Q + q0 * ldq, ldq, Q_slm,
                ugemm_kq_wg_tile_m, //layout.N
                d, ugemm_kq_wg_tile_m, ugemm_kq_wg_tile_n, 0, 0, 0, sg_i_qdSt,
                sg_j_qdSt, (local char *)ugemm_slm);
#else
        a_tile_type dK_tile1;
#endif
        barrier(CLK_LOCAL_MEM_FENCE);

        tile_binary(dK_tile, dK_tile1, binary_add);

        uint sg_i0_dk = sg_i_qdSt * ugemm_qdSt_sg_tile_m;
        uint sg_j0_dk = sg_j_qdSt * ugemm_qdSt_sg_tile_n;

        //// dk slm tile
        //a_tile_type dK_tile_slm;
        //tile_load_full(
            //&dK_tile_slm, dK_slm, ugemm_kq_wg_tile_m, sg_j0_dk, sg_i0_dk);
        //tile_binary(dK_tile_slm, dK_tile1, binary_add);
        //tile_store_full(dK_tile_slm, dK_slm, ugemm_kq_wg_tile_m, sg_j0_dk, sg_i0_dk);

        // dQ = dS * K
#if DO_MM
        ktq_tile_type dQ_tile;
            dQ_tile = ugemm_ktq(K + k0, ldk, Q_slm,
                    ugemm_kq_wg_tile_m, d, ugemm_kq_wg_tile_n, ugemm_kq_wg_tile_m,
                    0, 0, 0, sg_i_ktq, sg_j_ktq, (local char *)ugemm_slm);
#else
        ktq_tile_type dQ_tile;
#endif
        uint sg_i0_dq = sg_i_ktq * ugemm_ktq_sg_tile_m; //todo: needs +k0 ??
        uint sg_j0_dq = sg_j_ktq * ugemm_ktq_sg_tile_n;

        tile_atomic_add(dQ_tile, dQ + q0 * ldq, d, q, ldq, sg_i0_dq, sg_j0_dq);
    }

    //TODO: wg_i0 = k0??

    //   // update dV
    uint sg_i0_vs = sg_i_vs * ugemm_vs_sg_tile_m;
    uint sg_j0_vs = sg_j_vs * ugemm_vs_sg_tile_n;

    //dv_tile_type dV_tile_slm;
    //barrier(CLK_LOCAL_MEM_FENCE);
    //tile_load_full(
        //&dV_tile_slm, dV_slm, ugemm_kq_wg_tile_m, sg_i0_vs, sg_j0_vs);
    //tile_store(dV_tile_slm, dV + wg_i0 * ldv, d, q, ldv, sg_i0_vs, sg_j0_vs);

    tile_store(dV_tile, dV + wg_i0 * ldv, d, q, ldv, sg_i0_vs, sg_j0_vs);

    // /update dV

    // update dK
    uint sg_i0_dk = sg_i_qdSt * ugemm_qdSt_sg_tile_m;
    uint sg_j0_dk = sg_j_qdSt * ugemm_qdSt_sg_tile_n;

    //a_tile_type dK_tile_slm;
    //tile_load_full(
        //&dK_tile_slm, dK_slm, ugemm_kq_wg_tile_m, sg_j0_dk, sg_i0_dk);
    //tile_store(dK_tile_slm, dK + wg_i0, k, d, ldk, sg_j0_dk, sg_i0_dk);

    tile_store(dK_tile, dK + wg_i0, k, d, ldk, sg_j0_dk, sg_i0_dk);
}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) kernel void
preprocess_Di(const global KEY_DATA_T *K, const global QRY_DATA_T *Q,
        const global VAL_DATA_T *V, global float *ws,
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

    uint lda = DST_S2;
    uint ldq = QRY_S2;

    // TODO: change to q?
    uint wg_q = get_group_id(0);

    uint sg_ij = sub_group_broadcast(get_local_id(1), 0);
    uint sg_i_kq = sg_ij % ugemm_kq_sg_per_wg_m;
    uint sg_j_kq = sg_ij / ugemm_kq_sg_per_wg_m;

    uint b0, b1;
    b0 = get_group_id(1);
    b1 = get_group_id(2);

    const uint preprocess_batch = b1 * (DST_D1 * 2 * q) + b0 * (2 * q);

    const size_t q_offset = QRY_BATCH(b1, b0);
    const size_t a_offset = DST_BATCH(b1, b0);

    /* Locate dQ/A/dA matrices within batch */
    dQ += q_offset;

    A += a_offset;
    dA += a_offset;

    global float *ws_Di = ws + q;
    ws_Di += preprocess_batch;


    uint wg_j0 = wg_q * ugemm_kq_wg_tile_n;

#define Di_slm_size (ugemm_kq_wg_tile_n * sizeof(VAL_DATA_T))
    local char slm[Di_slm_size];

    local VAL_DATA_T *Di_slm = (local VAL_DATA_T *)&slm[0];

    uint sg_i0_kq = sg_i_kq * ugemm_kq_sg_tile_m;
    uint sg_j0_kq = sg_j_kq * ugemm_kq_sg_tile_n;

    //s_sum_tile_type D_i;
    //tile_fill(D_i, 0.0f);
    if (q > 0) {
        q_tile_type dA_tile1, A_tile; // for D_i calculation
        q_tile_type zero_dQ_tile; // zeroing dQ for atomics
        tile_fill(zero_dQ_tile, 0.f);

        uint q0_copy = q_tile_sg_n * sg_ij;

        tile_store(zero_dQ_tile, (global FMA_TYPE *)dQ, d, q, ldq, 0,
                wg_j0 + q0_copy);
        // printf("wg_m,n %d,%d  sg_ij %d, sg_i,j_kq %d,%d q0_copy%d \n", ugemm_kq_wg_tile_m, ugemm_kq_wg_tile_n, sg_ij, sg_i_kq, sg_j_kq, q0_copy);

        // TODO: fixtype, load dA_type
        tile_load(&dA_tile1, (global FMA_TYPE *)dA, d, q, lda, 0,
                wg_j0 + q0_copy);
        tile_load(&A_tile, (global FMA_TYPE *)A, d, q, lda, 0, wg_j0 + q0_copy);
#define binary_mul(x, y) ((x) * (y))
        tile_binary(A_tile, dA_tile1, binary_mul);

        for (int j = 0; j < q_tile_sg_n; j++) {
            float r = 0.f;
            for (int i0 = 0; i0 < D_MAX; i0 += SUBGROUP_SIZE) {
                r += sub_group_reduce_add(
                        tile_access(A_tile, i0, j, SUBGROUP_SIZE, D_MAX, 1, 1));
            }
            Di_slm[j + q0_copy] = r;
        }
        barrier(CLK_LOCAL_MEM_FENCE); //unneeded, no sharing, only caching
        // tile_load_full(&D_i, Di_slm, ugemm_kq_wg_tile_n, q0_copy, 0);

        for (int i = get_local_id(0); i < ugemm_kq_wg_tile_n;
                i += get_local_size(0)) {
            if (get_local_id(1) == 0) { ws_Di[wg_j0 + i] = Di_slm[i]; }
        }
        //tile_store(D_i, ws_Di, q, 1, q, wg_j0 + q0_copy,
        //0); //should be 1d write //TODO:wg_j correct?
    }
}
