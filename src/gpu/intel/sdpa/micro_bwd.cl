/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#define sg_per_wg_BcBr \
    (ugemm_kq_sg_per_wg_m * ugemm_kq_sg_per_wg_n) // same for kq, vtdA
#define sg_per_wg_BcD \
    (ugemm_vs_sg_per_wg_m * ugemm_vs_sg_per_wg_n) // same for qdSt and vs
#define sg_per_wg_BrD (ugemm_ktq_sg_per_wg_m * ugemm_ktq_sg_per_wg_n)
#define sg_per_wg MAX(sg_per_wg_BcBr, MAX(sg_per_wg_BcD, sg_per_wg_BrD))

#define q_tile_sg_n DIV_UP(ugemm_kq_wg_tile_n, sg_per_wg)
#define dmax_tile_sg_n DIV_UP(D_MAX, sg_per_wg)

/* Instantiate tile types and operations */
typedef ugemm_kq_c_type s_tile_type; // Bc*Br tile
typedef ugemm_qdSt_c_type a_tile_type; // Bc*D tile
typedef ugemm_vtdA_c_type p_tile_type; // Br*Bc tile (.T)
typedef ugemm_vs_c_type dv_tile_type; // D*Bc tile
typedef ugemm_ktq_c_type ktq_tile_type; // D*Br tile

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

DECLARE_2D_TILE(q_tile_type, FMA_TYPE, SUBGROUP_SIZE, D_MAX, 1, 1, q_tile_sg_n)

DECLARE_2D_TILE(dq_tile_type, float, SUBGROUP_SIZE, D_MAX, 1, 1, q_tile_sg_n)
DECLARE_2D_TILE_BLOCK_OPS(
        dq_tile_type, float, SUBGROUP_SIZE, D_MAX, 1, 1, q_tile_sg_n)
DECLARE_2D_TILE_COPY_REBLOCK(q_tile_type, SUBGROUP_SIZE, D_MAX, 1, 1,
        q_tile_sg_n, dq_tile_type, SUBGROUP_SIZE, D_MAX, 1, 1, q_tile_sg_n,
        CONVERT_FLOAT_T)

DECLARE_2D_TILE(k_tile_type, FMA_TYPE, SUBGROUP_SIZE, ugemm_kq_wg_tile_m, 1, 1,
        dmax_tile_sg_n)
#if BLOCK_K
DECLARE_2D_TILE_BLOCK_OPS(k_tile_type, FMA_TYPE, SUBGROUP_SIZE,
        ugemm_kq_wg_tile_m, 1, 1, dmax_tile_sg_n)
#endif

DECLARE_2D_TILE(s_tile_type_packed, uint, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1 / 2, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1)
DECLARE_2D_TILE(s_tile_type_packed_t, uint, SUBGROUP_SIZE,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_block0 / 2,
        ugemm_kq_c_type_nblock1, ugemm_kq_c_type_nblock0)

DECLARE_2D_TILE(s_tile_type_reblock, FMA_TYPE, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_m, 1, 1, ugemm_kq_sg_tile_n)
DECLARE_2D_TILE_BLOCK_OPS(s_tile_type_reblock, FMA_TYPE, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_m, 1, 1, ugemm_kq_sg_tile_n)

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

DECLARE_2D_TILE(qmask_tile_type_float, float, SUBGROUP_SIZE, ugemm_kq_sg_tile_n,
        1, 1, 1)
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

DECLARE_2D_TILE(a_tile_type_dst, DST_DATA_T, SUBGROUP_SIZE,
        ugemm_qdSt_sg_tile_m, 1, 1, ugemm_qdSt_sg_tile_n)
DECLARE_2D_TILE_BLOCK_OPS(a_tile_type_dst, DST_DATA_T, SUBGROUP_SIZE,
        ugemm_qdSt_sg_tile_m, 1, 1, ugemm_qdSt_sg_tile_n)
DECLARE_2D_TILE_COPY_REBLOCK(a_tile_type, SUBGROUP_SIZE,
        ugemm_qdSt_c_type_block0, ugemm_qdSt_c_type_block1,
        ugemm_qdSt_c_type_nblock0, ugemm_qdSt_c_type_nblock1, a_tile_type_dst,
        SUBGROUP_SIZE, ugemm_qdSt_sg_tile_m, 1, 1, ugemm_qdSt_sg_tile_n,
        CONVERT_DATA_T)

DECLARE_2D_TILE(dv_tile_type_dst, DST_DATA_T, SUBGROUP_SIZE, ugemm_vs_sg_tile_m,
        1, 1, ugemm_vs_sg_tile_n)
DECLARE_2D_TILE_BLOCK_OPS(dv_tile_type_dst, DST_DATA_T, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_m, 1, 1, ugemm_vs_sg_tile_n)
DECLARE_2D_TILE_COPY_REBLOCK(dv_tile_type, SUBGROUP_SIZE,
        ugemm_vs_c_type_block0, ugemm_vs_c_type_block1, ugemm_vs_c_type_nblock0,
        ugemm_vs_c_type_nblock1, dv_tile_type_dst, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_m, 1, 1, ugemm_vs_sg_tile_n, CONVERT_DATA_T)

DECLARE_2D_TILE(dq_tile_type_dst, DST_DATA_T, SUBGROUP_SIZE,
        ugemm_ktq_sg_tile_m, 1, 1, ugemm_ktq_sg_tile_n)
DECLARE_2D_TILE_BLOCK_OPS(dq_tile_type_dst, DST_DATA_T, SUBGROUP_SIZE,
        ugemm_ktq_sg_tile_m, 1, 1, ugemm_ktq_sg_tile_n)
DECLARE_2D_TILE_COPY_REBLOCK(ktq_tile_type, SUBGROUP_SIZE,
        ugemm_ktq_c_type_block0, ugemm_ktq_c_type_block1,
        ugemm_ktq_c_type_nblock0, ugemm_ktq_c_type_nblock1, dq_tile_type_dst,
        SUBGROUP_SIZE, ugemm_ktq_sg_tile_m, 1, 1, ugemm_ktq_sg_tile_n,
        CONVERT_DATA_T)

DECLARE_2D_TILE_COPY_REBLOCK(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, s_tile_type_reblock, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_m, 1, 1, ugemm_kq_sg_tile_n, CONVERT_DATA_T)
DECLARE_2D_TILE_COPY_REBLOCK(p_tile_type, SUBGROUP_SIZE,
        ugemm_vtdA_c_type_block0, ugemm_vtdA_c_type_block1,
        ugemm_vtdA_c_type_nblock0, ugemm_vtdA_c_type_nblock1,
        p_tile_type_reblock, SUBGROUP_SIZE, ugemm_vtdA_c_type_block0, 1,
        ugemm_vtdA_c_type_nblock0,
        ugemm_vtdA_c_type_block1 *ugemm_vtdA_c_type_nblock1, CONVERT_DATA_T)
DECLARE_2D_TILE_COPY_REBLOCK(p_tile_type_reblock, SUBGROUP_SIZE,
        ugemm_vtdA_c_type_block0, 1, ugemm_vtdA_c_type_nblock0,
        ugemm_vtdA_c_type_block1 *ugemm_vtdA_c_type_nblock1, p_tile_type,
        SUBGROUP_SIZE, ugemm_vtdA_c_type_block0, ugemm_vtdA_c_type_block1,
        ugemm_vtdA_c_type_nblock0, ugemm_vtdA_c_type_nblock1, CONVERT_FLOAT_T)

DECLARE_2D_TILE_VREDUCE(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, s_sum_tile_type, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_n, 1, 1, 1)
DECLARE_2D_TILE_VREDUCE(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, qmask_tile_type_float, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_n, 1, 1, 1)
DECLARE_2D_TILE_HREDUCE(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, qmask_tile_type_float, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_n, 1, 1, 1)
DECLARE_2D_TILE_HREDUCE(p_tile_type, SUBGROUP_SIZE, ugemm_vtdA_c_type_block0,
        ugemm_vtdA_c_type_block1, ugemm_vtdA_c_type_nblock0,
        ugemm_vtdA_c_type_nblock1, kmask_tile_type_float, SUBGROUP_SIZE,
        ugemm_vtdA_sg_tile_m, 1, 1, 1)
DECLARE_2D_TILE_VREDUCE(p_tile_type, SUBGROUP_SIZE, ugemm_vtdA_c_type_block0,
        ugemm_vtdA_c_type_block1, ugemm_vtdA_c_type_nblock0,
        ugemm_vtdA_c_type_nblock1, kmask_tile_type_float, SUBGROUP_SIZE,
        ugemm_vtdA_sg_tile_m, 1, 1, 1)

DECLARE_2D_TILE_HREDUCE(p_tile_type, SUBGROUP_SIZE, ugemm_vtdA_c_type_block0,
        ugemm_vtdA_c_type_block1, ugemm_vtdA_c_type_nblock0,
        ugemm_vtdA_c_type_nblock1, p_sum_tile_type, SUBGROUP_SIZE,
        ugemm_vtdA_sg_tile_n, 1, 1, 1)

DECLARE_2D_TILE_HREDUCE(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, s_sum_tile_type, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_n, 1, 1, 1)
#if WITH_ATTN_MASK
DECLARE_2D_TILE_VREDUCE(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, mask_tile_type_float, SUBGROUP_SIZE, mask_br,
        mask_bc, mask_nbr, mask_nbc)
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

inline void tile_load_k(k_tile_type *K_tile, const global KEY_DATA_T *K, int m,
        int n, int ldk, int offset_r, int offset_c, int load_rem) {
#if BLOCK_K
    // can ignore load_rem due to d_full requirement
    tile_load_block(K_tile, K, ldk, offset_r, offset_c);
#else
    tile_load(K_tile, K, m, n, ldk, offset_r, offset_c);
#endif
}

#if KV_GROUP_SIZE > 1
#define DST_DATA_T_DKDV float
#else
#define DST_DATA_T_DKDV DST_DATA_T
#endif

inline void tile_store_dV(dv_tile_type *dV_tile_slm, global DST_DATA_T_DKDV *dV,
        int m, int n, int ld, int offset_r, int offset_c, int rem) {

#if KV_GROUP_SIZE > 1 // GQA update
    tile_atomic_add(*dV_tile_slm, dV, m, n, ld, offset_r, offset_c);
#else // MHA update

    dv_tile_type_dst dV_tile_dst; // convert to half
    tile_copy_reblock(*dV_tile_slm, &dV_tile_dst);
#if BLOCK_DV
    tile_store_block_rem_q(dV_tile_dst, dV, n, ld, offset_r, offset_c, rem)
#else
    tile_store(dV_tile_dst, dV, m, n, ld, offset_r, offset_c);
#endif

#endif
}

inline void tile_store_dK(a_tile_type *dK_tile_slm, global DST_DATA_T_DKDV *dK,
        int m, int n, int ld, int offset_r, int offset_c) {

#if KV_GROUP_SIZE > 1 // GQA update
    tile_atomic_add(*dK_tile_slm, dK, m, n, ld, offset_r, offset_c);
#else // MHA update

    a_tile_type_dst dK_tile_dst; // convert to half
    tile_copy_reblock(*dK_tile_slm, &dK_tile_dst);
#if BLOCK_DK
    tile_store_block(dK_tile_dst, dK, ld, offset_r, offset_c);
#else
    tile_store(dK_tile_dst, dK, m, n, ld, offset_r, offset_c);
#endif

#endif
}

#define DO_MM 1

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) kernel void
micro_sdpa_bwd(const global KEY_DATA_T *K, const global QRY_DATA_T *Q,
        const global VAL_DATA_T *V, const global float *ws,
        const global float *Di, const global DST_DATA_T *A,
        const global DST_DATA_T *dA,
#if WITH_DS
        global DST_DATA_T *dS, // expensive, optional intermediate
#endif
        global DST_DATA_T_DKDV *dK, global float *dQ,
        global DST_DATA_T_DKDV *dV,
#if WITH_HOST_SCALE
        float scalar_scale, float inv_scalar_scale,
#else
        const global SCALE_DATA_T *scale_ptr,
#endif
        int d, int k, int q, const int attn_mask_type
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

    uint b1 = get_group_id(2);

    // TODO: batch q=1 cases to KV_GROUP_SIZE
    uint b0, b0_kv;
    b0 = get_group_id(1);
    b0_kv = b0 / KV_GROUP_SIZE;

    uint wg_i0 = wg_k * ugemm_kq_wg_tile_m;

    const uint preprocess_batch = b1 * (DST_D1 * q) + b0 * q;
    const global float *ws_logsumexp = ws + preprocess_batch;
    Di += preprocess_batch;

    /* Calculate the number of keys to process */
    int q0end = q;
    int qdiag0 = 0; // potentially offset starting idx in causal mask cases
#if WITH_CAUSAL_MASK
    if (attn_mask_type == ATTN_MASK_TOP_LEFT) {
        qdiag0 = max(0, (int)(wg_i0));
    } else {
        qdiag0 = max(0, (int)(wg_i0 + (q - k)));
    }
#endif

    /* Leading dimension for matrices */
    uint ldk = TRANSPOSE_K ? KEY_S3 : KEY_S2;
    uint ldq = QRY_S2;
    uint ldv = VAL_S2;
    uint lda = DST_S2;

    /* Subgroup IDs for each GEMM, although total number of
     * sg per wg may be shared
     * ordering may differ due to transposes */
    uint sg_i_kq = sg_ij % ugemm_kq_sg_per_wg_m;
    uint sg_j_kq = sg_ij / ugemm_kq_sg_per_wg_m;

    uint sg_i_vtdA = sg_ij % ugemm_vtdA_sg_per_wg_m;
    uint sg_j_vtdA = sg_ij / ugemm_vtdA_sg_per_wg_m;

    uint sg_i_vs = sg_ij % ugemm_vs_sg_per_wg_m;
    uint sg_j_vs = sg_ij / ugemm_vs_sg_per_wg_m;

    uint sg_i_qdSt = sg_ij % ugemm_qdSt_sg_per_wg_m;
    uint sg_j_qdSt = sg_ij / ugemm_qdSt_sg_per_wg_m;

    uint sg_i_ktq = sg_ij % ugemm_ktq_sg_per_wg_m;
    uint sg_j_ktq = sg_ij / ugemm_ktq_sg_per_wg_m;

    /* SLM allocations -- place in one array to work around compiler bug */
#define K_slm_size (ugemm_kq_wg_tile_m * D_MAX * sizeof(KEY_DATA_T))
#define S_slm_size (ugemm_kq_wg_tile_m * ugemm_kq_wg_tile_n * sizeof(FMA_TYPE))

#define dK_slm_size (ugemm_kq_wg_tile_m * D_MAX * sizeof(float))
#define dV_slm_size (ugemm_kq_wg_tile_m * D_MAX * sizeof(float))

#define ugemm_slm_size \
    MAX(MAX(MAX(MAX(ugemm_kq_slm_size, ugemm_vs_slm_size), \
                    ugemm_vtdA_slm_size), \
                ugemm_qdSt_slm_size), \
            ugemm_ktq_slm_size)

    local char slm[K_slm_size + S_slm_size + ugemm_slm_size + dK_slm_size
            + dV_slm_size];

    local KEY_DATA_T *K_slm = (local KEY_DATA_T *)&slm[0];

    // used for caching various A,B gemm tiles
    local FMA_TYPE *S_slm = (local FMA_TYPE *)&slm[K_slm_size];

    // ugemm scratch space
    local uint *ugemm_slm = (local uint *)&slm[K_slm_size + S_slm_size];

    // used for accumulation of dV, dK across q-loop
    local float *dK_slm
            = (local float *)&slm[K_slm_size + S_slm_size + ugemm_slm_size];
    local float *dV_slm = (local float *)&slm[K_slm_size + S_slm_size
            + ugemm_slm_size + dK_slm_size];

    const size_t k_offset = KEY_BATCH(b1, b0_kv);
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

#if WITH_DS
    dS += b1 * (DST_D1 * q * k) + b0 * (q * k);
#endif

#if WITH_ATTN_MASK
    msk += MSK_BATCH(b1 % MSK_D0, b0 % MSK_D1);
    int mask_aligned = (((size_t)msk) % 4) == 0;
    bool block_msk = (b1 < MSK_D0 - ceil((float)ugemm_kq_wg_tile_m / MSK_S2))
            && mask_aligned;
#endif

    if (qdiag0 < q0end) {
        /* Load K tile, destined for SLM */

        k_tile_type K_tile;
        tile_fill(K_tile, TO_DATA_T(0.f));

        uint k0_copy = dmax_tile_sg_n
                * sg_ij; //each sg will be responsible for dmax_tile_sg_n columns
        tile_load_k(&K_tile, K, k, d, ldk, wg_i0, k0_copy, remainder_k);
        ///* Store K tile to SLM */
#if USE_SYSTOLIC_UKERNEL
        tile_store_sys_src1(K_tile, &K_slm[0], SUBGROUP_SIZE, D_MAX,
                ugemm_kq_wg_tile_m, D_MAX, 0, k0_copy);
#else
        tile_store_packed_src1(
                K_tile, K_slm, ugemm_kq_sg_tile_m, D_MAX, 0, k0_copy);
#endif
    }

    /* Load scale */
    float scale = 1.f;
    float iscale = 1.f;
    if (qdiag0 < q0end) {
#if WITH_ATTN_SCALE
#if WITH_HOST_SCALE
#if INVERT_SCALE
        iscale = scalar_scale;
        scale = inv_scalar_scale;
#else
        scale = scalar_scale;
        iscale = inv_scalar_scale;
#endif
#else
#if INVERT_SCALE
        iscale = SCALES_TO_FLOAT(*scale_ptr);
        scale = native_recip(iscale);
#else
        scale = SCALES_TO_FLOAT(*scale_ptr);
        iscale = native_recip(scale);
#endif
#endif
#endif
    }

    /* Initialize dV, dK to zero */
#pragma unroll
    for (int i = get_local_id(0); i < ugemm_kq_wg_tile_m * D_MAX;
            i += get_local_size(0)) {
        dK_slm[i] = 0.f;
        dV_slm[i] = 0.f;
    }

    uint sg_i0_kq = sg_i_kq * ugemm_kq_sg_tile_m;
    uint sg_j0_kq = sg_j_kq * ugemm_kq_sg_tile_n;

    const int k0 = wg_i0;

    // make sure K_tile in SLM
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Main loop over k blocks */
    for (int q0 = qdiag0; q0 < q0end; q0 += ugemm_kq_wg_tile_n) {

        const bool first = (q0 == qdiag0);
        const int qnext = q0 + ugemm_kq_wg_tile_n;
        const bool last = (qnext >= q0end);

        qmask_tile_type_float q_mask;
        kmask_tile_type_float k_mask;

        int k_chunk = min(k - k0, ugemm_kq_wg_tile_m);
        int q_nchunk = min(q0end - q0, ugemm_kq_wg_tile_n);
        /* Calculate S = (K^T) * Q */
#if DO_MM
        s_tile_type S_tile
                = ugemm_kq(K_slm, D_MAX, Q + q0 * ldq, ldq, k_chunk, q_nchunk,
                        d, 0, 0, 0, sg_i_kq, sg_j_kq, (local char *)ugemm_slm);
#else
        s_tile_type S_tile;
#endif

        uint sg_i0_s2 = sg_i_kq * ugemm_kq_sg_tile_m + k0;
        uint sg_j0_s2 = sg_j_kq * ugemm_kq_sg_tile_n + q0;

        /* Apply attention mask */
#if WITH_ATTN_MASK
        mask_tile_type mask_tile;
#if BROADCAST_MASK_Q
        if (block_msk) {
            tile_load_block(&mask_tile, msk, MSK_S2, 0, k0 + sg_i0_kq, 0);
        } else {
            tile_load(&mask_tile, msk, k, 1, MSK_S2, k0 + sg_i0_kq, 0);
        }
#else
        tile_load(&mask_tile, msk, k, q, MSK_S2, k0 + sg_i0_kq, q0 + sg_j0_kq);
#endif

#define unscale(x) ((x) * iscale)
        mask_tile_type_float mask_tile_float;
        tile_copy_reblock(mask_tile, &mask_tile_float);
#if WITH_ATTN_SCALE
        tile_elementwise(mask_tile_float, unscale);
#endif
#undef unscale
#if BROADCAST_MASK_Q
        tile_vbroadcast_add(&S_tile, mask_tile_float);
#else
        tile_binary(S_tile, mask_tile_float, binary_add);
#endif
#endif

        /* Apply q mask */
        if (remainder_q) {
#pragma unroll
            for (int jj = get_sub_group_local_id(); jj < ugemm_kq_sg_tile_n;
                    jj += SUBGROUP_SIZE) {
                q_mask.x[0][jj / SUBGROUP_SIZE]
                        = ((q0 + sg_j0_kq + jj) < q0end) ? nan(0u) : -INFINITY;
            }
            tile_hbroadcast_min(&S_tile, q_mask);
        }

#if WITH_CAUSAL_MASK
#define less_than(offset_k, offset_q) (offset_q < offset_k)

        int col_offset = q0 + sg_j0_kq;
        if (q == 1) col_offset = 1;
        if (attn_mask_type == ATTN_MASK_BOTTOM_RIGHT) col_offset += k - q;

        /* Apply causal mask */
        const bool is_diag = (q0
                == qdiag0); // first iteration will be on diagonal, requiring partial masking
        if (is_diag) {
            tile_predicated_assignment(S_tile, k0 + sg_i0_kq, col_offset,
                    less_than, -INFINITY, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
                    ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
                    ugemm_kq_c_type_nblock1);
        }
#undef less_than
#endif

        s_sum_tile_type S_logsumexp_tile;
        tile_fill(S_logsumexp_tile, 0.f);
        tile_load(&S_logsumexp_tile, ws_logsumexp, q, 1, ugemm_kq_wg_tile_n,
                sg_j0_kq + q0, 0);
#define mulscale(x) (x * scale)
        tile_elementwise(S_tile, mulscale);
#undef mulscale
        tile_hbroadcast_sub(&S_tile, S_logsumexp_tile); //layout.N
        //tile_vbroadcast_sub(&S_tile, S_logsumexp_tile); //layout.T

/* Scale + exponentiate */
#define scaled_exp(x) native_vexp2(x * 1.44269504089f)
        tile_elementwise(S_tile, scaled_exp);
#undef scaled_exp

        s_tile_type_reblock S_tile_reblock;
        tile_copy_reblock(S_tile, &S_tile_reblock);
        uint sg_i0_ds = sg_i_kq * ugemm_kq_sg_tile_m;
        uint sg_j0_ds = sg_j_kq * ugemm_kq_sg_tile_n;

        barrier(CLK_LOCAL_MEM_FENCE);
#if USE_SYSTOLIC_UKERNEL
        tile_store_t_sys_src22(S_tile_reblock, (local FMA_TYPE *)S_slm,
                ugemm_vs_sg_tile_n, ugemm_kq_wg_tile_m, ugemm_kq_wg_tile_n,
                sg_i0_kq, sg_j0_kq);
#else
        tile_store_packed_src1(S_tile_reblock, S_slm, ugemm_vs_sg_tile_n,
                ugemm_kq_wg_tile_n, sg_i0_kq, sg_j0_kq);
#endif
        barrier(CLK_LOCAL_MEM_FENCE);

#if DO_MM
        dv_tile_type dV_tile1;
        dV_tile1 = ugemm_vs(dA + q0 * lda, lda, (local FMA_TYPE *)S_slm,
                ugemm_kq_wg_tile_n, d, k_chunk, q_nchunk, 0, 0, 0, sg_i_vs,
                sg_j_vs, (local char *)ugemm_slm);
#else
        dv_tile_type dV_tile1;
#endif
        uint sg_i0_vs = sg_i_vs * ugemm_vs_sg_tile_m;
        uint sg_j0_vs = sg_j_vs * ugemm_vs_sg_tile_n;

        //slm dv tile
        dv_tile_type dV_tile_slm;
        tile_load(&dV_tile_slm, dV_slm, D_MAX, ugemm_kq_wg_tile_m, D_MAX,
                sg_i0_vs, sg_j0_vs);
        tile_binary(dV_tile_slm, dV_tile1, binary_add);
        tile_store(dV_tile_slm, dV_slm, D_MAX, ugemm_kq_wg_tile_m, D_MAX,
                sg_i0_vs, sg_j0_vs);

#if DO_MM
        p_tile_type dP_tile = ugemm_vtdA(V + k0 * ldv, ldv, dA + q0 * lda, lda,
                k_chunk, q_nchunk, d, 0, 0, 0, sg_i_kq, sg_j_kq,
                (local char *)ugemm_slm);
#else
        p_tile_type dP_tile;
#endif

        // get D_i tile
        p_sum_tile_type D_i;
        tile_fill(D_i, 0.0f);
        tile_load(&D_i, Di, q0end, 1, q0end, q0 + sg_j0_kq, 0);

        tile_hbroadcast_sub(&dP_tile,
                D_i); // needs output to be transposed from vtdA layout.C = N

        // reload softmax from SLM since ugemm_vtdA() will clobber registers
        p_tile_type S2_tile;
        p_tile_type_reblock S2_tile_reblock;

#if USE_SYSTOLIC_UKERNEL
        tile_load_t_sys_src2(&S2_tile_reblock, (local FMA_TYPE *)S_slm,
                ugemm_vs_sg_tile_n, ugemm_kq_wg_tile_n, sg_j0_kq, sg_i0_kq);
#else
        tile_load_packed_src1(&S2_tile_reblock, S_slm, ugemm_vs_sg_tile_n,
                ugemm_kq_wg_tile_n, sg_i0_kq, sg_j0_kq);
#endif
        tile_copy_reblock(S2_tile_reblock, &S2_tile);

#define binary_mul_scale(x, y) ((x) * (y) * scale)
        tile_binary(dP_tile, S2_tile, binary_mul_scale);

        if (remainder_k) {
#pragma unroll
            for (int ii = 0; ii < ugemm_kq_sg_tile_m / SUBGROUP_SIZE; ii++) {
                k_mask.x[0][ii] = (k0 + sg_i0_kq + ii * SUBGROUP_SIZE
                                                  + get_sub_group_local_id()
                                          < k)
                        ? 1
                        : 0;
            }
            tile_vbroadcast_mul(&dP_tile, k_mask);
        }

        p_tile_type_reblock P_tile_reblock;
        tile_copy_reblock(dP_tile, &P_tile_reblock);
#if WITH_DS
        tile_store(P_tile_reblock, dS, k_chunk, q_nchunk, k, k0 + sg_i0_kq,
                q0 + sg_j0_kq);
#endif

        // SLM for dK = dS^t * Q
        local FMA_TYPE *dS_slm = (local FMA_TYPE *)S_slm;
        barrier(CLK_LOCAL_MEM_FENCE);
#if USE_SYSTOLIC_UKERNEL
        tile_store_sys_src1(P_tile_reblock, dS_slm, SUBGROUP_SIZE,
                ugemm_kq_wg_tile_n, ugemm_kq_wg_tile_m, ugemm_kq_wg_tile_n,
                sg_i0_kq, sg_j0_kq);
#else
        tile_store_packed_src1(P_tile_reblock, dS_slm, ugemm_qdSt_sg_tile_m,
                ugemm_kq_wg_tile_n, sg_i0_kq, sg_j0_kq);
#endif
        barrier(CLK_LOCAL_MEM_FENCE);

#if DO_MM
        a_tile_type dK_tile1;
        dK_tile1 = ugemm_qdSt(dS_slm, ugemm_kq_wg_tile_n, Q + q0 * ldq, ldq,
                k_chunk, d, q_nchunk, 0, 0, 0, sg_i_qdSt, sg_j_qdSt,
                (local char *)ugemm_slm); // dS^t * Q -> Bc x d
#else
        a_tile_type dK_tile1;
#endif
        uint sg_i0_dk = sg_i_qdSt * ugemm_qdSt_sg_tile_m;
        uint sg_j0_dk = sg_j_qdSt * ugemm_qdSt_sg_tile_n;

        //// dk slm tile
        a_tile_type dK_tile_slm;
        tile_load(&dK_tile_slm, dK_slm, ugemm_kq_wg_tile_m, D_MAX,
                ugemm_kq_wg_tile_m, sg_i0_dk, sg_j0_dk);
        tile_binary(dK_tile_slm, dK_tile1, binary_add);
        tile_store(dK_tile_slm, dK_slm, ugemm_kq_wg_tile_m, D_MAX,
                ugemm_kq_wg_tile_m, sg_i0_dk, sg_j0_dk);

        p_tile_type_reblock dS_transpose_tile;
#if USE_SYSTOLIC_UKERNEL
        tile_load_sys_src1(&dS_transpose_tile, dS_slm, SUBGROUP_SIZE,
                ugemm_kq_wg_tile_n, ugemm_kq_wg_tile_m, ugemm_kq_wg_tile_n,
                sg_i0_kq, sg_j0_kq);
        barrier(CLK_LOCAL_MEM_FENCE);
        tile_store_sys_src22(dS_transpose_tile, dS_slm, ugemm_ktq_sg_tile_n,
                ugemm_kq_wg_tile_m, ugemm_kq_wg_tile_n, sg_i0_kq, sg_j0_kq);
#else
        tile_load_packed_src1(&dS_transpose_tile, dS_slm, ugemm_qdSt_sg_tile_m,
                ugemm_kq_wg_tile_n, sg_i0_kq, sg_j0_kq);
        barrier(CLK_LOCAL_MEM_FENCE);
        tile_store_t_packed_src1(dS_transpose_tile, dS_slm, ugemm_ktq_sg_tile_n,
                ugemm_kq_wg_tile_m, sg_j0_kq, sg_i0_kq);
#endif
        barrier(CLK_LOCAL_MEM_FENCE);

        // dQ = dS * K
#if DO_MM
        ktq_tile_type dQ_tile;
        dQ_tile = ugemm_ktq(K + k0, ldk, dS_slm, ugemm_kq_wg_tile_m, d,
                q_nchunk, k_chunk, 0, 0, 0, sg_i_ktq, sg_j_ktq,
                (local char *)ugemm_slm);
#else
        ktq_tile_type dQ_tile;
#endif
        uint sg_i0_dq = sg_i_ktq * ugemm_ktq_sg_tile_m;
        uint sg_j0_dq = sg_j_ktq * ugemm_ktq_sg_tile_n + q0;

        tile_atomic_add(dQ_tile, dQ, d, q, ldq, sg_i0_dq, sg_j0_dq);
    }

    //////// update dV
    uint sg_i0_vs = sg_i_vs * ugemm_vs_sg_tile_m;
    uint sg_j0_vs = sg_j_vs * ugemm_vs_sg_tile_n;

    // ensure all loops done writing to SLM
    barrier(CLK_LOCAL_MEM_FENCE);

    dv_tile_type dV_tile_slm;
    tile_load(&dV_tile_slm, dV_slm, D_MAX, ugemm_kq_wg_tile_m, D_MAX, sg_i0_vs,
            sg_j0_vs);

    tile_store_dV(&dV_tile_slm, dV, d, k, ldv, sg_i0_vs, wg_i0 + sg_j0_vs,
            remainder_k);
    // /update dV

    //////// update dK
    uint sg_i0_dk = sg_i_qdSt * ugemm_qdSt_sg_tile_m;
    uint sg_j0_dk = sg_j_qdSt * ugemm_qdSt_sg_tile_n;

    a_tile_type dK_tile_slm;
    tile_load(&dK_tile_slm, dK_slm, ugemm_kq_wg_tile_m, D_MAX,
            ugemm_kq_wg_tile_m, sg_i0_dk, sg_j0_dk);

    int wg_k_chunk = min(k - k0, ugemm_kq_wg_tile_m);
    tile_store_dK(
            &dK_tile_slm, dK + wg_i0, wg_k_chunk, d, ldk, sg_i0_dk, sg_j0_dk);
    // /update dK
}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) kernel void
preprocess_Di(global float *Di, const global DST_DATA_T *A,
        const global DST_DATA_T *dA, int d, int k, int q, QRY_OFFSETS,
        DST_OFFSETS) {

    uint lda = DST_S2;
    uint ldq = QRY_S2;

    uint sg_ij = sub_group_broadcast(get_local_id(1), 0);
    uint sg_i_kq = sg_ij % ugemm_kq_sg_per_wg_m;
    uint sg_j_kq = sg_ij / ugemm_kq_sg_per_wg_m;

    uint b0, b1;
    b0 = get_group_id(1);
    b1 = get_group_id(2);

    const uint preprocess_batch = b1 * (DST_D1 * q) + b0 * q;

    const size_t q_offset = QRY_BATCH(b1, b0);
    const size_t a_offset = DST_BATCH(b1, b0);

    /* Locate A/dA matrices within batch */
    A += a_offset;
    dA += a_offset;

    Di += preprocess_batch;

    uint wg_q = get_group_id(0);
    uint wg_j0 = wg_q * ugemm_kq_wg_tile_n;

#define Di_slm_size (ugemm_kq_wg_tile_n * sizeof(float))
    local char slm[Di_slm_size];

    local float *Di_slm = (local float *)&slm[0];

    uint sg_i0_kq = sg_i_kq * ugemm_kq_sg_tile_m;
    uint sg_j0_kq = sg_j_kq * ugemm_kq_sg_tile_n;

    uint q0_copy = q_tile_sg_n * sg_ij;

    if (q > 0) {
        // D_i calculation
#if QRY_DT_F32
        dq_tile_type dA_tile, A_tile;
        tile_fill(A_tile, 0.f);
        tile_fill(dA_tile, 0.f);
        tile_load(
                &dA_tile, (global FMA_TYPE *)dA, d, q, lda, 0, wg_j0 + q0_copy);
        tile_load(&A_tile, (global FMA_TYPE *)A, d, q, lda, 0, wg_j0 + q0_copy);
#else
        dq_tile_type dA_tile, A_tile;
        q_tile_type dA_tile_reblock, A_tile_reblock; // load native type
        tile_fill(A_tile_reblock, TO_DATA_T(0.f));
        tile_fill(dA_tile_reblock, TO_DATA_T(0.f));

        tile_load(&dA_tile_reblock, (global FMA_TYPE *)dA, d, q, lda, 0,
                wg_j0 + q0_copy);
        tile_load(&A_tile_reblock, (global FMA_TYPE *)A, d, q, lda, 0,
                wg_j0 + q0_copy);

        // convert to float for calculation
        tile_copy_reblock(dA_tile_reblock, &dA_tile);
        tile_copy_reblock(A_tile_reblock, &A_tile);
#endif

#define binary_mul(x, y) ((x) * (y))
        tile_binary(A_tile, dA_tile, binary_mul);

        // reduce tile across D_MAX
        for (int j = 0; j < q_tile_sg_n; j++) {
            float r = 0.f;
            for (int i0 = 0; i0 < D_MAX; i0 += SUBGROUP_SIZE) {
                r += sub_group_reduce_add(
                        tile_access(A_tile, i0, j, SUBGROUP_SIZE, D_MAX, 1, 1));
            }
            Di_slm[j + q0_copy] = r;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = get_local_id(0); i < ugemm_kq_wg_tile_n;
                i += get_local_size(0)) {
            if (get_local_id(1) == 0 && (wg_j0 + i) < q) {
                Di[wg_j0 + i] = Di_slm[i];
            }
        }
    }
}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) kernel void zero_dQ(
        global float *dst, int nelems, QRY_OFFSETS) {
    uint b0 = get_group_id(1);
    uint b1 = get_group_id(2);

    const size_t offset = QRY_BATCH(b1, b0);

    dst += offset;
    size_t idx = get_global_id(0);
    if (idx < nelems) { dst[idx] = 0.f; }
}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) kernel void
postprocess_dQ(global DST_DATA_T *dst, global const float *src, int nelems,
        QRY_OFFSETS) {
    uint b0 = get_group_id(1);
    uint b1 = get_group_id(2);

    const size_t offset = QRY_BATCH(b1, b0);

    /* Locate dQ/dV/dK matrices within batch */
    src += offset;
    dst += offset;
    size_t idx = get_global_id(0);
    if (idx < (nelems)) { dst[idx] = TO_DATA_T(src[idx]); }
}
