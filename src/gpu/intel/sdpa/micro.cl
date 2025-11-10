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

#include "gpu/intel/include/conversion.h"
#include "gpu/intel/include/tile_ops.h"
#include "gpu/intel/include/types_interop.h"
#include "gpu/intel/sdpa/utils.h"

/* Microkernel headers -- generated at runtime */
#include "gemm_kq.h"
#include "gemm_vs.h"

/* The quantization parameter may be unique for each token/element */
#define QUANTIZE_2D 2

/* The quantization parameter shares the same value across the work-group */
#define QUANTIZE_COMMON 3

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define DIV_UP(x, y) (((x) + (y)-1) / (y))

#define sg_per_wg (ugemm_kq_sg_per_wg_m * ugemm_kq_sg_per_wg_n)
#define q_tile_sg_n DIV_UP(ugemm_kq_wg_tile_n, sg_per_wg)

/* Instantiate tile types and operations */
typedef ugemm_kq_c_type s_tile_type;
typedef ugemm_vs_c_type a_tile_type;

// Tile debugging example for s_tile
//
// example: declare print tile function macro for S_tile
// DECLARE_2D_TILE_PRINT(s_tile_type, float, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
//                       ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
//                       ugemm_kq_c_type_nblock1)
//
// example: Prints the entire S_tile in the (0, 1, 0) work group
// print_tile(S_tile, "%7.2f", 0, 1, 0, ugemm_kq_sg_per_wg_m, ugemm_kq_sg_per_wg_n);

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

#if KQ_F16_ACC
DECLARE_2D_TILE(s_tile_type_float, float, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1)
DECLARE_2D_TILE_COPY_REBLOCK(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, s_tile_type_float, SUBGROUP_SIZE,
        ugemm_kq_c_type_block0, ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, CONVERT_FLOAT_T)
#endif

#if VS_F16_ACC
DECLARE_2D_TILE(a_tile_type_float, float, SUBGROUP_SIZE, ugemm_vs_c_type_block0,
        ugemm_vs_c_type_block1, ugemm_vs_c_type_nblock0,
        ugemm_vs_c_type_nblock1)
DECLARE_2D_TILE_COPY_REBLOCK(a_tile_type, SUBGROUP_SIZE, ugemm_vs_c_type_block0,
        ugemm_vs_c_type_block1, ugemm_vs_c_type_nblock0,
        ugemm_vs_c_type_nblock1, a_tile_type_float, SUBGROUP_SIZE,
        ugemm_vs_c_type_block0, ugemm_vs_c_type_block1, ugemm_vs_c_type_nblock0,
        ugemm_vs_c_type_nblock1, CONVERT_FLOAT_T)
#endif

DECLARE_2D_TILE(s_tile_type_packed, uint, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1 / 2, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1)
DECLARE_2D_TILE(s_tile_type_reblock, FMA_TYPE, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_n, 1, ugemm_kq_sg_tile_n / ugemm_vs_sg_tile_n,
        ugemm_kq_sg_tile_m)
DECLARE_2D_TILE_BLOCK_OPS(s_tile_type_reblock, FMA_TYPE, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_n, 1, ugemm_kq_sg_tile_n / ugemm_vs_sg_tile_n,
        ugemm_kq_sg_tile_m)

DECLARE_2D_TILE(
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

#if KQ_F16_ACC

DECLARE_2D_TILE_COPY_REBLOCK(s_tile_type_float, SUBGROUP_SIZE,
        ugemm_kq_c_type_block0, ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, s_tile_type_reblock, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_n, 1, ugemm_kq_sg_tile_n / ugemm_vs_sg_tile_n,
        ugemm_kq_sg_tile_m, CONVERT_DATA_T)

DECLARE_2D_TILE_VREDUCE(s_tile_type_float, SUBGROUP_SIZE,
        ugemm_kq_c_type_block0, ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, s_sum_tile_type, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_n, 1, 1, 1)

DECLARE_2D_TILE_HREDUCE(s_tile_type_float, SUBGROUP_SIZE,
        ugemm_kq_c_type_block0, ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, kmask_tile_type_float, SUBGROUP_SIZE,
        ugemm_kq_sg_tile_m, 1, 1, 1)
#if WITH_ATTN_MASK
DECLARE_2D_TILE_HREDUCE(s_tile_type_float, SUBGROUP_SIZE,
        ugemm_kq_c_type_block0, ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, mask_tile_type_float, SUBGROUP_SIZE, mask_br,
        mask_bc, mask_nbr, mask_nbc)
#endif

#else

DECLARE_2D_TILE_COPY_REBLOCK(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
        ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
        ugemm_kq_c_type_nblock1, s_tile_type_reblock, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_n, 1, ugemm_kq_sg_tile_n / ugemm_vs_sg_tile_n,
        ugemm_kq_sg_tile_m, CONVERT_DATA_T)

DECLARE_2D_TILE_VREDUCE(s_tile_type, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
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
#endif

#if VS_F16_ACC

#if BLOCK_A
DECLARE_2D_TILE_COPY_REBLOCK(a_tile_type_float, SUBGROUP_SIZE,
        ugemm_vs_c_type_block0, ugemm_vs_c_type_block1, ugemm_vs_c_type_nblock0,
        ugemm_vs_c_type_nblock1, a_tile_type_dst, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_m, 1, 1, ugemm_vs_sg_tile_n, CONVERT_DATA_T)
#else
DECLARE_2D_TILE_COPY_REBLOCK(a_tile_type_float, SUBGROUP_SIZE,
        ugemm_vs_c_type_block0, ugemm_vs_c_type_block1, ugemm_vs_c_type_nblock0,
        ugemm_vs_c_type_nblock1, a_tile_type_dst, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_m, 8, 1, ugemm_vs_sg_tile_n / 8, CONVERT_DATA_T)
#endif

DECLARE_2D_TILE_HREDUCE(a_tile_type_float, SUBGROUP_SIZE,
        ugemm_vs_c_type_block0, ugemm_vs_c_type_block1, ugemm_vs_c_type_nblock0,
        ugemm_vs_c_type_nblock1, a_scale_tile_type, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_n, 1, 1, 1)

#else

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

DECLARE_2D_TILE_HREDUCE(a_tile_type, SUBGROUP_SIZE, ugemm_vs_c_type_block0,
        ugemm_vs_c_type_block1, ugemm_vs_c_type_nblock0,
        ugemm_vs_c_type_nblock1, a_scale_tile_type, SUBGROUP_SIZE,
        ugemm_vs_sg_tile_n, 1, 1, 1)
#endif

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

#if REMAINDER_Q
#define tile_load_block_rem_q(t, ptr, n, ld, off_r, off_c) \
    tile_load_block(t, ptr, n, ld, off_r, off_c);
#define tile_store_block_rem_q(t, ptr, n, ld, off_r, off_c) \
    tile_store_block(t, ptr, n, ld, off_r, off_c);
#else
#define tile_load_block_rem_q(t, ptr, n, ld, off_r, off_c) \
    tile_load_block(t, ptr, ld, off_r, off_c);
#define tile_store_block_rem_q(t, ptr, n, ld, off_r, off_c) \
    tile_store_block(t, ptr, ld, off_r, off_c);
#endif

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
        int m, int n, int ldq, int offset_r, int offset_c) {

#if USE_SYSTOLIC_UKERNEL

#if BLOCK_Q
    tile_load_block_rem_q(
            Q_tile, (global uint *)Q, n, ldq >> 1, offset_r, offset_c);
#elif Q_ALIGN >= 4
    tile_load(Q_tile, (global uint *)Q, (m + 1) >> 1, n, ldq >> 1, offset_r,
            offset_c);
#else
    tile_load_packed_vec2(Q_tile, Q, m, n, ldq, offset_r, offset_c);
#endif

#else // FMA

#if BLOCK_Q
    tile_load_block_rem_q(Q_tile, Q, n, ldq, offset_r, offset_c);
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

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) kernel void
micro_sdpa(const global KEY_DATA_T *K, const global QRY_DATA_T *Q,
        const global VAL_DATA_T *V, global DST_DATA_T *A,
#if WITH_HOST_SCALE
        float scalar_scale, float inv_scalar_scale,
#else
        const global SCALE_DATA_T *scale_ptr,
#endif
        int d, int k, int q, const global KEY_ATTR_SCALES_DATA_T *K_scales,
        const global KEY_ATTR_ZP_DATA_T *K_zp,
        const global VAL_ATTR_SCALES_DATA_T *V_scales,
        const global VAL_ATTR_ZP_DATA_T *V_zp, const int attn_mask_type
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
        const int remainder_k) {

    uint sg_ij = sub_group_broadcast(get_local_id(1), 0);
    uint b1 = get_group_id(2);

    uint b0, b0_kv;
    uint wg_j0 = get_group_id(0) * ugemm_kq_wg_tile_n;

    uint q_group_size;
    if (q == 1 && KV_GROUP_SIZE > 1) {
        // For second token Grouped Query Attention(GQA) cases, we batch the
        // kernel across the KV heads instead of the q heads. This allows us to
        // batch multiple queries into a single work group.
        b0_kv = get_group_id(1);
        b0 = b0_kv * KV_GROUP_SIZE;
        q_group_size = KV_GROUP_SIZE;
    } else {
        b0 = get_group_id(1);
        b0_kv = b0 / KV_GROUP_SIZE;
        q_group_size = q;
    }

    /* Calculate the number of keys to process */
    int k0end = k;
#if WITH_CAUSAL_MASK
    if (attn_mask_type == ATTN_MASK_TOP_LEFT) {
        k0end = min(k, (int)(wg_j0 + ugemm_kq_wg_tile_n));
        if (q == 1) k0end = 1;
    } else {
        k0end = min(k, (int)(wg_j0 + ugemm_kq_wg_tile_n) - (q - k));
    }
#endif

    /* Leading dimension for matrices */
    uint ldk = TRANSPOSE_K ? KEY_S3 : KEY_S2;
    uint ldq = QRY_S2;
    uint ldv = VAL_S2;
    uint lda = DST_S2;

#if KEY_SCALES || KEY_ZERO_POINTS
    uint ldkq = KEY_D3;
    uint num_key_groups = d / KEY_GROUP_SIZE;
#endif
#if VAL_SCALES || VAL_ZERO_POINTS
    uint ldvq = div_up(d, VAL_GROUP_SIZE);
    uint num_val_groups = d / VAL_GROUP_SIZE;
#endif

    /* Subgroup IDs for each GEMM */
    uint sg_i_kq = sg_ij % ugemm_kq_sg_per_wg_m;
    uint sg_j_kq = sg_ij / ugemm_kq_sg_per_wg_m;

    uint sg_i_vs = sg_ij % ugemm_vs_sg_per_wg_m;
    uint sg_j_vs = sg_ij / ugemm_vs_sg_per_wg_m;

    /* SLM allocations -- place in one array to work around compiler bug */
#define Q_slm_size (D_MAX * ugemm_kq_wg_tile_n * sizeof(QRY_DATA_T))
#define S_slm_size \
    (ugemm_kq_wg_tile_m * ugemm_kq_wg_tile_n * sizeof(QRY_DATA_T))
#define S_sum_slm_size \
    (ugemm_kq_wg_tile_n * ugemm_kq_sg_per_wg_m * sizeof(float))
#define S_max_slm_size (ugemm_kq_wg_tile_n * sizeof(float))
#define ugemm_slm_size MAX(ugemm_kq_slm_size, ugemm_vs_slm_size)

    local char slm[Q_slm_size + S_slm_size + S_sum_slm_size + S_max_slm_size
            + ugemm_slm_size];

    local QRY_DATA_T *Q_slm = (local QRY_DATA_T *)&slm[0];
    local QRY_DATA_T *S_slm = (local QRY_DATA_T *)&slm[Q_slm_size];
    local float *S_sum_slm = (local float *)&slm[Q_slm_size + S_slm_size];
    local float *S_max_slm
            = (local float *)&slm[Q_slm_size + S_slm_size + S_sum_slm_size];
    local uint *ugemm_slm = (local uint *)&slm[Q_slm_size + S_slm_size
            + S_sum_slm_size + S_max_slm_size];

    const bool need_sum_barrier = (ugemm_vs_barrier_count == 0);

    /* Convert to half precision and store */
    const size_t k_offset = KEY_BATCH(b1, b0_kv);
    const size_t v_offset = VAL_BATCH(b1, b0_kv);
    /* Locate K/Q/V/A matrices within batch */
    K += k_offset / KEY_ELEMENTS_PER_BYTE;
    Q += QRY_BATCH(b1, b0);
    V += v_offset / VAL_ELEMENTS_PER_BYTE;
    A += DST_BATCH(b1, b0);
#if WITH_ATTN_MASK
    msk += MSK_BATCH(b1 % MSK_D0, b0 % MSK_D1);
    int mask_aligned = (((size_t)msk) % 4) == 0;
    bool block_msk = (b1 < MSK_D0 - ceil((float)ugemm_kq_wg_tile_m / k))
            && mask_aligned;
#endif

#if KEY_SCALES
    K_scales += k_offset / KEY_GROUP_SIZE;
#endif
#if KEY_SCALES == QUANTIZE_COMMON
    float k_scale = KEY_SCALES_TO_FLOAT(*K_scales);
#endif
#if KEY_ZERO_POINTS
    K_zp += k_offset / KEY_GROUP_SIZE / KEY_ZP_ELEMENTS_PER_BYTE;
#endif
#if VAL_SCALES
    V_scales += v_offset / VAL_GROUP_SIZE;
#endif
#if VAL_SCALES == QUANTIZE_COMMON
    float v_scale = VAL_SCALES_TO_FLOAT(*V_scales);
#endif
#if VAL_ZERO_POINTS
    V_zp += v_offset / VAL_GROUP_SIZE / VAL_ZP_ELEMENTS_PER_BYTE;
#endif

    if (k0end > 0) {
        /* Load Q tile, destined for SLM */
        q_tile_type Q_tile;
        uint q0_copy = q_tile_sg_n * sg_ij;

        tile_load_src1(&Q_tile, Q, d, q_group_size, ldq, 0, wg_j0 + q0_copy);

        /* Store Q tile to SLM */
        tile_store_t_slm_src1(
                &Q_tile, Q_slm, ugemm_kq_sg_tile_n, D_MAX, q0_copy, 0);

#if Q_ARRIVE_AWAIT_BARRIER
        intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);
#endif
    }

    /* Load scale */
    float scale = 1.f;
    float iscale = 1.f;
    if (k0end > 0) {
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
        scale *= 1.442695f; // log2(e)
    }

#if PREFETCH_K0
    if (k0end > 0) {
        /* Prefetch first K tile. */
        cooperative_prefetch_2d_k(
                /* ptr */ K,
                /* r */ k,
                /* c */ d,
                /* rmax */ ugemm_kq_wg_tile_m,
                /* cmax */ PREFETCH_D_MAX,
                /* ld */ ldk,
                /* sg_id */ sg_ij,
                /* n_sg */ sg_per_wg,
                /* sg_size */ SUBGROUP_SIZE,
                /* cache */ LSC_LDCC_L1C_L3C);

#if KEY_SCALES == QUANTIZE_2D
        cooperative_prefetch_2d_maybe_rem(
                /* ptr */ K_scales,
                /* r */ k,
                /* c */ num_key_groups,
                /* rmax */ ugemm_kq_wg_tile_m,
                /* cmax */ D_MAX / KEY_GROUP_SIZE,
                /* ld */ ldkq,
                /* sg_id */ sg_ij,
                /* n_sg */ sg_per_wg,
                /* sg_size */ SUBGROUP_SIZE,
                /* cache */ LSC_LDCC_L1C_L3C);
#endif
#if KEY_ZERO_POINTS == QUANTIZE_2D
        cooperative_prefetch_2d_maybe_rem(
                /* ptr */ K_zp,
                /* r */ k,
                /* c */ num_key_groups,
                /* rmax */ ugemm_kq_wg_tile_m,
                /* cmax */ D_MAX / KEY_GROUP_SIZE,
                /* ld */ ldkq,
                /* sg_id */ sg_ij,
                /* n_sg */ sg_per_wg,
                /* sg_size */ SUBGROUP_SIZE,
                /* cache */ LSC_LDCC_L1C_L3C);
#endif
    }
#endif

    if (k0end > 0) {
        /* Initialize S column sums in SLM to -inf */
        const uint n_col_sg
                = DIV_UP(ugemm_kq_wg_tile_n, SUBGROUP_SIZE * sg_per_wg);
        const float neg_inf = -INFINITY;

#pragma unroll
        for (int q = 0; q < n_col_sg; q++)
            intel_sub_group_block_write(
                    (local uint *)&S_max_slm[(q + sg_ij * n_col_sg)
                            * SUBGROUP_SIZE],
                    as_uint(neg_inf));
    }

#if VS_F16_ACC
    a_tile_type_float A_tile;
#else
    a_tile_type A_tile;
#endif
    s_sum_tile_type S_sum_tile;
    s_sum_tile_type S_max_tile, S_max_tile_old;

    if (k0end > 0) {
        /* Clear accumulator */
        tile_fill(A_tile, 0.0f);

        /* Clear S column sums/maxes */
        tile_fill(S_sum_tile, 0.0f);
        tile_fill(S_max_tile, -INFINITY);

        /* Wait for Q data to reach SLM */
#if Q_ARRIVE_AWAIT_BARRIER
        intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
#else
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
    }

    uint sg_i0_kq = sg_i_kq * ugemm_kq_sg_tile_m;
    uint sg_j0_kq = sg_j_kq * ugemm_kq_sg_tile_n;

    /* Main loop over k blocks */
    for (int k0 = 0; k0 < k0end; k0 += ugemm_kq_wg_tile_m) {
        bool first = (k0 == 0);
        int knext = k0 + ugemm_kq_wg_tile_m;
        bool last = (knext >= k0end);

#if WITH_ATTN_MASK
        /* Load mask. No remainder handling needed assuming k block size is a power of 2. */
        mask_tile_type mask_tile;
#if BROADCAST_MASK_Q
        if (block_msk) {
            tile_load_block(&mask_tile, msk, k, 0, k0 + sg_i0_kq, 0);
        } else {
            tile_load(&mask_tile, msk, k, 1, k0 + sg_i0_kq, 0);
        }
#else
        tile_load_t(&mask_tile, msk, q, k, sg_j0_kq + wg_j0, k0 + sg_i0_kq);
#endif
#endif

        /* Prepare k mask: NaN in bounds, -inf out of bounds */
        kmask_tile_type_float k_mask;
        if (remainder_k) {
#pragma unroll
            for (int ii = 0; ii < ugemm_kq_sg_tile_m / SUBGROUP_SIZE; ii++) {
                k_mask.x[0][ii] = (k0 + sg_i0_kq + ii * SUBGROUP_SIZE
                                                  + get_sub_group_local_id()
                                          < k0end)
                        ? nan(0u)
                        : -INFINITY;
            }
        }

        /* Calculate S = (K^T) * Q */
#if KQ_F16_ACC
        s_tile_type S_tile_f16
#else
        s_tile_type S_tile
#endif
                = ugemm_kq(K, ldk, Q_slm, D_MAX, k0end, ugemm_kq_wg_tile_n, d,
                        k0, 0, 0, sg_i_kq, sg_j_kq, (local char *)ugemm_slm
#if KEY_SCALES == QUANTIZE_2D
                        ,
                        K_scales
#endif
#if KEY_ZERO_POINTS
                        ,
                        K_zp
#endif
#if (KEY_SCALES == QUANTIZE_2D) || KEY_ZERO_POINTS
                        ,
                        ldkq
#endif
                );
#if KQ_F16_ACC
        s_tile_type_float S_tile;
        tile_copy_reblock(S_tile_f16, &S_tile);
#endif

#if KEY_SCALES == QUANTIZE_COMMON
#define k_scale_op(x) ((x)*k_scale)
        tile_elementwise(S_tile, k_scale_op);
#endif

        /* Apply attention mask */
#if WITH_ATTN_MASK
#define unscale(x) ((x)*iscale)
        mask_tile_type_float mask_tile_float;
        tile_copy_reblock(mask_tile, &mask_tile_float);
#if WITH_ATTN_SCALE
        tile_elementwise(mask_tile_float, unscale);
#endif
#if BROADCAST_MASK_Q
        tile_hbroadcast_add(&S_tile, mask_tile_float);
#else
        tile_binary(S_tile, mask_tile_float, binary_add);
#endif
#endif

        /* Apply k mask */
        if (remainder_k) { tile_hbroadcast_min(&S_tile, k_mask); }

#if WITH_CAUSAL_MASK
#define less_than(offset_k, offset_q) (offset_q < offset_k)

        int col_offset = wg_j0 + sg_j0_kq;
        if (q == 1) col_offset = 1;
        if (attn_mask_type == ATTN_MASK_BOTTOM_RIGHT) col_offset += k - q;

        /* Apply causal mask */
        tile_predicated_assignment_t(S_tile, k0 + sg_i0_kq, col_offset,
                less_than, -INFINITY, SUBGROUP_SIZE, ugemm_kq_c_type_block0,
                ugemm_kq_c_type_block1, ugemm_kq_c_type_nblock0,
                ugemm_kq_c_type_nblock1);
#endif

        /* Before softmax, we will need to scale columns by maximum values to avoid overflow. */

        /* Compute our maxima and reduce across SLM */
        tile_vreduce_max(S_tile, &S_max_tile);
        tile_atomic_max_full(
                S_max_tile, S_max_slm, ugemm_kq_wg_tile_n, sg_j0_kq, 0);
        intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);

        int k_chunk = min(k0end - k0, ugemm_kq_wg_tile_m);
#if PREFETCH_V
        /* Prefetch V tile. */
        cooperative_prefetch_2d_maybe_rem(
                /* ptr */ V,
                /* r */ d,
                /* c */ k0end - k0,
                /* rmax */ PREFETCH_D_MAX,
                /* cmax */ ugemm_kq_wg_tile_m,
                /* ld */ ldv,
                /* sg_id */ sg_ij,
                /* n_sg */ sg_per_wg,
                /* sg_size */ SUBGROUP_SIZE,
                /* cache */ LSC_LDCC_L1C_L3C);

#if VAL_SCALES == QUANTIZE_2D
        /* Prefetch V scales. */
        cooperative_prefetch_2d_maybe_rem(
                /* ptr */ V_scales,
                /* r */ num_val_groups,
                /* c */ k0end - k0,
                /* rmax */ PREFETCH_D_MAX / VAL_GROUP_SIZE,
                /* cmax */ k_chunk,
                /* ld */ ldvq,
                /* sg_id */ sg_ij,
                /* n_sg */ sg_per_wg,
                /* sg_size */ SUBGROUP_SIZE,
                /* cache */ LSC_LDCC_L1C_L3C);
#endif
#if VAL_ZERO_POINTS == QUANTIZE_2D
        /* Prefetch V zero points. */
        cooperative_prefetch_2d_maybe_rem(
                /* ptr */ V_zp,
                /* r */ num_val_groups,
                /* c */ k0end - k0,
                /* rmax */ PREFETCH_D_MAX / VAL_GROUP_SIZE,
                /* cmax */ k_chunk,
                /* ld */ ldvq,
                /* sg_id */ sg_ij,
                /* n_sg */ sg_per_wg,
                /* sg_size */ SUBGROUP_SIZE,
                /* cache */ LSC_LDCC_L1C_L3C);
#endif
#endif
#ifndef ALT_MAX
        /* Read back WG-wide maxima */
        intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
        tile_load_full(&S_max_tile, S_max_slm, ugemm_kq_wg_tile_n, sg_j0_kq, 0);
#endif

#if SOFTMAX_INF_AS_ZERO
#define set_zeros(v) vselect(-FLT_MAX, v, visfinite(v))
        tile_elementwise(S_max_tile, set_zeros);
#endif

        tile_vbroadcast_sub(&S_tile, S_max_tile);

/* Scale + exponentiate */
#define scaled_exp(x) native_vexp2(x *scale)
        tile_elementwise(S_tile, scaled_exp);

#ifdef ALT_MAX
        /* Read back WG-wide maxima and adjust S to match */
        intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);
        s_sum_tile_type S_max_tile1;
        tile_copy(S_max_tile, S_max_tile1);
        tile_load_full(&S_max_tile, S_max_slm, ugemm_kq_wg_tile_n, sg_j0_kq, 0);

#define binary_exp_neg(x, y) native_vexp2(scale *((x) - (y)))
        tile_binary(S_max_tile1, S_max_tile, binary_exp_neg);
        tile_vbroadcast_mul(&S_tile, S_max_tile1);
#endif

        /* Accumulate sums. S tile is transposed for easy summation. */
        s_sum_tile_type S_sum_tile1;
        tile_fill(S_sum_tile1, 0.0f);
        tile_vreduce_add(S_tile, &S_sum_tile1);

#if USE_SYSTOLIC_UKERNEL
        /* Convert to half or bf16, VNNI format */
        s_tile_type_packed S_tile_packed;
        tile_copy_to_vec2(S_tile, S_tile_packed, VEC_TYPE2);

        /* Store to SLM, in packed format */
        tile_store_t_sys_src2(S_tile_packed, (local uint *)S_slm,
                ugemm_vs_sg_tile_n, ugemm_kq_wg_tile_m / 2, sg_i0_kq / 2,
                sg_j0_kq);
#else
        /* Reblock and store to SLM */
        s_tile_type_reblock S_tile_reblock;
        tile_copy_reblock(S_tile, &S_tile_reblock);
        tile_store_block_packed(S_tile_reblock, S_slm, ugemm_vs_sg_tile_n,
                ugemm_kq_wg_tile_m, sg_j0_kq, sg_i0_kq);
#endif

        intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);

        /* Rescale existing accumulator and sums to match new maxima */
        if (!first) {
#define binary_exp_sub(x, y) native_vexp2(scale *((x) - (y)))
#define binary_mul(x, y) ((x) * (y))
            tile_binary(S_max_tile_old, S_max_tile, binary_exp_sub);
            tile_binary(S_sum_tile, S_max_tile_old, binary_mul);

            /* Find the subset of sums that applies to the accumulation tile */
            a_scale_tile_type A_scale_tile;
#if ugemm_kq_wg_tile_n == ugemm_vs_wg_tile_n \
        && ugemm_kq_sg_tile_n == ugemm_vs_sg_tile_n
            tile_copy(S_max_tile_old, A_scale_tile);
#elif ugemm_kq_wg_tile_n == ugemm_vs_wg_tile_n \
        && (ugemm_kq_sg_tile_n % ugemm_vs_sg_tile_n) == 0
            tile_rselect(&A_scale_tile, S_max_tile_old,
                    sg_j_vs % (ugemm_kq_sg_tile_n / ugemm_vs_sg_tile_n));
#else
#error unimplemented
#endif
            tile_hbroadcast_mul(&A_tile, A_scale_tile);
        }

        /* Accumulate sums */
        tile_binary(S_sum_tile, S_sum_tile1, binary_add);

        /* Save maxima */
        tile_copy(S_max_tile, S_max_tile_old);

        /* Last iteration: store column sums in SLM */
        if (last) {
            tile_store_full(S_sum_tile, S_sum_slm, ugemm_kq_wg_tile_n, sg_j0_kq,
                    sg_i_kq);
        }

#if PREFETCH_K
        /* Prefetch next K tile. */
        if (!last) {
#if TRANSPOSE_K
            const uint stride_k = ldk;
#else
            const uint stride_k = 1;
#endif

            const global KEY_DATA_T *K_next = K + (knext)*stride_k;
            cooperative_prefetch_2d_k(
                    /* ptr */ K_next,
                    /* r */ k0end - k0 - ugemm_kq_wg_tile_m,
                    /* c */ d,
                    /* rmax */ ugemm_kq_wg_tile_m,
                    /* cmax */ D_MAX,
                    /* ld*/ ldk,
                    /* sg_id */ sg_ij,
                    /* n_sg */ sg_per_wg,
                    /* sg_size */ SUBGROUP_SIZE,
                    /* cache*/ LSC_LDCC_L1C_L3C);
#if KEY_SCALES == QUANTIZE_2D
            const global KEY_ATTR_SCALES_DATA_T *K_scales_next
                    = K_scales + knext;
            cooperative_prefetch_2d_maybe_rem(
                    /* ptr */ K_scales_next,
                    /* r */ k0end - k0 - ugemm_kq_wg_tile_m,
                    /* c */ num_key_groups,
                    /* rmax */ ugemm_kq_wg_tile_m,
                    /* cmax */ D_MAX / KEY_GROUP_SIZE,
                    /* ld */ ldkq,
                    /* sg_id */ sg_ij,
                    /* n_sg */ sg_per_wg,
                    /* sg_size */ SUBGROUP_SIZE,
                    /* cache */ LSC_LDCC_L1C_L3C);
#endif
#if KEY_ZERO_POINTS == QUANTIZE_2D
            const global KEY_ATTR_ZP_DATA_T *K_zp_next = K_zp + knext;
            cooperative_prefetch_2d_maybe_rem(
                    /* ptr */ K_zp_next,
                    /* r */ k0end - k0 - ugemm_kq_wg_tile_m,
                    /* c */ num_key_groups,
                    /* rmax */ ugemm_kq_wg_tile_m,
                    /* cmax */ D_MAX / KEY_GROUP_SIZE,
                    /* ld */ ldkq,
                    /* sg_id */ sg_ij,
                    /* n_sg */ sg_per_wg,
                    /* sg_size */ SUBGROUP_SIZE,
                    /* cache */ LSC_LDCC_L1C_L3C);
#endif
        }
#endif

#if WITH_ATTN_MASK && PREFETCH_MASK
        /* Prefetch next mask tile. */
        if (!last) {
#if BROADCAST_MASK_Q
            cooperative_prefetch_2d_maybe_rem(
                    /* ptr */ msk + knext,
                    /* r */ k0end - k0 - ugemm_kq_wg_tile_m,
                    /* c */ 1,
                    /* rmax */ ugemm_kq_wg_tile_m,
                    /* cmax */ 1,
                    /* ld */ 0,
                    /* sg_id */ sg_ij,
                    /* n_sg */ sg_per_wg,
                    /* sg_size */ SUBGROUP_SIZE,
                    /* cache */ LSC_LDCC_L1C_L3C);
#else
            cooperative_prefetch_2d_maybe_rem(
                    /* ptr */ msk + k0 + ugemm_kq_sg_tile_m + (wg_j0)*MSK_S2,
                    /* r */ k0end - k0 - ugemm_kq_wg_tile_m,
                    /* c */ q - wg_j0,
                    /* rmax */ ugemm_kq_wg_tile_m,
                    /* cmax */ (ugemm_kq_wg_tile_n * PREFETCH_D_MAX) / D_MAX,
                    /* ld */ MSK_S2,
                    /* sg_id */ sg_ij,
                    /* n_sg */ sg_per_wg,
                    /* sg_size */ SUBGROUP_SIZE,
                    /* cache */ LSC_LDCC_L1UC_L3C);
#endif
        }
#endif

        /* Wait for S stores */
        intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);

        /* Last iteration: signal column sums are ready */
        if (last && need_sum_barrier)
            intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE);

            /* Accumulate A += V * S */
#if VS_F16_ACC
        a_tile_type A_tile1_f16
#else
        a_tile_type A_tile1
#endif
                = ugemm_vs(V, ldv, S_slm, ugemm_kq_wg_tile_m, d,
                        ugemm_kq_wg_tile_n, k_chunk, 0, 0, 0, sg_i_vs, sg_j_vs,
                        (local char *)ugemm_slm
#if VAL_SCALES == QUANTIZE_2D
                        ,
                        V_scales
#endif
#if VAL_ZERO_POINTS
                        ,
                        V_zp
#endif
#if (VAL_SCALES == QUANTIZE_2D) || VAL_ZERO_POINTS
                        ,
                        ldvq
#endif
                );
#if VS_F16_ACC
        a_tile_type_float A_tile1;
        tile_copy_reblock(A_tile1_f16, &A_tile1);
#endif

        V += ldv * ugemm_kq_wg_tile_m / VAL_ELEMENTS_PER_BYTE;
#if VAL_SCALES == QUANTIZE_2D
        V_scales += ldvq * ugemm_kq_wg_tile_m;
#endif
#if VAL_ZERO_POINTS == QUANTIZE_2D
        V_zp += ldvq * ugemm_kq_wg_tile_m / VAL_ZP_ELEMENTS_PER_BYTE;
#endif
        tile_binary(A_tile, A_tile1, binary_add);
    }

    if (k0end > 0) {
        /* Wait for column sums to be ready */
        if (need_sum_barrier)
            intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE);

        /* Load column sums from SLM + reduce in registers */
        a_scale_tile_type A_scale_tile, A_scale_tile_load;
        tile_fill(A_scale_tile, 0.0f);

#pragma unroll
        for (uint sg1 = 0; sg1 < ugemm_kq_sg_per_wg_m; sg1++) {
            tile_load_full(&A_scale_tile_load, S_sum_slm, ugemm_kq_wg_tile_n,
                    ugemm_vs_sg_tile_n * sg_j_vs, sg1);
            tile_binary(A_scale_tile, A_scale_tile_load, binary_add);
        }
#if VAL_SCALES == QUANTIZE_COMMON
#define v_scale_op(x) ((x)*v_scale)
        tile_elementwise(A_tile, v_scale_op);
#endif

        /* Rescale by 1 / (column sums) */
#if SOFTMAX_INF_AS_ZERO
#define set_zeros2(v) (vselect(native_vrecip(v), 1.f, v == 0))
        tile_elementwise(A_scale_tile, set_zeros2);
#else
        tile_elementwise(A_scale_tile, native_vrecip);
#endif
        tile_hbroadcast_mul(&A_tile, A_scale_tile);
    }

    a_tile_type_dst A_tile_dst;
    if (k0end > 0) {
        /* Convert to half precision and store */
        tile_copy_reblock(A_tile, &A_tile_dst);
    } else {
        tile_fill(A_tile_dst, 0.0f);
    }

    uint sg_i0_vs = sg_i_vs * ugemm_vs_sg_tile_m;
    uint sg_j0_vs = sg_j_vs * ugemm_vs_sg_tile_n + wg_j0;

#if BLOCK_2D_A
    tile_store_block2d(A_tile_dst, A, d, q_group_size, lda, sg_i0_vs, sg_j0_vs);
#elif BLOCK_A
    tile_store_block_rem_q(
            A_tile_dst, A, q_group_size, lda, sg_i0_vs, sg_j0_vs);
#else
    tile_store(A_tile_dst, A, d, q_group_size, lda, sg_i0_vs, sg_j0_vs);
#endif
}
