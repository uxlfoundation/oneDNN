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

#include "gpu/intel/include/dispatch.h"
#include "gpu/intel/include/io.h"
#include "gpu/intel/include/post_ops.h"
#include "gpu/intel/include/types.h"

// Read functions.
inline VECT_N(FLT_ACC_DATA_T)
        read_vect_c_block(int idx, const __global DATA_T *ptr, off_t c,
                off_t blocks_stride, int chunks_per_block);
inline VECT_N(int) read_vect_c_block_int(int idx, const __global int *ptr,
        off_t c, off_t blocks_stride, int chunks_per_block);

// Write functions.
inline void write_vect_c_block(int idx, __global DATA_T *ptr, off_t c,
        off_t blocks_stride, int chunks_per_block,
        VECT_N(FLT_ACC_DATA_T) block);
inline void write_vect_c_block_int(int idx, __global int *ptr, off_t c,
        off_t blocks_stride, int chunks_per_block, VECT_N(int) block);

#if IS_FWD
KERNEL_ATTR
__kernel void xe_pooling_fwd(__global DATA_T *src, __global int *ws,
        __global DATA_T *dst, const dim_t batch_id POST_OP_ARGS) {

    if (GWS_OVERFLOW) return;

    const off_t mb0 = MB_BLOCK_SIZE * batch_id + GWS_GET_MB();
#if UNROLL_MB_COUNT > 1
    const off_t mb1 = mb0 + MB / 2;
#endif
    const off_t c = GWS_GET_C();
    const off_t od = GWS_GET_OD();
    const off_t oh = GWS_GET_OH();
    const off_t ow = GWS_GET_OW();

    // Calculate number of subgroup chunks inside C block
    // and stride between consecutive MB/C blocks
#if USE_MB_C_BLOCK
    const off_t src_stride = (SRC_SB0 > 1) ? SRC_SB0 : SRC_S0;
    const off_t dst_stride = (DST_SB0 > 1) ? DST_SB0 : DST_S0;
    const int src_chunks_per_c_block = CHUNKS_PER_C_BLOCK;
    const int dst_chunks_per_c_block = CHUNKS_PER_C_BLOCK;
#elif USE_ONLY_C_BLOCK
    const off_t src_stride = (SRC_B1 > 1) ? SRC_S1 : SUB_GROUP_SIZE;
    const off_t dst_stride = (DST_B1 > 1) ? DST_S1 : SUB_GROUP_SIZE;
    const int src_chunks_per_c_block
            = (SRC_B1 > 1) ? (SRC_B1 / SUB_GROUP_SIZE) : 1;
    const int dst_chunks_per_c_block
            = (DST_B1 > 1) ? (DST_B1 / SUB_GROUP_SIZE) : 1;
#endif

    const off_t ws_stride = dst_stride;
    const int ws_chunks_per_c_block = dst_chunks_per_c_block;

    if (mb0 >= SRC_D0) {
        VECT_N(FLT_ACC_DATA_T) dst_zero = 0;
        VECT_N(int) ws_zero = 0;
        off_t off = DST_OFF(mb0, c, od, oh, ow);
        write_vect_c_block(
                0, &dst[off], c, dst_stride, dst_chunks_per_c_block, dst_zero);
        write_vect_c_block(
                1, &dst[off], c, dst_stride, dst_chunks_per_c_block, dst_zero);
#if ALG_MAX && IS_TRAINING
        write_vect_c_block_int(
                0, &ws[off], c, ws_stride, ws_chunks_per_c_block, ws_zero);
        write_vect_c_block_int(
                1, &ws[off], c, ws_stride, ws_chunks_per_c_block, ws_zero);
#endif // ALG_MAX && IS_TRAINING

        return;
    }

    const off_t id = od * SD - PD;
    const off_t ih = oh * SH - PH;
    const off_t iw = ow * SW - PW;
    DATA_T d_type_dummy;
    FLT_ACC_DATA_T d_min = ALG_MAX
            ? CONCAT2(into_, FLT_ACC_DATA_T)(min_val(d_type_dummy))
            : (FLT_ACC_DATA_T)0;
    VECT_N(FLT_ACC_DATA_T) D0 = d_min;
    VECT_N(FLT_ACC_DATA_T) D1 = d_min;
    VECT_N(int) WS0 = 0, WS1 = 0;

    for (int kd = 0; kd < KD; ++kd) {
        if (id + kd < 0 || id + kd >= ID) continue;
        for (int kh = 0; kh < KH; ++kh) {
            if (ih + kh < 0 || ih + kh >= IH) continue;
            for (int kw = 0; kw < KW; ++kw) {
                if (iw + kw < 0 || iw + kw >= IW) continue;

                off_t src_off0 = SRC_OFF(mb0, c, id + kd, ih + kh, iw + kw);
#if UNROLL_MB_COUNT > 1
                off_t src_off1 = SRC_OFF(mb1, c, id + kd, ih + kh, iw + kw);
#endif
                VECT_N(FLT_ACC_DATA_T)
                S0 = read_vect_c_block(0, &src[src_off0], c, src_stride,
                        src_chunks_per_c_block);
#if UNROLL_MB_COUNT > 1
                VECT_N(FLT_ACC_DATA_T)
                S1 = read_vect_c_block(0, &src[src_off1], c, src_stride,
                        src_chunks_per_c_block);
#else
                VECT_N(FLT_ACC_DATA_T)
                S1 = read_vect_c_block(1, &src[src_off0], c, src_stride,
                        src_chunks_per_c_block);
#endif

#if ALG_MAX
#if IS_TRAINING
                // select() mask width must match the value type width.
                // For f64 (doubleN), use longN; for f32 (floatN), use intN.
                VECT_N(int)
                CMP0 = CONCAT2(convert_, VECT_N(int))(isless(D0, S0));
                WS0 = select(WS0, kd * KH * KW + kh * KW + kw, CMP0);
#if DT_F64
                D0 = select(D0, S0, CONCAT2(convert_, VECT_N(long))(CMP0));
#else
                D0 = select(D0, S0, CMP0);
#endif

                VECT_N(int)
                CMP1 = CONCAT2(convert_, VECT_N(int))(isless(D1, S1));
                WS1 = select(WS1, kd * KH * KW + kh * KW + kw, CMP1);
#if DT_F64
                D1 = select(D1, S1, CONCAT2(convert_, VECT_N(long))(CMP1));
#else
                D1 = select(D1, S1, CMP1);
#endif

#else // TRAINING
                D0 = max(D0, S0);
                D1 = max(D1, S1);
#endif // TRAINING
#else // ALG_MAX
                D0 += S0;
                D1 += S1;
#endif // ALG_MAX
            }
        }
    }

#if ALG_AVG_P
    D0 = D0 / (KD * KH * KW);
    D1 = D1 / (KD * KH * KW);

#endif // ALG_AVG_P

#if ALG_AVG_NP
    const off_t id_start = max(od * SD - PD, (off_t)0);
    const off_t ih_start = max(oh * SH - PH, (off_t)0);
    const off_t iw_start = max(ow * SW - PW, (off_t)0);
    const off_t id_end = min(od * SD - PD + KD, (off_t)ID);
    const off_t ih_end = min(oh * SH - PH + KH, (off_t)IH);
    const off_t iw_end = min(ow * SW - PW + KW, (off_t)IW);
    const int num_summands = (int)(ih_end - ih_start) * (int)(iw_end - iw_start)
            * (int)(id_end - id_start);
    D0 = D0 / num_summands;
    D1 = D1 / num_summands;
#endif // ALG_AVG_NP

    off_t dst_off0 = DST_OFF(mb0, c, od, oh, ow);
#if UNROLL_MB_COUNT > 1
    off_t dst_off1 = DST_OFF(mb1, c, od, oh, ow);
#endif
    VECT_N(FLT_ACC_DATA_T) sum0;
    VECT_N(FLT_ACC_DATA_T) sum1;
#if WITH_SUM
    sum0 = read_vect_c_block(
            0, &dst[dst_off0], c, dst_stride, dst_chunks_per_c_block);
#if UNROLL_MB_COUNT > 1
    sum1 = read_vect_c_block(
            0, &dst[dst_off1], c, dst_stride, dst_chunks_per_c_block);
#else
    sum1 = read_vect_c_block(
            1, &dst[dst_off0], c, dst_stride, dst_chunks_per_c_block);
#endif
#endif

    const int local_id = get_sub_group_local_id();

#if VECT_DT_N == 1
    const off_t po_mb = mb0;
    const off_t po_oc = c + local_id;
    if (po_oc < C_WO_PADDING) {
        POST_OP_DATA_T po_sum0 = sum0;
        POST_OP_DATA_T po_D0 = D0;
        APPLY_POST_OPS_SERIAL(po_D0, po_sum0, po_mb, po_oc, 0, 0, 0, 0);
        D0 = po_D0;

        POST_OP_DATA_T po_sum1 = sum1;
        POST_OP_DATA_T po_D1 = D1;
        APPLY_POST_OPS_SERIAL(po_D1, po_sum1, po_mb, po_oc, 0, 0, 0, 0);
        D1 = po_D1;
    }

#else
    for (int idx = 0; idx < VECT_DT_N; ++idx) {
#if USE_MB_C_BLOCK
        int c_sub_block_id = idx % CHUNKS_PER_C_BLOCK;
        int mb_sub_block_id = idx / CHUNKS_PER_C_BLOCK;
        off_t po_oc = c + c_sub_block_id * SUB_GROUP_SIZE + local_id;
        off_t po_mb = (mb0 + mb_sub_block_id);
#else // USE_MB_C_BLOCK
        off_t po_oc = c + idx * SUB_GROUP_SIZE + local_id;
        off_t po_mb = mb0;
#endif // USE_MB_C_BLOCK

        POST_OP_DATA_T d0_i = D0[idx];
        POST_OP_DATA_T sum0_i = sum0[idx];
        if (po_mb >= MB_WO_PADDING || po_oc >= C_WO_PADDING) {
            D0[idx] = 0;
            WS0[idx] = 0;
        } else {
            APPLY_POST_OPS_SERIAL(d0_i, sum0_i, po_mb, po_oc, 0, 0, 0, 0);
            D0[idx] = d0_i;
        }

        POST_OP_DATA_T d1_i = D1[idx];
        POST_OP_DATA_T sum1_i = sum1[idx];
        if (UNROLL_MB_COUNT > 1)
            po_mb += MB / 2;
        else {
            if (USE_MB_C_BLOCK)
                po_oc += (VECT_DT_N % CHUNKS_PER_C_BLOCK) * SUB_GROUP_SIZE;
            else
                po_oc += VECT_DT_N * SUB_GROUP_SIZE;
        }
        if (po_mb >= MB_WO_PADDING || po_oc >= C_WO_PADDING) {
            D1[idx] = 0;
            WS1[idx] = 0;
        } else {
            APPLY_POST_OPS_SERIAL(d1_i, sum1_i, po_mb, po_oc, 0, 0, 0, 0);
            D1[idx] = d1_i;
        }
    }
#endif // #if VECT_DT_N == 1
    write_vect_c_block(
            0, &dst[dst_off0], c, dst_stride, dst_chunks_per_c_block, D0);
#if UNROLL_MB_COUNT > 1
    write_vect_c_block(
            0, &dst[dst_off1], c, dst_stride, dst_chunks_per_c_block, D1);
#else
    write_vect_c_block(
            1, &dst[dst_off0], c, dst_stride, dst_chunks_per_c_block, D1);
#endif

#if ALG_MAX && IS_TRAINING
    off_t ws_off0 = dst_off0;
#if UNROLL_MB_COUNT > 1
    off_t ws_off1 = dst_off1;
#endif
    write_vect_c_block_int(
            0, &ws[ws_off0], c, ws_stride, ws_chunks_per_c_block, WS0);
#if UNROLL_MB_COUNT > 1
    write_vect_c_block_int(
            0, &ws[ws_off1], c, ws_stride, ws_chunks_per_c_block, WS1);
#else
    write_vect_c_block_int(
            1, &ws[ws_off0], c, ws_stride, ws_chunks_per_c_block, WS1);
#endif
#endif // ALG_MAX && IS_TRAINING
}
#endif

#if IS_BWD
KERNEL_ATTR
__kernel void xe_pooling_bwd(__global DATA_T *diff_src, __global int *ws,
        __global DATA_T *diff_dst) {

    if (GWS_OVERFLOW) return;

    const off_t mb0 = GWS_GET_MB();
#if UNROLL_MB_COUNT > 1
    off_t mb[UNROLL_MB_COUNT];
    mb[0] = GWS_GET_MB();
    unroll_for(int i = 1; i < UNROLL_MB_COUNT; i++) {
        mb[i] = mb[i - 1] + MB / UNROLL_MB_COUNT;
    }
#endif
    const off_t c = GWS_GET_C();
    const off_t id = GWS_GET_ID();
    const off_t ih = GWS_GET_IH();
    const off_t iw = GWS_GET_IW();

    // Calculate number of subgroup chunks inside C block
    // and stride between consecutive MB/C blocks
#if USE_MB_C_BLOCK
    const off_t src_stride = (SRC_SB0 > 1) ? SRC_SB0 : SRC_S0;
    const off_t dst_stride = (DST_SB0 > 1) ? DST_SB0 : DST_S0;
    const int src_chunks_per_c_block = CHUNKS_PER_C_BLOCK;
    const int dst_chunks_per_c_block = CHUNKS_PER_C_BLOCK;
#elif USE_ONLY_C_BLOCK
    const off_t src_stride = (SRC_B1 > 1) ? SRC_S1 : SUB_GROUP_SIZE;
    const off_t dst_stride = (DST_B1 > 1) ? DST_S1 : SUB_GROUP_SIZE;
    const int src_chunks_per_c_block
            = (SRC_B1 > 1) ? (SRC_B1 / SUB_GROUP_SIZE) : 1;
    const int dst_chunks_per_c_block
            = (DST_B1 > 1) ? (DST_B1 / SUB_GROUP_SIZE) : 1;
#endif

    const off_t ws_stride = dst_stride;
    const int ws_chunks_per_c_block = dst_chunks_per_c_block;

    VECT_N(FLT_ACC_DATA_T) S0 = 0, S1 = 0;
#if UNROLL_MB_COUNT > 1
    VECT_N(FLT_ACC_DATA_T) S[UNROLL_MB_COUNT] = {0};
#endif
    for (int kd = 0; kd < KD; kd++) {
        off_t od = (id + PD - kd);
        if (od % SD != 0) continue;
        od /= SD;
        if (od < 0 || od >= OD) continue;

        for (int kh = 0; kh < KH; kh++) {
            off_t oh = (ih + PH - kh);
            if (oh % SH != 0) continue;
            oh /= SH;
            if (oh < 0 || oh >= OH) continue;

            for (int kw = 0; kw < KW; kw++) {
                off_t ow = (iw + PW - kw);
                if (ow % SW != 0) continue;
                ow /= SW;
                if (ow < 0 || ow >= OW) continue;

                const off_t dst_off0 = DST_OFF(mb0, c, od, oh, ow);
#if UNROLL_MB_COUNT > 1
                off_t dst_off[UNROLL_MB_COUNT];
                unroll_for(int i = 0; i < UNROLL_MB_COUNT; i++) {
                    dst_off[i] = DST_OFF(mb[i], c, od, oh, ow);
                }
#endif
                VECT_N(FLT_ACC_DATA_T)
                D0 = read_vect_c_block(0, &diff_dst[dst_off0], c, dst_stride,
                        dst_chunks_per_c_block);
                VECT_N(FLT_ACC_DATA_T)
                D1 = read_vect_c_block(1, &diff_dst[dst_off0], c, dst_stride,
                        dst_chunks_per_c_block);
#if UNROLL_MB_COUNT > 1
                VECT_N(FLT_ACC_DATA_T) D[UNROLL_MB_COUNT];
                unroll_for(int i = 0; i < UNROLL_MB_COUNT; i++) {
                    D[i] = read_vect_c_block(0, &diff_dst[dst_off[i]], c,
                            dst_stride, dst_chunks_per_c_block);
                }
#endif

#if ALG_MAX
                VECT_N(int)
                WS0 = read_vect_c_block_int(
                        0, &ws[dst_off0], c, ws_stride, ws_chunks_per_c_block);
                VECT_N(int)
                WS1 = read_vect_c_block_int(
                        1, &ws[dst_off0], c, ws_stride, ws_chunks_per_c_block);
#if UNROLL_MB_COUNT > 1
                VECT_N(int) WS[UNROLL_MB_COUNT];
                unroll_for(int i = 0; i < UNROLL_MB_COUNT; i++) {
                    WS[i] = read_vect_c_block_int(0, &ws[dst_off[i]], c,
                            ws_stride, ws_chunks_per_c_block);
                }
#endif

                // select(FLT_ACC_DATA_TN, ..., intN) is invalid for double
                // (needs longN mask). Multiply by abs(isequal(...)) instead:
                // isequal returns -1 (equal) or 0 for vectors, 1 or 0 for
                // scalars; abs() normalises both to 1/0.
                VECT_N(int)
                CMP0 = isequal(CONCAT2(as_, VECT_FLOAT_T)(
                                       WS0 - kd * KH * KW - kh * KW - kw),
                        (VECT_FLOAT_T)0);
                D0 *= CONCAT2(convert_, VECT_N(FLT_ACC_DATA_T))(abs(CMP0));

                VECT_N(int)
                CMP1 = isequal(CONCAT2(as_, VECT_FLOAT_T)(
                                       WS1 - kd * KH * KW - kh * KW - kw),
                        (VECT_FLOAT_T)0);
                D1 *= CONCAT2(convert_, VECT_N(FLT_ACC_DATA_T))(abs(CMP1));

#if UNROLL_MB_COUNT > 1
                VECT_N(int) CMP[UNROLL_MB_COUNT];
                unroll_for(int i = 0; i < UNROLL_MB_COUNT; i++) {
                    CMP[i] = isequal(CONCAT2(as_, VECT_FLOAT_T)(WS[i]
                                             - kd * KH * KW - kh * KW - kw),
                            (VECT_FLOAT_T)0);
                    D[i] *= CONCAT2(convert_, VECT_N(FLT_ACC_DATA_T))(
                            abs(CMP[i]));
                }
#endif
#endif
#if ALG_AVG_NP
                const off_t id_start = max(id - kd, (off_t)0);
                const off_t ih_start = max(ih - kh, (off_t)0);
                const off_t iw_start = max(iw - kw, (off_t)0);
                const off_t id_end = min(id - kd + KD, (off_t)ID);
                const off_t ih_end = min(ih - kh + KH, (off_t)IH);
                const off_t iw_end = min(iw - kw + KW, (off_t)IW);
                const int num_summands = (int)(ih_end - ih_start)
                        * (int)(iw_end - iw_start) * (int)(id_end - id_start);
                D0 /= num_summands;
                D1 /= num_summands;
#if UNROLL_MB_COUNT > 1
                unroll_for(int i = 0; i < UNROLL_MB_COUNT; i++) {
                    D[i] /= num_summands;
                }
#endif
#endif
                S0 += D0;
                S1 += D1;
#if UNROLL_MB_COUNT > 1
                unroll_for(int i = 0; i < UNROLL_MB_COUNT; i++) {
                    S[i] += D[i];
                }
#endif
            }
        }
    }
#if ALG_AVG_P
    S0 /= KD * KH * KW;
    S1 /= KD * KH * KW;
#if UNROLL_MB_COUNT > 1
    unroll_for(int i = 0; i < UNROLL_MB_COUNT; i++) {
        S[i] /= KD * KH * KW;
    }
#endif
#endif

    off_t src_off0 = SRC_OFF(mb0, c, id, ih, iw);
#if UNROLL_MB_COUNT > 1
    off_t src_off[UNROLL_MB_COUNT];
    unroll_for(int i = 0; i < UNROLL_MB_COUNT; i++) {
        src_off[i] = SRC_OFF(mb[i], c, id, ih, iw);
    }
#endif
    write_vect_c_block(
            0, &diff_src[src_off0], c, src_stride, src_chunks_per_c_block, S0);
#if UNROLL_MB_COUNT > 1
    unroll_for(int i = 0; i < UNROLL_MB_COUNT; i++) {
        write_vect_c_block(0, &diff_src[src_off[i]], c, src_stride,
                src_chunks_per_c_block, S[i]);
    }
#else
    write_vect_c_block(
            1, &diff_src[src_off0], c, src_stride, src_chunks_per_c_block, S1);
#endif
}
#endif

inline FLT_ACC_DATA_T read_c_block(const __global DATA_T *ptr, off_t c) {
#if C_W_PADDING % SUB_GROUP_SIZE != 0
    int local_id = get_sub_group_local_id();
    off_t tail = C_WO_PADDING - c;
    FLT_ACC_DATA_T result;
    return (local_id < tail) ? load(result, ptr, local_id) : (FLT_ACC_DATA_T)0;
#else
    FLT_ACC_DATA_T result = block_load(result, ptr);
    return result;
#endif
}

#define CALC_VECT_LEN() \
    ({ \
        off_t size; \
        if (USE_ONLY_C_BLOCK == 1 \
                && VECT_DT_N > C_WO_PADDING / SUB_GROUP_SIZE + 1) \
            size = C_WO_PADDING / SUB_GROUP_SIZE + 1; \
        else \
            size = VECT_DT_N; \
        size; \
    })

inline VECT_N(FLT_ACC_DATA_T)
        read_vect_c_block(int idx, const __global DATA_T *ptr, off_t c,
                off_t blocks_stride, int chunks_per_block) {
    if (idx >= NVECT) return 0;

    if ((blocks_stride == chunks_per_block * SUB_GROUP_SIZE)
            && (C_WO_PADDING % (chunks_per_block * SUB_GROUP_SIZE) == 0)) {
        VECT_N(FLT_ACC_DATA_T) result;
        block_load(&result, ptr + idx * VECT_DT_N * SUB_GROUP_SIZE);
        return result;
    } else {
        VECT_N(FLT_ACC_DATA_T) ret;
        for (int i = 0; i < CALC_VECT_LEN(); i++) {
            const int offset_index = (idx * VECT_DT_N + i);
            const int local_c_block_index = offset_index % chunks_per_block;
            const int global_c_block_index = offset_index / chunks_per_block;
            const off_t ptr_offset = local_c_block_index * SUB_GROUP_SIZE
                    + global_c_block_index * blocks_stride;
            const int c_off
                    = (USE_ONLY_C_BLOCK ? offset_index * SUB_GROUP_SIZE
                                        : local_c_block_index * SUB_GROUP_SIZE);
#if VECT_DT_N == 1
            ret = read_c_block(ptr + ptr_offset, c + c_off);
#else
            ret[i] = read_c_block(ptr + ptr_offset, c + c_off);
#endif
        }
#if VECT_DT_N > 1
        for (int i = CALC_VECT_LEN(); i < VECT_DT_N; ++i) {
            ret[i] = 0;
        }
#endif
        return ret;
    }
}

inline int read_c_block_int(const __global int *ptr, off_t c) {
#if C_W_PADDING % SUB_GROUP_SIZE != 0
    int local_id = get_sub_group_local_id();
    off_t tail = C_WO_PADDING - c;
    return (local_id < tail) ? ptr[local_id] : 0;
#else
    int result;
    block_load(&result, ptr);
    return result;
#endif
}

inline VECT_N(int) read_vect_c_block_int(int idx, const __global int *ptr,
        off_t c, off_t blocks_stride, int chunks_per_block) {
    if (idx >= NVECT) return 0;

    if ((blocks_stride == chunks_per_block * SUB_GROUP_SIZE)
            && (C_WO_PADDING % (chunks_per_block * SUB_GROUP_SIZE) == 0)) {
        VECT_N(int) result;
        block_load(&result, ptr + idx * VECT_DT_N * SUB_GROUP_SIZE);
        return result;
    } else {
        VECT_N(int) ret;
        for (int i = 0; i < VECT_DT_N; i++) {
            const int offset_index = (idx * VECT_DT_N + i);
            const int local_c_block_index = offset_index % chunks_per_block;
            const int global_c_block_index = offset_index / chunks_per_block;
            const off_t ptr_offset = local_c_block_index * SUB_GROUP_SIZE
                    + global_c_block_index * blocks_stride;
            const int c_off
                    = (USE_ONLY_C_BLOCK ? offset_index * SUB_GROUP_SIZE
                                        : local_c_block_index * SUB_GROUP_SIZE);
#if VECT_DT_N == 1
            ret = read_c_block_int(ptr + ptr_offset, c + c_off);
#else
            ret[i] = read_c_block_int(ptr + ptr_offset, c + c_off);
#endif
        }
        return ret;
    }
}

inline void write_c_block(__global DATA_T *ptr, off_t c, FLT_ACC_DATA_T value) {
#if C_W_PADDING % SUB_GROUP_SIZE != 0
    int local_id = get_sub_group_local_id();
    off_t tail = C_WO_PADDING - c;

    if (local_id < tail) write(ptr + local_id, value);
#else
#if C_WO_PADDING % SUB_GROUP_SIZE != 0
    int local_id = get_sub_group_local_id();
    if (local_id >= C_WO_PADDING - c && local_id < C_W_PADDING - c) value = 0;
#endif
    if (c >= C_WO_PADDING) {
        FLT_ACC_DATA_T zero = 0;
        block_write(ptr, &zero);
        return;
    }
    block_write(ptr, &value);
#endif
}

inline void write_vect_c_block(int idx, __global DATA_T *ptr, off_t c,
        off_t blocks_stride, int chunks_per_block,
        VECT_N(FLT_ACC_DATA_T) block) {
    if (idx >= NVECT) return;

    if ((blocks_stride == chunks_per_block * SUB_GROUP_SIZE)
            && (C_WO_PADDING % (chunks_per_block * SUB_GROUP_SIZE) == 0)) {
        block_write(ptr + idx * VECT_DT_N * SUB_GROUP_SIZE, &block);
    } else {
        for (int i = 0; i < VECT_DT_N; i++) {
            const int offset_index = (idx * VECT_DT_N + i);
            const int local_c_block_index = offset_index % chunks_per_block;
            const int global_c_block_index = offset_index / chunks_per_block;
            const off_t ptr_offset = local_c_block_index * SUB_GROUP_SIZE
                    + global_c_block_index * blocks_stride;
            const int c_off
                    = (USE_ONLY_C_BLOCK ? offset_index * SUB_GROUP_SIZE
                                        : local_c_block_index * SUB_GROUP_SIZE);
#if VECT_DT_N == 1
            write_c_block(ptr + ptr_offset, c + c_off, block);
#else
            write_c_block(ptr + ptr_offset, c + c_off, block[i]);
#endif
        }
    }
}

inline void write_c_block_int(__global int *ptr, off_t c, int value) {
#if C_WO_PADDING % SUB_GROUP_SIZE != 0
    int local_id = get_sub_group_local_id();
    off_t tail = C_WO_PADDING - c;
    if (local_id < tail)
        ptr[local_id] = value;
    else if (local_id < C_W_PADDING - c) {
        ptr[local_id] = 0;
    } else
        return;
#else
    if (c >= C_WO_PADDING) {
        int zero = 0;
        block_write(ptr, &zero);
        return;
    }
    block_write(ptr, &value);
#endif
}

inline void write_vect_c_block_int(int idx, __global int *ptr, off_t c,
        off_t blocks_stride, int chunks_per_block, VECT_N(int) block) {
    if (idx >= NVECT) return;

    if ((blocks_stride == chunks_per_block * SUB_GROUP_SIZE)
            && (C_WO_PADDING % (chunks_per_block * SUB_GROUP_SIZE) == 0)) {
        block_write(ptr + idx * VECT_DT_N * SUB_GROUP_SIZE, &block);
    } else {
        for (int i = 0; i < VECT_DT_N; i++) {
            const int offset_index = (idx * VECT_DT_N + i);
            const int local_c_block_index = offset_index % chunks_per_block;
            const int global_c_block_index = offset_index / chunks_per_block;
            const off_t ptr_offset = local_c_block_index * SUB_GROUP_SIZE
                    + global_c_block_index * blocks_stride;
            const int c_off
                    = (USE_ONLY_C_BLOCK ? offset_index * SUB_GROUP_SIZE
                                        : local_c_block_index * SUB_GROUP_SIZE);
#if VECT_DT_N == 1
            write_c_block_int(ptr + ptr_offset, c + c_off, block);
#else
            write_c_block_int(ptr + ptr_offset, c + c_off, block[i]);
#endif
        }
    }
}
