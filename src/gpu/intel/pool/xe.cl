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

// Read functions. All operate in float to avoid type-specific branches.
inline float read_c_block_f(const __global DATA_T *ptr, off_t c);
inline void read_vect_c_block_f(float *ret, int idx, const __global DATA_T *ptr,
        off_t c, off_t blocks_stride, int chunks_per_block);
inline int read_c_block_int(const __global int *ptr, off_t c);
inline void read_vect_c_block_int(int *ret, int idx, const __global int *ptr,
        off_t c, off_t blocks_stride, int chunks_per_block);

// Write functions.
inline void write_c_block_f(__global DATA_T *ptr, off_t c, float value);
inline void write_vect_c_block_f(int idx, __global DATA_T *ptr, off_t c,
        off_t blocks_stride, int chunks_per_block, float *block);
inline void write_c_block_int(__global int *ptr, off_t c, int value);
inline void write_vect_c_block_int(int idx, __global int *ptr, off_t c,
        off_t blocks_stride, int chunks_per_block, int *block);

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
        float dst_zero[VECT_DT_N];
        for (int i = 0; i < VECT_DT_N; i++)
            dst_zero[i] = 0.0f;
        int ws_zero[VECT_DT_N];
        for (int i = 0; i < VECT_DT_N; i++)
            ws_zero[i] = 0;
        off_t off = DST_OFF(mb0, c, od, oh, ow);
        write_vect_c_block_f(
                0, &dst[off], c, dst_stride, dst_chunks_per_c_block, dst_zero);
        write_vect_c_block_f(
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

    const float d_init = ALG_MAX ? DATA_TO_REF(DATA_MIN) : 0.0f;
    float D0[VECT_DT_N], D1[VECT_DT_N];
    for (int i = 0; i < VECT_DT_N; i++)
        D0[i] = D1[i] = d_init;
    int WS0[VECT_DT_N], WS1[VECT_DT_N];
    for (int i = 0; i < VECT_DT_N; i++)
        WS0[i] = WS1[i] = 0;

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
                float S0[VECT_DT_N], S1[VECT_DT_N];
                read_vect_c_block_f(S0, 0, &src[src_off0], c, src_stride,
                        src_chunks_per_c_block);
#if UNROLL_MB_COUNT > 1
                read_vect_c_block_f(S1, 0, &src[src_off1], c, src_stride,
                        src_chunks_per_c_block);
#else
                read_vect_c_block_f(S1, 1, &src[src_off0], c, src_stride,
                        src_chunks_per_c_block);
#endif

#if ALG_MAX
#if IS_TRAINING
                for (int i = 0; i < VECT_DT_N; i++) {
                    if (D0[i] < S0[i]) {
                        WS0[i] = kd * KH * KW + kh * KW + kw;
                        D0[i] = S0[i];
                    }
                    if (D1[i] < S1[i]) {
                        WS1[i] = kd * KH * KW + kh * KW + kw;
                        D1[i] = S1[i];
                    }
                }
#else // IS_TRAINING
                for (int i = 0; i < VECT_DT_N; i++) {
                    D0[i] = max(D0[i], S0[i]);
                    D1[i] = max(D1[i], S1[i]);
                }
#endif // IS_TRAINING
#else // ALG_MAX
                for (int i = 0; i < VECT_DT_N; i++) {
                    D0[i] += S0[i];
                    D1[i] += S1[i];
                }
#endif // ALG_MAX
            }
        }
    }

#if ALG_AVG_P
    for (int i = 0; i < VECT_DT_N; i++) {
        D0[i] /= KD * KH * KW;
        D1[i] /= KD * KH * KW;
    }
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
    for (int i = 0; i < VECT_DT_N; i++) {
        D0[i] /= num_summands;
        D1[i] /= num_summands;
    }
#endif // ALG_AVG_NP

    off_t dst_off0 = DST_OFF(mb0, c, od, oh, ow);
#if UNROLL_MB_COUNT > 1
    off_t dst_off1 = DST_OFF(mb1, c, od, oh, ow);
#endif
    float sum0[VECT_DT_N], sum1[VECT_DT_N];
    for (int i = 0; i < VECT_DT_N; i++)
        sum0[i] = sum1[i] = 0.0f;
#if WITH_SUM
    read_vect_c_block_f(
            sum0, 0, &dst[dst_off0], c, dst_stride, dst_chunks_per_c_block);
#if UNROLL_MB_COUNT > 1
    read_vect_c_block_f(
            sum1, 0, &dst[dst_off1], c, dst_stride, dst_chunks_per_c_block);
#else
    read_vect_c_block_f(
            sum1, 1, &dst[dst_off0], c, dst_stride, dst_chunks_per_c_block);
#endif
#endif // WITH_SUM

    const int local_id = get_sub_group_local_id();

#if VECT_DT_N == 1
    const off_t po_mb = mb0;
    const off_t po_oc = c + local_id;
    if (po_oc < C_WO_PADDING) {
        float po_D0 = D0[0];
        APPLY_POST_OPS_SERIAL(po_D0, sum0[0], po_mb, po_oc, 0, 0, 0, 0);
        D0[0] = po_D0;
        float po_D1 = D1[0];
        APPLY_POST_OPS_SERIAL(po_D1, sum1[0], po_mb, po_oc, 0, 0, 0, 0);
        D1[0] = po_D1;
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

        if (po_mb >= MB_WO_PADDING || po_oc >= C_WO_PADDING) {
            D0[idx] = 0.0f;
            WS0[idx] = 0;
        } else {
            float d0 = D0[idx];
            APPLY_POST_OPS_SERIAL(d0, sum0[idx], po_mb, po_oc, 0, 0, 0, 0);
            D0[idx] = d0;
        }

#if UNROLL_MB_COUNT > 1
        po_mb += MB / 2;
#else
#if USE_MB_C_BLOCK
        po_oc += (VECT_DT_N % CHUNKS_PER_C_BLOCK) * SUB_GROUP_SIZE;
#else
        po_oc += VECT_DT_N * SUB_GROUP_SIZE;
#endif
#endif
        if (po_mb >= MB_WO_PADDING || po_oc >= C_WO_PADDING) {
            D1[idx] = 0.0f;
            WS1[idx] = 0;
        } else {
            float d1 = D1[idx];
            APPLY_POST_OPS_SERIAL(d1, sum1[idx], po_mb, po_oc, 0, 0, 0, 0);
            D1[idx] = d1;
        }
    }
#endif // VECT_DT_N == 1

    write_vect_c_block_f(
            0, &dst[dst_off0], c, dst_stride, dst_chunks_per_c_block, D0);
#if UNROLL_MB_COUNT > 1
    write_vect_c_block_f(
            0, &dst[dst_off1], c, dst_stride, dst_chunks_per_c_block, D1);
#else
    write_vect_c_block_f(
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
#endif // IS_FWD

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

    float S0[VECT_DT_N], S1[VECT_DT_N];
    for (int i = 0; i < VECT_DT_N; i++)
        S0[i] = S1[i] = 0.0f;
#if UNROLL_MB_COUNT > 1
    float S[UNROLL_MB_COUNT][VECT_DT_N];
    for (int i = 0; i < UNROLL_MB_COUNT; i++)
        for (int j = 0; j < VECT_DT_N; j++)
            S[i][j] = 0.0f;
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
                float D0[VECT_DT_N], D1[VECT_DT_N];
                read_vect_c_block_f(D0, 0, &diff_dst[dst_off0], c, dst_stride,
                        dst_chunks_per_c_block);
                read_vect_c_block_f(D1, 1, &diff_dst[dst_off0], c, dst_stride,
                        dst_chunks_per_c_block);
#if UNROLL_MB_COUNT > 1
                float D[UNROLL_MB_COUNT][VECT_DT_N];
                unroll_for(int i = 0; i < UNROLL_MB_COUNT; i++) {
                    read_vect_c_block_f(D[i], 0, &diff_dst[dst_off[i]], c,
                            dst_stride, dst_chunks_per_c_block);
                }
#endif

#if ALG_MAX
                int WS0[VECT_DT_N], WS1[VECT_DT_N];
                read_vect_c_block_int(WS0, 0, &ws[dst_off0], c, ws_stride,
                        ws_chunks_per_c_block);
                read_vect_c_block_int(WS1, 1, &ws[dst_off0], c, ws_stride,
                        ws_chunks_per_c_block);
#if UNROLL_MB_COUNT > 1
                int WS[UNROLL_MB_COUNT][VECT_DT_N];
                unroll_for(int i = 0; i < UNROLL_MB_COUNT; i++) {
                    read_vect_c_block_int(WS[i], 0, &ws[dst_off[i]], c,
                            ws_stride, ws_chunks_per_c_block);
                }
#endif
                const int ws_target = kd * KH * KW + kh * KW + kw;
                for (int i = 0; i < VECT_DT_N; i++) {
                    if (WS0[i] != ws_target) D0[i] = 0.0f;
                    if (WS1[i] != ws_target) D1[i] = 0.0f;
                }
#if UNROLL_MB_COUNT > 1
                unroll_for(int i = 0; i < UNROLL_MB_COUNT; i++) {
                    for (int j = 0; j < VECT_DT_N; j++) {
                        if (WS[i][j] != ws_target) D[i][j] = 0.0f;
                    }
                }
#endif
#endif // ALG_MAX

#if ALG_AVG_NP
                const off_t id_start = max(id - kd, (off_t)0);
                const off_t ih_start = max(ih - kh, (off_t)0);
                const off_t iw_start = max(iw - kw, (off_t)0);
                const off_t id_end = min(id - kd + KD, (off_t)ID);
                const off_t ih_end = min(ih - kh + KH, (off_t)IH);
                const off_t iw_end = min(iw - kw + KW, (off_t)IW);
                const int num_summands = (int)(ih_end - ih_start)
                        * (int)(iw_end - iw_start) * (int)(id_end - id_start);
                for (int i = 0; i < VECT_DT_N; i++) {
                    D0[i] /= num_summands;
                    D1[i] /= num_summands;
                }
#if UNROLL_MB_COUNT > 1
                unroll_for(int i = 0; i < UNROLL_MB_COUNT; i++) {
                    for (int j = 0; j < VECT_DT_N; j++)
                        D[i][j] /= num_summands;
                }
#endif
#endif // ALG_AVG_NP

#if UNROLL_MB_COUNT > 1
                unroll_for(int i = 0; i < UNROLL_MB_COUNT; i++) {
                    for (int j = 0; j < VECT_DT_N; j++)
                        S[i][j] += D[i][j];
                }
#else
                for (int i = 0; i < VECT_DT_N; i++) {
                    S0[i] += D0[i];
                    S1[i] += D1[i];
                }
#endif
            }
        }
    }

#if ALG_AVG_P
#if UNROLL_MB_COUNT > 1
    unroll_for(int i = 0; i < UNROLL_MB_COUNT; i++) {
        for (int j = 0; j < VECT_DT_N; j++)
            S[i][j] /= KD * KH * KW;
    }
#else
    for (int i = 0; i < VECT_DT_N; i++) {
        S0[i] /= KD * KH * KW;
        S1[i] /= KD * KH * KW;
    }
#endif
#endif // ALG_AVG_P

    off_t src_off0 = SRC_OFF(mb0, c, id, ih, iw);
#if UNROLL_MB_COUNT > 1
    off_t src_off[UNROLL_MB_COUNT];
    unroll_for(int i = 0; i < UNROLL_MB_COUNT; i++) {
        src_off[i] = SRC_OFF(mb[i], c, id, ih, iw);
    }
    unroll_for(int i = 0; i < UNROLL_MB_COUNT; i++) {
        write_vect_c_block_f(0, &diff_src[src_off[i]], c, src_stride,
                src_chunks_per_c_block, S[i]);
    }
#else
    write_vect_c_block_f(
            0, &diff_src[src_off0], c, src_stride, src_chunks_per_c_block, S0);
    write_vect_c_block_f(
            1, &diff_src[src_off0], c, src_stride, src_chunks_per_c_block, S1);
#endif
}
#endif // IS_BWD

// ===== Helper function implementations =====

inline float read_c_block_f(const __global DATA_T *ptr, off_t c) {
#if C_W_PADDING % SUB_GROUP_SIZE != 0
    int local_id = get_sub_group_local_id();
    off_t tail = C_WO_PADDING - c;
    return (local_id < tail) ? DATA_TO_REF(ptr[local_id]) : 0.0f;
#else
    float val;
    block_load(&val, (__global DATA_T *)ptr);
    return val;
#endif
}

inline void read_vect_c_block_f(float *ret, int idx, const __global DATA_T *ptr,
        off_t c, off_t blocks_stride, int chunks_per_block) {
    if (idx >= NVECT) {
        for (int i = 0; i < VECT_DT_N; i++)
            ret[i] = 0.0f;
        return;
    }
    if ((blocks_stride == chunks_per_block * SUB_GROUP_SIZE)
            && (C_WO_PADDING % (chunks_per_block * SUB_GROUP_SIZE) == 0)) {
        block_load(ret,
                (__global DATA_T *)ptr + idx * VECT_DT_N * SUB_GROUP_SIZE,
                VECT_DT_N);
    } else {
        for (int i = 0; i < CALC_VECT_LEN(); i++) {
            const int offset_index = (idx * VECT_DT_N + i);
            const int local_c_block_index = offset_index % chunks_per_block;
            const int global_c_block_index = offset_index / chunks_per_block;
            const off_t ptr_offset = local_c_block_index * SUB_GROUP_SIZE
                    + global_c_block_index * blocks_stride;
            const int c_off = USE_ONLY_C_BLOCK
                    ? offset_index * SUB_GROUP_SIZE
                    : local_c_block_index * SUB_GROUP_SIZE;
            ret[i] = read_c_block_f(ptr + ptr_offset, c + c_off);
        }
        for (int i = CALC_VECT_LEN(); i < VECT_DT_N; i++)
            ret[i] = 0.0f;
    }
}

inline int read_c_block_int(const __global int *ptr, off_t c) {
#if C_W_PADDING % SUB_GROUP_SIZE != 0
    int local_id = get_sub_group_local_id();
    off_t tail = C_WO_PADDING - c;
    return (local_id < tail) ? ptr[local_id] : 0;
#else
    int val;
    block_load(&val, (__global int *)ptr);
    return val;
#endif
}

inline void read_vect_c_block_int(int *ret, int idx, const __global int *ptr,
        off_t c, off_t blocks_stride, int chunks_per_block) {
    if (idx >= NVECT) {
        for (int i = 0; i < VECT_DT_N; i++)
            ret[i] = 0;
        return;
    }
    if ((blocks_stride == chunks_per_block * SUB_GROUP_SIZE)
            && (C_WO_PADDING % (chunks_per_block * SUB_GROUP_SIZE) == 0)) {
        block_load(ret, (__global int *)ptr + idx * VECT_DT_N * SUB_GROUP_SIZE,
                VECT_DT_N);
    } else {
        for (int i = 0; i < VECT_DT_N; i++) {
            const int offset_index = (idx * VECT_DT_N + i);
            const int local_c_block_index = offset_index % chunks_per_block;
            const int global_c_block_index = offset_index / chunks_per_block;
            const off_t ptr_offset = local_c_block_index * SUB_GROUP_SIZE
                    + global_c_block_index * blocks_stride;
            const int c_off = USE_ONLY_C_BLOCK
                    ? offset_index * SUB_GROUP_SIZE
                    : local_c_block_index * SUB_GROUP_SIZE;
            ret[i] = read_c_block_int(ptr + ptr_offset, c + c_off);
        }
    }
}

inline void write_c_block_f(__global DATA_T *ptr, off_t c, float value) {
#if C_W_PADDING % SUB_GROUP_SIZE != 0
    int local_id = get_sub_group_local_id();
    off_t tail = C_WO_PADDING - c;
    if (local_id < tail) write(ptr + local_id, value);
#else
#if C_WO_PADDING % SUB_GROUP_SIZE != 0
    int local_id = get_sub_group_local_id();
    if (local_id >= C_WO_PADDING - c && local_id < C_W_PADDING - c)
        value = 0.0f;
#endif
    if (c >= C_WO_PADDING) {
        float zero = 0.0f;
        block_write(ptr, &zero, 1);
        return;
    }
    block_write(ptr, &value, 1);
#endif
}

inline void write_vect_c_block_f(int idx, __global DATA_T *ptr, off_t c,
        off_t blocks_stride, int chunks_per_block, float *block) {
    if (idx >= NVECT) return;

    if ((blocks_stride == chunks_per_block * SUB_GROUP_SIZE)
            && (C_WO_PADDING % (chunks_per_block * SUB_GROUP_SIZE) == 0)) {
        block_write(ptr + idx * VECT_DT_N * SUB_GROUP_SIZE, block, VECT_DT_N);
    } else {
        for (int i = 0; i < VECT_DT_N; i++) {
            const int offset_index = (idx * VECT_DT_N + i);
            const int local_c_block_index = offset_index % chunks_per_block;
            const int global_c_block_index = offset_index / chunks_per_block;
            const off_t ptr_offset = local_c_block_index * SUB_GROUP_SIZE
                    + global_c_block_index * blocks_stride;
            const int c_off = USE_ONLY_C_BLOCK
                    ? offset_index * SUB_GROUP_SIZE
                    : local_c_block_index * SUB_GROUP_SIZE;
            write_c_block_f(ptr + ptr_offset, c + c_off, block[i]);
        }
    }
}

inline void write_c_block_int(__global int *ptr, off_t c, int value) {
#if C_WO_PADDING % SUB_GROUP_SIZE != 0
    int local_id = get_sub_group_local_id();
    off_t tail = C_WO_PADDING - c;
    if (local_id < tail)
        ptr[local_id] = value;
    else if (local_id < C_W_PADDING - c)
        ptr[local_id] = 0;
#else
    if (c >= C_WO_PADDING) {
        int zero = 0;
        block_write((__global int *)ptr, &zero, 1);
        return;
    }
    block_write((__global int *)ptr, &value, 1);
#endif
}

inline void write_vect_c_block_int(int idx, __global int *ptr, off_t c,
        off_t blocks_stride, int chunks_per_block, int *block) {
    if (idx >= NVECT) return;

    if ((blocks_stride == chunks_per_block * SUB_GROUP_SIZE)
            && (C_WO_PADDING % (chunks_per_block * SUB_GROUP_SIZE) == 0)) {
        block_write((__global int *)ptr + idx * VECT_DT_N * SUB_GROUP_SIZE,
                block, VECT_DT_N);
    } else {
        for (int i = 0; i < VECT_DT_N; i++) {
            const int offset_index = (idx * VECT_DT_N + i);
            const int local_c_block_index = offset_index % chunks_per_block;
            const int global_c_block_index = offset_index / chunks_per_block;
            const off_t ptr_offset = local_c_block_index * SUB_GROUP_SIZE
                    + global_c_block_index * blocks_stride;
            const int c_off = USE_ONLY_C_BLOCK
                    ? offset_index * SUB_GROUP_SIZE
                    : local_c_block_index * SUB_GROUP_SIZE;
            write_c_block_int(ptr + ptr_offset, c + c_off, block[i]);
        }
    }
}
