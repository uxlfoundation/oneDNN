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

#ifndef GPU_INTEL_OCL_SDPA_UTILS_H
#define GPU_INTEL_OCL_SDPA_UTILS_H

#define _4D_OFF(tag, x0, x1, x2, x3) \
    (((x0) % tag##_B0) * tag##_SB0 + ((x0) / tag##_B0) * tag##_S0 \
            + ((x1) % tag##_B1) * tag##_SB1 + ((x1) / tag##_B1) * tag##_S1 \
            + ((x2) % tag##_B2) * tag##_SB2 + ((x2) / tag##_B2) * tag##_S2 \
            + ((x3) % tag##_B3) * tag##_SB3 + ((x3) / tag##_B3) * tag##_S3)

#define QRY_OFF(x0, x1, x2, x3) _4D_OFF(QRY, x0, x1, x2, x3)
#define KEY_OFF(x0, x1, x2, x3) _4D_OFF(KEY, x0, x1, x2, x3)
#define VAL_OFF(x0, x1, x2, x3) _4D_OFF(VAL, x0, x1, x2, x3)
#define MSK_OFF(x0, x1, x2, x3) _4D_OFF(MSK, x0, x1, x2, x3)

#define _BATCH_OFF(tag, x0, x1) \
    (((x0) % tag##_B0) * tag##_SB0 + ((x0) / tag##_B0) * tag##_S0 \
            + ((x1) % tag##_B1) * tag##_SB1 + ((x1) / tag##_B1) * tag##_S1)

#define QRY_BATCH(x0, x1) _BATCH_OFF(QRY, x0, x1)
#define KEY_BATCH(x0, x1) _BATCH_OFF(KEY, x0, x1)
#define VAL_BATCH(x0, x1) _BATCH_OFF(VAL, x0, x1)
#define DST_BATCH(x0, x1) _BATCH_OFF(DST, x0, x1)
#define MSK_BATCH(x0, x1) _BATCH_OFF(MSK, x0, x1)

#define REDUCE_STAGE_0(cat, f)
#define REDUCE_STAGE_1(cat, f) f(0)
#define REDUCE_STAGE_2(cat, f) cat(REDUCE_STAGE_1(cat, f), f(1))
#define REDUCE_STAGE_3(cat, f) cat(REDUCE_STAGE_2(cat, f), f(2))
#define REDUCE_STAGE_4(cat, f) cat(REDUCE_STAGE_3(cat, f), f(3))
#define REDUCE2(n, cat, f) REDUCE_STAGE_##n(cat, f)
#define REDUCE(n, cat, f) REDUCE2(n, cat, f)
#define INTERNAL_CAT(a, b) a##b
#define CAT(a, b) INTERNAL_CAT(a, b)
#define JOIN_COMMA(x, y) x, y
#define JOIN_SEMICOLON(x, y) \
    x; \
    y
#define CS_PARAM(p0, p1, p2, p3) \
    JOIN_COMMA(p0, JOIN_COMMA(p1, JOIN_COMMA(p2, p3)))

#define IDX(varname, n) const off_t varname##n

#define KEY_DIMS(n) IDX(KEY_D, n)
#define KEY_STRIDES(n) IDX(KEY_S, n)
#define KEY_BLOCKS(n) IDX(KEY_B, n)
#define KEY_BLOCK_STRIDES(n) IDX(KEY_SB, n)

#define QRY_DIMS(n) IDX(QRY_D, n)
#define QRY_STRIDES(n) IDX(QRY_S, n)
#define QRY_BLOCKS(n) IDX(QRY_B, n)
#define QRY_BLOCK_STRIDES(n) IDX(QRY_SB, n)

#define VAL_DIMS(n) IDX(VAL_D, n)
#define VAL_STRIDES(n) IDX(VAL_S, n)
#define VAL_BLOCKS(n) IDX(VAL_B, n)
#define VAL_BLOCK_STRIDES(n) IDX(VAL_SB, n)

#define DST_DIMS(n) IDX(DST_D, n)
#define DST_STRIDES(n) IDX(DST_S, n)
#define DST_BLOCKS(n) IDX(DST_B, n)
#define DST_BLOCK_STRIDES(n) IDX(DST_SB, n)

#define MSK_DIMS(n) IDX(MSK_D, n)
#define MSK_STRIDES(n) IDX(MSK_S, n)
#define MSK_BLOCKS(n) IDX(MSK_B, n)
#define MSK_BLOCK_STRIDES(n) IDX(MSK_SB, n)

#define KEY_OFFSETS \
    JOIN_COMMA(REDUCE(NDIMS, JOIN_COMMA, KEY_DIMS), \
            JOIN_COMMA(REDUCE(NDIMS, JOIN_COMMA, KEY_STRIDES), \
                    JOIN_COMMA(REDUCE(NDIMS, JOIN_COMMA, KEY_BLOCKS), \
                            REDUCE(NDIMS, JOIN_COMMA, KEY_BLOCK_STRIDES))))

#define QRY_OFFSETS \
    JOIN_COMMA(REDUCE(NDIMS, JOIN_COMMA, QRY_DIMS), \
            JOIN_COMMA(REDUCE(NDIMS, JOIN_COMMA, QRY_STRIDES), \
                    JOIN_COMMA(REDUCE(NDIMS, JOIN_COMMA, QRY_BLOCKS), \
                            REDUCE(NDIMS, JOIN_COMMA, QRY_BLOCK_STRIDES))))

#define VAL_OFFSETS \
    JOIN_COMMA(REDUCE(NDIMS, JOIN_COMMA, VAL_DIMS), \
            JOIN_COMMA(REDUCE(NDIMS, JOIN_COMMA, VAL_STRIDES), \
                    JOIN_COMMA(REDUCE(NDIMS, JOIN_COMMA, VAL_BLOCKS), \
                            REDUCE(NDIMS, JOIN_COMMA, VAL_BLOCK_STRIDES))))

#define DST_OFFSETS \
    JOIN_COMMA(REDUCE(NDIMS, JOIN_COMMA, DST_DIMS), \
            JOIN_COMMA(REDUCE(NDIMS, JOIN_COMMA, DST_STRIDES), \
                    JOIN_COMMA(REDUCE(NDIMS, JOIN_COMMA, DST_BLOCKS), \
                            REDUCE(NDIMS, JOIN_COMMA, DST_BLOCK_STRIDES))))

#define MSK_OFFSETS \
    JOIN_COMMA(REDUCE(NDIMS, JOIN_COMMA, MSK_DIMS), \
            JOIN_COMMA(REDUCE(NDIMS, JOIN_COMMA, MSK_STRIDES), \
                    JOIN_COMMA(REDUCE(NDIMS, JOIN_COMMA, MSK_BLOCKS), \
                            REDUCE(NDIMS, JOIN_COMMA, MSK_BLOCK_STRIDES))))

#endif
