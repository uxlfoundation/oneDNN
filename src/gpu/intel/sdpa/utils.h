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

#ifndef GPU_INTEL_SDPA_UTILS_H
#define GPU_INTEL_SDPA_UTILS_H

#define _4D_OFF(tag, x0, x1, x2, x3) \
    (((x0) % tag##_B0) * tag##_SB0 + ((x0) / tag##_B0) * tag##_S0 \
            + ((x1) % tag##_B1) * tag##_SB1 + ((x1) / tag##_B1) * tag##_S1 \
            + ((x2) % tag##_B2) * tag##_SB2 + ((x2) / tag##_B2) * tag##_S2 \
            + ((x3) % tag##_B3) * tag##_SB3 + ((x3) / tag##_B3) * tag##_S3)

#define QRY_OFF(x0, x1, x2, x3) _4D_OFF(QRY, x0, x1, x2, x3)
#define KEY_OFF(x0, x1, x2, x3) _4D_OFF(KEY, x0, x1, x2, x3)
#define VAL_OFF(x0, x1, x2, x3) _4D_OFF(VAL, x0, x1, x2, x3)
#define MSK_OFF(x0, x1, x2, x3) _4D_OFF(MSK, x0, x1, x2, x3)

#define _BATCH_OFF(tag, x0, x1) ((x0) * tag##_S0 + (x1) * tag##_S1)

#define QRY_BATCH(x0, x1) _BATCH_OFF(QRY, x0, x1)
#define KEY_BATCH(x0, x1) _BATCH_OFF(KEY, x0, x1)
#define VAL_BATCH(x0, x1) _BATCH_OFF(VAL, x0, x1)
#define DST_BATCH(x0, x1) _BATCH_OFF(DST, x0, x1)
#define MSK_BATCH(x0, x1) _BATCH_OFF(MSK, x0, x1)
#define DA_BATCH(x0, x1) _BATCH_OFF(DA, x0, x1)
#define DK_BATCH(x0, x1) _BATCH_OFF(DK, x0, x1)
#define DQ_BATCH(x0, x1) _BATCH_OFF(DQ, x0, x1)
#define DV_BATCH(x0, x1) _BATCH_OFF(DV, x0, x1)

/* Forward kernel offset macros */
#define KEY_OFFSETS \
    const long KEY_S0, const long KEY_S1, const long KEY_S2, \
            const long KEY_S3, const long KEY_D3
#define QRY_OFFSETS const long QRY_S0, const long QRY_S1, const long QRY_S2
#define VAL_OFFSETS const long VAL_S0, const long VAL_S1, const long VAL_S2
#define DST_OFFSETS \
    const long DST_S0, const long DST_S1, const long DST_S2, const long DST_D1
#define MSK_OFFSETS \
    const long MSK_S0, const long MSK_S1, const long MSK_S2, \
            const long MSK_D0, const long MSK_D1

/* Backward kernel offset/stride macros for preprocess kernel */
#define BWD_QRY_OFFSETS const long QRY_S0, const long QRY_S1, const long QRY_S2
#define BWD_DST_OFFSETS \
    const long DST_S0, const long DST_S1, const long DST_S2, const long DST_D1
#define BWD_DA_OFFSETS const long DA_S0, const long DA_S1, const long DA_S2

#define BWD_UNPACK_STRIDE_PARAMS(buf) \
    long KEY_S0 = buf[0]; \
    long KEY_S1 = buf[1]; \
    long KEY_S2 = buf[2]; \
    long KEY_S3 = buf[3]; \
    long QRY_S0 = buf[4]; \
    long QRY_S1 = buf[5]; \
    long QRY_S2 = buf[6]; \
    long VAL_S0 = buf[7]; \
    long VAL_S1 = buf[8]; \
    long VAL_S2 = buf[9]; \
    long DST_S0 = buf[10]; \
    long DST_S1 = buf[11]; \
    long DST_S2 = buf[12]; \
    long DST_D1 = buf[13]; \
    long DA_S0 = buf[14]; \
    long DA_S1 = buf[15]; \
    long DA_S2 = buf[16]; \
    long DK_S0 = buf[17]; \
    long DK_S1 = buf[18]; \
    long DK_S2 = buf[19]; \
    long DK_S3 = buf[20]; \
    long DQ_S0 = buf[21]; \
    long DQ_S1 = buf[22]; \
    long DQ_S2 = buf[23]; \
    long DV_S0 = buf[24]; \
    long DV_S1 = buf[25]; \
    long DV_S2 = buf[26];

#define BWD_UNPACK_MSK_PARAMS(buf) \
    long MSK_S0 = buf[27]; \
    long MSK_S1 = buf[28]; \
    long MSK_S2 = buf[29]; \
    long MSK_D0 = buf[30]; \
    long MSK_D1 = buf[31];

#define FULL_QRY_OFFSETS \
    const long QRY_D0, const long QRY_D1, const long QRY_D2, \
            const long QRY_D3, const long QRY_S0, const long QRY_S1, \
            const long QRY_S2, const long QRY_S3
#define DA_OFFSETS const long DA_S0, const long DA_S1, const long DA_S2
#define DK_STRIDES \
    const long DK_S0, const long DK_S1, const long DK_S2, const long DK_S3
#define DQ_STRIDES \
    const long DQ_S0, const long DQ_S1, const long DQ_S2, const long DQ_S3
#define DV_STRIDES \
    const long DV_S0, const long DV_S1, const long DV_S2, const long DV_S3

#endif
