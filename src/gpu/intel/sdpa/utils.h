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

#define _BATCH_OFF(tag, x0, x1) \
    ((x0) * tag##_S.array[0] + (x1) * tag##_S.array[1])

#define QRY_BATCH(x0, x1) _BATCH_OFF(QRY, x0, x1)
#define KEY_BATCH(x0, x1) _BATCH_OFF(KEY, x0, x1)
#define VAL_BATCH(x0, x1) _BATCH_OFF(VAL, x0, x1)
#define DST_BATCH(x0, x1) _BATCH_OFF(DST, x0, x1)
#define MSK_BATCH(x0, x1) _BATCH_OFF(MSK, x0, x1)

#define JOIN_COMMA(x, y) x, y

#define RT_DIM4(varname) const int64x4_t varname
#define RT_OFFSETS(basename) \
    JOIN_COMMA(RT_DIM4(basename##_D), RT_DIM4(basename##_S))

#define KEY_OFFSETS RT_OFFSETS(KEY)
#define QRY_OFFSETS RT_OFFSETS(QRY)
#define VAL_OFFSETS RT_OFFSETS(VAL)
#define DST_OFFSETS RT_OFFSETS(DST)
#define MSK_OFFSETS RT_OFFSETS(MSK)

// helper shorthands for accessing individual dimensions
#define KEY_D3 KEY_D.array[3]
#define KEY_S3 KEY_S.array[3]
#define KEY_S2 KEY_S.array[2]
#define QRY_S2 QRY_S.array[2]
#define VAL_S2 VAL_S.array[2]
#define DST_S2 DST_S.array[2]
#define MSK_D0 MSK_D.array[0]
#define MSK_D1 MSK_D.array[1]
#define MSK_S2 MSK_S.array[2]

#endif
