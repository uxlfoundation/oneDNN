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

#endif
