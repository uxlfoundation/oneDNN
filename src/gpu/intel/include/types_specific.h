/*******************************************************************************
* Copyright 2025 Intel Corporation
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

// See ocl_types.h for details about macros naming and organization.

#ifndef GPU_INTEL_INCLUDE_TYPES_SPECIFIC_H
#define GPU_INTEL_INCLUDE_TYPES_SPECIFIC_H

#ifdef DST_DATA_T
#if DST_DT_BF16
#define DST_DATA_FMAX cvt_bf16_to_f32((ushort)0x7F7F)
#define DST_DATA_FMIN cvt_bf16_to_f32((ushort)0x0080)
#define DST_DATA_FLOW cvt_bf16_to_f32((ushort)0xFF7F)
#elif DST_DT_F16
#define DST_DATA_FMAX convert_float(HALF_MAX)
#define DST_DATA_FMIN convert_float(HALF_MIN)
#define DST_DATA_FLOW -DST_DATA_FMAX
#elif DST_DT_BF8
#define DST_DATA_FMAX convert_float(cvt_f8_e5m2_to_hf((uchar)0x7B))
#define DST_DATA_FMIN convert_float(cvt_f8_e5m2_to_hf((uchar)0x04))
#define DST_DATA_FLOW convert_float(cvt_f8_e5m2_to_hf((uchar)0xFB))
#elif DST_DT_HF8
#define DST_DATA_FMAX convert_float(cvt_f8_e4m3_to_hf((uchar)0x7E))
#define DST_DATA_FMIN convert_float(cvt_f8_e4m3_to_hf((uchar)0x08))
#define DST_DATA_FLOW convert_float(cvt_f8_e4m3_to_hf((uchar)0xFE))
#elif DST_DT_U4
#define DST_DATA_FMAX 15.0
#define DST_DATA_FMIN 0.0
#define DST_DATA_FLOW DST_DATA_FMIN
#elif DST_DT_S4
#define DST_DATA_FMAX 7.0
#define DST_DATA_FMIN 1
#define DST_DATA_FLOW -8.0
#elif DST_DT_F4_E2M1
#define DST_DATA_FMAX 6.0
#define DST_DATA_FMIN 1.0
#define DST_DATA_FLOW -6.0
#elif DST_DT_F4_E3M0
#define DST_DATA_FMAX 16.0
#define DST_DATA_FMIN 0.25
#define DST_DATA_FLOW -16.0
#elif DST_DT_U8
#define DST_DATA_FMAX convert_float(UCHAR_MAX)
#define DST_DATA_FMIN 1
#define DST_DATA_FLOW 0
#elif DST_DT_S8
#define DST_DATA_FMAX convert_float(CHAR_MAX)
#define DST_DATA_FMIN 1
#define DST_DATA_FLOW -DST_DATA_FMAX
#elif DST_DT_S32
#define DST_DATA_FMAX convert_float(INT_MAX)
#define DST_DATA_FMIN 1
#define DST_DATA_FLOW -DST_DATA_FMAX
#elif DST_DT_F32
#define DST_DATA_FMAX convert_float(FLT_MAX)
#define DST_DATA_FMIN convert_float(FLT_MIN)
#define DST_DATA_FLOW -DST_DATA_FMAX
#elif DST_DT_F64
#define DST_DATA_FMAX convert_float(FLT_MAX)
#define DST_DATA_FMIN convert_float(FLT_MIN)
#define DST_DATA_FLOW -DST_DATA_FMAX
#else
#error "Not expected"
#endif
#endif

#if NDIMS == 2
#define SRC_OFF(x0, x1, d, h, w) \
    (((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0 \
            + ((x1) % SRC_B1) * SRC_SB1 + ((x1) / SRC_B1) * SRC_S1)

#if WITH_GROUPS == 1
#define WEI_OFF(x0, x1, x2, d, h, w) \
    (((x0) % WEI_B0) * WEI_SB0 + ((x0) / WEI_B0) * WEI_S0 \
            + ((x1) % WEI_B1) * WEI_SB1 + ((x1) / WEI_B1) * WEI_S1 \
            + ((x2) % WEI_B2) * WEI_SB2 + ((x2) / WEI_B2) * WEI_S2)
#else
#define WEI_OFF(g, x0, x1, d, h, w) \
    (((x0) % WEI_B0) * WEI_SB0 + ((x0) / WEI_B0) * WEI_S0 \
            + ((x1) % WEI_B1) * WEI_SB1 + ((x1) / WEI_B1) * WEI_S1)
#endif

#define DST_OFF(x0, x1, d, h, w) \
    (((x0) % DST_B0) * DST_SB0 + ((x0) / DST_B0) * DST_S0 \
            + ((x1) % DST_B1) * DST_SB1 + ((x1) / DST_B1) * DST_S1)
#elif NDIMS == 3
#define SRC_OFF(x0, x1, d, h, x2) \
    (((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0 \
            + ((x1) % SRC_B1) * SRC_SB1 + ((x1) / SRC_B1) * SRC_S1 \
            + ((x2) % SRC_B2) * SRC_SB2 + ((x2) / SRC_B2) * SRC_S2)

#if WITH_GROUPS == 1
#define WEI_OFF(x0, x1, x2, d, h, x3) \
    (((x0) % WEI_B0) * WEI_SB0 + ((x0) / WEI_B0) * WEI_S0 \
            + ((x1) % WEI_B1) * WEI_SB1 + ((x1) / WEI_B1) * WEI_S1 \
            + ((x2) % WEI_B2) * WEI_SB2 + ((x2) / WEI_B2) * WEI_S2 \
            + ((x3) % WEI_B3) * WEI_SB3 + ((x3) / WEI_B3) * WEI_S3)
#else
#define WEI_OFF(g, x0, x1, d, h, x2) \
    (((x0) % WEI_B0) * WEI_SB0 + ((x0) / WEI_B0) * WEI_S0 \
            + ((x1) % WEI_B1) * WEI_SB1 + ((x1) / WEI_B1) * WEI_S1 \
            + ((x2) % WEI_B2) * WEI_SB2 + ((x2) / WEI_B2) * WEI_S2)
#endif

#define DST_OFF(x0, x1, d, h, x2) \
    (((x0) % DST_B0) * DST_SB0 + ((x0) / DST_B0) * DST_S0 \
            + ((x1) % DST_B1) * DST_SB1 + ((x1) / DST_B1) * DST_S1 \
            + ((x2) % DST_B2) * DST_SB2 + ((x2) / DST_B2) * DST_S2)
#elif NDIMS == 4
#define SRC_OFF(x0, x1, d, x2, x3) \
    (((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0 \
            + ((x1) % SRC_B1) * SRC_SB1 + ((x1) / SRC_B1) * SRC_S1 \
            + ((x2) % SRC_B2) * SRC_SB2 + ((x2) / SRC_B2) * SRC_S2 \
            + ((x3) % SRC_B3) * SRC_SB3 + ((x3) / SRC_B3) * SRC_S3)

#if WITH_GROUPS == 1
#define WEI_OFF(x0, x1, x2, d, x3, x4) \
    (((x0) % WEI_B0) * WEI_SB0 + ((x0) / WEI_B0) * WEI_S0 \
            + ((x1) % WEI_B1) * WEI_SB1 + ((x1) / WEI_B1) * WEI_S1 \
            + ((x2) % WEI_B2) * WEI_SB2 + ((x2) / WEI_B2) * WEI_S2 \
            + ((x3) % WEI_B3) * WEI_SB3 + ((x3) / WEI_B3) * WEI_S3 \
            + ((x4) % WEI_B4) * WEI_SB4 + ((x4) / WEI_B4) * WEI_S4)
#else
#define WEI_OFF(g, x1, x2, d, x3, x4) \
    (((x1) % WEI_B0) * WEI_SB0 + ((x1) / WEI_B0) * WEI_S0 \
            + ((x2) % WEI_B1) * WEI_SB1 + ((x2) / WEI_B1) * WEI_S1 \
            + ((x3) % WEI_B2) * WEI_SB2 + ((x3) / WEI_B2) * WEI_S2 \
            + ((x4) % WEI_B3) * WEI_SB3 + ((x4) / WEI_B3) * WEI_S3)
#endif

#define DST_OFF(x0, x1, d, x2, x3) \
    (((x0) % DST_B0) * DST_SB0 + ((x0) / DST_B0) * DST_S0 \
            + ((x1) % DST_B1) * DST_SB1 + ((x1) / DST_B1) * DST_S1 \
            + ((x2) % DST_B2) * DST_SB2 + ((x2) / DST_B2) * DST_S2 \
            + ((x3) % DST_B3) * DST_SB3 + ((x3) / DST_B3) * DST_S3)
#elif NDIMS == 5
#define SRC_OFF(x0, x1, x2, x3, x4) \
    (((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0 \
            + ((x1) % SRC_B1) * SRC_SB1 + ((x1) / SRC_B1) * SRC_S1 \
            + ((x2) % SRC_B2) * SRC_SB2 + ((x2) / SRC_B2) * SRC_S2 \
            + ((x3) % SRC_B3) * SRC_SB3 + ((x3) / SRC_B3) * SRC_S3 \
            + ((x4) % SRC_B4) * SRC_SB4 + ((x4) / SRC_B4) * SRC_S4)

#if WITH_GROUPS == 1
#define WEI_OFF(x0, x1, x2, x3, x4, x5) \
    (((x0) % WEI_B0) * WEI_SB0 + ((x0) / WEI_B0) * WEI_S0 \
            + ((x1) % WEI_B1) * WEI_SB1 + ((x1) / WEI_B1) * WEI_S1 \
            + ((x2) % WEI_B2) * WEI_SB2 + ((x2) / WEI_B2) * WEI_S2 \
            + ((x3) % WEI_B3) * WEI_SB3 + ((x3) / WEI_B3) * WEI_S3 \
            + ((x4) % WEI_B4) * WEI_SB4 + ((x4) / WEI_B4) * WEI_S4 \
            + ((x5) % WEI_B5) * WEI_SB5 + ((x5) / WEI_B5) * WEI_S5)
#else
#define WEI_OFF(g, x1, x2, x3, x4, x5) \
    (((x1) % WEI_B0) * WEI_SB0 + ((x1) / WEI_B0) * WEI_S0 \
            + ((x2) % WEI_B1) * WEI_SB1 + ((x2) / WEI_B1) * WEI_S1 \
            + ((x3) % WEI_B2) * WEI_SB2 + ((x3) / WEI_B2) * WEI_S2 \
            + ((x4) % WEI_B3) * WEI_SB3 + ((x4) / WEI_B3) * WEI_S3 \
            + ((x5) % WEI_B4) * WEI_SB4 + ((x5) / WEI_B4) * WEI_S4)
#endif

#define DST_OFF(x0, x1, x2, x3, x4) \
    (((x0) % DST_B0) * DST_SB0 + ((x0) / DST_B0) * DST_S0 \
            + ((x1) % DST_B1) * DST_SB1 + ((x1) / DST_B1) * DST_S1 \
            + ((x2) % DST_B2) * DST_SB2 + ((x2) / DST_B2) * DST_S2 \
            + ((x3) % DST_B3) * DST_SB3 + ((x3) / DST_B3) * DST_S3 \
            + ((x4) % DST_B4) * DST_SB4 + ((x4) / DST_B4) * DST_S4)
#endif

#endif
