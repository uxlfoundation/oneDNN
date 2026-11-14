/*******************************************************************************
* Copyright 2026 Intel Corporation
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

// Enables the low-precision conversion helpers for every type.
#define MATH_UTILS_DECLARE_BF8 1
#define MATH_UTILS_DECLARE_HF8 1
#define MATH_UTILS_DECLARE_F4_E2M1 1
#define MATH_UTILS_DECLARE_BF16 1
#define MATH_UTILS_DECLARE_E8M0 1

#include "gpu/intel/include/philox.h"

// Values match dnnl_data_type_t.
enum {
    fill_dt_undef = 0,
    fill_dt_f16 = 1,
    fill_dt_bf16 = 2,
    fill_dt_f32 = 3,
    fill_dt_s32 = 4,
    fill_dt_s8 = 5,
    fill_dt_u8 = 6,
    fill_dt_f64 = 7,
    fill_dt_boolean = 8,
    fill_dt_f8_e5m2 = 9,
    fill_dt_f8_e4m3 = 10,
    fill_dt_s4 = 11,
    fill_dt_u4 = 12,
    fill_dt_e8m0 = 13,
    fill_dt_f4_e2m1 = 14,
    fill_dt_s64 = 16,
    fill_dt_u2 = 17,
    fill_dt_u3 = 18,
};

int fill_dt_bits(int dt) {
    switch (dt) {
        case fill_dt_f64:
        case fill_dt_s64: return 64;
        case fill_dt_f32:
        case fill_dt_s32: return 32;
        case fill_dt_f16:
        case fill_dt_bf16: return 16;
        case fill_dt_s4:
        case fill_dt_u4:
        case fill_dt_f4_e2m1: return 4;
        case fill_dt_u3: return 3;
        case fill_dt_u2: return 2;
        default: return 8;
    }
}

// Four N(0, 1) samples via Box-Muller.
float4 normal4(ulong ctr, uint seed) {
    uint4 r = philox_4x32_vec4_w_offset(ctr << 2, seed, 0);
    float2 u1 = (convert_float2(r.s02 >> 8) + 0.5f) * 0x1p-24f;
    float2 u2 = convert_float2(r.s13 >> 8) * 0x1p-24f;
    float2 rad = sqrt(-2.0f * log(u1));
    float2 c;
    float2 s = sincos(2.0f * M_PI_F * u2, &c);
    return (float4)(rad.s0 * c.s0, rad.s0 * s.s0, rad.s1 * c.s1, rad.s1 * s.s1);
}

// Avoids double arithmetic: not every device supports fp64.
ulong f32_to_f64_bits(float f) {
    uint b = as_uint(f);
    ulong sign = (ulong)(b >> 31) << 63;
    uint e = (b >> 23) & 0xff;
    if (e == 0) return sign;
    return sign | ((ulong)(e + 896) << 52) | ((ulong)(b & 0x7fffff) << 29);
}

// Integers hold N(0, 1) scaled so that +/-4 sigma spans the type range.
ulong encode_normal(int dt, float z) {
    switch (dt) {
        case fill_dt_f16: return as_ushort((half)z);
        case fill_dt_bf16: return cvt_f32_to_bf16(z);
        case fill_dt_f32: return as_uint(z);
        case fill_dt_f64: return f32_to_f64_bits(z);
        case fill_dt_f8_e5m2: return cvt_hf_to_f8_e5m2((half)z);
        case fill_dt_f8_e4m3: return cvt_hf_to_f8_e4m3((half)z);
        case fill_dt_f4_e2m1: return cvt_f32_to_f4_e2m1(z);
        case fill_dt_e8m0: return cvt_f32_to_e8m0(fabs(z));
        case fill_dt_boolean: return z > 0.0f;
        case fill_dt_s8: return as_uchar(convert_char_sat_rte(z * 32.0f));
        case fill_dt_u8: return convert_uchar_sat_rte(z * 32.0f + 127.5f);
        case fill_dt_s32: return as_uint(convert_int_sat_rte(z * 0x1p29f));
        case fill_dt_s64: return as_ulong(convert_long_sat_rte(z * 0x1p61f));
        case fill_dt_s4: return as_uchar(cvt_f32_to_s4(z * 2.0f));
        case fill_dt_u4: return cvt_f32_to_u4(z * 2.0f + 7.5f);
        case fill_dt_u3: return clamp(convert_int_sat_rte(z + 3.5f), 0, 7);
        case fill_dt_u2: return cvt_f32_to_u2(z * 0.5f + 1.5f);
        default: return 0;
    }
}

// Packs N(0, 1) samples into the first 16 bytes (12 bytes for u3).
uint4 fill_normal(int dt, ulong item, uint seed) {
    const int bits = fill_dt_bits(dt);
    const int n = (bits == 3 ? 96 : 128) / bits;
    uint w[4] = {0, 0, 0, 0};
    for (int j = 0; j < n; j += 4) {
        float4 z = normal4(item * 16 + j / 4, seed);
        for (int k = 0; k < 4 && j + k < n; k++) {
            ulong v = encode_normal(dt, z[k]);
            int off = (j + k) * bits;
            int idx = off >> 5, sh = off & 31;
            w[idx] |= (uint)(v << sh);
            if (sh + bits > 32) w[idx + 1] |= (uint)(v >> (32 - sh));
        }
    }
    return (uint4)(w[0], w[1], w[2], w[3]);
}

// Fills a buffer with pseudo-random data using Philox RNG. Each work-item
// writes 16 bytes (12 bytes for u3 with normal data).
// - normal_dt == fill_dt_undef: uniform bits; the mask clears exponent LSBs to
//   prevent NaN/Inf for FP types.
// - Otherwise: N(0, 1) samples converted to normal_dt.
__kernel void fill_random(__global uint *buf, uint seed, ulong byte_count,
        uint mask, int normal_dt) {
    const ulong item_bytes
            = (normal_dt != fill_dt_undef && fill_dt_bits(normal_dt) == 3) ? 12
                                                                           : 16;
    const ulong start = (ulong)get_global_id(0) * item_bytes;
    if (start >= byte_count) return;

    uint4 rnd;
    if (normal_dt == fill_dt_undef) {
        uint b = (uint)(start >> 2);
        rnd = philox_4x32_vec4(b, b ^ seed) & (uint4)(mask);
    } else {
        rnd = fill_normal(normal_dt, get_global_id(0), seed);
    }

    if (start + item_bytes <= byte_count) {
        if (item_bytes == 12)
            vstore3(rnd.s012, get_global_id(0), buf);
        else
            vstore4(rnd, get_global_id(0), buf);
    } else {
        __global uchar *p = (__global uchar *)buf;
        uchar16 bytes = as_uchar16(rnd);
        for (ulong i = 0; i < byte_count - start; i++)
            p[start + i] = bytes[i];
    }
}
