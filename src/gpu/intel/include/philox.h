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

#ifndef GPU_INTEL_INCLUDE_PHILOX_H
#define GPU_INTEL_INCLUDE_PHILOX_H

#define DT_UNDEF 1
#include "gpu/intel/include/types.h"

uint philox_4x32_s64(ulong idx, ulong seed, ulong offset) {
#define PHILOX_4UINT_ROUND(mul, ctr, key) \
    as_uint4(convert_ulong2(ctr.s02) * mul).s3210 \
            ^ (uint4)(ctr.s1 ^ key.s0, 0, ctr.s3 ^ key.s1, 0)

    ulong x = (idx & ~3L);
    uint4 ctr = (uint4)((uint)offset, (uint)(offset >> 32), (uint)x,
            (uint)(x >> 32));
    uint seed_lo = (uint)seed;
    uint seed_hi = (uint)(seed >> 32);
    const ulong seeds = as_ulong((uint2)(seed_lo, seed_hi));

    const ulong2 PHILOX_M4x32 = (ulong2)(0xD2511F53uL, 0xCD9E8D57uL);
    const ulong PHILOX_W4x32 = as_ulong((uint2)(0x9E3779B9u, 0xBB67AE85u));
    const uint16 key0 = as_uint16((ulong8)(seeds))
            + as_uint16((ulong8)(PHILOX_W4x32))
                    * (uint16)(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    const uint4 key1 = as_uint4((ulong2)seeds)
            + as_uint4((ulong2)(PHILOX_W4x32)) * (uint4)(8, 8, 9, 9);

    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, key0.s01);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, key0.s23);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, key0.s45);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, key0.s67);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, key0.s89);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, key0.sAB);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, key0.sCD);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, key0.sEF);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, key1.s01);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, key1.s23);
    return ctr[idx & 3L];
}

uint philox_4x32(uint idx, uint seed) {
    // Note: this is for compatibility with impls that don't support s64 rand
    ulong x = idx & ~3L;
    ulong idx_64 = ((x + 3) << 32) + (x + 2);
    ulong offset_64 = ((x + 1) << 32) + x;
    ulong seed_64 = ((ulong)(seed) << 32) + seed;
    return philox_4x32_s64(idx_64, seed_64, offset_64);
}

ushort philox_8x16(long idx, uint seed) {
    return as_ushort2(philox_4x32(idx >> 1, seed))[idx & 1];
}

uchar philox_16x8(long idx, uint seed) {
    return as_uchar4(philox_4x32(idx >> 2, seed))[idx & 3];
}

#if WITH_SROUND

#if DST_DT_DIGITS > 24
#error "Invalid dst digits"
#endif

float stochastic_round_fwd(float s, long idx, uint seed) {
    if (isnan(s) || isinf(s)) return s;
    uint truncation_mask = 0xffffffff << (24 - DST_DT_DIGITS);
    uint bias_val = sizeof(DST_DATA_T) == 2 ? philox_16x8(idx, seed)
                                            : philox_8x16(idx, seed);
    uint rnd_bias = (uint)(bias_val & ~truncation_mask);
    float r = as_float((as_uint(s) + rnd_bias) & truncation_mask);
    r = fmin(fmax((float)DST_DATA_FLOW, r), (float)DST_DATA_FMAX);
    if (fabs(r) > 0 && fabs(r) < DST_DATA_FMIN) r = 0;
    return r;
}
#endif

#endif
