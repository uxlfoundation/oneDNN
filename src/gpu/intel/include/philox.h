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

// Defined for every kernel by kernel_ctx_t from math::philox_dropout_rounds.
#ifndef PHILOX_DROPOUT_ROUNDS
#error "PHILOX_DROPOUT_ROUNDS is not defined"
#endif
#if PHILOX_DROPOUT_ROUNDS != 7
#error "The dropout functions below implement 7 Philox rounds"
#endif

/* Philox-4x32 generators. Dropout uses 7 rounds, everything else the
   standard 10. The dropout round count is checked above against
   math::philox_dropout_rounds in src/common/math_utils.hpp.

   A block is the output of one Philox call: four 32-bit words (128 bits).
   A value is the random number of one element: a 32-bit word, or a 16-bit
   half-word of a block. Dropout compares a value against
   get_dropout_threshold(); 16-bit values are the top half of a 32-bit one.

   Function                        Rounds Returns / used by
   philox_4x32_key_schedule          -    keys for *_ctr_w_keys(), per seed
   philox_4x32_vec4_ctr_w_keys       7    one block from a counter
   philox_4x32_vec4x2_ctr_w_keys     7    two blocks, two chains (ILP)
   philox_4x32_ctr_w_offset          -    counter with offset (idx & ~3)
   philox_4x32_ctr_legacy            -    counter without offset
   philox_window4 / philox_window8   -    shifted window; unaligned SDPA
   philox_4x32_dropout_w_offset      7    one value; dropout attr, offset
   philox_4x32_dropout               7    one value; dropout attr, legacy
   philox_4x32_vec4_w_offset        10    one block; generic
   philox_4x32_w_offset             10    one value; generic
   philox_4x32 / philox_4x32_vec4   10    one value / block; generic
   philox_8x16 / philox_16x8        10    16 / 8 bits; stochastic rounding
   The grouped SDPA fill builds its generators from the blocks above. */

/* Key schedule depends only on seed: compute once, then use *_w_keys(). */
typedef struct {
    uint16 key0;
    uint4 key1;
} philox_key_schedule_t;

philox_key_schedule_t philox_4x32_key_schedule(ulong seed) {
    uint seed_lo = (uint)seed;
    uint seed_hi = (uint)(seed >> 32);
    const ulong seeds = as_ulong((uint2)(seed_lo, seed_hi));
    const ulong PHILOX_W4x32 = as_ulong((uint2)(0x9E3779B9u, 0xBB67AE85u));
    philox_key_schedule_t ks;
    ks.key0 = as_uint16((ulong8)(seeds))
            + as_uint16((ulong8)(PHILOX_W4x32))
                    * (uint16)(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    ks.key1 = as_uint4((ulong2)seeds)
            + as_uint4((ulong2)(PHILOX_W4x32)) * (uint4)(8, 8, 9, 9);
    return ks;
}

/* No native 64-bit multiply: use 32-bit mul + mul_hi for each product. */
#define PHILOX_4UINT_ROUND(mul, ctr, key) \
    ({ \
        uint lo0 = ctr.s0 * mul.s0; \
        uint hi0 = mul_hi(ctr.s0, mul.s0); \
        uint lo1 = ctr.s2 * mul.s1; \
        uint hi1 = mul_hi(ctr.s2, mul.s1); \
        (uint4)(hi1 ^ ctr.s1 ^ key.s0, lo1, hi0 ^ ctr.s3 ^ key.s1, lo0); \
    })

/* 7 rounds, for dropout. */
uint4 philox_4x32_vec4_ctr_w_keys(uint4 ctr, uint16 key0, uint4 key1) {
    const uint2 PHILOX_M4x32 = (uint2)(0xD2511F53u, 0xCD9E8D57u);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, key0.s01);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, key0.s23);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, key0.s45);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, key0.s67);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, key0.s89);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, key0.sAB);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, key0.sCD);
    return ctr;
}

/* Two independent vec4 chains per call. Their instructions interleave, so
   the GPU overlaps one chain's round latency with the other chain's work
   (ILP, instruction-level parallelism). */
uint8 philox_4x32_vec4x2_ctr_w_keys(
        uint4 ctrA, uint4 ctrB, uint16 key0, uint4 key1) {
    const uint2 PHILOX_M4x32 = (uint2)(0xD2511F53u, 0xCD9E8D57u);

#define PHILOX_4UINT_ROUND_X2(key) \
    do { \
        ctrA = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctrA, (key)); \
        ctrB = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctrB, (key)); \
    } while (0)

    PHILOX_4UINT_ROUND_X2(key0.s01);
    PHILOX_4UINT_ROUND_X2(key0.s23);
    PHILOX_4UINT_ROUND_X2(key0.s45);
    PHILOX_4UINT_ROUND_X2(key0.s67);
    PHILOX_4UINT_ROUND_X2(key0.s89);
    PHILOX_4UINT_ROUND_X2(key0.sAB);
    PHILOX_4UINT_ROUND_X2(key0.sCD);
#undef PHILOX_4UINT_ROUND_X2
    return (uint8)(ctrA, ctrB);
}

uint4 philox_4x32_ctr_w_offset(ulong idx, ulong offset) {
    ulong x = (idx & ~3L);
    return (uint4)((uint)offset, (uint)(offset >> 32), (uint)x,
            (uint)(x >> 32));
}

/* Standard 10 rounds: the 7 dropout rounds and three more. */
uint4 philox_4x32_vec4_w_offset(ulong idx, ulong seed, ulong offset) {
    const uint2 PHILOX_M4x32 = (uint2)(0xD2511F53u, 0xCD9E8D57u);
    const philox_key_schedule_t ks = philox_4x32_key_schedule(seed);
    uint4 ctr = philox_4x32_vec4_ctr_w_keys(
            philox_4x32_ctr_w_offset(idx, offset), ks.key0, ks.key1);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, ks.key0.sEF);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, ks.key1.s01);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, ks.key1.s23);
    return ctr;
}

/* Legacy (no offset) generator, as in philox_4x32(): the counter comes from a
   32-bit idx alone and all indices sharing idx & ~3 get word 2 of the block. */
ulong philox_4x32_legacy_seed(long seed) {
    return ((ulong)(uint)seed << 32) + (uint)seed;
}

uint4 philox_4x32_ctr_legacy(uint idx) {
    uint x = idx & ~3u;
    return (uint4)(x, x + 1, x, x + 3);
}

/* Words [a, a + 4) of the 8-word window (lo, hi), for a in [0, 3]. */
uint4 philox_window4(uint4 lo, uint4 hi, int a) {
    return a == 0    ? lo
            : a == 1 ? (uint4)(lo.s123, hi.s0)
            : a == 2 ? (uint4)(lo.s23, hi.s01)
                     : (uint4)(lo.s3, hi.s012);
}

/* Halfwords [a, a + 8) of the 16-halfword window (lo, hi), for a in [0, 7]. */
ushort8 philox_window8(ushort8 lo, ushort8 hi, int a) {
    const uint4 l = as_uint4(lo), h = as_uint4(hi);
    const int b = a >> 1;
    const uint4 w = philox_window4(l, h, b);
    const uint e = b == 0 ? h.s0 : b == 1 ? h.s1 : b == 2 ? h.s2 : h.s3;
    return as_ushort8((a & 1) ? (w >> 16) | ((uint4)(w.s123, e) << 16) : w);
}

uint philox_4x32_w_offset(long idx, long seed, long offset) {
    return philox_4x32_vec4_w_offset(
            (ulong)idx, (ulong)seed, (ulong)offset)[idx & 3L];
}

/* Dropout with offset: one block gives eight 16-bit values. idx uses
   half-word (idx & 7) of the block with counter (idx & ~7). Matches
   philox4x32_dropout() in math_utils.hpp. */
uint philox_4x32_dropout_w_offset(long idx, long seed, long offset) {
    const philox_key_schedule_t ks = philox_4x32_key_schedule((ulong)seed);
    const uint4 blk = philox_4x32_vec4_ctr_w_keys(
            philox_4x32_ctr_w_offset((ulong)idx & ~7UL, (ulong)offset), ks.key0,
            ks.key1);
    return (uint)as_ushort8(blk)[idx & 7L] << 16;
}

uint philox_4x32(long idx, long seed) {
    // Note: this is for compatibility with impls that don't support s64 rand
    // Truncate to uint first to handle negative longs correctly
    uint idx_u = (uint)idx;
    uint seed_u = (uint)seed;
    ulong x = idx_u & ~3UL;
    ulong idx_64 = ((x + 3) << 32) + (x + 2);
    ulong offset_64 = ((x + 1) << 32) + x;
    ulong seed_64 = ((ulong)seed_u << 32) + seed_u;
    return philox_4x32_w_offset(idx_64, seed_64, offset_64);
}

/* Dropout without offset: the counter of philox_4x32(), 7 rounds. */
uint philox_4x32_dropout(long idx, long seed) {
    const philox_key_schedule_t ks
            = philox_4x32_key_schedule(philox_4x32_legacy_seed(seed));
    return philox_4x32_vec4_ctr_w_keys(
            philox_4x32_ctr_legacy((uint)idx), ks.key0, ks.key1)
            .s2;
}

uint4 philox_4x32_vec4(uint idx, uint seed) {
    ulong x = idx & ~3L;
    ulong idx_64 = ((x + 3) << 32) + (x + 2);
    ulong offset_64 = ((x + 1) << 32) + x;
    ulong seed_64 = ((ulong)(seed) << 32) + seed;
    return philox_4x32_vec4_w_offset(idx_64, seed_64, offset_64);
}

ushort philox_8x16(long idx, long seed) {
    return as_ushort2(philox_4x32((idx >> 1), seed))[idx & 1];
}

uchar philox_16x8(long idx, long seed) {
    return as_uchar4(philox_4x32((idx >> 2), seed))[idx & 3];
}

#if WITH_SROUND

#if DST_DT_DIGITS > 24
#error "Invalid dst digits"
#endif

float stochastic_round_fwd(float s, long idx, uint seed) {
    if (isnan(s) || isinf(s)) return s;
    uint truncation_mask = 0xffffffff << (24 - DST_DT_DIGITS);
    uint bias_val = sizeof(DST_DATA_T) == 2 ? philox_16x8(idx, (long)seed)
                                            : philox_8x16(idx, (long)seed);
    uint rnd_bias = (uint)(bias_val & ~truncation_mask);
    float r = as_float((as_uint(s) + rnd_bias) & truncation_mask);
    r = fmin(fmax((float)DST_DATA_FLOW, r), (float)DST_DATA_FMAX);
    if (fabs(r) > 0 && fabs(r) < DST_DATA_FMIN) r = 0;
    return r;
}
#endif

#if WITH_DROPOUT
// No need to enable fp64 extensions just to compute (double)p * 0xFFFFFFFFu
uint get_dropout_threshold(float p) {
    if (p >= 1.f) return 0xFFFFFFFFu;
    char exponent = 126 - ((as_uint(p) >> 23) & 0x7F);
    if ((p <= 0.f) || (exponent > 31)) return 0u;
    uint mantissa = (as_uint(p) << 8) | 0x80000000u;
    if (!exponent) return (convert_ulong(mantissa) * 0xFFFFFFFFuL) >> 32;
    return ((convert_ulong(mantissa >> exponent) * 0xFFFFFFFFuL) >> 32)
            + !!(mantissa & ((1u << exponent) - 1u));
}
#endif
#endif
