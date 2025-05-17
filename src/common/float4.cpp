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

#include <array>

#include "common/bit_cast.hpp"
#include "common/dnnl_thread.hpp"
#include "common/float16.hpp"
#include "common/float4.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {

uint8_t float2e2m1(float f) {
    uint32_t f_raw = float2int(f);
    uint32_t sign = f_raw & 0x80000000;

    // There is no NaN or infinity in e2m1, for now we just return zero
    // TODO: figure if there is a standard value to return
    uint32_t naninf_mask = 0x7f800000;
    if ((f_raw & naninf_mask) == naninf_mask) return 0x07 | (sign >> 28);

    // we convert with naive closest value computation out of 8
    float e2m1_val_table[8] = {0.0f, .5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

    float abs_f = fmin(e2m1_val_table[7], int2float(f_raw ^ sign));

    int idx = 0;
    float min_diff = ::fabsf(e2m1_val_table[idx] - abs_f);
    uint8_t raw_bits = idx;
    for (++idx; idx < 8; ++idx) {
        float diff = ::fabsf(e2m1_val_table[idx] - abs_f);
        if (diff < min_diff) {
            min_diff = diff;
            raw_bits = idx;
        }
        // Special case for midpoint, we round to even (so even index)
        if ((diff == min_diff) && !(idx & 1)) raw_bits = idx;
    }
    assert(raw_bits < 8);
    // reapply sign
    if (sign) raw_bits = raw_bits | 0x08;
    assert(raw_bits < 16);
    return raw_bits;
}

float4_e2m1_t &float4_e2m1_t::operator=(bfloat16_t f) {
    float f32 = f;
    raw_bits_ = float2e2m1(f32);
    return *this;
}

float4_e2m1_t &float4_e2m1_t::operator=(float16_t f) {
    float f32 = f;
    raw_bits_ = float2e2m1(f32);
    return *this;
}

float4_e2m1_t &float4_e2m1_t::operator=(float f) {
    raw_bits_ = float2e2m1(f);
    return *this;
}

float4_e2m1_t::operator float() const {
    // List of e2m1 values. The index of each value maps to its encoding.
    static const float e2m1_table[16] = {0.0f, .5f, 1.0f, 1.5f, 2.0f, 3.0f,
            4.0f, 6.0f, -0.0f, -.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    assert(raw_bits_ < 16);
    return e2m1_table[raw_bits_];
}

float4_e2m1_t::operator float16_t() const {
    // List of e2m1 values. The index of each value maps to its encoding.
    static const float16_t e2m1_table[16] = {0.0f, .5f, 1.0f, 1.5f, 2.0f, 3.0f,
            4.0f, 6.0f, -0.0f, -.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    assert(raw_bits_ < 16);
    return e2m1_table[raw_bits_];
}

uint8_t float2e3m0(float f) {
    uint32_t f_raw = float2int(f);
    uint32_t sign = f_raw & 0x80000000;

    // There is no NaN or infinity in e3m0, we just return maxval
    uint32_t naninf_mask = 0x7f800000;
    if ((f_raw & naninf_mask) == naninf_mask) return 0x7 | (sign >> 28);

    // we convert with naive closest value computation out of 8
    float e3m0_val_table[8] = {0.0f, .25f, .5f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f};

    float abs_f = fmin(e3m0_val_table[7], int2float(f_raw ^ sign));

    int idx = 0;
    float min_diff = ::fabsf(e3m0_val_table[idx] - abs_f);
    uint8_t raw_bits = idx;
    for (++idx; idx < 8; ++idx) {
        float diff = ::fabsf(e3m0_val_table[idx] - abs_f);
        if (diff < min_diff) {
            min_diff = diff;
            raw_bits = idx;
        }
        // Special case for midpoint, we round to even (so even index)
        if ((diff == min_diff) && !(idx & 1)) raw_bits = idx;
    }
    assert(raw_bits < 8);
    // reapply sign
    if (sign) raw_bits = raw_bits | 0x08;
    assert(raw_bits < 16);
    return raw_bits;
}

float4_e3m0_t &float4_e3m0_t::operator=(bfloat16_t f) {
    float f32 = f;
    raw_bits_ = float2e3m0(f32);
    return *this;
}

float4_e3m0_t &float4_e3m0_t::operator=(float16_t f) {
    float f32 = f;
    raw_bits_ = float2e3m0(f32);
    return *this;
}

float4_e3m0_t &float4_e3m0_t::operator=(float f) {
    raw_bits_ = float2e3m0(f);
    return *this;
}

float4_e3m0_t::operator float() const {
    // List of e3m0 values. The index of each value maps to its encoding.
    static const float e3m0_table[16]
            = {0.0f, .25f, .5f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f, -0.0f, -.25f,
                    -.5f, -1.0f, -2.0f, -4.0f, -8.0f, -16.0f};
    assert(raw_bits_ < 16);
    return e3m0_table[raw_bits_];
}

float4_e3m0_t::operator float16_t() const {
    // List of e3m0 values. The index of each value maps to its encoding.
    static const float16_t e3m0_table[16]
            = {0.0f, .25f, .5f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f, -0.0f, -.25f,
                    -.5f, -1.0f, -2.0f, -4.0f, -8.0f, -16.0f};
    assert(raw_bits_ < 16);
    return e3m0_table[raw_bits_];
}

static const float nf4_table[16] = {
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
};

uint8_t float2nf4(float f) {
    if (f == nf4_table[0x7]) return 0x7; /* fast path for zero */
    if (f <= nf4_table[0x0]) return 0x0;
    if (f >= nf4_table[0xF]) return 0xF;

    // There is no NaN or infinity in e2m1, for now we just return zero
    uint32_t f_raw = float2int(f);
    uint32_t naninf_mask = 0x7f800000;
    if ((f_raw & naninf_mask) == naninf_mask) f = 0.0f;

    int idx = 0;
    float min_diff = ::fabsf(nf4_table[idx] - f);
    uint8_t raw_bits = idx;
    for (++idx; idx < 16; ++idx) {
        float diff = ::fabsf(nf4_table[idx] - f);
        if (diff < min_diff) {
            min_diff = diff;
            raw_bits = idx;
        }
        // Special case for midpoint, we round to even (so even index)
        if ((diff == min_diff) && !(idx & 1)) raw_bits = idx;
    }
    assert(raw_bits < 16);
    return raw_bits;
}

nf4_t &nf4_t::operator=(bfloat16_t f) {
    float f32 = f;
    raw_bits_ = float2nf4(f32);
    return *this;
}

nf4_t &nf4_t::operator=(float16_t f) {
    float f32 = f;
    raw_bits_ = float2nf4(f32);
    return *this;
}

nf4_t &nf4_t::operator=(float f) {
    raw_bits_ = float2nf4(f);
    return *this;
}

nf4_t::operator float() const {
    assert(raw_bits_ < 16);
    return nf4_table[raw_bits_];
}

nf4_t::operator float16_t() const {
    assert(raw_bits_ < 16);
    return float16_t(nf4_table[raw_bits_]);
}

} // namespace impl
} // namespace dnnl
