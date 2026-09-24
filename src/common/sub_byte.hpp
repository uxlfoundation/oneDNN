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

#ifndef COMMON_SUB_BYTE_HPP
#define COMMON_SUB_BYTE_HPP

#include <cassert>
#include <cstdint>

#include "c_types_map.hpp"

namespace dnnl {
namespace impl {

// Returns element bit width for sub-byte data types and 0 otherwise
constexpr int sub_byte_bits(data_type_t dt) {
    return dt == data_type::u2 ? 2
            : (dt == data_type::s4 || dt == data_type::u4
                      || dt == data_type::f4_e2m1)
            ? 4
            : 0;
}

// Returns smallest number of elements that fill whole bytes
constexpr int sub_byte_nelems(int bits) {
    return bits > 0 ? 8 / (bits & -bits) : 1;
}

// Sub-byte elements are packed contiguously, such that element `i` occupies bits
// [i*bits, i*bits + bits) starting from least significant bit, straddling
// into the next byte when the width does not divide the byte boundary

// Returns the raw bits of element idx
template <int bits>
uint8_t sub_byte_get(const uint8_t *base, int64_t idx) {
    static_assert(bits > 1 && bits <= 4, "unsupported sub-byte width");
    assert(idx >= 0);
    constexpr uint8_t mask = static_cast<uint8_t>((1 << bits) - 1);
    const int64_t bit = idx * bits, byte = bit >> 3;
    const int sh = static_cast<int>(bit & 7);
    uint8_t v = base[byte] >> sh;
    if (sh + bits > 8) v |= base[byte + 1] << (8 - sh); // straddle
    return v & mask;
}

// Writes the low bits of value into element idx
template <int bits>
void sub_byte_set(uint8_t *base, int64_t idx, uint8_t val) {
    static_assert(bits > 1 && bits <= 4, "unsupported sub-byte width");
    assert(idx >= 0);
    constexpr uint8_t mask = static_cast<uint8_t>((1 << bits) - 1);
    const int64_t bit = idx * bits, byte = bit >> 3;
    const int sh = static_cast<int>(bit & 7);
    val &= mask;
    base[byte] = static_cast<uint8_t>(
            (base[byte] & ~((mask << sh) & 0xFF)) | ((val << sh) & 0xFF));
    if (sh + bits > 8) { // high bits straddle into the next byte
        const int lo = 8 - sh;
        base[byte + 1] = static_cast<uint8_t>(
                (base[byte + 1] & ~((1 << (bits - lo)) - 1)) | (val >> lo));
    }
}

} // namespace impl
} // namespace dnnl

#endif
