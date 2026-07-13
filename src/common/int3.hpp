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

#ifndef COMMON_INT3_HPP
#define COMMON_INT3_HPP

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace dnnl {
namespace impl {

struct uint3_t {
    template <typename IntegerType,
            typename SFINAE = typename std::enable_if<
                    std::is_integral<IntegerType>::value>::type>
    constexpr uint3_t(IntegerType raw) : raw_bits_(static_cast<uint8_t>(raw)) {
#if __cplusplus >= 201402L
        assert(0 <= raw && raw <= std::numeric_limits<uint8_t>::max());
#endif
    }
    uint3_t(float val_f32) {
        uint8_t val_uint8 = static_cast<uint8_t>(val_f32);
        raw_bits_ = val_uint8 & 0x7;
    }

    operator float() const { return (float)raw_bits_; }

    uint8_t raw_bits_;
};

static_assert(sizeof(uint3_t) == 1, "uint3_t must be 1 byte");

// u3 uses the OV transposed layout: each group of 8 values packs into 3 bytes
// such that the low 2 bits of values 0-3 are in byte 0, the low 2 bits of
// values 4-7 are in byte 1, and all 8 MSBs are in byte 2. Decodes the 3-bit
// value at logical index `idx` from the packed buffer.
inline uint8_t uint3_unpack(const uint8_t *packed, int64_t idx) {
    const int64_t base = (idx / 8) * 3;
    const int pos = idx % 8;
    const int low2 = (packed[base + pos / 4] >> (6 - 2 * (pos % 4))) & 0x3;
    const int msb = (packed[base + 2] >> (7 - pos)) & 0x1;
    return static_cast<uint8_t>(low2 | (msb << 2));
}

} // namespace impl
} // namespace dnnl

#endif
