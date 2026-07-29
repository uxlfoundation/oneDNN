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

// u3 uses the OV transposed layout:
//
//         bit7 bit6 bit5 bit4 bit3 bit2 bit1 bit0
//        ┌────┬────┬────┬────┬────┬────┬────┬────┐
// byte0  │v0.1│v0.0│v1.1│v1.0│v2.1│v2.0│v3.1│v3.0│ low 2 bits of v0..v3
//        ├────┼────┼────┼────┼────┼────┼────┼────┤
// byte1  │v4.1│v4.0│v5.1│v5.0│v6.1│v6.0│v7.1│v7.0│ low 2 bits of v4..v7
//        ├────┼────┼────┼────┼────┼────┼────┼────┤
// byte2  │v0.2│v1.2│v2.2│v3.2│v4.2│v5.2│v6.2│v7.2│ MSB of all
//        └────┴────┴────┴────┴────┴────┴────┴────┘
//
inline uint8_t uint3_unpack(const uint8_t *packed, int64_t idx) {
    const int64_t base = (idx / 8) * 3;
    const int pos = idx % 8;
    const int low2 = (packed[base + pos / 4] >> (6 - 2 * (pos % 4))) & 0x3;
    const int msb = (packed[base + 2] >> (7 - pos)) & 0x1;
    return static_cast<uint8_t>(low2 | (msb << 2));
}
inline void uint3_pack(uint8_t *packed, int64_t idx, uint8_t v) {
    const int64_t base = (idx / 8) * 3;
    const int pos = idx % 8;
    const int lshift = 6 - 2 * (pos % 4);
    packed[base + pos / 4]
            = static_cast<uint8_t>((packed[base + pos / 4] & ~(0x3 << lshift))
                    | ((v & 0x3) << lshift));
    const int hshift = 7 - pos;
    packed[base + 2]
            = static_cast<uint8_t>((packed[base + 2] & ~(0x1 << hshift))
                    | (((v >> 2) & 0x1) << hshift));
}

} // namespace impl
} // namespace dnnl

#endif
