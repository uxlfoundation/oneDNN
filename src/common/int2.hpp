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

#ifndef COMMON_INT2_HPP
#define COMMON_INT2_HPP

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace dnnl {
namespace impl {

struct uint2_t {
    template <typename IntegerType,
            typename SFINAE = typename std::enable_if<
                    std::is_integral<IntegerType>::value>::type>
    constexpr uint2_t(IntegerType raw) : raw_bits_(static_cast<uint8_t>(raw)) {
#if __cplusplus >= 201402L
        assert(0 <= raw && raw <= std::numeric_limits<uint8_t>::max());
#endif
    }
    uint2_t(float val_f32) {
        uint8_t val_uint8 = static_cast<uint8_t>(val_f32);
        raw_bits_ = val_uint8 & 0x3;
    }

    operator float() const { return (float)raw_bits_; }

    uint8_t raw_bits_;
};

static_assert(sizeof(uint2_t) == 1, "uint2_t must be 1 byte");

struct int2_t {
    template <typename IntegerType,
            typename SFINAE = typename std::enable_if<
                    std::is_integral<IntegerType>::value>::type>
    constexpr int2_t(IntegerType i) : raw_bits_(static_cast<uint8_t>(i)) {}
    int2_t(float val_f32) {
        int8_t val_int8 = static_cast<int8_t>(val_f32);
        bool negative = val_f32 < 0;
        // positive numbers have the most significant bit set to 0
        // negative numbers have the most significant bit set to 1
        raw_bits_ = negative ? (val_int8 & 0x3) | 0x2 : val_int8 & 0x1;
    }

    operator float() const {
        float sign = (raw_bits_ & (1 << 1)) ? -1.f : 1.f;
        return sign * (float)(sign == -1 ? (~raw_bits_ & 0x3) + 1 : raw_bits_);
    }

    uint8_t raw_bits_;
};

static_assert(sizeof(int2_t) == 1, "int2_t must be 1 byte");

} // namespace impl
} // namespace dnnl

#endif
