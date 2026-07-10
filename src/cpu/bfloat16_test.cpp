/*******************************************************************************
* Copyright 2020 Intel Corporation
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

// Next-to-source internal unit test for common/bfloat16.hpp.
//
// Unlike the legacy tests under tests/gtests/internals/, this file lives right
// next to the code it exercises and depends on nothing but the header under
// test and gtest. The build system links it against a static archive of the
// dnnl_common object library, from which the linker pulls only bfloat16.o.

#include <cfloat>
#include <limits>
#include <type_traits>

#include "gtest/gtest.h"

#include "common/bfloat16.hpp"

namespace dnnl {
namespace impl {
namespace {

template <typename IntegerType>
void assert_same_bits_converted_from_integer_as_from_float(
        const IntegerType value) {
    const bfloat16_t constructed_from_float {static_cast<float>(value)};
    const bfloat16_t constructed_from_integer {value};
    ASSERT_EQ(constructed_from_integer.raw_bits_,
            constructed_from_float.raw_bits_);
}

template <typename IntegerType>
void assert_same_bits_assigned_from_integer_as_from_float(
        const IntegerType value) {
    bfloat16_t assigned_from_float;
    assigned_from_float = static_cast<float>(value);
    bfloat16_t assigned_from_integer;
    assigned_from_integer = value;
    ASSERT_EQ(assigned_from_integer.raw_bits_, assigned_from_float.raw_bits_);
}

template <typename IntegerType>
void assert_same_bits_from_integer_as_from_float(const IntegerType value) {
    assert_same_bits_converted_from_integer_as_from_float(value);
    assert_same_bits_assigned_from_integer_as_from_float(value);
}

template <typename IntegerType>
void assert_same_bits_from_nonnegative_integer_as_from_float() {
    assert_same_bits_from_integer_as_from_float<IntegerType>(0);
    assert_same_bits_from_integer_as_from_float<IntegerType>(1);
    constexpr auto max_value = std::numeric_limits<IntegerType>::max();
    assert_same_bits_from_integer_as_from_float(max_value - 1);
    assert_same_bits_from_integer_as_from_float(max_value);
}

template <typename SignedType>
void assert_same_bits_from_integer_as_from_float() {
    constexpr auto min_value = std::numeric_limits<SignedType>::min();
    assert_same_bits_from_integer_as_from_float(min_value);
    assert_same_bits_from_integer_as_from_float(min_value + 1);
    assert_same_bits_from_integer_as_from_float<SignedType>(-1);

    assert_same_bits_from_nonnegative_integer_as_from_float<SignedType>();
    using UnsignedType = typename std::make_unsigned<SignedType>::type;
    assert_same_bits_from_nonnegative_integer_as_from_float<UnsignedType>();
}

} // namespace

TEST(test_bfloat16_plus_float, TestDenormF32) {
    const float denorm_f32 {FLT_MIN / 2.0f};
    const bfloat16_t initial_value_bf16 {FLT_MIN};

    bfloat16_t expect_bf16 {float {initial_value_bf16} + denorm_f32};
    ASSERT_GT(float {expect_bf16}, float {initial_value_bf16});

    bfloat16_t bf16_plus_equal_f32 = initial_value_bf16;
    bf16_plus_equal_f32 += denorm_f32;
    ASSERT_EQ(float {bf16_plus_equal_f32}, float {expect_bf16});

    bfloat16_t bf16_plus_f32 = initial_value_bf16 + denorm_f32;
    ASSERT_EQ(float {bf16_plus_f32}, float {expect_bf16});
}

TEST(test_bfloat16_denorm_f32, BitsFromDoubleSameAsFromFloat) {
    constexpr float denorm_f32 {FLT_MIN / 2.0f};
    const auto expected_bits = bfloat16_t {denorm_f32}.raw_bits_;

    EXPECT_EQ(bfloat16_t {double {denorm_f32}}.raw_bits_, expected_bits);

    bfloat16_t bf16;
    bf16 = double {denorm_f32};
    EXPECT_EQ(bf16.raw_bits_, expected_bits);
}

TEST(test_bfloat16_converting_constructor_and_assignment,
        BitsFromIntegerSameAsFromFloat) {
    assert_same_bits_from_integer_as_from_float<signed char>();
    assert_same_bits_from_integer_as_from_float<short>();
    assert_same_bits_from_integer_as_from_float<int>();
    assert_same_bits_from_integer_as_from_float<long>();
    assert_same_bits_from_integer_as_from_float<long long>();
}

TEST(test_bfloat16_rounding, RoundToNearestEven) {
    constexpr float halfway_odd_lsb {1.01171875f};
    ASSERT_EQ(bfloat16_t {halfway_odd_lsb}.raw_bits_, 0x3F82u);

    constexpr float halfway_even_lsb {1.00390625f};
    ASSERT_EQ(bfloat16_t {halfway_even_lsb}.raw_bits_, 0x3F80u);
}

} // namespace impl
} // namespace dnnl
