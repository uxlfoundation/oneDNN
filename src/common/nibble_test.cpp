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

#include "gtest/gtest.h"

#include "common/dnnl_thread.hpp"
#include "common/float4.hpp"
#include "common/int4.hpp"
#include "common/nibble.hpp"
#include "common/nstl.hpp"

namespace dnnl {
namespace impl {

template <typename T>
void test_limits(float max, float lowest, float epsilon) {
    ASSERT_EQ(max, static_cast<float>(nstl::numeric_limits<T>::max()));
    ASSERT_EQ(lowest, static_cast<float>(nstl::numeric_limits<T>::lowest()));
    ASSERT_EQ(epsilon, static_cast<float>(nstl::numeric_limits<T>::epsilon()));
}

TEST(test_limits, int4) {
    test_limits<int4_t>(7.f, -8.f, 0.f);
}

TEST(test_limits, uint4) {
    test_limits<uint4_t>(15.f, 0.f, 0.f);
}

TEST(test_limits, f4_e2m1) {
    test_limits<float4_e2m1_t>(6.0f, -6.0f, 1.0f);
}

template <typename T>
void test_conversions() {
    parallel_nd(0xff, [=](uint8_t u8) {
        // Each uint8_t contains a pair of 4-bit numbers.
        // Convert T -> f32 and back again,
        // expecting bitwise identical values.
        nibble2_t T_pair(u8);
        float num1 = static_cast<T>(T_pair.get(0));
        float num2 = static_cast<T>(T_pair.get(1));
        // Check that the all numbers are in the range
        float T_lowest = static_cast<float>(nstl::numeric_limits<T>::lowest());
        float T_max = static_cast<float>(nstl::numeric_limits<T>::max());
        ASSERT_TRUE(num1 >= T_lowest && num1 <= T_max);
        ASSERT_TRUE(num2 >= T_lowest && num2 <= T_max);

        // Check that the numbers are extracted in the right order
        if (u8 <= 0xf)
            ASSERT_TRUE(num2 == 0);
        else
            // only case for num2 == -num2 is with fp types and signed 0.
            ASSERT_TRUE(num2 != 0 || num2 == -num2);

        // The target value must be initialized
        nibble2_t new_T_pair(static_cast<T>(T_pair.get(0)).raw_bits_,
                static_cast<T>(T_pair.get(1)).raw_bits_);
        ASSERT_EQ(T_pair.get(), new_T_pair.get());
    });
}

TEST(test_int4_conversion, int4) {
    test_conversions<int4_t>();
}

TEST(test_int4_conversion, uint4) {
    test_conversions<uint4_t>();
}

TEST(test_e2m1_conversion, f4_e2m1) {
    test_conversions<float4_e2m1_t>();
}

} // namespace impl
} // namespace dnnl
