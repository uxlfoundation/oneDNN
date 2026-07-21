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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include <cstdint>
#include <vector>

#include "src/common/serialization.hpp"

namespace dnnl {

// An element count read from the stream that exceeds the destination capacity
// must be refused instead of copied. quant_entry_t::deserialize relies on this
// to keep group_ndims from overrunning the fixed group_dims_ array.
TEST(test_serialization, pop_array_rejects_oversized_count) {
    using namespace impl;

    constexpr size_t capacity = 12; // DNNL_MAX_NDIMS
    constexpr size_t bogus_count = 100;
    constexpr int64_t sentinel = 0x5A5A5A5A5A5A5A5ALL;

    serialization_stream_t s;
    s.append(bogus_count);
    for (size_t i = 0; i < bogus_count; i++)
        s.append<int64_t>(static_cast<int64_t>(0x4141414141414141LL));

    // Buffer larger than `capacity`; the slots past `capacity` are canaries
    // that a copy honoring the capacity bound can never touch.
    std::vector<int64_t> buf(capacity + 8, sentinel);

    deserializer_t d(s);
    size_t n = 0;
    d.pop_array(n, buf.data(), capacity);

    EXPECT_EQ(n, 0u);
    for (size_t i = capacity; i < buf.size(); i++)
        EXPECT_EQ(buf[i], sentinel) << "write past capacity at index " << i;
}

// A well-formed array whose count fits the destination round-trips unchanged.
TEST(test_serialization, pop_array_roundtrips_valid_count) {
    using namespace impl;

    constexpr size_t capacity = 12; // DNNL_MAX_NDIMS
    const int64_t src[3] = {5, 7, 9};

    serialization_stream_t s;
    s.append_array(3, src);

    int64_t dst[capacity] = {0};
    deserializer_t d(s);
    size_t n = 0;
    d.pop_array(n, dst, capacity);

    ASSERT_EQ(n, 3u);
    EXPECT_EQ(dst[0], 5);
    EXPECT_EQ(dst[1], 7);
    EXPECT_EQ(dst[2], 9);
}

} // namespace dnnl
