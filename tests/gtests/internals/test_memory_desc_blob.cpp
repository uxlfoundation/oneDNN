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

#include <cstdint>
#include <vector>

#include "tests/gtests/dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "dnnl.hpp"

#include "common/memory_desc.hpp"

namespace dnnl {

// dnnl_memory_desc_create_with_blob copies a raw memory_desc_t out of the
// caller-supplied blob, so every field is attacker-controlled. Consumers such
// as dnnl_memory_desc_get_size index fixed-size dims_t arrays by these fields
// (compute_blocks does `blocks[inner_idxs[i]] *= ...`). A blob whose ndims or
// blocking.inner_idxs are out of range must be rejected rather than driving an
// out-of-bounds access.

namespace {
std::vector<uint8_t> blob_of(dnnl_memory_desc_t md) {
    size_t blob_sz = 0;
    EXPECT_EQ(dnnl_memory_desc_get_blob(nullptr, &blob_sz, md), dnnl_success);
    std::vector<uint8_t> blob(blob_sz);
    EXPECT_EQ(
            dnnl_memory_desc_get_blob(blob.data(), &blob_sz, md), dnnl_success);
    return blob;
}
} // namespace

TEST(test_memory_desc_blob, ValidBlobRoundTrips) {
    dnnl_memory_desc_t md {};
    dnnl_dims_t dims = {2, 16, 4, 4};
    ASSERT_EQ(dnnl_memory_desc_create_with_tag(
                      &md, 4, dims, dnnl_f32, dnnl_aBcd16b),
            dnnl_success);

    const std::vector<uint8_t> blob = blob_of(md);

    dnnl_memory_desc_t rt {};
    ASSERT_EQ(
            dnnl_memory_desc_create_with_blob(&rt, blob.data()), dnnl_success);
    EXPECT_EQ(dnnl_memory_desc_get_size(rt), dnnl_memory_desc_get_size(md));

    dnnl_memory_desc_destroy(rt);
    dnnl_memory_desc_destroy(md);
}

TEST(test_memory_desc_blob, RejectsOutOfBoundsInnerIdx) {
    dnnl_memory_desc_t md {};
    dnnl_dims_t dims = {2, 16, 4, 4};
    ASSERT_EQ(dnnl_memory_desc_create_with_tag(
                      &md, 4, dims, dnnl_f32, dnnl_aBcd16b),
            dnnl_success);

    std::vector<uint8_t> blob = blob_of(md);
    reinterpret_cast<impl::memory_desc_t *>(blob.data())
            ->format_desc.blocking.inner_idxs[0]
            = 100000;

    dnnl_memory_desc_t bad = nullptr;
    EXPECT_EQ(dnnl_memory_desc_create_with_blob(&bad, blob.data()),
            dnnl_invalid_arguments);
    EXPECT_EQ(bad, nullptr);

    dnnl_memory_desc_destroy(md);
}

TEST(test_memory_desc_blob, RejectsOutOfBoundsNdims) {
    dnnl_memory_desc_t md {};
    dnnl_dims_t dims = {2, 16, 4, 4};
    ASSERT_EQ(dnnl_memory_desc_create_with_tag(
                      &md, 4, dims, dnnl_f32, dnnl_aBcd16b),
            dnnl_success);

    std::vector<uint8_t> blob = blob_of(md);
    reinterpret_cast<impl::memory_desc_t *>(blob.data())->ndims = 1000;

    dnnl_memory_desc_t bad = nullptr;
    EXPECT_EQ(dnnl_memory_desc_create_with_blob(&bad, blob.data()),
            dnnl_invalid_arguments);
    EXPECT_EQ(bad, nullptr);

    dnnl_memory_desc_destroy(md);
}

} // namespace dnnl
