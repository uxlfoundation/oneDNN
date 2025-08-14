/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {

class memory_desc_test_t : public ::testing::Test {};

HANDLE_EXCEPTIONS_FOR_TEST(memory_desc_test_t, TestQueryDefaultConstructor) {
    memory::desc md;
    EXPECT_EQ(md.get_ndims(), 0);
    EXPECT_EQ(md.get_size(), size_t(0));
    EXPECT_EQ(md.get_submemory_offset(), 0);
    EXPECT_EQ(md.get_inner_nblks(), 0);
    EXPECT_EQ(md.get_format_kind(), memory::format_kind::undef);
    EXPECT_EQ(md.get_data_type(), memory::data_type::undef);

    EXPECT_TRUE(md.is_zero());
    EXPECT_TRUE(md.get_dims().empty());
    EXPECT_TRUE(md.get_padded_dims().empty());
    EXPECT_TRUE(md.get_padded_offsets().empty());
    EXPECT_TRUE(md.get_strides().empty());
    EXPECT_TRUE(md.get_inner_blks().empty());
    EXPECT_TRUE(md.get_inner_idxs().empty());
}

HANDLE_EXCEPTIONS_FOR_TEST(memory_desc_test_t, TestQueryBlockedFormat) {
    const memory::dims dims = {32, 64, 3, 3};
    const memory::data_type dt = memory::data_type::f32;
    auto md = memory::desc(dims, dt, memory::format_tag::ABcd16a16b);

    const auto exp_format_kind = memory::format_kind::blocked;
    const memory::dim exp_submemory_offset = 0;
    const memory::dims exp_padded_offsets = {0, 0, 0, 0};
    const memory::dims exp_inner_blks = {16, 16};
    const memory::dims exp_inner_idxs = {0, 1};
    const memory::dims exp_strides = {9216, 2304, 768, 256};
    const size_t exp_size = 73728;

    EXPECT_EQ(md.get_ndims(), (int)dims.size());
    EXPECT_EQ(md.get_submemory_offset(), exp_submemory_offset);
    EXPECT_EQ(md.get_inner_nblks(), (int)exp_inner_blks.size());
    EXPECT_EQ(md.get_format_kind(), exp_format_kind);
    EXPECT_EQ(md.get_data_type(), dt);

    EXPECT_EQ(md.get_dims(), dims);
    EXPECT_EQ(md.get_padded_dims(), dims);
    EXPECT_EQ(md.get_padded_offsets(), exp_padded_offsets);
    EXPECT_EQ(md.get_strides(), exp_strides);
    EXPECT_EQ(md.get_inner_blks(), exp_inner_blks);
    EXPECT_EQ(md.get_inner_idxs(), exp_inner_idxs);
    EXPECT_EQ(md.get_size(), exp_size);

    EXPECT_TRUE(!md.is_zero());
}

HANDLE_EXCEPTIONS_FOR_TEST(memory_desc_test_t, TestComparison) {
    auto zero_md = memory::desc();
    auto blocked_md = memory::desc({32, 64, 3, 3}, memory::data_type::f32,
            memory::format_tag::ABcd16a16b);
    auto plain_md = memory::desc(
            {32, 64, 3, 3}, memory::data_type::f32, memory::format_tag::abcd);

    EXPECT_EQ(zero_md, memory::desc {});
    EXPECT_EQ(zero_md, zero_md);
    EXPECT_EQ(plain_md, plain_md);
    EXPECT_EQ(blocked_md, blocked_md);

    EXPECT_NE(zero_md, plain_md);
    EXPECT_NE(zero_md, blocked_md);
    EXPECT_NE(plain_md, blocked_md);
}

TEST(c_api_host_scalar_md, TestHostScalarF32) {
    dnnl_memory_desc_t scalar_md = nullptr;
    DNNL_CHECK(dnnl_memory_desc_create_host_scalar(&scalar_md, dnnl_f32));

    int ndims = -1;
    dnnl_data_type_t dtype = dnnl_data_type_undef;
    dnnl_format_kind_t fmt_kind = dnnl_format_kind_undef;

    DNNL_CHECK(dnnl_memory_desc_query(scalar_md, dnnl_query_ndims_s32, &ndims));
    EXPECT_EQ(ndims, 1);

    DNNL_CHECK(dnnl_memory_desc_query(scalar_md, dnnl_query_data_type, &dtype));
    EXPECT_EQ(dtype, dnnl_f32);

    DNNL_CHECK(dnnl_memory_desc_query(
            scalar_md, dnnl_query_format_kind, &fmt_kind));
    EXPECT_EQ(fmt_kind, dnnl_format_kind_host_scalar);

    size_t sz = dnnl_memory_desc_get_size(scalar_md);
    EXPECT_EQ(sz, sizeof(float));

    DNNL_CHECK(dnnl_memory_desc_destroy(scalar_md));
}

TEST(c_api_host_scalar_md, TestHostScalarInvalidType) {
    dnnl_memory_desc_t scalar_md = nullptr;

    ASSERT_EQ(dnnl_memory_desc_create_host_scalar(
                      &scalar_md, (dnnl_data_type_t)999),
            dnnl_invalid_arguments);

    ASSERT_EQ(dnnl_memory_desc_create_host_scalar(
                      &scalar_md, dnnl_data_type_undef),
            dnnl_invalid_arguments);
}

TEST(c_api_host_scalar_md, TestHostScalarNullPtr) {
    dnnl_status_t status
            = dnnl_memory_desc_create_host_scalar(nullptr, dnnl_f32);
    ASSERT_EQ(status, dnnl_invalid_arguments);
}

TEST(cpp_api_host_scalar_md, TestHostScalarF32) {
    using namespace dnnl;
    auto md = memory::desc::host_scalar(memory::data_type::f32);

    EXPECT_EQ(md.get_ndims(), 1);
    EXPECT_EQ(md.get_data_type(), memory::data_type::f32);
    EXPECT_EQ(md.get_format_kind(), memory::format_kind::host_scalar);
    EXPECT_EQ(md.get_size(), sizeof(float));
}

TEST(cpp_api_host_scalar_md, TestHostScalarInvalidType) {
    using namespace dnnl;

    EXPECT_THROW(
            memory::desc::host_scalar((memory::data_type)999), dnnl::error);
    EXPECT_THROW(
            memory::desc::host_scalar(memory::data_type::undef), dnnl::error);
}

} // namespace dnnl
