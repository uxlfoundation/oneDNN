/*******************************************************************************
* Copyright 2025 Intel Corporation
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

using dt = memory::data_type;

class iface_grouped_test_t : public ::testing::Test {};

TEST(iface_grouped_test_t, TestGroupedMDCreation) {
    const int num_groups = 3;
    const int K = 256;

    memory::desc md;

    // shared group_dims
    memory::dims shared_dims = {DNNL_RUNTIME_DIM_VAL, K};
    ASSERT_NO_THROW(md
            = memory::desc::grouped({9, K}, dt::f32, num_groups, shared_dims));

    // per-group dims (resolved M per group)
    memory::dims per_group_dims = {1, K, 3, K, 5, K};
    ASSERT_NO_THROW(md = memory::desc::grouped(
                            {9, K}, dt::f32, num_groups, per_group_dims));

    // per-group dims with empty dimension (resolved M per group)
    memory::dims per_group_dims_empty = {1, K, 0, K, 8, K};
    ASSERT_NO_THROW(md = memory::desc::grouped(
                            {9, K}, dt::f32, num_groups, per_group_dims_empty));

    // row-major layout
    ASSERT_NO_THROW(md = memory::desc::grouped({9, K}, dt::f32, num_groups,
                            shared_dims, dt::s32, {K, 1}));

    // row-major with "padding"
    ASSERT_NO_THROW(memory::desc::grouped(
            {4, K}, dt::f32, num_groups, shared_dims, dt::s32, {K + 1, 1}));

    // different data types
    ASSERT_NO_THROW(md
            = memory::desc::grouped({9, K}, dt::f16, num_groups, shared_dims));
    ASSERT_NO_THROW(md
            = memory::desc::grouped({9, K}, dt::bf16, num_groups, shared_dims));
}

TEST(iface_grouped_test_t, TestGroupedMDInvalidArgs) {
    const int num_groups = 2;
    const int K = 256;
    memory::dims valid_group_dims = {1, K, 3, K};

    // 3D is not supported
    memory::dims dims_3d = {4, K, 10};
    EXPECT_THROW(memory::desc::grouped(
                         dims_3d, dt::f32, num_groups, valid_group_dims),
            dnnl::error);

    // non-uniform K dimensions (group_dims[1] != group_dims[3])
    memory::dims non_uniform_k = {1, K, 3, K + 1};
    EXPECT_THROW(
            memory::desc::grouped({4, K}, dt::f32, num_groups, non_uniform_k),
            dnnl::error);

    // RUNTIME_DIM_VAL in K dimension is not supported
    memory::dims runtime_k = {1, DNNL_RUNTIME_DIM_VAL, 3, K};
    EXPECT_THROW(memory::desc::grouped({4, K}, dt::f32, num_groups, runtime_k),
            dnnl::error);

    // column-major is not supported
    EXPECT_THROW(memory::desc::grouped({4, K}, dt::f32, num_groups,
                         valid_group_dims, dt::s32, {1, 0}),
            dnnl::error);

    // invalid strides
    EXPECT_THROW(memory::desc::grouped({4, K}, dt::f32, num_groups,
                         valid_group_dims, dt::s32, {2, 2}),
            dnnl::error);
    EXPECT_THROW(memory::desc::grouped({4, K}, dt::f32, num_groups,
                         valid_group_dims, dt::s32, {K - 1, 1}),
            dnnl::error);
    EXPECT_THROW(memory::desc::grouped({4, K}, dt::f32, num_groups,
                         valid_group_dims, dt::s32, {K, 0}),
            dnnl::error);

    // invalid group count
    EXPECT_THROW(memory::desc::grouped({4, K}, dt::f32, 0, valid_group_dims),
            dnnl::error);
    EXPECT_THROW(memory::desc::grouped({4, K}, dt::f32, -1, valid_group_dims),
            dnnl::error);

    // invalid inner block dims
    memory::dims single_k = {K};
    EXPECT_THROW(memory::desc::grouped({4, K}, dt::f32, num_groups, single_k),
            dnnl::error);
}

TEST(iface_grouped_test_t, TestGroupedMDQueries) {
    const int num_groups = 3;
    const int K = 256;
    const int total_tokens = 9;
    const memory::dims dims = {total_tokens, K};
    const memory::dims group_dims = {1, K, 3, K, 5, K};
    const memory::data_type data_type = dt::f32;
    const memory::data_type offsets_dt = dt::s32;

    memory::desc md;
    ASSERT_NO_THROW(md = memory::desc::grouped(dims, data_type, num_groups,
                            group_dims, offsets_dt));

    ASSERT_EQ(md.get_dims(), dims);
    ASSERT_EQ(md.get_data_type(), data_type);
    ASSERT_EQ(md.get_data_type(0), data_type);
    ASSERT_EQ(md.get_format_kind(), memory::format_kind::sparse);
    ASSERT_EQ(md.get_sparse_encoding(), memory::sparse_encoding::grouped);
    ASSERT_EQ(md.get_data_type(1), offsets_dt);
}

TEST(iface_grouped_test_t, TestGroupedMDComparison) {
    const int num_groups = 2;
    const int K = 256;
    memory::dims group_dims = {1, K, 3, K};

    memory::desc md1, md2;

    // equal descriptors
    ASSERT_NO_THROW(md1
            = memory::desc::grouped({4, K}, dt::f32, num_groups, group_dims));
    ASSERT_NO_THROW(md2
            = memory::desc::grouped({4, K}, dt::f32, num_groups, group_dims));
    ASSERT_EQ(md1, md2);

    // explicit vs inferred row-major
    ASSERT_NO_THROW(md1
            = memory::desc::grouped({4, K}, dt::f32, num_groups, group_dims));
    ASSERT_NO_THROW(md2 = memory::desc::grouped({4, K}, dt::f32, num_groups,
                            group_dims, dt::s32, {K, 1}));
    ASSERT_EQ(md1, md2);

    // different data types
    ASSERT_NO_THROW(md1
            = memory::desc::grouped({4, K}, dt::f32, num_groups, group_dims));
    ASSERT_NO_THROW(md2
            = memory::desc::grouped({4, K}, dt::f16, num_groups, group_dims));
    ASSERT_NE(md1, md2);

    // different num_groups
    ASSERT_NO_THROW(md1
            = memory::desc::grouped({4, K}, dt::f32, num_groups, group_dims));
    memory::dims group_dims_3 = {1, K, 1, K, 2, K};
    ASSERT_NO_THROW(
            md2 = memory::desc::grouped({4, K}, dt::f32, 3, group_dims_3));
    ASSERT_NE(md1, md2);
}

TEST(iface_grouped_test_t, TestGroupedMDSize) {
    const int num_groups = 3;
    const int K = 256;
    const int total_tokens = 9;
    const memory::dims group_dims = {1, K, 3, K, 5, K};

    memory::desc md;
    ASSERT_NO_THROW(md = memory::desc::grouped({total_tokens, K}, dt::f32,
                            num_groups, group_dims, dt::s32));

    // Size of values buffer (buffer 0): total_tokens * K elements
    size_t ref_values_size = total_tokens * K * memory::data_type_size(dt::f32);
    ASSERT_EQ(md.get_size(0), ref_values_size);

    // Size of offsets buffer (buffer 1): num_groups + 1
    size_t ref_offsets_size
            = (num_groups + 1) * memory::data_type_size(dt::s32);
    ASSERT_EQ(md.get_size(1), ref_offsets_size);
}

} // namespace dnnl
