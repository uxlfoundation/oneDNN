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

    // invalid strides
    EXPECT_THROW(memory::desc::grouped({4, K}, dt::f32, num_groups,
                         valid_group_dims, dt::s32, {1, 0}),
            dnnl::error);
    EXPECT_THROW(memory::desc::grouped({4, K}, dt::f32, num_groups,
                         valid_group_dims, dt::s32, {K - 1, 1}),
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

    // Test with f32 and s32 offsets
    {
        const memory::data_type data_type = dt::f32;
        const memory::data_type offsets_dt = dt::s32;

        memory::desc md;
        ASSERT_NO_THROW(md = memory::desc::grouped(dims, data_type, num_groups,
                                group_dims, offsets_dt));

        // Basic queries
        ASSERT_EQ(md.get_dims(), dims);
        ASSERT_EQ(md.get_data_type(), data_type);
        ASSERT_EQ(md.get_data_type(0), data_type);
        ASSERT_EQ(md.get_format_kind(), memory::format_kind::sparse);
        ASSERT_EQ(md.get_sparse_encoding(), memory::sparse_encoding::grouped);
        ASSERT_EQ(md.get_data_type(1), offsets_dt);

        // Query nnz
        memory::dim nnz = md.get_nnz();
        ASSERT_EQ(nnz, total_tokens * K);

        // Query strides
        memory::dims strides = md.get_strides();
        ASSERT_EQ(strides.size(), 2);
        ASSERT_EQ(strides[0], K);
        ASSERT_EQ(strides[1], 1);
    }
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

    // different data types
    ASSERT_NO_THROW(md1
            = memory::desc::grouped({4, K}, dt::f32, num_groups, group_dims));
    ASSERT_NO_THROW(md2
            = memory::desc::grouped({4, K}, dt::f16, num_groups, group_dims));
    ASSERT_NE(md1, md2);

    // different offsets data types
    ASSERT_NO_THROW(md1 = memory::desc::grouped(
                            {4, K}, dt::f32, num_groups, group_dims, dt::s32));
    ASSERT_NO_THROW(md2 = memory::desc::grouped(
                            {4, K}, dt::f32, num_groups, group_dims, dt::s8));
    ASSERT_NE(md1, md2);

    // different num_groups
    ASSERT_NO_THROW(md1
            = memory::desc::grouped({4, K}, dt::f32, num_groups, group_dims));
    memory::dims group_dims_3 = {1, K, 1, K, 2, K};
    ASSERT_NO_THROW(
            md2 = memory::desc::grouped({4, K}, dt::f32, 3, group_dims_3));
    ASSERT_NE(md1, md2);

    // different strides (same K but different padding)
    ASSERT_NO_THROW(md1 = memory::desc::grouped({4, K}, dt::f32, num_groups,
                            group_dims, dt::s32, {K, 1}));
    ASSERT_NO_THROW(md2 = memory::desc::grouped({4, K}, dt::f32, num_groups,
                            group_dims, dt::s32, {K + 16, 1}));
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

HANDLE_EXCEPTIONS_FOR_TEST(iface_grouped_test_t, TestGroupedMemoryCreation) {
    engine eng = get_test_engine();

    const bool is_unimplemented = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL);
    if (is_unimplemented) return;

    const int num_groups = 3;
    const int K = 256;
    const int total_tokens = 9;
    const memory::dims group_dims = {1, K, 3, K, 5, K};

    memory::desc md;
    ASSERT_NO_THROW(md = memory::desc::grouped({total_tokens, K}, dt::f32,
                            num_groups, group_dims));

    memory mem;

    // default
    mem = memory(md, eng);

    // user provided buffers (2 buffers: values and offsets)
    {
        const int total_elements = total_tokens * K;
        std::vector<float> values(total_elements);
        std::vector<int32_t> offsets(num_groups + 1);
        offsets = {0, 1, 4, 9};

        EXPECT_NO_THROW(mem = memory(md, eng, {values.data(), offsets.data()}));
    }

    // same, but skipping one group
    {
        const int total_elements = total_tokens * K;
        std::vector<float> values(total_elements);
        std::vector<int32_t> offsets(num_groups + 1);
        offsets = {0, 1, 1, 9}; // skip group 2

        EXPECT_NO_THROW(mem = memory(md, eng, {values.data(), offsets.data()}));
    }
}

HANDLE_EXCEPTIONS_FOR_TEST(
        iface_grouped_test_t, TestGroupedMemorySetGetDataHandles) {
    engine eng = get_test_engine();

    const bool is_unimplemented = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL);
    if (is_unimplemented) return;

    const int num_groups = 3;
    const int K = 256;
    const int total_tokens = 9;
    const memory::dims group_dims = {1, K, 3, K, 5, K};

    memory::desc md;
    ASSERT_NO_THROW(md = memory::desc::grouped({total_tokens, K}, dt::f32,
                            num_groups, group_dims));

    memory mem = memory(md, eng);

    {
        const int total_elements = total_tokens * K;
        std::vector<float> values(total_elements);
        std::vector<int32_t> offsets(num_groups + 1);

        ASSERT_NO_THROW(mem.set_data_handle(values.data(), 0));
        ASSERT_NO_THROW(mem.set_data_handle(offsets.data(), 1));

        ASSERT_EQ(mem.get_data_handle(0), values.data());
        ASSERT_EQ(mem.get_data_handle(1), offsets.data());
    }
}

TEST(iface_grouped_test_t, TestGroupedMemoryMapUnmap) {
    engine eng = get_test_engine();

    const bool is_unimplemented = (eng.get_kind() == engine::kind::gpu
            || DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL);
    if (is_unimplemented) return;

    const int num_groups = 2;
    const int K = 4;
    const int total_tokens = 3;
    const memory::dims group_dims = {1, K, 2, K};
    ASSERT_EQ(num_groups * 2, group_dims.size());

    memory::desc md;
    ASSERT_NO_THROW(md = memory::desc::grouped({total_tokens, K}, dt::f32,
                            num_groups, group_dims, dt::s32));

    const int total_elements = total_tokens * K;
    std::vector<float> values(total_elements);
    for (int i = 0; i < total_elements; i++)
        values[i] = static_cast<float>(i) * 0.5f;

    // compute cumulative offsets based on group_dims
    std::vector<int32_t> offsets = {0};
    for (int g = 0; g < num_groups; g++) {
        int M = group_dims[g * 2];
        offsets.push_back(offsets[g] + M);
    }
    ASSERT_EQ(offsets[num_groups], total_tokens);

    memory mem(md, eng, {values.data(), offsets.data()});

    float *mapped_values = nullptr;
    int32_t *mapped_offsets = nullptr;

    ASSERT_NO_THROW(mapped_values = mem.map_data<float>(0));
    ASSERT_NO_THROW(mapped_offsets = mem.map_data<int32_t>(1));

    for (size_t i = 0; i < values.size(); i++)
        ASSERT_EQ(values[i], mapped_values[i]);

    for (size_t i = 0; i < offsets.size(); i++)
        ASSERT_EQ(offsets[i], mapped_offsets[i]);

    ASSERT_NO_THROW(mem.unmap_data(mapped_values, 0));
    ASSERT_NO_THROW(mem.unmap_data(mapped_offsets, 1));
}

TEST(c_api_grouped_md, TestGroupedMDQueries) {
    const int num_groups = 3;
    const int K = 256;
    const int total_tokens = 9;
    const int total_elements = total_tokens * K;

    dnnl_memory_desc_t md = nullptr;
    dnnl_dims_t dims = {total_tokens, K};
    dnnl_dims_t group_dims = {1, K, 3, K, 5, K};

    DNNL_CHECK(dnnl_memory_desc_create_with_grouped_encoding(&md, 2, dims,
            dnnl_f32, num_groups, 6, group_dims, dnnl_s32, nullptr));
    ASSERT_NE(md, nullptr);

    // Query all properties
    int ndims = -1;
    DNNL_CHECK(dnnl_memory_desc_query(md, dnnl_query_ndims_s32, &ndims));
    EXPECT_EQ(ndims, 2);

    dnnl_data_type_t dtype = dnnl_data_type_undef;
    DNNL_CHECK(dnnl_memory_desc_query(md, dnnl_query_data_type, &dtype));
    EXPECT_EQ(dtype, dnnl_f32);

    dnnl_format_kind_t fmt_kind = dnnl_format_kind_undef;
    DNNL_CHECK(dnnl_memory_desc_query(md, dnnl_query_format_kind, &fmt_kind));
    EXPECT_EQ(fmt_kind, dnnl_format_kind_sparse);

    dnnl_sparse_encoding_t encoding = dnnl_sparse_encoding_undef;
    DNNL_CHECK(dnnl_memory_desc_query_v2(
            md, dnnl_query_sparse_encoding, 0, &encoding));
    EXPECT_EQ(encoding, dnnl_grouped);

    dnnl_dim_t nnz = -1;
    DNNL_CHECK(dnnl_memory_desc_query_v2(md, dnnl_query_nnz_s64, 0, &nnz));
    EXPECT_EQ(nnz, total_elements);

    size_t values_size = dnnl_memory_desc_get_size_v2(md, 0);
    EXPECT_EQ(values_size, total_elements * sizeof(float));

    size_t offsets_size = dnnl_memory_desc_get_size_v2(md, 1);
    EXPECT_EQ(offsets_size, (num_groups + 1) * sizeof(int32_t));

    DNNL_CHECK(dnnl_memory_desc_destroy(md));
}

TEST(c_api_grouped_md, TestGroupedMDWithRuntimeDims) {
    dnnl_memory_desc_t md = nullptr;
    const int K = 256;
    dnnl_dims_t dims = {4, K};
    dnnl_dims_t group_dims = {DNNL_RUNTIME_DIM_VAL, K};

    DNNL_CHECK(dnnl_memory_desc_create_with_grouped_encoding(
            &md, 2, dims, dnnl_f32, 2, 2, group_dims, dnnl_s32, nullptr));
    ASSERT_NE(md, nullptr);

    DNNL_CHECK(dnnl_memory_desc_destroy(md));
}

TEST(c_api_grouped_md, TestGroupedMDInvalidArgs) {
    dnnl_memory_desc_t md = nullptr;
    const int K = 256;
    dnnl_dims_t dims = {4, K};
    dnnl_dims_t group_dims = {1, K, 3, K};

    // 3D is not supported
    dnnl_dims_t dims_3d = {4, K, 10};
    ASSERT_EQ(dnnl_memory_desc_create_with_grouped_encoding(&md, 3, dims_3d,
                      dnnl_f32, 2, 4, group_dims, dnnl_s32, nullptr),
            dnnl_unimplemented);
    EXPECT_EQ(md, nullptr);

    // non-uniform K dimensions
    dnnl_dims_t non_uniform_k = {1, K, 3, K + 1};
    ASSERT_EQ(dnnl_memory_desc_create_with_grouped_encoding(&md, 2, dims,
                      dnnl_f32, 2, 4, non_uniform_k, dnnl_s32, nullptr),
            dnnl_invalid_arguments);
    EXPECT_EQ(md, nullptr);

    // RUNTIME_DIM_VAL in K dimension is not supported
    dnnl_dims_t runtime_k = {1, DNNL_RUNTIME_DIM_VAL, 3, K};
    ASSERT_EQ(dnnl_memory_desc_create_with_grouped_encoding(&md, 2, dims,
                      dnnl_f32, 2, 4, runtime_k, dnnl_s32, nullptr),
            dnnl_invalid_arguments);
    EXPECT_EQ(md, nullptr);

    // invalid group count
    ASSERT_EQ(dnnl_memory_desc_create_with_grouped_encoding(&md, 2, dims,
                      dnnl_f32, 0, 4, group_dims, dnnl_s32, nullptr),
            dnnl_invalid_arguments);
    EXPECT_EQ(md, nullptr);

    ASSERT_EQ(dnnl_memory_desc_create_with_grouped_encoding(&md, 2, dims,
                      dnnl_f32, -1, 4, group_dims, dnnl_s32, nullptr),
            dnnl_invalid_arguments);
    EXPECT_EQ(md, nullptr);

    // null pointer md
    dnnl_status_t status = dnnl_memory_desc_create_with_grouped_encoding(
            nullptr, 2, dims, dnnl_f32, 2, 4, group_dims, dnnl_s32, nullptr);
    ASSERT_EQ(status, dnnl_invalid_arguments);
}

} // namespace dnnl
