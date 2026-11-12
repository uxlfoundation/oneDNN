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

#include "oneapi/dnnl/dnnl_config.h"

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY

#include "dnnl_test_common.hpp"
#include "dnnl_test_macros.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {

using dt = memory::data_type;
using tag = memory::format_tag;

class iface_grouped_test_t : public ::testing::Test {};

TEST(iface_grouped_test_t, TestGroupedMDCreation) {
    const int ngroups = 3;
    const int K = 256;
    const int total_tokens = 9;

    // Variable M and variable K with the default layout
    ASSERT_NO_THROW(memory::desc::grouped(
            {ngroups, DNNL_VARIABLE_DIM_VAL, K}, dt::f32, total_tokens));
    ASSERT_NO_THROW(memory::desc::grouped(
            {ngroups, DNNL_VARIABLE_DIM_VAL, K}, dt::f16, 4, {}, dt::s32));
    ASSERT_NO_THROW(memory::desc::grouped(
            {ngroups, K, DNNL_VARIABLE_DIM_VAL}, dt::f32, total_tokens));

    // Variable dim outermost or contiguous, with a padded leading dim
    ASSERT_NO_THROW(memory::desc::grouped({ngroups, DNNL_VARIABLE_DIM_VAL, K},
            dt::f32, total_tokens, {0, K + 16, 1}));
    ASSERT_NO_THROW(memory::desc::grouped({ngroups, DNNL_VARIABLE_DIM_VAL, K},
            dt::f32, total_tokens, {0, 1, total_tokens}));
    ASSERT_NO_THROW(memory::desc::grouped({ngroups, K, DNNL_VARIABLE_DIM_VAL},
            dt::f32, total_tokens, {0, total_tokens, 1}));
    ASSERT_NO_THROW(memory::desc::grouped({ngroups, K, DNNL_VARIABLE_DIM_VAL},
            dt::f32, total_tokens, {0, 1, K + 3}));
}

TEST(iface_grouped_test_t, TestGroupedMDInvalidArgs) {
    const int ngroups = 2;
    const int total_tokens = 100, K = 256;

    // Only 3D is supported
    EXPECT_THROW(memory::desc::grouped(
                         {DNNL_VARIABLE_DIM_VAL, K}, dt::f32, total_tokens),
            dnnl::error);
    EXPECT_THROW(memory::desc::grouped({ngroups, DNNL_VARIABLE_DIM_VAL, K, 10},
                         dt::f32, total_tokens),
            dnnl::error);

    // Exactly one matrix dim is variable, the group dim never is
    EXPECT_THROW(memory::desc::grouped(
                         {ngroups, total_tokens, K}, dt::f32, total_tokens),
            dnnl::error);
    EXPECT_THROW(memory::desc::grouped({ngroups, DNNL_VARIABLE_DIM_VAL,
                                               DNNL_VARIABLE_DIM_VAL},
                         dt::f32, total_tokens),
            dnnl::error);
    EXPECT_THROW(memory::desc::grouped({DNNL_VARIABLE_DIM_VAL, total_tokens, K},
                         dt::f32, total_tokens),
            dnnl::error);

    // Invalid group count, total_tokens, and fixed dim: zero, negative, runtime
    EXPECT_THROW(memory::desc::grouped(
                         {0, DNNL_VARIABLE_DIM_VAL, K}, dt::f32, total_tokens),
            dnnl::error);
    EXPECT_THROW(memory::desc::grouped(
                         {-1, DNNL_VARIABLE_DIM_VAL, K}, dt::f32, total_tokens),
            dnnl::error);
    EXPECT_THROW(memory::desc::grouped(
                         {ngroups, DNNL_VARIABLE_DIM_VAL, K}, dt::f32, 0),
            dnnl::error);
    EXPECT_THROW(memory::desc::grouped(
                         {ngroups, DNNL_VARIABLE_DIM_VAL, K}, dt::f32, -1),
            dnnl::error);
    EXPECT_THROW(memory::desc::grouped({ngroups, DNNL_VARIABLE_DIM_VAL, K},
                         dt::f32, DNNL_RUNTIME_DIM_VAL),
            dnnl::error);
    EXPECT_THROW(memory::desc::grouped({ngroups, DNNL_VARIABLE_DIM_VAL, 0},
                         dt::f32, total_tokens),
            dnnl::error);
    EXPECT_THROW(memory::desc::grouped(
                         {ngroups, DNNL_VARIABLE_DIM_VAL, DNNL_RUNTIME_DIM_VAL},
                         dt::f32, total_tokens),
            dnnl::error);

    // Strides: G has a zero stride, one matrix dim is contiguous, and the
    // leading dim covers it
    EXPECT_THROW(memory::desc::grouped({ngroups, DNNL_VARIABLE_DIM_VAL, K},
                         dt::f32, total_tokens, {total_tokens * K, K, 1}),
            dnnl::error);
    EXPECT_THROW(memory::desc::grouped({ngroups, DNNL_VARIABLE_DIM_VAL, K},
                         dt::f32, total_tokens, {0, 2 * K, 2}),
            dnnl::error);
    EXPECT_THROW(memory::desc::grouped({ngroups, DNNL_VARIABLE_DIM_VAL, K},
                         dt::f32, total_tokens, {0, K - 1, 1}),
            dnnl::error);
    EXPECT_THROW(memory::desc::grouped({ngroups, DNNL_VARIABLE_DIM_VAL, K},
                         dt::f32, total_tokens, {0, 1, total_tokens - 1}),
            dnnl::error);
    EXPECT_THROW(memory::desc::grouped({ngroups, DNNL_VARIABLE_DIM_VAL, K},
                         dt::f32, total_tokens, {0, K}),
            dnnl::error);

    // Valid offsets type: s32; invalid: f32
    ASSERT_NO_THROW(memory::desc::grouped({ngroups, DNNL_VARIABLE_DIM_VAL, K},
            dt::f32, total_tokens, {}, dt::s32));
    EXPECT_THROW(memory::desc::grouped({ngroups, DNNL_VARIABLE_DIM_VAL, K},
                         dt::f32, total_tokens, {}, dt::f32),
            dnnl::error);
}

TEST(iface_grouped_test_t, TestGroupedMDQueries) {
    const int ngroups = 3;
    const int K = 256;
    const int total_tokens = 9;
    const size_t f32_size = memory::data_type_size(dt::f32);

    {
        // Variable M, the default layout has the variable dim outermost
        const memory::dims dims = {ngroups, DNNL_VARIABLE_DIM_VAL, K};
        memory::desc md;
        ASSERT_NO_THROW(md = memory::desc::grouped(
                                dims, dt::f32, total_tokens, {}, dt::s32));

        ASSERT_EQ(md.get_ndims(), 3);
        ASSERT_EQ(md.get_dims(), dims);
        ASSERT_EQ(md.get_data_type(), dt::f32);
        ASSERT_EQ(md.get_data_type(0), dt::f32);
        ASSERT_EQ(md.get_data_type(1), dt::s32);
        ASSERT_EQ(md.get_format_kind(), memory::format_kind::sparse);
        ASSERT_EQ(md.get_sparse_encoding(), memory::sparse_encoding::grouped);
        ASSERT_EQ(md.get_num_handles(), 2);
        ASSERT_EQ(md.get_nnz(), total_tokens * K);
        ASSERT_EQ(md.get_strides(), memory::dims({0, K, 1}));
        ASSERT_EQ(md.get_size(0), total_tokens * K * f32_size);
        ASSERT_EQ(md.get_size(1), ngroups * memory::data_type_size(dt::s32));
    }

    {
        // Variable K, the default layout is column-major
        const memory::dims dims = {ngroups, K, DNNL_VARIABLE_DIM_VAL};
        memory::desc md;
        ASSERT_NO_THROW(
                md = memory::desc::grouped(dims, dt::f32, total_tokens));
        ASSERT_EQ(md.get_dims(), dims);
        ASSERT_EQ(md.get_nnz(), K * total_tokens);
        ASSERT_EQ(md.get_strides(), memory::dims({0, 1, K}));
        ASSERT_EQ(md.get_size(0), K * total_tokens * f32_size);
    }

    {
        // Variable K contiguous with a padded leading dim, the size covers
        // the padding but nnz does not
        const memory::dim ld = total_tokens + 7;
        memory::desc md;
        ASSERT_NO_THROW(
                md = memory::desc::grouped({ngroups, K, DNNL_VARIABLE_DIM_VAL},
                        dt::f32, total_tokens, {0, ld, 1}));
        ASSERT_EQ(md.get_strides(), memory::dims({0, ld, 1}));
        ASSERT_EQ(md.get_nnz(), K * total_tokens);
        ASSERT_EQ(md.get_size(0), K * ld * f32_size);
    }
}

TEST(iface_grouped_test_t, TestGroupedMDComparison) {
    const int ngroups = 2;
    const int K = 256;
    const int total_tokens = 4;

    memory::desc md = memory::desc::grouped(
            {ngroups, DNNL_VARIABLE_DIM_VAL, K}, dt::f32, total_tokens);

    // Equal descriptors, the default strides are the block ones
    ASSERT_EQ(md,
            memory::desc::grouped({ngroups, DNNL_VARIABLE_DIM_VAL, K}, dt::f32,
                    total_tokens));
    ASSERT_EQ(md,
            memory::desc::grouped({ngroups, DNNL_VARIABLE_DIM_VAL, K}, dt::f32,
                    total_tokens, {0, K, 1}));

    // Different data types, group counts, totals, variable dims, and strides
    ASSERT_NE(md,
            memory::desc::grouped({ngroups, DNNL_VARIABLE_DIM_VAL, K}, dt::f16,
                    total_tokens));
    ASSERT_NE(md,
            memory::desc::grouped(
                    {3, DNNL_VARIABLE_DIM_VAL, K}, dt::f32, total_tokens));
    ASSERT_NE(md,
            memory::desc::grouped({ngroups, DNNL_VARIABLE_DIM_VAL, K}, dt::f32,
                    total_tokens + 1));
    ASSERT_NE(md,
            memory::desc::grouped({ngroups, K, DNNL_VARIABLE_DIM_VAL}, dt::f32,
                    total_tokens));
    ASSERT_NE(md,
            memory::desc::grouped({ngroups, DNNL_VARIABLE_DIM_VAL, K}, dt::f32,
                    total_tokens, {0, 1, total_tokens}));
    ASSERT_NE(md,
            memory::desc::grouped({ngroups, DNNL_VARIABLE_DIM_VAL, K}, dt::f32,
                    total_tokens, {0, K + 16, 1}));
}

HANDLE_EXCEPTIONS_FOR_TEST(iface_grouped_test_t, TestGroupedMemoryCreation) {
    engine eng = get_test_engine();

    SKIP_IF(eng.get_kind() == engine::kind::gpu
                    || DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL,
            "Test requires host-allocated memory.");

    const int ngroups = 3;
    const int K = 256;
    const int total_tokens = 9;

    memory::desc md;
    ASSERT_NO_THROW(
            md = memory::desc::grouped({ngroups, DNNL_VARIABLE_DIM_VAL, K},
                    dt::f32, total_tokens));

    memory mem;
    mem = memory(md, eng);

    // user provided buffers (2 buffers: values and offsets)
    {
        const int total_elements = total_tokens * K;
        std::vector<float> values(total_elements);
        std::vector<int32_t> offsets(ngroups);
        offsets = {1, 4, 9};

        EXPECT_NO_THROW(mem = memory(md, eng, {values.data(), offsets.data()}));
    }

    // same, but skipping one group
    {
        const int total_elements = total_tokens * K;
        std::vector<float> values(total_elements);
        std::vector<int32_t> offsets(ngroups);
        offsets = {1, 1, 9}; // skip group 1

        EXPECT_NO_THROW(mem = memory(md, eng, {values.data(), offsets.data()}));
    }
}

HANDLE_EXCEPTIONS_FOR_TEST(
        iface_grouped_test_t, TestGroupedMemorySetGetDataHandles) {
    engine eng = get_test_engine();

    SKIP_IF(eng.get_kind() == engine::kind::gpu
                    || DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL,
            "Test requires host-allocated memory.");

    const int ngroups = 3;
    const int K = 256;
    const int total_tokens = 9;

    memory::desc md;
    ASSERT_NO_THROW(
            md = memory::desc::grouped({ngroups, DNNL_VARIABLE_DIM_VAL, K},
                    dt::f32, total_tokens));

    memory mem = memory(md, eng);

    {
        const int total_elements = total_tokens * K;
        std::vector<float> values(total_elements);
        std::vector<int32_t> offsets(ngroups);

        ASSERT_NO_THROW(mem.set_data_handle(values.data(), 0));
        ASSERT_NO_THROW(mem.set_data_handle(offsets.data(), 1));

        ASSERT_EQ(mem.get_data_handle(0), values.data());
        ASSERT_EQ(mem.get_data_handle(1), offsets.data());
    }
}

TEST(iface_grouped_test_t, TestGroupedMemoryMapUnmap) {
    engine eng = get_test_engine();

    SKIP_IF(eng.get_kind() == engine::kind::gpu
                    || DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL,
            "Test requires host-allocated memory.");

    const int ngroups = 2;
    const int K = 4;
    const int total_tokens = 3;

    memory::desc md;
    ASSERT_NO_THROW(
            md = memory::desc::grouped({ngroups, DNNL_VARIABLE_DIM_VAL, K},
                    dt::f32, total_tokens, {}, dt::s32));

    const int total_elements = total_tokens * K;
    std::vector<float> values(total_elements);
    for (int i = 0; i < total_elements; i++)
        values[i] = static_cast<float>(i) * 0.5f;

    std::vector<int32_t> offsets = {1, 3};
    ASSERT_EQ(offsets[ngroups - 1], total_tokens);

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
    const int ngroups = 3;
    const int K = 256;
    const int total_tokens = 9;
    const int total_elements = total_tokens * K;

    dnnl_memory_desc_t md = nullptr;
    dnnl_dims_t dims = {ngroups, DNNL_VARIABLE_DIM_VAL, K};

    DNNL_CHECK(dnnl_memory_desc_create_with_grouped_encoding(
            &md, 3, dims, dnnl_f32, total_tokens, nullptr, dnnl_s32));
    ASSERT_NE(md, nullptr);

    // Query all properties
    int ndims = -1;
    DNNL_CHECK(dnnl_memory_desc_query(md, dnnl_query_ndims_s32, &ndims));
    EXPECT_EQ(ndims, 3);

    const dnnl_dims_t *queried_dims = nullptr;
    DNNL_CHECK(dnnl_memory_desc_query(md, dnnl_query_dims, &queried_dims));
    for (int d = 0; d < 3; d++)
        EXPECT_EQ((*queried_dims)[d], dims[d]);

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

    const dnnl_dims_t *strides = nullptr;
    DNNL_CHECK(dnnl_memory_desc_query(md, dnnl_query_strides, &strides));
    EXPECT_EQ((*strides)[0], 0);
    EXPECT_EQ((*strides)[1], K);
    EXPECT_EQ((*strides)[2], 1);

    size_t values_size = dnnl_memory_desc_get_size_v2(md, 0);
    EXPECT_EQ(values_size, total_elements * sizeof(float));

    size_t offsets_size = dnnl_memory_desc_get_size_v2(md, 1);
    EXPECT_EQ(offsets_size, (ngroups) * sizeof(int32_t));

    DNNL_CHECK(dnnl_memory_desc_destroy(md));

    // User strides are kept as is
    const dnnl_dims_t window_strides = {0, 1, total_tokens};
    DNNL_CHECK(dnnl_memory_desc_create_with_grouped_encoding(
            &md, 3, dims, dnnl_f32, total_tokens, window_strides, dnnl_s32));
    DNNL_CHECK(dnnl_memory_desc_query(md, dnnl_query_strides, &strides));
    for (int d = 0; d < 3; d++)
        EXPECT_EQ((*strides)[d], window_strides[d]);
    DNNL_CHECK(dnnl_memory_desc_destroy(md));
}

TEST(c_api_grouped_md, TestGroupedMDInvalidArgs) {
    dnnl_memory_desc_t md = nullptr;
    const int K = 256, total_tokens = 4;
    dnnl_dims_t dims = {2, DNNL_VARIABLE_DIM_VAL, K};

    // Only 3D is supported
    dnnl_dims_t dims_2d = {DNNL_VARIABLE_DIM_VAL, K};
    ASSERT_EQ(dnnl_memory_desc_create_with_grouped_encoding(&md, 2, dims_2d,
                      dnnl_f32, total_tokens, nullptr, dnnl_s32),
            dnnl_unimplemented);
    EXPECT_EQ(md, nullptr);

    // No variable dim
    dnnl_dims_t dims_fixed = {2, total_tokens, K};
    ASSERT_EQ(dnnl_memory_desc_create_with_grouped_encoding(&md, 3, dims_fixed,
                      dnnl_f32, total_tokens, nullptr, dnnl_s32),
            dnnl_invalid_arguments);
    EXPECT_EQ(md, nullptr);

    // Invalid group count and total_tokens
    dnnl_dims_t dims_no_groups = {0, DNNL_VARIABLE_DIM_VAL, K};
    ASSERT_EQ(
            dnnl_memory_desc_create_with_grouped_encoding(&md, 3,
                    dims_no_groups, dnnl_f32, total_tokens, nullptr, dnnl_s32),
            dnnl_invalid_arguments);
    EXPECT_EQ(md, nullptr);
    ASSERT_EQ(dnnl_memory_desc_create_with_grouped_encoding(
                      &md, 3, dims, dnnl_f32, 0, nullptr, dnnl_s32),
            dnnl_invalid_arguments);
    EXPECT_EQ(md, nullptr);

    // Non-zero group stride
    const dnnl_dims_t strides = {total_tokens * K, K, 1};
    ASSERT_EQ(dnnl_memory_desc_create_with_grouped_encoding(
                      &md, 3, dims, dnnl_f32, total_tokens, strides, dnnl_s32),
            dnnl_invalid_arguments);
    EXPECT_EQ(md, nullptr);

    // null pointer md
    dnnl_status_t status = dnnl_memory_desc_create_with_grouped_encoding(
            nullptr, 3, dims, dnnl_f32, total_tokens, nullptr, dnnl_s32);
    ASSERT_EQ(status, dnnl_invalid_arguments);
}

TEST(iface_grouped_test_t, TestGroupedMatmulValidation) {
    engine eng = get_test_engine();
    const int ngroups = 3;
    const int M = 100, K = 256, N = 512;

    auto src_md = memory::desc::grouped(
            {ngroups, DNNL_VARIABLE_DIM_VAL, K}, dt::f32, M);
    auto dst_md = memory::desc::grouped(
            {ngroups, DNNL_VARIABLE_DIM_VAL, N}, dt::f32, M);
    auto wei_md = memory::desc({ngroups, K, N}, dt::f32, tag::abc);
    EXPECT_NO_THROW(matmul::primitive_desc(eng, src_md, wei_md, dst_md));

    // Invalid: 2D weights
    auto wei_2d = memory::desc({K, N}, dt::f32, tag::ab);
    EXPECT_THROW(
            matmul::primitive_desc(eng, src_md, wei_2d, dst_md), dnnl::error);

    // Invalid: mismatched number of groups
    auto dst_4groups
            = memory::desc::grouped({4, DNNL_VARIABLE_DIM_VAL, N}, dt::f32, M);
    auto wei_4groups = memory::desc({4, K, N}, dt::f32, tag::abc);
    EXPECT_THROW(matmul::primitive_desc(eng, src_md, wei_md, dst_4groups),
            dnnl::error);
    EXPECT_THROW(matmul::primitive_desc(eng, src_md, wei_4groups, dst_md),
            dnnl::error);

    // Invalid: mismatched totals and K
    auto dst_other_total = memory::desc::grouped(
            {ngroups, DNNL_VARIABLE_DIM_VAL, N}, dt::f32, M + 1);
    EXPECT_THROW(matmul::primitive_desc(eng, src_md, wei_md, dst_other_total),
            dnnl::error);
    auto wei_other_k = memory::desc({ngroups, K + 1, N}, dt::f32, tag::abc);
    EXPECT_THROW(matmul::primitive_desc(eng, src_md, wei_other_k, dst_md),
            dnnl::error);
}

TEST(iface_grouped_test_t, TestGroupedMatmulPatterns) {
    engine eng = get_test_engine();

    const int ngroups = 2;
    const int M = 4, K = 8, N = 6;

    // Shapes to mix and match:

    // Grouped, variable M, K, or N
    auto src_var_m = memory::desc::grouped(
            {ngroups, DNNL_VARIABLE_DIM_VAL, K}, dt::f32, M);
    auto dst_var_m = memory::desc::grouped(
            {ngroups, DNNL_VARIABLE_DIM_VAL, N}, dt::f32, M);
    auto src_var_k = memory::desc::grouped(
            {ngroups, M, DNNL_VARIABLE_DIM_VAL}, dt::f32, K);
    auto wei_var_k = memory::desc::grouped(
            {ngroups, DNNL_VARIABLE_DIM_VAL, N}, dt::f32, K);
    auto wei_var_n = memory::desc::grouped(
            {ngroups, K, DNNL_VARIABLE_DIM_VAL}, dt::f32, N);

    // Dense 3D shapes
    auto src_dense_3d
            = memory::desc({ngroups, M / ngroups, K}, dt::f32, tag::abc);
    auto wei_dense_3d = memory::desc({ngroups, K, N}, dt::f32, tag::abc);
    auto dst_dense_3d = memory::desc({ngroups, M, N}, dt::f32, tag::abc);

    // Valid cases:

    // grouped(varM) * dense 3D -> grouped(varM) (MoE forward)
    EXPECT_NO_THROW(
            matmul::primitive_desc(eng, src_var_m, wei_dense_3d, dst_var_m));

    // grouped(varK) * grouped(varK) -> dense 3D (MoE backward)
    EXPECT_NO_THROW(
            matmul::primitive_desc(eng, src_var_k, wei_var_k, dst_dense_3d));

    // Column-major src and dst (varM), K-major src and wei (varK)
    EXPECT_NO_THROW(matmul::primitive_desc(eng,
            memory::desc::grouped(
                    {ngroups, DNNL_VARIABLE_DIM_VAL, K}, dt::f32, M, {0, 1, M}),
            wei_dense_3d,
            memory::desc::grouped({ngroups, DNNL_VARIABLE_DIM_VAL, N}, dt::f32,
                    M, {0, 1, M})));
    EXPECT_NO_THROW(matmul::primitive_desc(eng,
            memory::desc::grouped(
                    {ngroups, M, DNNL_VARIABLE_DIM_VAL}, dt::f32, K, {0, K, 1}),
            memory::desc::grouped(
                    {ngroups, DNNL_VARIABLE_DIM_VAL, N}, dt::f32, K, {0, 1, K}),
            dst_dense_3d));

    // Unsupported cases:

    // grouped(varM) * dense 3D -> dense 3D
    EXPECT_THROW(
            matmul::primitive_desc(eng, src_var_m, wei_dense_3d, dst_dense_3d),
            dnnl::error);

    // dense 3D * dense 3D -> grouped
    EXPECT_THROW(
            matmul::primitive_desc(eng, src_dense_3d, wei_dense_3d, dst_var_m),
            dnnl::error);

    // grouped(varM) * grouped(varK) -> dense 3D
    EXPECT_THROW(
            matmul::primitive_desc(eng, src_var_m, wei_var_k, dst_dense_3d),
            dnnl::error);

    // grouped(varK) * grouped(varN) -> dense 3D
    EXPECT_THROW(
            matmul::primitive_desc(eng, src_var_k, wei_var_n, dst_dense_3d),
            dnnl::error);
}

TEST(iface_grouped_test_t, TestGroupedMatmulScales) {
    engine eng = get_test_engine();
    SKIP_IF(unsupported_data_type(dt::f8_e4m3, dt::bf16),
            "Engine does not support f8_e4m3 or bf16.");

    const int ngroups = 2;
    const int M = 32, K = 64, N = 16;

    // Masks are over src {G, M, K} and wei {G, K, N} in both patterns
    const int src_rowwise = (1 << 0) | (1 << 1);
    const int src_k_blocked = src_rowwise | (1 << 2);
    const int wei_colwise = (1 << 0) | (1 << 2);
    const int wei_k_blocked = wei_colwise | (1 << 1);

    for (bool var_k : {false, true}) {
        // K-major src and wei for variable K
        auto src_md = var_k
                ? memory::desc::grouped({ngroups, M, DNNL_VARIABLE_DIM_VAL},
                          dt::f8_e4m3, K, {0, K, 1})
                : memory::desc::grouped(
                          {ngroups, DNNL_VARIABLE_DIM_VAL, K}, dt::f8_e4m3, M);
        auto wei_md = var_k
                ? memory::desc::grouped({ngroups, DNNL_VARIABLE_DIM_VAL, N},
                          dt::f8_e4m3, K, {0, 1, K})
                : memory::desc({ngroups, K, N}, dt::f8_e4m3, tag::acb);
        auto dst_md = var_k
                ? memory::desc({ngroups, M, N}, dt::bf16, tag::abc)
                : memory::desc::grouped(
                          {ngroups, DNNL_VARIABLE_DIM_VAL, N}, dt::bf16, M);

        // Row-wise src and column-wise wei scales (FP8 row-wise)
        {
            primitive_attr attr;
            attr.set_scales_mask(DNNL_ARG_SRC, src_rowwise);
            attr.set_scales_mask(DNNL_ARG_WEIGHTS, wei_colwise);
            EXPECT_NO_THROW(
                    matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr));
        }

        // K-blocked src and wei scales (MXFP8)
        {
            primitive_attr attr;
            attr.set_scales(DNNL_ARG_SRC, src_k_blocked, {1, 32}, dt::e8m0);
            attr.set_scales(DNNL_ARG_WEIGHTS, wei_k_blocked, {32, 1}, dt::e8m0);
            EXPECT_NO_THROW(
                    matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr));
        }

        // A mask covering the variable dim must cover G too
        {
            primitive_attr attr;
            attr.set_scales_mask(
                    DNNL_ARG_SRC, var_k ? (1 << 1) | (1 << 2) : (1 << 1));
            EXPECT_THROW(
                    matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr),
                    dnnl::error);
        }

        // Groups go with K-blocked masks only
        {
            primitive_attr attr;
            attr.set_scales(DNNL_ARG_SRC, src_rowwise, {1, 32}, dt::e8m0);
            EXPECT_THROW(
                    matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr),
                    dnnl::error);
        }
    }
}

TEST(iface_grouped_test_t, TestGroupedMatmulBias) {
    engine eng = get_test_engine();

    const int ngroups = 3;
    const int M = 12, K = 8, N = 4;

    auto src_md = memory::desc::grouped(
            {ngroups, DNNL_VARIABLE_DIM_VAL, K}, dt::f32, M);
    auto wei_md = memory::desc({ngroups, K, N}, dt::f32, tag::abc);
    auto dst_md = memory::desc::grouped(
            {ngroups, DNNL_VARIABLE_DIM_VAL, N}, dt::f32, M);

    // Bias is [G, 1, N]
    auto bia_md = memory::desc({ngroups, 1, N}, dt::f32, tag::abc);
    EXPECT_NO_THROW(
            matmul::primitive_desc(eng, src_md, wei_md, bia_md, dst_md));

    auto bia_2d = memory::desc({ngroups, N}, dt::f32, tag::ab);
    EXPECT_THROW(matmul::primitive_desc(eng, src_md, wei_md, bia_2d, dst_md),
            dnnl::error);
    auto bia_shared = memory::desc({1, 1, N}, dt::f32, tag::abc);
    EXPECT_THROW(
            matmul::primitive_desc(eng, src_md, wei_md, bia_shared, dst_md),
            dnnl::error);

    // Variable K takes no bias
    auto src_var_k = memory::desc::grouped(
            {ngroups, M, DNNL_VARIABLE_DIM_VAL}, dt::f32, K);
    auto wei_var_k = memory::desc::grouped(
            {ngroups, DNNL_VARIABLE_DIM_VAL, N}, dt::f32, K);
    auto dst_dense = memory::desc({ngroups, M, N}, dt::f32, tag::abc);
    EXPECT_THROW(matmul::primitive_desc(
                         eng, src_var_k, wei_var_k, bia_md, dst_dense),
            dnnl::error);
}

HANDLE_EXCEPTIONS_FOR_TEST(iface_grouped_test_t, TestMaxGroupMHint) {
    engine eng = get_test_engine();
    SKIP_IF(eng.get_kind() == engine::kind::gpu
                    || DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL,
            "Test requires host-allocated memory.");

    auto hint_md = memory::desc::host_scalar(dt::s32);
    memory hint_mem(hint_md, int32_t(3));

    // Grouped matmul: hint is accepted
    {
        const int ngroups = 2;
        const int M = 6, K = 4, N = 4;

        auto src_md = memory::desc::grouped(
                {ngroups, DNNL_VARIABLE_DIM_VAL, K}, dt::f32, M);
        auto wei_md = memory::desc({ngroups, K, N}, dt::f32, tag::abc);
        auto dst_md = memory::desc::grouped(
                {ngroups, DNNL_VARIABLE_DIM_VAL, N}, dt::f32, M);

        matmul::primitive_desc pd(eng, src_md, wei_md, dst_md);
        matmul prim(pd);
        stream strm(eng);

        std::vector<float> src_data(M * K, 1.f);
        std::vector<float> wei_data(ngroups * K * N, 1.f);
        std::vector<float> dst_data(M * N, 0.f);
        std::vector<int32_t> offsets = {3, 6};

        memory src_mem(src_md, eng, {src_data.data(), offsets.data()});
        memory wei_mem(wei_md, eng, wei_data.data());
        memory dst_mem(dst_md, eng, {dst_data.data(), offsets.data()});

        EXPECT_NO_THROW(prim.execute(strm,
                {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, wei_mem},
                        {DNNL_ARG_DST, dst_mem},
                        {DNNL_ARG_HINT_MAX_GROUP_SIZE, hint_mem}}));
        strm.wait();
    }

    // Non-grouped matmul
    {
        const int M = 4, K = 8, N = 6;
        auto src_md = memory::desc({M, K}, dt::f32, tag::ab);
        auto wei_md = memory::desc({K, N}, dt::f32, tag::ab);
        auto dst_md = memory::desc({M, N}, dt::f32, tag::ab);

        matmul::primitive_desc pd(eng, src_md, wei_md, dst_md);
        matmul prim(pd);
        stream strm(eng);

        std::vector<float> src_data(M * K, 1.f);
        std::vector<float> wei_data(K * N, 1.f);
        std::vector<float> dst_data(M * N, 0.f);
        memory src_mem(src_md, eng, src_data.data());
        memory wei_mem(wei_md, eng, wei_data.data());
        memory dst_mem(dst_md, eng, dst_data.data());

        // Hint is unused and ignored
        EXPECT_NO_THROW(prim.execute(strm,
                {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, wei_mem},
                        {DNNL_ARG_DST, dst_mem},
                        {DNNL_ARG_HINT_MAX_GROUP_SIZE, hint_mem}}));
        strm.wait();

        // Hint is passed as input instead of wei, so should be rejected
        EXPECT_THROW(prim.execute(strm,
                             {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem},
                                     {DNNL_ARG_HINT_MAX_GROUP_SIZE, hint_mem}}),
                dnnl::error);
    }
}

HANDLE_EXCEPTIONS_FOR_TEST(iface_grouped_test_t, TestGroupedMatmulKMajor) {
    engine eng = get_test_engine();
    SKIP_IF(eng.get_kind() == engine::kind::gpu
                    || DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL,
            "Test requires host-allocated memory.");

    // Variable K with K-major src [M, total_K] and wei [total_K, N] given in
    // PyTorch's layout, i.e., src row-major and wei column-major
    const int ngroups = 2;
    const int M = 2, N = 3, total_K = 5;
    std::vector<int32_t> offsets = {2, 5};

    auto src_md = memory::desc::grouped({ngroups, M, DNNL_VARIABLE_DIM_VAL},
            dt::f32, total_K, {0, total_K, 1});
    auto wei_md = memory::desc::grouped({ngroups, DNNL_VARIABLE_DIM_VAL, N},
            dt::f32, total_K, {0, 1, total_K});
    auto dst_md = memory::desc({ngroups, M, N}, dt::f32, tag::abc);

    std::vector<float> src_data(M * total_K), wei_data(total_K * N);
    for (int i = 0; i < M * total_K; i++)
        src_data[i] = static_cast<float>(i % 7 - 3);
    for (int i = 0; i < total_K * N; i++)
        wei_data[i] = static_cast<float>(i % 5 - 2);
    std::vector<float> dst_data(ngroups * M * N, -1.f);

    memory src_mem(src_md, eng, {src_data.data(), offsets.data()});
    memory wei_mem(wei_md, eng, {wei_data.data(), offsets.data()});
    memory dst_mem(dst_md, eng, dst_data.data());

    matmul::primitive_desc pd(eng, src_md, wei_md, dst_md);
    stream strm(eng);
    matmul(pd).execute(strm,
            {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, wei_mem},
                    {DNNL_ARG_DST, dst_mem}});
    strm.wait();

    for_(int g = 0; g < ngroups; g++)
    for_(int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
        float ref = 0.f;
        for (int k = g == 0 ? 0 : offsets[g - 1]; k < offsets[g]; k++)
            ref += src_data[m * total_K + k] * wei_data[n * total_K + k];
        EXPECT_EQ(dst_data[(g * M + m) * N + n], ref);
    }
}

HANDLE_EXCEPTIONS_FOR_TEST(iface_grouped_test_t, TestBinaryPostOpPDCreation) {
    engine eng = get_test_engine();

    SKIP_IF(eng.get_kind() == engine::kind::gpu
                    || DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL,
            "Test requires host-allocated memory.");

    const int ngroups = 2;
    const int M = 6, K = 4, N = 4;
    auto src_md = memory::desc::grouped(
            {ngroups, DNNL_VARIABLE_DIM_VAL, K}, dt::f32, M);
    auto wei_md = memory::desc({ngroups, K, N}, dt::f32, tag::abc);
    auto dst_md = memory::desc::grouped(
            {ngroups, DNNL_VARIABLE_DIM_VAL, N}, dt::f32, M);

    // A grouped operand matches dst's groups and has the default layout
    for (const auto &bin_md : {
                 memory::desc::grouped(
                         {ngroups, DNNL_VARIABLE_DIM_VAL, N}, dt::f32, M),
                 memory::desc::grouped(
                         {ngroups, DNNL_VARIABLE_DIM_VAL, 1}, dt::f32, M),
                 memory::desc::grouped(
                         {3, DNNL_VARIABLE_DIM_VAL, N}, dt::f32, M),
                 memory::desc::grouped(
                         {ngroups, DNNL_VARIABLE_DIM_VAL, N}, dt::f32, M + 1),
                 memory::desc::grouped({ngroups, DNNL_VARIABLE_DIM_VAL, N},
                         dt::f32, M, {0, 1, M}),
         }) {
        const bool ok = bin_md.get_dims()[0] == ngroups
                && bin_md.get_nnz() == M * bin_md.get_dims()[2]
                && bin_md.get_strides()[2] == 1;
        post_ops po;
        po.append_binary(algorithm::binary_mul, bin_md);
        primitive_attr attr;
        attr.set_post_ops(po);
        if (ok)
            EXPECT_NO_THROW(
                    matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr));
        else
            EXPECT_THROW(
                    matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr),
                    dnnl::error);
    }
}

TEST(iface_grouped_test_t, TestPostOps) {
    engine eng = get_test_engine();

    SKIP_IF(eng.get_kind() == engine::kind::gpu
                    || DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL,
            "Test requires host-allocated memory.");

    const int ngroups = 2;
    const int M = 6, K = 4, N = 4;
    auto src_md = memory::desc::grouped(
            {ngroups, DNNL_VARIABLE_DIM_VAL, K}, dt::f32, M);
    auto wei_md = memory::desc({ngroups, K, N}, dt::f32, tag::abc);
    auto dst_md = memory::desc::grouped(
            {ngroups, DNNL_VARIABLE_DIM_VAL, N}, dt::f32, M);
    auto bin_md = memory::desc::grouped(
            {ngroups, DNNL_VARIABLE_DIM_VAL, N}, dt::f32, M);

    // Any eltwise algorithm is accepted as an activation post-op
    for (auto alg : {algorithm::eltwise_relu, algorithm::eltwise_tanh,
                 algorithm::eltwise_elu, algorithm::eltwise_square,
                 algorithm::eltwise_abs, algorithm::eltwise_sqrt,
                 algorithm::eltwise_linear, algorithm::eltwise_soft_relu,
                 algorithm::eltwise_logistic, algorithm::eltwise_exp,
                 algorithm::eltwise_gelu_tanh, algorithm::eltwise_gelu_erf,
                 algorithm::eltwise_log, algorithm::eltwise_clip,
                 algorithm::eltwise_clip_v2, algorithm::eltwise_pow,
                 algorithm::eltwise_round, algorithm::eltwise_mish,
                 algorithm::eltwise_hardswish,
                 algorithm::eltwise_hardsigmoid}) {
        post_ops po;
        po.append_eltwise(alg, 0.f, 1.f);
        primitive_attr attr;
        attr.set_post_ops(po);
        EXPECT_NO_THROW(
                matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr));
    }

    // eltwise_swish + binary_mul is accepted
    {
        post_ops po;
        po.append_eltwise(algorithm::eltwise_swish, 1.f, 0.f);
        po.append_binary(algorithm::binary_mul, bin_md);
        primitive_attr attr;
        attr.set_post_ops(po);
        EXPECT_NO_THROW(
                matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr));
    }

    // binary_add is not supported for grouped matmul
    {
        post_ops po;
        po.append_binary(algorithm::binary_add, bin_md);
        primitive_attr attr;
        attr.set_post_ops(po);
        EXPECT_THROW(matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr),
                dnnl::error);
    }

    // sum is not supported
    {
        post_ops po;
        po.append_sum(1.0f);
        primitive_attr attr;
        attr.set_post_ops(po);
        EXPECT_THROW(matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr),
                dnnl::error);
    }
}

HANDLE_EXCEPTIONS_FOR_TEST(
        iface_grouped_test_t, TestBinaryPostOpRejectsMismatchedOffsets) {
    engine eng = get_test_engine();

    SKIP_IF(eng.get_kind() == engine::kind::gpu
                    || DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL,
            "Test requires host-allocated memory.");

    const int ngroups = 2;
    const int total_m = 6;
    const int K = 4;
    const int N = 4;

    auto src_md = memory::desc::grouped(
            {ngroups, DNNL_VARIABLE_DIM_VAL, K}, dt::f32, total_m);
    auto wei_md = memory::desc({ngroups, K, N}, dt::f32, tag::abc);
    auto dst_md = memory::desc::grouped(
            {ngroups, DNNL_VARIABLE_DIM_VAL, N}, dt::f32, total_m);
    auto bin_md = memory::desc::grouped(
            {ngroups, DNNL_VARIABLE_DIM_VAL, N}, dt::f32, total_m);

    post_ops po;
    po.append_binary(algorithm::binary_mul, bin_md);
    primitive_attr attr;
    attr.set_post_ops(po);

    matmul::primitive_desc pd(eng, src_md, wei_md, dst_md, attr);
    matmul prim(pd);
    stream strm(eng);

    std::vector<float> src_data(total_m * K, 1.f);
    std::vector<float> wei_data(ngroups * K * N, 1.f);
    std::vector<float> dst_data(total_m * N, 0.f);
    std::vector<float> bin_data(total_m * N, 1.f);
    std::vector<int32_t> src_offsets = {3, 6};
    std::vector<int32_t> dst_offsets = {3, 6};
    std::vector<int32_t> bin_offsets = {2, 6};

    memory src_mem(src_md, eng, {src_data.data(), src_offsets.data()});
    memory wei_mem(wei_md, eng, wei_data.data());
    memory dst_mem(dst_md, eng, {dst_data.data(), dst_offsets.data()});
    memory bin_mem(bin_md, eng, {bin_data.data(), bin_offsets.data()});

    EXPECT_THROW(
            prim.execute(strm,
                    {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, wei_mem},
                            {DNNL_ARG_DST, dst_mem},
                            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                    bin_mem}}),
            dnnl::error);
}

TEST(iface_grouped_test_t, TestBinaryPostOpDenseShapes) {
    engine eng = get_test_engine();

    const int ngroups = 4;
    const int total_m = 32;
    const int K = 64;
    const int N = 32;

    auto src_md = memory::desc::grouped(
            {ngroups, DNNL_VARIABLE_DIM_VAL, K}, dt::f32, total_m);
    auto wei_md = memory::desc({ngroups, K, N}, dt::f32, tag::abc);
    auto dst_md = memory::desc::grouped(
            {ngroups, DNNL_VARIABLE_DIM_VAL, N}, dt::f32, total_m);

    // A dense operand is supported if it does not vary along the variable dim:
    // per tensor, per column, per group, per group and column
    for (const auto &bin_dims : {memory::dims {1, 1, 1}, memory::dims {1, 1, N},
                 memory::dims {ngroups, 1, 1}, memory::dims {ngroups, 1, N}}) {
        post_ops po;
        po.append_binary(algorithm::binary_mul,
                memory::desc(bin_dims, dt::f32, tag::abc));
        primitive_attr attr;
        attr.set_post_ops(po);
        EXPECT_NO_THROW(
                matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr));
    }

    // Varying along the variable dim takes a grouped operand, and 2D operands
    // don't match the dst rank
    for (const auto &bin_md : {memory::desc({1, total_m, 1}, dt::f32, tag::abc),
                 memory::desc({ngroups, 1}, dt::f32, tag::ab)}) {
        post_ops po;
        po.append_binary(algorithm::binary_mul, bin_md);
        primitive_attr attr;
        attr.set_post_ops(po);
        EXPECT_THROW(matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr),
                dnnl::error);
    }
}

HANDLE_EXCEPTIONS_FOR_TEST(iface_grouped_test_t, TestBiasAndPostOpsExecution) {
    engine eng = get_test_engine();
    SKIP_IF(eng.get_kind() == engine::kind::gpu
                    || DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL,
            "Test requires host-allocated memory.");

    // Variable M with bias [G, 1, N], a per-group [G, 1, 1] and a per-row
    // grouped {G, *, 1} binary_mul
    const int ngroups = 2;
    const int M = 3, K = 2, N = 2;
    std::vector<int32_t> offsets = {1, 3};

    auto src_md = memory::desc::grouped(
            {ngroups, DNNL_VARIABLE_DIM_VAL, K}, dt::f32, M);
    auto wei_md = memory::desc({ngroups, K, N}, dt::f32, tag::abc);
    auto bia_md = memory::desc({ngroups, 1, N}, dt::f32, tag::abc);
    auto dst_md = memory::desc::grouped(
            {ngroups, DNNL_VARIABLE_DIM_VAL, N}, dt::f32, M);
    auto per_group_md = memory::desc({ngroups, 1, 1}, dt::f32, tag::abc);
    auto per_row_md = memory::desc::grouped(
            {ngroups, DNNL_VARIABLE_DIM_VAL, 1}, dt::f32, M);

    post_ops po;
    po.append_binary(algorithm::binary_mul, per_group_md);
    po.append_binary(algorithm::binary_mul, per_row_md);
    primitive_attr attr;
    attr.set_post_ops(po);

    std::vector<float> src_data = {1, 2, 3, 4, 5, 6};
    std::vector<float> wei_data = {1, 0, 0, 1, 2, 1, 1, 2};
    std::vector<float> bia_data = {10, 20, 30, 40};
    std::vector<float> per_group_data = {2, 3};
    std::vector<float> per_row_data = {1, -1, 2};
    std::vector<float> dst_data(M * N, 0.f);

    memory src_mem(src_md, eng, {src_data.data(), offsets.data()});
    memory wei_mem(wei_md, eng, wei_data.data());
    memory bia_mem(bia_md, eng, bia_data.data());
    memory dst_mem(dst_md, eng, {dst_data.data(), offsets.data()});
    memory per_group_mem(per_group_md, eng, per_group_data.data());
    memory per_row_mem(per_row_md, eng, {per_row_data.data(), offsets.data()});

    matmul::primitive_desc pd(eng, src_md, wei_md, bia_md, dst_md, attr);
    stream strm(eng);
    matmul(pd).execute(strm,
            {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, wei_mem},
                    {DNNL_ARG_BIAS, bia_mem}, {DNNL_ARG_DST, dst_mem},
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                            per_group_mem},
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                            per_row_mem}});
    strm.wait();

    for_(int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
        const int g = m < offsets[0] ? 0 : 1;
        float ref = bia_data[g * N + n];
        for (int k = 0; k < K; k++)
            ref += src_data[m * K + k] * wei_data[(g * K + k) * N + n];
        ref *= per_group_data[g] * per_row_data[m];
        EXPECT_EQ(dst_data[m * N + n], ref);
    }
}

} // namespace dnnl

#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY
