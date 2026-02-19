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

#include "src/xpu/ze/usm_utils.hpp"

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl_ze.hpp"

#include <cstdint>
#include <vector>

using namespace dnnl::impl::xpu;

namespace dnnl {

namespace {
void fill_data(void *usm_ptr, memory::dim n, const engine &eng) {
    ze_memory_allocation_properties_t memory_allocation_properties = {};
    memory_allocation_properties.stype
            = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
    TEST_ZE_CHECK(zeMemGetAllocProperties(ze_interop::get_context(eng), usm_ptr,
            &memory_allocation_properties, nullptr));

    auto alloc_kind = memory_allocation_properties.type;
    if (alloc_kind == ZE_MEMORY_TYPE_HOST
            || alloc_kind == ZE_MEMORY_TYPE_SHARED) {
        for (int i = 0; i < n; i++)
            ((float *)usm_ptr)[i] = float(i);
    } else {
        std::vector<float> host_ptr(n);
        for (int i = 0; i < n; i++)
            host_ptr[i] = float(i);

        auto s = stream(eng);
        ze::memcpy(s.get(), usm_ptr, host_ptr.data(), n * sizeof(float));
        s.wait();
    }
}

using usm_unique_ptr_t = std::unique_ptr<void, std::function<void(void *)>>;
usm_unique_ptr_t allocate_usm(size_t size, const engine &eng) {
    return usm_unique_ptr_t(ze::malloc_shared(eng.get(), size),
            [&](void *ptr) { ze::free(eng.get(), ptr); });
}

} // namespace

class ze_memory_test_t : public ::testing::Test {};

HANDLE_EXCEPTIONS_FOR_TEST(ze_memory_test_t, Constructor) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "Engine not found.");

    engine eng(engine::kind::gpu, 0);
    memory::dim n = 100;
    memory::desc mem_d({n}, memory::data_type::f32, memory::format_tag::x);

    auto ptr = allocate_usm(sizeof(float) * n, eng);

    auto mem = ze_interop::make_memory(mem_d, eng, ptr.get());

    ASSERT_EQ(ptr.get(), mem.get_data_handle());

    for (int i = 0; i < n; i++) {
        ((float *)ptr.get())[i] = float(i);
    }

    float *ptr_f32 = (float *)mem.get_data_handle();
    GTEST_EXPECT_NE(ptr_f32, nullptr);
    for (int i = 0; i < n; i++) {
        ASSERT_EQ(ptr_f32[i], float(i));
    }
}

HANDLE_EXCEPTIONS_FOR_TEST(ze_memory_test_t, ConstructorNone) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "Engine not found.");

    engine eng(engine::kind::gpu, 0);
    memory::desc mem_d({0}, memory::data_type::f32, memory::format_tag::x);

    auto mem = ze_interop::make_memory(mem_d, eng, DNNL_MEMORY_NONE);

    ASSERT_EQ(nullptr, mem.get_data_handle());
}

HANDLE_EXCEPTIONS_FOR_TEST(ze_memory_test_t, ConstructorAllocate) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "Engine not found.");

    engine eng(engine::kind::gpu, 0);
    memory::dim n = 100;
    memory::desc mem_d({n}, memory::data_type::f32, memory::format_tag::x);

    auto mem = ze_interop::make_memory(mem_d, eng, DNNL_MEMORY_ALLOCATE);

    void *ptr = mem.get_data_handle();
    GTEST_EXPECT_NE(ptr, nullptr);
    fill_data(ptr, n, eng);

    float *mapped_ptr = mem.map_data<float>();
    GTEST_EXPECT_NE(mapped_ptr, nullptr);
    for (int i = 0; i < n; i++) {
        ASSERT_EQ(mapped_ptr[i], float(i));
    }
    mem.unmap_data(mapped_ptr);
}

HANDLE_EXCEPTIONS_FOR_TEST(ze_memory_test_t, DefaultConstructor) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "Engine not found.");

    engine eng(engine::kind::gpu, 0);
    memory::dim n = 100;
    memory::desc mem_d({n}, memory::data_type::f32, memory::format_tag::x);

    auto mem = ze_interop::make_memory(mem_d, eng);

    void *ptr = mem.get_data_handle();
    GTEST_EXPECT_NE(ptr, nullptr);
    fill_data(ptr, n, eng);

    float *mapped_ptr = mem.map_data<float>();
    GTEST_EXPECT_NE(mapped_ptr, nullptr);
    for (int i = 0; i < n; i++) {
        ASSERT_EQ(mapped_ptr[i], float(i));
    }
    mem.unmap_data(mapped_ptr);
}

HANDLE_EXCEPTIONS_FOR_TEST(ze_memory_test_t, HostScalarConstructor) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "Engine not found.");

    engine eng(engine::kind::gpu, 0);

    auto scalar_md = memory::desc::host_scalar(memory::data_type::f32);

    EXPECT_THROW(
            memory mem = ze_interop::make_memory(scalar_md, eng), dnnl::error);
}

template <typename AllocFuncT, typename FreeFuncT>
void test_usm_map_unmap(
        const AllocFuncT &alloc_func, const FreeFuncT &free_func) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "Engine not found.");

    engine eng(engine::kind::gpu, 0);
    memory::dim n = 100;
    memory::desc mem_d({n}, memory::data_type::f32, memory::format_tag::x);

    auto *ptr = alloc_func(eng.get(), mem_d.get_size());
    ASSERT_NE(ptr, nullptr);

    auto mem = ze_interop::make_memory(mem_d, eng, ptr);

    {
        float *mapped_ptr = mem.template map_data<float>();
        GTEST_EXPECT_NE(mapped_ptr, nullptr);
        fill_data(mapped_ptr, n, eng);
        mem.unmap_data(mapped_ptr);
    }

    {
        float *mapped_ptr = mem.template map_data<float>();
        GTEST_EXPECT_NE(mapped_ptr, nullptr);
        for (int i = 0; i < n; i++) {
            ASSERT_EQ(mapped_ptr[i], float(i));
        }
        mem.unmap_data(mapped_ptr);
    }
    free_func(eng.get(), ptr);
}

/// This test checks if passing system allocated memory(e.g. using malloc)
/// will throw if passed into the make_memory, unless shared system USM is supported
TEST(ze_memory_usm_test, ErrorMakeMemoryUsingSystemMemory) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "Engine not found.");

    engine eng(engine::kind::gpu, 0);
    memory::dim n = 100;
    memory::desc mem_d({n}, memory::data_type::f32, memory::format_tag::x);

    std::vector<float> system_buf(n);
    bool system_memory_supported = false;

    ze_driver_handle_t ze_driver = nullptr;
    ze_device_handle_t ze_dev = nullptr;
    ze_context_handle_t ze_ctx = nullptr;
    find_ze_device(&ze_driver, &ze_dev, &ze_ctx);

    ze_device_memory_access_properties_t device_memory_access_properties = {};
    device_memory_access_properties.stype
            = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES;

    TEST_ZE_CHECK(zeDeviceGetMemoryAccessProperties(
            ze_dev, &device_memory_access_properties));
    system_memory_supported
            = device_memory_access_properties.sharedSystemAllocCapabilities;

    if (system_memory_supported) {
        EXPECT_NO_THROW(memory mem
                = ze_interop::make_memory(mem_d, eng, system_buf.data()));
    } else {
        EXPECT_THROW(memory mem
                = ze_interop::make_memory(mem_d, eng, system_buf.data()),
                dnnl::error);
    }

    if (ze_ctx) { zeContextDestroy(ze_ctx); }
}

HANDLE_EXCEPTIONS_FOR_TEST(ze_memory_test_t, DeviceMapUnmap) {
    test_usm_map_unmap(ze::malloc_device, ze::free);
}

HANDLE_EXCEPTIONS_FOR_TEST(ze_memory_test_t, SharedMapUnmap) {
    test_usm_map_unmap(ze::malloc_shared, ze::free);
}

HANDLE_EXCEPTIONS_FOR_TEST(ze_memory_test_t, TestSparseMemoryCreation) {
    engine eng(engine::kind::gpu, 0);
    const int nnz = 12;
    memory::desc md;

    ASSERT_NO_THROW(md = memory::desc::coo({64, 128}, memory::data_type::f32,
                            nnz, memory::data_type::s32));

    memory mem;
    // Default memory constructor.
    EXPECT_NO_THROW(mem = memory(md, eng));
    // Default interop API to create a memory object.
    EXPECT_NO_THROW(mem = ze_interop::make_memory(md, eng));
    // User provided buffers.
    auto ze_values = allocate_usm(md.get_size(0), eng);
    ASSERT_NE(ze_values, nullptr);

    auto ze_row_indices = allocate_usm(md.get_size(1), eng);
    ASSERT_NE(ze_row_indices, nullptr);

    auto ze_col_indices = allocate_usm(md.get_size(2), eng);
    ASSERT_NE(ze_col_indices, nullptr);

    EXPECT_NO_THROW(mem = ze_interop::make_memory(md, eng,
                            {ze_values.get(), ze_row_indices.get(),
                                    ze_col_indices.get()}));

    ASSERT_NO_THROW(mem.set_data_handle(nullptr, 0));
    ASSERT_NO_THROW(mem.set_data_handle(nullptr, 1));
    ASSERT_NO_THROW(mem.set_data_handle(nullptr, 2));

    ASSERT_EQ(mem.get_data_handle(0), nullptr);
    ASSERT_EQ(mem.get_data_handle(1), nullptr);
    ASSERT_EQ(mem.get_data_handle(2), nullptr);
}

HANDLE_EXCEPTIONS_FOR_TEST(ze_memory_test_t, TestSparseMemoryMapUnmap) {
    engine eng(engine::kind::gpu, 0);

    const int nnz = 2;
    memory::desc md;

    // COO.
    ASSERT_NO_THROW(md = memory::desc::coo({2, 2}, memory::data_type::f32, nnz,
                            memory::data_type::s32));

    // User provided buffers.
    std::vector<float> coo_values = {1.5, 2.5};
    std::vector<int> row_indices = {0, 1};
    std::vector<int> col_indices = {0, 1};

    // User provided buffers.
    auto ze_values = allocate_usm(md.get_size(0), eng);
    ASSERT_NE(ze_values, nullptr);

    auto ze_row_indices = allocate_usm(md.get_size(1), eng);
    ASSERT_NE(ze_row_indices, nullptr);

    auto ze_col_indices = allocate_usm(md.get_size(2), eng);
    ASSERT_NE(ze_col_indices, nullptr);

    auto s = stream(eng);
    ze::memcpy(s.get(), ze_values.get(), coo_values.data(), md.get_size(0));
    ze::memcpy(
            s.get(), ze_row_indices.get(), row_indices.data(), md.get_size(1));
    ze::memcpy(
            s.get(), ze_col_indices.get(), col_indices.data(), md.get_size(2));
    s.wait();

    memory coo_mem;
    EXPECT_NO_THROW(coo_mem = ze_interop::make_memory(md, eng,
                            {ze_values.get(), ze_row_indices.get(),
                                    ze_col_indices.get()}));

    float *mapped_coo_values = nullptr;
    int *mapped_row_indices = nullptr;
    int *mapped_col_indices = nullptr;

    ASSERT_NO_THROW(mapped_coo_values = coo_mem.map_data<float>(0));
    ASSERT_NO_THROW(mapped_row_indices = coo_mem.map_data<int>(1));
    ASSERT_NO_THROW(mapped_col_indices = coo_mem.map_data<int>(2));

    for (size_t i = 0; i < coo_values.size(); i++)
        ASSERT_EQ(coo_values[i], mapped_coo_values[i]);

    for (size_t i = 0; i < row_indices.size(); i++)
        ASSERT_EQ(row_indices[i], mapped_row_indices[i]);

    for (size_t i = 0; i < col_indices.size(); i++)
        ASSERT_EQ(col_indices[i], mapped_col_indices[i]);

    ASSERT_NO_THROW(coo_mem.unmap_data(mapped_coo_values, 0));
    ASSERT_NO_THROW(coo_mem.unmap_data(mapped_row_indices, 1));
    ASSERT_NO_THROW(coo_mem.unmap_data(mapped_col_indices, 2));
}

} // namespace dnnl
