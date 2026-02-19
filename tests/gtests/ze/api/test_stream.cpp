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

#include "oneapi/dnnl/dnnl_ze.hpp"

namespace dnnl {
class ze_stream_test_c_t : public ::testing::Test {
protected:
    void SetUp() override {
        auto found = find_ze_device();
        if (!found) return;

        DNNL_CHECK(dnnl_engine_create(&eng, dnnl_gpu, 0));
        DNNL_CHECK(dnnl_ze_interop_engine_get_driver(eng, &ze_driver));
        DNNL_CHECK(dnnl_ze_interop_engine_get_device(eng, &ze_dev));
        DNNL_CHECK(dnnl_ze_interop_engine_get_context(eng, &ze_ctx));
    }

    void TearDown() override {
        if (eng) { DNNL_CHECK(dnnl_engine_destroy(eng)); }
    }

    dnnl_engine_t eng = nullptr;
    ze_driver_handle_t ze_driver = nullptr;
    ze_device_handle_t ze_dev = nullptr;
    ze_context_handle_t ze_ctx = nullptr;
};

class ze_stream_test_cpp_t : public ::testing::Test {
protected:
    void SetUp() override {
        auto found = find_ze_device();
        if (!found) return;

        eng = engine(engine::kind::gpu, 0);

        ze_driver = ze_interop::get_driver(eng);
        ze_dev = ze_interop::get_device(eng);
        ze_ctx = ze_interop::get_context(eng);
    }

    engine eng;
    ze_driver_handle_t ze_driver = nullptr;
    ze_device_handle_t ze_dev = nullptr;
    ze_context_handle_t ze_ctx = nullptr;
};

TEST_F(ze_stream_test_c_t, CreateC) {
    SKIP_IF(!ze_dev, "Level Zero GPU devices not found.");

    dnnl_stream_t stream;
    DNNL_CHECK(dnnl_stream_create(&stream, eng, dnnl_stream_default_flags));

    ze_command_list_handle_t ze_list;
    DNNL_CHECK(dnnl_ze_interop_stream_get_list(stream, &ze_list));

    ze_device_handle_t ze_list_dev = nullptr;
    ze_context_handle_t ze_list_ctx = nullptr;
    TEST_ZE_CHECK(zeCommandListGetDeviceHandle(ze_list, &ze_list_dev));
    TEST_ZE_CHECK(zeCommandListGetContextHandle(ze_list, &ze_list_ctx));

    ASSERT_EQ(ze_dev, ze_list_dev);
    ASSERT_EQ(ze_ctx, ze_list_ctx);

    DNNL_CHECK(dnnl_stream_destroy(stream));
}

TEST_F(ze_stream_test_cpp_t, CreateCpp) {
    SKIP_IF(!ze_dev, "Level Zero GPU devices not found.");

    stream s(eng);
    auto ze_list = ze_interop::get_list(s);

    ze_device_handle_t ze_list_dev = nullptr;
    ze_context_handle_t ze_list_ctx = nullptr;
    TEST_ZE_CHECK(zeCommandListGetDeviceHandle(ze_list, &ze_list_dev));
    TEST_ZE_CHECK(zeCommandListGetContextHandle(ze_list, &ze_list_ctx));

    ASSERT_EQ(ze_dev, ze_list_dev);
    ASSERT_EQ(ze_ctx, ze_list_ctx);
}

TEST_F(ze_stream_test_c_t, BasicInteropC) {
    SKIP_IF(!ze_dev, "Level Zero GPU devices not found.");

    ze_command_list_handle_t interop_ze_list
            = create_command_list(ze_ctx, ze_dev);

    dnnl_stream_t stream;
    DNNL_CHECK(dnnl_ze_interop_stream_create(
            &stream, eng, interop_ze_list, false));

    ze_command_list_handle_t ze_list;
    DNNL_CHECK(dnnl_ze_interop_stream_get_list(stream, &ze_list));
    ASSERT_EQ(ze_list, interop_ze_list);

    DNNL_CHECK(dnnl_stream_destroy(stream));

    ze_bool_t is_immediate;
    TEST_ZE_CHECK(zeCommandListIsImmediate(interop_ze_list, &is_immediate));
    ASSERT_EQ(is_immediate, true);

    TEST_ZE_CHECK(zeCommandListDestroy(interop_ze_list));
}

TEST_F(ze_stream_test_cpp_t, BasicInteropCpp) {
    SKIP_IF(!ze_dev, "Level Zero GPU devices not found.");

    ze_command_list_handle_t interop_ze_list
            = create_command_list(ze_ctx, ze_dev);

    {
        auto s = ze_interop::make_stream(eng, interop_ze_list);

        auto ze_list = ze_interop::get_list(s);
        ASSERT_EQ(ze_list, interop_ze_list);
    }

    ze_bool_t is_immediate;
    TEST_ZE_CHECK(zeCommandListIsImmediate(interop_ze_list, &is_immediate));
    ASSERT_EQ(is_immediate, true);

    TEST_ZE_CHECK(zeCommandListDestroy(interop_ze_list));
}

TEST_F(ze_stream_test_cpp_t, out_of_order_queue) {
    SKIP_IF(!ze_dev, "Level Zero GPU devices not found.");

    ze_command_list_handle_t interop_ze_list
            = create_command_list(ze_ctx, ze_dev,
                    /* in_order = */ false);

    memory::dims dims = {16, 16};
    memory::desc mem_d(dims, memory::data_type::f32, memory::format_tag::ab);

    auto matmul_pd = matmul::primitive_desc(eng, mem_d, mem_d, mem_d);
    auto mm = matmul(matmul_pd);

    auto mem_src = ze_interop::make_memory(mem_d, eng);
    auto mem_wei = ze_interop::make_memory(mem_d, eng);
    auto mem_dst = ze_interop::make_memory(mem_d, eng);

    auto stream = ze_interop::make_stream(eng, interop_ze_list);

    const int size = std::accumulate(dims.begin(), dims.end(),
            (dnnl::memory::dim)1, std::multiplies<dnnl::memory::dim>());

    std::vector<float> host_data_src(size);
    for (int i = 0; i < size; i++)
        host_data_src[i] = 1.f;

    std::vector<float> host_data_wei(size);
    for (int i = 0; i < size; i++)
        host_data_wei[i] = 1.f;

    ze_event_pool_handle_t event_pool = nullptr;
    ze_event_pool_desc_t event_pool_desc {};
    event_pool_desc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
    event_pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    event_pool_desc.count = 16;
    TEST_ZE_CHECK(zeEventPoolCreate(
            ze_ctx, &event_pool_desc, 0, nullptr, &event_pool));

    uint32_t event_idx = 0;
    ze_event_handle_t write_buffer_event0 = nullptr;
    ze_event_desc_t write_event_desc0 {};
    write_event_desc0.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
    write_event_desc0.index = event_idx++;
    write_event_desc0.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    write_event_desc0.wait = ZE_EVENT_SCOPE_FLAG_HOST;
    TEST_ZE_CHECK(zeEventCreate(
            event_pool, &write_event_desc0, &write_buffer_event0));

    TEST_ZE_CHECK(zeCommandListAppendMemoryCopy(interop_ze_list,
            mem_src.get_data_handle(), host_data_src.data(),
            size * sizeof(float), write_buffer_event0, 0, nullptr));

    ze_event_handle_t write_buffer_event1 = nullptr;
    ze_event_desc_t write_event_desc1 {};
    write_event_desc1.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
    write_event_desc1.index = event_idx++;
    write_event_desc1.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    write_event_desc1.wait = ZE_EVENT_SCOPE_FLAG_HOST;
    TEST_ZE_CHECK(zeEventCreate(
            event_pool, &write_event_desc1, &write_buffer_event1));

    TEST_ZE_CHECK(zeCommandListAppendMemoryCopy(interop_ze_list,
            mem_wei.get_data_handle(), host_data_wei.data(),
            size * sizeof(float), write_buffer_event1, 0, nullptr));

    ze_event_handle_t matmul_event = ze_interop::execute(mm, stream,
            {{DNNL_ARG_SRC, mem_src}, {DNNL_ARG_WEIGHTS, mem_wei},
                    {DNNL_ARG_DST, mem_dst}},
            {write_buffer_event0, write_buffer_event1});

    // Note: Level Zero API doesn't allow to query if the command list was
    // created as in-order or out-of-order. oneDNN assumes it's always in-order,
    // thus, doesn't return any event, though for out-of-order it's expected.
    // Once it starts, this condition should fail and be removed signalling
    // about correct flow.
    EXPECT_EQ(matmul_event, nullptr);

    if (matmul_event) {
        // Check results.
        ze_event_handle_t read_buffer_event = nullptr;
        ze_event_desc_t read_event_desc {};
        read_event_desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
        read_event_desc.index = event_idx++;
        read_event_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
        read_event_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
        TEST_ZE_CHECK(zeEventCreate(
                event_pool, &read_event_desc, &read_buffer_event));

        std::vector<float> host_data_dst(size, -1);
        TEST_ZE_CHECK(zeCommandListAppendMemoryCopy(interop_ze_list,
                host_data_dst.data(), mem_dst.get_data_handle(),
                size * sizeof(float), read_buffer_event, 1, &matmul_event));

        TEST_ZE_CHECK(zeEventHostSynchronize(read_buffer_event, UINT64_MAX));

        for (int i = 0; i < size; i++) {
            float exp_value = size;
            EXPECT_EQ(host_data_dst[i], exp_value);
        }

        TEST_ZE_CHECK(zeEventDestroy(read_buffer_event));
    }

    TEST_ZE_CHECK(zeEventDestroy(write_buffer_event0));
    TEST_ZE_CHECK(zeEventDestroy(write_buffer_event1));
    TEST_ZE_CHECK(zeEventPoolDestroy(event_pool));
    TEST_ZE_CHECK(zeCommandListDestroy(interop_ze_list));
}

#ifdef DNNL_EXPERIMENTAL_PROFILING
TEST_F(ze_stream_test_cpp_t, TestProfilingAPIUserQueue) {
    SKIP_IF(!ze_dev, "Level Zero GPU devices not found.");

    ze_command_list_handle_t interop_ze_list
            = create_command_list(ze_ctx, ze_dev);

    memory::dims dims = {16, 16};
    memory::desc mem_d(dims, memory::data_type::f32, memory::format_tag::ab);

    auto matmul_pd = matmul::primitive_desc(eng, mem_d, mem_d, mem_d);
    auto mm = matmul(matmul_pd);

    auto mem_src = ze_interop::make_memory(mem_d, eng);
    auto mem_wei = ze_interop::make_memory(mem_d, eng);
    auto mem_dst = ze_interop::make_memory(mem_d, eng);

    auto stream = ze_interop::make_stream(
            eng, interop_ze_list, /* profiling = */ true);

    // Reset profiler's state.
    ASSERT_NO_THROW(reset_profiling(stream));

    mm.execute(stream,
            {{DNNL_ARG_SRC, mem_src}, {DNNL_ARG_WEIGHTS, mem_wei},
                    {DNNL_ARG_DST, mem_dst}});
    stream.wait();

    // Query profiling data.
    std::vector<uint64_t> nsec;
    ASSERT_NO_THROW(
            nsec = get_profiling_data(stream, profiling_data_kind::time));
    ASSERT_FALSE(nsec.empty());

    // Reset profiler's state.
    ASSERT_NO_THROW(reset_profiling(stream));
    // Test that the profiler's state was reset.
    ASSERT_NO_THROW(
            nsec = get_profiling_data(stream, profiling_data_kind::time));
    ASSERT_TRUE(nsec.empty());

    TEST_ZE_CHECK(zeCommandListDestroy(interop_ze_list));
}
#endif

#ifdef DNNL_EXPERIMENTAL_PROFILING
TEST_F(ze_stream_test_cpp_t, TestProfilingAPILibraryQueue) {
    SKIP_IF(!ze_dev, "Level Zero GPU devices not found.");

    memory::dims dims = {16, 16};
    memory::desc mem_d(dims, memory::data_type::f32, memory::format_tag::ab);

    auto matmul_pd = matmul::primitive_desc(eng, mem_d, mem_d, mem_d);
    auto mm = matmul(matmul_pd);

    auto mem_src = ze_interop::make_memory(mem_d, eng);
    auto mem_wei = ze_interop::make_memory(mem_d, eng);
    auto mem_dst = ze_interop::make_memory(mem_d, eng);

    auto stream = dnnl::stream(eng, stream::flags::profiling);

    // Reset profiler's state.
    ASSERT_NO_THROW(reset_profiling(stream));

    mm.execute(stream,
            {{DNNL_ARG_SRC, mem_src}, {DNNL_ARG_WEIGHTS, mem_wei},
                    {DNNL_ARG_DST, mem_dst}});
    stream.wait();

    // Query profiling data.
    std::vector<uint64_t> nsec;
    ASSERT_NO_THROW(
            nsec = get_profiling_data(stream, profiling_data_kind::time));
    ASSERT_FALSE(nsec.empty());

    // Reset profiler's state.
    ASSERT_NO_THROW(reset_profiling(stream));
    // Test that the profiler's state was reset.
    ASSERT_NO_THROW(
            nsec = get_profiling_data(stream, profiling_data_kind::time));
    ASSERT_TRUE(nsec.empty());
}
#endif

#ifdef DNNL_EXPERIMENTAL_PROFILING
TEST_F(ze_stream_test_cpp_t, TestProfilingAPIOutOfOrderQueue) {
    SKIP_IF(!ze_dev, "Level Zero GPU devices not found.");

    ze_command_list_handle_t interop_ze_list
            = create_command_list(ze_ctx, ze_dev,
                    /* in_order = */ false);

    // Create stream with a user provided queue.
    // Note: Level Zero API doesn't allow to query if the command list was
    // created as in-order or out-of-order. Thus, no ability to restrict this
    // combination. Change when it becomes true.
    ASSERT_NO_THROW(auto stream = ze_interop::make_stream(
                            eng, interop_ze_list, /* profiling = */ true));
    // Create a stream with a library provided queue.
    ASSERT_ANY_THROW(
            auto stream = dnnl::stream(eng,
                    stream::flags::out_of_order | stream::flags::profiling));

    TEST_ZE_CHECK(zeCommandListDestroy(interop_ze_list));
}
#endif

} // namespace dnnl
