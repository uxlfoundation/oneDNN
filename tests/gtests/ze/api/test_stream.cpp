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

#include "oneapi/dnnl/dnnl_ze.h"
#include "oneapi/dnnl/dnnl_ze.hpp"

#include <memory>

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

    ze_command_queue_desc_t command_queue_desc = {};
    command_queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    command_queue_desc.ordinal = 0;
    command_queue_desc.index = 0;
    command_queue_desc.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
    command_queue_desc.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
    command_queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ze_command_list_handle_t interop_ze_list = nullptr;
    TEST_ZE_CHECK(zeCommandListCreateImmediate(
            ze_ctx, ze_dev, &command_queue_desc, &interop_ze_list));

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

// TEST_F(ze_stream_test_cpp_t, BasicInteropCpp) {
//     SKIP_IF(!find_ze_device(CL_DEVICE_TYPE_GPU),
//             "Level Zero GPU devices not found.");
//
//     cl_int err;
// #ifdef CL_VERSION_2_0
//     cl_command_queue interop_ze_queue
//             = clCreateCommandQueueWithProperties(ze_ctx, ze_dev, nullptr, &err);
// #else
//     cl_command_queue interop_ze_queue
//             = clCreateCommandQueue(ze_ctx, ze_dev, 0, &err);
// #endif
//     TEST_ZE_CHECK(err);
//
//     {
//         auto s = ze_interop::make_stream(eng, interop_ze_queue);
//
//         cl_uint ref_count;
//         TEST_ZE_CHECK(clGetCommandQueueInfo(interop_ze_queue,
//                 CL_QUEUE_REFERENCE_COUNT, sizeof(cl_uint), &ref_count,
//                 nullptr));
//         int i_ref_count = int(ref_count);
//         ASSERT_EQ(i_ref_count, 2);
//
//         cl_command_queue ze_queue = ze_interop::get_command_queue(s);
//         ASSERT_EQ(ze_queue, interop_ze_queue);
//     }
//
//     cl_uint ref_count;
//     TEST_ZE_CHECK(clGetCommandQueueInfo(interop_ze_queue,
//             CL_QUEUE_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr));
//     int i_ref_count = int(ref_count);
//     ASSERT_EQ(i_ref_count, 1);
//
//     TEST_ZE_CHECK(clReleaseCommandQueue(interop_ze_queue));
// }
//
// TEST_F(ze_stream_test_c_t, InteropIncompatibleQueueC) {
//     SKIP_IF(!find_ze_device(CL_DEVICE_TYPE_GPU),
//             "Level Zero GPU devices not found.");
//
//     cl_device_id cpu_ze_dev = find_ze_device(CL_DEVICE_TYPE_CPU);
//     SKIP_IF(!cpu_ze_dev, "Level Zero CPU devices not found.");
//
//     cl_int err;
//     cl_context cpu_ze_ctx
//             = clCreateContext(nullptr, 1, &cpu_ze_dev, nullptr, nullptr, &err);
//     TEST_ZE_CHECK(err);
//
// #ifdef CL_VERSION_2_0
//     cl_command_queue cpu_ze_queue = clCreateCommandQueueWithProperties(
//             cpu_ze_ctx, cpu_ze_dev, nullptr, &err);
// #else
//     cl_command_queue cpu_ze_queue
//             = clCreateCommandQueue(cpu_ze_ctx, cpu_ze_dev, 0, &err);
// #endif
//     TEST_ZE_CHECK(err);
//
//     dnnl_stream_t stream;
//     dnnl_status_t status
//             = dnnl_ze_interop_stream_create(&stream, eng, cpu_ze_queue);
//     ASSERT_EQ(status, dnnl_invalid_arguments);
//
//     TEST_ZE_CHECK(clReleaseCommandQueue(cpu_ze_queue));
// }
//
// TEST_F(ze_stream_test_cpp_t, InteropIncompatibleQueueCpp) {
//     SKIP_IF(!find_ze_device(CL_DEVICE_TYPE_GPU),
//             "Level Zero GPU devices not found.");
//
//     cl_device_id cpu_ze_dev = find_ze_device(CL_DEVICE_TYPE_CPU);
//     SKIP_IF(!cpu_ze_dev, "Level Zero CPU devices not found.");
//
//     cl_int err;
//     cl_context cpu_ze_ctx
//             = clCreateContext(nullptr, 1, &cpu_ze_dev, nullptr, nullptr, &err);
//     TEST_ZE_CHECK(err);
//
// #ifdef CL_VERSION_2_0
//     cl_command_queue cpu_ze_queue = clCreateCommandQueueWithProperties(
//             cpu_ze_ctx, cpu_ze_dev, nullptr, &err);
// #else
//     cl_command_queue cpu_ze_queue
//             = clCreateCommandQueue(cpu_ze_ctx, cpu_ze_dev, 0, &err);
// #endif
//     TEST_ZE_CHECK(err);
//
//     catch_expected_failures([&] { ze_interop::make_stream(eng, cpu_ze_queue); },
//             true, dnnl_invalid_arguments);
//
//     TEST_ZE_CHECK(clReleaseCommandQueue(cpu_ze_queue));
// }
//
// TEST_F(ze_stream_test_cpp_t, out_of_order_queue) {
//     SKIP_IF(!find_ze_device(CL_DEVICE_TYPE_GPU),
//             "Level Zero GPU devices not found.");
//
//     cl_int err;
//
// #ifdef CL_VERSION_2_0
//     cl_queue_properties properties[]
//             = {CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0};
//     cl_command_queue ze_queue = clCreateCommandQueueWithProperties(
//             ze_ctx, ze_dev, properties, &err);
// #else
//     cl_command_queue_properties properties
//             = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
//     cl_command_queue ze_queue
//             = clCreateCommandQueue(ze_ctx, ze_dev, properties, &err);
// #endif
//     TEST_ZE_CHECK(err);
//
//     memory::dims dims = {2, 3, 4, 5};
//     memory::desc mem_d(dims, memory::data_type::f32, memory::format_tag::nchw);
//
//     auto eltwise_pd = eltwise_forward::primitive_desc(eng, prop_kind::forward,
//             algorithm::eltwise_relu, mem_d, mem_d, 0.0f);
//     auto eltwise = eltwise_forward(eltwise_pd);
//
//     auto mem = ze_interop::make_memory(
//             mem_d, eng, ze_interop::memory_kind::buffer);
//
//     auto stream = ze_interop::make_stream(eng, ze_queue);
//
//     const int size = std::accumulate(dims.begin(), dims.end(),
//             (dnnl::memory::dim)1, std::multiplies<dnnl::memory::dim>());
//
//     std::vector<float> host_data_src(size);
//     for (int i = 0; i < size; i++)
//         host_data_src[i] = static_cast<float>(i - size / 2);
//
//     cl_event write_buffer_event;
//     TEST_ZE_CHECK(
//             clEnqueueWriteBuffer(ze_queue, ze_interop::get_mem_object(mem),
//                     /* blocking */ CL_FALSE, 0, size * sizeof(float),
//                     host_data_src.data(), 0, nullptr, &write_buffer_event));
//
//     cl_event eltwise_event = ze_interop::execute(eltwise, stream,
//             {{DNNL_ARG_SRC, mem}, {DNNL_ARG_DST, mem}}, {write_buffer_event});
//
//     // Check results.
//     std::vector<float> host_data_dst(size, -1);
//     cl_event read_buffer_event;
//     TEST_ZE_CHECK(clEnqueueReadBuffer(ze_queue, ze_interop::get_mem_object(mem),
//             /* blocking */ CL_FALSE, 0, size * sizeof(float),
//             host_data_dst.data(), 1, &eltwise_event, &read_buffer_event));
//     TEST_ZE_CHECK(clWaitForEvents(1, &read_buffer_event));
//
//     for (int i = 0; i < size; i++) {
//         float exp_value
//                 = static_cast<float>((i - size / 2) <= 0 ? 0 : (i - size / 2));
//         EXPECT_EQ(host_data_dst[i], exp_value);
//     }
//
//     TEST_ZE_CHECK(clReleaseEvent(read_buffer_event));
//     TEST_ZE_CHECK(clReleaseEvent(write_buffer_event));
//     TEST_ZE_CHECK(clReleaseEvent(eltwise_event));
//     TEST_ZE_CHECK(clReleaseCommandQueue(ze_queue));
// }
//
// #ifdef DNNL_EXPERIMENTAL_PROFILING
// TEST_F(ze_stream_test_cpp_t, TestProfilingAPIUserQueue) {
//     SKIP_IF(!find_ze_device(CL_DEVICE_TYPE_GPU),
//             "Level Zero GPU devices not found.");
//
//     memory::dims dims = {2, 3, 4, 5};
//     memory::desc md(dims, memory::data_type::f32, memory::format_tag::nchw);
//
//     auto eltwise_pd = eltwise_forward::primitive_desc(
//             eng, prop_kind::forward, algorithm::eltwise_relu, md, md, 0.0f);
//     auto eltwise = eltwise_forward(eltwise_pd);
//     auto mem = memory(md, eng);
//
//     cl_int err;
// #ifdef CL_VERSION_2_0
//     cl_queue_properties properties[]
//             = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
//     cl_command_queue ze_queue = clCreateCommandQueueWithProperties(
//             ze_ctx, ze_dev, properties, &err);
// #else
//     cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
//     cl_command_queue ze_queue
//             = clCreateCommandQueue(ze_ctx, ze_dev, properties, &err);
// #endif
//
//     TEST_ZE_CHECK(err);
//
//     auto stream = ze_interop::make_stream(eng, ze_queue);
//     TEST_ZE_CHECK(clReleaseCommandQueue(ze_queue));
//
//     // Reset profiler's state.
//     ASSERT_NO_THROW(reset_profiling(stream));
//
//     eltwise.execute(stream, {{DNNL_ARG_SRC, mem}, {DNNL_ARG_DST, mem}});
//     stream.wait();
//
//     // Query profiling data.
//     std::vector<uint64_t> nsec;
//     ASSERT_NO_THROW(
//             nsec = get_profiling_data(stream, profiling_data_kind::time));
//     ASSERT_FALSE(nsec.empty());
//
//     // Reset profiler's state.
//     ASSERT_NO_THROW(reset_profiling(stream));
//     // Test that the profiler's state was reset.
//     ASSERT_NO_THROW(
//             nsec = get_profiling_data(stream, profiling_data_kind::time));
//     ASSERT_TRUE(nsec.empty());
// }
//
// TEST_F(ze_stream_test_cpp_t, TestProfilingAPILibraryQueue) {
//     SKIP_IF(!find_ze_device(CL_DEVICE_TYPE_GPU),
//             "Level Zero GPU devices not found.");
//
//     memory::dims dims = {2, 3, 4, 5};
//     memory::desc md(dims, memory::data_type::f32, memory::format_tag::nchw);
//
//     auto eltwise_pd = eltwise_forward::primitive_desc(
//             eng, prop_kind::forward, algorithm::eltwise_relu, md, md, 0.0f);
//     auto eltwise = eltwise_forward(eltwise_pd);
//     auto mem = memory(md, eng);
//
//     auto stream = dnnl::stream(eng, stream::flags::profiling);
//
//     // Reset profiler's state.
//     ASSERT_NO_THROW(reset_profiling(stream));
//
//     eltwise.execute(stream, {{DNNL_ARG_SRC, mem}, {DNNL_ARG_DST, mem}});
//     stream.wait();
//
//     // Query profiling data.
//     std::vector<uint64_t> nsec;
//     ASSERT_NO_THROW(
//             nsec = get_profiling_data(stream, profiling_data_kind::time));
//     ASSERT_FALSE(nsec.empty());
//
//     // Reset profiler's state.
//     ASSERT_NO_THROW(reset_profiling(stream));
//     // Test that the profiler's state was reset.
//     ASSERT_NO_THROW(
//             nsec = get_profiling_data(stream, profiling_data_kind::time));
//     ASSERT_TRUE(nsec.empty());
// }
//
// TEST_F(ze_stream_test_cpp_t, TestProfilingAPIOutOfOrderQueue) {
//     SKIP_IF(!find_ze_device(CL_DEVICE_TYPE_GPU),
//             "Level Zero GPU devices not found.");
//     cl_int err;
// #ifdef CL_VERSION_2_0
//     cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES,
//             CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
//             0};
//     cl_command_queue ze_queue = clCreateCommandQueueWithProperties(
//             ze_ctx, ze_dev, properties, &err);
// #else
//     cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE
//             | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
//     cl_command_queue ze_queue
//             = clCreateCommandQueue(ze_ctx, ze_dev, properties, &err);
// #endif
//     TEST_ZE_CHECK(err);
//
//     // Create stream with a user provided queue.
//     ASSERT_ANY_THROW(auto stream = ze_interop::make_stream(eng, ze_queue));
//     TEST_ZE_CHECK(clReleaseCommandQueue(ze_queue));
//     // Create a stream with a library provided queue.
//     ASSERT_ANY_THROW(
//             auto stream = dnnl::stream(eng,
//                     stream::flags::out_of_order | stream ::flags::profiling));
// }
//
// #if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
// TEST_F(ze_stream_test_cpp_t, TestProfilingAPICPU) {
//     auto eng = engine(engine::kind::cpu, 0);
//     ASSERT_ANY_THROW(auto stream = dnnl::stream(eng, stream::flags::profiling));
// }
// #endif
//
// #endif
//
// #ifndef DNNL_EXPERIMENTAL_PROFILING
// extern "C" dnnl_status_t dnnl_reset_profiling(dnnl_stream_t stream);
// #endif
//
// TEST_F(ze_stream_test_cpp_t, TestProfilingAPIDisabledAndEnabled) {
//     SKIP_IF(!find_ze_device(CL_DEVICE_TYPE_GPU),
//             "Level Zero GPU devices not found.");
//
//     memory::dims dims = {2, 3, 4, 5};
//     memory::desc md(dims, memory::data_type::f32, memory::format_tag::nchw);
//
//     auto eltwise_pd = eltwise_forward::primitive_desc(
//             eng, prop_kind::forward, algorithm::eltwise_relu, md, md, 0.0f);
//     auto eltwise = eltwise_forward(eltwise_pd);
//     auto mem = memory(md, eng);
//
//     cl_int err;
// #ifdef CL_VERSION_2_0
//     cl_queue_properties properties[]
//             = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
//     cl_command_queue ze_queue = clCreateCommandQueueWithProperties(
//             ze_ctx, ze_dev, properties, &err);
// #else
//     cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
//     cl_command_queue ze_queue
//             = clCreateCommandQueue(ze_ctx, ze_dev, properties, &err);
// #endif
//
//     TEST_ZE_CHECK(err);
//
//     auto stream = ze_interop::make_stream(eng, ze_queue);
//     TEST_ZE_CHECK(clReleaseCommandQueue(ze_queue));
//
//     eltwise.execute(stream, {{DNNL_ARG_SRC, mem}, {DNNL_ARG_DST, mem}});
//     stream.wait();
//
//     auto st = dnnl_reset_profiling(stream.get());
//
// // If the experimental profiling API is not enabled then the library should not
// // enable profiling regardless of the queue's properties.
// #ifdef DNNL_EXPERIMENTAL_PROFILING
//     EXPECT_EQ(st, dnnl_success);
// #else
//     EXPECT_EQ(st, dnnl_invalid_arguments);
// #endif
// }

} // namespace dnnl
