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
#include "oneapi/dnnl/dnnl_sycl.hpp"

#include <vector>

using namespace dnnl;

class sycl_host_scalar_reorder_test_t : public ::testing::Test {};

TEST_F(sycl_host_scalar_reorder_test_t, HostScalarToGPU_f32) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "GPU not found.");

    engine gpu_eng(engine::kind::gpu, 0);

    const float scalar_value = 42.0f;

    // Create host scalar memory (no engine required).
    auto scalar_md = memory::desc::host_scalar(memory::data_type::f32);
    memory scalar_mem(scalar_md, scalar_value);

    // Create 1-element GPU destination memory.
    memory::desc dst_md({1}, memory::data_type::f32, memory::format_tag::x);
    memory dst_mem(dst_md, gpu_eng);

    // Create reorder from memory objects (host scalar has no engine).
    reorder::primitive_desc rpd(scalar_mem, dst_mem);
    reorder r(rpd);

    // Execute.
    stream s(gpu_eng);
    r.execute(s, {{DNNL_ARG_FROM, scalar_mem}, {DNNL_ARG_TO, dst_mem}});
    s.wait();

    // Read back from GPU and verify.
    auto q = sycl_interop::get_queue(stream(gpu_eng));
    float result = 0.0f;
    q.memcpy(&result, dst_mem.get_data_handle(), sizeof(float)).wait();
    ASSERT_EQ(result, scalar_value);
}

TEST_F(sycl_host_scalar_reorder_test_t, HostScalarToGPU_f16) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "GPU not found.");

    engine gpu_eng(engine::kind::gpu, 0);

    // f16 is 2 bytes; use uint16_t to hold the value.
    // f16 representation of 1.0 is 0x3C00.
    const uint16_t scalar_value = 0x3C00;

    auto scalar_md = memory::desc::host_scalar(memory::data_type::f16);
    memory scalar_mem(scalar_md, scalar_value);

    memory::desc dst_md({1}, memory::data_type::f16, memory::format_tag::x);
    memory dst_mem(dst_md, gpu_eng);

    reorder::primitive_desc rpd(scalar_mem, dst_mem);
    reorder r(rpd);

    stream s(gpu_eng);
    r.execute(s, {{DNNL_ARG_FROM, scalar_mem}, {DNNL_ARG_TO, dst_mem}});
    s.wait();

    auto q = sycl_interop::get_queue(stream(gpu_eng));
    uint16_t result = 0;
    q.memcpy(&result, dst_mem.get_data_handle(), sizeof(uint16_t)).wait();
    ASSERT_EQ(result, scalar_value);
}

TEST_F(sycl_host_scalar_reorder_test_t, HostScalarToGPU_s32) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "GPU not found.");

    engine gpu_eng(engine::kind::gpu, 0);

    const int32_t scalar_value = -123;

    auto scalar_md = memory::desc::host_scalar(memory::data_type::s32);
    memory scalar_mem(scalar_md, scalar_value);

    memory::desc dst_md({1}, memory::data_type::s32, memory::format_tag::x);
    memory dst_mem(dst_md, gpu_eng);

    reorder::primitive_desc rpd(scalar_mem, dst_mem);
    reorder r(rpd);

    stream s(gpu_eng);
    r.execute(s, {{DNNL_ARG_FROM, scalar_mem}, {DNNL_ARG_TO, dst_mem}});
    s.wait();

    auto q = sycl_interop::get_queue(stream(gpu_eng));
    int32_t result = 0;
    q.memcpy(&result, dst_mem.get_data_handle(), sizeof(int32_t)).wait();
    ASSERT_EQ(result, scalar_value);
}

TEST_F(sycl_host_scalar_reorder_test_t, RejectDifferentDataTypes) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "GPU not found.");

    engine gpu_eng(engine::kind::gpu, 0);

    const float dummy = 0.0f;
    auto scalar_md = memory::desc::host_scalar(memory::data_type::f32);
    memory scalar_mem(scalar_md, dummy);
    memory::desc dst_md({1}, memory::data_type::s32, memory::format_tag::x);
    memory dst_mem(dst_md, gpu_eng);

    // Should fail: different data types.
    EXPECT_ANY_THROW(reorder::primitive_desc(scalar_mem, dst_mem));
}

TEST_F(sycl_host_scalar_reorder_test_t, RejectMultiElementDst) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "GPU not found.");

    engine gpu_eng(engine::kind::gpu, 0);

    const float dummy = 0.0f;
    auto scalar_md = memory::desc::host_scalar(memory::data_type::f32);
    memory scalar_mem(scalar_md, dummy);
    memory::desc dst_md({4}, memory::data_type::f32, memory::format_tag::x);
    memory dst_mem(dst_md, gpu_eng);

    // Should fail: destination has more than 1 element.
    EXPECT_ANY_THROW(reorder::primitive_desc(scalar_mem, dst_mem));
}
