/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl.hpp"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "oneapi/dnnl/dnnl_ocl.hpp"
#endif

#include <cstring>

extern "C" void dnnl_impl_gpu_intel_clear_device_info_cache();

namespace dnnl {

class persistent_cache_api_test_t : public ::testing::Test {};

void check_engine_usable(const engine &eng) {
    memory::desc md({4, 4}, memory::data_type::f32, memory::format_tag::ab);
    auto prim = matmul(matmul::primitive_desc {eng, md, md, md});

    auto src = test::make_memory(md, eng);
    auto wei = test::make_memory(md, eng);
    auto dst = test::make_memory(md, eng);

    stream strm(eng);
    prim.execute(strm,
            {{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, wei},
                    {DNNL_ARG_DST, dst}});
    strm.wait();
}

// The engine blob starts with the version (three ints and the length-prefixed
// hash) followed by the runtime kind.
std::vector<uint8_t> change_engine_blob_runtime(
        std::vector<uint8_t> cache_blob) {
    const size_t runtime_offset = 3 * sizeof(int) + sizeof(size_t)
            + std::strlen(dnnl_version()->hash);
    cache_blob.at(runtime_offset)++;
    return cache_blob;
}

HANDLE_EXCEPTIONS_FOR_TEST(
        persistent_cache_api_test_t, TestPersistentCacheAPI) {
    engine e = get_test_engine();
    auto pd = convolution_forward::primitive_desc {e,
            prop_kind::forward_training, algorithm::convolution_direct,
            {{2, 16, 16, 16}, memory::data_type::f32, memory::format_tag::nchw},
            {{16, 16, 3, 3}, memory::data_type::f32, memory::format_tag::oihw},
            {{2, 16, 14, 14}, memory::data_type::f32, memory::format_tag::nchw},
            {1, 1}, {0, 0}, {0, 0}};
    auto p = convolution_forward(pd);

    std::vector<uint8_t> cache_blob_id;
    std::vector<uint8_t> cache_blob;

    ASSERT_NO_THROW(cache_blob_id = pd.get_cache_blob_id());
    ASSERT_EQ(cache_blob_id, pd.get_cache_blob_id());

    if (get_test_engine_kind() != engine::kind::gpu
            || (get_test_engine_kind() == engine::kind::gpu
                    && (DNNL_GPU_RUNTIME != DNNL_RUNTIME_OCL
                            && DNNL_GPU_RUNTIME != DNNL_RUNTIME_ZE))) {
        ASSERT_EQ(cache_blob_id.empty(), true);
        EXPECT_ANY_THROW(cache_blob = p.get_cache_blob());
        ASSERT_EQ(cache_blob.empty(), true);
        EXPECT_ANY_THROW(convolution_forward(pd, cache_blob));
    } else {
        ASSERT_EQ(cache_blob_id.empty(), false);
        ASSERT_NO_THROW(cache_blob = p.get_cache_blob());
        ASSERT_EQ(cache_blob.empty(), false);
        ASSERT_NO_THROW(p = convolution_forward(pd, cache_blob));
        ASSERT_EQ(cache_blob, p.get_cache_blob());

        // The blob starts with the oneDNN version.
        auto corrupted_blob = cache_blob;
        corrupted_blob[0] ^= 0xFF;
        EXPECT_ANY_THROW(convolution_forward(pd, corrupted_blob));
    }
}

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
HANDLE_EXCEPTIONS_FOR_TEST(
        persistent_cache_api_test_t, TestPersistentCacheAPIEngine) {
    using namespace dnnl::ocl_interop;
    engine test_engine = get_test_engine();

    if (get_test_engine_kind() != engine::kind::gpu) {
        ASSERT_ANY_THROW(get_engine_cache_blob(test_engine));
        return;
    }

    std::vector<uint8_t> cache_blob;
    std::vector<uint8_t> cache_blob_id;

    ASSERT_NO_THROW(cache_blob = get_engine_cache_blob(test_engine));
    ASSERT_NO_THROW(
            cache_blob_id = get_engine_cache_blob_id(get_device(test_engine)));

    ASSERT_EQ(get_engine_cache_blob(test_engine), cache_blob);
    ASSERT_EQ(get_engine_cache_blob_id(get_device(test_engine)), cache_blob_id);

    ASSERT_TRUE(!cache_blob.empty());
    ASSERT_TRUE(!cache_blob_id.empty());

    // Force init_from_cache_blob, otherwise the device info is reused from the
    // internal cache.
    dnnl_impl_gpu_intel_clear_device_info_cache();

    // The blob starts with the oneDNN version.
    auto corrupted_blob = cache_blob;
    corrupted_blob[0] ^= 0xFF;
    EXPECT_ANY_THROW(make_engine(
            get_device(test_engine), get_context(test_engine), corrupted_blob));

    // A failed engine creation prints the verbose header, which creates an
    // engine and populates the device info cache again.
    dnnl_impl_gpu_intel_clear_device_info_cache();
    EXPECT_ANY_THROW(make_engine(get_device(test_engine),
            get_context(test_engine), change_engine_blob_runtime(cache_blob)));

    dnnl_impl_gpu_intel_clear_device_info_cache();
    auto eng = make_engine(
            get_device(test_engine), get_context(test_engine), cache_blob);

    ASSERT_EQ(get_engine_cache_blob(eng), cache_blob);
    ASSERT_EQ(get_engine_cache_blob_id(get_device(eng)), cache_blob_id);

    check_engine_usable(eng);
}
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
HANDLE_EXCEPTIONS_FOR_TEST(
        persistent_cache_api_test_t, TestPersistentCacheAPIEngine) {
    using namespace dnnl::ze_interop;
    engine test_engine = get_test_engine();

    if (get_test_engine_kind() != engine::kind::gpu) {
        ASSERT_ANY_THROW(get_engine_cache_blob(test_engine));
        return;
    }

    std::vector<uint8_t> cache_blob;
    std::vector<uint8_t> cache_blob_id;

    ASSERT_NO_THROW(cache_blob = get_engine_cache_blob(test_engine));
    ASSERT_NO_THROW(cache_blob_id = get_engine_cache_blob_id(
                            get_driver(test_engine), get_device(test_engine)));

    ASSERT_EQ(get_engine_cache_blob(test_engine), cache_blob);
    ASSERT_EQ(get_engine_cache_blob_id(
                      get_driver(test_engine), get_device(test_engine)),
            cache_blob_id);

    ASSERT_TRUE(!cache_blob.empty());
    ASSERT_TRUE(!cache_blob_id.empty());

    // Force init_from_cache_blob, otherwise the device info is reused from the
    // internal cache.
    dnnl_impl_gpu_intel_clear_device_info_cache();

    // The blob starts with the oneDNN version.
    auto corrupted_blob = cache_blob;
    corrupted_blob[0] ^= 0xFF;
    EXPECT_ANY_THROW(make_engine(get_driver(test_engine),
            get_device(test_engine), get_context(test_engine), corrupted_blob));

    // A failed engine creation prints the verbose header, which creates an
    // engine and populates the device info cache again.
    dnnl_impl_gpu_intel_clear_device_info_cache();
    EXPECT_ANY_THROW(make_engine(get_driver(test_engine),
            get_device(test_engine), get_context(test_engine),
            change_engine_blob_runtime(cache_blob)));

    dnnl_impl_gpu_intel_clear_device_info_cache();
    auto eng = make_engine(get_driver(test_engine), get_device(test_engine),
            get_context(test_engine), cache_blob);

    ASSERT_EQ(get_engine_cache_blob(eng), cache_blob);
    ASSERT_EQ(
            get_engine_cache_blob_id(get_driver(test_engine), get_device(eng)),
            cache_blob_id);

    check_engine_usable(eng);
}

#endif

HANDLE_EXCEPTIONS_FOR_TEST(
        persistent_cache_api_test_t, TestPersistentCacheAPIDevicelessEngine) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL || DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    using namespace dnnl::ocl_interop;
    const auto deviceless_engine_create
            = dnnl_ocl_interop_deviceless_engine_create;
#else
    using namespace dnnl::ze_interop;
    const auto deviceless_engine_create
            = dnnl_ze_interop_deviceless_engine_create;
#endif
    engine test_engine = get_test_engine();

    if (get_test_engine_kind() != engine::kind::gpu) return;

    const auto cache_blob = get_engine_cache_blob(test_engine);
    ASSERT_TRUE(!cache_blob.empty());

    dnnl_engine_t c_engine = nullptr;
    EXPECT_EQ(deviceless_engine_create(
                      nullptr, cache_blob.size(), cache_blob.data()),
            dnnl_invalid_arguments);
    EXPECT_EQ(deviceless_engine_create(&c_engine, cache_blob.size(), nullptr),
            dnnl_invalid_arguments);
    EXPECT_EQ(deviceless_engine_create(&c_engine, 0, cache_blob.data()),
            dnnl_invalid_arguments);
    EXPECT_ANY_THROW(make_deviceless_engine({}));

    // The blob starts with the oneDNN version.
    auto corrupted_blob = cache_blob;
    corrupted_blob[0] ^= 0xFF;
    EXPECT_EQ(deviceless_engine_create(
                      &c_engine, corrupted_blob.size(), corrupted_blob.data()),
            dnnl_invalid_arguments);

    const auto foreign_runtime_blob = change_engine_blob_runtime(cache_blob);
    EXPECT_EQ(deviceless_engine_create(&c_engine, foreign_runtime_blob.size(),
                      foreign_runtime_blob.data()),
            dnnl_invalid_arguments);

    auto eng = make_deviceless_engine(cache_blob);
    ASSERT_EQ(eng.get_kind(), engine::kind::gpu);

    // Phase 2 re-exports the blob it was given.
    ASSERT_EQ(get_engine_cache_blob(eng), cache_blob);

    // Phase 3 finds the primitives compiled in phase 2 by this ID.
    memory::desc eltwise_md(
            {2, 16, 4, 4}, memory::data_type::f32, memory::format_tag::nchw);
    auto make_eltwise_pd = [&](const engine &e) {
        return eltwise_forward::primitive_desc(e, prop_kind::forward_inference,
                algorithm::eltwise_relu, eltwise_md, eltwise_md, 0.f);
    };
    EXPECT_EQ(make_eltwise_pd(eng).get_cache_blob_id(),
            make_eltwise_pd(test_engine).get_cache_blob_id());

    // Phase 3 runs the primitive compiled in phase 2, so both engines must
    // dispatch to the same implementation.
    memory::desc matmul_md(
            {64, 64}, memory::data_type::f32, memory::format_tag::ab);
    auto deviceless_matmul_pd
            = matmul::primitive_desc(eng, matmul_md, matmul_md, matmul_md);
    auto matmul_pd = matmul::primitive_desc(
            test_engine, matmul_md, matmul_md, matmul_md);
    ASSERT_STREQ(
            deviceless_matmul_pd.impl_info_str(), matmul_pd.impl_info_str());

    std::vector<uint8_t> primitive_blob;
    ASSERT_NO_THROW(
            primitive_blob = matmul(deviceless_matmul_pd).get_cache_blob());

    // Otherwise the primitive cache hands back the imported primitive as the
    // reference.
    const int primitive_cache_capacity = get_primitive_cache_capacity();
    set_primitive_cache_capacity(0);
    matmul imported_matmul;
    ASSERT_NO_THROW(imported_matmul = matmul(matmul_pd, primitive_blob));
    auto ref_matmul = matmul(matmul_pd);
    set_primitive_cache_capacity(primitive_cache_capacity);

    auto src = test::make_memory(matmul_md, test_engine);
    auto wei = test::make_memory(matmul_md, test_engine);
    auto dst = test::make_memory(matmul_md, test_engine);
    auto ref_dst = test::make_memory(matmul_md, test_engine);
    fill_data<float>(matmul_md.get_size() / sizeof(float), src);
    fill_data<float>(matmul_md.get_size() / sizeof(float), wei);

    stream strm(test_engine);
    imported_matmul.execute(strm,
            {{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, wei},
                    {DNNL_ARG_DST, dst}});
    ref_matmul.execute(strm,
            {{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, wei},
                    {DNNL_ARG_DST, ref_dst}});
    strm.wait();
    compare_data<float>(ref_dst, dst);

    dnnl_stream_t c_stream = nullptr;
    EXPECT_EQ(
            dnnl_stream_create(&c_stream, eng.get(), dnnl_stream_default_flags),
            dnnl_unimplemented);

    memory::desc md({4, 4}, memory::data_type::f32, memory::format_tag::ab);
    dnnl_memory_t c_memory = nullptr;
    EXPECT_NE(dnnl_memory_create(
                      &c_memory, md.get(), eng.get(), DNNL_MEMORY_ALLOCATE),
            dnnl_success);
#endif
}

HANDLE_EXCEPTIONS_FOR_TEST(
        persistent_cache_api_test_t, TestPersistentCacheAPIMemoryDesc) {
    engine e = get_test_engine();
    auto pd = convolution_forward::primitive_desc {e,
            prop_kind::forward_training, algorithm::convolution_direct,
            {{2, 16, 16, 16}, memory::data_type::f32, memory::format_tag::any},
            {{16, 16, 3, 3}, memory::data_type::f32, memory::format_tag::any},
            {{2, 16, 14, 14}, memory::data_type::f32, memory::format_tag::any},
            {1, 1}, {0, 0}, {0, 0}};

    auto wei_desc = pd.weights_desc();

    // C API check
    {
        auto c_wei_desc = wei_desc.get();

        // serialization
        size_t size;
        dnnl_memory_desc_get_blob(nullptr, &size, c_wei_desc);
        std::vector<uint8_t> serialized_wei_desc(size);
        dnnl_memory_desc_get_blob(
                serialized_wei_desc.data(), &size, c_wei_desc);

        // deserialization
        dnnl_memory_desc_t deserialized_wei_desc;
        dnnl_memory_desc_create_with_blob(
                &deserialized_wei_desc, serialized_wei_desc.data());

        ASSERT_EQ(pd.weights_desc(), deserialized_wei_desc);
    }

    // C++ API check
    {
        // serialization
        std::vector<uint8_t> serialized_wei_desc = wei_desc.get_blob();

        //  deserialization
        auto deserialized_desc = memory::desc(serialized_wei_desc);

        ASSERT_EQ(pd.weights_desc(), deserialized_desc);
    }
}

} // namespace dnnl
