/*******************************************************************************
* Copyright 2019 Intel Corporation
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

extern "C" bool dnnl_impl_gpu_intel_mayiuse_ngen_kernels(dnnl_engine_t engine);

namespace dnnl {
namespace {

enum class dev_kind { null, cpu, gpu };
enum class ctx_kind { null, cpu, gpu };

} // namespace

struct ze_engine_test_t_params_t {
    dev_kind adev_kind;
    ctx_kind actx_kind;
    dnnl_status_t expected_status;
};

class ze_engine_test_t
    : public ::testing::TestWithParam<ze_engine_test_t_params_t> {
protected:
    void SetUp() override { find_ze_device(&ze_driver, &ze_dev, &ze_ctx); }

    void TearDown() override {
        if (ze_ctx) { zeContextDestroy(ze_ctx); }
    }

    ze_driver_handle_t ze_driver = nullptr;
    ze_device_handle_t ze_dev = nullptr;
    ze_context_handle_t ze_ctx = nullptr;
};

TEST_P(ze_engine_test_t, BasicInteropC) {
    auto p = GetParam();

    SKIP_IF(p.adev_kind != dev_kind::null && !ze_dev,
            "Required Level Zero device not found.");
    SKIP_IF(p.actx_kind != ctx_kind::null && !ze_ctx,
            "Required Level Zero context not found.");

    dnnl_engine_t eng = nullptr;
    dnnl_status_t s
            = dnnl_ze_interop_engine_create(&eng, ze_driver, ze_dev, ze_ctx);

    ASSERT_EQ(s, p.expected_status);

    if (s == dnnl_success) {
        ze_driver_handle_t driver = nullptr;
        ze_device_handle_t dev = nullptr;
        ze_context_handle_t ctx = nullptr;

        DNNL_CHECK(dnnl_ze_interop_engine_get_driver(eng, &driver));
        DNNL_CHECK(dnnl_ze_interop_engine_get_device(eng, &dev));
        DNNL_CHECK(dnnl_ze_interop_engine_get_context(eng, &ctx));

        ASSERT_EQ(driver, ze_driver);
        ASSERT_EQ(dev, ze_dev);
        ASSERT_EQ(ctx, ze_ctx);

        DNNL_CHECK(dnnl_engine_destroy(eng));

        // Verify that user's context wasn't destroyed.
        TEST_ZE_CHECK(zeContextGetStatus(ze_ctx));
    }
}

TEST_P(ze_engine_test_t, BasicInteropCpp) {
    auto p = GetParam();

    SKIP_IF(p.adev_kind != dev_kind::null && !ze_dev,
            "Required Level Zero device not found.");
    SKIP_IF(p.actx_kind != ctx_kind::null && !ze_ctx,
            "Required Level Zero context not found.");

    catch_expected_failures([&]() {
        {
            auto eng = ze_interop::make_engine(ze_driver, ze_dev, ze_ctx);
            if (p.expected_status != dnnl_success) {
                FAIL() << "Success not expected";
            }

            auto driver = ze_interop::get_driver(eng);
            auto dev = ze_interop::get_device(eng);
            auto ctx = ze_interop::get_context(eng);
            ASSERT_EQ(driver, ze_driver);
            ASSERT_EQ(dev, ze_dev);
            ASSERT_EQ(ctx, ze_ctx);
        }
        // Verify that user's context wasn't destroyed.
        TEST_ZE_CHECK(zeContextGetStatus(ze_ctx));
    }, p.expected_status != dnnl_success, p.expected_status);
}

TEST_P(ze_engine_test_t, BinaryKernels) {
    auto p = GetParam();
    SKIP_IF(p.adev_kind != dev_kind::null && !ze_dev,
            "Required Level Zero device not found.");
    SKIP_IF(p.actx_kind != ctx_kind::null && !ze_ctx,
            "Required Level Zero context not found.");

    dnnl_engine_t eng = nullptr;
    dnnl_status_t s
            = dnnl_ze_interop_engine_create(&eng, ze_driver, ze_dev, ze_ctx);

    ASSERT_EQ(s, p.expected_status);

//DNNL_ENABLE_MEM_DEBUG forces allocation fail, causing mayiuse to fail
#ifndef DNNL_ENABLE_MEM_DEBUG
    if (s == dnnl_success) {
        ASSERT_EQ(dnnl_impl_gpu_intel_mayiuse_ngen_kernels(eng), true);
    }
#endif

    if (s == dnnl_success) { DNNL_CHECK(dnnl_engine_destroy(eng)); }
}

INSTANTIATE_TEST_SUITE_P(Simple, ze_engine_test_t,
        ::testing::Values(ze_engine_test_t_params_t {
                dev_kind::gpu, ctx_kind::gpu, dnnl_success}));

} // namespace dnnl
