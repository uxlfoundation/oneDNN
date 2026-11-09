/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#include "gtest/gtest.h"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <string>

#include "dnnl_test_common.hpp"

#include "gtest/gtest.h"

using namespace testing;

static std::atomic<bool> g_is_current_test_failed(false);
bool is_current_test_failed() {
    return g_is_current_test_failed;
}

class assert_fail_handler_t : public EmptyTestEventListener {
protected:
    void OnTestStart(const TestInfo &test_info) override {
        g_is_current_test_failed = false;
    }
    void OnTestPartResult(const testing::TestPartResult &part_result) override {
        if (part_result.type() == testing::TestPartResult::kFatalFailure) {
            g_is_current_test_failed = true;
        }
    }
};

class dnnl_environment_t : public ::testing::Environment {
public:
    void SetUp() override;
    void TearDown() override;
};

static void test_init(int argc, char *argv[]);

int main(int argc, char *argv[]) {
    int result;
    {
        ::testing::InitGoogleTest(&argc, argv);

        // Parse oneDNN command line arguments
        test_init(argc, argv);

        TestEventListeners &listeners = UnitTest::GetInstance()->listeners();

        auto *fail_handler = new assert_fail_handler_t();
        listeners.Append(fail_handler);

        ::testing::AddGlobalTestEnvironment(new dnnl_environment_t());

#if defined(_WIN32)
        // Safety cleanup.
        system("where /q umdh && del pre_cpu.txt");
        system("where /q umdh && del post_cpu.txt");
        system("where /q umdh && del memdiff_cpu.txt");

        // Get first snapshot.
        system("where /q umdh && umdh -pn:tests.exe -f:pre_cpu.txt");
#endif

        result = RUN_ALL_TESTS();
    }

#if defined(_WIN32)
    // Get second snapshot.
    system("where /q umdh && umdh -pn:tests.exe -f:post_cpu.txt");

    // Prepare memory diff.
    system("where /q umdh && umdh pre_cpu.txt post_cpu.txt -f:memdiff_cpu.txt");

    // Cleanup.
    system("where /q umdh && del pre_cpu.txt");
    system("where /q umdh && del post_cpu.txt");
#endif

    finalize_tbb();
    return result;
}

// A mistyped option would otherwise silently run the tests on the default CPU
// engine, and a release build compiles out `assert`.
[[noreturn]] static void reject_cmd_option(const std::string &arg) {
    std::cerr << "Unsupported option: " << arg << std::endl;
    std::exit(EXIT_FAILURE);
}

inline dnnl::engine::kind to_engine_kind(const std::string &str) {
    if (str.empty() || str == "cpu") return dnnl::engine::kind::cpu;

    if (str == "gpu") return dnnl::engine::kind::gpu;

    reject_cmd_option("--engine=" + str);
}

// test_engine can be accessed only from tests compiled with
// DNNL_TEST_WITH_ENGINE_PARAM macro
#ifdef DNNL_TEST_WITH_ENGINE_PARAM
static dnnl::engine::kind test_engine_kind;
static std::unique_ptr<dnnl::engine> test_engine;

dnnl::engine::kind get_test_engine_kind() {
    return test_engine_kind;
}

dnnl::engine get_test_engine() {
    return *test_engine;
}

void dnnl_environment_t::SetUp() {
    test_engine.reset(new dnnl::engine(get_test_engine_kind(), 0));
}

void dnnl_environment_t::TearDown() {
    test_engine.reset();
}
#else
void dnnl_environment_t::SetUp() {}
void dnnl_environment_t::TearDown() {}
#endif

void test_init(int argc, char *argv[]) {
#ifdef DNNL_TEST_WITH_ENGINE_PARAM
    const std::string engine_option = "--engine=";
    std::string engine_str;
#endif
    // `InitGoogleTest` has already removed the gtest flags from `argv`.
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
#ifdef DNNL_TEST_WITH_ENGINE_PARAM
        if (arg.compare(0, engine_option.size(), engine_option) == 0) {
            engine_str = arg.substr(engine_option.size());
            continue;
        }
#endif
        reject_cmd_option(arg);
    }
#ifdef DNNL_TEST_WITH_ENGINE_PARAM
    test_engine_kind = to_engine_kind(engine_str);

    std::string filter_str = ::testing::GTEST_FLAG(filter);
    if (test_engine_kind == dnnl::engine::kind::cpu) {
        // Exclude non-CPU tests
        ::testing::GTEST_FLAG(filter) = filter_str + ":-*_GPU*";
    } else if (test_engine_kind == dnnl::engine::kind::gpu) {
        // Exclude non-GPU tests
        ::testing::GTEST_FLAG(filter) = filter_str + ":-*_CPU*";
    }
#endif
}
