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

#include <cstdlib>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

#include "test_api_common.hpp"

using namespace testing;

// A mistyped option would otherwise silently run the tests on the default CPU
// engine, and a release build compiles out `assert`.
[[noreturn]] static void reject_cmd_option(const std::string &arg) {
    std::cerr << "Unsupported option: " << arg << std::endl;
    std::exit(EXIT_FAILURE);
}

static dnnl_engine_kind_t to_engine_kind(const std::string &str) {
    if (str.empty() || str == "cpu") return dnnl_cpu;

    if (str == "gpu") return dnnl_gpu;

    reject_cmd_option("--engine=" + str);
}

int main(int argc, char *argv[]) {
    int result;

    const std::string engine_option = "--engine=";
    const auto is_engine_option = [&](const std::string &arg) {
        return arg.compare(0, engine_option.size(), engine_option) == 0;
    };

    // `InitGoogleTest` evaluates parametrized test values, which read
    // `api_test_engine_kind`, so it must be set before the call.
    std::string engine_str;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (is_engine_option(arg))
            engine_str = arg.substr(engine_option.size());
    }
    api_test_engine_kind = to_engine_kind(engine_str);

    ::testing::InitGoogleTest(&argc, argv);

    // Only after `InitGoogleTest` has removed the gtest flags from `argv` can
    // the remaining options be validated.
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (!is_engine_option(arg)) reject_cmd_option(arg);
    }

#ifndef DNNL_GPU_RUNTIME
    if (api_test_engine_kind == dnnl_graph_gpu) {
        std::cout << "GPU runtime is not enabled" << std::endl;
        return 0;
    }
#endif

    result = RUN_ALL_TESTS();

    return result;
}
