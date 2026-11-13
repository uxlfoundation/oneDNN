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

#include "gtest/gtest.h"
#include "gtest/internal/gtest-internal.h"

#include "unit_test_common.hpp"

#include "interface/c_types_map.hpp"

using namespace testing;
namespace graph = dnnl::impl::graph;

// A mistyped option would otherwise silently run the tests on the default CPU
// engine, and a release build compiles out `assert`.
[[noreturn]] static void reject_cmd_option(const std::string &arg) {
    std::cerr << "Unsupported option: " << arg << std::endl;
    std::exit(EXIT_FAILURE);
}

inline graph::engine_kind_t to_engine_kind(const std::string &str) {
    if (str.empty() || str == "cpu") return graph::engine_kind::cpu;

    if (str == "gpu") return graph::engine_kind::gpu;

    reject_cmd_option("--engine=" + str);
}

int main(int argc, char *argv[]) {
    int result;

    (void)(::testing::GTEST_FLAG(death_test_style) = "threadsafe");
    // Runs first to remove the gtest flags from `argv`.
    ::testing::InitGoogleTest(&argc, argv);

    const std::string engine_option = "--engine=";
    std::string engine_str;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg.compare(0, engine_option.size(), engine_option) != 0)
            reject_cmd_option(arg);
        engine_str = arg.substr(engine_option.size());
    }
    set_test_engine_kind(to_engine_kind(engine_str));
    const graph::engine_kind_t engine_kind = get_test_engine_kind();

#if !defined(DNNL_GPU_RUNTIME) || DNNL_GPU_RUNTIME == DNNL_RUNTIME_NONE
    if (engine_kind == graph::engine_kind::gpu) {
        std::cout << "GPU runtime is not enabled" << std::endl;
        return 0;
    }
#endif

#if !defined(DNNL_CPU_RUNTIME) || DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE
    if (engine_kind == graph::engine_kind::cpu) {
        std::cout << "CPU runtime is not enabled" << std::endl;
        return 0;
    }
#endif

    std::string filter_str = ::testing::GTEST_FLAG(filter);
    if (engine_kind == graph::engine_kind::cpu) {
        // Exclude non-CPU tests
        ::testing::GTEST_FLAG(filter) = filter_str + ":-*_GPU*";
    } else if (engine_kind == graph::engine_kind::gpu) {
        // Exclude non-GPU tests
        ::testing::GTEST_FLAG(filter) = filter_str + ":-*_CPU*";
    } else {
        std::cout << "unsupported engine kind" << std::endl;
        return 0;
    }

    result = RUN_ALL_TESTS();

    return result;
}
