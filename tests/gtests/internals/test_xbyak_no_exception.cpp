/*******************************************************************************
* Copyright 2026 Google LLC
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

#include "tests/gtests/dnnl_test_common.hpp"
#include "gtest/gtest.h"

// Explicitly disable exceptions in Xbyak so that errors are written to TLS.
#ifndef XBYAK_NO_EXCEPTION
#define XBYAK_NO_EXCEPTION
#endif

#include "src/cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace {

struct dummy_jit_t : public impl::cpu::x64::jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(dummy_jit_t)

    dummy_jit_t() : jit_generator_t(jit_name(), impl::cpu::x64::isa_all) {}

    void generate() override { ret(); }
};

TEST(jit_generator_test, thread_local_error_isolation) {
    // Pretend a prior failed JIT operation left an error on this thread.
    Xbyak::local::SetError(Xbyak::ERR_BAD_COMBINATION);
    ASSERT_EQ(Xbyak::GetError(), Xbyak::ERR_BAD_COMBINATION);

    // A new generator should start with a clean error state.
    dummy_jit_t generator;
    impl::status_t status = generator.create_kernel();

    EXPECT_EQ(status, impl::status::success);
    EXPECT_NE(generator.jit_ker(), nullptr);
    EXPECT_EQ(Xbyak::GetError(), Xbyak::ERR_NONE);
}

} // namespace
} // namespace dnnl
