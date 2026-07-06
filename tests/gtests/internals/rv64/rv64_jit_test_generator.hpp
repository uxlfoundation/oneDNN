/*******************************************************************************
* Copyright 2026 Barcelona Supercomputing Center
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

#ifndef TESTS_GTESTS_INTERNALS_RV64_JIT_TEST_GENERATOR_HPP
#define TESTS_GTESTS_INTERNALS_RV64_JIT_TEST_GENERATOR_HPP

#include <cstdint>
#include <utility>

#include "xbyak_riscv/xbyak_riscv.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

/// Minimal JIT harness for rvjit component tests
///
/// @details Stands in for the production `jit_generator_t`, which needs
///     symbols (`register_jit_code`, `get_max_cpu_isa`) hidden by
///     `-fvisibility=internal` in the shared library; these tests only need
///     to JIT and execute a probe kernel, so this harness talks to
///     `Xbyak_riscv::CodeGenerator` directly instead
class rv64_jit_test_generator_t : public Xbyak_riscv::CodeGenerator {
public:
    bool create_kernel() {
        generate();
        ready();
        return getCode() != nullptr;
    }

    template <typename... kernel_args_t>
    void operator()(kernel_args_t... args) const {
        using jit_kernel_func_t = void (*)(const kernel_args_t...);
        auto *fptr = reinterpret_cast<jit_kernel_func_t>(
                const_cast<uint8_t *>(getCode()));
        (*fptr)(std::forward<kernel_args_t>(args)...);
    }

protected:
    virtual void generate() = 0;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
