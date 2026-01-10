/*******************************************************************************
* Copyright 2025 ZTE Corporation
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

#ifndef CPU_RV64_JIT_GENERATOR_HPP
#define CPU_RV64_JIT_GENERATOR_HPP

#include <cstdint>
#include <utility>

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/rv64/cpu_isa_traits.hpp"
#include "xbyak_riscv/xbyak_riscv.hpp"

#define DECLARE_CPU_JIT_AUX_FUNCTIONS(gen_name) \
    const char *name() const override { \
        return STRINGIFY(gen_name); \
    } \
    const char *source_file() const override { \
        return __FILE__; \
    }

#define JIT_ASSERT(condition) \
    do { \
        assert(condition); \
        if (!(condition)) XBYAK_RISCV_THROW(Xbyak_riscv::ERR_INTERNAL); \
    } while (false)

#define JIT_ASSERT_RET(condition, ret) \
    do { \
        assert(condition); \
        if (!(condition)) \
            XBYAK_RISCV_THROW_RET(Xbyak_riscv::ERR_INTERNAL, ret); \
    } while (false)

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

// Simple helper to check subset relation between two ISA masks.
inline bool is_subset(cpu_isa_t isa, cpu_isa_t max_isa) {
    using u_t = typename std::underlying_type<cpu_isa_t>::type;
    return (static_cast<u_t>(isa) & static_cast<u_t>(max_isa))
            == static_cast<u_t>(isa);
}

// Minimal RV64 JIT generator base class.
class jit_generator_t : public Xbyak_riscv::CodeGenerator, public c_compatible {
public:
    using c_compatible::operator new;
    using c_compatible::operator new[];
    using c_compatible::operator delete;
    using c_compatible::operator delete[];

    // All JIT kernels must override these to provide a stable name used for
    // debug/logging and jit code registration.
    virtual const char *name() const = 0;
    virtual const char *source_file() const = 0;

    explicit jit_generator_t(const char * /*unused_name*/,
            cpu_isa_t max_cpu_isa = get_max_cpu_isa())
        : Xbyak_riscv::CodeGenerator(max_code_size)
        , max_cpu_isa_(max_cpu_isa) {}

    ~jit_generator_t() override = default;

    const uint8_t *jit_ker() const { return jit_ker_; }

    template <typename... kernel_args_t>
    void operator()(kernel_args_t... args) const {
        using jit_kernel_func_t = void (*)(const kernel_args_t...);
        auto *fptr = reinterpret_cast<jit_kernel_func_t>(jit_ker_);
        (*fptr)(std::forward<kernel_args_t>(args)...);
    }

    virtual status_t create_kernel() {
        try {
            generate();
        } catch (...) { return status::runtime_error; }

        this->ready(Xbyak_riscv::CodeArray::PROTECT_RWE);

        jit_ker_ = Xbyak_riscv::CodeGenerator::getCode();

        if (jit_ker_) {
            // TODO: integrate RV64 JIT code registration with cpu::jit_utils
            // once profiling / dumping support is enabled for this backend.
            return status::success;
        }

        return status::runtime_error;
    }

    inline cpu_isa_t max_cpu_isa() const noexcept { return max_cpu_isa_; }

    // Helper to check that a requested ISA is both within the per‑kernel limit
    // and supported by the current CPU.
    inline bool is_valid_isa(cpu_isa_t isa) const {
        return is_subset(isa, max_cpu_isa_) && mayiuse(isa);
    }

protected:
    virtual void generate() = 0;

private:
    static constexpr unsigned max_code_size = 256 * 1024;

    const cpu_isa_t max_cpu_isa_;
    const uint8_t *jit_ker_ = nullptr;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
