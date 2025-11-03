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

#ifndef GPU_INTEL_JIT_IR_INCLUDE_KERNEL_HPP
#define GPU_INTEL_JIT_IR_INCLUDE_KERNEL_HPP

#include <vector>

#include "gpu/intel/jit/ir/include/hw.hpp"
#include "gpu/intel/jit/ir/include/object.hpp"
#include "gpu/intel/jit/ir/include/type.hpp"

#include "ngen_debuginfo.hpp"

// NOLINTBEGIN(readability-identifier-naming)
namespace ngen {
class InterfaceHandler;
}
// NOLINTEND(readability-identifier-naming)

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

struct codegen_extension_interface_t;
using codegen_extension_handler_t
        = void (*)(const object_t &, codegen_extension_interface_t &);

namespace kernel {

// Representation for a kernel's function prototype
class iface_t {
public:
    iface_t(std::string name) : kernel_name_(std::move(name)) {}
    iface_t(const ngen::InterfaceHandler &iface);

    size_t nargs() const { return args_.size(); }
    const std::string &kernel_name() const { return kernel_name_; }
    const expr_t &operator[](size_t idx) const;
    bool has(const std::string &name) const { return find_arg_impl(name); }

    expr_t find_arg(const std::string &name, bool allow_empty = false) const;
    size_t index(const std::string &name) const;
    void register_arg(const expr_t &var) { args_.emplace_back(var); }
    void register_arg(const std::string &name, const type_t &type);

private:
    struct arg_t {
        arg_t() = default;
        arg_t(const expr_t &var) : var(var) {}
        const std::string &name() const;
        bool is_ptr() const { return var.type().is_ptr(); }

        expr_t var;
    };

    const arg_t *find_arg_impl(const std::string &name) const {
        for (size_t i = 0; i < nargs(); i++) {
            if (args_[i].name() == name) return &args_[i];
        }
        return nullptr;
    }

    std::string kernel_name_;
    std::vector<arg_t> args_;
};

extern codegen_extension_handler_t default_extension_handler;

// Compilation options used for IR generation and lowering
class options_t {
public:
    options_t() = default;
    options_t(const hw_t &hw) : hw_(hw) {}
    options_t(const hw_t &hw, int regs, int simd)
        : hw_(hw), regs_(regs), simd_(simd) {}

    const hw_t &hw() const { return hw_; }
    int regs() const { return regs_; }
    int simd() const { return simd_; }
    int grf_size() const { return hw_.grf_size(); }
    void set_regs(int regs) { regs_ = regs; }
    void set_simd(int simd) { simd_ = simd; }

    codegen_extension_handler_t extension_handler() const {
        return extension_handler_;
    }
    void set_extension_handler(codegen_extension_handler_t extension_handler) {
        extension_handler_ = extension_handler;
    }

    std::string str() const {
        ostringstream_t oss;
        oss << hw_.str();
        oss << ", SIMD: " << simd();
        oss << ", regs: " << regs();
        return oss.str();
    }

private:
    hw_t hw_;
    int regs_ = 0;
    int simd_ = 0;
    codegen_extension_handler_t extension_handler_ = default_extension_handler;
};

} // namespace kernel

struct kernel_t {
    kernel_t() : iface("invalid_dsl_kernel") {}
    kernel_t(kernel::iface_t iface, stmt_t body,
            const kernel::options_t &options)
        : iface(std::move(iface)), body(std::move(body)), options(options) {}

    kernel::iface_t iface;
    stmt_t body;
    kernel::options_t options;
    ngen::DebugConfig debug_cfg;
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
