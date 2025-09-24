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

#include "gpu/intel/jit/ir/include/object.hpp"
#include "gpu/intel/jit/ir/include/type.hpp"

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
namespace kernel {

// Representation for a kernel's function prototype
class iface_t {
public:
    iface_t(std::string name) : kernel_name_(std::move(name)) {}
    iface_t(const ngen::InterfaceHandler &iface);

    size_t nargs() const { return args_.size(); }
    const std::string &kernel_name() const { return kernel_name_; }
    const expr_t &operator[](size_t idx) const {
        gpu_assert(idx < nargs());
        return args_[idx].var;
    }
    bool has(const std::string &name) const { return find_arg_impl(name); }

    expr_t find_arg(const std::string &name, bool allow_empty = false) const {
        auto *arg = find_arg_impl(name);
        if (arg) return arg->var;
        if (!allow_empty)
            gpu_error_not_expected() << "Argument not found: " << name;
        return expr_t();
    }

    size_t index(const std::string &name) const {
        for (size_t i = 0; i < nargs(); i++) {
            if (args_[i].name() == name) return i;
        }
        return -1;
    }

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
} // namespace kernel

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
