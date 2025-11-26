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

#ifndef GPU_INTEL_JIT_IR_KERNEL_INFO_HPP
#define GPU_INTEL_JIT_IR_KERNEL_INFO_HPP

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/primitive_exec_types.hpp"
#include "gpu/intel/jit/ir/core.hpp"
#include "gpu/intel/jit/ir/include/kernel.hpp"
#include "gpu/intel/jit/ir/kernel_desc.hpp"
#include "gpu/intel/jit/utils/type_bridge.hpp"
#include "gpu/intel/primitive.hpp"
#include "ngen_interface.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

class memory_storage_ptr_t {
public:
    memory_storage_ptr_t(std::unique_ptr<memory_storage_t> &&ptr)
        : unique_ptr_(std::move(ptr)) {}
    memory_storage_ptr_t(const memory_storage_t *ptr) : raw_ptr_(ptr) {}
    memory_storage_ptr_t(const memory_storage_ptr_t &) = delete;
    memory_storage_ptr_t &operator=(const memory_storage_ptr_t &) = delete;
    ~memory_storage_ptr_t() = default;

    const memory_storage_t *get() const {
        if (unique_ptr_) return unique_ptr_.get();
        return raw_ptr_;
    }

private:
    std::unique_ptr<memory_storage_t> unique_ptr_; // Owning pointer.
    const memory_storage_t *raw_ptr_ = nullptr; // Non-owning pointer.
};

class memory_storage_wrapper_t {
public:
    memory_storage_wrapper_t() = default;
    memory_storage_wrapper_t(std::unique_ptr<memory_storage_t> &&ptr)
        : ptr_(new memory_storage_ptr_t(std::move(ptr))) {}
    memory_storage_wrapper_t(const memory_storage_t *ptr)
        : ptr_(new memory_storage_ptr_t(ptr)) {}
    memory_storage_wrapper_t(const memory_storage_t &ref)
        : memory_storage_wrapper_t(&ref) {}

    const memory_storage_t *get() const {
        if (!ptr_) return nullptr;
        return ptr_->get();
    }

private:
    std::shared_ptr<memory_storage_ptr_t> ptr_;
};

enum class kernel_id_t {
    undef,
    convolution,
    pre_reorder,
    post_reorder,
    zero_out,
    zp_precalc,
};

// Kernel information, includes:
// - Kernel identifier
// - Kernel arguments
// - ND-range for submission (optional)
// Kernel arguments can be of the following kinds:
// - Internal arguments: only scalar
//   - Examples: common output scales (contain a single value)
// - User arguments: passed by the user at run time
//   - Examples: source, weights, destination
class kernel_info_t {
public:
    void set_id(kernel_id_t id) { id_ = id; }

    kernel_id_t id() const { return id_; }

    // Returns stage ID, kernels with smaller stage IDs are executed first.
    int stage_id() const {
        switch (id()) {
            case kernel_id_t::zero_out: return 0;
            case kernel_id_t::zp_precalc: return 1;
            case kernel_id_t::pre_reorder: return 2;
            case kernel_id_t::convolution: return 3;
            case kernel_id_t::post_reorder: return 4;
            default: gpu_error_not_expected();
        }
        return -1;
    }

    void set_nd_range(const compute::nd_range_t &nd_range) {
        nd_range_ = nd_range;
    }

    const compute::nd_range_t &nd_range() const { return nd_range_; }

    void register_immediate_arg(const expr_t &var,
            const expr_t &value = expr_t(), int dnnl_arg = DNNL_ARG_UNDEF) {
        gpu_assert(value.is_empty() || (dnnl_arg == DNNL_ARG_UNDEF));
        register_arg(var, arg_kind_t::immediate, dnnl_arg, /*is_input=*/true);
        set_immediate_arg(var.as<var_t>().name, value);
    }

    void set_immediate_arg(const std::string &name, const expr_t &value) {
        auto *arg = find_arg_impl(name);
        gpu_assert(arg) << "Cannot find argument: " << name;
        arg->value = value;
    }

    void register_user_arg(const expr_t &var, int dnnl_arg, bool is_input) {
        register_arg(var, arg_kind_t::user, dnnl_arg, is_input);
    }

    void register_scratchpad_arg(
            const expr_t &var, int key, bool is_input, size_t size) {
        register_arg(var, arg_kind_t::scratchpad, key, is_input, size);
    }

    const std::string &arg_name(int idx) const {
        gpu_assert(idx >= 0 && idx < nargs());
        return args_[idx].name();
    }

    const expr_t &arg_var(int idx) const {
        gpu_assert(idx >= 0 && idx < nargs());
        return args_[idx].var;
    }

    const type_t &arg_type(int idx) const { return arg_var(idx).type(); }

    expr_t find_arg(const std::string &name, bool allow_empty = false) const {
        auto *arg = find_arg_impl(name);
        if (arg) return arg->var;
        if (!allow_empty)
            gpu_error_not_expected() << "Argument not found: " << name;
        return expr_t();
    }

    int key(int idx) const {
        gpu_assert(idx >= 0 && idx < nargs());
        return args_[idx].key;
    }

    int key(const std::string &name) const {
        for (int i = 0; i < nargs(); i++) {
            if (arg_name(i) == name) return key(i);
        }
        gpu_error_not_expected() << "Argument not found: " << name;
        return -1;
    }

    int nargs() const { return int(args_.size()); }

    bool is_scratchpad(int idx) const {
        gpu_assert(idx >= 0 && idx < nargs());
        return args_[idx].kind == arg_kind_t::scratchpad;
    }

    bool is_user(int idx) const {
        gpu_assert(idx >= 0 && idx < nargs());
        return args_[idx].kind == arg_kind_t::user;
    }

    bool is_input(int idx) const {
        gpu_assert(idx >= 0 && idx < nargs());
        return args_[idx].is_input;
    }

    bool is_output(int idx) const { return !is_input(idx); }

    kernel::iface_t iface(const std::string &name) const {
        kernel::iface_t iface(name);
        for (int i = 0; i < nargs(); i++) {
            iface.register_arg(args_[i].var);
        }
        return iface;
    }

    memory_storage_wrapper_t arg_storage(int idx, const exec_ctx_t &ctx,
            const primitive_t *primitive) const {
        gpu_assert(idx >= 0 && idx < nargs());
        bool is_input = args_[idx].is_input;
        int key = args_[idx].key;
        switch (args_[idx].kind) {
            case arg_kind_t::scratchpad:
                return ctx.get_scratchpad_grantor().get_memory_storage(key);
            case arg_kind_t::user:
                if (!is_input) return CTX_OUT_STORAGE(key);
                return CTX_IN_STORAGE(key);
            // No storage for immediate arguments unless host-scalar
            case arg_kind_t::immediate:
                if (key == DNNL_ARG_UNDEF) return memory_storage_wrapper_t();
                return CTX_IN_STORAGE(key);
            default: gpu_error_not_expected();
        }
        return memory_storage_wrapper_t();
    }

    size_t arg_size(int idx, const primitive_t *primitive) const {
        switch (args_[idx].kind) {
            case arg_kind_t::user: {
                auto *md = primitive->pd()->arg_md(key(idx));
                return memory_desc_wrapper(md).size();
            }
            case arg_kind_t::scratchpad: return args_[idx].scratchpad_size;
            default: gpu_error_not_expected();
        }
        return std::numeric_limits<size_t>::max();
    }

    void init_memory_storage_list(std::vector<memory_storage_wrapper_t> &list,
            const exec_ctx_t &ctx, const primitive_t *primitive) const {
        list = std::vector<memory_storage_wrapper_t>(nargs());
        for (int i = 0; i < nargs(); i++) {
            list[i] = arg_storage(i, ctx, primitive);
        }
    }

    void set_args(compute::kernel_arg_list_t &arg_list,
            const std::vector<memory_storage_wrapper_t> &storage_list) const {
#define CASE_IF(type, ir_type, cpp_type) \
    if ((type) == type_t::ir_type()) { \
        CASE(cpp_type); \
        break; \
    }
#define ALL_CASES(type) \
    do { \
        CASE_IF(type, f32, float) \
        CASE_IF(type, s8, int8_t) \
        CASE_IF(type, s16, int16_t) \
        CASE_IF(type, s32, int32_t) \
        CASE_IF(type, s64, int64_t) \
        CASE_IF(type, u8, uint8_t) \
        CASE_IF(type, u16, uint16_t) \
        CASE_IF(type, u32, uint32_t) \
        CASE_IF(type, u64, uint64_t) \
        gpu_error_not_expected() << (type); \
    } while (false);

        for (int i = 0; i < nargs(); i++) {
            switch (args_[i].kind) {
                case arg_kind_t::immediate: {
                    auto value = args_[i].value;
                    if (value.is_empty()) {
                        auto storage = storage_list[i].get();
                        gpu_assert(storage && storage->is_host_scalar());
                        auto *host_storage = utils::downcast<
                                const host_scalar_memory_storage_t *>(storage);
                        auto type = to_ir(host_storage->data_type());
#define CASE(cpp_type) \
    cpp_type buf = 0; \
    auto status = host_storage->get_scalar_value(&buf, sizeof(buf)); \
    gpu_assert(status == status::success); \
    value = expr_t(buf);
                        ALL_CASES(type)
#undef CASE
                    }
                    auto &type = args_[i].var.type();
#define CASE(cpp_type) arg_list.set(i, to_cpp<cpp_type>(value));
                    ALL_CASES(type)
#undef CASE
                    break;
                }
                case arg_kind_t::scratchpad:
                case arg_kind_t::user: {
                    arg_list.set(i, *storage_list[i].get());
                    break;
                }
                default: gpu_error_not_expected();
            }
        }
#undef ALL_CASES
#undef CASE_IF
    }

private:
    enum class arg_kind_t { immediate, scratchpad, user };

    struct arg_t {
        arg_t(const expr_t &var, arg_kind_t kind, int key, bool is_input,
                size_t scratchpad_size)
            : var(var)
            , kind(kind)
            , key(key)
            , is_input(is_input)
            , scratchpad_size(scratchpad_size) {}

        const std::string &name() const { return var.as<var_t>().name; }

        expr_t var;
        arg_kind_t kind;
        int key; // Unique key across arguments with the same kind.
        bool is_input;
        expr_t value; // For immediate arguments; must be a constant.
        size_t scratchpad_size; // For scratchpad arguments only.
    };

    void register_arg(const expr_t &var, arg_kind_t kind, int key,
            bool is_input, size_t scratchpad_size = 0) {
        gpu_assert(is_var(var)) << "Expected var, got: " << var;
        args_.emplace_back(var, kind, key, is_input, scratchpad_size);
    }

    const arg_t *find_arg_impl(const std::string &name) const {
        for (int i = 0; i < nargs(); i++) {
            if (args_[i].name() == name) return &args_[i];
        }
        return nullptr;
    }

    arg_t *find_arg_impl(const std::string &name) {
        auto *arg
                = const_cast<const kernel_info_t *>(this)->find_arg_impl(name);
        return const_cast<arg_t *>(arg);
    }

    kernel_id_t id_ = kernel_id_t::undef;
    compute::nd_range_t nd_range_;

    std::vector<arg_t> args_;
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
