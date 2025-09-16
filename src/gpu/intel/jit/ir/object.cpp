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

#include <typeindex>

#include "gpu/intel/jit/ir/core.hpp"
#include "gpu/intel/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
expr_t const_fold_non_recursive(const expr_t &expr);

object_t object::impl_t::_mutate(ir_mutator_t &mutator) const {
    return *this;
}
void object::impl_t::_visit(ir_visitor_t &visitor) const {}

const void *object::impl_t::get_uid(const std::type_info &info) {
    static std::unordered_map<std::type_index, const void *> type_registry;
    static std::mutex mutex;

    const std::lock_guard<std::mutex> guard(mutex);
    auto result = type_registry.emplace(std::type_index(info), &info);
    return result.first->second;
}

static void stmt_seq_flatten(std::vector<stmt_t> &out, const stmt_t &s) {
    if (auto *seq = s.as_ptr<stmt_seq_t>()) {
        out.insert(out.end(), seq->vec.begin(), seq->vec.end());
        return;
    }
    out.push_back(s);
}

expr_t expr_t::operator[](const expr_t &off) const {
    if (is<shuffle_t>()) {
        gpu_assert(is_const(off)) << "Offset is not constant.";
        auto &shuffle = as<shuffle_t>();
        int i_off = to_cpp<int>(off);
        gpu_assert(i_off < (int)shuffle.idx.size());
        int idx = shuffle.idx[i_off];
        return shuffle.vec[idx];
    }
    if (type().is_ptr()) return shift_ptr(op_kind_t::_add, *this, off);
    if (is<var_t>() || is<ref_t>()) {
        gpu_assert(is_const(off)) << "var/ref requires constant offset.";
        return ref_t::make(*this, to_cpp<int>(off), 1);
    }
    gpu_error_not_expected() << "Unexpected expression: " << str();
    return expr_t();
}

expr_t expr_t::ptr(const expr_t &off) const {
    if (is<var_t>()) return ptr_t::make(*this, off);
    if (auto *ref = as_ptr<ref_t>()) {
        return ptr_t::make(ref->var, ref->off + off);
    }
    gpu_error_not_expected() << "Unexpected expression: " << str();
    return expr_t();
}

expr_t::expr_t(bool value) : object_t(new bool_imm_t(value)) {}
expr_t::expr_t(float value) : object_t(new float_imm_t(value)) {}
expr_t::expr_t(double value)
    : object_t(new float_imm_t(value, type_t::f64())) {}
expr_t::expr_t(int16_t value) : object_t(new int_imm_t(value)) {}
expr_t::expr_t(int32_t value) : object_t(new int_imm_t(value)) {}
expr_t::expr_t(int64_t value) : object_t(new int_imm_t(value)) {}
expr_t::expr_t(uint16_t value) : object_t(new int_imm_t(value)) {}
expr_t::expr_t(uint32_t value) : object_t(new int_imm_t(value)) {}
expr_t::expr_t(uint64_t value) : object_t(new int_imm_t(value)) {}

bool is_const(const expr_t &e) {
    return e.is<bool_imm_t>() || e.is<int_imm_t>() || e.is<float_imm_t>();
}

bool to_bool(const expr_t &e) {
    return to_cpp<bool>(e);
}

expr_t operator-(const expr_t &a) {
    return const_fold_non_recursive(unary_op_t::make(op_kind_t::_minus, a));
}

expr_t div_up(const expr_t &a, const expr_t &b) {
    return const_fold_non_recursive(
            binary_op_t::make(op_kind_t::_div_up, a, b));
}

#define DEFINE_BINARY_OPERATOR(op, op_kind) \
    expr_t operator op(const expr_t &a, const expr_t &b) { \
        if (a.type().is_ptr()) return shift_ptr(op_kind, a, b); \
        return const_fold_non_recursive(binary_op_t::make(op_kind, a, b)); \
    }

DEFINE_BINARY_OPERATOR(+, op_kind_t::_add)
DEFINE_BINARY_OPERATOR(-, op_kind_t::_sub)
DEFINE_BINARY_OPERATOR(*, op_kind_t::_mul)
DEFINE_BINARY_OPERATOR(/, op_kind_t::_div)
DEFINE_BINARY_OPERATOR(%, op_kind_t::_mod)
DEFINE_BINARY_OPERATOR(<<, op_kind_t::_shl)
DEFINE_BINARY_OPERATOR(>>, op_kind_t::_shr)

DEFINE_BINARY_OPERATOR(==, op_kind_t::_eq)
DEFINE_BINARY_OPERATOR(!=, op_kind_t::_ne)
DEFINE_BINARY_OPERATOR(>, op_kind_t::_gt)
DEFINE_BINARY_OPERATOR(>=, op_kind_t::_ge)
DEFINE_BINARY_OPERATOR(<, op_kind_t::_lt)
DEFINE_BINARY_OPERATOR(<=, op_kind_t::_le)

DEFINE_BINARY_OPERATOR(&, op_kind_t::_and)
DEFINE_BINARY_OPERATOR(|, op_kind_t::_or)
DEFINE_BINARY_OPERATOR(^, op_kind_t::_xor)

#undef DEFINE_BINARY_OPERATOR

#define DEFINE_BINARY_ASSIGN_OPERATOR(op) \
    expr_t &expr_t::operator op##=(const expr_t &rhs) { \
        auto tmp = (*this)op rhs; \
        *this = std::move(tmp); \
        return *this; \
    }

DEFINE_BINARY_ASSIGN_OPERATOR(+)
DEFINE_BINARY_ASSIGN_OPERATOR(-)
DEFINE_BINARY_ASSIGN_OPERATOR(*)
DEFINE_BINARY_ASSIGN_OPERATOR(/)
DEFINE_BINARY_ASSIGN_OPERATOR(%)
DEFINE_BINARY_ASSIGN_OPERATOR(&)

#undef DEFINE_BINARY_ASSIGN_OPERATOR
stmt_t stmt_t::append(const stmt_t &s) const {
    if (is_empty()) return s;
    if (s.is_empty()) return *this;
    std::vector<stmt_t> vec;
    stmt_seq_flatten(vec, *this);
    stmt_seq_flatten(vec, s);
    return stmt_seq_t::make(vec);
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
