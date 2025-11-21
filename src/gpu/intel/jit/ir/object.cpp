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
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
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
    if (type().is_ptr()) return *this + off;
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
    : object_t(new float_imm_t(value, dsl::type_t::f64())) {}
expr_t::expr_t(int16_t value) : object_t(new int_imm_t(value)) {}
expr_t::expr_t(int32_t value) : object_t(new int_imm_t(value)) {}
expr_t::expr_t(int64_t value) : object_t(new int_imm_t(value)) {}
expr_t::expr_t(uint16_t value) : object_t(new int_imm_t(value)) {}
expr_t::expr_t(uint32_t value) : object_t(new int_imm_t(value)) {}
expr_t::expr_t(uint64_t value) : object_t(new int_imm_t(value)) {}

bool expr_t::is(int value) const {
    if (!is_const(*this)) return false;
    return is_equal(to_expr(value, type()));
}

bool is_const(const expr_t &e) {
    return e.is<bool_imm_t>() || e.is<int_imm_t>() || e.is<float_imm_t>();
}

bool to_bool(const expr_t &e) {
    return to_cpp<bool>(e);
}

expr_t normalized_neg(const expr_t &a) {
    if (!a.type().is_scalar()) {
        int elems = a.type().elems();
        std::vector<expr_t> ret;
        ret.reserve(elems);
        for (int i = 0; i < elems; i++) {
            ret.push_back(normalized_neg(a[i]));
        }
        return shuffle_t::make(ret);
    }

    if (!is_const(a)) return unary_op_t::make(op_kind_t::_minus, a);

#define CASE(ir_type, cpp_type) \
    if (a.type() == dsl::type_t::ir_type()) return to_expr(-to_cpp<cpp_type>(a))

    CASE(f32, float);
    CASE(s16, int16_t);
    CASE(s32, int32_t);
    CASE(s64, int64_t);

#undef CASE

    gpu_error_not_expected() << "Cannot handle type: " << a;
    return expr_t();
}

template <op_kind_t>
struct evaluator_t {};

#define DECL_EVALUATOR(name, op) \
    template <> \
    struct evaluator_t<name> { \
        template <typename T> \
        static decltype(std::declval<T>() op std::declval<T>()) eval( \
                T a, T b) { \
            return a op b; \
        } \
    }

DECL_EVALUATOR(op_kind_t::_add, +);
DECL_EVALUATOR(op_kind_t::_sub, -);
DECL_EVALUATOR(op_kind_t::_mul, *);
DECL_EVALUATOR(op_kind_t::_div, /);
DECL_EVALUATOR(op_kind_t::_mod, %);
DECL_EVALUATOR(op_kind_t::_eq, ==);
DECL_EVALUATOR(op_kind_t::_ne, !=);
DECL_EVALUATOR(op_kind_t::_gt, >);
DECL_EVALUATOR(op_kind_t::_ge, >=);
DECL_EVALUATOR(op_kind_t::_lt, <);
DECL_EVALUATOR(op_kind_t::_le, <=);
DECL_EVALUATOR(op_kind_t::_shl, <<);
DECL_EVALUATOR(op_kind_t::_shr, >>);
DECL_EVALUATOR(op_kind_t::_and, &);
DECL_EVALUATOR(op_kind_t::_or, |);
DECL_EVALUATOR(op_kind_t::_xor, |);

#undef DECL_EVALUATOR

template <op_kind_t op>
expr_t const_binary(const expr_t &a, const expr_t &b) {
    auto compute_type = common_type(a, b);
    if (!compute_type.is_scalar()) {
        int elems = compute_type.elems();
        std::vector<expr_t> ret;
        ret.reserve(elems);
        for (int i = 0; i < elems; i++) {
            ret.push_back(const_binary<op>(a[i], b[i]));
        }
        return shuffle_t::make(ret);
    }

#define CASE(ir_type, cpp_type) \
    if (compute_type == dsl::type_t::ir_type()) { \
        auto _a = to_cpp<cpp_type>(a); \
        auto _b = to_cpp<cpp_type>(b); \
        return evaluator_t<op>::eval(_a, _b); \
    }

    CASE(s16, int16_t)
    CASE(s32, int32_t)
    CASE(s64, int64_t)
    CASE(u16, uint16_t)
    CASE(u32, uint32_t)
    CASE(u64, uint64_t)

#undef CASE

    if (compute_type == dsl::type_t::_bool()) {
        auto _a = to_cpp<bool>(a);
        auto _b = to_cpp<bool>(b);
        if (op == op_kind_t::_and) return to_expr(_a & _b);
        if (op == op_kind_t::_or) return to_expr(_a | _b);
        if (op == op_kind_t::_xor) return to_expr(_a ^ _b);
        gpu_error_not_expected();
    }

    if (compute_type == dsl::type_t::f32()) {
        auto _a = to_cpp<float>(a);
        auto _b = to_cpp<float>(b);
        if (op == op_kind_t::_add) return to_expr(_a + _b);
        if (op == op_kind_t::_sub) return to_expr(_a - _b);
        if (op == op_kind_t::_mul) return to_expr(_a * _b);
        if (op == op_kind_t::_div) return to_expr(_a / _b);
        gpu_error_not_expected();
    }

    gpu_error_not_expected() << "Unknown type.";
    return expr_t();
}

expr_t normalized_mul(const expr_t &a, const expr_t &b) {
    gpu_assert(!a.type().is_ptr() && !b.type().is_ptr());
    if (a.is(0) || b.is(0)) return 0;
    if (a.is(1)) return b;
    if (b.is(1)) return a;
    if (is_const(a) && is_const(b)) return const_binary<op_kind_t::_mul>(a, b);
    return binary_op_t::make(op_kind_t::_mul, a, b);
}

expr_t normalized_div(const expr_t &a, const expr_t &b) {
    gpu_assert(!a.type().is_ptr() && !b.type().is_ptr());
    gpu_assert(!b.is(0));
    if (a.is(0)) return 0;
    if (b.is(1)) return a;
    if (a.is_equal(b)) return 1;
    if (is_const(a) && is_const(b)) return const_binary<op_kind_t::_div>(a, b);
    return binary_op_t::make(op_kind_t::_div, a, b);
}

expr_t normalized_mod(const expr_t &a, const expr_t &b) {
    gpu_assert(!a.type().is_ptr() && !b.type().is_ptr());
    gpu_assert(!b.is(0));
    if (a.is(0) || b.is(1) || a.is_equal(b)) return 0;
    if (is_const(a) && is_const(b)) return const_binary<op_kind_t::_mod>(a, b);
    return binary_op_t::make(op_kind_t::_mod, a, b);
}

expr_t normalized_add(const expr_t &a, const expr_t &b) {
    if (a.is<ptr_t>()) {
        gpu_assert(b.type().is_int());
        auto &ptr = a.as<ptr_t>();
        return ptr_t::make(ptr.base, normalized_add(ptr.off, b));
    }
    if (a.is<ref_t>()) {
        gpu_assert(b.type().is_int());
        auto &ref = a.as<ref_t>();
        return ref_t::make(ref.var, ref.off + to_cpp<int>(b), ref.elems);
    }
    if (a.is(0)) return b;
    if (b.is(0)) return a;
    if (is_const(a) && is_const(b)) return const_binary<op_kind_t::_add>(a, b);
    if (a.is_equal(b)) return normalized_mul(2, a);
    return binary_op_t::make(op_kind_t::_add, a, b);
}

expr_t normalized_sub(const expr_t &a, const expr_t &b) {
    if (a.is<ptr_t>()) {
        gpu_assert(!b.type().is_int());
        auto &ptr = a.as<ptr_t>();
        return ptr_t::make(ptr.base, normalized_sub(ptr.off, b));
    }
    if (a.is<ref_t>()) {
        gpu_assert(!b.type().is_int());
        auto &ref = a.as<ref_t>();
        return ref_t::make(ref.var, ref.off - to_cpp<int>(b), ref.elems);
    }
    if (b.is(0)) return a;
    if (is_const(a) && is_const(b)) return const_binary<op_kind_t::_sub>(a, b);
    if (a.is_equal(b)) return 0;
    if (a.is(0)) return unary_op_t::make(op_kind_t::_sub, b);
    return binary_op_t::make(op_kind_t::_sub, a, b);
}

expr_t normalized_shl(const expr_t &a, const expr_t &b) {
    if (b.is(0)) return a;
    if (is_const(a) && is_const(b)) return const_binary<op_kind_t::_shl>(a, b);
    return binary_op_t::make(op_kind_t::_shl, a, b);
}

expr_t normalized_shr(const expr_t &a, const expr_t &b) {
    if (b.is(0)) return a;
    if (is_const(a) && is_const(b)) return const_binary<op_kind_t::_shr>(a, b);
    return binary_op_t::make(op_kind_t::_shr, a, b);
}

expr_t normalized_eq(const expr_t &a, const expr_t &b) {
    if (a.is_equal(b)) return true;
    if (is_const(a) && is_const(b)) return const_binary<op_kind_t::_eq>(a, b);
    return binary_op_t::make(op_kind_t::_eq, a, b);
}

expr_t normalized_ne(const expr_t &a, const expr_t &b) {
    if (a.is_equal(b)) return false;
    if (is_const(a) && is_const(b)) return const_binary<op_kind_t::_ne>(a, b);
    return binary_op_t::make(op_kind_t::_ne, a, b);
}

expr_t normalized_gt(const expr_t &a, const expr_t &b) {
    if (a.is_equal(b)) return false;
    if (is_const(a) && is_const(b)) return const_binary<op_kind_t::_gt>(a, b);
    return binary_op_t::make(op_kind_t::_gt, a, b);
}

expr_t normalized_ge(const expr_t &a, const expr_t &b) {
    if (a.is_equal(b)) return true;
    if (is_const(a) && is_const(b)) return const_binary<op_kind_t::_ge>(a, b);
    return binary_op_t::make(op_kind_t::_ge, a, b);
}

expr_t normalized_lt(const expr_t &a, const expr_t &b) {
    if (a.is_equal(b)) return false;
    if (is_const(a) && is_const(b)) return const_binary<op_kind_t::_lt>(a, b);
    return binary_op_t::make(op_kind_t::_lt, a, b);
}

expr_t normalized_le(const expr_t &a, const expr_t &b) {
    if (a.is_equal(b)) return true;
    if (is_const(a) && is_const(b)) return const_binary<op_kind_t::_le>(a, b);
    return binary_op_t::make(op_kind_t::_le, a, b);
}

expr_t normalized_and(const expr_t &a, const expr_t &b) {
    if (is_const(a)) return to_cpp<bool>(a) ? b : false;
    if (is_const(b)) return to_cpp<bool>(b) ? a : false;
    return binary_op_t::make(op_kind_t::_and, a, b);
}

expr_t normalized_or(const expr_t &a, const expr_t &b) {
    if (is_const(a)) return to_cpp<bool>(a) ? true : b;
    if (is_const(b)) return to_cpp<bool>(b) ? true : a;
    return binary_op_t::make(op_kind_t::_or, a, b);
}

expr_t normalized_xor(const expr_t &a, const expr_t &b) {
    if (is_const(a) && is_const(b)) return const_binary<op_kind_t::_xor>(a, b);
    return binary_op_t::make(op_kind_t::_xor, a, b);
}

expr_t operator-(const expr_t &a) {
    return normalized_neg(a);
}

#define DEFINE_BINARY_OPERATOR(op, op_kind) \
    expr_t operator op(const expr_t &a, const expr_t &b) { \
        return normalized_##op_kind(a, b); \
    }

DEFINE_BINARY_OPERATOR(+, add)
DEFINE_BINARY_OPERATOR(-, sub)
DEFINE_BINARY_OPERATOR(*, mul)
DEFINE_BINARY_OPERATOR(/, div)
DEFINE_BINARY_OPERATOR(%, mod)
DEFINE_BINARY_OPERATOR(<<, shl)
DEFINE_BINARY_OPERATOR(>>, shr)

DEFINE_BINARY_OPERATOR(==, eq)
DEFINE_BINARY_OPERATOR(!=, ne)
DEFINE_BINARY_OPERATOR(>, gt)
DEFINE_BINARY_OPERATOR(>=, ge)
DEFINE_BINARY_OPERATOR(<, lt)
DEFINE_BINARY_OPERATOR(<=, le)

DEFINE_BINARY_OPERATOR(&, and)
DEFINE_BINARY_OPERATOR(|, or)
DEFINE_BINARY_OPERATOR(^, xor)

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
