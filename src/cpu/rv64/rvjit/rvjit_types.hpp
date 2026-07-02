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

#ifndef CPU_RV64_RVJIT_RVJIT_TYPES_HPP
#define CPU_RV64_RVJIT_RVJIT_TYPES_HPP

#include <algorithm>
#include <array>
#include <functional>
#include <utility>
#include <initializer_list>
#include <type_traits>

#ifndef XBYAK_RISCV_V
#define XBYAK_RISCV_V 1
#endif

#include "xbyak_riscv/xbyak_riscv.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace rvjit {

// Fundamental Code-generation Types

using codegen_t = Xbyak_riscv::CodeGenerator;
using Reg = Xbyak_riscv::Reg;
using FReg = Xbyak_riscv::FReg;
using VReg = Xbyak_riscv::VReg;
using Label = Xbyak_riscv::Label;
using LMUL = Xbyak_riscv::LMUL;
using SEW = Xbyak_riscv::SEW;
using VM = Xbyak_riscv::VM;
using VTA = Xbyak_riscv::VTA;
using VMA = Xbyak_riscv::VMA;

/// Non-owning handle to the underlying code generator used by the components
class emitter_t {
public:
    explicit emitter_t(codegen_t &cg) : cg_(&cg) {}

    codegen_t *operator->() const { return cg_; }
    codegen_t &cg() const { return *cg_; }

private:
    codegen_t *cg_;
};

// Forward declarations from rvjit_utils.hpp
bool is_writeable(const Reg &r);
bool is_not_zero(const Reg &r);
bool is_simm12(int i);
bool is_pow2(int64_t v);

/// Element kind used across rvjit's typed memory-move and macc components
///
/// @note Scope is exactly the 8 values rvjit ever handles; there is no
///     entry for wider oneDNN dtypes since those already no-op today
enum class optype_t { s8, u8, s32, s64, f16, bf16, f32, f64 };

// Callback types

/// Callback type to emit code for a parametrized code block
template <typename... Args>
using callback_t = std::function<void(Args...)>;

// Callback type to emit code for an unparametrized basic block
using on_enter_t = callback_t<>;

// Callback type to emit code for a conditional basic block
using on_cond_t = callback_t<bool>;

// Callback type to emit code to set a value into an int register
using on_set_t = callback_t<const Reg &>;

// Callback type to emit code for a code block unrolled by some amount
using on_unrolled_t = callback_t<int>;

// Callback type to emit code for an indexed basic block
//
/// @details The first argument is the block index
using on_case_t = callback_t<int>;

// Data types

/// Constant value during code generation, stored as an imm12 field, a reg,
/// or (for an imm-kind value that overflows imm12) both at once
struct const_t {

    const_t() = default;
    const_t(const Reg &reg) : kind(kind_t::reg), reg_(reg) {}
    const_t(int imm) : kind(kind_t::imm), imm_(imm) {}
    /// Attaches `reg` only if `imm` actually overflows imm12
    const_t(int imm, const Reg &reg) : kind(kind_t::imm), imm_(imm) {
        if (needs_reg()) reg_ = reg;
    }

    bool is_reg() const { return kind == kind_t::reg; }
    bool is_imm() const { return kind == kind_t::imm; }
    Reg reg() const { return reg_; }
    const_t &reg(const Reg &r) {
        reg_ = r;
        return *this;
    }
    int imm() const { return imm_; }

    /// Whether this constant's value overflows imm12
    bool needs_reg() const { return is_imm() && !is_simm12(imm_); }

    /// Whether a register is available to address this constant with
    bool has_reg() const { return is_reg() || is_not_zero(reg_); }

    bool is_valid() const { return !needs_reg() || is_not_zero(reg_); }

    bool is_zero() const {
        static const auto rzero = Xbyak_riscv::zero;
        return (is_imm() && !imm_) || (is_reg() && reg_ == rzero);
    }

protected:
    enum class kind_t { reg, imm };

    kind_t kind = kind_t::imm;
    Reg reg_ {};
    int imm_ = 0;
};

/// Storage for a value (`reg`) plus an optional strategy to load it (`get`)
struct runtime_value_t {
    Reg reg {};
    on_set_t get {};

    runtime_value_t() = default;
    runtime_value_t(const Reg &r) : reg(r) {}
    runtime_value_t(const Reg &r, on_set_t get) : reg(r), get(std::move(get)) {}

    /// Accepts anything convertible to `on_set_t` (a raw lambda included)
    ///
    /// @note Templated so a lambda converts to `runtime_value_t` in a single
    ///     step; a plain `on_set_t` parameter would need two chained
    ///     conversions, which C++ disallows at the call site.
    /// @note Constrained to types `on_set_t` builds from, otherwise this
    ///     would out-match the dedicated `Reg`/copy constructors below.
    template <typename F,
            typename = typename std::enable_if<
                    std::is_constructible<on_set_t, F>::value>::type>
    runtime_value_t(F &&f) : get(std::forward<F>(f)) {}

    bool is_eager() const { return get == nullptr; }
    bool is_lazy() const { return get != nullptr; }

    /// Whether `reg` still needs to be materialized (e.g. via
    /// `register_pool_t::new_runtime_value`)
    bool need_reg() const { return !is_not_zero(reg); }

    bool is_valid() const { return !need_reg(); }
};

// Architectural Collection Types

/// Non-owning 2D view into a contiguous slice of a collection of T
template <typename T>
struct block_t {
    block_t() = default;
    block_t(const T *data, int cols) : data_(data), rows_(1), cols_(cols) {}
    block_t(const T *data, int rows, int cols)
        : data_(data), rows_(rows), cols_(cols) {}

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int size() const { return rows_ * cols_; }

    T operator()(int i) const { return data_[i % cols_]; }
    T operator()(int row, int col) const {
        return data_[(row % rows_) * cols_ + (col % cols_)];
    }
    T operator[](int i) const { return data_[i % size()]; }

    const T *begin() const { return data_; }
    const T *end() const { return data_ + size(); }

    /// Repurposes the resources within the allocated block to a new shape
    ///
    /// @pre The new shape must have the same, or smaller total size
    block_t reshaped(int new_rows, int new_cols) const {
        if (new_rows * new_cols > size()) return {};
        return block_t(data_, new_rows, new_cols);
    }

private:
    const T *data_ = nullptr;
    int rows_ = 0;
    int cols_ = 0;
};

using x_block_t = block_t<Reg>;
using f_block_t = block_t<FReg>;
#if XBYAK_RISCV_V
using v_block_t = block_t<VReg>;
#endif

// Control Flow Types

/// Components describing a branching instruction
struct branch_t {
    /// Branch condition types
    enum class cond_t { eq, ne, lt, le, gt, ge };

    /// Distance based on the addressing range of control flow instructions
    enum class distance_t {
        short_, // 12-bit simm (branch)
        medium // 20-bit simm (jump)
    };

    Reg lhs;
    runtime_value_t rhs;
    cond_t cond;
    distance_t dist;

    branch_t() = default;
    branch_t(const Reg &lhs, runtime_value_t rhs, cond_t cond,
            distance_t dist = distance_t::short_)
        : lhs(lhs), rhs(std::move(rhs)), cond(cond), dist(dist) {}
    branch_t(const Reg &lhs, cond_t cond, distance_t dist = distance_t::short_)
        : lhs(lhs), rhs(Xbyak_riscv::zero), cond(cond), dist(dist) {}

    static branch_t eq(const Reg &lhs, runtime_value_t rhs,
            distance_t dist = distance_t::short_) {
        return branch_t(lhs, std::move(rhs), cond_t::eq, dist);
    }

    static branch_t ne(const Reg &lhs, runtime_value_t rhs,
            distance_t dist = distance_t::short_) {
        return branch_t(lhs, std::move(rhs), cond_t::ne, dist);
    }

    static branch_t lt(const Reg &lhs, runtime_value_t rhs,
            distance_t dist = distance_t::short_) {
        return branch_t(lhs, std::move(rhs), cond_t::lt, dist);
    }

    static branch_t le(const Reg &lhs, runtime_value_t rhs,
            distance_t dist = distance_t::short_) {
        return branch_t(lhs, std::move(rhs), cond_t::le, dist);
    }

    static branch_t gt(const Reg &lhs, runtime_value_t rhs,
            distance_t dist = distance_t::short_) {
        return branch_t(lhs, std::move(rhs), cond_t::gt, dist);
    }

    static branch_t ge(const Reg &lhs, runtime_value_t rhs,
            distance_t dist = distance_t::short_) {
        return branch_t(lhs, std::move(rhs), cond_t::ge, dist);
    }

    static branch_t eqz(const Reg &lhs, distance_t dist = distance_t::short_) {
        return branch_t(lhs, cond_t::eq, dist);
    }

    static branch_t nez(const Reg &lhs, distance_t dist = distance_t::short_) {
        return branch_t(lhs, cond_t::ne, dist);
    }

    static branch_t ltz(const Reg &lhs, distance_t dist = distance_t::short_) {
        return branch_t(lhs, cond_t::lt, dist);
    }

    static branch_t lez(const Reg &lhs, distance_t dist = distance_t::short_) {
        return branch_t(lhs, cond_t::le, dist);
    }

    static branch_t gtz(const Reg &lhs, distance_t dist = distance_t::short_) {
        return branch_t(lhs, cond_t::gt, dist);
    }

    static branch_t gez(const Reg &lhs, distance_t dist = distance_t::short_) {
        return branch_t(lhs, cond_t::ge, dist);
    }
};

/// A small, ordered set of loop-unrolling factors
struct unroll_t {
    static constexpr int MAX_FACTORS = 4;

    int nfactors = 0;
    std::array<int, MAX_FACTORS> factors {};

    unroll_t() = default;
    unroll_t(int ur) : nfactors(1), factors {ur} {}
    unroll_t(std::initializer_list<int> urs) {
        for (int u : urs)
            if (nfactors < MAX_FACTORS) factors[nfactors++] = u;
    }

    template <typename It>
    unroll_t(It s, It e) {
        for (auto it = s; it != e && nfactors < MAX_FACTORS; ++it)
            factors[nfactors++] = *it;
    }

    /// nfactors in [1, MAX_FACTORS] and every factor >= 1
    bool is_valid() const {
        if (nfactors <= 0 || nfactors > MAX_FACTORS) return false;
        for (int i = 0; i < nfactors; ++i)
            if (factors[i] < 1) return false;
        return true;
    }
};

/// Whether `c` takes `const_folding_t::round_down`'s fast (shift-based)
/// path, which can alias `rd`/`rs1` and so needs no distinct tmp register
///
/// @details `1` never invokes `round_down` at all; any other power of two
///     takes the shift fast path. Anything else falls back to `rem`, which
///     needs `rd`/`rs1` to differ.
inline bool may_fast_div(int c) {
    return c == 1 || is_pow2(c);
}

/// Checks whether any unroll factor in `[s, e)` needs a tmp register
/// distinct from `limit` (i.e. isn't handled by `may_fast_div`)
///
/// @details A lazy `loop_t` aliases `tmp` with `limit`, which is only safe
///     if every factor allows the fast path
template <typename It>
inline bool needs_distinct_tmp(It s, It e) {
    for (auto it = s; it != e; ++it)
        if (!may_fast_div(*it)) return true;
    return false;
}

inline bool needs_distinct_tmp(int ur) {
    return needs_distinct_tmp(&ur, &ur + 1);
}

inline bool needs_distinct_tmp(const unroll_t &u) {
    return needs_distinct_tmp(
            u.factors.begin(), u.factors.begin() + u.nfactors);
}

/// Components for a control structure for implementating an unrolled loop
struct loop_t {

    /// Dispatch mode for this loop structure
    enum class mode_t { literal, switch_, unroll, unroll_and_switch };

    // Loop implementation kind
    mode_t mode = mode_t::unroll;

    // `lhs` is the iteration counter (unroll/unroll_and_switch only); `rhs`
    // is the mode-relevant runtime value — the dispatch id for switch_, the
    // iteration limit for unroll/unroll_and_switch (mutually exclusive by
    // mode, so they share this one `runtime_value_t`). `cond` is always
    // `lt`, the only condition `unrolled_loop()` implements.
    branch_t branch {Reg {}, runtime_value_t {}, branch_t::cond_t::lt};

private:
    unroll_t unroll_factors_;
    Reg tmp_ {};

public:
    Reg iter() const { return branch.lhs; }
    Reg tmp() const { return tmp_; }
    Reg limit() const { return value().reg; }
    Reg id() const { return value().reg; }

    /// The mode-relevant runtime value backing `branch.rhs`; a mode-agnostic
    /// accessor so callers (e.g. `register_pool_t::new_loop`) don't need to
    /// reach into `branch.rhs` directly
    const runtime_value_t &value() const { return branch.rhs; }

    loop_t &value(runtime_value_t v) {
        branch.rhs = std::move(v);
        return *this;
    }

    unroll_t &unrolling() { return unroll_factors_; }
    const unroll_t &unrolling() const { return unroll_factors_; }

    loop_t &unrolling(unroll_t u) {
        std::sort(u.factors.begin(), u.factors.begin() + u.nfactors,
                std::greater<int>());
        unroll_factors_ = std::move(u);
        return *this;
    }

    /// Sugar over `unrolling(unroll_t(n))` for `switch_`'s dispatch count
    loop_t &cases(int n) { return unrolling(unroll_t(n)); }

    loop_t &iter(const Reg &r) {
        branch.lhs = r;
        return *this;
    }
    loop_t &tmp(const Reg &r) {
        tmp_ = r;
        return *this;
    }
    loop_t &distance(branch_t::distance_t d) {
        branch.dist = d;
        return *this;
    }
    /// Sugar over `value(v)` for unroll/unroll_and_switch's iteration limit
    loop_t &limit(runtime_value_t v) { return value(std::move(v)); }
    /// Sugar over `value(v)` for switch_'s dispatch id
    loop_t &id(runtime_value_t v) { return value(std::move(v)); }

    /// Checks whether this object contains a callback to initialize `value()`
    bool is_lazy() const {
        return mode != mode_t::literal && value().is_lazy();
    }

    /// Checks whether this object expresses a loop with a pre-loaded value
    bool is_eager() const { return !is_lazy(); }

    bool need_unrolling() const { return !unrolling().is_valid(); }

    bool need_id() const {
        return mode == mode_t::switch_ && !is_not_zero(id());
    }

    bool need_limit() const {
        return (mode == mode_t::unroll || mode == mode_t::unroll_and_switch)
                && !is_not_zero(limit());
    }

    bool needs_iter() const {
        return (mode == mode_t::unroll || mode == mode_t::unroll_and_switch)
                && !is_not_zero(iter());
    }

    bool needs_tmp() const {
        return mode != mode_t::literal && !is_not_zero(tmp_);
    }

    /// Checks if the loop is valid for its dispatch mode
    bool is_valid() const {
        if (need_unrolling()) return false;
        switch (mode) {
            case mode_t::literal: return true;
            case mode_t::switch_:
                return unrolling().factors[0] <= 16 && !need_id()
                        && !needs_tmp() && id() != tmp();
            case mode_t::unroll: return is_core_valid_();
            case mode_t::unroll_and_switch:
                return is_core_valid_() && unrolling().factors[0] <= 17;
        }
        return false;
    }

    /// Loop that is always fully-unrolled by a literal factor
    static loop_t literal() {
        loop_t p;
        p.mode = mode_t::literal;
        return p;
    }

    /// Loop fully-unrolled by a runtime factor between [1,n]
    static loop_t switch_(unroll_t u = unroll_t()) {
        loop_t p;
        p.mode = mode_t::switch_;
        if (u.is_valid()) p.unrolling(std::move(u));
        return p;
    }

    /// Single (or sequence of) unrolled `while_` loop(s)
    static loop_t unroll(unroll_t u = unroll_t()) {
        loop_t p;
        p.mode = mode_t::unroll;
        if (u.is_valid()) p.unrolling(std::move(u));
        return p;
    }

    /// Main unrolled loop(s) with a tail switch_
    static loop_t unroll_and_switch(unroll_t u = unroll_t()) {
        loop_t p;
        p.mode = mode_t::unroll_and_switch;
        if (u.is_valid()) p.unrolling(std::move(u));
        return p;
    }

private:
    /// Checks the core loop-control registers and unroll factors
    bool is_core_valid_() const {
        if (needs_iter() || need_limit() || needs_tmp()) return false;
        if (!is_writeable(iter())) return false;
        if (iter() == limit() || tmp() == iter()) return false;
        if (is_eager() && tmp() == limit()) return false;
        if (is_lazy() && tmp() == limit() && needs_distinct_tmp(unrolling()))
            return false;
        return true;
    }
};

} // namespace rvjit
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
