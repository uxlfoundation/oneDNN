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

#ifndef CPU_RV64_RVJIT_RVJIT_CONTROL_FLOW_HPP
#define CPU_RV64_RVJIT_RVJIT_CONTROL_FLOW_HPP

#include "cpu/rv64/rvjit/rvjit_const_folding.hpp"
#include "cpu/rv64/rvjit/rvjit_types.hpp"

#if defined(RVJIT_DEBUG)
#include "common/verbose.hpp"
#define DEBUg(...) \
    do { \
        if (get_verbose(verbose_t::debuginfo) > 1) { __VA_ARGS__ } \
    } while (0)
#else
#define DEBUg(...)
#endif
#define DEBUG(...) DEBUg(__VA_ARGS__)

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace rvjit {

/// Component expressing common control flow patterns.
class control_flow_t {
public:
    control_flow_t(emitter_t e, const const_folding_t &arith)
        : e_(e), arith_(arith) {}

    /// If-then pattern emitter
    ///
    /// @details Preserves the `lhs` and `rhs` branch registers. Reloads
    ///     `branch.rhs` via its own loader first if lazy.
    void if_(const branch_t &branch, const on_enter_t &body) {
        reload_if_lazy_(branch.rhs);
        Label skip;
        emit_branch_(branch, skip);
        body();
        e_->L(skip);
    }

    /// If-then-else pattern emitter
    ///
    /// @details Preserves the `lhs` and `rhs` branch registers. Reloads
    ///     `branch.rhs` via its own loader first if lazy.
    void if_(const branch_t &branch, const on_cond_t &cb) {
        if (!cb) return;
        reload_if_lazy_(branch.rhs);
        Label skip, done;
        emit_branch_(branch, skip);
        cb(true);
        e_->j_(done);
        e_->L(skip);
        cb(false);
        e_->L(done);
    }

    /// While loop pattern emitter
    ///
    /// @pre Caller must ensure `iter` progress in `cb`
    ///
    /// @details Preserves the `iter` and `limit` registers. Reloads
    ///     `branch.rhs` via its own loader once, before the loop, if lazy —
    ///     not on every iteration.
    void while_(const branch_t &branch, const on_enter_t &cb) {
        if (!cb) return;
        reload_if_lazy_(branch.rhs);
        Label head, end;
        e_->L(head);
        emit_branch_(branch, end);
        cb();
        e_->j_(head);
        e_->L(end);
    }

    /// While loop with step pattern emitter; advances `iter` by `step`
    /// before each call to `cb`
    ///
    /// @pre Caller must preserve `iter` if clobbered in `cb`
    /// @note Preserves the `end` register; clobbers `iter`. Reloads `b.rhs`
    ///     via its own loader once, before the loop, if lazy — not on every
    ///     iteration.
    void while_(const branch_t &b, const const_t step, const on_enter_t &cb) {
        if (!cb) return;
        reload_if_lazy_(b.rhs);
        Label head, end;
        e_->L(head);
        emit_branch_(b, end);
        arith_.add_const(b.lhs, b.lhs, step);
        cb();
        e_->j_(head);
        e_->L(end);
    }

    /// Switch-case emitter covering runtime `id` values in [1, N] (N <= 16)
    ///
    /// @pre Caller must ensure `id` fits that range
    /// @note Clobbers the `id` and `t` registers. Reloads `id` via its own
    ///     loader if lazy.
    void switch_case(int N, const runtime_value_t &id, const Reg &t,
            const on_case_t &cb) {
        static constexpr int MAXN = 16;
        if (N <= 0 || N > MAXN || id.reg == Xbyak_riscv::x0) {
            DEBUG({
                verbose_printf(verbose_t::debuginfo,
                        "rvjit: Failed control_flow_t::switch_case() due to "
                        "an invalid N or id\n");
            });
            return;
        }
        if (id.is_lazy()) id.get(id.reg);

        Label end;
        Label cases[MAXN];
        const Reg &address = t;
        const Reg &entry = id.reg;

        // Byte offset from auipc to the selected case entry (jump table
        // starts 3 instructions after auipc; entry i maps to case N-i):
        // offset = ((3 + N) * 4) - (id * 4)
        e_->li(address, (3 + N) * sizeof(uint32_t));
        e_->slli(entry, id.reg, 2);
        e_->sub(entry, address, entry);

        e_->auipc(address, 0);
        e_->add(address, address, entry);
        e_->jr(address);

        for (int i = N; i > 0; --i)
            e_->j_(cases[i - 1]);

        for (int i = N; i > 0; --i) {
            e_->L(cases[i - 1]);
            cb(i);
            if (i > 1) e_->j_(end);
        }

        e_->L(end);
    }

    /// Emits an unrolled loop / dispatch structure around a parametrized cb
    ///
    /// @pre Caller must ensure `loop` passes the mode-specific validity check
    void unrolled_loop(const loop_t &loop, const on_case_t &cb) {
        if (!loop.is_valid()) {
            DEBUG({
                verbose_printf(verbose_t::debuginfo,
                        "rvjit: control_flow_t::unrolled_loop() failed due to "
                        "invalid loop\n");
            });
            return;
        }
        using m = loop_t::mode_t;
        switch (loop.mode) {
            case m::literal: cb(loop.unrolling().factors[0]); break;
            case m::switch_:
                switch_case(loop.unrolling().factors[0], loop.value(),
                        loop.tmp(), cb);
                break;
            case m::unroll:
            case m::unroll_and_switch:
                const Reg &iter = loop.iter();
                const Reg &limit = loop.limit();
                const Reg &tail = loop.tmp();
                const Reg &tmp = loop.tmp();
                const auto &dist = loop.branch.dist;

                // A tmp/limit alias needs a fresh reload before every use past
                // the first, since round_down's aliased path overwrites
                // `limit` in place (its distinct-tmp path preserves it).
                const bool lazy_alias = loop.is_lazy() && tmp == limit;

                // Main collection of loops
                e_->mv(iter, Xbyak_riscv::zero);
                for (int i = 0; i < loop.unrolling().nfactors; ++i) {
                    const int ur = loop.unrolling().factors[i];
                    if (ur == 1) {
                        // while_ reloads loop.branch.rhs itself if lazy
                        while_(loop.branch, ur, [&] { cb(ur); });
                    } else if (i == 0) {
                        // Feeds round_down's input, not a branch comparison,
                        // so while_'s own reload doesn't cover this
                        if (loop.is_lazy()) loop.value().get(limit);
                        arith_.round_down(tmp, limit, ur);
                        while_(branch_t::lt(iter, tmp, dist), ur,
                                [&] { cb(ur); });
                    } else {
                        // Stages past the first must round down what earlier
                        // stages left behind (limit - iter)
                        if (lazy_alias) loop.value().get(limit);
                        e_->sub(limit, limit, iter);
                        arith_.round_down(tmp, limit, ur);
                        if (tmp != limit) e_->add(limit, limit, iter);
                        e_->add(tmp, tmp, iter);
                        while_(branch_t::lt(iter, tmp, dist), ur,
                                [&] { cb(ur); });
                    }
                }

                if (loop.mode == m::unroll) break;

                // Number of tail cases to generate
                const unroll_t &unrolling = loop.unrolling();
                const int cases = unrolling.factors[unrolling.nfactors - 1] - 1;

                // Feeds the tail-count subtraction, not switch_case's id
                // directly, so switch_case's own reload doesn't cover this
                if (lazy_alias) loop.value().get(limit);
                e_->sub(tail, limit, iter);
                if_(branch_t::nez(tail, branch_t::distance_t::medium),
                        [&] { switch_case(cases, tail, iter, cb); });
                break;
        }
    }

private:
    emitter_t e_;
    const const_folding_t &arith_;

    /// Invokes `rv`'s loader if lazy
    ///
    /// @note Called once by `if_`/`while_` before any label/loop they build,
    ///     not from `emit_branch_` — `emit_branch_`'s emitted instructions
    ///     run on every iteration of a `while_` loop, so reloading there
    ///     would reload on every iteration instead of once
    static void reload_if_lazy_(const runtime_value_t &rv) {
        if (rv.is_lazy()) rv.get(rv.reg);
    }

    /// @pre Caller has already reloaded `b.rhs` if lazy (see `reload_if_lazy_`)
    void emit_branch_(const branch_t &b, const Label &not_taken) {
        const Reg &lhs = b.lhs;
        const Reg &rhs = b.rhs.reg;
        const branch_t::cond_t &c = b.cond;
        const branch_t::distance_t &d = b.dist;

        using cg_t = codegen_t;
        using fn_t = void (cg_t::*)(const Reg &, const Reg &, const Label &);

        // Organized in the same order elements are declared
        static constexpr fn_t branch_table[] = {
                &cg_t::beq,
                &cg_t::bne,
                &cg_t::blt,
                &cg_t::ble,
                &cg_t::bgt,
                &cg_t::bge,
        };

        // Emit a condition-agnostic branch instruction
        cg_t &cg = e_.cg();
        const auto emit = [&](const branch_t::cond_t &cond, const Label &lbl) {
            (cg.*branch_table[static_cast<int>(cond)])(lhs, rhs, lbl);
        };

        // Get the reverse branching condition
        const auto inverse = [&]() {
            switch (c) {
                case branch_t::cond_t::eq: return branch_t::cond_t::ne;
                case branch_t::cond_t::ne: return branch_t::cond_t::eq;
                case branch_t::cond_t::lt: return branch_t::cond_t::ge;
                case branch_t::cond_t::ge: return branch_t::cond_t::lt;
                case branch_t::cond_t::le: return branch_t::cond_t::gt;
                case branch_t::cond_t::gt: return branch_t::cond_t::le;
                default: return branch_t::cond_t::ne;
            }
        };

        if (d == branch_t::distance_t::short_) {
            // emits the inverse condition as a single branch instruction
            emit(inverse(), not_taken);
        } else {
            // Use jump when "taken" path is further than the 4 KB branch range
            Label taken;
            emit(c, taken);
            e_->j_(not_taken);
            e_->L(taken);
        }
    }
};

} // namespace rvjit
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#undef DEBUg
#undef DEBUG

#endif
