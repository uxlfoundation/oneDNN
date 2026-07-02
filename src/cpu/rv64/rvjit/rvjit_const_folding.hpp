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

#ifndef CPU_RV64_RVJIT_RVJIT_CONST_FOLDING_HPP
#define CPU_RV64_RVJIT_RVJIT_CONST_FOLDING_HPP

#include <iterator>
#include <initializer_list>

#include "cpu/rv64/rvjit/rvjit_types.hpp"
#include "cpu/rv64/rvjit/rvjit_utils.hpp"

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

/// Component for constant-folded arithmetic
struct const_folding_t {
    explicit const_folding_t(emitter_t e) : e_(e) {}

    /// Writes `c`'s value into its backing register, if it has one assigned
    const_t init_constant(const const_t &c) const {
        if (c.is_imm() && is_not_zero(c.reg())) e_->li(c.reg(), c.imm());
        return c;
    }

    /// rd = rs1 + c
    ///
    /// @note Preserves `rs1`; no-op if c is zero
    void add_const(const Reg &rd, const Reg &rs1, const const_t &c) const {
        if (!c.is_zero()) {
            if (c.has_reg())
                e_->add(rd, rs1, c.reg());
            else
                e_->addi(rd, rs1, c.imm());
        }
    }

    /// rd = floor(rs1 / c)
    ///
    /// @pre `rd` must not alias `rs1` unless c is a power of two
    /// @note Preserves `rs1` and `c`'s register contents
    void div_const(const Reg &rd, const Reg &rs1, const const_t &c) const {
        // Reject divison by zero
        if (c.is_zero()) {
            DEBUG({
                verbose_printf(verbose_t::debuginfo,
                        "rvjit: Failed const_folding_t::div_const (div by "
                        "zero)\n");
            });
            return;
        }

        // Constant from register
        if (c.has_reg()) {
            e_->div(rd, rs1, c.reg());
            return;
        }

        // Constant from immediate
        const auto imm = c.imm();
        const auto abs_imm = imm < 0 ? -imm : imm;

        if (is_pow2(abs_imm)) {
            // Quick division if power of two
            const int shammt = ilog2q(abs_imm);
            e_->srli(rd, rs1, shammt);
            if (imm < 0) e_->sub(rd, Xbyak_riscv::zero, rd);
        } else if (rd != rs1) {
            // Fallback to div if rd and rs1 are not aliased
            e_->li(rd, imm);
            e_->div(rd, rs1, rd);
            return;
        } else {
            // Illegal alias of rd and rs1
            DEBUG({
                verbose_printf(verbose_t::debuginfo,
                        "rvjit: Failed const_folding_t::div_const due to "
                        "rd alias\n");
            });
        }
    }

    /// rd = rs1 - (rs1 % c)
    ///
    /// @pre `rd` must not alias `rs1` unless c is a power of two
    /// @note Preserves `rs1` and `c`'s register contents
    void round_down(const Reg &rd, const Reg &rs1, const const_t &c) const {
        // Reject divison by zero
        if (c.is_zero()) {
            DEBUG({
                verbose_printf(verbose_t::debuginfo,
                        "rvjit: Failed const_folding_t::round_down due to "
                        "division by zero\n");
            });
            return;
        }

        // Constant from register
        if (c.has_reg()) {
            e_->div(rd, rs1, c.reg());
            e_->mul(rd, rd, c.reg());
            return;
        }

        // Constant from immediate
        const auto imm = c.imm();
        const auto abs_imm = imm < 0 ? -imm : imm;

        if (is_pow2(abs_imm)) {
            // Quick division and multiplication
            const int shammt = ilog2q(abs_imm);
            e_->srli(rd, rs1, shammt);
            e_->slli(rd, rd, shammt);
        } else if (rd != rs1) {
            // Default to rem
            e_->li(rd, imm);
            e_->rem(rd, rs1, rd);
            e_->sub(rd, rs1, rd);
        } else {
            // Illegal alias of rd and rs1
            DEBUG({
                verbose_printf(verbose_t::debuginfo,
                        "rvjit: Failed const_folding_t::round_down due to "
                        "illegal rd and rs1 alias\n");
            });
        }
    }

private:
    emitter_t e_;
};

} // namespace rvjit
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#undef DEBUg
#undef DEBUG

#endif
