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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_ARITHMETIC_DIV_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_ARITHMETIC_DIV_HPP

#include "rvjit/implementation/rv64/isa/operation/arithmetic/uniform.hpp"
#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operation/arithmetic.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace rv64 {
namespace arithmetic {

/// General scalar divide instruction, `rd = rs2 / rs1`
///
/// @details Emits `divw` on s32, `div` on the other signed integers, `divu`
///     on the unsigned ones, or `fdiv.{h,s,d}` on floats, from uniform scalar
///     shapes only
///
/// @note An integer quotient is extended like its operands, except two cases
///     exact only in the low `width` bits of `rd`: the minimum s8 or s16
///     divided by -1, and a division by zero (all ones, the ISA's result);
///     dividing at the next wider type, after `conversion_t::extend`, avoids
///     the first
///
/// @pre Each integer operand holds its value extended to 64 bits by its
///     type's signedness, as the scalar loads and `conversion_t::extend`
///     leave it
struct div_t : public rvjit::arithmetic::expression_impl_t {
    static constexpr const char *NAME = "rvjit::rv64::arithmetic";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit div_t(rvjit::emitter_t e) : e_(e) {}

    /// Whether `op` maps to a supported instruction
    ///
    /// @param op The operation checked
    bool is_valid(const rvjit::arithmetic::op_t &op) const override {
        return is_uniform_scalar(op);
    }

    void emit(const rvjit::arithmetic::op_t &op,
            const rvjit::arithmetic::args_t &args) const override {
        CHECK_OR_RET(args.named(3), , "invalid operand descriptor");
        CHECK_OR_RET(div_t::is_valid(op), , "invalid descriptor");

        if (op.rd.is_int()) {
            const Reg rd(args.rd), divisor(args.rs1), dividend(args.rs2);
            if (!op.rd.is_signed())
                e_->divu(rd, dividend, divisor);
            else if (op.rd.width == 4)
                e_->divw(rd, dividend, divisor);
            else
                e_->div(rd, dividend, divisor);
            return;
        }
        const FReg fd(args.rd), divisor(args.rs1), dividend(args.rs2);
        switch (op.rd.width) {
            case 2: e_->fdiv_h(fd, dividend, divisor); break;
            case 4: e_->fdiv_s(fd, dividend, divisor); break;
            default: e_->fdiv_d(fd, dividend, divisor); break;
        }
    }

private:
    emitter_t e_;
};

} // namespace arithmetic
} // namespace rv64
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
