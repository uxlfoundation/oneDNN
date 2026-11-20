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

#ifndef RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_ARITHMETIC_DIV_HPP
#define RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_ARITHMETIC_DIV_HPP

#include "rvjit/implementation/rv64/isa/operation/arithmetic/div.hpp"
#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/operation/arithmetic.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace rvv {
namespace arithmetic {

/// General divide instruction, `rd = rs2 / rs1`, extending the RV64 scalar
/// one with vectors
///
/// @details Dispatches to `vdiv`/`vdivu` on signed/unsigned int vector forms,
///     or `vfdiv` on float vector forms, from uniform shapes only; leaves a
///     scalar `rd` to `rv64::arithmetic::div_t`, under its conditions on integers
///
/// @note The scalar operand of a vector form, if any, is `rs1`, the divisor
///
/// @pre The caller must issue `vsetvli` prior to calling a vector form of
///     this expression
struct div_t : public rv64::arithmetic::div_t {
    static constexpr const char *NAME = "rvjit::rvv::arithmetic";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit div_t(rvjit::emitter_t e) : rv64::arithmetic::div_t(e), e_(e) {}

    bool is_valid(const rvjit::arithmetic::op_t &op) const override {
        if (!op.rd.is_vector()) return rv64::arithmetic::div_t::is_valid(op);
        if (!op.is_ternary() || !op.is_uniform()) return false;
        if (!op.rd.is_defined() || !op.rs1.is_defined() || !op.rs2.is_defined())
            return false;
        if (!op.rd.has_valid_width() || !op.rs2.is_vector()) return false;
        if (op.rd.type != op.rs1.type || op.rd.type != op.rs2.type)
            return false;
        if (op.rd.is_int()) return true;
        const rvjit::type_t t = op.rd.type;
        return t == rvjit::type_t::f16 || t == rvjit::type_t::f32
                || t == rvjit::type_t::f64;
    }

    void emit(const rvjit::arithmetic::op_t &op,
            const rvjit::arithmetic::args_t &args) const override {
        if (!op.rd.is_vector()) {
            rv64::arithmetic::div_t::emit(op, args);
            return;
        }
        CHECK_OR_RET(
                op.is_ternary(), , "expected a ternary operand descriptor");
        CHECK_OR_RET(all_valid(op.rd, op.rs1, op.rs2) && args.named(3), ,
                "invalid operand descriptor");
        CHECK_OR_RET(is_valid(op), , "invalid descriptor");

        const VReg vd(args.rd), vs2(args.rs2);
        const VM vm = rv64::isa_traits_base_t::vm(op.predicate);
        const bool sca = op.rs1.is_scalar();
        if (op.rd.is_float()) {
            if (sca)
                e_->vfdiv_vf(vd, vs2, FReg(args.rs1), vm);
            else
                e_->vfdiv_vv(vd, vs2, VReg(args.rs1), vm);
        } else if (op.rd.is_signed()) {
            if (sca)
                e_->vdiv_vx(vd, vs2, Reg(args.rs1), vm);
            else
                e_->vdiv_vv(vd, vs2, VReg(args.rs1), vm);
        } else {
            if (sca)
                e_->vdivu_vx(vd, vs2, Reg(args.rs1), vm);
            else
                e_->vdivu_vv(vd, vs2, VReg(args.rs1), vm);
        }
    }

private:
    rv64::emitter_t e_;
};

} // namespace arithmetic
} // namespace rvv
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
