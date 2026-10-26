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

#ifndef RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_ARITHMETIC_MUL_HPP
#define RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_ARITHMETIC_MUL_HPP

#include "rvjit/implementation/rv64/isa/operation/arithmetic/mul.hpp"
#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/operation/arithmetic.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace rvv {
namespace arithmetic {

/// General multiply instruction, extending the RV64 scalar one with vectors
///
/// @details Dispatches to `vmul` or `vwmul{,u,su}` on int vector forms, and
///     `vfmul` or `vfwmul` on float vector forms, from uniform or widening
///     shapes; leaves a scalar `rd` to `rv64::arithmetic::mul_t`, under its conditions on integers
///
/// @note The scalar operand of a vector form, if any, is `rs1`
///
/// @pre The caller must issue `vsetvli` prior to calling a vector form of
///     this expression, at the SEW of `rs1`
struct mul_t : public rv64::arithmetic::mul_t {
    static constexpr const char *NAME = "rvjit::rvv::arithmetic";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit mul_t(rvjit::emitter_t e) : rv64::arithmetic::mul_t(e), e_(e) {}

    bool is_valid(const rvjit::arithmetic::op_t &op) const override {
        if (!op.rd.is_vector()) return rv64::arithmetic::mul_t::is_valid(op);
        return select_(op) != sel_t::none;
    }

    void emit(const rvjit::arithmetic::op_t &op,
            const rvjit::arithmetic::args_t &args) const override {
        if (!op.rd.is_vector()) {
            rv64::arithmetic::mul_t::emit(op, args);
            return;
        }
        CHECK_OR_RET(
                op.is_ternary(), , "expected a ternary operand descriptor");
        CHECK_OR_RET(all_valid(op.rd, op.rs1, op.rs2) && args.named(3), ,
                "invalid operand descriptor");
        const sel_t sel = select_(op);
        CHECK_OR_RET(sel != sel_t::none, , "invalid descriptor");

        const VReg vd(args.rd), vs2(args.rs2);
        const VM vm = rv64::isa_traits_base_t::vm(op.predicate);
        if (op.rs1.is_scalar() && op.rs1.is_int()) {
            const Reg rs1(args.rs1);
            switch (sel) {
                case sel_t::vmul: e_->vmul_vx(vd, vs2, rs1, vm); break;
                case sel_t::vwmul: e_->vwmul_vx(vd, vs2, rs1, vm); break;
                case sel_t::vwmulu: e_->vwmulu_vx(vd, vs2, rs1, vm); break;
                case sel_t::vwmulsu: e_->vwmulsu_vx(vd, vs2, rs1, vm); break;
                default: return;
            }
            return;
        }
        if (op.rs1.is_scalar()) {
            const FReg fs1(args.rs1);
            switch (sel) {
                case sel_t::vfmul: e_->vfmul_vf(vd, vs2, fs1, vm); break;
                case sel_t::vfwmul: e_->vfwmul_vf(vd, vs2, fs1, vm); break;
                default: return;
            }
            return;
        }
        const VReg vs1(args.rs1);
        switch (sel) {
            case sel_t::vmul: e_->vmul_vv(vd, vs2, vs1, vm); break;
            case sel_t::vwmul: e_->vwmul_vv(vd, vs2, vs1, vm); break;
            case sel_t::vwmulu: e_->vwmulu_vv(vd, vs2, vs1, vm); break;
            case sel_t::vwmulsu: e_->vwmulsu_vv(vd, vs2, vs1, vm); break;
            case sel_t::vwmulus: e_->vwmulsu_vv(vd, vs1, vs2, vm); break;
            case sel_t::vfmul: e_->vfmul_vv(vd, vs2, vs1, vm); break;
            case sel_t::vfwmul: e_->vfwmul_vv(vd, vs2, vs1, vm); break;
            default: return;
        }
    }

private:
    rv64::emitter_t e_;

    // `vwmulus` is `vwmulsu` with its sources swapped
    enum class sel_t {
        none,
        vmul,
        vwmul,
        vwmulu,
        vwmulsu,
        vwmulus,
        vfmul,
        vfwmul
    };

    // `vwmulsu` multiplies a signed `vs2` by an unsigned `vs1`; a signed
    // scalar times an unsigned vector has no instruction
    static sel_t select_int_(const rvjit::arithmetic::op_t &op, bool vx) {
        if (op.is_uniform()) {
            const bool same
                    = op.rd.type == op.rs1.type && op.rd.type == op.rs2.type;
            return same ? sel_t::vmul : sel_t::none;
        }
        if (!op.is_widening()) return sel_t::none;
        const bool s1 = op.rs1.is_signed();
        const bool s2 = op.rs2.is_signed();
        if (s1 && s2) return sel_t::vwmul;
        if (!s1 && !s2) return sel_t::vwmulu;
        if (s2) return sel_t::vwmulsu;
        return vx ? sel_t::none : sel_t::vwmulus;
    }

    static sel_t select_float_(const rvjit::arithmetic::op_t &op) {
        const rvjit::type_t t = op.rs1.type;
        if (op.rs2.type != t) return sel_t::none;
        const bool narrow = t == rvjit::type_t::f16 || t == rvjit::type_t::f32;
        if (op.is_uniform() && op.rd.type == t
                && (narrow || t == rvjit::type_t::f64))
            return sel_t::vfmul;
        if (op.is_widening() && narrow && op.rd.type == rvjit::natural_wide(t))
            return sel_t::vfwmul;
        return sel_t::none;
    }

    static sel_t select_(const rvjit::arithmetic::op_t &op) {
        if (!op.is_ternary() || !op.rd.is_vector() || !op.rs2.is_vector())
            return sel_t::none;
        if (!op.rd.is_defined() || !op.rs1.is_defined() || !op.rs2.is_defined())
            return sel_t::none;
        if (!op.rd.has_valid_width() || !op.rs1.has_valid_width()
                || !op.rs2.has_valid_width())
            return sel_t::none;
        const bool all_int
                = op.rd.is_int() && op.rs1.is_int() && op.rs2.is_int();
        const bool all_float
                = op.rd.is_float() && op.rs1.is_float() && op.rs2.is_float();
        if (all_int) return select_int_(op, op.rs1.is_scalar());
        return all_float ? select_float_(op) : sel_t::none;
    }
};

} // namespace arithmetic
} // namespace rvv
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
