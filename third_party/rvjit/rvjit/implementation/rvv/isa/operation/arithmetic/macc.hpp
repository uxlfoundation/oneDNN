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

#ifndef RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_ARITHMETIC_MACC_HPP
#define RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_ARITHMETIC_MACC_HPP

#include "rvjit/implementation/rv64/isa/operation/arithmetic/macc.hpp"
#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/operation/arithmetic.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace rvv {
namespace arithmetic {

/// General multiply-accumulate instruction, extending the RV64 scalar one
/// with vectors
///
/// @note Dispatches code emission based on the operands data object parameter
///
/// @details Dispatches to v{,w}macc{,u,su,us} on int vector forms
/// @details Dispatches to vfmacc/vfwmacc/vfwmaccbf16 on float vector forms
/// @details Leaves a scalar `rd` to `rv64::arithmetic::macc_t`
///
/// @pre The caller must issue `vsetvli` prior to calling a vector form of
///     this expression and the `vtype` must be configured using the inputs
///     SEW configuration
struct macc_t : public rv64::arithmetic::macc_t {
    static constexpr const char *NAME = "rvjit::rvv::arithmetic";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit macc_t(rvjit::emitter_t e) : rv64::arithmetic::macc_t(e), e_(e) {}

    bool is_valid(const rvjit::arithmetic::op_t &op) const override {
        if (!op.rd.is_vector()) return rv64::arithmetic::macc_t::is_valid(op);
        return select_(op) != sel_t::none;
    }

    void emit(const rvjit::arithmetic::op_t &op,
            const rvjit::arithmetic::args_t &args) const override {
        if (!op.rd.is_vector()) {
            rv64::arithmetic::macc_t::emit(op, args);
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
                case sel_t::vmacc: e_->vmacc_vx(vd, rs1, vs2, vm); break;
                case sel_t::vwmacc: e_->vwmacc_vx(vd, rs1, vs2, vm); break;
                case sel_t::vwmaccu: e_->vwmaccu_vx(vd, rs1, vs2, vm); break;
                case sel_t::vwmaccsu: e_->vwmaccsu_vx(vd, rs1, vs2, vm); break;
                case sel_t::vwmaccus: e_->vwmaccus_vx(vd, rs1, vs2, vm); break;
                default: return;
            }
            return;
        }
        if (op.rs1.is_scalar()) {
            const FReg fs1(args.rs1);
            switch (sel) {
                case sel_t::vfmacc: e_->vfmacc_vf(vd, fs1, vs2, vm); break;
                case sel_t::vfwmacc: e_->vfwmacc_vf(vd, fs1, vs2, vm); break;
                case sel_t::vfwmaccbf16:
                    e_->vfwmaccbf16_vf(vd, fs1, vs2, vm);
                    break;
                default: return;
            }
            return;
        }
        const VReg vs1(args.rs1);
        switch (sel) {
            case sel_t::vmacc: e_->vmacc_vv(vd, vs1, vs2, vm); break;
            case sel_t::vwmacc: e_->vwmacc_vv(vd, vs1, vs2, vm); break;
            case sel_t::vwmaccu: e_->vwmaccu_vv(vd, vs1, vs2, vm); break;
            case sel_t::vwmaccsu: e_->vwmaccsu_vv(vd, vs1, vs2, vm); break;
            case sel_t::vfmacc: e_->vfmacc_vv(vd, vs1, vs2, vm); break;
            case sel_t::vfwmacc: e_->vfwmacc_vv(vd, vs1, vs2, vm); break;
            case sel_t::vfwmaccbf16:
                e_->vfwmaccbf16_vv(vd, vs1, vs2, vm);
                break;
            default: return;
        }
    }

private:
    rv64::emitter_t e_;

    enum class sel_t {
        none,
        vmacc,
        vwmacc,
        vwmaccu,
        vwmaccsu,
        vwmaccus,
        vfmacc,
        vfwmacc,
        vfwmaccbf16
    };

    // The `us` combination has no `_vv` form, only `_vx`
    static sel_t select_int_(const rvjit::arithmetic::op_t &op, bool vx) {
        if (op.is_uniform()) return sel_t::vmacc;
        if (!op.is_widening()) return sel_t::none;
        const bool s1 = op.rs1.is_signed();
        const bool s2 = op.rs2.is_signed();
        if (s1 && s2) return sel_t::vwmacc;
        if (!s1 && !s2) return sel_t::vwmaccu;
        if (s1) return sel_t::vwmaccsu;
        return vx ? sel_t::vwmaccus : sel_t::none;
    }

    static sel_t select_float_(const rvjit::arithmetic::op_t &op) {
        const rvjit::type_t t = op.rs1.type;
        if (t != op.rs2.type) return sel_t::none;
        if (op.is_uniform()
                && (t == rvjit::type_t::f16 || t == rvjit::type_t::f32
                        || t == rvjit::type_t::f64))
            return sel_t::vfmacc;
        if (op.is_widening() && t == rvjit::type_t::bf16)
            return sel_t::vfwmaccbf16;
        if (op.is_widening()
                && (t == rvjit::type_t::f16 || t == rvjit::type_t::f32))
            return sel_t::vfwmacc;
        return sel_t::none;
    }

    static sel_t select_(const rvjit::arithmetic::op_t &op) {
        if (!op.is_ternary() || !op.rd.is_vector()) return sel_t::none;
        if (!op.rd.is_defined() || !op.rs1.is_defined() || !op.rs2.is_defined())
            return sel_t::none;
        if (!op.rd.has_valid_width() || !op.rs1.has_valid_width()
                || !op.rs2.has_valid_width())
            return sel_t::none;
        if (op.rs1.is_scalar()) {
            if (!op.rs1.is_int()) return select_float_(op);
            return op.rs2.is_int() ? select_int_(op, true) : sel_t::none;
        }
        if (!op.rs1.is_int() || !op.rs2.is_int()) return select_float_(op);
        return select_int_(op, false);
    }
};

} // namespace arithmetic
} // namespace rvv
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
