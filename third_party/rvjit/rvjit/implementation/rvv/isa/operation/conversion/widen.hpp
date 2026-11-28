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

#ifndef RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_CONVERSION_WIDEN_HPP
#define RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_CONVERSION_WIDEN_HPP

#include "rvjit/implementation/rv64/isa/operation/conversion/widen.hpp"
#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/operation/conversion.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace rvv {
namespace conversion {

/// Conversion into a type twice as wide, extending the RV64 scalar integer
/// widening with vector-to-vector conversions
///
/// @details Dispatches to `vfwcvt_f_f_v` (f16 to f32, f32 to f64),
///     `vfwcvtbf16_f_f_v` (bf16 to f32), and `vwadd_vx` / `vwaddu_vx` of the
///     zero register on integers, which extend the sign or the zero bits
///
/// @note Scalar-to-scalar conversions are left to `rv64::conversion::widen_t`,
///     which widens integers only, under its conditions; mixed forms are
///     invalid
///
/// @pre The caller must issue `vsetvli` prior to calling a vector form of
///     this expression and the `vtype` must be configured using the source's
///     SEW and LMUL
/// @pre On vector forms, `dst` is not `src`: the source group is the
///     highest-numbered part of the destination group, or does not overlap it
struct widen_t : public rv64::conversion::widen_t {
    static constexpr const char *NAME = "rvjit::rvv::conversion::widen";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit widen_t(rvjit::emitter_t e)
        : rv64::conversion::widen_t(e), e_(e) {}

    bool is_valid(const rvjit::conversion::op_t &op) const override {
        if (op.dst.is_scalar() && op.src.is_scalar())
            return rv64::conversion::widen_t::is_valid(op);
        return select_(op) != sel_t::none;
    }

    void emit(const rvjit::conversion::op_t &op,
            const rvjit::conversion::args_t &args) const override {
        if (op.dst.is_scalar() && op.src.is_scalar()) {
            rv64::conversion::widen_t::emit(op, args);
            return;
        }
        const sel_t sel = select_(op);
        CHECK_OR_RET(sel != sel_t::none, , "invalid descriptor");
        CHECK_OR_RET(all_valid(op.dst, op.src) && args.named(), ,
                "invalid operand descriptor");

        const VReg vd(args.dst), vs2(args.src);
        const VM vm = rv64::isa_traits_base_t::vm(op.predicate);
        switch (sel) {
            case sel_t::vfwcvt: e_->vfwcvt_f_f_v(vd, vs2, vm); break;
            case sel_t::vfwcvtbf16: e_->vfwcvtbf16_f_f_v(vd, vs2, vm); break;
            case sel_t::vwadd: e_->vwadd_vx(vd, vs2, zero, vm); break;
            case sel_t::vwaddu: e_->vwaddu_vx(vd, vs2, zero, vm); break;
            default: break;
        }
    }

private:
    rv64::emitter_t e_;

    enum class sel_t { none, vfwcvt, vfwcvtbf16, vwadd, vwaddu };

    static sel_t select_(const rvjit::conversion::op_t &op) {
        if (!op.is_widening() || !op.dst.is_vector() || !op.src.is_vector())
            return sel_t::none;
        if (!op.dst.has_valid_width() || !op.src.has_valid_width())
            return sel_t::none;

        const rvjit::type_t d = op.dst.type, s = op.src.type;
        if (op.dst.is_float() && op.src.is_float()) {
            if (s == rvjit::type_t::f16 && d == rvjit::type_t::f32)
                return sel_t::vfwcvt;
            if (s == rvjit::type_t::f32 && d == rvjit::type_t::f64)
                return sel_t::vfwcvt;
            if (s == rvjit::type_t::bf16 && d == rvjit::type_t::f32)
                return sel_t::vfwcvtbf16;
            return sel_t::none;
        }
        if (!op.dst.is_int() || !op.src.is_int()) return sel_t::none;

        // A signed source needs a signed destination; an unsigned source
        // extends with zeros into either
        if (op.src.is_signed())
            return op.dst.is_signed() ? sel_t::vwadd : sel_t::none;
        return sel_t::vwaddu;
    }
};

} // namespace conversion
} // namespace rvv
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
