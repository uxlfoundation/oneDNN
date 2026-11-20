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

#ifndef RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_CONVERSION_CAST_HPP
#define RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_CONVERSION_CAST_HPP

#include "rvjit/implementation/rv64/isa/operation/conversion/cast.hpp"
#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/operation/conversion.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace rvv {
namespace conversion {

/// Conversion between an integer and a float type, extending the RV64 scalar
/// cast with vector-to-vector conversions of one width
///
/// @details Dispatches vectors to `vfcvt.f.x[u].v` and `vfcvt.x[u].f.v`, and
///     scalars to `rv64::conversion::cast_t`; both round as the dynamic
///     rounding mode says
///
/// @note Vectors convert f16, f32 or f64 at one width; scalars follow
///     `rv64::conversion::cast_t`; a scalar and a vector operand are never
///     mixed
///
/// @pre On vector forms, the caller must issue `vsetvli` prior to calling
///     this expression with the operands' SEW and LMUL
/// @pre On vector forms, `dst` may name `src`, casting in place
struct cast_t : public rv64::conversion::cast_t {
    static constexpr const char *NAME = "rvjit::rvv::conversion::cast";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit cast_t(rvjit::emitter_t e) : rv64::conversion::cast_t(e), e_(e) {}

    bool is_valid(const rvjit::conversion::op_t &op) const override {
        return vector_valid_(op) || rv64::conversion::cast_t::is_valid(op);
    }

    void emit(const rvjit::conversion::op_t &op,
            const rvjit::conversion::args_t &args) const override {
        if (!op.dst.is_vector() || !op.src.is_vector()) {
            rv64::conversion::cast_t::emit(op, args);
            return;
        }
        CHECK_OR_RET(vector_valid_(op), , "invalid descriptor");
        CHECK_OR_RET(all_valid(op.dst, op.src) && args.named(), ,
                "invalid operand descriptor");

        const VReg vd(args.dst), vs2(args.src);
        const VM vm = rv64::isa_traits_base_t::vm(op.predicate);
        if (op.dst.is_float()) {
            if (op.src.is_signed())
                e_->vfcvt_f_x_v(vd, vs2, vm);
            else
                e_->vfcvt_f_xu_v(vd, vs2, vm);
        } else {
            if (op.dst.is_signed())
                e_->vfcvt_x_f_v(vd, vs2, vm);
            else
                e_->vfcvt_xu_f_v(vd, vs2, vm);
        }
    }

private:
    rv64::emitter_t e_;

    static bool vector_valid_(const rvjit::conversion::op_t &op) {
        if (!op.is_casting() || !op.dst.has_valid_width()
                || !op.src.has_valid_width())
            return false;
        if (!op.dst.is_vector() || !op.src.is_vector()) return false;
        const rvjit::type_t f = op.dst.is_float() ? op.dst.type : op.src.type;
        const bool float_ok = f == rvjit::type_t::f16 || f == rvjit::type_t::f32
                || f == rvjit::type_t::f64;
        return op.dst.width == op.src.width && float_ok;
    }
};

} // namespace conversion
} // namespace rvv
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
