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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_CONVERSION_CAST_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_CONVERSION_CAST_HPP

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/operation/conversion.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace rv64 {
namespace conversion {

/// Scalar conversion between an integer and a float type
///
/// @details Dispatches to the `fcvt` of the f16, f32 or f64 and the 32- or
///     64-bit integer, rounding as the dynamic rounding mode says
///
/// @note Only scalar-to-scalar conversions are valid
struct cast_t : public rvjit::conversion::expression_impl_t {
    static constexpr const char *NAME = "rvjit::rv64::conversion::cast";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit cast_t(rvjit::emitter_t e) : e_(e) {}

    bool is_valid(const rvjit::conversion::op_t &op) const override {
        if (!op.is_casting() || !op.dst.has_valid_width()
                || !op.src.has_valid_width())
            return false;
        if (!op.dst.is_scalar() || !op.src.is_scalar()) return false;
        const rvjit::type_t f = op.dst.is_float() ? op.dst.type : op.src.type;
        const bool float_ok = f == rvjit::type_t::f16 || f == rvjit::type_t::f32
                || f == rvjit::type_t::f64;
        const int int_width = op.dst.is_int() ? op.dst.width : op.src.width;
        return float_ok && int_width >= 4;
    }

    void emit(const rvjit::conversion::op_t &op,
            const rvjit::conversion::args_t &args) const override {
        CHECK_OR_RET(cast_t::is_valid(op), , "invalid descriptor");
        CHECK_OR_RET(all_valid(op.dst, op.src) && args.named(), ,
                "invalid operand descriptor");

        const bool to_float = op.dst.is_float();
        const int fw = to_float ? op.dst.width : op.src.width;
        const bool wide = (to_float ? op.src.width : op.dst.width) == 8;
        const bool sgn = to_float ? op.src.is_signed() : op.dst.is_signed();
        if (to_float) {
            const FReg fd(args.dst);
            const Reg rs(args.src);
            if (fw == 2)
                wide ? (sgn ? e_->fcvt_h_l(fd, rs) : e_->fcvt_h_lu(fd, rs))
                     : (sgn ? e_->fcvt_h_w(fd, rs) : e_->fcvt_h_wu(fd, rs));
            else if (fw == 4)
                wide ? (sgn ? e_->fcvt_s_l(fd, rs) : e_->fcvt_s_lu(fd, rs))
                     : (sgn ? e_->fcvt_s_w(fd, rs) : e_->fcvt_s_wu(fd, rs));
            else
                wide ? (sgn ? e_->fcvt_d_l(fd, rs) : e_->fcvt_d_lu(fd, rs))
                     : (sgn ? e_->fcvt_d_w(fd, rs) : e_->fcvt_d_wu(fd, rs));
            return;
        }
        const Reg rd(args.dst);
        const FReg fs(args.src);
        if (fw == 2)
            wide ? (sgn ? e_->fcvt_l_h(rd, fs) : e_->fcvt_lu_h(rd, fs))
                 : (sgn ? e_->fcvt_w_h(rd, fs) : e_->fcvt_wu_h(rd, fs));
        else if (fw == 4)
            wide ? (sgn ? e_->fcvt_l_s(rd, fs) : e_->fcvt_lu_s(rd, fs))
                 : (sgn ? e_->fcvt_w_s(rd, fs) : e_->fcvt_wu_s(rd, fs));
        else
            wide ? (sgn ? e_->fcvt_l_d(rd, fs) : e_->fcvt_lu_d(rd, fs))
                 : (sgn ? e_->fcvt_w_d(rd, fs) : e_->fcvt_wu_d(rd, fs));
    }

private:
    rv64::emitter_t e_;
};

} // namespace conversion
} // namespace rv64
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
