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

#ifndef RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_CONVERSION_EXTEND_HPP
#define RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_CONVERSION_EXTEND_HPP

#include "rvjit/implementation/rv64/isa/operation/conversion/extend.hpp"
#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/operation/conversion.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace rvv {
namespace conversion {

/// Integer extension into a type 2, 4 or 8 times as wide, extending the RV64
/// scalar extension with vector-to-vector conversions
///
/// @details Dispatches to `vsext.vf2/vf4/vf8` on a signed source and to
///     `vzext.vf2/vf4/vf8` on an unsigned one
///
/// @note Scalar-to-scalar conversions are left to `rv64::conversion::extend_t`,
///     under its conditions; mixed forms are invalid
///
/// @pre The caller must issue `vsetvli` prior to calling a vector form of
///     this expression and the `vtype` must be configured using the
///     destination's SEW and LMUL, unlike `widen_t`
/// @pre On vector forms, the source group is the highest-numbered part of the
///     destination group, or does not overlap it
struct extend_t : public rv64::conversion::extend_t {
    static constexpr const char *NAME = "rvjit::rvv::conversion::extend";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit extend_t(rvjit::emitter_t e)
        : rv64::conversion::extend_t(e), e_(e) {}

    bool is_valid(const rvjit::conversion::op_t &op) const override {
        if (op.dst.is_scalar() && op.src.is_scalar())
            return rv64::conversion::extend_t::is_valid(op);
        return select_(op) != 0;
    }

    void emit(const rvjit::conversion::op_t &op,
            const rvjit::conversion::args_t &args) const override {
        if (op.dst.is_scalar() && op.src.is_scalar()) {
            rv64::conversion::extend_t::emit(op, args);
            return;
        }
        const int sel = select_(op);
        CHECK_OR_RET(sel != 0, , "invalid descriptor");
        CHECK_OR_RET(all_valid(op.dst, op.src) && args.named(), ,
                "invalid operand descriptor");

        const VReg vd(args.dst), vs2(args.src);
        const VM vm = rv64::isa_traits_base_t::vm(op.predicate);
        switch (sel) {
            case 2: e_->vsext_vf2(vd, vs2, vm); break;
            case 4: e_->vsext_vf4(vd, vs2, vm); break;
            case 8: e_->vsext_vf8(vd, vs2, vm); break;
            case -2: e_->vzext_vf2(vd, vs2, vm); break;
            case -4: e_->vzext_vf4(vd, vs2, vm); break;
            case -8: e_->vzext_vf8(vd, vs2, vm); break;
            default: break;
        }
    }

private:
    rv64::emitter_t e_;

    // The ratio, negative for a zero extension, or 0 when unsupported
    static int select_(const rvjit::conversion::op_t &op) {
        if (!op.is_extending() || !op.dst.is_vector() || !op.src.is_vector())
            return 0;
        if (!op.dst.has_valid_width() || !op.src.has_valid_width()) return 0;
        if (!op.dst.is_int() || !op.src.is_int()) return 0;

        // A signed source needs a signed destination; an unsigned source
        // extends with zeros into either
        const int ratio = op.dst.width / op.src.width;
        if (op.src.is_signed()) return op.dst.is_signed() ? ratio : 0;
        return -ratio;
    }
};

} // namespace conversion
} // namespace rvv
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
