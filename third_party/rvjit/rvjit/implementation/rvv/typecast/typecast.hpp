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

#ifndef RVJIT_IMPLEMENTATION_RVV_TYPECAST_TYPECAST_HPP
#define RVJIT_IMPLEMENTATION_RVV_TYPECAST_TYPECAST_HPP

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/implementation/rvv/rvv.hpp"
#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/operation/conversion.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/interface/typecast/typecast.hpp"
#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace rvv {
namespace typecast {

/// The vector configuration a position of a typecast runs in
struct vconfig_t {
    SEW sew = SEW::e8;
    LMUL lmul = LMUL::m1;
};

/// RVV typecast, switching the element width between steps
///
/// @details Each step runs at the SEW of one of its operands: its source's
///     for a widening, which reads at the narrow width, and its destination's
///     otherwise. The LMUL scales the output configuration of `vtype` by the
///     ratio of the widths, keeping the vector length. A position whose SEW
///     differs from the one before it (the state hint's, for position 0)
///     interposes `vsetvli x0, x0, sew, lmul` with `vtype`'s policies, and
///     position `nsteps()` returns to the state hint's SEW
///
///     A widening reads the narrow value from the highest-numbered part of
///     the wide register group and a narrowing leaves it in the lowest part
///
/// @note A scalar typecast runs outside `vtype`: it needs no `state_hint` and
///     interposes nothing. `configure` requires a vector `state_hint` for a
///     non-empty vector typecast, and fails when an interposed configuration
///     is fractional or beyond m8
///
/// @pre `vtype` holds the output configuration the kernel runs at before
///     `configure`, and the vector length in effect is that configuration's
struct typecast_t : public rvjit::typecast::typecast_t {
    static constexpr const char *NAME = "rvjit::rvv::typecast";

    /// Creates an empty, invalid typecast
    ///
    /// @param cvt The RVV conversion dialect
    /// @param e The emitter the configurations are emitted through
    /// @param vtype The vector type settings whose output configuration the
    ///     positions scale
    ///
    /// @pre `cvt` and `vtype` outlive the typecast
    typecast_t(const operation::conversion_t &cvt, rvjit::emitter_t e,
            const vtype_t &vtype)
        : rvjit::typecast::typecast_t(cvt), e_(e), vtype_(&vtype) {}

    bool configure(const rvjit::conversion::op_t &op,
            const operand_t &state_hint = absent_operand()) override {
        if (!rvjit::typecast::typecast_t::configure(op, state_hint))
            return false;
        if (empty() || !dst().is_vector()) return true;
        const bool hinted
                = state_hint.is_vector() && state_hint.type != type_t::none;
        const bool fits = hinted && fits_();
        if (!fits) clear_();
        CHECK(hinted, "a typecast without a vector state hint");
        CHECK(fits, "a configuration beyond the register groups");
        return true;
    }

    /// The SEW position `i` runs at
    ///
    /// @param i The position, from 0 to `nsteps()`
    SEW sew(int i) const {
        if (i >= nsteps()) return sew_for(state_hint().type);
        const step_t &s = step(i);
        return sew_for(s.kind == step_t::widen ? s.op.src.type : s.op.dst.type);
    }

    /// The configuration position `i` runs in
    ///
    /// @param i The position, from 0 to `nsteps()`
    ///
    /// @note The LMUL is m1 when no LMUL keeps the vector length
    vconfig_t config(int i) const {
        vconfig_t c;
        c.sew = sew(i);
        keep_vl_lmul(*vtype_, c.sew, c.lmul);
        return c;
    }

    bool interposes(int i) const override {
        if (!is_valid() || empty() || !dst().is_vector()) return false;
        if (i < 0 || i > nsteps()) return false;
        const SEW before = i == 0 ? sew(nsteps()) : sew(i - 1);
        return sewb(sew(i)) != sewb(before);
    }

    /// Emits the `vsetvli` of position `i`
    bool interpose(int i) const override {
        CHECK(is_valid() && i >= 0 && i <= nsteps(), "an invalid position");
        LMUL lmul = LMUL::m1;
        CHECK(keep_vl_lmul(*vtype_, sew(i), lmul),
                "a configuration beyond the register groups");
        e_->vsetvli(
                ignore_out, ignore_out, sew(i), lmul, vtype_->vt, vtype_->vm);
        return true;
    }

private:
    rv64::emitter_t e_;
    const vtype_t *vtype_;

    bool fits_() const {
        LMUL lmul = LMUL::m1;
        for (int i = 0; i <= nsteps(); ++i)
            if (interposes(i) && !keep_vl_lmul(*vtype_, sew(i), lmul))
                return false;
        return true;
    }
};

} // namespace typecast
} // namespace rvv
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
