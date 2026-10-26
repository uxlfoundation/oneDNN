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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_CONVERSION_EXTEND_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_CONVERSION_EXTEND_HPP

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/operation/conversion.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace rv64 {
namespace conversion {

/// Scalar integer extension into a type 2, 4 or 8 times as wide
///
/// @details Extends the low `width` bits of the source to 64 bits, by its
///     sign when signed and by zeros otherwise: `addiw` (s32) or `andi` (u8)
///     alone, else a left shift and a right shift; the bits above the
///     source's width are never read
///
/// @note A signed source needs a signed destination; an unsigned source
///     extends with zeros into either
/// @note The destination may be the source register
/// @note Uses base RV64I only: Zba/Zbb would make every case one instruction
struct extend_t : public rvjit::conversion::expression_impl_t {
    static constexpr const char *NAME = "rvjit::rv64::conversion::extend";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit extend_t(rvjit::emitter_t e) : e_(e) {}

    bool is_valid(const rvjit::conversion::op_t &op) const override {
        if (!op.is_extending() || !op.dst.is_scalar() || !op.src.is_scalar())
            return false;
        if (!op.dst.is_int() || !op.src.is_int()) return false;

        // A signed source needs a signed destination; an unsigned source
        // extends with zeros into either
        return !op.src.is_signed() || op.dst.is_signed();
    }

    void emit(const rvjit::conversion::op_t &op,
            const rvjit::conversion::args_t &args) const override {
        CHECK_OR_RET(extend_t::is_valid(op), , "invalid descriptor");
        CHECK_OR_RET(all_valid(op.dst, op.src) && args.named(), ,
                "invalid operand descriptor");

        const Reg rd(args.dst), rs(args.src);
        const uint32_t shift = 64 - 8 * op.src.width;
        if (op.src.is_signed() && op.src.width == 4) {
            e_->addiw(rd, rs, 0);
        } else if (!op.src.is_signed() && op.src.width == 1) {
            e_->andi(rd, rs, 0xff);
        } else {
            e_->slli(rd, rs, shift);
            if (op.src.is_signed())
                e_->srai(rd, rd, shift);
            else
                e_->srli(rd, rd, shift);
        }
    }

private:
    rv64::emitter_t e_;
};

} // namespace conversion
} // namespace rv64
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
