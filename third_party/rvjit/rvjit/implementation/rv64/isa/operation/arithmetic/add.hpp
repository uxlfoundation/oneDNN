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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_ARITHMETIC_ADD_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_ARITHMETIC_ADD_HPP

#include "rvjit/implementation/rv64/isa/operation/arithmetic/uniform.hpp"
#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operation/arithmetic.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace rv64 {
namespace arithmetic {

/// General scalar add instruction, `rd = rs1 + rs2`
///
/// @details Emits `addw` on 32-bit integers, `add` on the other integers, or
///     `fadd.{h,s,d}` on floats, from uniform scalar shapes only
///
/// @note An integer sum reads only the low `width` bits of its operands and
///     is exact in the low `width` bits of `rd`; an s32, s64 or u64 sum is
///     also extended by its sign, while the bits above an s8, u8, s16, u16 or
///     u32 sum are unspecified (`conversion_t::extend` restores them)
struct add_t : public rvjit::arithmetic::expression_impl_t {
    static constexpr const char *NAME = "rvjit::rv64::arithmetic";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit add_t(rvjit::emitter_t e) : e_(e) {}

    /// Whether `op` maps to a supported instruction
    ///
    /// @param op The operation checked
    bool is_valid(const rvjit::arithmetic::op_t &op) const override {
        return is_uniform_scalar(op);
    }

    void emit(const rvjit::arithmetic::op_t &op,
            const rvjit::arithmetic::args_t &args) const override {
        CHECK_OR_RET(args.named(3), , "invalid operand descriptor");
        CHECK_OR_RET(add_t::is_valid(op), , "invalid descriptor");

        if (op.rd.is_int()) {
            const Reg rd(args.rd), rs1(args.rs1), rs2(args.rs2);
            if (op.rd.width == 4)
                e_->addw(rd, rs1, rs2);
            else
                e_->add(rd, rs1, rs2);
            return;
        }
        const FReg fd(args.rd), fs1(args.rs1), fs2(args.rs2);
        switch (op.rd.width) {
            case 2: e_->fadd_h(fd, fs1, fs2); break;
            case 4: e_->fadd_s(fd, fs1, fs2); break;
            default: e_->fadd_d(fd, fs1, fs2); break;
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
