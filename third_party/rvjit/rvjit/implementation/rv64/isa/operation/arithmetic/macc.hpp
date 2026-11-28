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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_ARITHMETIC_MACC_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_ARITHMETIC_MACC_HPP

#include "rvjit/implementation/rv64/isa/operation/arithmetic/uniform.hpp"
#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operation/arithmetic.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace rv64 {
namespace arithmetic {

/// General scalar multiply-accumulate instruction, `rd += rs1 * rs2`
///
/// @details Emits the fused `fmadd.{h,s,d} rd, rs1, rs2, rd` on uniform
///     f16/f32/f64 scalar shapes; RV64 has no integer multiply-accumulate
struct macc_t : public rvjit::arithmetic::expression_impl_t {
    static constexpr const char *NAME = "rvjit::rv64::arithmetic";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit macc_t(rvjit::emitter_t e) : e_(e) {}

    /// Whether `op` maps to a supported instruction
    ///
    /// @param op The operation checked
    bool is_valid(const rvjit::arithmetic::op_t &op) const override {
        return is_uniform_scalar(op) && op.rd.is_float();
    }

    void emit(const rvjit::arithmetic::op_t &op,
            const rvjit::arithmetic::args_t &args) const override {
        CHECK_OR_RET(args.named(3), , "invalid operand descriptor");
        CHECK_OR_RET(macc_t::is_valid(op), , "invalid descriptor");

        const FReg fd(args.rd), fs1(args.rs1), fs2(args.rs2);
        switch (op.rd.width) {
            case 2: e_->fmadd_h(fd, fs1, fs2, fd); break;
            case 4: e_->fmadd_s(fd, fs1, fs2, fd); break;
            default: e_->fmadd_d(fd, fs1, fs2, fd); break;
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
