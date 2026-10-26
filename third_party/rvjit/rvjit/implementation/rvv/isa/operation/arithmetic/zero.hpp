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

#ifndef RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_ARITHMETIC_ZERO_HPP
#define RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_ARITHMETIC_ZERO_HPP

#include "rvjit/implementation/rv64/isa/operation/arithmetic/zero.hpp"
#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operation/arithmetic.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace rvv {
namespace arithmetic {

/// General register-clearing operation, extending the RV64 scalar one with
/// vectors
///
/// @details Emits `vmv.v.i` on a vector `op`, and leaves scalars to
///     `rv64::arithmetic::zero_t` (`mv`/`fmv.d.x` against the zero register)
struct zero_t : public rv64::arithmetic::zero_t {
    static constexpr const char *NAME = "rvjit::rvv::arithmetic";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit zero_t(rvjit::emitter_t e) : rv64::arithmetic::zero_t(e), e_(e) {}

    bool is_valid(const rvjit::arithmetic::op_t &op) const override {
        if (op.is_unary() && op.rd.is_defined() && op.rd.is_vector())
            return true;
        return rv64::arithmetic::zero_t::is_valid(op);
    }

    void emit(const rvjit::arithmetic::op_t &op,
            const rvjit::arithmetic::args_t &args) const override {
        if (!op.rd.is_vector()) {
            rv64::arithmetic::zero_t::emit(op, args);
            return;
        }
        CHECK_OR_RET(op.is_unary(), , "expected a unary operand descriptor");
        CHECK_OR_RET(op.rd.is_valid() && args.named(1), ,
                "invalid operand descriptor");
        e_->vmv_v_i(vreg_t(args.rd), 0);
    }

private:
    rv64::emitter_t e_;
};

} // namespace arithmetic
} // namespace rvv
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
