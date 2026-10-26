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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_ARITHMETIC_UNIFORM_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_ARITHMETIC_UNIFORM_HPP

#include "rvjit/interface/isa/operation/arithmetic.hpp"
#include "rvjit/interface/isa/types.hpp"

namespace rvjit {
namespace rv64 {
namespace arithmetic {

/// Whether `op` is a ternary scalar operation whose operands share a type
/// RV64 computes on
///
/// @param op The operation checked
///
/// @details The type is an integer of any width, or f16/f32/f64; RV64 has no
///     bf16 or widening scalar arithmetic
inline bool is_uniform_scalar(const rvjit::arithmetic::op_t &op) {
    if (!op.is_ternary() || !op.rd.is_scalar() || !op.rs1.is_scalar()
            || !op.rs2.is_scalar())
        return false;
    const rvjit::type_t t = op.rd.type;
    if (op.rs1.type != t || op.rs2.type != t) return false;
    if (op.rd.is_int()) return op.rd.has_valid_width();
    return t == rvjit::type_t::f16 || t == rvjit::type_t::f32
            || t == rvjit::type_t::f64;
}

} // namespace arithmetic
} // namespace rv64
} // namespace rvjit

#endif
