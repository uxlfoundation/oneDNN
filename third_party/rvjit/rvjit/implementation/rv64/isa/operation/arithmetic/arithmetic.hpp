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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_ARITHMETIC_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_ARITHMETIC_HPP

#include "rvjit/implementation/rv64/isa/operation/arithmetic/add.hpp"
#include "rvjit/implementation/rv64/isa/operation/arithmetic/bitcast.hpp"
#include "rvjit/implementation/rv64/isa/operation/arithmetic/div.hpp"
#include "rvjit/implementation/rv64/isa/operation/arithmetic/macc.hpp"
#include "rvjit/implementation/rv64/isa/operation/arithmetic/mul.hpp"
#include "rvjit/implementation/rv64/isa/operation/arithmetic/zero.hpp"
#include "rvjit/interface/isa/operation/arithmetic.hpp"

namespace rvjit {
namespace rv64 {
namespace arithmetic {

/// RV64 arithmetic dialect: scalar macc, zero, mul, add, div and bitcast
using arithmetic_t = operation::arithmetic_compose_t<macc_t, zero_t, mul_t,
        add_t, bitcast_t, div_t>;

} // namespace arithmetic
} // namespace rv64
} // namespace rvjit

#endif
