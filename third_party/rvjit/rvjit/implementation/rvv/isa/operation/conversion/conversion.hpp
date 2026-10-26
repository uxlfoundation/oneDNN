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

#ifndef RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_CONVERSION_HPP
#define RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_CONVERSION_HPP

#include "rvjit/implementation/rvv/isa/operation/conversion/cast.hpp"
#include "rvjit/implementation/rvv/isa/operation/conversion/extend.hpp"
#include "rvjit/implementation/rvv/isa/operation/conversion/narrow.hpp"
#include "rvjit/implementation/rvv/isa/operation/conversion/widen.hpp"
#include "rvjit/interface/isa/operation/conversion.hpp"

namespace rvjit {
namespace rvv {
namespace conversion {

/// RVV conversion dialect: vector-to-vector narrowings, widenings, extensions
/// and casts, extending the RV64 dialect's scalar widen, extend and cast
///
/// @note Each scalar-to-scalar conversion is the RV64 expression's, under its
///     conditions; there is no scalar narrowing, and mixed forms are invalid
using conversion_t
        = operation::conversion_compose_t<narrow_t, widen_t, extend_t, cast_t>;

} // namespace conversion
} // namespace rvv
} // namespace rvjit

#endif
