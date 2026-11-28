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

#ifndef RVJIT_IMPLEMENTATION_RV64_TYPECAST_TYPECAST_HPP
#define RVJIT_IMPLEMENTATION_RV64_TYPECAST_TYPECAST_HPP

#include "rvjit/implementation/rv64/isa/operation/conversion/conversion.hpp"
#include "rvjit/interface/typecast/typecast.hpp"

namespace rvjit {
namespace rv64 {
namespace typecast {

/// RV64 typecast over `rv64::conversion::conversion_t`
///
/// @details Scalar steps need no machine state, so nothing is interposed and
///     the state hint is ignored. Only the single-step casts between an f16,
///     f32 or f64 and a 32- or 64-bit integer are valid, the dialect
///     implementing no other conversion
using typecast_t = rvjit::typecast::typecast_t;

} // namespace typecast
} // namespace rv64
} // namespace rvjit

#endif
