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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_CONVERSION_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_CONVERSION_HPP

#include "rvjit/implementation/rv64/isa/operation/conversion/cast.hpp"
#include "rvjit/implementation/rv64/isa/operation/conversion/extend.hpp"
#include "rvjit/implementation/rv64/isa/operation/conversion/widen.hpp"
#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operation/conversion.hpp"
#include "rvjit/interface/isa/types.hpp"

namespace rvjit {
namespace rv64 {
namespace conversion {

/// RV64 conversion dialect, scalar-to-scalar only
///
/// @note Only `widen`, `extend` and `cast` are implemented
struct conversion_t : public operation::conversion_t {
    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit conversion_t(rvjit::emitter_t e)
        : widen_impl_(e), extend_impl_(e), cast_impl_(e) {}

    const rvjit::conversion::expression_impl_t &widen_() const override {
        return widen_impl_;
    }
    const rvjit::conversion::expression_impl_t &extend_() const override {
        return extend_impl_;
    }
    const rvjit::conversion::expression_impl_t &cast_() const override {
        return cast_impl_;
    }

protected:
    widen_t widen_impl_;
    extend_t extend_impl_;
    cast_t cast_impl_;
};

} // namespace conversion
} // namespace rv64
} // namespace rvjit

#endif
