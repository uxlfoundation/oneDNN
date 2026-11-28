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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_CONVERSION_WIDEN_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_CONVERSION_WIDEN_HPP

#include "rvjit/implementation/rv64/isa/operation/conversion/extend.hpp"
#include "rvjit/interface/isa/operation/conversion.hpp"
#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace rv64 {
namespace conversion {

/// Scalar integer conversion into a type twice as wide, `extend_t` at 2x
///
/// @details Emits what `extend_t` emits: the low `width` bits of the source,
///     extended to 64 bits by its sign when signed and by zeros otherwise
///
/// @note The bits above the source's width are never read, so the result is
///     exact whatever they hold, e.g. after a narrow `add` or `mul`
/// @note A signed source needs a signed destination; an unsigned source
///     widens into either. Floats are invalid
/// @note The destination may be the source register
struct widen_t : public extend_t {
    static constexpr const char *NAME = "rvjit::rv64::conversion::widen";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit widen_t(rvjit::emitter_t e) : extend_t(e) {}

    bool is_valid(const rvjit::conversion::op_t &op) const override {
        return op.is_widening() && extend_t::is_valid(op);
    }

    void emit(const rvjit::conversion::op_t &op,
            const rvjit::conversion::args_t &args) const override {
        CHECK_OR_RET(widen_t::is_valid(op), , "invalid descriptor");
        extend_t::emit(op, args);
    }
};

} // namespace conversion
} // namespace rv64
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
