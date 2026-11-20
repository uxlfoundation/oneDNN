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

#ifndef RVJIT_INTERFACE_ISA_OPERATION_EXPRESSION_HPP
#define RVJIT_INTERFACE_ISA_OPERATION_EXPRESSION_HPP

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace operation {

/// The ISA logic of one kind of expression, over operations `Op` emitted on
/// the registers `Args`
///
/// @details A dialect owns one per kind of expression and binds it to the
///     expressions it creates
template <typename Op, typename Args>
struct expression_impl_t {
    static constexpr const char *NAME = "rvjit::operation::expression_impl";

    virtual ~expression_impl_t() = default;

    /// Whether `op` can be emitted
    ///
    /// @param op The operation checked
    virtual bool is_valid(const Op &op) const {
        (void)op;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Emits `op` on the registers `args`
    ///
    /// @param op The operation
    /// @param args The registers of its operands
    ///
    /// @pre `is_valid(op)` is true
    virtual void emit(const Op &op, const Args &args) const {
        (void)op;
        (void)args;
        LOG_FAIL(__func__, "unimplemented");
    }
};

/// An operation of a dialect, bound to the ISA logic emitting it
///
/// @details Holds the forms and types of the operation (its `Op` members),
///     never its registers: one expression emits any number of times, each
///     call naming the registers, which construct an `Args`
///
/// @pre The dialect that created the expression outlives it
template <typename Op, typename Args>
struct expression_t : public Op {
    static constexpr const char *NAME = "rvjit::operation::expression";

    /// An unbound expression, never valid
    expression_t() = default;

    /// Binds `op` to the logic emitting it
    ///
    /// @param impl The ISA logic
    /// @param op The operation
    expression_t(const expression_impl_t<Op, Args> &impl, const Op &op)
        : Op(op), impl_(&impl) {}

    /// Whether the operation can be emitted
    bool is_valid() const { return impl_ && impl_->is_valid(*this); }

    /// Emits the operation on the registers `names`
    ///
    /// @param names The registers, in the order of the `Args` constructor
    ///
    /// @pre `is_valid()` is true
    template <typename... Names>
    void operator()(const Names &...names) const {
        CHECK_OR_RET(impl_, , "an unbound expression");
        impl_->emit(*this, Args(names...));
    }

private:
    const expression_impl_t<Op, Args> *impl_ = nullptr;
};

} // namespace operation
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
