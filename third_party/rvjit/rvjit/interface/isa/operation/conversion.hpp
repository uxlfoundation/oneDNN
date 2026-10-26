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

#ifndef RVJIT_INTERFACE_ISA_OPERATION_CONVERSION_HPP
#define RVJIT_INTERFACE_ISA_OPERATION_CONVERSION_HPP

#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/operation/expression.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/utils.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace conversion {

using operand_t = rvjit::operand_t;

/// Conversion of a source operand into a destination operand of another type
///
/// @details Describes the forms and types of both operands; the registers
///     are the arguments of the expression emitting it (`args_t`)
///
/// @note The type traits `typed_t` answers are the destination's
struct op_t : public typed_t {
    operand_t src = absent_();
    operand_t dst = absent_();
    predicate_t predicate = predicate_t::unmasked;

    op_t() = default;

    /// Creates a conversion between two operands
    ///
    /// @param dst The destination operand
    /// @param src The source operand
    op_t(const operand_t &dst, const operand_t &src) : src(src), dst(dst) {}

    /// Creates a conversion between two types of one form
    ///
    /// @param dst The destination data type
    /// @param src The source data type
    /// @param form The storage category of both operands
    op_t(type_t dst, type_t src, rvjit::form_t form)
        : src(form, src), dst(form, dst) {}

    bool is_scalar() const override { return dst.is_scalar(); }
    bool is_vector() const override { return dst.is_vector(); }
    bool is_int() const override { return dst.is_int(); }
    bool is_float() const override { return dst.is_float(); }
    bool is_signed() const override { return dst.is_signed(); }

    /// Whether both operands are present
    bool is_complete() const { return is_present_(src) && is_present_(dst); }

    /// Whether the destination is half as wide as the source
    bool is_narrowing() const {
        return is_complete() && 2 * dst.width == src.width;
    }

    /// Whether the destination is twice as wide as the source
    bool is_widening() const {
        return is_complete() && dst.width == 2 * src.width;
    }

    /// Whether the destination is 2, 4 or 8 times as wide as the source
    bool is_extending() const {
        if (!is_complete() || src.width <= 0) return false;
        const int ratio = dst.width / src.width;
        return dst.width % src.width == 0
                && (ratio == 2 || ratio == 4 || ratio == 8);
    }

    /// Whether the conversion changes between integer and float
    bool is_casting() const {
        return is_complete() && dst.is_float() != src.is_float();
    }

private:
    static operand_t absent_() {
        return operand_t(rvjit::form_t::undefined, type_t::none);
    }
    static bool is_present_(const operand_t &o) {
        return o.type != type_t::none;
    }
};

/// The registers a conversion expression is emitted on
struct args_t {
    name_t dst;
    name_t src;

    /// Names the registers of both operands
    ///
    /// @param dst The destination register
    /// @param src The source register
    args_t(name_t dst, name_t src) : dst(dst), src(src) {}

    /// Whether both registers are named
    bool named() const { return dst != no_name && src != no_name; }
};

using expression_t = operation::expression_t<op_t, args_t>;
using expression_impl_t = operation::expression_impl_t<op_t, args_t>;

} // namespace conversion
} // namespace rvjit

namespace rvjit {
namespace operation {

/// Conversion expressions
///
/// @details A factory of `conversion::expression_t`: each method binds a
///     conversion to the ISA logic of its kind, and the expression is then
///     called with the registers, e.g. `cast(form, to, from)(dst, src)`. Each
///     kind takes the whole `conversion::op_t`, two types of one form, or two
///     operands. Which forms an expression accepts is the implementation's
///
/// @note An implementation overrides the protected accessors of the kinds it
///     implements; the others are never valid
struct conversion_t {
    static constexpr const char *NAME = "rvjit::operation::conversion";

    virtual ~conversion_t() = default;

    /// Conversion of `src` into a type half as wide, `dst = narrow(src)`;
    /// integers saturate, floats round to the nearest even
    ///
    /// @param op The conversion
    ///
    /// @pre The `op.is_narrowing()` expression is true
    /// @pre `dst` may be `src`, narrowing in place
    conversion::expression_t narrow(const conversion::op_t &op) const {
        return conversion::expression_t(narrow_(), op);
    }

    /// Narrowing between two types of one form
    ///
    /// @param form The storage category of both operands
    /// @param to The destination data type
    /// @param from The source data type
    conversion::expression_t narrow(form_t form, type_t to, type_t from) const {
        return narrow(conversion::op_t(to, from, form));
    }

    /// Narrowing between two operands
    ///
    /// @param dst The destination operand
    /// @param src The source operand
    conversion::expression_t narrow(
            const operand_t &dst, const operand_t &src) const {
        return narrow(conversion::op_t(dst, src));
    }

    /// Conversion of `src` into a type twice as wide, `dst = widen(src)`; an
    /// integer extends by its sign when signed and by zeros otherwise
    ///
    /// @param op The conversion
    ///
    /// @pre The `op.is_widening()` expression is true
    conversion::expression_t widen(const conversion::op_t &op) const {
        return conversion::expression_t(widen_(), op);
    }

    /// Widening between two types of one form
    ///
    /// @param form The storage category of both operands
    /// @param to The destination data type
    /// @param from The source data type
    conversion::expression_t widen(form_t form, type_t to, type_t from) const {
        return widen(conversion::op_t(to, from, form));
    }

    /// Widening between two operands
    ///
    /// @param dst The destination operand
    /// @param src The source operand
    conversion::expression_t widen(
            const operand_t &dst, const operand_t &src) const {
        return widen(conversion::op_t(dst, src));
    }

    /// Integer extension of `src` into a type 2, 4 or 8 times as wide,
    /// `dst = extend(src)`, by the sign of a signed source and by zeros
    /// otherwise
    ///
    /// @param op The conversion
    ///
    /// @pre The `op.is_extending()` expression is true
    conversion::expression_t extend(const conversion::op_t &op) const {
        return conversion::expression_t(extend_(), op);
    }

    /// Extension between two types of one form
    ///
    /// @param form The storage category of both operands
    /// @param to The destination data type
    /// @param from The source data type
    conversion::expression_t extend(form_t form, type_t to, type_t from) const {
        return extend(conversion::op_t(to, from, form));
    }

    /// Extension between two operands
    ///
    /// @param dst The destination operand
    /// @param src The source operand
    conversion::expression_t extend(
            const operand_t &dst, const operand_t &src) const {
        return extend(conversion::op_t(dst, src));
    }

    /// Conversion of `src` between an integer and a float type,
    /// `dst = cast(src)`, rounding as the dynamic rounding mode says (round to
    /// nearest even unless changed)
    ///
    /// @param op The conversion
    ///
    /// @pre The `op.is_casting()` expression is true
    conversion::expression_t cast(const conversion::op_t &op) const {
        return conversion::expression_t(cast_(), op);
    }

    /// Cast between two types of one form
    ///
    /// @param form The storage category of both operands
    /// @param to The destination data type
    /// @param from The source data type
    conversion::expression_t cast(form_t form, type_t to, type_t from) const {
        return cast(conversion::op_t(to, from, form));
    }

    /// Cast between two operands
    ///
    /// @param dst The destination operand
    /// @param src The source operand
    conversion::expression_t cast(
            const operand_t &dst, const operand_t &src) const {
        return cast(conversion::op_t(dst, src));
    }

protected:
    /// The logic of `narrow`
    virtual const conversion::expression_impl_t &narrow_() const {
        return unimplemented_();
    }

    /// The logic of `widen`
    virtual const conversion::expression_impl_t &widen_() const {
        return unimplemented_();
    }

    /// The logic of `extend`
    virtual const conversion::expression_impl_t &extend_() const {
        return unimplemented_();
    }

    /// The logic of `cast`
    virtual const conversion::expression_impl_t &cast_() const {
        return unimplemented_();
    }

private:
    static const conversion::expression_impl_t &unimplemented_() {
        static const conversion::expression_impl_t none;
        return none;
    }
};

/// Assembles a `conversion_t` from individual expression implementations
///
/// @details Each implementation derives from `conversion::expression_impl_t`
///     and is constructible from `emitter_t`
template <typename NarrowOp, typename WidenOp, typename ExtendOp,
        typename CastOp>
struct conversion_compose_t : public conversion_t {
    /// Constructs every expression implementation
    ///
    /// @param e The emitter the expressions emit through
    explicit conversion_compose_t(emitter_t e)
        : narrow_impl_(e), widen_impl_(e), extend_impl_(e), cast_impl_(e) {}

protected:
    const conversion::expression_impl_t &narrow_() const override {
        return narrow_impl_;
    }
    const conversion::expression_impl_t &widen_() const override {
        return widen_impl_;
    }
    const conversion::expression_impl_t &extend_() const override {
        return extend_impl_;
    }
    const conversion::expression_impl_t &cast_() const override {
        return cast_impl_;
    }

    NarrowOp narrow_impl_;
    WidenOp widen_impl_;
    ExtendOp extend_impl_;
    CastOp cast_impl_;
};

} // namespace operation
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
