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

#ifndef RVJIT_INTERFACE_ISA_OPERATION_ARITHMETIC_HPP
#define RVJIT_INTERFACE_ISA_OPERATION_ARITHMETIC_HPP

#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/operation/expression.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/utils.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace arithmetic {

using operand_t = rvjit::operand_t;

/// Generic arithmetic operation involving up to three operands
///
/// @details Describes the forms and types of the operands; the registers are
///     the arguments of the expression emitting it (`args_t`)
struct op_t : public typed_t {
    operand_t rd = absent_();
    operand_t rs1 = absent_();
    operand_t rs2 = absent_();
    predicate_t predicate = predicate_t::unmasked;

    op_t() = default;

    /// Creates a unary operation
    ///
    /// @param dst The destination operand
    explicit op_t(const operand_t &dst) : rd(dst) {}

    /// Creates a ternary operation
    ///
    /// @param dst The destination operand
    /// @param src1 The first source operand
    /// @param src2 The second source operand
    op_t(const operand_t &dst, const operand_t &src1, const operand_t &src2)
        : rd(dst), rs1(src1), rs2(src2) {}

    /// Creates a ternary operation from data types
    ///
    /// @param dst The destination data type
    /// @param src1 The first source data type
    /// @param src2 The second source data type
    /// @param form The storage category of every operand
    op_t(type_t dst, type_t src1, type_t src2,
            rvjit::form_t form = rvjit::form_t::undefined)
        : rd(form, dst), rs1(form, src1), rs2(form, src2) {}

    /// Number of present operands
    int size() const {
        if (!is_present_(rd)) return 0;
        if (!is_present_(rs1)) return 1;
        return is_present_(rs2) ? 3 : 2;
    }

    bool is_unary() const { return size() == 1; }
    bool is_binary() const { return size() == 2; }
    bool is_ternary() const { return size() == 3; }
    bool is_scalar() const override { return rd.is_scalar(); }
    bool is_vector() const override { return rd.is_vector(); }
    bool is_int() const override { return rd.is_int(); }
    bool is_float() const override { return rd.is_float(); }
    bool is_signed() const override { return rd.is_signed(); }

    /// Whether every present operand has the same width
    bool is_uniform() const {
        return is_unary() || (rd.width == rs1.width && rs1.width == rs2.width);
    }

    /// Whether the destination is twice as wide as both equally wide sources
    bool is_widening() const {
        return !is_unary() && rd.width == 2 * rs1.width
                && rs1.width == rs2.width;
    }

    /// Whether the destination and the second source are twice as wide as
    /// the first source
    bool is_wide_operand() const {
        return is_ternary() && rd.width == rs2.width
                && rd.width == 2 * rs1.width;
    }

    /// Sets every operand to the scalar form
    op_t &scalar() {
        rd.form = rs1.form = rs2.form = rvjit::form_t::scalar;
        return *this;
    }

    /// Sets every operand to the vector form
    ///
    /// @param vm The vector mask policy
    op_t &vector(predicate_t vm = predicate_t::unmasked) {
        rd.form = rs1.form = rs2.form = rvjit::form_t::vector;
        predicate = vm;
        return *this;
    }

    /// Sets the first source to the scalar form and the rest to the vector form
    ///
    /// @param vm The vector mask policy
    op_t &vector_scalar(predicate_t vm = predicate_t::unmasked) {
        rd.form = rvjit::form_t::vector;
        rs1.form = rvjit::form_t::scalar;
        rs2.form = rvjit::form_t::vector;
        predicate = vm;
        return *this;
    }

    /// Creates a ternary operation whose operands all have the type `dt`
    ///
    /// @param dt The data type of every operand
    /// @param form The storage category of every operand
    static op_t uniform(type_t dt, rvjit::form_t form) {
        const operand_t o(form, dt);
        return op_t(o, o, o);
    }

    /// Creates a binary operation of the operands `dst` and `src`
    ///
    /// @param dst The destination operand
    /// @param src The source operand
    static op_t binary(const operand_t &dst, const operand_t &src) {
        return op_t(dst, src, absent_());
    }

    /// Creates a ternary operation whose sources are `dt` and destination
    /// is the natural 2x-wide type of `dt`
    ///
    /// @param dt The data type of both sources
    /// @param form The storage category of every operand
    static op_t widening(type_t dt, rvjit::form_t form) {
        const operand_t wide(form, natural_wide(dt));
        const operand_t narrow(form, dt);
        return op_t(wide, narrow, narrow);
    }

    /// Creates a ternary operation whose first source is `dt` and whose
    /// destination and second source are the natural 2x-wide type of `dt`
    ///
    /// @param dt The data type of the first source
    /// @param form The storage category of every operand
    static op_t wide_operand(type_t dt, rvjit::form_t form) {
        const operand_t wide(form, natural_wide(dt));
        const operand_t narrow(form, dt);
        return op_t(wide, narrow, wide);
    }

private:
    static operand_t absent_() {
        return operand_t(rvjit::form_t::undefined, type_t::none);
    }
    static bool is_present_(const operand_t &o) {
        return o.type != type_t::none;
    }
};

/// The registers an arithmetic expression is emitted on
struct args_t {
    name_t rd;
    name_t rs1;
    name_t rs2;

    /// Names the registers of the operands, absent ones `no_name`
    ///
    /// @param rd The destination register
    /// @param rs1 The first source register
    /// @param rs2 The second source register
    args_t(name_t rd, name_t rs1 = no_name, name_t rs2 = no_name)
        : rd(rd), rs1(rs1), rs2(rs2) {}

    /// Whether the first `n` registers are named
    ///
    /// @param n The number of operands, 1 to 3
    bool named(int n) const {
        return rd != no_name && (n < 2 || rs1 != no_name)
                && (n < 3 || rs2 != no_name);
    }
};

using expression_t = operation::expression_t<op_t, args_t>;
using expression_impl_t = operation::expression_impl_t<op_t, args_t>;

/// The implementation of a kind a target lacks, never valid
struct unimplemented_t : public expression_impl_t {
    /// Creates the placeholder
    explicit unimplemented_t(emitter_t) {}
};

} // namespace arithmetic
} // namespace rvjit

namespace rvjit {
namespace operation {

/// Arithmetic expressions
///
/// @details A factory of `arithmetic::expression_t`: each method binds an
///     operation to the ISA logic of its kind, and the expression is then
///     called with the registers, e.g. `macc(form, dt)(rd, rs1, rs2)`. Each
///     kind takes the whole `arithmetic::op_t`, or the shapes of its common
///     forms
///
/// @note An implementation overrides the protected accessors of the kinds it
///     implements; the others are never valid
/// @note What a scalar register holds above a narrow integer's width is the
///     implementation's contract: each kind documents what it reads there
///     and leaves there
struct arithmetic_t {
    static constexpr const char *NAME = "rvjit::operation::arithmetic";

    virtual ~arithmetic_t() = default;

    /// Multiply-accumulate expression, `rd += rs1 * rs2`
    ///
    /// @param op The operation
    ///
    /// @pre The `op.is_ternary()` expression is true
    arithmetic::expression_t macc(const arithmetic::op_t &op) const {
        return arithmetic::expression_t(macc_(), op);
    }

    /// Multiply-accumulate expression whose operands share a form and type
    ///
    /// @param form The storage category of every operand
    /// @param dt The data type of every operand
    arithmetic::expression_t macc(form_t form, type_t dt) const {
        return macc(arithmetic::op_t::uniform(dt, form));
    }

    /// Multiply-accumulate expression of three operands
    ///
    /// @param rd The destination operand
    /// @param rs1 The first source operand
    /// @param rs2 The second source operand
    arithmetic::expression_t macc(const operand_t &rd, const operand_t &rs1,
            const operand_t &rs2) const {
        return macc(arithmetic::op_t(rd, rs1, rs2));
    }

    /// Clear buffer expression, clearing `rd`
    ///
    /// @param op The operation
    ///
    /// @pre The `op.is_unary()` expression is true
    arithmetic::expression_t zero(const arithmetic::op_t &op) const {
        return arithmetic::expression_t(zero_(), op);
    }

    /// Clear buffer expression of an operand
    ///
    /// @param form The storage category of `rd`
    /// @param dt The data type of `rd`
    arithmetic::expression_t zero(form_t form, type_t dt) const {
        return zero(arithmetic::op_t(operand_t(form, dt)));
    }

    /// Multiplication expression, `rd = rs1 * rs2`
    ///
    /// @param op The operation
    ///
    /// @pre The `op.is_ternary()` expression is true, and `op.is_uniform()`
    ///     or `op.is_widening()` is true
    arithmetic::expression_t mul(const arithmetic::op_t &op) const {
        return arithmetic::expression_t(mul_(), op);
    }

    /// Multiplication expression whose operands share a form and type
    ///
    /// @param form The storage category of every operand
    /// @param dt The data type of every operand
    arithmetic::expression_t mul(form_t form, type_t dt) const {
        return mul(arithmetic::op_t::uniform(dt, form));
    }

    /// Multiplication expression of three operands
    ///
    /// @param rd The destination operand
    /// @param rs1 The first source operand
    /// @param rs2 The second source operand
    arithmetic::expression_t mul(const operand_t &rd, const operand_t &rs1,
            const operand_t &rs2) const {
        return mul(arithmetic::op_t(rd, rs1, rs2));
    }

    /// Addition expression, `rd = rs1 + rs2`
    ///
    /// @param op The operation
    ///
    /// @pre The `op.is_ternary()` expression is true, and `op.is_uniform()`,
    ///     `op.is_widening()` or `op.is_wide_operand()` is true
    arithmetic::expression_t add(const arithmetic::op_t &op) const {
        return arithmetic::expression_t(add_(), op);
    }

    /// Addition expression whose operands share a form and type
    ///
    /// @param form The storage category of every operand
    /// @param dt The data type of every operand
    arithmetic::expression_t add(form_t form, type_t dt) const {
        return add(arithmetic::op_t::uniform(dt, form));
    }

    /// Addition expression of three operands
    ///
    /// @param rd The destination operand
    /// @param rs1 The first source operand
    /// @param rs2 The second source operand
    arithmetic::expression_t add(const operand_t &rd, const operand_t &rs1,
            const operand_t &rs2) const {
        return add(arithmetic::op_t(rd, rs1, rs2));
    }

    /// Division expression, `rd = rs2 / rs1`
    ///
    /// @param op The operation
    ///
    /// @note A scalar `rs1` is the divisor
    ///
    /// @pre The `op.is_ternary()` and `op.is_uniform()` expressions are true
    arithmetic::expression_t div(const arithmetic::op_t &op) const {
        return arithmetic::expression_t(div_(), op);
    }

    /// Division expression whose operands share a form and type
    ///
    /// @param form The storage category of every operand
    /// @param dt The data type of every operand
    arithmetic::expression_t div(form_t form, type_t dt) const {
        return div(arithmetic::op_t::uniform(dt, form));
    }

    /// Division expression of three operands
    ///
    /// @param rd The destination operand
    /// @param rs1 The divisor operand
    /// @param rs2 The dividend operand
    arithmetic::expression_t div(const operand_t &rd, const operand_t &rs1,
            const operand_t &rs2) const {
        return div(arithmetic::op_t(rd, rs1, rs2));
    }

    /// Bit-pattern move from an integer `rs1` into a float `rd`
    ///
    /// @param op The operation
    ///
    /// @pre The `op.is_binary()` expression is true and both widths match
    arithmetic::expression_t bitcast(const arithmetic::op_t &op) const {
        return arithmetic::expression_t(bitcast_(), op);
    }

    /// Bit-pattern move between two operands
    ///
    /// @param rd The float destination operand
    /// @param rs1 The integer source operand
    arithmetic::expression_t bitcast(
            const operand_t &rd, const operand_t &rs1) const {
        return bitcast(arithmetic::op_t::binary(rd, rs1));
    }

protected:
    /// The logic of `macc`
    virtual const arithmetic::expression_impl_t &macc_() const {
        return unimplemented_();
    }

    /// The logic of `zero`
    virtual const arithmetic::expression_impl_t &zero_() const {
        return unimplemented_();
    }

    /// The logic of `mul`
    virtual const arithmetic::expression_impl_t &mul_() const {
        return unimplemented_();
    }

    /// The logic of `add`
    virtual const arithmetic::expression_impl_t &add_() const {
        return unimplemented_();
    }

    /// The logic of `div`
    virtual const arithmetic::expression_impl_t &div_() const {
        return unimplemented_();
    }

    /// The logic of `bitcast`
    virtual const arithmetic::expression_impl_t &bitcast_() const {
        return unimplemented_();
    }

private:
    static const arithmetic::expression_impl_t &unimplemented_() {
        static const arithmetic::expression_impl_t none;
        return none;
    }
};

/// Assembles an `arithmetic_t` from individual expression implementations
///
/// @details Each implementation derives from `arithmetic::expression_impl_t`
///     and is constructible from `emitter_t`
///
/// @note A target without `div` leaves `DivOp` to `arithmetic::unimplemented_t`
template <typename MaccOp, typename ZeroOp, typename MulOp, typename AddOp,
        typename BitcastOp, typename DivOp = arithmetic::unimplemented_t>
struct arithmetic_compose_t : public arithmetic_t {
    /// Constructs every expression implementation
    ///
    /// @param e The emitter the expressions emit through
    explicit arithmetic_compose_t(emitter_t e)
        : macc_impl_(e)
        , zero_impl_(e)
        , mul_impl_(e)
        , add_impl_(e)
        , bitcast_impl_(e)
        , div_impl_(e) {}

protected:
    const arithmetic::expression_impl_t &macc_() const override {
        return macc_impl_;
    }
    const arithmetic::expression_impl_t &zero_() const override {
        return zero_impl_;
    }
    const arithmetic::expression_impl_t &mul_() const override {
        return mul_impl_;
    }
    const arithmetic::expression_impl_t &add_() const override {
        return add_impl_;
    }
    const arithmetic::expression_impl_t &bitcast_() const override {
        return bitcast_impl_;
    }
    const arithmetic::expression_impl_t &div_() const override {
        return div_impl_;
    }

    MaccOp macc_impl_;
    ZeroOp zero_impl_;
    MulOp mul_impl_;
    AddOp add_impl_;
    BitcastOp bitcast_impl_;
    DivOp div_impl_;
};

} // namespace operation
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
