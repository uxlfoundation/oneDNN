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

#ifndef RVJIT_INTERFACE_TYPECAST_TYPECAST_HPP
#define RVJIT_INTERFACE_TYPECAST_TYPECAST_HPP

#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/operation/conversion.hpp"
#include "rvjit/interface/isa/types.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace typecast {

/// The steps converting a value between two arbitrary types
///
/// @details `configure` plans a chain of conversion steps, each one
///     expression of a conversion dialect, and validates every step against
///     it. A widening chain changes width at most once, in its first step,
///     and then converts in place; an integer growth is an extension at every
///     ratio, a doubling included; a narrowing one converts in place first and
///     then halves its width in place, step by step. Intermediate operands
///     take the destination's form
///
///     An implementation may need instructions around the steps that are not
///     conversions, such as a change of the machine state a step runs in. It
///     announces them through `interposes(i)`, for the positions
///     `i = 0 .. nsteps()`: position `i < nsteps()` comes before step `i`, and
///     position `nsteps()` follows the last step, returning to the state the
///     caller named by the `state_hint` of `configure`. A caller emitting the
///     steps of one or many values
///
///     1. calls `interpose(i)` before the first emission of step `i`, when
///        `interposes(i)` is true,
///     2. emits step `i` of every value through `emit(i, ...)`,
///     3. calls `interpose(nsteps())`, when `interposes(nsteps())` is true,
///        once nothing else needs the last step's state
///
///     Whatever the caller emits between two positions runs in that state:
///     the loads of the converted values may run in the state of step 0
///     before the steps, and their stores in the state of the last step
///     before position `nsteps()`. `interpose(i)` enters the state of
///     position `i` from any other, so a caller skipping positions applies
///     only the last of the interposing ones it skips. `emit(dst, src)`
///     follows these steps for a single value
///
/// @note An empty typecast converts nothing: the types match, and nothing is
///     interposed
struct typecast_t {
    static constexpr const char *NAME = "rvjit::typecast";
    static constexpr int CAPACITY = 4;

    /// One step of a typecast
    struct step_t {
        /// The conversion expression a step emits
        enum kind_t {
            narrow, // A type half as wide, saturating integers
            widen, // A type twice as wide
            extend, // An integer 2x, 4x or 8x as wide, by sign or zeros
            cast, // Between an integer and a float of one width
        };

        kind_t kind = cast;
        conversion::op_t op;
    };

    /// Creates an empty, invalid typecast over a conversion dialect
    ///
    /// @param cvt The dialect every step is validated and emitted with
    ///
    /// @pre `cvt` outlives the typecast
    explicit typecast_t(const operation::conversion_t &cvt) : cvt_(&cvt) {}

    virtual ~typecast_t() = default;

    /// Plans the steps converting `op.src` into `op.dst`
    ///
    /// @param op The conversion, with no register names
    /// @param state_hint An operand describing the state the caller runs in
    ///     around the typecast, which the last interposition returns to;
    ///     absent when the caller names none
    ///
    /// @note Returns false, leaving the typecast invalid, when no chain of
    ///     steps the dialect supports converts the pair; an implementation
    ///     may also require a `state_hint`
    virtual bool configure(const conversion::op_t &op,
            const operand_t &state_hint = absent_operand()) {
        clear_();
        op_ = op;
        state_hint_ = state_hint;
        if (op.src.type == op.dst.type) {
            valid_ = true;
            return true;
        }

        // Two types of one width and family, e.g. s8 and u8, need a step too
        const bool ok
                = plan_(op.src.type, op.dst.type) && n_ > 0 && valid_steps_();
        if (!ok) clear_();
        CHECK(ok, "no typecast");
        valid_ = true;
        return true;
    }

    /// Whether the last `configure` succeeded
    bool is_valid() const { return valid_; }

    /// Whether there is nothing to convert
    bool empty() const { return n_ == 0; }

    /// The number of steps
    int nsteps() const { return n_; }

    /// The step `i`
    ///
    /// @param i The step's index
    const step_t &step(int i) const { return steps_[i]; }

    /// The operand converted
    const operand_t &src() const { return op_.src; }

    /// The operand produced
    const operand_t &dst() const { return op_.dst; }

    /// The state hint of the last `configure`
    const operand_t &state_hint() const { return state_hint_; }

    /// How many times wider the widest type is than the narrowest one
    int ratio() const {
        if (!n_) return 1;
        const int a = op_.src.width, b = op_.dst.width;
        return a > b ? a / b : b / a;
    }

    /// The expression of step `i`, called with its destination and source
    ///
    /// @param i The step's index
    conversion::expression_t expression(int i) const {
        const conversion::op_t &op = steps_[i].op;
        switch (steps_[i].kind) {
            case step_t::narrow: return cvt_->narrow(op);
            case step_t::widen: return cvt_->widen(op);
            case step_t::extend: return cvt_->extend(op);
            default: return cvt_->cast(op);
        }
    }

    /// Emits step `i` from the register `src` into `dst`
    ///
    /// @param i The step's index
    /// @param dst The destination register
    /// @param src The source register
    ///
    /// @pre The typecast is valid, and the caller is in the state of step `i`
    void emit(int i, name_t dst, name_t src) const { expression(i)(dst, src); }

    /// Emits every step and interposition converting the value in `src`
    /// into `dst`
    ///
    /// @param dst The destination register
    /// @param src The source register
    ///
    /// @details Interposes at each position that `interposes` and emits each
    ///     step after its position, from the register the step before wrote.
    ///     A step writes into `dst` when its result lives in `dst`'s register
    ///     file (vector, scalar float or scalar integer), else in place in
    ///     `src`, e.g. the extension before a scalar integer-to-float cast
    ///
    /// @note Returns false, having emitted the positions before, when the
    ///     typecast is invalid or an interposition fails
    /// @note An empty typecast emits nothing and leaves `dst` unwritten
    ///
    /// @pre The caller runs in the state of `state_hint`, which the last
    ///     interposition returns to
    /// @pre `src` may be overwritten, and both registers satisfy every step's
    ///     expression, e.g. the overlap rules of a vector widening
    bool emit(name_t dst, name_t src) const {
        if (!valid_) return false;
        name_t from = src;
        for (int i = 0; i < n_; ++i) {
            if (interposes(i) && !interpose(i)) return false;
            const name_t to
                    = same_file_(steps_[i].op.dst, op_.dst) ? dst : from;
            emit(i, to, from);
            from = to;
        }
        return !interposes(n_) || interpose(n_);
    }

    /// Whether instructions are interposed at position `i`
    ///
    /// @param i The position, from 0 to `nsteps()`
    virtual bool interposes(int i) const {
        (void)i;
        return false;
    }

    /// Emits the instructions interposed at position `i`
    ///
    /// @param i The position, from 0 to `nsteps()`
    ///
    /// @note Returns false when the instructions could not be emitted
    virtual bool interpose(int i) const {
        (void)i;
        return true;
    }

    /// An operand naming no state, the default `state_hint`
    static operand_t absent_operand() {
        return operand_t(rvjit::form_t::undefined, type_t::none);
    }

protected:
    /// Forgets every step, leaving the typecast invalid
    void clear_() {
        n_ = 0;
        valid_ = false;
    }

private:
    const operation::conversion_t *cvt_;
    step_t steps_[CAPACITY];
    int n_ = 0;
    bool valid_ = false;
    conversion::op_t op_;
    operand_t state_hint_ = absent_operand();

    // Whether two operands live in the same register file
    static bool same_file_(const operand_t &a, const operand_t &b) {
        if (a.is_vector() || b.is_vector())
            return a.is_vector() && b.is_vector();
        return a.is_float() == b.is_float();
    }

    void push_(step_t::kind_t k, type_t from, type_t to) {
        step_t &s = steps_[n_++];
        s.kind = k;
        s.op = conversion::op_t(operand_t(op_.dst.form, to),
                operand_t(n_ == 1 ? op_.src.form : op_.dst.form, from));
        s.op.predicate = op_.predicate;
    }

    // The integer type of `width` bytes and the signedness asked for
    static type_t int_of_(int width, bool sign) {
        switch (width) {
            case 1: return sign ? type_t::s8 : type_t::u8;
            case 2: return sign ? type_t::s16 : type_t::u16;
            case 4: return sign ? type_t::s32 : type_t::u32;
            case 8: return sign ? type_t::s64 : type_t::u64;
            default: return type_t::none;
        }
    }

    bool plan_(type_t from, type_t to) {
        const int wf = sewb(from), wt = sewb(to);
        CHECK(wf > 0 && wt > 0, "a typecast of an unsized type");
        const bool fi = is_int(from), ti = is_int(to);

        // Floats change width by two at most
        if (!fi && !ti) {
            CHECK(wf == 2 * wt || wt == 2 * wf, "a float pair beyond 2x");
            push_(wt > wf ? step_t::widen : step_t::narrow, from, to);
            return true;
        }

        // Integers grow once, by an extension, which runs in the destination's
        // configuration as the steps around it do; they shrink by halves
        if (fi && ti) {
            if (wt > wf) {
                push_(step_t::extend, from, to);
                return true;
            }
            const bool sign = is_signed_int(from);
            CHECK(sign == is_signed_int(to) || wt == wf,
                    "a narrowing that changes signedness");
            return narrow_ints_(from, to);
        }

        // An integer reaches the float's width first, then casts
        if (fi) {
            CHECK(wt >= wf, "an integer wider than its float");
            const type_t at = int_of_(wt, is_signed_int(from));
            CHECK(at != type_t::none, "no integer of the float's width");
            if (at != from) push_(step_t::extend, from, at);
            push_(step_t::cast, at, to);
            return true;
        }

        // A float casts at its width, then halves down to the integer's
        CHECK(wf >= wt, "an integer wider than the float converted");
        const type_t at = int_of_(wf, is_signed_int(to));
        CHECK(at != type_t::none, "no integer of the float's width");
        push_(step_t::cast, from, at);
        return narrow_ints_(at, to);
    }

    bool narrow_ints_(type_t from, type_t to) {
        const bool sign = is_signed_int(to);
        type_t cur = from;
        while (sewb(cur) > sewb(to)) {
            const type_t next = int_of_(sewb(cur) / 2, sign);
            CHECK(n_ < CAPACITY, "too many steps");
            push_(step_t::narrow, cur, next);
            cur = next;
        }
        return true;
    }

    bool valid_steps_() const {
        for (int i = 0; i < n_; ++i)
            CHECK(expression(i).is_valid(), "an unsupported step");
        return true;
    }
};

} // namespace typecast
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
