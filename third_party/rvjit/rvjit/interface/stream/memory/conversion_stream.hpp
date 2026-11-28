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

#ifndef RVJIT_INTERFACE_STREAM_MEMORY_CONVERSION_STREAM_HPP
#define RVJIT_INTERFACE_STREAM_MEMORY_CONVERSION_STREAM_HPP

#include <memory>
#include <utility>

#include "rvjit/interface/core/index.hpp"
#include "rvjit/interface/stream/stream_impl.hpp"
#include "rvjit/interface/typecast/typecast.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace stream {
namespace memory_operand {

/// A stream converting the values another stream's registers hold
///
/// @details Each position converts the register of the same position of the
///     `regs` stream through a typecast. A narrowing (or same-width) typecast
///     runs in place; a widening one reads the narrow value `offset`
///     registers above the wide register it fills, the highest part of its
///     group, and continues in place. A position emits the selected step, or
///     every step
///
/// @note The stream tracks the typecast position whose state is in effect,
///     starting from the state hint's, and walks forward from it through the
///     cycle hint, step 0, ..., hint: `configure(s)` interposes when a
///     position on the way to step `s` does, and `restore()` returns to the
///     hint's state the same way
struct conversion_stream_t : public stream_impl_t {
    static constexpr const char *NAME
            = "rvjit::stream::memory_operand::conversion";

    conversion_stream_t() = default;

    /// Configures the stream over the registers of `regs`
    ///
    /// @param tc The typecast, configured and valid, which the stream owns
    /// @param regs The stream whose registers hold the values
    /// @param offset The registers between a wide register and the narrow
    ///     part a widening reads, 0 unless the typecast widens
    bool init(std::unique_ptr<typecast::typecast_t> tc, stream_impl_t *regs,
            int offset = 0) {
        CHECK(regs, "no register stream");
        CHECK(tc && tc->is_valid(), "an invalid typecast");
        const bool widens = !tc->empty() && tc->dst().width > tc->src().width;
        CHECK(!widens || offset > 0,
                "a widening needs the upper part of a group");
        tc_ = std::move(tc);
        regs_ = regs;
        offset_ = widens ? offset : 0;
        at_ = tc_->nsteps();
        step_ = -1;
        return true;
    }

    /// The typecast
    ///
    /// @pre The stream is initialized
    const typecast::typecast_t &typecast() const { return *tc_; }

    /// Whether there is anything to convert
    bool converts() const { return tc_ && !tc_->empty(); }

    /// Makes `iteration()` emit step `s` only, or every step for -1
    ///
    /// @param s The step
    void select(int s) { step_ = s; }

    /// Enters the state of step `s`, interposing when it differs
    ///
    /// @param s The step
    bool configure(int s) { return enter_(s); }

    /// Returns to the state hint's state, interposing when it differs
    bool restore() { return converts() ? enter_(tc_->nsteps()) : true; }

    /// Emits steps `first` up to `last` (excluded) of every position, step by
    /// step, each in its own state
    ///
    /// @param first The first step
    /// @param last The step after the last one
    void emit_steps(int first, int last) {
        for (int s = first; s < last; ++s) {
            configure(s);
            select(s);
            stream_impl_t::emit();
        }
        select(-1);
    }

    /// Emits steps `first` up to `last` (excluded) of `n` positions, step by
    /// step, each in its own state
    ///
    /// @param first The first step
    /// @param last The step after the last one
    /// @param at The callable giving the `j`-th position, `index_t(int)`
    /// @param n The number of positions
    template <typename At>
    void emit_steps(int first, int last, const At &at, int n) {
        for (int s = first; s < last; ++s) {
            configure(s);
            select(s);
            for (int j = 0; j < n; ++j)
                iteration(at(j));
        }
        select(-1);
    }

    int ndims() const override { return regs_->ndims(); }
    int size(int d) const override { return regs_->size(d); }
    bool is_valid() const override { return regs_ && regs_->is_valid(); }
    bool resize(const index_t &shape) override { return regs_->resize(shape); }

    /// The register holding the converted value at `idx`
    name_t get(const index_t &idx) const override {
        return static_cast<name_t>(regs_->get(idx) - offset_);
    }

    const buffer::buffer_t &buffer() const override { return regs_->buffer(); }

protected:
    void produce_(const index_t &idx) override {
        const int first = step_ < 0 ? 0 : step_;
        const int last = step_ < 0 ? tc_->nsteps() : step_ + 1;
        const name_t wide = get(idx);
        for (int s = first; s < last; ++s) {
            const name_t src = s == 0 ? regs_->get(idx) : wide;
            tc_->emit(s, wide, src);
        }
    }

private:
    std::unique_ptr<typecast::typecast_t> tc_;
    stream_impl_t *regs_ = nullptr;
    int offset_ = 0;
    int at_ = 0; // The position in effect, `nsteps()` for the hint's state
    int step_ = -1;

    // Whether a position interposes on the way from `at_` to `s`, walking
    // forward through the cycle hint, 0, ..., nsteps() (the hint again)
    bool changes_(int s) const {
        const int n = tc_->nsteps() + 1;
        for (int k = (at_ + 1) % n;; k = (k + 1) % n) {
            if (tc_->interposes(k)) return true;
            if (k == s) return false;
        }
    }

    bool enter_(int s) {
        if (!converts() || s == at_) return true;
        if (changes_(s)) CHECK(tc_->interpose(s), "no state of the step");
        at_ = s;
        return true;
    }
};

} // namespace memory_operand
} // namespace stream
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
