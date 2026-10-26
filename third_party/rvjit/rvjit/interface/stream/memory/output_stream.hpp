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

#ifndef RVJIT_INTERFACE_STREAM_MEMORY_OUTPUT_STREAM_HPP
#define RVJIT_INTERFACE_STREAM_MEMORY_OUTPUT_STREAM_HPP

#include <memory>
#include <utility>

#include "rvjit/implementation/common/stream/interleave.hpp"
#include "rvjit/interface/core/index.hpp"
#include "rvjit/interface/isa/operation/conversion.hpp"
#include "rvjit/interface/isa/parameter.hpp"
#include "rvjit/interface/stream/memory/conversion_stream.hpp"
#include "rvjit/interface/stream/memory/memory_operand.hpp"
#include "rvjit/interface/stream/scheduler.hpp"
#include "rvjit/interface/stream/stream_impl.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace stream {
namespace memory_operand {

/// A stream storing accumulators, converting them first when C's type differs
///
/// @details It drives a store memory operand whose ring holds the
///     accumulators, which convert in place (see `conversion_stream_t`). Every
///     step but the last runs over all the positions first, each step in its
///     own state; the last is interleaved with the stores, `delay`
///     positions ahead, so that an independent conversion separates a store
///     from the instruction producing its data
///
/// @note Each position owns its register and converts in place, so staggering
///     them needs no extra register
///
/// @note The stores run in the state of the last step; the code around the
///     stream runs in the state of the accumulators' type, which the typecast
///     is hinted with
///
/// @pre The caller owns the store stream
struct output_stream_t : public stream_impl_t {
    static constexpr const char *NAME = "rvjit::stream::memory_operand::output";

    output_stream_t() = default;

    /// A stream that is never valid
    static output_stream_t *invalid() {
        static output_stream_t none;
        return &none;
    }

    /// Configures the stream over a store operand
    ///
    /// @param tc An empty typecast, which the stream configures and owns
    /// @param sched The scheduler resolving the delay
    /// @param store The store operand, which the stream drives
    /// @param t The stored type (the operand's) and the type of the
    ///     accumulators held by the store's ring
    /// @param pol The policy of the delay between a conversion and its store
    ///
    /// @note Returns false for a type pair the typecast does not convert
    bool init(std::unique_ptr<typecast::typecast_t> tc,
            const scheduler_t &sched, memory_operand_impl_t *store,
            const type_parameter_t &t,
            const policy_t &pol = policy_t::arithmetic()) {
        CHECK(store && store->is_valid(), "invalid store operand");
        CHECK(store->op().dir == memory::dir_t::store, "not a store operand");
        const memory::op_t &op = store->op();
        CHECK(op.data.type == t.mem, "a store of another type");
        CHECK(!t.narrows(), "a store wider than the accumulators");
        CHECK(tc, "no typecast");
        const operand_t reg(op.data.form, t.reg);
        const conversion::op_t cast(operand_t(op.data.form, t.mem), reg);
        CHECK(tc->configure(cast, reg),
                "unsupported conversion of the accumulators");
        CHECK(conv_.init(std::move(tc), store), "invalid conversion");
        sched_ = sched;
        pol_ = pol;
        store_ = store;
        return true;
    }

    /// Whether the stored type differs from the accumulators'
    bool converts() const { return conv_.converts(); }

    int ndims() const override { return store_->ndims(); }
    int size(int d) const override { return store_->size(d); }
    bool is_valid() const override { return store_ && store_->is_valid(); }
    bool resize(const index_t &shape) override {
        CHECK(is_valid(), "invalid stream");
        return store_->resize(shape);
    }

    /// The accumulator at `idx`
    name_t get(const index_t &idx) const override { return store_->get(idx); }
    const buffer::buffer_t &buffer() const override { return store_->buffer(); }

    /// Emits the conversion of position `idx`, nothing when C has the
    /// accumulators' type
    void convert(const index_t &idx) { conv_.iteration(idx); }

    /// Emits the store of position `idx`
    void store(const index_t &idx) { store_->iteration(idx); }

    /// Emits the conversions and the stores of every position, the last step
    /// of the conversions `delay` positions ahead
    ///
    /// @note The delay is the policy's, up to the number of positions
    void emit() override {
        if (!is_valid()) return;
        if (!conv_.converts()) {
            store_->emit();
            return;
        }
        const int last = conv_.typecast().nsteps() - 1;
        conv_.emit_steps(0, last);
        conv_.configure(last);
        conv_.select(last);
        const common::stream::interleave_t interleave(sched_);
        interleave(
                pol_, [&](const index_t &idx) { store(idx); }, {conv_.ref()});
        conv_.select(-1);
        conv_.restore();
    }

protected:
    void produce_(const index_t &idx) override {
        convert(idx);
        store(idx);
    }

private:
    conversion_stream_t conv_;
    scheduler_t sched_;
    policy_t pol_;
    memory_operand_impl_t *store_ = nullptr;
};

} // namespace memory_operand
} // namespace stream
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
