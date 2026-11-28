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

#ifndef RVJIT_INTERFACE_LINALG_MATMUL_POSTOP_BUILTIN_HPP
#define RVJIT_INTERFACE_LINALG_MATMUL_POSTOP_BUILTIN_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "rvjit/interface/isa/operation/arithmetic.hpp"
#include "rvjit/interface/isa/operation/control_flow.hpp"
#include "rvjit/interface/isa/operation/memory.hpp"
#include "rvjit/interface/isa/pointer.hpp"
#include "rvjit/interface/isa/register.hpp"
#include "rvjit/interface/linalg/matmul/postop.hpp"
#include "rvjit/interface/stream/memory/input_stream.hpp"
#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace linalg {
namespace matmul {
namespace postop {

/// The bit pattern of `v` as a `dt` scalar
///
/// @param v The value
/// @param dt The scalar data type
/// @param bits Receives the bit pattern
///
/// @note Returns false if `dt` has no scalar form or the bits exceed 32
inline bool scalar_bits(double v, type_t dt, int64_t &bits) {
    switch (dt) {
        case type_t::f32: {
            const float f = static_cast<float>(v);
            uint32_t u;
            std::memcpy(&u, &f, sizeof(u));
            bits = static_cast<int32_t>(u);
            return true;
        }
        case type_t::f16: {
            // Round to nearest even, no subnormal results
            const float f = static_cast<float>(v);
            uint32_t u;
            std::memcpy(&u, &f, sizeof(u));
            const uint32_t sign = (u >> 16) & 0x8000u;
            const int exp = static_cast<int>((u >> 23) & 0xff) - 127 + 15;
            if (std::isnan(f)) {
                bits = sign | 0x7e00u;
            } else if (exp >= 31) {
                bits = sign | 0x7c00u;
            } else if (exp <= 0) {
                bits = sign;
            } else {
                uint32_t h = sign | static_cast<uint32_t>(exp) << 10
                        | (u & 0x7fffffu) >> 13;
                const uint32_t rest = u & 0x1fffu;
                if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) ++h;
                bits = h;
            }
            return true;
        }
        case type_t::f64: {
            uint64_t u;
            std::memcpy(&u, &v, sizeof(u));
            bits = static_cast<int64_t>(u);
            return u <= 0x7fffffffu;
        }
        default: {
            if (!rvjit::is_int(dt)) return false;
            bits = static_cast<int64_t>(v);
            return bits >= INT32_MIN && bits <= INT32_MAX;
        }
    }
}

/// How a scalar parameter folds at codegen time
enum class scalar_mode_t {
    skip, // Not planned yet
    zero, // A literal 0, which absorbs
    one, // A literal 1, the identity
    value, // Any other literal, or a runtime value
};

/// An integer type of the same width as `dt`
///
/// @param dt The data type
inline type_t bits_type(type_t dt) {
    switch (sewb(dt)) {
        case 1: return type_t::s8;
        case 2: return type_t::s16;
        case 4: return type_t::s32;
        default: return type_t::s64;
    }
}

/// A scalar held as bits in an integer register and, for a float type,
/// moved into a float register when it is used; a runtime scalar in memory is
/// instead loaded in its own type, straight into the register it is used from
struct scalar_t {
    static constexpr const char *NAME = "rvjit::linalg::matmul::postop";

    /// Registers a use needs from the scratch
    ///
    /// @param dt The scalar data type
    static needs_t needs(type_t dt) {
        needs_t n;
        n.ints = 1;
        n.floats = rvjit::is_int(dt) ? 0 : 1;
        return n;
    }

    /// The fold of a parameter with the identity `one` and absorbing `zero`
    ///
    /// @param p The parameter
    static scalar_mode_t classify(const scalar_parameter_t &p) {
        if (!p.has_literal()) return scalar_mode_t::value;
        if (p.literal() == 0.0) return scalar_mode_t::zero;
        if (p.literal() == 1.0) return scalar_mode_t::one;
        return scalar_mode_t::value;
    }

    /// Creates the bits variable on `x`, from the literal or runtime source
    ///
    /// @param env The environment of the matmul
    /// @param p The parameter
    /// @param x The register holding the bits
    ///
    /// @note A scalar in memory is not read here: `x` is only reserved for
    ///     its bits, which `bits` loads
    bool plan(postop_env_t &env, const scalar_parameter_t &p, const reg_t &x) {
        dt_ = env.plan.epi_type();
        in_memory_ = false;
        if (p.has_literal()) {
            int64_t bits = 0;
            CHECK(scalar_bits(p.literal(), dt_, bits),
                    "an unrepresentable literal scalar");
            var_ = env.pm.isa().buffers().new_variable(
                    source_t::immediate(static_cast<int>(bits)), x);
        } else if (p.source().is_load()) {
            ptr_ = p.source().memory();
            in_memory_ = true;
            var_ = env.pm.isa().buffers().new_variable(source_t::none(), x);
        } else {
            var_ = env.pm.isa().buffers().new_variable(p.source(), x);
        }
        CHECK(var_.is_valid(), "an invalid scalar parameter");
        float_ = !rvjit::is_int(dt_);
        return true;
    }

    /// The variable holding the bits, loaded in an integer register of the
    /// scalar's width when the scalar is in memory
    ///
    /// @param env The environment of the matmul
    ///
    /// @note A literal or register scalar is materialized by whatever uses
    ///     the variable
    const variable_t &bits(postop_env_t &env) const {
        if (!in_memory_) return var_;
        const auto load
                = env.pm.isa().memory().load(form_t::scalar, bits_type(dt_));
        load(reg_name(var_.buffer()), ptr_);
        return var_;
    }

    /// Returns the register holding the scalar, loading it unless `loaded`
    ///
    /// @param env The environment of the matmul
    /// @param f The float register a float scalar is held in
    /// @param loaded Whether the bits were loaded already
    ///
    /// @details A float scalar in memory is loaded into `f` in its own type
    name_t load(postop_env_t &env, const freg_t &f, bool loaded = false) const {
        if (in_memory_) {
            if (float_) {
                const auto load
                        = env.pm.isa().memory().load(form_t::scalar, dt_);
                load(reg_name(f), ptr_);
                return reg_name(f);
            }
            if (!loaded) bits(env);
            return reg_name(var_.buffer());
        }
        if (!loaded) env.pm.isa().transfer().init(var_);
        if (!float_) return reg_name(var_.buffer());
        const auto to_float = env.pm.isa().arithmetic().bitcast(
                {form_t::scalar, dt_}, {form_t::scalar, bits_type(dt_)});
        to_float(reg_name(f), reg_name(var_.buffer()));
        return reg_name(f);
    }

private:
    variable_t var_;
    pointer_t ptr_;
    type_t dt_ = type_t::f32;
    bool float_ = true;
    bool in_memory_ = false;
};

/// Scales the accumulators by alpha
///
/// @details A literal 0 clears them, so no Inf/NaN reaches C, and a literal 1
///     emits nothing
struct scale_t : public postop_t {
    int flag() const override { return with_alpha; }

    needs_t needs(const postop_shape_t &shape) const override {
        if (scalar_t::classify(shape.plan.alpha) != scalar_mode_t::value)
            return {};
        return scalar_t::needs(shape.plan.epi_type());
    }

    shortfall_t plan(postop_env_t &env) override {
        mode_ = scalar_t::classify(env.plan.alpha);
        if (mode_ != scalar_mode_t::value) return {};
        if (!alpha_.plan(env, env.plan.alpha, env.scratch.x(0)))
            return shortfall_t::of(resource_t::integers);
        return {};
    }

    bool emit(postop_tile_t &t) override {
        auto &pm = t.env.pm;
        const type_t dtc = t.env.plan.epi_type();
        const form_t form = t.env.c_form;
        if (mode_ == scalar_mode_t::one) return true;
        if (mode_ == scalar_mode_t::zero) {
            const auto zero = pm.isa().arithmetic().zero(form, dtc);
            t.for_each_acc([&](name_t acc, const index_t &) { zero(acc); });
            return true;
        }
        const name_t alpha = alpha_.load(t.env,
                t.env.scratch.floats.size() > 0 ? t.env.scratch.f(0)
                                                : freg_t());
        const auto scale = pm.isa().arithmetic().mul(
                {form, dtc}, {form_t::scalar, dtc}, {form, dtc});
        t.for_each_acc(
                [&](name_t acc, const index_t &) { scale(acc, alpha, acc); });
        return true;
    }

private:
    scalar_mode_t mode_ = scalar_mode_t::skip;
    scalar_t alpha_;
};

/// Adds beta times the old C to the accumulators
///
/// @details A literal 0 reads no C and a literal 1 adds it unscaled. A runtime
///     beta equal to 0 skips the C read too, as oneDNN's gemm kernel does
struct accumulate_t : public postop_t {
    int flag() const override { return with_beta; }

    needs_t needs(const postop_shape_t &shape) const override {
        needs_t n;
        const scalar_mode_t m = scalar_t::classify(shape.plan.beta);
        if (m == scalar_mode_t::zero) return n;
        n.accs = 1;
        n.accs_wanted = 2;
        n.ints = 1;
        if (m == scalar_mode_t::value)
            n.merge(scalar_t::needs(shape.plan.epi_type()));
        return n;
    }

    shortfall_t plan(postop_env_t &env) override {
        mode_ = scalar_t::classify(env.plan.beta);
        if (mode_ == scalar_mode_t::zero) return {};
        if (mode_ == scalar_mode_t::value
                && !beta_.plan(env, env.plan.beta, env.scratch.x(0)))
            return shortfall_t::of(resource_t::integers);

        // A ring of up to two registers lets a load overlap the previous add
        ring_ = std::min(2, env.scratch.accs.size());
        name_t ids[2];
        for (int i = 0; i < ring_; ++i)
            ids[i] = env.scratch.acc(i);
        const buffer::buffer_t regs(ids, ring_);

        // A narrower C is loaded into the upper part of the wide registers
        // and widened from there
        const type_t acc = env.plan.epi_type();
        const type_parameter_t c(env.plan.dtc.mem, acc);
        stream::memory_operand::variant_t variant
                = env.c_variant(buffer_map_t(regs, {env.nb, env.mb})
                                        .layout({env.mb, 1}, ring_));
        if (c.widens()) {
            const operand_t acc_operand(env.c_form, acc);
            variant.offset = c.offset(
                    env.pm.isa()
                            .buffers()
                            .allocator(acc_operand)
                            .group_size(buffer::allocation_t::scratch));
        }
        const int offset = variant.offset;
        load_ = env.create_c(env.c_spec(memory::dir_t::load,
                                     stream::iterator::update_t::preserve()),
                variant);
        if (!load_->is_valid()) return env.ms.miss();
        if (!in_.init(env.pm.typecast(), load_, c, offset))
            return shortfall_t::of(resource_t::integers);
        return {};
    }

    bool emit(postop_tile_t &t) override {
        if (mode_ == scalar_mode_t::zero) return true;
        auto &env = t.env;
        auto &pm = env.pm;
        const type_t dtc = env.plan.epi_type();

        // Shape the C ring to the dispatched tile
        in_.resize({t.nb, t.mb});
        // Positions load n-major, so the ring cycles in that order
        in_.regs().layout({t.mb, 1}, ring_);
        const int total = t.nb * t.mb;

        // Fold each loaded C granule into its accumulator, scaled by beta
        const auto body = [&](name_t beta) {
            const auto &arith = pm.isa().arithmetic();
            const form_t c = env.c_form;
            const bool scaled = mode_ == scalar_mode_t::value;
            const auto fold = scaled
                    ? arith.macc({c, dtc}, {form_t::scalar, dtc}, {c, dtc})
                    : arith.add(c, dtc);
            const auto at = [&](int i) { return index_t {i / t.mb, i % t.mb}; };
            const auto combine = [&](int i) {
                const index_t idx = at(i);
                const auto acc
                        = static_cast<name_t>(t.acc.get({idx[1], idx[0]}));
                if (scaled)
                    fold(acc, beta, in_.get(idx));
                else
                    fold(acc, in_.get(idx), acc);
            };
            if (in_.converts()) {
                // The loads of a ring's worth of positions run ahead of their
                // conversions; the folds follow in the accumulators'
                // configuration
                for (int base = 0; base < total; base += ring_) {
                    const int n = std::min(ring_, total - base);
                    in_.fetch_convert([&](int j) { return at(base + j); }, n);
                    for (int j = 0; j < n; ++j)
                        combine(base + j);
                }
                return;
            }

            // Load i, then fold i - 1, so a load overlaps the previous fold
            for (int i = 0; i < total; ++i) {
                in_.fetch(at(i));
                if (ring_ < 2) {
                    combine(i);
                } else if (i > 0) {
                    combine(i - 1);
                }
            }
            if (ring_ >= 2 && total > 0) combine(total - 1);
        };

        if (mode_ == scalar_mode_t::one) {
            body(no_name);
            return true;
        }
        const freg_t f
                = env.scratch.floats.size() > 0 ? env.scratch.f(0) : freg_t();
        pm.isa().control_flow().if_(control_flow::branch_t::nez(beta_.bits(env),
                                            control_flow::distance_t::short_),
                [&] { body(beta_.load(env, f, /*loaded=*/true)); });
        return true;
    }

private:
    scalar_mode_t mode_ = scalar_mode_t::skip;
    scalar_t beta_;
    memory_stream_t *load_ = nullptr;
    stream::memory_operand::input_stream_t in_;
    int ring_ = 1;
};

/// Adds the bias array to every column of the accumulators
///
/// @details The array holds `bias_type()` elements, converted to the
///     epilogue's type as they are loaded; a scalar bias is a single element
///     loaded once and added to every M granule
struct bias_add_t : public postop_t {
    int flag() const override { return with_bias; }

    needs_t needs(const postop_shape_t &shape) const override {
        needs_t n;
        const bool scalar = shape.plan.bias_kind == bias_kind_t::scalar;
        n.accs = n.accs_wanted = scalar ? 1 : shape.mb;
        n.ints = 1;
        n.m_offset = !scalar;
        return n;
    }

    shortfall_t plan(postop_env_t &env) override {
        const buffer::buffer_t regs = env.scratch.accs;

        // The tile's bias pointer is rebuilt from the base address and the
        // M offset, so only the base is read in place or loaded per tile
        base_ = env.pm.isa().buffers().new_constant(env.plan.bias.source());
        if (!base_.is_valid() || !base_.is_reg())
            return shortfall_t::of(resource_t::integers);
        ptr_ = env.scratch.x(0);

        // One axis of M granules, or a single element read for all of them
        const type_t epi = env.plan.epi_type(), mem = env.plan.bias_type();
        scalar_ = env.plan.bias_kind == bias_kind_t::scalar;
        rows_ = scalar_ ? 1 : env.mb;
        const int c_bytes = std::max(env.sewb_c, 1);
        num_ = sewb(mem);
        den_ = c_bytes;
        m_stride_ = env.c_m_stride;
        if (sewb(mem) != c_bytes) {
            m_stride_ = const_t(env.m_grain * sewb(mem));
            if (!env.pm.isa().buffers().allocate(m_stride_))
                return shortfall_t::of(resource_t::integers);
        }

        stream::memory_operand::spec_t spec;
        spec.op = memory::op_t::load(env.c_form, mem);
        if (scalar_) {
            spec.op.strided();
            spec.args = reg_name(reg_t::zero());
        }
        spec.origin = pointer_t(ptr_);
        spec.space = stream::iterator::space_t(
                {1, rows_}, {const_t(0), scalar_ ? const_t(0) : m_stride_});
        spec.update = stream::iterator::update_t::advance(0);

        // A narrower array is loaded into the upper part of the wide
        // registers and widened from there
        stream::memory_operand::variant_t variant;
        variant.ring = buffer_map_t(regs, {1, rows_}).layout({0, 1});
        const type_parameter_t b(mem, epi);
        if (b.widens()) {
            const operand_t epi_operand(env.c_form, epi);
            variant.offset = b.offset(
                    env.pm.isa()
                            .buffers()
                            .allocator(epi_operand)
                            .group_size(buffer::allocation_t::scratch));
        }
        const int offset = variant.offset;
        bias_ = env.create_c(spec, variant);
        if (!bias_->is_valid()) return env.ms.miss();
        if (!in_.init(env.pm.typecast(), bias_, b, offset))
            return shortfall_t::of(resource_t::integers);
        return {};
    }

    bool emit(postop_tile_t &t) override {
        auto &pm = t.env.pm;
        auto &ca = pm.isa().const_arithmetic();
        pm.isa().transfer().init(base_);
        const reg_t off = scalar_ ? reg_t::zero() : t.m_offset();
        const reg_t base = base_.buffer();
        if (is_not_zero(off) && num_ > den_) {
            // The M offset counts bytes of C, narrower than the array's
            ca.mul(ptr_, off, const_t(num_ / den_));
            ca.add(ptr_, base, const_t(ptr_), false);
        } else if (is_not_zero(off) && num_ < den_) {
            // The M offset counts bytes of C, wider than the array's
            ca.div(ptr_, off, const_t(den_ / num_));
            ca.add(ptr_, base, const_t(ptr_), false);
        } else {
            ca.add(ptr_, base, is_not_zero(off) ? const_t(off) : const_t(0),
                    ptr_ == base);
        }
        const int rows = scalar_ ? 1 : t.mb;
        in_.resize({1, rows});
        in_.fetch_convert([](int j) { return index_t {0, j}; }, rows);

        const auto add = pm.isa().arithmetic().add(
                t.env.c_form, t.env.plan.epi_type());
        t.for_each_acc([&](name_t acc, const index_t &mn) {
            add(acc, in_.get({0, scalar_ ? 0 : mn[0]}), acc);
        });
        return true;
    }

private:
    memory_stream_t *bias_ = nullptr;
    stream::memory_operand::input_stream_t in_;
    const_t base_; // The bias array's address
    const_t m_stride_; // Bytes between the array's M granules
    int num_ = 1, den_ = 1; // Bytes of an array element per bytes of C's
    int rows_ = 1;
    bool scalar_ = false;
    reg_t ptr_; // The tile's bias pointer
};

} // namespace postop
} // namespace matmul
} // namespace linalg
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
