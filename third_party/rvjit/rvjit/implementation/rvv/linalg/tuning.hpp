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

#ifndef RVJIT_IMPLEMENTATION_RVV_LINALG_TUNING_HPP
#define RVJIT_IMPLEMENTATION_RVV_LINALG_TUNING_HPP

#include <algorithm>
#include <functional>

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/implementation/rvv/rvv.hpp"
#include "rvjit/interface/isa/buffer/buffer.hpp"
#include "rvjit/interface/isa/buffer/buffer_pool.hpp"
#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/interface/linalg/matmul/needs.hpp"
#include "rvjit/interface/linalg/matmul/plan.hpp"
#include "rvjit/interface/linalg/matmul/tuned.hpp"
#include "rvjit/utils/utils.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace rvv {
namespace linalg {
namespace matmul {

/// What `tuning_t` decides for a plan: the generic decisions, an M granule
/// being one vector, and the vtype that realizes them
struct tuned_t : public rvjit::linalg::matmul::tuned_t {
    /// The SEW and LMUL of the inputs and the accumulators
    vtype_t vtype;
};

/// The RVV decisions about a matmul plan: vector grouping and block sizes
///
/// @details A function of the plan and the hardware alone, holding no emitter
///     or programming model, so a host can ask what a kernel for a plan will
///     do before it builds one. The blocks it returns, handed back in
///     `plan_t::blocks`, are honored as they are
struct tuning_t {
    static constexpr const char *NAME = "rvjit::rvv::linalg::matmul::tuning";

    /// Creates the tuning for a machine
    ///
    /// @param model The hardware model
    explicit tuning_t(const rvjit::rvv::vpu_t &model) : model_(model) {}

    /// Forces the input LMUL instead of the model-driven choice
    ///
    /// @param m The input LMUL
    ///
    /// @note The accumulator LMUL is derived from it; a widening plan whose
    ///     accumulator LMUL would exceed m8 is rejected, as is a plan with a
    ///     conversion narrower than a whole register group. A forced LMUL is
    ///     not bound by `max_lmul()`
    void set_lmul(LMUL m) {
        lmul_ = m;
        has_lmul_ = true;
    }

    /// The largest register group the model's own choice reaches
    ///
    /// @details The smallest group that spans 2048 bits, m8 when the vector
    ///     registers are too short for any
    LMUL max_lmul() const {
        for (LMUL m : {LMUL::m1, LMUL::m2, LMUL::m4}) {
            if (model_.vlenb * 8 * vgroup_size(m) >= vlong_bits_) return m;
        }
        return LMUL::m8;
    }

    /// The vector registers only inputs may draw, keeping `v0` (the mask
    /// register) away from accumulators and epilogue values
    static rvjit::buffer::buffer_t reserved_vectors() {
        const name_t v0 = 0;
        return rvjit::buffer::buffer_t(&v0, 1);
    }

    /// The plan as this backend runs it
    ///
    /// @param p The plan
    ///
    /// @details Integer data at most half as wide as integer accumulators of
    ///     its registers' type is multiplied at half their width, by a widening
    ///     multiply-accumulate (`vwmacc`, `vwmaccu`, `vwmaccsu`, `vwmaccus`):
    ///     A's and B's registers take the type of half the accumulators'
    ///     width their memory type widens to (`natural_wide`). s8 or u8 data
    ///     into s32 accumulators is thus extended to s16 or u16, not s32; into
    ///     s16 accumulators it is used as it is. Every other plan, and any
    ///     plan under a forced LMUL, is returned as it is
    ///
    /// @note Idempotent: the plan returned no longer accumulates uniformly
    rvjit::linalg::matmul::plan_t backend_plan(
            const rvjit::linalg::matmul::plan_t &p) const {
        const rvjit::type_t acc = p.dtc.reg;
        if (has_lmul_ || !rvjit::is_int(acc) || p.dta.reg != acc
                || p.dtb.reg != acc)
            return p;
        const int half = rvjit::sewb(acc) / 2;
        const auto narrow_int = [&](rvjit::type_t mem) {
            return rvjit::is_int(mem) && rvjit::sewb(mem) <= half;
        };
        if (half < 1 || !narrow_int(p.dta.mem) || !narrow_int(p.dtb.mem))
            return p;
        const auto widened = [&](rvjit::type_t mem) {
            rvjit::type_t t = mem;
            while (t != rvjit::type_t::none && rvjit::sewb(t) < half)
                t = rvjit::natural_wide(t);
            return t;
        };
        const rvjit::type_t a = widened(p.dta.mem), b = widened(p.dtb.mem);
        if (a == rvjit::type_t::none || b == rvjit::type_t::none) return p;
        rvjit::linalg::matmul::plan_t q = p;
        q.dta.reg = a;
        q.dtb.reg = b;
        return q;
    }

    /// How the vector register file must be grouped to run `given`
    ///
    /// @param given The plan, run as `backend_plan(given)`
    ///
    /// @details The input groups (A's pipeline buffers) and the scratch groups
    ///     (the epilogue's C and bias buffers) are the accumulator groups,
    ///     unless a value is loaded narrower than the type it is widened to
    ///     and takes a fraction of a register: it may not overlap the group
    ///     it is widened into, so the buffer takes room beside it. The
    ///     epilogue reuses A's buffers once the K loop is done, so they also
    ///     take the size of a scratch buffer larger than the accumulator
    ///     group, leaving it one aligned buffer to draw
    ///
    /// @note An invalid layout is returned when the plan has none
    rvjit::buffer::vector_layout_t vector_layout(
            const rvjit::linalg::matmul::plan_t &given) const {
        using namespace rvjit::linalg::matmul;
        const plan_t p = backend_plan(given);
        if (!accumulator_lmul_fits_(p.dta.reg, p.dtc.reg)) return {0, 0, {}};
        const vtype_t vt = vtype_of_(p);
        const int out = vgroup_size(vt.lmulo);

        // The registers of a buffer a value `ratio` times narrower widens
        // into a group of `group` registers in (`fraction` is the denominator
        // of a fractional group): the group when the narrow value takes whole
        // registers of it, else twice the group, or the ratio if larger, so
        // that the highest register the value is loaded into
        // (`type_parameter_t::offset`) lies beside the wide group
        const auto room = [](int group, int fraction, int ratio) {
            if (group % (fraction * ratio) == 0) return group;
            return std::max(2 * group, ratio);
        };

        int in = vgroup_size(vt.lmuli);
        if (p.dta.widens())
            in = room(in, lmul_fraction(vt.lmuli), p.dta.ratio());

        // C and the bias are loaded at their memory type and widened to the
        // epilogue's
        int scratch = out;
        const auto widen = [&](rvjit::type_t mem) {
            const rvjit::type_parameter_t t(mem, p.epi_type());
            if (t.widens())
                scratch = std::max(scratch, room(out, 1, t.ratio()));
        };
        if (p.flags & with_beta) widen(p.dtc.mem);
        if (p.flags & with_bias) widen(p.bias_type());

        if (scratch > out) in = std::max(in, scratch);

        return {in, out, reserved_vectors(), scratch == out ? 0 : scratch};
    }

    /// The SEW and LMUL pair `given`'s data types run at
    ///
    /// @param given The plan, run as `backend_plan(given)`
    ///
    /// @details Unless the LMUL is forced, the accumulator group is the
    ///     preference (m4, or m2 inputs widened) within `max_lmul()`. Every
    ///     value a conversion of `p` runs at takes the accumulators' vector
    ///     length, so it takes a whole number of registers of its own width
    ///     when the group is a multiple of the width ratio; the group is
    ///     raised, within `max_lmul()`, until that holds, and is kept at
    ///     `max_lmul()` otherwise, where the value takes a fraction of a
    ///     register. It is then narrowed while a bounded M fits the half
    ///     group and the conversions stay whole. A widening keeps its inputs
    ///     at m1 or more unless `max_lmul()` is below the widening ratio,
    ///     where they take the fraction of one register it leaves them
    ///
    /// @note No group exceeds `max_lmul()` unless the LMUL is forced
    vtype_t vtype_of(const rvjit::linalg::matmul::plan_t &given) const {
        return vtype_of_(backend_plan(given));
    }

    /// Decides the vector grouping and the blocks for `given`
    ///
    /// @param given The plan, run as `backend_plan(given)`; the blocks in
    ///     `given.blocks` are honored as they are or the plan is rejected
    /// @param pool The register pool, whose vector file is grouped as
    ///     `vector_layout(given)` says
    /// @param epilogue What the epilogue needs for the given M vectors, nothing
    ///     when empty
    /// @param out The decisions
    ///
    /// @note Returns false, logging the reason, when the plan has no valid
    ///     decision. The accumulators take the groups the pipeline leaves, at
    ///     most `nb_ceiling_` of them; the epilogue reuses the input groups,
    ///     `v0`'s included unless it needs the mask, and only a minimum that
    ///     does not fit shrinks them. An unpinned M takes at most `mb_max`
    ///     vectors, and is unrolled only when the integer registers left after
    ///     the epilogue's reach `m_unroll_min_ints`. When B's rows are far
    ///     apart (`strided_`) and a bounded M spans two tiles, it takes two tiles beside
    ///     at most four accumulators, as many as fit, unless the columns of a
    ///     call fit in one pass of that many
    bool operator()(const rvjit::linalg::matmul::plan_t &given,
            rvjit::buffer::buffer_pool_t &pool,
            const std::function<rvjit::linalg::matmul::needs_t(int)> &epilogue,
            tuned_t &out) const {
        const rvjit::linalg::matmul::plan_t p = backend_plan(given);
        CHECK(accumulator_lmul_fits_(p.dta.reg, p.dtc.reg),
                "an accumulator LMUL beyond m8");

        // Configure vtype according to the plan
        out = tuned_t();
        out.vtype = vtype_of_(p);
        const vtype_t &vt = out.vtype;
        const rvjit::buffer::vector_layout_t layout = vector_layout(given);

        // A forced LMUL runs every conversion at a width whose LMUL, scaled
        // from the accumulators' to keep the vector length, is a whole
        // register group. The model's own choice may leave one a fraction of
        // a register, which `vector_layout` gives room to
        if (has_lmul_) {
            const auto fits = [&](rvjit::type_t t) {
                const int group = vgroup_size(out.vtype.lmulo);
                const int from = rvjit::rvv::sewb(out.vtype.sewo);
                const int to = rvjit::sewb(t);
                return to >= from || group * to % from == 0;
            };
            CHECK(fits(p.dtc.mem), "an output narrower than a register group");
            CHECK(fits(p.epi_type()), "an epilogue type narrower than a group");
            CHECK(fits(p.dta.mem), "an A narrower than a register group");
            CHECK(!(p.flags & rvjit::linalg::matmul::with_bias)
                            || fits(p.bias_type()),
                    "a bias narrower than a register group");
        }

        // Derive the MAXVL and SEWB from the vtype configurations
        const int maxvl
                = get_maximum_vector_length(model_.vlenb, vt.sewi, vt.lmuli);
        const int sewib = rvjit::rvv::sewb(vt.sewi);
        const int sewob = rvjit::rvv::sewb(vt.sewo);
        out.m_grain = maxvl;

        const auto needs = [&](int mb) {
            return epilogue ? epilogue(mb) : rvjit::linalg::matmul::needs_t();
        };

        // Determine the K loop unrolling factor (reduce loop)
        // The empirical value 8 shows good results on different micro-arch
        static constexpr int kb_max = 8;
        out.kb = kb_max;
        if (p.blocks.kb > 0) {
            CHECK(p.blocks.kb <= kb_max, "a pinned K unroll beyond %d", kb_max);
            out.kb = p.blocks.kb;
        }

        // Determine the K loop unrolling factor for the N loop tail(s)
        // Use shorter unrolling to reduce code size on lower impact code paths
        static constexpr int ktb_max = 4;
        out.ktb = ktb_max;
        while (out.ktb > 1 && out.ktb >= out.kb)
            out.ktb /= 2;

        // The accumulator groups left by A's two pipeline buffers: two groups,
        // one when mixed precision or m8, and more when the buffers take room
        // for a narrow value beside them (`layout.in`). The epilogue reuses
        // the input groups once the K loop is done, keeping `v0`'s group clear
        // when it needs the mask; a scratch buffer larger than a group
        // (`layout.scratch`) counts as the groups it spans
        const int out_groups = get_nvgroups(vt.lmulo);
        const int group = vgroup_size(vt.lmulo);
        const int scratch_groups = std::max(layout.scratch, group) / group;
        const auto epilogue_groups = [&](int mb) {
            const rvjit::linalg::matmul::needs_t n = needs(mb);
            return n.accs * scratch_groups + (n.mask ? 1 : 0);
        };
        CHECK(out_groups - 1 >= epilogue_groups(1),
                "the epilogue leaves no vector register for the accumulators");
        const int spare = vt.is_uniform() && vt.lmulo == LMUL::m8
                ? 1
                : (2 * layout.in + group - 1) / group;
        const int ceiling
                = std::min(out_groups - spare, out_groups - epilogue_groups(1));
        const int max_acc = std::min(nb_ceiling_(p), std::max(1, ceiling));
        CHECK(p.blocks.nb <= max_acc,
                "a pinned N block of %d beyond the accumulator ceiling %d",
                p.blocks.nb, max_acc);
        out.nb = max_acc;
        if (p.blocks.nb > 0)
            out.nb = p.blocks.nb;
        else if (p.n.has_bound())
            out.nb = std::min(p.n.bound(), out.nb);

        // M: unroll as many vector tiles as M and the registers left after
        // N's accumulators and A's double buffer allow, up to `mb_max`; the
        // tile pointer steps must fold into one simm12 `addi`. A pinned count
        // is honored as is. An unbounded M is unrolled as a large bound is,
        // its tails left to the M loop
        static constexpr int mb_max = 4;
        const int m_step_max = 2047 / (maxvl * std::max(sewib, sewob));
        const bool m_pinned = p.blocks.mb > 0;
        CHECK(!m_pinned || p.blocks.mb <= std::max(m_step_max, 1),
                "a pinned M block of %d vectors beyond the pointer step range",
                p.blocks.mb);
        int m_ur = 1;
        if (m_pinned) {
            m_ur = p.blocks.mb;
        } else {
            const int m_tiles = p.m.has_bound() ? p.m.bound() / maxvl : mb_max;
            m_ur = std::min({m_tiles, m_step_max, mb_max});
        }

        // An unrolled M tile addresses each M-indexed stream through its own
        // pointer, which only kernels with integer registers to spare afford
        // (empirical, from the register-pressure sweeps)
        static constexpr int m_unroll_min_ints = 20;
        // A's buffers wider than the input group share the registers with the
        // accumulators, which the input and output views of the pool count
        // apart: each unrolled tile takes at least one
        const int a_group = layout.in;
        const bool wide_a = a_group > vgroup_size(vt.lmuli);
        const int nregs = get_nvgroups(LMUL::m1);
        const auto fits = [&](int nb, int mr) {
            const bool fits_pool = pool.vectors().may_reserve(nb * mr, 2)
                    && (!wide_a || nb * mr * group + mr * a_group <= nregs);
            const bool fits_epilogue
                    = nb * mr + epilogue_groups(mr) <= out_groups;
            const int ints = pool.integers().available() - needs(mr).ints;
            const bool fits_ints = m_pinned || ints >= m_unroll_min_ints;
            return fits_pool && fits_epilogue && fits_ints;
        };

        // The most accumulators beside two tiles of M
        static constexpr int nb_max_strided = 4;
        // The caller pinned neither the M nor the N block
        const bool unpinned = !m_pinned && p.blocks.nb == 0;
        // M is bounded and spans at least two tiles
        const bool m_spans_two = p.m.has_bound() && m_ur >= 2;
        // The columns of a call fit in one pass of the accumulators
        const bool one_pass = p.n.has_bound() && p.n.bound() <= nb_max_strided;
        // B's rows are far apart (`strided_`)
        const bool far_apart = strided_(p);
        // The vectors are shorter than 2048 bits
        const bool short_vectors = model_.vlenb * 8 < vlong_bits_;
        const bool two_tiles = unpinned && m_spans_two && !one_pass && far_apart
                && short_vectors;
        if (two_tiles) {
            m_ur = 2;
            for (int nb = std::min(out.nb, nb_max_strided); nb >= 1; --nb) {
                if (!fits(nb, m_ur)) continue;
                out.nb = nb;
                break;
            }
        }
        while (m_ur > 1) {
            if (fits(out.nb, m_ur)) break;
            CHECK(!m_pinned, "a pinned M block of %d vectors that does not fit",
                    p.blocks.mb);
            --m_ur;
        }
        out.mb = std::max(m_ur, 1);
        return true;
    }

private:
    // Vector length (bits) from which an LMUL group and the N block are capped
    static constexpr int vlong_bits_ = 2048;

    /// Whether the rows of a non-transposed B are far apart
    ///
    /// @param p The plan
    ///
    /// @details Each K step reads one scalar from each of B's rows, `ldb`
    ///     bytes apart. B is far apart when `ldb` spans at least 2048 elements
    ///     or has a power-of-two factor of at least 8 KiB. An unknown `ldb` is
    ///     not far apart
    bool strided_(const rvjit::linalg::matmul::plan_t &p) const {
        // The stride, in elements, from which B is far apart
        static constexpr int strided_ldb_elems = 2048;
        // The power-of-two factor of the stride, in bytes, from which it is
        static constexpr int aliased_ldb_factor_hard = 8192;
        // A transposed B or an unknown stride is not far apart
        if (p.is_transb || !p.ldb.is_literal()) return false;
        const int ldb = p.ldb.value();
        const int elem = rvjit::sewb(p.dtb.mem);
        if (ldb <= 0 || elem <= 0) return false;
        return ldb / elem >= strided_ldb_elems
                || (ldb & -ldb) >= aliased_ldb_factor_hard;
    }

    /// The SEW and LMUL pair `p` runs at, as `vtype_of` describes
    ///
    /// @param p The plan, as the backend runs it (`backend_plan`)
    vtype_t vtype_of_(const rvjit::linalg::matmul::plan_t &p) const {
        const rvjit::type_t dta = p.dta.reg, dtc = p.dtc.reg;
        const SEW sewi = sew_for(dta);
        const SEW sewo = sew_for(dtc);

        vtype_t vt;
        if (has_lmul_) {
            if (dta == dtc)
                vt.for_uniform(sewo, lmul_);
            else
                vt.for_mixed(sewi, sewo, lmul_);
            return vt;
        }

        // The accumulator group, in registers: the preference within the cap
        const int wo = rvjit::sewb(dtc);
        const int ratio = std::max(1, wo / rvjit::sewb(dta));
        const int cap = vgroup_size(max_lmul());
        static constexpr int uniform_group = 4, widening_input_group = 2;
        int g = std::min(
                dta == dtc ? uniform_group : widening_input_group * ratio, cap);

        // Whether every conversion takes whole registers of a group of `n`
        const auto whole = [&](int n) {
            const auto fits = [&](rvjit::type_t t) {
                const int to = rvjit::sewb(t);
                return to >= wo || n * to % wo == 0;
            };
            return fits(p.dtc.mem) && fits(p.epi_type()) && fits(p.dta.mem)
                    && (!(p.flags & rvjit::linalg::matmul::with_bias)
                            || fits(p.bias_type()));
        };
        while (g < cap && !whole(g))
            g *= 2;

        // A bounded M that half the group still holds leaves lanes idle
        const int m_bound = p.m.bound();
        while (m_bound > 0 && g / 2 >= ratio && whole(g / 2)
                && model_.vlenb * (g / 2) / wo >= m_bound)
            g /= 2;

        if (dta == dtc)
            vt.for_uniform(sewo, lmul_for(g));
        else
            vt.for_mixed(sewi, sewo, lmul_for(g, ratio));
        return vt;
    }

    /// The most N columns (accumulator groups) a tile of `p` unrolls
    ///
    /// @param p The plan
    ///
    /// @details A non-transposed B reads one scalar from each of its N rows,
    ///     `ldb` bytes apart, per K step. When `ldb` is a multiple of a large
    ///     power of two, those rows fall in the same cache sets, and more of
    ///     them than the cache's ways evict each other: on a 1024-bit A100,
    ///     14 rows at a stride with a power-of-two factor of 4 KiB run at a
    ///     third of their rate or below, where 6 or 7 rows lose no more than
    ///     the shape costs anyway. The tile then takes at most
    ///     `nb_max_aliased` columns, as it does when `ldb` is only known at
    ///     run time; otherwise `nb_max`, or `nb_max_aliased` from a VLEN of
    ///     2048 bits
    int nb_ceiling_(const rvjit::linalg::matmul::plan_t &p) const {
        static constexpr int nb_max = 12, nb_max_aliased = 8;
        // The largest power-of-two factor of `ldb`, in bytes, B's rows are
        // read at without aliasing
        static constexpr int aliased_ldb_factor = 2048;
        if (model_.vlenb * 8 >= vlong_bits_) return nb_max_aliased;
        if (p.is_transb) return nb_max;
        if (!p.ldb.is_literal()) return nb_max_aliased;
        const int ldb = p.ldb.value();
        const int pow2 = ldb & -ldb;
        return pow2 > aliased_ldb_factor ? nb_max_aliased : nb_max;
    }

    /// Whether the accumulators of `dta` x `dtc` fit a group of at most m8
    ///
    /// @details The model's own choice never exceeds `max_lmul`; a forced
    ///     input LMUL widens into `sewb(dtc) / sewb(dta)` times its group,
    ///     which `lmul_for` would otherwise silently map to m1
    bool accumulator_lmul_fits_(rvjit::type_t dta, rvjit::type_t dtc) const {
        if (dta == dtc || !has_lmul_) return true;
        const int in = vgroup_size(lmul_);
        const int out = in * (rvjit::sewb(dtc) / rvjit::sewb(dta));
        CHECK(out <= vgroup_size(LMUL::m8),
                "a widening accumulator group of %d registers (input group "
                "%d) beyond m8",
                out, in);
        return true;
    }

    rvjit::rvv::vpu_t model_;
    LMUL lmul_ = LMUL::m1; // Forced input LMUL, if has_lmul_
    bool has_lmul_ = false;
};

} // namespace matmul
} // namespace linalg
} // namespace rvv
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
