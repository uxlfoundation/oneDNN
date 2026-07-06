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

#ifndef CPU_RV64_RVJIT_RVJIT_MATMUL_HPP
#define CPU_RV64_RVJIT_RVJIT_MATMUL_HPP

#include "cpu/rv64/rvjit/rvjit_arithmetic.hpp"
#include "cpu/rv64/rvjit/rvjit_const_folding.hpp"
#include "cpu/rv64/rvjit/rvjit_control_flow.hpp"
#include "cpu/rv64/rvjit/rvjit_memory_move.hpp"
#include "cpu/rv64/rvjit/rvjit_register_pool.hpp"
#include "cpu/rv64/rvjit/rvjit_rvv.hpp"
#include "cpu/rv64/rvjit/rvjit_types.hpp"
#include "cpu/rv64/rvjit/rvjit_utils.hpp"

#if defined(RVJIT_DEBUG)
#include "common/verbose.hpp"
#define DEBUg(...) \
    do { \
        if (get_verbose(verbose_t::debuginfo) > 1) { __VA_ARGS__ } \
    } while (0)
#else
#define DEBUg(...)
#endif
#define DEBUG(...) DEBUg(__VA_ARGS__)

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace rvjit {

#if XBYAK_RISCV_V

/// Leading dimension strides for matrix operands
///
/// @details Differentiate between compile-time constants and runtime values
struct matmul_strides_t {
    struct entry_t {
        bool prematerialized = false;
        int64_t bytes = 0;
        const_t value;

        static entry_t bytes_of(int64_t b) {
            entry_t e;
            e.bytes = b;
            return e;
        }

        static entry_t materialized(const const_t &c) {
            entry_t e;
            e.prematerialized = true;
            e.value = c;
            return e;
        }
    };

    entry_t lda, ldb, ldc;

    static matmul_strides_t from_bytes(int64_t lda, int64_t ldb, int64_t ldc) {
        return matmul_strides_t {entry_t::bytes_of(lda), entry_t::bytes_of(ldb),
                entry_t::bytes_of(ldc)};
    }

    static matmul_strides_t from_regs(
            const Reg &lda, const Reg &ldb, const Reg &ldc) {
        return matmul_strides_t {entry_t::materialized(const_t(lda)),
                entry_t::materialized(const_t(ldb)),
                entry_t::materialized(const_t(ldc))};
    }
};

/// Declarative request for `rvv_matmul_engine_t::configure`
struct matmul_plan_t {
    optype_t dti = optype_t::f32;
    optype_t dto = optype_t::f32;
    optype_t dtb = optype_t::f32;

    Reg ptra {};
    Reg ptrb {};
    Reg ptrc {};

    bool is_transa = false;
    bool is_transb = false;

    matmul_strides_t strides;

    loop_t n_loop;
    loop_t k_loop;

    // If not set (0), unrolling factors are determined by an optimizer
    int max_n_ur = 0;
    int k_ur = 0;
    int k_ur_tail = 0;

    // Vector length (AVL)
    runtime_value_t avl;
};

/// Callback invoked after the matmul dense loops
using on_matmul_postop_t = callback_t<v_block_t, v_block_t>;

/// RVV matmul code generator, driven by a declarative `matmul_plan_t`
class rvv_matmul_engine_t {
public:
    explicit rvv_matmul_engine_t(emitter_t e, register_pool_t &pool,
            memory_move_t &mem, rvv_arithmetic_t &mc, const_folding_t &ca,
            control_flow_t &cf, const rvv_t &model)
        : e_(e)
        , pool_(pool)
        , mem_(mem)
        , mc_(mc)
        , ca_(ca)
        , cf_(cf)
        , model_(model) {}

    /// @pre configure() has been called
    bool is_mixed_precision() const { return plan_.dti != plan_.dto; }

    /// @pre configure() has returned true
    const loop_t &n_loop() const { return n_loop_cfg_; }
    /// @pre configure() has returned true
    const loop_t &k_loop() const { return k_loop_cfg_; }
    /// @pre configure() has returned true
    const const_t &ldc() const { return ldc_; }
    /// @pre configure() has returned true
    const const_t &ldb() const { return ldb_; }

    /// Sizes N/K unrolling, materializes strides, allocates the float/vector
    /// register files, resolves the N/K loops, and configures macc state;
    /// returns whether the plan was valid
    bool configure(const matmul_plan_t &plan) {
        // Determine SEW settings
        const SEW sewi = sew_for(plan.dti);
        const SEW sewo = sew_for(plan.dto);
        const bool is_mixed = plan.dti != plan.dto;

        // Determine LMUL settings
        LMUL m = is_mixed ? LMUL::m2 : LMUL::m4;
        if (is_lmul_gte(m, model_.vpu.max_lmul)) m = model_.vpu.max_lmul;
        const LMUL m_acc = lmul_for(m, sewi, sewo);

        // Determine n-loop unrolling
        const int ceiling = model_.vpu.max_n_accumulators(sewi, sewo);
        const int n_ur = plan.max_n_ur > 0 ? std::min(plan.max_n_ur, ceiling)
                                           : ceiling;
        loop_t n_loop = plan.n_loop;
        n_loop.unrolling(n_ur);
        pool_.new_loop(n_loop);

        // Determine k-loop unrolling
        const int k_ur = plan.k_ur > 0 ? plan.k_ur : 8;
        const int k_ur_tail = plan.k_ur_tail > 0 ? plan.k_ur_tail : 4;
        loop_t k_loop = plan.k_loop;
        k_loop.unrolling({k_ur, 1});
        pool_.new_loop(k_loop);

        // Allocate (but do not yet write) stride constants that don't fit a
        // 12-bit immediate. Every write is deferred to generate()
        //
        // `ldb` additionally needs special handling: when B isn't
        // transposed, load_b_'s Case A/B addressing can bake
        // `ni * ldb + (ki - kb) * elemsize` into a single immediate (ni in
        // [0, n_ur), ki-kb in (-k_ur, 0]), so its overflow bound must cover
        // both the N and K contributions, not just ldb's own byte value.
        const const_t lda = allocate_stride_(plan.strides.lda);
        const const_t ldc = allocate_stride_(plan.strides.ldc);
        const bool ldb_overflows = !plan.is_transb
                && !plan.strides.ldb.prematerialized
                && !is_simm12(
                        n_ur * plan.strides.ldb.bytes + k_ur * sewb(plan.dtb));
        Reg ldb_reg {};
        bool ldb_ok = true;
        if (ldb_overflows) {
            ldb_reg = pool_.new_int();
            ldb_ok = is_not_zero(ldb_reg);
            if (!ldb_ok) {
                DEBUG({
                    verbose_printf(verbose_t::debuginfo,
                            "rvjit: rvv_matmul_engine_t::configure() failed "
                            "due to insufficient registers to materialize "
                            "ldb\n");
                });
            }
        }
        const const_t ldb = ldb_overflows
                ? const_t(plan.strides.ldb.bytes).reg(ldb_reg)
                : allocate_stride_(plan.strides.ldb);

        // Allocate float and vector register files
        pool_.float_register_file();
        pool_.vector_register_file(m, m_acc);

        // Setup the strategy to obtain the application vector length
        const runtime_value_t avl = pool_.new_runtime_value(plan.avl);
        bool ok = avl.is_valid();
        if (!ok) {
            DEBUG({
                verbose_printf(verbose_t::debuginfo,
                        "rvjit: rvv_matmul_engine_t::configure() failed due "
                        "to an invalid avl\n");
            });
        }
        ok = ok && ldb_ok;
        avl_ = avl.reg;

        // When the N-loop materializes as a loop instead of a dispatch
        const bool needs_ptra_base = n_loop.mode == loop_t::mode_t::unroll
                || n_loop.mode == loop_t::mode_t::unroll_and_switch;
        ptra_base_ = needs_ptra_base ? pool_.new_int() : Reg {};

        ok = configure_(n_ur, plan.dti, plan.dto, plan.dtb, plan.ptra, lda,
                     plan.is_transa, plan.ptrb, ldb, plan.is_transb, plan.ptrc)
                && ok;
        scratch_ = scratch_vreg_();

        plan_ = plan;
        n_ur_ = n_ur;
        k_ur_ = k_ur;
        k_ur_tail_ = k_ur_tail;
        sewi_ = sewi;
        sewo_ = sewo;
        mi_ = m;
        mo_ = m_acc;
        lda_ = lda;
        ldc_ = ldc;
        ldb_ = ldb;
        n_loop_cfg_ = n_loop;
        k_loop_cfg_ = k_loop;
        return ok;
    }

    /// Emits the innermost N-loop per the configured plan
    ///
    /// @pre configure() has been called and the calling kernel's
    ///     `register_pool_t::preserve()` has already run, since this writes
    ///     any stride register configure() forced but deferred
    void generate(const on_matmul_postop_t &cb) {
        if (!plan_.strides.lda.prematerialized) ca_.init_constant(lda_);
        if (!plan_.strides.ldc.prematerialized) ca_.init_constant(ldc_);
        if (!plan_.strides.ldb.prematerialized) ca_.init_constant(ldb_);
        ca_.init_constant(b_outer_stride_);

        if (plan_.avl.is_lazy()) plan_.avl.get(avl_);

        const bool has_ptra_base = is_not_zero(ptra_base_);
        if (has_ptra_base) e_->mv(ptra_base_, plan_.ptra);
        if (!is_mixed_precision()) wide_vset_();

        cf_.unrolled_loop(n_loop_cfg_, [&](int nb) {
            loop_t k_loops = k_loop_cfg_;
            // Bypasses unrolling(unroll_t)'s sort — factors[1] stays 1 and
            // k_ur_/k_ur_tail_ are always >= 1, so descending order holds.
            k_loops.unrolling().factors[0] = nb == n_ur_ ? k_ur_ : k_ur_tail_;

            if (has_ptra_base) e_->mv(plan_.ptra, ptra_base_);

            if (is_mixed_precision()) wide_vset_();
            zero_accumulators_(nb);

            if (is_mixed_precision()) narrow_vset_();
            dense_loop_(nb, k_loops);

            if (is_mixed_precision()) wide_vset_();
            cb(c_data_view_().reshaped(1, nb), v_block_t(&scratch_, 1));
        });
    }

private:
    emitter_t e_;
    register_pool_t &pool_;
    memory_move_t &mem_;
    rvv_arithmetic_t &mc_;
    const_folding_t &ca_;
    control_flow_t &cf_;
    rvv_t model_;

    // Low-level tile state
    int N_ = 0;
    fma_t op_ = fma_t::uniform(optype_t::f32);

    Reg ptra_ {}, ptrb_ {}, ptrc_ {};
    const_t a_outer_stride_, a_inner_stride_;
    const_t b_outer_stride_, b_inner_stride_;

    vaddr_t amode_ {};
    v_block_t c_data_, a_data_;
    f_block_t b_data_;
    x_block_t b_data_x_;
    x_block_t pivots_;

    // Plan-driven state
    matmul_plan_t plan_;
    Reg avl_ {};
    Reg ptra_base_ {};
    VReg scratch_ {};

    // Configuration resolved by configure(), consumed by generate()
    int n_ur_ = 0;
    int k_ur_ = 0;
    int k_ur_tail_ = 0;
    SEW sewi_ = SEW::e8;
    SEW sewo_ = SEW::e8;
    LMUL mi_ = LMUL::m1;
    LMUL mo_ = LMUL::m1;
    const_t lda_;
    const_t ldc_;
    const_t ldb_;
    loop_t n_loop_cfg_;
    loop_t k_loop_cfg_;

    /// Allocates data registers and stores the addressing configuration
    ///
    /// @param N  Largest N-loop unrolling factor used on dense loops
    ///
    /// @pre Caller has set up the pool's vector register file
    bool configure_(int N, optype_t idt, optype_t adt, optype_t bdt,
            const Reg &ptra, const const_t &lda, bool is_transa,
            const Reg &ptrb, const const_t &ldb, bool is_transb,
            const Reg &ptrc) {
        N_ = N;
        ptra_ = ptra;
        ptrb_ = ptrb;
        ptrc_ = ptrc;

        const bool same_sew = sewb(idt) == sewb(adt);
        op_ = same_sew ? fma_t::uniform(idt) : fma_t::widening(idt);

        // A's on-disk width drives its own stride; B's on-disk width (bdt)
        // may differ from idt when A has been pre-widened by the caller
        // (see matmul_plan_t::dtb).
        const int sewb_a = sewb(idt);
        const int sewb_b = sewb(bdt);
        a_outer_stride_ = is_transa ? const_t(sewb_a) : lda;
        a_inner_stride_ = is_transa ? lda : const_t(sewb_a);
        b_outer_stride_ = is_transb ? const_t(sewb_b) : ldb;
        b_inner_stride_ = is_transb ? ldb : const_t(sewb_b);

        // load_b_'s Case A/B addressing bakes `ni * b_outer_stride_` into a
        // raw immediate at JIT-generation time (ni in [0, N-1]); new_const()
        // only validated the unscaled stride, so re-check against the actual
        // multiplier range here and fall back to a register if it overflows.
        // Allocation only, same as lda/ldc/ldb above — the write is deferred
        // to generate() so it lands after the calling kernel's
        // register_pool_t::preserve(), not before it.
        if (b_outer_stride_.is_imm() && N > 1
                && !is_simm12((N - 1) * b_outer_stride_.imm())) {
            const Reg r = pool_.new_int();
            if (!is_not_zero(r)) {
                DEBUG({
                    verbose_printf(verbose_t::debuginfo,
                            "rvjit: Failed rvv_matmul_engine_t::configure() "
                            "due to insufficient registers to materialize "
                            "b_outer_stride\n");
                });
                return false;
            }
            b_outer_stride_.reg(r);
        }

        amode_ = a_inner_stride_.has_reg()
                ? vaddr_t::strided(ptra, a_inner_stride_.reg())
                : vaddr_t::unit(ptra);

        if (pool_.v_available() < N) {
            DEBUG({
                verbose_printf(verbose_t::debuginfo,
                        "rvjit: Failed rvv_matmul_engine_t::configure() due "
                        "to insufficient accumulator registers\n");
            });
            return false;
        }
        c_data_ = pool_.new_vector_accumulator(N, 1);

        // Try to use a double-buffer for pipelining
        if (pool_.v_available() >= 2)
            a_data_ = pool_.new_vector(2, 1);
        else if (pool_.v_available() >= 1)
            a_data_ = pool_.new_vector(1, 1);
        else {
            DEBUG({
                verbose_printf(verbose_t::debuginfo,
                        "rvjit: Failed rvv_matmul_engine_t::configure() due "
                        "to insufficient vector registers\n");
            });
            return false;
        }

        if (is_int_optype(bdt)) {
            if (pool_.x_available() >= 2 * N)
                b_data_x_ = pool_.new_int(2, N);
            else if (pool_.x_available() >= N)
                b_data_x_ = pool_.new_int(1, N);
            else {
                DEBUG({
                    verbose_printf(verbose_t::debuginfo,
                            "rvjit: Failed rvv_matmul_engine_t::configure() "
                            "due to insufficient int registers\n");
                });
                return false;
            }
        } else if (pool_.f_available() >= 2 * N)
            b_data_ = pool_.new_float(2, N);
        else if (pool_.f_available() >= N)
            b_data_ = pool_.new_float(1, N);
        else {
            DEBUG({
                verbose_printf(verbose_t::debuginfo,
                        "rvjit: Failed rvv_matmul_engine_t::configure() due "
                        "to insufficient float registers\n");
            });
            return false;
        }

        // B pivots: N independent pointers (Case C) when b_outer is a reg;
        // fall back to 1 pointer (Case D) if N registers are unavailable
        const int want_pivots = b_outer_stride_.has_reg() ? N : 1;
        pivots_ = pool_.new_int(want_pivots);
        if (!pivots_.size() && b_outer_stride_.has_reg())
            pivots_ = pool_.new_int(1);
        if (!pivots_.size()) {
            DEBUG({
                verbose_printf(verbose_t::debuginfo,
                        "rvjit: Failed rvv_matmul_engine_t::configure() due "
                        "to insufficient pointer registers\n");
            });
            return false;
        }

        return true;
    }

    /// Zeroes the accumulator registers for this tile and primes pointers
    ///
    /// @param nb  N-loop unroll for this tile (≤ N passed to configure_)
    ///
    /// @pre Caller has configured vsetvli for the wide (accumulator) LMUL:
    ///     `vmv_v_i` only clears as many physical registers as the active
    ///     LMUL spans, and the accumulator group is sized by the wide LMUL
    void zero_accumulators_(int nb) {
        // Interleave `vmv` and `add_const` for in-order dual issue processors
        e_->vmv_v_i(c_data_(0, 0), 0);
        e_->mv(pivots_[0], ptrb_);

        for (int ni = 1; ni < nb; ++ni) {
            e_->vmv_v_i(c_data_(ni, 0), 0);
            if (b_outer_stride_.has_reg() && pivots_.size() > 1)
                ca_.add_const(pivots_[ni], pivots_[ni - 1], b_outer_stride_);
        }
    }

    /// Emits a dense arithmetic loop for matrix tile multiplication
    ///
    /// @param nb       N-loop unroll for this tile (≤ N passed to configure_)
    /// @param delay    Load iterations before the FMAs (clamped to [0,nb]);
    ///     defaults to 5, a typical L1D cache-line load latency in cycles
    ///
    /// @pre zero_accumulators_(nb) has run for this tile, and vsetvli is
    ///     configured for the narrow (input) LMUL
    void dense_loop_(int nb, const loop_t &k_loops, int delay = 5) {
        const auto d = std::min(nb, std::max(0, delay));
        cf_.unrolled_loop(k_loops, [&](int ku) { k_loop_(nb, ku, d); });
    }

    v_block_t c_data_view_() const { return c_data_; }
    VReg scratch_vreg_() const { return a_data_(0, 0); }

    void k_loop_(int nb, int kb, int delay) {
        if (a_data_.rows() >= 2 && b_data_.rows() >= 2 && delay > 0)
            k_loop_pipelined_(nb, kb, delay);
        else
            k_loop_seq_(nb, kb);
    }

    // Pipelined K-loop: `delay` B loads are issued before the first FMA;
    // each FMA(kk,nn) then issues the B load at linear position
    // kk*nb+nn+delay, wrapping across k-step boundaries.
    //
    // @pre a_data_ and b_data_ have >= 2 rows (2-row ring covers kk, kk+1)
    void k_loop_pipelined_(int nb, int kb, int delay) {
        load_a_(0, kb);
        for (int d = 0; d < delay; ++d)
            load_b_(0, kb, d, nb);

        for (int kk = 0; kk < kb; ++kk) {
            load_a_(kk + 1, kb);
            for (int nn = 0; nn < nb; ++nn) {
                macc_(kk, nn);
                load_b_(kk + (nn + delay) / nb, kb, (nn + delay) % nb, nb);
            }
        }
    }

    // Sequential K-loop: used when a_data_/b_data_ have only 1 row (WAR
    // hazard prevents pipelining) or delay == 0.
    void k_loop_seq_(int nb, int kb) {
        for (int kk = 0; kk < kb; ++kk) {
            load_a_(kk, kb);
            for (int nn = 0; nn < nb; ++nn)
                load_b_(kk, kb, nn, nb);
            for (int nn = 0; nn < nb; ++nn)
                macc_(kk, nn);
        }
    }

    // Dispatches the accumulate step to the int or float macc backend based
    // on B's on-disk dtype (plan_.dtb) — the register type b_data_/b_data_x_
    // holds the loaded B scalar in dictates which overload applies.
    void macc_(int kk, int nn) {
        if (is_int_optype(plan_.dtb))
            mc_.fmacc_int(
                    c_data_(nn, 0), b_data_x_(kk, nn), a_data_(kk, 0), op_);
        else
            mc_.fmacc_float(
                    c_data_(nn, 0), b_data_(kk, nn), a_data_(kk, 0), op_);
    }

    // Loads A[kk] into ring slot ki%2 and advances ptra; no-op if ki >= kb
    void load_a_(int ki, int kb) {
        if (ki >= kb) return;
        mem_.vload(a_data_(ki, 0), amode_, sewi_);
        ca_.add_const(ptra_, ptra_, a_outer_stride_);
    }

    // Loads a scalar B element into b_data_/b_data_x_, whichever matches
    // B's on-disk dtype (plan_.dtb) — int path reads via a sign/zero
    // extending xload, float path via fload. Uses sew_for(plan_.dtb), not
    // sewi_, since B's storage width can differ from A's (see dtb).
    void load_b_scalar_(int ki, int ni, const Reg &base, int off) {
        const SEW sewb_dt = sew_for(plan_.dtb);
        if (is_int_optype(plan_.dtb))
            mem_.xload(b_data_x_(ki, ni), base, sewb_dt,
                    is_signed_int(plan_.dtb), off);
        else
            mem_.fload(b_data_(ki, ni), base, sewb_dt, off);
    }

    // Loads B[kk][nn] from the appropriate pivot; no-op if ki >= kb.
    //
    // Case D (fallback): 1 pivot, columns advanced sequentially by b_outer,
    //     net-stepped to the next k-row after the last column.
    // Case C: N independent pivots, one per column.
    // Case B: 1 pivot advancing per k-step.
    // Case A: 1 shared pivot, immediate offsets per column.
    void load_b_(int ki, int kb, int ni, int nb) {
        if (ki >= kb) return;
        if (b_outer_stride_.has_reg() && pivots_.size() < N_) {
            // Case D
            load_b_scalar_(ki, ni, pivots_[0], 0);
            if (ni < nb - 1) {
                ca_.add_const(pivots_[0], pivots_[0], b_outer_stride_);
            } else {
                ca_.add_const(pivots_[0], pivots_[0], b_inner_stride_);
                for (int i = 0; i < nb - 1; ++i)
                    e_->sub(pivots_[0], pivots_[0], b_outer_stride_.reg());
            }
            return;
        }
        if (b_outer_stride_.has_reg()) {
            // Case C
            load_b_scalar_(ki, ni, pivots_[ni],
                    ki == 0 ? 0 : (ki - kb) * b_inner_stride_.imm());
        } else if (b_inner_stride_.has_reg()) {
            // Case B
            load_b_scalar_(ki, ni, pivots_[0], ni * b_outer_stride_.imm());
        } else {
            // Case A
            load_b_scalar_(ki, ni, pivots_[0],
                    ki == 0 ? ni * b_outer_stride_.imm()
                            : ni * b_outer_stride_.imm()
                                    + (ki - kb) * b_inner_stride_.imm());
        }

        // Cases A/C advance the pivot right after the ki==0 load, so later
        // ki>0 loads use a negative (ki-kb)*b_inner offset instead of a
        // growing positive one; Case B advances once per k-step instead.
        if (!b_inner_stride_.has_reg()) {
            if (ki == 0) {
                if (b_outer_stride_.has_reg()) {
                    ca_.add_const(pivots_[ni], pivots_[ni],
                            const_t(kb * b_inner_stride_.imm()));
                } else if (ni == nb - 1) {
                    ca_.add_const(pivots_[0], pivots_[0],
                            const_t(kb * b_inner_stride_.imm()));
                }
            }
        } else if (ni == nb - 1) {
            ca_.add_const(pivots_[0], pivots_[0], b_inner_stride_);
        }
    }

    /// Allocates a register for `e` if it doesn't fit a 12-bit immediate
    ///
    /// @note Does not write to the allocated register — see the `@pre` on
    ///     generate(), which performs that write
    const_t allocate_stride_(const matmul_strides_t::entry_t &e) {
        if (e.prematerialized) return e.value;
        return pool_.new_const(e.bytes);
    }

    void wide_vset_() const {
        e_->vsetvli(Xbyak_riscv::x0, avl_, sewo_, mo_, VTA::ta, VMA::ma);
    }
    void narrow_vset_() const {
        e_->vsetvli(Xbyak_riscv::x0, avl_, sewi_, mi_, VTA::ta, VMA::ma);
    }
};

#endif // XBYAK_RISCV_V

} // namespace rvjit
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#undef DEBUg
#undef DEBUG

#endif
