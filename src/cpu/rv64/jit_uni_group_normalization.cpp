/*******************************************************************************
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
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

#include <math.h>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/platform.hpp"

#include "cpu/rv64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/rv64/jit_generator.hpp"
#include "cpu/rv64/jit_uni_group_normalization.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;
// NB: intentionally no `using namespace data_type;` — it would bring the 4-bit
// and 8-bit integer data-type names (s4/u4/s8/u8) into scope and shadow the
// homonymous Xbyak_riscv saved registers used by the kernels.

using pd_t = jit_uni_group_normalization_fwd_t::pd_t;
using stat_call_params_t
        = jit_uni_group_normalization_fwd_t::stat_call_params_t;
using norm_call_params_t
        = jit_uni_group_normalization_fwd_t::norm_call_params_t;

namespace {

// ============================ stats kernel ===============================
// Reduces one group to (sum, sum_of_squares) written as doubles. f32 accumulates
// in f64 (vfwadd.wv / vfwmacc.vv); f16 widens f16->f32->f64 first. Both layouts:
//   ncsp - the group is one contiguous C_PER_G*SP run, consumed by a VLA loop
//          (partial-vl vsetvli passes VTA::tu explicitly so the tail safely
//          accumulates into the running f64 vector before the final reduction).
//   nspc - channel chunks (vl fixed per chunk) accumulated across spatial; lanes
//          are independent partial sums reduced together at the end.
template <cpu_isa_t isa, data_type_t d_type>
struct stat_kernel_t
    : public jit_uni_group_normalization_fwd_t::kernel_stat_base_t,
      public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(stat_kernel_t)

    stat_kernel_t(const pd_t *apd)
        : jit_generator_t("jit_rvv_gnorm_stat", isa), is_ncsp_(apd->is_ncsp_) {}

    status_t create_kernel() override {
        return jit_generator_t::create_kernel();
    }
    void operator()(const stat_call_params_t *p) const override {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    const bool is_ncsp_;
};

template <cpu_isa_t isa, data_type_t d_type>
void stat_kernel_t<isa, d_type>::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    constexpr bool is_f16 = (d_type == data_type::f16);
    const Reg reg_param = a0;
    const Reg reg_src = a1, reg_sum = a2, reg_sumsq = a3;
    const Reg reg_cpg = a4, reg_sp = a5, reg_c = a6;
    const Reg reg_vl = t0, reg_tmp = t1, reg_tmp2 = t2;

    ld(reg_src, reg_param, (int)offsetof(stat_call_params_t, src));
    ld(reg_sum, reg_param, (int)offsetof(stat_call_params_t, sum));
    ld(reg_sumsq, reg_param, (int)offsetof(stat_call_params_t, sumsq));
    ld(reg_cpg, reg_param, (int)offsetof(stat_call_params_t, c_per_g));
    ld(reg_sp, reg_param, (int)offsetof(stat_call_params_t, sp));
    ld(reg_c, reg_param, (int)offsetof(stat_call_params_t, c));

    if (is_ncsp_ && !is_f16) {
        // ncsp f32: 4x-unrolled f64 reduction at VLMAX + VLA tail, matching the
        // ILP of the previous intrinsic stats_reduction. Four independent f64m2
        // accumulator pairs hide the widening-FMA latency; combined and reduced
        // at the end. reg_tmp == 4*VLMAX is reused as the per-load byte stride
        // since f32 is 4 bytes (4*VLMAX elements == VLMAX*4 bytes).
        const VReg vS0(8), vS1(10), vS2(12), vS3(14);
        const VReg vQ0(16), vQ1(18), vQ2(20), vQ3(22);
        const VReg vx0(4), vx1(5), vx2(6), vx3(7);
        const VReg v_zero(24), v_red(25);
        const Reg reg_n = a7, reg_p = t3, reg_t = t4;

        vsetvli(reg_vl, x0, SEW::e64, LMUL::m2, VTA::ta,
                VMA::ma); // zero accumulators
        vmv_v_x(vS0, x0);
        vmv_v_x(vS1, x0);
        vmv_v_x(vS2, x0);
        vmv_v_x(vS3, x0);
        vmv_v_x(vQ0, x0);
        vmv_v_x(vQ1, x0);
        vmv_v_x(vQ2, x0);
        vmv_v_x(vQ3, x0);

        mul(reg_n, reg_cpg, reg_sp); // group_len = C_PER_G * SP
        vsetvli(reg_vl, x0, SEW::e32, LMUL::m1, VTA::ta,
                VMA::ma); // vtype e32/m1, reg_vl = VLMAX
        slli(reg_tmp, reg_vl, 2); // 4*VLMAX (== VLMAX*4 bytes for f32)
        slli(reg_tmp2, reg_tmp, 2); // 16*VLMAX bytes = src advance per iter

        Label main, main_done;
        L(main);
        blt(reg_n, reg_tmp, main_done); // while remaining >= 4*VLMAX
        mv(reg_p, reg_src);
        vle32_v(vx0, reg_p);
        add(reg_p, reg_p, reg_tmp);
        vle32_v(vx1, reg_p);
        add(reg_p, reg_p, reg_tmp);
        vle32_v(vx2, reg_p);
        add(reg_p, reg_p, reg_tmp);
        vle32_v(vx3, reg_p);
        vfwadd_wv(vS0, vS0, vx0);
        vfwadd_wv(vS1, vS1, vx1);
        vfwadd_wv(vS2, vS2, vx2);
        vfwadd_wv(vS3, vS3, vx3);
        vfwmacc_vv(vQ0, vx0, vx0);
        vfwmacc_vv(vQ1, vx1, vx1);
        vfwmacc_vv(vQ2, vx2, vx2);
        vfwmacc_vv(vQ3, vx3, vx3);
        add(reg_src, reg_src, reg_tmp2);
        sub(reg_n, reg_n, reg_tmp);
        j_(main);
        L(main_done);

        // Combine the four accumulator pairs (under e64/m2).
        vsetvli(reg_t, x0, SEW::e64, LMUL::m2, VTA::ta, VMA::ma);
        vfadd_vv(vS0, vS0, vS1);
        vfadd_vv(vS2, vS2, vS3);
        vfadd_vv(vS0, vS0, vS2);
        vfadd_vv(vQ0, vQ0, vQ1);
        vfadd_vv(vQ2, vQ2, vQ3);
        vfadd_vv(vQ0, vQ0, vQ2);

        // Remainder (< 4*VLMAX): VLA tail into vS0/vQ0 (tu keeps lanes).
        Label tail, tail_done;
        L(tail);
        beqz(reg_n, tail_done);
        vsetvli(reg_vl, reg_n, SEW::e32, LMUL::m1, VTA::tu, VMA::ma);
        vle32_v(vx0, reg_src);
        vfwadd_wv(vS0, vS0, vx0);
        vfwmacc_vv(vQ0, vx0, vx0);
        slli(reg_t, reg_vl, 2);
        add(reg_src, reg_src, reg_t);
        sub(reg_n, reg_n, reg_vl);
        j_(tail);
        L(tail_done);

        vsetvli(reg_t, x0, SEW::e64, LMUL::m1, VTA::ta, VMA::ma);
        vmv_v_x(v_zero, x0);
        vsetvli(reg_t, x0, SEW::e64, LMUL::m2, VTA::ta, VMA::ma);
        vfredusum_vs(v_red, vS0, v_zero);
        vfmv_f_s(fa0, v_red);
        fsd(fa0, reg_sum, 0);
        vfredusum_vs(v_red, vQ0, v_zero);
        vfmv_f_s(fa0, v_red);
        fsd(fa0, reg_sumsq, 0);
        ret();
        return;
    }

    // Single-accumulator f64 path: nspc (f32/f16) and ncsp f16.
    // Accumulators: f32 path uses f64m2, f16 path widens to f64m4.
    const LMUL acc_lmul = is_f16 ? LMUL::m4 : LMUL::m2;
    const VReg v_sum(8);
    const VReg v_sq = is_f16 ? VReg(12) : VReg(10);
    const VReg v_x(4); // f32 load (m1) for f32 path
    const VReg v_h(2); // f16 load (m1)
    const VReg v_f(4); // widened f32 (m2) for f16 path
    const VReg v_zero = is_f16 ? VReg(16) : VReg(12);
    const VReg v_red = is_f16 ? VReg(17) : VReg(13);

    vsetvli(reg_vl, x0, SEW::e64, acc_lmul, VTA::ta, VMA::ma);
    vmv_v_x(v_sum, x0);
    vmv_v_x(v_sq, x0);

    auto accumulate = [&]() {
        // One chunk: load (reg_src/ptr), widen-accumulate into v_sum/v_sq.
        if (!is_f16) {
            vle32_v(v_x, reg_src);
            vfwadd_wv(v_sum, v_sum, v_x);
            vfwmacc_vv(v_sq, v_x, v_x);
        } else {
            vle16_v(v_h, reg_src);
            vfwcvt_f_f_v(v_f, v_h); // f16 -> f32 (m2), under e16/m1
            vsetvli(reg_tmp, reg_vl, SEW::e32, LMUL::m2, VTA::tu, VMA::ma);
            vfwadd_wv(v_sum, v_sum, v_f); // f32 (m2) -> f64 (m4)
            vfwmacc_vv(v_sq, v_f, v_f);
        }
    };

    const SEW load_sew = is_f16 ? SEW::e16 : SEW::e32;
    const int dtshift = is_f16 ? 1 : 2;

    if (is_ncsp_) {
        const Reg reg_n = a7;
        mul(reg_n, reg_cpg, reg_sp); // group_len = C_PER_G * SP
        Label loop, done;
        L(loop);
        beqz(reg_n, done);
        vsetvli(reg_vl, reg_n, load_sew, LMUL::m1, VTA::tu, VMA::ma);
        accumulate();
        slli(reg_tmp, reg_vl, dtshift);
        add(reg_src, reg_src, reg_tmp);
        sub(reg_n, reg_n, reg_vl);
        j_(loop);
        L(done);
    } else {
        // nspc: outer channel-block (vl fixed within a block), inner spatial.
        slli(reg_tmp2, reg_c, dtshift); // spatial stride (bytes) = C * dt_size
        const Reg reg_crem = a7, reg_cbase = t3, reg_sprem = t4, reg_ptr = t5;
        mv(reg_crem, reg_cpg);
        mv(reg_cbase, reg_src);
        Label cblk, creduce;
        L(cblk);
        beqz(reg_crem, creduce);
        vsetvli(reg_vl, reg_crem, load_sew, LMUL::m1, VTA::tu, VMA::ma);
        mv(reg_sprem, reg_sp);
        mv(reg_ptr, reg_cbase);
        Label sp_loop, sp_done;
        L(sp_loop);
        beqz(reg_sprem, sp_done);
        mv(reg_src, reg_ptr); // accumulate() reads reg_src
        accumulate();
        if (is_f16)
            vsetvli(reg_vl, reg_crem, load_sew, LMUL::m1, VTA::tu,
                    VMA::ma); // restore vtype
        add(reg_ptr, reg_ptr, reg_tmp2);
        addi(reg_sprem, reg_sprem, -1);
        j_(sp_loop);
        L(sp_done);
        slli(reg_tmp, reg_vl, dtshift);
        add(reg_cbase, reg_cbase, reg_tmp);
        sub(reg_crem, reg_crem, reg_vl);
        j_(cblk);
        L(creduce);
    }

    // Horizontal reduction of the f64 accumulators -> scalar sum / sumsq.
    vsetvli(reg_vl, x0, SEW::e64, LMUL::m1, VTA::ta, VMA::ma);
    vmv_v_x(v_zero, x0);
    vsetvli(reg_vl, x0, SEW::e64, acc_lmul, VTA::ta, VMA::ma);
    vfredusum_vs(v_red, v_sum, v_zero);
    vfmv_f_s(fa0, v_red);
    fsd(fa0, reg_sum, 0);
    vfredusum_vs(v_red, v_sq, v_zero);
    vfmv_f_s(fa0, v_red);
    fsd(fa0, reg_sumsq, 0);
    ret();
#else
    ret();
#endif
}

// ========================== normalize kernel ============================
// dst = (src - mean) * inv_std * gamma + beta, then the fused post-op chain,
// over a whole group in one call. ncsp loops channels (scalar gamma/beta, vector
// over spatial); nspc loops spatial positions (vector gamma/beta, vector over
// channels). f16 computes at f32 (widen on load, narrow on store).
template <cpu_isa_t isa, data_type_t d_type>
struct norm_kernel_t
    : public jit_uni_group_normalization_fwd_t::kernel_norm_base_t,
      public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(norm_kernel_t)

    norm_kernel_t(const pd_t *apd)
        : jit_generator_t("jit_rvv_gnorm_norm", isa)
        , is_ncsp_(apd->is_ncsp_)
        , use_scale_(apd->use_scale())
        , use_shift_(apd->use_shift())
        , post_ops_(apd->attr()->post_ops_) {
        with_postops_ = post_ops_.len() != 0;
        with_binary_ = post_ops_.find(primitive_kind::binary) != -1;
    }

    status_t create_kernel() override {
        return jit_generator_t::create_kernel();
    }
    void operator()(const norm_call_params_t *p) const override {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    const bool is_ncsp_;
    const bool use_scale_;
    const bool use_shift_;
    post_ops_t post_ops_;
    bool with_postops_ = false;
    bool with_binary_ = false;
};

template <cpu_isa_t isa, data_type_t d_type>
void norm_kernel_t<isa, d_type>::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    constexpr bool is_f16 = (d_type == data_type::f16);
    const int dtshift = is_f16 ? 1 : 2;
    const Reg reg_param = a0;

    // Prologue: preserve the callee-saved registers used as loop invariants.
    const int stack = 64;
    addi(sp, sp, -stack);
    sd(s0, sp, 0);
    sd(s1, sp, 8);
    sd(s2, sp, 16);
    sd(s3, sp, 24);
    sd(s4, sp, 32);
    sd(s5, sp, 40);
    sd(s6, sp, 48);
    sd(s7, sp, 56);

    ld(s0, reg_param, (int)offsetof(norm_call_params_t, src));
    ld(s1, reg_param, (int)offsetof(norm_call_params_t, dst));
    flw(ft0, reg_param, (int)offsetof(norm_call_params_t, mean));
    flw(ft1, reg_param, (int)offsetof(norm_call_params_t, inv_std));
    if (use_scale_) ld(s2, reg_param, (int)offsetof(norm_call_params_t, scale));
    if (use_shift_) ld(s3, reg_param, (int)offsetof(norm_call_params_t, shift));
    if (with_binary_)
        ld(s4, reg_param,
                (int)offsetof(norm_call_params_t, post_ops_binary_rhs));
    ld(s5, reg_param, (int)offsetof(norm_call_params_t, c_per_g));
    ld(s6, reg_param, (int)offsetof(norm_call_params_t, sp));

    // In-kernel post-op injector. Scratch is host-owned free vector/scalar regs;
    // binary uses indirect (rhs pointer array) scalar broadcast (off unused).
    eltwise_injector::static_params_t esp(
            VReg(16), VReg(20), VReg(24), fa4, fa5, t3, /*is_fwd=*/true);
    binary_injector::static_params_t bsp(VReg(28), fa6, s4, x0, t4);
    injector::jit_uni_postops_injector_t<isa> po_inj(
            this, post_ops_, esp, with_binary_ ? &bsp : nullptr);

    const Reg reg_vl = t0, reg_tmp = t1, reg_tmp2 = t2;
    const Reg reg_runrem = a2, reg_rsrc = a3, reg_rdst = a4;
    const Reg reg_gptr = a5, reg_bptr = a6;
    const VReg v_dst(8), v_g(4), v_b(6), v_h(2);

    // Process one chunk of the active run. gamma_vec selects per-lane (nspc)
    // vs broadcast (ncsp) scale/shift.
    auto emit_chunk = [&](bool gamma_vec) {
        if (!is_f16) {
            vsetvli(reg_vl, reg_runrem, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
            vle32_v(v_dst, reg_rsrc);
        } else {
            vsetvli(reg_vl, reg_runrem, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
            vle16_v(v_h, reg_rsrc);
            vfwcvt_f_f_v(v_dst, v_h); // f16 -> f32 (m2)
            vsetvli(reg_tmp, reg_vl, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
        }
        vfsub_vf(v_dst, v_dst, ft0); // - mean
        vfmul_vf(v_dst, v_dst, ft1); // * inv_std
        if (use_scale_) {
            if (gamma_vec) {
                vle32_v(v_g, reg_gptr);
                vfmul_vv(v_dst, v_dst, v_g);
            } else
                vfmul_vf(v_dst, v_dst, ft2);
        }
        if (use_shift_) {
            if (gamma_vec) {
                vle32_v(v_b, reg_bptr);
                vfadd_vv(v_dst, v_dst, v_b);
            } else
                vfadd_vf(v_dst, v_dst, ft3);
        }
        if (with_postops_) po_inj.compute_vector(v_dst.getIdx());
        if (!is_f16) {
            vse32_v(v_dst, reg_rdst);
        } else {
            vsetvli(reg_tmp, reg_vl, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
            vfncvt_f_f_w(v_h, v_dst);
            vse16_v(v_h, reg_rdst);
        }
        slli(reg_tmp, reg_vl, dtshift);
        add(reg_rsrc, reg_rsrc, reg_tmp);
        add(reg_rdst, reg_rdst, reg_tmp);
        if (gamma_vec) {
            slli(reg_tmp2, reg_vl, 2); // scale/shift are f32 arrays
            if (use_scale_) add(reg_gptr, reg_gptr, reg_tmp2);
            if (use_shift_) add(reg_bptr, reg_bptr, reg_tmp2);
        }
        sub(reg_runrem, reg_runrem, reg_vl);
    };

    if (is_ncsp_) {
        // Outer over channels; each run is this channel's contiguous SP.
        slli(s7, s6, dtshift); // channel stride (bytes) = SP * dt_size
        const Reg reg_chrem = a1;
        mv(reg_chrem, s5);
        Label ch_loop, ch_done;
        L(ch_loop);
        beqz(reg_chrem, ch_done);
        if (use_scale_) flw(ft2, s2, 0); // gamma[c]
        if (use_shift_) flw(ft3, s3, 0); // beta[c]
        mv(reg_runrem, s6);
        mv(reg_rsrc, s0);
        mv(reg_rdst, s1);
        Label run_loop, run_done;
        L(run_loop);
        beqz(reg_runrem, run_done);
        emit_chunk(/*gamma_vec=*/false);
        j_(run_loop);
        L(run_done);
        add(s0, s0, s7);
        add(s1, s1, s7);
        if (use_scale_) addi(s2, s2, 4);
        if (use_shift_) addi(s3, s3, 4);
        addi(reg_chrem, reg_chrem, -1);
        j_(ch_loop);
        L(ch_done);
    } else {
        // Outer over spatial positions; each run is this position's C_PER_G
        // contiguous channels with per-lane gamma/beta.
        ld(s7, reg_param, (int)offsetof(norm_call_params_t, c));
        slli(s7, s7, dtshift); // position stride (bytes) = C * dt_size
        const Reg reg_posrem = a1;
        mv(reg_posrem, s6);
        Label pos_loop, pos_done;
        L(pos_loop);
        beqz(reg_posrem, pos_done);
        mv(reg_runrem, s5);
        mv(reg_rsrc, s0);
        mv(reg_rdst, s1);
        if (use_scale_) mv(reg_gptr, s2);
        if (use_shift_) mv(reg_bptr, s3);
        Label run_loop, run_done;
        L(run_loop);
        beqz(reg_runrem, run_done);
        emit_chunk(/*gamma_vec=*/true);
        j_(run_loop);
        L(run_done);
        add(s0, s0, s7);
        add(s1, s1, s7);
        addi(reg_posrem, reg_posrem, -1);
        j_(pos_loop);
        L(pos_done);
    }

    ld(s0, sp, 0);
    ld(s1, sp, 8);
    ld(s2, sp, 16);
    ld(s3, sp, 24);
    ld(s4, sp, 32);
    ld(s5, sp, 40);
    ld(s6, sp, 48);
    ld(s7, sp, 56);
    addi(sp, sp, stack);
    ret();
#else
    ret();
#endif
}

} // namespace

// ============================== factories ===============================

jit_uni_group_normalization_fwd_t::kernel_stat_base_t *
jit_uni_group_normalization_fwd_t::kernel_stat_base_t::create(const pd_t *pd) {
    const auto dt = pd->src_md()->data_type;
    if (dt == data_type::f16 && mayiuse(zvfh))
        return new stat_kernel_t<zvfh, data_type::f16>(pd);
    if (dt == data_type::f32 && mayiuse(v))
        return new stat_kernel_t<v, data_type::f32>(pd);
    return nullptr;
}

jit_uni_group_normalization_fwd_t::kernel_norm_base_t *
jit_uni_group_normalization_fwd_t::kernel_norm_base_t::create(const pd_t *pd) {
    const auto dt = pd->src_md()->data_type;
    if (dt == data_type::f16 && mayiuse(zvfh))
        return new norm_kernel_t<zvfh, data_type::f16>(pd);
    if (dt == data_type::f32 && mayiuse(v))
        return new norm_kernel_t<v, data_type::f32>(pd);
    return nullptr;
}

// =============================== pd::init ===============================

status_t jit_uni_group_normalization_fwd_t::pd_t::init(engine_t *engine) {
    using namespace format_tag;
    using skip_mask_t = primitive_attr_t::skip_mask_t;

    const auto sdt = src_md()->data_type;
    const auto ddt = dst_md()->data_type;

    VDISPATCH_GNORM(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_GNORM(mayiuse(v), VERBOSE_UNSUPPORTED_ISA);
    VDISPATCH_GNORM(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_GNORM(
            utils::one_of(sdt, data_type::f32, data_type::f16) && sdt == ddt,
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_GNORM(IMPLICATION(sdt == data_type::f16, mayiuse(zvfh)),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_GNORM(
            platform::has_data_type_support(sdt), VERBOSE_UNSUPPORTED_DT);
    isa_ = (sdt == data_type::f16) ? zvfh : v; // selects kernels + impl name
    VDISPATCH_GNORM(check_scale_shift_data_type(), VERBOSE_UNSUPPORTED_FEATURE,
            "unsupported scale or shift data type");
    VDISPATCH_GNORM(set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);

    const bool ncsp_ok
            = memory_desc_matches_one_of_tag(*src_md(), ncdhw, nchw, ncw, nc)
            && memory_desc_matches_one_of_tag(*dst_md(), ncdhw, nchw, ncw, nc);
    const bool nspc_ok
            = memory_desc_matches_one_of_tag(*src_md(), ndhwc, nhwc, nwc, nc)
            && memory_desc_matches_one_of_tag(*dst_md(), ndhwc, nhwc, nwc, nc);
    VDISPATCH_GNORM(ncsp_ok || nspc_ok, VERBOSE_UNSUPPORTED_TAG);
    // `nc` matches both sets and the two layouts coincide there; prefer the nspc
    // path (a single channel-vectorized pass) for it.
    is_ncsp_ = ncsp_ok && !nspc_ok;

    VDISPATCH_GNORM(impl::is_dense_format_kind({src_md(), dst_md()}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);
    VDISPATCH_GNORM(attr()->has_default_values(skip_mask_t::post_ops),
            VERBOSE_UNSUPPORTED_ATTR);

    // post-ops: any injector-supported eltwise chain plus binary restricted to
    // scalar (per_tensor) broadcast, mirroring x64 (per-channel binary offset
    // cannot be passed scalably). Everything else falls back to ncsp/ref.
    // Concretize any format_any binary src1 descriptor first (x64 parity), so
    // its dims/strides are well-defined before it is inspected below.
    VDISPATCH_GNORM(attr_.set_default_formats(dst_md(0)) == status::success,
            VERBOSE_UNSUPPORTED_POSTOP);
    const auto &po = attr()->post_ops_;
    VDISPATCH_GNORM(injector::jit_uni_postops_injector_t<v>::post_ops_ok(po),
            VERBOSE_UNSUPPORTED_POSTOP);
    for (int i = 0; i < po.len(); i++) {
        if (!po.entry_[i].is_binary()) continue;
        const memory_desc_wrapper s1_d(po.entry_[i].binary.src1_desc);
        // The in-kernel binary injector reads the rhs as f32 (flw/vle32) and the
        // host advances it by sizeof(float); only a scalar (per_tensor) f32 src1
        // is read correctly, so reject anything else (-> ncsp/ref).
        VDISPATCH_GNORM(
                s1_d.nelems() == 1 && s1_d.data_type() == data_type::f32,
                VERBOSE_UNSUPPORTED_POSTOP);
    }

    return status::success;
}

// =========================== primitive init =============================

status_t jit_uni_group_normalization_fwd_t::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_stat_, kernel_stat_base_t::create(pd())));
    CHECK(safe_ptr_assign(kernel_norm_, kernel_norm_base_t::create(pd())));
    CHECK(kernel_stat_->create_kernel());
    CHECK(kernel_norm_->create_kernel());
    return status::success;
}

// ============================== execute =================================

status_t jit_uni_group_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    const float *scale = pd()->use_scale()
            ? CTX_IN_MEM(const float *, DNNL_ARG_SCALE)
            : nullptr;
    const float *shift = pd()->use_shift()
            ? CTX_IN_MEM(const float *, DNNL_ARG_SHIFT)
            : nullptr;

    float *mean = pd()->stats_is_src()
            ? const_cast<float *>(CTX_IN_MEM(const float *, DNNL_ARG_MEAN))
            : CTX_OUT_MEM(float *, DNNL_ARG_MEAN);
    float *variance = pd()->stats_is_src()
            ? const_cast<float *>(CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE))
            : CTX_OUT_MEM(float *, DNNL_ARG_VARIANCE);

    const auto N = pd()->MB();
    const auto C = pd()->C();
    const size_t SP = (size_t)pd()->D() * pd()->H() * pd()->W();
    const auto G = pd()->desc()->groups;
    const auto eps = pd()->desc()->group_norm_epsilon;
    const auto C_PER_G = C / G;
    const bool calculate_stats = !pd()->stats_is_src();
    const bool save_stats = pd()->is_training();
    const bool is_ncsp = pd()->is_ncsp_;
    const auto dt_size = types::data_type_size(src_d.data_type());
    // Shift to each tensor's logical element 0 so a submemory src/dst is read
    // from its origin (off_l(0) is 0 for plain tensors). The binary rhs base is
    // shifted equivalently below.
    const dim_t src_off0 = src_d.off_l(0);
    const dim_t dst_off0 = dst_d.off_l(0);

    // Binary post-op src1 bases (scalar broadcast), one per binary in attr order.
    const auto &po = pd()->attr()->post_ops_;
    std::vector<const void *> po_rhs;
    for (int i = 0; i < po.len(); i++)
        if (po.entry_[i].is_binary()) {
            const memory_desc_wrapper s1_d(po.entry_[i].binary.src1_desc);
            const auto *base = static_cast<const char *>(ctx.host_ptr(
                    DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1));
            po_rhs.push_back(base + s1_d.off_l(0) * sizeof(float));
        }
    const void *const *po_rhs_arr = po_rhs.empty() ? nullptr : po_rhs.data();

    parallel_nd(N, G, [&](dim_t n, dim_t g) {
        const dim_t c_start = g * C_PER_G;
        const size_t off_elems = is_ncsp ? ((size_t)n * C + c_start) * SP
                                         : ((size_t)n * SP * C + c_start);
        const char *src_g = static_cast<const char *>(src)
                + (src_off0 + off_elems) * dt_size;
        char *dst_g
                = static_cast<char *>(dst) + (dst_off0 + off_elems) * dt_size;

        float v_mean = 0.f, v_var = 0.f;
        if (calculate_stats) {
            double sum = 0.0, sumsq = 0.0;
            stat_call_params_t sp;
            sp.src = src_g;
            sp.sum = &sum;
            sp.sumsq = &sumsq;
            sp.c_per_g = C_PER_G;
            sp.sp = (dim_t)SP;
            sp.c = C;
            (*kernel_stat_)(&sp);

            const double n_elems = (double)C_PER_G * (double)SP;
            const double mean_d = sum / n_elems;
            double var_d = sumsq / n_elems - mean_d * mean_d;
            if (var_d < 0) var_d = 0;
            v_mean = (float)mean_d;
            v_var = (float)var_d;
            if (save_stats) {
                mean[n * G + g] = v_mean;
                variance[n * G + g] = v_var;
            }
        } else {
            v_mean = mean[n * G + g];
            v_var = variance[n * G + g];
        }

        const float inv_std = 1.f / sqrtf(v_var + eps);

        norm_call_params_t np;
        np.src = src_g;
        np.dst = dst_g;
        np.mean = v_mean;
        np.inv_std = inv_std;
        np.scale = scale ? scale + c_start : nullptr;
        np.shift = shift ? shift + c_start : nullptr;
        np.post_ops_binary_rhs = po_rhs_arr;
        np.c_per_g = C_PER_G;
        np.sp = (dim_t)SP;
        np.c = C;
        (*kernel_norm_)(&np);
    });

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
