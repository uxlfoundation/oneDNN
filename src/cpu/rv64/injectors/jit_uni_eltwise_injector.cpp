/*******************************************************************************
* Copyright 2019 Intel Corporation
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
#include <cstring>
#include <limits>

#include "common/utils.hpp"

#include "cpu/rv64/injectors/jit_uni_eltwise_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

namespace eltwise_injector {

bool is_alg_supported(alg_kind_t alg) {
    using namespace alg_kind;
    // The *_use_dst_for_bwd variants are included (like x64/aarch64): their
    // forward math equals the base algorithm and their derivative is
    // implemented. eltwise_pow is not supported (like aarch64) -> ref.
    return utils::one_of(alg, eltwise_relu, eltwise_tanh, eltwise_elu,
            eltwise_square, eltwise_abs, eltwise_sqrt, eltwise_linear,
            eltwise_logistic, eltwise_mish, eltwise_exp, eltwise_gelu_tanh,
            eltwise_hardsigmoid, eltwise_hardswish, eltwise_swish, eltwise_clip,
            eltwise_clip_v2, eltwise_round, eltwise_relu_use_dst_for_bwd,
            eltwise_tanh_use_dst_for_bwd, eltwise_elu_use_dst_for_bwd,
            eltwise_sqrt_use_dst_for_bwd, eltwise_logistic_use_dst_for_bwd,
            eltwise_exp_use_dst_for_bwd, eltwise_clip_v2_use_dst_for_bwd);
}

bool needs_extra_aux(alg_kind_t alg) {
    using namespace alg_kind;
    // soft_relu (exp then log, keeping x live across both), log (keeps x live
    // across the poly to patch 0/inf/negative), gelu_erf (x live across erf).
    return utils::one_of(alg, eltwise_soft_relu, eltwise_log, eltwise_gelu_erf);
}

bool is_supported(alg_kind_t alg) {
    // A single algorithm set drives both directions (as on x64/aarch64): the
    // base 3-aux algorithms plus the ones needing a 4th aux. pow is the only
    // gap vs x64 (unsupported, like aarch64). eltwise_round is forward-only
    // (no derivative), but the common layer rejects round + backward before
    // this runs, so it is not special-cased here (matching x64).
    return is_alg_supported(alg) || needs_extra_aux(alg);
}

} // namespace eltwise_injector

template <cpu_isa_t isa>
size_t jit_uni_eltwise_injector_t<isa>::aux_vecs_count(
        alg_kind_t alg, bool is_fwd) {
    // Contract granularity, not exact per-algorithm usage: 3 covers the base
    // post-op budget, 4 adds v_aux3 (soft_relu/log/gelu_erf forward), and
    // backward is served only by the standalone 5-aux host.
    if (!is_fwd) return 5;
    return eltwise_injector::needs_extra_aux(alg) ? 4 : 3;
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::load_f32_const(const FReg &f, float val) {
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(bits));
    h_->li(gpr_aux0_, bits);
    h_->fmv_w_x(f, gpr_aux0_);
}

// NaN-preserving clamp(v, lo, hi). Comparisons with NaN are false, so NaN lanes
// keep their original value instead of being replaced by the clamp bound.
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::clamp(
        const Vmm &vmm_src, float lo, float hi) {
    load_f32_const(f_aux0_, lo);
    h_->vmflt_vf(vmm_mask_, vmm_src, f_aux0_);
    h_->vfmerge_vfm(vmm_src, vmm_src, f_aux0_);
    load_f32_const(f_aux1_, hi);
    h_->vmfgt_vf(vmm_mask_, vmm_src, f_aux1_);
    h_->vfmerge_vfm(vmm_src, vmm_src, f_aux1_);
}

// log(x) via Cephes single-precision logf: frexp into mantissa in
// [sqrt(1/2), sqrt(2)) and integer exponent, then a degree-8 minimax
// polynomial. Constants inline. Uses v_aux0 (poly/scratch), v_aux2 (exponent
// e) and v0 (mantissa-range mask); leaves v_aux1 free. ~1 ULP, well within the
// LOG benchdnn tolerance. Assumes x > 0; log_compute_vector_fwd patches the
// domain edges and soft_relu feeds 1 + exp(x) > 0.
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::log_compute_vector(const Vmm &vmm_src) {
    const Vmm &poly = v_aux0_;
    const Vmm &e = v_aux2_;
    const Vmm &tmp = v_aux1_; // free during the frexp branch and poly tail

    // frexpf: e = ((bits >> 23) & 0xff) - 126; mantissa in [0.5, 1).
    h_->li(gpr_aux0_, 23);
    h_->vsrl_vx(e, vmm_src, gpr_aux0_);
    h_->li(gpr_aux0_, 0xff);
    h_->vand_vx(e, e, gpr_aux0_);
    h_->li(gpr_aux0_, 126);
    h_->vsub_vx(e, e, gpr_aux0_); // e (int exponent)
    // mantissa m = (bits & 0x807fffff) | 0x3f000000  -> [0.5, 1)
    h_->li(gpr_aux0_, 0x807fffff);
    h_->vand_vx(vmm_src, vmm_src, gpr_aux0_);
    h_->li(gpr_aux0_, 0x3f000000);
    h_->vor_vx(vmm_src, vmm_src, gpr_aux0_);
    h_->vfcvt_f_x_v(e, e); // (float)e

    // if (m < SQRTHF) { e -= 1; m = m + m - 1; } else { m = m - 1; }. The host
    // vtype is mask-agnostic, so branch with explicit merges (which write every
    // body lane) rather than masked arithmetic.
    load_f32_const(f_aux0_, 0.707106781186547524f); // SQRTHF
    h_->vmflt_vf(vmm_mask_, vmm_src, f_aux0_); // mask: m < SQRTHF
    load_f32_const(f_aux0_, 1.f);
    h_->vfsub_vf(tmp, e, f_aux0_); // e - 1
    h_->vmerge_vvm(e, e, tmp); // e := mask ? e-1 : e
    h_->vfadd_vv(tmp, vmm_src, vmm_src); // 2m
    h_->vmerge_vvm(vmm_src, vmm_src, tmp); // m := mask ? 2m : m
    load_f32_const(f_aux0_, 1.f);
    h_->vfsub_vf(vmm_src, vmm_src, f_aux0_); // m := m - 1 (== 2m-1 on masked)

    // poly = m^3 * P(m); P is Horner over the Cephes logf coefficients.
    load_f32_const(f_aux0_, 7.0376836292e-2f);
    h_->vfmv_v_f(poly, f_aux0_);
    const float p[] = {-1.1514610310e-1f, 1.1676998740e-1f, -1.2420140846e-1f,
            1.4249322787e-1f, -1.6668057665e-1f, 2.0000714765e-1f,
            -2.4999993993e-1f, 3.3333331174e-1f};
    for (float c : p) {
        h_->vfmul_vv(poly, poly, vmm_src);
        load_f32_const(f_aux0_, c);
        h_->vfadd_vf(poly, poly, f_aux0_);
    }
    h_->vfmul_vv(poly, poly, vmm_src);
    h_->vfmul_vv(poly, poly, vmm_src);
    h_->vfmul_vv(poly, poly, vmm_src); // poly *= m^3

    // result = m + (poly + ln2lo*e - 0.5*m^2) + ln2hi*e  (ln2 split hi/lo)
    load_f32_const(f_aux0_, -2.12194440e-4f); // ln2 lo
    h_->vfmacc_vf(poly, f_aux0_, e); // poly += ln2lo * e
    h_->vfmul_vv(tmp, vmm_src, vmm_src); // m^2
    load_f32_const(f_aux0_, 0.5f);
    h_->vfnmsac_vf(poly, f_aux0_, tmp); // poly -= 0.5*m^2
    h_->vfadd_vv(vmm_src, vmm_src, poly); // v = m + poly
    load_f32_const(f_aux0_, 0.693359375f); // ln2 hi
    h_->vfmacc_vf(vmm_src, f_aux0_, e); // v += ln2hi * e
}

// erf(x) via Abramowitz & Stegun 7.1.26: erf(|x|) = 1 - P(t)*exp(-x^2),
// t = 1/(1 + p|x|). It returns the non-negative magnitude; callers restore the
// odd sign when needed. Max abs error is ~1.5e-7, inside the gelu_erf
// tolerance. Uses v_aux0/v_aux1/v_aux2 (v_aux2 as exp scratch) and v0. A
// caller that needs the original input keeps it live outside those groups
// (forward uses v_aux3).
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::erf_compute_vector(const Vmm &vmm_src) {
    const Vmm &ax = v_aux0_; // |x|
    const Vmm &t = v_aux1_; // t and later the poly

    // Sign-free: return erf(|v|) >= 0; callers fold the sign in (gelu_erf uses
    // x*sign(x) == |x|). This keeps erf inside 3 aux (aux0/aux1/aux2) so it
    // fits the post-op budget when the host supplies a 4th aux for the outer
    // alg.
    h_->li(gpr_aux0_, 0x7fffffff);
    h_->vand_vx(ax, vmm_src, gpr_aux0_); // |x|

    // t = 1 / (1 + p*|x|)
    load_f32_const(f_aux0_, 0.3275911f);
    h_->vfmv_v_f(t, f_aux0_);
    h_->vfmul_vv(t, t, ax);
    load_f32_const(f_aux0_, 1.f);
    h_->vfadd_vf(t, t, f_aux0_);
    load_f32_const(f_aux0_, 1.f);
    h_->vfrdiv_vf(t, t, f_aux0_); // t = 1/(1+p|x|)

    // poly = ((((a5*t + a4)*t + a3)*t + a2)*t + a1) * t   (in v)
    load_f32_const(f_aux0_, 1.061405429f); // a5
    h_->vfmv_v_f(vmm_src, f_aux0_);
    const float a[] = {
            -1.453152027f, 1.421413741f, -0.284496736f, 0.254829592f}; // a4..a1
    for (float c : a) {
        h_->vfmul_vv(vmm_src, vmm_src, t);
        load_f32_const(f_aux0_, c);
        h_->vfadd_vf(vmm_src, vmm_src, f_aux0_);
    }
    h_->vfmul_vv(vmm_src, vmm_src, t); // * t  -> poly, in v
    h_->vmv_v_v(t, vmm_src); // stash poly (survives exp, which uses aux0/aux2)

    // e = exp(-x^2) in v (exp clobbers v_aux0 and v_aux2)
    h_->vfmul_vv(vmm_src, ax, ax); // x^2
    h_->vfneg_v(vmm_src, vmm_src); // -x^2
    exp_compute_vector_fwd(vmm_src); // v = exp(-x^2)

    // erf(|x|) = 1 - poly*e  (in [0, 1))
    h_->vfmul_vv(vmm_src, vmm_src, t); // poly*e
    load_f32_const(f_aux0_, 1.f);
    h_->vfrsub_vf(vmm_src, vmm_src, f_aux0_); // 1 - poly*e
}

// exp(x) via base-2 range reduction + degree-5 minimax polynomial (the classic
// Cephes/sse_mathfun expf), with all constants materialized inline as FP
// scalars (RVV vf-form ops take an FReg directly, so no constant table is
// needed). Uses two aux vector groups (v_aux0_ scratch/poly, v_aux2_ integer
// exponent); v_aux1_ is left free so callers can stash x across the call.
// Doubles as the building block for the other transcendentals.
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::exp_compute_vector_fwd(
        const Vmm &vmm_src) {
    const Vmm &a0 = v_aux0_; // poly accumulator / scratch
    const Vmm &a2
            = v_aux2_; // integer exponent n (must survive to the 2^n step)

    // clamp x to [ln(FLT_MIN), ln(FLT_MAX)] to keep the result finite. The
    // 2*2^(n-1) reconstruction (below) makes the lowest exponent bucket (n ==
    // -126) come out as exactly 0: (n-1)+127 == 0 builds +0.0, not subnormal
    // 2^-127. This matches x64 (which masks the < ln(FLT_MIN) lanes to 0); the
    // error vs the true value (~1e-38) is negligible.
    clamp(vmm_src, -87.33654475f, 88.72283905f); // MINLOGF, MAXLOGF

    // n = round(x * log2e); z = (float)n. Add a signed half and truncate with
    // an explicit mode so float-to-int rounding does not use the application's
    // current FRM or serialize the hot loop with per-vector FRM CSR updates.
    load_f32_const(f_aux0_, 1.44269504088896341f); // log2e
    h_->vfmul_vf(a0, vmm_src, f_aux0_);
    load_f32_const(f_aux0_, 0.5f);
    h_->vfmv_v_f(a2, f_aux0_);
    h_->vfsgnj_vv(a2, a2, a0); // copysign(0.5f, x * log2e)
    h_->vfadd_vv(a0, a0, a2);
    h_->vfcvt_rtz_x_f_v(a2, a0); // nearest integer, ties away from zero
    h_->vfcvt_f_x_v(a0, a2); // z = (float)n

    // r = x - z*C1 - z*C2  (extended-precision ln2), r in [-ln2/2, ln2/2]
    load_f32_const(f_aux0_, 0.693359375f); // C1
    h_->vfnmsac_vf(vmm_src, f_aux0_, a0); // v -= C1*z
    load_f32_const(f_aux0_, -2.12194440e-4f); // C2
    h_->vfnmsac_vf(vmm_src, f_aux0_, a0); // v -= C2*z  => v = r

    // poly5(r) by Horner; coefficients p0..p5 (Cephes expf)
    load_f32_const(f_aux0_, 1.9875691500e-4f);
    h_->vfmv_v_f(a0, f_aux0_); // y = p0
    const float p[] = {1.3981999507e-3f, 8.3334519073e-3f, 4.1665795894e-2f,
            1.6666665459e-1f, 5.0000001201e-1f};
    for (float c : p) {
        h_->vfmul_vv(a0, a0, vmm_src); // y *= r
        load_f32_const(f_aux0_, c);
        h_->vfadd_vf(a0, a0, f_aux0_); // y += c
    }
    // y = y*r^2 + r + 1, computed as ((y*r)*r) + r + 1 to avoid a 3rd aux reg
    h_->vfmul_vv(a0, a0, vmm_src);
    h_->vfmul_vv(a0, a0, vmm_src);
    h_->vfadd_vv(a0, a0, vmm_src);
    load_f32_const(f_aux0_, 1.f);
    h_->vfadd_vf(a0, a0, f_aux0_);

    // 2^n applied as 2 * 2^(n-1): for x near MAXLOGF, n rounds up to 128 and a
    // direct 2^128 = ((128+127)<<23) is +inf, poisoning the otherwise-finite
    // result (exp(MAXLOGF) ~= FLT_MAX). 2^(n-1) stays representable (<= 2^127),
    // so we scale by it and double afterwards (the x64 injector uses the same
    // trick). 2^(n-1) = reinterpret(((n-1) + 127) << 23) = reinterpret((n+126)<<23).
    h_->li(gpr_aux0_, 126);
    h_->vadd_vx(a2, a2, gpr_aux0_);
    h_->li(gpr_aux0_, 23);
    h_->vsll_vx(a2, a2, gpr_aux0_); // a2 = 2^(n-1)

    h_->vfmul_vv(vmm_src, a0, a2); // poly * 2^(n-1)
    load_f32_const(f_aux0_, 2.f);
    h_->vfmul_vf(vmm_src, vmm_src, f_aux0_); // * 2 = poly * 2^n = exp(x)
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::relu_compute_vector_fwd(
        const Vmm &vmm_src) {
    // leaky relu = x>0 ? x : alpha*x. NaN takes the false branch, but
    // alpha*NaN is still NaN.
    load_f32_const(f_aux0_, 0.f);
    load_f32_const(f_aux1_, alpha_);
    h_->vfmul_vf(v_aux0_, vmm_src, f_aux1_);
    h_->vmfgt_vf(vmm_mask_, vmm_src, f_aux0_);
    h_->vmerge_vvm(vmm_src, v_aux0_, vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::relu_zero_ns_compute_vector_fwd(
        const Vmm &vmm_src) {
    // relu(x) = x < 0 ? 0 : x. Keep NaN lanes unchanged.
    load_f32_const(f_aux0_, 0.f);
    h_->vmflt_vf(vmm_mask_, vmm_src, f_aux0_);
    h_->vfmerge_vfm(vmm_src, vmm_src, f_aux0_);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::elu_compute_vector_fwd(
        const Vmm &vmm_src) {
    // x>0 ? x : alpha*(exp(x)-1). NaN takes the false branch and remains NaN
    // through exp/sub/mul.
    h_->vmv_v_v(v_aux1_, vmm_src); // save x
    load_f32_const(f_aux0_, 0.f);
    h_->vmfgt_vf(vmm_mask_, v_aux1_, f_aux0_);
    h_->vfmerge_vfm(vmm_src, vmm_src, f_aux0_); // positive lanes use exp(0)
    exp_compute_vector_fwd(vmm_src); // exp(x)
    load_f32_const(f_aux0_, 1.f);
    h_->vfsub_vf(vmm_src, vmm_src, f_aux0_); // exp(...) - 1
    load_f32_const(f_aux0_, alpha_);
    h_->vfmul_vf(vmm_src, vmm_src, f_aux0_); // alpha*(...)
    load_f32_const(f_aux0_, 0.f);
    h_->vmfgt_vf(vmm_mask_, v_aux1_, f_aux0_);
    h_->vmerge_vvm(vmm_src, vmm_src, v_aux1_);
}

// tanh(x) = 2*sigmoid(2x) - 1, with a linear blend toward tanh(x) ~= x for
// small |x| to avoid catastrophic cancellation (the sigmoid tail has ~1e-7
// absolute error, which is a large *relative* error when tanh(x) is tiny).
// Uses all three aux groups (v_aux0/v_aux1/v_aux2).
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::tanh_compute_vector_fwd(
        const Vmm &vmm_src) {
    constexpr float t1 = 0.002f, t2 = 0.008f; // blend band (in |x|)
    h_->vmv_v_v(v_aux1_, vmm_src); // save x
    load_f32_const(f_aux0_, 2.f);
    h_->vfmul_vf(vmm_src, vmm_src, f_aux0_);
    logistic_compute_vector_fwd(vmm_src);
    load_f32_const(f_aux0_, 2.f);
    h_->vfmul_vf(vmm_src, vmm_src, f_aux0_);
    load_f32_const(f_aux0_, -1.f);
    h_->vfadd_vf(vmm_src, vmm_src, f_aux0_); // v = t = 2*sigmoid(2x)-1
    h_->vfsub_vv(v_aux0_, v_aux1_, vmm_src); // v_aux0 = x - t
    h_->vfmul_vv(v_aux2_, v_aux1_, v_aux1_); // v_aux2 = x^2
    load_f32_const(f_aux0_, t2 * t2);
    h_->vfrsub_vf(v_aux2_, v_aux2_, f_aux0_); // t2^2 - x^2
    load_f32_const(f_aux0_, 1.f / (t2 * t2 - t1 * t1));
    h_->vfmul_vf(v_aux2_, v_aux2_, f_aux0_); // w (unclamped)
    clamp(v_aux2_, 0.f, 1.f); // w in [0,1]
    h_->vfmacc_vv(vmm_src, v_aux2_, v_aux0_); // v = t + w*(x - t)
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::square_compute_vector_fwd(
        const Vmm &vmm_src) {
    // x * x
    h_->vfmul_vv(vmm_src, vmm_src, vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::abs_compute_vector_fwd(
        const Vmm &vmm_src) {
    // clear the IEEE-754 sign bit of each f32 lane (mask-free, no aux
    // vector). Operates on the raw bits and assumes vtype SEW=e32 (the
    // 0x7fffffff mask clears bit 31 of each 32-bit lane).
    h_->li(gpr_aux0_, 0x7fffffff);
    h_->vand_vx(vmm_src, vmm_src, gpr_aux0_);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::sqrt_compute_vector_fwd(
        const Vmm &vmm_src) {
    h_->vfsqrt_v(vmm_src, vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::linear_compute_vector_fwd(
        const Vmm &vmm_src) {
    // alpha * x + beta, fused (single rounding): the reference build
    // contracts this expression to fmadd.s (gcc -ffp-contract=fast)
    // and the x64/aarch64 injectors use fused FMA too. The extra
    // rounding of an unfused mul+add is observable once the f32
    // result is converted to an integer dst (half-integer boundary,
    // e.g. alpha*x+beta landing exactly on n+0.5 only because the
    // product was pre-rounded).
    load_f32_const(f_aux0_, alpha_);
    load_f32_const(f_aux1_, beta_);
    h_->vfmv_v_f(v_aux0_, f_aux1_);
    h_->vfmadd_vf(vmm_src, f_aux0_, v_aux0_); // v = alpha * v + beta
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::soft_relu_compute_vector_fwd(
        const Vmm &vmm_src) {
    // (1/alpha) * log1p(exp(alpha*x)); for alpha*x >= exp_overflow the
    // reference returns alpha*x/alpha == x, so blend that in to avoid the
    // clamped-exp plateau diverging at large inputs.
    constexpr float exp_ovf = 88.72283172607421875f;
    // log() uses all three aux, so keep x in v_aux3 (standalone-only).
    h_->vmv_v_v(v_aux3_, vmm_src); // save x (== in/alpha)
    load_f32_const(f_aux0_, alpha_);
    h_->vfmul_vf(vmm_src, vmm_src, f_aux0_); // in = alpha*x
    exp_compute_vector_fwd(vmm_src); // exp(in) (input clamped internally)
    load_f32_const(f_aux0_, 1.f);
    h_->vfadd_vf(vmm_src, vmm_src, f_aux0_); // 1 + exp(in)
    log_compute_vector(vmm_src); // log1p(exp(in))
    load_f32_const(f_aux0_, 1.f / alpha_);
    h_->vfmul_vf(vmm_src, vmm_src, f_aux0_); // / alpha
    // recompute in = alpha*x and select x where in >= exp_overflow
    load_f32_const(f_aux0_, alpha_);
    h_->vfmul_vf(v_aux0_, v_aux3_, f_aux0_); // in
    load_f32_const(f_aux0_, exp_ovf);
    h_->vmfge_vf(vmm_mask_, v_aux0_, f_aux0_);
    h_->vmerge_vvm(vmm_src, vmm_src, v_aux3_); // in>=ovf ? x : soft_relu
    // log_compute_vector bit-decomposes its positive-domain input. Test
    // the scaled input so 0 * (+/-inf), as well as an input NaN, is
    // restored to the public NaN result.
    h_->vmfne_vv(vmm_mask_, v_aux0_, v_aux0_);
    load_f32_const(f_aux0_, std::numeric_limits<float>::quiet_NaN());
    h_->vfmerge_vfm(vmm_src, vmm_src, f_aux0_);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::mish_compute_vector_fwd(
        const Vmm &vmm_src) {
    // mish(x) = x * tanh(softplus(x)). With w = 1 + exp(x),
    // tanh(softplus(x)) = 1 - 2/(w^2 + 1) (overflow-safe: w^2 -> inf
    // gives 1 - 0 = 1, i.e. mish -> x, matching the reference).
    h_->vmv_v_v(v_aux1_, vmm_src); // save x
    exp_compute_vector_fwd(vmm_src); // exp(x)
    load_f32_const(f_aux0_, 1.f);
    h_->vfadd_vf(vmm_src, vmm_src, f_aux0_); // w = 1 + exp(x)
    h_->vfmul_vv(vmm_src, vmm_src, vmm_src); // w^2
    load_f32_const(f_aux0_, 1.f);
    h_->vfadd_vf(vmm_src, vmm_src, f_aux0_); // w^2 + 1
    load_f32_const(f_aux0_, 2.f);
    h_->vfrdiv_vf(vmm_src, vmm_src, f_aux0_); // 2 / (w^2 + 1)
    load_f32_const(f_aux0_, 1.f);
    h_->vfrsub_vf(vmm_src, vmm_src, f_aux0_); // 1 - 2/(w^2+1) = tanh(sp(x))
    h_->vfmul_vv(vmm_src, vmm_src, v_aux1_); // * x
}

// sigmoid(x) = 1 / (1 + exp(-x)); numerically stable for both tails.
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::logistic_compute_vector_fwd(
        const Vmm &vmm_src) {
    h_->vfneg_v(vmm_src, vmm_src); // -x
    exp_compute_vector_fwd(vmm_src); // exp(-x)
    load_f32_const(f_aux0_, 1.f);
    h_->vfadd_vf(vmm_src, vmm_src, f_aux0_); // 1 + exp(-x)
    load_f32_const(f_aux0_, 1.f);
    h_->vfrdiv_vf(vmm_src, vmm_src, f_aux0_); // 1 / (1 + exp(-x))
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::gelu_tanh_compute_vector_fwd(
        const Vmm &vmm_src) {
    // 0.5*x*(1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) )). The tanh is
    // emitted raw (2*sigmoid(2g)-1, no small-|x| blend): v_aux1 must hold x
    // across the call, which the blending tanh building block would clobber.
    h_->vmv_v_v(v_aux1_, vmm_src); // save x
    h_->vfmul_vv(vmm_src, vmm_src, vmm_src); // x^2
    load_f32_const(f_aux0_, 0.044715f);
    h_->vfmul_vf(vmm_src, vmm_src, f_aux0_); // 0.044715*x^2
    load_f32_const(f_aux0_, 1.f);
    h_->vfadd_vf(vmm_src, vmm_src, f_aux0_); // 1 + 0.044715*x^2
    h_->vfmul_vv(vmm_src, vmm_src, v_aux1_); // x*(1 + 0.044715*x^2)
    load_f32_const(f_aux0_, 0.7978845608028654f); // sqrt(2/pi)
    h_->vfmul_vf(vmm_src, vmm_src, f_aux0_); // inner argument
    // tanh(inner): 2*sigmoid(2*inner) - 1
    load_f32_const(f_aux0_, 2.f);
    h_->vfmul_vf(vmm_src, vmm_src, f_aux0_);
    logistic_compute_vector_fwd(vmm_src);
    load_f32_const(f_aux0_, 2.f);
    h_->vfmul_vf(vmm_src, vmm_src, f_aux0_);
    load_f32_const(f_aux0_, -1.f);
    h_->vfadd_vf(vmm_src, vmm_src, f_aux0_); // tanh(inner)
    load_f32_const(f_aux0_, 1.f);
    h_->vfadd_vf(vmm_src, vmm_src, f_aux0_); // 1 + tanh
    h_->vfmul_vv(vmm_src, vmm_src, v_aux1_); // x*(1+tanh)
    load_f32_const(f_aux0_, 0.5f);
    h_->vfmul_vf(vmm_src, vmm_src, f_aux0_); // 0.5*x*(1+tanh)
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::swish_compute_vector_fwd(
        const Vmm &vmm_src) {
    // x * sigmoid(alpha * x)
    h_->vmv_v_v(v_aux1_, vmm_src); // save x (exp/logistic keep aux1 free)
    load_f32_const(f_aux0_, alpha_);
    h_->vfmul_vf(vmm_src, vmm_src, f_aux0_); // alpha*x
    logistic_compute_vector_fwd(vmm_src); // sigmoid(alpha*x)
    h_->vfmul_vv(vmm_src, vmm_src, v_aux1_); // x * sigmoid(alpha*x)
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::log_compute_vector_fwd(
        const Vmm &vmm_src) {
    // Cephes poly is accurate for finite x > 0; patch the domain edges
    // to match libm/reference: x<0 (incl -inf) -> NaN, x==0 -> -inf,
    // x==+inf -> +inf. Keep the original x in v_aux3 (standalone-only).
    h_->vmv_v_v(v_aux3_, vmm_src); // save x
    log_compute_vector(vmm_src);
    load_f32_const(f_aux0_, std::numeric_limits<float>::quiet_NaN());
    load_f32_const(f_aux1_, 0.f);
    h_->vmflt_vf(vmm_mask_, v_aux3_, f_aux1_); // x < 0
    h_->vfmerge_vfm(vmm_src, vmm_src, f_aux0_); // -> NaN
    h_->vmfne_vv(vmm_mask_, v_aux3_, v_aux3_); // x is NaN
    h_->vfmerge_vfm(vmm_src, vmm_src, f_aux0_); // -> NaN
    load_f32_const(f_aux0_, -std::numeric_limits<float>::infinity());
    h_->vmfeq_vf(vmm_mask_, v_aux3_, f_aux1_); // x == 0
    h_->vfmerge_vfm(vmm_src, vmm_src, f_aux0_); // -> -inf
    load_f32_const(f_aux0_, std::numeric_limits<float>::infinity());
    h_->vmfeq_vf(vmm_mask_, v_aux3_, f_aux0_); // x == +inf
    h_->vfmerge_vfm(vmm_src, vmm_src, f_aux0_); // -> +inf
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::clip_compute_vector_fwd(
        const Vmm &vmm_src) {
    // clamp(x, alpha, beta)
    clamp(vmm_src, alpha_, beta_);
}

// Unlike x64 (whose single clip method serves both algorithms via the same
// max/min pair), rv64 keeps clip and clip_v2 separate: clip preserves NaN
// lanes (compare+merge), while clip_v2 follows maxNum/minNum behavior.
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::clip_v2_compute_vector_fwd(
        const Vmm &vmm_src) {
    // clip_v2 follows maxNum/minNum behavior: unlike clip, a NaN input
    // selects alpha, matching the reference's ordered comparisons.
    load_f32_const(f_aux0_, alpha_);
    h_->vfmax_vf(vmm_src, vmm_src, f_aux0_);
    load_f32_const(f_aux0_, beta_);
    h_->vfmin_vf(vmm_src, vmm_src, f_aux0_);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::gelu_erf_compute_vector_fwd(
        const Vmm &vmm_src) {
    // 0.5*x*(1 + erf(x/sqrt2)) = 0.5*(x + |x|*erf(|x/sqrt2|)), since
    // x*sign(x) == |x|. The sign-free erf keeps everything in 4 aux
    // (erf uses v_aux0..2, x lives in v_aux3), so it works as a post-op.
    h_->vmv_v_v(v_aux3_, vmm_src); // save x
    load_f32_const(f_aux0_, 0.707106769084930419921875f); // 1/sqrt(2)
    h_->vfmul_vf(vmm_src, vmm_src, f_aux0_);
    erf_compute_vector(vmm_src); // erf(|x/sqrt2|), uses aux0..2
    h_->li(gpr_aux0_, 0x7fffffff);
    h_->vand_vx(v_aux0_, v_aux3_, gpr_aux0_); // |x|
    h_->vfmul_vv(vmm_src, vmm_src, v_aux0_); // |x| * erf
    h_->vfadd_vv(vmm_src, vmm_src, v_aux3_); // + x
    load_f32_const(f_aux0_, 0.5f);
    h_->vfmul_vf(vmm_src, vmm_src, f_aux0_); // 0.5*(x + |x|*erf)
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::round_compute_vector_fwd(
        const Vmm &vmm_src) {
    // nearbyint semantics: vfcvt follows the caller's current FRM, like
    // the reference and x64 MXCSR path. |s| >= 2^23 is already integral;
    // restore it after the i32 round-trip to avoid conversion overflow.
    // Restore NaNs as well, and copy the input sign to preserve -0.
    h_->vmv_v_v(v_aux0_, vmm_src); // save s
    h_->vfcvt_x_f_v(vmm_src, vmm_src); // f32 -> i32 (current FRM)
    h_->vfcvt_f_x_v(vmm_src, vmm_src); // i32 -> f32 = round(s)
    h_->li(gpr_aux0_, 0x7fffffff);
    h_->vand_vx(v_aux1_, v_aux0_, gpr_aux0_); // |s|
    load_f32_const(f_aux0_, 8388608.0f); // 2^23
    h_->vmfge_vf(vmm_mask_, v_aux1_, f_aux0_); // |s| >= 2^23
    h_->vmfne_vv(v_aux1_, v_aux0_, v_aux0_); // input is NaN
    h_->vmor_mm(vmm_mask_, vmm_mask_, v_aux1_);
    h_->vmerge_vvm(vmm_src, vmm_src, v_aux0_); // restore s where large
    h_->vfsgnj_vv(vmm_src, vmm_src, v_aux0_); // preserve the sign of zero
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::hardswish_compute_vector_fwd(
        const Vmm &vmm_src) {
    // x * clamp(alpha * x + beta, 0, 1)
    h_->vmv_v_v(v_aux0_, vmm_src); // save x
    load_f32_const(f_aux0_, alpha_);
    h_->vfmul_vf(vmm_src, vmm_src, f_aux0_);
    load_f32_const(f_aux1_, beta_);
    h_->vfadd_vf(vmm_src, vmm_src, f_aux1_);
    clamp(vmm_src, 0.f, 1.f);
    h_->vfmul_vv(vmm_src, vmm_src, v_aux0_); // x * hardsigmoid(x)
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::hardsigmoid_compute_vector_fwd(
        const Vmm &vmm_src) {
    // clamp(alpha * x + beta, 0, 1)
    load_f32_const(f_aux0_, alpha_);
    h_->vfmul_vf(vmm_src, vmm_src, f_aux0_);
    load_f32_const(f_aux1_, beta_);
    h_->vfadd_vf(vmm_src, vmm_src, f_aux1_);
    clamp(vmm_src, 0.f, 1.f);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::exp_compute_vector_bwd(
        const Vmm &vmm_src) {
    // exp'(s) = exp(s); the use-dst form receives d == exp(s) directly.
    if (!use_dst_) exp_compute_vector_fwd(vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::relu_compute_vector_bwd(
        const Vmm &vmm_src) {
    // s > 0 ? 1 : alpha. relu preserves sign (d>0 <=> s>0), so the use-dst
    // form evaluates the same formula on the forward output.
    const Vmm &a0 = v_aux0_;
    load_f32_const(f_aux0_, 0.f);
    h_->vmfgt_vf(vmm_mask_, vmm_src, f_aux0_);
    load_f32_const(f_aux0_, alpha_);
    h_->vfmv_v_f(a0, f_aux0_);
    load_f32_const(f_aux0_, 1.f);
    h_->vfmv_v_f(vmm_src, f_aux0_);
    h_->vmerge_vvm(vmm_src, a0, vmm_src); // mask ? 1 : alpha
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::elu_compute_vector_bwd(
        const Vmm &vmm_src) {
    if (use_dst_) {
        // d > 0 ? 1 : d + alpha
        load_f32_const(f_aux0_, 0.f);
        h_->vmfgt_vf(vmm_mask_, vmm_src, f_aux0_); // mask: d>0
        load_f32_const(f_aux0_, alpha_);
        h_->vfadd_vf(vmm_src, vmm_src, f_aux0_); // d + alpha
        load_f32_const(f_aux0_, 1.f);
        h_->vfmerge_vfm(vmm_src, vmm_src, f_aux0_); // d>0 -> 1
    } else {
        // s > 0 ? 1 : alpha*exp(s)
        const Vmm &a1 = v_aux1_;
        h_->vmv_v_v(a1, vmm_src); // save s (exp's internal clamp clobbers v0)
        exp_compute_vector_fwd(vmm_src); // exp(s)
        load_f32_const(f_aux0_, alpha_);
        h_->vfmul_vf(vmm_src, vmm_src, f_aux0_); // alpha*exp(s)
        load_f32_const(f_aux0_, 0.f);
        h_->vmfgt_vf(vmm_mask_, a1, f_aux0_); // recompute mask: s>0
        load_f32_const(f_aux0_, 1.f);
        h_->vfmerge_vfm(vmm_src, vmm_src, f_aux0_); // s>0 -> 1
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::tanh_compute_vector_bwd(
        const Vmm &vmm_src) {
    // tanh'(s) = 1 - tanh(s)^2; the use-dst form receives d = tanh(s).
    if (!use_dst_) tanh_compute_vector_fwd(vmm_src);
    h_->vfmul_vv(vmm_src, vmm_src, vmm_src); // d^2
    load_f32_const(f_aux0_, 1.f);
    h_->vfrsub_vf(vmm_src, vmm_src, f_aux0_); // 1 - d^2
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::square_compute_vector_bwd(
        const Vmm &vmm_src) {
    // 2 * s
    load_f32_const(f_aux0_, 2.f);
    h_->vfmul_vf(vmm_src, vmm_src, f_aux0_);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::abs_compute_vector_bwd(
        const Vmm &vmm_src) {
    // s > 0 ? 1 : s < 0 ? -1 : 0
    const Vmm &a0 = v_aux0_;
    const Vmm &a1 = v_aux1_;
    h_->vmv_v_v(a0, vmm_src); // save s
    load_f32_const(f_aux0_, 0.f);
    h_->vfmv_v_f(vmm_src, f_aux0_); // 0
    h_->vmfgt_vf(vmm_mask_, a0, f_aux0_);
    load_f32_const(f_aux1_, 1.f);
    h_->vfmv_v_f(a1, f_aux1_);
    h_->vmerge_vvm(vmm_src, vmm_src, a1); // s>0 -> 1
    h_->vmflt_vf(vmm_mask_, a0, f_aux0_);
    load_f32_const(f_aux1_, -1.f);
    h_->vfmv_v_f(a1, f_aux1_);
    h_->vmerge_vvm(vmm_src, vmm_src, a1); // s<0 -> -1
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::sqrt_compute_vector_bwd(
        const Vmm &vmm_src) {
    // 1 / (2 * sqrt(s)); the use-dst form receives d = sqrt(s).
    if (!use_dst_) h_->vfsqrt_v(vmm_src, vmm_src);
    load_f32_const(f_aux0_, 2.f);
    h_->vfmul_vf(vmm_src, vmm_src, f_aux0_);
    load_f32_const(f_aux0_, 1.f);
    h_->vfrdiv_vf(vmm_src, vmm_src, f_aux0_);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::linear_compute_vector_bwd(
        const Vmm &vmm_src) {
    // alpha
    load_f32_const(f_aux0_, alpha_);
    h_->vfmv_v_f(vmm_src, f_aux0_);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::soft_relu_compute_vector_bwd(
        const Vmm &vmm_src) {
    // srelu'(s) = sigmoid(alpha*s)
    load_f32_const(f_aux0_, alpha_);
    h_->vfmul_vf(vmm_src, vmm_src, f_aux0_);
    logistic_compute_vector_fwd(vmm_src);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::logistic_compute_vector_bwd(
        const Vmm &vmm_src) {
    // sig'(s) = sig(s)*(1 - sig(s)); the use-dst form receives d = sig(s).
    if (!use_dst_) logistic_compute_vector_fwd(vmm_src);
    const Vmm &a0 = v_aux0_;
    load_f32_const(f_aux0_, 1.f);
    h_->vfrsub_vf(a0, vmm_src, f_aux0_); // 1 - d
    h_->vfmul_vv(vmm_src, vmm_src, a0); // d*(1 - d)
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::mish_compute_vector_bwd(
        const Vmm &vmm_src) {
    // mish'(s) = th + s*sig*(1 - th^2), th = tanh(softplus(s)),
    // sig = sigmoid(s). With w = 1 + exp(s) and e = exp(s):
    //   sig = e/w         (== (w-1)/w, no cancellation for e<<1)
    //   th  = (w^2-1)/(w^2+1), w^2-1 = e*(w+1)  (no cancellation)
    // Overflow-safe: e,w -> inf gives sig=1, th=1.
    const Vmm &a0 = v_aux0_;
    const Vmm &a1 = v_aux1_;
    const Vmm &a2 = v_aux2_;
    const Vmm &s = v_aux3_, &w = v_aux4_;
    h_->vmv_v_v(s, vmm_src); // save s
    exp_compute_vector_fwd(vmm_src); // e = exp(s) in v (uses a0/a2)
    // cap e so w^2 stays finite; beyond this th,sig == 1 to f32 anyway.
    load_f32_const(f_aux0_, 1e18f);
    h_->vfmin_vf(vmm_src, vmm_src, f_aux0_);
    load_f32_const(f_aux0_, 1.f);
    h_->vfadd_vf(w, vmm_src, f_aux0_); // w = 1 + e
    h_->vfdiv_vv(a1, vmm_src, w); // sig = e/w
    h_->vfmul_vv(a0, w, w); // w^2
    h_->vfadd_vf(a2, w, f_aux0_); // w + 1
    h_->vfmul_vv(a2, a2, vmm_src); // e*(w+1) = w^2 - 1
    h_->vfadd_vf(vmm_src, a0, f_aux0_); // w^2 + 1
    h_->vfdiv_vv(a0, a2, vmm_src); // th = (w^2-1)/(w^2+1)
    h_->vfmul_vv(vmm_src, a0, a0); // th^2
    h_->vfrsub_vf(vmm_src, vmm_src, f_aux0_); // 1 - th^2
    h_->vfmul_vv(vmm_src, vmm_src, a1); // sig*(1-th^2)
    h_->vfmul_vv(vmm_src, vmm_src, s); // s*sig*(1-th^2)
    h_->vfadd_vv(vmm_src, vmm_src, a0); // + th
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::gelu_tanh_compute_vector_bwd(
        const Vmm &vmm_src) {
    // ds = 0.5*(1+t)*(1 + s*(1-t)*dg), t=tanh(g),
    // g = k*s*(1+c*s^2), dg = k*(1+3c*s^2), k=sqrt(2/pi), c=0.044715.
    const Vmm &a0 = v_aux0_;
    const Vmm &s = v_aux3_, &dg = v_aux4_;
    constexpr float k = 0.79788458347320556640625f, c = 0.044715f;
    h_->vmv_v_v(s, vmm_src); // save s
    h_->vfmul_vv(a0, vmm_src, vmm_src); // s^2
    load_f32_const(f_aux0_, 3.f * c);
    h_->vfmul_vf(dg, a0, f_aux0_); // 3c*s^2
    load_f32_const(f_aux0_, 1.f);
    h_->vfadd_vf(dg, dg, f_aux0_); // 1 + 3c*s^2
    load_f32_const(f_aux0_, k);
    h_->vfmul_vf(dg, dg, f_aux0_); // dg
    load_f32_const(f_aux0_, c);
    h_->vfmul_vf(a0, a0, f_aux0_); // c*s^2
    load_f32_const(f_aux0_, 1.f);
    h_->vfadd_vf(a0, a0, f_aux0_); // 1 + c*s^2
    h_->vfmul_vv(vmm_src, vmm_src, a0); // s*(1+c*s^2)
    load_f32_const(f_aux0_, k);
    h_->vfmul_vf(vmm_src, vmm_src, f_aux0_); // g
    tanh_compute_vector_fwd(vmm_src); // t = tanh(g) (uses a0/a1/a2)
    load_f32_const(f_aux0_, 1.f);
    h_->vfrsub_vf(a0, vmm_src, f_aux0_); // 1 - t
    h_->vfmul_vv(a0, a0, dg); // dg*(1 - t)
    h_->vfmul_vv(a0, a0, s); // s*dg*(1 - t)
    load_f32_const(f_aux0_, 1.f);
    h_->vfadd_vf(a0, a0, f_aux0_); // 1 + s*(1-t)*dg
    load_f32_const(f_aux0_, 1.f);
    h_->vfadd_vf(vmm_src, vmm_src, f_aux0_); // 1 + t
    load_f32_const(f_aux0_, 0.5f);
    h_->vfmul_vf(vmm_src, vmm_src, f_aux0_); // 0.5*(1+t)
    h_->vfmul_vv(vmm_src, vmm_src, a0); // ds
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::swish_compute_vector_bwd(
        const Vmm &vmm_src) {
    // v = sig(alpha*s); ds = v + s*alpha*v*(1 - v). logistic clobbers
    // a0/a2, so keep s in a1 (free across the logistic building block).
    const Vmm &a0 = v_aux0_;
    const Vmm &a1 = v_aux1_;
    h_->vmv_v_v(a1, vmm_src); // save s
    load_f32_const(f_aux0_, alpha_);
    h_->vfmul_vf(vmm_src, vmm_src, f_aux0_); // alpha*s
    logistic_compute_vector_fwd(vmm_src); // v = sig(alpha*s)
    load_f32_const(f_aux0_, 1.f);
    h_->vfrsub_vf(a0, vmm_src, f_aux0_); // 1 - v
    h_->vfmul_vv(a0, a0, vmm_src); // v*(1 - v)
    load_f32_const(f_aux0_, alpha_);
    h_->vfmul_vf(a0, a0, f_aux0_); // alpha*v*(1-v)
    h_->vfmul_vv(a0, a0, a1); // s*alpha*v*(1-v)
    h_->vfadd_vv(vmm_src, vmm_src, a0); // + v
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::log_compute_vector_bwd(
        const Vmm &vmm_src) {
    // log'(s) = 1/s
    load_f32_const(f_aux0_, 1.f);
    h_->vfrdiv_vf(vmm_src, vmm_src, f_aux0_);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::clip_compute_vector_bwd(
        const Vmm &vmm_src) {
    // (alpha < s && s <= beta) ? 1 : 0
    const Vmm &a0 = v_aux0_;
    const Vmm &a1 = v_aux1_;
    h_->vmv_v_v(a0, vmm_src); // save s
    load_f32_const(f_aux0_, 1.f);
    h_->vfmv_v_f(vmm_src, f_aux0_); // 1
    load_f32_const(f_aux1_, 0.f);
    h_->vfmv_v_f(a1, f_aux1_); // 0
    load_f32_const(f_aux0_, alpha_);
    h_->vmfle_vf(vmm_mask_, a0, f_aux0_);
    h_->vmerge_vvm(vmm_src, vmm_src, a1); // s<=alpha -> 0
    load_f32_const(f_aux0_, beta_);
    h_->vmfgt_vf(vmm_mask_, a0, f_aux0_);
    h_->vmerge_vvm(vmm_src, vmm_src, a1); // s>beta -> 0
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::clip_v2_compute_vector_bwd(
        const Vmm &vmm_src) {
    // (alpha < s && s < beta) ? 1 : 0; the use-dst form evaluates the same
    // formula on d.
    const Vmm &a0 = v_aux0_;
    const Vmm &a1 = v_aux1_;
    h_->vmv_v_v(a0, vmm_src); // save value
    load_f32_const(f_aux0_, 1.f);
    h_->vfmv_v_f(vmm_src, f_aux0_); // 1
    load_f32_const(f_aux1_, 0.f);
    h_->vfmv_v_f(a1, f_aux1_); // 0
    load_f32_const(f_aux0_, alpha_);
    h_->vmfle_vf(vmm_mask_, a0, f_aux0_);
    h_->vmerge_vvm(vmm_src, vmm_src, a1); // s<=alpha -> 0
    load_f32_const(f_aux0_, beta_);
    h_->vmfge_vf(vmm_mask_, a0, f_aux0_);
    h_->vmerge_vvm(vmm_src, vmm_src, a1); // s>=beta -> 0
    h_->vmfne_vv(vmm_mask_, a0, a0);
    h_->vmerge_vvm(vmm_src, vmm_src, a1); // unordered (NaN) -> 0
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::gelu_erf_compute_vector_bwd(
        const Vmm &vmm_src) {
    // gelu_erf'(s) = 0.5*(1 + erf(u) + u*C*exp(-u^2)),
    // u = s*sqrt(2)/2, C = 2/sqrt(pi).
    constexpr float c = 0.707106769084930419921875f; // 1/sqrt(2)
    constexpr float C = 1.12837922573089599609375f; // 2/sqrt(pi)
    const Vmm &a0 = v_aux0_;
    const Vmm &a1 = v_aux1_;
    const Vmm &s = v_aux3_, &u = v_aux4_;
    h_->vmv_v_v(s, vmm_src); // save s
    load_f32_const(f_aux0_, c);
    h_->vfmul_vf(vmm_src, vmm_src, f_aux0_); // u = s/sqrt2
    h_->vmv_v_v(u, vmm_src); // save u
    erf_compute_vector(vmm_src); // erf(|u|), uses a0..a2
    h_->li(gpr_aux0_, 0x80000000);
    h_->vand_vx(a0, s, gpr_aux0_); // sign(s) == sign(u)
    h_->li(gpr_aux0_, 0x7fffffff);
    h_->vand_vx(vmm_src, vmm_src, gpr_aux0_);
    h_->vor_vv(vmm_src, vmm_src, a0); // erf(u) = copysign(erf(|u|), u)
    h_->vfmul_vv(a1, u, u); // u^2
    h_->vfneg_v(a1, a1); // -u^2
    exp_compute_vector_fwd(a1); // exp(-u^2) (uses a0/a2)
    h_->vfmul_vv(a1, a1, u); // u*exp(-u^2)
    load_f32_const(f_aux0_, C);
    h_->vfmul_vf(a1, a1, f_aux0_); // u*C*exp(-u^2)
    h_->vfadd_vv(vmm_src, vmm_src, a1); // erf(u) + term
    load_f32_const(f_aux0_, 1.f);
    h_->vfadd_vf(vmm_src, vmm_src, f_aux0_); // 1 + ...
    load_f32_const(f_aux0_, 0.5f);
    h_->vfmul_vf(vmm_src, vmm_src, f_aux0_); // 0.5*(...)
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::hardswish_compute_vector_bwd(
        const Vmm &vmm_src) {
    // v = alpha*s+beta; w = 2*alpha*s+beta; v<=0?0 : v>=1?1 : w
    const Vmm &a0 = v_aux0_;
    const Vmm &a1 = v_aux1_;
    load_f32_const(f_aux0_, alpha_);
    h_->vfmul_vf(a0, vmm_src, f_aux0_); // alpha*s
    load_f32_const(f_aux1_, 2.f);
    h_->vfmul_vf(vmm_src, a0, f_aux1_); // 2*alpha*s
    load_f32_const(f_aux0_, beta_);
    h_->vfadd_vf(a0, a0, f_aux0_); // a0 = v
    h_->vfadd_vf(vmm_src, vmm_src, f_aux0_); // v = w
    load_f32_const(f_aux1_, 1.f);
    h_->vfmv_v_f(a1, f_aux1_); // 1
    load_f32_const(f_aux0_, 1.f);
    h_->vmfge_vf(vmm_mask_, a0, f_aux0_);
    h_->vmerge_vvm(vmm_src, vmm_src, a1); // v>=1 -> 1
    load_f32_const(f_aux1_, 0.f);
    h_->vfmv_v_f(a1, f_aux1_); // 0
    load_f32_const(f_aux0_, 0.f);
    h_->vmfle_vf(vmm_mask_, a0, f_aux0_);
    h_->vmerge_vvm(vmm_src, vmm_src, a1); // v<=0 -> 0
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::hardsigmoid_compute_vector_bwd(
        const Vmm &vmm_src) {
    // v = alpha*s + beta; (v<=0 || v>=1) ? 0 : alpha
    const Vmm &a0 = v_aux0_;
    const Vmm &a1 = v_aux1_;
    load_f32_const(f_aux0_, alpha_);
    h_->vfmul_vf(a0, vmm_src, f_aux0_);
    load_f32_const(f_aux0_, beta_);
    h_->vfadd_vf(a0, a0, f_aux0_); // a0 = v
    load_f32_const(f_aux0_, alpha_);
    h_->vfmv_v_f(vmm_src, f_aux0_); // alpha
    load_f32_const(f_aux1_, 0.f);
    h_->vfmv_v_f(a1, f_aux1_); // 0
    load_f32_const(f_aux0_, 1.f);
    h_->vmfge_vf(vmm_mask_, a0, f_aux0_);
    h_->vmerge_vvm(vmm_src, vmm_src, a1); // v>=1 -> 0
    load_f32_const(f_aux0_, 0.f);
    h_->vmfle_vf(vmm_mask_, a0, f_aux0_);
    h_->vmerge_vvm(vmm_src, vmm_src, a1); // v<=0 -> 0
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::compute_body(const Vmm &vmm_src) {
    using namespace alg_kind;

    if (is_fwd_) {
        // The *_use_dst_for_bwd variants share the forward math of their base
        // algorithm (the forward companion of a use-dst backward training
        // pair), so they fold into the same case, as on x64/aarch64.
        switch (alg_) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu:
                if (alpha_ == 0.f)
                    relu_zero_ns_compute_vector_fwd(vmm_src);
                else
                    relu_compute_vector_fwd(vmm_src);
                break;
            case eltwise_elu_use_dst_for_bwd:
            case eltwise_elu: elu_compute_vector_fwd(vmm_src); break;
            case eltwise_tanh_use_dst_for_bwd:
            case eltwise_tanh: tanh_compute_vector_fwd(vmm_src); break;
            case eltwise_square: square_compute_vector_fwd(vmm_src); break;
            case eltwise_abs: abs_compute_vector_fwd(vmm_src); break;
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt: sqrt_compute_vector_fwd(vmm_src); break;
            case eltwise_swish: swish_compute_vector_fwd(vmm_src); break;
            case eltwise_linear: linear_compute_vector_fwd(vmm_src); break;
            case eltwise_soft_relu:
                soft_relu_compute_vector_fwd(vmm_src);
                break;
            case eltwise_mish: mish_compute_vector_fwd(vmm_src); break;
            case eltwise_logistic_use_dst_for_bwd:
            case eltwise_logistic: logistic_compute_vector_fwd(vmm_src); break;
            case eltwise_exp_use_dst_for_bwd:
            case eltwise_exp: exp_compute_vector_fwd(vmm_src); break;
            case eltwise_gelu_tanh:
                gelu_tanh_compute_vector_fwd(vmm_src);
                break;
            case eltwise_log: log_compute_vector_fwd(vmm_src); break;
            // clip and clip_v2 stay separate (unlike x64's shared method):
            // clip preserves NaN lanes, clip_v2 follows maxNum/minNum.
            case eltwise_clip: clip_compute_vector_fwd(vmm_src); break;
            case eltwise_clip_v2_use_dst_for_bwd:
            case eltwise_clip_v2: clip_v2_compute_vector_fwd(vmm_src); break;
            case eltwise_gelu_erf: gelu_erf_compute_vector_fwd(vmm_src); break;
            case eltwise_round: round_compute_vector_fwd(vmm_src); break;
            case eltwise_hardswish:
                hardswish_compute_vector_fwd(vmm_src);
                break;
            case eltwise_hardsigmoid:
                hardsigmoid_compute_vector_fwd(vmm_src);
                break;
            default: assert(!"unsupported eltwise algorithm");
        }
    } else {
        switch (alg_) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu: relu_compute_vector_bwd(vmm_src); break;
            case eltwise_elu_use_dst_for_bwd:
            case eltwise_elu: elu_compute_vector_bwd(vmm_src); break;
            case eltwise_tanh_use_dst_for_bwd:
            case eltwise_tanh: tanh_compute_vector_bwd(vmm_src); break;
            case eltwise_square: square_compute_vector_bwd(vmm_src); break;
            case eltwise_abs: abs_compute_vector_bwd(vmm_src); break;
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt: sqrt_compute_vector_bwd(vmm_src); break;
            case eltwise_linear: linear_compute_vector_bwd(vmm_src); break;
            case eltwise_soft_relu:
                soft_relu_compute_vector_bwd(vmm_src);
                break;
            case eltwise_logistic_use_dst_for_bwd:
            case eltwise_logistic: logistic_compute_vector_bwd(vmm_src); break;
            case eltwise_mish: mish_compute_vector_bwd(vmm_src); break;
            case eltwise_exp_use_dst_for_bwd:
            case eltwise_exp: exp_compute_vector_bwd(vmm_src); break;
            case eltwise_gelu_tanh:
                gelu_tanh_compute_vector_bwd(vmm_src);
                break;
            case eltwise_swish: swish_compute_vector_bwd(vmm_src); break;
            case eltwise_log: log_compute_vector_bwd(vmm_src); break;
            case eltwise_clip: clip_compute_vector_bwd(vmm_src); break;
            case eltwise_clip_v2_use_dst_for_bwd:
            case eltwise_clip_v2: clip_v2_compute_vector_bwd(vmm_src); break;
            case eltwise_gelu_erf: gelu_erf_compute_vector_bwd(vmm_src); break;
            case eltwise_hardswish:
                hardswish_compute_vector_bwd(vmm_src);
                break;
            case eltwise_hardsigmoid:
                hardsigmoid_compute_vector_bwd(vmm_src);
                break;
            default: assert(!"unsupported eltwise algorithm");
        }
    }

    // eltwise post-op scale: result = scale * alg(x)
    if (scale_ != 1.f) {
        load_f32_const(f_aux0_, scale_);
        h_->vfmul_vf(vmm_src, vmm_src, f_aux0_);
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::compute_vector_range(
        size_t start_idx, size_t end_idx) {
    for (size_t i = start_idx; i < end_idx; i++)
        compute_body(Vmm(i));
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs) {
    for (size_t idx : vmm_idxs)
        compute_body(Vmm(idx));
}

template struct jit_uni_eltwise_injector_t<v>;
template struct jit_uni_eltwise_injector_t<zvfh>;

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
