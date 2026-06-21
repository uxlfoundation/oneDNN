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
#include <cstring>

#include "cpu/rv64/injectors/jit_uni_eltwise_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

namespace eltwise_injector {

bool is_alg_supported(alg_kind_t alg) {
    using namespace alg_kind;
    switch (alg) {
        // arithmetic, table-free, mask-free
        case eltwise_relu:
        case eltwise_square:
        case eltwise_abs:
        case eltwise_sqrt:
        case eltwise_linear:
        case eltwise_clip:
        case eltwise_hardsigmoid:
        case eltwise_hardswish:
        // transcendental, built on the inline-coefficient exp() primitive
        case eltwise_exp:
        case eltwise_logistic:
        case eltwise_tanh:
        case eltwise_elu:
        case eltwise_swish:
        case eltwise_gelu_tanh: return true;
        default: return false;
    }
}

} // namespace eltwise_injector

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
void jit_uni_eltwise_injector_t<isa>::clamp(const Vmm &v, float lo, float hi) {
    const Xbyak_riscv::VReg v_mask(0);
    load_f32_const(f_aux0_, lo);
    h_->vmflt_vf(v_mask, v, f_aux0_);
    h_->vfmerge_vfm(v, v, f_aux0_);
    load_f32_const(f_aux1_, hi);
    h_->vmfgt_vf(v_mask, v, f_aux1_);
    h_->vfmerge_vfm(v, v, f_aux1_);
}

// exp(x) via base-2 range reduction + degree-5 minimax polynomial (the classic
// Cephes/sse_mathfun expf), with all constants materialized inline as FP
// scalars (RVV vf-form ops take an FReg directly, so no constant table is
// needed). Uses two aux vector groups (v_aux0_ scratch/poly, v_aux2_ integer
// exponent); v_aux1_ is left free so callers can stash x across the call.
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::exp_compute_vector(const Vmm &v) {
    const Vmm &a0 = v_aux0_; // poly accumulator / scratch
    const Vmm &a2
            = v_aux2_; // integer exponent n (must survive to the 2^n step)

    // clamp x to [ln(FLT_MIN), ln(FLT_MAX)] to keep the result finite. The
    // 2*2^(n-1) reconstruction (below) makes the lowest exponent bucket (n ==
    // -126) come out as exactly 0: (n-1)+127 == 0 builds +0.0, not subnormal
    // 2^-127. This matches x64 (which masks the < ln(FLT_MIN) lanes to 0); the
    // error vs the true value (~1e-38) is negligible.
    clamp(v, -87.33654475f, 88.72283905f); // MINLOGF, MAXLOGF

    // n = round(x * log2e); z = (float)n
    load_f32_const(f_aux0_, 1.44269504088896341f); // log2e
    h_->vfmul_vf(a0, v, f_aux0_);
    h_->vfcvt_x_f_v(a2, a0); // n (int, round-to-nearest)
    h_->vfcvt_f_x_v(a0, a2); // z = (float)n

    // r = x - z*C1 - z*C2  (extended-precision ln2), r in [-ln2/2, ln2/2]
    load_f32_const(f_aux0_, 0.693359375f); // C1
    h_->vfnmsac_vf(v, f_aux0_, a0); // v -= C1*z
    load_f32_const(f_aux0_, -2.12194440e-4f); // C2
    h_->vfnmsac_vf(v, f_aux0_, a0); // v -= C2*z  => v = r

    // poly5(r) by Horner; coefficients p0..p5 (Cephes expf)
    load_f32_const(f_aux0_, 1.9875691500e-4f);
    h_->vfmv_v_f(a0, f_aux0_); // y = p0
    const float p[] = {1.3981999507e-3f, 8.3334519073e-3f, 4.1665795894e-2f,
            1.6666665459e-1f, 5.0000001201e-1f};
    for (float c : p) {
        h_->vfmul_vv(a0, a0, v); // y *= r
        load_f32_const(f_aux0_, c);
        h_->vfadd_vf(a0, a0, f_aux0_); // y += c
    }
    // y = y*r^2 + r + 1, computed as ((y*r)*r) + r + 1 to avoid a 3rd aux reg
    h_->vfmul_vv(a0, a0, v);
    h_->vfmul_vv(a0, a0, v);
    h_->vfadd_vv(a0, a0, v);
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

    h_->vfmul_vv(v, a0, a2); // poly * 2^(n-1)
    load_f32_const(f_aux0_, 2.f);
    h_->vfmul_vf(v, v, f_aux0_); // * 2 = poly * 2^n = exp(x)
}

// sigmoid(x) = 1 / (1 + exp(-x)); numerically stable for both tails.
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::logistic_compute_vector(const Vmm &v) {
    h_->vfneg_v(v, v); // -x
    exp_compute_vector(v); // exp(-x)
    load_f32_const(f_aux0_, 1.f);
    h_->vfadd_vf(v, v, f_aux0_); // 1 + exp(-x)
    load_f32_const(f_aux0_, 1.f);
    h_->vfrdiv_vf(v, v, f_aux0_); // 1 / (1 + exp(-x))
}

// Backward: transform a register holding `s` into alg'(s). Uses v0 as the RVV
// mask register (the standalone eltwise primitive keeps v0 free) plus the two
// aux vector groups. The caller multiplies the resulting alg'(s) by diff_dst.
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::compute_vector_bwd(const Vmm &v) {
    using namespace alg_kind;
    const Xbyak_riscv::VReg v_mask(0);
    const Vmm &a0 = v_aux0_;
    const Vmm &a1 = v_aux1_;

    switch (alg_) {
        case eltwise_relu:
            // s > 0 ? 1 : alpha
            load_f32_const(f_aux0_, 0.f);
            h_->vmfgt_vf(v_mask, v, f_aux0_);
            load_f32_const(f_aux0_, alpha_);
            h_->vfmv_v_f(a0, f_aux0_);
            load_f32_const(f_aux0_, 1.f);
            h_->vfmv_v_f(v, f_aux0_);
            h_->vmerge_vvm(v, a0, v); // mask ? 1 : alpha
            break;

        case eltwise_square:
            // 2 * s
            load_f32_const(f_aux0_, 2.f);
            h_->vfmul_vf(v, v, f_aux0_);
            break;

        case eltwise_abs:
            // s > 0 ? 1 : s < 0 ? -1 : 0
            h_->vmv_v_v(a0, v); // save s
            load_f32_const(f_aux0_, 0.f);
            h_->vfmv_v_f(v, f_aux0_); // 0
            h_->vmfgt_vf(v_mask, a0, f_aux0_);
            load_f32_const(f_aux1_, 1.f);
            h_->vfmv_v_f(a1, f_aux1_);
            h_->vmerge_vvm(v, v, a1); // s>0 -> 1
            h_->vmflt_vf(v_mask, a0, f_aux0_);
            load_f32_const(f_aux1_, -1.f);
            h_->vfmv_v_f(a1, f_aux1_);
            h_->vmerge_vvm(v, v, a1); // s<0 -> -1
            break;

        case eltwise_sqrt:
            // 1 / (2 * sqrt(s))
            h_->vfsqrt_v(v, v);
            load_f32_const(f_aux0_, 2.f);
            h_->vfmul_vf(v, v, f_aux0_);
            load_f32_const(f_aux0_, 1.f);
            h_->vfrdiv_vf(v, v, f_aux0_);
            break;

        case eltwise_linear:
            // alpha
            load_f32_const(f_aux0_, alpha_);
            h_->vfmv_v_f(v, f_aux0_);
            break;

        case eltwise_clip:
            // (alpha < s && s <= beta) ? 1 : 0
            h_->vmv_v_v(a0, v); // save s
            load_f32_const(f_aux0_, 1.f);
            h_->vfmv_v_f(v, f_aux0_); // 1
            load_f32_const(f_aux1_, 0.f);
            h_->vfmv_v_f(a1, f_aux1_); // 0
            load_f32_const(f_aux0_, alpha_);
            h_->vmfle_vf(v_mask, a0, f_aux0_);
            h_->vmerge_vvm(v, v, a1); // s<=alpha -> 0
            load_f32_const(f_aux0_, beta_);
            h_->vmfgt_vf(v_mask, a0, f_aux0_);
            h_->vmerge_vvm(v, v, a1); // s>beta -> 0
            break;

        case eltwise_hardsigmoid:
            // v = alpha*s + beta; (v<=0 || v>=1) ? 0 : alpha
            load_f32_const(f_aux0_, alpha_);
            h_->vfmul_vf(a0, v, f_aux0_);
            load_f32_const(f_aux0_, beta_);
            h_->vfadd_vf(a0, a0, f_aux0_); // a0 = v
            load_f32_const(f_aux0_, alpha_);
            h_->vfmv_v_f(v, f_aux0_); // alpha
            load_f32_const(f_aux1_, 0.f);
            h_->vfmv_v_f(a1, f_aux1_); // 0
            load_f32_const(f_aux0_, 1.f);
            h_->vmfge_vf(v_mask, a0, f_aux0_);
            h_->vmerge_vvm(v, v, a1); // v>=1 -> 0
            load_f32_const(f_aux0_, 0.f);
            h_->vmfle_vf(v_mask, a0, f_aux0_);
            h_->vmerge_vvm(v, v, a1); // v<=0 -> 0
            break;

        case eltwise_hardswish:
            // v = alpha*s+beta; w = 2*alpha*s+beta; v<=0?0 : v>=1?1 : w
            load_f32_const(f_aux0_, alpha_);
            h_->vfmul_vf(a0, v, f_aux0_); // alpha*s
            load_f32_const(f_aux1_, 2.f);
            h_->vfmul_vf(v, a0, f_aux1_); // 2*alpha*s
            load_f32_const(f_aux0_, beta_);
            h_->vfadd_vf(a0, a0, f_aux0_); // a0 = v
            h_->vfadd_vf(v, v, f_aux0_); // v = w
            load_f32_const(f_aux1_, 1.f);
            h_->vfmv_v_f(a1, f_aux1_); // 1
            load_f32_const(f_aux0_, 1.f);
            h_->vmfge_vf(v_mask, a0, f_aux0_);
            h_->vmerge_vvm(v, v, a1); // v>=1 -> 1
            load_f32_const(f_aux1_, 0.f);
            h_->vfmv_v_f(a1, f_aux1_); // 0
            load_f32_const(f_aux0_, 0.f);
            h_->vmfle_vf(v_mask, a0, f_aux0_);
            h_->vmerge_vvm(v, v, a1); // v<=0 -> 0
            break;

        default: assert(!"unsupported eltwise bwd alg"); break;
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_t<isa>::compute_body(const Vmm &v) {
    using namespace alg_kind;

    if (!is_fwd_) {
        compute_vector_bwd(v);
        return;
    }

    switch (alg_) {
        case eltwise_relu:
            if (alpha_ == 0.f) {
                // relu(x) = x < 0 ? 0 : x. Keep NaN lanes unchanged.
                const Xbyak_riscv::VReg v_mask(0);
                load_f32_const(f_aux0_, 0.f);
                h_->vmflt_vf(v_mask, v, f_aux0_);
                h_->vfmerge_vfm(v, v, f_aux0_);
            } else {
                // leaky relu = x>0 ? x : alpha*x. NaN takes the false branch,
                // but alpha*NaN is still NaN.
                const Xbyak_riscv::VReg v_mask(0);
                load_f32_const(f_aux0_, 0.f);
                load_f32_const(f_aux1_, alpha_);
                h_->vfmul_vf(v_aux0_, v, f_aux1_);
                h_->vmfgt_vf(v_mask, v, f_aux0_);
                h_->vmerge_vvm(v, v_aux0_, v);
            }
            break;

        case eltwise_square:
            // x * x
            h_->vfmul_vv(v, v, v);
            break;

        case eltwise_abs:
            // clear the IEEE-754 sign bit of each f32 lane (mask-free, no aux
            // vector). Operates on the raw bits and assumes vtype SEW=e32 (the
            // 0x7fffffff mask clears bit 31 of each 32-bit lane).
            h_->li(gpr_aux0_, 0x7fffffff);
            h_->vand_vx(v, v, gpr_aux0_);
            break;

        case eltwise_sqrt: h_->vfsqrt_v(v, v); break;

        case eltwise_linear:
            // alpha * x + beta
            load_f32_const(f_aux0_, alpha_);
            h_->vfmul_vf(v, v, f_aux0_);
            load_f32_const(f_aux1_, beta_);
            h_->vfadd_vf(v, v, f_aux1_);
            break;

        case eltwise_clip:
            // clamp(x, alpha, beta)
            clamp(v, alpha_, beta_);
            break;

        case eltwise_hardsigmoid:
            // clamp(alpha * x + beta, 0, 1)
            load_f32_const(f_aux0_, alpha_);
            h_->vfmul_vf(v, v, f_aux0_);
            load_f32_const(f_aux1_, beta_);
            h_->vfadd_vf(v, v, f_aux1_);
            clamp(v, 0.f, 1.f);
            break;

        case eltwise_hardswish:
            // x * clamp(alpha * x + beta, 0, 1)
            h_->vmv_v_v(v_aux0_, v); // save x
            load_f32_const(f_aux0_, alpha_);
            h_->vfmul_vf(v, v, f_aux0_);
            load_f32_const(f_aux1_, beta_);
            h_->vfadd_vf(v, v, f_aux1_);
            clamp(v, 0.f, 1.f);
            h_->vfmul_vv(v, v, v_aux0_); // x * hardsigmoid(x)
            break;

        case eltwise_exp: exp_compute_vector(v); break;

        case eltwise_logistic: logistic_compute_vector(v); break;

        case eltwise_tanh: {
            // tanh(x) = 2 * sigmoid(2x) - 1. Reuses the logistic/exp polynomial
            // (no dedicated tanh approximation like x64), so ULP error is a bit
            // larger but within benchdnn tolerance (gelu_tanh below reuses it).
            //
            // For small |x| this formula loses relative accuracy: tanh(x) ~= x
            // is tiny, yet it is built as 2*sigmoid(2x)-1 ~= 2*0.5 - 1, so the
            // exp polynomial's ~1e-7 absolute error becomes a large *relative*
            // error (catastrophic cancellation). Blend in the linear approx
            // tanh(x) ~= x for small |x| (where the x^3/3 error is negligible),
            // mask-free so the forward path never touches v0: result =
            // t + w*(x - t), with w a clamped ramp 1->0 over T1<|x|<T2. Inside
            // the ramp both x and t already match tanh to <2e-5, so the convex
            // blend is accurate regardless of w.
            constexpr float t1 = 0.002f, t2 = 0.008f; // blend band (in |x|)
            h_->vmv_v_v(v_aux1_, v); // save x
            load_f32_const(f_aux0_, 2.f);
            h_->vfmul_vf(v, v, f_aux0_);
            logistic_compute_vector(v);
            load_f32_const(f_aux0_, 2.f);
            h_->vfmul_vf(v, v, f_aux0_);
            load_f32_const(f_aux0_, -1.f);
            h_->vfadd_vf(v, v, f_aux0_); // v = t = 2*sigmoid(2x)-1
            h_->vfsub_vv(v_aux0_, v_aux1_, v); // v_aux0 = x - t
            h_->vfmul_vv(v_aux2_, v_aux1_, v_aux1_); // v_aux2 = x^2
            load_f32_const(f_aux0_, t2 * t2);
            h_->vfrsub_vf(v_aux2_, v_aux2_, f_aux0_); // t2^2 - x^2
            load_f32_const(f_aux0_, 1.f / (t2 * t2 - t1 * t1));
            h_->vfmul_vf(v_aux2_, v_aux2_, f_aux0_); // w (unclamped)
            clamp(v_aux2_, 0.f, 1.f); // w in [0,1]
            h_->vfmacc_vv(v, v_aux2_, v_aux0_); // v = t + w*(x - t)
            break;
        }

        case eltwise_elu: {
            // x>0 ? x : alpha*(exp(x)-1). NaN takes the false branch and
            // remains NaN through exp/sub/mul.
            const Xbyak_riscv::VReg v_mask(0);
            h_->vmv_v_v(v_aux1_, v); // save x
            load_f32_const(f_aux0_, 0.f);
            h_->vmfgt_vf(v_mask, v_aux1_, f_aux0_);
            exp_compute_vector(v); // exp(x)
            load_f32_const(f_aux0_, 1.f);
            h_->vfsub_vf(v, v, f_aux0_); // exp(...) - 1
            load_f32_const(f_aux0_, alpha_);
            h_->vfmul_vf(v, v, f_aux0_); // alpha*(...)
            load_f32_const(f_aux0_, 0.f);
            h_->vmfgt_vf(v_mask, v_aux1_, f_aux0_);
            h_->vmerge_vvm(v, v, v_aux1_);
            break;
        }

        case eltwise_swish:
            // x * sigmoid(alpha * x)
            h_->vmv_v_v(v_aux1_, v); // save x (exp/logistic keep aux1 free)
            load_f32_const(f_aux0_, alpha_);
            h_->vfmul_vf(v, v, f_aux0_); // alpha*x
            logistic_compute_vector(v); // sigmoid(alpha*x)
            h_->vfmul_vv(v, v, v_aux1_); // x * sigmoid(alpha*x)
            break;

        case eltwise_gelu_tanh:
            // 0.5*x*(1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) ))
            h_->vmv_v_v(v_aux1_, v); // save x
            h_->vfmul_vv(v, v, v); // x^2
            load_f32_const(f_aux0_, 0.044715f);
            h_->vfmul_vf(v, v, f_aux0_); // 0.044715*x^2
            load_f32_const(f_aux0_, 1.f);
            h_->vfadd_vf(v, v, f_aux0_); // 1 + 0.044715*x^2
            h_->vfmul_vv(v, v, v_aux1_); // x*(1 + 0.044715*x^2)
            load_f32_const(f_aux0_, 0.7978845608028654f); // sqrt(2/pi)
            h_->vfmul_vf(v, v, f_aux0_); // inner argument
            // tanh(inner): 2*sigmoid(2*inner) - 1
            load_f32_const(f_aux0_, 2.f);
            h_->vfmul_vf(v, v, f_aux0_);
            logistic_compute_vector(v);
            load_f32_const(f_aux0_, 2.f);
            h_->vfmul_vf(v, v, f_aux0_);
            load_f32_const(f_aux0_, -1.f);
            h_->vfadd_vf(v, v, f_aux0_); // tanh(inner)
            load_f32_const(f_aux0_, 1.f);
            h_->vfadd_vf(v, v, f_aux0_); // 1 + tanh
            h_->vfmul_vv(v, v, v_aux1_); // x*(1+tanh)
            load_f32_const(f_aux0_, 0.5f);
            h_->vfmul_vf(v, v, f_aux0_); // 0.5*x*(1+tanh)
            break;

        default: assert(!"unsupported eltwise alg"); break;
    }

    // eltwise post-op scale: result = scale * alg(x)
    if (scale_ != 1.f) {
        load_f32_const(f_aux0_, scale_);
        h_->vfmul_vf(v, v, f_aux0_);
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
