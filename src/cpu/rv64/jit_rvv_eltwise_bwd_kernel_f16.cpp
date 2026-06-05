#include <assert.h>

#include "cpu/rv64/jit_rvv_eltwise_bwd_kernel_f16.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

namespace {

template <alg_kind_t alg>
void dispatch_jit_eltwise_bwd_f16(
        const jit_rvv_eltwise_bwd_kernel_f16_t::call_params_t *p) {
    static const jit_rvv_eltwise_bwd_kernel_f16_t kernel(alg);
    kernel(p);
}

} // namespace

jit_rvv_eltwise_bwd_kernel_f16_t::jit_rvv_eltwise_bwd_kernel_f16_t(
        alg_kind_t alg)
    : jit_generator_t("jit_rvv_eltwise_bwd_kernel_f16"), alg_(alg) {
    create_kernel();
}

bool jit_rvv_eltwise_bwd_f16_supported(alg_kind_t alg) {
    switch (alg) {
        case alg_kind::eltwise_abs:
        case alg_kind::eltwise_clip:
        case alg_kind::eltwise_hardsigmoid:
        case alg_kind::eltwise_hardswish:
        case alg_kind::eltwise_linear:
        case alg_kind::eltwise_relu:
        case alg_kind::eltwise_sqrt:
        case alg_kind::eltwise_square: return true;
        default: return false;
    }
}

void jit_rvv_eltwise_apply_bwd_f16(alg_kind_t alg, void *diff_src,
        const void *diff_dst, const void *src, dim_t len, float alpha,
        float beta) {
    const jit_rvv_eltwise_bwd_kernel_f16_t::call_params_t p {
            diff_src, diff_dst, src, len, alpha, beta};
    switch (alg) {
        case alg_kind::eltwise_abs:
            dispatch_jit_eltwise_bwd_f16<alg_kind::eltwise_abs>(&p);
            break;
        case alg_kind::eltwise_clip:
            dispatch_jit_eltwise_bwd_f16<alg_kind::eltwise_clip>(&p);
            break;
        case alg_kind::eltwise_hardsigmoid:
            dispatch_jit_eltwise_bwd_f16<alg_kind::eltwise_hardsigmoid>(&p);
            break;
        case alg_kind::eltwise_hardswish:
            dispatch_jit_eltwise_bwd_f16<alg_kind::eltwise_hardswish>(&p);
            break;
        case alg_kind::eltwise_linear:
            dispatch_jit_eltwise_bwd_f16<alg_kind::eltwise_linear>(&p);
            break;
        case alg_kind::eltwise_relu:
            dispatch_jit_eltwise_bwd_f16<alg_kind::eltwise_relu>(&p);
            break;
        case alg_kind::eltwise_sqrt:
            dispatch_jit_eltwise_bwd_f16<alg_kind::eltwise_sqrt>(&p);
            break;
        case alg_kind::eltwise_square:
            dispatch_jit_eltwise_bwd_f16<alg_kind::eltwise_square>(&p);
            break;
        default: assert(!"unsupported f16 eltwise bwd JIT alg");
    }
}

void jit_rvv_eltwise_bwd_kernel_f16_t::compute_vector(const VReg &v_dst,
        const VReg &v_dd, const VReg &v_s, const VReg &v_tmp,
        const FReg &f_alpha, const FReg &f_beta, const FReg &f_zero,
        const FReg &f_one) {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    // Compute runs at SEW=e32 LMUL=m4 (the wide f32 side of widen-narrow).
    const VReg v_mask(0);
    const VReg v_mask_tmp(1); // secondary mask reg used by clip_bwd
    switch (alg_) {
        case alg_kind::eltwise_relu:
            // s > 0 ? dd : dd * alpha
            vmfgt_vf(v_mask, v_s, f_zero);
            vfmul_vf(v_tmp, v_dd, f_alpha);
            vmerge_vvm(v_dst, v_tmp, v_dd);
            break;
        case alg_kind::eltwise_square:
            // dd * 2 * s. Compute 2*s via add to keep a single rounding.
            vfadd_vv(v_tmp, v_s, v_s);
            vfmul_vv(v_dst, v_dd, v_tmp);
            break;
        case alg_kind::eltwise_abs:
            // s > 0 ? dd : s < 0 ? -dd : 0
            vfneg_v(v_tmp, v_dd);
            vmflt_vf(v_mask, v_s, f_zero);
            vfmv_v_f(v_dst, f_zero);
            vmerge_vvm(v_dst, v_dst, v_tmp); // (s<0) ? -dd : 0
            vmfgt_vf(v_mask, v_s, f_zero);
            vmerge_vvm(v_dst, v_dst, v_dd); // (s>0) ? dd : prev
            break;
        case alg_kind::eltwise_sqrt:
            // dd / (2 * sqrt(s))
            vfsqrt_v(v_tmp, v_s);
            vfadd_vv(v_tmp, v_tmp, v_tmp);
            vfdiv_vv(v_dst, v_dd, v_tmp);
            break;
        case alg_kind::eltwise_linear:
            // dd * alpha
            vfmul_vf(v_dst, v_dd, f_alpha);
            break;
        case alg_kind::eltwise_clip:
            // (alpha < s && s <= beta) ? dd : 0
            vmfgt_vf(v_mask, v_s, f_alpha);
            vmfle_vf(v_mask_tmp, v_s, f_beta);
            vmand_mm(v_mask, v_mask, v_mask_tmp);
            vfmv_v_f(v_dst, f_zero);
            vmerge_vvm(v_dst, v_dst, v_dd);
            break;
        case alg_kind::eltwise_hardsigmoid:
            // v = alpha*s + beta; v >= 1 || v <= 0 ? 0 : dd * alpha
            vfmul_vf(v_tmp, v_s, f_alpha);
            vfadd_vf(v_tmp, v_tmp, f_beta); // tmp = v
            vfmul_vf(v_dst, v_dd, f_alpha); // dst = dd * alpha
            vmfge_vf(v_mask, v_tmp, f_one);
            vfmerge_vfm(v_dst, v_dst, f_zero); // dst[v>=1] = 0
            vmfle_vf(v_mask, v_tmp, f_zero);
            vfmerge_vfm(v_dst, v_dst, f_zero); // dst[v<=0] = 0
            break;
        case alg_kind::eltwise_hardswish:
            // v = alpha*s + beta; w = 2*alpha*s + beta;
            // v >= 1 ? dd : v <= 0 ? 0 : dd * w
            vfmul_vf(v_tmp, v_s, f_alpha); // tmp = alpha*s
            vfadd_vv(v_dst, v_tmp, v_tmp); // dst = 2*alpha*s
            vfadd_vf(v_dst, v_dst, f_beta); // dst = w
            vfadd_vf(v_tmp, v_tmp, f_beta); // tmp = v
            vfmul_vv(v_dst, v_dd, v_dst); // dst = dd * w
            vmfge_vf(v_mask, v_tmp, f_one);
            vmerge_vvm(v_dst, v_dst, v_dd); // dst[v>=1] = dd
            vmfle_vf(v_mask, v_tmp, f_zero);
            vfmerge_vfm(v_dst, v_dst, f_zero); // dst[v<=0] = 0
            break;
        default: assert(!"unsupported f16 eltwise bwd JIT alg");
    }
#else
    UNUSED(v_dst, v_dd, v_s, v_tmp, f_alpha, f_beta, f_zero, f_one);
#endif
}

void jit_rvv_eltwise_bwd_kernel_f16_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    // call_params_t layout:
    //  0: diff_src, 8: diff_dst, 16: src, 24: len, 32: alpha, 36: beta
    const Reg reg_param = a0;
    const Reg reg_diff_src = a1;
    const Reg reg_diff_dst = a2;
    const Reg reg_src = a3;
    const Reg reg_len = a4;
    const Reg reg_vl = t0;
    const Reg reg_bytes = t1;
    const Reg reg_tmp = t2;

    const FReg f_alpha = fa0;
    const FReg f_beta = fa1;
    const FReg f_zero = fa2;
    const FReg f_one = fa3;

    // Reg layout (widen-narrow at SEW=e32 LMUL=m4 compute):
    //   v_dd_f16  (m2, regs 2-3)   ← diff_dst f16 load
    //   v_dd      (m4, regs 4-7)   ← widened diff_dst
    //   v_s       (m4, regs 8-11)  ← widened src
    //   v_tmp     (m4, regs 12-15) ← scratch
    //   v_dst     (m4, regs 16-19) ← compute output
    //   v_s_f16   (m2, regs 20-21) ← src f16 load
    //   v_dst_f16 (m2, regs 22-23) ← narrow store buffer
    const VReg v_dd_f16(2);
    const VReg v_dd(4);
    const VReg v_s(8);
    const VReg v_tmp(12);
    const VReg v_dst(16);
    const VReg v_s_f16(20);
    const VReg v_dst_f16(22);

    ld(reg_diff_src, reg_param, 0);
    ld(reg_diff_dst, reg_param, 8);
    ld(reg_src, reg_param, 16);
    ld(reg_len, reg_param, 24);
    flw(f_alpha, reg_param, 32);
    flw(f_beta, reg_param, 36);
    fmv_w_x(f_zero, x0);
    li(reg_tmp, 0x3f800000);
    fmv_w_x(f_one, reg_tmp);

    Label loop, done;
    L(loop);
    beqz(reg_len, done);
    // Three vsetvli phases: e16/m2 for the two widens, e32/m4 for the f32
    // compute, e16/m2 again for the narrow + store. SEW conventions for
    // vfwcvt/vfncvt are documented in the fwd kernel.
    vsetvli(reg_vl, reg_len, SEW::e16, LMUL::m2);
    vle16_v(v_dd_f16, reg_diff_dst);
    vfwcvt_f_f_v(v_dd, v_dd_f16);
    vle16_v(v_s_f16, reg_src);
    vfwcvt_f_f_v(v_s, v_s_f16);
    vsetvli(reg_vl, reg_vl, SEW::e32, LMUL::m4);
    compute_vector(v_dst, v_dd, v_s, v_tmp, f_alpha, f_beta, f_zero, f_one);
    vsetvli(reg_vl, reg_vl, SEW::e16, LMUL::m2);
    vfncvt_f_f_w(v_dst_f16, v_dst);
    vse16_v(v_dst_f16, reg_diff_src);

    // 2 bytes per f16 element.
    slli(reg_bytes, reg_vl, 1);
    add(reg_diff_src, reg_diff_src, reg_bytes);
    add(reg_diff_dst, reg_diff_dst, reg_bytes);
    add(reg_src, reg_src, reg_bytes);
    sub(reg_len, reg_len, reg_vl);
    j_(loop);

    L(done);
    ret();
#else
    ret();
#endif
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
