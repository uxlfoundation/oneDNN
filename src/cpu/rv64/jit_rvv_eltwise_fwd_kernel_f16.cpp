#include <assert.h>

#include "cpu/rv64/jit_rvv_eltwise_fwd_kernel_f16.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

namespace {

template <alg_kind_t alg>
void dispatch_jit_eltwise_fwd_f16(
        const jit_rvv_eltwise_fwd_kernel_f16_t::call_params_t *p) {
    static const jit_rvv_eltwise_fwd_kernel_f16_t kernel(alg);
    kernel(p);
}

} // namespace

jit_rvv_eltwise_fwd_kernel_f16_t::jit_rvv_eltwise_fwd_kernel_f16_t(
        alg_kind_t alg)
    : jit_generator_t("jit_rvv_eltwise_fwd_kernel_f16"), alg_(alg) {
    create_kernel();
}

bool jit_rvv_eltwise_fwd_f16_supported(alg_kind_t alg) {
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

void jit_rvv_eltwise_apply_fwd_f16(alg_kind_t alg, const void *src, void *dst,
        dim_t len, float alpha, float beta) {
    const jit_rvv_eltwise_fwd_kernel_f16_t::call_params_t p {
            src, dst, len, alpha, beta};
    switch (alg) {
        case alg_kind::eltwise_abs:
            dispatch_jit_eltwise_fwd_f16<alg_kind::eltwise_abs>(&p);
            break;
        case alg_kind::eltwise_clip:
            dispatch_jit_eltwise_fwd_f16<alg_kind::eltwise_clip>(&p);
            break;
        case alg_kind::eltwise_hardsigmoid:
            dispatch_jit_eltwise_fwd_f16<alg_kind::eltwise_hardsigmoid>(&p);
            break;
        case alg_kind::eltwise_hardswish:
            dispatch_jit_eltwise_fwd_f16<alg_kind::eltwise_hardswish>(&p);
            break;
        case alg_kind::eltwise_linear:
            dispatch_jit_eltwise_fwd_f16<alg_kind::eltwise_linear>(&p);
            break;
        case alg_kind::eltwise_relu:
            dispatch_jit_eltwise_fwd_f16<alg_kind::eltwise_relu>(&p);
            break;
        case alg_kind::eltwise_sqrt:
            dispatch_jit_eltwise_fwd_f16<alg_kind::eltwise_sqrt>(&p);
            break;
        case alg_kind::eltwise_square:
            dispatch_jit_eltwise_fwd_f16<alg_kind::eltwise_square>(&p);
            break;
        default: assert(!"unsupported f16 eltwise fwd JIT alg");
    }
}

void jit_rvv_eltwise_fwd_kernel_f16_t::compute_vector(const VReg &v_dst,
        const VReg &v_src, const VReg &v_tmp, const FReg &f_alpha,
        const FReg &f_beta, const FReg &f_zero, const FReg &f_one) {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    // Compute runs at SEW=e32, LMUL=m4 — same operands and shape as the
    // upstream f32 eltwise kernel, since the widening makes v_src an f32 group.
    const VReg v_mask(0);
    switch (alg_) {
        case alg_kind::eltwise_abs: vfabs_v(v_dst, v_src); break;
        case alg_kind::eltwise_clip:
            vfmax_vf(v_dst, v_src, f_alpha);
            vfmin_vf(v_dst, v_dst, f_beta);
            break;
        case alg_kind::eltwise_hardsigmoid:
            vfmul_vf(v_dst, v_src, f_alpha);
            vfadd_vf(v_dst, v_dst, f_beta);
            vfmax_vf(v_dst, v_dst, f_zero);
            vfmin_vf(v_dst, v_dst, f_one);
            break;
        case alg_kind::eltwise_hardswish:
            vfmul_vf(v_tmp, v_src, f_alpha);
            vfadd_vf(v_tmp, v_tmp, f_beta);
            vfmax_vf(v_tmp, v_tmp, f_zero);
            vfmin_vf(v_tmp, v_tmp, f_one);
            vfmul_vv(v_dst, v_src, v_tmp);
            break;
        case alg_kind::eltwise_linear:
            vfmul_vf(v_dst, v_src, f_alpha);
            vfadd_vf(v_dst, v_dst, f_beta);
            break;
        case alg_kind::eltwise_relu:
            vmfgt_vf(v_mask, v_src, f_zero);
            vfmul_vf(v_tmp, v_src, f_alpha);
            vmerge_vvm(v_dst, v_tmp, v_src);
            break;
        case alg_kind::eltwise_sqrt: vfsqrt_v(v_dst, v_src); break;
        case alg_kind::eltwise_square: vfmul_vv(v_dst, v_src, v_src); break;
        default: assert(!"unsupported f16 eltwise fwd JIT alg");
    }
#else
    UNUSED(v_dst, v_src, v_tmp, f_alpha, f_beta, f_zero, f_one);
#endif
}

void jit_rvv_eltwise_fwd_kernel_f16_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const Reg reg_src = a1;
    const Reg reg_dst = a2;
    const Reg reg_len = a3;
    const Reg reg_vl = t0;
    const Reg reg_bytes = t1;
    const Reg reg_tmp = t2;

    const FReg f_alpha = fa0;
    const FReg f_beta = fa1;
    const FReg f_zero = fa2;
    const FReg f_one = fa3;

    // Reg layout (widen-narrow at SEW=e32 LMUL=m4 compute):
    //   v_in_f16  (m2, regs 2-3)   ← f16 load
    //   v_src     (m4, regs 4-7)   ← widened f32 input
    //   v_tmp     (m4, regs 8-11)  ← scratch
    //   v_dst     (m4, regs 12-15) ← compute output
    //   v_out_f16 (m2, regs 16-17) ← narrow store buffer
    const VReg v_in_f16(2);
    const VReg v_src(4);
    const VReg v_tmp(8);
    const VReg v_dst(12);
    const VReg v_out_f16(16);

    ld(reg_src, reg_param, 0);
    ld(reg_dst, reg_param, 8);
    ld(reg_len, reg_param, 16);
    flw(f_alpha, reg_param, 24);
    flw(f_beta, reg_param, 28);
    fmv_w_x(f_zero, x0);
    li(reg_tmp, 0x3f800000);
    fmv_w_x(f_one, reg_tmp);

    Label loop, done;
    L(loop);
    beqz(reg_len, done);
    // Three vsetvli phases. vfwcvt reads SEW as the source narrow width;
    // vfncvt reads SEW as the *destination* narrow width — opposite
    // convention. The compute runs at the wide f32 SEW. VLMAX matches at
    // e16/m2 and e32/m4 for VLEN >= 64, so reg_vl is preserved.
    vsetvli(reg_vl, reg_len, SEW::e16, LMUL::m2);
    vle16_v(v_in_f16, reg_src);
    vfwcvt_f_f_v(v_src, v_in_f16);
    vsetvli(reg_vl, reg_vl, SEW::e32, LMUL::m4);
    compute_vector(v_dst, v_src, v_tmp, f_alpha, f_beta, f_zero, f_one);
    vsetvli(reg_vl, reg_vl, SEW::e16, LMUL::m2);
    vfncvt_f_f_w(v_out_f16, v_dst);
    vse16_v(v_out_f16, reg_dst);

    // 2 bytes per f16 element.
    slli(reg_bytes, reg_vl, 1);
    add(reg_src, reg_src, reg_bytes);
    add(reg_dst, reg_dst, reg_bytes);
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
