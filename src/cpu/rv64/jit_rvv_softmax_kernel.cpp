/*******************************************************************************
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
* Copyright 2026 SpacemiT Corporation
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

#include <cstddef>

#include "cpu/rv64/jit_rvv_softmax_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

#define F16_AFFINE_OFF(field) \
    static_cast<int32_t>(offsetof( \
            jit_rvv_softmax_f16_affine_kernel_t::call_params_t, field))
#define F16_STRIDED_OFF(field) \
    static_cast<int32_t>(offsetof( \
            jit_rvv_softmax_f16_strided_kernel_t::call_params_t, field))
#define F16_EXP_SUB_SUM_OFF(field) \
    static_cast<int32_t>(offsetof( \
            jit_rvv_softmax_f16_exp_sub_sum_kernel_t::call_params_t, field))

namespace {

template <bool src_f32>
void dispatch_f16_affine(
        const jit_rvv_softmax_f16_affine_kernel_t::call_params_t *p) {
    static const jit_rvv_softmax_f16_affine_kernel_t kernel(src_f32);
    kernel(p);
}

template <bool gather>
void dispatch_f16_strided(
        const jit_rvv_softmax_f16_strided_kernel_t::call_params_t *p) {
    static const jit_rvv_softmax_f16_strided_kernel_t kernel(gather);
    kernel(p);
}

void dispatch_f16_exp_sub_sum(
        const jit_rvv_softmax_f16_exp_sub_sum_kernel_t::call_params_t *p) {
    static const jit_rvv_softmax_f16_exp_sub_sum_kernel_t kernel;
    kernel(p);
}

} // namespace

jit_rvv_softmax_affine_kernel_t::jit_rvv_softmax_affine_kernel_t()
    : jit_generator_t("jit_rvv_softmax_affine_kernel") {
    create_kernel();
}

void jit_rvv_softmax_affine_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const Reg reg_src = a1;
    const Reg reg_dst = a2;
    const Reg reg_len = a3;
    const Reg reg_vl = t0;
    const Reg reg_bytes = t1;
    const Reg reg_tmp = t2;

    const FReg f_sub = fa0;
    const FReg f_mul = fa1;

    const VReg v_src(0);

    // call_params_t layout:
    //  0: src, 8: dst, 16: len, 24: sub, 28: mul
    ld(reg_src, reg_param, 0);
    ld(reg_dst, reg_param, 8);
    ld(reg_len, reg_param, 16);

    lw(reg_tmp, reg_param, 24);
    fmv_w_x(f_sub, reg_tmp);
    lw(reg_tmp, reg_param, 28);
    fmv_w_x(f_mul, reg_tmp);

    Label loop, done;
    L(loop);
    beqz(reg_len, done);
    vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
    vle32_v(v_src, reg_src);
    vfsub_vf(v_src, v_src, f_sub);
    vfmul_vf(v_src, v_src, f_mul);
    vse32_v(v_src, reg_dst);
    slli(reg_bytes, reg_vl, 2);
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

jit_rvv_softmax_f16_affine_kernel_t::jit_rvv_softmax_f16_affine_kernel_t(
        bool src_f32)
    : jit_generator_t("jit_rvv_softmax_f16_affine_kernel"), src_f32_(src_f32) {
    create_kernel();
}

jit_rvv_softmax_f16_strided_kernel_t::jit_rvv_softmax_f16_strided_kernel_t(
        bool gather)
    : jit_generator_t("jit_rvv_softmax_f16_strided_kernel"), gather_(gather) {
    create_kernel();
}

void jit_rvv_softmax_f16_affine_from_f16(const dnnl::impl::float16_t *src,
        dnnl::impl::float16_t *dst, dim_t len, float sub, float mul) {
    const jit_rvv_softmax_f16_affine_kernel_t::call_params_t p {
            src, dst, len, sub, mul};
    dispatch_f16_affine<false>(&p);
}

void jit_rvv_softmax_f16_affine_from_f32(const float *src,
        dnnl::impl::float16_t *dst, dim_t len, float sub, float mul) {
    const jit_rvv_softmax_f16_affine_kernel_t::call_params_t p {
            src, dst, len, sub, mul};
    dispatch_f16_affine<true>(&p);
}

void jit_rvv_softmax_f16_gather(const dnnl::impl::float16_t *src,
        dnnl::impl::float16_t *dst, dim_t len, dim_t stride_bytes) {
    const jit_rvv_softmax_f16_strided_kernel_t::call_params_t p {
            src, dst, len, stride_bytes};
    dispatch_f16_strided<true>(&p);
}

void jit_rvv_softmax_f16_scatter(const dnnl::impl::float16_t *src,
        dnnl::impl::float16_t *dst, dim_t len, dim_t stride_bytes) {
    const jit_rvv_softmax_f16_strided_kernel_t::call_params_t p {
            src, dst, len, stride_bytes};
    dispatch_f16_strided<false>(&p);
}

void jit_rvv_softmax_f16_exp_sub_sum(const dnnl::impl::float16_t *src,
        float *tmp, dim_t len, float sub, float *sum) {
    const jit_rvv_softmax_f16_exp_sub_sum_kernel_t::call_params_t p {
            src, tmp, len, sub, sum};
    dispatch_f16_exp_sub_sum(&p);
}

void jit_rvv_softmax_f16_affine_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const Reg reg_src = a1;
    const Reg reg_dst = a2;
    const Reg reg_len = a3;
    const Reg reg_vl = t0;
    const Reg reg_src_bytes = t1;
    const Reg reg_dst_bytes = t2;

    const FReg f_sub = fa0;
    const FReg f_mul = fa1;

    const VReg v_src(4);
    const VReg v_f32(8);
    const VReg v_dst(4);

    ld(reg_src, reg_param, F16_AFFINE_OFF(src));
    ld(reg_dst, reg_param, F16_AFFINE_OFF(dst));
    ld(reg_len, reg_param, F16_AFFINE_OFF(len));
    flw(f_sub, reg_param, F16_AFFINE_OFF(sub));
    flw(f_mul, reg_param, F16_AFFINE_OFF(mul));

    Label loop, done;
    L(loop);
    beqz(reg_len, done);

    if (src_f32_) {
        vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
        vle32_v(v_f32, reg_src);
    } else {
        vsetvli(reg_vl, reg_len, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
        vle16_v(v_src, reg_src);
        vfwcvt_f_f_v(v_f32, v_src);
        vsetvli(reg_vl, reg_vl, SEW::e32, LMUL::m2, VTA::ta, VMA::ma);
    }

    vfsub_vf(v_f32, v_f32, f_sub);
    vfmul_vf(v_f32, v_f32, f_mul);
    vsetvli(reg_vl, reg_vl, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
    vfncvt_f_f_w(v_dst, v_f32);
    vse16_v(v_dst, reg_dst);

    slli(reg_dst_bytes, reg_vl, 1);
    add(reg_dst, reg_dst, reg_dst_bytes);
    if (src_f32_) {
        slli(reg_src_bytes, reg_vl, 2);
    } else {
        slli(reg_src_bytes, reg_vl, 1);
    }
    add(reg_src, reg_src, reg_src_bytes);
    sub(reg_len, reg_len, reg_vl);
    j_(loop);

    L(done);
    ret();
#else
    ret();
#endif
}

void jit_rvv_softmax_f16_strided_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const Reg reg_src = a1;
    const Reg reg_dst = a2;
    const Reg reg_len = a3;
    const Reg reg_stride = a4;
    const Reg reg_vl = t0;
    const Reg reg_bytes = t1;

    const VReg v_data(4);

    ld(reg_src, reg_param, F16_STRIDED_OFF(src));
    ld(reg_dst, reg_param, F16_STRIDED_OFF(dst));
    ld(reg_len, reg_param, F16_STRIDED_OFF(len));
    ld(reg_stride, reg_param, F16_STRIDED_OFF(stride_bytes));

    Label loop, done;
    L(loop);
    beqz(reg_len, done);

    vsetvli(reg_vl, reg_len, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
    if (gather_) {
        vlse16_v(v_data, reg_src, reg_stride);
        vse16_v(v_data, reg_dst);
        mul(reg_bytes, reg_vl, reg_stride);
        add(reg_src, reg_src, reg_bytes);
        slli(reg_bytes, reg_vl, 1);
        add(reg_dst, reg_dst, reg_bytes);
    } else {
        vle16_v(v_data, reg_src);
        vsse16_v(v_data, reg_dst, reg_stride);
        slli(reg_bytes, reg_vl, 1);
        add(reg_src, reg_src, reg_bytes);
        mul(reg_bytes, reg_vl, reg_stride);
        add(reg_dst, reg_dst, reg_bytes);
    }

    sub(reg_len, reg_len, reg_vl);
    j_(loop);

    L(done);
    ret();
#else
    ret();
#endif
}

jit_rvv_softmax_f16_exp_sub_sum_kernel_t::
        jit_rvv_softmax_f16_exp_sub_sum_kernel_t()
    : jit_generator_t("jit_rvv_softmax_f16_exp_sub_sum_kernel") {
    create_kernel();
}

void jit_rvv_softmax_f16_exp_sub_sum_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const Reg reg_src = a1;
    const Reg reg_tmp = a2;
    const Reg reg_len = a3;
    const Reg reg_sum = a4;
    const Reg reg_sub_tmp = t2;
    const Reg reg_vl = t0;
    const Reg reg_bytes = t1;
    const Reg reg_maxexp = t4;
    const Reg reg_minexp = t5;
    const Reg reg_imm = t6;

    const FReg f_zero = ft0;
    const FReg f_lower = ft1;
    const FReg f_upper = ft2;
    const FReg f_round = ft3;
    const FReg f_log2_recip = ft4;
    const FReg f_log2_high = ft5;
    const FReg f_log2_low = ft6;
    const FReg f_sub = fa0;
    const FReg f_poly = ft7;
    const FReg f_poly_coeff = ft8;
    const FReg f_sum = ft10;

    const VReg v_in16(0);
    const VReg v_x(8);
    const VReg v_bias(12);
    const VReg v_poly(16);
    const VReg v_tmpv(20);
    const VReg v_acc(24);
    const VReg v_red(28);
    const VReg v_mask(0);

    auto load_f32_bits = [&](const FReg &freg, uint32_t bits) {
        li(reg_imm, static_cast<int64_t>(bits));
        fmv_w_x(freg, reg_imm);
    };
    auto load_f32 = [&](const FReg &freg, float value) {
        load_f32_bits(freg, utils::bit_cast<uint32_t>(value));
    };

    ld(reg_src, reg_param, F16_EXP_SUB_SUM_OFF(src));
    ld(reg_tmp, reg_param, F16_EXP_SUB_SUM_OFF(tmp));
    ld(reg_len, reg_param, F16_EXP_SUB_SUM_OFF(len));
    ld(reg_sum, reg_param, F16_EXP_SUB_SUM_OFF(sum));
    lw(reg_sub_tmp, reg_param, F16_EXP_SUB_SUM_OFF(sub));
    fmv_w_x(f_sub, reg_sub_tmp);

    fmv_w_x(f_zero, x0);
    load_f32(f_lower, -103.9720840454f);
    load_f32(f_upper, 88.7762626647950f);
    load_f32_bits(f_round, 0x4b400000u);
    load_f32_bits(f_log2_recip, 0x3fb8aa3bu);
    load_f32_bits(f_log2_high, 0xbf317200u);
    load_f32_bits(f_log2_low, 0xb5bfbe8eu);
    load_f32_bits(f_poly, 0x3ab4a000u);
    li(reg_minexp, static_cast<int64_t>(0xC1000000u));
    li(reg_maxexp, static_cast<int64_t>(0x3F800000u));

    vsetvli(reg_vl, x0, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
    vfmv_v_f(v_acc, f_zero);

    Label loop, finish;
    L(loop);
    beqz(reg_len, finish);

    vsetvli(reg_vl, reg_len, SEW::e16, LMUL::m2, VTA::ta, VMA::ma);
    vle16_v(v_in16, reg_src);
    slli(reg_bytes, reg_vl, 1);
    add(reg_src, reg_src, reg_bytes);
    vfwcvt_f_f_v(v_x, v_in16);

    vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m4, VTA::tu, VMA::ma);
    vfsub_vf(v_x, v_x, f_sub);
    vfmv_v_f(v_bias, f_round);
    vmflt_vf(v_mask, v_x, f_lower);
    vfmerge_vfm(v_x, v_x, f_lower);
    vmfgt_vf(v_mask, v_x, f_upper);
    vfmerge_vfm(v_x, v_x, f_upper);
    vfmacc_vf(v_bias, f_log2_recip, v_x);
    vfsub_vf(v_tmpv, v_bias, f_round);
    vfmacc_vf(v_x, f_log2_high, v_tmpv);
    vfmacc_vf(v_x, f_log2_low, v_tmpv);
    vfmv_v_f(v_poly, f_poly);
    load_f32_bits(f_poly_coeff, 0x3c092f6eu);
    vfmul_vv(v_poly, v_poly, v_x);
    vfadd_vf(v_poly, v_poly, f_poly_coeff);
    load_f32_bits(f_poly_coeff, 0x3d2aadadu);
    vfmul_vv(v_poly, v_poly, v_x);
    vfadd_vf(v_poly, v_poly, f_poly_coeff);
    load_f32_bits(f_poly_coeff, 0x3e2aaa28u);
    vfmul_vv(v_poly, v_poly, v_x);
    vfadd_vf(v_poly, v_poly, f_poly_coeff);
    load_f32_bits(f_poly_coeff, 0x3efffffbu);
    vfmul_vv(v_poly, v_poly, v_x);
    vfadd_vf(v_poly, v_poly, f_poly_coeff);
    load_f32_bits(f_poly_coeff, 0x3f800000u);
    vfmul_vv(v_poly, v_poly, v_x);
    vfadd_vf(v_poly, v_poly, f_poly_coeff);
    vsll_vi(v_bias, v_bias, 23);
    vmin_vx(v_tmpv, v_bias, reg_maxexp);
    vmax_vx(v_tmpv, v_tmpv, reg_minexp);
    vsub_vv(v_bias, v_bias, v_tmpv);
    vadd_vx(v_bias, v_bias, reg_maxexp);
    vadd_vx(v_tmpv, v_tmpv, reg_maxexp);
    vfmul_vv(v_x, v_x, v_bias);
    vfmadd_vv(v_poly, v_x, v_bias);
    vfmul_vv(v_poly, v_poly, v_tmpv);
    vfadd_vv(v_acc, v_acc, v_poly);
    vse32_v(v_poly, reg_tmp);
    slli(reg_bytes, reg_vl, 2);
    add(reg_tmp, reg_tmp, reg_bytes);
    sub(reg_len, reg_len, reg_vl);
    j_(loop);

    L(finish);
    vsetvli(reg_vl, x0, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
    vfmv_v_f(v_red, f_zero);
    vfredosum_vs(v_red, v_acc, v_red);
    vfmv_f_s(f_sum, v_red);
    fsw(f_sum, reg_sum, 0);
    ret();
#else
    ret();
#endif
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
