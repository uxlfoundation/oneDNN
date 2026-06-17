/*******************************************************************************
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

#include "cpu/rv64/jit_rvv_dwconv_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

#define DWCONV_OFF(field) \
    static_cast<int32_t>(offsetof(jit_rvv_dwconv_kernel_t::call_params_t, field))

namespace {

VReg wei_v(int idx) {
    return VReg(idx);
}

VReg src_v(int idx) {
    return VReg(9 + idx);
}

VReg acc_v(int idx) {
    return VReg(24 + idx);
}

} // namespace

void jit_rvv_dwconv_kernel_t::preload_dwconv3x3s1_f16(const Reg &r0,
        const Reg &r1, const Reg &r2, const Reg &lhs_stride_1) {
    vle16_v(src_v(6), r0);
    add(r0, r0, lhs_stride_1);
    vle16_v(src_v(7), r1);
    add(r1, r1, lhs_stride_1);
    vle16_v(src_v(8), r2);
    add(r2, r2, lhs_stride_1);
    vle16_v(src_v(9), r0);
    add(r0, r0, lhs_stride_1);
    vle16_v(src_v(10), r1);
    add(r1, r1, lhs_stride_1);
    vle16_v(src_v(11), r2);
    add(r2, r2, lhs_stride_1);
    vle16_v(src_v(12), r0);
    add(r0, r0, lhs_stride_1);
    vle16_v(src_v(13), r1);
    add(r1, r1, lhs_stride_1);
    vle16_v(src_v(14), r2);
    add(r2, r2, lhs_stride_1);
}

void jit_rvv_dwconv_kernel_t::compute_dwconv3x3s1_f16_m5(const Reg &r0,
        const Reg &r1, const Reg &r2, const Reg &lhs_stride_1) {
    vfwmul_vv(acc_v(1), wei_v(0), src_v(0));
    vfwmul_vv(acc_v(2), wei_v(0), src_v(3));
    vfwmul_vv(acc_v(3), wei_v(0), src_v(6));
    vfwmul_vv(acc_v(4), wei_v(0), src_v(9));
    vfwmul_vv(acc_v(5), wei_v(0), src_v(12));
    vle16_v(src_v(0), r0);
    add(r0, r0, lhs_stride_1);
    vfwmacc_vv(acc_v(1), wei_v(1), src_v(1));
    vfwmacc_vv(acc_v(2), wei_v(1), src_v(4));
    vfwmacc_vv(acc_v(3), wei_v(1), src_v(7));
    vfwmacc_vv(acc_v(4), wei_v(1), src_v(10));
    vfwmacc_vv(acc_v(5), wei_v(1), src_v(13));
    vle16_v(src_v(1), r1);
    add(r1, r1, lhs_stride_1);
    vfwmacc_vv(acc_v(1), wei_v(2), src_v(2));
    vfwmacc_vv(acc_v(2), wei_v(2), src_v(5));
    vfwmacc_vv(acc_v(3), wei_v(2), src_v(8));
    vfwmacc_vv(acc_v(4), wei_v(2), src_v(11));
    vfwmacc_vv(acc_v(5), wei_v(2), src_v(14));
    vle16_v(src_v(2), r2);
    add(r2, r2, lhs_stride_1);
    vfwmacc_vv(acc_v(1), wei_v(3), src_v(3));
    vfwmacc_vv(acc_v(2), wei_v(3), src_v(6));
    vfwmacc_vv(acc_v(3), wei_v(3), src_v(9));
    vfwmacc_vv(acc_v(4), wei_v(3), src_v(12));
    vfwmacc_vv(acc_v(5), wei_v(3), src_v(0));
    vle16_v(src_v(3), r0);
    add(r0, r0, lhs_stride_1);
    vfwmacc_vv(acc_v(1), wei_v(4), src_v(4));
    vfwmacc_vv(acc_v(2), wei_v(4), src_v(7));
    vfwmacc_vv(acc_v(3), wei_v(4), src_v(10));
    vfwmacc_vv(acc_v(4), wei_v(4), src_v(13));
    vfwmacc_vv(acc_v(5), wei_v(4), src_v(1));
    vle16_v(src_v(4), r1);
    add(r1, r1, lhs_stride_1);
    vfwmacc_vv(acc_v(1), wei_v(5), src_v(5));
    vfwmacc_vv(acc_v(2), wei_v(5), src_v(8));
    vfwmacc_vv(acc_v(3), wei_v(5), src_v(11));
    vfwmacc_vv(acc_v(4), wei_v(5), src_v(14));
    vfwmacc_vv(acc_v(5), wei_v(5), src_v(2));
    vle16_v(src_v(5), r2);
    add(r2, r2, lhs_stride_1);
    vfwmacc_vv(acc_v(1), wei_v(6), src_v(6));
    vfwmacc_vv(acc_v(2), wei_v(6), src_v(9));
    vfwmacc_vv(acc_v(3), wei_v(6), src_v(12));
    vfwmacc_vv(acc_v(4), wei_v(6), src_v(0));
    vfwmacc_vv(acc_v(5), wei_v(6), src_v(3));

    vfwmacc_vv(acc_v(1), wei_v(7), src_v(7));
    vfwmacc_vv(acc_v(2), wei_v(7), src_v(10));
    vfwmacc_vv(acc_v(3), wei_v(7), src_v(13));
    vfwmacc_vv(acc_v(4), wei_v(7), src_v(1));
    vfwmacc_vv(acc_v(5), wei_v(7), src_v(4));

    vfwmacc_vv(acc_v(1), wei_v(8), src_v(8));
    vfwmacc_vv(acc_v(2), wei_v(8), src_v(11));
    vfwmacc_vv(acc_v(3), wei_v(8), src_v(14));
    vfwmacc_vv(acc_v(4), wei_v(8), src_v(2));
    vfwmacc_vv(acc_v(5), wei_v(8), src_v(5));
}

void jit_rvv_dwconv_kernel_t::add_bias_m(const Reg &vl, int count) {
    vsetvli(x0, vl, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
    for (int i = 1; i <= count; ++i)
        vfadd_vv(acc_v(i), acc_v(i), acc_v(0));
    vsetvli(x0, vl, SEW::e16, LMUL::mf2, VTA::ta, VMA::ma);
}

void jit_rvv_dwconv_kernel_t::narrow_m(int count) {
    for (int i = 1; i <= count; ++i)
        vfncvt_f_f_w(acc_v(i), acc_v(i));
}

void jit_rvv_dwconv_kernel_t::store_m(const Reg &out,
        const Reg &out_stride_1, const Reg &ratio_bytes, int count) {
    for (int i = 1; i <= count; ++i) {
        vsse16_v(acc_v(i), out, ratio_bytes);
        add(out, out, out_stride_1);
    }
}

void jit_rvv_dwconv_kernel_t::compute_one_output(int dst_idx, int src_start) {
    auto src_reg = [](int idx) { return src_v(idx < 15 ? idx : idx - 15); };
    vfwmul_vv(acc_v(dst_idx), wei_v(0), src_reg(src_start));
    for (int k = 1; k < 9; ++k)
        vfwmacc_vv(acc_v(dst_idx), wei_v(k), src_reg(src_start + k));
}

void jit_rvv_dwconv_kernel_t::load_tail_extra_cols(const Reg &r0,
        const Reg &r1, const Reg &r2, const Reg &lhs_stride_1, int cols) {
    for (int col = 0; col < cols; ++col) {
        const int base = 6 + col * 3;
        vle16_v(src_v(base), r0);
        add(r0, r0, lhs_stride_1);
        vle16_v(src_v(base + 1), r1);
        add(r1, r1, lhs_stride_1);
        vle16_v(src_v(base + 2), r2);
        add(r2, r2, lhs_stride_1);
    }
}

void jit_rvv_dwconv_kernel_t::compute_tail(const Reg &r0, const Reg &r1,
        const Reg &r2, const Reg &lhs_stride_1, const Reg &vl,
        const Reg &out, const Reg &out_stride_1, const Reg &ratio_bytes,
        int count) {
    if (count == 4) {
        load_tail_extra_cols(r0, r1, r2, lhs_stride_1, 3);
        for (int i = 1; i <= 3; ++i)
            compute_one_output(i, (i - 1) * 3);
        vle16_v(src_v(0), r0);
        vle16_v(src_v(1), r1);
        vle16_v(src_v(2), r2);
        compute_one_output(4, 9);
    } else {
        load_tail_extra_cols(r0, r1, r2, lhs_stride_1, count);
        for (int i = 1; i <= count; ++i)
            compute_one_output(i, (i - 1) * 3);
    }

    add_bias_m(vl, count);
    narrow_m(count);
    store_m(out, out_stride_1, ratio_bytes, count);
}

jit_rvv_dwconv_kernel_t::jit_rvv_dwconv_kernel_t()
    : jit_generator_t("jit_rvv_dwconv_kernel") {
    create_kernel();
}

void jit_rvv_dwconv_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    Label loop_c;
    Label end_c;
    Label bypass_bias;
    Label loop_h;
    Label loop_w;
    Label tail_w;
    Label tail_w5;
    Label tail_w1;
    Label tail_w2;
    Label tail_w3;
    Label tail_w4;
    Label w_end;

    mv(a7, a0);
    li(t3, 0);
    ld(t5, a7, DWCONV_OFF(lhs_stride_0));
    ld(t6, a7, DWCONV_OFF(lhs_stride_1));
    ld(t4, a7, DWCONV_OFF(out_stride_1));

    L(loop_c);
    ld(t0, a7, DWCONV_OFF(c));
    sub(t0, t0, t3);
    blez(t0, end_c);
    vsetvli(t1, t0, SEW::e16, LMUL::mf2, VTA::ta, VMA::ma);

    slli(t0, t3, 1);
    ld(a1, a7, DWCONV_OFF(rhs));
    add(a1, a1, t0);
    ld(a4, a7, DWCONV_OFF(rhs_stride_0));
    ld(a5, a7, DWCONV_OFF(rhs_stride_1));

    add(a2, a1, a4);
    add(a3, a2, a4);

    vle16_v(wei_v(0), a1);
    add(a1, a1, a5);
    vle16_v(wei_v(1), a2);
    add(a2, a2, a5);
    vle16_v(wei_v(2), a3);
    add(a3, a3, a5);

    vle16_v(wei_v(3), a1);
    add(a1, a1, a5);
    vle16_v(wei_v(4), a2);
    add(a2, a2, a5);
    vle16_v(wei_v(5), a3);
    add(a3, a3, a5);

    vsetvli(x0, t1, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
    vxor_vv(acc_v(0), acc_v(0), acc_v(0));
    ld(t0, a7, DWCONV_OFF(bias));
    beqz(t0, bypass_bias);
    slli(t2, t3, 2);
    add(t0, t0, t2);
    vle32_v(acc_v(0), t0);
    L(bypass_bias);
    vsetvli(x0, t1, SEW::e16, LMUL::mf2, VTA::ta, VMA::ma);

    vle16_v(wei_v(6), a1);
    vle16_v(wei_v(7), a2);
    vle16_v(wei_v(8), a3);
    ld(a5, a7, DWCONV_OFF(ratio_bytes));
    ld(a4, a7, DWCONV_OFF(lhs));
    slli(t0, t3, 1);
    add(a4, a4, t0);
    ld(a6, a7, DWCONV_OFF(out));
    mul(t2, t3, a5);
    add(a6, a6, t2);
    ld(t2, a7, DWCONV_OFF(h));

    L(loop_h);
    ld(t0, a7, DWCONV_OFF(w));
    li(a1, 5);
    divw(t0, t0, a1);

    mv(a0, a6);
    mv(a1, a4);
    add(a2, a4, t5);
    add(a3, a2, t5);

    vle16_v(src_v(0), a1);
    add(a1, a1, t6);
    vle16_v(src_v(1), a2);
    add(a2, a2, t6);
    vle16_v(src_v(2), a3);
    add(a3, a3, t6);

    vle16_v(src_v(3), a1);
    add(a1, a1, t6);
    vle16_v(src_v(4), a2);
    add(a2, a2, t6);
    vle16_v(src_v(5), a3);
    add(a3, a3, t6);

    blez(t0, tail_w);
    preload_dwconv3x3s1_f16(a1, a2, a3, t6);
    addi(t0, t0, -1);
    blez(t0, tail_w5);

    L(loop_w);
    addi(t0, t0, -1);
    compute_dwconv3x3s1_f16_m5(a1, a2, a3, t6);
    add_bias_m(t1, 5);
    narrow_m(5);
    preload_dwconv3x3s1_f16(a1, a2, a3, t6);
    store_m(a0, t4, a5, 5);
    bnez(t0, loop_w);

    L(tail_w5);
    compute_dwconv3x3s1_f16_m5(a1, a2, a3, t6);
    add_bias_m(t1, 5);
    narrow_m(5);
    store_m(a0, t4, a5, 5);

    L(tail_w);
    ld(t0, a7, DWCONV_OFF(w));
    li(a5, 5);
    remw(t0, t0, a5);
    ld(a5, a7, DWCONV_OFF(ratio_bytes));
    addi(t0, t0, -1);
    beqz(t0, tail_w1);
    addi(t0, t0, -1);
    beqz(t0, tail_w2);
    addi(t0, t0, -1);
    beqz(t0, tail_w3);
    addi(t0, t0, -1);
    beqz(t0, tail_w4);
    j_(w_end);

    L(tail_w1);
    compute_tail(a1, a2, a3, t6, t1, a0, t4, a5, 1);
    j_(w_end);

    L(tail_w2);
    compute_tail(a1, a2, a3, t6, t1, a0, t4, a5, 2);
    j_(w_end);

    L(tail_w3);
    compute_tail(a1, a2, a3, t6, t1, a0, t4, a5, 3);
    j_(w_end);

    L(tail_w4);
    compute_tail(a1, a2, a3, t6, t1, a0, t4, a5, 4);
    j_(w_end);

    L(w_end);
    ld(t0, a7, DWCONV_OFF(out_stride_0));
    add(a6, a6, t0);
    add(a4, a4, t5);
    addi(t2, t2, -1);
    bnez(t2, loop_h);

    add(t3, t3, t1);
    j_(loop_c);
    L(end_c);
    ret();
#else
    ret();
#endif
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
