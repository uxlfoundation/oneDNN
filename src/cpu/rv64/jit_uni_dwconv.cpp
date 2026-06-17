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

#include "cpu/rv64/jit_uni_dwconv.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "common/dnnl_thread.hpp"
#include "common/float16.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "cpu/rv64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace dnnl::impl::data_type;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;
using namespace Xbyak_riscv;

struct jit_uni_dwconv_kernel_t : public jit_generator_t {
    struct call_params_t {
        const float16_t *lhs;
        dim_t lhs_stride_0;
        dim_t lhs_stride_1;
        const float16_t *rhs;
        dim_t rhs_stride_0;
        dim_t rhs_stride_1;
        float16_t *out;
        dim_t out_stride_0;
        dim_t out_stride_1;
        dim_t h;
        dim_t w;
        dim_t c;
        dim_t ratio_bytes;
        const float *bias;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_dwconv_kernel_t)

    jit_uni_dwconv_kernel_t(int stride);

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    void preload_dwconv3x3s1_f16(const Xbyak_riscv::Reg &r0,
            const Xbyak_riscv::Reg &r1, const Xbyak_riscv::Reg &r2,
            const Xbyak_riscv::Reg &lhs_stride_1);
    void compute_dwconv3x3s1_f16_m5(const Xbyak_riscv::Reg &r0,
            const Xbyak_riscv::Reg &r1, const Xbyak_riscv::Reg &r2,
            const Xbyak_riscv::Reg &lhs_stride_1);
    void compute_dwconv3x3s2_f16_m5(const Xbyak_riscv::Reg &r0,
            const Xbyak_riscv::Reg &r1, const Xbyak_riscv::Reg &r2,
            const Xbyak_riscv::Reg &lhs_stride_1);
    void compute_dwconv3x3s2_f16_m(const Xbyak_riscv::Reg &r0,
            const Xbyak_riscv::Reg &r1, const Xbyak_riscv::Reg &r2,
            const Xbyak_riscv::Reg &lhs_stride_1, int count);
    void add_bias_m(const Xbyak_riscv::Reg &vl, int count);
    void narrow_m(int count);
    void store_m(const Xbyak_riscv::Reg &out,
            const Xbyak_riscv::Reg &out_stride_1,
            const Xbyak_riscv::Reg &ratio_bytes, int count);
    void compute_one_output(int dst_idx, int src_start);
    void load_tail_extra_cols(const Xbyak_riscv::Reg &r0,
            const Xbyak_riscv::Reg &r1, const Xbyak_riscv::Reg &r2,
            const Xbyak_riscv::Reg &lhs_stride_1, int cols);
    void compute_tail(const Xbyak_riscv::Reg &r0, const Xbyak_riscv::Reg &r1,
            const Xbyak_riscv::Reg &r2, const Xbyak_riscv::Reg &lhs_stride_1,
            const Xbyak_riscv::Reg &vl, const Xbyak_riscv::Reg &out,
            const Xbyak_riscv::Reg &out_stride_1,
            const Xbyak_riscv::Reg &ratio_bytes, int count);
    void compute_tail_s2(const Xbyak_riscv::Reg &r0, const Xbyak_riscv::Reg &r1,
            const Xbyak_riscv::Reg &r2, const Xbyak_riscv::Reg &lhs_stride_1,
            const Xbyak_riscv::Reg &vl, const Xbyak_riscv::Reg &out,
            const Xbyak_riscv::Reg &out_stride_1,
            const Xbyak_riscv::Reg &ratio_bytes, int count);

    const int stride_;
};

#define DWCONV_OFF(field) \
    static_cast<int32_t>( \
            offsetof(jit_uni_dwconv_kernel_t::call_params_t, field))

namespace {

VReg wei_v(int idx) {
    return VReg(idx);
}

VReg src_v(int idx) {
    return VReg(9 + idx);
}

VReg acc_v(int idx) {
    return VReg(26 + idx);
}

} // namespace

void jit_uni_dwconv_kernel_t::preload_dwconv3x3s1_f16(
        const Reg &r0, const Reg &r1, const Reg &r2, const Reg &lhs_stride_1) {
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

void jit_uni_dwconv_kernel_t::compute_dwconv3x3s1_f16_m5(
        const Reg &r0, const Reg &r1, const Reg &r2, const Reg &lhs_stride_1) {
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

void jit_uni_dwconv_kernel_t::compute_dwconv3x3s2_f16_m5(
        const Reg &r0, const Reg &r1, const Reg &r2, const Reg &lhs_stride_1) {
    compute_dwconv3x3s2_f16_m(r0, r1, r2, lhs_stride_1, 5);
}

void jit_uni_dwconv_kernel_t::compute_dwconv3x3s2_f16_m(const Reg &r0,
        const Reg &r1, const Reg &r2, const Reg &lhs_stride_1, int count) {
    for (int i = 1; i <= count; ++i) {
        vle16_v(src_v(0), r0);
        add(r0, r0, lhs_stride_1);
        vle16_v(src_v(1), r1);
        add(r1, r1, lhs_stride_1);
        vle16_v(src_v(2), r2);
        add(r2, r2, lhs_stride_1);

        vle16_v(src_v(3), r0);
        add(r0, r0, lhs_stride_1);
        vle16_v(src_v(4), r1);
        add(r1, r1, lhs_stride_1);
        vle16_v(src_v(5), r2);
        add(r2, r2, lhs_stride_1);

        vle16_v(src_v(6), r0);
        vle16_v(src_v(7), r1);
        vle16_v(src_v(8), r2);

        vfwmul_vv(acc_v(i), wei_v(0), src_v(0));
        for (int k = 1; k < 9; ++k)
            vfwmacc_vv(acc_v(i), wei_v(k), src_v(k));
    }
}

void jit_uni_dwconv_kernel_t::add_bias_m(const Reg &vl, int count) {
    vsetvli(x0, vl, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
    for (int i = 1; i <= count; ++i)
        vfadd_vv(acc_v(i), acc_v(i), acc_v(0));
    vsetvli(x0, vl, SEW::e16, LMUL::mf2, VTA::ta, VMA::ma);
}

void jit_uni_dwconv_kernel_t::narrow_m(int count) {
    for (int i = 1; i <= count; ++i)
        vfncvt_f_f_w(acc_v(i), acc_v(i));
}

void jit_uni_dwconv_kernel_t::store_m(const Reg &out, const Reg &out_stride_1,
        const Reg &ratio_bytes, int count) {
    for (int i = 1; i <= count; ++i) {
        vsse16_v(acc_v(i), out, ratio_bytes);
        add(out, out, out_stride_1);
    }
}

void jit_uni_dwconv_kernel_t::compute_one_output(int dst_idx, int src_start) {
    auto src_reg = [](int idx) { return src_v(idx < 15 ? idx : idx - 15); };
    vfwmul_vv(acc_v(dst_idx), wei_v(0), src_reg(src_start));
    for (int k = 1; k < 9; ++k)
        vfwmacc_vv(acc_v(dst_idx), wei_v(k), src_reg(src_start + k));
}

void jit_uni_dwconv_kernel_t::load_tail_extra_cols(const Reg &r0, const Reg &r1,
        const Reg &r2, const Reg &lhs_stride_1, int cols) {
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

void jit_uni_dwconv_kernel_t::compute_tail(const Reg &r0, const Reg &r1,
        const Reg &r2, const Reg &lhs_stride_1, const Reg &vl, const Reg &out,
        const Reg &out_stride_1, const Reg &ratio_bytes, int count) {
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

void jit_uni_dwconv_kernel_t::compute_tail_s2(const Reg &r0, const Reg &r1,
        const Reg &r2, const Reg &lhs_stride_1, const Reg &vl, const Reg &out,
        const Reg &out_stride_1, const Reg &ratio_bytes, int count) {
    compute_dwconv3x3s2_f16_m(r0, r1, r2, lhs_stride_1, count);
    add_bias_m(vl, count);
    narrow_m(count);
    store_m(out, out_stride_1, ratio_bytes, count);
}

jit_uni_dwconv_kernel_t::jit_uni_dwconv_kernel_t(int stride)
    : jit_generator_t("jit_uni_dwconv_kernel"), stride_(stride) {
    create_kernel();
}

void jit_uni_dwconv_kernel_t::generate() {
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

    if (stride_ == 1) {
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
    } else {
        blez(t0, tail_w);

        L(loop_w);
        addi(t0, t0, -1);
        compute_dwconv3x3s2_f16_m5(a1, a2, a3, t6);
        add_bias_m(t1, 5);
        narrow_m(5);
        store_m(a0, t4, a5, 5);
        bnez(t0, loop_w);

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
        compute_tail_s2(a1, a2, a3, t6, t1, a0, t4, a5, 1);
        j_(w_end);

        L(tail_w2);
        compute_tail_s2(a1, a2, a3, t6, t1, a0, t4, a5, 2);
        j_(w_end);

        L(tail_w3);
        compute_tail_s2(a1, a2, a3, t6, t1, a0, t4, a5, 3);
        j_(w_end);

        L(tail_w4);
        compute_tail_s2(a1, a2, a3, t6, t1, a0, t4, a5, 4);
        j_(w_end);
    }

    L(w_end);
    ld(t0, a7, DWCONV_OFF(out_stride_0));
    add(a6, a6, t0);
    add(a4, a4, t5);
    if (stride_ == 2) add(a4, a4, t5);
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

namespace {

static dim_t tensor_nhwc_elems(dim_t n, dim_t h, dim_t w, dim_t c) {
    return n * h * w * c;
}

static void pack_input_nhwc(const float16_t *src,
        const memory_desc_wrapper &src_d, dim_t n, dim_t ih, dim_t iw,
        dim_t channels, dim_t padded_h, dim_t padded_w, dim_t t_pad,
        dim_t l_pad, float16_t *packed) {
    const float16_t zero(0.0f);
    std::fill(packed,
            packed + tensor_nhwc_elems(1, padded_h, padded_w, channels), zero);

    for (dim_t h = 0; h < ih; ++h) {
        for (dim_t w = 0; w < iw; ++w) {
            float16_t *dst = packed
                    + ((h + t_pad) * padded_w + (w + l_pad)) * channels;
            const float16_t *src_ptr = src + src_d.off(n, 0, h, w);
            std::copy_n(src_ptr, channels, dst);
        }
    }
}

static void pack_weights_goihw(const float16_t *weights,
        const memory_desc_wrapper &wei_d, dim_t groups, dim_t oc_per_group,
        std::vector<float16_t> &packed) {
    packed.resize(oc_per_group * 3 * 3 * groups);
    for (dim_t oc = 0; oc < oc_per_group; ++oc) {
        float16_t *oc_base = packed.data() + oc * 9 * groups;
        for (dim_t kh = 0; kh < 3; ++kh) {
            for (dim_t kw = 0; kw < 3; ++kw) {
                float16_t *k_base = oc_base + (kh * 3 + kw) * groups;
                for (dim_t g = 0; g < groups; ++g)
                    k_base[g] = weights[wei_d.off(g, oc, 0, kh, kw)];
            }
        }
    }
}

static void prepare_bias(const void *bias, bool bias_is_f32, dim_t channels,
        std::vector<float> &bias_fp32) {
    if (bias == nullptr) {
        bias_fp32.clear();
        return;
    }

    bias_fp32.resize(channels);
    if (bias_is_f32) {
        const auto *bias_data = static_cast<const float *>(bias);
        std::copy_n(bias_data, channels, bias_fp32.begin());
    } else {
        const auto *bias_data = static_cast<const float16_t *>(bias);
        for (dim_t c = 0; c < channels; ++c)
            bias_fp32[c] = static_cast<float>(bias_data[c]);
    }
}

} // namespace

status_t jit_uni_dwconv_fwd_t::execute(const exec_ctx_t &ctx) const {
    const auto *src = CTX_IN_MEM(const float16_t *, DNNL_ARG_SRC);
    const auto *wei = CTX_IN_MEM(const float16_t *, DNNL_ARG_WEIGHTS);
    const auto *bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
    auto *dst = CTX_OUT_MEM(float16_t *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper wei_d(pd()->weights_md(0));
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const convolution_desc_t *cd = pd()->desc();
    const dim_t mb = src_d.dims()[0];
    const dim_t ih = src_d.dims()[2];
    const dim_t iw = src_d.dims()[3];
    const dim_t groups = wei_d.dims()[0];
    const dim_t oc_per_group = wei_d.dims()[1];
    const dim_t oh = dst_d.dims()[2];
    const dim_t ow = dst_d.dims()[3];
    const dim_t t_pad = cd->padding[0][0];
    const dim_t l_pad = cd->padding[0][1];
    const dim_t b_pad = cd->padding[1][0];
    const dim_t r_pad = cd->padding[1][1];
    const dim_t padded_h = ih + t_pad + b_pad;
    const dim_t padded_w = iw + l_pad + r_pad;
    const dim_t stride_h = cd->strides[0];

    std::vector<float16_t> packed_weights;
    pack_weights_goihw(wei, wei_d, groups, oc_per_group, packed_weights);

    std::vector<float> bias_fp32;
    const bool bias_is_f32 = pd()->with_bias()
            && pd()->weights_md(1)->data_type == data_type::f32;
    prepare_bias(bias, bias_is_f32, groups * oc_per_group, bias_fp32);

    parallel_nd(mb, oc_per_group, [&](dim_t n, dim_t oc) {
        std::vector<float16_t> packed_src(
                tensor_nhwc_elems(1, padded_h, padded_w, groups));
        pack_input_nhwc(src, src_d, n, ih, iw, groups, padded_h, padded_w,
                t_pad, l_pad, packed_src.data());

        jit_uni_dwconv_kernel_t::call_params_t args;
        args.lhs = packed_src.data();
        args.lhs_stride_0 = padded_w * groups * (dim_t)sizeof(float16_t);
        args.lhs_stride_1 = groups * (dim_t)sizeof(float16_t);
        args.rhs = packed_weights.data() + oc * 9 * groups;
        args.rhs_stride_0 = 3 * groups * (dim_t)sizeof(float16_t);
        args.rhs_stride_1 = groups * (dim_t)sizeof(float16_t);
        args.out = dst + dst_d.off(n, oc, 0, 0);
        args.out_stride_0
                = dst_d.blocking_desc().strides[2] * (dim_t)sizeof(float16_t);
        args.out_stride_1
                = dst_d.blocking_desc().strides[3] * (dim_t)sizeof(float16_t);
        args.h = oh;
        args.w = ow;
        args.c = groups;
        args.ratio_bytes = oc_per_group * (dim_t)sizeof(float16_t);
        args.bias
                = bias_fp32.empty() ? nullptr : bias_fp32.data() + oc * groups;
        static const jit_uni_dwconv_kernel_t kernel_s1(1);
        static const jit_uni_dwconv_kernel_t kernel_s2(2);
        const auto &kernel = stride_h == 1 ? kernel_s1 : kernel_s2;
        kernel(&args);
    });

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
