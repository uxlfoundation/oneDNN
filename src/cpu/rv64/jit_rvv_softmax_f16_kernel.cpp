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

#include <cstddef>

#include "cpu/rv64/jit_rvv_softmax_f16_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

#define AFFINE_OFF(field) \
    static_cast<int32_t>(offsetof( \
            jit_rvv_softmax_f16_affine_kernel_t::call_params_t, field))
#define STRIDED_OFF(field) \
    static_cast<int32_t>(offsetof( \
            jit_rvv_softmax_f16_strided_kernel_t::call_params_t, field))

namespace {

template <bool src_f32>
void dispatch_affine(
        const jit_rvv_softmax_f16_affine_kernel_t::call_params_t *p) {
    static const jit_rvv_softmax_f16_affine_kernel_t kernel(src_f32);
    kernel(p);
}

template <bool gather>
void dispatch_strided(
        const jit_rvv_softmax_f16_strided_kernel_t::call_params_t *p) {
    static const jit_rvv_softmax_f16_strided_kernel_t kernel(gather);
    kernel(p);
}

} // namespace

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
    dispatch_affine<false>(&p);
}

void jit_rvv_softmax_f16_affine_from_f32(const float *src,
        dnnl::impl::float16_t *dst, dim_t len, float sub, float mul) {
    const jit_rvv_softmax_f16_affine_kernel_t::call_params_t p {
            src, dst, len, sub, mul};
    dispatch_affine<true>(&p);
}

void jit_rvv_softmax_f16_gather(const dnnl::impl::float16_t *src,
        dnnl::impl::float16_t *dst, dim_t len, dim_t stride_bytes) {
    const jit_rvv_softmax_f16_strided_kernel_t::call_params_t p {
            src, dst, len, stride_bytes};
    dispatch_strided<true>(&p);
}

void jit_rvv_softmax_f16_scatter(const dnnl::impl::float16_t *src,
        dnnl::impl::float16_t *dst, dim_t len, dim_t stride_bytes) {
    const jit_rvv_softmax_f16_strided_kernel_t::call_params_t p {
            src, dst, len, stride_bytes};
    dispatch_strided<false>(&p);
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

    ld(reg_src, reg_param, AFFINE_OFF(src));
    ld(reg_dst, reg_param, AFFINE_OFF(dst));
    ld(reg_len, reg_param, AFFINE_OFF(len));
    flw(f_sub, reg_param, AFFINE_OFF(sub));
    flw(f_mul, reg_param, AFFINE_OFF(mul));

    Label loop, done;
    L(loop);
    beqz(reg_len, done);

    if (src_f32_) {
        vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m2);
        vle32_v(v_f32, reg_src);
    } else {
        vsetvli(reg_vl, reg_len, SEW::e16, LMUL::m1);
        vle16_v(v_src, reg_src);
        vfwcvt_f_f_v(v_f32, v_src);
        vsetvli(reg_vl, reg_vl, SEW::e32, LMUL::m2);
    }

    vfsub_vf(v_f32, v_f32, f_sub);
    vfmul_vf(v_f32, v_f32, f_mul);
    vsetvli(reg_vl, reg_vl, SEW::e16, LMUL::m1);
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

    ld(reg_src, reg_param, STRIDED_OFF(src));
    ld(reg_dst, reg_param, STRIDED_OFF(dst));
    ld(reg_len, reg_param, STRIDED_OFF(len));
    ld(reg_stride, reg_param, STRIDED_OFF(stride_bytes));

    Label loop, done;
    L(loop);
    beqz(reg_len, done);

    vsetvli(reg_vl, reg_len, SEW::e16, LMUL::m1);
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

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
