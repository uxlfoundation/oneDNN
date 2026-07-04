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

#include "cpu/rv64/jit_rvv_gemm_convolution_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

#define COPY_OFF(field) \
    static_cast<int32_t>(offsetof( \
            jit_rvv_gemm_convolution_copy_kernel_t::call_params_t, field))
#define POST_OFF(field) \
    static_cast<int32_t>(offsetof( \
            jit_rvv_gemm_convolution_post_kernel_t::call_params_t, field))

namespace {

#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
void dispatch_gemm_convolution_copy(
        const jit_rvv_gemm_convolution_copy_kernel_t::call_params_t *p) {
    static const jit_rvv_gemm_convolution_copy_kernel_t kernel;
    kernel(p);
}
#endif

template <bool vector_bias, bool with_relu, bool relu_alpha_zero,
        bool with_scale>
void dispatch_gemm_convolution_post(
        const jit_rvv_gemm_convolution_post_kernel_t::call_params_t *p) {
    static const jit_rvv_gemm_convolution_post_kernel_t kernel(
            vector_bias, with_relu, relu_alpha_zero, with_scale);
    kernel(p);
}

} // namespace

jit_rvv_gemm_convolution_copy_kernel_t::jit_rvv_gemm_convolution_copy_kernel_t()
    : jit_generator_t("jit_rvv_gemm_convolution_copy_kernel") {
    create_kernel();
}

void jit_rvv_gemm_convolution_copy_f32(
        const float *src, float *dst, dim_t len) {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const jit_rvv_gemm_convolution_copy_kernel_t::call_params_t p {
            src, dst, len};
    dispatch_gemm_convolution_copy(&p);
#else
    for (dim_t i = 0; i < len; ++i)
        dst[i] = src[i];
#endif
}

void jit_rvv_gemm_convolution_copy_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const Reg reg_src = a1;
    const Reg reg_dst = a2;
    const Reg reg_len = a3;
    const Reg reg_vl = t0;
    const Reg reg_bytes = t1;

    const VReg v_data(4);

    ld(reg_src, reg_param, COPY_OFF(src));
    ld(reg_dst, reg_param, COPY_OFF(dst));
    ld(reg_len, reg_param, COPY_OFF(len));

    Label loop, done;
    L(loop);
    beqz(reg_len, done);

    vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
    vle32_v(v_data, reg_src);
    vse32_v(v_data, reg_dst);

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

jit_rvv_gemm_convolution_post_kernel_t::jit_rvv_gemm_convolution_post_kernel_t(
        bool vector_bias, bool with_relu, bool relu_alpha_zero, bool with_scale)
    : jit_generator_t("jit_rvv_gemm_convolution_post_kernel")
    , vector_bias_(vector_bias)
    , with_relu_(with_relu)
    , relu_alpha_zero_(relu_alpha_zero)
    , with_scale_(with_scale) {
    create_kernel();
}

void jit_rvv_gemm_convolution_apply_bias(
        float *dst, const float *bias, dim_t len) {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const jit_rvv_gemm_convolution_post_kernel_t::call_params_t p {
            dst, bias, len, 0.0f, 0.0f, 1.0f};
    dispatch_gemm_convolution_post<true, false, true, false>(&p);
#else
    for (dim_t i = 0; i < len; ++i)
        dst[i] += bias[i];
#endif
}

void jit_rvv_gemm_convolution_apply_scalar_bias(
        float *dst, dim_t len, float bias) {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const jit_rvv_gemm_convolution_post_kernel_t::call_params_t p {
            dst, nullptr, len, bias, 0.0f, 1.0f};
    dispatch_gemm_convolution_post<false, false, true, false>(&p);
#else
    for (dim_t i = 0; i < len; ++i)
        dst[i] += bias;
#endif
}

void jit_rvv_gemm_convolution_apply_scalar_bias_relu(
        float *dst, dim_t len, float bias, float relu_alpha, float scale) {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const jit_rvv_gemm_convolution_post_kernel_t::call_params_t p {
            dst, nullptr, len, bias, relu_alpha, scale};
    const bool alpha_zero = relu_alpha == 0.0f;
    if (alpha_zero) {
        if (scale != 1.0f)
            dispatch_gemm_convolution_post<false, true, true, true>(&p);
        else
            dispatch_gemm_convolution_post<false, true, true, false>(&p);
    } else {
        dispatch_gemm_convolution_post<false, true, false, true>(&p);
    }
#else
    for (dim_t i = 0; i < len; ++i) {
        float value = dst[i] + bias;
        if (relu_alpha == 0.0f)
            value = value < 0.0f ? 0.0f : value;
        else if (value < 0.0f)
            value *= relu_alpha;
        dst[i] = value * scale;
    }
#endif
}

void jit_rvv_gemm_convolution_post_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const Reg reg_dst = a1;
    const Reg reg_bias = a2;
    const Reg reg_len = a3;
    const Reg reg_vl = t0;
    const Reg reg_bytes = t1;

    const FReg f_bias = fa0;
    const FReg f_alpha = fa1;
    const FReg f_scale = fa2;
    const FReg f_zero = fa3;

    const VReg v_mask(0);
    const VReg v_dst(4);
    const VReg v_bias(8);
    const VReg v_tmp(12);

    ld(reg_dst, reg_param, POST_OFF(dst));
    ld(reg_bias, reg_param, POST_OFF(bias));
    ld(reg_len, reg_param, POST_OFF(len));
    flw(f_bias, reg_param, POST_OFF(scalar_bias));
    flw(f_alpha, reg_param, POST_OFF(relu_alpha));
    flw(f_scale, reg_param, POST_OFF(scale));
    fmv_w_x(f_zero, x0);

    Label loop, done;
    L(loop);
    beqz(reg_len, done);

    vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m4, VTA::ta, VMA::ma);
    vle32_v(v_dst, reg_dst);

    if (vector_bias_) {
        vle32_v(v_bias, reg_bias);
        vfadd_vv(v_dst, v_dst, v_bias);
    } else {
        vfadd_vf(v_dst, v_dst, f_bias);
    }

    if (with_relu_) {
        if (relu_alpha_zero_) {
            vmflt_vf(v_mask, v_dst, f_zero);
            vfmerge_vfm(v_dst, v_dst, f_zero);
        } else {
            vmflt_vf(v_mask, v_dst, f_zero);
            vfmul_vf(v_tmp, v_dst, f_alpha);
            vmerge_vvm(v_dst, v_dst, v_tmp);
        }
        if (with_scale_) vfmul_vf(v_dst, v_dst, f_scale);
    }

    vse32_v(v_dst, reg_dst);

    slli(reg_bytes, reg_vl, 2);
    add(reg_dst, reg_dst, reg_bytes);
    if (vector_bias_) add(reg_bias, reg_bias, reg_bytes);
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
