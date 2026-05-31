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

#include "cpu/rv64/jit_rvv_matmul_post_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace matmul {

using namespace Xbyak_riscv;

#define GET_OFF(field) \
    static_cast<int32_t>( \
            offsetof(jit_rvv_matmul_post_kernel_t::call_params_t, field))

namespace {

template <bool with_bias, bool scalar_bias, bool with_relu>
void dispatch_jit_matmul_post(
        const jit_rvv_matmul_post_kernel_t::call_params_t *p) {
    static const jit_rvv_matmul_post_kernel_t kernel(
            with_bias, scalar_bias, with_relu);
    kernel(p);
}

} // namespace

jit_rvv_matmul_post_kernel_t::jit_rvv_matmul_post_kernel_t(
        bool with_bias, bool scalar_bias, bool with_relu)
    : jit_generator_t("jit_rvv_matmul_post_kernel")
    , with_bias_(with_bias)
    , scalar_bias_(scalar_bias)
    , with_relu_(with_relu) {
    create_kernel();
}

void jit_rvv_matmul_post_apply(float *dst, const float *bias, dim_t len,
        bool with_bias, bool scalar_bias, bool with_relu, float relu_alpha) {
    const jit_rvv_matmul_post_kernel_t::call_params_t p {
            dst, bias, len, relu_alpha};
    if (with_bias) {
        if (scalar_bias) {
            if (with_relu)
                dispatch_jit_matmul_post<true, true, true>(&p);
            else
                dispatch_jit_matmul_post<true, true, false>(&p);
        } else {
            if (with_relu)
                dispatch_jit_matmul_post<true, false, true>(&p);
            else
                dispatch_jit_matmul_post<true, false, false>(&p);
        }
    } else {
        if (with_relu)
            dispatch_jit_matmul_post<false, false, true>(&p);
        else
            dispatch_jit_matmul_post<false, false, false>(&p);
    }
}

void jit_rvv_matmul_post_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const Reg reg_dst = a1;
    const Reg reg_bias = a2;
    const Reg reg_len = a3;
    const Reg reg_vl = t0;
    const Reg reg_bytes = t1;

    const FReg f_bias = fa0;
    const FReg f_alpha = fa1;
    const FReg f_zero = fa2;

    const VReg v_mask(0);
    const VReg v_dst(4);
    const VReg v_bias(8);
    const VReg v_tmp(12);

    ld(reg_dst, reg_param, GET_OFF(dst));
    ld(reg_bias, reg_param, GET_OFF(bias));
    ld(reg_len, reg_param, GET_OFF(len));
    flw(f_alpha, reg_param, GET_OFF(relu_alpha));
    fmv_w_x(f_zero, x0);
    if (with_bias_ && scalar_bias_) flw(f_bias, reg_bias, 0);

    Label loop, done;
    L(loop);
    beqz(reg_len, done);

    vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m4);
    vle32_v(v_dst, reg_dst);

    if (with_bias_) {
        if (scalar_bias_) {
            vfadd_vf(v_dst, v_dst, f_bias);
        } else {
            vle32_v(v_bias, reg_bias);
            vfadd_vv(v_dst, v_dst, v_bias);
        }
    }

    if (with_relu_) {
        vmfgt_vf(v_mask, v_dst, f_zero);
        vfmul_vf(v_tmp, v_dst, f_alpha);
        vmerge_vvm(v_dst, v_tmp, v_dst);
    }

    vse32_v(v_dst, reg_dst);

    slli(reg_bytes, reg_vl, 2);
    add(reg_dst, reg_dst, reg_bytes);
    if (with_bias_ && !scalar_bias_) add(reg_bias, reg_bias, reg_bytes);
    sub(reg_len, reg_len, reg_vl);
    j_(loop);

    L(done);
    ret();
#else
    ret();
#endif
}

} // namespace matmul
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
