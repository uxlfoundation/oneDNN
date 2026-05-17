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

#include <assert.h>

#include "cpu/rv64/jit_rvv_eltwise_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

namespace {

template <alg_kind_t alg>
void dispatch_jit_eltwise_f32(
        const jit_rvv_eltwise_fwd_kernel_t::call_params_t *p) {
    static const jit_rvv_eltwise_fwd_kernel_t kernel(alg);
    kernel(p);
}

} // namespace

jit_rvv_eltwise_fwd_kernel_t::jit_rvv_eltwise_fwd_kernel_t(alg_kind_t alg)
    : jit_generator_t("jit_rvv_eltwise_fwd_kernel"), alg_(alg) {
    create_kernel();
}

bool jit_rvv_eltwise_f32_supported(alg_kind_t alg) {
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

void jit_rvv_eltwise_apply_f32(alg_kind_t alg, const float *src, float *dst,
        dim_t len, float alpha, float beta) {
    const jit_rvv_eltwise_fwd_kernel_t::call_params_t p {
            src, dst, len, alpha, beta};
    switch (alg) {
        case alg_kind::eltwise_abs:
            dispatch_jit_eltwise_f32<alg_kind::eltwise_abs>(&p);
            break;
        case alg_kind::eltwise_clip:
            dispatch_jit_eltwise_f32<alg_kind::eltwise_clip>(&p);
            break;
        case alg_kind::eltwise_hardsigmoid:
            dispatch_jit_eltwise_f32<alg_kind::eltwise_hardsigmoid>(&p);
            break;
        case alg_kind::eltwise_hardswish:
            dispatch_jit_eltwise_f32<alg_kind::eltwise_hardswish>(&p);
            break;
        case alg_kind::eltwise_linear:
            dispatch_jit_eltwise_f32<alg_kind::eltwise_linear>(&p);
            break;
        case alg_kind::eltwise_relu:
            dispatch_jit_eltwise_f32<alg_kind::eltwise_relu>(&p);
            break;
        case alg_kind::eltwise_sqrt:
            dispatch_jit_eltwise_f32<alg_kind::eltwise_sqrt>(&p);
            break;
        case alg_kind::eltwise_square:
            dispatch_jit_eltwise_f32<alg_kind::eltwise_square>(&p);
            break;
        default: assert(!"unsupported f32 eltwise JIT alg");
    }
}

void jit_rvv_eltwise_fwd_kernel_t::compute_vector(const VReg &v_dst,
        const VReg &v_src, const VReg &v_tmp, const FReg &f_alpha,
        const FReg &f_beta, const FReg &f_zero, const FReg &f_one) {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
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
        default: assert(!"unsupported f32 eltwise JIT alg");
    }
#else
    UNUSED(v_dst, v_src, v_tmp, f_alpha, f_beta, f_zero, f_one);
#endif
}

void jit_rvv_eltwise_fwd_kernel_t::generate() {
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

    const VReg v_src(4);
    const VReg v_tmp(8);
    const VReg v_dst(12);

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
    vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m4);
    vle32_v(v_src, reg_src);
    compute_vector(v_dst, v_src, v_tmp, f_alpha, f_beta, f_zero, f_one);
    vse32_v(v_dst, reg_dst);
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

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
