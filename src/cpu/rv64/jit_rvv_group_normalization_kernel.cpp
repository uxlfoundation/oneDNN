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

#include "cpu/rv64/jit_rvv_group_normalization_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

namespace {

template <bool use_scale, bool use_shift>
void dispatch_jit_group_normalization_f32(
        const jit_rvv_group_normalization_fwd_kernel_t::call_params_t *p) {
    static const jit_rvv_group_normalization_fwd_kernel_t kernel(
            use_scale, use_shift);
    kernel(p);
}

} // namespace

jit_rvv_group_normalization_fwd_kernel_t::
        jit_rvv_group_normalization_fwd_kernel_t(bool use_scale, bool use_shift)
    : jit_generator_t("jit_rvv_group_normalization_fwd_kernel")
    , use_scale_(use_scale)
    , use_shift_(use_shift) {
    create_kernel();
}

void jit_rvv_group_normalization_apply_f32(const float *src, float *dst,
        dim_t len, float mean, float inv_std, float gamma, float beta,
        bool use_scale, bool use_shift) {
    const jit_rvv_group_normalization_fwd_kernel_t::call_params_t p {
            src, dst, len, mean, inv_std, gamma, beta};
    if (use_scale) {
        if (use_shift)
            dispatch_jit_group_normalization_f32<true, true>(&p);
        else
            dispatch_jit_group_normalization_f32<true, false>(&p);
    } else {
        if (use_shift)
            dispatch_jit_group_normalization_f32<false, true>(&p);
        else
            dispatch_jit_group_normalization_f32<false, false>(&p);
    }
}

void jit_rvv_group_normalization_fwd_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const Reg reg_src = a1;
    const Reg reg_dst = a2;
    const Reg reg_len = a3;
    const Reg reg_vl = t0;
    const Reg reg_bytes = t1;

    const FReg f_mean = fa0;
    const FReg f_inv_std = fa1;
    const FReg f_gamma = fa2;
    const FReg f_beta = fa3;

    const VReg v_src(4);

    ld(reg_src, reg_param, 0);
    ld(reg_dst, reg_param, 8);
    ld(reg_len, reg_param, 16);
    flw(f_mean, reg_param, 24);
    flw(f_inv_std, reg_param, 28);
    flw(f_gamma, reg_param, 32);
    flw(f_beta, reg_param, 36);

    Label loop, done;
    L(loop);
    beqz(reg_len, done);
    vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m1);
    vle32_v(v_src, reg_src);
    vfsub_vf(v_src, v_src, f_mean);
    vfmul_vf(v_src, v_src, f_inv_std);
    if (use_scale_) vfmul_vf(v_src, v_src, f_gamma);
    if (use_shift_) vfadd_vf(v_src, v_src, f_beta);
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

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
