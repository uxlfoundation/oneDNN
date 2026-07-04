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

#include "cpu/rv64/jit_rvv_batch_normalization_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

namespace {

template <bool per_elem_params, bool with_relu>
void dispatch_jit_batch_normalization_f32(
        const jit_rvv_batch_normalization_fwd_kernel_t::call_params_t *p) {
    static const jit_rvv_batch_normalization_fwd_kernel_t kernel(
            per_elem_params, with_relu);
    kernel(p);
}

} // namespace

jit_rvv_batch_normalization_fwd_kernel_t::
        jit_rvv_batch_normalization_fwd_kernel_t(
                bool per_elem_params, bool with_relu)
    : jit_generator_t("jit_rvv_batch_normalization_fwd_kernel")
    , per_elem_params_(per_elem_params)
    , with_relu_(with_relu) {
    create_kernel();
}

void jit_rvv_batch_normalization_apply_f32(const float *src, float *dst,
        dim_t len, const float *mean, const float *scale_mul,
        const float *scale_add, bool per_elem_params, bool with_relu) {
    const jit_rvv_batch_normalization_fwd_kernel_t::call_params_t p {
            src, dst, len, mean, scale_mul, scale_add};
    if (per_elem_params) {
        if (with_relu)
            dispatch_jit_batch_normalization_f32<true, true>(&p);
        else
            dispatch_jit_batch_normalization_f32<true, false>(&p);
    } else {
        if (with_relu)
            dispatch_jit_batch_normalization_f32<false, true>(&p);
        else
            dispatch_jit_batch_normalization_f32<false, false>(&p);
    }
}

void jit_rvv_batch_normalization_fwd_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const Reg reg_src = a1;
    const Reg reg_dst = a2;
    const Reg reg_len = a3;
    const Reg reg_mean = a4;
    const Reg reg_sm = a5;
    const Reg reg_sv = a6;
    const Reg reg_vl = t0;
    const Reg reg_bytes = t1;

    const FReg f_mean = fa0;
    const FReg f_sm = fa1;
    const FReg f_sv = fa2;
    const FReg f_zero = fa3;

    const VReg v_mask(0);
    const VReg v_src(4);
    const VReg v_mean(8);
    const VReg v_sm(12);
    const VReg v_sv(16);

    ld(reg_src, reg_param, 0);
    ld(reg_dst, reg_param, 8);
    ld(reg_len, reg_param, 16);
    ld(reg_mean, reg_param, 24);
    ld(reg_sm, reg_param, 32);
    ld(reg_sv, reg_param, 40);

    if (!per_elem_params_) {
        flw(f_mean, reg_mean, 0);
        flw(f_sm, reg_sm, 0);
        flw(f_sv, reg_sv, 0);
    }
    fmv_w_x(f_zero, x0);

    Label loop, done;
    L(loop);
    beqz(reg_len, done);
    vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
    vle32_v(v_src, reg_src);
    if (per_elem_params_) {
        vle32_v(v_mean, reg_mean);
        vle32_v(v_sm, reg_sm);
        vle32_v(v_sv, reg_sv);
        vfsub_vv(v_src, v_src, v_mean);
        vfmul_vv(v_src, v_src, v_sm);
        vfadd_vv(v_src, v_src, v_sv);
    } else {
        vfsub_vf(v_src, v_src, f_mean);
        vfmul_vf(v_src, v_src, f_sm);
        vfadd_vf(v_src, v_src, f_sv);
    }
    if (with_relu_) {
        vmflt_vf(v_mask, v_src, f_zero);
        vfmerge_vfm(v_src, v_src, f_zero);
    }
    vse32_v(v_src, reg_dst);

    slli(reg_bytes, reg_vl, 2);
    add(reg_src, reg_src, reg_bytes);
    add(reg_dst, reg_dst, reg_bytes);
    if (per_elem_params_) {
        add(reg_mean, reg_mean, reg_bytes);
        add(reg_sm, reg_sm, reg_bytes);
        add(reg_sv, reg_sv, reg_bytes);
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
