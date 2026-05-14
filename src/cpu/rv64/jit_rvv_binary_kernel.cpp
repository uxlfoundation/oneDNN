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

#include "cpu/rv64/jit_rvv_binary_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

namespace {

template <alg_kind_t alg>
void dispatch_jit_binary_f32(const jit_rvv_binary_kernel_t::call_params_t *p) {
    static const jit_rvv_binary_kernel_t kernel(alg);
    kernel(p);
}

} // namespace

jit_rvv_binary_kernel_t::jit_rvv_binary_kernel_t(alg_kind_t alg)
    : jit_generator_t("jit_rvv_binary_kernel"), alg_(alg) {
    create_kernel();
}

bool jit_rvv_binary_f32_supported(alg_kind_t alg) {
    switch (alg) {
        case alg_kind::binary_add:
        case alg_kind::binary_div:
        case alg_kind::binary_eq:
        case alg_kind::binary_ge:
        case alg_kind::binary_gt:
        case alg_kind::binary_le:
        case alg_kind::binary_lt:
        case alg_kind::binary_max:
        case alg_kind::binary_min:
        case alg_kind::binary_mul:
        case alg_kind::binary_ne:
        case alg_kind::binary_select:
        case alg_kind::binary_sub: return true;
        default: return false;
    }
}

void jit_rvv_binary_apply_f32(alg_kind_t alg, const float *src0,
        const float *src1, const int8_t *src2, float *dst, dim_t len) {
    const jit_rvv_binary_kernel_t::call_params_t p {src0, src1, src2, dst, len};
    switch (alg) {
        case alg_kind::binary_add:
            dispatch_jit_binary_f32<alg_kind::binary_add>(&p);
            break;
        case alg_kind::binary_div:
            dispatch_jit_binary_f32<alg_kind::binary_div>(&p);
            break;
        case alg_kind::binary_eq:
            dispatch_jit_binary_f32<alg_kind::binary_eq>(&p);
            break;
        case alg_kind::binary_ge:
            dispatch_jit_binary_f32<alg_kind::binary_ge>(&p);
            break;
        case alg_kind::binary_gt:
            dispatch_jit_binary_f32<alg_kind::binary_gt>(&p);
            break;
        case alg_kind::binary_le:
            dispatch_jit_binary_f32<alg_kind::binary_le>(&p);
            break;
        case alg_kind::binary_lt:
            dispatch_jit_binary_f32<alg_kind::binary_lt>(&p);
            break;
        case alg_kind::binary_max:
            dispatch_jit_binary_f32<alg_kind::binary_max>(&p);
            break;
        case alg_kind::binary_min:
            dispatch_jit_binary_f32<alg_kind::binary_min>(&p);
            break;
        case alg_kind::binary_mul:
            dispatch_jit_binary_f32<alg_kind::binary_mul>(&p);
            break;
        case alg_kind::binary_ne:
            dispatch_jit_binary_f32<alg_kind::binary_ne>(&p);
            break;
        case alg_kind::binary_select:
            dispatch_jit_binary_f32<alg_kind::binary_select>(&p);
            break;
        case alg_kind::binary_sub:
            dispatch_jit_binary_f32<alg_kind::binary_sub>(&p);
            break;
        default: assert(!"unsupported f32 binary JIT alg");
    }
}

void jit_rvv_binary_kernel_t::compute_vector(const VReg &v_dst,
        const VReg &v_src0, const VReg &v_src1, const FReg &f_zero,
        const FReg &f_one) {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const VReg v_mask(0);
    switch (alg_) {
        case alg_kind::binary_add: vfadd_vv(v_dst, v_src0, v_src1); break;
        case alg_kind::binary_div: vfdiv_vv(v_dst, v_src0, v_src1); break;
        case alg_kind::binary_max: vfmax_vv(v_dst, v_src0, v_src1); break;
        case alg_kind::binary_min: vfmin_vv(v_dst, v_src0, v_src1); break;
        case alg_kind::binary_mul: vfmul_vv(v_dst, v_src0, v_src1); break;
        case alg_kind::binary_sub: vfsub_vv(v_dst, v_src0, v_src1); break;
        case alg_kind::binary_eq:
            vmfeq_vv(v_mask, v_src0, v_src1);
            vfmv_v_f(v_dst, f_zero);
            vfmerge_vfm(v_dst, v_dst, f_one);
            break;
        case alg_kind::binary_ne:
            vmfne_vv(v_mask, v_src0, v_src1);
            vfmv_v_f(v_dst, f_zero);
            vfmerge_vfm(v_dst, v_dst, f_one);
            break;
        case alg_kind::binary_lt:
            vmflt_vv(v_mask, v_src0, v_src1);
            vfmv_v_f(v_dst, f_zero);
            vfmerge_vfm(v_dst, v_dst, f_one);
            break;
        case alg_kind::binary_le:
            vmfle_vv(v_mask, v_src0, v_src1);
            vfmv_v_f(v_dst, f_zero);
            vfmerge_vfm(v_dst, v_dst, f_one);
            break;
        case alg_kind::binary_gt:
            vmfgt_vv(v_mask, v_src0, v_src1);
            vfmv_v_f(v_dst, f_zero);
            vfmerge_vfm(v_dst, v_dst, f_one);
            break;
        case alg_kind::binary_ge:
            vmfge_vv(v_mask, v_src0, v_src1);
            vfmv_v_f(v_dst, f_zero);
            vfmerge_vfm(v_dst, v_dst, f_one);
            break;
        case alg_kind::binary_select:
            // Handled in generate() because it needs the src2 condition input.
            break;
        default: assert(!"unsupported f32 binary JIT alg");
    }
#else
    UNUSED(v_dst, v_src0, v_src1, f_zero, f_one);
#endif
}

void jit_rvv_binary_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const Reg reg_src0 = a1;
    const Reg reg_src1 = a2;
    const Reg reg_src2 = a3;
    const Reg reg_dst = a4;
    const Reg reg_len = a5;
    const Reg reg_vl = t0;
    const Reg reg_bytes = t1;
    const Reg reg_tmp = t2;

    const FReg f_zero = fa0;
    const FReg f_one = fa1;

    const VReg v_mask(0);
    const VReg v_src0(4);
    const VReg v_src1(8);
    const VReg v_dst(12);

    // call_params_t layout:
    //  0: src0, 8: src1, 16: src2, 24: dst, 32: len
    ld(reg_src0, reg_param, 0);
    ld(reg_src1, reg_param, 8);
    ld(reg_src2, reg_param, 16);
    ld(reg_dst, reg_param, 24);
    ld(reg_len, reg_param, 32);

    if (alg_ == alg_kind::binary_select) {
        const VReg v_src0_m1(1);
        const VReg v_src1_m1(2);
        const VReg v_dst_m1(3);
        const VReg v_src2_mf4(4);

        Label select_loop, select_done;
        L(select_loop);
        beqz(reg_len, select_done);

        vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m1);
        vle32_v(v_src0_m1, reg_src0);
        vle32_v(v_src1_m1, reg_src1);

        vsetvli(reg_vl, reg_vl, SEW::e8, LMUL::mf4);
        vle8_v(v_src2_mf4, reg_src2);
        vmsne_vi(v_mask, v_src2_mf4, 0);

        vsetvli(reg_vl, reg_vl, SEW::e32, LMUL::m1);
        vmerge_vvm(v_dst_m1, v_src1_m1, v_src0_m1);
        vse32_v(v_dst_m1, reg_dst);

        slli(reg_bytes, reg_vl, 2);
        add(reg_src0, reg_src0, reg_bytes);
        add(reg_src1, reg_src1, reg_bytes);
        add(reg_src2, reg_src2, reg_vl);
        add(reg_dst, reg_dst, reg_bytes);
        sub(reg_len, reg_len, reg_vl);
        j_(select_loop);

        L(select_done);
        ret();
    }

    li(reg_tmp, 0);
    fmv_w_x(f_zero, reg_tmp);
    li(reg_tmp, 0x3f800000);
    fmv_w_x(f_one, reg_tmp);

    Label loop, done;
    L(loop);
    beqz(reg_len, done);
    vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m4);
    vle32_v(v_src0, reg_src0);
    vle32_v(v_src1, reg_src1);
    compute_vector(v_dst, v_src0, v_src1, f_zero, f_one);
    vse32_v(v_dst, reg_dst);
    slli(reg_bytes, reg_vl, 2);
    add(reg_src0, reg_src0, reg_bytes);
    add(reg_src1, reg_src1, reg_bytes);
    add(reg_dst, reg_dst, reg_bytes);
    sub(reg_len, reg_len, reg_vl);
    j_(loop);

    L(done);
    ret();

    UNUSED(v_mask);
#else
    ret();
#endif
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
