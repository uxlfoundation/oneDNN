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

#include "cpu/rv64/jit_rvv_layernorm_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

jit_rvv_layernorm_kernel_t::jit_rvv_layernorm_kernel_t(
        bool with_scale, bool with_shift)
    : jit_generator_t("jit_rvv_layernorm_kernel")
    , with_scale_(with_scale)
    , with_shift_(with_shift) {
    create_kernel();
}

void jit_rvv_layernorm_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const Reg reg_src = a1;
    const Reg reg_dst = a2;
    const Reg reg_scale = a3;
    const Reg reg_shift = a4;
    const Reg reg_len = a5;
    const Reg reg_vl = t0;
    const Reg reg_bytes = t1;
    const Reg reg_tmp = t2;
    const Reg reg_vlmax = t3;
    const Reg reg_block = t4;
    const Reg reg_bytes4 = t5;

    const FReg f_mean = fa0;
    const FReg f_inv = fa1;

    const VReg v_in0(0);
    const VReg v_in1(1);
    const VReg v_in2(2);
    const VReg v_in3(3);
    const VReg v_mean(4);
    const VReg v_inv(5);
    const VReg v_scale0(6);
    const VReg v_scale1(7);
    const VReg v_scale2(8);
    const VReg v_scale3(9);
    const VReg v_shift0(10);
    const VReg v_shift1(11);
    const VReg v_shift2(12);
    const VReg v_shift3(13);

    // call_params_t layout:
    //  0: src, 8: dst, 16: scale, 24: shift, 32: len, 40: mean, 44: inv_std
    ld(reg_src, reg_param, 0);
    ld(reg_dst, reg_param, 8);
    ld(reg_scale, reg_param, 16);
    ld(reg_shift, reg_param, 24);
    ld(reg_len, reg_param, 32);

    lw(reg_tmp, reg_param, 40);
    fmv_w_x(f_mean, reg_tmp);
    lw(reg_tmp, reg_param, 44);
    fmv_w_x(f_inv, reg_tmp);

    // vlmax for main unrolled loop
    vsetvli(reg_vlmax, x0, SEW::e32, LMUL::m1);
    slli(reg_bytes, reg_vlmax, 2); // bytes for 1 * vlmax
    slli(reg_bytes4, reg_bytes, 2); // bytes for 4 * vlmax
    slli(reg_block, reg_vlmax, 2); // elements for 4 * vlmax

    Label main_loop, tail_loop, done;

    L(main_loop);
    blt(reg_len, reg_block, tail_loop);

    vfmv_v_f(v_mean, f_mean);
    vfmv_v_f(v_inv, f_inv);

    // load 4 vectors
    vle32_v(v_in0, reg_src);
    add(reg_tmp, reg_src, reg_bytes);
    vle32_v(v_in1, reg_tmp);
    add(reg_tmp, reg_tmp, reg_bytes);
    vle32_v(v_in2, reg_tmp);
    add(reg_tmp, reg_tmp, reg_bytes);
    vle32_v(v_in3, reg_tmp);

    // normalize
    vfsub_vv(v_in0, v_in0, v_mean);
    vfsub_vv(v_in1, v_in1, v_mean);
    vfsub_vv(v_in2, v_in2, v_mean);
    vfsub_vv(v_in3, v_in3, v_mean);
    vfmul_vv(v_in0, v_in0, v_inv);
    vfmul_vv(v_in1, v_in1, v_inv);
    vfmul_vv(v_in2, v_in2, v_inv);
    vfmul_vv(v_in3, v_in3, v_inv);

    if (with_scale_) {
        vle32_v(v_scale0, reg_scale);
        add(reg_tmp, reg_scale, reg_bytes);
        vle32_v(v_scale1, reg_tmp);
        add(reg_tmp, reg_tmp, reg_bytes);
        vle32_v(v_scale2, reg_tmp);
        add(reg_tmp, reg_tmp, reg_bytes);
        vle32_v(v_scale3, reg_tmp);
        if (with_shift_) {
            vle32_v(v_shift0, reg_shift);
            add(reg_tmp, reg_shift, reg_bytes);
            vle32_v(v_shift1, reg_tmp);
            add(reg_tmp, reg_tmp, reg_bytes);
            vle32_v(v_shift2, reg_tmp);
            add(reg_tmp, reg_tmp, reg_bytes);
            vle32_v(v_shift3, reg_tmp);
            vfmacc_vv(v_shift0, v_scale0, v_in0);
            vfmacc_vv(v_shift1, v_scale1, v_in1);
            vfmacc_vv(v_shift2, v_scale2, v_in2);
            vfmacc_vv(v_shift3, v_scale3, v_in3);
            vse32_v(v_shift0, reg_dst);
            add(reg_tmp, reg_dst, reg_bytes);
            vse32_v(v_shift1, reg_tmp);
            add(reg_tmp, reg_tmp, reg_bytes);
            vse32_v(v_shift2, reg_tmp);
            add(reg_tmp, reg_tmp, reg_bytes);
            vse32_v(v_shift3, reg_tmp);
        } else {
            vfmul_vv(v_in0, v_in0, v_scale0);
            vfmul_vv(v_in1, v_in1, v_scale1);
            vfmul_vv(v_in2, v_in2, v_scale2);
            vfmul_vv(v_in3, v_in3, v_scale3);
            vse32_v(v_in0, reg_dst);
            add(reg_tmp, reg_dst, reg_bytes);
            vse32_v(v_in1, reg_tmp);
            add(reg_tmp, reg_tmp, reg_bytes);
            vse32_v(v_in2, reg_tmp);
            add(reg_tmp, reg_tmp, reg_bytes);
            vse32_v(v_in3, reg_tmp);
        }
    } else if (with_shift_) {
        vle32_v(v_shift0, reg_shift);
        add(reg_tmp, reg_shift, reg_bytes);
        vle32_v(v_shift1, reg_tmp);
        add(reg_tmp, reg_tmp, reg_bytes);
        vle32_v(v_shift2, reg_tmp);
        add(reg_tmp, reg_tmp, reg_bytes);
        vle32_v(v_shift3, reg_tmp);
        vfadd_vv(v_in0, v_in0, v_shift0);
        vfadd_vv(v_in1, v_in1, v_shift1);
        vfadd_vv(v_in2, v_in2, v_shift2);
        vfadd_vv(v_in3, v_in3, v_shift3);
        vse32_v(v_in0, reg_dst);
        add(reg_tmp, reg_dst, reg_bytes);
        vse32_v(v_in1, reg_tmp);
        add(reg_tmp, reg_tmp, reg_bytes);
        vse32_v(v_in2, reg_tmp);
        add(reg_tmp, reg_tmp, reg_bytes);
        vse32_v(v_in3, reg_tmp);
    } else {
        vse32_v(v_in0, reg_dst);
        add(reg_tmp, reg_dst, reg_bytes);
        vse32_v(v_in1, reg_tmp);
        add(reg_tmp, reg_tmp, reg_bytes);
        vse32_v(v_in2, reg_tmp);
        add(reg_tmp, reg_tmp, reg_bytes);
        vse32_v(v_in3, reg_tmp);
    }

    add(reg_src, reg_src, reg_bytes4);
    add(reg_dst, reg_dst, reg_bytes4);
    if (with_scale_) add(reg_scale, reg_scale, reg_bytes4);
    if (with_shift_) add(reg_shift, reg_shift, reg_bytes4);
    sub(reg_len, reg_len, reg_block);
    j_(main_loop);

    L(tail_loop);
    beqz(reg_len, done);
    vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m1);
    vfmv_v_f(v_mean, f_mean);
    vfmv_v_f(v_inv, f_inv);

    vle32_v(v_in0, reg_src);
    vfsub_vv(v_in0, v_in0, v_mean);
    vfmul_vv(v_in0, v_in0, v_inv);
    if (with_scale_) {
        vle32_v(v_scale0, reg_scale);
        if (with_shift_) {
            vle32_v(v_shift0, reg_shift);
            vfmacc_vv(v_shift0, v_scale0, v_in0);
            vse32_v(v_shift0, reg_dst);
        } else {
            vfmul_vv(v_in0, v_in0, v_scale0);
            vse32_v(v_in0, reg_dst);
        }
    } else if (with_shift_) {
        vle32_v(v_shift0, reg_shift);
        vfadd_vv(v_in0, v_in0, v_shift0);
        vse32_v(v_in0, reg_dst);
    } else {
        vse32_v(v_in0, reg_dst);
    }
    slli(reg_bytes, reg_vl, 2);
    add(reg_src, reg_src, reg_bytes);
    add(reg_dst, reg_dst, reg_bytes);
    if (with_scale_) add(reg_scale, reg_scale, reg_bytes);
    if (with_shift_) add(reg_shift, reg_shift, reg_bytes);
    sub(reg_len, reg_len, reg_vl);
    j_(tail_loop);

    L(done);
    ret();
#endif
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
