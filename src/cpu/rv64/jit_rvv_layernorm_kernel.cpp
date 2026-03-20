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

#include "cpu/rv64/jit_rvv_layernorm_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

#define GET_FUSED_OFF(field) \
    static_cast<int32_t>( \
            offsetof(jit_rvv_layernorm_fused_kernel_t::call_params_t, field))
#define GET_DATA_OFF(field) \
    static_cast<int32_t>( \
            offsetof(jit_rvv_layernorm_data_kernel_t::call_params_t, field))

jit_rvv_layernorm_fused_kernel_t::jit_rvv_layernorm_fused_kernel_t(
        bool with_scale, bool with_shift)
    : jit_generator_t("jit_rvv_layernorm_fused_kernel")
    , with_scale_(with_scale)
    , with_shift_(with_shift) {
    create_kernel();
}

void jit_rvv_layernorm_fused_kernel_t::generate() {
    const Reg reg_param = a0;
    const Reg reg_src = a1;
    const Reg reg_len = a2;
    const Reg reg_dst = a3;
    const Reg reg_scale = a4;
    const Reg reg_shift = a5;
    const Reg reg_mean_ptr = a6;
    const Reg reg_var_ptr = a7;
    const Reg reg_vl = t0;
    const Reg reg_vlmax = t1;
    const Reg reg_bytes = t2;
    const Reg reg_bytes4 = t3;
    const Reg reg_block = t4;
    const Reg reg_tmp = t5;

    const FReg f_zero = ft0;
    const FReg f_sum = ft1;
    const FReg f_var_sum = ft2;
    const FReg f_len = ft3;
    const FReg f_mean = ft4;
    const FReg f_var = ft5;
    const FReg f_inv = ft6;
    const FReg f_aux = ft7;
    const FReg f_one = ft8;

    const VReg v_in0(0);
    const VReg v_in1(1);
    const VReg v_in2(2);
    const VReg v_in3(3);
    const VReg v_tmp0(4);
    const VReg v_tmp1(5);
    const VReg v_tmp2(6);
    const VReg v_tmp3(7);
    const VReg v_mean(12);
    const VReg v_inv(13);
    const VReg v_scale0(14);
    const VReg v_scale1(15);
    const VReg v_scale2(16);
    const VReg v_scale3(17);
    const VReg v_shift0(18);
    const VReg v_shift1(19);
    const VReg v_shift2(20);
    const VReg v_shift3(21);
    fmv_w_x(f_zero, x0);

    vsetvli(reg_vlmax, x0, SEW::e32, LMUL::m1);
    slli(reg_bytes, reg_vlmax, 2);
    slli(reg_bytes4, reg_bytes, 2);
    slli(reg_block, reg_vlmax, 2);

    Label mean_main_loop, mean_vec_tail, mean_reduce_loop, mean_done;
    Label var_main_loop, var_vec_tail, var_reduce_loop, var_done;
    Label store_var, start_data, data_main_loop, data_tail_loop, done;

    ld(reg_src, reg_param, GET_FUSED_OFF(src));
    ld(reg_len, reg_param, GET_FUSED_OFF(len));
    fsgnj_s(f_sum, f_zero, f_zero);
    vfmv_v_f(v_tmp0, f_zero);
    vfmv_v_f(v_tmp1, f_zero);
    vfmv_v_f(v_tmp2, f_zero);
    vfmv_v_f(v_tmp3, f_zero);

    vsetvli(x0, reg_vlmax, SEW::e32, LMUL::m1);
    L(mean_main_loop);
    blt(reg_len, reg_block, mean_vec_tail);
    vle32_v(v_in0, reg_src);
    add(reg_tmp, reg_src, reg_bytes);
    vle32_v(v_in1, reg_tmp);
    add(reg_tmp, reg_tmp, reg_bytes);
    vle32_v(v_in2, reg_tmp);
    add(reg_tmp, reg_tmp, reg_bytes);
    vle32_v(v_in3, reg_tmp);
    vfadd_vv(v_tmp0, v_tmp0, v_in0);
    vfadd_vv(v_tmp1, v_tmp1, v_in1);
    vfadd_vv(v_tmp2, v_tmp2, v_in2);
    vfadd_vv(v_tmp3, v_tmp3, v_in3);
    add(reg_src, reg_src, reg_bytes4);
    sub(reg_len, reg_len, reg_block);
    j_(mean_main_loop);

    L(mean_vec_tail);
    beqz(reg_len, mean_reduce_loop);
    vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m1);
    slli(reg_tmp, reg_vl, 2);
    vle32_v(v_in0, reg_src);
    vfadd_vv(v_tmp0, v_tmp0, v_in0);
    add(reg_src, reg_src, reg_tmp);
    sub(reg_len, reg_len, reg_vl);
    j_(mean_vec_tail);

    L(mean_reduce_loop);
    vsetvli(x0, reg_vlmax, SEW::e32, LMUL::m1);
    vfadd_vv(v_tmp0, v_tmp0, v_tmp1);
    vfadd_vv(v_tmp2, v_tmp2, v_tmp3);
    vfadd_vv(v_tmp0, v_tmp0, v_tmp2);
    vfmv_v_f(v_in0, f_zero);
    vfredusum_vs(v_in1, v_tmp0, v_in0);
    vfmv_f_s(f_aux, v_in1);
    fadd_s(f_sum, f_sum, f_aux);

    L(mean_done);
    ld(reg_len, reg_param, GET_FUSED_OFF(len));
    fcvt_s_l(f_len, reg_len);
    fdiv_s(f_mean, f_sum, f_len);

    ld(reg_src, reg_param, GET_FUSED_OFF(src));
    ld(reg_len, reg_param, GET_FUSED_OFF(len));
    fsgnj_s(f_var_sum, f_zero, f_zero);
    vfmv_v_f(v_tmp0, f_zero);
    vfmv_v_f(v_tmp1, f_zero);
    vfmv_v_f(v_tmp2, f_zero);
    vfmv_v_f(v_tmp3, f_zero);
    vsetvli(x0, reg_vlmax, SEW::e32, LMUL::m1);
    vfmv_v_f(v_mean, f_mean);

    L(var_main_loop);
    blt(reg_len, reg_block, var_vec_tail);
    vle32_v(v_in0, reg_src);
    add(reg_tmp, reg_src, reg_bytes);
    vle32_v(v_in1, reg_tmp);
    add(reg_tmp, reg_tmp, reg_bytes);
    vle32_v(v_in2, reg_tmp);
    add(reg_tmp, reg_tmp, reg_bytes);
    vle32_v(v_in3, reg_tmp);
    vfsub_vv(v_in0, v_in0, v_mean);
    vfsub_vv(v_in1, v_in1, v_mean);
    vfsub_vv(v_in2, v_in2, v_mean);
    vfsub_vv(v_in3, v_in3, v_mean);
    vfmacc_vv(v_tmp0, v_in0, v_in0);
    vfmacc_vv(v_tmp1, v_in1, v_in1);
    vfmacc_vv(v_tmp2, v_in2, v_in2);
    vfmacc_vv(v_tmp3, v_in3, v_in3);
    add(reg_src, reg_src, reg_bytes4);
    sub(reg_len, reg_len, reg_block);
    j_(var_main_loop);

    L(var_vec_tail);
    beqz(reg_len, var_reduce_loop);
    vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m1);
    slli(reg_tmp, reg_vl, 2);
    vle32_v(v_in0, reg_src);
    vfsub_vv(v_in0, v_in0, v_mean);
    vfmacc_vv(v_tmp0, v_in0, v_in0);
    add(reg_src, reg_src, reg_tmp);
    sub(reg_len, reg_len, reg_vl);
    j_(var_vec_tail);

    L(var_reduce_loop);
    vsetvli(x0, reg_vlmax, SEW::e32, LMUL::m1);
    vfadd_vv(v_tmp0, v_tmp0, v_tmp1);
    vfadd_vv(v_tmp2, v_tmp2, v_tmp3);
    vfadd_vv(v_tmp0, v_tmp0, v_tmp2);
    vfmv_v_f(v_in0, f_zero);
    vfredusum_vs(v_in1, v_tmp0, v_in0);
    vfmv_f_s(f_aux, v_in1);
    fadd_s(f_var_sum, f_var_sum, f_aux);

    L(var_done);
    ld(reg_len, reg_param, GET_FUSED_OFF(len));
    fcvt_s_l(f_len, reg_len);
    fdiv_s(f_var, f_var_sum, f_len);
    fmax_s(f_var, f_var, f_zero);

    ld(reg_mean_ptr, reg_param, GET_FUSED_OFF(mean));
    beqz(reg_mean_ptr, store_var);
    fsw(f_mean, reg_mean_ptr, 0);

    L(store_var);
    ld(reg_var_ptr, reg_param, GET_FUSED_OFF(variance));
    beqz(reg_var_ptr, start_data);
    fsw(f_var, reg_var_ptr, 0);

    L(start_data);
    flw(f_aux, reg_param, GET_FUSED_OFF(eps));
    fadd_s(f_aux, f_var, f_aux);
    fsqrt_s(f_inv, f_aux);
    li(reg_tmp, 0x3f800000);
    fmv_w_x(f_one, reg_tmp);
    fdiv_s(f_inv, f_one, f_inv);

    ld(reg_src, reg_param, GET_FUSED_OFF(src));
    ld(reg_dst, reg_param, GET_FUSED_OFF(dst));
    ld(reg_scale, reg_param, GET_FUSED_OFF(scale));
    ld(reg_shift, reg_param, GET_FUSED_OFF(shift));
    ld(reg_len, reg_param, GET_FUSED_OFF(len));

    vsetvli(reg_vlmax, x0, SEW::e32, LMUL::m1);
    slli(reg_bytes, reg_vlmax, 2);
    slli(reg_bytes4, reg_bytes, 2);
    slli(reg_block, reg_vlmax, 2);
    vfmv_v_f(v_mean, f_mean);
    vfmv_v_f(v_inv, f_inv);

    L(data_main_loop);
    blt(reg_len, reg_block, data_tail_loop);

    vle32_v(v_in0, reg_src);
    add(reg_tmp, reg_src, reg_bytes);
    vle32_v(v_in1, reg_tmp);
    add(reg_tmp, reg_tmp, reg_bytes);
    vle32_v(v_in2, reg_tmp);
    add(reg_tmp, reg_tmp, reg_bytes);
    vle32_v(v_in3, reg_tmp);

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
    j_(data_main_loop);

    L(data_tail_loop);
    beqz(reg_len, done);
    vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m1);
    slli(reg_tmp, reg_vl, 2);
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
    add(reg_src, reg_src, reg_tmp);
    add(reg_dst, reg_dst, reg_tmp);
    if (with_scale_) add(reg_scale, reg_scale, reg_tmp);
    if (with_shift_) add(reg_shift, reg_shift, reg_tmp);
    sub(reg_len, reg_len, reg_vl);
    j_(data_tail_loop);

    L(done);
    ret();
}

jit_rvv_layernorm_data_kernel_t::jit_rvv_layernorm_data_kernel_t(
        bool with_scale, bool with_shift)
    : jit_generator_t("jit_rvv_layernorm_data_kernel")
    , with_scale_(with_scale)
    , with_shift_(with_shift) {
    create_kernel();
}

void jit_rvv_layernorm_data_kernel_t::generate() {
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

    ld(reg_src, reg_param, GET_DATA_OFF(src));
    ld(reg_dst, reg_param, GET_DATA_OFF(dst));
    ld(reg_scale, reg_param, GET_DATA_OFF(scale));
    ld(reg_shift, reg_param, GET_DATA_OFF(shift));
    ld(reg_len, reg_param, GET_DATA_OFF(len));
    flw(f_mean, reg_param, GET_DATA_OFF(mean));
    flw(f_inv, reg_param, GET_DATA_OFF(inv_std));

    vsetvli(reg_vlmax, x0, SEW::e32, LMUL::m1);
    slli(reg_bytes, reg_vlmax, 2);
    slli(reg_bytes4, reg_bytes, 2);
    slli(reg_block, reg_vlmax, 2);

    Label main_loop, tail_loop, done;

    L(main_loop);
    blt(reg_len, reg_block, tail_loop);

    vfmv_v_f(v_mean, f_mean);
    vfmv_v_f(v_inv, f_inv);

    vle32_v(v_in0, reg_src);
    add(reg_tmp, reg_src, reg_bytes);
    vle32_v(v_in1, reg_tmp);
    add(reg_tmp, reg_tmp, reg_bytes);
    vle32_v(v_in2, reg_tmp);
    add(reg_tmp, reg_tmp, reg_bytes);
    vle32_v(v_in3, reg_tmp);

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
    slli(reg_tmp, reg_vl, 2);
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
    add(reg_src, reg_src, reg_tmp);
    add(reg_dst, reg_dst, reg_tmp);
    if (with_scale_) add(reg_scale, reg_scale, reg_tmp);
    if (with_shift_) add(reg_shift, reg_shift, reg_tmp);
    sub(reg_len, reg_len, reg_vl);
    j_(tail_loop);

    L(done);
    ret();
}

#undef GET_FUSED_OFF
#undef GET_DATA_OFF

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
