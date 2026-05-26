/*******************************************************************************
* Copyright 2026 ZTE Corporation
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

#include "cpu/rv64/jit_rvv_pooling_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

jit_rvv_pooling_fwd_kernel_t::jit_rvv_pooling_fwd_kernel_t(alg_t alg)
    : jit_generator_t(alg == alg_t::max_pool ? "jit_rvv_pooling_fwd_max"
                      : alg == alg_t::avg_include
                      ? "jit_rvv_pooling_fwd_avg_inc"
                      : "jit_rvv_pooling_fwd_avg_exc")
    , alg_(alg) {
    create_kernel();
}

void jit_rvv_pooling_fwd_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const VReg v_mask(0), v_acc(4), v_tmp(8);

    // Save callee-saved regs (s0-s7) to hold loop invariants across the
    // nested id/ih/iw/channel loops (src/dst ptrs, strides, range bounds)
    const int stack_size = 8 * 8;
    addi(sp, sp, -stack_size);
    sd(s0, sp, 0);
    sd(s1, sp, 8);
    sd(s2, sp, 16);
    sd(s3, sp, 24);
    sd(s4, sp, 32);
    sd(s5, sp, 40);
    sd(s6, sp, 48);
    sd(s7, sp, 56);

    // Load params from call_params_t
    using p_t = call_params_t;
    ld(s0, reg_param, static_cast<int>(offsetof(p_t, src)));
    ld(s1, reg_param, static_cast<int>(offsetof(p_t, dst)));
    ld(s2, reg_param, static_cast<int>(offsetof(p_t, channels)));
    ld(s6, reg_param, static_cast<int>(offsetof(p_t, id_start)));
    ld(s7, reg_param, static_cast<int>(offsetof(p_t, ih_start)));
    ld(a1, reg_param, static_cast<int>(offsetof(p_t, iw_start)));
    ld(t5, reg_param, static_cast<int>(offsetof(p_t, id_end)));
    ld(t6, reg_param, static_cast<int>(offsetof(p_t, ih_end)));
    ld(a2, reg_param, static_cast<int>(offsetof(p_t, iw_end)));
    ld(t2, reg_param, static_cast<int>(offsetof(p_t, inW_stride)));
    ld(t3, reg_param, static_cast<int>(offsetof(p_t, inD_stride)));
    flw(fa0, reg_param, static_cast<int>(offsetof(p_t, init_val)));
    flw(ft1, reg_param, static_cast<int>(offsetof(p_t, scale_val)));
    flw(fa2, reg_param, static_cast<int>(offsetof(p_t, relu_alpha)));

    fmv_w_x(fa1, x0); // f_zero = 0.0

    // Compute byte strides
    slli(s4, t2, 2); // inW_stride_bytes = inW_stride * 4
    slli(s5, t3, 2); // inD_stride_bytes = inD_stride * 4

    // Byte stride to advance one pixel in W: channels * sizeof(float)
    slli(s3, s2, 2); // W_spatial_byte_stride = channels * 4

    // Load with_relu flag (t3 is free after inD_stride consumed into s5)
    lbu(t3, reg_param, static_cast<int>(offsetof(p_t, with_relu)));

    // Channel loop: process channels in vector chunks
    Label ch_loop, ch_done;
    L(ch_loop);
    beqz(s2, ch_done);

    // Set vector length for remaining channels
    vsetvli(t0, s2, SEW::e32, LMUL::m1);

    // Initialize accumulator
    if (alg_ == alg_t::max_pool) {
        // init_val = -FLT_MAX for max pool
        vfmv_v_f(v_acc, fa0);
    } else {
        vfmv_v_f(v_acc, fa1); // zero
    }

    // Depth (id) loop
    mv(a3, s6); // reg_id = id_start
    mul(t1, a3, s5);
    add(t1, s0, t1); // t1 = depth_ptr = src + id * inD_stride_bytes

    Label id_loop, id_done;
    L(id_loop);
    bge(a3, t5, id_done); // if id >= id_end, done

    // Height (ih) loop
    mv(a4, s7); // reg_ih = ih_start
    mul(t2, a4, s4);
    add(a6, t1, t2); // a6 = row_ptr

    Label ih_loop, ih_done;
    L(ih_loop);
    bge(a4, t6, ih_done); // if ih >= ih_end, done

    // Width (iw) loop
    mv(a5, a1); // reg_iw = iw_start
    mul(t2, a5, s3);
    add(a7, a6, t2); // a7 = src_ptr

    Label iw_loop, iw_done;
    L(iw_loop);
    bge(a5, a2, iw_done); // if iw >= iw_end, done

    // Load and accumulate
    vle32_v(v_tmp, a7);
    if (alg_ == alg_t::max_pool) {
        vfmax_vv(v_acc, v_acc, v_tmp);
    } else {
        vfadd_vv(v_acc, v_acc, v_tmp);
    }

    addi(a5, a5, 1); // iw++
    add(a7, a7, s3); // advance src_ptr by channels * 4
    j_(iw_loop);
    L(iw_done);

    addi(a4, a4, 1); // ih++
    add(a6, a6, s4); // advance row_ptr by inW_stride_bytes
    j_(ih_loop);
    L(ih_done);

    addi(a3, a3, 1); // id++
    add(t1, t1, s5); // advance depth_ptr by inD_stride_bytes
    j_(id_loop);
    L(id_done);

    // Apply avg pooling divide
    if (alg_ != alg_t::max_pool) { vfmul_vf(v_acc, v_acc, ft1); }

    // Apply ReLU post-op (t3 = with_relu, loaded before channel loop)
    Label relu_done, relu_alpha_zero;
    beqz(t3, relu_done);

    // Check if alpha == 0
    fmv_w_x(ft0, x0);
    feq_s(t2, fa2, ft0);
    bnez(t2, relu_alpha_zero);
    // Alpha != 0: mask = v > 0; neg = v * alpha; merge
    vmfgt_vf(v_mask, v_acc, fa1);
    vfmul_vf(v_tmp, v_acc, fa2);
    vmerge_vvm(v_acc, v_tmp, v_acc);
    j_(relu_done);
    L(relu_alpha_zero);
    vfmax_vf(v_acc, v_acc, fa1);
    L(relu_done);
    // Store result
    vse32_v(v_acc, s1);

    // Advance src/dst pointers by vl * 4 bytes
    slli(t1, t0, 2);
    add(s0, s0, t1);
    add(s1, s1, t1);
    sub(s2, s2, t0); // channels -= vl

    j_(ch_loop);
    L(ch_done);

    // Restore callee-saved regs
    ld(s0, sp, 0);
    ld(s1, sp, 8);
    ld(s2, sp, 16);
    ld(s3, sp, 24);
    ld(s4, sp, 32);
    ld(s5, sp, 40);
    ld(s6, sp, 48);
    ld(s7, sp, 56);
    addi(sp, sp, stack_size);

    ret();
#else
    ret();
#endif
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
