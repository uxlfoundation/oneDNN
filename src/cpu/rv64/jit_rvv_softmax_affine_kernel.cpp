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

#include "cpu/rv64/jit_rvv_softmax_affine_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

jit_rvv_softmax_affine_kernel_t::jit_rvv_softmax_affine_kernel_t()
    : jit_generator_t("jit_rvv_softmax_affine_kernel") {
    create_kernel();
}

void jit_rvv_softmax_affine_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const Reg reg_src = a1;
    const Reg reg_dst = a2;
    const Reg reg_len = a3;
    const Reg reg_vl = t0;
    const Reg reg_bytes = t1;
    const Reg reg_tmp = t2;

    const FReg f_sub = fa0;
    const FReg f_mul = fa1;

    const VReg v_src(0);

    // call_params_t layout:
    //  0: src, 8: dst, 16: len, 24: sub, 28: mul
    ld(reg_src, reg_param, 0);
    ld(reg_dst, reg_param, 8);
    ld(reg_len, reg_param, 16);

    lw(reg_tmp, reg_param, 24);
    fmv_w_x(f_sub, reg_tmp);
    lw(reg_tmp, reg_param, 28);
    fmv_w_x(f_mul, reg_tmp);

    Label loop, done;
    L(loop);
    beqz(reg_len, done);
    vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m1);
    vle32_v(v_src, reg_src);
    vfsub_vf(v_src, v_src, f_sub);
    vfmul_vf(v_src, v_src, f_mul);
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
