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

#include "cpu/rv64/jit_rvv_inner_product_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

#define GET_OFF(field) \
    static_cast<int32_t>( \
            offsetof(jit_rvv_inner_product_kernel_t::call_params_t, field))

namespace {

template <jit_rvv_inner_product_kernel_t::dt_pair_t dt_pair>
float dispatch_jit_inner_product(
        const jit_rvv_inner_product_kernel_t::call_params_t *p) {
    static const jit_rvv_inner_product_kernel_t kernel(dt_pair);
    kernel(p);
    return *p->acc;
}

} // namespace

jit_rvv_inner_product_kernel_t::jit_rvv_inner_product_kernel_t(
        dt_pair_t dt_pair)
    : jit_generator_t("jit_rvv_inner_product_kernel"), dt_pair_(dt_pair) {
    create_kernel();
}

float jit_rvv_inner_product_fwd_s8_s8(
        const void *src, const void *weights, dim_t len) {
    float acc = 0.0f;
    const jit_rvv_inner_product_kernel_t::call_params_t p {
            src, weights, len, &acc};
    return dispatch_jit_inner_product<
            jit_rvv_inner_product_kernel_t::dt_pair_t::s8_s8>(&p);
}

float jit_rvv_inner_product_fwd_u8_s8(
        const void *src, const void *weights, dim_t len) {
    float acc = 0.0f;
    const jit_rvv_inner_product_kernel_t::call_params_t p {
            src, weights, len, &acc};
    return dispatch_jit_inner_product<
            jit_rvv_inner_product_kernel_t::dt_pair_t::u8_s8>(&p);
}

void jit_rvv_inner_product_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    switch (dt_pair_) {
        case dt_pair_t::s8_s8: generate_x8_s8(false); break;
        case dt_pair_t::u8_s8: generate_x8_s8(true); break;
    }
#else
    ret();
#endif
}

void jit_rvv_inner_product_kernel_t::generate_x8_s8(bool src_is_u8) {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const Reg reg_src = a1;
    const Reg reg_wei = a2;
    const Reg reg_len = a3;
    const Reg reg_acc_ptr = a4;
    const Reg reg_vl = t0;
    const Reg reg_part = t1;
    const Reg reg_acc = t2;

    const FReg f_acc = fa0;

    const VReg v_src(4);
    const VReg v_wei(5);
    const VReg v_prod(8);
    const VReg v_zero(12);
    const VReg v_sum(13);

    ld(reg_src, reg_param, GET_OFF(src));
    ld(reg_wei, reg_param, GET_OFF(weights));
    ld(reg_len, reg_param, GET_OFF(len));
    ld(reg_acc_ptr, reg_param, GET_OFF(acc));
    li(reg_acc, 0);

    Label loop, done;
    L(loop);
    beqz(reg_len, done);

    vsetvli(reg_vl, reg_len, SEW::e8, LMUL::m1, VTA::ta, VMA::ma);
    vle8_v(v_src, reg_src);
    vle8_v(v_wei, reg_wei);
    if (src_is_u8)
        vwmulsu_vv(v_prod, v_wei, v_src);
    else
        vwmul_vv(v_prod, v_src, v_wei);

    vsetvli(reg_vl, reg_vl, SEW::e16, LMUL::m2, VTA::ta, VMA::ma);
    vmv_v_x(v_zero, x0);
    vwredsum_vs(v_sum, v_prod, v_zero);
    vmv_x_s(reg_part, v_sum);
    add(reg_acc, reg_acc, reg_part);

    add(reg_src, reg_src, reg_vl);
    add(reg_wei, reg_wei, reg_vl);
    sub(reg_len, reg_len, reg_vl);
    j_(loop);

    L(done);
    fcvt_s_w(f_acc, reg_acc);
    fsw(f_acc, reg_acc_ptr, 0);
    ret();
#else
    ret();
#endif
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
