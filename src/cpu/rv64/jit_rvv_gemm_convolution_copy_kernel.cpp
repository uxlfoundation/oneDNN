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

#include "cpu/rv64/jit_rvv_gemm_convolution_copy_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

#define GET_OFF(field) \
    static_cast<int32_t>(offsetof( \
            jit_rvv_gemm_convolution_copy_kernel_t::call_params_t, field))

namespace {

void dispatch_gemm_convolution_copy(
        const jit_rvv_gemm_convolution_copy_kernel_t::call_params_t *p) {
    static const jit_rvv_gemm_convolution_copy_kernel_t kernel;
    kernel(p);
}

} // namespace

jit_rvv_gemm_convolution_copy_kernel_t::
        jit_rvv_gemm_convolution_copy_kernel_t()
    : jit_generator_t("jit_rvv_gemm_convolution_copy_kernel") {
    create_kernel();
}

void jit_rvv_gemm_convolution_copy_f32(
        const float *src, float *dst, dim_t len) {
    const jit_rvv_gemm_convolution_copy_kernel_t::call_params_t p {
            src, dst, len};
    dispatch_gemm_convolution_copy(&p);
}

void jit_rvv_gemm_convolution_copy_kernel_t::generate() {
#if defined(XBYAK_RISCV_V) && XBYAK_RISCV_V == 1
    const Reg reg_param = a0;
    const Reg reg_src = a1;
    const Reg reg_dst = a2;
    const Reg reg_len = a3;
    const Reg reg_vl = t0;
    const Reg reg_bytes = t1;

    const VReg v_data(4);

    ld(reg_src, reg_param, GET_OFF(src));
    ld(reg_dst, reg_param, GET_OFF(dst));
    ld(reg_len, reg_param, GET_OFF(len));

    Label loop, done;
    L(loop);
    beqz(reg_len, done);

    vsetvli(reg_vl, reg_len, SEW::e32, LMUL::m4);
    vle32_v(v_data, reg_src);
    vse32_v(v_data, reg_dst);

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
