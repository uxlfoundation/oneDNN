/******************************************************************************
* Copyright 2023 Intel Corporation
* Copyright 2025 ISCAS
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

#include "cpu/rv64/jit_rvv_nchw_pooling.hpp"
#include "common/dnnl_thread.hpp"
#include "xbyak_riscv/xbyak_riscv.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;

struct jit_pool_call_s {
    const void *src;
    void *dst;
    size_t ow;
    size_t stride_w_bytes;
    size_t kw;
    size_t dt_size;
    float init_val;
    float div_val;
};

struct jit_rvv_nchw_pool_kernel_t : public CodeGenerator {
    bool is_fp16;
    bool is_max;
    bool is_avg;

    // ABI Register Mapping (RISC-V standard convention)
    const Reg reg_src = a0;
    const Reg reg_dst = a1;
    const Reg reg_ow = a2;
    const Reg reg_stride_bytes = a3;
    const Reg reg_kw = a4;
    const Reg reg_dt_size = a5;
    const FReg freg_init_val = fa0;
    const FReg freg_div_val = fa1;

    // Temporary Registers
    const Reg reg_tmp_src = t0;
    const Reg reg_tmp_kw = t1;
    const Reg reg_vl = t2;
    const Reg reg_tmp_stride = t3;
    const Reg reg_offset = t4;

    // Vector Registers
    const VReg v_acc = v0;
    const VReg v_src = v2;

    jit_rvv_nchw_pool_kernel_t(const jit_rvv_nchw_pooling_fwd_t::pd_t *pd)
        : CodeGenerator() {

        is_fp16 = pd->src_md()->data_type == data_type::f16;
        is_max = pd->desc()->alg_kind == alg_kind::pooling_max;
        is_avg = !is_max;

        generate();
    }

    void generate() {
        SEW sew = is_fp16 ? SEW::e16 : SEW::e32;
        LMUL lmul = LMUL::m1;

        Label l_ow_loop;

        L(l_ow_loop);
        {
            vsetvli(reg_vl, reg_ow, sew, lmul, VTA::ta, VMA::ma);

            vfmv_v_f(v_acc, freg_init_val);

            mv(reg_tmp_src, reg_src);
            mv(reg_tmp_kw, reg_kw);

            Label l_kw_loop;
            L(l_kw_loop);
            {
                if (is_fp16) {
                    vlse16_v(v_src, reg_tmp_src, reg_stride_bytes);
                } else {
                    vlse32_v(v_src, reg_tmp_src, reg_stride_bytes);
                }

                if (is_max) {
                    vfmax_vv(v_acc, v_acc, v_src);
                } else {
                    vfadd_vv(v_acc, v_acc, v_src);
                }

                add(reg_tmp_src, reg_tmp_src, reg_dt_size);

                addi(reg_tmp_kw, reg_tmp_kw, -1);
                bnez(reg_tmp_kw, l_kw_loop);
            }

            if (is_avg) { vfmul_vf(v_acc, v_acc, freg_div_val); }

            if (is_fp16) {
                vse16_v(v_acc, reg_dst);
                slli(reg_tmp_stride, reg_vl, 1); // Bytes processed: vl * 2
            } else {
                vse32_v(v_acc, reg_dst);
                slli(reg_tmp_stride, reg_vl, 2); // Bytes processed: vl * 4
            }

            add(reg_dst, reg_dst, reg_tmp_stride);

            mul(reg_tmp_stride, reg_vl, reg_stride_bytes);
            add(reg_src, reg_src, reg_tmp_stride);

            sub(reg_ow, reg_ow, reg_vl);
            bnez(reg_ow, l_ow_loop);
        }

        ret();
    }
};

jit_rvv_nchw_pooling_fwd_t::jit_rvv_nchw_pooling_fwd_t(const pd_t *apd)
    : primitive_t(apd) {
    kernel_ = std::unique_ptr<jit_rvv_nchw_pool_kernel_t>(
            new jit_rvv_nchw_pool_kernel_t(apd));
    kernel_->ready(); // Finalize code generation and set execute permissions
}

jit_rvv_nchw_pooling_fwd_t::~jit_rvv_nchw_pooling_fwd_t() = default;

status_t jit_rvv_nchw_pooling_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const int mb = pd()->MB();
    const int c = pd()->OC();
    const int od = pd()->OD();
    const int oh = pd()->OH();
    const int ow = pd()->OW();

    const int stride_w = pd()->KSW();
    const int kw = pd()->KW();

    const bool is_fp16 = pd()->src_md()->data_type == data_type::f16;
    const int data_size = is_fp16 ? 2 : 4;

    float init_val = 0.0f;
    if (pd()->desc()->alg_kind == alg_kind::pooling_max) {
        if (is_fp16) {
            init_val = -std::numeric_limits<float>::infinity();
        } else {
            init_val = -std::numeric_limits<float>::infinity();
        }
    }

    float div_val = 1.0f / (pd()->KD() * pd()->KH() * pd()->KW());

    auto ker = kernel_->getCode<void (*)(const void *, void *, size_t, size_t,
            size_t, size_t, float, float)>();

    parallel_nd(mb, c, od, oh, [&](int n, int ch, int d, int h) {
        auto dst_off = dst_d.blk_off(n, ch, d, h, 0) * data_size;

        int id_start = d * pd()->KSD() - pd()->padFront();
        int ih_start = h * pd()->KSH() - pd()->padT();
        int iw_start_base = -pd()->padL();

        const char *src_base = src + src_d.blk_off(n, ch, 0, 0, 0) * data_size;
        char *dst_ptr = dst + dst_off;

        for (int kd = 0; kd < pd()->KD(); ++kd) {
            int id = id_start + kd;
            if (id < 0 || id >= pd()->ID()) continue;

            for (int kh = 0; kh < pd()->KH(); ++kh) {
                int ih = ih_start + kh;
                if (ih < 0 || ih >= pd()->IH()) continue;

                size_t src_offset
                        = (id * pd()->IH() * pd()->IW() + ih * pd()->IW())
                        * data_size;
                const char *src_row
                        = src_base + src_offset + (iw_start_base * data_size);

                ker((const void *)src_row, (void *)dst_ptr, ow,
                        stride_w * data_size, kw, data_size, init_val, div_val);

                goto finish_kernel;
            }
        }
    finish_kernel:;
    });

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
