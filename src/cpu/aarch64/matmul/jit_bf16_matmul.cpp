/*******************************************************************************
* Copyright 2023 Intel Corporation
* Copyright 2025 FUJITSU LIMITED
* Copyright 2025 Arm Ltd. and affiliates
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

#include <cassert>

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/tag_traits.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/cpu_primitive.hpp"
#include "cpu/matmul/matmul_utils.hpp"
#include "cpu/scale_utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/primitive_attr_postops.hpp"

#include "cpu/aarch64/matmul/jit_bf16_matmul.hpp"

#define GET_OFF(field) (uint32_t) offsetof(call_params_t, field)

#define LDR_IMM(reg, addr, off) \
    { \
        const uint64_t IMM12_MASK = ~uint64_t(0xfff); \
        if (((off) & IMM12_MASK) == 0) { \
            ldr(reg, ptr(addr, off)); \
        } else { \
            add_imm(X_DEFAULT_ADDR, addr, off, X_TMP_0); \
            ldr(reg, ptr(X_DEFAULT_ADDR)); \
        } \
    }

#define VCHECK_BG(f, msg, ...) \
    VCHECK(primitive, create, dispatch, brgemm_matmul, f, msg, ##__VA_ARGS__);

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

using namespace Xbyak_aarch64;
using namespace dnnl::impl::cpu::matmul;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;

using namespace data_type;

struct jit_bf16_matmul_kernel_t : public jit_generator_t {

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_bf16_matmul_kernel_t)

    struct call_params_t {
        const uint16_t *src, *wei;
        float *dst;
        dim_t M, K, N;
    };

    XReg reg_param = abi_param1;
    XReg reg_a = x3;
    XReg reg_b = x4;
    XReg reg_c = x5;
    XReg reg_aux_a = x6;
    XReg reg_aux_b = x7;
    XReg reg_aux_c = x8;
    XReg reg_rd_loop = x9;
    XReg reg_tmp = x10;
    XReg reg_tmp_1 = x11;
    PReg prd_st = p1;

    call_params_t inp;

    void operator()(const call_params_t *p) {
        return jit_generator_t::operator()(p);
    }

    ZReg loadb(int ld) { return ZReg(ld + 1); }

    ZReg acc(int bd, int ld) const {
        return ZReg(bd * brg.ld_block + ld + brg.ld_block + 1);
    }

    void zero_regs() {
        for (int bd = 0; bd < brg.bd_block / 2; bd++)
            for (int ld = 0; ld < brg.ld_block; ld++)
                eor(acc(bd, ld).d, acc(bd, ld).d, acc(bd, ld).d);
    }

    void store_regs(int bdb, int blb) {
        for (int bd = 0; bd < bdb; bd++) {
            for (int ld = 0; ld < ldb; ld += 2) {
                if (ld + 1 < ldb) {
                    uzp1(z31.d, acc(bd, ld).d, acc(bd, ld + 1).d);
                    uzp2(acc(bd, ld + 1).d, acc(bd, ld).d, acc(bd, ld + 1).d);
                    mov(acc(bd, ld).d, z31.d);
                } else {
                    uzp1(z31.d, acc(bd, ld).d, acc(bd, ld).d);
                    uzp2(acc(bd, ld + 1).d, acc(bd, ld).d, acc(bd, ld).d);
                    mov(acc(bd, ld).d, z31.d);
                }
            }
        }

        mov(reg_tmp, reg_aux_c);
        add_imm(reg_tmp_1, reg_aux_c, brg.N * brg.dst_dt_sz, X_TMP_0);

        for (int bd = 0; bd < bdb; bd++) {
            for (int ld = 0; ld < ldb; ld += 2) {
                PReg p = (brg.is_n_tail && ld >= ldb - 2) ? prd_st : P_ALL_ONE;
                int vl = ld / 2;
                if (brg.with_sum_po) {
                    ld1w(z0.s, p, ptr(reg_tmp, vl, MUL_VL));
                    fadd(acc(bd, ld).s, z0.s, acc(bd, ld).s);
                }
                st1w(acc(bd, ld).s, p, ptr(reg_tmp, vl, MUL_VL));
                if (bd >= bdb - 1 && brg.is_m_tail) {
                    if (brg.m_tail % 2 == 0) {
                        if (brg.with_sum_po) {
                            ld1w(z0.s, p, ptr(reg_tmp_1, vl, MUL_VL));
                            fadd(acc(bd, ld + 1).s, z0.s, acc(bd, ld + 1).s);
                        }
                        st1w(acc(bd, ld + 1).s, p, ptr(reg_tmp_1, vl, MUL_VL));
                    }
                } else {
                    if (brg.with_sum_po) {
                        ld1w(z0.s, p, ptr(reg_tmp_1, vl, MUL_VL));
                        fadd(acc(bd, ld + 1).s, z0.s, acc(bd, ld + 1).s);
                    }
                    st1w(acc(bd, ld + 1).s, p, ptr(reg_tmp_1, vl, MUL_VL));
                }
            }
            add_imm(reg_tmp, reg_tmp, 2 * brg.N * brg.dst_dt_sz, X_TMP_0);
            add_imm(reg_tmp_1, reg_tmp_1, 2 * brg.N * brg.dst_dt_sz, X_TMP_0);
        }
    }

    void microkernel(int rdb, int bdb, int ldb) {
        int a_off = 0, rd, ld, bd;
        mov(reg_tmp, reg_aux_b);
        for (rd = 0; rd < rdb; rd++) {
            for (ld = 0; ld < ldb; ld += 2) {
                ld1h(loadb(ld).h, P_ALL_ONE, ptr(reg_tmp, 0, MUL_VL));
                ld1h(loadb(ld + 1).h, P_ALL_ONE, ptr(reg_tmp, 1, MUL_VL));
                add_imm(reg_tmp, reg_tmp,
                        div_up(brg.K, brg.k_blk) * brg.k_blk * brg.k_blk * 2
                                * brg.bf16_dt_sz,
                        X_TMP_0);
            }
            for (bd = 0; bd < bdb; bd++) {
                add_imm(X_DEFAULT_ADDR, reg_aux_a,
                        a_off
                                + ((div_up(brg.K, brg.k_blk) * brg.k_blk * 2
                                           * brg.bf16_dt_sz)
                                        * bd),
                        X_TMP_0);
                ld1rqh(z0.h, P_ALL_ONE, ptr(X_DEFAULT_ADDR));

                for (ld = 0; ld < ldb; ld++) {
                    bfmmla(acc(bd, ld).s, z0.h, loadb(ld).h);
                }
            }
            a_off += brg.k_blk * 2 * brg.bf16_dt_sz;
            add_imm(reg_tmp, reg_aux_b,
                    brg.k_blk * brg.n_blk * 2 * brg.bf16_dt_sz * (rd + 1),
                    X_TMP_0);
        }
    }

    void loop_k(int bdb, int ldb) {
        zero_regs();
        if (k_full_blks > 0) {
            mov(reg_rd_loop, k_full_blks);
            Label l0;
            L(l0);
            microkernel(brg.rd_block, bdb, ldb);
            add_imm(reg_aux_a, reg_aux_a,
                    brg.k_blk * brg.rd_block * 2 * brg.bf16_dt_sz, X_TMP_0);
            add_imm(reg_aux_b, reg_aux_b,
                    brg.k_blk * brg.rd_block * brg.n_blk * 2 * brg.bf16_dt_sz,
                    X_TMP_0);
            sub(reg_rd_loop, reg_rd_loop, 1);
            cmp(reg_rd_loop, 0);
            b(GT, l0);
        }

        if (k_tail_blks > 0) {
            microkernel(k_tail_blks, bdb, ldb);
            add_imm(reg_aux_a, reg_aux_a,
                    brg.k_blk * k_tail_blks * 2 * brg.bf16_dt_sz, X_TMP_0);
            add_imm(reg_aux_b, reg_aux_b,
                    brg.k_blk * k_tail_blks * brg.n_blk * 2 * brg.bf16_dt_sz,
                    X_TMP_0);
        }

        if (k_residual_blk > 0) { microkernel(1, bdb, ldb); }

        store_regs(bdb, ldb);
    }

    void config() {
        int pred_st = 0, sv_len = 8;
        k_full_blks = brg.K / (brg.k_blk * brg.rd_block);
        k_tail_blks = (brg.K % (brg.k_blk * brg.rd_block)) / brg.k_blk;
        k_residual_blk = (brg.K % (brg.k_blk * brg.rd_block)) % brg.k_blk;
        ldb = (brg.is_n_tail) ? div_up(brg.n_tail, 4) : brg.ld_block;
        bdb = (brg.is_m_tail) ? div_up(brg.m_tail, 2) : brg.bd_block / 2;

        if (brg.is_n_tail) {
            if (brg.n_tail % brg.n_blk == 0) {
                pred_st = (brg.n_tail % (brg.n_blk * 2) == 0) ? sv_len
                                                              : sv_len / 2;
            } else {
                pred_st = (ldb % 2 == 0)
                        ? (sv_len / 2) + (brg.n_tail % brg.n_blk)
                        : (brg.n_tail % brg.n_blk);
            }
        }
        set_preg(prd_st.s, pred_st, X_TMP_0, X_TMP_1);
    }

    void generate() override {
        preamble();
        config();

        LDR_IMM(reg_a, reg_param, GET_OFF(src));
        LDR_IMM(reg_b, reg_param, GET_OFF(wei));
        LDR_IMM(reg_c, reg_param, GET_OFF(dst));

        mov(reg_aux_a, reg_a);
        mov(reg_aux_b, reg_b);
        mov(reg_aux_c, reg_c);
        loop_k(bdb, ldb);

        postamble();
    }

    jit_bf16_matmul_kernel_t(
            const dnnl::impl::cpu::aarch64::matmul::brg_bf16_t &k)
        : brg(k) {}
    ~jit_bf16_matmul_kernel_t() override = default;

    dnnl::impl::cpu::aarch64::matmul::brg_bf16_t brg;

    int ldb;
    int bdb;
    int rdb;
    int k_full_blks;
    int k_tail_blks;
    int k_residual_blk;
};

status_t jit_bf16_matmul_t::pd_t::init(engine_t *engine) {
    const memory_desc_wrapper src_d(src_md_);
    const memory_desc_wrapper weights_d(weights_md_);
    const memory_desc_wrapper dst_d(dst_md_);

    const bool no_runtime_dims_or_strides
            = !(src_d.has_runtime_dims_or_strides()
                    || weights_d.has_runtime_dims_or_strides());

    VDISPATCH_MATMUL(
            no_runtime_dims_or_strides, VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    const auto src_type = src_md(0)->data_type;
    const auto wei_type = weights_md(0)->data_type;
    const auto dst_type = dst_md(0)->data_type;

    bool is_bf16 = utils::everyone_is(bf16, src_type, wei_type);
    const bool dt_correct = (is_bf16) && utils::everyone_is(f32, dst_type);

    VDISPATCH_MATMUL(dt_correct, VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_MATMUL(attr()->has_default_values(
                             primitive_attr_t::skip_mask_t::post_ops, dst_type),
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_MATMUL(formats_ok(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_MATMUL(mayiuse(sve_256), VERBOSE_UNSUPPORTED_ISA);
    VDISPATCH_MATMUL(!with_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);

    int dims = src_d.ndims();

    switch (dims) {
        case 2: {
            VCHECK_BG(memory_desc_init_by_tag(src_md_, format_tag::AB2a4b),
                    VERBOSE_UNSUPPORTED_TAG);
            VCHECK_BG(memory_desc_init_by_tag(weights_md_, format_tag::BA8b4a),
                    VERBOSE_UNSUPPORTED_TAG);
            VCHECK_BG(memory_desc_init_by_tag(dst_md_, format_tag::ab),
                    VERBOSE_UNSUPPORTED_TAG);
            break;
        }
        default: return status::unimplemented;
    }

    const auto &post_ops = attr()->post_ops_;
    const auto sum_idx = post_ops.find(primitive_kind::sum);
    brg.with_sum_po = sum_idx != -1;
    const auto eltwise_idx = post_ops.find(primitive_kind::eltwise);
    brg.with_eltwise_po = eltwise_idx != -1;
    const auto binary_idx = post_ops.find(primitive_kind::binary);
    brg.with_binary_po = binary_idx != -1;
    const auto prelu_idx = post_ops.find(primitive_kind::prelu);
    brg.with_prelu_po = prelu_idx != -1;

    //currently, only sum post-op is supported
    if (brg.with_eltwise_po || brg.with_binary_po || brg.with_prelu_po)
        return status::unimplemented;

    if (brg.with_sum_po) {
        const auto &sum_po = post_ops.entry_[sum_idx].sum;
        if (sum_po.scale != 1.0f || sum_po.zero_point != 0) {
            return status::unimplemented;
        }
    }

    matmul_helper_t helper(src_d, weights_d, dst_d);
    brg.K = helper.K();
    brg.M = helper.M();
    brg.N = helper.N();
    brg.dst_dt_sz = 4;
    brg.bf16_dt_sz = 2;
    brg.m_tail = brg.M % brg.m_blk;
    brg.k_tail = brg.K % (brg.k_blk * brg.rd_block);
    brg.n_tail = brg.N % (brg.n_blk * brg.ld_block);

    return status::success;
}

status_t jit_bf16_matmul_t::init(engine_t *engine) {

    auto &b1 = pd()->get_b();

    dnnl::impl::cpu::aarch64::matmul::brg_bf16_t b;
    b.M = b1.M;
    b.K = b1.K;
    b.N = b1.N;
    b.m_tail = b1.m_tail;
    b.n_tail = b1.n_tail;
    b.k_tail = b1.k_tail;
    b.dst_dt_sz = b1.dst_dt_sz;
    b.bf16_dt_sz = b1.bf16_dt_sz;
    b.is_bf16 = b1.is_bf16;
    b.with_sum_po = b1.with_sum_po;
    b.with_eltwise_po = b1.with_eltwise_po;
    b.with_binary_po = b1.with_binary_po;
    b.with_prelu_po = b1.with_prelu_po;

    for (int m = 0; m < 2; m++)
        for (int n = 0; n < 2; n++)
            for (int k = 0; k < 2; k++) {
                int idx = pd()->get_idx(m, k, n, b);
                if (idx == -1) continue;
                b.is_m_tail = m;
                b.is_k_tail = k;
                b.is_n_tail = n;
                bf16_kernels_[idx] = std::unique_ptr<jit_bf16_matmul_kernel_t> {
                        new jit_bf16_matmul_kernel_t(b)};
                if (!bf16_kernels_[idx]) return status::runtime_error;
                CHECK(bf16_kernels_[idx]->create_kernel());
            }

    return status::success;
}

jit_bf16_matmul_t::jit_bf16_matmul_t(const pd_t *apd) : primitive_t(apd) {}
jit_bf16_matmul_t::~jit_bf16_matmul_t() = default;

status_t jit_bf16_matmul_t::execute(const exec_ctx_t &ctx) const {
    const auto *weights = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
    const auto *src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    int dims = src_d.ndims();

    const dim_t M = dst_d.dims()[dims - 2];
    const dim_t N = dst_d.dims()[dims - 1];
    const dim_t K = src_d.dims()[dims - 1];

    auto &b = pd()->get_b();

    int num_threads = dnnl_get_current_num_threads();
    int num_a_blocks = div_up(M, b.m_blk);
    int num_b_blocks = div_up(N, (b.n_blk * b.ld_block));
    int ktail = (b.k_tail == 0) ? 0 : 1;
    int parallel_work = num_a_blocks * num_b_blocks;

    parallel(num_threads, [&](const int ithr, const int nthr) {
        int start {0}, end {0};
        balance211(parallel_work, num_threads, ithr, start, end);

        while (start < end) {

            int m_block = start % num_a_blocks;
            int n_block = start / num_a_blocks;
            int mtail = (b.m_tail != 0 && m_block == num_a_blocks - 1) ? 1 : 0;
            int ntail = (b.n_tail != 0 && n_block == num_b_blocks - 1) ? 1 : 0;
            int dst_adr = (m_block * b.m_blk * N
                    + n_block * (b.n_blk * b.ld_block));
            int m_blk_adr = m_block * b.m_blk * div_up(K, b.k_blk) * b.k_blk;
            int n_blk_adr = n_block * (b.n_blk * b.ld_block)
                    * div_up(K, b.k_blk) * b.k_blk;

            int idx = pd()->get_idx(mtail, ktail, ntail, b);

            jit_bf16_matmul_kernel_t::call_params_t p;
            p.src = (uint16_t *)src + m_blk_adr;
            p.wei = (uint16_t *)weights + n_blk_adr;
            p.dst = dst + dst_adr;
            p.M = M;
            p.N = N;
            p.K = K;
            (*bf16_kernels_[idx])(&p);
            start++;
        }
    });

    return status::success;
}

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
