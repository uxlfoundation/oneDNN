/*******************************************************************************
* Copyright 2023 Intel Corporation
* Copyright 2025 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_MATMUL_JIT_BF16_MATMUL_HPP
#define CPU_AARCH64_MATMUL_JIT_BF16_MATMUL_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/primitive_attr_postops.hpp"

#include "cpu/matmul/cpu_matmul_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

struct jit_bf16_matmul_kernel_t;
struct brg_bf16_t {
    int M, K, N;
    const int m_blk = 8, k_blk = 4, n_blk = 4;
    const int bd_block = 8, rd_block = 4, ld_block = 6;
    int m_tail, n_tail, k_tail;
    int is_m_tail, is_k_tail, is_n_tail;
    int dst_dt_sz, bf16_dt_sz;
    bool is_bf16, fp_mathmode_is_bf16;
    bool with_sum_po, with_eltwise_po, with_binary_po, with_prelu_po;
};

struct jit_bf16_matmul_t : public primitive_t {
    struct pd_t : public dnnl::impl::cpu::matmul::cpu_matmul_pd_t {
        using ::dnnl::impl::cpu::matmul::cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("jit:bf16", jit_bf16_matmul_t);

        status_t init(engine_t *engine);

        bool formats_ok() const {

            const memory_desc_wrapper src_d(src_md_);
            const memory_desc_wrapper weights_d(weights_md_);
            const memory_desc_wrapper dst_d(dst_md_);
            const bool is_src = src_d.matches_one_of_tag(format_tag::AB2a4b)
                            != format_tag::undef
                    || src_d.format_kind() == format_kind::any;
            const bool is_wei = weights_d.matches_one_of_tag(format_tag::BA8b4a)
                            != format_tag::undef
                    || weights_d.format_kind() == format_kind::any;
            const bool is_dst = dst_d.matches_one_of_tag(format_tag::ab)
                            != format_tag::undef
                    || dst_d.format_kind() == format_kind::any;

            return is_dst && is_wei && is_src;
        }

        const brg_bf16_t &get_b() const { return brg; }

        int get_idx(int m, int k, int n, brg_bf16_t b) const {

            int mt = b.M % b.m_blk;
            int kt = b.K % (b.k_blk * b.rd_block);
            int nt = b.N % (b.n_blk * b.ld_block);
            if ((m == 1 && mt == 0) || (k == 1 && kt == 0)
                    || (n == 1 && nt == 0) || (k == 0 && kt == 1))
                return -1;
            return k + n * 2 + m * 2 * 2;
        }

    private:
        brg_bf16_t brg;
    };

    jit_bf16_matmul_t(const pd_t *apd);
    ~jit_bf16_matmul_t() override;
    int get_idx(int m, int k, int n, int M, int K, int N);
    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<jit_bf16_matmul_kernel_t> bf16_kernels_[8];
};

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif
