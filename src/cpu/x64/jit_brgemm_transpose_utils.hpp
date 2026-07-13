/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef CPU_X64_JIT_BRGEMM_TRANSPOSE_UTILS_HPP
#define CPU_X64_JIT_BRGEMM_TRANSPOSE_UTILS_HPP

#include "cpu/x64/jit_brgemm_primitive_conf.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_brgemm_trans_src_t {
    struct ctx_t {
        const void *src;
        const void *tr_src;

        dim_t current_gemm_batch;
        dim_t current_M, current_K;
    };

    virtual void operator()(const ctx_t *ctx) = 0;
    virtual status_t create_kernel() = 0;

    jit_brgemm_trans_src_t(const jit_brgemm_primitive_conf_t *conf)
        : conf_(conf) {}
    virtual ~jit_brgemm_trans_src_t() = default;

    const jit_brgemm_primitive_conf_t *conf_;
};

struct jit_brgemm_relo_copy_to_wbuffer_t : public jit_generator_t {
    struct cfg_t {
        data_type_t wei_dt {data_type_t::dnnl_data_type_undef};
        int out_oc_block {0};
        int inp_oc_block {0};
        int rd {0};
        bool is_rd_padded_to_block {false};
        int inp_ocb_offs {0};
        int last_occ_to_copy {0};
    };

    struct ctx_t {
        const char *src {nullptr};
        char *dst {nullptr};
        size_t last_ocb {0};
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_relo_copy_to_wbuffer_t)

    using reg64_t = Xbyak::Reg64;

    jit_brgemm_relo_copy_to_wbuffer_t(const cfg_t &ajcp)
        : jit_generator_t(jit_name(), avx512_core_amx), wjcp(ajcp) {}

private:
    cfg_t wjcp;

    const reg64_t reg_src = rax;
    const reg64_t reg_dst = rbx;
    const reg64_t aux_reg_src = r10;
    const reg64_t aux_reg_dst = r11;
    const reg64_t reg_tmp = rdx;

    const Xbyak::Opmask kmask_load = k2;

    const Xbyak::Zmm zmm_src = zmm0;
    const Xbyak::Zmm zmm_dst = zmm1;
    const Xbyak::Zmm zmm_zero = zmm2;
    const Xbyak::Zmm zmm_idx = zmm3;

    void generate() override;
};

status_t create_brgemm_trans_src(
        std::unique_ptr<jit_brgemm_trans_src_t> &trans_ker,
        const jit_brgemm_primitive_conf_t *conf);
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
