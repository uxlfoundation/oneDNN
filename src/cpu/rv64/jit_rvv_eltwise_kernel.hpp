/*******************************************************************************
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
* Copyright 2026 openKylin community
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

#ifndef CPU_RV64_JIT_RVV_ELTWISE_KERNEL_HPP
#define CPU_RV64_JIT_RVV_ELTWISE_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "cpu/rv64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct jit_rvv_eltwise_fwd_kernel_t : public jit_generator_t {
    struct call_params_t {
        const float *src;
        float *dst;
        dim_t len;
        float alpha;
        float beta;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_eltwise_fwd_kernel_t)

    explicit jit_rvv_eltwise_fwd_kernel_t(alg_kind_t alg);

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    void compute_vector(const Xbyak_riscv::VReg &v_dst,
            const Xbyak_riscv::VReg &v_src, const Xbyak_riscv::VReg &v_tmp,
            const Xbyak_riscv::FReg &f_alpha, const Xbyak_riscv::FReg &f_beta,
            const Xbyak_riscv::FReg &f_zero, const Xbyak_riscv::FReg &f_one);

    alg_kind_t alg_;
};

bool jit_rvv_eltwise_f32_supported(alg_kind_t alg);

void jit_rvv_eltwise_apply_f32(alg_kind_t alg, const float *src, float *dst,
        dim_t len, float alpha, float beta);

// f16 eltwise forward kernel (widen-narrow: load f16 → widen to f32 →
// reuse the f32 compute → narrow on store; keeps alpha/beta in f32 so
// hardsigmoid/hardswish clamp boundaries match the reference bit-exactly).
struct jit_rvv_eltwise_fwd_kernel_f16_t : public jit_generator_t {
    struct call_params_t {
        const void *src;
        void *dst;
        dim_t len;
        float alpha;
        float beta;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_eltwise_fwd_kernel_f16_t)

    explicit jit_rvv_eltwise_fwd_kernel_f16_t(alg_kind_t alg);

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    void compute_vector(const Xbyak_riscv::VReg &v_dst,
            const Xbyak_riscv::VReg &v_src, const Xbyak_riscv::VReg &v_tmp,
            const Xbyak_riscv::FReg &f_alpha, const Xbyak_riscv::FReg &f_beta,
            const Xbyak_riscv::FReg &f_zero, const Xbyak_riscv::FReg &f_one);

    alg_kind_t alg_;
};

bool jit_rvv_eltwise_fwd_f16_supported(alg_kind_t alg);

void jit_rvv_eltwise_apply_fwd_f16(alg_kind_t alg, const void *src, void *dst,
        dim_t len, float alpha, float beta);

// f16 eltwise backward kernel (widen-narrow). The bwd direction had no JIT
// scaffold previously; this class adds one f16-only — the f32 bwd path
// stays on the existing intrinsic implementation.
struct jit_rvv_eltwise_bwd_kernel_f16_t : public jit_generator_t {
    struct call_params_t {
        void *diff_src;
        const void *diff_dst;
        const void *src;
        dim_t len;
        float alpha;
        float beta;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_eltwise_bwd_kernel_f16_t)

    explicit jit_rvv_eltwise_bwd_kernel_f16_t(alg_kind_t alg);

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    void compute_vector(const Xbyak_riscv::VReg &v_dst,
            const Xbyak_riscv::VReg &v_dd, const Xbyak_riscv::VReg &v_s,
            const Xbyak_riscv::VReg &v_tmp, const Xbyak_riscv::FReg &f_alpha,
            const Xbyak_riscv::FReg &f_beta, const Xbyak_riscv::FReg &f_zero,
            const Xbyak_riscv::FReg &f_one);

    alg_kind_t alg_;
};

bool jit_rvv_eltwise_bwd_f16_supported(alg_kind_t alg);

void jit_rvv_eltwise_apply_bwd_f16(alg_kind_t alg, void *diff_src,
        const void *diff_dst, const void *src, dim_t len, float alpha,
        float beta);

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_JIT_RVV_ELTWISE_KERNEL_HPP
