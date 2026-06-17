/*******************************************************************************
* Copyright 2026 SpacemiT Corporation
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

#ifndef CPU_RV64_JIT_RVV_DWCONV_KERNEL_HPP
#define CPU_RV64_JIT_RVV_DWCONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/float16.hpp"
#include "cpu/rv64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct jit_rvv_dwconv_kernel_t : public jit_generator_t {
    struct call_params_t {
        const float16_t *lhs;
        dim_t lhs_stride_0;
        dim_t lhs_stride_1;
        const float16_t *rhs;
        dim_t rhs_stride_0;
        dim_t rhs_stride_1;
        float16_t *out;
        dim_t out_stride_0;
        dim_t out_stride_1;
        dim_t h;
        dim_t w;
        dim_t c;
        dim_t ratio_bytes;
        const float *bias;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rvv_dwconv_kernel_t)

    jit_rvv_dwconv_kernel_t(int stride);

    void operator()(const call_params_t *p) const {
        jit_generator_t::operator()(p);
    }

protected:
    void generate() override;

private:
    void preload_dwconv3x3s1_f16(const Xbyak_riscv::Reg &r0,
            const Xbyak_riscv::Reg &r1, const Xbyak_riscv::Reg &r2,
            const Xbyak_riscv::Reg &lhs_stride_1);
    void compute_dwconv3x3s1_f16_m5(const Xbyak_riscv::Reg &r0,
            const Xbyak_riscv::Reg &r1, const Xbyak_riscv::Reg &r2,
            const Xbyak_riscv::Reg &lhs_stride_1);
    void compute_dwconv3x3s2_f16_m5(const Xbyak_riscv::Reg &r0,
            const Xbyak_riscv::Reg &r1, const Xbyak_riscv::Reg &r2,
            const Xbyak_riscv::Reg &lhs_stride_1);
    void compute_dwconv3x3s2_f16_m(const Xbyak_riscv::Reg &r0,
            const Xbyak_riscv::Reg &r1, const Xbyak_riscv::Reg &r2,
            const Xbyak_riscv::Reg &lhs_stride_1, int count);
    void add_bias_m(const Xbyak_riscv::Reg &vl, int count);
    void narrow_m(int count);
    void store_m(const Xbyak_riscv::Reg &out,
            const Xbyak_riscv::Reg &out_stride_1,
            const Xbyak_riscv::Reg &ratio_bytes, int count);
    void compute_one_output(int dst_idx, int src_start);
    void load_tail_extra_cols(const Xbyak_riscv::Reg &r0,
            const Xbyak_riscv::Reg &r1, const Xbyak_riscv::Reg &r2,
            const Xbyak_riscv::Reg &lhs_stride_1, int cols);
    void compute_tail(const Xbyak_riscv::Reg &r0, const Xbyak_riscv::Reg &r1,
            const Xbyak_riscv::Reg &r2, const Xbyak_riscv::Reg &lhs_stride_1,
            const Xbyak_riscv::Reg &vl, const Xbyak_riscv::Reg &out,
            const Xbyak_riscv::Reg &out_stride_1,
            const Xbyak_riscv::Reg &ratio_bytes, int count);
    void compute_tail_s2(const Xbyak_riscv::Reg &r0, const Xbyak_riscv::Reg &r1,
            const Xbyak_riscv::Reg &r2, const Xbyak_riscv::Reg &lhs_stride_1,
            const Xbyak_riscv::Reg &vl, const Xbyak_riscv::Reg &out,
            const Xbyak_riscv::Reg &out_stride_1,
            const Xbyak_riscv::Reg &ratio_bytes, int count);

    const int stride_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_JIT_RVV_DWCONV_KERNEL_HPP
