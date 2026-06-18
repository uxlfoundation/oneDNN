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

#ifndef CPU_RV64_JIT_UNI_REDUCTION_KERNEL_HPP
#define CPU_RV64_JIT_UNI_REDUCTION_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"

#include "cpu/rv64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct jit_reduction_conf_t {
    data_type_t src_type;
    data_type_t dst_type;
    alg_kind_t alg;
    size_t src_dt_size;
    size_t dst_dt_size;
    dim_t idle_size;
    dim_t reduce_size;
};

struct jit_uni_reduction_args_t {
    const void *src;
    void *dst;
    dim_t reduce_size;
};

struct jit_uni_reduction_kernel_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_reduction_kernel)

    explicit jit_uni_reduction_kernel_t(const jit_reduction_conf_t &conf);
    ~jit_uni_reduction_kernel_t() override = default;

private:
    void generate() override;

    void load_params();
    void load_f32_const(const Xbyak_riscv::FReg &freg, float value);
    void load_f16_const(const Xbyak_riscv::FReg &freg, uint16_t raw_value);
    void advance_src(int shift);

    enum class src_kind_t { f32, f16 };
    enum class acc_kind_t { f32, f16 };
    enum class scalar_kind_t { f32, f16 };
    enum class reduce_op_t { max, min, sum, mean };

    src_kind_t src_kind() const;
    scalar_kind_t dst_kind() const;
    reduce_op_t reduce_op() const;
    acc_kind_t acc_kind(src_kind_t src_kind) const;
    bool is_f16_widen_acc() const;

    void emit_reduce(src_kind_t src_kind);
    void emit_init(src_kind_t src_kind);
    void emit_loop(src_kind_t src_kind);
    void emit_finalize(src_kind_t src_kind);

    void emit_update_acc(src_kind_t src_kind);
    void emit_horizontal_reduce(src_kind_t src_kind);
    void emit_mean_scale_if_needed();
    void emit_store_scalar(scalar_kind_t scalar_kind);

    void emit_finalize_f16_minmax();
    void emit_finalize_f16_widen_sum_or_mean();

    const jit_reduction_conf_t conf_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
