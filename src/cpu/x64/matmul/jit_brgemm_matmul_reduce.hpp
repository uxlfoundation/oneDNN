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

#ifndef CPU_X64_MATMUL_JIT_BRGEMM_MATMUL_REDUCE_HPP
#define CPU_X64_MATMUL_JIT_BRGEMM_MATMUL_REDUCE_HPP

#include "common/c_types_map.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"
#include "cpu/x64/matmul/brgemm_matmul_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

struct brgemm_kernel_reduce_args_t {
    brgemm_kernel_reduce_args_t()
        : ptr_in(nullptr), ptr_acc(nullptr), ptr_out(nullptr), flags(0) {};

    void *ptr_in;
    void *ptr_acc;
    void *ptr_out;
    int flags;
};

template <typename Vmm>
struct jit_brgemm_kernel_reduce_t : public jit_generator_t {
    jit_brgemm_kernel_reduce_t(
            const brgemm_matmul_conf_t &bgmmc, const brgemm_desc_t &abrg);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_kernel_reduce_t)

private:
    brgemm_desc_t brg_;
    matmul_reduce_kind_t reduce_kind_;
    data_type_t in_dt_;
    data_type_t out_dt_;
    data_type_t acc_dt_;

    int in_typesize_;
    int out_typesize_;
    int acc_typesize_;

    using Vmm_lower_t = typename vreg_traits_t<Vmm>::Vmm_lower_t;
    using reg64_t = const Xbyak::Reg64;
    // Register decomposition
    const reg64_t param1 = abi_param1;
    const reg64_t reg_in = r15;
    const reg64_t reg_out = r14;
    const reg64_t reg_acc = r13;
    const reg64_t aux_reg_in = r12;
    const reg64_t reg_k_iter = r11;
    const reg64_t reg_flag = r10;
    const reg64_t reg_mask = rax;

    Xbyak::Label f16_perm_table_;
    Xbyak::Label mask_label_;
    Xbyak::Opmask k_full_mask = Xbyak::Opmask(2);
    Xbyak::Opmask k_tail_mask = Xbyak::Opmask(3);
    Xbyak::Opmask k_f16_perm_mask = Xbyak::Opmask(4);
    Xbyak::Opmask k_store_mask = Xbyak::Opmask(5);
    Vmm vreg_unit = Vmm(31);
    Vmm vreg_perm = Vmm(30);
    Vmm vmm_tail_mask = Vmm(15); // use for avx tail loads

    const int n_max_regs_ = 4;

    Vmm vmm_mask(const Vmm vmm_in, bool mask_flag, bool store,
            Xbyak::Opmask ktail_mask);
    Vmm get_acc_reg(int n) const { return Vmm(n); }
    Vmm_lower_t get_acc_reg_lower(int n) const { return Vmm_lower_t(n); }
    Vmm get_in_reg(int n) const { return Vmm(n + n_max_regs_); }
    Vmm get_workspace_reg() const {
        assert(reduce_kind_ == matmul_reduce_kind::src);
        return Vmm(1);
    }

    void accumulate(bool mask_flag);
    void loop_by_K();
    void init_masks(int tail_length);
    void generate() override;

    void generate_reduce();
};

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
