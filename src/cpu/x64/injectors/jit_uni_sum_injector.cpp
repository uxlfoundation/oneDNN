/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "common/utils.hpp"

#include "cpu/x64/injectors/jit_uni_sum_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, typename Vmm>
jit_uni_sum_injector_t<isa, Vmm>::jit_uni_sum_injector_t(jit_generator_t *host,
        const post_ops_t::entry_t::sum_t &sum, data_type_t dst_dt,
        binary_injector::jit_uni_binary_injector_t<Vmm> *binary_injector,
        size_t scratch_vmm_idx, bool preserve_scratch_vmm)
    : host_(host)
    , binary_injector_(binary_injector)
    , sum_dt_(sum.dt == data_type::undef ? dst_dt : sum.dt)
    , scale_val_(sum.scale)
    , zp_val_((float)sum.zero_point)
    , has_scale_(scale_val_ != 1.f)
    , has_zp_(sum.zero_point != 0)
    , scratch_vmm_idx_(scratch_vmm_idx)
    , preserve_scratch_vmm_(preserve_scratch_vmm) {

    // The inputs are borrowed from the binary injector's static params so
    // we want to check that the caller populated them for real use. An undef
    // read type means `dst_d` was left unset.
    assert(binary_injector_ && "native sum requires the binary injector");
    assert(sum_dt_ != data_type::undef
            && "native sum read type is undef: binary static params unset");
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_sum_injector_t<isa, Vmm>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs,
        const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params) {

    assert(binary_injector_ && "native sum needs the binary injector");
    if (vmm_idxs.empty()) return;

    // If a user-provided scratch register happens to be one of the
    // accumulators, pick a new free register.
    const int max_idx = cpu_isa_traits_t<isa>::n_vregs - 1;
    const int scratch = (int)scratch_vmm_idx_;

    int tmp_idx = scratch;
    bool scratch_changed = false;

    if (vmm_idxs.find((size_t)scratch) != vmm_idxs.end()) {
        for (int idx = max_idx; idx >= 0; idx--) {
            if (vmm_idxs.find((size_t)idx) == vmm_idxs.end()) {
                // Found a free register.
                tmp_idx = idx;
                scratch_changed = true;
                break;
            }
        }
    }

    assert(vmm_idxs.find((size_t)tmp_idx) == vmm_idxs.end()
            && "native sum could not find a free scratch vector register");

    // Preserve when the caller asked to or whenever we picked a new scratch.
    // The new scratch register is not the caller's designated scratch so its
    // prior contents must be saved regardless of `preserve_scratch_vmm_`.
    const bool preserve_tmp = preserve_scratch_vmm_ || scratch_changed;
    const injector_utils::conditional_register_preserve_guard_t tmp_guard {
            preserve_tmp, host_, {}, {Vmm(tmp_idx)}};

    const Vmm prev(tmp_idx);

    for (const auto vmm_idx : vmm_idxs) {
        const int idx = (int)vmm_idx;
        const Vmm acc(idx);

        const auto reg_it = rhs_arg_params.vmm_idx_to_out_reg.find(idx);
        assert(reg_it != rhs_arg_params.vmm_idx_to_out_reg.end()
                && "native sum needs the destination base-address register per "
                   "accumulator");

        const auto off_it
                = rhs_arg_params.vmm_idx_to_out_elem_off_val.find(idx);
        const size_t off
                = off_it != rhs_arg_params.vmm_idx_to_out_elem_off_val.end()
                ? off_it->second
                : 0;

        const bool with_tail = rhs_arg_params.vmm_tail_idx_.count(idx) > 0;

        binary_injector_->load_acc_as_f32(
                prev, reg_it->second, off, sum_dt_, with_tail);

        if (has_zp_)
            host_->uni_vsubps(prev, prev, host_->ptr[host_->rip + zp_lbl_]);

        if (has_scale_) {
            host_->uni_vfmadd231ps(
                    acc, prev, host_->ptr[host_->rip + scale_lbl_]);
        } else
            host_->uni_vaddps(acc, acc, prev);
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_sum_injector_t<isa, Vmm>::prepare_table(bool gen_table) {
    if (!gen_table || (!has_scale_ && !has_zp_)) return;

    const int simd = cpu_isa_traits_t<isa>::vlen / (int)sizeof(float);
    const auto store_value = [&](float val) {
        for (int i = 0; i < simd; i++)
            host_->dd(float2int(val));
    };
    host_->align(64);
    if (has_scale_) {
        host_->L(scale_lbl_);
        store_value(scale_val_);
    }
    if (has_zp_) {
        host_->L(zp_lbl_);
        store_value(zp_val_);
    }
}

template class jit_uni_sum_injector_t<avx512_core_fp16, Xbyak::Zmm>;
template class jit_uni_sum_injector_t<avx512_core_fp16, Xbyak::Ymm>;
template class jit_uni_sum_injector_t<avx512_core_fp16, Xbyak::Xmm>;
template class jit_uni_sum_injector_t<avx512_core_bf16, Xbyak::Zmm>;
template class jit_uni_sum_injector_t<avx512_core, Xbyak::Zmm>;
template class jit_uni_sum_injector_t<avx512_core, Xbyak::Ymm>;
template class jit_uni_sum_injector_t<avx512_core, Xbyak::Xmm>;
template class jit_uni_sum_injector_t<avx2_vnni_2, Xbyak::Ymm>;
template class jit_uni_sum_injector_t<avx2_vnni_2, Xbyak::Xmm>;
template class jit_uni_sum_injector_t<avx2, Xbyak::Ymm>;
template class jit_uni_sum_injector_t<avx2, Xbyak::Xmm>;
template class jit_uni_sum_injector_t<avx, Xbyak::Ymm>;
template class jit_uni_sum_injector_t<avx, Xbyak::Xmm>;
template class jit_uni_sum_injector_t<sse41, Xbyak::Xmm>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
