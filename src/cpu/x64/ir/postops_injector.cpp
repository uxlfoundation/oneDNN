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

#include "common/memory_desc_wrapper.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/ir/postops_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

template <typename Vmm>
using injector_base_t = injector::jit_uni_postops_injector_base_t<Vmm>;

namespace {

// Create an injector for the target ISA and return it type-erased.
template <typename Vmm>
std::shared_ptr<void> create_injector(jit_generator_t *host, cpu_isa_t isa,
        const post_ops_t &post_ops,
        const binary_injector::static_params_t &bsp) {
    return std::shared_ptr<injector_base_t<Vmm>>(
            injector_base_t<Vmm>::create(host, isa, post_ops, bsp));
}

template <typename Vmm>
injector_base_t<Vmm> *cast2tgt(void *injector) {
    return static_cast<injector_base_t<Vmm> *>(injector);
}

} // namespace

postops_injector_t::postops_injector_t(jit_generator_t *host, cpu_isa_t isa,
        const post_ops_t &post_ops, const memory_desc_t &dst_md,
        const Xbyak::Reg64 &param_reg)
    : is_zmm_(is_superset(isa, avx512_core)) {
    const memory_desc_wrapper dst_d(dst_md);

    // The injector preserves the state of the gpr and vec registers it borrows.
    static constexpr bool preserve_gpr = true;
    static constexpr bool preserve_vmm = true;
    static constexpr bool use_exact_tail_scalar_bcast = false;

    // Right-hand-side argument config for binary post-ops. Eltwise does not use
    // it.
    // TODO: pass the rhs-arg offset and tail size in when binary support lands.
    const std::size_t rhs_dt_helper_vmm_idx = 0;
    const std::size_t rhs_arg_offset = 0;
    const std::size_t dst_orig_offset = 0;
    const std::size_t tail_size = 0;

    const binary_injector::rhs_arg_static_params_t rhs_sp {
            rhs_dt_helper_vmm_idx, host->r14, host->r15, host->r13,
            preserve_gpr, preserve_vmm, rhs_arg_offset, dst_orig_offset, dst_d,
            tail_size, use_exact_tail_scalar_bcast};

    const binary_injector::static_params_t bsp {param_reg,
            binary_injector::get_all_strategies_supported_by_injector(),
            rhs_sp};

    injector_ = is_zmm_ ? create_injector<Xbyak::Zmm>(gen, isa, post_ops, bsp)
                        : create_injector<Xbyak::Ymm>(gen, isa, post_ops, bsp);
    assert(injector_ && "ir post-ops injector creation failed");

    const auto eltwise_ind = post_ops.find(primitive_kind::eltwise);
    with_eltwise_ = eltwise_ind != -1;
}

void postops_injector_t::apply(const std::vector<int> &acc_phys) {
    injector_utils::vmm_index_set_t vmm_idxs;
    for (int idx : acc_phys)
        vmm_idxs.insert((size_t)idx);

    if (is_zmm_)
        cast2tgt<Xbyak::Zmm>(injector_.get())->compute_vector_range(vmm_idxs);
    else
        cast2tgt<Xbyak::Ymm>(injector_.get())->compute_vector_range(vmm_idxs);
}

void postops_injector_t::maybe_prepare_table() {
    // The constant table is only needed for eltwise.
    if (!with_eltwise_) return;

    const bool generate = true;
    if (is_zmm_)
        cast2tgt<Xbyak::Zmm>(injector_.get())->prepare_table(generate);
    else
        cast2tgt<Xbyak::Ymm>(injector_.get())->prepare_table(generate);
}

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
