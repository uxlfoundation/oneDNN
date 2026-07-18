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
std::shared_ptr<void> create_injector(jit_generator_t &gen, cpu_isa_t isa,
        const post_ops_t &post_ops,
        const binary_injector::static_params_t &bsp) {
    return std::shared_ptr<injector_base_t<Vmm>>(
            injector_base_t<Vmm>::create(&gen, isa, post_ops, bsp));
}

template <typename Vmm>
injector_base_t<Vmm> *cast2tgt(void *injector) {
    return static_cast<injector_base_t<Vmm> *>(injector);
}

} // namespace

postops_injector_t::postops_injector_t(jit_generator_t &gen, cpu_isa_t isa,
        const post_ops_t &post_ops, const memory_desc_t &dst_md,
        const Xbyak::Reg64 &param_reg, size_t rhs_arg_offset,
        size_t dst_orig_off, int tail_elems)
    : is_zmm_(is_superset(isa, avx512_core)), tail_elems_(tail_elems) {
    const memory_desc_wrapper dst_d(dst_md);

    // The injector preserves the state of the gpr and vec registers it borrows.
    static constexpr bool preserve_gpr = true;
    static constexpr bool preserve_vmm = true;
    static constexpr bool use_exact_tail_scalar_bcast = false;

    const size_t rhs_dt_helper_vmm_idx = 0;

    // Right-hand-side argument config for binary post-ops. Eltwise does not use
    // it. `rhs_arg_offset` locates the argument pointer array in the parameter
    // struct. `dst_orig_off` locates the destination origin, used to turn an
    // accumulator address into its destination position. `tail_elems` is the
    // element count a partial load reads on avx2. The avx512 opmask path is not
    // enabled yet.
    const binary_injector::rhs_arg_static_params_t rhs_sp {
            rhs_dt_helper_vmm_idx, gen.r14, gen.r15, gen.r13, preserve_gpr,
            preserve_vmm, rhs_arg_offset, dst_orig_off, dst_d,
            (size_t)tail_elems, use_exact_tail_scalar_bcast};

    const binary_injector::static_params_t bsp {param_reg,
            binary_injector::get_all_strategies_supported_by_injector(),
            rhs_sp};

    injector_ = is_zmm_ ? create_injector<Xbyak::Zmm>(gen, isa, post_ops, bsp)
                        : create_injector<Xbyak::Ymm>(gen, isa, post_ops, bsp);
    assert(injector_ && "ir post-ops injector creation failed");

    with_eltwise_ = post_ops.find(primitive_kind::eltwise) != -1;
    // Prelu is injected through the binary injector and needs the same
    // right-hand-side arguments, so treat it like a binary post-op.
    with_binary_ = post_ops.find(primitive_kind::binary) != -1
            || post_ops.find(primitive_kind::prelu) != -1;
}

void postops_injector_t::apply(const std::vector<int> &acc_phys, int base_phys,
        const std::vector<dim_t> &out_byte_off, int mask_phys, int elems) {
    injector_utils::vmm_index_set_t vmm_idxs;
    for (int idx : acc_phys)
        vmm_idxs.insert((size_t)idx);

    if (!with_binary_) {
        // Eltwise-only chains have no right-hand-side arguments to address.
        if (is_zmm_)
            cast2tgt<Xbyak::Zmm>(injector_.get())
                    ->compute_vector_range(vmm_idxs);
        else
            cast2tgt<Xbyak::Ymm>(injector_.get())
                    ->compute_vector_range(vmm_idxs);
        return;
    }

    // `mask_phys` is a K register (avx512 opmask). The avx512 emitter is not
    // enabled yet, so it is currently unused.
    MAYBE_UNUSED(mask_phys);

    // `elems == -1` is a full vector. A positive count is a partial load of
    // `elems` right-hand-side elements and must match `tail_elems_`.
    JIT_ASSERT((elems == -1 || elems > 0)
            && "inject_postops: elems must be -1 or a positive count");
    const bool is_tail = elems > 0;
    JIT_ASSERT((!is_tail || elems == tail_elems_)
            && "inject_postops: tail count does not match the injector");

    // Map each accumulator to its destination address so the injector locates
    // the matching right-hand-side slice.
    binary_injector::rhs_arg_dynamic_params_t rhs_args;
    const Xbyak::Reg64 out_reg(base_phys);
    for (size_t i = 0; i < acc_phys.size(); i++) {
        rhs_args.vmm_idx_to_out_reg.emplace(acc_phys[i], out_reg);
        rhs_args.vmm_idx_to_out_elem_off_val.emplace(
                acc_phys[i], (size_t)out_byte_off[i]);
        if (is_tail) rhs_args.vmm_tail_idx_.emplace(acc_phys[i]);
    }

    if (is_zmm_)
        cast2tgt<Xbyak::Zmm>(injector_.get())
                ->compute_vector_range(vmm_idxs, rhs_args);
    else
        cast2tgt<Xbyak::Ymm>(injector_.get())
                ->compute_vector_range(vmm_idxs, rhs_args);
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
