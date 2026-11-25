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
#include "common/utils.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/ir/postops_injector.hpp"
#include "cpu/x64/jit_avx512_core_fp8cvt.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

template <typename Vmm>
using injector_t = injector::jit_uni_postops_injector_t<Vmm>;

namespace {

// Create an injector and return it type-erased. The ISA it generates for comes
// from the host generator.
template <typename Vmm>
std::shared_ptr<void> create_injector(jit_generator_t &gen,
        const post_ops_t &post_ops, const binary_injector::static_params_t &bsp,
        const eltwise_injector::static_params_t &esp) {
    return std::make_shared<injector_t<Vmm>>(
            &gen, post_ops, bsp, esp, /* inject_sum = */ true);
}

template <typename Vmm>
injector_t<Vmm> *cast2tgt(void *injector) {
    return static_cast<injector_t<Vmm> *>(injector);
}

// Whether a binary post-op of `post_ops` reads a right-hand side of type `dt`.
bool has_binary_rhs_of(const post_ops_t &post_ops, data_type_t dt) {
    for (int i = 0; i < post_ops.len(); i++) {
        const auto &e = post_ops.entry_[i];
        if (e.is_binary() && e.binary.src1_desc.data_type == dt) return true;
    }
    return false;
}

} // namespace

bool postops_injector_t::needs_fp8_cvt(const post_ops_t &post_ops) {
    return has_binary_rhs_of(post_ops, data_type::f8_e5m2)
            || has_binary_rhs_of(post_ops, data_type::f8_e4m3);
}

postops_injector_t::fp8_cvt_regs_t postops_injector_t::fp8_cvt_regs(
        cpu_isa_t isa) {
    // The binary injector takes an fp8 right-hand side on AVX-512 only (see
    // `binary_injector::is_data_type_supported()`).
    assert(is_superset(isa, avx512_core));

    fp8_cvt_regs_t regs;
    // The e4m3 converter takes five scratch vectors, the e5m2 one three of
    // them. The sixth is the binary injector's right-hand-side vector: the
    // injector picks a free register for it from the top of the file when
    // its hint is taken, so a fixed one keeps it clear of the scratch. The
    // highest register is left to the converting stores (see
    // `make_reg_config()`).
    const int n_vec = isa_num_vregs(isa);
    for (int i = n_vec - 7; i < n_vec - 1; i++)
        regs.vecs.push_back(i);
    // `k6` and `k7` belong to the eltwise and binary injectors.
    regs.opmask = 5;
    // The binary injector borrows `r13` to `r15`, `rax`, `rdx`, `r8` and `r9`
    // around the right-hand-side load that runs the converters, so the table
    // pointer has to be another register.
    regs.gpr = Xbyak::Operand::R12;
    return regs;
}

postops_injector_t::~postops_injector_t() = default;

postops_injector_t::postops_injector_t(jit_generator_t &gen, cpu_isa_t isa,
        const post_ops_t &post_ops, const memory_desc_t &dst_md,
        const bcast_set_t &bcast_set, const Xbyak::Reg64 &param_reg,
        int rhs_arg_offset, dim_t dst_orig_off, int tail_elems,
        int eltwise_opmask, int binary_tail_opmask,
        const fp8_cvt_regs_t &fp8_regs)
    : is_zmm_(is_superset(isa, avx512_core))
    , tail_elems_(tail_elems)
    , binary_tail_opmask_(binary_tail_opmask) {
    const memory_desc_wrapper dst_d(dst_md);

    // The eltwise injector overwrites its mask and the binary injector reads
    // its own across the same call, so sharing one register would destroy the
    // tail pattern.
    JIT_ASSERT(eltwise_opmask != binary_tail_opmask
            && "postops_injector: eltwise and binary opmask must differ");
    // `init()` builds the tail pattern with a 64-bit shift.
    JIT_ASSERT(tail_elems < 64 && "postops_injector: tail is too wide");

    // The injector preserves the state of the gpr and vec registers it borrows.
    static constexpr bool preserve_gpr = true;
    static constexpr bool preserve_vmm = true;
    static constexpr bool use_exact_tail_scalar_bcast = false;

    // The fp8 converters of the right-hand sides the chain reads. Each takes
    // its scratch from `fp8_regs` (see `fp8_cvt_regs_t`).
    int rhs_dt_helper_vmm_idx = 0;
    if (needs_fp8_cvt(post_ops)) {
        JIT_ASSERT(is_zmm_ && fp8_regs.vecs.size() == 6 && fp8_regs.opmask >= 0
                && fp8_regs.gpr >= 0
                && "postops_injector: fp8 right-hand side needs its registers");
        const auto &v = fp8_regs.vecs;
        const Xbyak::Reg64 gpr(fp8_regs.gpr);
        if (has_binary_rhs_of(post_ops, data_type::f8_e5m2))
            f8_e5m2_cvt_ = utils::make_unique<fp8_conversion_e5m2_t>(&gen,
                    Xbyak::Xmm(v[0]), Xbyak::Xmm(v[1]), Xbyak::Xmm(v[2]),
                    Xbyak::Opmask(fp8_regs.opmask), gpr);
        if (has_binary_rhs_of(post_ops, data_type::f8_e4m3))
            f8_e4m3_cvt_ = utils::make_unique<fp8_conversion_e4m3_t>(&gen,
                    Xbyak::Xmm(v[0]), Xbyak::Xmm(v[1]), Xbyak::Xmm(v[2]),
                    Xbyak::Xmm(v[3]), Xbyak::Xmm(v[4]), gpr);
        rhs_dt_helper_vmm_idx = v[5];
    }

    // Right-hand-side argument config, used by binary post-ops and, for the
    // load path only, by sum. Eltwise does not use it.
    // `rhs_arg_offset` locates the argument pointer array in the parameter
    // struct. `dst_orig_off` locates the destination origin, used to turn an
    // accumulator address into its destination position. `tail_elems` is the
    // element count a partial load reads, as an integer on avx2 and through
    // `binary_tail_opmask` on avx512.
    const binary_injector::rhs_arg_static_params_t rhs_sp {
            rhs_dt_helper_vmm_idx, gen.r14, gen.r15, gen.r13, preserve_gpr,
            preserve_vmm, rhs_arg_offset, dst_orig_off, dst_d, tail_elems,
            Xbyak::Opmask(binary_tail_opmask), use_exact_tail_scalar_bcast};

    const binary_injector::static_params_t bsp {param_reg, bcast_set, rhs_sp,
            f8_e5m2_cvt_.get(), f8_e4m3_cvt_.get()};

    // Everything but the mask keeps the eltwise injector's own defaults. The
    // mask has to be named because its default, `k1`, is a register the IR
    // allocator hands out.
    const eltwise_injector::static_params_t esp {/*save_state=*/true,
            /*p_table=*/Xbyak::Reg64(Xbyak::Operand::RAX),
            Xbyak::Opmask(eltwise_opmask)};

    injector_ = is_zmm_ ? create_injector<Xbyak::Zmm>(gen, post_ops, bsp, esp)
                        : create_injector<Xbyak::Ymm>(gen, post_ops, bsp, esp);
    assert(injector_ && "ir post-ops injector creation failed");

    with_eltwise_ = post_ops.find(primitive_kind::eltwise) != -1;
    with_sum_ = post_ops.find(primitive_kind::sum) != -1;
    // Prelu is injected through the binary injector and needs the same
    // right-hand-side arguments, so treat it like a binary post-op. The sum
    // reads the previous destination value through those same arguments.
    needs_rhs_args_ = post_ops.find(primitive_kind::binary) != -1
            || post_ops.find(primitive_kind::prelu) != -1 || with_sum_;
}

void postops_injector_t::init(jit_generator_t &gen) const {
    // The tail opmask is read only by the binary injector. An eltwise-only
    // chain never reaches it, and on avx2* the tail is an integer inside the
    // injector, so neither case has anything to set up.
    if (!is_zmm_ || !needs_rhs_args_ || tail_elems_ <= 0) return;

    // `kxnorq` sets all 64 bits and `kshiftrq` keeps the low `tail_elems_` of
    // them. Building the mask in the k-register file needs no gpr.
    const Xbyak::Opmask k(binary_tail_opmask_);
    gen.kxnorq(k, k, k);
    gen.kshiftrq(k, k, (uint8_t)(64 - tail_elems_));
}

void postops_injector_t::inject(const std::vector<int> &acc_phys, int base_phys,
        const std::vector<dim_t> &out_byte_off, int n_elems) {
    injector_utils::vmm_index_set_t vmm_idxs;
    for (int idx : acc_phys)
        vmm_idxs.insert(idx);

    if (!needs_rhs_args_) {
        // Chains without a binary, prelu or sum post-op have nothing to address
        // relative to the destination.
        if (is_zmm_)
            cast2tgt<Xbyak::Zmm>(injector_.get())
                    ->compute_vector_range(vmm_idxs);
        else
            cast2tgt<Xbyak::Ymm>(injector_.get())
                    ->compute_vector_range(vmm_idxs);
        return;
    }

    // An accumulator holding the tail count has fewer valid elements than a
    // full vector, so a right-hand-side load has to stop at that count to stay
    // in bounds. `init()` has put that pattern in the tail opmask.
    const bool is_tail = tail_elems_ > 0 && n_elems == tail_elems_;

    // Map each accumulator to its destination address. A binary post-op uses it
    // to locate the corresponding right-hand-side slice, and the sum operation
    // uses it to access the previous destination value.
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
    // The constant table is needed for eltwise and sum.
    if (with_eltwise_ || with_sum_) {
        const bool generate = true;
        if (is_zmm_)
            cast2tgt<Xbyak::Zmm>(injector_.get())->prepare_table(generate);
        else
            cast2tgt<Xbyak::Ymm>(injector_.get())->prepare_table(generate);
    }

    if (f8_e5m2_cvt_) f8_e5m2_cvt_->prepare_table();
    if (f8_e4m3_cvt_) f8_e4m3_cvt_->prepare_table();
}

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
