/*******************************************************************************
* Copyright 2020 Intel Corporation
* Copyright 2022 FUJITSU LIMITED
* Copyright 2025-2026 Arm Ltd. and affiliates
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
#include <numeric>
#include <vector>
#include "cpu/aarch64/injectors/injector_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace injector_utils {

template <cpu_isa_t isa>
register_preserve_guard_t<isa>::register_preserve_guard_t(jit_generator_t *host,
        std::initializer_list<Xbyak_aarch64::XReg> reg64_to_preserve,
        std::initializer_list<Xbyak_aarch64::VReg> vmm_to_preserve)
    : host_(host)
    , gpr_regs_(reg64_to_preserve)
    , vec_regs_(vmm_to_preserve)
    , vmm_to_preserve_size_bytes_(
              calc_vmm_to_preserve_size_bytes(vmm_to_preserve)) {

    if (reg64_to_preserve.size() != 0) {
        const auto store_pairs = reg64_to_preserve.size() / 2;
        const bool has_lone_store = reg64_to_preserve.size() % 2;
        uint32_t i = 0;
        for (; i < store_pairs; ++i)
            host_->stp(gpr_regs_[2 * i], gpr_regs_[2 * i + 1],
                    pre_ptr(host_->X_SP, -16));

        if (has_lone_store) {
            // The hardware requires the stack pointer to be quad-word aligned
            // https://github.com/ARM-software/abi-aa/blob/e2534ac15f02fa2e03b7f336f069f7cf392257d9/aapcs64/aapcs64.rst?#645the-stack
            host_->str(gpr_regs_[2 * i], pre_ptr(host_->X_SP, -16));
        }
    }

    if (!vec_regs_.empty()) {
        host_->sub(host_->X_SP, host_->X_SP, vmm_to_preserve_size_bytes_);

        uint32_t stack_offset = vmm_to_preserve_size_bytes_;
        for (const auto &vmm : vmm_to_preserve) {
            const uint32_t vlen_bytes = simd_bytes(isa);
            stack_offset -= vlen_bytes;
            const auto idx = vmm.getIdx();
            if (vlen_bytes > 16) {
                if (stack_offset % vlen_bytes == 0) {
                    host_->st1w(Xbyak_aarch64::ZRegS(idx), host_->P_ALL_ONE,
                            ptr(host_->X_SP, stack_offset / vlen_bytes,
                                    Xbyak_aarch64::MUL_VL));
                } else {
                    host_->add_imm(host_->X_DEFAULT_ADDR, host_->X_SP,
                            stack_offset, host_->X_TMP_0);
                    host_->st1w(Xbyak_aarch64::ZRegS(idx), host_->P_ALL_ONE,
                            ptr(host_->X_DEFAULT_ADDR));
                }
            } else {
                host_->str(Xbyak_aarch64::QReg(idx),
                        ptr(host_->X_SP, stack_offset));
            }
        }
    }
}

template <cpu_isa_t isa>
register_preserve_guard_t<isa>::~register_preserve_guard_t() {

    uint32_t tmp_stack_offset = 0;

    while (!vec_regs_.empty()) {
        const Xbyak_aarch64::VReg &vmm = vec_regs_.back();
        const uint32_t vlen_bytes = simd_bytes(isa);
        const auto idx = vmm.getIdx();
        if (vlen_bytes > 16) {
            if (tmp_stack_offset % vlen_bytes == 0) {
                host_->ld1w(Xbyak_aarch64::ZRegS(idx), host_->P_ALL_ONE,
                        ptr(host_->X_SP, tmp_stack_offset / vlen_bytes,
                                Xbyak_aarch64::MUL_VL));
            } else {
                host_->add_imm(host_->X_DEFAULT_ADDR, host_->X_SP,
                        tmp_stack_offset, host_->X_TMP_0);
                host_->ld1w(Xbyak_aarch64::ZRegS(idx), host_->P_ALL_ONE,
                        ptr(host_->X_SP, host_->X_DEFAULT_ADDR));
            }
        } else {
            host_->ldr(Xbyak_aarch64::QReg(idx),
                    ptr(host_->X_SP, tmp_stack_offset));
        }

        tmp_stack_offset += vlen_bytes;
        vec_regs_.pop_back();
    }

    if (vmm_to_preserve_size_bytes_)
        host_->add_imm(host_->X_SP, host_->X_SP, vmm_to_preserve_size_bytes_,
                host_->X_TMP_0);

    if (!gpr_regs_.empty()) {
        const bool has_lone_store = gpr_regs_.size() % 2;

        if (has_lone_store) {
            host_->ldr(gpr_regs_.back(), post_ptr(host_->X_SP, 16));
            gpr_regs_.pop_back();
        }

        for (int i = gpr_regs_.size() - 1; i > 0; --i) {
            host_->ldp(
                    gpr_regs_[i - 1], gpr_regs_[i], post_ptr(host_->X_SP, 16));
        }
    }
}

template <cpu_isa_t isa>
size_t register_preserve_guard_t<isa>::calc_vmm_to_preserve_size_bytes(
        const std::initializer_list<Xbyak_aarch64::VReg> &vmm_to_preserve)
        const {

    return std::accumulate(vmm_to_preserve.begin(), vmm_to_preserve.end(),
            std::size_t(0u),
            [](std::size_t accum, const Xbyak_aarch64::VReg &vmm) {
        return accum + simd_bytes(isa);
    });
}

template <cpu_isa_t isa>
size_t register_preserve_guard_t<isa>::stack_space_occupied() const {
    constexpr static size_t reg64_size = 8;
    const size_t stack_space_occupied
            = vmm_to_preserve_size_bytes_ + gpr_regs_.size() * reg64_size;

    return stack_space_occupied;
}

template <cpu_isa_t isa>
conditional_register_preserve_guard_t<
        isa>::conditional_register_preserve_guard_t(bool condition_to_be_met,
        jit_generator_t *host,
        std::initializer_list<Xbyak_aarch64::XReg> reg64_to_preserve,
        std::initializer_list<Xbyak_aarch64::VReg> vmm_to_preserve)
    : register_preserve_guard_t<isa> {condition_to_be_met
                      ? register_preserve_guard_t<isa> {host, reg64_to_preserve,
                                vmm_to_preserve}
                      : register_preserve_guard_t<isa> {nullptr, {}, {}}} {};

template class register_preserve_guard_t<sve>;
template class conditional_register_preserve_guard_t<sve>;

} // namespace injector_utils
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
