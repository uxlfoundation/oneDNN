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
#include <numeric>
#include "common/broadcast_strategy.hpp"
#include "cpu/x64/injectors/injector_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace injector_utils {

static std::size_t get_vmm_size_bytes(const Xbyak::Xmm &vmm) {
    static constexpr int byte_size_bits = 8;
    return vmm.getBit() / byte_size_bits;
}

static std::size_t calc_vmm_to_preserve_size_bytes(
        const std::initializer_list<Xbyak::Xmm> &vmm_to_preserve) {

    return std::accumulate(vmm_to_preserve.begin(), vmm_to_preserve.end(),
            std::size_t(0u), [](std::size_t accum, const Xbyak::Xmm &vmm) {
        return accum + get_vmm_size_bytes(vmm);
    });
}

register_preserve_guard_t::register_preserve_guard_t(jit_generator_t *host,
        std::initializer_list<Xbyak::Reg64> reg64_to_preserve,
        std::initializer_list<Xbyak::Xmm> vmm_to_preserve)
    : host_(host)
    , reg64_stack_(reg64_to_preserve)
    , vmm_stack_(vmm_to_preserve)
    , vmm_to_preserve_size_bytes_(
              calc_vmm_to_preserve_size_bytes(vmm_to_preserve)) {

    for (const auto &reg : reg64_to_preserve)
        host_->push(reg);

    if (!vmm_stack_.empty()) {
        host_->sub(host_->rsp, vmm_to_preserve_size_bytes_);

        auto stack_offset = vmm_to_preserve_size_bytes_;
        for (const auto &vmm : vmm_to_preserve) {
            stack_offset -= get_vmm_size_bytes(vmm);
            const auto idx = vmm.getIdx();
            if (vmm.isXMM())
                host_->uni_vmovups(
                        host_->ptr[host_->rsp + stack_offset], Xbyak::Xmm(idx));
            else if (vmm.isYMM())
                host_->uni_vmovups(
                        host_->ptr[host_->rsp + stack_offset], Xbyak::Ymm(idx));
            else
                host_->uni_vmovups(
                        host_->ptr[host_->rsp + stack_offset], Xbyak::Zmm(idx));
        }
    }
}

register_preserve_guard_t::~register_preserve_guard_t() {

    auto tmp_stack_offset = 0;

    while (!vmm_stack_.empty()) {
        const Xbyak::Xmm &vmm = vmm_stack_.top();
        const auto idx = vmm.getIdx();
        if (vmm.isXMM())
            host_->uni_vmovups(
                    Xbyak::Xmm(idx), host_->ptr[host_->rsp + tmp_stack_offset]);
        else if (vmm.isYMM())
            host_->uni_vmovups(
                    Xbyak::Ymm(idx), host_->ptr[host_->rsp + tmp_stack_offset]);
        else
            host_->uni_vmovups(
                    Xbyak::Zmm(idx), host_->ptr[host_->rsp + tmp_stack_offset]);

        tmp_stack_offset += get_vmm_size_bytes(vmm);
        vmm_stack_.pop();
    }

    if (vmm_to_preserve_size_bytes_)
        host_->add(host_->rsp, vmm_to_preserve_size_bytes_);

    while (!reg64_stack_.empty()) {
        host_->pop(reg64_stack_.top());
        reg64_stack_.pop();
    }
}

size_t register_preserve_guard_t::stack_space_occupied() const {
    constexpr static size_t reg64_size = 8;
    const size_t stack_space_occupied
            = vmm_to_preserve_size_bytes_ + reg64_stack_.size() * reg64_size;

    return stack_space_occupied;
}

conditional_register_preserve_guard_t::conditional_register_preserve_guard_t(
        bool condition_to_be_met, jit_generator_t *host,
        std::initializer_list<Xbyak::Reg64> reg64_to_preserve,
        std::initializer_list<Xbyak::Xmm> vmm_to_preserve)
    : register_preserve_guard_t {condition_to_be_met
                      ? register_preserve_guard_t {host, reg64_to_preserve,
                                vmm_to_preserve}
                      : register_preserve_guard_t {nullptr, {}, {}}} {};

/*
* Registry scratchpad code
*/

int registry_scratchpad_t::book(int size /*= DefaultRegSize*/) {
    const auto currentOffset = size_;
    size_ += size;
    return currentOffset;
}

void registry_scratchpad_t::saveReg(
        const Xbyak::Reg64 &reg, int booking) const {
    if (booking < 0) return;
    jit_.mov(jit_.ptr[jit_.rsp + booking], reg);
}

void registry_scratchpad_t::restoreReg(
        const Xbyak::Reg64 &reg, int booking) const {
    if (booking < 0) return;
    jit_.mov(reg, jit_.ptr[jit_.rsp + booking]);
}

void registry_scratchpad_t::restoreReg(
        const Xbyak::Reg32 &reg, int booking) const {
    if (booking < 0) return;
    jit_.mov(reg, jit_.ptr[jit_.rsp + booking]);
}

void registry_scratchpad_t::leaToReg(
        const Xbyak::Reg64 &reg, int booking) const {
    if (booking < 0) return;
    jit_.lea(reg, jit_.ptr[jit_.rsp + booking]);
}

Xbyak::Address registry_scratchpad_t::getPtr(int booking) const {
    return jit_.ptr[jit_.rsp + booking];
}

reg64_savable_t::reg64_savable_t(registry_scratchpad_t &regscratchpad,
        const Xbyak::Reg64 &reg, bool is_storable)
    : Xbyak::Reg64(reg), regscratchpad_(regscratchpad), storable_(is_storable) {
    if (storable_) booking_ = regscratchpad_.book();
}

reg64_savable_t::reg64_savable_t(registry_scratchpad_t &regscratchpad,
        const Xbyak::Reg64 &reg, const Xbyak::Reg64 &alternative_reg,
        bool use_alternative)
    : reg64_savable_t(regscratchpad, use_alternative ? alternative_reg : reg,
              !use_alternative) {}

reg64_savable_t::reg64_savable_t(registry_scratchpad_t &regscratchpad,
        const Xbyak::Reg64 &reg, const Xbyak::Reg64 &ext_reg)
    : reg64_savable_t(
              regscratchpad, reg, ext_reg, regscratchpad.ExtendedRegisters()) {
    // We expect ext_reg from extended registers set
    assert(16 <= ext_reg.getIdx() && ext_reg.getIdx() <= 31);
}

void reg64_savable_t::save() const {
    regscratchpad_.saveReg(*this, booking_);
}

void reg64_savable_t::saveTo(const reg64_savable_t &regsavable) const {
    if (regsavable.booking() >= 0)
        regscratchpad_.saveReg(*this, regsavable.booking());
    else
        regscratchpad_.jit().mov(regsavable, *this);
}

void reg64_savable_t::restore() const {
    regscratchpad_.restoreReg(*this, booking_);
}

void reg64_savable_t::lea() const {
    regscratchpad_.leaToReg(*this, booking_);
}

void reg64_savable_t::restoreTo(const Xbyak::Reg64 &reg) const {
    if (booking_ >= 0)
        regscratchpad_.restoreReg(reg, booking_);
    else
        regscratchpad_.jit().mov(reg, *this);
}

void reg64_savable_t::restoreTo(const Xbyak::Reg32 &reg32) const {
    if (booking_ >= 0)
        regscratchpad_.restoreReg(reg32, booking_);
    else
        regscratchpad_.jit().mov(reg32, *this);
}

void reg64_savable_t::addTo(const Xbyak::Reg64 &reg) const {
    if (booking_ >= 0)
        regscratchpad_.jit().add(reg, getStoragePtr());
    else
        regscratchpad_.jit().add(reg, *this);
}

void reg64_savable_t::imulTo(const Xbyak::Reg64 &reg, int imm) const {
    if (booking_ >= 0)
        regscratchpad_.jit().imul(reg, getStoragePtr(), imm);
    else
        regscratchpad_.jit().imul(reg, *this, imm);
}

reg64_savable_guard_t::reg64_savable_guard_t(
        std::initializer_list<const reg64_savable_t *> regs,
        bool condition /*= true*/) {
    if (!condition) return;

    // Reserve space
    regs_.reserve(regs.size());
    for (const auto &reg : regs) {
        if (std::find(regs_.begin(), regs_.end(), reg) == regs_.end()) {
            regs_.push_back(reg);
            reg->save();
        }
    }
}

reg64_savable_guard_t::reg64_savable_guard_t(std::initializer_list<
        std::pair<std::initializer_list<const reg64_savable_t *>, bool>>
                init_list) {
    // Reserve space
    size_t max_total_regs = 0;
    for (const auto &pair : init_list) {
        if (pair.second) max_total_regs += pair.first.size();
    }
    regs_.reserve(max_total_regs);

    for (const auto &pair : init_list) {
        if (pair.second) {
            for (const auto &reg : pair.first) {
                if (std::find(regs_.begin(), regs_.end(), reg) == regs_.end()) {
                    regs_.push_back(reg);
                    reg->save();
                }
            }
        }
    }
}

reg64_savable_guard_t::~reg64_savable_guard_t() {
    // Restore in reverse order
    for (auto it = regs_.rbegin(); it != regs_.rend(); ++it)
        (*it)->restore();
}

} // namespace injector_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
