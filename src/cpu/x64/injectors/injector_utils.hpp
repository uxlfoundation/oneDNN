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

#ifndef CPU_X64_INJECTORS_INJECTOR_UTILS_HPP
#define CPU_X64_INJECTORS_INJECTOR_UTILS_HPP

#include <array>
#include <cstddef>
#include <set>
#include <stack>

#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace injector_utils {

using vmm_index_set_t = typename std::set<size_t>;
using vmm_index_set_iterator_t = typename std::set<size_t>::iterator;

enum class layout_t { ncsp, c_blocked, nspc, cspn, unsupported };

inline layout_t get_layout_type(const memory_desc_wrapper &dst_d) {
    const auto strides = dst_d.blocking_desc().strides;
    if (!dst_d.is_plain()) return layout_t::c_blocked;
    if (strides[0] >= strides[1]
            && IMPLICATION(dst_d.ndims() >= 3, strides[1] >= strides[2]))
        return layout_t::ncsp;
    if (strides[1] == 1) return layout_t::nspc;
    if (strides[0] == 1) return layout_t::cspn;
    return layout_t::unsupported;
}

/*
 * Scope guard for general purpose register and vector registers preservation.
 * Pushes registers to stack during construction and pops during destruction.
 */
class register_preserve_guard_t {

public:
    register_preserve_guard_t(jit_generator_t *host,
            std::initializer_list<Xbyak::Reg64> reg64_to_preserve,
            std::initializer_list<Xbyak::Xmm> vmm_to_preserve = {});
    register_preserve_guard_t(register_preserve_guard_t &&other) = default;
    register_preserve_guard_t &operator=(register_preserve_guard_t &&other)
            = default;
    DNNL_DISALLOW_COPY_AND_ASSIGN(register_preserve_guard_t);
    ~register_preserve_guard_t();
    size_t stack_space_occupied() const;

private:
    jit_generator_t *host_;
    std::stack<Xbyak::Reg64> reg64_stack_;
    std::stack<Xbyak::Xmm> vmm_stack_;
    size_t vmm_to_preserve_size_bytes_;
};

class conditional_register_preserve_guard_t : public register_preserve_guard_t {
public:
    conditional_register_preserve_guard_t(bool condition_to_be_met,
            jit_generator_t *host,
            std::initializer_list<Xbyak::Reg64> reg64_to_preserve,
            std::initializer_list<Xbyak::Xmm> vmm_to_preserve = {});
    DNNL_DISALLOW_COPY_AND_ASSIGN(conditional_register_preserve_guard_t);
};

/*
 * This class provides functionality to book scratchpad space (based on stack
 * register), as well as save and restore general-purpose registers to/from the
 * scratchpad.
 */
class registry_scratchpad_t {
    static constexpr int DefaultReg64Size {8};

public:
    explicit registry_scratchpad_t(jit_generator_t &jit, cpu_isa_t isa)
        : jit_(jit), isa_(isa) {}
    DNNL_DISALLOW_COPY_AND_ASSIGN(registry_scratchpad_t);

    inline jit_generator_t &jit() const { return jit_; }
    inline bool ExtendedRegisters() const {
        return is_superset(isa_, avx512_core)
                && (mayiuse(avx10_2_512) || mayiuse(avx10_2_512_amx_2));
    }

    inline int Size() const { return size_; }

private:
    jit_generator_t &jit_;
    cpu_isa_t isa_ {isa_undef};
    int size_ {0};

    friend class reg64_savable_t;
    int book(int size = DefaultReg64Size);
    void saveReg(const Xbyak::Reg64 &reg, int booking) const;
    void restoreReg(const Xbyak::Reg64 &reg, int booking) const;
    void restoreReg(const Xbyak::Reg32 &reg, int booking) const;
    void leaToReg(const Xbyak::Reg64 &reg, int booking) const;
    Xbyak::Address getPtr(int booking) const;
};
/*
 * This functionality extends Xbyak::Reg64 by allowing the register to be
 * temporarily saved to a scratchpad space (booked via registry_scratchpad_t)
 * and later restored. It can be used in JIT kernels for preserving register
 * values in sections of code where the register might be overwritten.
 */
class reg64_savable_t : public Xbyak::Reg64 {
public:
    DNNL_DISALLOW_COPY_AND_ASSIGN(reg64_savable_t);

    // base constructor
    reg64_savable_t(registry_scratchpad_t &regscratchpad,
            const Xbyak::Reg64 &reg, bool is_storable = true);

    // This constructor provides conditional substitution:
    // choose a different register when a condition is met
    // (e.g., to avoid booking scratchpad by using a truly free register).
    reg64_savable_t(registry_scratchpad_t &regscratchpad,
            const Xbyak::Reg64 &reg, const Xbyak::Reg64 &alternative_reg,
            bool use_alternative);

    // This constructor is used to utilize extended registers (e.g. r16-r31).
    // If second register is provided and the ISA allows using extended registers
    // then we use it and then operations of saving/restoring will be dummy.
    reg64_savable_t(registry_scratchpad_t &regscratchpad,
            const Xbyak::Reg64 &reg, const Xbyak::Reg64 &ext_reg);

    inline int booking() const { return booking_; }

    inline bool savable() const { return booking_ >= 0; }

    virtual void save() const;
    virtual void restore() const;
    void saveTo(const reg64_savable_t &regsavable) const;
    void lea() const;
    void restoreTo(const Xbyak::Reg64 &reg) const;
    void restoreTo(const Xbyak::Reg32 &reg32) const;
    void addTo(const Xbyak::Reg64 &reg) const;
    void imulTo(const Xbyak::Reg64 &reg, int imm) const;
    Xbyak::Address getStoragePtr() const {
        return regscratchpad_.getPtr(booking_);
    }

    registry_scratchpad_t &regscratchpad_;

private:
    int booking_ {-1};
    bool storable_ {false};
};

class reg64_savable_backup_t : public reg64_savable_t {
public:
    DNNL_DISALLOW_COPY_AND_ASSIGN(reg64_savable_backup_t);
    reg64_savable_backup_t(const reg64_savable_t &other)
        : reg64_savable_t(other.regscratchpad_, other), other_(other) {}

    reg64_savable_backup_t(
            const reg64_savable_t &other, const Xbyak::Reg64 &ext_reg)
        : reg64_savable_t(other.regscratchpad_,
                  other.regscratchpad_.ExtendedRegisters() ? ext_reg : other)
        , other_(other) {}

    void save() const override { other_.saveTo(*this); }
    void restore() const override { restoreTo(other_); }

private:
    const reg64_savable_t &other_;
};

/*
 * This class takes a list of RegSavable objects and, depending on a condition,
 * saves their values upon construction and restores them upon destruction.
 */
class reg64_savable_guard_t {
public:
    reg64_savable_guard_t(std::initializer_list<const reg64_savable_t *> regs,
            bool condition = true);

    reg64_savable_guard_t(std::initializer_list<
            std::pair<std::initializer_list<const reg64_savable_t *>, bool>>
                    init_list);

    ~reg64_savable_guard_t();
    DNNL_DISALLOW_COPY_AND_ASSIGN(reg64_savable_guard_t);

private:
    std::vector<const reg64_savable_t *> regs_;
};

} // namespace injector_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
