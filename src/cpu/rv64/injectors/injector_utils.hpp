/*******************************************************************************
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
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
#ifndef CPU_RV64_INJECTORS_INJECTOR_UTILS_HPP
#define CPU_RV64_INJECTORS_INJECTOR_UTILS_HPP

#include <cstddef>
#include <set>
#include <vector>
#include <initializer_list>

#include "common/memory_desc_wrapper.hpp"
#include "common/utils.hpp"
#include "cpu/rv64/cpu_isa_traits.hpp"
#include "cpu/rv64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

// JIT register-type traits, the codegen companion to cpu_isa_traits.hpp (which
// stays free of the xbyak_riscv code-generator include for the intrinsics
// build). RVV has a single architectural vector-register type; the active
// element count is governed at run time by vsetvli rather than by the register
// type, so unlike x64/aarch64 there is no per-isa vector-width type. The traits
// are still keyed on isa for structural parity with x64/aarch64 and to leave
// room for future isa-specific register budgets.
template <cpu_isa_t isa>
struct jit_isa_traits_t {
    using Vmm = Xbyak_riscv::VReg;
    static constexpr int n_vregs = 32;
};

namespace injector_utils {

using vmm_index_set_t = typename std::set<size_t>;
using vmm_index_set_iterator_t = typename std::set<size_t>::iterator;

// Builds the vmm-index set for a run of accumulators that occupy the half-open
// PHYSICAL register span [start_idx, end_idx), where each accumulator is a
// register group of group_stride vector registers (its LMUL/EMUL, one of
// 1/2/4; consecutive accumulator bases are therefore group_stride registers
// apart). This is the rv64-native replacement for x64's implicit stride-1
// register numbering: unlike x64, where every vmm index is an independent
// full-width register, an rv64 run of m>1 accumulators does NOT occupy
// consecutive indices. A caller converting an x64 compute_vector_range(start,
// end) must therefore state its group stride here (group_stride == 1 reproduces
// x64's m1 behavior). The stride is mandatory precisely so that such a
// conversion cannot compile as a silent stride-1 enumeration. Every enumerated
// base is group-aligned when start_idx is, so the result is always a legal set
// of RVV register-group bases. An illegal request is a host bug: JIT_ASSERT
// keeps the checks in Release, where the throw fails kernel creation instead of
// enumerating reserved or overlapping register indices.
//
// m8 (group_stride == 8) is intentionally rejected: the binary injector's rhs
// helper, per-oc gather, and narrow-dtype staging are all e32/acc_lmul groups
// (up to m4), so at m8 they plus the m8 accumulator and the v0 mask cannot fit
// the 32-register file (m8 acc + m8 helper + m8 gather + m8 narrow + v0 > 32).
// Compute binary post-ops at LMUL <= m4.
template <cpu_isa_t isa>
inline vmm_index_set_t make_vmm_group_set(
        size_t start_idx, size_t end_idx, size_t group_stride) {
    // JIT_ASSERT throws Xbyak_riscv::Error; make it visible to the macro here
    // (the injector .cpp files get it from a file-scope using-directive, but
    // this helper lives in a header ahead of any such directive).
    using Xbyak_riscv::Error;
    JIT_ASSERT(group_stride == 1 || group_stride == 2 || group_stride == 4);
    JIT_ASSERT(start_idx <= end_idx && start_idx % group_stride == 0
            && (end_idx - start_idx) % group_stride == 0);
    JIT_ASSERT(end_idx <= (size_t)jit_isa_traits_t<isa>::n_vregs);
    vmm_index_set_t vmm_idxs;
    for (size_t i = start_idx; i < end_idx; i += group_stride)
        vmm_idxs.emplace(i);
    return vmm_idxs;
}

// Validates an already-built set of accumulator group bases (as passed to the
// explicit set overloads): group_stride is each accumulator's register count
// (LMUL/EMUL, 1/2/4), and every base must be group-aligned, the groups must not
// overlap, and the top group must fit in the register file. Same
// JIT_ASSERT/Release contract as make_vmm_group_set (an illegal host set fails
// kernel creation instead of emitting overlapping or out-of-range registers).
template <cpu_isa_t isa>
inline void validate_vmm_group_set(
        const vmm_index_set_t &bases, size_t group_stride) {
    using Xbyak_riscv::Error;
    JIT_ASSERT(group_stride == 1 || group_stride == 2 || group_stride == 4);
    size_t prev_group_end = 0;
    bool first = true;
    for (size_t base : bases) { // std::set iterates ascending
        JIT_ASSERT(base % group_stride == 0);
        JIT_ASSERT(
                base + group_stride <= (size_t)jit_isa_traits_t<isa>::n_vregs);
        JIT_ASSERT(first || base >= prev_group_end);
        prev_group_end = base + group_stride;
        first = false;
    }
}

// Mirrors x64/aarch64 injector_utils::layout_t / get_layout_type: classifies a
// dst descriptor so the binary injector's per-channel/per-spatial address math
// (calculate_*_{ncsp,blocked,nspc,cspn}) can pick the right stride formula.
enum class layout_t { ncsp, c_blocked, nspc, cspn, unsupported };

inline layout_t get_layout_type(const memory_desc_wrapper &dst_d) {
    const auto strides = dst_d.blocking_desc().strides;
    if (!dst_d.is_plain()) {
        // x64/aarch64 classify every non-plain descriptor as c_blocked, but the
        // channel formulas that consume this (calculate_oc_blocked and the
        // per-lane gather) both assume the single inner block IS the channel
        // one. A descriptor blocked on another dimension (e.g. Abcd16a, blocked
        // on N) would silently produce a wrong channel, so report it as
        // unsupported and let the caller decline.
        const auto &bd = dst_d.blocking_desc();
        return bd.inner_nblks == 1 && bd.inner_idxs[0] == 1
                ? layout_t::c_blocked
                : layout_t::unsupported;
    }
    if (strides[0] >= strides[1]
            && IMPLICATION(dst_d.ndims() >= 3, strides[1] >= strides[2]))
        return layout_t::ncsp;
    if (strides[1] == 1) return layout_t::nspc;
    if (strides[0] == 1) return layout_t::cspn;
    return layout_t::unsupported;
}

// Scope guard that preserves scalar (GPR), vector (VReg), and floating-point
// (FReg) registers across a region by spilling them to the stack on
// construction and restoring them on destruction (x64's
// register_preserve_guard_t). The binary injector constructs it around each
// injection: preserve_gpr_helpers shields the helper GPRs and the fixed X_TMP
// scratch the way x64 shields rax/rdx/r8/r9, and preserve_vmm_helper shields
// the rhs dt helper vmm the way x64 pushes Vmm(vmm_hint).
//
// Vector entries are spilled as whole register GROUPS of vmm_group_stride
// registers (the injector's helper is an e32/acc_lmul group, so its LMUL is the
// accumulator's; the VReg must be group_stride-aligned) with the vl-independent
// whole-register moves vs<N>r.v/vl<N>re8.v (N = group_stride, one of 1/2/4).
// The frame is sized at RUN time from csrr vlenb (group_stride * vlenb per
// group; vlenb >= 16 for VLEN >= 128 keeps sp 16-byte aligned); t0 serves as
// the vlenb scratch and is saved and restored by the guard itself. v0 must
// never be passed: it is the architectural mask register and the injector's
// compare/select paths write it by design.
class register_preserve_guard_t {
public:
    register_preserve_guard_t(jit_generator_t *host,
            std::initializer_list<Xbyak_riscv::Reg> gpr_to_preserve,
            std::initializer_list<Xbyak_riscv::VReg> vmm_to_preserve = {},
            std::initializer_list<Xbyak_riscv::FReg> freg_to_preserve = {},
            size_t vmm_group_stride = 4);
    register_preserve_guard_t(register_preserve_guard_t &&) = default;
    register_preserve_guard_t &operator=(register_preserve_guard_t &&)
            = default;
    DNNL_DISALLOW_COPY_AND_ASSIGN(register_preserve_guard_t);
    ~register_preserve_guard_t();

    // Number of compile-time stack bytes the guard occupies (16B-aligned;
    // excludes the runtime-sized vector frame).
    size_t stack_space_occupied() const { return stack_bytes_; }

private:
    jit_generator_t *host_;
    std::vector<Xbyak_riscv::Reg> gpr_regs_;
    std::vector<Xbyak_riscv::VReg> vmm_regs_;
    std::vector<Xbyak_riscv::FReg> freg_regs_;
    size_t vmm_group_stride_ = 4;
    size_t stack_bytes_;
};

} // namespace injector_utils
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_INJECTORS_INJECTOR_UTILS_HPP
