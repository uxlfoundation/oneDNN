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
#include "cpu/rv64/injectors/injector_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace injector_utils {

using namespace Xbyak_riscv;

register_preserve_guard_t::register_preserve_guard_t(jit_generator_t *host,
        std::initializer_list<Reg> gpr_to_preserve,
        std::initializer_list<VReg> vmm_to_preserve,
        std::initializer_list<FReg> freg_to_preserve, size_t vmm_group_stride)
    : host_(host)
    , gpr_regs_(gpr_to_preserve)
    , vmm_regs_(vmm_to_preserve)
    , freg_regs_(freg_to_preserve)
    , vmm_group_stride_(vmm_group_stride)
    , stack_bytes_(0) {

    const size_t n_slots = gpr_regs_.size() + freg_regs_.size();
    if (n_slots > 0) {
        // Keep sp 16-byte aligned per the RISC-V calling convention; each
        // register takes one 8-byte slot.
        stack_bytes_ = utils::rnd_up(n_slots * 8, 16);
        host_->addi(sp, sp, -static_cast<int>(stack_bytes_));

        int off = 0;
        for (const auto &r : gpr_regs_) {
            host_->sd(r, sp, off);
            off += 8;
        }
        for (const auto &f : freg_regs_) {
            host_->fsd(f, sp, off);
            off += 8;
        }
    }

    if (!vmm_regs_.empty()) {
        // Whole-group spills with a runtime VLEN-sized frame (vs<N>r/vl<N>re8
        // move vmm_group_stride * vlenb bytes regardless of vl); t0 is the
        // vlenb scratch, saved and restored by the guard itself. group_stride
        // is the spilled groups' LMUL (1/2/4); group_stride * vlenb (vlenb >= 16
        // for VLEN >= 128) keeps sp 16-byte aligned.
        const int lg = vmm_group_stride_ == 4 ? 2 : (int)vmm_group_stride_ - 1;
        host_->addi(sp, sp, -16);
        host_->sd(t0, sp, 0);
        host_->csrr(t0, CSR::vlenb);
        host_->slli(t0, t0, lg); // group_stride * vlenb per group
        for (const auto &vreg : vmm_regs_) {
            // Each entry must be a legal whole group: not v0 (the live mask
            // register), group_stride-aligned, and fully inside the register
            // file. JIT_ASSERT keeps the check in Release builds, failing
            // kernel creation instead of emitting an illegal group spill.
            JIT_ASSERT(vreg.getIdx() != 0
                    && vreg.getIdx() % vmm_group_stride_ == 0
                    && vreg.getIdx() + vmm_group_stride_
                            <= (size_t)jit_isa_traits_t<v>::n_vregs);
            host_->sub(sp, sp, t0);
            if (vmm_group_stride_ == 1)
                host_->vs1r_v(vreg, sp);
            else if (vmm_group_stride_ == 2)
                host_->vs2r_v(vreg, sp);
            else
                host_->vs4r_v(vreg, sp);
        }
    }
}

register_preserve_guard_t::~register_preserve_guard_t() {
    if (!vmm_regs_.empty()) {
        // t0 may have been clobbered by the guarded region: recompute the
        // group size, restore the groups in reverse, then t0 itself.
        const int lg = vmm_group_stride_ == 4 ? 2 : (int)vmm_group_stride_ - 1;
        host_->csrr(t0, CSR::vlenb);
        host_->slli(t0, t0, lg);
        for (auto it = vmm_regs_.rbegin(); it != vmm_regs_.rend(); ++it) {
            if (vmm_group_stride_ == 1)
                host_->vl1re8_v(*it, sp);
            else if (vmm_group_stride_ == 2)
                host_->vl2re8_v(*it, sp);
            else
                host_->vl4re8_v(*it, sp);
            host_->add(sp, sp, t0);
        }
        host_->ld(t0, sp, 0);
        host_->addi(sp, sp, 16);
    }

    if (stack_bytes_ == 0) return;

    int off = 0;
    for (const auto &r : gpr_regs_) {
        host_->ld(r, sp, off);
        off += 8;
    }
    for (const auto &f : freg_regs_) {
        host_->fld(f, sp, off);
        off += 8;
    }
    host_->addi(sp, sp, static_cast<int>(stack_bytes_));
}

} // namespace injector_utils
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
