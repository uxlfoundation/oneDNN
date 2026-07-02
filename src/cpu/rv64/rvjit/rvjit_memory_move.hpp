/*******************************************************************************
* Copyright 2026 Barcelona Supercomputing Center
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

#ifndef CPU_RV64_RVJIT_RVJIT_MEMORY_MOVE_HPP
#define CPU_RV64_RVJIT_RVJIT_MEMORY_MOVE_HPP

#include "cpu/rv64/rvjit/rvjit_types.hpp"

#if defined(RVJIT_DEBUG)
#include "common/verbose.hpp"
#define DEBUg(...) \
    do { \
        if (get_verbose(verbose_t::debuginfo) > 1) { __VA_ARGS__ } \
    } while (0)
#else
#define DEBUg(...)
#endif
#define DEBUG(...) DEBUg(__VA_ARGS__)

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace rvjit {

#if XBYAK_RISCV_V

/// Vector memory op: unit-stride, constant stride, or indexed
struct vaddr_t {
    enum class kind_t { unit, strided, indexed };

    kind_t kind = kind_t::unit;

    // All kinds
    Reg base;

    // Strided
    Reg stride;

    // Indexed
    VReg index;
    SEW isew = SEW::e32;
    bool ordered = true;

    /// Factory for unit-stride vector memory operation descriptor
    static vaddr_t unit(const Reg &base) {
        vaddr_t a;
        a.kind = kind_t::unit;
        a.base = base;
        return a;
    }

    /// Factory for constant stride vector memory operation descriptor
    static vaddr_t strided(const Reg &base, const Reg &stride) {
        vaddr_t a;
        a.kind = kind_t::strided;
        a.base = base;
        a.stride = stride;
        return a;
    }

    /// Factory for indexed vector memory operation descriptor
    static vaddr_t indexed(const Reg &base, const VReg &index,
            SEW isew = SEW::e32, bool ordered = true) {
        vaddr_t a;
        a.kind = kind_t::indexed;
        a.base = base;
        a.index = index;
        a.isew = isew;
        a.ordered = ordered;
        return a;
    }
};

#endif

/// Component for width-adaptive memory move instructions
class memory_move_t {
public:
    explicit memory_move_t(emitter_t e) : e_(e) {}

    /// Scalar float load (flh/flw/fld) adapted to `sew`
    ///
    /// @note Emits nothing if `sew` has no supported float load
    void fload(const FReg &fd, const Reg &addr, SEW sew, int imm = 0) const {
        switch (sew) {
            case SEW::e16: e_->flh(fd, addr, imm); break;
            case SEW::e32: e_->flw(fd, addr, imm); break;
            case SEW::e64: e_->fld(fd, addr, imm); break;
            default:
                DEBUG({
                    verbose_printf(verbose_t::debuginfo,
                            "rvjit: Failed memory_move_t.fload due to "
                            "unsupported width\n");
                });
                break;
        }
    }

    /// Scalar integer load (lb/lbu/lw/ld) adapted to `sew`
    ///
    /// @param is_signed  Selects `lb` vs `lbu` at e8; e32/e64 always sign
    ///     extend via `lw`/`ld` (there is no unsigned counterpart in scope)
    ///
    /// @note Emits nothing if `sew` has no supported integer load
    void xload(const Reg &rd, const Reg &addr, SEW sew, bool is_signed = true,
            int imm = 0) const {
        switch (sew) {
            case SEW::e8:
                is_signed ? e_->lb(rd, addr, imm) : e_->lbu(rd, addr, imm);
                break;
            case SEW::e32: e_->lw(rd, addr, imm); break;
            case SEW::e64: e_->ld(rd, addr, imm); break;
            default:
                DEBUG({
                    verbose_printf(verbose_t::debuginfo,
                            "rvjit: Failed memory_move_t.xload due to "
                            "unsupported width\n");
                });
                break;
        }
    }

#if XBYAK_RISCV_V

    /// Unit-stride vector load (vle{8,16,32,64}_v) adapted to `sew`
    ///
    /// @pre Caller has already configured a compatible SEW/LMUL via vsetvli
    void vle(const VReg &vd, const Reg &base, SEW sew,
            VM vm = VM::unmasked) const {
        switch (sew) {
            case SEW::e8: e_->vle8_v(vd, base, vm); break;
            case SEW::e16: e_->vle16_v(vd, base, vm); break;
            case SEW::e32: e_->vle32_v(vd, base, vm); break;
            case SEW::e64: e_->vle64_v(vd, base, vm); break;
        }
    }

    /// Unit-stride vector store (vse{8,16,32,64}_v) adapted to `sew`
    ///
    /// @pre Caller has already configured a compatible SEW/LMUL via vsetvli
    void vse(const VReg &vs, const Reg &base, SEW sew,
            VM vm = VM::unmasked) const {
        switch (sew) {
            case SEW::e8: e_->vse8_v(vs, base, vm); break;
            case SEW::e16: e_->vse16_v(vs, base, vm); break;
            case SEW::e32: e_->vse32_v(vs, base, vm); break;
            case SEW::e64: e_->vse64_v(vs, base, vm); break;
        }
    }

    /// Strided vector load (vlse{8,16,32,64}_v) adapted to `sew`
    ///
    /// @pre Caller has already configured a compatible SEW/LMUL via vsetvli
    void vlse(const VReg &vd, const Reg &base, const Reg &stride, SEW sew,
            VM vm = VM::unmasked) const {
        switch (sew) {
            case SEW::e8: e_->vlse8_v(vd, base, stride, vm); break;
            case SEW::e16: e_->vlse16_v(vd, base, stride, vm); break;
            case SEW::e32: e_->vlse32_v(vd, base, stride, vm); break;
            case SEW::e64: e_->vlse64_v(vd, base, stride, vm); break;
        }
    }

    /// Strided vector load (vsse{8,16,32,64}_v) adapted to `sew`
    ///
    /// @pre Caller has already configured a compatible SEW/LMUL via vsetvli
    void vsse(const VReg &vs, const Reg &base, const Reg &stride, SEW sew,
            VM vm = VM::unmasked) const {
        switch (sew) {
            case SEW::e8: e_->vsse8_v(vs, base, stride, vm); break;
            case SEW::e16: e_->vsse16_v(vs, base, stride, vm); break;
            case SEW::e32: e_->vsse32_v(vs, base, stride, vm); break;
            case SEW::e64: e_->vsse64_v(vs, base, stride, vm); break;
        }
    }

    /// Ordered indexed vector load (vloxei{8,16,32,64}_v) adapted to `isew`
    ///
    /// @note `isew` is the width of each element of `index`
    void vloxei(const VReg &vd, const Reg &base, const VReg &index, SEW isew,
            VM vm = VM::unmasked) const {
        switch (isew) {
            case SEW::e8: e_->vloxei8_v(vd, base, index, vm); break;
            case SEW::e16: e_->vloxei16_v(vd, base, index, vm); break;
            case SEW::e32: e_->vloxei32_v(vd, base, index, vm); break;
            case SEW::e64: e_->vloxei64_v(vd, base, index, vm); break;
        }
    }

    /// Unordered indexed vector load (vluxei{8,16,32,64}_v) adapted to `isew`
    ///
    /// @note `isew` is the width of each element of `index`
    void vluxei(const VReg &vd, const Reg &base, const VReg &index, SEW isew,
            VM vm = VM::unmasked) const {
        switch (isew) {
            case SEW::e8: e_->vluxei8_v(vd, base, index, vm); break;
            case SEW::e16: e_->vluxei16_v(vd, base, index, vm); break;
            case SEW::e32: e_->vluxei32_v(vd, base, index, vm); break;
            case SEW::e64: e_->vluxei64_v(vd, base, index, vm); break;
        }
    }

    /// Ordered indexed vector store (vsoxei{8,16,32,64}_v) adapted to `isew`
    ///
    /// @note `isew` is the width of each element of `index`
    void vsoxei(const VReg &vs, const Reg &base, const VReg &index, SEW isew,
            VM vm = VM::unmasked) const {
        switch (isew) {
            case SEW::e8: e_->vsoxei8_v(vs, base, index, vm); break;
            case SEW::e16: e_->vsoxei16_v(vs, base, index, vm); break;
            case SEW::e32: e_->vsoxei32_v(vs, base, index, vm); break;
            case SEW::e64: e_->vsoxei64_v(vs, base, index, vm); break;
        }
    }

    /// Unordered indexed vector load (vsuxei{8,16,32,64}_v) adapted to `isew`
    ///
    /// @note `isew` is the width of each element of `index`
    void vsuxei(const VReg &vs, const Reg &base, const VReg &index, SEW isew,
            VM vm = VM::unmasked) const {
        switch (isew) {
            case SEW::e8: e_->vsuxei8_v(vs, base, index, vm); break;
            case SEW::e16: e_->vsuxei16_v(vs, base, index, vm); break;
            case SEW::e32: e_->vsuxei32_v(vs, base, index, vm); break;
            case SEW::e64: e_->vsuxei64_v(vs, base, index, vm); break;
        }
    }

    /// Mode- and width-adaptable vector load instruction
    void vload(const VReg &vd, const vaddr_t &addr, SEW sew,
            VM vm = VM::unmasked) const {
        switch (addr.kind) {
            case vaddr_t::kind_t::unit: vle(vd, addr.base, sew, vm); break;
            case vaddr_t::kind_t::strided:
                vlse(vd, addr.base, addr.stride, sew, vm);
                break;
            case vaddr_t::kind_t::indexed:
                if (addr.ordered)
                    vloxei(vd, addr.base, addr.index, addr.isew, vm);
                else
                    vluxei(vd, addr.base, addr.index, addr.isew, vm);
                break;
        }
    }

    /// Mode- and width-adaptable vector store instruction
    void vstore(const VReg &vs, const vaddr_t &addr, SEW sew,
            VM vm = VM::unmasked) const {
        switch (addr.kind) {
            case vaddr_t::kind_t::unit: vse(vs, addr.base, sew, vm); break;
            case vaddr_t::kind_t::strided:
                vsse(vs, addr.base, addr.stride, sew, vm);
                break;
            case vaddr_t::kind_t::indexed:
                if (addr.ordered)
                    vsoxei(vs, addr.base, addr.index, addr.isew, vm);
                else
                    vsuxei(vs, addr.base, addr.index, addr.isew, vm);
                break;
        }
    }

#endif

private:
    emitter_t e_;
};

} // namespace rvjit
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#undef DEBUg
#undef DEBUG

#endif
