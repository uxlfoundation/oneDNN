/*******************************************************************************
* Copyright 2025 Intel Corporation
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

/*
 * Do not #include this file directly; ngen uses it internally.
 */

#if XE4
template <typename DT = void> void abs_(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
    abs_<DT>(defaultMods(), dst, src0, loc);
}
#endif

    template <typename DT = void> void add(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        add<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void> void add(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        add<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void> void add3(const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        add3<DT>(defaultMods(), dst, src0, src1, src2, loc);
    }
    template <typename DT = void> void add3(const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        add3<DT>(defaultMods(), dst, src0, src1, src2, loc);
    }
#if XE4
    template <typename DT = void>
    void addc(const RegData &dst, const FlagRegister &carryOut, const RegData &src0, const RegData &src1, const FlagRegister &carryIn = FlagRegister(), SourceLocation loc = {}) {
        addc<DT>(InstructionModifier(), dst, carryOut, src0, src1, carryIn, loc);
    }
    template <typename DT = void>
    void addc(const RegData &dst, const FlagRegister &carryOut, const RegData &src0, const Immediate &src1, const FlagRegister &carryIn = FlagRegister(), SourceLocation loc = {}) {
        addc<DT>(InstructionModifier(), dst, carryOut, src0, src1, carryIn, loc);
    }
#endif
    template <typename DT = void>
    void and_(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        and_<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void>
    void and_(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        and_<DT>(defaultMods(), dst, src0, src1, loc);
    }
#ifndef NGEN_NO_OP_NAMES
    template <typename DT = void>
    void and(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        and_<DT>(mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void and(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        and_<DT>(mod, dst, src0, src1, loc);
    }
#endif
    template <typename DT = void>
    void asr(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        asr<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void>
    void asr(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        asr<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void>
    void avg(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        avg<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void>
    void avg(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        avg<DT>(defaultMods(), dst, src0, src1, loc);
    }
#if XE4
    template <typename DT = uint32_t>
    void bfe(int width, int offset, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        bfe<DT>(defaultMods(), width, offset, dst, src0, loc);
    }
    template <typename DT = uint32_t>
    void bfegen(const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        bfe<DT>(defaultMods(), dst, src0, src1, src2, loc);
    }
    template <typename DT = uint32_t>
    void bfegen(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        bfe<DT>(mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = uint32_t>
    void bfi(int width, int offset, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        bfi<DT>(defaultMods(), width, offset, dst, src0, src1, loc);
    }
    template <typename DT = uint32_t>
    void bfi(int width, int offset, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        bfi<DT>(defaultMods(), width, offset, dst, src0, src1, loc);
    }
    template <typename DT = uint32_t>
    void bfigen(const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        bfigen<DT>(defaultMods(), dst, src0, src1, src2, loc);
    }
    template <typename DT = uint32_t>
    void bfigen(const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &src2, SourceLocation loc = {}) {
        bfigen<DT>(defaultMods(), dst, src0, src1, src2, loc);
    }
    template <typename DT = void> void bfn2(uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        bfn2<DT>(defaultMods(), ctrl, dst, src0, src1, loc);
    }
    template <typename DT = void> void bfn2(uint8_t ctrl, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        bfn2<DT>(defaultMods(), ctrl, dst, src0, src1, loc);
    }
    template <typename DT = void> void bfn3(uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        bfn3<DT>(defaultMods(), ctrl, dst, src0, src1, src2, loc);
    }
    template <typename DT = void> void bfn3(uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        bfn3<DT>(defaultMods(), ctrl, dst, src0, src1, src2, loc);
    }
    template <typename DT = void> void bfn3(uint8_t ctrl, const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &src2, SourceLocation loc = {}) {
        bfn3<DT>(defaultMods(), ctrl, dst, src0, src1, src2, loc);
    }
#endif
    template <typename DT = void> void bfrev(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        bfrev<DT>(defaultMods(), dst, src0, loc);
    }
#if XE4
    template <typename DT = void> void brepgen(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        brepgen<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void> void brepgen(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        brepgen<DT>(defaultMods(), dst, src0, src1, loc);
    }
#endif
    void call(const RegData &dst, Label &jip, SourceLocation loc = {}) {
        call(defaultMods(), dst, jip, loc);
    }
    void call(const RegData &dst, const RegData &jip, SourceLocation loc = {}) {
        call(defaultMods(), dst, jip, loc);
    }
#if XE4
    void calla(const RegData &dst, uint64_t jip = 0, SourceLocation loc = {}) {
        calla(defaultMods(), dst, jip, loc);
    }
    void callad(const RegData &dst, uint64_t jip = 0, SourceLocation loc = {}) {
        callad(defaultMods(), dst, jip, loc);
    }
    void calld(const RegData &dst, Label &jip, SourceLocation loc = {}) {
        calld(InstructionModifier(), dst, jip, loc);
    }
#endif
    template <typename DT = void> void cbit(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        cbit<DT>(defaultMods(), dst, src0, loc);
    }
    template <typename DT = void> void cmp(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        cmp<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void> void cmp(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        cmp<DT>(defaultMods(), dst, src0, src1, loc);
    }
#if XE4
    void cnvg(uint8_t cid, bool exp, const RegData &src0, SourceLocation loc = {})   { cnvg(InstructionModifier(), cid, exp, src0, loc); }
    void cnvg(uint8_t cid, bool exp, const Immediate &src0, SourceLocation loc = {}) { cnvg(InstructionModifier(), cid, exp, src0, loc); }
    void cnvg(bool exp, const RegData &src0, SourceLocation loc = {})                { cnvg(InstructionModifier(), exp, src0, loc); }
    void cnvg(bool exp, const Immediate &src0, SourceLocation loc = {})              { cnvg(InstructionModifier(), exp, src0, loc); }
    template <typename DT = void> void cvt(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        cvt<DT>(defaultMods(), dst, src0, loc);
    }
    template <typename DT = void> void cvt2(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        cvt2<DT>(defaultMods(), dst, src0, src1, loc);
    }
#endif
    template <typename DT = void> void dp4a(const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        dp4a<DT>(InstructionModifier(), dst, src0, src1, src2, loc);
    }
    template <typename DT = void> void dp4a(const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        dp4a<DT>(InstructionModifier(), dst, src0, src1, src2, loc);
    }
#if XE4
    template <typename DT = void> void dp4a(const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &src2, SourceLocation loc = {}) {
        dp4a<DT>(InstructionModifier(), dst, src0, src1, src2, loc);
    }
    template <typename DT = void> void emcos(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        emcos<DT>(defaultMods(), dst, src0, loc);
    }
    template <typename DT = void> void emexp2(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        emexp2<DT>(defaultMods(), dst, src0, loc);
    }
    template <typename DT = void> void eminv(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        eminv<DT>(defaultMods(), dst, src0, loc);
    }
    template <typename DT = void> void eminvm(const ExtendedReg &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        eminvm<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void> void emlog2(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        emlog2<DT>(defaultMods(), dst, src0, loc);
    }
    template <typename DT = void> void emrsqt(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        emrsqt<DT>(defaultMods(), dst, src0, loc);
    }
    template <typename DT = void> void emrsqtm(const ExtendedReg &dst, const RegData &src0, SourceLocation loc = {}) {
        emrsqtm<DT>(defaultMods(), dst, src0, loc);
    }
    template <typename DT = void> void emsin(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        emsin<DT>(defaultMods(), dst, src0, loc);
    }
    template <typename DT = void> void emsgmd(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        emsgmd<DT>(defaultMods(), dst, src0, loc);
    }
    template <typename DT = void> void emsqt(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        emsqt<DT>(defaultMods(), dst, src0, loc);
    }
    template <typename DT = void> void emtanh(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        emtanh<DT>(defaultMods(), dst, src0, loc);
    }
#endif
    template <typename DT = void> void fbh(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        fbh<DT>(defaultMods(), dst, src0, loc);
    }
    template <typename DT = void> void fbl(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        fbl<DT>(defaultMods(), dst, src0, loc);
    }
    void goto_(Label &jip, Label &uip, bool branchCtrl = false, SourceLocation loc = {}) {
        goto_(defaultMods(), jip, uip, branchCtrl, loc);
    }
    void goto_(Label &jip, SourceLocation loc = {}) {
        goto_(defaultMods(), jip, jip, false, loc);
    }
    void jmpi(Label &jip, SourceLocation loc = {}) {
#if XE4
        jmpi((hardware >= HW::Xe4) ? 32 : 1, jip, loc);
#else
        jmpi(1, jip, loc);
#endif
    }
    void jmpi(const RegData &jip, SourceLocation loc = {}) {
#if XE4
        jmpi((hardware >= HW::Xe4) ? 32 : 1, jip, loc);
#else
        jmpi(1, jip, loc);
#endif
    }
    void join(Label &jip, SourceLocation loc = {}) { join(defaultMods(), jip, loc); }
    void join(SourceLocation loc = {})             { join(defaultMods(), loc); }
    template <typename DT = void> void mad(const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        mad<DT>(InstructionModifier(), dst, src0, src1, src2, loc);
    }
    template <typename DT = void> void mad(const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        mad<DT>(InstructionModifier(), dst, src0, src1, src2, loc);
    }
#if XE4
    template <typename DT = void> void mad(const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &src2, SourceLocation loc = {}) {
        mad<DT>(InstructionModifier(), dst, src0, src1, src2, loc);
    }
    template <typename DT = void> void mad(const RegData &dst, const RegData &src0, const Immediate &src1, const Immediate &src2, SourceLocation loc = {}) {
        mad<DT>(InstructionModifier(), dst, src0, src1, src2, loc);
    }
    template <typename DT = void> void madc(const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, const FlagRegister &carryIn = FlagRegister(), SourceLocation loc = {}) {
        madc<DT>(defaultMods(), dst, src0, src1, src2, carryIn, loc);
    }
    template <typename DT = void> void madc(const RegData &dst, const FlagRegister &carryOut, const RegData &src0, const RegData &src1, const RegData &src2, const FlagRegister &carryIn = FlagRegister(), SourceLocation loc = {}) {
        madc<DT>(defaultMods(), dst, carryOut, src0, src1, src2, carryIn, loc);
    }
    template <typename DT = void> void madlh(const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        madlh<DT>(defaultMods(), dst, src0, src1, src2, loc);
    }
    template <typename DT = void> void madlh(const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        madlh<DT>(defaultMods(), dst, src0, src1, src2, loc);
    }
    template <typename DT = void> void madlh(const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &src2, SourceLocation loc = {}) {
        madlh<DT>(defaultMods(), dst, src0, src1, src2, loc);
    }
    template <typename DT = void> void madlh(const RegData &dst, const RegData &src0, const Immediate &src1, const Immediate &src2, SourceLocation loc = {}) {
        madlh<DT>(defaultMods(), dst, src0, src1, src2, loc);
    }
#endif
    template <typename DT = void> void madm(const ExtendedReg &dst, const ExtendedReg &src0, const ExtendedReg &src1, const ExtendedReg &src2, SourceLocation loc = {}) {
        madm<DT>(defaultMods(), dst, src0, src1, src2, loc);
    }
    template <typename DT = void> void max_(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        max_<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void> void max_(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        max_<DT>(defaultMods(), dst, src0, src1, loc);
    }
#ifndef NGEN_WINDOWS_COMPAT
    template <typename DT = void> void max(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        max_<DT>(mod | ge | f0[0], dst, src0, src1, loc);
    }
    template <typename DT = void> void max(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        max_<DT>(mod | ge | f0[0], dst, src0, src1, loc);
    }
    template <typename DT = void> void max(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        max_<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void> void max(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        max_<DT>(defaultMods(), dst, src0, src1, loc);
    }
#endif
    template <typename DT = void> void min_(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        min_<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void> void min_(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        min_<DT>(defaultMods(), dst, src0, src1, loc);
    }
#ifndef NGEN_WINDOWS_COMPAT
    template <typename DT = void> void min(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        min_<DT>(mod, dst, src0, src1, loc);
    }
    template <typename DT = void> void min(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        min_<DT>(mod, dst, src0, src1, loc);
    }
    template <typename DT = void> void min(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        min_<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void> void min(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        min_<DT>(defaultMods(), dst, src0, src1, loc);
    }
#endif
    template <typename DT = void> void mov(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        mov<DT>(defaultMods(), dst, src0, loc);
    }
    template <typename DT = void> void mov(const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        mov<DT>(defaultMods(), dst, src0, loc);
    }
#if XE4
    template <typename DT = void> void movb(const RegData &dst, const RegData &src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        movb<DT>(defaultMods(), dst, src0, lanemask, loc);
    }
    template <typename DT = void> void movb(const RegData &dst, const RegData &src0, RegData lanemask, SourceLocation loc = {}) {
        movb<DT>(defaultMods(), dst, src0, lanemask, loc);
    }
    template <typename DT = void> void movb(const RegData &dst, const Immediate &src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        movb<DT>(defaultMods(), dst, src0, lanemask, loc);
    }
    template <typename DT = void> void movb(const RegData &dst, const Immediate &src0, RegData lanemask, SourceLocation loc = {}) {
        movb<DT>(defaultMods(), dst, src0, lanemask, loc);
    }
    template <typename DT = void> void movg(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        movg<DT>(defaultMods(), dst, src0, loc);
    }
    template <typename DT = void> void movs(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        movs<DT>(defaultMods(), dst, src0, loc);
    }
#endif
    template <typename DT = uint32_t> void msk(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        msk<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = uint32_t> void msk(const RegData &dst, const RegData &src0, uint32_t src1, SourceLocation loc = {}) {
        msk<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = uint32_t> void msk(const RegData &dst, uint32_t src0, const RegData &src1, SourceLocation loc = {}) {
        msk<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = uint32_t> void msk(const RegData &dst, uint32_t src0, uint32_t src1, SourceLocation loc = {}) {
        msk<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void> void mul(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        mul<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void> void mul(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        mul<DT>(defaultMods(), dst, src0, src1, loc);
    }
#if XE3P
    template <typename DT = void> void mullh(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        mullh<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void> void mullh(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        mullh<DT>(defaultMods(), dst, src0, src1, loc);
    }
#endif
    template <typename DT = void> void not_(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        not_<DT>(defaultMods(), dst, src0, loc);
    }
#ifndef NGEN_NO_OP_NAMES
    template <typename DT = void>
    void not(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        not_<DT>(mod, dst, src0, loc);
    }
    template <typename DT = void> void not(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        not_<DT>(defaultMods(), dst, src0, loc);
    }
#endif
    template <typename DT = void> void or_(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        or_<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void> void or_(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        or_<DT>(defaultMods(), dst, src0, src1, loc);
    }
#ifndef NGEN_NO_OP_NAMES
    template <typename DT = void>
    void or(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        or_<DT>(mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void or(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        or_<DT>(mod, dst, src0, src1, loc);
    }
    template <typename DT = void> void or(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        or_<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void> void or(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        or_<DT>(defaultMods(), dst, src0, src1, loc);
    }
#endif
#if XE4
    template <typename DT = uint32_t> void redand(const RegData &dst, const RegData &src0, const RegData &lanemask, SourceLocation loc = {}) {
        redand<DT>(defaultMods(), dst, src0, lanemask, loc);
    }
    template <typename DT = uint32_t> void redand(const RegData &dst, const RegData &src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        redand<DT>(defaultMods(), dst, src0, lanemask, loc);
    }
    template <typename DT = uint32_t> void redfirst(const RegData &dst, const RegData &src0, const RegData &lanemask, SourceLocation loc = {}) {
        redfirst<DT>(defaultMods(), dst, src0, lanemask, loc);
    }
    template <typename DT = uint32_t> void redfirst(const RegData &dst, const RegData &src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        redfirst<DT>(defaultMods(), dst, src0, lanemask, loc);
    }
    template <typename DT = uint32_t> void redfirstidx(const RegData &dst, const RegData &src0, const RegData &lanemask, SourceLocation loc = {}) {
        redfirstidx<DT>(defaultMods(), dst, src0, lanemask, loc);
    }
    template <typename DT = uint32_t> void redfirstidx(const RegData &dst, const RegData &src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        redfirstidx<DT>(defaultMods(), dst, src0, lanemask, loc);
    }
    template <typename DT = void> void redmax(const RegData &dst, const RegData &src0, const RegData &lanemask, SourceLocation loc = {}) {
        redmax<DT>(defaultMods(), dst, src0, lanemask, loc);
    }
    template <typename DT = void> void redmax(const RegData &dst, const RegData &src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        redmax<DT>(defaultMods(), dst, src0, lanemask, loc);
    }
    template <typename DT = void> void redmin(const RegData &dst, const RegData &src0, const RegData &lanemask, SourceLocation loc = {}) {
        redmin<DT>(defaultMods(), dst, src0, lanemask, loc);
    }
    template <typename DT = void> void redmin(const RegData &dst, const RegData &src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        redmin<DT>(defaultMods(), dst, src0, lanemask, loc);
    }
    template <typename DT = uint32_t> void redor(const RegData &dst, const RegData &src0, const RegData &lanemask, SourceLocation loc = {}) {
        redor<DT>(defaultMods(), dst, src0, lanemask, loc);
    }
    template <typename DT = uint32_t> void redor(const RegData &dst, const RegData &src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        redor<DT>(defaultMods(), dst, src0, lanemask, loc);
    }
    template <typename DT = void> void redsum(const RegData &dst, const RegData &src0, const RegData &lanemask, SourceLocation loc = {}) {
        redsum<DT>(defaultMods(), dst, src0, lanemask, loc);
    }
    template <typename DT = void> void redsum(const RegData &dst, const RegData &src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        redsum<DT>(defaultMods(), dst, src0, lanemask, loc);
    }
    template <typename DT = uint32_t> void redxor(const RegData &dst, const RegData &src0, const RegData &lanemask, SourceLocation loc = {}) {
        redxor<DT>(defaultMods(), dst, src0, lanemask, loc);
    }
    template <typename DT = uint32_t> void redxor(const RegData &dst, const RegData &src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        redxor<DT>(defaultMods(), dst, src0, lanemask, loc);
    }
#endif
    void ret(const RegData &src0, SourceLocation loc = {}) { ret(defaultMods(), src0, loc); }
#if XE4
    void retd(const RegData &src0, SourceLocation loc = {}) { retd(InstructionModifier(), src0, loc); }
    template <typename DT = void> void rnd(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        rnd<DT>(defaultMods(), dst, src0, loc);
    }
#endif
    template <typename DT = void> void rol(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        rol<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void> void rol(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        rol<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void> void ror(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        ror<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void> void ror(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        ror<DT>(defaultMods(), dst, src0, src1, loc);
    }
#if XE4
    template <typename DT = uint32_t> void sel(const RegData &dst, const RegData &src0, const RegData &src1, FlagRegister predicate, SourceLocation loc = {}) {
        sel<DT>(defaultMods(), dst, src0, src1, predicate, loc);
    }
    template <typename DT = uint32_t> void sel(const RegData &dst, const RegData &src0, const Immediate &src1, FlagRegister predicate, SourceLocation loc = {}) {
        sel<DT>(defaultMods(), dst, src0, src1, predicate, loc);
    }
    template <typename DT = uint32_t> void shfld(const RegData &dst, const RegData &src0, const RegData &src1, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        shfld<DT>(InstructionModifier(), dst, src0, src1, lanemask, loc);
    }
    template <typename DT = uint32_t> void shfld(const RegData &dst, const RegData &src0, const Immediate &src1, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        shfld<DT>(InstructionModifier(), dst, src0, src1, lanemask, loc);
    }
    template <typename DT = uint32_t> void shfli(const RegData &dst, const RegData &src0, const RegData &src1, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        shfli<DT>(InstructionModifier(), dst, src0, src1, lanemask, loc);
    }
    template <typename DT = uint32_t> void shfli(const RegData &dst, const RegData &src0, const Immediate &src1, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        shfli<DT>(InstructionModifier(), dst, src0, src1, lanemask, loc);
    }
    template <typename DT = uint32_t> void shflu(const RegData &dst, const RegData &src0, const RegData &src1, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        shflu<DT>(InstructionModifier(), dst, src0, src1, lanemask, loc);
    }
    template <typename DT = uint32_t> void shflu(const RegData &dst, const RegData &src0, const Immediate &src1, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        shflu<DT>(InstructionModifier(), dst, src0, src1, lanemask, loc);
    }
    template <typename DT = uint32_t> void shflx(const RegData &dst, const RegData &src0, const RegData &src1, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        shflx<DT>(InstructionModifier(), dst, src0, src1, lanemask, loc);
    }
    template <typename DT = uint32_t> void shflx(const RegData &dst, const RegData &src0, const Immediate &src1, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        shflx<DT>(InstructionModifier(), dst, src0, src1, lanemask, loc);
    }
#endif
    template <typename DT = void> void shl(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        shl<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void> void shl(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        shl<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void> void shr(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        shr<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void> void shr(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        shr<DT>(defaultMods(), dst, src0, src1, loc);
    }
#if XE4
    template <typename DT = void>
    void subb(const RegData &dst, const FlagRegister &carryOut, const RegData &src0, const RegData &src1, const FlagRegister &carryIn = FlagRegister(), SourceLocation loc = {}) {
        subb<DT>(InstructionModifier(), dst, carryOut, src0, src1, carryIn, loc);
    }
    template <typename DT = void>
    void subb(const RegData &dst, const FlagRegister &carryOut, const RegData &src0, const Immediate &src1, const FlagRegister &carryIn = FlagRegister(), SourceLocation loc = {}) {
        subb<DT>(InstructionModifier(), dst, carryOut, src0, src1, carryIn, loc);
    }
#endif
    template <typename DT = void> void xor_(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        xor_<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void> void xor_(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        xor_<DT>(defaultMods(), dst, src0, src1, loc);
    }
#ifndef NGEN_NO_OP_NAMES
    template <typename DT = void>
    void xor(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        xor_<DT>(mod, dst, src0, src1);
    }
    template <typename DT = void>
    void xor(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        xor_<DT>(mod, dst, src0, src1);
    }
    template <typename DT = void> void xor(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        xor_<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void> void xor(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        xor_<DT>(defaultMods(), dst, src0, src1, loc);
    }
#endif

#if XE4
    template <typename DT = void> void sadd(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        sadd<DT>(InstructionModifier(), dst, src0, src1, loc);
    }
    template <typename DT = void> void sadd(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        sadd<DT>(InstructionModifier(), dst, src0, src1, loc);
    }
    template <typename DT = void> void sasr(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        sasr<DT>(InstructionModifier(), dst, src0, src1, loc);
    }
    template <typename DT = void> void sasr(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        sasr<DT>(InstructionModifier(), dst, src0, src1, loc);
    }
    template <typename DT = void> void sbfn2(uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        sbfn2<DT>(InstructionModifier(), ctrl, dst, src0, src1, loc);
    }
    template <typename DT = void> void sbfn2(uint8_t ctrl, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        sbfn2<DT>(InstructionModifier(), ctrl, dst, src0, src1, loc);
    }
    template <typename DT = void> void sbfn3(uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        sbfn3<DT>(InstructionModifier(), ctrl, dst, src0, src1, src2, loc);
    }
    template <typename DT = void> void sbfn3(uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        sbfn3<DT>(InstructionModifier(), ctrl, dst, src0, src1, src2, loc);
    }
    template <typename DT = void> void sbfn3(uint8_t ctrl, const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &src2, SourceLocation loc = {}) {
        sbfn3<DT>(InstructionModifier(), ctrl, dst, src0, src1, src2, loc);
    }
    template <typename DT = uint32_t>
    void sbfe(int width, int offset, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        sbfe<DT>(InstructionModifier(), width, offset, dst, src0, loc);
    }
    template <typename DT = uint32_t>
    void sbfegen(const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        sbfegen<DT>(InstructionModifier(), dst, src0, src1, src2, loc);
    }
    template <typename DT = uint32_t> void sbfi(int width, int offset, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        sbfi<DT>(InstructionModifier(), width, offset, dst, src0, src1, loc);
    }
    template <typename DT = uint32_t> void sbfi(int width, int offset, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        sbfi<DT>(InstructionModifier(), width, offset, dst, src0, src1, loc);
    }
    template <typename DT = uint32_t> void sbfia(const InstructionModifier &mod, unsigned width, unsigned offset, IndirectARF dst, const RegData &src1, SourceLocation loc = {}) {
        sbfia<DT>(mod, width, offset, dst, dst, src1, loc);
    }
    template <typename DT = uint32_t> void sbfia(const InstructionModifier &mod, unsigned width, unsigned offset, IndirectARF dst, const Immediate &src1, SourceLocation loc = {}) {
        sbfia<DT>(mod, width, offset, dst, dst, src1, loc);
    }
    template <typename DT = uint32_t> void sbfia(int width, int offset, IndirectARF dst, IndirectARF src0, const RegData &src1, SourceLocation loc = {}) {
        sbfia<DT>(InstructionModifier(), width, offset, dst, src0, src1, loc);
    }
    template <typename DT = uint32_t> void sbfia(int width, int offset, IndirectARF dst, IndirectARF src0, const Immediate &src1, SourceLocation loc = {}) {
        sbfia<DT>(InstructionModifier(), width, offset, dst, src0, src1, loc);
    }
    template <typename DT = uint32_t> void sbfia(int width, int offset, IndirectARF dst, const RegData &src1, SourceLocation loc = {}) {
        sbfia<DT>(InstructionModifier(), width, offset, dst, dst, src1, loc);
    }
    template <typename DT = uint32_t> void sbfia(int width, int offset, IndirectARF dst, const Immediate &src1, SourceLocation loc = {}) {
        sbfia<DT>(InstructionModifier(), width, offset, dst, dst, src1, loc);
    }
    template <typename DT = uint32_t>
    void sbfigen(const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        sbfigen<DT>(InstructionModifier(), dst, src0, src1, src2, loc);
    }
    template <typename DT = uint32_t>
    void sbfigen(const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &src2, SourceLocation loc = {}) {
        sbfigen<DT>(InstructionModifier(), dst, src0, src1, src2, loc);
    }
    template <typename DT = void> void sbfrev(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        sbfrev<DT>(InstructionModifier(), dst, src0, loc);
    }
    template <typename DT = void> void sbrepgen(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        sbrepgen<DT>(InstructionModifier(), dst, src0, src1, loc);
    }
    template <typename DT = void> void sbrepgen(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        sbrepgen<DT>(InstructionModifier(), dst, src0, src1, loc);
    }
    template <typename DT = void> void scmp(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        scmp<DT>(InstructionModifier(), dst, src0, src1, loc);
    }
    template <typename DT = void> void scmp(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        scmp<DT>(InstructionModifier(), dst, src0, src1, loc);
    }
    template <typename DT = void> void sfbh(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        sfbh<DT>(InstructionModifier(), dst, src0, loc);
    }
    template <typename DT = void> void sfbl(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        sfbl<DT>(InstructionModifier(), dst, src0, loc);
    }
    template <typename DT = uint32_t> void sgeta(const RegData &dst, IndirectARF src0, SourceLocation loc = {}) {
        sgeta<DT>(InstructionModifier(), dst, src0, loc);
    }
    template <typename DT = void> void smad(const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        smad<DT>(InstructionModifier(), dst, src0, src1, src2, loc);
    }
    template <typename DT = void> void smad(const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        smad<DT>(InstructionModifier(), dst, src0, src1, src2, loc);
    }
    template <typename DT = void> void smad(const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &src2, SourceLocation loc = {}) {
        smad<DT>(InstructionModifier(), dst, src0, src1, src2, loc);
    }
    template <typename DT = void> void smad(const RegData &dst, const RegData &src0, const Immediate &src1, const Immediate &src2, SourceLocation loc = {}) {
        smad<DT>(InstructionModifier(), dst, src0, src1, src2, loc);
    }
    template <typename DT = void> void smov(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        smov<DT>(InstructionModifier(), dst, src0, loc);
    }
    template <typename DT = void> void smov(const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        smov<DT>(InstructionModifier(), dst, src0, loc);
    }
    template <typename DT = uint32_t> void smsk(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        smsk<DT>(InstructionModifier(), dst, src0, src1, loc);
    }
    template <typename DT = uint32_t> void smsk(const RegData &dst, const RegData &src0, uint32_t src1, SourceLocation loc = {}) {
        smsk<DT>(InstructionModifier(), dst, src0, src1, loc);
    }
    template <typename DT = uint32_t> void smsk(const RegData &dst, uint32_t src0, const RegData &src1, SourceLocation loc = {}) {
        smsk<DT>(InstructionModifier(), dst, src0, src1, loc);
    }
    template <typename DT = uint32_t> void smsk(const RegData &dst, uint32_t src0, uint32_t src1, SourceLocation loc = {}) {
        smsk<DT>(InstructionModifier(), dst, src0, src1, loc);
    }
    template <typename DT = void> void smul(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        smul<DT>(InstructionModifier(), dst, src0, src1, loc);
    }
    template <typename DT = void> void smul(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        smul<DT>(InstructionModifier(), dst, src0, src1, loc);
    }
    template <typename DT = void> void smullh(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        smullh<DT>(InstructionModifier(), dst, src0, src1, loc);
    }
    template <typename DT = void> void smullh(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        smullh<DT>(InstructionModifier(), dst, src0, src1, loc);
    }
    template <typename DT = uint32_t> void ssel(const RegData &dst, const RegData &src0, const RegData &src1, FlagRegister predicate, SourceLocation loc = {}) {
        ssel<DT>(InstructionModifier(), dst, src0, src1, predicate, loc);
    }
    template <typename DT = uint32_t> void ssel(const RegData &dst, const RegData &src0, const Immediate &src1, FlagRegister predicate, SourceLocation loc = {}) {
        ssel<DT>(InstructionModifier(), dst, src0, src1, predicate, loc);
    }
    template <typename DT = uint32_t> void sseta(IndirectARF dst, const RegData &src0, SourceLocation loc = {}) {
        sseta<DT>(InstructionModifier(), dst, src0, loc);
    }
    template <typename DT = uint32_t> void sseta(IndirectARF dst, const Immediate &src0, SourceLocation loc = {}) {
        sseta<DT>(InstructionModifier(), dst, src0, loc);
    }
    template <typename DT = void> void sshl(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        sshl<DT>(InstructionModifier(), dst, src0, src1, loc);
    }
    template <typename DT = void> void sshl(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        sshl<DT>(InstructionModifier(), dst, src0, src1, loc);
    }
    template <typename DT = void> void sshr(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        sshr<DT>(InstructionModifier(), dst, src0, src1, loc);
    }
    template <typename DT = void> void sshr(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        sshr<DT>(InstructionModifier(), dst, src0, src1, loc);
    }
#endif