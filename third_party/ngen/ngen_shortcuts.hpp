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
    template <typename DT = void> void bfrev(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        bfrev<DT>(defaultMods(), dst, src0, loc);
    }
    void call(const RegData &dst, Label &jip, SourceLocation loc = {}) {
        call(defaultMods(), dst, jip, loc);
    }
    void call(const RegData &dst, const RegData &jip, SourceLocation loc = {}) {
        call(defaultMods(), dst, jip, loc);
    }
    template <typename DT = void> void cbit(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        cbit<DT>(defaultMods(), dst, src0, loc);
    }
    template <typename DT = void> void cmp(const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        cmp<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void> void cmp(const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        cmp<DT>(defaultMods(), dst, src0, src1, loc);
    }
    template <typename DT = void> void dp4a(const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        dp4a<DT>(InstructionModifier(), dst, src0, src1, src2, loc);
    }
    template <typename DT = void> void dp4a(const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        dp4a<DT>(InstructionModifier(), dst, src0, src1, src2, loc);
    }
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
        jmpi(1, jip, loc);
    }
    void jmpi(const RegData &jip, SourceLocation loc = {}) {
        jmpi(1, jip, loc);
    }
    void join(Label &jip, SourceLocation loc = {}) { join(defaultMods(), jip, loc); }
    void join(SourceLocation loc = {})             { join(defaultMods(), loc); }
    template <typename DT = void> void mad(const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        mad<DT>(InstructionModifier(), dst, src0, src1, src2, loc);
    }
    template <typename DT = void> void mad(const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        mad<DT>(InstructionModifier(), dst, src0, src1, src2, loc);
    }
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
    void ret(const RegData &src0, SourceLocation loc = {}) { ret(defaultMods(), src0, loc); }
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

