/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

// nGEN: a C++ library for runtime Gen assembly generation.
//
// Macros that control nGEN's interface:
//    NGEN_SAFE             if defined, enables run-time safety checks. Exceptions will be thrown if checks fail.
//    NGEN_SHORT_NAMES      if defined, enables some short names (r[...] for indirect addressing, W for NoMask)
//    NGEN_GLOBAL_REGS      if defined, register names and instruction modifiers (r7, cr0, Switch, etc.) are
//                           global variables in the ngen namespace. Otherwise, they are members of the code
//                           generator classes
//    NGEN_CPP11            if defined, ngen is C++11-compatible (C++17 not required)

#ifndef NGEN_HPP
#define NGEN_HPP

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-int-conversion"
#endif

#include "ngen_config_internal.hpp"

#include <array>
#include <cstring>
#include <type_traits>
#include <vector>

#include "ngen_core.hpp"
#include "ngen_auto_swsb.hpp"
#include "ngen_debuginfo.hpp"
// -----------------------------------------------------------------------
// Binary formats, split between pre-Gen12 and post-Gen12.
#include "ngen_gen8.hpp"
#include "ngen_gen12.hpp"
#if XE4
#include "ngen_xe4.hpp"
#endif
// -----------------------------------------------------------------------

#include "ngen_asm.hpp"

namespace NGEN_NAMESPACE {

// Forward declarations.
template <HW hw> class BinaryCodeGenerator;
template <HW hw> class ELFCodeGenerator;


template <HW hw> struct Instruction12Dispatch       { using type = Instruction12;    };
template <> struct Instruction12Dispatch<HW::XeHPC> { using type = InstructionXeHPC; };
template <> struct Instruction12Dispatch<HW::Xe2>   { using type = InstructionXeHPC; };
template <> struct Instruction12Dispatch<HW::Xe3>   { using type = InstructionXeHPC; };
#if XE3P
template <> struct Instruction12Dispatch<HW::XE3P_35_10>  { using type = InstructionXe3p;  };
template <> struct Instruction12Dispatch<HW::XE3P_35_11>  { using type = InstructionXe3p;  };
template <> struct Instruction12Dispatch<HW::XE3P_UNKNOWN>  { using type = InstructionXe3p;  };
#endif
#if XE4
template <> struct Instruction12Dispatch<HW::Xe4>   { using type = InstructionXe4;   };
#endif

// MSVC v140 workaround for enum comparison in template arguments.
static constexpr bool hwLT(HW hw1, HW hw2) { return hw1 < hw2; }
static constexpr bool hwLE(HW hw1, HW hw2) { return hw1 <= hw2; }
static constexpr bool hwGE(HW hw1, HW hw2) { return hw1 >= hw2; }
static constexpr bool hwGT(HW hw1, HW hw2) { return hw1 > hw2; }

class LabelFixup {
public:
    uint32_t labelID;
    int32_t anchor;
    int32_t offset;

    LabelFixup(uint32_t labelID_, int32_t offset_) : labelID(labelID_), anchor(0), offset(offset_) {}

    static constexpr auto JIPOffset = 12;
    static constexpr auto JIPOffsetJMPI = -4;
    static constexpr auto UIPOffset = 8;

#if XE4
    static constexpr auto JIPOffsetXe4 = 9;
    static constexpr auto UIPOffsetXe4 = 5;
#endif
};

#if defined(NGEN_GLOBAL_REGS) && !defined(NGEN_GLOBAL_REGS_DEFINED)
#define NGEN_GLOBAL_REGS_DEFINED
#include "ngen_registers.hpp"
#endif

template <HW hw>
class BinaryCodeGenerator
{
    friend class ELFCodeGenerator<hw>;

public:
    using RootCodeGenerator = BinaryCodeGenerator;
    static constexpr HW hardware = hw;
    static constexpr HW getHardware() { return hardware; }
    void cancelAutoSWSB() { cancelAutoSWSB_ = true; }

protected:
    class InstructionStream {
        friend class BinaryCodeGenerator;

        std::vector<LabelFixup> fixups;
        std::vector<uint32_t> labels;
        std::vector<uint64_t> code;
#if XE4
        std::vector<uint32_t> savedBPs;
#endif
        bool appended = false;

        int length() const { return int(code.size() * sizeof(uint64_t)); }

        void db(const Instruction8 &i) {
            code.push_back(i.qword[0]);
            code.push_back(i.qword[1]);
        }

        void db(const Instruction12 &i) {
            code.push_back(i.qword[0]);
            code.push_back(i.qword[1]);
        }

#if XE4
        void db(const InstructionXe4 &i, bool isBP = false) {
            if (isBP) savedBPs.push_back(static_cast<uint32_t>(code.size()) >> 1);
            code.push_back(i.qword[0]);
            code.push_back(i.qword[1]);
        }

        void db(const InstructionXe4 &i, const InstructionModifier &mod) { db(i, mod.isBreakpoint()); }
#endif

        void addFixup(LabelFixup fixup) {
            fixup.anchor = length();
            fixups.push_back(fixup);
        }

        void mark(Label &label, LabelManager &man) {
            uint32_t id = label.getID(man);

            man.setTarget(id, length());
            labels.push_back(id);
        }

        void fixLabels(LabelManager &man) {
            for (const auto &fixup : fixups) {
                int32_t target = man.getTarget(fixup.labelID);
                uint8_t *field = ((uint8_t *) code.data()) + fixup.anchor + fixup.offset;
                *((int32_t *) field) = target - fixup.anchor;
            }
        }

        void append(InstructionStream &other, LabelManager &man) {
            auto offset = length();
            auto sz = code.size();

            code.resize(sz + other.code.size());
            std::copy(other.code.begin(), other.code.end(), code.begin() + sz);

            sz = labels.size();
            labels.resize(sz + other.labels.size());
            std::copy(other.labels.begin(), other.labels.end(), labels.begin() + sz);

            for (LabelFixup fixup : other.fixups) {
                fixup.anchor += offset;
                fixups.push_back(fixup);
            }

#ifdef NGEN_SAFE
            if (other.appended && !other.labels.empty())
                throw multiple_label_exception();
#endif

            for (uint32_t id : other.labels)
                man.offsetTarget(id, offset);

#if XE4
            if (hw >= HW::Xe4) {
                sz = savedBPs.size();
                savedBPs.resize(sz + other.savedBPs.size());
                for (auto ibp: other.savedBPs)
                    savedBPs[sz++] = ibp + (offset / sizeof(InstructionXe4));
            }
#endif

            other.appended = true;
        }

        InstructionStream() {}
    };

    class Program {
        friend class BinaryCodeGenerator;
        using Instruction = typename Instruction12Dispatch<hw>::type;
        std::vector<uint64_t> &code;

        Program(InstructionStream &stream) : code(stream.code) {};

    public:
        size_t size() const                               { return code.size() >> 1; }
        Instruction &operator[](size_t index)             { return *reinterpret_cast<Instruction *>(&code[index * 2]); }
        const Instruction &operator[](size_t index) const { return *reinterpret_cast<Instruction *>(&code[index * 2]); }
    };

    static constexpr bool isGen12 = (hw >= HW::Gen12LP);
    Product product;
    int declaredGRFs = 128;

    InterfaceLabels _interfaceLabels;

    Label _lastFenceLabel;
    RegData _lastFenceDst;

    DebugLine debugLine;

    std::atomic<bool> cancelAutoSWSB_;

private:
    InstructionModifier defaultModifier;
#if XE3P
    bool useEfficient64Bit = (hw >= HW::XE3P_35_10);
#endif

    LabelManager labelManager;
    InstructionStream rootStream;
    std::vector<InstructionStream*> streamStack;

    template <typename Instruction>
    void db(const Instruction &i, SourceLocation loc) {
        debugLine.add(rootStream.length(), loc);    /* FIXME: stream support */
        db(i);
    }

    void db(const Instruction8 &i)   { streamStack.back()->db(i); }
    void db(const Instruction12 &i)  { streamStack.back()->db(i); }
#if XE4
    void db(const InstructionXe4 &i) { streamStack.back()->db(i); }
    void db(const InstructionXe4 &i, const InstructionModifier &mod, SourceLocation loc) {
        debugLine.add(rootStream.length(), loc);    /* FIXME: stream support */
        streamStack.back()->db(i, mod);
    }
#endif
    void addFixup(LabelFixup fixup)  { streamStack.back()->addFixup(fixup); }

    template <bool forceWE = false, typename D, typename S0, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, SourceLocation loc);
    template <bool forceWE = false, typename D, typename S0, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, SourceLocation loc);
    template <bool forceWE = false, typename D, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, const Immediate &src0, SourceLocation loc);
    template <bool forceWE = false, typename D, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, const Immediate &src0, SourceLocation loc);

    template <bool forceWE = false, typename D, typename S0, typename S1, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, SourceLocation loc);
    template <bool forceWE = false, typename D, typename S0, typename S1, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, SourceLocation loc);
    template <bool forceWE = false, typename D, typename S0, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, const Immediate &src1, SourceLocation loc);
    template <bool forceWE = false, typename D, typename S0, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, const Immediate &src1, SourceLocation loc);

    template <HW hw_ = hw>
    typename std::enable_if<hwLE(hw_, HW::Gen9)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, RegData dst, RegData src0, RegData src1, RegData src2, SourceLocation loc);
    template <HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, Align16Operand dst, Align16Operand src0, Align16Operand src1, Align16Operand src2, SourceLocation loc);
    template <typename D, typename S0, typename S1, typename S2, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2, SourceLocation loc);
    template <typename D, typename S0, typename S1, typename S2, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2, SourceLocation loc);

    template <typename DS0>
    void opMath(Opcode op, DataType defaultType, const InstructionModifier &mod, MathFunction fc, DS0 dst, DS0 src0, SourceLocation loc);
    template <typename DS0, typename S1>
    void opMath(Opcode op, DataType defaultType, const InstructionModifier &mod, MathFunction fc, DS0 dst, DS0 src0, S1 src1, SourceLocation loc);

    template <typename D, typename S0, typename S2>
    void opBfn(Opcode op, DataType defaultType, const InstructionModifier &mod, int bfnCtrl, D dst, S0 src0, RegData src1, S2 src2, SourceLocation loc);
    void opDpas(Opcode op, DataType defaultType, const InstructionModifier &mod, int sdepth, int rcount, RegData dst, RegData src0, RegData src1, RegData src2, SourceLocation loc);

#if XE3P
    void opBdpas(Opcode op, DataType defaultType, const InstructionModifier &mod, int sdepth, int rcount, RegData dst, RegData src0, RegData src1, RegData src2, RegData src3, RegData src4, SourceLocation loc);
#endif

    template <typename D, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opSend(Opcode op, const InstructionModifier &mod, SharedFunction sfid, const RegData &dst, const RegData &src0, const RegData &src1, int src1Length, uint32_t exdesc, D desc, SourceLocation loc);
    template <typename D, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opSend(Opcode op, const InstructionModifier &mod, SharedFunction sfid, const RegData &dst, const RegData &src0, const RegData &src1, int src1Length, const RegData &exdesc, D desc, SourceLocation loc);
    template <typename ED, typename D, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opSend(Opcode op, const InstructionModifier &mod, SharedFunction sfid, const RegData &dst, const RegData &src0, const RegData &src1, int src1Length, ED exdesc, D desc, SourceLocation loc);

    template <HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opSend(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc, SourceLocation loc);
    template <HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opSend(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc, SourceLocation loc);
    template <typename D, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opSend(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, D desc, SourceLocation loc);

    template <typename ED, typename D, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opSends(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, ED exdesc, D desc, SourceLocation loc);
    template <typename D, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opSends(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, D desc, SourceLocation loc);
    template <typename D, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opSends(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, RegData exdesc, D desc, SourceLocation loc);

#if XE3P
    void opSendg(Opcode op, const InstructionModifier &mod, SharedFunction sfid, const RegData &dst, RegData src0, int src0Len, const RegData &src1, int src1Len, RegData ind0, RegData ind1, uint64_t desc, SourceLocation loc);
#endif

    template <HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, int32_t uip, SourceLocation loc);
    template <HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, int32_t uip, SourceLocation loc);
    template <bool forceWE = false, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, SourceLocation loc);
    template <bool forceWE = false, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, SourceLocation loc);
    template <bool forceWE = false, bool small12 = true, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc);
    template <bool forceWE = false, bool small12 = true, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc);

    void opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, Label &jip, Label &uip, SourceLocation loc);
    template <bool forceWE = false>
    void opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, Label &jip, SourceLocation loc);
    void opCall(Opcode op, const InstructionModifier &mod, const RegData &dst, Label &jip, SourceLocation loc);

    template <HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opJmpi(Opcode op, const InstructionModifier &mod, const RegData &dst, RegData src0, uint32_t jip, SourceLocation loc);
    template <HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opJmpi(Opcode op, const InstructionModifier &mod, const RegData &dst, RegData src0, uint32_t jip, SourceLocation loc);
    void opJmpi(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, Label &jip, SourceLocation loc);

#if XE3P
    void opShflLfsr(Opcode op, uint8_t fc, DataType defaultType, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc);
    void opShflLfsr(Opcode op, uint8_t fc, DataType defaultType, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc);
#endif

    void opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod, SourceLocation loc);
    void opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod, RegData src0, SourceLocation loc);
    void opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod, const Immediate &src0, SourceLocation loc);

    void opNop(Opcode op, SourceLocation loc);

    template <typename S1>
    void opDirective(Directive directive, RegData src0, S1 src1, SourceLocation loc);

#if XE4
    template <typename D, typename S0, typename S1, typename S2>
    Opcode preprocessXe4(OpcodeClassXe4 opclass, DataType &defaultType, InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2);

    template <typename D, typename S0>
    void op128A(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, D dst, S0 src0, SourceLocation loc) {
        op128A(opclass, defaultType, mod, dst, src0, NullRegister(), NullRegister(), loc);
    }
    template <typename D, typename S0, typename S1>
    void op128A(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, D dst, S0 src0, S1 src1, SourceLocation loc) {
        op128A(opclass, defaultType, mod, dst, src0, src1, NullRegister(), loc);
    }
    template <typename D, typename S0, typename S1, typename S2>
    void op128A(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, D dst, S0 src0, S1 src1, S2 src2, SourceLocation loc);
    template <typename D, typename S0, typename S1, typename S2>
    void op128A(Opcode op, InstructionModifier mod, D dst, S0 src0, S1 src1, S2 src2, SourceLocation loc);

    template <typename S1, typename S2>
    void opMullh(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, RegData dst, RegData src0, S1 src1, S2 src2, SourceLocation loc);

    void op128AD(OpcodeClassXe4 opclassA, OpcodeClassXe4 opclassD, DataType defaultType, InstructionModifier mod, RegData dst, RegData src0, Immediate src1, SourceLocation loc) {
        op128AD(opclassA, opclassD, defaultType, mod, dst, src0, src1, NullRegister(), loc);
    }
    template <typename S1, typename S2>
    void op128AD(OpcodeClassXe4 opclassA, OpcodeClassXe4 opclassD, DataType defaultType, InstructionModifier mod, RegData dst, RegData src0, S1 src1, S2 src2, SourceLocation loc);

    void op128B(OpcodeClassXe4 opclass, InstructionModifier mod, RegData dst, RegData src0, uint32_t jip, uint32_t uip, SourceLocation loc);
    void op128B(OpcodeClassXe4 opclass, InstructionModifier mod, RegData dst, Label &jip, SourceLocation loc);
    void op128B(OpcodeClassXe4 opclass, InstructionModifier mod, RegData dst, Label &jip, Label &uip, SourceLocation loc);

    void op128C(Opcode op, InstructionModifier mod, SharedFunction sfid, uint64_t desc, RegData dst, RegData src0, RegData src1, RegData ind, bool int2, SourceLocation loc);

    template <typename S1, typename S2>
    void op128D(Opcode op, InstructionModifier mod, RegData dst, RegData src0, S1 src1, S2 src2, SourceLocation loc);

    template <typename S1, typename S2>
    void op128E(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, uint8_t ctrl, RegData dst, RegData src0, S1 src1, S2 src2, SourceLocation loc);

    template <typename S1>
    void op128F(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, RegData dst, RegData src0, S1 src1, SourceLocation loc) {
        op128F(opclass, defaultType, mod, dst, src0, null, src1, loc);
    };

    template <typename S1, typename S2>
    void op128F(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, RegData dst, RegData src0, S1 src1, S2 src2, SourceLocation loc, uint8_t sgran = 0);

    template <typename DS0, typename S1>
    void op128G(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, unsigned width, unsigned offset, DS0 dst, DS0 src0, S1 src1, SourceLocation loc);

    void op128H(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, ExtendedReg dst, ExtendedReg src0, ExtendedReg src1, ExtendedReg src2, SourceLocation loc);

    template <typename S0, typename S1>
    void op128I(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, RegData dst, S0 src0, S1 src1, SourceLocation loc);

    template <typename S1>
    void op128J(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, RegData dst, RegData src0, S1 src1, FlagRegister predicate, SourceLocation loc);

    template <typename S1>
    void op128K(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, RegData dst, FlagRegister carryOut, RegData src0, S1 src1, FlagRegister carryIn, SourceLocation loc) {
        op128K(opclass, defaultType, mod, dst, carryOut, src0, src1, null, carryIn, loc);
    }

    template <typename S1, typename S2>
    void op128K(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, RegData dst, FlagRegister carryOut, RegData src0, S1 src1, S2 src2, FlagRegister carryIn, SourceLocation loc);

    template <typename S0>
    void op128L(OpcodeClassXe4 opclass, InstructionModifier mod, bool withID, uint8_t cid, bool exp, S0 src0, SourceLocation loc);

    void op128O(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, RegData dst, RegData src0, RegData src1, SourceLocation loc);

    void op128P(OpcodeClassXe4 opclass, InstructionModifier mod, RegData dst, uint64_t cip, SourceLocation loc);

    template <typename S1, typename S2>
    void op128Q(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, RegData dst, RegData src0, S1 src1, S2 src2, SourceLocation loc);

    template <typename D, typename S0>
    void op128R(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, D dst, S0 src0, SourceLocation loc);

    template <typename S1>
    void op128S(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, RegData dst, RegData src0, S1 src1, SourceLocation loc);

    void op64A(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, RegData dst, RegData src0, SourceLocation loc) {
        op64A(opclass, defaultType, mod, dst, src0, null, loc);
    }

    template <typename S1>
    void op64A(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, RegData dst, RegData src0, S1 src1, SourceLocation loc);

    void op64D(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, uint8_t ctrl, RegData dst, RegData src0, RegData src1, SourceLocation loc);

    void op64E(Opcode op, InstructionModifier mod, SyncFunction fc, SourceLocation loc);
    void op64E(Opcode op, InstructionModifier mod, SyncFunction fc, RegData src0, SourceLocation loc);
    void op64E(Opcode op, InstructionModifier mod, SyncFunction fc, Immediate src0, SourceLocation loc);
    void op64E(Opcode op, InstructionModifier mod, SyncFunction fc, std::array<SWSBItem, 5> items, SourceLocation loc);

    template <typename S0, typename S1>
    void op64F(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, RegData dst, S0 src0, S1 src1, SourceLocation loc);

    void op64G(OpcodeClassXe4 opclass, InstructionModifier mod, SourceLocation loc) {
        op64F(opclass, DataType::invalid, mod, null, null, null, loc);
    }
#endif

    static constexpr14 InstructionModifier defaultMods() {
#if XE4
        if (hw >= HW::Xe4) return InstructionModifier{};
#endif
        return GRF::bytes(hw) >> 2;
    }

    inline void unsupported();

#include "ngen_compiler_fix.hpp"

public:
    explicit BinaryCodeGenerator(Product product_, DebugConfig debugConfig = {})
        : product{product_}, debugLine(debugConfig), cancelAutoSWSB_(false), defaultModifier{}, labelManager{},

#if XE3P
                                                     lfsr{this}, shfl{this},
#endif
                                                     sync{this}, load{this}, store{this}, atomic{this}
    {
        _workaround_();
        pushStream(rootStream);
    }

    explicit BinaryCodeGenerator(int stepping_ = 0, DebugConfig debugConfig = {}) : BinaryCodeGenerator({genericProductFamily(hw), stepping_, PlatformType::Unknown}, debugConfig) {}

    ~BinaryCodeGenerator() {
        for (size_t sn = 1; sn < streamStack.size(); sn++)
            delete streamStack[sn];
    }

    std::vector<uint8_t> getCode();
    size_t getRootStreamLength() const { return rootStream.length(); }

    Product getProduct() const { return product; }
    ProductFamily getProductFamily() const { return product.family; }
    int getStepping() const { return product.stepping; }

    void setProduct(Product product_) { product = product_; }
    void setProductFamily(ProductFamily family_) { product.family = family_; }
    void setStepping(int stepping_) { product.stepping = stepping_; }

protected:
    // Configuration.
    void setDefaultNoMask(bool def = true)          { defaultModifier.setWrEn(def); }
    void setDefaultAutoSWSB(bool def = true)        { defaultModifier.setAutoSWSB(def); }
    bool getDefaultNoMask() const                   { return defaultModifier.isWrEn(); }
    bool getDefaultAutoSWSB() const                 { return defaultModifier.isAutoSWSB(); }

#if XE3P
    void setEfficient64Bit(bool def = true)         { useEfficient64Bit = def; }
    bool getEfficient64Bit() const                  { return useEfficient64Bit; }
#endif

    // Stream handling.
    void pushStream()                               { pushStream(new InstructionStream()); }
    void pushStream(InstructionStream *s)           { streamStack.push_back(s); }
    void pushStream(InstructionStream &s)           { pushStream(&s); }

    InstructionStream *popStream();

    void appendStream(InstructionStream *s)         { appendStream(*s); }
    void appendStream(InstructionStream &s)         { streamStack.back()->append(s, labelManager); }
    void appendCurrentStream()                      { InstructionStream *s = popStream(); appendStream(s); delete s; }

    void discardStream()                            { delete popStream(); }

    void requireGRF(int grfs)                       { declaredGRFs = grfs; }

public:
    template <typename String>
    void comment(String)                            {}

    // Registers.
#ifndef NGEN_GLOBAL_REGS
#include "ngen_registers.hpp"
#endif

    // Labels.
    void mark(Label &label)            { streamStack.back()->mark(label, labelManager); }
    void markIfUndefined(Label &label) { if (!label.defined(labelManager)) mark(label); }

    // Instructions.
#if XE4
    template <typename DT = void>
    void abs_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::abs, getDataType<DT>(), mod, dst, src0, loc);
        else
            mov(mod, dst, abs(src0), loc);
    }
#endif
    template <typename DT = void>
    void add(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::add_128A, getDataType<DT>(), mod, dst, src0, src1, loc);
        else
#endif
        opX(Opcode::add, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void add(const InstructionModifier &mod, const RegData &dst,
             const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128AD(OpcodeClassXe4::add_128A, OpcodeClassXe4::add_128D, getDataType<DT>(), mod, dst, src0, src1, loc);
        else
#endif
        opX(Opcode::add, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void addc(const InstructionModifier &mod, const RegData &dst,
              const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128K(OpcodeClassXe4::addc, getDataType<DT>(), mod, dst, FlagRegister(), src0, src1, FlagRegister(), loc);
        else
#endif
        opX(Opcode::addc, getDataType<DT>(), mod | AccWrEn, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void addc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128K(OpcodeClassXe4::addc, getDataType<DT>(), mod, dst, FlagRegister(), src0, src1, FlagRegister(), loc);
        else
#endif
        opX(Opcode::addc, getDataType<DT>(), mod | AccWrEn, dst, src0, src1, loc);
    }
#if XE4
    template <typename DT = void>
    void addc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const FlagRegister &carryIn, SourceLocation loc = {}) {
        op128K(OpcodeClassXe4::addc, getDataType<DT>(), mod, dst, FlagRegister(), src0, src1, carryIn, loc);
    }
    template <typename DT = void>
    void addc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const FlagRegister &carryIn, SourceLocation loc = {}) {
        op128K(OpcodeClassXe4::addc, getDataType<DT>(), mod, dst, FlagRegister(), src0, src1, carryIn, loc);
    }
    template <typename DT = void>
    void addc(const InstructionModifier &mod, const RegData &dst, const FlagRegister &carryOut, const RegData &src0, const RegData &src1, const FlagRegister &carryIn = FlagRegister(), SourceLocation loc = {}) {
        op128K(OpcodeClassXe4::addc, getDataType<DT>(), mod, dst, carryOut, src0, src1, carryIn, loc);
    }
    template <typename DT = void>
    void addc(const InstructionModifier &mod, const RegData &dst, const FlagRegister &carryOut, const RegData &src0, const Immediate &src1, const FlagRegister &carryIn = FlagRegister(), SourceLocation loc = {}) {
        op128K(OpcodeClassXe4::addc, getDataType<DT>(), mod, dst, carryOut, src0, src1, carryIn, loc);
    }
    template <typename DT = void>
    void addc(const RegData &dst, const RegData &src0, const RegData &src1, const FlagRegister &carryIn = FlagRegister(), SourceLocation loc = {}) {
        addc<DT>(InstructionModifier(), dst, src0, src1, carryIn, loc);
    }
    template <typename DT = void>
    void addc(const RegData &dst, const RegData &src0, const Immediate &src1, const FlagRegister &carryIn = FlagRegister(), SourceLocation loc = {}) {
        addc<DT>(InstructionModifier(), dst, src0, src1, carryIn, loc);
    }
#endif
    template <typename DT = void>
    void add3(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        if (hw < HW::XeHP) unsupported();
#if XE4
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::add3, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
        else
#endif
        opX(Opcode::add3, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void add3(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        if (hw < HW::XeHP) unsupported();
        opX(Opcode::add3, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void add3(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        if (hw < HW::XeHP) unsupported();
#if XE4
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::add3, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
        else
#endif
        opX(Opcode::add3, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void add3(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        if (hw < HW::XeHP) unsupported();
        opX(Opcode::add3, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void and_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            bfn2<DT>(mod, 0x88, dst, src0, src1, loc);
        else
#endif
        opX(isGen12 ? Opcode::and_gen12 : Opcode::and_, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void and_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            bfn2<DT>(mod, 0x88, dst, src0, src1, loc);
        else
#endif
        opX(isGen12 ? Opcode::and_gen12 : Opcode::and_, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void asr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::asr, getDataType<DT>(), mod, dst, src0, src1, loc);
        else
#endif
        opX(isGen12 ? Opcode::asr_gen12 : Opcode::asr, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void asr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::asr, getDataType<DT>(), mod, dst, src0, src1, loc);
        else
#endif
        opX(isGen12 ? Opcode::asr_gen12 : Opcode::asr, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void avg(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::avg, getDataType<DT>(), mod, dst, src0, src1, loc);
        else
#endif
        opX(Opcode::avg, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void avg(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::avg, getDataType<DT>(), mod, dst, src0, src1, loc);
        else
#endif
        opX(Opcode::avg, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void bfe(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::bfegen, getDataType<DT>(), mod, dst, src2, src0, src1, loc);
        else
#endif
        opX(isGen12 ? Opcode::bfe_gen12 : Opcode::bfe, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void bfe(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfe_gen12 : Opcode::bfe, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void bfe(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfe_gen12 : Opcode::bfe, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void bfe(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfe_gen12 : Opcode::bfe, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
#if XE4
    template <typename DT = uint32_t>
    void bfe(const InstructionModifier &mod, unsigned width, unsigned offset, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        op128G(OpcodeClassXe4::bfe, getDataType<DT>(), mod, width, offset, dst, src0, null, loc);
    }
#endif
    template <typename DT = void>
    void bfi1(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfi1_gen12 : Opcode::bfi1, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void bfi1(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfi1_gen12 : Opcode::bfi1, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void bfi2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            bfigen(mod, dst, src2, src1, src0, loc);
        else
#endif
        opX(isGen12 ? Opcode::bfi2_gen12 : Opcode::bfi2, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void bfi2(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfi2_gen12 : Opcode::bfi2, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void bfi2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfi2_gen12 : Opcode::bfi2, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void bfi2(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfi2_gen12 : Opcode::bfi2, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
#if XE4
    template <typename DT = uint32_t>
    void bfi(const InstructionModifier &mod, unsigned width, unsigned offset, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        op128G(OpcodeClassXe4::bfi, getDataType<DT>(), mod, width, offset, dst, src0, src1, loc);
    }
    template <typename DT = uint32_t>
    void bfi(const InstructionModifier &mod, unsigned width, unsigned offset, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        op128G(OpcodeClassXe4::bfi, getDataType<DT>(), mod, width, offset, dst, src0, src1, loc);
    }
    template <typename DT = uint32_t>
    void bfigen(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        op128A(OpcodeClassXe4::bfigen, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = uint32_t>
    void bfigen(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &src2, SourceLocation loc = {}) {
        op128A(OpcodeClassXe4::bfigen, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
#endif
    template <typename DT = void>
    void bfn(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        if (hw < HW::XeHP) unsupported();
#if XE4
        if (hardware >= HW::Xe4)
            bfn3<DT>(mod, ctrl, dst, src0, src1, src2, loc);
        else
#endif
        opBfn(Opcode::bfn, getDataType<DT>(), mod, ctrl, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void bfn(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        if (hw < HW::XeHP) unsupported();
        opBfn(Opcode::bfn, getDataType<DT>(), mod, ctrl, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void bfn(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        if (hw < HW::XeHP) unsupported();
#if XE4
        if (hardware >= HW::Xe4)
            bfn3<DT>(mod, ctrl, dst, src0, src1, src2, loc);
        else
#endif
        opBfn(Opcode::bfn, getDataType<DT>(), mod, ctrl, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void bfn(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        if (hw < HW::XeHP) unsupported();
        opBfn(Opcode::bfn, getDataType<DT>(), mod, ctrl, dst, src0, src1, src2, loc);
    }
#if XE4
    template <typename DT = void>
    void bfn2(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        op64D(OpcodeClassXe4::bfn2, getDataType<DT>(), mod, ctrl, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void bfn2(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        bfn3<DT>(mod, ctrl, dst, src0, src1, null, loc);
    }
    template <typename DT = void>
    void bfn3(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        op128E(OpcodeClassXe4::bfn3, getDataType<DT>(), mod, ctrl, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void bfn3(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        op128E(OpcodeClassXe4::bfn3, getDataType<DT>(), mod, ctrl, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void bfn3(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &src2, SourceLocation loc = {}) {
        op128E(OpcodeClassXe4::bfn3, getDataType<DT>(), mod, ctrl, dst, src0, src1, src2, loc);
    }
#endif
    template <typename DT = void>
    void bfrev(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128G(OpcodeClassXe4::bfrev, getDataType<DT>(), mod, 0, 0, dst, src0, NullRegister(), loc);
        else
#endif
        opX(isGen12 ? Opcode::bfrev_gen12 : Opcode::bfrev, getDataType<DT>(), mod, dst, src0, loc);
    }
    void brc(const InstructionModifier &mod, Label &jip, Label &uip, SourceLocation loc = {}) {
        opBranch(Opcode::brc, mod, isGen12 ? null.ud() : ip.d(), jip, uip, loc);
    }
    void brc(const InstructionModifier &mod, RegData src0, SourceLocation loc = {}) {
        src0.setRegion(2, 2, 1);
        opBranch<true, true>(Opcode::brc, mod, isGen12 ? null.ud() : ip.d(), src0, loc);
    }
    void brd(const InstructionModifier &mod, Label &jip, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128B(OpcodeClassXe4::brd, mod, null, jip, loc);
        else
#endif
        opBranch(Opcode::brd, mod, isGen12 ? null.ud() : ip.d(), jip, loc);
    }
    void brd(const InstructionModifier &mod, RegData src0, SourceLocation loc = {}) {
        src0.setRegion(2, 2, 1);
        opBranch<true, true>(Opcode::brd, mod, isGen12 ? null.ud() : ip.d(), src0, loc);
    }
#if XE4
    void brd(const InstructionModifier &mod, Label &jip, bool branchCtrl, SourceLocation loc = {}) {
        auto emod = mod;
        emod.setBranchCtrl(branchCtrl);
        if (branchCtrl && hardware < HW::Xe4) unsupported();
        brd(emod, jip, loc);
    }
    template <typename DT = void>
    void brepgen(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        op128A(OpcodeClassXe4::brepgen, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void brepgen(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        op128A(OpcodeClassXe4::brepgen, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
#endif
    void break_(const InstructionModifier &mod, Label &jip, Label &uip, SourceLocation loc = {}) {
        opBranch(Opcode::break_, mod, null, jip, uip, loc);
    }
    void call(const InstructionModifier &mod, const RegData &dst, Label &jip, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128B(OpcodeClassXe4::call, mod, dst, jip, loc);
        else
#endif
        opCall(Opcode::call, mod, dst, jip, loc);
    }
    void call(const InstructionModifier &mod, const RegData &dst, RegData jip, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128B(OpcodeClassXe4::call, mod, dst, jip, 0, 0, loc);
        else
#endif
        if (isGen12)
            opBranch<true, true>(Opcode::call, mod, dst, jip, loc);
        else {
            jip.setRegion(0, 1, 0);
            opX<true>(Opcode::call, DataType::d, mod, dst, null.ud(0)(0, 1, 0), jip, loc);
        }
    }
#if XE4
    void calla(const InstructionModifier &mod, const RegData &dst, uint64_t jip = 0, SourceLocation loc = {}) {
        if (hardware >= HW::Xe4)
            op128P(OpcodeClassXe4::calla, mod, dst, jip, loc);
        else if (isGen12)
            opBranch<true>(Opcode::calla, mod, dst, jip, loc);
        else
            opX<true>(Opcode::calla, DataType::d, mod, dst, (hw <= HW::Gen9) ? null.ud(0)(2,2,1) : null.ud(0)(0,1,0), Immediate::d(jip), loc);
    }
#else
    void calla(const InstructionModifier &mod, const RegData &dst, int32_t jip, SourceLocation loc = {}) {
        if (isGen12)
            opBranch<true>(Opcode::calla, mod, dst, jip, loc);
        else
            opX<true>(Opcode::calla, DataType::d, mod, dst, (hw <= HW::Gen9) ? null.ud(0)(2,2,1) : null.ud(0)(0,1,0), Immediate::d(jip), loc);
    }
#endif
    void calla(const InstructionModifier &mod, const RegData &dst, RegData jip, SourceLocation loc = {}) {
        if (isGen12)
            opBranch<true, true>(Opcode::calla, mod, dst, jip, loc);
        else {
            jip.setRegion(0, 1, 0);
            opX<true>(Opcode::calla, DataType::d, mod, dst, null.ud(0)(0, 1, 0), jip, loc);
        }
    }
#if XE4
    void callad(const InstructionModifier &mod, const RegData &dst, uint64_t jip = 0, SourceLocation loc = {}) {
        op128P(OpcodeClassXe4::callad, mod, dst, jip, loc);
    }
    void calld(const InstructionModifier &mod, const RegData &dst, Label &jip, SourceLocation loc = {}) {
        op128B(OpcodeClassXe4::calld, mod, dst, jip, loc);
    }
    void calld(const InstructionModifier &mod, const RegData &dst, const RegData &jip, SourceLocation loc = {}) {
        op128B(OpcodeClassXe4::calld, mod, dst, jip, 0, 0, loc);
    }
#endif
    template <typename DT = void>
    void cbit(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::cbit, getDataType<DT>(), mod, dst, src0, loc);
        else
#endif
        opX(Opcode::cbit, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void cmp(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128S(OpcodeClassXe4::cmp_128S, getDataType<DT>(), mod, dst, src0, src1, loc);
        else
#endif
        opX(isGen12 ? Opcode::cmp_gen12 : Opcode::cmp, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void cmp(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128S(OpcodeClassXe4::cmp_128S, getDataType<DT>(), mod, dst, src0, src1, loc);
        else
#endif
        opX(isGen12 ? Opcode::cmp_gen12 : Opcode::cmp, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void cmpn(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128S(OpcodeClassXe4::cmp_128S, getDataType<DT>(), mod, dst, src0, src1, loc);
        else
#endif
        opX(isGen12 ? Opcode::cmpn_gen12 : Opcode::cmpn, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
#if XE4
    void cnvg(const InstructionModifier &mod, uint8_t cid, bool exp, const RegData &src0, SourceLocation loc = {}) {
        op128L(OpcodeClassXe4::cnvg, mod, true, cid, exp, src0, loc);
    }
    void cnvg(const InstructionModifier &mod, uint8_t cid, bool exp, const Immediate &src0, SourceLocation loc = {}) {
        op128L(OpcodeClassXe4::cnvg, mod, true, cid, exp, src0, loc);
    }
    void cnvg(const InstructionModifier &mod, bool exp, const RegData &src0, SourceLocation loc = {}) {
        op128L(OpcodeClassXe4::cnvg, mod, false, 0, exp, src0, loc);
    }
    void cnvg(const InstructionModifier &mod, bool exp, const Immediate &src0, SourceLocation loc = {}) {
        op128L(OpcodeClassXe4::cnvg, mod, false, 0, exp, src0, loc);
    }
#endif
    template <typename DT = void>
    void csel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::csel_gen12 : Opcode::csel, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void csel(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::csel_gen12 : Opcode::csel, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void csel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::csel_gen12 : Opcode::csel, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void csel(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::csel_gen12 : Opcode::csel, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    void cont(const InstructionModifier &mod, Label &jip, Label &uip, SourceLocation loc = {}) {
        opBranch(Opcode::cont, mod, null, jip, uip, loc);
    }
#if XE4
    template <typename DT = void>
    void cvt(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        if (hw >= HW::Xe4)
            op128O(OpcodeClassXe4::cvt, getDataType<DT>(), mod, dst, src0, null, loc);
        else
            mov<DT>(mod, dst, src0, loc);
    }
    template <typename DT = void>
    void cvt2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        op128O(OpcodeClassXe4::cvt2, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
#endif
    template <typename DT = void>
    void dp2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::dp2, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void dp2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::dp2, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void dp3(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::dp3, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void dp3(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::dp3, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void dp4(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::dp4, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void dp4(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::dp4, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        if (hw < HW::Gen12LP) unsupported();
#if XE4
        if (hardware >= HW::Xe4) {
            if (mod.hasExecSize())
                op128Q(OpcodeClassXe4::dp4a, getDataType<DT>(), mod, dst, src1, src2, src0, loc);
            else
                op128Q(OpcodeClassXe4::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
        } else
#endif
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        if (hw < HW::Gen12LP) unsupported();
#if XE4
        if (hardware >= HW::Xe4) {
            if (mod.hasExecSize())
                op128Q(OpcodeClassXe4::dp4a, getDataType<DT>(), mod, dst, src1, src2, src0, loc);
            else
                unsupported();
        } else
#endif
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        if (hw < HW::Gen12LP) unsupported();
#if XE4
        if (hardware >= HW::Xe4) {
            if (mod.hasExecSize())
                op128Q(OpcodeClassXe4::dp4a, getDataType<DT>(), mod, dst, src1, src2, src0, loc);
            else
                op128Q(OpcodeClassXe4::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
        } else
#endif
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        if (hw < HW::Gen12LP) unsupported();
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
#if XE4
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &src2, SourceLocation loc = {}) {
        if (mod.hasExecSize()) unsupported();
        op128Q(OpcodeClassXe4::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
#endif
    template <typename DT = void>
    void dpas(const InstructionModifier &mod, uint8_t sdepth, uint8_t rcount, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opDpas(Opcode::dpas, getDataType<DT>(), mod, sdepth, rcount, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void dpasw(const InstructionModifier &mod, uint8_t sdepth, uint8_t rcount, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opDpas(Opcode::dpasw, getDataType<DT>(), mod, sdepth, rcount, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void dph(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::dph, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void dph(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::dph, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    void else_(InstructionModifier mod, Label &jip, Label &uip, bool branchCtrl, SourceLocation loc = {}) {
        mod.setBranchCtrl(branchCtrl);
        opBranch(Opcode::else_, mod, null, jip, uip, loc);
    }
    void else_(InstructionModifier mod, Label &jip, Label &uip, SourceLocation loc = {}) {
        else_(mod, jip, uip, false, loc);
    }
    void else_(InstructionModifier mod, Label &jip, SourceLocation loc = {}) {
        else_(mod, jip, jip, false, loc);
    }
#if XE4
    template <typename DT = void>
    void emcos(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        if (hw >= HW::Xe4)
            op128A(OpcodeClassXe4::emcos, getDataType<DT>(), mod, dst, src0, loc);
        else
            math<DT>(mod, MathFunction::cos, dst, src0, loc);
    }
    template <typename DT = void>
    void emexp2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        if (hw >= HW::Xe4)
            op128A(OpcodeClassXe4::emexp2, getDataType<DT>(), mod, dst, src0, loc);
        else
            math<DT>(mod, MathFunction::exp, dst, src0, loc);
    }
    template <typename DT = void>
    void eminv(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        if (hw >= HW::Xe4)
            op128A(OpcodeClassXe4::eminv, getDataType<DT>(), mod, dst, src0, loc);
        else
            math<DT>(mod, MathFunction::inv, dst, src0, loc);
    }
    template <typename DT = void>
    void eminvm(const InstructionModifier &mod, const ExtendedReg &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        if (hardware >= HW::Xe4)
            op128H(OpcodeClassXe4::eminvm, getDataType<DT>(), mod, dst, src0 | nomme, src1 | nomme, NullRegister() | nomme, loc);
        else
            math<DT>(mod, MathFunction::invm, dst, src0 | nomme, src1 | nomme, loc);
    }
    template <typename DT = void>
    void emlog2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        if (hw >= HW::Xe4)
            op128A(OpcodeClassXe4::emlog2, getDataType<DT>(), mod, dst, src0, loc);
        else
            math<DT>(mod, MathFunction::log, dst, src0, loc);
    }
    template <typename DT = void>
    void emrsqt(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        if (hw >= HW::Xe4)
            op128A(OpcodeClassXe4::emrsqt, getDataType<DT>(), mod, dst, src0, loc);
        else
            math<DT>(mod, MathFunction::rsqt, dst, src0, loc);
    }
    template <typename DT = void>
    void emrsqtm(const InstructionModifier &mod, const ExtendedReg &dst, const RegData &src0, SourceLocation loc = {}) {
        if (hardware >= HW::Xe4)
            op128H(OpcodeClassXe4::emrsqtm, getDataType<DT>(), mod, dst, src0 | nomme, NullRegister() | nomme, NullRegister() | nomme, loc);
        else
            math<DT>(mod, MathFunction::rsqtm, dst, src0 | nomme, loc);
    }
    template <typename DT = void>
    void emsin(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        if (hw >= HW::Xe4)
            op128A(OpcodeClassXe4::emsin, getDataType<DT>(), mod, dst, src0, loc);
        else
            math<DT>(mod, MathFunction::sin, dst, src0, loc);
    }
    template <typename DT = void>
    void emsgmd(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        if (hw >= HW::Xe4)
            op128A(OpcodeClassXe4::emsgmd, getDataType<DT>(), mod, dst, src0, loc);
        else
            math<DT>(mod, MathFunction::sigm, dst, src0, loc);
    }
    template <typename DT = void>
    void emsqt(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        if (hw >= HW::Xe4)
            op128A(OpcodeClassXe4::emsqt, getDataType<DT>(), mod, dst, src0, loc);
        else
            math<DT>(mod, MathFunction::sqt, dst, src0, loc);
    }
    template <typename DT = void>
    void emtanh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        if (hw >= HW::Xe4)
            op128A(OpcodeClassXe4::emtanh, getDataType<DT>(), mod, dst, src0, loc);
        else
            math<DT>(mod, MathFunction::tanh, dst, src0, loc);
    }
#endif
    void endif(const InstructionModifier &mod, Label &jip, SourceLocation loc = {}) {
        opBranch(Opcode::endif, mod, null, jip, loc);
    }
    void endif(const InstructionModifier &mod, SourceLocation loc = {}) {
        opBranch(Opcode::endif, mod, null, sizeof(Instruction8), loc);
    }
    template <typename DT = void>
    void fbh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::fbh, getDataType<DT>(), mod, dst, src0, loc);
        else
#endif
        opX(Opcode::fbh, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void fbl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::fbl, getDataType<DT>(), mod, dst, src0, loc);
        else
#endif
        opX(Opcode::fbl, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void frc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::frc, getDataType<DT>(), mod, dst, src0, loc);
        else
#endif
        opX(Opcode::frc, getDataType<DT>(), mod, dst, src0, loc);
    }
    void goto_(InstructionModifier mod, Label &jip, Label &uip, bool branchCtrl, SourceLocation loc = {}) {
        mod.setBranchCtrl(branchCtrl);
#if XE4
        if (hardware >= HW::Xe4)
            op128B(OpcodeClassXe4::goto_, mod, null, jip, uip, loc);
        else
#endif
        opBranch(Opcode::goto_, mod, null, jip, uip, loc);
    }
    void goto_(const InstructionModifier &mod, Label &jip, Label &uip, SourceLocation loc = {}) {
        goto_(mod, jip, uip, false, loc);
    }
    void goto_(const InstructionModifier &mod, Label &jip, SourceLocation loc = {}) {
        goto_(mod, jip, jip, loc);
    }
    void halt(const InstructionModifier &mod, Label &jip, Label &uip, SourceLocation loc = {}) {
        opBranch(Opcode::halt, mod, null, jip, uip, loc);
    }
    void halt(const InstructionModifier &mod, Label &jip, SourceLocation loc = {}) {
        halt(mod, jip, jip, loc);
    }
    void if_(InstructionModifier mod, Label &jip, Label &uip, bool branchCtrl, SourceLocation loc = {}) {
        mod.setBranchCtrl(branchCtrl);
        opBranch(Opcode::if_, mod, null, jip, uip, loc);
    }
    void if_(const InstructionModifier &mod, Label &jip, Label &uip, SourceLocation loc = {}) {
        if_(mod, jip, uip, false, loc);
    }
    void if_(const InstructionModifier &mod, Label &jip, SourceLocation loc = {}) {
        if_(mod, jip, jip, false, loc);
    }
    void illegal(SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::illegal, DataType::invalid, InstructionModifier(), null, null, loc);
        else
#endif
        opX(Opcode::illegal, DataType::invalid, InstructionModifier(), null, null, null, loc);
    }
    void join(InstructionModifier mod, Label &jip, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128B(OpcodeClassXe4::join, mod, null, jip, loc);
        else
#endif
        opBranch(Opcode::join, mod, null, jip, loc);
    }
    void join(InstructionModifier mod, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128B(OpcodeClassXe4::join, mod, null, null, sizeof(InstructionXe4), 0, loc);
        else
#endif
        opBranch(Opcode::join, mod, null, sizeof(Instruction8), loc);
    }
    void jmpi(const InstructionModifier &mod, Label &jip, SourceLocation loc = {}) {
        auto dst = isGen12 ? ARF(null) : ARF(ip);
#if XE4
        if (hardware >= HW::Xe4)
            op128B(OpcodeClassXe4::jmpi, mod, null, jip, loc);
        else
#endif
        opJmpi(Opcode::jmpi, mod, dst, dst, jip, loc);
    }
    void jmpi(const InstructionModifier &mod, const RegData &jip, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (!isGen12 && jip.getType() != DataType::d && jip.getType() != DataType::invalid)
            throw invalid_type_exception();
#endif
#if XE4
        if (hardware >= HW::Xe4)
            op128B(OpcodeClassXe4::jmpi, mod, null, jip, 0, 0, loc);
        else
#endif
        if (isGen12)
            opBranch<true, false>(Opcode::jmpi, mod, null, jip, loc);
        else
            opX(Opcode::jmpi, DataType::d, mod, ip, ip, jip, loc);
    }
    template <typename DT = void>
    void line(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        if (hw >= HW::Gen11) unsupported();
        opX(Opcode::line, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void line(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        if (hw >= HW::Gen11) unsupported();
        opX(Opcode::line, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void lrp(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(Opcode::lrp, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void lzd(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(Opcode::lzd, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void lzd(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        opX(Opcode::lzd, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void mac(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE3P
#ifdef NGEN_SAFE
        if (hardware >= HW::XE3P_35_10) unsupported();
#endif
#endif
        opX(Opcode::mac, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void mac(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE3P
#ifdef NGEN_SAFE
        if (hardware >= HW::XE3P_35_10) unsupported();
#endif
#endif
        opX(Opcode::mac, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void mach(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE3P
#ifdef NGEN_SAFE
        if (hardware >= HW::XE3P_35_10) unsupported();
#endif
#endif
        opX(Opcode::mach, getDataType<DT>(), (hw >= HW::XeHPC) ? mod : (mod | AccWrEn), dst, src0, src1, loc);
    }
    template <typename DT = void>
    void mach(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE3P
#ifdef NGEN_SAFE
        if (hardware >= HW::XE3P_35_10) unsupported();
#endif
#endif
        opX(Opcode::mach, getDataType<DT>(), (hw >= HW::XeHPC) ? mod : (mod | AccWrEn), dst, src0, src1, loc);
    }
    template <typename DT = void>
    void macl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
#if XE3P
        if (hardware >= HW::XE3P_35_10) unsupported();
#endif
        if (hw < HW::Gen10) unsupported();
#endif
        opX((hw >= HW::XeHPC) ? Opcode::macl : Opcode::mach, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void macl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
#if XE3P
        if (hardware >= HW::XE3P_35_10) unsupported();
#endif
        if (hw < HW::Gen10) unsupported();
#endif
        opX((hw >= HW::XeHPC) ? Opcode::macl : Opcode::mach, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4) {
            if (mod.hasExecSize())
                op128A(OpcodeClassXe4::mad_128A, getDataType<DT>(), mod, dst, src1, src2, src0, loc);
            else
                op128A(OpcodeClassXe4::mad_128A, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
        } else
#endif
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4) {
            if (mod.hasExecSize())
                op128AD(OpcodeClassXe4::mad_128A, OpcodeClassXe4::mad_128D, getDataType<DT>(), mod, dst, src1, src2, src0, loc);
            else
                unsupported();
        } else
#endif
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            if (mod.hasExecSize())
                op128AD(OpcodeClassXe4::mad_128A, OpcodeClassXe4::mad_128D, getDataType<DT>(), mod, dst, src1, src2, src0, loc);
            else
                op128AD(OpcodeClassXe4::mad_128A, OpcodeClassXe4::mad_128D, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
        else
#endif
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4) {
            if (mod.hasExecSize())
                op128AD(OpcodeClassXe4::mad_128A, OpcodeClassXe4::mad_128D, getDataType<DT>(), mod, dst, src1, src2, src0, loc);
            else
                unsupported();
        } else
#endif
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
#if XE4
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &src2, SourceLocation loc = {}) {
        if (mod.hasExecSize()) unsupported();
        op128AD(OpcodeClassXe4::mad_128A, OpcodeClassXe4::mad_128D, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const Immediate &src2, SourceLocation loc = {}) {
        if (mod.hasExecSize()) unsupported();
        op128AD(OpcodeClassXe4::mad_128A, OpcodeClassXe4::mad_128D, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void madc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, const FlagRegister &carryIn = FlagRegister(), SourceLocation loc = {}) {
        op128K(OpcodeClassXe4::madc, getDataType<DT>(), mod, dst, FlagRegister(), src0, src1, src2, carryIn, loc);
    }
    template <typename DT = void>
    void madc(const InstructionModifier &mod, const RegData &dst, const FlagRegister &carryOut, const RegData &src0, const RegData &src1, const RegData &src2, const FlagRegister &carryIn = FlagRegister(), SourceLocation loc = {}) {
        op128K(OpcodeClassXe4::madc, getDataType<DT>(), mod, dst, carryOut, src0, src1, src2, carryIn, loc);
    }
    template <typename DT = void>
    void madlh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opMullh(OpcodeClassXe4::madlh, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void madlh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opMullh(OpcodeClassXe4::madlh, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void madlh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &src2, SourceLocation loc = {}) {
        opMullh(OpcodeClassXe4::madlh, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void madlh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const Immediate &src2, SourceLocation loc = {}) {
        opMullh(OpcodeClassXe4::madlh, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
#endif
    template <typename DT = void, HW hw_ = hw>
    typename std::enable_if<hwLE(hw_, HW::Gen9)>::type
    madm(const InstructionModifier &mod, const ExtendedReg &dst, const ExtendedReg &src0, const ExtendedReg &src1, const ExtendedReg &src2, SourceLocation loc = {}) {
        opX(Opcode::madm, getDataType<DT>(), mod, extToAlign16(dst), extToAlign16(src0), extToAlign16(src1), extToAlign16(src2), loc);
    }
    template <typename DT = void, HW hw_ = hw>
    typename std::enable_if<hwGT(hw_, HW::Gen9)>::type
    madm(const InstructionModifier &mod, const ExtendedReg &dst, ExtendedReg src0, ExtendedReg src1, const ExtendedReg &src2, SourceLocation loc = {}) {
        src0.getBase().setRegion(4,4,1);
        src1.getBase().setRegion(4,4,1);
#if XE4
        if (hardware >= HW::Xe4)
            op128H(OpcodeClassXe4::madm, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
        else
#endif
        opX(Opcode::madm, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void math(const InstructionModifier &mod, MathFunction fc, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (mathArgCount(hw, fc) != 1) throw invalid_operand_count_exception();
#endif
#if XE4
        if (hardware >= HW::Xe4) switch (fc) {
            case MathFunction::cos:  emcos<DT>(mod, dst, src0); break;
            case MathFunction::exp:  emexp2<DT>(mod, dst, src0); break;
            case MathFunction::inv:  eminv<DT>(mod, dst, src0); break;
            case MathFunction::log:  emlog2<DT>(mod, dst, src0); break;
            case MathFunction::rsqt: emrsqt<DT>(mod, dst, src0); break;
            case MathFunction::sigm: emsgmd<DT>(mod, dst, src0); break;
            case MathFunction::sin:  emsin<DT>(mod, dst, src0); break;
            case MathFunction::sqt:  emsqt<DT>(mod, dst, src0); break;
            case MathFunction::tanh: emtanh<DT>(mod, dst, src0); break;
            default: unsupported();
        } else
#endif
        opMath(Opcode::math, getDataType<DT>(), mod, fc, dst, src0, loc);
    }
    template <typename DT = void>
    void math(const InstructionModifier &mod, MathFunction fc, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (mathArgCount(hw, fc) != 2) throw invalid_operand_count_exception();
#endif
        opMath(Opcode::math, getDataType<DT>(), mod, fc, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void math(const InstructionModifier &mod, MathFunction fc, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (fc == MathFunction::invm || fc == MathFunction::rsqtm) throw invalid_operand_exception();
#endif
        opMath(Opcode::math, getDataType<DT>(), mod, fc, dst, src0, src1.forceInt32(), loc);
    }
    template <typename DT = void, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen11)>::type
    math(const InstructionModifier &mod, MathFunction fc, const ExtendedReg &dst, const ExtendedReg &src0, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (fc != MathFunction::rsqtm) throw invalid_operand_exception();
#endif
        opMath(Opcode::math, getDataType<DT>(), mod, fc, extToAlign16(dst), extToAlign16(src0), loc);
    }
    template <typename DT = void, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen11)>::type
    math(const InstructionModifier &mod, MathFunction fc, const ExtendedReg &dst, ExtendedReg src0, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (fc != MathFunction::rsqtm) throw invalid_operand_exception();
#endif
        if (hw == HW::Gen11)
            src0.getBase().setRegion(2,2,1);
        else
            src0.getBase().setRegion(1,1,0);
#if XE4
        if (hardware >= HW::Xe4)
            emrsqtm<DT>(mod, dst, src0.getBase());
        else
#endif
        opMath(Opcode::math, getDataType<DT>(), mod, fc, dst, src0, loc);
    }
    template <typename DT = void, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen11)>::type
    math(const InstructionModifier &mod, MathFunction fc, const ExtendedReg &dst, const ExtendedReg &src0, const ExtendedReg &src1, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (fc != MathFunction::invm) throw invalid_operand_exception();
#endif
        opMath(Opcode::math, getDataType<DT>(), mod, fc, extToAlign16(dst), extToAlign16(src0), extToAlign16(src1), loc);
    }
    template <typename DT = void, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen11)>::type
    math(const InstructionModifier &mod, MathFunction fc, const ExtendedReg &dst, ExtendedReg src0, ExtendedReg src1, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (fc != MathFunction::invm) throw invalid_operand_exception();
#endif
        if (hw == HW::Gen11) {
            src0.getBase().setRegion(2,2,1);
            src1.getBase().setRegion(2,2,1);
        } else {
            src0.getBase().setRegion(1,1,0);
            src1.getBase().setRegion(1,1,0);
        }
#if XE4
        if (hardware >= HW::Xe4)
            eminvm<DT>(mod, dst, src0.getBase(), src1.getBase());
        else
#endif
        opMath(Opcode::math, getDataType<DT>(), mod, fc, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void max_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::max_, getDataType<DT>(), mod, dst, src0, src1, loc);
        else
#endif
        sel<DT>(mod | ge | f0[0], dst, src0, src1, loc);
    }
    template <typename DT = void>
    void max_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::max_, getDataType<DT>(), mod, dst, src0, src1, loc);
        else
#endif
        sel<DT>(mod | ge | f0[0], dst, src0, src1, loc);
    }
    template <typename DT = void>
    void min_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::min_, getDataType<DT>(), mod, dst, src0, src1, loc);
        else
#endif
        sel<DT>(mod | lt | f0[0], dst, src0, src1, loc);
    }
    template <typename DT = void>
    void min_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::min_, getDataType<DT>(), mod, dst, src0, src1, loc);
        else
#endif
        sel<DT>(mod | lt | f0[0], dst, src0, src1, loc);
    }
    template <typename DT = void>
    void mov(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4) {
            auto dt = dst.getType(), st = src0.getType();
            if (dt == DataType::invalid) dt = getDataType<DT>();
            if (st == DataType::invalid) st = getDataType<DT>();
            if (dt == st || dt == DataType::invalid || st == DataType::invalid)
                op128R(OpcodeClassXe4::mov_128R, getDataType<DT>(), mod, dst, src0, loc);
            else
                cvt<DT>(mod, dst, src0, loc);
        } else
#endif
        opX(isGen12 ? Opcode::mov_gen12 : Opcode::mov, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void mov(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128R(OpcodeClassXe4::mov_128R, getDataType<DT>(), mod, dst, src0, loc);
        else
#endif
        opX(isGen12 ? Opcode::mov_gen12 : Opcode::mov, getDataType<DT>(), mod, dst, src0, loc);
    }
#if XE4
    template <typename DT = void>
    void movb(const InstructionModifier &mod, RegData dst, RegData src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        canonicalizeSRF(dst);
        canonicalizeSRF(src0);
#ifdef NGEN_SAFE
        if (dst.isSRF() == src0.isSRF())
            throw invalid_operand_exception();
#endif
        op128I(OpcodeClassXe4::movb, getDataType<DT>(), mod, dst, src0, Immediate::ud(lanemask), loc);
    }
    template <typename DT = void>
    void movb(const InstructionModifier &mod, RegData dst, RegData src0, RegData lanemask, SourceLocation loc = {}) {
        canonicalizeSRF(dst);
        canonicalizeSRF(src0);
#ifdef NGEN_SAFE
        if (dst.isSRF() == src0.isSRF())
            throw invalid_operand_exception();
#endif
        lanemask.setType(DataType::b32);
        op128I(OpcodeClassXe4::movb, getDataType<DT>(), mod, dst, src0, lanemask, loc);
    }
    template <typename DT = void>
    void movb(const InstructionModifier &mod, RegData dst, Immediate src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        op128I(OpcodeClassXe4::movb, getDataType<DT>(), mod, dst, src0, Immediate::ud(lanemask), loc);
    }
    template <typename DT = void>
    void movb(const InstructionModifier &mod, RegData dst, Immediate src0, RegData lanemask, SourceLocation loc = {}) {
        lanemask.setType(DataType::b32);
        op128I(OpcodeClassXe4::movb, getDataType<DT>(), mod, dst, src0, lanemask, loc);
    }
    template <typename DT = void>
    void movg(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        validateIndXe4(src0);
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::movg, getDataType<DT>(), mod, dst, src0.getIndirectBaseRegXe4(), src0.getIndirectRegXe4(), loc);
        else
            mov(mod, dst, src0);
    }
    template <typename DT = void>
    void movs(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        validateIndXe4(dst);
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::movs, getDataType<DT>(), mod, dst.getIndirectBaseRegXe4(), src0, dst.getIndirectRegXe4(), loc);
        else
            mov(mod, dst, src0);
    }
#endif
    template <typename DT = void>
    void movi(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (!src0.isIndirect()) throw invalid_address_mode_exception();
#endif
        if (hardware >= HW::Gen10)
            movi<DT>(mod, dst, src0, null.ud(0)(1,1,0));
        else
            opX(isGen12 ? Opcode::movi_gen12 : Opcode::movi, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void movi(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (hardware < HW::Gen10) throw unsupported_instruction();
        if (!src0.isIndirect()) throw invalid_address_mode_exception();
#endif
        opX(isGen12 ? Opcode::movi_gen12 : Opcode::movi, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void movi(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (hardware < HW::Gen10) throw unsupported_instruction();
#endif
        opX(isGen12 ? Opcode::movi_gen12 : Opcode::movi, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
#if XE4
    template <typename DT = uint32_t>
    void msk(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        op64F(OpcodeClassXe4::msk, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = uint32_t>
    void msk(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t src1, SourceLocation loc = {}) {
        op64F(OpcodeClassXe4::msk, getDataType<DT>(), mod, dst, src0, Immediate::ud(src1), loc);
    }
    template <typename DT = uint32_t>
    void msk(const InstructionModifier &mod, const RegData &dst, uint32_t src0, const RegData &src1, SourceLocation loc = {}) {
        op64F(OpcodeClassXe4::msk, getDataType<DT>(), mod, dst, Immediate::ud(src0), src1, loc);
    }
    template <typename DT = uint32_t>
    void msk(const InstructionModifier &mod, const RegData &dst, uint32_t src0, uint32_t src1, SourceLocation loc = {}) {
        op64F(OpcodeClassXe4::msk, getDataType<DT>(),  mod, dst, Immediate::ud(src0), Immediate::ud(src1), loc);
    }
#endif
    template <typename DT = void>
    void mul(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::mul_128A, getDataType<DT>(), mod, dst, src0, src1, loc);
        else
#endif
        opX(Opcode::mul, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void mul(const InstructionModifier &mod, const RegData &dst, const RegData &src0, Immediate src1, SourceLocation loc = {}) {
        if (dst.getBytes() == 8)
            src1 = src1.forceInt32();
#if XE4
        if (hardware >= HW::Xe4)
            op128AD(OpcodeClassXe4::mul_128A, OpcodeClassXe4::mul_128D, getDataType<DT>(), mod, dst, src0, src1, loc);
        else
#endif
        opX(Opcode::mul, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
#if XE3P
    template <typename DT = void>
    void mullh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opMullh(OpcodeClassXe4::mullh, getDataType<DT>(), mod, dst, src0, src1, null, loc);
        else
#endif
        opX(Opcode::mullh, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void mullh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, Immediate src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opMullh(OpcodeClassXe4::mullh, getDataType<DT>(), mod, dst, src0, src1, null, loc);
        else
#endif
        opX(Opcode::mullh, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
#endif /* XE3P */
    void nop(SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            nop128(loc);
        else
#endif
        opNop(isGen12 ? Opcode::nop_gen12 : Opcode::nop, loc);
    }
    void nop(const InstructionModifier &mod, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::nop_gen12 : Opcode::nop, DataType::invalid, mod, null, null, null, loc);
    }
#if XE4
    void nop128(SourceLocation loc = {}) {
        op128A(OpcodeClassXe4::nop128, DataType::invalid, InstructionModifier(), null, null, loc);
    }
    void nop64(SourceLocation loc = {}) {
        op64A(OpcodeClassXe4::nop64, DataType::invalid, InstructionModifier(), null, null, loc);
    }
#endif
    template <typename DT = void>
    void not_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            bfn2<DT>(mod, 0x55, dst, src0, null, loc);
        else
#endif
        opX(isGen12 ? Opcode::not_gen12 : Opcode::not_, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void or_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            bfn2<DT>(mod, 0xEE, dst, src0, src1, loc);
        else
#endif
        opX(isGen12 ? Opcode::or_gen12 : Opcode::or_, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void or_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            bfn2<DT>(mod, 0xEE, dst, src0, src1, loc);
        else
#endif
        opX(isGen12 ? Opcode::or_gen12 : Opcode::or_, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void pln(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        if (hw >= HW::Gen11) unsupported();
        opX(Opcode::pln, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
#if XE4
    template <typename DT = uint32_t>
    void redand(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &lanemask, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::redand, getDataType<DT>(), mod, dst, src0, lanemask, loc);
    }
    template <typename DT = uint32_t>
    void redand(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::redand, getDataType<DT>(), mod, dst, src0, Immediate::ud(lanemask), loc);
    }
    template <typename DT = uint32_t>
    void redfirst(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &lanemask, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::redfirst, getDataType<DT>(), mod, dst, src0, lanemask, loc);
    }
    template <typename DT = uint32_t>
    void redfirst(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::redfirst, getDataType<DT>(), mod, dst, src0, Immediate::ud(lanemask), loc);
    }
    template <typename DT = uint32_t>
    void redfirstidx(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &lanemask, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::redfirstidx, getDataType<DT>(), mod, dst, src0, lanemask, loc);
    }
    template <typename DT = uint32_t>
    void redfirstidx(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::redfirstidx, getDataType<DT>(), mod, dst, src0, Immediate::ud(lanemask), loc);
    }
    template <typename DT = void>
    void redmax(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &lanemask, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::redmax, getDataType<DT>(), mod, dst, src0, lanemask, loc);
    }
    template <typename DT = void>
    void redmax(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::redmax, getDataType<DT>(), mod, dst, src0, Immediate::ud(lanemask), loc);
    }
    template <typename DT = void>
    void redmin(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &lanemask, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::redmin, getDataType<DT>(), mod, dst, src0, lanemask, loc);
    }
    template <typename DT = void>
    void redmin(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::redmin, getDataType<DT>(), mod, dst, src0, Immediate::ud(lanemask), loc);
    }
    template <typename DT = uint32_t>
    void redor(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &lanemask, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::redor, getDataType<DT>(), mod, dst, src0, lanemask, loc);
    }
    template <typename DT = uint32_t>
    void redor(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::redor, getDataType<DT>(), mod, dst, src0, Immediate::ud(lanemask), loc);
    }
    template <typename DT = void>
    void redsum(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &lanemask, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::redsum, getDataType<DT>(), mod, dst, src0, lanemask, loc);
    }
    template <typename DT = void>
    void redsum(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::redsum, getDataType<DT>(), mod, dst, src0, Immediate::ud(lanemask), loc);
    }
    template <typename DT = uint32_t>
    void redxor(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &lanemask, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::redxor, getDataType<DT>(), mod, dst, src0, lanemask, loc);
    }
    template <typename DT = uint32_t>
    void redxor(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::redxor, getDataType<DT>(), mod, dst, src0, Immediate::ud(lanemask), loc);
    }
#endif
    void ret(const InstructionModifier &mod, RegData src0, SourceLocation loc = {}) {
        src0.setRegion(2,2,1);
#if XE4
        if (hardware >= HW::Xe4)
            op128B(OpcodeClassXe4::ret, mod, null, src0, 0, 0, loc);
        else
#endif
        if (isGen12)
            opBranch<true, true>(Opcode::ret, mod, null, src0, loc);
        else
            opX<true>(Opcode::ret, DataType::ud, mod, null, src0, loc);
    }
#if XE4
    void retd(const InstructionModifier &mod, const RegData &src0, SourceLocation loc = {}) {
        op128B(OpcodeClassXe4::retd, mod, null, src0, 0, 0, loc);
    }
    template <typename DT = void>
    void rnd(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        op128A(OpcodeClassXe4::rnd, getDataType<DT>(), mod, dst, src0, loc);
    }
#endif
    template <typename DT = void>
    void rndd(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            rnd<DT>(mod | rd, dst, src0, loc);
        else
#endif
        opX(Opcode::rndd, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void rnde(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            rnd<DT>(mod | rne, dst, src0, loc);
        else
#endif
        opX(Opcode::rnde, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void rndu(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            rnd<DT>(mod | ru, dst, src0, loc);
        else
#endif
        opX(Opcode::rndu, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void rndz(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            rnd<DT>(mod | rtz, dst, src0, loc);
        else
#endif
        opX(Opcode::rndz, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void rol(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::rol, getDataType<DT>(), mod, dst, src0, src1, loc);
        else
#endif
        opX(isGen12 ? Opcode::rol_gen12 : Opcode::rol, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void rol(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::rol, getDataType<DT>(), mod, dst, src0, src1, loc);
        else
#endif
        opX(isGen12 ? Opcode::rol_gen12 : Opcode::rol, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void ror(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::ror, getDataType<DT>(), mod, dst, src0, src1, loc);
        else
#endif
        opX(isGen12 ? Opcode::ror_gen12 : Opcode::ror, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void ror(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::ror, getDataType<DT>(), mod, dst, src0, src1, loc);
        else
#endif
        opX(isGen12 ? Opcode::ror_gen12 : Opcode::ror, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void sad2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        if (hw >= HW::Gen12LP) unsupported();
        opX(Opcode::sad2, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void sad2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        if (hw >= HW::Gen12LP) unsupported();
        opX(Opcode::sad2, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void sada2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        if (hw >= HW::Gen12LP) unsupported();
        opX(Opcode::sada2, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void sada2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        if (hw >= HW::Gen12LP) unsupported();
        opX(Opcode::sada2, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void sel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::sel_gen12 : Opcode::sel, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void sel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::sel_gen12 : Opcode::sel, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
#if XE4
    template <typename DT = uint32_t>
    void sel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, FlagRegister predicate, SourceLocation loc = {}) {
        if (hardware >= HW::Xe4)
            op128J(OpcodeClassXe4::sel, getDataType<DT>(), mod, dst, src0, src1, predicate, loc);
        else
            sel(mod | predicate, dst, src0, src1, loc);
    }
    template <typename DT = uint32_t>
    void sel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, FlagRegister predicate, SourceLocation loc = {}) {
        if (hardware >= HW::Xe4)
            op128J(OpcodeClassXe4::sel, getDataType<DT>(), mod, dst, src0, src1, predicate, loc);
        else
            sel(mod | predicate, dst, src0, src1, loc);
    }
#endif

    /* Gen12-style sends */
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(Opcode::send, mod, sf, dst, src0, src1, -1, exdesc, desc, loc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(Opcode::send, mod, sf, dst, src0, src1, -1, exdesc, desc, loc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const GRFRange &src1, const RegData &exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(Opcode::send, mod, sf, dst, src0, src1[0], src1.getLen(), exdesc, desc, loc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(Opcode::send, mod, sf, dst, src0, src1, -1, exdesc, desc, loc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(Opcode::send, mod, sf, dst, src0, src1, -1, exdesc, desc, loc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const GRFRange &src1, const RegData &exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(Opcode::send, mod, sf, dst, src0, src1[0], src1.getLen(), exdesc, desc, loc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(Opcode::send, mod, sf, dst, src0, NullRegister(), 0, exdesc, desc, loc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(Opcode::send, mod, sf, dst, src0, NullRegister(), 0, exdesc, desc, loc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(Opcode::send, mod, sf, dst, src0, NullRegister(), 0, exdesc, desc, loc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(Opcode::send, mod, sf, dst, src0, NullRegister(), 0, exdesc, desc, loc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(Opcode::sendc, mod, sf, dst, src0, src1, -1, exdesc, desc, loc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(Opcode::sendc, mod, sf, dst, src0, src1, -1, exdesc, desc, loc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const GRFRange &src1, const RegData &exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(Opcode::sendc, mod, sf, dst, src0, src1[0], src1.getLen(), exdesc, desc, loc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(Opcode::sendc, mod, sf, dst, src0, src1, -1, exdesc, desc, loc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(Opcode::sendc, mod, sf, dst, src0, src1, -1, exdesc, desc, loc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const GRFRange &src1, const RegData &exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(Opcode::sendc, mod, sf, dst, src0, src1[0], src1.getLen(), exdesc, desc, loc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(Opcode::sendc, mod, sf, dst, src0, NullRegister(), 0, exdesc, desc, loc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(Opcode::sendc, mod, sf, dst, src0, NullRegister(), 0, exdesc, desc, loc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(Opcode::sendc, mod, sf, dst, src0, NullRegister(), 0, exdesc, desc, loc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(Opcode::sendc, mod, sf, dst, src0, NullRegister(), 0, exdesc, desc, loc);
    }
    /* Pre-Gen12-style sends; also supported on Gen12. */
    void send(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(Opcode::send, mod, dst, src0, exdesc, desc, loc);
    }
    void send(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(Opcode::send, mod, dst, src0, exdesc, desc, loc);
    }
    void sendc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(Opcode::sendc, mod, dst, src0, exdesc, desc, loc);
    }
    void sendc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(Opcode::sendc, mod, dst, src0, exdesc, desc, loc);
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSends(Opcode::sends, mod, dst, src0, src1, exdesc, desc, loc);
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSends(Opcode::sends, mod, dst, src0, src1, exdesc, desc, loc);
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSends(Opcode::sends, mod, dst, src0, src1, exdesc, desc, loc);
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSends(Opcode::sends, mod, dst, src0, src1, exdesc, desc, loc);
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSends(Opcode::sendsc, mod, dst, src0, src1, exdesc, desc, loc);
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSends(Opcode::sendsc, mod, dst, src0, src1, exdesc, desc, loc);
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSends(Opcode::sendsc, mod, dst, src0, src1, exdesc, desc, loc);
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSends(Opcode::sendsc, mod, dst, src0, src1, exdesc, desc, loc);
    }
#if XE3P
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, NullRegister(), NullRegister(), desc, loc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, ind0, NullRegister(), desc, loc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, ind0, ind1, desc, loc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegisterRange &src1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), NullRegister(), NullRegister(), desc, loc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegisterRange &src1, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), ind0, NullRegister(), desc, loc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegisterRange &src1, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), ind0, ind1, desc, loc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src0Len, NullRegister(), 0, NullRegister(), NullRegister(), desc, loc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src0Len, NullRegister(), 0, ind0, NullRegister(), desc, loc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src0Len, NullRegister(), 0, ind0, ind1, desc, loc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, NullRegister(), NullRegister(), desc, loc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, ind0, NullRegister(), desc, loc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, ind0, ind1, desc, loc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegisterRange &src1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), NullRegister(), NullRegister(), desc, loc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegisterRange &src1, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), ind0, NullRegister(), desc, loc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegisterRange &src1, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), ind0, ind1, desc, loc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src0Len, NullRegister(), 0, NullRegister(), NullRegister(), desc, loc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src0Len, NullRegister(), 0, ind0, NullRegister(), desc, loc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src0Len, NullRegister(), 0, ind0, ind1, desc, loc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, NullRegister(), NullRegister(), desc, loc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, ind0, NullRegister(), desc, loc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, ind0, ind1, desc, loc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegisterRange &src1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), NullRegister(), NullRegister(), desc, loc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegisterRange &src1, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), ind0, NullRegister(), desc, loc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegisterRange &src1, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), ind0, ind1, desc, loc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src0Len, NullRegister(), 0, NullRegister(), NullRegister(), desc, loc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src0Len, NullRegister(), 0, ind0, NullRegister(), desc, loc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src0Len, NullRegister(), 0, ind0, ind1, desc, loc);
    }
    void sendgxc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgxc, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, NullRegister(), NullRegister(), desc, loc);
    }
    void sendgxc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgxc, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, ind0, NullRegister(), desc, loc);
    }
    void sendgxc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgxc, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, ind0, ind1, desc, loc);
    }
    void sendgxc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegisterRange &src1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgxc, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), NullRegister(), NullRegister(), desc, loc);
    }
    void sendgxc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegisterRange &src1, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgxc, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), ind0, NullRegister(), desc, loc);
    }
    void sendgxc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegisterRange &src1, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgxc, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), ind0, ind1, desc, loc);
    }
    void sendgxc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgxc, mod, sf, dst, src0, src0Len, NullRegister(), 0, NullRegister(), NullRegister(), desc, loc);
    }
    void sendgxc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgxc, mod, sf, dst, src0, src0Len, NullRegister(), 0, ind0, NullRegister(), desc, loc);
    }
    void sendgxc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgxc, mod, sf, dst, src0, src0Len, NullRegister(), 0, ind0, ind1, desc, loc);
    }
#endif
#if XE4
    template <typename DT = uint32_t>
    void shfld(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &lanemask, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::shfld, getDataType<DT>(), mod, dst, src0, src1, lanemask, loc);
    }
    template <typename DT = uint32_t>
    void shfld(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &lanemask, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::shfld, getDataType<DT>(), mod, dst, src0, src1, lanemask, loc);
    }
    template <typename DT = uint32_t>
    void shfld(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::shfld, getDataType<DT>(), mod, dst, src0, src1, Immediate::ud(lanemask), loc);
    }
    template <typename DT = uint32_t>
    void shfld(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::shfld, getDataType<DT>(), mod, dst, src0, src1, Immediate::ud(lanemask), loc);
    }
    template <typename DT = uint32_t>
    void shfli(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &lanemask, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::shfli, getDataType<DT>(), mod, dst, src0, src1, lanemask, loc);
    }
    template <typename DT = uint32_t>
    void shfli(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &lanemask, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::shfli, getDataType<DT>(), mod, dst, src0, src1, lanemask, loc);
    }
    template <typename DT = uint32_t>
    void shfli(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::shfli, getDataType<DT>(), mod, dst, src0, src1, Immediate::ud(lanemask), loc);
    }
    template <typename DT = uint32_t>
    void shfli(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::shfli, getDataType<DT>(), mod, dst, src0, src1, Immediate::ud(lanemask), loc);
    }
    template <typename DT = uint32_t>
    void shflu(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &lanemask, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::shflu, getDataType<DT>(), mod, dst, src0, src1, lanemask, loc);
    }
    template <typename DT = uint32_t>
    void shflu(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &lanemask, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::shflu, getDataType<DT>(), mod, dst, src0, src1, lanemask, loc);
    }
    template <typename DT = uint32_t>
    void shflu(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::shflu, getDataType<DT>(), mod, dst, src0, src1, Immediate::ud(lanemask), loc);
    }
    template <typename DT = uint32_t>
    void shflu(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::shflu, getDataType<DT>(), mod, dst, src0, src1, Immediate::ud(lanemask), loc);
    }
    template <typename DT = uint32_t>
    void shflx(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &lanemask, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::shflx, getDataType<DT>(), mod, dst, src0, src1, lanemask, loc);
    }
    template <typename DT = uint32_t>
    void shflx(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &lanemask, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::shflx, getDataType<DT>(), mod, dst, src0, src1, lanemask, loc);
    }
    template <typename DT = uint32_t>
    void shflx(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::shflx, getDataType<DT>(), mod, dst, src0, src1, Immediate::ud(lanemask), loc);
    }
    template <typename DT = uint32_t>
    void shflx(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        op128F(OpcodeClassXe4::shflx, getDataType<DT>(), mod, dst, src0, src1, Immediate::ud(lanemask), loc);
    }
#endif
    template <typename DT = void>
    void shl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::shl, getDataType<DT>(), mod, dst, src0, src1, loc);
        else
#endif
        opX(isGen12 ? Opcode::shl_gen12 : Opcode::shl, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void shl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::shl, getDataType<DT>(), mod, dst, src0, src1, loc);
        else
#endif
        opX(isGen12 ? Opcode::shl_gen12 : Opcode::shl, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void shr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::shr, getDataType<DT>(), mod, dst, src0, src1, loc);
        else
#endif
        opX(isGen12 ? Opcode::shr_gen12 : Opcode::shr, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void shr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128A(OpcodeClassXe4::shr, getDataType<DT>(), mod, dst, src0, src1, loc);
        else
#endif
        opX(isGen12 ? Opcode::shr_gen12 : Opcode::shr, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void smov(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::smov_gen12 : Opcode::smov, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void srnd(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::srnd, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void srnd(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::srnd, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void subb(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128K(OpcodeClassXe4::subb, getDataType<DT>(), mod, dst, FlagRegister(), src0, src1, FlagRegister(), loc);
        else
#endif
        opX(Opcode::subb, getDataType<DT>(), mod | AccWrEn, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void subb(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            op128K(OpcodeClassXe4::subb, getDataType<DT>(), mod, dst, FlagRegister(), src0, src1, FlagRegister(), loc);
        else
#endif
        opX(Opcode::subb, getDataType<DT>(), mod | AccWrEn, dst, src0, src1, loc);
    }
#if XE4
    template <typename DT = void>
    void subb(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const FlagRegister &carryIn, SourceLocation loc = {}) {
        op128K(OpcodeClassXe4::subb, getDataType<DT>(), mod, dst, FlagRegister(), src0, src1, carryIn, loc);
    }
    template <typename DT = void>
    void subb(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const FlagRegister &carryIn, SourceLocation loc = {}) {
        op128K(OpcodeClassXe4::subb, getDataType<DT>(), mod, dst, FlagRegister(), src0, src1, carryIn, loc);
    }
    template <typename DT = void>
    void subb(const InstructionModifier &mod, const RegData &dst, const FlagRegister &carryOut, const RegData &src0, const RegData &src1, const FlagRegister &carryIn = FlagRegister(), SourceLocation loc = {}) {
        op128K(OpcodeClassXe4::subb, getDataType<DT>(), mod, dst, carryOut, src0, src1, carryIn, loc);
    }
    template <typename DT = void>
    void subb(const InstructionModifier &mod, const RegData &dst, const FlagRegister &carryOut, const RegData &src0, const Immediate &src1, const FlagRegister &carryIn = FlagRegister(), SourceLocation loc = {}) {
        op128K(OpcodeClassXe4::subb, getDataType<DT>(), mod, dst, carryOut, src0, src1, carryIn, loc);
    }
    template <typename DT = void>
    void subb(const RegData &dst, const RegData &src0, const RegData &src1, const FlagRegister &carryIn = FlagRegister(), SourceLocation loc = {}) {
        subb<DT>(InstructionModifier(), dst, src0, src1, carryIn, loc);
    }
    template <typename DT = void>
    void subb(const RegData &dst, const RegData &src0, const Immediate &src1, const FlagRegister &carryIn = FlagRegister(), SourceLocation loc = {}) {
        subb<DT>(InstructionModifier(), dst, src0, src1, carryIn, loc);
    }
    void tarb(const InstructionModifier &mod = InstructionModifier(), SourceLocation loc = {}) {
        op64G(OpcodeClassXe4::tarb, mod, loc);
    }
#endif
    void wait(const InstructionModifier &mod, const RegData &nreg, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (!nreg.isARF() || nreg.getARFType() != ARFType::n) throw invalid_arf_exception();
#endif
        opX(Opcode::wait, DataType::invalid, mod, nreg, nreg, loc);
    }
    void while_(const InstructionModifier &mod, Label &jip, SourceLocation loc = {}) {
        opBranch(Opcode::while_, mod, null, jip, loc);
    }
    template <typename DT = void>
    void xor_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            bfn2<DT>(mod, 0x66, dst, src0, src1, loc);
        else
#endif
        opX(isGen12 ? Opcode::xor_gen12 : Opcode::xor_, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void xor_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            bfn2<DT>(mod, 0x66, dst, src0, src1, loc);
        else
#endif
        opX(isGen12 ? Opcode::xor_gen12 : Opcode::xor_, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
#if XE4
    void yield(const InstructionModifier &mod = InstructionModifier(), SourceLocation loc = {}) {
        op64G(OpcodeClassXe4::yield, mod, loc);
    }
#endif

#if XE3P
    template <typename DT = void>
    void bdpas(const InstructionModifier &mod, uint8_t sdepth, uint8_t rcount, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, const RegData &src3, const RegData &src4, SourceLocation loc = {}) {
        auto emod = mod | defaultModifier;
        if (emod.isAutoSWSB()) {
            if (!src3.isARF()) wrdep(GRF(src3.getBase()), loc);
            if (!src4.isARF()) wrdep(GRF(src4.getBase()), loc);
        }
        opBdpas(Opcode::bdpas, getDataType<DT>(), mod, sdepth, rcount, dst, src0, src1, src2, src3, src4, loc);
    }
    template <typename DT = void>
    void dnscl(const InstructionModifier &mod, uint8_t mode, RoundingType rnd, RegData dst, RegData src0, RegData src1, const RegData &src2, SourceLocation loc = {}) {
        auto ctrl = encodeDnsclCtrl(mode, rnd, dst, src0, src1);
        opBfn(Opcode::dnscl, getDataType<DT>(), mod, ctrl, dst, src0, src1, src2, loc);
    }

private:
    struct LFSR {
        BinaryCodeGenerator<hw> &parent;

        LFSR(BinaryCodeGenerator<hw> *parent_) : parent(*parent_) {}

        void operator()(LFSRFunction fc, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
            parent.opShflLfsr(Opcode::lfsr, static_cast<uint8_t>(fc), DataType::invalid, mod, dst, src0, src1, loc);
        }

        template <typename DT = void>
        void b32(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
            parent.opShflLfsr(Opcode::lfsr, static_cast<uint8_t>(LFSRFunction::b32), getDataType<DT>(), mod, dst, src0, src1, loc);
        }
        template <typename DT = void>
        void b32(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
            parent.opShflLfsr(Opcode::lfsr, static_cast<uint8_t>(LFSRFunction::b32), getDataType<DT>(), mod, dst, src0, src1, loc);
        }
        template <typename DT = void>
        void b16v2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
            parent.opShflLfsr(Opcode::lfsr, static_cast<uint8_t>(LFSRFunction::b16v2), getDataType<DT>(), mod, dst, src0, src1, loc);
        }
        template <typename DT = void>
        void b16v2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
            parent.opShflLfsr(Opcode::lfsr, static_cast<uint8_t>(LFSRFunction::b16v2), getDataType<DT>(), mod, dst, src0, src1, loc);
        }
        template <typename DT = void>
        void b8v4(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
            parent.opShflLfsr(Opcode::lfsr, static_cast<uint8_t>(LFSRFunction::b8v4), getDataType<DT>(), mod, dst, src0, src1, loc);
        }
        template <typename DT = void>
        void b8v4(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
            parent.opShflLfsr(Opcode::lfsr, static_cast<uint8_t>(LFSRFunction::b8v4), getDataType<DT>(), mod, dst, src0, src1, loc);
        }
    };
public:
    LFSR lfsr;

private:
    struct Shfl {
        BinaryCodeGenerator<hw> &parent;

        Shfl(BinaryCodeGenerator<hw> *parent_) : parent(*parent_) {}

        void operator()(ShuffleFunction fc, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
            parent.opShflLfsr(Opcode::shfl, static_cast<uint8_t>(fc), DataType::invalid, mod, dst, src0, src1, loc);
        }

        template <typename DT = void>
        void idx4(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
            parent.opShflLfsr(Opcode::shfl, static_cast<uint8_t>(ShuffleFunction::idx4), getDataType<DT>(), mod, dst, src0, src1, loc);
        }
    };
public:
    Shfl shfl;
#endif

#if XE4
    /* Scalar instructions */
    template <typename DT = void>
    void sadd(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        op128A(OpcodeClassXe4::sadd_128A, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void sadd(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        op128A(OpcodeClassXe4::sadd_128A, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void sasr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        op128A(OpcodeClassXe4::sasr, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void sasr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        op128A(OpcodeClassXe4::sasr, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void sbfn2(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        op64D(OpcodeClassXe4::sbfn2, getDataType<DT>(), mod, ctrl, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void sbfn2(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        sbfn3(mod, ctrl, dst, src0, src1, null, loc);
    }
    template <typename DT = void>
    void sbfn3(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        op128E(OpcodeClassXe4::bfn2, getDataType<DT>(), mod, ctrl, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void sbfn3(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        op128E(OpcodeClassXe4::bfn2, getDataType<DT>(), mod, ctrl, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void sbfn3(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &src2, SourceLocation loc = {}) {
        op128E(OpcodeClassXe4::bfn2, getDataType<DT>(), mod, ctrl, dst, src0, src1, src2, loc);
    }
    template <typename DT = uint32_t>
    void sbfe(const InstructionModifier &mod, unsigned width, unsigned offset, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        op128G(OpcodeClassXe4::sbfe, getDataType<DT>(), mod, width, offset, dst, src0, null, loc);
    }
    template <typename DT = uint32_t>
    void sbfegen(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        op128A(OpcodeClassXe4::sbfegen, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = uint32_t>
    void sbfi(const InstructionModifier &mod, unsigned width, unsigned offset, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        op128G(OpcodeClassXe4::sbfi, getDataType<DT>(), mod, width, offset, dst, src0, src1, loc);
    }
    template <typename DT = uint32_t>
    void sbfi(const InstructionModifier &mod, unsigned width, unsigned offset, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        op128G(OpcodeClassXe4::sbfi, getDataType<DT>(), mod, width, offset, dst, src0, src1, loc);
    }
    template <typename DT = uint32_t>
    void sbfia(const InstructionModifier &mod, unsigned width, unsigned offset, IndirectARF dst, IndirectARF src0, const RegData &src1, SourceLocation loc = {}) {
        op128G(OpcodeClassXe4::sbfia, getDataType<DT>(), mod, width, offset, dst, src0, src1, loc);
    }
    template <typename DT = uint32_t>
    void sbfia(const InstructionModifier &mod, unsigned width, unsigned offset, IndirectARF dst, IndirectARF src0, const Immediate &src1, SourceLocation loc = {}) {
        op128G(OpcodeClassXe4::sbfia, getDataType<DT>(), mod, width, offset, dst, src0, src1, loc);
    }
    template <typename DT = uint32_t>
    void sbfigen(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        op128A(OpcodeClassXe4::sbfigen, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = uint32_t>
    void sbfigen(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &src2, SourceLocation loc = {}) {
        op128A(OpcodeClassXe4::sbfigen, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void sbfrev(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        op128G(OpcodeClassXe4::sbfrev, getDataType<DT>(), mod, 0, 0, dst, src0, null, loc);
    }
    template <typename DT = void>
    void sbrepgen(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        op128A(OpcodeClassXe4::sbrepgen, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void sbrepgen(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        op128A(OpcodeClassXe4::sbrepgen, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = uint32_t>
    void scmp(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        op128S(OpcodeClassXe4::scmp_128S, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = uint32_t>
    void scmp(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        op128S(OpcodeClassXe4::scmp_128S, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void sfbh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        op128A(OpcodeClassXe4::sfbh, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void sfbl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        op128A(OpcodeClassXe4::sfbl, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = uint32_t>
    void sgeta(const InstructionModifier &mod, const RegData &dst, IndirectARF src0, SourceLocation loc = {}) {
        op128R(OpcodeClassXe4::sgeta, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void smad(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        op128A(OpcodeClassXe4::smad_128A, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void smad(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        op128A(OpcodeClassXe4::smad_128A, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void smad(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &src2, SourceLocation loc = {}) {
        op128A(OpcodeClassXe4::smad_128A, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void smad(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const Immediate &src2, SourceLocation loc = {}) {
        op128A(OpcodeClassXe4::smad_128A, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void smov(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        op128R(OpcodeClassXe4::smov_128R, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void smov(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        op128R(OpcodeClassXe4::smov_128R, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = uint32_t>
    void smsk(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        op64F(OpcodeClassXe4::smsk, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = uint32_t>
    void smsk(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t src1, SourceLocation loc = {}) {
        op64F(OpcodeClassXe4::smsk, getDataType<DT>(), mod, dst, src0, Immediate::ud(src1), loc);
    }
    template <typename DT = uint32_t>
    void smsk(const InstructionModifier &mod, const RegData &dst, uint32_t src0, const RegData &src1, SourceLocation loc = {}) {
        op64F(OpcodeClassXe4::smsk, getDataType<DT>(), mod, dst, Immediate::ud(src0), src1, loc);
    }
    template <typename DT = uint32_t>
    void smsk(const InstructionModifier &mod, const RegData &dst, uint32_t src0, uint32_t src1, SourceLocation loc = {}) {
        op64F(OpcodeClassXe4::smsk, getDataType<DT>(),  mod, dst, Immediate::ud(src0), Immediate::ud(src1), loc);
    }
    template <typename DT = void>
    void smul(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        op128A(OpcodeClassXe4::smul_128A, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void smul(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        op128A(OpcodeClassXe4::smul_128A, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void smullh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opMullh(OpcodeClassXe4::smullh, getDataType<DT>(), mod, dst, src0, src1, null, loc);
    }
    template <typename DT = void>
    void smullh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opMullh(OpcodeClassXe4::smullh, getDataType<DT>(), mod, dst, src0, src1, null, loc);
    }
    template <typename DT = uint32_t>
    void ssel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, FlagRegister predicate, SourceLocation loc = {}) {
        op128J(OpcodeClassXe4::ssel, getDataType<DT>(), mod, dst, src0, src1, predicate, loc);
    }
    template <typename DT = uint32_t>
    void ssel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, FlagRegister predicate, SourceLocation loc = {}) {
        op128J(OpcodeClassXe4::ssel, getDataType<DT>(), mod, dst, src0, src1, predicate, loc);
    }
    template <typename DT = uint32_t>
    void sseta(const InstructionModifier &mod, IndirectARF dst, const RegData &src0, SourceLocation loc = {}) {
        op128R(OpcodeClassXe4::sseta, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = uint32_t>
    void sseta(const InstructionModifier &mod, IndirectARF dst, const Immediate &src0, SourceLocation loc = {}) {
        op128R(OpcodeClassXe4::sseta, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void sshl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        op128A(OpcodeClassXe4::sshl, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void sshl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        op128A(OpcodeClassXe4::sshl, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void sshr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        op128A(OpcodeClassXe4::sshr, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void sshr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        op128A(OpcodeClassXe4::sshr, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
#endif

private:
    struct Sync {
        BinaryCodeGenerator<hw> &parent;

        Sync(BinaryCodeGenerator<hw> *parent_) : parent(*parent_) {}

        void operator()(SyncFunction fc, const InstructionModifier &mod = InstructionModifier(), SourceLocation loc = {}) {
            parent.opSync(Opcode::sync, fc, mod, loc);
        }
        void operator()(SyncFunction fc, const RegData &src0, SourceLocation loc = {}) {
            this->operator()(fc, InstructionModifier(), src0, loc);
        }
        void operator()(SyncFunction fc, const InstructionModifier &mod, const RegData &src0, SourceLocation loc = {}) {
            parent.opSync(Opcode::sync, fc, mod, src0, loc);
        }
        void operator()(SyncFunction fc, int src0, SourceLocation loc = {}) {
            this->operator()(fc, InstructionModifier(), src0, loc);
        }
        void operator()(SyncFunction fc, const InstructionModifier &mod, uint32_t src0, SourceLocation loc = {}) {
            parent.opSync(Opcode::sync, fc, mod, Immediate::ud(src0), loc);
        }
        void allrd(SourceLocation loc = {}) {
            allrd(null.ud(0)(0, 1, 1), loc);
        }
        void allrd(const InstructionModifier &mod, SourceLocation loc = {}) {
            allrd(mod, null.ud(0)(0, 1, 1), loc);
        }
        void allrd(const RegData &src0, SourceLocation loc = {}) {
            allrd(InstructionModifier(), src0, loc);
        }
        void allrd(const InstructionModifier &mod, const RegData &src0, SourceLocation loc = {}) {
            this->operator()(SyncFunction::allrd, mod, src0, loc);
        }
        void allrd(uint32_t src0, SourceLocation loc = {}) {
            allrd(InstructionModifier(), src0, loc);
        }
        void allrd(const InstructionModifier &mod, uint32_t src0, SourceLocation loc = {}) {
            this->operator()(SyncFunction::allrd, mod, src0, loc);
        }
        void allwr(SourceLocation loc = {}) {
            allwr(null, loc);
        }
        void allwr(const InstructionModifier &mod, SourceLocation loc = {}) {
            allwr(mod, null, loc);
        }
        void allwr(const RegData &src0, SourceLocation loc = {}) {
            allwr(InstructionModifier(), src0, loc);
        }
        void allwr(const InstructionModifier &mod, const RegData &src0, SourceLocation loc = {}) {
            this->operator()(SyncFunction::allwr, mod, src0, loc);
        }
        void allwr(uint32_t src0, SourceLocation loc = {}) {
            allwr(InstructionModifier(), src0, loc);
        }
        void allwr(const InstructionModifier &mod, uint32_t src0, SourceLocation loc = {}) {
            this->operator()(SyncFunction::allwr, mod, src0, loc);
        }
        void bar(const InstructionModifier &mod = InstructionModifier(), SourceLocation loc = {}) {
            this->operator()(SyncFunction::bar, mod, loc);
        }
        void bar(const InstructionModifier &mod, uint32_t src0, SourceLocation loc = {}) {
            this->operator()(SyncFunction::bar, mod, src0, loc);
        }
        void bar(const InstructionModifier &mod, const RegData &src0, SourceLocation loc = {}) {
            this->operator()(SyncFunction::bar, mod, src0, loc);
        }
        void bar(uint32_t src0, SourceLocation loc = {}) {
            this->operator()(SyncFunction::bar, InstructionModifier(), src0, loc);
        }
        void bar(const RegData &src0, SourceLocation loc = {}) {
            this->operator()(SyncFunction::bar, InstructionModifier(), src0, loc);
        }
        void flush(SourceLocation loc = {}) {
            flush(InstructionModifier(), loc);
        }
        void flush(const InstructionModifier &mod, SourceLocation loc = {}) {
            this->operator()(SyncFunction::flush, InstructionModifier(), null, loc);
        }
        void host(const InstructionModifier &mod = InstructionModifier(), SourceLocation loc = {}) {
            this->operator()(SyncFunction::host, mod, loc);
        }
        void nop(const InstructionModifier &mod = InstructionModifier(), SourceLocation loc = {}) {
            this->operator()(SyncFunction::nop, mod, loc);
        }
#if XE4
        void barflush(const InstructionModifier &mod = InstructionModifier(), SourceLocation loc = {}) {
            flush(loc);
        }
        void barid(uint32_t src0, SourceLocation loc = {})                                         { bar(src0, loc); }
        void barid(const InstructionModifier &mod, uint32_t src0, SourceLocation loc = {})         { bar(mod, src0, loc); }
        void barsrc(const RegData &src0, SourceLocation loc = {})                                  { bar(src0, loc); }
        void barsrc(const InstructionModifier &mod, const RegData &src0, SourceLocation loc = {})  { bar(src0, loc); }
        void dstmsk(uint32_t src0, SourceLocation loc = {})                                        { allwr(src0, loc); }
        void dstmsk(const InstructionModifier &mod, uint32_t src0, SourceLocation loc = {})        { allwr(src0, loc); }
        void srcmsk(uint32_t src0, SourceLocation loc = {})                                        { allrd(src0, loc); }
        void srcmsk(const InstructionModifier &mod, uint32_t src0, SourceLocation loc = {})        { allrd(src0, loc); }
        void none(const InstructionModifier &mod = InstructionModifier(), SourceLocation loc = {}) {
            nop(mod, loc);
        }
        void none(SWSBItem swsb0, SWSBItem swsb1, SourceLocation loc = {}) {
            none(swsb0, swsb1, {}, {}, {}, loc);
        }
        void none(SWSBItem swsb0, SWSBItem swsb1, SWSBItem swsb2, SourceLocation loc = {}) {
            none(swsb0, swsb1, swsb2, {}, {}, loc);
        }
        void none(SWSBItem swsb0, SWSBItem swsb1, SWSBItem swsb2, SWSBItem swsb3, SourceLocation loc = {}) {
            none(swsb0, swsb1, swsb2, swsb3, {}, loc);
        }
        void none(SWSBItem swsb0, SWSBItem swsb1, SWSBItem swsb2, SWSBItem swsb3, SWSBItem swsb4, SourceLocation loc = {}) {
            parent.op64E(Opcode::sync_64E, InstructionModifier(), SyncFunction::none, {swsb0, swsb1, swsb2, swsb3, swsb4}, loc);
        }
#endif
    };
public:
    Sync sync;

    void ignoredep(Operand op, SourceLocation loc = {}) {
        if (hw >= HW::Gen12LP)
            opDirective(static_cast<Directive>(op), null, null, loc);
    }
    void subdep(Operand op, const GRFRange &r, SourceLocation loc = {}) {
        if (op == Operand::dst && !r.isEmpty()) {
#ifdef NGEN_SAFE
            if (r.getLen() > 32) throw invalid_directive_exception();
#endif
            opDirective(Directive::subdep_dst, r[0], r[r.getLen() - 1], loc);
        } else {
            ignoredep(op, loc);
            wrdep(r, loc);
        }
    }
    void subdep(Operand op, const GRF &r, SourceLocation loc = {}) {
        subdep(op, r-r, loc);
    }
    void wrdep(const GRFRange &r, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (hw < HW::Gen12LP) throw unsupported_instruction();
#endif
        int len = r.getLen();
        for (int o = 0; o < len; o += 32) {
            int thisLen = std::min(len - o, 32);
            opDirective(Directive::wrdep, r[o], r[o + thisLen - 1], loc);
        }
    }
    void wrdep(const GRF &r, SourceLocation loc = {}) {
        wrdep(r-r, loc);
    }
    void fencedep(Label &fenceLocation, SourceLocation loc = {}) {
        addFixup(LabelFixup(fenceLocation.getID(labelManager), LabelFixup::JIPOffset));
        opDirective(Directive::fencedep, null, Immediate::ud(0), loc);
    }
    void disablePVCWARWA(SourceLocation loc = {}) {
        opDirective(Directive::pvcwarwa, null, null, loc);
    }

    using _self = BinaryCodeGenerator<hw>;

#include "ngen_shortcuts.hpp"
#include "ngen_pseudo.hpp"
};

#define NGEN_FORWARD(hw) NGEN_FORWARD_SCOPE(NGEN_NAMESPACE::BinaryCodeGenerator<hw>)

#define NGEN_FORWARD_SCOPE(scope) \
NGEN_FORWARD_SCOPE_NO_ELF_OVERRIDES(scope) \
void requireGRF(int grfs) { scope::requireGRF(grfs); }

#define NGEN_NILARY_OP(op, scope) void op(NGEN_NAMESPACE::SourceLocation loc = {}) {scope::op(loc);}
#define NGEN_UNARY_OP(op, scope) template <typename A0> void op(A0 &&a0, NGEN_NAMESPACE::SourceLocation loc = {}) {scope::op(std::forward<A0>(a0), loc);}
#define NGEN_BINARY_OP(op, scope) template <typename A0, typename A1> void op(A0 &&a0, A1 &&a1, NGEN_NAMESPACE::SourceLocation loc = {}) {scope::op(std::forward<A0>(a0), std::forward<A1>(a1), loc);}
#define NGEN_TERNARY_OP(op, scope) template <typename A0, typename A1, typename A2> void op(A0 &&a0, A1 &&a1, A2 &&a2, NGEN_NAMESPACE::SourceLocation loc = {}) {scope::op(std::forward<A0>(a0), std::forward<A1>(a1), std::forward<A2>(a2), loc);}
#define NGEN_QUADRARY_OP(op, scope) template <typename A0, typename A1, typename A2, typename A3> void op(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, NGEN_NAMESPACE::SourceLocation loc = {}) {scope::op(std::forward<A0>(a0), std::forward<A1>(a1), std::forward<A2>(a2), std::forward<A3>(a3), loc);}
#define NGEN_PENTARY_OP(op, scope) template <typename A0, typename A1, typename A2, typename A3, typename A4> void op(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4, NGEN_NAMESPACE::SourceLocation loc = {}) {scope::op(std::forward<A0>(a0), std::forward<A1>(a1), std::forward<A2>(a2), std::forward<A3>(a3), std::forward<A4>(a4), loc);}
#define NGEN_HEXARY_OP(op, scope) template <typename A0, typename A1, typename A2, typename A3, typename A4, typename A5> void op(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4, A5 &&a5, NGEN_NAMESPACE::SourceLocation loc = {}) {scope::op(std::forward<A0>(a0), std::forward<A1>(a1), std::forward<A2>(a2), std::forward<A3>(a3), std::forward<A4>(a4), std::forward<A5>(a5), loc);}
#define NGEN_SEPTARY_OP(op, scope) template <typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6> void op(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4, A5 &&a5, A6 &&a6, NGEN_NAMESPACE::SourceLocation loc = {}) {scope::op(std::forward<A0>(a0), std::forward<A1>(a1), std::forward<A2>(a2), std::forward<A3>(a3), std::forward<A4>(a4), std::forward<A5>(a5), std::forward<A6>(a6), loc);}

#define NGEN_FORWARD_SCOPE_OP(op, scope) \
    NGEN_UNARY_OP(op, scope)       \
    NGEN_BINARY_OP(op, scope)      \
    NGEN_TERNARY_OP(op, scope)     \
    NGEN_QUADRARY_OP(op, scope)    \
    NGEN_PENTARY_OP(op, scope)     \
    NGEN_HEXARY_OP(op, scope)      \
    NGEN_SEPTARY_OP(op, scope)     \

#define NGEN_BINARY_DT_OP(op, scope) template <typename DT = void, typename A0, typename A1> void op(A0 &&a0, A1 &&a1, NGEN_NAMESPACE::SourceLocation loc = {}) {scope::template op<DT>(std::forward<A0>(a0), std::forward<A1>(a1), loc);}
#define NGEN_TERNARY_DT_OP(op, scope) template <typename DT = void, typename A0, typename A1, typename A2> void op(A0 &&a0, A1 &&a1, A2 &&a2, NGEN_NAMESPACE::SourceLocation loc = {}) {scope::template op<DT>(std::forward<A0>(a0), std::forward<A1>(a1), std::forward<A2>(a2), loc);}
#define NGEN_QUADRARY_DT_OP(op, scope) template <typename DT = void, typename A0, typename A1, typename A2, typename A3> void op(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, NGEN_NAMESPACE::SourceLocation loc = {}) {scope::template op<DT>(std::forward<A0>(a0), std::forward<A1>(a1), std::forward<A2>(a2), std::forward<A3>(a3), loc);}
#define NGEN_PENTARY_DT_OP(op, scope) template <typename DT = void, typename A0, typename A1, typename A2, typename A3, typename A4> void op(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4, NGEN_NAMESPACE::SourceLocation loc = {}) {scope::template op<DT>(std::forward<A0>(a0), std::forward<A1>(a1), std::forward<A2>(a2), std::forward<A3>(a3), std::forward<A4>(a4), loc);}
#define NGEN_HEXARY_DT_OP(op, scope) template <typename DT = void, typename A0, typename A1, typename A2, typename A3, typename A4, typename A5> void op(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4, A5 &&a5, NGEN_NAMESPACE::SourceLocation loc = {}) {scope::template op<DT>(std::forward<A0>(a0), std::forward<A1>(a1), std::forward<A2>(a2), std::forward<A3>(a3), std::forward<A4>(a4), std::forward<A5>(a5), loc);}
#define NGEN_HEPTARY_DT_OP(op, scope) template <typename DT = void, typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6> void op(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4, A5 &&a5, A6 &&a6, NGEN_NAMESPACE::SourceLocation loc = {}) {scope::template op<DT>(std::forward<A0>(a0), std::forward<A1>(a1), std::forward<A2>(a2), std::forward<A3>(a3), std::forward<A4>(a4), std::forward<A5>(a5), std::forward<A6>(a6), loc);}
#define NGEN_NONARY_DT_OP(op, scope) template <typename DT = void, typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8> void op(A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4, A5 &&a5, A6 &&a6, A7 &&a7, A8 &&a8, NGEN_NAMESPACE::SourceLocation loc = {}) {scope::template op<DT>(std::forward<A0>(a0), std::forward<A1>(a1), std::forward<A2>(a2), std::forward<A3>(a3), std::forward<A4>(a4), std::forward<A5>(a5), std::forward<A6>(a6), std::forward<A7>(a7), std::forward<A8>(a8), loc);}

#define NGEN_FORWARD_SCOPE_DT_OP(op, scope) \
    NGEN_BINARY_DT_OP(op, scope)      \
    NGEN_TERNARY_DT_OP(op, scope)     \
    NGEN_QUADRARY_DT_OP(op, scope)    \
    NGEN_PENTARY_DT_OP(op, scope)     \
    NGEN_HEXARY_DT_OP(op, scope)      \
    NGEN_HEPTARY_DT_OP(op, scope)     \
    NGEN_NONARY_DT_OP(op, scope)      \

#define NGEN_FORWARD_SCOPE_NO_ELF_OVERRIDES(scope) \
using scope::isGen12; \
constexpr NGEN_NAMESPACE::HW getHardware() const { return scope::getHardware(); } \
NGEN_FORWARD_SCOPE_DT_OP(add, scope) \
NGEN_FORWARD_SCOPE_DT_OP(addc, scope) \
NGEN_FORWARD_SCOPE_DT_OP(add3, scope) \
NGEN_FORWARD_SCOPE_DT_OP(and_, scope) \
NGEN_FORWARD_SCOPE_DT_OP(asr, scope) \
NGEN_FORWARD_SCOPE_DT_OP(avg, scope) \
NGEN_FORWARD_SCOPE_DT_OP(bfe, scope) \
NGEN_FORWARD_SCOPE_DT_OP(bfi1, scope) \
NGEN_FORWARD_SCOPE_DT_OP(bfi2, scope) \
NGEN_FORWARD_SCOPE_DT_OP(bfn, scope) \
NGEN_FORWARD_SCOPE_DT_OP(bfrev, scope) \
NGEN_FORWARD_SCOPE_DT_OP(cbit, scope) \
NGEN_FORWARD_SCOPE_DT_OP(cmp, scope) \
NGEN_FORWARD_SCOPE_DT_OP(cmpn, scope) \
NGEN_FORWARD_SCOPE_DT_OP(csel, scope) \
NGEN_FORWARD_SCOPE_DT_OP(dp2, scope) \
NGEN_FORWARD_SCOPE_DT_OP(dp3, scope) \
NGEN_FORWARD_SCOPE_DT_OP(dp4, scope) \
NGEN_FORWARD_SCOPE_DT_OP(dp4a, scope) \
NGEN_FORWARD_SCOPE_DT_OP(dpas, scope) \
NGEN_FORWARD_SCOPE_DT_OP(dpasw, scope) \
NGEN_FORWARD_SCOPE_DT_OP(dph, scope) \
NGEN_FORWARD_SCOPE_DT_OP(fbh, scope) \
NGEN_FORWARD_SCOPE_DT_OP(fbl, scope) \
NGEN_FORWARD_SCOPE_DT_OP(frc, scope) \
NGEN_FORWARD_SCOPE_DT_OP(line, scope) \
NGEN_FORWARD_SCOPE_DT_OP(lrp, scope) \
NGEN_FORWARD_SCOPE_DT_OP(lzd, scope) \
NGEN_FORWARD_SCOPE_DT_OP(mac, scope) \
NGEN_FORWARD_SCOPE_DT_OP(macl, scope) \
NGEN_FORWARD_SCOPE_DT_OP(mach, scope) \
NGEN_FORWARD_SCOPE_DT_OP(mad, scope) \
NGEN_FORWARD_SCOPE_DT_OP(madm, scope) \
NGEN_FORWARD_SCOPE_DT_OP(math, scope) \
NGEN_FORWARD_SCOPE_DT_OP(mov, scope) \
NGEN_FORWARD_SCOPE_DT_OP(movi, scope) \
NGEN_FORWARD_SCOPE_DT_OP(mul, scope) \
NGEN_FORWARD_SCOPE_DT_OP(not_, scope) \
NGEN_FORWARD_SCOPE_DT_OP(or_, scope) \
NGEN_FORWARD_SCOPE_DT_OP(pln, scope) \
NGEN_FORWARD_SCOPE_DT_OP(rndd, scope) \
NGEN_FORWARD_SCOPE_DT_OP(rnde, scope) \
NGEN_FORWARD_SCOPE_DT_OP(rndu, scope) \
NGEN_FORWARD_SCOPE_DT_OP(rndz, scope) \
NGEN_FORWARD_SCOPE_DT_OP(rol, scope) \
NGEN_FORWARD_SCOPE_DT_OP(ror, scope) \
NGEN_FORWARD_SCOPE_DT_OP(sad2, scope) \
NGEN_FORWARD_SCOPE_DT_OP(sada2, scope) \
NGEN_FORWARD_SCOPE_DT_OP(sel, scope) \
NGEN_FORWARD_SCOPE_DT_OP(shl, scope) \
NGEN_FORWARD_SCOPE_DT_OP(shr, scope) \
NGEN_FORWARD_SCOPE_DT_OP(smov, scope) \
NGEN_FORWARD_SCOPE_DT_OP(subb, scope) \
NGEN_FORWARD_SCOPE_DT_OP(xor_, scope) \
NGEN_FORWARD_SCOPE_OP(brc, scope) \
NGEN_FORWARD_SCOPE_OP(brd, scope) \
NGEN_FORWARD_SCOPE_OP(break_, scope) \
NGEN_FORWARD_SCOPE_OP(call, scope) \
NGEN_FORWARD_SCOPE_OP(calla, scope) \
NGEN_FORWARD_SCOPE_OP(cont, scope) \
NGEN_FORWARD_SCOPE_OP(else_, scope) \
NGEN_FORWARD_SCOPE_OP(endif, scope) \
NGEN_FORWARD_SCOPE_OP(goto_, scope) \
NGEN_FORWARD_SCOPE_OP(halt, scope) \
NGEN_FORWARD_SCOPE_OP(if_, scope) \
NGEN_NILARY_OP(illegal, scope) \
NGEN_FORWARD_SCOPE_OP(join, scope) \
NGEN_FORWARD_SCOPE_OP(jmpi, scope) \
NGEN_NILARY_OP(nop, scope) \
NGEN_UNARY_OP(nop, scope) \
NGEN_FORWARD_SCOPE_OP(ret, scope) \
NGEN_FORWARD_SCOPE_OP(send, scope) \
NGEN_FORWARD_SCOPE_OP(sendc, scope) \
NGEN_FORWARD_SCOPE_OP(sends, scope) \
NGEN_FORWARD_SCOPE_OP(sendsc, scope) \
using scope::sync; \
NGEN_FORWARD_SCOPE_OP(wait, scope) \
NGEN_FORWARD_SCOPE_OP(while_, scope) \
NGEN_FORWARD_SCOPE_OP(ignoredep, scope) \
NGEN_FORWARD_SCOPE_OP(subdep, scope) \
NGEN_FORWARD_SCOPE_OP(wrdep, scope) \
NGEN_FORWARD_SCOPE_OP(fencedep, scope) \
NGEN_NILARY_OP(disablePVCWARWA, scope) \
NGEN_FORWARD_SCOPE_DT_OP(min_, scope) \
NGEN_FORWARD_SCOPE_DT_OP(max_, scope) \
NGEN_FORWARD_SCOPE_DT_OP(bfi, scope) \
NGEN_FORWARD_SCOPE_DT_OP(cos, scope) \
NGEN_FORWARD_SCOPE_DT_OP(exp, scope) \
NGEN_FORWARD_SCOPE_DT_OP(fdiv, scope) \
NGEN_FORWARD_SCOPE_DT_OP(idiv, scope) \
NGEN_FORWARD_SCOPE_DT_OP(inv, scope) \
NGEN_FORWARD_SCOPE_DT_OP(invm, scope) \
NGEN_FORWARD_SCOPE_DT_OP(iqot, scope) \
NGEN_FORWARD_SCOPE_DT_OP(irem, scope) \
NGEN_FORWARD_SCOPE_DT_OP(log, scope) \
NGEN_FORWARD_SCOPE_DT_OP(pow, scope) \
NGEN_FORWARD_SCOPE_DT_OP(rsqt, scope) \
NGEN_FORWARD_SCOPE_DT_OP(rsqtm, scope) \
NGEN_FORWARD_SCOPE_DT_OP(sin, scope) \
NGEN_FORWARD_SCOPE_DT_OP(sqt, scope) \
template <typename DT = void, typename... Targs> void fdiv_ieee(Targs&&... args) { scope::template fdiv_ieee<DT>(std::forward<Targs>(args)...); } \
NGEN_FORWARD_SCOPE_DT_OP(inv_ieee, scope) \
NGEN_FORWARD_SCOPE_DT_OP(sqt_ieee, scope) \
NGEN_FORWARD_SCOPE_OP(threadend, scope) \
template <typename... Targs> void barrierheader(Targs&&... args) { scope::barrierheader(std::forward<Targs>(args)...); } \
NGEN_FORWARD_SCOPE_OP(barriermsg, scope)                                           \
template <typename... Targs> void barriersignal(Targs&&... args) { scope::barriersignal(std::forward<Targs>(args)...); } \
NGEN_NILARY_OP(barrierwait, scope) \
NGEN_FORWARD_SCOPE_OP(barrierwait, scope) \
template <typename... Targs> void barrier(Targs&&... args) { scope::barrier(std::forward<Targs>(args)...); } \
using scope::load; \
using scope::store; \
using scope::atomic; \
NGEN_FORWARD_SCOPE_OP(memfence, scope) \
NGEN_FORWARD_SCOPE_OP(slmfence, scope) \
NGEN_NILARY_OP(fencewait, scope) \
NGEN_FORWARD_SCOPE_OP(loadlid, scope) \
NGEN_FORWARD_SCOPE_OP(loadargs, scope) \
template <typename... Targs> void epilogue(int GRFCount, bool hasSLM, const NGEN_NAMESPACE::RegData &r0_info, NGEN_NAMESPACE::SourceLocation loc = {}) { scope::epilogue(GRFCount, hasSLM, r0_info, loc); } \
template <typename... Targs> void pushStream(Targs&&... args) { scope::pushStream(std::forward<Targs>(args)...); } \
template <typename... Targs> void appendStream(Targs&&... args) { scope::appendStream(std::forward<Targs>(args)...); } \
template <typename... Targs> void appendCurrentStream(Targs&&... args) { scope::appendCurrentStream(std::forward<Targs>(args)...); } \
template <typename... Targs> void discardStream(Targs&&... args) { scope::discardStream(std::forward<Targs>(args)...); } \
template <typename... Targs> void mark(Targs&&... args) { scope::mark(std::forward<Targs>(args)...); } \
template <typename... Targs> void comment(Targs&&... args) { scope::comment(std::forward<Targs>(args)...); } \
template <typename... Targs> void setDefaultNoMask(Targs&&... args) { scope::setDefaultNoMask(std::forward<Targs>(args)...); } \
template <typename... Targs> void setDefaultAutoSWSB(Targs&&... args) { scope::setDefaultAutoSWSB(std::forward<Targs>(args)...); } \
bool getDefaultNoMask() const { return scope::getDefaultNoMask(); } \
bool getDefaultAutoSWSB() const { return scope::getDefaultAutoSWSB(); } \
using scope::product; \
NGEN_NAMESPACE::Product getProduct() const { return scope::getProduct(); } \
NGEN_NAMESPACE::ProductFamily getProductFamily() const { return scope::getProductFamily(); } \
int getStepping() const { return scope::getStepping(); } \
void setProduct(NGEN_NAMESPACE::Product product_) { scope::setProduct(product_); } \
void setProductFamily(NGEN_NAMESPACE::ProductFamily family_) { scope::setProductFamily(family_); } \
void setStepping(int stepping_) { scope::setStepping(stepping_); } \
NGEN_FORWARD_SCOPE_EXTRA1(scope) \
NGEN_FORWARD_SCOPE_EXTRA2(scope) \
NGEN_FORWARD_SCOPE_OP_NAMES(scope) \
NGEN_FORWARD_SCOPE_MIN_MAX(scope) \
NGEN_FORWARD_SCOPE_REGISTERS(scope)

#if !XE3P
#define NGEN_FORWARD_SCOPE_EXTRA1(scope)
#define NGEN_FORWARD_SCOPE_EXTRA_ELF_OVERRIDES(hw)
#else
#define NGEN_FORWARD_SCOPE_EXTRA1(scope)                                       \
  NGEN_FORWARD_SCOPE_DT_OP(bdpas, scope)                                       \
  NGEN_FORWARD_SCOPE_DT_OP(dnscl, scope)                                       \
  NGEN_FORWARD_SCOPE_DT_OP(mullh, scope)                                       \
  using scope::lfsr;                                                           \
  using scope::shfl;                                                           \
  NGEN_FORWARD_SCOPE_OP(sendg, scope)                                          \
  NGEN_FORWARD_SCOPE_OP(sendgc, scope)                                         \
  NGEN_FORWARD_SCOPE_OP(sendgx, scope)                                         \
  NGEN_FORWARD_SCOPE_OP(sendgxc, scope)                                        \
  NGEN_FORWARD_SCOPE_DT_OP(sigm, scope)                                        \
  bool getEfficient64Bit() { return scope::getEfficient64Bit(); }              \
  void setEfficient64Bit(bool def = true) {return scope::setEfficient64Bit(def);}

#define NGEN_FORWARD_EXTRA_ELF_OVERRIDES(hw)                                   \
  template <typename... Targs> void setEfficient64Bit(Targs &&...args) {       \
    NGEN_NAMESPACE::BinaryCodeGenerator<hw>::setEfficient64Bit(                \
        std::forward<Targs>(args)...);                                         \
  }
#endif

#if !XE4
#define NGEN_FORWARD_SCOPE_EXTRA2(scope)
#else
#define NGEN_FORWARD_SCOPE_EXTRA2(scope) \
NGEN_FORWARD_SCOPE_DT_OP(abs_, scope) \
NGEN_FORWARD_SCOPE_DT_OP(bfegen, scope) \
NGEN_FORWARD_SCOPE_DT_OP(bfigen, scope) \
NGEN_FORWARD_SCOPE_DT_OP(bfn2, scope) \
NGEN_FORWARD_SCOPE_DT_OP(bfn3, scope) \
NGEN_FORWARD_SCOPE_OP(cnvg, scope) \
NGEN_FORWARD_SCOPE_DT_OP(cvt, scope) \
NGEN_FORWARD_SCOPE_DT_OP(cvt2, scope) \
NGEN_FORWARD_SCOPE_DT_OP(emcos, scope) \
NGEN_FORWARD_SCOPE_DT_OP(emexp2, scope) \
NGEN_FORWARD_SCOPE_DT_OP(eminv, scope) \
NGEN_FORWARD_SCOPE_DT_OP(eminvm, scope) \
NGEN_FORWARD_SCOPE_DT_OP(emlog2, scope) \
NGEN_FORWARD_SCOPE_DT_OP(emrsqt, scope) \
NGEN_FORWARD_SCOPE_DT_OP(emrsqtm, scope) \
NGEN_FORWARD_SCOPE_DT_OP(emsin, scope) \
NGEN_FORWARD_SCOPE_DT_OP(emsgmd, scope) \
NGEN_FORWARD_SCOPE_DT_OP(emsqt, scope) \
NGEN_FORWARD_SCOPE_DT_OP(emtanh, scope) \
NGEN_FORWARD_SCOPE_DT_OP(madc, scope) \
NGEN_FORWARD_SCOPE_DT_OP(madlh, scope) \
NGEN_FORWARD_SCOPE_DT_OP(movb, scope) \
NGEN_FORWARD_SCOPE_DT_OP(movg, scope) \
NGEN_FORWARD_SCOPE_DT_OP(movs, scope) \
NGEN_FORWARD_SCOPE_DT_OP(msk, scope) \
NGEN_NILARY_OP(nop64, scope) \
NGEN_NILARY_OP(nop128, scope) \
NGEN_FORWARD_SCOPE_DT_OP(redand, scope) \
NGEN_FORWARD_SCOPE_DT_OP(redfirst, scope) \
NGEN_FORWARD_SCOPE_DT_OP(redfirstidx, scope) \
NGEN_FORWARD_SCOPE_DT_OP(redmax, scope) \
NGEN_FORWARD_SCOPE_DT_OP(redmin, scope) \
NGEN_FORWARD_SCOPE_DT_OP(redor, scope) \
NGEN_FORWARD_SCOPE_DT_OP(redsum, scope) \
NGEN_FORWARD_SCOPE_DT_OP(redxor, scope) \
NGEN_FORWARD_SCOPE_OP(retd, scope) \
NGEN_FORWARD_SCOPE_DT_OP(rnd, scope) \
NGEN_FORWARD_SCOPE_DT_OP(shfld, scope) \
NGEN_FORWARD_SCOPE_DT_OP(shfli, scope) \
NGEN_FORWARD_SCOPE_DT_OP(shflu, scope) \
NGEN_FORWARD_SCOPE_DT_OP(shflx, scope) \
NGEN_FORWARD_SCOPE_OP(tarb, scope) \
NGEN_FORWARD_SCOPE_OP(yield, scope) \
NGEN_FORWARD_SCOPE_DT_OP(sadd, scope) \
NGEN_FORWARD_SCOPE_DT_OP(sasr, scope) \
NGEN_FORWARD_SCOPE_DT_OP(sbfn2, scope) \
NGEN_FORWARD_SCOPE_DT_OP(sbfn3, scope) \
NGEN_FORWARD_SCOPE_DT_OP(sbfe, scope) \
NGEN_FORWARD_SCOPE_DT_OP(sbfegen, scope) \
NGEN_FORWARD_SCOPE_DT_OP(sbfi, scope) \
NGEN_FORWARD_SCOPE_DT_OP(sbfia, scope) \
NGEN_FORWARD_SCOPE_DT_OP(sbfigen, scope) \
NGEN_FORWARD_SCOPE_DT_OP(sbfrev, scope) \
NGEN_FORWARD_SCOPE_DT_OP(sbrepgen, scope) \
NGEN_FORWARD_SCOPE_DT_OP(scmp, scope) \
NGEN_FORWARD_SCOPE_DT_OP(sfbh, scope) \
NGEN_FORWARD_SCOPE_DT_OP(sgeta, scope) \
NGEN_FORWARD_SCOPE_DT_OP(smad, scope) \
NGEN_FORWARD_SCOPE_DT_OP(smsk, scope) \
NGEN_FORWARD_SCOPE_DT_OP(smul, scope) \
NGEN_FORWARD_SCOPE_DT_OP(smullh, scope) \
NGEN_FORWARD_SCOPE_DT_OP(ssel, scope) \
NGEN_FORWARD_SCOPE_DT_OP(sseta, scope) \
NGEN_FORWARD_SCOPE_DT_OP(sshl, scope) \
NGEN_FORWARD_SCOPE_DT_OP(sshr, scope) \
template <typename... Targs> void cbarrier(Targs&&... args) { scope::cbarrier(std::forward<Targs>(args)...); } \
NGEN_FORWARD_SCOPE_OP(abarrierinit, scope) \
NGEN_FORWARD_SCOPE_OP(abarrierarrive, scope) \
NGEN_FORWARD_SCOPE_OP(abarrierexpect, scope) \
NGEN_FORWARD_SCOPE_OP(abarrierarriveexp, scope) \
NGEN_FORWARD_SCOPE_OP(abarriercomplete, scope) \
NGEN_FORWARD_SCOPE_OP(abarrierpoll, scope) \
NGEN_FORWARD_SCOPE_OP(abarriertry, scope) \
NGEN_FORWARD_SCOPE_OP(abarrierinval, scope) \
NGEN_FORWARD_SCOPE_OP(abarrierquery, scope) \
NGEN_FORWARD_SCOPE_OP(abarrierwait, scope) \
NGEN_FORWARD_SCOPE_OP(admall2r, scope) \
NGEN_FORWARD_SCOPE_OP(admall2g, scope) \
NGEN_FORWARD_SCOPE_OP(admalpf, scope) \
NGEN_FORWARD_SCOPE_OP(admalg2l, scope) \
NGEN_FORWARD_SCOPE_OP(admalrl2r, scope) \
NGEN_FORWARD_SCOPE_OP(admalrl2g, scope) \
NGEN_FORWARD_SCOPE_OP(admatl2g, scope) \
NGEN_FORWARD_SCOPE_OP(admatpf, scope) \
NGEN_FORWARD_SCOPE_OP(admatg2l, scope) \
NGEN_FORWARD_SCOPE_OP(admatrl2g, scope) \
NGEN_FORWARD_SCOPE_OP(admarl2g, scope) \
NGEN_FORWARD_SCOPE_OP(admarpf, scope) \
NGEN_FORWARD_SCOPE_OP(admarg2l, scope) \
NGEN_FORWARD_SCOPE_OP(admarrl2g, scope) \
NGEN_FORWARD_SCOPE_OP(amma, scope) \
NGEN_FORWARD_SCOPE_OP(ammaerrorclr, scope) \
NGEN_FORWARD_SCOPE_OP(ammaerrorquery, scope)
#endif

#ifdef NGEN_NO_OP_NAMES
#define NGEN_FORWARD_SCOPE_OP_NAMES(scope)
#else
#define NGEN_FORWARD_SCOPE_OP_NAMES(scope) \
NGEN_FORWARD_SCOPE_DT_OP(and, scope) \
NGEN_FORWARD_SCOPE_DT_OP(not, scope) \
NGEN_FORWARD_SCOPE_DT_OP(or, scope) \
NGEN_FORWARD_SCOPE_DT_OP(xor, scope)
#endif

#ifdef NGEN_WINDOWS_COMPAT
#define NGEN_FORWARD_SCOPE_MIN_MAX(scope)
#else
#define NGEN_FORWARD_SCOPE_MIN_MAX(scope) \
NGEN_FORWARD_SCOPE_DT_OP(min, scope)     \
NGEN_FORWARD_SCOPE_DT_OP(max, scope)
#endif

#ifdef NGEN_GLOBAL_REGS
#define NGEN_FORWARD_SCOPE_REGISTERS(scope)
#else
#define NGEN_FORWARD_SCOPE_REGISTERS_BASE(scope) \
using scope::indirect; \
using scope::r0; using scope::r1; using scope::r2; using scope::r3; \
using scope::r4; using scope::r5; using scope::r6; using scope::r7; \
using scope::r8; using scope::r9; using scope::r10; using scope::r11; \
using scope::r12; using scope::r13; using scope::r14; using scope::r15; \
using scope::r16; using scope::r17; using scope::r18; using scope::r19; \
using scope::r20; using scope::r21; using scope::r22; using scope::r23; \
using scope::r24; using scope::r25; using scope::r26; using scope::r27; \
using scope::r28; using scope::r29; using scope::r30; using scope::r31; \
using scope::r32; using scope::r33; using scope::r34; using scope::r35; \
using scope::r36; using scope::r37; using scope::r38; using scope::r39; \
using scope::r40; using scope::r41; using scope::r42; using scope::r43; \
using scope::r44; using scope::r45; using scope::r46; using scope::r47; \
using scope::r48; using scope::r49; using scope::r50; using scope::r51; \
using scope::r52; using scope::r53; using scope::r54; using scope::r55; \
using scope::r56; using scope::r57; using scope::r58; using scope::r59; \
using scope::r60; using scope::r61; using scope::r62; using scope::r63; \
using scope::r64; using scope::r65; using scope::r66; using scope::r67; \
using scope::r68; using scope::r69; using scope::r70; using scope::r71; \
using scope::r72; using scope::r73; using scope::r74; using scope::r75; \
using scope::r76; using scope::r77; using scope::r78; using scope::r79; \
using scope::r80; using scope::r81; using scope::r82; using scope::r83; \
using scope::r84; using scope::r85; using scope::r86; using scope::r87; \
using scope::r88; using scope::r89; using scope::r90; using scope::r91; \
using scope::r92; using scope::r93; using scope::r94; using scope::r95; \
using scope::r96; using scope::r97; using scope::r98; using scope::r99; \
using scope::r100; using scope::r101; using scope::r102; using scope::r103; \
using scope::r104; using scope::r105; using scope::r106; using scope::r107; \
using scope::r108; using scope::r109; using scope::r110; using scope::r111; \
using scope::r112; using scope::r113; using scope::r114; using scope::r115; \
using scope::r116; using scope::r117; using scope::r118; using scope::r119; \
using scope::r120; using scope::r121; using scope::r122; using scope::r123; \
using scope::r124; using scope::r125; using scope::r126; using scope::r127; \
using scope::r128; using scope::r129; using scope::r130; using scope::r131; \
using scope::r132; using scope::r133; using scope::r134; using scope::r135; \
using scope::r136; using scope::r137; using scope::r138; using scope::r139; \
using scope::r140; using scope::r141; using scope::r142; using scope::r143; \
using scope::r144; using scope::r145; using scope::r146; using scope::r147; \
using scope::r148; using scope::r149; using scope::r150; using scope::r151; \
using scope::r152; using scope::r153; using scope::r154; using scope::r155; \
using scope::r156; using scope::r157; using scope::r158; using scope::r159; \
using scope::r160; using scope::r161; using scope::r162; using scope::r163; \
using scope::r164; using scope::r165; using scope::r166; using scope::r167; \
using scope::r168; using scope::r169; using scope::r170; using scope::r171; \
using scope::r172; using scope::r173; using scope::r174; using scope::r175; \
using scope::r176; using scope::r177; using scope::r178; using scope::r179; \
using scope::r180; using scope::r181; using scope::r182; using scope::r183; \
using scope::r184; using scope::r185; using scope::r186; using scope::r187; \
using scope::r188; using scope::r189; using scope::r190; using scope::r191; \
using scope::r192; using scope::r193; using scope::r194; using scope::r195; \
using scope::r196; using scope::r197; using scope::r198; using scope::r199; \
using scope::r200; using scope::r201; using scope::r202; using scope::r203; \
using scope::r204; using scope::r205; using scope::r206; using scope::r207; \
using scope::r208; using scope::r209; using scope::r210; using scope::r211; \
using scope::r212; using scope::r213; using scope::r214; using scope::r215; \
using scope::r216; using scope::r217; using scope::r218; using scope::r219; \
using scope::r220; using scope::r221; using scope::r222; using scope::r223; \
using scope::r224; using scope::r225; using scope::r226; using scope::r227; \
using scope::r228; using scope::r229; using scope::r230; using scope::r231; \
using scope::r232; using scope::r233; using scope::r234; using scope::r235; \
using scope::r236; using scope::r237; using scope::r238; using scope::r239; \
using scope::r240; using scope::r241; using scope::r242; using scope::r243; \
using scope::r244; using scope::r245; using scope::r246; using scope::r247; \
using scope::r248; using scope::r249; using scope::r250; using scope::r251; \
using scope::r252; using scope::r253; using scope::r254; using scope::r255; \
using scope::null; \
using scope::a0; \
using scope::acc0; using scope::acc1; using scope::acc2; using scope::acc3; \
using scope::acc4; using scope::acc5; using scope::acc6; using scope::acc7; \
using scope::acc8; using scope::acc9; \
using scope::mme0; using scope::mme1; using scope::mme2; using scope::mme3; \
using scope::mme4; using scope::mme5; using scope::mme6; using scope::mme7; \
using scope::noacc; using scope::nomme; \
using scope::f0; using scope::f1; using scope::f2; using scope::f3; \
using scope::f0_0; using scope::f0_1; using scope::f1_0; using scope::f1_1; \
using scope::ce0; using scope::sp; using scope::sr0; using scope::sr1; \
using scope::cr0; using scope::n0; using scope::ip; using scope::tdr0; \
using scope::tm0; using scope::tm1; using scope::tm2; using scope::tm3; \
using scope::tm4; using scope::pm0; using scope::tp0; using scope::dbg0; \
using scope::fc0; using scope::fc1; using scope::fc2; using scope::fc3; \
using scope::NoDDClr; using scope::NoDDChk; \
using scope::AccWrEn; using scope::NoSrcDepSet; using scope::Breakpoint; using scope::sat; \
using scope::NoMask; \
using scope::ExBSO; \
using scope::Serialize; using scope::EOT; \
using scope::Atomic; using scope::Switch; using scope::NoPreempt; \
using scope::anyv; using scope::allv; using scope::any2h; using scope::all2h; \
using scope::any4h; using scope::all4h; using scope::any8h; using scope::all8h; \
using scope::any16h; using scope::all16h; using scope::any32h; using scope::all32h; \
using scope::any; using scope::all; \
using scope::x_repl; using scope::y_repl; using scope::z_repl; using scope::w_repl; \
using scope::ze; using scope::eq; using scope::nz; using scope::ne; \
using scope::gt; using scope::ge; using scope::lt; using scope::le; \
using scope::ov; using scope::un; using scope::eo; \
using scope::M0; using scope::M4; using scope::M8; using scope::M12; \
using scope::M16; using scope::M20; using scope::M24; using scope::M28; \
using scope::sb0; using scope::sb1; using scope::sb2; using scope::sb3; \
using scope::sb4; using scope::sb5; using scope::sb6; using scope::sb7; \
using scope::sb8; using scope::sb9; using scope::sb10; using scope::sb11; \
using scope::sb12; using scope::sb13; using scope::sb14; using scope::sb15; \
using scope::sb16; using scope::sb17; using scope::sb18; using scope::sb19; \
using scope::sb20; using scope::sb21; using scope::sb22; using scope::sb23; \
using scope::sb24; using scope::sb25; using scope::sb26; using scope::sb27; \
using scope::sb28; using scope::sb29; using scope::sb30; using scope::sb31; \
using scope::NoAccSBSet; \
using scope::A32; using scope::A32NC; using scope::A64; using scope::A64NC; \
using scope::SLM; \
template <typename... Targs> NGEN_NAMESPACE::InstructionModifier ExecutionOffset(Targs&&... args) { return scope::ExecutionOffset(std::forward<Targs>(args)...); } \
template <typename... Targs> NGEN_NAMESPACE::AddressBase Surface(Targs&&... args) { return scope::Surface(std::forward<Targs>(args)...); } \
template <typename... Targs> NGEN_NAMESPACE::AddressBase CC(Targs&&... args) { return scope::CC(std::forward<Targs>(args)...); } \
template <typename... Targs> NGEN_NAMESPACE::AddressBase SC(Targs&&... args) { return scope::SC(std::forward<Targs>(args)...); } \
using scope::D8; using scope::D16; using scope::D32; using scope::D64; \
using scope::D8U32; using scope::D16U32; \
using scope::D8T; using scope::D16T; using scope::D32T; using scope::D64T; \
using scope::D8U32T; using scope::D16U32T; \
using scope::V1; using scope::V2; using scope::V3; using scope::V4; \
using scope::V8; using scope::V16; using scope::V32; using scope::V64; \
using scope::V1T; using scope::V2T; using scope::V3T; using scope::V4T; \
using scope::V8T; using scope::V16T; using scope::V32T; using scope::V64T; \
using scope::transpose; \
using scope::vnni; \
using scope::L1UC_L3UC; using scope::L1UC_L3C; using scope::L1C_L3UC; using scope::L1C_L3C; \
using scope::L1S_L3UC; using scope::L1S_L3C; using scope::L1IAR_L3C; using scope::L1UC_L3WB; \
using scope::L1WT_L3UC; using scope::L1WT_L3WB; using scope::L1S_L3WB; using scope::L1WB_L3WB; \
using scope::L1C_L3CC; using scope::L1UC_L3CC; \
using scope::s0;
#if !XE3P
#define NGEN_FORWARD_SCOPE_REGISTERS_EXTRA1(scope)
#else
#define NGEN_FORWARD_SCOPE_REGISTERS_EXTRA1(scope) \
using scope::Fwd; \
using scope::r256; using scope::r257; using scope::r258; using scope::r259; \
using scope::r260; using scope::r261; using scope::r262; using scope::r263; \
using scope::r264; using scope::r265; using scope::r266; using scope::r267; \
using scope::r268; using scope::r269; using scope::r270; using scope::r271; \
using scope::r272; using scope::r273; using scope::r274; using scope::r275; \
using scope::r276; using scope::r277; using scope::r278; using scope::r279; \
using scope::r280; using scope::r281; using scope::r282; using scope::r283; \
using scope::r284; using scope::r285; using scope::r286; using scope::r287; \
using scope::r288; using scope::r289; using scope::r290; using scope::r291; \
using scope::r292; using scope::r293; using scope::r294; using scope::r295; \
using scope::r296; using scope::r297; using scope::r298; using scope::r299; \
using scope::r300; using scope::r301; using scope::r302; using scope::r303; \
using scope::r304; using scope::r305; using scope::r306; using scope::r307; \
using scope::r308; using scope::r309; using scope::r310; using scope::r311; \
using scope::r312; using scope::r313; using scope::r314; using scope::r315; \
using scope::r316; using scope::r317; using scope::r318; using scope::r319; \
using scope::r320; using scope::r321; using scope::r322; using scope::r323; \
using scope::r324; using scope::r325; using scope::r326; using scope::r327; \
using scope::r328; using scope::r329; using scope::r330; using scope::r331; \
using scope::r332; using scope::r333; using scope::r334; using scope::r335; \
using scope::r336; using scope::r337; using scope::r338; using scope::r339; \
using scope::r340; using scope::r341; using scope::r342; using scope::r343; \
using scope::r344; using scope::r345; using scope::r346; using scope::r347; \
using scope::r348; using scope::r349; using scope::r350; using scope::r351; \
using scope::r352; using scope::r353; using scope::r354; using scope::r355; \
using scope::r356; using scope::r357; using scope::r358; using scope::r359; \
using scope::r360; using scope::r361; using scope::r362; using scope::r363; \
using scope::r364; using scope::r365; using scope::r366; using scope::r367; \
using scope::r368; using scope::r369; using scope::r370; using scope::r371; \
using scope::r372; using scope::r373; using scope::r374; using scope::r375; \
using scope::r376; using scope::r377; using scope::r378; using scope::r379; \
using scope::r380; using scope::r381; using scope::r382; using scope::r383; \
using scope::r384; using scope::r385; using scope::r386; using scope::r387; \
using scope::r388; using scope::r389; using scope::r390; using scope::r391; \
using scope::r392; using scope::r393; using scope::r394; using scope::r395; \
using scope::r396; using scope::r397; using scope::r398; using scope::r399; \
using scope::r400; using scope::r401; using scope::r402; using scope::r403; \
using scope::r404; using scope::r405; using scope::r406; using scope::r407; \
using scope::r408; using scope::r409; using scope::r410; using scope::r411; \
using scope::r412; using scope::r413; using scope::r414; using scope::r415; \
using scope::r416; using scope::r417; using scope::r418; using scope::r419; \
using scope::r420; using scope::r421; using scope::r422; using scope::r423; \
using scope::r424; using scope::r425; using scope::r426; using scope::r427; \
using scope::r428; using scope::r429; using scope::r430; using scope::r431; \
using scope::r432; using scope::r433; using scope::r434; using scope::r435; \
using scope::r436; using scope::r437; using scope::r438; using scope::r439; \
using scope::r440; using scope::r441; using scope::r442; using scope::r443; \
using scope::r444; using scope::r445; using scope::r446; using scope::r447; \
using scope::r448; using scope::r449; using scope::r450; using scope::r451; \
using scope::r452; using scope::r453; using scope::r454; using scope::r455; \
using scope::r456; using scope::r457; using scope::r458; using scope::r459; \
using scope::r460; using scope::r461; using scope::r462; using scope::r463; \
using scope::r464; using scope::r465; using scope::r466; using scope::r467; \
using scope::r468; using scope::r469; using scope::r470; using scope::r471; \
using scope::r472; using scope::r473; using scope::r474; using scope::r475; \
using scope::r476; using scope::r477; using scope::r478; using scope::r479; \
using scope::r480; using scope::r481; using scope::r482; using scope::r483; \
using scope::r484; using scope::r485; using scope::r486; using scope::r487; \
using scope::r488; using scope::r489; using scope::r490; using scope::r491; \
using scope::r492; using scope::r493; using scope::r494; using scope::r495; \
using scope::r496; using scope::r497; using scope::r498; using scope::r499; \
using scope::r500; using scope::r501; using scope::r502; using scope::r503; \
using scope::r504; using scope::r505; using scope::r506; using scope::r507; \
using scope::r508; using scope::r509; using scope::r510; using scope::r511; \
using scope::A64_A32U; using scope::A64_A32S; using scope::Overfetch; \
using scope::L1UC_L2UC_L3UC;    using scope::L1UC_L2UC_L3C;  using scope::L1UC_L2C_L3UC; \
using scope::L1UC_L2C_L3C;      using scope::L1C_L2UC_L3UC;  using scope::L1C_L2UC_L3C; \
using scope::L1C_L2C_L3UC;      using scope::L1C_L2C_L3C;    using scope::L1S_L2UC_L3UC; \
using scope::L1S_L2UC_L3C;      using scope::L1S_L2C_L3UC;   using scope::L1S_L2C_L3C; \
using scope::L1IAR_L2IAR_L3IAR; using scope::L1UC_L2UC_L3WB; using scope::L1UC_L2WB_L3UC; \
using scope::L1WT_L2UC_L3UC;    using scope::L1WT_L2UC_L3WB; using scope::L1WT_L2WB_L3UC; \
using scope::L1S_L2UC_L3WB;     using scope::L1S_L2WB_L3UC;  using scope::L1S_L2WB_L3WB; \
using scope::L1WB_L2WB_L3UC;    using scope::L1WB_L2UC_L3WB;
#endif
#if !XE4
#define NGEN_FORWARD_SCOPE_REGISTERS_EXTRA2(scope)
#else
#define NGEN_FORWARD_SCOPE_REGISTERS_EXTRA2(scope) \
using scope::s1; using scope::s2; using scope::s3; \
using scope::s4; using scope::s5; using scope::s6; using scope::s7; \
using scope::s8; using scope::s9; using scope::s10; using scope::s11; \
using scope::s12; using scope::s13; using scope::s14; using scope::s15; \
using scope::s16; using scope::s17; using scope::s18; using scope::s19; \
using scope::s20; using scope::s21; using scope::s22; using scope::s23; \
using scope::s24; using scope::s25; using scope::s26; using scope::s27; \
using scope::s28; using scope::s29; using scope::s30; using scope::s31; \
using scope::s32; using scope::s33; using scope::s34; using scope::s35; \
using scope::s36; using scope::s37; using scope::s38; using scope::s39; \
using scope::s40; using scope::s41; using scope::s42; using scope::s43; \
using scope::s44; using scope::s45; using scope::s46; using scope::s47; \
using scope::s48; using scope::s49; using scope::s50; using scope::s51; \
using scope::s52; using scope::s53; using scope::s54; using scope::s55; \
using scope::s56; using scope::s57; using scope::s58; using scope::s59; \
using scope::s60; using scope::s61; using scope::s62; using scope::s63; \
using scope::s64; using scope::s65; using scope::s66; using scope::s67; \
using scope::s68; using scope::s69; using scope::s70; using scope::s71; \
using scope::s72; using scope::s73; using scope::s74; using scope::s75; \
using scope::s76; using scope::s77; using scope::s78; using scope::s79; \
using scope::s80; using scope::s81; using scope::s82; using scope::s83; \
using scope::s84; using scope::s85; using scope::s86; using scope::s87; \
using scope::s88; using scope::s89; using scope::s90; using scope::s91; \
using scope::s92; using scope::s93; using scope::s94; using scope::s95; \
using scope::s96; using scope::s97; using scope::s98; using scope::s99; \
using scope::s100; using scope::s101; using scope::s102; using scope::s103; \
using scope::s104; using scope::s105; using scope::s106; using scope::s107; \
using scope::s108; using scope::s109; using scope::s110; using scope::s111; \
using scope::s112; using scope::s113; using scope::s114; using scope::s115; \
using scope::s116; using scope::s117; using scope::s118; using scope::s119; \
using scope::s120; using scope::s121; using scope::s122; using scope::s123; \
using scope::s124; using scope::s125; using scope::s126; using scope::s127; \
using scope::s128; using scope::s129; using scope::s130; using scope::s131; \
using scope::s132; using scope::s133; using scope::s134; using scope::s135; \
using scope::s136; using scope::s137; using scope::s138; using scope::s139; \
using scope::s140; using scope::s141; using scope::s142; using scope::s143; \
using scope::s144; using scope::s145; using scope::s146; using scope::s147; \
using scope::s148; using scope::s149; using scope::s150; using scope::s151; \
using scope::s152; using scope::s153; using scope::s154; using scope::s155; \
using scope::s156; using scope::s157; using scope::s158; using scope::s159; \
using scope::s160; using scope::s161; using scope::s162; using scope::s163; \
using scope::s164; using scope::s165; using scope::s166; using scope::s167; \
using scope::s168; using scope::s169; using scope::s170; using scope::s171; \
using scope::s172; using scope::s173; using scope::s174; using scope::s175; \
using scope::s176; using scope::s177; using scope::s178; using scope::s179; \
using scope::s180; using scope::s181; using scope::s182; using scope::s183; \
using scope::s184; using scope::s185; using scope::s186; using scope::s187; \
using scope::s188; using scope::s189; using scope::s190; using scope::s191; \
using scope::s192; using scope::s193; using scope::s194; using scope::s195; \
using scope::s196; using scope::s197; using scope::s198; using scope::s199; \
using scope::s200; using scope::s201; using scope::s202; using scope::s203; \
using scope::s204; using scope::s205; using scope::s206; using scope::s207; \
using scope::s208; using scope::s209; using scope::s210; using scope::s211; \
using scope::s212; using scope::s213; using scope::s214; using scope::s215; \
using scope::s216; using scope::s217; using scope::s218; using scope::s219; \
using scope::s220; using scope::s221; using scope::s222; using scope::s223; \
using scope::s224; using scope::s225; using scope::s226; using scope::s227; \
using scope::s228; using scope::s229; using scope::s230; using scope::s231; \
using scope::s232; using scope::s233; using scope::s234; using scope::s235; \
using scope::s236; using scope::s237; using scope::s238; using scope::s239; \
using scope::s240; using scope::s241; using scope::s242; using scope::s243; \
using scope::s244; using scope::s245; using scope::s246; using scope::s247; \
using scope::s248; using scope::s249; using scope::s250; using scope::s251; \
using scope::s252; using scope::s253; using scope::s254; using scope::s255; \
using scope::s256; using scope::s257; using scope::s258; using scope::s259; \
using scope::s260; using scope::s261; using scope::s262; using scope::s263; \
using scope::s264; using scope::s265; using scope::s266; using scope::s267; \
using scope::s268; using scope::s269; using scope::s270; using scope::s271; \
using scope::s272; using scope::s273; using scope::s274; using scope::s275; \
using scope::s276; using scope::s277; using scope::s278; using scope::s279; \
using scope::s280; using scope::s281; using scope::s282; using scope::s283; \
using scope::s284; using scope::s285; using scope::s286; using scope::s287; \
using scope::s288; using scope::s289; using scope::s290; using scope::s291; \
using scope::s292; using scope::s293; using scope::s294; using scope::s295; \
using scope::s296; using scope::s297; using scope::s298; using scope::s299; \
using scope::s300; using scope::s301; using scope::s302; using scope::s303; \
using scope::s304; using scope::s305; using scope::s306; using scope::s307; \
using scope::s308; using scope::s309; using scope::s310; using scope::s311; \
using scope::s312; using scope::s313; using scope::s314; using scope::s315; \
using scope::s316; using scope::s317; using scope::s318; using scope::s319; \
using scope::s320; using scope::s321; using scope::s322; using scope::s323; \
using scope::s324; using scope::s325; using scope::s326; using scope::s327; \
using scope::s328; using scope::s329; using scope::s330; using scope::s331; \
using scope::s332; using scope::s333; using scope::s334; using scope::s335; \
using scope::s336; using scope::s337; using scope::s338; using scope::s339; \
using scope::s340; using scope::s341; using scope::s342; using scope::s343; \
using scope::s344; using scope::s345; using scope::s346; using scope::s347; \
using scope::s348; using scope::s349; using scope::s350; using scope::s351; \
using scope::s352; using scope::s353; using scope::s354; using scope::s355; \
using scope::s356; using scope::s357; using scope::s358; using scope::s359; \
using scope::s360; using scope::s361; using scope::s362; using scope::s363; \
using scope::s364; using scope::s365; using scope::s366; using scope::s367; \
using scope::s368; using scope::s369; using scope::s370; using scope::s371; \
using scope::s372; using scope::s373; using scope::s374; using scope::s375; \
using scope::s376; using scope::s377; using scope::s378; using scope::s379; \
using scope::s380; using scope::s381; using scope::s382; using scope::s383; \
using scope::s384; using scope::s385; using scope::s386; using scope::s387; \
using scope::s388; using scope::s389; using scope::s390; using scope::s391; \
using scope::s392; using scope::s393; using scope::s394; using scope::s395; \
using scope::s396; using scope::s397; using scope::s398; using scope::s399; \
using scope::s400; using scope::s401; using scope::s402; using scope::s403; \
using scope::s404; using scope::s405; using scope::s406; using scope::s407; \
using scope::s408; using scope::s409; using scope::s410; using scope::s411; \
using scope::s412; using scope::s413; using scope::s414; using scope::s415; \
using scope::s416; using scope::s417; using scope::s418; using scope::s419; \
using scope::s420; using scope::s421; using scope::s422; using scope::s423; \
using scope::s424; using scope::s425; using scope::s426; using scope::s427; \
using scope::s428; using scope::s429; using scope::s430; using scope::s431; \
using scope::s432; using scope::s433; using scope::s434; using scope::s435; \
using scope::s436; using scope::s437; using scope::s438; using scope::s439; \
using scope::s440; using scope::s441; using scope::s442; using scope::s443; \
using scope::s444; using scope::s445; using scope::s446; using scope::s447; \
using scope::s448; using scope::s449; using scope::s450; using scope::s451; \
using scope::s452; using scope::s453; using scope::s454; using scope::s455; \
using scope::s456; using scope::s457; using scope::s458; using scope::s459; \
using scope::s460; using scope::s461; using scope::s462; using scope::s463; \
using scope::s464; using scope::s465; using scope::s466; using scope::s467; \
using scope::s468; using scope::s469; using scope::s470; using scope::s471; \
using scope::s472; using scope::s473; using scope::s474; using scope::s475; \
using scope::s476; using scope::s477; using scope::s478; using scope::s479; \
using scope::s480; using scope::s481; using scope::s482; using scope::s483; \
using scope::s484; using scope::s485; using scope::s486; using scope::s487; \
using scope::s488; using scope::s489; using scope::s490; using scope::s491; \
using scope::s492; using scope::s493; using scope::s494; using scope::s495; \
using scope::s496; using scope::s497; using scope::s498; using scope::s499; \
using scope::s500; using scope::s501; using scope::s502; using scope::s503; \
using scope::s504; using scope::s505; using scope::s506; using scope::s507; \
using scope::s508; using scope::s509; using scope::s510; using scope::s511; \
using scope::indirectSRF; \
using scope::lid; \
using scope::p0; using scope::p1; using scope::p2; using scope::p3; \
using scope::p4; using scope::p5; using scope::p6; using scope::p7; \
using scope::rne; using scope::ru; using scope::rd; using scope::rtz; using scope::rna; \
using scope::clmp; \
using scope::D4; using scope::D6; \
using scope::L2UC_L3UC; using scope::L2UC_L3C; using scope::L2C_L3UC; using scope::L2C_L3C; \
using scope::ABarrier; using scope::Multicast; using scope::NaNFill; using scope::CopySize; \
using scope::Type1; using scope::Type2; using scope::Type3; \
using scope::AScale; using scope::BScale; \
using scope::ATranspose; using scope::BTranspose; \
using scope::ATrack; using scope::BTrack; using scope::DTrack; \
using scope::AReuse;
#endif
#define NGEN_FORWARD_SCOPE_REGISTERS(scope)    \
    NGEN_FORWARD_SCOPE_REGISTERS_BASE(scope)   \
    NGEN_FORWARD_SCOPE_REGISTERS_EXTRA1(scope) \
    NGEN_FORWARD_SCOPE_REGISTERS_EXTRA2(scope)
#endif

template <HW hw>
inline void BinaryCodeGenerator<hw>::unsupported()
{
#ifdef NGEN_SAFE
    throw unsupported_instruction();
#endif
}

template <HW hw>
typename BinaryCodeGenerator<hw>::InstructionStream *BinaryCodeGenerator<hw>::popStream()
{
#ifdef NGEN_SAFE
    if (streamStack.size() <= 1) throw stream_stack_underflow();
#endif

    InstructionStream *result = streamStack.back();
    streamStack.pop_back();
    return result;
}

template <HW hw>
static inline Instruction12 encodeSyncInsertion(autoswsb::SyncInsertion &si)
{
    Instruction12 i;

#if XE4
    if (hw >= HW::Xe4) {
        InstructionXe4 ii{{}, Opcode::sync_64E};
        ii._64E.ctrl = encodeSyncFunctionXe4(si.fc);
        ii._64E.pf = encodePFXe4(InstructionModifier());
        if (si.fc == SyncFunction::none) {
            auto items = si.noneItems();
            ii._64.sb0 = encodeSWSBXe4(items[0]);
            ii._64E.sb1 = encodeSWSBXe4(items[1]);
            ii._64E.sb2 = encodeSWSBXe4(items[2]);
            ii._64E.sb3 = encodeSWSBXe4(items[3]);
            ii._64E.sb4 = encodeSWSBXe4(items[4]);
        } else {
            ii._64.sb0 = encodeSWSBXe4(si.swsb[0]);
            ii._64E.sb1 = encodeSWSBXe4(si.swsb[1]);
            if (si.fc == SyncFunction::srcmsk || si.fc == SyncFunction::dstmsk)
                ii._64EImm.imm = si.mask;
        }
        i.qword[0] = ii.qword[0];
        i.qword[1] = ii.qword[1];
        return i;
    }
#endif

    i.common.opcode = static_cast<int>(Opcode::sync);
    i.common.swsb = (hw >= HW::XeHPC) ? SWSBInfoXeHPC(si.swsb, Opcode::sync).raw()
                                      :    SWSBInfo12(si.swsb, Opcode::sync).raw();
    i.common.maskCtrl = true;
    i.binary.cmod = static_cast<int>(si.fc);

    if (si.mask) {
        i.binary.src0Type = getTypecode12(DataType::ud);
        i.binary.src0Imm = true;
        i.imm32.value = si.mask;
    }
    i.binary.dst = 1;

    return i;
}

template <HW hw>
static inline Instruction12 encodeDummyMovInsertion(autoswsb::DummyMovInsertion &mi)
{
    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};

    i.common.opcode = static_cast<int>(Opcode::mov_gen12);
    i.common.swsb = (hw >= HW::XeHPC) ? SWSBInfoXeHPC(mi.swsb, Opcode::sync).raw()
                                      :    SWSBInfo12(mi.swsb, Opcode::sync).raw();
    i.common.maskCtrl = true;
    i.binary.dst = 1;
    i.binary.dstType = i.binary.src0Type = getTypecode12(DataType::ud);

    if (mi.constant) {
        i.binary.src0Imm = true;
        i.imm32.value = 0;
    } else
        i.binary.src0 = encodeBinaryOperand12<0>(GRF(mi.grf).ud(0), tag).bits;

    return i;
}

template <HW hw>
std::vector<uint8_t> BinaryCodeGenerator<hw>::getCode()
{
#ifdef NGEN_SAFE
    if (streamStack.size() > 1) throw unfinished_stream_exception();
#endif
    rootStream.fixLabels(labelManager);

    Program program(rootStream);
    autoswsb::BasicBlockList analysis = autoswsb::autoSWSB(hw, declaredGRFs, program, cancelAutoSWSB_);
    std::vector<uint8_t> result;

    if (analysis.empty()) {
        result.resize(rootStream.length());
        std::memmove(result.data(), rootStream.code.data(), rootStream.length());
    } else {
        std::multimap<int32_t, autoswsb::SyncInsertion*> syncs;
        std::multimap<int32_t, autoswsb::DummyMovInsertion*> movs;

        for (auto &bb : analysis) {
            for (auto &sync : bb.syncs)
                syncs.insert(std::make_pair(sync.inum, &sync));
            for (auto &mov : bb.movs)
                movs.insert(std::make_pair(mov.inum, &mov));
        }

        result.resize(rootStream.length() + (syncs.size() + movs.size()) * sizeof(Instruction12));

        auto *psrc_start = reinterpret_cast<const Instruction12 *>(rootStream.code.data());
        auto *psrc = psrc_start;
        auto *pdst_start = reinterpret_cast<Instruction12 *>(result.data());
        auto *pdst = pdst_start;
        auto &srcLines = debugLine.srcLines;

        auto nextSync = syncs.begin();
        auto nextMov = movs.begin();

        for (uint32_t isrc = 0; isrc < program.size(); isrc++, psrc++) {
            if (psrc->opcode() == Opcode::directive)
                continue;
            while ((nextSync != syncs.end()) && (nextSync->second->inum == isrc))
                *pdst++ = encodeSyncInsertion<hw>(*(nextSync++)->second);
            while ((nextMov != movs.end()) && (nextMov->second->inum == isrc))
                *pdst++ = encodeDummyMovInsertion<hw>(*(nextMov++)->second);

            if(!srcLines.empty())
                srcLines[psrc - psrc_start].address = sizeof(*pdst) * (pdst - pdst_start);
            *pdst++ = *psrc;
        }

        result.resize(reinterpret_cast<uint8_t *>(pdst) - result.data());
    }

#if XE4
    if (hw >= HW::Xe4) {
        /* Restore breakpoint flags. */
        auto ibase = (InstructionXe4 *) result.data();
        for (auto ibp: rootStream.savedBPs)
            ibase[ibp].common.dbg = 1;

        /* Compress 64-bit instructions. */
        auto src = (const InstructionXe4 *) result.data();
        auto end = (const InstructionXe4 *) (result.data() + result.size());
        auto dst = (uint64_t *) result.data();
        for (; src < end; src++) {
            *dst++ = src->qword[0];
            if (!src->is64())
                *dst++ = src->qword[1];
        }
        result.resize((uint8_t *) dst - result.data());
    }
#endif

    return result;
}

template <HW hw>
template <bool forceWE, typename D, typename S0, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, SourceLocation loc)
{
    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    int ewidth = getExecWidth({defaultType, dst.getType(), src0.getType()});
    dst.fixup(hw, emod.getExecSize(), ewidth, defaultType, -1, 1);
    src0.fixup(hw, emod.getExecSize(), ewidth, defaultType, 0, 1);

    encodeCommon8(i, op, emod);
    i.common.accessMode = std::is_base_of<Align16Operand, D>::value;

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;

    if (dst.isIndirect())  i.binary.dstAddrImm9 = dst.getOffset() >> 9;
    if (src0.isIndirect()) i.binary.src0AddrImm9 = src0.getOffset() >> 9;

    i.binary.dstType = getTypecode<hw>(dst.getType());
    i.binary.src0Type = getTypecode<hw>(src0.getType());

    i.binary.dstRegFile = dst.getRegFile();
    i.binary.src0RegFile = src0.getRegFile();

    db(i, loc);
}

template <HW hw>
template <bool forceWE, typename D, typename S0, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, SourceLocation loc)
{
#if XE4
    if (hw >= HW::Xe4) unsupported();
#endif

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};

    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    int ewidth = getExecWidth({defaultType, dst.getType(), src0.getType()});
    dst.fixup(hw, emod.getExecSize(), ewidth, defaultType, -1, 1);
    src0.fixup(hw, emod.getExecSize(), ewidth, defaultType, 0, 1);

    encodeCommon12(i, op, emod, dst, tag);

    i.binary.dst  = encodeBinaryOperand12<-1>(dst, tag).bits;
    i.binary.src0 = encodeBinaryOperand12<0>(src0, tag).bits;

    i.binary.dstAddrMode = dst.isIndirect();
    i.binary.dstType  = getTypecode12(dst.getType());
    i.binary.src0Type = getTypecode12(src0.getType());

    i.binary.src0Mods = src0.getMods();

    i.binary.cmod = static_cast<int>(mod.getCMod());

#if XE3P
    if (hw >= HW::XE3P_35_10) {
        if (op == Opcode::math)
            i.binaryXe3pImm.src0Reg8 = 0;
        i.binaryXe3p.dstReg8 = getHighBit(dst);
        i.binaryXe3p.src0Reg8 = getHighBit(src0);
    }
#endif

    db(i, loc);
}

template <HW hw>
template <bool forceWE, typename D, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, const Immediate &src0, SourceLocation loc)
{
    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    int ewidth = getExecWidth({defaultType, dst.getType(), src0.getType()});
    dst.fixup(hw, emod.getExecSize(), ewidth, defaultType, -1, 1);
    src0.fixup(hw, emod.getExecSize(), ewidth, defaultType, 0, 1);

    encodeCommon8(i, op, emod);
    i.common.accessMode = std::is_base_of<Align16Operand, D>::value;

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;

    i.binary.dstType = getTypecode<hw>(dst.getType());
    i.binary.src0Type = getImmediateTypecode<hw>(src0.getType());

    i.binary.dstRegFile = dst.getRegFile();
    i.binary.src0RegFile = src0.getRegFile();

    if (dst.isIndirect())  i.binary.dstAddrImm9 = dst.getOffset() >> 9;

    if (getBytes(src0.getType()) == 8)
        i.imm64.value = static_cast<uint64_t>(src0);
    else
        i.imm32.value = static_cast<uint64_t>(src0);

    db(i, loc);
}

template <HW hw>
template <bool forceWE, typename D, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, const Immediate &src0, SourceLocation loc)
{
#if XE4
    if (hw >= HW::Xe4) unsupported();
#endif

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};

    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    int ewidth = getExecWidth({defaultType, dst.getType(), src0.getType()});
    dst.fixup(hw, emod.getExecSize(), ewidth, defaultType, -1, 1);
    src0.fixup(hw, emod.getExecSize(), ewidth, defaultType, 0, 1);

    encodeCommon12(i, op, emod, dst, tag);

    i.binary.dst = encodeBinaryOperand12<-1>(dst, tag).bits;

    i.binary.dstAddrMode = dst.isIndirect();

    i.binary.dstType  = getTypecode12(dst.getType());
    i.binary.src0Type = getTypecode12(src0.getType());

    i.binary.src0Imm = true;

    i.binary.cmod = static_cast<int>(mod.getCMod());

    auto val = static_cast<uint64_t>(src0);
    i.imm32.value = uint32_t(val);
    if (getBytes(src0.getType()) == 8) {
#ifdef NGEN_SAFE
        if (mod.getCMod() != ConditionModifier::none) throw invalid_modifiers_exception();
#endif
        i.imm64.high = val >> 32;
    }

#if XE3P
    if (hw >= HW::XE3P_35_10)
        i.unaryXe3pImm.dstReg8 = getHighBit(dst);
#endif

    db(i, loc);
}

template <HW hw>
template <bool forceWE, typename D, typename S0, typename S1, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, SourceLocation loc)
{
    Instruction8 i{};

    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    int ewidth = getExecWidth({defaultType, dst.getType(), src0.getType(), src1.getType()});
    dst.fixup(hw, emod.getExecSize(),  ewidth, defaultType, -1, 2);
    src0.fixup(hw, emod.getExecSize(), ewidth, defaultType, 0, 2);
    src1.fixup(hw, emod.getExecSize(), ewidth, defaultType, 1, 2);

    encodeCommon8(i, op, emod);
    i.common.accessMode = std::is_base_of<Align16Operand, D>::value;

    i.binary.dst  = encodeBinaryOperand8<true>(dst).bits;
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;
    i.binary.src1 = encodeBinaryOperand8<false>(src1).bits;

    if (dst.isIndirect())  i.binary.dstAddrImm9 = dst.getOffset() >> 9;
    if (src0.isIndirect()) i.binary.src0AddrImm9 = src0.getOffset() >> 9;
    if (src1.isIndirect()) i.binary.src1AddrImm9 = src1.getOffset() >> 9;

    i.binary.dstType  = getTypecode<hw>(dst.getType());
    i.binary.src0Type = getTypecode<hw>(src0.getType());
    i.binary.src1Type = getTypecode<hw>(src1.getType());

    i.binary.dstRegFile = dst.getRegFile();
    i.binary.src0RegFile = src0.getRegFile();
    i.binary.src1RegFile = src1.getRegFile();

#ifdef NGEN_SAFE
    if (src1.isARF() && op != Opcode::illegal && op != Opcode::movi && op != Opcode::directive)
        throw grf_expected_exception();
#endif

    db(i, loc);
}

template <HW hw>
template <bool forceWE, typename D, typename S0, typename S1, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, SourceLocation loc)
{
#if XE4
    if (hw >= HW::Xe4) unsupported();
#endif

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};

    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    int ewidth = getExecWidth({defaultType, dst.getType(), src0.getType(), src1.getType()});
    dst.fixup(hw,  emod.getExecSize(), ewidth, defaultType, -1, 2);
    src0.fixup(hw, emod.getExecSize(), ewidth, defaultType, 0, 2);
    src1.fixup(hw, emod.getExecSize(), ewidth, defaultType, 1, 2);

    encodeCommon12(i, op, emod, dst, tag);

    i.binary.dst  = encodeBinaryOperand12<-1>(dst, tag).bits;
    i.binary.src0 = encodeBinaryOperand12<0>(src0, tag).bits;
    i.binary.src1 = encodeBinaryOperand12<1>(src1, tag).bits;

    i.binary.dstAddrMode = dst.isIndirect();
    i.binary.dstType  = getTypecode12(dst.getType());
    i.binary.src0Type = getTypecode12(src0.getType());
    i.binary.src1Type = getTypecode12(src1.getType());

    i.binary.src0Mods = src0.getMods();
    i.binary.src1Mods = src1.getMods();

    i.binary.cmod = static_cast<int>(mod.getCMod());

#if XE3P
    if (hw >= HW::XE3P_35_10) {
        i.binaryXe3pImm.src0Reg8 = 0;
        i.binaryXe3p.dstReg8 = getHighBit(dst);
        i.binaryXe3p.src0Reg8 = getHighBit(src0);
        i.binaryXe3p.src1Reg8 = getHighBit(src1);
        i.binaryXe3p.src1Scalar = checkSrc1Scalar(op, src1, dst, tag);
    }
#endif

    db(i, loc);
}

template <HW hw>
template <bool forceWE, typename D, typename S0, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, const Immediate &src1, SourceLocation loc)
{
    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    int ewidth = getExecWidth({defaultType, dst.getType(), src0.getType(), src1.getType()});
    dst.fixup(hw,  emod.getExecSize(), ewidth, defaultType, -1, 2);
    src0.fixup(hw, emod.getExecSize(), ewidth, defaultType, 0, 2);
    src1.fixup(hw, emod.getExecSize(), ewidth, defaultType, 1, 2);

    encodeCommon8(i, op, emod);
    i.common.accessMode = std::is_base_of<Align16Operand, D>::value;

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;

    if (dst.isIndirect())  i.binary.dstAddrImm9 = dst.getOffset() >> 9;
    if (src0.isIndirect()) i.binary.src0AddrImm9 = src0.getOffset() >> 9;

    i.binary.dstType = getTypecode<hw>(dst.getType());
    i.binary.src0Type = getTypecode<hw>(src0.getType());
    i.binary.src1Type = getImmediateTypecode<hw>(src1.getType());

    i.binary.dstRegFile = dst.getRegFile();
    i.binary.src0RegFile = src0.getRegFile();
    i.binary.src1RegFile = src1.getRegFile();

    i.imm32.value = static_cast<uint64_t>(src1);

    db(i, loc);
}

template <HW hw>
template <bool forceWE, typename D, typename S0, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, const Immediate &src1, SourceLocation loc)
{
#if XE4
    if (hw >= HW::Xe4) unsupported();
#endif

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};

    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    int ewidth = getExecWidth({defaultType, dst.getType(), src0.getType(), src1.getType()});
    dst.fixup(hw,  emod.getExecSize(), ewidth, defaultType, -1, 2);
    src0.fixup(hw, emod.getExecSize(), ewidth, defaultType, 0, 2);
    src1.fixup(hw, emod.getExecSize(), ewidth, defaultType, 1, 2);

    encodeCommon12(i, op, emod, dst, tag);

    i.binary.dst  = encodeBinaryOperand12<-1>(dst, tag).bits;
    i.binary.src0 = encodeBinaryOperand12<0>(src0, tag).bits;
    i.binary.src1 = static_cast<uint64_t>(src1);

    i.binary.dstAddrMode = dst.isIndirect();
    i.binary.dstType  = getTypecode12(dst.getType());
    i.binary.src0Type = getTypecode12(src0.getType());
    i.binary.src1Type = getTypecode12(src1.getType());

    i.binary.src0Mods = src0.getMods();

    i.binary.cmod = static_cast<int>(mod.getCMod());

    i.binary.src1Imm = true;
    i.imm32.value = uint32_t(static_cast<uint64_t>(src1));

#if XE3P
    if (hw >= HW::XE3P_35_10) {
        i.binaryXe3pImm.dstReg8 = getHighBit(dst);
        i.binaryXe3pImm.src0Reg8 = getHighBit(src0);
    }
#endif

    db(i, loc);
}

template <HW hw>
template <HW hw_>
typename std::enable_if<hwLE(hw_, HW::Gen9)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, RegData dst, RegData src0, RegData src1, RegData src2, SourceLocation loc)
{
    opX(op, defaultType, mod, emulateAlign16Dst(dst),  emulateAlign16Src(src0),
                              emulateAlign16Src(src1), emulateAlign16Src(src2), loc);
}


template <HW hw>
template <HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, Align16Operand dst, Align16Operand src0, Align16Operand src1, Align16Operand src2, SourceLocation loc)
{
#ifdef NGEN_SAFE
    if (dst.getReg().isARF())  throw grf_expected_exception();
    if (src0.getReg().isARF()) throw grf_expected_exception();
    if (src1.getReg().isARF()) throw grf_expected_exception();
    if (src2.getReg().isARF()) throw grf_expected_exception();
#endif

    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier | Align16;

    int ewidth = getExecWidth({defaultType, dst.getType(), src0.getType(), src1.getType(), src2.getType()});
    dst.getReg().fixup(hw,  emod.getExecSize(), ewidth, defaultType, -1, 3);
    src0.getReg().fixup(hw, emod.getExecSize(), ewidth, defaultType, 0, 3);
    src1.getReg().fixup(hw, emod.getExecSize(), ewidth, defaultType, 1, 3);
    src2.getReg().fixup(hw, emod.getExecSize(), ewidth, defaultType, 2, 3);

    encodeCommon8(i, op, emod);

    i.ternary16.dstChanEn = dst.getChanEn();
    i.ternary16.dstRegNum = dst.getReg().getBase();
    i.ternary16.dstSubregNum2_4 = dst.getReg().getByteOffset() >> 2;
    i.ternary16.dstType = getTernary16Typecode8(dst.getReg().getType());

    i.ternary16.srcType = getTernary16Typecode8(src0.getReg().getType());

    bool isFOrHF = (src0.getReg().getType() == DataType::f
                 || src0.getReg().getType() == DataType::hf);

    i.ternary16.src1Type = isFOrHF && (src1.getReg().getType() == DataType::hf);
    i.ternary16.src2Type = isFOrHF && (src1.getReg().getType() == DataType::hf);

    encodeTernaryCommon8(i, src0, src1, src2);

    db(i, loc);
}

template <HW hw>
template <typename D, typename S0, typename S1, typename S2, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2, SourceLocation loc)
{
    if (hw < HW::Gen10)
        unsupported();

#ifdef NGEN_SAFE
    if (src0.isARF()) throw grf_expected_exception();
    if (src2.isARF()) throw grf_expected_exception();
#endif

    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;

    int ewidth = getExecWidth({defaultType, dst.getType(), src0.getType(), src1.getType(), src2.getType()});
    dst.fixup(hw,  emod.getExecSize(), ewidth, defaultType, -1, 3);
    src0.fixup(hw, emod.getExecSize(), ewidth, defaultType, 0, 3);
    src1.fixup(hw, emod.getExecSize(), ewidth, defaultType, 1, 3);
    src2.fixup(hw, emod.getExecSize(), ewidth, defaultType, 2, 3);

    encodeCommon8(i, op, emod);

    i.ternary1.src0RegFile = std::is_base_of<Immediate, S0>::value;
    i.ternary1.src1RegFile = src1.isARF();
    i.ternary1.src2RegFile = std::is_base_of<Immediate, S2>::value;

    encodeTernaryCommon8(i, src0, src1, src2);
    encodeTernary1Dst10(i, dst);

    db(i, loc);
}

template <HW hw>
template <typename D, typename S0,typename S1, typename S2, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2, SourceLocation loc)
{
#if XE4
    if (hw >= HW::Xe4) unsupported();
#endif

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;

    int ewidth = getExecWidth({defaultType, dst.getType(), src0.getType(), src1.getType(), src2.getType()});
    dst.fixup(hw,  emod.getExecSize(), ewidth, defaultType, -1, 3);
    src0.fixup(hw, emod.getExecSize(), ewidth, defaultType, 0, 3);
    src1.fixup(hw, emod.getExecSize(), ewidth, defaultType, 1, 3);
    src2.fixup(hw, emod.getExecSize(), ewidth, defaultType, 2, 3);

    encodeCommon12(i, op, emod, dst, tag);

    i.ternary.dst  = encodeTernaryOperand12<true>(dst, tag).bits;
    encodeTernarySrc0(i, src0, tag);
    encodeTernarySrc1(i, src1, tag);
    encodeTernarySrc2(i, src2, tag);
    encodeTernaryTypes(i, dst, src0, src1, src2);

    i.ternary.cmod = static_cast<int>(mod.getCMod());

#if XE3P
    encodeTernary512GRF(i, dst, src0, src1, src2, tag);
#endif

    db(i, loc);
}

template <HW hw>
template <typename DS0>
void BinaryCodeGenerator<hw>::opMath(Opcode op, DataType defaultType, const InstructionModifier &mod, MathFunction fc, DS0 dst, DS0 src0, SourceLocation loc)
{
    InstructionModifier mmod = mod;

    mmod.setCMod(static_cast<ConditionModifier>(fc));
    opX(op, defaultType, mmod, dst, src0, loc);
}

template <HW hw>
template <typename DS0, typename S1>
void BinaryCodeGenerator<hw>::opMath(Opcode op, DataType defaultType, const InstructionModifier &mod, MathFunction fc, DS0 dst, DS0 src0, S1 src1, SourceLocation loc)
{
    InstructionModifier mmod = mod;

    mmod.setCMod(static_cast<ConditionModifier>(fc));
    opX(op, defaultType, mmod, dst, src0, src1, loc);
}

template <HW hw>
template <typename D, typename S0, typename S2>
void BinaryCodeGenerator<hw>::opBfn(Opcode op, DataType defaultType, const InstructionModifier &mod, int bfnCtrl, D dst, S0 src0, RegData src1, S2 src2, SourceLocation loc)
{
#if XE4
    if (hw >= HW::Xe4) unsupported();
#endif
    if (hw < HW::XeHP) unsupported();

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;

    int ewidth = getExecWidth({defaultType, dst.getType(), src0.getType(), src1.getType(), src2.getType()});
    dst.fixup(hw,  emod.getExecSize(), ewidth, defaultType, -1, 3);
    src0.fixup(hw, emod.getExecSize(), ewidth, defaultType, 0, 3);
    src1.fixup(hw, emod.getExecSize(), ewidth, defaultType, 1, 3);
    src2.fixup(hw, emod.getExecSize(), ewidth, defaultType, 2, 3);

    encodeCommon12(i, op, emod, dst, tag);

    i.ternary.dst  = encodeTernaryOperand12<true>(dst, tag).bits;
    encodeTernarySrc0(i, src0, tag);
    encodeTernarySrc1(i, src1, tag);
    encodeTernarySrc2(i, src2, tag);
    encodeTernaryTypes(i, dst, src0, src1, src2);

    i.ternary.cmod = static_cast<int>(mod.getCMod());

    i.bfn.bfnCtrl03 = (bfnCtrl >> 0);
    i.bfn.bfnCtrl47 = (bfnCtrl >> 4);

#if XE3P
    encodeTernary512GRF(i, dst, src0, src1, src2, tag);
#endif

    db(i, loc);
}

template <HW hw>
static inline void encodeDPAS(Instruction12 &i, Opcode op, DataType defaultType, const InstructionModifier &emod, int sdepth, int rcount, RegData dst, RegData src0, RegData src1, RegData src2)
{
    typename EncodingTag12Dispatch<hw>::tag tag;

    dst.fixup(hw, emod.getExecSize(), 0, defaultType, -1, 3);
    src0.fixup(hw, emod.getExecSize(), 0, defaultType, 0, 3);
    src1.fixup(hw, emod.getExecSize(), 0, defaultType, 1, 3);
    src2.fixup(hw, emod.getExecSize(), 0, defaultType, 2, 3);

    encodeCommon12(i, op, emod, dst, tag);

    i.ternary.dst  = encodeTernaryOperand12<true,  false>(dst,  tag).bits;
    i.ternary.src0 = encodeTernaryOperand12<false, false>(src0, tag).bits;
    i.ternary.src1 = encodeTernaryOperand12<false, false>(src1, tag).bits;
    i.ternary.src2 = encodeTernaryOperand12<false, false>(src2, tag).bits;

    encodeTernaryTypes(i, dst, src0, src1, src2);
#if XE3P
    encodeTernary512GRF(i, dst, src0, src1, src2, tag);
#endif

    i.dpas.rcount = rcount - 1;
    i.dpas.sdepth = utils::log2(sdepth);

    i.dpas.src1SubBytePrecision = encodeSubBytePrecision12(src1.getType());
    i.dpas.src2SubBytePrecision = encodeSubBytePrecision12(src2.getType());

    i.ternary.cmod = static_cast<int>(emod.getCMod());
}

template <HW hw>
void BinaryCodeGenerator<hw>::opDpas(Opcode op, DataType defaultType, const InstructionModifier &mod, int sdepth, int rcount, RegData dst, RegData src0, RegData src1, RegData src2, SourceLocation loc)
{
#if XE4
    if (hw >= HW::Xe4) unsupported();
#endif
    if (hw < HW::XeHP) unsupported();

    Instruction12 i{};
    encodeDPAS<hw>(i, op, defaultType, mod | defaultModifier, sdepth, rcount, dst, src0, src1, src2);
    db(i, loc);
}

#if XE3P
template <HW hw>
void BinaryCodeGenerator<hw>::opBdpas(Opcode op, DataType defaultType, const InstructionModifier &mod, int sdepth, int rcount,
                                      RegData dst, RegData src0, RegData src1, RegData src2, RegData src3, RegData src4, SourceLocation loc)
{
#if XE4
    if (hw >= HW::Xe4) unsupported();
#endif
    if (hw < HW::XE3P_35_10) unsupported();

    Instruction12 i{};

    encodeDPAS<hw>(i, op, defaultType, mod | defaultModifier, sdepth, rcount, dst, src0, src1, src2);

    src3.fixup(hw, mod.getExecSize(), 0, DataType::ub, 3, 5);
    src4.fixup(hw, mod.getExecSize(), 0, DataType::ub, 4, 5);

    int s3r = src3.getBase(), s4r = src4.getBase();

    i.bdpas.src3RegFile = src3.getRegFile8();
    i.bdpas.src4RegFile = src4.getRegFile8();

    i.bdpas.src3Reg0 = s3r;
    i.bdpas.src3Reg1_2 = s3r >> 1;
    i.bdpas.src3Reg3_6 = s3r >> 3;
    i.bdpas.src3Reg7_8 = s3r >> 7;
    i.bdpas.src3SubReg4_5 = src3.getByteOffset() >> 4;

    i.bdpas.src4Reg0_3 = s4r;
    i.bdpas.src4Reg4_8 = s4r >> 4;
    i.bdpas.src4SubReg3_5 = src4.getByteOffset() >> 3;

    db(i, loc);
}
#endif

template <HW hw>
template <typename D, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSend(Opcode op, const InstructionModifier &mod, SharedFunction sfid, const RegData &dst, const RegData &src0, const RegData &src1, int src1Length, uint32_t exdesc, D desc, SourceLocation loc)
{
    exdesc |= uint32_t(static_cast<uint8_t>(sfid));
    opSends(static_cast<Opcode>(static_cast<uint8_t>(op) | 2), mod, dst, src0, src1, exdesc, desc, loc);
}

template <HW hw>
template <typename D, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSend(Opcode op, const InstructionModifier &mod, SharedFunction sfid, const RegData &dst, const RegData &src0, const RegData &src1, int src1Length, const RegData &exdesc, D desc, SourceLocation loc)
{
    opSends(static_cast<Opcode>(static_cast<uint8_t>(op) | 2), mod, dst, src0, src1, exdesc, desc, loc);
}

template <HW hw>
template <typename ED, typename D, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSend(Opcode op, const InstructionModifier &mod, SharedFunction sfid, const RegData &dst, const RegData &src0_, const RegData &src1, int src1Length, ED exdesc, D desc, SourceLocation loc)
{
#if XE4
    if (hw >= HW::Xe4) unsupported();
#endif

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;

    auto src0 = src0_;
    const bool src0Indirect = (hw >= HW::Xe3 && src0.isIndirect());
    if (src0Indirect)
        src0 = src0.getIndirectReg();

    encodeCommon12(i, op, emod, dst, tag);

    i.send.fusionCtrl = emod.isSerialized();

    i.send.dstReg = dst.getBase();
    i.send.src0Reg = src0.getBase();
    i.send.src1Reg = src1.getBase();

    i.send.dstRegFile = dst.getRegFile8();
    i.send.src0RegFile = src0.getRegFile8();
    i.send.src1RegFile = src1.getRegFile8();

    i.send.sfid = static_cast<int>(sfid) & 0xF;

    if (src1.isNull())
        src1Length = 0;

    encodeSendDesc(i, desc);
    encodeSendExDesc(i, exdesc, mod, src1Length, hw);

    if (src0Indirect)
        i.send.exDesc6_10 = src0.getOffset() >> 1;

#if XE3P
#ifdef NGEN_SAFE
    if (getHighBit(dst) || getHighBit(src0) || getHighBit(src1))
        throw limited_to_256_grf_exception();
#endif
#endif

    db(i, loc);
}

template <HW hw>
template <HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSend(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc, SourceLocation loc)
{
    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon8(i, op, emod);

    i.binary.dst  = encodeBinaryOperand8<true>(dst).bits;
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;

    i.sendsGen9.dstRegFile = dst.getRegFile8();
    i.binary.src0RegFile = src0.getRegFile();
    i.binary.src1RegFile = RegFileIMM;

    i.binary.dstType = getTypecode<hw>(dst.getType());

    i.sendsGen9.sfid = exdesc & 0xF;
    i.sendGen8.zero = 0;
    i.sendGen8.exDesc16_19 = (exdesc >> 16) & 0xF;
    i.sendGen8.exDesc20_23 = (exdesc >> 20) & 0xF;
    i.sendGen8.exDesc24_27 = (exdesc >> 24) & 0xF;
    i.sendGen8.exDesc28_31 = (exdesc >> 28) & 0xF;
    i.sendsGen9.desc = desc;

    i.sendsGen9.eot = (exdesc >> 5) & 1;
    if (dst.isIndirect()) i.sendsGen9.dstAddrImm9 = dst.getOffset() >> 9;

    db(i, loc);
}

template <HW hw>
template <HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSend(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc, SourceLocation loc)
{
#ifdef NGEN_SAFE
    // Only a0.0:ud is allowed for desc.
    if (!desc.isARF() || desc.getARFType() != ARFType::a || desc.getARFBase() != 0 || desc.getOffset() != 0)
        throw invalid_arf_exception();
#endif
    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon8(i, op, emod);

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;
    i.binary.src1 = encodeBinaryOperand8<false>(desc).bits;

    i.sendsGen9.dstRegFile = dst.getRegFile8();
    i.binary.src0RegFile = src0.getRegFile();
    i.binary.src1RegFile = desc.getRegFile();
    i.binary.src1Type = getTypecode<hw>(desc.getType());

    i.sendsGen9.sfid = exdesc & 0xF;
    i.sendGen8.zero = 0;
    i.sendGen8.exDesc16_19 = (exdesc >> 16) & 0xF;
    i.sendGen8.exDesc20_23 = (exdesc >> 20) & 0xF;
    i.sendGen8.exDesc24_27 = (exdesc >> 24) & 0xF;
    i.sendGen8.exDesc28_31 = (exdesc >> 28) & 0xF;

    i.sendsGen9.eot = (exdesc >> 5) & 1;
    if (dst.isIndirect()) i.sendsGen9.dstAddrImm9 = dst.getOffset() >> 9;

    db(i, loc);
}

template <HW hw>
template <typename D, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSend(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, D desc, SourceLocation loc)
{
    opSends(op, mod, dst, src0, null, exdesc, desc, loc);
}

template <HW hw>
template <typename ED, typename D, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSends(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, ED exdesc, D desc, SourceLocation loc)
{
    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon8(i, op, emod);

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;

    i.binary.src0RegFile = 0;                   // ?
    i.sendsGen9.dstRegFile = dst.getRegFile8();
    i.sendsGen9.src1RegFile = src1.getRegFile8();
    i.sendsGen9.src1RegNum = src1.getBase();

    if (dst.isIndirect())  i.sendsGen9.dstAddrImm9  =  dst.getOffset() >> 9;
    if (src0.isIndirect()) i.sendsGen9.src0AddrImm9 = src0.getOffset() >> 9;

    encodeSendsDesc(i, desc);
    encodeSendsExDesc(i, exdesc);

    db(i, loc);
}

template <HW hw>
template <typename D, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSends(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, RegData exdesc, D desc, SourceLocation loc)
{
#ifdef NGEN_SAFE
    throw sfid_needed_exception();
#endif
}

template <HW hw>
template <typename D, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSends(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, D desc, SourceLocation loc)
{
    Opcode mop = static_cast<Opcode>(static_cast<int>(op) & ~2);
    opSend(mop, mod, static_cast<SharedFunction>(exdesc & 0x1F), dst, src0, src1, -1, exdesc, desc, loc);
}

#if XE3P
static inline unsigned encodeSendgxRegNum(RegData r)
{
    if (r.isNull())
        return 0x1FF;
#ifdef NGEN_SAFE
    else if (r.isARF())
        throw invalid_arf_exception();
    else if (r.getBase() == 0x1FF)
        throw r511_not_allowed_exception();
#endif
    else
        return r.getBase();
}

template <HW hw>
void BinaryCodeGenerator<hw>::opSendg(Opcode op, const InstructionModifier &mod, SharedFunction sfid,
                                      const RegData &dst, RegData src0, int src0Len, const RegData &src1, int src1Len,
                                      RegData ind0, RegData ind1, uint64_t desc, SourceLocation loc)
{
#if XE4
    if (hw >= HW::Xe4) {
        switch (op) {
            case Opcode::sendg:
            case Opcode::sendgx:  op = Opcode::send_128C;  break;
            case Opcode::sendgc:
            case Opcode::sendgxc: op = Opcode::sendc_128C; break;
            default: unsupported();
        }
        canonicalizeSRF(ind0);
        canonicalizeSRF(ind1);
        bool twoInd = false;
        if (!ind1.isNull()) {
            if (ind1.getBase() == ind0.getBase() + 2)
                twoInd = true;
            else
                unsupported();
        }
        op128C(op, mod, sfid, desc, dst, src0, src1, ind0, twoInd, loc);
        return;
    }
#endif

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;

    bool src0Indirect = src0.isIndirect();
    if (src0Indirect)
        src0 = src0.getIndirectReg();

    encodeCommon12(i, op, emod, dst, tag);

    i.sendg.eot = emod.isEOT();

    if (op == Opcode::sendgx || op == Opcode::sendgxc) {
        unsigned dstReg = encodeSendgxRegNum(dst);
        unsigned src0Reg = encodeSendgxRegNum(src0);
        unsigned src1Reg = encodeSendgxRegNum(src1);

        i.sendg.dstReg = dstReg;
        i.sendg.src0Reg = src0Reg;
        i.sendg.src1Reg = src1Reg;

        i.sendgx.dstReg8 = dstReg >> 8;
        i.sendgx.src0Reg8 = src0Reg >> 8;
        i.sendgx.src1Reg8 = src1Reg >> 8;
    } else {
        i.sendg.dstReg = dst.getBase();
        i.sendg.src0Reg = src0.getBase();
        i.sendg.src1Reg = src1.getBase();

        i.sendg.dstRegFile = dst.getRegFile8();
        i.sendg.src0RegFile = src0.getRegFile8();
        i.sendg.src1RegFile = src1.getRegFile8();

#ifdef NGEN_SAFE
        if (getHighBit(dst) || getHighBit(src0) || getHighBit(src1))
            throw limited_to_256_grf_exception();
#endif
    }

    i.sendg.src0Len = src0Len;
    i.sendg.src1Len = src1Len;

    i.sendg.sfid = static_cast<int>(sfid) & 0xF;

    i.sendg.desc0_15 = desc;
    i.sendg.desc16_27 = desc >> 16;
    i.sendg.desc28_29 = desc >> 28;
    i.sendg.desc30_31 = desc >> 30;
    i.sendg.desc32_39 = desc >> 32;
    i.sendg.desc40_41 = desc >> 40;
    i.sendg.ind1_desc42_46 = desc >> 42;

    if (src0Indirect)
        i.sendg.src1Len = src0.getOffset() >> 1;

    i.sendg.ind0Present = !ind0.isNull();
    i.sendg.ind1Present = !ind1.isNull();

    if (i.sendg.ind0Present) {
#ifdef NGEN_SAFE
        if (!ind0.isARF() || ind0.getARFType() != ARFType::s)
            throw invalid_arf_exception();
#endif
        i.sendg.ind0 = ind0.getByteOffset() >> 3;
    }
    if (i.sendg.ind1Present) {
#ifdef NGEN_SAFE
        if (!ind1.isARF() || ind1.getARFType() != ARFType::s)
            throw invalid_arf_exception();
#endif
        i.sendg.ind1_desc42_46 = ind1.getByteOffset() >> 3;
    }

    db(i, loc);
}
#endif

template <HW hw>
template <HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, int32_t uip, SourceLocation loc)
{
    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon8(i, op, emod);

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.dstRegFile = dst.getRegFile();
    i.binary.dstType = getTypecode<hw>(dst.getType());
    i.binary.src0RegFile = Immediate().getRegFile();
    i.binary.src0Type = getTypecode<hw>(DataType::d);
    i.branches.jip = jip;
    i.branches.uip = uip;

    db(i, loc);
}

template <HW hw>
template <HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, int32_t uip, SourceLocation loc)
{
#if XE4
    if (hw >= HW::Xe4) unsupported();
#endif

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon12(i, op, emod, dst, tag);

    i.branches.branchCtrl = emod.getBranchCtrl();

    i.binary.dst = encodeBinaryOperand12<-1, false>(dst, tag).bits;

    i.binary.src0Imm = true;
    i.binary.src1Imm = true;

    i.branches.jip = jip;
    i.branches.uip = uip;

#if XE3P
    if (hw >= HW::XE3P_35_10)
        i.branchXe3p.dstReg8 = getHighBit(dst);
#endif

    db(i, loc);
}

template <HW hw>
template <bool forceWE, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, SourceLocation loc)
{
    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    encodeCommon8(i, op, emod);

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.dstRegFile = dst.getRegFile();
    i.binary.dstType = getTypecode<hw>(dst.getType());
    i.binary.src1RegFile = RegFileIMM;
    i.binary.src1Type = getTypecode<hw>(DataType::d);
    i.branches.jip = jip;

    db(i, loc);
}

template <HW hw>
template <bool forceWE, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, SourceLocation loc)
{
#if XE4
    if (hw >= HW::Xe4) unsupported();
#endif

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    encodeCommon12(i, op, emod, dst, tag);

    i.branches.branchCtrl = emod.getBranchCtrl();

    i.binary.dst = encodeBinaryOperand12<-1, false>(dst, tag).bits;
    i.binary.src0Imm = true;
    i.branches.jip = jip;

#if XE3P
    if (hw >= HW::XE3P_35_10)
        i.branchXe3p.dstReg8 = getHighBit(dst);
#endif

    db(i, loc);
}

template <HW hw>
template <bool forceWE, bool small12, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc)
{
    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    encodeCommon8(i, op, emod);

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.dstRegFile = dst.getRegFile();
    i.binary.dstType = getTypecode<hw>(DataType::d);
    i.binary.src0RegFile = src0.getRegFile();
    i.binary.src0Type = getTypecode<hw>(DataType::d);
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;

    db(i, loc);
}

template <HW hw>
template <bool forceWE, bool small12, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc)
{
#if XE4
    if (hw >= HW::Xe4) unsupported();
#endif

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    encodeCommon12(i, op, emod, dst, tag);

    i.branches.branchCtrl = emod.getBranchCtrl();

    i.binary.dst = encodeBinaryOperand12<-1, false>(dst, tag).bits;
    i.binary.src0 = encodeBinaryOperand12<0, false>(src0, tag).bits;
    if (small12)
        i.binary.src0 &= 0xFFFF;


#if XE3P
    if (hw >= HW::XE3P_35_10) {
        i.branchXe3p.dstReg8 = getHighBit(dst);
        i.branchXe3p.src0Reg8 = getHighBit(src0);
    }
#endif

    db(i, loc);
}

template <HW hw>
void BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, Label &jip, Label &uip, SourceLocation loc)
{
    addFixup(LabelFixup(jip.getID(labelManager), LabelFixup::JIPOffset));
    addFixup(LabelFixup(uip.getID(labelManager), LabelFixup::UIPOffset));
    opBranch(op, mod, dst, 0, 0, loc);
}

template <HW hw>
template <bool forceWE>
void BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, Label &jip, SourceLocation loc)
{
    addFixup(LabelFixup(jip.getID(labelManager), LabelFixup::JIPOffset));
    opBranch<forceWE>(op, mod, dst, 0, loc);
}

template <HW hw>
void BinaryCodeGenerator<hw>::opCall(Opcode op, const InstructionModifier &mod, const RegData &dst, Label &jip, SourceLocation loc)
{
    addFixup(LabelFixup(jip.getID(labelManager), LabelFixup::JIPOffset));
    if (isGen12)
        opBranch<true>(op, mod, dst, 0, loc);
    else
        opX<true>(op, DataType::d, mod, dst, null.ud(0)(0, 1, 0), Immediate::d(0), loc);
}

template <HW hw>
template <HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opJmpi(Opcode op, const InstructionModifier &mod, const RegData &dst, RegData src0, uint32_t jip, SourceLocation loc)
{
    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier | NoMask;

    encodeCommon8(i, op, emod);

    src0.fixup(hw, emod.getExecSize(), 0, DataType::d, 0, 2);

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;
    i.binary.src0RegFile = src0.getRegFile();
    i.binary.src1RegFile = RegFileIMM;
    i.binary.src1Type = getTypecode<hw>(DataType::d);

    i.branches.jip = jip;

    db(i, loc);
}

template <HW hw>
template <HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opJmpi(Opcode op, const InstructionModifier &mod, const RegData &dst, RegData src0, uint32_t jip, SourceLocation loc)
{
    opBranch<true>(op, mod, dst, jip, loc);
}

template <HW hw>
void BinaryCodeGenerator<hw>::opJmpi(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, Label &jip, SourceLocation loc)
{
    if (hw >= HW::Gen12LP)
        addFixup(LabelFixup(jip.getID(labelManager), LabelFixup::JIPOffset));
    opJmpi(op, mod, dst, src0, 0, loc);
    if (hw < HW::Gen12LP)
        addFixup(LabelFixup(jip.getID(labelManager), LabelFixup::JIPOffsetJMPI));
}

#if XE3P
template <HW hw>
void BinaryCodeGenerator<hw>::opShflLfsr(Opcode op, uint8_t fc, DataType defaultType, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc)
{
    InstructionModifier mmod = mod;

    mmod.setCMod(static_cast<ConditionModifier>(fc));
    opX(op, defaultType, mmod, dst, src0, src1, loc);
}

template <HW hw>
void BinaryCodeGenerator<hw>::opShflLfsr(Opcode op, uint8_t fc, DataType defaultType, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc)
{
    InstructionModifier mmod = mod;

    mmod.setCMod(static_cast<ConditionModifier>(fc));
    opX(op, defaultType, mmod, dst, src0, src1, loc);
}
#endif

template <HW hw>
void BinaryCodeGenerator<hw>::opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod, SourceLocation loc)
{
#if XE4
    if (hw >= HW::Xe4) {
        op64E(Opcode::sync_64E, mod, fc, loc);
        return;
    }
#endif
    if (hw < HW::Gen12LP) unsupported();

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon12(i, op, emod, null, tag);

    i.binary.dst = 0x1;
    i.binary.cmod = static_cast<int>(fc);

    db(i, loc);
}

template <HW hw>
void BinaryCodeGenerator<hw>::opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod, RegData src0, SourceLocation loc)
{
#if XE4
    if (hw >= HW::Xe4) {
        op64E(Opcode::sync_64E,mod,fc,src0,loc);
        return;
    }
#endif
    if (hw < HW::Gen12LP) unsupported();

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon12(i, op, emod, null, tag);

    i.binary.dst = 0x1;
    if (!src0.isNull()) {
        src0.setRegion(0, 1, 0);
        i.binary.src0 = encodeBinaryOperand12<0>(src0, tag).bits;
        i.binary.src0Type = getTypecode12(src0.getType());
    }
    i.binary.cmod = static_cast<int>(fc);

    db(i, loc);
}

template <HW hw>
void BinaryCodeGenerator<hw>::opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod, const Immediate &src0, SourceLocation loc)
{
#if XE4
    if (hw >= HW::Xe4) {
        op64E(Opcode::sync_64E,mod,fc,src0,loc);
        return;
    }
#endif
    if (hw < HW::Gen12LP) unsupported();

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon12(i, op, emod, null, tag);

    i.binary.dst = 0x1;
    i.binary.src0Type = getTypecode12(src0.getType());
    i.binary.src0Imm = true;
    i.binary.cmod = static_cast<int>(fc);

    i.imm32.value = uint32_t(static_cast<uint64_t>(src0));

    db(i, loc);
}

template <HW hw>
void BinaryCodeGenerator<hw>::opNop(Opcode op, SourceLocation loc)
{
#if XE4
    if (hw >= HW::Xe4) unsupported();
#endif

    Instruction8 i{};

    i.qword[0] = static_cast<int>(op);
    i.qword[1] = 0;

    db(i, loc);
}

template <HW hw>
template <typename S1>
void BinaryCodeGenerator<hw>::opDirective(Directive directive, RegData src0, S1 src1, SourceLocation loc)
{
#if XE4
    if (hw >= HW::Xe4)
        op128A(Opcode::directive_xe4, InstructionModifier::createAutoSWSB(), GRF(static_cast<int>(directive)), src0, src1, null, loc);
    else
#endif
    opX(Opcode::directive, DataType::ud, InstructionModifier::createAutoSWSB(), GRF(static_cast<int>(directive)), src0, src1, loc);
}

#if XE4
static inline void disallowMod(bool condition) {
#ifdef NGEN_SAFE
    if (condition) throw invalid_modifiers_exception();
#endif
}
static inline void disallowMod(PredCtrl pred)          { disallowMod(pred != PredCtrl::None); }
static inline void disallowMod(ConditionModifier cmod) { disallowMod(cmod != ConditionModifier::none); }
static inline void disallowMod(RoundingOverride rmo)   { disallowMod(rmo  != RoundingOverride::none); }

static inline bool getNeg(RegData rd)                  { return rd.getNeg(); }
static inline bool getNeg(Immediate i)                 { return false; }
static inline bool getNeg(IndirectARF iarf)            { return false; }

template <HW hw>
template <typename D, typename S0, typename S1, typename S2>
Opcode BinaryCodeGenerator<hw>::preprocessXe4(OpcodeClassXe4 opclass, DataType &defaultType, InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2)
{
    if (hw < HW::Xe4) unsupported();
    mod = mod | defaultModifier;
    if (allowScalarization(dst) && allowScalarization(src0) && allowScalarization(src1) && allowScalarization(src2))
        opclass = toScalar(opclass);
    validateXe4(mod, opclass);
    validateXe4(dst);
    validateXe4(src0);
    validateXe4(src1);
    validateXe4(src2);
    processTypesXe4(defaultType, dst, src0, src1, src2);
    validateBaseXe4(dst, src0, src1, src2);
    return opcodeXe4(opclass, defaultType);
}

template <HW hw>
template <typename D, typename S0, typename S1, typename S2>
void BinaryCodeGenerator<hw>::op128A(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, D dst, S0 src0, S1 src1, S2 src2, SourceLocation loc)
{
    auto op = preprocessXe4(opclass, defaultType, mod, dst, src0, src1, src2);
    op128A(op, mod, dst, src0, src1, src2, loc);
}

template <HW hw>
template <typename S1, typename S2>
void BinaryCodeGenerator<hw>::opMullh(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, RegData dst, RegData src0, S1 src1, S2 src2, SourceLocation loc)
{
    if (hw < HW::Xe4) unsupported();
    mod | defaultModifier;
    if (allowScalarization(dst) && allowScalarization(src0) && allowScalarization(src1) && allowScalarization(src2))
        opclass = toScalar(opclass);
    validateXe4(mod, opclass);
    validateXe4(dst);
    validateXe4(src0);
    validateXe4(src1);
    validateXe4(src2);
    processTypesXe4(defaultType, dst, src2);
    auto srcType = DataType::invalid;
    switch (defaultType) {
        case DataType::u64: srcType = DataType::u32; break;
        case DataType::s64: srcType = DataType::s32; break;
        default: break;
    }
    processTypesXe4(srcType, src0, src1);
    validateBaseXe4(dst, src0, src1, src2);

    op128A(opcodeXe4(opclass, defaultType), mod, dst, src0, src1, src2, loc);
}

template <typename D, typename S0, typename S1, typename S2>
InstructionXe4 encode128A(Opcode op, InstructionModifier mod, D dst, S0 src0, S1 src1, S2 src2)
{
    InstructionXe4 ii{mod, op};
    auto &i = ii._128;

    encodeSWSBXe4<2>(ii, mod);
    i.cmcf = encodeCMCFXe4(mod);
    i.ictrl =  unsigned(std::is_base_of<Immediate, S1>::value)
            | (unsigned(std::is_base_of<Immediate, S2>::value) << 1);
    i.s0inv = getNeg(src0);
    i.s1inv = getNeg(src1);
    i.s2inv = getNeg(src2);
    i.dsat = mod.isSaturate();
    i.ipred = mod.isPredInv();
    i.pf = encodePFXe4(mod);
    i.rmo = static_cast<unsigned>(mod.getRounding());
    i.w = mod.isWrEn();
    uint64_t imm = 0;
    auto esrc2 = encodeRegOrImmXe4<uint32_t, true>(src2, imm);
    i.csrc2_0_3 = esrc2 & 0xF;
    i.csrc2_4_11 = esrc2 >> 4;
    i.csrc1 = encodeRegOrImmXe4<uint32_t, false>(src1, imm);
    i.csrc0 = encodeRegXe4(src0);
    i.cdst = encodeRegXe4(dst);
    i.imm = imm;

    return ii;
}

template <HW hw>
template <typename D, typename S0, typename S1, typename S2>
void BinaryCodeGenerator<hw>::op128A(Opcode op, InstructionModifier mod, D dst, S0 src0, S1 src1, S2 src2, SourceLocation loc)
{
    db(encode128A(op, mod, dst, src0, src1, src2), mod, loc);
}

template <HW hw>
void BinaryCodeGenerator<hw>::op128B(OpcodeClassXe4 opclass, InstructionModifier mod, RegData dst, RegData src0, uint32_t jip, uint32_t uip, SourceLocation loc)
{
    auto defaultType = DataType::invalid;
    auto op = preprocessXe4(opclass, defaultType, mod, dst, src0, null, null);

    InstructionXe4 i{mod, op};

    encodeSWSBXe4<2>(i, mod);
    i._128.ipred = mod.isPredInv();
    i._128.pf = encodePFXe4(mod);
    i._128.w = mod.isWrEn();
    i._128B.brc = mod.getBranchCtrl();
    i._128B.sctrl = !src0.isNull();

    disallowMod(src0.getNeg());
    disallowMod(mod.isSaturate());
    disallowMod(mod.getRounding());
    disallowMod(mod.getCMod());

    i._128B.uip0_23 = uip & 0xFFF;
    i._128B.uip24_31 = uip >> 24;
    i._128B.jip = jip;

    i._128.csrc0 = encodeRegXe4(src0);
    i._128.cdst = encodeRegXe4(dst);

    db(i, mod, loc);
}

template <HW hw>
void BinaryCodeGenerator<hw>::op128B(OpcodeClassXe4 opclass, InstructionModifier mod, RegData dst, Label &jip, SourceLocation loc)
{
    addFixup(LabelFixup(jip.getID(labelManager), LabelFixup::JIPOffsetXe4));
    op128B(opclass, mod, dst, null, 0, 0, loc);
}

template <HW hw>
void BinaryCodeGenerator<hw>::op128B(OpcodeClassXe4 opclass, InstructionModifier mod, RegData dst, Label &jip, Label &uip, SourceLocation loc)
{
    addFixup(LabelFixup(jip.getID(labelManager), LabelFixup::JIPOffsetXe4));
    addFixup(LabelFixup(uip.getID(labelManager), LabelFixup::UIPOffsetXe4));
    op128B(opclass, mod, dst, null, 0, 0, loc);
}

template <HW hw>
void BinaryCodeGenerator<hw>::op128C(Opcode op, InstructionModifier mod, SharedFunction sfid, uint64_t desc, RegData dst, RegData src0, RegData src1, RegData ind, bool ind2, SourceLocation loc)
{
    mod = mod | defaultModifier;

    InstructionXe4 i{mod, op};

    i._128C.sbid = encodeSWSBWithSBIDXe4(i, mod);
    i._128C.ipred = mod.isPredInv();
    i._128C.pf = encodePFXe4(mod);
    i._128.w = mod.isWrEn();

    disallowMod(src0.getNeg() || src1.getNeg());
    disallowMod(mod.isSaturate());
    disallowMod(mod.getRounding());
    disallowMod(mod.getCMod());

    i._128.csrc1 = encodeRegXe4(src1);
    i._128.csrc0 = encodeRegXe4(src0);
    i._128.cdst = encodeRegXe4(dst);

    i._128C.eot = mod.isEOT();
    i._128C.sfid = static_cast<uint8_t>(sfid);

    i._128C.msgd0_21 = desc;
    i._128C.msgd22_46 = desc >> 22;

    i._128C.ind = ind2;
    i._128C.inda = encodeRegXe4(ind) >> 1;

    db(i, mod, loc);
}

template <HW hw>
template <typename S1, typename S2>
void BinaryCodeGenerator<hw>::op128AD(OpcodeClassXe4 opclassA, OpcodeClassXe4 opclassD, DataType defaultType, InstructionModifier mod, RegData dst, RegData src0, S1 src1, S2 src2, SourceLocation loc)
{
    bool imm1 = std::is_base_of<Immediate, S1>::value;
    bool imm2 = std::is_base_of<Immediate, S2>::value;

    auto opA = preprocessXe4(opclassA, defaultType, mod, dst, src0, src1, src2);

    bool needD = false;
    if (imm1 && imm2)
        needD = true;
    else if (imm1)
        needD = (getBytes(src1.getType()) == 8);
    else if (imm2)
        needD = (getBytes(src2.getType()) == 8);

    needD ? op128D(opcodeXe4(opclassD, defaultType), mod, dst, src0, src1, src2, loc)
          : op128A(opA, mod, dst, src0, src1, src2, loc);
}

template <HW hw>
template <typename S1, typename S2>
void BinaryCodeGenerator<hw>::op128D(Opcode op, InstructionModifier mod, RegData dst, RegData src0, S1 src1, S2 src2, SourceLocation loc)
{
    InstructionXe4 i{mod, op};

    encodeSWSBXe4<1>(i, mod);
    i._128D.cmcf = encodeCMCFXe4(mod);
    i._128D.ictrl =  unsigned(std::is_base_of<Immediate, S1>::value)
                  | (unsigned(std::is_base_of<Immediate, S2>::value) << 1);
    disallowMod(src0.getNeg() || getNeg(src1) || getNeg(src2) || mod.isSaturate() || mod.isPredInv());
    disallowMod(mod.getPredCtrl());
    disallowMod(mod.getRounding());
    disallowMod(mod.isWrEn());

    uint64_t imm = 0;
    if (i._128D.ictrl == 3) {
        uint64_t imm1, imm2;
        i._128D.csrc2 = encodeRegOrImmXe4<uint32_t, true>(src2, imm2);
        i._128.csrc1  = encodeRegOrImmXe4<uint32_t, false>(src1, imm1);
        imm = (imm1 << 32) | imm2;
    } else {
        i._128D.csrc2 = encodeRegOrImmXe4<uint64_t, true>(src2, imm);
        i._128.csrc1  = encodeRegOrImmXe4<uint64_t, false>(src1, imm);
    }
    i._128D.imm0_23 = imm;
    i._128D.imm24_51 = imm >> 24;
    i._128.csrc0 = encodeRegXe4(src0);
    i._128.cdst = encodeRegXe4(dst);

    db(i, mod, loc);
}

template <HW hw>
template <typename S1, typename S2>
void BinaryCodeGenerator<hw>::op128E(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, uint8_t ctrl, RegData dst, RegData src0, S1 src1, S2 src2, SourceLocation loc)
{
    auto op = preprocessXe4(opclass, defaultType, mod, dst, src0, src1, src2);
    auto i = encode128A(op, mod, dst, src0, src1, src2);
    i._128E.bctrl = ctrl;
    db(i, mod, loc);
}

template <HW hw>
template <typename S1, typename S2>
void BinaryCodeGenerator<hw>::op128F(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, RegData dst, RegData src0, S1 src1, S2 src2, SourceLocation loc, uint8_t sgran)
{
    auto op = preprocessXe4(opclass, defaultType, mod, dst, src0, null, src2);

    InstructionXe4 i{mod, op};

    encodeSWSBXe4<2>(i, mod);
    i._128.cmcf = encodeCMCFXe4(mod);
    i._128.ipred = mod.isPredInv();
    i._128.pf = encodePFXe4(mod);
    i._128.w = mod.isWrEn();
    i._128F.sgran = sgran;
    i._128F.cctrl = unsigned(std::is_base_of<Immediate, S1>::value);
    i._128F.lctrl = unsigned(std::is_base_of<Immediate, S2>::value);

    disallowMod(src0.getNeg() || getNeg(src1));
    disallowMod(mod.isSaturate());
    disallowMod(mod.getRounding());

    uint64_t imm = 0;
    auto esrc2 = encodeRegOrImmXe4<uint32_t, true>(src2, imm);
    i._128.imm = imm;
    i._128.csrc2_0_3 = esrc2 & 0xF;
    i._128.csrc2_4_11 = esrc2 >> 4;
    i._128.csrc1 = encodeRegOrImmXe4<uint32_t, true>(src1, imm);
    i._128.csrc0 = encodeRegXe4(src0);
    i._128.cdst = encodeRegXe4(dst);

    db(i, mod, loc);
}

template <HW hw>
template <typename DS0, typename S1>
void BinaryCodeGenerator<hw>::op128G(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, unsigned width, unsigned offset, DS0 dst, DS0 src0, S1 src1, SourceLocation loc)
{
    auto op = preprocessXe4(opclass, defaultType, mod, dst, src0, src1, NullRegister());
    auto i = encode128A(op, mod, dst, src0, src1, NullRegister());

    disallowMod(mod.getCMod());
    disallowMod(mod.isSaturate());
    disallowMod(mod.getPredCtrl());
    disallowMod(mod.getRounding());

    i._128.cmcf = 0;
    i._128G.bwidth = width;
    i._128G.boff = offset;
    db(i, mod, loc);
}

template <HW hw>
void BinaryCodeGenerator<hw>::op128H(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, ExtendedReg dst, ExtendedReg src0, ExtendedReg src1, ExtendedReg src2, SourceLocation loc)
{
    auto op = preprocessXe4(opclass, defaultType, mod, dst, src0, src1, src2);
    auto i = encode128A(op, mod, dst.getBase(), src0.getBase(), src1.getBase(), src2.getBase());

    disallowMod(mod.isSaturate());
    disallowMod(mod.getPredCtrl());

    i._128H.dmme =  encodeMMEXe4(dst);
    i._128H.s0mme = encodeMMEXe4(src0);
    i._128H.s1mme = encodeMMEXe4(src1);
    i._128H.s2mme = encodeMMEXe4(src2);
    db(i, mod, loc);
}

template <HW hw>
template <typename S0, typename S1>
void BinaryCodeGenerator<hw>::op128I(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, RegData dst, S0 src0, S1 src1, SourceLocation loc)
{
    auto op = preprocessXe4(opclass, defaultType, mod, dst, src0, null, null);

    InstructionXe4 i{mod, op};

    encodeSWSBXe4<2>(i, mod);
    i._128I.dctrl = std::is_base_of<Immediate, S0>::value;
    i._128.cmcf = encodeCMCFXe4(mod);
    i._128.ipred = mod.isPredInv();
    i._128.pf = encodePFXe4(mod);
    i._128.w = mod.isWrEn();
    i._128I.lctrl = std::is_base_of<Immediate, S1>::value;

    disallowMod(getNeg(src0) || getNeg(src1));
    disallowMod(mod.isSaturate());
    disallowMod(mod.getRounding());

    uint64_t imm = 0;
    i._128I.csrc2 = encodeRegOrImmXe4<uint32_t, true>(src1, imm);
    i._128I.imsk = imm;

    imm = 0;
    i._128.csrc0 = encodeRegOrImmXe4<uint32_t, false>(src0, imm);
    i._128I.dimm0 = imm;
    i._128I.dimm1_20 = imm >> 1;
    i._128.cdst = encodeRegXe4(dst);

    db(i, mod, loc);
}

template <HW hw>
template <typename S1>
void BinaryCodeGenerator<hw>::op128J(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, RegData dst, RegData src0, S1 src1, FlagRegister predicate, SourceLocation loc)
{
    auto op = preprocessXe4(opclass, defaultType, mod, dst, src0, src1, null);
    auto i = encode128A(op, mod, dst, src0, src1, null);
    disallowMod(mod.getCMod());
    i._128J.spf = predicate.getARFBase();
    i._128J.spinv = predicate.getNeg();
    db(i, mod, loc);
}

template <HW hw>
template <typename S1, typename S2>
void BinaryCodeGenerator<hw>::op128K(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, RegData dst, FlagRegister carryOut, RegData src0, S1 src1, S2 src2, FlagRegister carryIn, SourceLocation loc)
{
    auto op = preprocessXe4(opclass, defaultType, mod, dst, src0, src1, null);
    auto i = encode128A(op, mod, dst, src0, src1, null);
    disallowMod(mod.getCMod());
    i._128K.spf = carryIn.isNull()  ? 0xF : carryIn.getARFBase();
    i._128K.dpf = carryOut.isNull() ? 0xF : carryOut.getARFBase();
    db(i, mod, loc);
}

template <HW hw>
template <typename S0>
void BinaryCodeGenerator<hw>::op128L(OpcodeClassXe4 opclass, InstructionModifier mod, bool withID, uint8_t cid, bool exp, S0 src0, SourceLocation loc)
{
    auto defaultType = DataType::invalid;
    auto op = preprocessXe4(opclass, defaultType, mod, null, src0, null, null);

    auto i = encode128A(op, mod, null, null, src0, null);
    disallowMod(mod.getCMod());
    disallowMod(mod.isSaturate());

    i._128.csrc0 = i._128.csrc1;
    i._128L.imm = i._128.imm;
    i._128L.res = 0;

    i._128L.ctype = withID;
    i._128L.cid = cid;
    i._128L.exp = exp;

    db(i, mod, loc);
}

template <HW hw>
void BinaryCodeGenerator<hw>::op128O(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, RegData dst, RegData src0, RegData src1, SourceLocation loc)
{
    if (hw < HW::Xe4) unsupported();

    mod = mod | defaultModifier;

    auto srcType = DataType::invalid;

    if (allowScalarization(dst) && allowScalarization(src0) && allowScalarization(src1))
        opclass = toScalar(opclass);
    validateXe4(mod, opclass);
    validateXe4(dst);
    validateXe4(src0);
    validateXe4(src1);
    processTypesXe4(srcType, src0, src1);
    processTypesXe4(defaultType, dst);
    validateBaseXe4(dst, src0, src1);

    auto op = opcodeXe4(opclass, defaultType);
    auto i = encode128A(op, mod, dst, src0, src1, null);

    disallowMod(src0.getNeg() || src1.getNeg());
    disallowMod(mod.getCMod());

    i._128.cmcf = 0;
    i._128O.sfmt = encodeSrcFormatXe4(srcType);

    db(i, mod, loc);
}

template <HW hw>
void BinaryCodeGenerator<hw>::op128P(OpcodeClassXe4 opclass, InstructionModifier mod, RegData dst, uint64_t cip, SourceLocation loc)
{
    auto defaultType = DataType::invalid;
    auto op = preprocessXe4(opclass, defaultType, mod, dst, null, null, null);

    InstructionXe4 i{mod, op};

    encodeSWSBXe4<2>(i, mod);
    i._128.ipred = mod.isPredInv();
    i._128.pf = encodePFXe4(mod);
    i._128.w = mod.isWrEn();

    disallowMod(mod.isSaturate());
    disallowMod(mod.getCMod());

    i._128P.cip0_11 = cip;
    i._128P.cip12_63 = cip >> 12;
    i._128.cdst = encodeRegXe4(dst);

    db(i, mod, loc);
}

template <HW hw>
template <typename S1, typename S2>
void BinaryCodeGenerator<hw>::op128Q(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, RegData dst, RegData src0, S1 src1, S2 src2, SourceLocation loc)
{
    auto op = preprocessXe4(opclass, defaultType, mod, dst, null, null, src2);
    auto i = encode128A(op, mod, dst, src0, src1, src2);
    i._128.cmcf = 0;
    i._128Q.st0 = (src0.getType() == DataType::s8v4);
    i._128Q.st1 = (src1.getType() == DataType::s8v4);
    db(i, mod, loc);
}

template <HW hw>
template <typename D, typename S0>
void BinaryCodeGenerator<hw>::op128R(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, D dst, S0 src0, SourceLocation loc)
{
    auto op = preprocessXe4(opclass, defaultType, mod, dst, src0, null, null);

    InstructionXe4 i{mod, op};

    encodeSWSBXe4<2>(i, mod);
    i._128.ipred = mod.isPredInv();
    i._128.pf = encodePFXe4(mod);
    i._128.w = mod.isWrEn();
    i._128.ictrl = std::is_base_of<Immediate, S0>::value;

    disallowMod(getNeg(src0));
    disallowMod(mod.getCMod());
    disallowMod(mod.isSaturate());
    disallowMod(mod.getRounding());

    uint64_t imm = 0;
    i._128.csrc0 = encodeRegOrImmXe4<uint64_t, false>(src0, imm);
    i._128R.imm0_11 = imm;
    i._128R.imm12_51 = imm >> 12;
    i._128.cdst = encodeRegXe4(dst);

    db(i, mod, loc);
}

template <HW hw>
template <typename S1>
void BinaryCodeGenerator<hw>::op128S(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, RegData dst, RegData src0, S1 src1, SourceLocation loc)
{
    if (hw < HW::Xe4) unsupported();

    mod = mod | defaultModifier;

    auto dstType = DataType::b32;

    if (allowScalarization(dst) && allowScalarization(src0) && allowScalarization(src1))
        opclass = toScalar(opclass);
    validateXe4(mod, opclass);
    validateXe4(dst);
    validateXe4(src0);
    validateXe4(src1);
    processTypesXe4(defaultType, src0, src1);
    processTypesXe4(dstType, dst);
    validateBaseXe4(dst, src0, src1);

    auto op = opcodeXe4(opclass, defaultType);
    auto i = encode128A(op, mod, dst, src0, src1, null);

    i._128S.sfmt = encodeSrcFormatXe4(defaultType);

    db(i, mod, loc);
}

template <HW hw>
template <typename S1>
void BinaryCodeGenerator<hw>::op64A(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, RegData dst, RegData src0, S1 src1, SourceLocation loc)
{
    auto op = preprocessXe4(opclass, defaultType, mod, dst, src0, src1, null);

    disallowMod(mod.getCMod());
    disallowMod(mod.getPredCtrl());
    disallowMod(mod.getRounding());

    InstructionXe4 i{mod, op};

    encodeSWSB64Xe4(i, mod);

    i._64.s0inv = getNeg(src0);
    i._64.s1inv = getNeg(src1);
    i._64.dsat = mod.isSaturate();
    i._64.w = mod.isWrEn();
    i._64.ictrl = std::is_base_of<Immediate, S1>::value;

    uint64_t imm = 0;
    i._64.csrc1 = encodeRegOrImmXe4<uint16_t, false>(src1, imm);
    i._64.imm = imm;
    i._64.csrc0 = encodeRegXe4(src0);
    i._64.cdst = encodeRegXe4(dst);

    db(i, mod, loc);
}

template <HW hw>
void BinaryCodeGenerator<hw>::op64D(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, uint8_t ctrl, RegData dst, RegData src0, RegData src1, SourceLocation loc)
{
    auto op = preprocessXe4(opclass, defaultType, mod, dst, src0, src1, null);

    disallowMod(getNeg(src0) || getNeg(src1));
    disallowMod(mod.getCMod());
    disallowMod(mod.isSaturate());
    disallowMod(mod.getPredCtrl());
    disallowMod(mod.getRounding());

    InstructionXe4 i{mod, op};

    encodeSWSB64Xe4(i, mod);

    i._64.w = mod.isWrEn();
    i._64D.bctrl = ctrl;
    uint64_t imm = 0;
    i._64.csrc1 = encodeRegOrImmXe4<uint16_t, false>(src1, imm);
    i._64.imm = imm;
    i._64.csrc0 = encodeRegXe4(src0);
    i._64.cdst = encodeRegXe4(dst);

    db(i, mod, loc);
}

template <HW hw>
void BinaryCodeGenerator<hw>::op64E(Opcode op, InstructionModifier mod, SyncFunction fc, SourceLocation loc)
{
    InstructionXe4 i{mod, op};

    unsigned sb0, sb1, sbid;
    encodeSWSBXe4<2>(mod, sb0, sb1, sbid);
    i._64.sb0 = sb0;
    i._64E.sb1 = sb1;
    i._64E.ctrl = encodeSyncFunctionXe4(fc);
    i._64E.ipred = mod.isPredInv();
    i._64E.pf = encodePFXe4(mod);

    db(i, mod, loc);
}

template <HW hw>
void BinaryCodeGenerator<hw>::op64E(Opcode op, InstructionModifier mod, SyncFunction fc, RegData src0, SourceLocation loc)
{
    op64E(op, mod, fc, Immediate::ud(encodeRegXe4(src0)), loc);
}

template <HW hw>
void BinaryCodeGenerator<hw>::op64E(Opcode op, InstructionModifier mod, SyncFunction fc, Immediate src0, SourceLocation loc)
{
    mod = mod | defaultModifier;

    InstructionXe4 i{mod, op};

    unsigned sb0, sb1, sbid;
    i._64E.ctrl = encodeSyncFunctionXe4(fc);
    i._64E.ipred = mod.isPredInv();
    i._64E.pf = encodePFXe4(mod);

    if (fc == SyncFunction::barid) {
        encodeSWSBXe4<2>(mod, sb0, sb1, sbid);
        i._64.sb0 = sb0;
        i._64E.sb1 = sb1;
        i._64E.bar = static_cast<uint64_t>(src0);
    } else {
        encodeSWSBXe4<1>(mod, sb0, sb1, sbid);
        i._64.sb0 = sb0;
        i._64EImm.imm = static_cast<uint64_t>(src0);
    }

    db(i, mod, loc);
}

template <HW hw>
void BinaryCodeGenerator<hw>::op64E(Opcode op, InstructionModifier mod, SyncFunction fc, std::array<SWSBItem, 5> items, SourceLocation loc)
{
    mod = mod | defaultModifier;

    InstructionXe4 i{mod, op};

    i._64E.ctrl = encodeSyncFunctionXe4(fc);
    i._64E.ipred = mod.isPredInv();
    i._64E.pf = encodePFXe4(mod);

    i._64.sb0  = encodeSWSBXe4(items[0]);
    i._64E.sb1 = encodeSWSBXe4(items[1]);
    i._64E.sb2 = encodeSWSBXe4(items[2]);
    i._64E.sb3 = encodeSWSBXe4(items[3]);
    i._64E.sb4 = encodeSWSBXe4(items[4]);

    db(i, mod, loc);
}

template <HW hw>
template <typename S0, typename S1>
void BinaryCodeGenerator<hw>::op64F(OpcodeClassXe4 opclass, DataType defaultType, InstructionModifier mod, RegData dst, S0 src0, S1 src1, SourceLocation loc)
{
    auto op = preprocessXe4(opclass, defaultType, mod, dst, src0, src1, null);

    disallowMod(mod.getCMod());
    disallowMod(mod.isSaturate());
    disallowMod(mod.getRounding());

    InstructionXe4 i{mod, op};

    encodeSWSB64Xe4(i, mod);

    i._64.s1inv = getNeg(src1);
    i._64.s0inv = getNeg(src0);
    i._64.w = mod.isWrEn();
    i._64F.ipred = mod.isPredInv();
    i._64F.pf = encodePFXe4(mod);
    i._64F.ictrl0 = std::is_base_of<Immediate, S0>::value;
    i._64F.ictrl1 = std::is_base_of<Immediate, S1>::value;

    uint64_t imm;
    i._64.csrc1 = encodeRegOrImmXe4<uint16_t, true>(src1, imm);
    i._64.csrc0 = encodeRegOrImmXe4<uint16_t, true>(src0, imm);
    i._64.cdst = encodeRegXe4(dst);

    db(i, mod, loc);
}

#endif /* XE4 */

} /* namespace NGEN_NAMESPACE */

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#endif /* header guard */
