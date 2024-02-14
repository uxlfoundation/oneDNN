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
// -----------------------------------------------------------------------

#include "ngen_asm.hpp"

namespace NGEN_NAMESPACE {

// Forward declarations.
template <HW hw> class BinaryCodeGenerator;
template <HW hw> class ELFCodeGenerator;

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
    static constexpr HW hardware = hw;
    static constexpr HW getHardware() { return hardware; }

protected:
    class InstructionStream {
        friend class BinaryCodeGenerator;

        std::vector<LabelFixup> fixups;
        std::vector<uint32_t> labels;
        std::vector<uint64_t> code;
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

private:
    InstructionModifier defaultModifier;
#if XE3P
    bool useEfficient64Bit = (hw >= HW::Xe3p);
#endif

    LabelManager labelManager;
    InstructionStream rootStream;
    std::vector<InstructionStream*> streamStack;

    void db(const Instruction8 &i)  { streamStack.back()->db(i); }
    void db(const Instruction12 &i) { streamStack.back()->db(i); }
    void addFixup(LabelFixup fixup) { streamStack.back()->addFixup(fixup); }

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
    void opSendg(Opcode op, const InstructionModifier &mod, SharedFunction sfid, const RegData &dst, RegData src0, int src0Len, const RegData &src1, int src1Len, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc);
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

    inline void unsupported();

#include "ngen_compiler_fix.hpp"

public:
    explicit BinaryCodeGenerator(Product product_, DebugConfig debugConfig = {})
        : product{product_}, debugLine(debugConfig), defaultModifier{}, labelManager{},

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

    template <typename String>
    void comment(String)                            {}

    void requireGRF(int grfs)                       { declaredGRFs = grfs; }

    // Registers.
#ifndef NGEN_GLOBAL_REGS
#include "ngen_registers.hpp"
#endif

    // Labels.
    void mark(Label &label)            { streamStack.back()->mark(label, labelManager); }
    void markIfUndefined(Label &label) { if (!label.defined(labelManager)) mark(label); }

    // Instructions.
    template <typename DT = void>
    void add(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::add, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void add(const InstructionModifier &mod, const RegData &dst,
             const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::add, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void addc(const InstructionModifier &mod, const RegData &dst,
              const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::addc, getDataType<DT>(), mod | AccWrEn, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void addc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::addc, getDataType<DT>(), mod | AccWrEn, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void add3(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        if (hw < HW::XeHP) unsupported();
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
        opX(Opcode::add3, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void add3(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        if (hw < HW::XeHP) unsupported();
        opX(Opcode::add3, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void and_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::and_gen12 : Opcode::and_, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void and_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::and_gen12 : Opcode::and_, getDataType<DT>(), mod, dst, src0, src1, loc);
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
    void asr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::asr_gen12 : Opcode::asr, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void asr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::asr_gen12 : Opcode::asr, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void avg(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::avg, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void avg(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::avg, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void bfe(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
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
    template <typename DT = void>
    void bfn(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        if (hw < HW::XeHP) unsupported();
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
        opBfn(Opcode::bfn, getDataType<DT>(), mod, ctrl, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void bfn(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        if (hw < HW::XeHP) unsupported();
        opBfn(Opcode::bfn, getDataType<DT>(), mod, ctrl, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void bfrev(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfrev_gen12 : Opcode::bfrev, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void bfrev(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
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
        opBranch(Opcode::brd, mod, isGen12 ? null.ud() : ip.d(), jip, loc);
    }
    void brd(const InstructionModifier &mod, RegData src0, SourceLocation loc = {}) {
        src0.setRegion(2, 2, 1);
        opBranch<true, true>(Opcode::brd, mod, isGen12 ? null.ud() : ip.d(), src0, loc);
    }
    void break_(const InstructionModifier &mod, Label &jip, Label &uip, SourceLocation loc = {}) {
        opBranch(Opcode::break_, mod, null, jip, uip, loc);
    }
    void call(const InstructionModifier &mod, const RegData &dst, Label &jip, SourceLocation loc = {}) {
        opCall(Opcode::call, mod, dst, jip, loc);
    }
    void call(const InstructionModifier &mod, const RegData &dst, RegData jip, SourceLocation loc = {}) {
        if (isGen12)
            opBranch<true, true>(Opcode::call, mod, dst, jip, loc);
        else {
            jip.setRegion(0, 1, 0);
            opX<true>(Opcode::call, DataType::d, mod, dst, null.ud(0)(0, 1, 0), jip, loc);
        }
    }
    void calla(const InstructionModifier &mod, const RegData &dst, int32_t jip, SourceLocation loc = {}) {
        if (isGen12)
            opBranch<true>(Opcode::calla, mod, dst, jip, loc);
        else
            opX<true>(Opcode::calla, DataType::d, mod, dst, (hw <= HW::Gen9) ? null.ud(0)(2,2,1) : null.ud(0)(0,1,0), Immediate::d(jip), loc);
    }
    void calla(const InstructionModifier &mod, const RegData &dst, RegData jip, SourceLocation loc = {}) {
        if (isGen12)
            opBranch<true, true>(Opcode::calla, mod, dst, jip, loc);
        else {
            jip.setRegion(0, 1, 0);
            opX<true>(Opcode::calla, DataType::d, mod, dst, null.ud(0)(0, 1, 0), jip, loc);
        }
    }
    template <typename DT = void>
    void cbit(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(Opcode::cbit, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void cbit(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        opX(Opcode::cbit, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void cmp(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::cmp_gen12 : Opcode::cmp, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void cmp(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::cmp_gen12 : Opcode::cmp, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void cmpn(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::cmpn_gen12 : Opcode::cmpn, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
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
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        if (hw < HW::Gen12LP) unsupported();
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        if (hw < HW::Gen12LP) unsupported();
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        if (hw < HW::Gen12LP) unsupported();
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
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
    void endif(const InstructionModifier &mod, Label &jip, SourceLocation loc = {}) {
        opBranch(Opcode::endif, mod, null, jip, loc);
    }
    void endif(const InstructionModifier &mod, SourceLocation loc = {}) {
        opBranch(Opcode::endif, mod, null, sizeof(Instruction8), loc);
    }
    template <typename DT = void>
    void fbh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(Opcode::fbh, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void fbh(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        opX(Opcode::fbh, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void fbl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(Opcode::fbl, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void fbl(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        opX(Opcode::fbl, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void frc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(Opcode::frc, getDataType<DT>(), mod, dst, src0, loc);
    }
    void goto_(InstructionModifier mod, Label &jip, Label &uip, bool branchCtrl, SourceLocation loc = {}) {
        mod.setBranchCtrl(branchCtrl);
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
        opX(Opcode::illegal, DataType::invalid, InstructionModifier(), null, null, null, loc);
    }
    void join(InstructionModifier mod, Label &jip, SourceLocation loc = {}) {
        opBranch(Opcode::join, mod, null, jip, loc);
    }
    void join(InstructionModifier mod, SourceLocation loc = {}) {
        opBranch(Opcode::join, mod, null, sizeof(Instruction8), loc);
    }
    void jmpi(const InstructionModifier &mod, Label &jip, SourceLocation loc = {}) {
        auto dst = isGen12 ? ARF(null) : ARF(ip);
        opJmpi(Opcode::jmpi, mod, dst, dst, jip, loc);
    }
    void jmpi(const InstructionModifier &mod, const RegData &jip, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (!isGen12 && jip.getType() != DataType::d && jip.getType() != DataType::invalid)
            throw invalid_type_exception();
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
        if (hardware >= HW::Xe3p) unsupported();
#endif
#endif
        opX(Opcode::mac, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void mac(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE3P
#ifdef NGEN_SAFE
        if (hardware >= HW::Xe3p) unsupported();
#endif
#endif
        opX(Opcode::mac, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void mach(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE3P
#ifdef NGEN_SAFE
        if (hardware >= HW::Xe3p) unsupported();
#endif
#endif
        opX(Opcode::mach, getDataType<DT>(), (hw >= HW::XeHPC) ? mod : (mod | AccWrEn), dst, src0, src1, loc);
    }
    template <typename DT = void>
    void mach(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE3P
#ifdef NGEN_SAFE
        if (hardware >= HW::Xe3p) unsupported();
#endif
#endif
        opX(Opcode::mach, getDataType<DT>(), (hw >= HW::XeHPC) ? mod : (mod | AccWrEn), dst, src0, src1, loc);
    }
    template <typename DT = void>
    void macl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
#if XE3P
        if (hardware >= HW::Xe3p) unsupported();
#endif
        if (hw < HW::Gen10) unsupported();
#endif
        opX((hw >= HW::XeHPC) ? Opcode::macl : Opcode::mach, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void macl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
#if XE3P
        if (hardware >= HW::Xe3p) unsupported();
#endif
        if (hw < HW::Gen10) unsupported();
#endif
        opX((hw >= HW::XeHPC) ? Opcode::macl : Opcode::mach, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
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
        opX(Opcode::madm, getDataType<DT>(), mod, dst, src0, src1, src2, loc);
    }
    template <typename DT = void>
    void math(const InstructionModifier &mod, MathFunction fc, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (mathArgCount(hw, fc) != 1) throw invalid_operand_count_exception();
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
        opMath(Opcode::math, getDataType<DT>(), mod, fc, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void mov(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::mov_gen12 : Opcode::mov, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void mov(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::mov_gen12 : Opcode::mov, getDataType<DT>(), mod, dst, src0, loc);
    }
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
    template <typename DT = void>
    void mul(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::mul, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void mul(const InstructionModifier &mod, const RegData &dst, const RegData &src0, Immediate src1, SourceLocation loc = {}) {
        if (dst.getBytes() == 8)
            src1 = src1.forceInt32();
        opX(Opcode::mul, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
#if XE3P
    template <typename DT = void>
    void mullh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::mullh, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void mullh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, Immediate src1, SourceLocation loc = {}) {
        opX(Opcode::mullh, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
#endif
    void nop(SourceLocation loc = {}) {
        opNop(isGen12 ? Opcode::nop_gen12 : Opcode::nop, loc);
    }
    void nop(const InstructionModifier &mod, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::nop_gen12 : Opcode::nop, DataType::invalid, mod, null, null, null, loc);
    }
    template <typename DT = void>
    void not_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::not_gen12 : Opcode::not_, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void not_(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::not_gen12 : Opcode::not_, getDataType<DT>(), mod, dst, src0, loc);
    }
#ifndef NGEN_NO_OP_NAMES
    template <typename DT = void>
    void not(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        not_<DT>(mod, dst, src0, loc);
    }
    template <typename DT = void>
    void not(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        not_<DT>(mod, dst, src0, loc);
    }
#endif
    template <typename DT = void>
    void or_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::or_gen12 : Opcode::or_, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void or_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::or_gen12 : Opcode::or_, getDataType<DT>(), mod, dst, src0, src1, loc);
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
#endif
    template <typename DT = void>
    void pln(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        if (hw >= HW::Gen11) unsupported();
        opX(Opcode::pln, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    void ret(const InstructionModifier &mod, RegData src0, SourceLocation loc = {}) {
        src0.setRegion(2,2,1);
        if (isGen12)
            opBranch<true, true>(Opcode::ret, mod, null, src0, loc);
        else
            opX<true>(Opcode::ret, DataType::ud, mod, null, src0, loc);
    }
    template <typename DT = void>
    void rndd(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(Opcode::rndd, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void rndd(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        opX(Opcode::rndd, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void rnde(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(Opcode::rnde, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void rnde(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        opX(Opcode::rnde, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void rndu(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(Opcode::rndu, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void rndu(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        opX(Opcode::rndu, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void rndz(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(Opcode::rndz, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void rndz(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        opX(Opcode::rndz, getDataType<DT>(), mod, dst, src0, loc);
    }
    template <typename DT = void>
    void rol(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::rol_gen12 : Opcode::rol, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void rol(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::rol_gen12 : Opcode::rol, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void ror(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::ror_gen12 : Opcode::ror, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void ror(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
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
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, NullRegister(), NullRegister(), desc, loc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, ind0, NullRegister(), desc, loc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, ind0, ind1, desc, loc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const GRFRange &src1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), NullRegister(), NullRegister(), desc, loc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const GRFRange &src1, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), ind0, NullRegister(), desc, loc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const GRFRange &src1, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
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
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, NullRegister(), NullRegister(), desc, loc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, ind0, NullRegister(), desc, loc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, ind0, ind1, desc, loc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const GRFRange &src1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), NullRegister(), NullRegister(), desc, loc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const GRFRange &src1, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), ind0, NullRegister(), desc, loc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const GRFRange &src1, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
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
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, NullRegister(), NullRegister(), desc, loc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, ind0, NullRegister(), desc, loc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, ind0, ind1, desc, loc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const GRFRange &src1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), NullRegister(), NullRegister(), desc, loc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const GRFRange &src1, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), ind0, NullRegister(), desc, loc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const GRFRange &src1, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
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
    void sendgxc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgxc, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, NullRegister(), NullRegister(), desc, loc);
    }
    void sendgxc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgxc, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, ind0, NullRegister(), desc, loc);
    }
    void sendgxc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgxc, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, ind0, ind1, desc, loc);
    }
    void sendgxc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const GRFRange &src1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgxc, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), NullRegister(), NullRegister(), desc, loc);
    }
    void sendgxc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const GRFRange &src1, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgxc, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), ind0, NullRegister(), desc, loc);
    }
    void sendgxc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const GRFRange &src1, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
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

    template <typename DT = void>
    void shl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::shl_gen12 : Opcode::shl, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void shl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::shl_gen12 : Opcode::shl, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void shr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::shr_gen12 : Opcode::shr, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void shr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
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
        opX(Opcode::subb, getDataType<DT>(), mod | AccWrEn, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void subb(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::subb, getDataType<DT>(), mod | AccWrEn, dst, src0, src1, loc);
    }
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
        opX(isGen12 ? Opcode::xor_gen12 : Opcode::xor_, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void xor_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::xor_gen12 : Opcode::xor_, getDataType<DT>(), mod, dst, src0, src1, loc);
    }
#ifndef NGEN_NO_OP_NAMES
    template <typename DT = void>
    void xor(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        xor_<DT>(mod, dst, src0, src1, loc);
    }
    template <typename DT = void>
    void xor(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        xor_<DT>(mod, dst, src0, src1, loc);
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
    };
public:
    Sync sync;

    void ignoredep(Operand op, SourceLocation loc = {}) {
        if (hw >= HW::Gen12LP)
            opX(Opcode::directive, DataType::ud, InstructionModifier(), GRF(static_cast<int>(op)), NullRegister(), NullRegister(), loc);
    }
    void subdep(Operand op, const GRFRange &r, SourceLocation loc = {}) {
        if (op == Operand::dst && !r.isEmpty()) {
#ifdef NGEN_SAFE
            if (r.getLen() > 32) throw invalid_directive_exception();
#endif
            opX(Opcode::directive, DataType::ud, InstructionModifier::createAutoSWSB(), GRF(static_cast<int>(Directive::subdep_dst)), r[0], r[r.getLen() - 1], loc);
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
            opX(Opcode::directive, DataType::ud, InstructionModifier::createAutoSWSB(), GRF(static_cast<int>(Directive::wrdep)), r[o], r[o + thisLen - 1], loc);
        }
    }
    void wrdep(const GRF &r, SourceLocation loc = {}) {
        wrdep(r-r, loc);
    }
    void fencedep(Label &fenceLocation, SourceLocation loc = {}) {
        addFixup(LabelFixup(fenceLocation.getID(labelManager), LabelFixup::JIPOffset));
        opX(Opcode::directive, DataType::ud, InstructionModifier::createAutoSWSB(), GRF(static_cast<int>(Directive::fencedep)), Immediate::ud(0), loc);
    }
    void disablePVCWARWA(SourceLocation loc = {}) {
        opX(Opcode::directive, DataType::ud, InstructionModifier::createAutoSWSB(), GRF(static_cast<int>(Directive::pvcwarwa)), NullRegister(), loc);
    }

    using _self = BinaryCodeGenerator<hw>;
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

#define NGEN_BINARY_DT_OP(op, scope) template <typename DT = void, typename A0, typename A1> void op(const NGEN_NAMESPACE::InstructionModifier &mod, A0 &&a0, A1 &&a1, NGEN_NAMESPACE::SourceLocation loc = {}) {scope::template op<DT>(mod, std::forward<A0>(a0), std::forward<A1>(a1), loc);}
#define NGEN_TERNARY_DT_OP(op, scope) template <typename DT = void, typename A0, typename A1, typename A2> void op(const NGEN_NAMESPACE::InstructionModifier &mod, A0 &&a0, A1 &&a1, A2 &&a2, NGEN_NAMESPACE::SourceLocation loc = {}) {scope::template op<DT>(mod, std::forward<A0>(a0), std::forward<A1>(a1), std::forward<A2>(a2), loc);}
#define NGEN_QUADRARY_DT_OP(op, scope) template <typename DT = void, typename A0, typename A1, typename A2, typename A3> void op(const NGEN_NAMESPACE::InstructionModifier &mod, A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, NGEN_NAMESPACE::SourceLocation loc = {}) {scope::template op<DT>(mod, std::forward<A0>(a0), std::forward<A1>(a1), std::forward<A2>(a2), std::forward<A3>(a3), loc);}
#define NGEN_PENTARY_DT_OP(op, scope) template <typename DT = void, typename A0, typename A1, typename A2, typename A3, typename A4> void op(const NGEN_NAMESPACE::InstructionModifier &mod, A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4, NGEN_NAMESPACE::SourceLocation loc = {}) {scope::template op<DT>(mod, std::forward<A0>(a0), std::forward<A1>(a1), std::forward<A2>(a2), std::forward<A3>(a3), std::forward<A4>(a4), loc);}
#define NGEN_HEXARY_DT_OP(op, scope) template <typename DT = void, typename A0, typename A1, typename A2, typename A3, typename A4, typename A5> void op(const NGEN_NAMESPACE::InstructionModifier &mod, A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4, A5 &&a5, NGEN_NAMESPACE::SourceLocation loc = {}) {scope::template op<DT>(mod, std::forward<A0>(a0), std::forward<A1>(a1), std::forward<A2>(a2), std::forward<A3>(a3), std::forward<A4>(a4), std::forward<A5>(a5), loc);}
#define NGEN_OCTARY_DT_OP(op, scope) template <typename DT = void, typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7> void op(const NGEN_NAMESPACE::InstructionModifier &mod, A0 &&a0, A1 &&a1, A2 &&a2, A3 &&a3, A4 &&a4, A5 &&a5, A6 &&a6, A7 &&a7, NGEN_NAMESPACE::SourceLocation loc = {}) {scope::template op<DT>(mod, std::forward<A0>(a0), std::forward<A1>(a1), std::forward<A2>(a2), std::forward<A3>(a3), std::forward<A4>(a4), std::forward<A5>(a5), std::forward<A6>(a6), std::forward<A7>(a7), loc);}

#define NGEN_FORWARD_SCOPE_DT_OP(op, scope) \
    NGEN_BINARY_DT_OP(op, scope)      \
    NGEN_TERNARY_DT_OP(op, scope)     \
    NGEN_QUADRARY_DT_OP(op, scope)    \
    NGEN_PENTARY_DT_OP(op, scope)     \
    NGEN_HEXARY_DT_OP(op, scope)      \
    NGEN_OCTARY_DT_OP(op, scope)      \

#define NGEN_FORWARD_SCOPE_NO_ELF_OVERRIDES(scope)            \
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
template <typename... Targs> void memfence(Targs&&... args) { scope::memfence(std::forward<Targs>(args)...); } \
template <typename... Targs> void slmfence(Targs&&... args) { scope::slmfence(std::forward<Targs>(args)...); } \
NGEN_NILARY_OP(fencewait, scope) \
template <typename... Targs> void loadlid(Targs&&... args) { scope::loadlid(std::forward<Targs>(args)...); } \
template <typename... Targs> void loadargs(Targs&&... args) { scope::loadargs(std::forward<Targs>(args)...); } \
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
NGEN_FORWARD_SCOPE_EXTRA(scope) \
NGEN_FORWARD_SCOPE_OP_NAMES(scope) \
NGEN_FORWARD_SCOPE_MIN_MAX(scope) \
NGEN_FORWARD_SCOPE_REGISTERS(scope)

#if !XE3P
#define NGEN_FORWARD_SCOPE_EXTRA(scope)
#define NGEN_FORWARD_SCOPE_EXTRA_ELF_OVERRIDES(hw)
#else
#define NGEN_FORWARD_SCOPE_EXTRA(scope)                                        \
  NGEN_FORWARD_SCOPE_DT_OP(bdpas, scope)                                       \
  NGEN_FORWARD_SCOPE_DT_OP(dnscl, scope)                                       \
  using scope::lfsr;                                                           \
  using scope::shfl;                                                           \
  NGEN_FORWARD_SCOPE_OP(sendg, scope)                                          \
  NGEN_FORWARD_SCOPE_OP(sendgc, scope)                                         \
  NGEN_FORWARD_SCOPE_OP(sendgx, scope)                                         \
  NGEN_FORWARD_SCOPE_OP(sendgxc, scope)                                        \
  bool getEfficient64Bit() { return scope::getEfficient64Bit(); }

#define NGEN_FORWARD_SCOPE_EXTRA_ELF_OVERRIDES(hw)                                   \
  template <typename... Targs> void setEfficient64Bit(Targs &&...args) {       \
    NGEN_NAMESPACE::BinaryCodeGenerator<hw>::setEfficient64Bit(                \
        std::forward<Targs>(args)...);                                         \
  }
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
using scope::L1C_L3CC; using scope::L1UC_L3CC;
#define NGEN_FORWARD_SCOPE_REGISTERS_EXTRA1(scope) \
using scope::s0;
#define NGEN_FORWARD_SCOPE_REGISTERS_EXTRA2(scope)
#if !XE3P
#define NGEN_FORWARD_SCOPE_REGISTERS_EXTRA3(scope)
#else
#define NGEN_FORWARD_SCOPE_REGISTERS_EXTRA3(scope) \
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
#define NGEN_FORWARD_SCOPE_REGISTERS(scope) NGEN_FORWARD_SCOPE_REGISTERS_BASE(scope) NGEN_FORWARD_SCOPE_REGISTERS_EXTRA1(scope) NGEN_FORWARD_SCOPE_REGISTERS_EXTRA2(scope) NGEN_FORWARD_SCOPE_REGISTERS_EXTRA3(scope)
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
    autoswsb::BasicBlockList analysis = autoswsb::autoSWSB(hw, declaredGRFs, program);
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

    return result;
}

template <HW hw>
template <bool forceWE, typename D, typename S0, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);
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

    i.binary.dstRegFile = getRegFile(dst);
    i.binary.src0RegFile = getRegFile(src0);

    db(i);
}

template <HW hw>
template <bool forceWE, typename D, typename S0, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);
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
    if (hw >= HW::Xe3p) {
        if (op == Opcode::math)
            i.binaryXe3pImm.src0Reg8 = 0;
        i.binaryXe3p.dstReg8 = getHighBit(dst);
        i.binaryXe3p.src0Reg8 = getHighBit(src0);
    }
#endif

    db(i);
}

template <HW hw>
template <bool forceWE, typename D, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, const Immediate &src0, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

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

    i.binary.dstRegFile = getRegFile(dst);
    i.binary.src0RegFile = getRegFile(src0);

    if (dst.isIndirect())  i.binary.dstAddrImm9 = dst.getOffset() >> 9;

    if (getBytes(src0.getType()) == 8)
        i.imm64.value = static_cast<uint64_t>(src0);
    else
        i.imm32.value = static_cast<uint64_t>(src0);

    db(i);
}

template <HW hw>
template <bool forceWE, typename D, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, const Immediate &src0, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

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
    if (hw >= HW::Xe3p)
        i.unaryXe3pImm.dstReg8 = getHighBit(dst);
#endif

    db(i);
}

template <HW hw>
template <bool forceWE, typename D, typename S0, typename S1, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

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

    i.binary.dstRegFile = getRegFile(dst);
    i.binary.src0RegFile = getRegFile(src0);
    i.binary.src1RegFile = getRegFile(src1);

#ifdef NGEN_SAFE
    if (src1.isARF() && op != Opcode::illegal && op != Opcode::movi && op != Opcode::directive)
        throw grf_expected_exception();
#endif

    db(i);
}

template <HW hw>
template <bool forceWE, typename D, typename S0, typename S1, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};

    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    int ewidth = getExecWidth({defaultType, dst.getType(), src0.getType(), src1.isNull() ? src0.getType()  : src1.getType()});
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
    if (hw >= HW::Xe3p) {
        i.binaryXe3pImm.src0Reg8 = 0;
        i.binaryXe3p.dstReg8 = getHighBit(dst);
        i.binaryXe3p.src0Reg8 = getHighBit(src0);
        i.binaryXe3p.src1Reg8 = getHighBit(src1);
        i.binaryXe3p.src1Scalar = checkSrc1Scalar(op, src1, dst, tag);
    }
#endif

    db(i);
}

template <HW hw>
template <bool forceWE, typename D, typename S0, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, const Immediate &src1, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

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

    i.binary.dstRegFile = getRegFile(dst);
    i.binary.src0RegFile = getRegFile(src0);
    i.binary.src1RegFile = getRegFile(src1);

    i.imm32.value = static_cast<uint64_t>(src1);

    db(i);
}

template <HW hw>
template <bool forceWE, typename D, typename S0, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, const Immediate &src1, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

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
    if (hw >= HW::Xe3p) {
        i.binaryXe3pImm.dstReg8 = getHighBit(dst);
        i.binaryXe3pImm.src0Reg8 = getHighBit(src0);
    }
#endif

    db(i);
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
    debugLine.add(rootStream.length(), loc);

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

    db(i);
}

template <HW hw>
template <typename D, typename S0, typename S1, typename S2, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2, SourceLocation loc)
{
    if (hw < HW::Gen10)
        unsupported();

    debugLine.add(rootStream.length(), loc);

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

    db(i);
}

template <HW hw>
template <typename D, typename S0,typename S1, typename S2, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

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

    db(i);
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
    if (hw < HW::XeHP)
        unsupported();

    debugLine.add(rootStream.length(), loc);

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

    /* SYCL + GCC 12.3 workaround                    */
    /* encodeTernaryTypes(i, dst, src0, src1, src2); */
    Instruction12 i2;
    encodeTernaryTypes(i2, dst, src0, src1, src2);

    i.ternary.execType = i2.ternary.execType;
    i.ternary.dstType  = i2.ternary.dstType;
    i.ternary.src0Type = i2.ternary.src0Type;
    i.ternary.src1Type = i2.ternary.src1Type;
    i.ternary.src2Type = i2.ternary.src2Type;
    /*************************************************/

    i.ternary.cmod = static_cast<int>(mod.getCMod());

    i.bfn.bfnCtrl03 = (bfnCtrl >> 0);
    i.bfn.bfnCtrl47 = (bfnCtrl >> 4);

#if XE3P
    encodeTernary512GRF(i, dst, src0, src1, src2, tag);
#endif

    db(i);
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
    if (hw < HW::XeHP)
        unsupported();

    debugLine.add(rootStream.length(), loc);

    Instruction12 i{};
    encodeDPAS<hw>(i, op, defaultType, mod | defaultModifier, sdepth, rcount, dst, src0, src1, src2);
    db(i);
}

#if XE3P
template <HW hw>
void BinaryCodeGenerator<hw>::opBdpas(Opcode op, DataType defaultType, const InstructionModifier &mod, int sdepth, int rcount,
                                      RegData dst, RegData src0, RegData src1, RegData src2, RegData src3, RegData src4, SourceLocation loc)
{
    if (hw < HW::Xe3p)
        unsupported();

    debugLine.add(rootStream.length(), loc);

    Instruction12 i{};

    encodeDPAS<hw>(i, op, defaultType, mod | defaultModifier, sdepth, rcount, dst, src0, src1, src2);

    src3.fixup(hw, mod.getExecSize(), 0, DataType::ub, 3, 5);
    src4.fixup(hw, mod.getExecSize(), 0, DataType::ub, 4, 5);

    int s3r = src3.getBase(), s4r = src4.getBase();

    i.bdpas.src3RegFile = getRegFile(src3);
    i.bdpas.src4RegFile = getRegFile(src4);

    i.bdpas.src3Reg0 = s3r;
    i.bdpas.src3Reg1_2 = s3r >> 1;
    i.bdpas.src3Reg3_6 = s3r >> 3;
    i.bdpas.src3Reg7_8 = s3r >> 7;
    i.bdpas.src3SubReg4_5 = src3.getByteOffset() >> 4;

    i.bdpas.src4Reg0_3 = s4r;
    i.bdpas.src4Reg4_8 = s4r >> 4;
    i.bdpas.src4SubReg3_5 = src4.getByteOffset() >> 3;

    db(i);
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
    debugLine.add(rootStream.length(), loc);

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

    i.send.dstRegFile = getRegFile(dst);
    i.send.src0RegFile = getRegFile(src0);
    i.send.src1RegFile = getRegFile(src1);

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

    db(i);
}

template <HW hw>
template <HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSend(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon8(i, op, emod);

    i.binary.dst  = encodeBinaryOperand8<true>(dst).bits;
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;

    i.sendsGen9.dstRegFile = getRegFile(dst);
    i.binary.src0RegFile = getRegFile(src0);
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

    db(i);
}

template <HW hw>
template <HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSend(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

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

    i.sendsGen9.dstRegFile = getRegFile(dst);
    i.binary.src0RegFile = getRegFile(src0);
    i.binary.src1RegFile = getRegFile(desc);
    i.binary.src1Type = getTypecode<hw>(desc.getType());

    i.sendsGen9.sfid = exdesc & 0xF;
    i.sendGen8.zero = 0;
    i.sendGen8.exDesc16_19 = (exdesc >> 16) & 0xF;
    i.sendGen8.exDesc20_23 = (exdesc >> 20) & 0xF;
    i.sendGen8.exDesc24_27 = (exdesc >> 24) & 0xF;
    i.sendGen8.exDesc28_31 = (exdesc >> 28) & 0xF;

    i.sendsGen9.eot = (exdesc >> 5) & 1;
    if (dst.isIndirect()) i.sendsGen9.dstAddrImm9 = dst.getOffset() >> 9;

    db(i);
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
    debugLine.add(rootStream.length(), loc);

    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon8(i, op, emod);

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;

    i.binary.src0RegFile = 0;                   // ?
    i.sendsGen9.dstRegFile = getRegFile(dst);
    i.sendsGen9.src1RegFile = getRegFile(src1);
    i.sendsGen9.src1RegNum = src1.getBase();

    if (dst.isIndirect())  i.sendsGen9.dstAddrImm9  =  dst.getOffset() >> 9;
    if (src0.isIndirect()) i.sendsGen9.src0AddrImm9 = src0.getOffset() >> 9;

    encodeSendsDesc(i, desc);
    encodeSendsExDesc(i, exdesc);

    db(i);
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
                                      const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

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

        i.sendg.dstRegFile = getRegFile(dst);
        i.sendg.src0RegFile = getRegFile(src0);
        i.sendg.src1RegFile = getRegFile(src1);

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

    db(i);
}
#endif

template <HW hw>
template <HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, int32_t uip, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon8(i, op, emod);

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.dstRegFile = getRegFile(dst);
    i.binary.dstType = getTypecode<hw>(dst.getType());
    i.binary.src0RegFile = getRegFile(Immediate());
    i.binary.src0Type = getTypecode<hw>(DataType::d);
    i.branches.jip = jip;
    i.branches.uip = uip;

    db(i);
}

template <HW hw>
template <HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, int32_t uip, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

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
    if (hw >= HW::Xe3p)
        i.branchXe3p.dstReg8 = getHighBit(dst);
#endif

    db(i);
}

template <HW hw>
template <bool forceWE, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    encodeCommon8(i, op, emod);

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.dstRegFile = getRegFile(dst);
    i.binary.dstType = getTypecode<hw>(dst.getType());
    i.binary.src1RegFile = RegFileIMM;
    i.binary.src1Type = getTypecode<hw>(DataType::d);
    i.branches.jip = jip;

    db(i);
}

template <HW hw>
template <bool forceWE, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

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
    if (hw >= HW::Xe3p)
        i.branchXe3p.dstReg8 = getHighBit(dst);
#endif

    db(i);
}

template <HW hw>
template <bool forceWE, bool small12, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier;
    if (forceWE)
        emod |= NoMask;

    encodeCommon8(i, op, emod);

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.dstRegFile = getRegFile(dst);
    i.binary.dstType = getTypecode<hw>(DataType::d);
    i.binary.src0RegFile = getRegFile(src0);
    i.binary.src0Type = getTypecode<hw>(DataType::d);
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;

    db(i);
}

template <HW hw>
template <bool forceWE, bool small12, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

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
    if (hw >= HW::Xe3p) {
        i.branchXe3p.dstReg8 = getHighBit(dst);
        i.branchXe3p.src0Reg8 = getHighBit(src0);
    }
#endif

    db(i);
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
#ifdef NGEN_SAFE
        if (mod.getExecSize() != 1) throw invalid_modifiers_exception();
#endif
    debugLine.add(rootStream.length(), loc);

    Instruction8 i{};
    InstructionModifier emod = mod | defaultModifier | NoMask;

    encodeCommon8(i, op, emod);

    src0.fixup(hw, emod.getExecSize(), 0, DataType::d, 0, 2);

    i.binary.dst = encodeBinaryOperand8<true>(dst).bits;
    i.binary.src0 = encodeBinaryOperand8<false>(src0).bits;
    i.binary.src0RegFile = getRegFile(src0);
    i.binary.src1RegFile = RegFileIMM;
    i.binary.src1Type = getTypecode<hw>(DataType::d);

    i.branches.jip = jip;

    db(i);
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
    if (hw < HW::Gen12LP)
        unsupported();

    debugLine.add(rootStream.length(), loc);

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon12(i, op, emod, null, tag);

    i.binary.dst = 0x1;
    i.binary.cmod = static_cast<int>(fc);

    db(i);
}

template <HW hw>
void BinaryCodeGenerator<hw>::opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod, RegData src0, SourceLocation loc)
{
    typename EncodingTag12Dispatch<hw>::tag tag;
    if (hw < HW::Gen12LP)
        unsupported();

    debugLine.add(rootStream.length(), loc);

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

    db(i);
}

template <HW hw>
void BinaryCodeGenerator<hw>::opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod, const Immediate &src0, SourceLocation loc)
{
    if (hw < HW::Gen12LP)
        unsupported();

    debugLine.add(rootStream.length(), loc);

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon12(i, op, emod, null, tag);

    i.binary.dst = 0x1;
    i.binary.src0Type = getTypecode12(src0.getType());
    i.binary.src0Imm = true;
    i.binary.cmod = static_cast<int>(fc);

    i.imm32.value = uint32_t(static_cast<uint64_t>(src0));

    db(i);
}

template <HW hw>
void BinaryCodeGenerator<hw>::opNop(Opcode op, SourceLocation loc)
{
    debugLine.add(rootStream.length(), loc);

    Instruction8 i{};

    i.qword[0] = static_cast<int>(op);
    i.qword[1] = 0;

    db(i);
}

} /* namespace NGEN_NAMESPACE */

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#endif /* header guard */
