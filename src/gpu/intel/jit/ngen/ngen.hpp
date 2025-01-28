/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifdef ENABLE_LLVM_WCONVERSION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-int-conversion"
#endif

#include "ngen_config.hpp"

#include <array>
#include <cstring>
#include <type_traits>
#include <vector>

#include "ngen_core.hpp"
#include "ngen_auto_swsb.hpp"

namespace NGEN_NAMESPACE {

// Forward declarations.
template <HW hw> class BinaryCodeGenerator;
template <HW hw> class ELFCodeGenerator;

// MSVC v140 workaround for enum comparison in template arguments.
static constexpr bool hwLT(HW hw1, HW hw2) { return hw1 < hw2; }
static constexpr bool hwLE(HW hw1, HW hw2) { return hw1 <= hw2; }
static constexpr bool hwGE(HW hw1, HW hw2) { return hw1 >= hw2; }
static constexpr bool hwGT(HW hw1, HW hw2) { return hw1 > hw2; }

// -----------------------------------------------------------------------
// Binary formats, split between pre-Gen12 and post-Gen12.

#include "ngen_gen8.hpp"
#include "ngen_gen12.hpp"

// -----------------------------------------------------------------------


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

    Label _labelLocalIDsLoaded;
    Label _labelArgsLoaded;
    Label _lastFenceLabel;
    RegData _lastFenceDst;

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
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0);
    template <bool forceWE = false, typename D, typename S0, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0);
    template <bool forceWE = false, typename D, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, const Immediate &src0);
    template <bool forceWE = false, typename D, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, const Immediate &src0);

    template <bool forceWE = false, typename D, typename S0, typename S1, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1);
    template <bool forceWE = false, typename D, typename S0, typename S1, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1);
    template <bool forceWE = false, typename D, typename S0, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, const Immediate &src1);
    template <bool forceWE = false, typename D, typename S0, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, const Immediate &src1);

    template <HW hw_ = hw>
    typename std::enable_if<hwLE(hw_, HW::Gen9)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, RegData dst, RegData src0, RegData src1, RegData src2);
    template <HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, Align16Operand dst, Align16Operand src0, Align16Operand src1, Align16Operand src2);
    template <typename D, typename S0, typename S1, typename S2, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2);
    template <typename D, typename S0, typename S1, typename S2, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2);

    template <typename DS0>
    void opMath(Opcode op, DataType defaultType, const InstructionModifier &mod, MathFunction fc, DS0 dst, DS0 src0);
    template <typename DS0, typename S1>
    void opMath(Opcode op, DataType defaultType, const InstructionModifier &mod, MathFunction fc, DS0 dst, DS0 src0, S1 src1);

    template <typename D, typename S0, typename S2>
    void opBfn(Opcode op, DataType defaultType, const InstructionModifier &mod, int bfnCtrl, D dst, S0 src0, RegData src1, S2 src2);
    void opDpas(Opcode op, DataType defaultType, const InstructionModifier &mod, int sdepth, int rcount, RegData dst, RegData src0, RegData src1, RegData src2);

    template <typename D, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opSend(Opcode op, const InstructionModifier &mod, SharedFunction sfid, const RegData &dst, const RegData &src0, const RegData &src1, int src1Length, uint32_t exdesc, D desc);
    template <typename D, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opSend(Opcode op, const InstructionModifier &mod, SharedFunction sfid, const RegData &dst, const RegData &src0, const RegData &src1, int src1Length, const RegData &exdesc, D desc);
    template <typename ED, typename D, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opSend(Opcode op, const InstructionModifier &mod, SharedFunction sfid, const RegData &dst, const RegData &src0, const RegData &src1, int src1Length, ED exdesc, D desc);

    template <HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opSend(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc);
    template <HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opSend(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc);
    template <typename D, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opSend(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, D desc);

    template <typename ED, typename D, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opSends(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, ED exdesc, D desc);
    template <typename D, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opSends(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, D desc);
    template <typename D, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opSends(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, RegData exdesc, D desc);

#if XE3P
    void opSendg(Opcode op, const InstructionModifier &mod, SharedFunction sfid, const RegData &dst, RegData src0, int src0Len, const RegData &src1, int src1Len, const RegData &ind0, const RegData &ind1, uint64_t desc);
#endif

    template <HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, int32_t uip);
    template <HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, int32_t uip);
    template <bool forceWE = false, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip);
    template <bool forceWE = false, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip);
    template <bool forceWE = false, bool small12 = true, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0);
    template <bool forceWE = false, bool small12 = true, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0);

    void opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, Label &jip, Label &uip);
    template <bool forceWE = false>
    void opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, Label &jip);
    void opCall(Opcode op, const InstructionModifier &mod, const RegData &dst, Label &jip);

    template <HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type opJmpi(Opcode op, const InstructionModifier &mod, const RegData &dst, RegData src0, uint32_t jip);
    template <HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type opJmpi(Opcode op, const InstructionModifier &mod, const RegData &dst, RegData src0, uint32_t jip);
    void opJmpi(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, Label &jip);

#if XE3P
    void opShfl(Opcode op, ShuffleFunction fc, DataType defaultType, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1);
#endif

    void opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod);
    void opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod, RegData src0);
    void opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod, const Immediate &src0);

    void opNop(Opcode op);

    inline void unsupported();

#include "ngen_compiler_fix.hpp"

public:
    explicit BinaryCodeGenerator(Product product_) : product{product_}, defaultModifier{}, labelManager{},
#if XE3P
                                                     shfl{this},
#endif
                                                     sync{this}, load{this}, store{this}, atomic{this}
    {
        _workaround_();
        pushStream(rootStream);
    }

    explicit BinaryCodeGenerator(int stepping_ = 0) : BinaryCodeGenerator({genericProductFamily(hw), stepping_}) {}

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
    inline void mark(Label &label)          { streamStack.back()->mark(label, labelManager); }

    // Instructions.
    template <typename DT = void>
    void add(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::add, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void add(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::add, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void addc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::addc, getDataType<DT>(), mod | AccWrEn, dst, src0, src1);
    }
    template <typename DT = void>
    void addc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::addc, getDataType<DT>(), mod | AccWrEn, dst, src0, src1);
    }
    template <typename DT = void>
    void add3(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2) {
        if (hw < HW::XeHP) unsupported();
        opX(Opcode::add3, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void add3(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2) {
        if (hw < HW::XeHP) unsupported();
        opX(Opcode::add3, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void add3(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2) {
        if (hw < HW::XeHP) unsupported();
        opX(Opcode::add3, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void add3(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2) {
        if (hw < HW::XeHP) unsupported();
        opX(Opcode::add3, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void and_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::and_gen12 : Opcode::and_, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void and_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::and_gen12 : Opcode::and_, getDataType<DT>(), mod, dst, src0, src1);
    }
#ifndef NGEN_NO_OP_NAMES
    template <typename DT = void>
    void and(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        and_<DT>(mod, dst, src0, src1);
    }
    template <typename DT = void>
    void and(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        and_<DT>(mod, dst, src0, src1);
    }
#endif
    template <typename DT = void>
    void asr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::asr_gen12 : Opcode::asr, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void asr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::asr_gen12 : Opcode::asr, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void avg(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::avg, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void avg(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::avg, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void bfe(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2) {
        opX(isGen12 ? Opcode::bfe_gen12 : Opcode::bfe, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void bfe(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2) {
        opX(isGen12 ? Opcode::bfe_gen12 : Opcode::bfe, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void bfe(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2) {
        opX(isGen12 ? Opcode::bfe_gen12 : Opcode::bfe, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void bfe(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2) {
        opX(isGen12 ? Opcode::bfe_gen12 : Opcode::bfe, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void bfi1(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::bfi1_gen12 : Opcode::bfi1, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void bfi1(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::bfi1_gen12 : Opcode::bfi1, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void bfi2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2) {
        opX(isGen12 ? Opcode::bfi2_gen12 : Opcode::bfi2, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void bfi2(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2) {
        opX(isGen12 ? Opcode::bfi2_gen12 : Opcode::bfi2, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void bfi2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2) {
        opX(isGen12 ? Opcode::bfi2_gen12 : Opcode::bfi2, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void bfi2(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2) {
        opX(isGen12 ? Opcode::bfi2_gen12 : Opcode::bfi2, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void bfn(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2) {
        if (hw < HW::XeHP) unsupported();
        opBfn(Opcode::bfn, getDataType<DT>(), mod, ctrl, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void bfn(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2) {
        if (hw < HW::XeHP) unsupported();
        opBfn(Opcode::bfn, getDataType<DT>(), mod, ctrl, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void bfn(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2) {
        if (hw < HW::XeHP) unsupported();
        opBfn(Opcode::bfn, getDataType<DT>(), mod, ctrl, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void bfn(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2) {
        if (hw < HW::XeHP) unsupported();
        opBfn(Opcode::bfn, getDataType<DT>(), mod, ctrl, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void bfrev(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(isGen12 ? Opcode::bfrev_gen12 : Opcode::bfrev, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void bfrev(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(isGen12 ? Opcode::bfrev_gen12 : Opcode::bfrev, getDataType<DT>(), mod, dst, src0);
    }
    void brc(const InstructionModifier &mod, Label &jip, Label &uip) {
        opBranch(Opcode::brc, mod, isGen12 ? null.ud() : ip.d(), jip, uip);
    }
    void brc(const InstructionModifier &mod, RegData src0) {
        src0.setRegion(2, 2, 1);
        opBranch<true, true>(Opcode::brc, mod, isGen12 ? null.ud() : ip.d(), src0);
    }
    void brd(const InstructionModifier &mod, Label &jip) {
        opBranch(Opcode::brd, mod, isGen12 ? null.ud() : ip.d(), jip);
    }
    void brd(const InstructionModifier &mod, RegData src0) {
        src0.setRegion(2, 2, 1);
        opBranch<true, true>(Opcode::brd, mod, isGen12 ? null.ud() : ip.d(), src0);
    }
    void break_(const InstructionModifier &mod, Label &jip, Label &uip) {
        opBranch(Opcode::break_, mod, null, jip, uip);
    }
    void call(const InstructionModifier &mod, const RegData &dst, Label &jip) {
        opCall(Opcode::call, mod, dst, jip);
    }
    void call(const InstructionModifier &mod, const RegData &dst, RegData jip) {
        if (isGen12)
            opBranch<true, true>(Opcode::call, mod, dst, jip);
        else {
            jip.setRegion(0, 1, 0);
            opX<true>(Opcode::call, DataType::d, mod, dst, null.ud(0)(0, 1, 0), jip);
        }
    }
    void calla(const InstructionModifier &mod, const RegData &dst, int32_t jip) {
        if (isGen12)
            opBranch<true>(Opcode::calla, mod, dst, jip);
        else
            opX<true>(Opcode::calla, DataType::d, mod, dst, (hw <= HW::Gen9) ? null.ud(0)(2,2,1) : null.ud(0)(0,1,0), Immediate::d(jip));
    }
    void calla(const InstructionModifier &mod, const RegData &dst, RegData jip) {
        if (isGen12)
            opBranch<true, true>(Opcode::calla, mod, dst, jip);
        else {
            jip.setRegion(0, 1, 0);
            opX<true>(Opcode::calla, DataType::d, mod, dst, null.ud(0)(0, 1, 0), jip);
        }
    }
    template <typename DT = void>
    void cbit(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(Opcode::cbit, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void cbit(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(Opcode::cbit, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void cmp(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::cmp_gen12 : Opcode::cmp, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void cmp(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::cmp_gen12 : Opcode::cmp, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void cmpn(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::cmpn_gen12 : Opcode::cmpn, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void csel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2) {
        opX(isGen12 ? Opcode::csel_gen12 : Opcode::csel, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void csel(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2) {
        opX(isGen12 ? Opcode::csel_gen12 : Opcode::csel, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void csel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2) {
        opX(isGen12 ? Opcode::csel_gen12 : Opcode::csel, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void csel(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2) {
        opX(isGen12 ? Opcode::csel_gen12 : Opcode::csel, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    void cont(const InstructionModifier &mod, Label &jip, Label &uip) {
        opBranch(Opcode::cont, mod, null, jip, uip);
    }
    template <typename DT = void>
    void dp2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::dp2, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dp2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::dp2, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dp3(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::dp3, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dp3(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::dp3, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dp4(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::dp4, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dp4(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::dp4, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2) {
        if (hw < HW::Gen12LP) unsupported();
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2) {
        if (hw < HW::Gen12LP) unsupported();
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2) {
        if (hw < HW::Gen12LP) unsupported();
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2) {
        if (hw < HW::Gen12LP) unsupported();
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void dpas(const InstructionModifier &mod, uint8_t sdepth, uint8_t rcount, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2) {
        opDpas(Opcode::dpas, getDataType<DT>(), mod, sdepth, rcount, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void dpasw(const InstructionModifier &mod, uint8_t sdepth, uint8_t rcount, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2) {
        opDpas(Opcode::dpasw, getDataType<DT>(), mod, sdepth, rcount, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void dph(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::dph, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dph(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::dph, getDataType<DT>(), mod, dst, src0, src1);
    }
    void else_(InstructionModifier mod, Label &jip, Label &uip, bool branchCtrl = false) {
        mod.setBranchCtrl(branchCtrl);
        opBranch(Opcode::else_, mod, null, jip, uip);
    }
    void else_(InstructionModifier mod, Label &jip) {
        else_(mod, jip, jip);
    }
    void endif(const InstructionModifier &mod, Label &jip) {
        opBranch(Opcode::endif, mod, null, jip);
    }
    void endif(const InstructionModifier &mod) {
        opBranch(Opcode::endif, mod, null, sizeof(Instruction8));
    }
    template <typename DT = void>
    void fbh(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(Opcode::fbh, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void fbh(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(Opcode::fbh, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void fbl(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(Opcode::fbl, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void fbl(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(Opcode::fbl, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void frc(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(Opcode::frc, getDataType<DT>(), mod, dst, src0);
    }
    void goto_(InstructionModifier mod, Label &jip, Label &uip, bool branchCtrl = false) {
        mod.setBranchCtrl(branchCtrl);
        opBranch(Opcode::goto_, mod, null, jip, uip);
    }
    void goto_(const InstructionModifier &mod, Label &jip) {
        goto_(mod, jip, jip);
    }
    void halt(const InstructionModifier &mod, Label &jip, Label &uip) {
        opBranch(Opcode::halt, mod, null, jip, uip);
    }
    void halt(const InstructionModifier &mod, Label &jip) {
        halt(mod, jip, jip);
    }
    void if_(InstructionModifier mod, Label &jip, Label &uip, bool branchCtrl = false) {
        mod.setBranchCtrl(branchCtrl);
        opBranch(Opcode::if_, mod, null, jip, uip);
    }
    void if_(const InstructionModifier &mod, Label &jip) {
        if_(mod, jip, jip);
    }
    void illegal() {
        opX(Opcode::illegal, DataType::invalid, InstructionModifier(), null, null, null);
    }
    void join(InstructionModifier mod, Label &jip) {
        opBranch(Opcode::join, mod, null, jip);
    }
    void join(InstructionModifier mod) {
        opBranch(Opcode::join, mod, null, sizeof(Instruction8));
    }
    void jmpi(const InstructionModifier &mod, Label &jip) {
        auto dst = isGen12 ? ARF(null) : ARF(ip);
        opJmpi(Opcode::jmpi, mod, dst, dst, jip);
    }
    void jmpi(const InstructionModifier &mod, const RegData &jip) {
#ifdef NGEN_SAFE
        if (!isGen12 && jip.getType() != DataType::d && jip.getType() != DataType::invalid)
            throw invalid_type_exception();
#endif
        if (isGen12)
            opBranch<true, false>(Opcode::jmpi, mod, null, jip);
        else
            opX(Opcode::jmpi, DataType::d, mod, ip, ip, jip);
    }
    template <typename DT = void>
    void line(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        if (hw >= HW::Gen11) unsupported();
        opX(Opcode::line, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void line(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        if (hw >= HW::Gen11) unsupported();
        opX(Opcode::line, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void lrp(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2) {
        opX(Opcode::lrp, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void lzd(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(Opcode::lzd, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void lzd(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(Opcode::lzd, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void mac(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
#if XE3P
#ifdef NGEN_SAFE
        if (hardware >= HW::Xe3p) unsupported();
#endif
#endif
        opX(Opcode::mac, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void mac(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
#if XE3P
#ifdef NGEN_SAFE
        if (hardware >= HW::Xe3p) unsupported();
#endif
#endif
        opX(Opcode::mac, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void mach(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
#if XE3P
#ifdef NGEN_SAFE
        if (hardware >= HW::Xe3p) unsupported();
#endif
#endif
        opX(Opcode::mach, getDataType<DT>(), (hw >= HW::XeHPC) ? mod : (mod | AccWrEn), dst, src0, src1);
    }
    template <typename DT = void>
    void mach(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
#if XE3P
#ifdef NGEN_SAFE
        if (hardware >= HW::Xe3p) unsupported();
#endif
#endif
        opX(Opcode::mach, getDataType<DT>(), (hw >= HW::XeHPC) ? mod : (mod | AccWrEn), dst, src0, src1);
    }
    template <typename DT = void>
    void macl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
#ifdef NGEN_SAFE
#if XE3P
        if (hardware >= HW::Xe3p) unsupported();
#endif
        if (hw < HW::Gen10) unsupported();
#endif
        opX((hw >= HW::XeHPC) ? Opcode::macl : Opcode::mach, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void macl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
#ifdef NGEN_SAFE
#if XE3P
        if (hardware >= HW::Xe3p) unsupported();
#endif
        if (hw < HW::Gen10) unsupported();
#endif
        opX((hw >= HW::XeHPC) ? Opcode::macl : Opcode::mach, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2) {
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2) {
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2) {
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2) {
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void, HW hw_ = hw>
    typename std::enable_if<hwLE(hw_, HW::Gen9)>::type
    madm(const InstructionModifier &mod, const ExtendedReg &dst, const ExtendedReg &src0, const ExtendedReg &src1, const ExtendedReg &src2) {
        opX(Opcode::madm, getDataType<DT>(), mod, extToAlign16(dst), extToAlign16(src0), extToAlign16(src1), extToAlign16(src2));
    }
    template <typename DT = void, HW hw_ = hw>
    typename std::enable_if<hwGT(hw_, HW::Gen9)>::type
    madm(const InstructionModifier &mod, const ExtendedReg &dst, ExtendedReg src0, ExtendedReg src1, const ExtendedReg &src2) {
        src0.getBase().setRegion(4,4,1);
        src1.getBase().setRegion(4,4,1);
        opX(Opcode::madm, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void math(const InstructionModifier &mod, MathFunction fc, const RegData &dst, const RegData &src0) {
#ifdef NGEN_SAFE
        if (mathArgCount(hw, fc) != 1) throw invalid_operand_count_exception();
#endif
        opMath(Opcode::math, getDataType<DT>(), mod, fc, dst, src0);
    }
    template <typename DT = void>
    void math(const InstructionModifier &mod, MathFunction fc, const RegData &dst, const RegData &src0, const RegData &src1) {
#ifdef NGEN_SAFE
        if (mathArgCount(hw, fc) != 2) throw invalid_operand_count_exception();
#endif
        opMath(Opcode::math, getDataType<DT>(), mod, fc, dst, src0, src1);
    }
    template <typename DT = void>
    void math(const InstructionModifier &mod, MathFunction fc, const RegData &dst, const RegData &src0, const Immediate &src1) {
#ifdef NGEN_SAFE
        if (fc == MathFunction::invm || fc == MathFunction::rsqtm) throw invalid_operand_exception();
#endif
        opMath(Opcode::math, getDataType<DT>(), mod, fc, dst, src0, src1.forceInt32());
    }
    template <typename DT = void, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen11)>::type
    math(const InstructionModifier &mod, MathFunction fc, const ExtendedReg &dst, const ExtendedReg &src0) {
#ifdef NGEN_SAFE
        if (fc != MathFunction::rsqtm) throw invalid_operand_exception();
#endif
        opMath(Opcode::math, getDataType<DT>(), mod, fc, extToAlign16(dst), extToAlign16(src0));
    }
    template <typename DT = void, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen11)>::type
    math(const InstructionModifier &mod, MathFunction fc, const ExtendedReg &dst, ExtendedReg src0) {
#ifdef NGEN_SAFE
        if (fc != MathFunction::rsqtm) throw invalid_operand_exception();
#endif
        if (hw == HW::Gen11)
            src0.getBase().setRegion(2,2,1);
        else
            src0.getBase().setRegion(1,1,0);
        opMath(Opcode::math, getDataType<DT>(), mod, fc, dst, src0);
    }
    template <typename DT = void, HW hw_ = hw>
    typename std::enable_if<hwLT(hw_, HW::Gen11)>::type
    math(const InstructionModifier &mod, MathFunction fc, const ExtendedReg &dst, const ExtendedReg &src0, const ExtendedReg &src1) {
#ifdef NGEN_SAFE
        if (fc != MathFunction::invm) throw invalid_operand_exception();
#endif
        opMath(Opcode::math, getDataType<DT>(), mod, fc, extToAlign16(dst), extToAlign16(src0), extToAlign16(src1));
    }
    template <typename DT = void, HW hw_ = hw>
    typename std::enable_if<hwGE(hw_, HW::Gen11)>::type
    math(const InstructionModifier &mod, MathFunction fc, const ExtendedReg &dst, ExtendedReg src0, ExtendedReg src1) {
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
        opMath(Opcode::math, getDataType<DT>(), mod, fc, dst, src0, src1);
    }
    template <typename DT = void>
    void mov(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(isGen12 ? Opcode::mov_gen12 : Opcode::mov, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void mov(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(isGen12 ? Opcode::mov_gen12 : Opcode::mov, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void movi(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        if (hardware >= HW::Gen10)
            movi<DT>(mod, dst, src0, null.ud(0)(1,1,0));
        else
            opX(isGen12 ? Opcode::movi_gen12 : Opcode::movi, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void movi(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
#ifdef NGEN_SAFE
        if (hardware < HW::Gen10) throw unsupported_instruction();
#endif
        opX(isGen12 ? Opcode::movi_gen12 : Opcode::movi, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void movi(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
#ifdef NGEN_SAFE
        if (hardware < HW::Gen10) throw unsupported_instruction();
#endif
        opX(isGen12 ? Opcode::movi_gen12 : Opcode::movi, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void mul(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::mul, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void mul(const InstructionModifier &mod, const RegData &dst, const RegData &src0, Immediate src1) {
        if (dst.getBytes() == 8)
            src1 = src1.forceInt32();
        opX(Opcode::mul, getDataType<DT>(), mod, dst, src0, src1);
    }
#if XE3P
    template <typename DT = void>
    void mullh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::mullh, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void mullh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, Immediate src1) {
        opX(Opcode::mullh, getDataType<DT>(), mod, dst, src0, src1);
    }
#endif
    void nop() {
        opNop(isGen12 ? Opcode::nop_gen12 : Opcode::nop);
    }
    void nop(const InstructionModifier &mod) {
        opX(isGen12 ? Opcode::nop_gen12 : Opcode::nop, DataType::invalid, mod, null, null, null);
    }
    template <typename DT = void>
    void not_(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(isGen12 ? Opcode::not_gen12 : Opcode::not_, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void not_(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(isGen12 ? Opcode::not_gen12 : Opcode::not_, getDataType<DT>(), mod, dst, src0);
    }
#ifndef NGEN_NO_OP_NAMES
    template <typename DT = void>
    void not(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        not_<DT>(mod, dst, src0);
    }
    template <typename DT = void>
    void not(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        not_<DT>(mod, dst, src0);
    }
#endif
    template <typename DT = void>
    void or_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::or_gen12 : Opcode::or_, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void or_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::or_gen12 : Opcode::or_, getDataType<DT>(), mod, dst, src0, src1);
    }
#ifndef NGEN_NO_OP_NAMES
    template <typename DT = void>
    void or(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        or_<DT>(mod, dst, src0, src1);
    }
    template <typename DT = void>
    void or(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        or_<DT>(mod, dst, src0, src1);
    }
#endif
    template <typename DT = void>
    void pln(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        if (hw >= HW::Gen11) unsupported();
        opX(Opcode::pln, getDataType<DT>(), mod, dst, src0, src1);
    }
    void ret(const InstructionModifier &mod, RegData src0) {
        src0.setRegion(2,2,1);
        if (isGen12)
            opBranch<true, true>(Opcode::ret, mod, null, src0);
        else
            opX<true>(Opcode::ret, DataType::ud, mod, null, src0);
    }
    template <typename DT = void>
    void rndd(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(Opcode::rndd, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rndd(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(Opcode::rndd, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rnde(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(Opcode::rnde, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rnde(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(Opcode::rnde, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rndu(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(Opcode::rndu, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rndu(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(Opcode::rndu, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rndz(const InstructionModifier &mod, const RegData &dst, const RegData &src0) {
        opX(Opcode::rndz, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rndz(const InstructionModifier &mod, const RegData &dst, const Immediate &src0) {
        opX(Opcode::rndz, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rol(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::rol_gen12 : Opcode::rol, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void rol(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::rol_gen12 : Opcode::rol, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void ror(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::ror_gen12 : Opcode::ror, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void ror(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::ror_gen12 : Opcode::ror, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sad2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        if (hw >= HW::Gen12LP) unsupported();
        opX(Opcode::sad2, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sad2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        if (hw >= HW::Gen12LP) unsupported();
        opX(Opcode::sad2, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sada2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        if (hw >= HW::Gen12LP) unsupported();
        opX(Opcode::sada2, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sada2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        if (hw >= HW::Gen12LP) unsupported();
        opX(Opcode::sada2, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::sel_gen12 : Opcode::sel, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::sel_gen12 : Opcode::sel, getDataType<DT>(), mod, dst, src0, src1);
    }

    /* Gen12-style sends */
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, uint32_t desc) {
        opSend(Opcode::send, mod, sf, dst, src0, src1, -1, exdesc, desc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, uint32_t desc) {
        opSend(Opcode::send, mod, sf, dst, src0, src1, -1, exdesc, desc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const GRFRange &src1, const RegData &exdesc, uint32_t desc) {
        opSend(Opcode::send, mod, sf, dst, src0, src1[0], src1.getLen(), exdesc, desc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, const RegData &desc) {
        opSend(Opcode::send, mod, sf, dst, src0, src1, -1, exdesc, desc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, const RegData &desc) {
        opSend(Opcode::send, mod, sf, dst, src0, src1, -1, exdesc, desc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const GRFRange &src1, const RegData &exdesc, const RegData &desc) {
        opSend(Opcode::send, mod, sf, dst, src0, src1[0], src1.getLen(), exdesc, desc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc) {
        opSend(Opcode::send, mod, sf, dst, src0, NullRegister(), 0, exdesc, desc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &exdesc, uint32_t desc) {
        opSend(Opcode::send, mod, sf, dst, src0, NullRegister(), 0, exdesc, desc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc) {
        opSend(Opcode::send, mod, sf, dst, src0, NullRegister(), 0, exdesc, desc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &exdesc, const RegData &desc) {
        opSend(Opcode::send, mod, sf, dst, src0, NullRegister(), 0, exdesc, desc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, uint32_t desc) {
        opSend(Opcode::sendc, mod, sf, dst, src0, src1, -1, exdesc, desc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, uint32_t desc) {
        opSend(Opcode::sendc, mod, sf, dst, src0, src1, -1, exdesc, desc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const GRFRange &src1, const RegData &exdesc, uint32_t desc) {
        opSend(Opcode::sendc, mod, sf, dst, src0, src1[0], src1.getLen(), exdesc, desc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, const RegData &desc) {
        opSend(Opcode::sendc, mod, sf, dst, src0, src1, -1, exdesc, desc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, const RegData &desc) {
        opSend(Opcode::sendc, mod, sf, dst, src0, src1, -1, exdesc, desc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const GRFRange &src1, const RegData &exdesc, const RegData &desc) {
        opSend(Opcode::sendc, mod, sf, dst, src0, src1[0], src1.getLen(), exdesc, desc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc) {
        opSend(Opcode::sendc, mod, sf, dst, src0, NullRegister(), 0, exdesc, desc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &exdesc, uint32_t desc) {
        opSend(Opcode::sendc, mod, sf, dst, src0, NullRegister(), 0, exdesc, desc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc) {
        opSend(Opcode::sendc, mod, sf, dst, src0, NullRegister(), 0, exdesc, desc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &exdesc, const RegData &desc) {
        opSend(Opcode::sendc, mod, sf, dst, src0, NullRegister(), 0, exdesc, desc);
    }
    /* Pre-Gen12-style sends; also supported on Gen12. */
    void send(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc) {
        opSend(Opcode::send, mod, dst, src0, exdesc, desc);
    }
    void send(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc) {
        opSend(Opcode::send, mod, dst, src0, exdesc, desc);
    }
    void sendc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc) {
        opSend(Opcode::sendc, mod, dst, src0, exdesc, desc);
    }
    void sendc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc) {
        opSend(Opcode::sendc, mod, dst, src0, exdesc, desc);
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, uint32_t desc) {
        opSends(Opcode::sends, mod, dst, src0, src1, exdesc, desc);
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, const RegData &desc) {
        opSends(Opcode::sends, mod, dst, src0, src1, exdesc, desc);
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, uint32_t desc) {
        opSends(Opcode::sends, mod, dst, src0, src1, exdesc, desc);
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, const RegData &desc) {
        opSends(Opcode::sends, mod, dst, src0, src1, exdesc, desc);
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, uint32_t desc) {
        opSends(Opcode::sendsc, mod, dst, src0, src1, exdesc, desc);
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, const RegData &desc) {
        opSends(Opcode::sendsc, mod, dst, src0, src1, exdesc, desc);
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, uint32_t desc) {
        opSends(Opcode::sendsc, mod, dst, src0, src1, exdesc, desc);
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, const RegData &desc) {
        opSends(Opcode::sendsc, mod, dst, src0, src1, exdesc, desc);
    }
#if XE3P
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, uint64_t desc) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, NullRegister(), NullRegister(), desc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const RegData &ind0, uint64_t desc) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, ind0, NullRegister(), desc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const RegData &ind0, const RegData &ind1, uint64_t desc) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, ind0, ind1, desc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const GRFRange &src1, uint64_t desc) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), NullRegister(), NullRegister(), desc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const GRFRange &src1, const RegData &ind0, uint64_t desc) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), ind0, NullRegister(), desc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const GRFRange &src1, const RegData &ind0, const RegData &ind1, uint64_t desc) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), ind0, ind1, desc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, uint64_t desc) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src0Len, NullRegister(), 0, NullRegister(), NullRegister(), desc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, const RegData &ind0, uint64_t desc) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src0Len, NullRegister(), 0, ind0, NullRegister(), desc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, const RegData &ind0, const RegData &ind1, uint64_t desc) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src0Len, NullRegister(), 0, ind0, ind1, desc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, uint64_t desc) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, NullRegister(), NullRegister(), desc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const RegData &ind0, uint64_t desc) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, ind0, NullRegister(), desc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const RegData &ind0, const RegData &ind1, uint64_t desc) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, ind0, ind1, desc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const GRFRange &src1, uint64_t desc) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), NullRegister(), NullRegister(), desc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const GRFRange &src1, const RegData &ind0, uint64_t desc) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), ind0, NullRegister(), desc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const GRFRange &src1, const RegData &ind0, const RegData &ind1, uint64_t desc) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), ind0, ind1, desc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, uint64_t desc) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src0Len, NullRegister(), 0, NullRegister(), NullRegister(), desc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, const RegData &ind0, uint64_t desc) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src0Len, NullRegister(), 0, ind0, NullRegister(), desc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, const RegData &ind0, const RegData &ind1, uint64_t desc) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src0Len, NullRegister(), 0, ind0, ind1, desc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, uint64_t desc) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, NullRegister(), NullRegister(), desc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const RegData &ind0, uint64_t desc) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, ind0, NullRegister(), desc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const RegData &ind0, const RegData &ind1, uint64_t desc) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src0.getLen(), NullRegister(), 0, ind0, ind1, desc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const GRFRange &src1, uint64_t desc) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), NullRegister(), NullRegister(), desc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const GRFRange &src1, const RegData &ind0, uint64_t desc) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), ind0, NullRegister(), desc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const GRFRange &src0, const GRFRange &src1, const RegData &ind0, const RegData &ind1, uint64_t desc) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src0.getLen(), src1, src1.getLen(), ind0, ind1, desc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, uint64_t desc) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src0Len, NullRegister(), 0, NullRegister(), NullRegister(), desc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, const RegData &ind0, uint64_t desc) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src0Len, NullRegister(), 0, ind0, NullRegister(), desc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, const RegData &ind0, const RegData &ind1, uint64_t desc) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src0Len, NullRegister(), 0, ind0, ind1, desc);
    }
#endif

    template <typename DT = void>
    void shl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::shl_gen12 : Opcode::shl, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void shl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::shl_gen12 : Opcode::shl, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void shr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::shr_gen12 : Opcode::shr, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void shr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::shr_gen12 : Opcode::shr, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void smov(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::smov_gen12 : Opcode::smov, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void srnd(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::srnd, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void srnd(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::srnd, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void subb(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(Opcode::subb, getDataType<DT>(), mod | AccWrEn, dst, src0, src1);
    }
    template <typename DT = void>
    void subb(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(Opcode::subb, getDataType<DT>(), mod | AccWrEn, dst, src0, src1);
    }
    void wait(const InstructionModifier &mod, const RegData &nreg) {
#ifdef NGEN_SAFE
        if (!nreg.isARF() || nreg.getARFType() != ARFType::n) throw invalid_arf_exception();
#endif
        opX(Opcode::wait, DataType::invalid, mod, nreg, nreg);
    }
    void while_(const InstructionModifier &mod, Label &jip) {
        opBranch(Opcode::while_, mod, null, jip);
    }
    template <typename DT = void>
    void xor_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        opX(isGen12 ? Opcode::xor_gen12 : Opcode::xor_, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void xor_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        opX(isGen12 ? Opcode::xor_gen12 : Opcode::xor_, getDataType<DT>(), mod, dst, src0, src1);
    }
#ifndef NGEN_NO_OP_NAMES
    template <typename DT = void>
    void xor(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
        xor_<DT>(mod, dst, src0, src1);
    }
    template <typename DT = void>
    void xor(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
        xor_<DT>(mod, dst, src0, src1);
    }
#endif

#if XE3P
private:
    struct Shfl {
        BinaryCodeGenerator<hw> &parent;

        Shfl(BinaryCodeGenerator<hw> *parent_) : parent(*parent_) {}

        void operator()(ShuffleFunction fc, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
            parent.opShfl(Opcode::shfl, fc, DataType::invalid, mod, dst, src0, src1);
        }

        template <typename DT = void>
        void idx4(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
            parent.opShfl(Opcode::shfl, ShuffleFunction::idx4, getDataType<DT>(), mod, dst, src0, src1);
        }
    };
public:
    Shfl shfl;
#endif

private:
    struct Sync {
        BinaryCodeGenerator<hw> &parent;

        Sync(BinaryCodeGenerator<hw> *parent_) : parent(*parent_) {}

        void operator()(SyncFunction fc, const InstructionModifier &mod = InstructionModifier()) {
            parent.opSync(Opcode::sync, fc, mod);
        }
        void operator()(SyncFunction fc, const RegData &src0) {
            this->operator()(fc, InstructionModifier(), src0);
        }
        void operator()(SyncFunction fc, const InstructionModifier &mod, const RegData &src0) {
            parent.opSync(Opcode::sync, fc, mod, src0);
        }
        void operator()(SyncFunction fc, int src0) {
            this->operator()(fc, InstructionModifier(), src0);
        }
        void operator()(SyncFunction fc, const InstructionModifier &mod, uint32_t src0) {
            parent.opSync(Opcode::sync, fc, mod, Immediate::ud(src0));
        }
        void allrd() {
            allrd(null.ud(0)(0, 1, 1));
        }
        void allrd(const InstructionModifier &mod) {
            allrd(mod, null.ud(0)(0, 1, 1));
        }
        void allrd(const RegData &src0) {
            allrd(InstructionModifier(), src0);
        }
        void allrd(const InstructionModifier &mod, const RegData &src0) {
            this->operator()(SyncFunction::allrd, mod, src0);
        }
        void allrd(uint32_t src0) {
            allrd(InstructionModifier(), src0);
        }
        void allrd(const InstructionModifier &mod, uint32_t src0) {
            this->operator()(SyncFunction::allrd, mod, src0);
        }
        void allwr() {
            allwr(null);
        }
        void allwr(const InstructionModifier &mod) {
            allwr(mod, null);
        }
        void allwr(const RegData &src0) {
            allwr(InstructionModifier(), src0);
        }
        void allwr(const InstructionModifier &mod, const RegData &src0) {
            this->operator()(SyncFunction::allwr, mod, src0);
        }
        void allwr(uint32_t src0) {
            allwr(InstructionModifier(), src0);
        }
        void allwr(const InstructionModifier &mod, uint32_t src0) {
            this->operator()(SyncFunction::allwr, mod, src0);
        }
        void bar(const InstructionModifier &mod = InstructionModifier()) {
            this->operator()(SyncFunction::bar, mod);
        }
        void bar(const InstructionModifier &mod, uint32_t src0) {
            this->operator()(SyncFunction::bar, mod, src0);
        }
        void bar(const InstructionModifier &mod, const RegData &src0) {
            this->operator()(SyncFunction::bar, mod, src0);
        }
        void bar(uint32_t src0) {
            this->operator()(SyncFunction::bar, InstructionModifier(), src0);
        }
        void bar(const RegData &src0) {
            this->operator()(SyncFunction::bar, InstructionModifier(), src0);
        }
        void flush() {
            flush(InstructionModifier());
        }
        void flush(const InstructionModifier &mod) {
            this->operator()(SyncFunction::flush, InstructionModifier(), null);
        }
        void host(const InstructionModifier &mod = InstructionModifier()) {
            this->operator()(SyncFunction::host, mod);
        }
        void nop(const InstructionModifier &mod = InstructionModifier()) {
            this->operator()(SyncFunction::nop, mod);
        }
    };
public:
    Sync sync;

    void ignoredep(Operand op) {
        if (hw >= HW::Gen12LP)
            opX(Opcode::directive, DataType::ud, InstructionModifier(), GRF(static_cast<int>(op)), NullRegister(), NullRegister());
    }
    void subdep(Operand op, const GRFRange &r) {
        if (op == Operand::dst && !r.isEmpty()) {
#ifdef NGEN_SAFE
            if (r.getLen() > 32) throw invalid_directive_exception();
#endif
            opX(Opcode::directive, DataType::ud, InstructionModifier::createAutoSWSB(), GRF(static_cast<int>(Directive::subdep_dst)), r[0], r[r.getLen() - 1]);
        } else {
            ignoredep(op);
            wrdep(r);
        }
    }
    void subdep(Operand op, const GRF &r) {
        subdep(op, r-r);
    }
    void wrdep(const GRFRange &r) {
#ifdef NGEN_SAFE
        if (hw < HW::Gen12LP) throw unsupported_instruction();
#endif
        int len = r.getLen();
        for (int o = 0; o < len; o += 32) {
            int thisLen = std::min(len - o, 32);
            opX(Opcode::directive, DataType::ud, InstructionModifier::createAutoSWSB(), GRF(static_cast<int>(Directive::wrdep)), r[o], r[o + thisLen - 1]);
        }
    }
    void wrdep(const GRF &r) {
        wrdep(r-r);
    }
    void fencedep(Label &fenceLocation) {
        addFixup(LabelFixup(fenceLocation.getID(labelManager), LabelFixup::JIPOffset));
        opX(Opcode::directive, DataType::ud, InstructionModifier::createAutoSWSB(), GRF(static_cast<int>(Directive::fencedep)), Immediate::ud(0));
    }

    using _self = BinaryCodeGenerator<hw>;
#include "ngen_pseudo.hpp"
};

#define NGEN_FORWARD(hw) \
NGEN_FORWARD_NO_ELF_OVERRIDES(hw) \
NGEN_FORWARD_EXTRA_ELF_OVERRIDES(hw) \
void requireGRF(int grfs) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::requireGRF(grfs); }

#define NGEN_FORWARD_NO_ELF_OVERRIDES(hw) \
using InstructionStream = typename NGEN_NAMESPACE::BinaryCodeGenerator<hw>::InstructionStream; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::isGen12; \
template <typename DT = void, typename... Targs> void add(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template add<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void add3(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template add3<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void addc(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template addc<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void and_(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template and_<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void asr(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template asr<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void avg(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template avg<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void bfe(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template bfe<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void bfi1(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template bfi1<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void bfi2(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template bfi2<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void bfn(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template bfn<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void bfrev(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template bfrev<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void cbit(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template cbit<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void cmp(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template cmp<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void cmpn(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template cmpn<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void csel(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template csel<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void dp2(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template dp2<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void dp3(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template dp3<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void dp4(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template dp4<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void dp4a(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template dp4a<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void dpas(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template dpas<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void dpasw(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template dpasw<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void dph(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template dph<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void fbh(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template fbh<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void fbl(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template fbl<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void frc(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template frc<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void line(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template line<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void lrp(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template lrp<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void lzd(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template lzd<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void mac(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template mac<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void macl(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template macl<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void mach(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template mach<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void mad(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template mad<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void madm(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template madm<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void math(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template math<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void mov(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template mov<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void movi(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template movi<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void mul(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template mul<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void not_(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template not_<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void or_(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template or_<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void pln(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template pln<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void rndd(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template rndd<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void rnde(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template rnde<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void rndu(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template rndu<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void rndz(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template rndz<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void rol(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template rol<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void ror(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template ror<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void sad2(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template sad2<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void sada2(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template sada2<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void sel(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template sel<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void shl(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template shl<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void shr(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template shr<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void smov(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template smov<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void subb(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template subb<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void xor_(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template xor_<DT>(std::forward<Targs>(args)...); } \
template <typename... Targs> void brc(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::brc(std::forward<Targs>(args)...); } \
template <typename... Targs> void brd(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::brd(std::forward<Targs>(args)...); } \
template <typename... Targs> void break_(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::break_(std::forward<Targs>(args)...); } \
template <typename... Targs> void call(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::call(std::forward<Targs>(args)...); } \
template <typename... Targs> void calla(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::calla(std::forward<Targs>(args)...); } \
template <typename... Targs> void cont(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::cont(std::forward<Targs>(args)...); } \
template <typename... Targs> void else_(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::else_(std::forward<Targs>(args)...); } \
template <typename... Targs> void endif(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::endif(std::forward<Targs>(args)...); } \
template <typename... Targs> void goto_(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::goto_(std::forward<Targs>(args)...); } \
template <typename... Targs> void halt(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::halt(std::forward<Targs>(args)...); } \
template <typename... Targs> void if_(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::if_(std::forward<Targs>(args)...); } \
template <typename... Targs> void illegal(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::illegal(std::forward<Targs>(args)...); } \
template <typename... Targs> void join(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::join(std::forward<Targs>(args)...); } \
template <typename... Targs> void jmpi(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::jmpi(std::forward<Targs>(args)...); } \
template <typename... Targs> void nop(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::nop(std::forward<Targs>(args)...); } \
template <typename... Targs> void ret(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::ret(std::forward<Targs>(args)...); } \
template <typename... Targs> void send(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::send(std::forward<Targs>(args)...); } \
template <typename... Targs> void sendc(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sendc(std::forward<Targs>(args)...); } \
template <typename... Targs> void sends(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sends(std::forward<Targs>(args)...); } \
template <typename... Targs> void sendsc(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sendsc(std::forward<Targs>(args)...); } \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sync; \
template <typename... Targs> void wait(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::wait(std::forward<Targs>(args)...); } \
template <typename... Targs> void while_(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::while_(std::forward<Targs>(args)...); } \
template <typename... Targs> void ignoredep(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::ignoredep(std::forward<Targs>(args)...); } \
template <typename... Targs> void subdep(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::subdep(std::forward<Targs>(args)...); } \
template <typename... Targs> void wrdep(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::wrdep(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void min_(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template min_<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void max_(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template max_<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void bfi(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template bfi<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void cos(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template cos<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void exp(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template exp<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void fdiv(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template fdiv<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void idiv(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template idiv<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void inv(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template inv<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void invm(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template invm<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void iqot(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template iqot<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void irem(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template irem<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void log(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template log<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void pow(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template pow<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void rsqt(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template rsqt<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void rsqtm(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template rsqtm<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void sin(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template sin<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void sqt(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template sqt<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void fdiv_ieee(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template fdiv_ieee<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void inv_ieee(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template inv_ieee<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void sqt_ieee(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template sqt_ieee<DT>(std::forward<Targs>(args)...); } \
template <typename... Targs> void threadend(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::threadend(std::forward<Targs>(args)...); } \
template <typename... Targs> void barrierheader(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::barrierheader(std::forward<Targs>(args)...); } \
template <typename... Targs> void barriermsg(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::barriermsg(std::forward<Targs>(args)...); } \
template <typename... Targs> void barriersignal(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::barriersignal(std::forward<Targs>(args)...); } \
template <typename... Targs> void barrierwait(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::barrierwait(std::forward<Targs>(args)...); } \
template <typename... Targs> void barrier(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::barrier(std::forward<Targs>(args)...); } \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::load; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::store; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::atomic; \
template <typename... Targs> void memfence(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::memfence(std::forward<Targs>(args)...); } \
template <typename... Targs> void slmfence(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::slmfence(std::forward<Targs>(args)...); } \
template <typename... Targs> void fencewait(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::fencewait(std::forward<Targs>(args)...); } \
template <typename... Targs> void loadlid(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::loadlid(std::forward<Targs>(args)...); } \
template <typename... Targs> void loadargs(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::loadargs(std::forward<Targs>(args)...); } \
template <typename... Targs> void epilogue(int GRFCount, bool hasSLM, const NGEN_NAMESPACE::RegData &r0_info) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::epilogue(GRFCount, hasSLM, r0_info); } \
template <typename... Targs> void pushStream(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::pushStream(std::forward<Targs>(args)...); } \
template <typename... Targs> InstructionStream *popStream(Targs&&... args) { return NGEN_NAMESPACE::BinaryCodeGenerator<hw>::popStream(std::forward<Targs>(args)...); } \
template <typename... Targs> void appendStream(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::appendStream(std::forward<Targs>(args)...); } \
template <typename... Targs> void appendCurrentStream(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::appendCurrentStream(std::forward<Targs>(args)...); } \
template <typename... Targs> void discardStream(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::discardStream(std::forward<Targs>(args)...); } \
template <typename... Targs> void mark(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::mark(std::forward<Targs>(args)...); } \
template <typename... Targs> void comment(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::comment(std::forward<Targs>(args)...); } \
template <typename... Targs> void setDefaultNoMask(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::setDefaultNoMask(std::forward<Targs>(args)...); } \
template <typename... Targs> void setDefaultAutoSWSB(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::setDefaultAutoSWSB(std::forward<Targs>(args)...); } \
bool getDefaultNoMask() { return NGEN_NAMESPACE::BinaryCodeGenerator<hw>::getDefaultNoMask(); } \
bool getDefaultAutoSWSB() { return NGEN_NAMESPACE::BinaryCodeGenerator<hw>::getDefaultAutoSWSB(); } \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::product; \
NGEN_NAMESPACE::Product getProduct() { return NGEN_NAMESPACE::BinaryCodeGenerator<hw>::getProduct(); } \
NGEN_NAMESPACE::ProductFamily getProductFamily() { return NGEN_NAMESPACE::BinaryCodeGenerator<hw>::getProductFamily(); } \
int getStepping() { return NGEN_NAMESPACE::BinaryCodeGenerator<hw>::getStepping(); } \
void setProduct(NGEN_NAMESPACE::Product product_) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::setProduct(product_); } \
void setProductFamily(NGEN_NAMESPACE::ProductFamily family_) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::setProductFamily(family_); } \
void setStepping(int stepping_) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::setStepping(stepping_); } \
NGEN_FORWARD_EXTRA(hw) \
NGEN_FORWARD_OP_NAMES(hw) \
NGEN_FORWARD_MIN_MAX(hw) \
NGEN_FORWARD_REGISTERS(hw)

#if !XE3P
#define NGEN_FORWARD_EXTRA(hw)
#define NGEN_FORWARD_EXTRA_ELF_OVERRIDES(hw)
#else
#define NGEN_FORWARD_EXTRA(hw) \
template <typename... Targs> void sendg(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sendg(std::forward<Targs>(args)...); } \
template <typename... Targs> void sendgc(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sendgc(std::forward<Targs>(args)...); } \
template <typename... Targs> void sendgx(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sendgx(std::forward<Targs>(args)...); } \
bool getEfficient64Bit() { return NGEN_NAMESPACE::BinaryCodeGenerator<hw>::getEfficient64Bit(); }

#define NGEN_FORWARD_EXTRA_ELF_OVERRIDES(hw) \
template <typename... Targs> void setEfficient64Bit(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::setEfficient64Bit(std::forward<Targs>(args)...); }
#endif

#ifdef NGEN_NO_OP_NAMES
#define NGEN_FORWARD_OP_NAMES(hw)
#else
#define NGEN_FORWARD_OP_NAMES(hw) \
template <typename DT = void, typename... Targs> void and(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template and_<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void not(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template not_<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void or(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template or_<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void xor(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template xor_<DT>(std::forward<Targs>(args)...); }
#endif

#ifdef NGEN_WINDOWS_COMPAT
#define NGEN_FORWARD_MIN_MAX(hw)
#else
#define NGEN_FORWARD_MIN_MAX(hw) \
template <typename DT = void, typename... Targs> void min(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template min<DT>(std::forward<Targs>(args)...); } \
template <typename DT = void, typename... Targs> void max(Targs&&... args) { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template max<DT>(std::forward<Targs>(args)...); }
#endif

#ifdef NGEN_GLOBAL_REGS
#define NGEN_FORWARD_REGISTERS(hw)
#else
#define NGEN_FORWARD_REGISTERS_BASE(hw) \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::indirect; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r0; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r1; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r2; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r3; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r4; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r5; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r6; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r7; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r8; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r9; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r10; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r11; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r12; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r13; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r14; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r15; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r16; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r17; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r18; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r19; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r20; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r21; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r22; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r23; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r24; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r25; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r26; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r27; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r28; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r29; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r30; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r31; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r32; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r33; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r34; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r35; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r36; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r37; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r38; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r39; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r40; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r41; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r42; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r43; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r44; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r45; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r46; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r47; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r48; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r49; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r50; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r51; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r52; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r53; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r54; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r55; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r56; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r57; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r58; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r59; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r60; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r61; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r62; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r63; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r64; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r65; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r66; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r67; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r68; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r69; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r70; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r71; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r72; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r73; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r74; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r75; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r76; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r77; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r78; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r79; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r80; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r81; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r82; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r83; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r84; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r85; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r86; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r87; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r88; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r89; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r90; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r91; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r92; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r93; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r94; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r95; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r96; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r97; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r98; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r99; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r100; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r101; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r102; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r103; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r104; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r105; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r106; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r107; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r108; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r109; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r110; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r111; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r112; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r113; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r114; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r115; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r116; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r117; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r118; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r119; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r120; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r121; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r122; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r123; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r124; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r125; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r126; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r127; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r128; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r129; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r130; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r131; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r132; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r133; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r134; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r135; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r136; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r137; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r138; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r139; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r140; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r141; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r142; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r143; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r144; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r145; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r146; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r147; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r148; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r149; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r150; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r151; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r152; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r153; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r154; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r155; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r156; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r157; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r158; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r159; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r160; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r161; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r162; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r163; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r164; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r165; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r166; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r167; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r168; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r169; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r170; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r171; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r172; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r173; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r174; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r175; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r176; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r177; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r178; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r179; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r180; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r181; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r182; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r183; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r184; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r185; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r186; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r187; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r188; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r189; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r190; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r191; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r192; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r193; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r194; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r195; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r196; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r197; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r198; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r199; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r200; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r201; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r202; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r203; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r204; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r205; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r206; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r207; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r208; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r209; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r210; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r211; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r212; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r213; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r214; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r215; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r216; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r217; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r218; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r219; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r220; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r221; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r222; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r223; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r224; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r225; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r226; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r227; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r228; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r229; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r230; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r231; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r232; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r233; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r234; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r235; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r236; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r237; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r238; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r239; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r240; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r241; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r242; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r243; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r244; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r245; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r246; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r247; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r248; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r249; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r250; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r251; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r252; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r253; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r254; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r255; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::null; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::a0; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::acc0; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::acc1; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::acc2; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::acc3; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::acc4; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::acc5; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::acc6; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::acc7; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::acc8; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::acc9; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::mme0; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::mme1; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::mme2; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::mme3; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::mme4; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::mme5; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::mme6; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::mme7; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::noacc; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::nomme; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::f0; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::f1; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::f2; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::f3; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::f0_0; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::f0_1; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::f1_0; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::f1_1; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::ce0; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sp; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sr0; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sr1; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::cr0; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::n0; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::ip; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::tdr0; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::tm0; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::tm1; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::tm2; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::tm3; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::tm4; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::pm0; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::tp0; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::dbg0; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::fc0; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::fc1; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::fc2; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::fc3; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::NoDDClr; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::NoDDChk; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::AccWrEn; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::NoSrcDepSet; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::Breakpoint; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sat; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::NoMask; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::ExBSO; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::Serialize; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::EOT; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::Atomic; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::Switch; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::NoPreempt; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::anyv; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::allv; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::any2h; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::all2h; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::any4h; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::all4h; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::any8h; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::all8h; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::any16h; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::all16h; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::any32h; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::all32h; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::any; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::all; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::x_repl; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::y_repl; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::z_repl; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::w_repl; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::ze; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::eq; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::nz; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::ne; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::gt; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::ge; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::lt; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::le; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::ov; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::un; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::eo; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::M0; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::M4; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::M8; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::M12; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::M16; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::M20; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::M24; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::M28; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb0; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb1; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb2; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb3; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb4; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb5; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb6; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb7; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb8; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb9; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb10; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb11; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb12; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb13; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb14; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb15; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb16; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb17; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb18; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb19; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb20; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb21; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb22; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb23; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb24; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb25; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb26; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb27; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb28; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb29; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb30; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::sb31; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::NoAccSBSet; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::A32; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::A32NC; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::A64; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::A64NC; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::SLM; \
template <typename... Targs> NGEN_NAMESPACE::InstructionModifier ExecutionOffset(Targs&&... args) { return NGEN_NAMESPACE::BinaryCodeGenerator<hw>::ExecutionOffset(std::forward<Targs>(args)...); } \
template <typename... Targs> NGEN_NAMESPACE::AddressBase Surface(Targs&&... args) { return NGEN_NAMESPACE::BinaryCodeGenerator<hw>::Surface(std::forward<Targs>(args)...); } \
template <typename... Targs> NGEN_NAMESPACE::AddressBase CC(Targs&&... args) { return NGEN_NAMESPACE::BinaryCodeGenerator<hw>::CC(std::forward<Targs>(args)...); } \
template <typename... Targs> NGEN_NAMESPACE::AddressBase SC(Targs&&... args) { return NGEN_NAMESPACE::BinaryCodeGenerator<hw>::SC(std::forward<Targs>(args)...); } \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::D8; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::D16; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::D32; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::D64; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::D8U32; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::D16U32; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::D8T; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::D16T; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::D32T; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::D64T; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::D8U32T; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::D16U32T; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::V1; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::V2; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::V3; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::V4; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::V8; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::V16; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::V32; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::V64; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::V1T; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::V2T; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::V3T; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::V4T; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::V8T; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::V16T; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::V32T; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::V64T; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::transpose; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::vnni; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1UC_L3UC; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1UC_L3C; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1C_L3UC; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1C_L3C; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1S_L3UC; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1S_L3C; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1IAR_L3C; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1UC_L3WB; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1WT_L3UC; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1WT_L3WB; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1S_L3WB; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1WB_L3WB; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1C_L3CC; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1UC_L3CC;
#define NGEN_FORWARD_REGISTERS_EXTRA1(hw) \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::s0;
#define NGEN_FORWARD_REGISTERS_EXTRA2(hw) 
#if !XE3P
#define NGEN_FORWARD_REGISTERS_EXTRA3(hw)
#else
#define NGEN_FORWARD_REGISTERS_EXTRA3(hw) \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::Fwd; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r256; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r257; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r258; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r259; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r260; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r261; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r262; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r263; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r264; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r265; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r266; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r267; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r268; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r269; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r270; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r271; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r272; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r273; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r274; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r275; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r276; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r277; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r278; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r279; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r280; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r281; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r282; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r283; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r284; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r285; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r286; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r287; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r288; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r289; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r290; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r291; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r292; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r293; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r294; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r295; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r296; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r297; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r298; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r299; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r300; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r301; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r302; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r303; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r304; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r305; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r306; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r307; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r308; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r309; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r310; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r311; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r312; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r313; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r314; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r315; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r316; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r317; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r318; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r319; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r320; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r321; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r322; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r323; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r324; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r325; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r326; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r327; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r328; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r329; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r330; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r331; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r332; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r333; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r334; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r335; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r336; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r337; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r338; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r339; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r340; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r341; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r342; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r343; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r344; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r345; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r346; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r347; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r348; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r349; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r350; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r351; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r352; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r353; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r354; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r355; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r356; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r357; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r358; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r359; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r360; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r361; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r362; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r363; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r364; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r365; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r366; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r367; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r368; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r369; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r370; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r371; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r372; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r373; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r374; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r375; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r376; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r377; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r378; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r379; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r380; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r381; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r382; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r383; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r384; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r385; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r386; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r387; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r388; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r389; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r390; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r391; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r392; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r393; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r394; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r395; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r396; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r397; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r398; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r399; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r400; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r401; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r402; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r403; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r404; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r405; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r406; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r407; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r408; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r409; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r410; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r411; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r412; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r413; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r414; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r415; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r416; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r417; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r418; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r419; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r420; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r421; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r422; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r423; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r424; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r425; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r426; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r427; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r428; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r429; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r430; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r431; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r432; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r433; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r434; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r435; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r436; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r437; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r438; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r439; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r440; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r441; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r442; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r443; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r444; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r445; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r446; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r447; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r448; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r449; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r450; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r451; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r452; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r453; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r454; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r455; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r456; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r457; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r458; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r459; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r460; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r461; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r462; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r463; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r464; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r465; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r466; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r467; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r468; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r469; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r470; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r471; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r472; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r473; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r474; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r475; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r476; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r477; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r478; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r479; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r480; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r481; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r482; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r483; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r484; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r485; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r486; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r487; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r488; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r489; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r490; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r491; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r492; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r493; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r494; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r495; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r496; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r497; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r498; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r499; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r500; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r501; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r502; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r503; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r504; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r505; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r506; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r507; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r508; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r509; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r510; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::r511; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::A64_A32U; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::A64_A32S; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::Overfetch; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1UC_L2UC_L3UC;    using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1UC_L2UC_L3C;  using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1UC_L2C_L3UC; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1UC_L2C_L3C;      using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1C_L2UC_L3UC;  using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1C_L2UC_L3C; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1C_L2C_L3UC;      using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1C_L2C_L3C;    using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1S_L2UC_L3UC; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1S_L2UC_L3C;      using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1S_L2C_L3UC;   using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1S_L2C_L3C; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1IAR_L2IAR_L3IAR; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1UC_L2UC_L3WB; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1UC_L2WB_L3UC; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1WT_L2UC_L3UC;    using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1WT_L2UC_L3WB; using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1WT_L2WB_L3UC; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1S_L2UC_L3WB;     using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1S_L2WB_L3UC;  using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1S_L2WB_L3WB; \
using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1WB_L2WB_L3UC;    using NGEN_NAMESPACE::BinaryCodeGenerator<hw>::L1WB_L2UC_L3WB;
#endif
#define NGEN_FORWARD_REGISTERS(hw) NGEN_FORWARD_REGISTERS_BASE(hw) NGEN_FORWARD_REGISTERS_EXTRA1(hw) NGEN_FORWARD_REGISTERS_EXTRA2(hw) NGEN_FORWARD_REGISTERS_EXTRA3(hw)
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
                                      :   SWSBInfo12(si.swsb, Opcode::sync).raw();
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

        for (auto &bb : analysis)
            for (auto &sync : bb.syncs)
                syncs.insert(std::make_pair(sync.inum, &sync));

        result.resize(rootStream.length() + syncs.size() * sizeof(Instruction12));

        auto *psrc = reinterpret_cast<const Instruction12 *>(rootStream.code.data());
        auto *pdst = reinterpret_cast<Instruction12 *>(result.data());
        auto nextSync = syncs.begin();

        for (uint32_t isrc = 0; isrc < program.size(); isrc++, psrc++) {
            if (psrc->opcode() == Opcode::directive)
                continue;
            while ((nextSync != syncs.end()) && (nextSync->second->inum == isrc))
                *pdst++ = encodeSyncInsertion<hw>(*(nextSync++)->second);
            *pdst++ = *psrc;
        }

        result.resize(reinterpret_cast<uint8_t *>(pdst) - result.data());
    }

    return result;
}

template <HW hw>
template <bool forceWE, typename D, typename S0, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0)
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

    i.binary.dstRegFile = getRegFile(dst);
    i.binary.src0RegFile = getRegFile(src0);

    db(i);
}

template <HW hw>
template <bool forceWE, typename D, typename S0, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0)
{
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
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, const Immediate &src0)
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
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, const Immediate &src0)
{
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
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1)
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
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1)
{
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
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, const Immediate &src1)
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

    i.binary.dstRegFile = getRegFile(dst);
    i.binary.src0RegFile = getRegFile(src0);
    i.binary.src1RegFile = getRegFile(src1);

    i.imm32.value = static_cast<uint64_t>(src1);

    db(i);
}

template <HW hw>
template <bool forceWE, typename D, typename S0, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, const Immediate &src1)
{
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
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, RegData dst, RegData src0, RegData src1, RegData src2)
{
    opX(op, defaultType, mod, emulateAlign16Dst(dst),  emulateAlign16Src(src0),
                              emulateAlign16Src(src1), emulateAlign16Src(src2));
}


template <HW hw>
template <HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, Align16Operand dst, Align16Operand src0, Align16Operand src1, Align16Operand src2)
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

    db(i);
}

template <HW hw>
template <typename D, typename S0, typename S1, typename S2, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2)
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

    db(i);
}

template <HW hw>
template <typename D, typename S0,typename S1, typename S2, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2)
{
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
void BinaryCodeGenerator<hw>::opMath(Opcode op, DataType defaultType, const InstructionModifier &mod, MathFunction fc, DS0 dst, DS0 src0)
{
    InstructionModifier mmod = mod;

    mmod.setCMod(static_cast<ConditionModifier>(fc));
    opX(op, defaultType, mmod, dst, src0);
}

template <HW hw>
template <typename DS0, typename S1>
void BinaryCodeGenerator<hw>::opMath(Opcode op, DataType defaultType, const InstructionModifier &mod, MathFunction fc, DS0 dst, DS0 src0, S1 src1)
{
    InstructionModifier mmod = mod;

    mmod.setCMod(static_cast<ConditionModifier>(fc));
    opX(op, defaultType, mmod, dst, src0, src1);
}

template <HW hw>
template <typename D, typename S0, typename S2>
void BinaryCodeGenerator<hw>::opBfn(Opcode op, DataType defaultType, const InstructionModifier &mod, int bfnCtrl, D dst, S0 src0, RegData src1, S2 src2)
{
    if (hw < HW::XeHP)
        unsupported();

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

    db(i);
}

template <HW hw>
void BinaryCodeGenerator<hw>::opDpas(Opcode op, DataType defaultType, const InstructionModifier &mod, int sdepth, int rcount, RegData dst, RegData src0, RegData src1, RegData src2)
{
    if (hw < HW::XeHP)
        unsupported();

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;

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

    i.ternary.cmod = static_cast<int>(mod.getCMod());

    db(i);
}

template <HW hw>
template <typename D, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSend(Opcode op, const InstructionModifier &mod, SharedFunction sfid, const RegData &dst, const RegData &src0, const RegData &src1, int src1Length, uint32_t exdesc, D desc)
{
    exdesc |= uint32_t(static_cast<uint8_t>(sfid));
    opSends(static_cast<Opcode>(static_cast<uint8_t>(op) | 2), mod, dst, src0, src1, exdesc, desc);
}

template <HW hw>
template <typename D, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSend(Opcode op, const InstructionModifier &mod, SharedFunction sfid, const RegData &dst, const RegData &src0, const RegData &src1, int src1Length, const RegData &exdesc, D desc)
{
    opSends(static_cast<Opcode>(static_cast<uint8_t>(op) | 2), mod, dst, src0, src1, exdesc, desc);
}

template <HW hw>
template <typename ED, typename D, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSend(Opcode op, const InstructionModifier &mod, SharedFunction sfid, const RegData &dst, const RegData &src0_, const RegData &src1, int src1Length, ED exdesc, D desc)
{
    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;

    auto src0 = src0_;
    bool src0Indirect = (hw >= HW::Xe3 && src0.isIndirect());
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
BinaryCodeGenerator<hw>::opSend(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc)
{
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
BinaryCodeGenerator<hw>::opSend(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc)
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
BinaryCodeGenerator<hw>::opSend(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, D desc)
{
    opSends(op, mod, dst, src0, null, exdesc, desc);
}

template <HW hw>
template <typename ED, typename D, HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSends(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, ED exdesc, D desc)
{
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
BinaryCodeGenerator<hw>::opSends(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, RegData exdesc, D desc)
{
#ifdef NGEN_SAFE
    throw sfid_needed_exception();
#endif
}

template <HW hw>
template <typename D, HW hw_>
typename std::enable_if<hwGE(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opSends(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, D desc)
{
    Opcode mop = static_cast<Opcode>(static_cast<int>(op) & ~2);
    opSend(mop, mod, static_cast<SharedFunction>(exdesc & 0x1F), dst, src0, src1, -1, exdesc, desc);
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
                                      const RegData &ind0, const RegData &ind1, uint64_t desc)
{
    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;

    bool src0Indirect = src0.isIndirect();
    if (src0Indirect)
        src0 = src0.getIndirectReg();

    encodeCommon12(i, op, emod, dst, tag);

    i.sendg.eot = emod.isEOT();

    if (op == Opcode::sendgx) {
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
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, int32_t uip)
{
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
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip, int32_t uip)
{
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
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip)
{
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
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, int32_t jip)
{
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
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0)
{
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
BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0)
{
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
void BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, Label &jip, Label &uip)
{
    addFixup(LabelFixup(jip.getID(labelManager), LabelFixup::JIPOffset));
    addFixup(LabelFixup(uip.getID(labelManager), LabelFixup::UIPOffset));
    opBranch(op, mod, dst, 0, 0);
}

template <HW hw>
template <bool forceWE>
void BinaryCodeGenerator<hw>::opBranch(Opcode op, const InstructionModifier &mod, const RegData &dst, Label &jip)
{
    addFixup(LabelFixup(jip.getID(labelManager), LabelFixup::JIPOffset));
    opBranch<forceWE>(op, mod, dst, 0);
}

template <HW hw>
void BinaryCodeGenerator<hw>::opCall(Opcode op, const InstructionModifier &mod, const RegData &dst, Label &jip)
{
    addFixup(LabelFixup(jip.getID(labelManager), LabelFixup::JIPOffset));
    if (isGen12)
        opBranch<true>(op, mod, dst, 0);
    else
        opX<true>(op, DataType::d, mod, dst, null.ud(0)(0, 1, 0), Immediate::d(0));
}

template <HW hw>
template <HW hw_>
typename std::enable_if<hwLT(hw_, HW::Gen12LP)>::type
BinaryCodeGenerator<hw>::opJmpi(Opcode op, const InstructionModifier &mod, const RegData &dst, RegData src0, uint32_t jip)
{
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
BinaryCodeGenerator<hw>::opJmpi(Opcode op, const InstructionModifier &mod, const RegData &dst, RegData src0, uint32_t jip)
{
    opBranch<true>(op, mod, dst, jip);
}

template <HW hw>
void BinaryCodeGenerator<hw>::opJmpi(Opcode op, const InstructionModifier &mod, const RegData &dst, const RegData &src0, Label &jip)
{
    if (hw >= HW::Gen12LP)
        addFixup(LabelFixup(jip.getID(labelManager), LabelFixup::JIPOffset));
    opJmpi(op, mod, dst, src0, 0);
    if (hw < HW::Gen12LP)
        addFixup(LabelFixup(jip.getID(labelManager), LabelFixup::JIPOffsetJMPI));
}

#if XE3P
template <HW hw>
void BinaryCodeGenerator<hw>::opShfl(Opcode op, ShuffleFunction fc, DataType defaultType, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1)
{
    InstructionModifier mmod = mod;

    mmod.setCMod(static_cast<ConditionModifier>(fc));
    opX(op, defaultType, mmod, dst, src0, src1);
}
#endif

template <HW hw>
void BinaryCodeGenerator<hw>::opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod)
{
    if (hw < HW::Gen12LP)
        unsupported();

    typename EncodingTag12Dispatch<hw>::tag tag;
    Instruction12 i{};
    InstructionModifier emod = mod | defaultModifier;

    encodeCommon12(i, op, emod, null, tag);

    i.binary.dst = 0x1;
    i.binary.cmod = static_cast<int>(fc);

    db(i);
}

template <HW hw>
void BinaryCodeGenerator<hw>::opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod, RegData src0)
{
    typename EncodingTag12Dispatch<hw>::tag tag;
    if (hw < HW::Gen12LP)
        unsupported();

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
void BinaryCodeGenerator<hw>::opSync(Opcode op, SyncFunction fc, const InstructionModifier &mod, const Immediate &src0)
{
    if (hw < HW::Gen12LP)
        unsupported();

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
void BinaryCodeGenerator<hw>::opNop(Opcode op)
{
    Instruction8 i{};

    i.qword[0] = static_cast<int>(op);
    i.qword[1] = 0;

    db(i);
}

} /* namespace NGEN_NAMESPACE */

#ifdef ENABLE_LLVM_WCONVERSION
#pragma clang diagnostic pop
#endif

#endif /* header guard */
