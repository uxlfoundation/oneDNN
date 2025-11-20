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

#ifndef NGEN_ASM_HPP
#define NGEN_ASM_HPP

#include "ngen_config_internal.hpp"

#ifdef NGEN_ASM

#include <array>
#include <cstdint>
#include <sstream>
#include <string>

#include "ngen_core.hpp"
#include "ngen_debuginfo.hpp"
#include "ngen_gen12.hpp"

namespace NGEN_NAMESPACE {


inline void RegData::outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const
{
#ifdef NGEN_SAFE
    if (isInvalid()) throw invalid_object_exception();
#endif
#if XE4
    const bool xe4 = (detail <= PrintDetail::xe4);
#endif

    auto vs = getVS();
    if (detail == PrintDetail::vs_hs)
        if (vs > 8 && (getHS() != 0))
            vs = 8;

    if (getNeg()) str << '-';
    if (getAbs()) str << "(abs)";

    if (isARF()) {
#if XE4
        if (xe4 && getARFType() == ARFType::f)
            str << 'p';
        else
#endif
        str << getARFType();
        switch (getARFType()) {
            case ARFType::null:
            case ARFType::sp:
            case ARFType::ip:
#if XE4
            case ARFType::lid:
#endif
                break;
            default:
                str << getARFBase();
        }
    } else {
#if XE4
        char file = isSRF() ? 's' : 'r';
#else
        char file = 'r';
#endif
        if (isIndirect()) {
            str << file << '[';
#if XE4
            if (xe4) {
                getIndirectRegXe4().outputText(str, PrintDetail::xe4, man);
                if (getOffset())
                    str << " + " << getOffset();
            } else
#endif
            {
                getIndirectReg().outputText(str, PrintDetail::sub_no_type, man);
                if (getOffset())
                    str << ',' << getOffset();
            }
            str << ']';
        } else {
#if XE4
            if (xe4 && getBytes() == 8)
                str << '(' << file << base << ',' << file << (base + 1) << ')';
            else
                str << file << base;
#else
            str << file << base;
#endif
        }
    }

#if XE4
    if (detail == PrintDetail::xe4_type)
        str << '.' << DataTypeXe4{getType()};
    if (detail == PrintDetail::xe4_dst && isLUOrC())
        str << ".c";
    else if (detail <= PrintDetail::xe4 && isLUOrC())
        str << ".lu";
#endif

    if (detail <= PrintDetail::base) return;

    if (!isIndirect() && !isNull())
        str << '.' << getOffset();

    if (detail <= PrintDetail::sub_no_type) return;

    if (detail >= PrintDetail::hs && !isNull()) {
        str << '<';
        if (detail >= PrintDetail::vs_hs && !isVxIndirect())
            str << vs << ';';
        if (detail == PrintDetail::full)
            str << getWidth() << ',';
        str << getHS();
        str << '>';
    }

    str << ':' << getType();
}

static inline std::ostream& operator<<(std::ostream &str, const RegData &r)
{
    LabelManager man;
    r.outputText(str, PrintDetail::full, man);
    return str;
}

inline void Immediate::outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const
{
#if XE4
    if (detail == PrintDetail::xe4_hide) return;
#endif

    uint64_t nbytes = getBytes(getType());
    uint64_t val;

    if (nbytes == 8)
        val = payload;
    else
        val = payload & ((uint64_t(1) << (nbytes * 8)) - 1);

    str << "0x" << std::hex << val << std::dec;
    if (!hiddenType && detail >= PrintDetail::sub)
        str << ':' << type;
}

inline void ExtendedReg::outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const
{
#ifdef NGEN_SAFE
    if (isInvalid()) throw invalid_object_exception();
#endif

#if XE4
    base.outputText(str, (detail <= PrintDetail::xe4) ? PrintDetail::xe4 : PrintDetail::base, man);
    if (detail == PrintDetail::xe4_dst) return;
#else
    base.outputText(str, PrintDetail::base, man);
#endif
    str << '.';
    if (mmeNum == 8)
        str << "nomme";
    else
        str << "mme" << int(mmeNum);

    if (detail >= PrintDetail::sub)
        str << ':' << base.getType();
}

inline void Align16Operand::outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const
{
#ifdef NGEN_SAFE
    if (isInvalid()) throw invalid_object_exception();
    throw iga_align16_exception();
#else
    str << "<unsupported Align16 operand>";
#endif
}

inline void GRFRange::outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const
{
#if XE4
    if (detail <= PrintDetail::xe4)
#endif
    str << 'r' << int(base) << ':' << int(len);
}

#if XE4
inline void RegisterRange::outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const
{
    auto rf = srf ? 's' : 'r';
    if (detail > PrintDetail::xe4)
        GRFRange(base, len).outputText(str, detail, man);
    else if (len > 5)
        str << '(' << rf << int(base) << ':' << rf << int(base + len - 1) << ')';
    else if (len > 1) {
        str << '(' << rf << int(base);
        for (int i = 1; i < len; i++)
            str << ',' << rf << int(base + i);
        str << ')';
    } else
        str << rf << int(base);
}
#endif

inline void Label::outputText(std::ostream &str, PrintDetail detail, LabelManager &man) {
    str << 'L' << getID(man);
}

struct NoOperand {
    static const bool emptyOp = true;
    void fixup(HW hw, int esize, int ewidth, DataType defaultType, int srcN, int arity) const {}
    constexpr DataType getType() const { return DataType::invalid; }
    constexpr bool isScalar() const { return true; }

    void outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const {}
};

struct AsmOperand {
    union {
        RegData reg;
        ExtendedReg ereg;
        Immediate imm;
        Label label;
        RegisterRange range;
#if XE4
        IndirectARF iarf;
#endif
    };
    enum class Type : uint8_t {
        none = 0,
        reg = 1,
        ereg = 2,
        imm = 3,
        label = 4,
        range = 5,
#if XE4
        iarf = 6,
#endif
    } type;

    AsmOperand()                     : type{Type::none} {}
    AsmOperand(NoOperand)            : AsmOperand() {}
    AsmOperand(RegData reg_)         : reg{reg_}, type{Type::reg} {}
    AsmOperand(ExtendedReg ereg_)    : ereg{ereg_}, type{Type::ereg} {}
    AsmOperand(Immediate imm_)       : imm{imm_}, type{Type::imm} {}
    AsmOperand(Label label_)         : label{label_}, type{Type::label} {}
    AsmOperand(GRFRange range_)      : range{range_}, type{Type::range} {}
    AsmOperand(uint32_t imm_)        : imm{imm_}, type{Type::imm} {}
#if XE3P
    AsmOperand(uint64_t imm_)        : imm{imm_}, type{Type::imm} {}
#endif
#if XE4
    AsmOperand(RegisterRange range_) : range{range_}, type{Type::range} {}
    AsmOperand(IndirectARF iarf_)    : iarf{iarf_}, type{Type::iarf} {}
#endif

    void outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const {
        switch (type) {
            case Type::none:    break;
            case Type::ereg:    ereg.outputText(str, detail, man); break;
            case Type::reg:     reg.outputText(str, detail, man); break;
            case Type::imm:     imm.outputText(str, detail, man); break;
            case Type::label: {
                auto clone = label;
                clone.outputText(str, detail, man);
                break;
            }
            case Type::range:   range.outputText(str, detail, man); break;
#if XE4
            case Type::iarf:    str << iarf; break;
#endif
        }
    }

    bool operator!() const { return (type == Type::none); }
    operator bool() const  { return !!*this; }

    bool isEmptyOrNull() const {
        switch (type) {
            case Type::none: return true;
            case Type::reg: return reg.isNull();
            case Type::ereg: return ereg.getBase().isNull();
            case Type::range: return range.isEmpty();
            default: return false;
        }
    }
};

struct AsmInstruction {
    Opcode op;
    uint16_t ext;
    uint32_t inum;
    InstructionModifier mod;
    AsmOperand dst, src[5];
    LabelManager *labelManager;
    std::string comment;

    AsmInstruction(Opcode op_, uint16_t ext_, uint32_t inum_, InstructionModifier mod_, LabelManager *man,
        AsmOperand dst_ = NoOperand(), AsmOperand src0 = NoOperand(), AsmOperand src1 = NoOperand(),
        AsmOperand src2 = NoOperand(), AsmOperand src3 = NoOperand(), AsmOperand src4 = NoOperand())
            : op(op_), ext(ext_), inum(inum_), mod(mod_), dst(dst_), src{src0, src1, src2, src3, src4}, labelManager{man}, comment{} {}

    explicit AsmInstruction(uint32_t inum_, const std::string &comment_)
            : op(Opcode::illegal), ext(0), inum(inum_), mod{}, dst{}, src{}, labelManager{nullptr}, comment{comment_} {}
    inline AsmInstruction(HW hw, const autoswsb::SyncInsertion &si);
    inline AsmInstruction(const autoswsb::DummyMovInsertion &mi);

    bool isLabel() const   { return (op == Opcode::illegal) && (dst.type == AsmOperand::Type::label); }
    bool isComment() const { return (op == Opcode::illegal) && !comment.empty(); }

    // Auto-SWSB interface.
    bool autoSWSB() const       { return mod.isAutoSWSB(); }
    SWSBInfo swsb() const       { return mod.getSWSB(); }
    void setSWSB(SWSBInfo swsb) { mod.setSWSB(swsb); }
    void clearAutoSWSB()        { mod.setAutoSWSB(false); }
    Opcode opcode() const       { return op; }
    SyncFunction syncFC() const { return static_cast<SyncFunction>(ext & 0xF); }
    SharedFunction sfid() const { return static_cast<SharedFunction>(ext & 0xF); }
    bool eot() const            { return mod.isEOT(); }
    bool predicated() const     { return !mod.isWrEn() || (mod.getPredCtrl() != PredCtrl::None); }
    bool atomic() const         { return mod.isAtomic(); }
    static bool is64()          { return false; }

    inline unsigned dstTypecode()  const { return getTypecode(dst); }
    inline unsigned src0Typecode() const { return getTypecode(src[0]); }
    inline unsigned src1Typecode() const { return getTypecode(src[1]); }
    inline autoswsb::DestinationMask destinations(int &jip, int &uip) const;
    inline bool getOperandRegion(autoswsb::DependencyRegion &region, int opNum) const;
    inline bool getCModDepRegion(autoswsb::DependencyRegion &region) const;

    void shiftJIP(int32_t shift) const {}
    void shiftUIP(int32_t shift) const {}

    bool getImm32(uint32_t &imm, int opNum = 0) const {
        if (src[opNum].type == AsmOperand::Type::imm) {
            imm = uint32_t(static_cast<uint64_t>(src[opNum].imm));
            return true;
        } else
            return false;
    }
    bool getARFType(ARFType &arfType, int opNum, HW hw) const {
        auto &opd = (opNum < 0) ? dst : src[opNum];
        if (opd.type == AsmOperand::Type::reg && opd.reg.isARF()) {
            arfType = opd.reg.getARFType();
            return true;
        } else
            return false;
    }
    bool getSendDesc(MessageDescriptor &desc) const { return getImm32(desc.all, 3); }
    int getFencedepJIP() const {
        if (src[0].type == AsmOperand::Type::label) {
            auto label = src[0].label;
            return labelManager->getTarget(label.getID(*labelManager)) - inum + 1;
        } else
            return 0;
    }

protected:
    static inline unsigned getTypecode(const AsmOperand &op);
};

AsmInstruction::AsmInstruction(HW hw, const autoswsb::SyncInsertion &si)
{
#if XE4
    op = (hw >= HW::Xe4) ? Opcode::sync_64E : Opcode::sync;
#else
    op = Opcode::sync;
#endif
    ext = static_cast<uint8_t>(si.fc);
    mod = InstructionModifier::createMaskCtrl(true);
    mod.setSWSB(si.swsb);
    dst = NoOperand();
    for (auto n = 0; n < 4; n++)
        src[n] = NoOperand();
    if (si.mask)
        src[0] = Immediate::ud(si.mask);
    else
        src[0] = NullRegister();
}

AsmInstruction::AsmInstruction(const autoswsb::DummyMovInsertion &mi)
{
    op = Opcode::mov_gen12;
    ext = 0;
    mod = 1 | InstructionModifier::createMaskCtrl(true);
    mod.setSWSB(mi.swsb);
    dst = NullRegister().retype(mi.dt);
    for (auto n = 1; n < 4; n++)
        src[n] = NoOperand();
    if (mi.constant) {
        src[0] = Immediate::zero(mi.dt);
    } else
        src[0] = GRF(mi.grf).sub(0, mi.dt);
}

unsigned AsmInstruction::getTypecode(const AsmOperand &op)
{
    DataType dt = DataType::invalid;

    switch (op.type) {
        case AsmOperand::Type::reg:  dt = op.reg.getType(); break;
        case AsmOperand::Type::ereg: dt = op.ereg.getType(); break;
        default: break;
    }

    return getTypecode12(dt);
}

autoswsb::DestinationMask AsmInstruction::destinations(int &jip, int &uip) const
{
    using namespace autoswsb;

    if (!isBranch(op))
        return eot() ? DestNone : DestNextIP;

    if (src[0].type == AsmOperand::Type::reg)
        return DestUnknown;

    DestinationMask mask = DestNextIP;
    if (src[0].type == AsmOperand::Type::label) {
        auto label = src[0].label;
        mask |= DestJIP;
        jip = labelManager->getTarget(label.getID(*labelManager)) - inum;
    }

    if (src[1].type == AsmOperand::Type::label) {
        auto label = src[1].label;
        mask |= DestUIP;
        uip = labelManager->getTarget(label.getID(*labelManager)) - inum;
    }

    if (op == Opcode::jmpi && mod.getPredCtrl() == PredCtrl::None)
        mask &= ~DestNextIP;

    return mask;
}

bool AsmInstruction::getOperandRegion(autoswsb::DependencyRegion &region, int opNum) const
{
    using namespace autoswsb;
    const AsmOperand &operand = (opNum < 0) ? dst : src[opNum];
    RegData rd;
    auto hw = region.hw;

    switch (operand.type) {
        case AsmOperand::Type::reg:    rd = operand.reg; break;
        case AsmOperand::Type::ereg:   rd = operand.ereg.getBase(); break;
        case AsmOperand::Type::range:
            if (!operand.range.isValid()) return false;
            region = DependencyRegion(hw, operand.range);
            return true;
        case AsmOperand::Type::none:
            if (hw >= HW::Xe3 && (op == Opcode::send || op == Opcode::sendc) && opNum == 1
                    && src[0].type == AsmOperand::Type::reg && src[0].reg.isIndirect()
                    && src[3].type == AsmOperand::Type::imm) {
                auto desc = static_cast<MessageDescriptor>(uint32_t(static_cast<uint64_t>(src[3].imm)));
                auto sreg = src[0].reg.getIndirectReg();
                sreg.setRegion(1, 1, 0);
                region = DependencyRegion(hw, desc.parts.messageLen, sreg);
                return true;
            }
#if XE3P
            if ((op == Opcode::sendg || op == Opcode::sendgc) && opNum == 1
                    && src[0].type == AsmOperand::Type::reg && src[0].reg.isIndirect()) {
                auto sreg = src[0].reg.getIndirectReg();
                sreg.setRegion(1, 1, 0);
                region = DependencyRegion(hw, ext >> 8, sreg);
                return true;
            }
#endif
            return false;
        default: return false;
    }

    if (rd.isARF() && !autoswsb::trackableARF(rd.getARFType()))
        return false;

    if (rd.isIndirect())
        region = DependencyRegion();
    else if (op == Opcode::send || op == Opcode::sendc) {
        int len = 0;
        if (opNum <= 0) {
            if (src[3].type == AsmOperand::Type::imm) {
                MessageDescriptor desc;
                desc.all = uint32_t(static_cast<uint64_t>(src[3].imm));
                len = (opNum < 0) ? desc.parts.responseLen : desc.parts.messageLen;
                if (len == 31) len++;       // 32 GRF responses are encoded as 31. Conservatively use the higher value.
            } else
                len = -1;
        } else if (opNum == 1) {
            bool exdescImm = (src[2].type == AsmOperand::Type::imm);
            if (exdescImm && (hw >= HW::XeHPG))
                len = ext >> 8;
            else if (exdescImm) {
                ExtendedMessageDescriptor exdesc;
                exdesc.all = uint32_t(static_cast<uint64_t>(src[2].imm));
                len = exdesc.parts.extMessageLen;
            } else
                len = -1;
        }
        if (len == 0)
            return false;
        else if (len == -1)
            region = DependencyRegion();
        else
            region = DependencyRegion(hw, GRFRange(rd.getBase(), len));
#if XE3P
    } else if (op == Opcode::sendg || op == Opcode::sendgc || op == Opcode::sendgx || op == Opcode::sendgxc) {
        if (opNum == -1 && !rd.isNull() && src[4].type == AsmOperand::Type::imm) {
            SendgMessageDescriptor desc;
            desc.all = static_cast<uint64_t>(src[4].imm);
            int execSize = mod.getExecSize();
#if XE4
            if (dst.type == AsmOperand::Type::reg && dst.reg.isSRF())
                execSize = 1;
#endif
            int len = desc.dstLen(hw, execSize, static_cast<SharedFunction>(ext & 0xF));
            if (len == -1)
                region = DependencyRegion();
            else
                region = DependencyRegion(hw, GRFRange(rd.getBase(), len));
        } else
            region = DependencyRegion(hw, mod.getExecSize(), rd);
#endif
    } else if (op == Opcode::dpas || op == Opcode::dpasw) {
        unsigned sdepth = ext >> 8;
        unsigned rcount = ext & 0xFF;
        unsigned len;

        switch (opNum) {
            case -1:
            case 0: len = GRF::bytesToGRFs(hw, rcount * operand.reg.getBytes() * mod.getExecSize()); break;
            case 1: len = sdepth; break;
            case 2:
                if (op == Opcode::dpasw) rcount = (rcount + 1) >> 1;
#if XE3P
                len = GRF::bytesToGRFs(hw, operand.reg.getByteOffset() + sdepth * rcount * 4 * operand.reg.getDwords());
#else
                len = GRF::bytesToGRFs(hw, operand.reg.getByteOffset() + sdepth * rcount * 4);
#endif
                break;
            default: return false;
        }

        region = DependencyRegion(hw, GRFRange(operand.reg.getBase(), len));
    } else
        region = DependencyRegion(hw, mod.getExecSize(), rd);

    return true;
}

bool AsmInstruction::getCModDepRegion(autoswsb::DependencyRegion &region) const
{
    if (mod.getCMod() == ConditionModifier::none)
        return false;

    region = autoswsb::DependencyRegion(region.hw, 1, mod.getFlagReg());
    return true;
}

#if defined(NGEN_GLOBAL_REGS) && !defined(NGEN_GLOBAL_REGS_DEFINED)
#include "ngen_registers.hpp"
#endif

class AsmCodeGenerator {
private:
#include "ngen_compiler_fix.hpp"
public:
    explicit AsmCodeGenerator(Product product_) : hardware(getCore(product_.family)), product(product_), defaultOutput{nullptr}, cancelAutoSWSB_(false),
#if XE3P
                                                  lfsr{this}, shfl{this},
#endif
                                                  sync{this}, load{this}, store{this}, atomic{this}
    {
        isGen12 = (hardware >= HW::Gen12LP);
        _workaround_();
        streamStack.push_back(new InstructionStream());
#if XE3P
        useEfficient64Bit = (hardware >= HW::Xe3p);
#endif
    }

    explicit AsmCodeGenerator(HW hardware_, int stepping_ = 0) : AsmCodeGenerator({genericProductFamily(hardware_), 0, PlatformType::Unknown}) {}

    AsmCodeGenerator(HW hardware_, std::ostream &defaultOutput_, int stepping_ = 0) : AsmCodeGenerator(hardware_, stepping_) {
        defaultOutput = &defaultOutput_;
    }
    ~AsmCodeGenerator() noexcept(false) {
        if (defaultOutput != nullptr)
            getCode(*defaultOutput);
        for (auto &s : streamStack)
            delete s;
    }

    using RootCodeGenerator = AsmCodeGenerator;

    constexpr HW getHardware() const { return hardware; }

    inline void getCode(std::ostream &out);
    inline void getPartialCode(std::ostream &out);
    void enableLineNumbers(bool enable = true) { lineNumbers = enable; }
    inline void disableAutoSWSB();
    void cancelAutoSWSB() { cancelAutoSWSB_ = true; }

    Product getProduct() const { return product; }
    ProductFamily getProductFamily() const { return product.family; }
    int getStepping() const { return product.stepping; }

    void setProduct(Product product_) { product = product_; }
    void setProductFamily(ProductFamily family_) { product.family = family_; }
    void setStepping(int stepping_) { product.stepping = stepping_; }

protected:
    struct InstructionStream {
        std::vector<AsmInstruction> buffer;
        std::vector<uint32_t> labels;

        template <typename... Remaining>
        AsmInstruction &append(Opcode op, uint16_t ext, Remaining&&... args) {
            buffer.emplace_back(op, ext, 0, std::forward<Remaining>(args)...);
            return buffer.back();
        }

        void appendComment(const std::string &str) { buffer.emplace_back(0, str); }

        void mark(Label &label, LabelManager &man) {
            uint32_t id = label.getID(man);

            man.setTarget(id, uint32_t(buffer.size()));
            labels.push_back(id);
            buffer.emplace_back(Opcode::illegal, 0, 0, InstructionModifier(), &man, label);
        }

        void append(InstructionStream &other, LabelManager &man) {
            for (uint32_t id : other.labels)
                man.offsetTarget(id, uint32_t(buffer.size()));

            buffer.insert(buffer.end(), other.buffer.begin(), other.buffer.end());
            labels.insert(labels.end(), other.labels.begin(), other.labels.end());
        }
    };

    HW hardware;
    Product product;
    bool isGen12;
    int declaredGRFs = 128;
    std::ostream *defaultOutput;
    bool lineNumbers = false;

    InterfaceLabels _interfaceLabels;

    Label _lastFenceLabel;
    RegData _lastFenceDst;

private:
    InstructionModifier defaultModifier;
    LabelManager labelManager;
    std::vector<InstructionStream*> streamStack;
    std::atomic<bool> cancelAutoSWSB_;

#if XE3P
    bool useEfficient64Bit = false;
#endif

    inline void unsupported();

    // Output functions.
    template <typename D, typename S0, typename S1, typename S2>
    inline void opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2, uint16_t ext);

    template <typename D, typename S0, typename S1, typename S2> void opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2) {
        opX(op, defaultType, mod, dst, src0, src1, src2, 0);
    }
    template <typename D, typename S0, typename S1> void opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1) {
        opX(op, defaultType, mod, dst, src0, src1, NoOperand());
    }
    template <typename D, typename S0, typename S1> void opX(Opcode op, const InstructionModifier &mod, D dst, S0 src0, S1 src1) {
        opX(op, DataType::invalid, mod, dst, src0, src1);
    }
    template <typename D, typename S0> void opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0) {
        opX(op, defaultType, mod, dst, src0, NoOperand());
    }
    template <typename D, typename S0> void opX(Opcode op, const InstructionModifier &mod, D dst, S0 src0) {
        opX(op, DataType::invalid, mod, dst, src0);
    }
    template <typename D> void opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst) {
        opX(op, defaultType, mod, dst, NoOperand());
    }
    template <typename D> void opX(Opcode op, const InstructionModifier &mod, D dst) {
        opX(op, DataType::invalid, mod, dst);
    }
    void opX(Opcode op) {
        opX(op, InstructionModifier(), NoOperand());
    }
    void opX(Opcode op, const InstructionModifier &mod, Label &jip) {
        (void) jip.getID(labelManager);
        opX(op, DataType::invalid, mod, NoOperand(), jip);
    }
    void opX(Opcode op, const InstructionModifier &mod, Label &jip, Label &uip) {
        (void) jip.getID(labelManager);
        (void) uip.getID(labelManager);
        opX(op, DataType::invalid, mod, NoOperand(), jip, uip, NoOperand());
    }

    template <typename S1, typename ED, typename D>
    void opSend(Opcode op, const InstructionModifier &mod, SharedFunction sf, RegData dst, RegData src0, S1 src1, ED exdesc, D desc) {
        if (!(hardware >= HW::Xe3 && src0.isIndirect()))
        if (src1.emptyOp && (isGen12 || op == Opcode::sends || op == Opcode::sendsc)) {
            opSend(op, mod, sf, dst, src0, null, exdesc, desc);
            return;
        }

        auto &i = streamStack.back()->append(op, static_cast<uint8_t>(sf), mod | defaultModifier, &labelManager, dst, src0, src1, exdesc, desc);
        if (i.src[2].type == AsmOperand::Type::imm && i.src[1].type != AsmOperand::Type::none) {
            uint32_t exdesc = uint32_t(static_cast<uint64_t>(i.src[2].imm));
            if (isGen12) {
                if (hardware >= HW::XeHPG) {
                    i.ext |= 0x80 | (((exdesc >> 6) & 0x1F) << 8);
                    i.src[2].imm = uint32_t(exdesc & ~0x7EF);
                } else
                i.src[2].imm = uint32_t(exdesc & ~0x2F);
            } else
                i.src[2].imm = uint32_t(exdesc | static_cast<uint8_t>(sf));
        }
    }
#if XE3P
    template <typename T>
    static inline T uniformizeInd(T r)              { return r; }
    static inline RegData uniformizeInd(RegData r)  { r.setOffset(r.getByteOffset()); r.setType(DataType::ub); return r; }
    template <typename T>
    static inline void applyLen(T &r, int len)             {}
    static inline void applyLen(RegisterRange &r, int len) { if (len > 0 && r.isValid()) r = RegisterRange(r[0], len); }

    template <typename S0, typename S1, typename I0, typename I1>
    void opSendg(Opcode op, const InstructionModifier &mod, SharedFunction sf, RegData dst, const S0 &src0, const S1 &src1, I0 ind0, I1 ind1, uint64_t desc) {
#if XE4
        if (hardware >= HW::Xe4) {
            switch (op) {
                case Opcode::sendg:
                case Opcode::sendgx:  op = Opcode::send_128C;  break;
                case Opcode::sendgc:
                case Opcode::sendgxc: op = Opcode::sendc_128C; break;
                default: unsupported();
            }
            canonicalizeSRF(ind0);
            canonicalizeSRF(ind1);
            auto mdesc = static_cast<SendgMessageDescriptor>(desc);
            auto msrc0 = src0;
            auto msrc1 = src1;
            auto execSize = dst.isSRF() ? 1 : 32;
            int dlen = mdesc.dstLen(hardware, execSize, sf);
            applyLen(msrc0, mdesc.src0Len(hardware, execSize, sf));
            applyLen(msrc1, mdesc.src1Len(hardware, execSize, sf));
            if (dst.isValid() && !dst.isNull() && dlen > 0)
                (void) streamStack.back()->append(op, static_cast<uint8_t>(sf), mod | defaultModifier, &labelManager, RegisterRange(dst, dlen), msrc0, msrc1, ind0, ind1, Immediate::uq(desc));
            else
                (void) streamStack.back()->append(op, static_cast<uint8_t>(sf), mod | defaultModifier, &labelManager, dst, msrc0, msrc1, ind0, ind1, Immediate::uq(desc));
        } else
#endif
        (void) streamStack.back()->append(op, static_cast<uint8_t>(sf), mod | defaultModifier, &labelManager, dst, src0, src1, uniformizeInd(ind0), uniformizeInd(ind1), desc);
    }
    template <typename I0, typename I1>
    void opSendg(Opcode op, const InstructionModifier &mod, SharedFunction sf, RegData dst, const RegData &src0, int src0Len, const I0 &ind0, const I1 &ind1, uint64_t desc) {
        if (src0.isIndirect())
            (void) streamStack.back()->append(op, static_cast<uint8_t>(sf) | 0x80 | (src0Len << 8), mod | defaultModifier, &labelManager, dst, src0, NoOperand(), ind0, ind1, desc);
        else
            opSendg(op, mod, sf, dst, GRFRange(src0.getBase(), src0Len), NullRegister(), uniformizeInd(ind0), uniformizeInd(ind1), desc);
    }
#endif
    void opDpas(Opcode op, DataType defaultType, const InstructionModifier &mod, int sdepth, int rcount, RegData dst, RegData src0, RegData src1, RegData src2) {
        dst.fixup(hardware, 1, 0, defaultType, -1, 3);
        src0.fixup(hardware, 1, 0, defaultType, 0, 3);
        src1.fixup(hardware, 1, 0, defaultType, 1, 3);
        src2.fixup(hardware, 1, 0, defaultType, 2, 3);
        (void) streamStack.back()->append(op, (sdepth << 8) | rcount, mod | defaultModifier, &labelManager, dst, src0, src1, src2);
    }
#if XE3P
    void opBdpas(Opcode op, DataType defaultType, const InstructionModifier &mod, int sdepth, int rcount, RegData dst, RegData src0, RegData src1, RegData src2, RegData src3, RegData src4) {
        dst.fixup(hardware, 1, 0, defaultType, -1, 3);
        src0.fixup(hardware, 1, 0, defaultType, 0, 3);
        src1.fixup(hardware, 1, 0, defaultType, 1, 3);
        src2.fixup(hardware, 1, 0, defaultType, 2, 3);
        src3.fixup(hardware, 1, 0, DataType::ub, 3, 5);
        src4.fixup(hardware, 1, 0, DataType::ub, 4, 5);
        (void) streamStack.back()->append(op, (sdepth << 8) | rcount, mod | defaultModifier, &labelManager, dst, src0, src1, src2, src3, src4);
    }
#endif
    template <typename D, typename S0> void opCall(Opcode op, const InstructionModifier &mod, D dst, S0 src0) {
        (void) streamStack.back()->append(op, 0, mod | defaultModifier | NoMask, &labelManager, dst, src0);
    }
    template <typename S1> void opJmpi(Opcode op, const InstructionModifier &mod, S1 src1) {
        (void) streamStack.back()->append(op, 0, mod | defaultModifier | NoMask, &labelManager, NoOperand(), src1);
    }
    template <typename S0> void opSync(SyncFunction fc, const InstructionModifier &mod, S0 src0) {
#if XE4
        auto op = (hardware >= HW::Xe4) ? Opcode::sync_64E : Opcode::sync;
#else
        auto op = Opcode::sync;
#endif
        (void) streamStack.back()->append(op, static_cast<uint8_t>(fc), mod | defaultModifier, &labelManager, NoOperand(), src0);
    }
    template <typename S0> void opDirective(Directive directive, const S0 &src0) {
#if XE4
        if (hardware >= HW::Xe4)
            streamStack.back()->append(Opcode::directive_xe4, 0, InstructionModifier::createAutoSWSB(), &labelManager, GRF(static_cast<int>(directive)), src0);
        else
#endif
        opX(Opcode::directive, DataType::ud, InstructionModifier::createAutoSWSB(), GRF(static_cast<int>(directive)), src0);
    }
#if XE4
    template <typename D, typename S0, typename S1, typename S2> inline void opX(OpcodeClassXe4 op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2, uint16_t ext = 0);
    template <typename D, typename S0, typename S1> inline void opX(OpcodeClassXe4 op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1) {
        opX(op, defaultType, mod, dst, src0, src1, NoOperand());
    }
    template <typename D, typename S0> inline void opX(OpcodeClassXe4 op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0) {
        opX(op, defaultType, mod, dst, src0, NoOperand());
    }
    void opX(OpcodeClassXe4 op, const InstructionModifier &mod, Label &jip) {
        (void) jip.getID(labelManager);
        opX(op, DataType::invalid, mod, NoOperand(), jip);
    }
    void opX(OpcodeClassXe4 op) {
        opX(op, DataType::invalid, InstructionModifier(), NoOperand(), NoOperand());
    }
    template <typename S1> void opCvt(OpcodeClassXe4 opclass, DataType defaultType, const InstructionModifier &mod, RegData dst, RegData src0, S1 src1);
    template <typename S1, typename S2> void opDP4A(OpcodeClassXe4 opclass, DataType defaultType, const InstructionModifier &mod, RegData dst, RegData src0, S1 src1, S2 src2);
    template <typename S0, typename S1> void opMovb(OpcodeClassXe4 opclass, DataType defaultType, const InstructionModifier &mod, RegData dst, S0 src0, S1 lanemask);
    template <typename S0, typename S1, typename S2> void opMullh(OpcodeClassXe4 opclass, DataType defaultType, const InstructionModifier &mod, RegData dst, S0 src0, S1 src1, S2 src2);
#endif

    inline void finalize();

    enum class ModPlacementType {
        Pre, Mid, Post,
#if XE4
        PreXe4, MidXe4, PostXe4,
#endif
    };
    inline void outX(std::ostream &out, const AsmInstruction &i, int &lineNo);
    inline void outExt(std::ostream &out, const AsmInstruction &i);
    inline void outMods(std::ostream &out, const InstructionModifier &mod, Opcode op, ModPlacementType location, uint16_t ext = 0, uint32_t ext2 = 0);
    inline void outComment(std::ostream &out, const AsmInstruction &i);

#if XE4
    inline void outXe4(std::ostream &out, const AsmInstruction &i, int &lineNo);
    inline void outExtSendXe4(std::ostream &out, const AsmInstruction &i);
    inline bool outSpecialOps(std::ostream &out, const AsmInstruction &i);
#endif

    InstructionModifier defaultMods() const {
#if XE4
        if (hardware >= HW::Xe4) return InstructionModifier{};
#endif
        return GRF::bytes(hardware) >> 2;
    }

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
    void pushStream(InstructionStream &s)           { pushStream(&s); }
    void pushStream(InstructionStream *s)           { streamStack.push_back(s); }

    inline InstructionStream *popStream();

    void appendStream(InstructionStream *s)         { appendStream(*s); }
    void appendStream(InstructionStream &s)         { streamStack.back()->append(s, labelManager); }
    void appendCurrentStream()                      { InstructionStream *s = popStream(); appendStream(s); delete s; }

    void discardStream()                            { delete popStream(); }

    void requireGRF(int grfs)                       { declaredGRFs = grfs; }

public:
    void comment(const std::string &str)            { streamStack.back()->appendComment(str); }

    // Instructions.
#if XE4
    template <typename DT = void>
    void abs_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::abs, getDataType<DT>(), mod, dst, src0);
        else
            mov(mod, dst, abs(src0));
    }
#endif
    template <typename DT = void>
    void add(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::add_128A, getDataType<DT>(), mod, dst, src0, src1);
        else
#endif
            opX(Opcode::add, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void add(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::add_128A, getDataType<DT>(), mod, dst, src0, src1);
        else
#endif
            opX(Opcode::add, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void addc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::addc, getDataType<DT>(), mod, dst, src0, src1);
        else
#endif
            opX(Opcode::addc, getDataType<DT>(), (hardware >= HW::XeHPC) ? mod : (mod | AccWrEn), dst, src0, src1);
    }
    template <typename DT = void>
    void addc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::addc, getDataType<DT>(), mod, dst, src0, src1);
        else
#endif
        opX(Opcode::addc, getDataType<DT>(), (hardware >= HW::XeHPC) ? mod : (mod | AccWrEn), dst, src0, src1);
    }
#if XE4
    template <typename DT = void>
    void addc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const FlagRegister &carryIn, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::addc, getDataType<DT>(), mod, dst, src0, src1, carryIn);
    }
    template <typename DT = void>
    void addc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const FlagRegister &carryIn, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::addc, getDataType<DT>(), mod, dst, src0, src1, carryIn);
    }
    template <typename DT = void>
    void addc(const InstructionModifier &mod, const RegData &dst, const FlagRegister &carryOut, const RegData &src0, const RegData &src1, const FlagRegister &carryIn = FlagRegister(), SourceLocation loc = {}) {
        if (carryIn.isValid())
            opX(OpcodeClassXe4::addc, getDataType<DT>(), mod | eo | carryOut, dst, src0, src1, carryIn);
        else
            opX(OpcodeClassXe4::addc, getDataType<DT>(), mod | eo | carryOut, dst, src0, src1);
    }
    template <typename DT = void>
    void addc(const InstructionModifier &mod, const RegData &dst, const FlagRegister &carryOut, const RegData &src0, const Immediate &src1, const FlagRegister &carryIn = FlagRegister(), SourceLocation loc = {}) {
        if (carryIn.isValid())
            opX(OpcodeClassXe4::addc, getDataType<DT>(), mod | eo | carryOut, dst, src0, src1, carryIn);
        else
            opX(OpcodeClassXe4::addc, getDataType<DT>(), mod | eo | carryOut, dst, src0, src1);
    }
    template <typename DT = void>
    void addc(const RegData &dst, const RegData &src0, const RegData &src1, const FlagRegister &carryIn = FlagRegister(), SourceLocation loc = {}) {
        if (carryIn.isValid())
            addc<DT>(InstructionModifier(), dst, src0, src1, carryIn);
        else
            addc<DT>(InstructionModifier(), dst, src0, src1);
    }
    template <typename DT = void>
    void addc(const RegData &dst, const RegData &src0, const Immediate &src1, const FlagRegister &carryIn = FlagRegister(), SourceLocation loc = {}) {
        if (carryIn.isValid())
            addc<DT>(InstructionModifier(), dst, src0, src1, carryIn);
        else
            addc<DT>(InstructionModifier(), dst, src0, src1);
    }
#endif
    template <typename DT = void>
    void add3(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::add3, getDataType<DT>(), mod, dst, src0, src1, src2);
        else
#endif
        opX(Opcode::add3, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void add3(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(Opcode::add3, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void add3(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::add3, getDataType<DT>(), mod, dst, src0, src1, src2);
        else
#endif
         opX(Opcode::add3, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void add3(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(Opcode::add3, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void and_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            bfn2<DT>(mod, 0x88, dst, src0, src1);
        else
#endif
        opX(isGen12 ? Opcode::and_gen12 : Opcode::and_, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void and_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            bfn2<DT>(mod, 0x88, dst, src0, src1);
        else
#endif
        opX(isGen12 ? Opcode::and_gen12 : Opcode::and_, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void asr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::asr, getDataType<DT>(), mod, dst, src0, src1);
        else
#endif
        opX(isGen12 ? Opcode::asr_gen12 : Opcode::asr, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void asr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::asr, getDataType<DT>(), mod, dst, src0, src1);
        else
#endif
        opX(isGen12 ? Opcode::asr_gen12 : Opcode::asr, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
        void avg(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::avg, getDataType<DT>(), mod, dst, src0, src1);
        else
#endif
        opX(Opcode::avg, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void avg(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::avg, getDataType<DT>(), mod, dst, src0, src1);
        else
#endif
        opX(Opcode::avg, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void bfe(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::bfegen, getDataType<DT>(), mod, dst, src2, src0, src1);
        else
#endif
        opX(isGen12 ? Opcode::bfe_gen12 : Opcode::bfe, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void bfe(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfe_gen12 : Opcode::bfe, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void bfe(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfe_gen12 : Opcode::bfe, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void bfe(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfe_gen12 : Opcode::bfe, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
#if XE4
    template <typename DT = uint32_t>
    void bfe(const InstructionModifier &mod, unsigned width, unsigned offset, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::bfe, getDataType<DT>(), mod, dst, src0, NoOperand(), NoOperand(), width | (offset << 8));
    }
#endif
    template <typename DT = void>
    void bfi1(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfi1_gen12 : Opcode::bfi1, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void bfi1(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfi1_gen12 : Opcode::bfi1, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void bfi2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            bfigen(mod, dst, src2, src1, src0);
        else
#endif
        opX(isGen12 ? Opcode::bfi2_gen12 : Opcode::bfi2, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void bfi2(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfi2_gen12 : Opcode::bfi2, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void bfi2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfi2_gen12 : Opcode::bfi2, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void bfi2(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::bfi2_gen12 : Opcode::bfi2, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
#if XE4
    template <typename DT = uint32_t>
    void bfi(const InstructionModifier &mod, unsigned width, unsigned offset, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::bfi, getDataType<DT>(), mod, dst, src0, src1, NoOperand(), width | (offset << 8));
    }
    template <typename DT = uint32_t>
    void bfi(const InstructionModifier &mod, unsigned width, unsigned offset, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::bfi, getDataType<DT>(), mod, dst, src0, src1, NoOperand(), width | (offset << 8));
    }
    template <typename DT = uint32_t>
    void bfigen(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::bfigen, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = uint32_t>
    void bfigen(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::bfigen, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
#endif
    template <typename DT = void>
    void bfn(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            bfn3<DT>(mod, ctrl, dst, src0, src1, src2);
        else
#endif
        opX(Opcode::bfn, getDataType<DT>(), mod, dst, src0, src1, src2, ctrl);
    }
    template <typename DT = void>
    void bfn(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(Opcode::bfn, getDataType<DT>(), mod, dst, src0, src1, src2, ctrl);
    }
    template <typename DT = void>
    void bfn(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            bfn3<DT>(mod, ctrl, dst, src0, src1, src2);
        else
#endif
        opX(Opcode::bfn, getDataType<DT>(), mod, dst, src0, src1, src2, ctrl);
    }
    template <typename DT = void>
    void bfn(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(Opcode::bfn, getDataType<DT>(), mod, dst, src0, src1, src2, ctrl);
    }
#if XE4
    template <typename DT = void>
    void bfn2(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::bfn2, getDataType<DT>(), mod, dst, src0, src1, NoOperand(), ctrl);
    }
    template <typename DT = void>
    void bfn2(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        bfn3<DT>(mod, ctrl, dst, src0, src1, null, loc);
    }
    template <typename DT = void>
    void bfn3(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::bfn3, getDataType<DT>(), mod, dst, src0, src1, src2, ctrl);
    }
    template <typename DT = void>
    void bfn3(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::bfn3, getDataType<DT>(), mod, dst, src0, src1, src2, ctrl);
    }
    template <typename DT = void>
    void bfn3(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::bfn3, getDataType<DT>(), mod, dst, src0, src1, src2, ctrl);
    }
#endif
    template <typename DT = void>
    void bfrev(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::bfrev, getDataType<DT>(), mod, dst, src0);
        else
#endif
        opX(isGen12 ? Opcode::bfrev_gen12 : Opcode::bfrev, getDataType<DT>(), mod, dst, src0);
    }
    void brc(const InstructionModifier &mod, Label &jip, Label &uip, SourceLocation loc = {}) {
        (void) jip.getID(labelManager);
        (void) uip.getID(labelManager);
        opX(Opcode::brc, mod, jip, uip);
    }
    void brc(const InstructionModifier &mod, const RegData &src0, SourceLocation loc = {}) {
        opCall(Opcode::brc, mod, NoOperand(), src0);
    }
    void brd(const InstructionModifier &mod, Label &jip, SourceLocation loc = {}) {
        (void) jip.getID(labelManager);
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::brd, mod, jip);
        else
#endif
        opX(Opcode::brd, mod, jip);
    }
    void brd(const InstructionModifier &mod, const RegData &src0, SourceLocation loc = {}) {
        opCall(Opcode::brd, mod, NoOperand(), src0);
    }
#if XE4
    void brd(const InstructionModifier &mod, Label &jip, bool branchCtrl, SourceLocation loc = {}) {
        auto emod = mod;
        emod.setBranchCtrl(branchCtrl);
        if (branchCtrl && hardware < HW::Xe4) unsupported();
        brd(emod, jip);
    }
    template <typename DT = void>
    void brepgen(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::brepgen, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void brepgen(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::brepgen, getDataType<DT>(), mod, dst, src0, src1);
    }
#endif
    void break_(const InstructionModifier &mod, Label &jip, Label &uip, SourceLocation loc = {}) {
        (void) jip.getID(labelManager);
        (void) uip.getID(labelManager);
        opX(Opcode::break_, mod, jip, uip);
    }
    void call(const InstructionModifier &mod, const RegData &dst, Label &jip, SourceLocation loc = {}) {
        (void) jip.getID(labelManager);
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::call, DataType::invalid, mod, RegisterRange(dst, 3), jip);
        else
#endif
        opCall(Opcode::call, mod, dst, jip);
    }
    void call(const InstructionModifier &mod, const RegData &dst, const RegData &jip, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::call, DataType::invalid, mod, RegisterRange(dst, 3), jip);
        else
#endif
        opCall(Opcode::call, mod, dst, jip);
    }
#if XE4
    void calla(const InstructionModifier &mod, const RegData &dst, uint64_t jip = 0, SourceLocation loc = {}) {
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::calla, DataType::invalid, mod, RegisterRange(dst, 3), Immediate::uq(jip));
        else
            opCall(Opcode::calla, mod, dst, Immediate::ud(uint32_t(jip)));
    }
#else
    void calla(const InstructionModifier &mod, const RegData &dst, int32_t jip, SourceLocation loc = {}) {
        opCall(Opcode::calla, mod, dst, Immediate::ud(jip));
    }
#endif
    void calla(const InstructionModifier &mod, const RegData &dst, const RegData &jip, SourceLocation loc = {}) {
        opCall(Opcode::calla, mod, dst, jip);
    }
#if XE4
    void callad(const InstructionModifier &mod, const RegData &dst, uint64_t jip = 0, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::callad, DataType::invalid, mod, RegisterRange(dst, 3), Immediate::uq(jip));
    }
    void calld(const InstructionModifier &mod, const RegData &dst, Label &jip, SourceLocation loc = {}) {
        (void) jip.getID(labelManager);
        opX(OpcodeClassXe4::calld, DataType::invalid, mod, RegisterRange(dst, 3), jip);
    }
    void calld(const InstructionModifier &mod, const RegData &dst, const RegData &jip, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::calld, DataType::invalid, mod, RegisterRange(dst, 3), jip);
    }
#endif
    template <typename DT = void>
    void cbit(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::cbit, getDataType<DT>(), mod, dst, src0);
        else
#endif
        opX(Opcode::cbit, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void cmp(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::cmp_128S, getDataType<DT>(), mod, dst, src0, src1);
        else
#endif
        opX(isGen12 ? Opcode::cmp_gen12 : Opcode::cmp, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void cmp(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::cmp_128S, getDataType<DT>(), mod, dst, src0, src1);
        else
#endif
        opX(isGen12 ? Opcode::cmp_gen12 : Opcode::cmp, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void cmpn(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::cmp_128S, getDataType<DT>(), mod, dst, src0, src1);
        else
#endif
        opX(isGen12 ? Opcode::cmpn_gen12 : Opcode::cmpn, getDataType<DT>(), mod, dst, src0, src1);
    }
#if XE4
    void cnvg(const InstructionModifier &mod, uint8_t cid, bool exp, const RegData &src0, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::cnvg, DataType::invalid, mod, NoOperand(), src0, NoOperand(), NoOperand(), 0x200 | cid | (uint16_t(exp) << 8));
    }
    void cnvg(const InstructionModifier &mod, uint8_t cid, bool exp, const Immediate &src0, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::cnvg, DataType::invalid, mod, NoOperand(), src0, NoOperand(), NoOperand(), 0x200 | cid | (uint16_t(exp) << 8));
    }
    void cnvg(const InstructionModifier &mod, bool exp, const RegData &src0, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::cnvg, DataType::invalid, mod, NoOperand(), src0, NoOperand(), NoOperand(), uint16_t(exp) << 8);
    }
    void cnvg(const InstructionModifier &mod, bool exp, const Immediate &src0, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::cnvg, DataType::invalid, mod, NoOperand(), src0, NoOperand(), NoOperand(), uint16_t(exp) << 8);
    }
#endif
    template <typename DT = void>
    void csel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::csel_gen12 : Opcode::csel, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void csel(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::csel_gen12 : Opcode::csel, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void csel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::csel_gen12 : Opcode::csel, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void csel(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::csel_gen12 : Opcode::csel, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    void cont(const InstructionModifier &mod, Label &jip, Label &uip, SourceLocation loc = {}) {
        (void) jip.getID(labelManager);
        (void) uip.getID(labelManager);
        opX(Opcode::cont, mod, jip, uip);
    }
#if XE4
    template <typename DT = void>
    void cvt(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        if (hardware >= HW::Xe4)
            opCvt(OpcodeClassXe4::cvt, getDataType<DT>(), mod, dst, src0, NoOperand());
        else
            mov<DT>(mod, dst, src0, loc);
    }
    template <typename DT = void>
    void cvt2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opCvt(OpcodeClassXe4::cvt2, getDataType<DT>(), mod, dst, src0, src1);
    }
#endif
    template <typename DT = void>
    void dp2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::dp2, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dp2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::dp2, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dp3(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::dp3, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dp3(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::dp3, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dp4(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::dp4, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dp4(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::dp4, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4) {
            if (mod.hasExecSize())
                opDP4A(OpcodeClassXe4::dp4a, getDataType<DT>(), mod, dst, src1, src2, src0);
            else
                opDP4A(OpcodeClassXe4::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2);
        } else
#endif
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4) {
            if (mod.hasExecSize())
                opDP4A(OpcodeClassXe4::dp4a, getDataType<DT>(), mod, dst, src1, src2, src0);
            else
                unsupported();
        } else
#endif
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4) {
            if (mod.hasExecSize())
                opDP4A(OpcodeClassXe4::dp4a, getDataType<DT>(), mod, dst, src1, src2, src0);
            else
                opDP4A(OpcodeClassXe4::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2);
        } else
#endif
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(Opcode::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
#if XE4
    template <typename DT = void>
    void dp4a(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &src2, SourceLocation loc = {}) {
        if (mod.hasExecSize()) unsupported();
        opDP4A(OpcodeClassXe4::dp4a, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
#endif
    template <typename DT = void>
    void dpas(const InstructionModifier &mod, uint8_t sdepth, uint8_t rcount, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opDpas(Opcode::dpas, getDataType<DT>(), mod, sdepth, rcount, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void dpasw(const InstructionModifier &mod, uint8_t sdepth, uint8_t rcount, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opDpas(Opcode::dpasw, getDataType<DT>(), mod, sdepth, rcount, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void dph(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::dph, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void dph(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::dph, getDataType<DT>(), mod, dst, src0, src1);
    }
    void else_(const InstructionModifier &mod, Label &jip, Label &uip, bool branchCtrl = false, SourceLocation loc = {}) {
        (void) jip.getID(labelManager);
        (void) uip.getID(labelManager);
        opX(Opcode::else_, DataType::invalid, mod, NoOperand(), jip, uip, NoOperand(), branchCtrl);
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
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::emcos, getDataType<DT>(), mod, dst, src0);
        else
            math<DT>(mod, MathFunction::cos, dst, src0);
    }
    template <typename DT = void>
    void emexp2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::emexp2, getDataType<DT>(), mod, dst, src0);
        else
            math<DT>(mod, MathFunction::exp, dst, src0);
    }
    template <typename DT = void>
    void eminv(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::eminv, getDataType<DT>(), mod, dst, src0);
        else
            math<DT>(mod, MathFunction::inv, dst, src0);
    }
    template <typename DT = void>
    void eminvm(const InstructionModifier &mod, const ExtendedReg &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::eminvm, getDataType<DT>(), mod, dst, src0, src1);
        else
            math<DT>(mod, MathFunction::invm, dst, src0 | nomme, src1 | nomme);
    }
    template <typename DT = void>
    void emlog2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::emlog2, getDataType<DT>(), mod, dst, src0);
        else
            math<DT>(mod, MathFunction::log, dst, src0);
    }
    template <typename DT = void>
    void emrsqt(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::emrsqt, getDataType<DT>(), mod, dst, src0);
        else
            math<DT>(mod, MathFunction::rsqt, dst, src0);
    }
    template <typename DT = void>
    void emrsqtm(const InstructionModifier &mod, const ExtendedReg &dst, const RegData &src0, SourceLocation loc = {}) {
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::emrsqtm, getDataType<DT>(), mod, dst, src0);
        else
            math<DT>(mod, MathFunction::rsqtm, dst, src0 | nomme);
    }
    template <typename DT = void>
    void emsin(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::emsin, getDataType<DT>(), mod, dst, src0);
        else
            math<DT>(mod, MathFunction::sin, dst, src0);
    }
    template <typename DT = void>
    void emsgmd(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::emsgmd, getDataType<DT>(), mod, dst, src0);
        else
            math<DT>(mod, MathFunction::sigm, dst, src0);
    }
    template <typename DT = void>
    void emsqt(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::emsqt, getDataType<DT>(), mod, dst, src0);
        else
            math<DT>(mod, MathFunction::sqt, dst, src0);
    }
    template <typename DT = void>
    void emtanh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::emtanh, getDataType<DT>(), mod, dst, src0);
        else
            math<DT>(mod, MathFunction::tanh, dst, src0);
    }
#endif
    void endif(const InstructionModifier &mod, Label &jip, SourceLocation loc = {}) {
        (void) jip.getID(labelManager);
        opX(Opcode::endif, mod, NoOperand(), jip);
    }
    void endif(const InstructionModifier &mod, SourceLocation loc = {}) {
        Label next;
        endif(mod, next);
        mark(next);
    }
    template <typename DT = void>
    void fbh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::fbh, getDataType<DT>(), mod, dst, src0);
        else
#endif
        opX(Opcode::fbh, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void fbl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::fbl, getDataType<DT>(), mod, dst, src0);
        else
#endif
        opX(Opcode::fbl, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void frc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::frc, getDataType<DT>(), mod, dst, src0);
        else
#endif
        opX(Opcode::frc, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void> void frc(const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        frc<DT>(defaultMods(), dst, src0);
    }
    void goto_(const InstructionModifier &mod, Label &jip, Label &uip, bool branchCtrl = false, SourceLocation loc = {}) {
        (void) jip.getID(labelManager);
        (void) uip.getID(labelManager);
#if XE4
        if (hardware >= HW::Xe4) {
            auto emod = mod;
            emod.setBranchCtrl(branchCtrl);
            opX(OpcodeClassXe4::goto_, DataType::invalid, emod, NoOperand(), jip, uip);
        } else
#endif
        opX(Opcode::goto_, DataType::invalid, mod, NoOperand(), jip, uip, NoOperand(), branchCtrl);
    }
    void goto_(const InstructionModifier &mod, Label &jip, SourceLocation loc = {}) {
        goto_(mod, jip, jip);
    }
    void halt(const InstructionModifier &mod, Label &jip, Label &uip, SourceLocation loc = {}) {
        (void) jip.getID(labelManager);
        (void) uip.getID(labelManager);
        opX(Opcode::halt, mod, jip, uip);
    }
    void halt(const InstructionModifier &mod, Label &jip, SourceLocation loc = {}) {
        halt(mod, jip, jip);
    }
    void if_(InstructionModifier mod, Label &jip, Label &uip, bool branchCtrl, SourceLocation loc = {}) {
        (void) jip.getID(labelManager);
        (void) uip.getID(labelManager);
        opX(Opcode::if_, DataType::invalid, mod, NoOperand(), jip, uip, NoOperand(), branchCtrl);
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
            opX(OpcodeClassXe4::illegal);
        else
#endif
        opX(Opcode::illegal);
    }
    void jmpi(const InstructionModifier &mod, Label &jip, SourceLocation loc = {}) {
        (void) jip.getID(labelManager);
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::jmpi, mod, jip);
        else
#endif
        opJmpi(Opcode::jmpi, mod, jip);
    }
    void jmpi(const InstructionModifier &mod, const RegData &jip, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::jmpi, DataType::invalid, mod, NoOperand(), jip);
        else
#endif
        opJmpi(Opcode::jmpi, mod, jip);
    }
    void join(const InstructionModifier &mod, Label &jip, SourceLocation loc = {}) {
        (void) jip.getID(labelManager);
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::join, mod, jip);
        else
#endif
        opX(Opcode::join, mod, jip);
    }
    void join(const InstructionModifier &mod, SourceLocation loc = {}) {
        Label next;
        join(mod, next);
        mark(next);
    }
    template <typename DT = void>
    void line(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::line, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void line(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::line, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void lrp(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(Opcode::lrp, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void lzd(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(Opcode::lzd, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void lzd(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        opX(Opcode::lzd, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void mac(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE3P
        if (hardware >= HW::Xe3p) unsupported();
#endif
        opX(Opcode::mac, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void mac(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE3P
        if (hardware >= HW::Xe3p) unsupported();
#endif
        opX(Opcode::mac, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void mach(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE3P
        if (hardware >= HW::Xe3p) unsupported();
#endif
        opX(Opcode::mach, getDataType<DT>(), (hardware >= HW::XeHPC) ? mod : (mod | AccWrEn), dst, src0, src1);
    }
    template <typename DT = void>
    void mach(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE3P
        if (hardware >= HW::Xe3p) unsupported();
#endif
        opX(Opcode::mach, getDataType<DT>(), (hardware >= HW::XeHPC) ? mod : (mod | AccWrEn), dst, src0, src1);
    }
    template <typename DT = void>
    void macl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE3P
        if (hardware >= HW::Xe3p) unsupported();
#endif
        if (hardware < HW::Gen10) unsupported();
        opX((hardware >= HW::XeHPC) ? Opcode::macl : Opcode::mach, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void macl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE3P
        if (hardware >= HW::Xe3p) unsupported();
#endif
        if (hardware < HW::Gen10) unsupported();
        opX((hardware >= HW::XeHPC) ? Opcode::macl : Opcode::mach, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4) {
            if (mod.hasExecSize())
                opX(OpcodeClassXe4::mad_128A, getDataType<DT>(), mod, dst, src1, src2, src0);
            else
                opX(OpcodeClassXe4::mad_128A, getDataType<DT>(), mod, dst, src0, src1, src2);
        } else
#endif
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const Align16Operand &dst, const Align16Operand &src0, const Align16Operand &src1, const Align16Operand &src2, SourceLocation loc = {}) {
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4) {
            if (mod.hasExecSize())
                opX(OpcodeClassXe4::mad_128A, getDataType<DT>(), mod, dst, src1, src2, src0);
            else
                unsupported();
        } else
#endif
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            if (mod.hasExecSize())
                opX(OpcodeClassXe4::mad_128A, getDataType<DT>(), mod, dst, src1, src2, src0);
            else
                opX(OpcodeClassXe4::mad_128A, getDataType<DT>(), mod, dst, src0, src1, src2);
        else
#endif
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4) {
            if (mod.hasExecSize())
                opX(OpcodeClassXe4::mad_128A, getDataType<DT>(), mod, dst, src1, src2, src0);
            else
                unsupported();
        } else
#endif
        opX(Opcode::mad, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
#if XE4
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &src2, SourceLocation loc = {}) {
        if (mod.hasExecSize()) unsupported();
        opX(OpcodeClassXe4::mad_128A, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void mad(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const Immediate &src2, SourceLocation loc = {}) {
        if (mod.hasExecSize()) unsupported();
        opX(OpcodeClassXe4::mad_128A, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void madc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, const FlagRegister &carryIn = FlagRegister(), SourceLocation loc = {}) {
        if (carryIn.isValid())
            opX(OpcodeClassXe4::madc, getDataType<DT>(), mod, dst, src0, src1, src2, carryIn);
        else
            opX(OpcodeClassXe4::madc, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void madc(const InstructionModifier &mod, const RegData &dst, const FlagRegister &carryOut, const RegData &src0, const RegData &src1, const RegData &src2, const FlagRegister &carryIn = FlagRegister(), SourceLocation loc = {}) {
        madc<DT>(mod | eo | carryOut, dst, src0, src1, src2, carryIn);
    }
    // TODO: further madc overrides with immediates. Make life easier with RegOrImm wrapper.
    template <typename DT = void>
    void madlh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opMullh(OpcodeClassXe4::madlh, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void madlh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opMullh(OpcodeClassXe4::madlh, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void madlh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &src2, SourceLocation loc = {}) {
        opMullh(OpcodeClassXe4::madlh, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void madlh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const Immediate &src2, SourceLocation loc = {}) {
        opMullh(OpcodeClassXe4::madlh, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
#endif
    template <typename DT = void>
    void madm(const InstructionModifier &mod, const ExtendedReg &dst, const ExtendedReg &src0, const ExtendedReg &src1, const ExtendedReg &src2, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::madm, getDataType<DT>(), mod, dst, src0, src1, src2);
        else
#endif
        opX(Opcode::madm, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void math(const InstructionModifier &mod, MathFunction fc, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (mathArgCount(hardware, fc) != 1) throw invalid_operand_count_exception();
#endif
        if (fc == MathFunction::rsqtm)
            math<DT>(mod, fc, dst | nomme, src0 | nomme);
        else
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
            opX(Opcode::math, getDataType<DT>(), mod, dst, src0, NoOperand(), NoOperand(), static_cast<uint8_t>(fc));
    }
    template <typename DT = void>
    void math(const InstructionModifier &mod, MathFunction fc, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (mathArgCount(hardware, fc) != 2) throw invalid_operand_count_exception();
#endif
        if (fc == MathFunction::invm)
            math<DT>(mod, fc, dst | nomme, src0 | nomme, src1 | nomme);
        else
            opX(Opcode::math, getDataType<DT>(), mod, dst, src0, src1, NoOperand(), static_cast<uint8_t>(fc));
    }
    template <typename DT = void>
    void math(const InstructionModifier &mod, MathFunction fc, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (fc == MathFunction::invm || fc == MathFunction::rsqtm) throw invalid_operand_exception();
#endif
        opX(Opcode::math, getDataType<DT>(), mod, dst, src0, src1.forceInt32(), NoOperand(), static_cast<uint8_t>(fc));
    }
    template <typename DT = void>
    void math(InstructionModifier mod, MathFunction fc, const ExtendedReg &dst, const ExtendedReg &src0, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (fc != MathFunction::rsqtm) throw invalid_operand_exception();
#endif
#if XE4
        if (hardware >= HW::Xe4)
            emrsqtm<DT>(mod, dst, src0.getBase());
        else
#endif
        {
            mod.setCMod(ConditionModifier::eo);
            opX(Opcode::math, getDataType<DT>(), mod, dst, src0, NoOperand(), NoOperand(), static_cast<uint8_t>(fc));
        }
    }
    template <typename DT = void>
    void math(InstructionModifier mod, MathFunction fc, const ExtendedReg &dst, const ExtendedReg &src0, const ExtendedReg &src1, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (fc != MathFunction::invm) throw invalid_operand_exception();
#endif
#if XE4
        if (hardware >= HW::Xe4)
            eminvm<DT>(mod, dst, src0.getBase(), src1.getBase());
        else
#endif
        {
            mod.setCMod(ConditionModifier::eo);
            opX(Opcode::math, getDataType<DT>(), mod, dst, src0, src1, NoOperand(), static_cast<uint8_t>(fc));
        }
    }
    template <typename DT = void>
    void max_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::max_, getDataType<DT>(), mod, dst, src0, src1);
        else
#endif
        sel<DT>(mod | ge | f0[0], dst, src0, src1);
    }
    template <typename DT = void>
    void max_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::max_, getDataType<DT>(), mod, dst, src0, src1);
        else
#endif
        sel<DT>(mod | ge | f0[0], dst, src0, src1);
    }
    template <typename DT = void>
    void min_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::min_, getDataType<DT>(), mod, dst, src0, src1);
        else
#endif
        sel<DT>(mod | lt | f0[0], dst, src0, src1);
    }
    template <typename DT = void>
    void min_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::min_, getDataType<DT>(), mod, dst, src0, src1);
        else
#endif
        sel<DT>(mod | lt | f0[0], dst, src0, src1);
    }
    template <typename DT = void>
    void mov(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4) {
            auto dt = dst.getType(), st = src0.getType();
            if (dt == DataType::invalid) dt = getDataType<DT>();
            if (st == DataType::invalid) st = getDataType<DT>();
            if (dt == st || dt == DataType::invalid || st == DataType::invalid)
                opX(OpcodeClassXe4::mov_128R, getDataType<DT>(), mod, dst, src0);
            else
                cvt<DT>(mod, dst, src0);
        } else
#endif
        opX(isGen12 ? Opcode::mov_gen12 : Opcode::mov, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void mov(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::mov_128R, getDataType<DT>(), mod, dst, src0);
        else
#endif
        opX(isGen12 ? Opcode::mov_gen12 : Opcode::mov, getDataType<DT>(), mod, dst, src0);
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
        opMovb(OpcodeClassXe4::movb, getDataType<DT>(), mod, dst, src0, lanemask);
    }
    template <typename DT = void>
    void movb(const InstructionModifier &mod, RegData dst, RegData src0, RegData lanemask, SourceLocation loc = {}) {
        canonicalizeSRF(dst);
        canonicalizeSRF(src0);
#ifdef NGEN_SAFE
        if (dst.isSRF() == src0.isSRF())
            throw invalid_operand_exception();
#endif
        opMovb(OpcodeClassXe4::movb, getDataType<DT>(), mod, dst, src0, lanemask);
    }
    template <typename DT = void>
    void movb(const InstructionModifier &mod, RegData dst, Immediate src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        opMovb(OpcodeClassXe4::movb, getDataType<DT>(), mod, dst, src0, lanemask);
    }
    template <typename DT = void>
    void movb(const InstructionModifier &mod, RegData dst, Immediate src0, RegData lanemask, SourceLocation loc = {}) {
        opMovb(OpcodeClassXe4::movb, getDataType<DT>(), mod, dst, src0, lanemask);
    }
    template <typename DT = void>
    void movg(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::movg, getDataType<DT>(), mod, dst, src0);
        else
            mov(mod, dst, src0);
    }
    template <typename DT = void>
    void movs(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::movs, getDataType<DT>(), mod, dst, src0);
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
            movi<DT>(mod, dst, src0, null);
        else
            opX(isGen12 ? Opcode::movi_gen12 : Opcode::movi, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void movi(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        if (hardware < HW::Gen10) unsupported();
#ifdef NGEN_SAFE
        if (!src0.isIndirect()) throw invalid_address_mode_exception();
#endif
        opX(isGen12 ? Opcode::movi_gen12 : Opcode::movi, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void movi(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        if (hardware < HW::Gen10) unsupported();
        opX(isGen12 ? Opcode::movi_gen12 : Opcode::movi, getDataType<DT>(), mod, dst, src0, src1);
    }
#if XE4
    template <typename DT = uint32_t>
    void msk(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::msk, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = uint32_t>
    void msk(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::msk, getDataType<DT>(), mod, dst, src0, Immediate::ud(src1));
    }
    template <typename DT = uint32_t>
    void msk(const InstructionModifier &mod, const RegData &dst, uint32_t src0, const RegData &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::msk, getDataType<DT>(), mod, dst, Immediate::ud(src0), src1);
    }
    template <typename DT = uint32_t>
    void msk(const InstructionModifier &mod, const RegData &dst, uint32_t src0, uint32_t src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::msk,getDataType<DT>(),  mod, dst, Immediate::ud(src0), Immediate::ud(src1));
    }
#endif
    template <typename DT = void>
    void mul(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::mul_128A, getDataType<DT>(), mod, dst, src0, src1);
        else
#endif
        opX(Opcode::mul, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void mul(const InstructionModifier &mod, const RegData &dst, const RegData &src0, Immediate src1, SourceLocation loc = {}) {
        if (dst.getBytes() == 8)
            src1 = src1.forceInt32();
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::mul_128A, getDataType<DT>(), mod, dst, src0, src1);
        else
#endif
        opX(Opcode::mul, getDataType<DT>(), mod, dst, src0, src1);
    }
#if XE3P
    template <typename DT = void>
    void mullh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opMullh(OpcodeClassXe4::mullh, getDataType<DT>(), mod, dst, src0, src1, NoOperand());
        else
#endif
        opX(Opcode::mullh, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void mullh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opMullh(OpcodeClassXe4::mullh, getDataType<DT>(), mod, dst, src0, src1, NoOperand());
        else
#endif
        opX(Opcode::mullh, getDataType<DT>(), mod, dst, src0, src1);
    }
#endif /* XE3P */
    void nop(SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            nop128();
        else
#endif
        opX(isGen12 ? Opcode::nop_gen12 : Opcode::nop);
    }
#if XE4
    void nop128(SourceLocation loc = {}) { opX(OpcodeClassXe4::nop128); }
    void nop64(SourceLocation loc = {})  { opX(OpcodeClassXe4::nop64);  }
#endif
    template <typename DT = void>
    void not_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            bfn2<DT>(mod, 0x55, dst, src0, NullRegister());
        else
#endif
        opX(isGen12 ? Opcode::not_gen12 : Opcode::not_, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void or_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            bfn2<DT>(mod, 0xEE, dst, src0, src1);
        else
#endif
        opX(isGen12 ? Opcode::or_gen12 : Opcode::or_, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void or_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            bfn2<DT>(mod, 0xEE, dst, src0, src1);
        else
#endif
        opX(isGen12 ? Opcode::or_gen12 : Opcode::or_, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void pln(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::pln, getDataType<DT>(), mod, dst, src0, src1);
    }
#if XE4
    template <typename DT = uint32_t>
    void redand(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &lanemask, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::redand, getDataType<DT>(), mod, dst, src0, lanemask);
    }
    template <typename DT = uint32_t>
    void redand(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::redand, getDataType<DT>(), mod, dst, src0, lanemask);
    }
    template <typename DT = uint32_t>
    void redfirst(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &lanemask, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::redfirst, getDataType<DT>(), mod, dst, src0, lanemask);
    }
    template <typename DT = uint32_t>
    void redfirst(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::redfirst, getDataType<DT>(), mod, dst, src0, lanemask);
    }
    template <typename DT = uint32_t>
    void redfirstidx(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &lanemask, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::redfirstidx, getDataType<DT>(), mod, dst, src0, lanemask);
    }
    template <typename DT = uint32_t>
    void redfirstidx(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::redfirstidx, getDataType<DT>(), mod, dst, src0, lanemask);
    }
    template <typename DT = void>
    void redmax(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &lanemask, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::redmax, getDataType<DT>(), mod, dst, src0, lanemask);
    }
    template <typename DT = void>
    void redmax(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::redmax, getDataType<DT>(), mod, dst, src0, lanemask);
    }
    template <typename DT = void>
    void redmin(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &lanemask, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::redmin, getDataType<DT>(), mod, dst, src0, lanemask);
    }
    template <typename DT = void>
    void redmin(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::redmin, getDataType<DT>(), mod, dst, src0, lanemask);
    }
    template <typename DT = uint32_t>
    void redor(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &lanemask, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::redor, getDataType<DT>(), mod, dst, src0, lanemask);
    }
    template <typename DT = uint32_t>
    void redor(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::redor, getDataType<DT>(), mod, dst, src0, lanemask);
    }
    template <typename DT = void>
    void redsum(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &lanemask, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::redsum, getDataType<DT>(), mod, dst, src0, lanemask);
    }
    template <typename DT = void>
    void redsum(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::redsum, getDataType<DT>(), mod, dst, src0, lanemask);
    }
    template <typename DT = uint32_t>
    void redxor(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &lanemask, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::redxor, getDataType<DT>(), mod, dst, src0, lanemask);
    }
    template <typename DT = uint32_t>
    void redxor(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::redxor, getDataType<DT>(), mod, dst, src0, lanemask);
    }
#endif
    void ret(const InstructionModifier &mod, const RegData &src0, SourceLocation loc = {}) {
        opJmpi(Opcode::ret, mod, src0);
    }
#if XE4
    void retd(const InstructionModifier &mod, const RegData &src0, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::retd, DataType::invalid, mod, NoOperand(), src0);
    }
    template <typename DT = void>
    void rnd(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::rnd, getDataType<DT>(), mod, dst, src0);
    }
#endif
    template <typename DT = void>
    void rndd(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            rnd<DT>(mod | rd, dst, src0);
        else
#endif
        opX(Opcode::rndd, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rnde(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            rnd<DT>(mod | rne, dst, src0);
        else
#endif
        opX(Opcode::rnde, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rndu(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            rnd<DT>(mod | ru, dst, src0);
        else
#endif
        opX(Opcode::rndu, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rndz(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            rnd<DT>(mod | rtz, dst, src0);
        else
#endif
        opX(Opcode::rndz, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void rol(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::rol, getDataType<DT>(), mod, dst, src0, src1);
        else
#endif
        opX(isGen12 ? Opcode::rol_gen12 : Opcode::rol, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void rol(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::rol, getDataType<DT>(), mod, dst, src0, src1);
        else
#endif
        opX(isGen12 ? Opcode::rol_gen12 : Opcode::rol, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void ror(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::ror, getDataType<DT>(), mod, dst, src0, src1);
        else
#endif
        opX(isGen12 ? Opcode::ror_gen12 : Opcode::ror, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void ror(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::ror, getDataType<DT>(), mod, dst, src0, src1);
        else
#endif
        opX(isGen12 ? Opcode::ror_gen12 : Opcode::ror, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sad2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::sad2, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sad2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::sad2, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sada2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::sada2, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sada2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::sada2, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::sel_gen12 : Opcode::sel, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::sel_gen12 : Opcode::sel, getDataType<DT>(), mod, dst, src0, src1);
    }
#if XE4
    template <typename DT = uint32_t>
    void sel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, FlagRegister predicate, SourceLocation loc = {}) {
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::sel, getDataType<DT>(), mod, dst, src0, src1, predicate);
        else
            sel(mod | predicate, dst, src0, src1);
    }
    template <typename DT = uint32_t>
    void sel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, FlagRegister predicate, SourceLocation loc = {}) {
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::sel, getDataType<DT>(), mod, dst, src0, src1, predicate);
        else
            sel(mod | predicate, dst, src0, src1);
    }
#endif

    /* Gen12-style sends */
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(isGen12 ? Opcode::send : Opcode::sends, mod, sf, dst, src0, src1, Immediate::ud(exdesc), Immediate::ud(desc));
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(isGen12 ? Opcode::send : Opcode::sends, mod, sf, dst, src0, src1, exdesc, Immediate::ud(desc));
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const GRFRange &src1, const RegData &exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(isGen12 ? Opcode::send : Opcode::sends, mod, sf, dst, src0, src1, exdesc, Immediate::ud(desc));
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(isGen12 ? Opcode::send : Opcode::sends, mod, sf, dst, src0, NoOperand(), Immediate::ud(exdesc), Immediate::ud(desc));
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(isGen12 ? Opcode::send : Opcode::sends, mod, sf, dst, src0, NoOperand(), exdesc, Immediate::ud(desc));
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(isGen12 ? Opcode::send : Opcode::sends, mod, sf, dst, src0, src1, Immediate::ud(exdesc), desc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(isGen12 ? Opcode::send : Opcode::sends, mod, sf, dst, src0, src1, exdesc, desc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const GRFRange &src1, const RegData &exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(isGen12 ? Opcode::send : Opcode::sends, mod, sf, dst, src0, src1, exdesc, desc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(isGen12 ? Opcode::send : Opcode::sends, mod, sf, dst, src0, NoOperand(), Immediate::ud(exdesc), desc);
    }
    void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(isGen12 ? Opcode::send : Opcode::sends, mod, sf, dst, src0, NoOperand(), exdesc, desc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(isGen12 ? Opcode::sendc : Opcode::sendsc, mod, sf, dst, src0, src1, Immediate::ud(exdesc), Immediate::ud(desc));
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(isGen12 ? Opcode::sendc : Opcode::sendsc, mod, sf, dst, src0, src1, exdesc, Immediate::ud(desc));
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const GRFRange &src1, const RegData &exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(isGen12 ? Opcode::sendc : Opcode::sendsc, mod, sf, dst, src0, src1, exdesc, Immediate::ud(desc));
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(isGen12 ? Opcode::sendc : Opcode::sendsc, mod, sf, dst, src0, NoOperand(), Immediate::ud(exdesc), Immediate::ud(desc));
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &exdesc, uint32_t desc, SourceLocation loc = {}) {
        opSend(isGen12 ? Opcode::sendc : Opcode::sendsc, mod, sf, dst, src0, NoOperand(), exdesc, Immediate::ud(desc));
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(isGen12 ? Opcode::sendc : Opcode::sendsc, mod, sf, dst, src0, src1, Immediate::ud(exdesc), desc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(isGen12 ? Opcode::sendc : Opcode::sendsc, mod, sf, dst, src0, src1, exdesc, desc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const GRFRange &src1, const RegData &exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(isGen12 ? Opcode::sendc : Opcode::sendsc, mod, sf, dst, src0, src1, exdesc, desc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(isGen12 ? Opcode::sendc : Opcode::sendsc, mod, sf, dst, src0, NoOperand(), Immediate::ud(exdesc), desc);
    }
    void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, const RegData &exdesc, const RegData &desc, SourceLocation loc = {}) {
        opSend(isGen12 ? Opcode::sendc : Opcode::sendsc, mod, sf, dst, src0, NoOperand(), exdesc, desc);
    }
    template <typename T1, typename T2> void send(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, NoOperand src1, T1 exdesc, T2 desc, SourceLocation loc = {}) {
        opSend(Opcode::send, mod, sf, dst, src0, src1, exdesc, desc);
    }
    template <typename T1, typename T2> void sendc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, NoOperand src1, T1 exdesc, T2 desc, SourceLocation loc = {}) {
        opSend(Opcode::sendc, mod, sf, dst, src0, src1, exdesc, desc);
    }
    /* Pre-Gen12 style sends */
    void send(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc, SourceLocation loc = {}) {
        if (isGen12)
            send(mod, static_cast<SharedFunction>(exdesc & 0xF), dst, src0, null, exdesc, desc);
        else
            send(mod, SharedFunction::null, dst, src0, NoOperand(), Immediate::ud(exdesc), Immediate::ud(desc));
    }
    void send(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc, SourceLocation loc = {}) {
        if (isGen12)
            send(mod, static_cast<SharedFunction>(exdesc & 0xF), dst, src0, null, exdesc, desc);
        else
            send(mod, SharedFunction::null, dst, src0, NoOperand(), Immediate::ud(exdesc), desc);
    }
    void sendc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, uint32_t desc, SourceLocation loc = {}) {
        if (isGen12)
            sendc(mod, static_cast<SharedFunction>(exdesc & 0xF), dst, src0, null, exdesc, desc);
        else
            sendc(mod, SharedFunction::null, dst, src0, NoOperand(), Immediate::ud(exdesc), Immediate::ud(desc));
    }
    void sendc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t exdesc, const RegData &desc, SourceLocation loc = {}) {
        if (isGen12)
            sendc(mod, static_cast<SharedFunction>(exdesc & 0xF), dst, src0, null, exdesc, desc);
        else
            sendc(mod, SharedFunction::null, dst, src0, NoOperand(), Immediate::ud(exdesc), desc);
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, uint32_t desc, SourceLocation loc = {}) {
        send(mod, static_cast<SharedFunction>(exdesc & 0xF), dst, src0, src1, exdesc, desc);
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, const RegData &desc, SourceLocation loc = {}) {
        send(mod, static_cast<SharedFunction>(exdesc & 0xF), dst, src0, src1, exdesc, desc);
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, uint32_t desc, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (isGen12) throw sfid_needed_exception();
#endif
        send(mod, static_cast<SharedFunction>(0), dst, src0, src1, exdesc, desc);
    }
    void sends(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, const RegData &desc, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (isGen12) throw sfid_needed_exception();
#endif
        send(mod, static_cast<SharedFunction>(0), dst, src0, src1, exdesc, desc);
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, uint32_t desc, SourceLocation loc = {}) {
        sendc(mod, static_cast<SharedFunction>(exdesc & 0xF), dst, src0, src1, exdesc, desc);
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t exdesc, const RegData &desc, SourceLocation loc = {}) {
        sendc(mod, static_cast<SharedFunction>(exdesc & 0xF), dst, src0, src1, exdesc, desc);
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, uint32_t desc, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (isGen12) throw sfid_needed_exception();
#endif
        sendc(mod, static_cast<SharedFunction>(0), dst, src0, src1, exdesc, desc);
    }
    void sendsc(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &exdesc, const RegData &desc, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (isGen12) throw sfid_needed_exception();
#endif
        sendc(mod, static_cast<SharedFunction>(0), dst, src0, src1, exdesc, desc);
    }
#if XE3P
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, NullRegister(), NoOperand(), NoOperand(), desc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, NullRegister(), ind0, NoOperand(), desc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, NullRegister(), ind0, ind1, desc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegisterRange &src1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src1, NoOperand(), NoOperand(), desc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegisterRange &src1, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src1, ind0, NoOperand(), desc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegisterRange &src1, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src1, ind0, ind1, desc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src0Len, NoOperand(), NoOperand(), desc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src0Len, ind0, NoOperand(), desc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendg, mod, sf, dst, src0, src0Len, ind0, ind1, desc);
    }
#if XE4
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const GRFRange &src1, uint64_t desc, SourceLocation loc = {}) {
        sendg(mod, sf, dst, src0, RegisterRange(src1), desc);
    }
    void sendg(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const GRFRange &src1, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        sendg(mod, sf, dst, src0, RegisterRange(src1), ind0, desc);
    }
#endif
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, NullRegister(), NoOperand(), NoOperand(), desc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, NullRegister(), ind0, NoOperand(), desc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, NullRegister(), ind0, ind1, desc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegisterRange &src1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src1, NoOperand(), NoOperand(), desc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegisterRange &src1, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src1, ind0, NoOperand(), desc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegisterRange &src1, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src1, ind0, ind1, desc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src0Len, NoOperand(), NoOperand(), desc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src0Len, ind0, NoOperand(), desc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgc, mod, sf, dst, src0, src0Len, ind0, ind1, desc);
    }
#if XE4
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const GRFRange &src1, uint64_t desc, SourceLocation loc = {}) {
        sendgc(mod, sf, dst, src0, RegisterRange(src1), desc);
    }
    void sendgc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const GRFRange &src1, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        sendgc(mod, sf, dst, src0, RegisterRange(src1), ind0, desc);
    }
#endif
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, NullRegister(), NoOperand(), NoOperand(), desc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        if (ind0.isNull())
            opSendg(Opcode::sendgx, mod, sf, dst, src0, NullRegister(), NoOperand(), NoOperand(), desc);
        else
            opSendg(Opcode::sendgx, mod, sf, dst, src0, NullRegister(), ind0, NoOperand(), desc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, NullRegister(), ind0, ind1, desc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegisterRange &src1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src1, NoOperand(), NoOperand(), desc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegisterRange &src1, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        if (ind0.isNull())
            opSendg(Opcode::sendgx, mod, sf, dst, src0, src1, NoOperand(), NoOperand(), desc);
        else
            opSendg(Opcode::sendgx, mod, sf, dst, src0, src1, ind0, NoOperand(), desc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegisterRange &src1, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src1, ind0, ind1, desc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src0Len, NoOperand(), NoOperand(), desc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src0Len, ind0, NoOperand(), desc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgx, mod, sf, dst, src0, src0Len, ind0, ind1, desc);
    }
#if XE4
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const GRFRange &src1, uint64_t desc, SourceLocation loc = {}) {
        sendgx(mod, sf, dst, src0, RegisterRange(src1), desc);
    }
    void sendgx(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const GRFRange &src1, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        sendgx(mod, sf, dst, src0, RegisterRange(src1), ind0, desc);
    }
#endif
    void sendgxc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgxc, mod, sf, dst, src0, NullRegister(), NoOperand(), NoOperand(), desc);
    }
    void sendgxc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        if (ind0.isNull())
            opSendg(Opcode::sendgxc, mod, sf, dst, src0, NullRegister(), NoOperand(), NoOperand(), desc);
        else
            opSendg(Opcode::sendgxc, mod, sf, dst, src0, NullRegister(), ind0, NoOperand(), desc);
    }
    void sendgxc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgxc, mod, sf, dst, src0, NullRegister(), ind0, ind1, desc);
    }
    void sendgxc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegisterRange &src1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgxc, mod, sf, dst, src0, src1, NoOperand(), NoOperand(), desc);
    }
    void sendgxc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegisterRange &src1, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        if (ind0.isNull())
            opSendg(Opcode::sendgxc, mod, sf, dst, src0, src1, NoOperand(), NoOperand(), desc);
        else
            opSendg(Opcode::sendgxc, mod, sf, dst, src0, src1, ind0, NoOperand(), desc);
    }
    void sendgxc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const RegisterRange &src1, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgxc, mod, sf, dst, src0, src1, ind0, ind1, desc);
    }
    void sendgxc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgxc, mod, sf, dst, src0, src0Len, NoOperand(), NoOperand(), desc);
    }
    void sendgxc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgxc, mod, sf, dst, src0, src0Len, ind0, NoOperand(), desc);
    }
    void sendgxc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegData &src0, int src0Len, const RegData &ind0, const RegData &ind1, uint64_t desc, SourceLocation loc = {}) {
        opSendg(Opcode::sendgxc, mod, sf, dst, src0, src0Len, ind0, ind1, desc);
    }
#if XE4
    void sendgxc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const GRFRange &src1, uint64_t desc, SourceLocation loc = {}) {
        sendgxc(mod, sf, dst, src0, RegisterRange(src1), desc);
    }
    void sendgxc(const InstructionModifier &mod, SharedFunction sf, const RegData &dst, const RegisterRange &src0, const GRFRange &src1, const RegData &ind0, uint64_t desc, SourceLocation loc = {}) {
        sendgxc(mod, sf, dst, src0, RegisterRange(src1), ind0, desc);
    }
#endif
#endif
#if XE4
    template <typename DT = uint32_t>
    void shfld(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &lanemask, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::shfld, getDataType<DT>(), mod, dst, src0, src1, lanemask);
    }
    template <typename DT = uint32_t>
    void shfld(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &lanemask, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::shfld, getDataType<DT>(), mod, dst, src0, src1, lanemask);
    }
    template <typename DT = uint32_t>
    void shfld(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::shfld, getDataType<DT>(), mod, dst, src0, src1, lanemask);
    }
    template <typename DT = uint32_t>
    void shfld(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::shfld, getDataType<DT>(), mod, dst, src0, src1, lanemask);
    }
    template <typename DT = uint32_t>
    void shfli(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &lanemask, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::shfli, getDataType<DT>(), mod, dst, src0, src1, lanemask);
    }
    template <typename DT = uint32_t>
    void shfli(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &lanemask, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::shfli, getDataType<DT>(), mod, dst, src0, src1, lanemask);
    }
    template <typename DT = uint32_t>
    void shfli(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::shfli, getDataType<DT>(), mod, dst, src0, src1, lanemask);
    }
    template <typename DT = uint32_t>
    void shfli(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::shfli, getDataType<DT>(), mod, dst, src0, src1, lanemask);
    }
    // TODO: shflsb
    template <typename DT = uint32_t>
    void shflu(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &lanemask, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::shflu, getDataType<DT>(), mod, dst, src0, src1, lanemask);
    }
    template <typename DT = uint32_t>
    void shflu(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &lanemask, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::shflu, getDataType<DT>(), mod, dst, src0, src1, lanemask);
    }
    template <typename DT = uint32_t>
    void shflu(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::shflu, getDataType<DT>(), mod, dst, src0, src1, lanemask);
    }
    template <typename DT = uint32_t>
    void shflu(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::shflu, getDataType<DT>(), mod, dst, src0, src1, lanemask);
    }
    template <typename DT = uint32_t>
    void shflx(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &lanemask, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::shflx, getDataType<DT>(), mod, dst, src0, src1, lanemask);
    }
    template <typename DT = uint32_t>
    void shflx(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &lanemask, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::shflx, getDataType<DT>(), mod, dst, src0, src1, lanemask);
    }
    template <typename DT = uint32_t>
    void shflx(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::shflx, getDataType<DT>(), mod, dst, src0, src1, lanemask);
    }
    template <typename DT = uint32_t>
    void shflx(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, uint32_t lanemask = 0xFFFFFFFF, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::shflx, getDataType<DT>(), mod, dst, src0, src1, lanemask);
    }
#endif
    template <typename DT = void>
    void shl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::shl, getDataType<DT>(), mod, dst, src0, src1);
        else
#endif
        opX(isGen12 ? Opcode::shl_gen12 : Opcode::shl, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void shl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::shl, getDataType<DT>(), mod, dst, src0, src1);
        else
#endif
        opX(isGen12 ? Opcode::shl_gen12 : Opcode::shl, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void shr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::shr, getDataType<DT>(), mod, dst, src0, src1);
        else
#endif
        opX(isGen12 ? Opcode::shr_gen12 : Opcode::shr, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void shr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::shr, getDataType<DT>(), mod, dst, src0, src1);
        else
#endif
        opX(isGen12 ? Opcode::shr_gen12 : Opcode::shr, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void smov(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(isGen12 ? Opcode::smov_gen12 : Opcode::smov, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void srnd(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(Opcode::srnd, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void srnd(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(Opcode::srnd, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void subb(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::subb, getDataType<DT>(), mod, dst, src0, src1);
        else
#endif
        opX(Opcode::subb, getDataType<DT>(), (hardware >= HW::XeHPC) ? mod : (mod | AccWrEn), dst, src0, src1);
    }
    template <typename DT = void>
    void subb(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            opX(OpcodeClassXe4::subb, getDataType<DT>(), mod, dst, src0, src1);
        else
#endif
        opX(Opcode::subb, getDataType<DT>(), (hardware >= HW::XeHPC) ? mod : (mod | AccWrEn), dst, src0, src1);
    }
#if XE4
    template <typename DT = void>
    void subb(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const FlagRegister &carryIn, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::subb, getDataType<DT>(), mod, dst, src0, src1, carryIn);
    }
    template <typename DT = void>
    void subb(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const FlagRegister &carryIn, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::subb, getDataType<DT>(), mod, dst, src0, src1, carryIn);
    }
    template <typename DT = void>
    void subb(const InstructionModifier &mod, const RegData &dst, const FlagRegister &carryOut, const RegData &src0, const RegData &src1, const FlagRegister &carryIn = FlagRegister(), SourceLocation loc = {}) {
        if (carryIn.isValid())
            opX(OpcodeClassXe4::subb, getDataType<DT>(), mod | eo | carryOut, dst, src0, src1, carryIn);
        else
            opX(OpcodeClassXe4::subb, getDataType<DT>(), mod | eo | carryOut, dst, src0, src1);
    }
    template <typename DT = void>
    void subb(const InstructionModifier &mod, const RegData &dst, const FlagRegister &carryOut, const RegData &src0, const Immediate &src1, const FlagRegister &carryIn = FlagRegister(), SourceLocation loc = {}) {
        if (carryIn.isValid())
            opX(OpcodeClassXe4::subb, getDataType<DT>(), mod | eo | carryOut, dst, src0, src1, carryIn);
        else
            opX(OpcodeClassXe4::subb, getDataType<DT>(), mod | eo | carryOut, dst, src0, src1);
    }
    template <typename DT = void>
    void subb(const RegData &dst, const RegData &src0, const RegData &src1, const FlagRegister &carryIn = FlagRegister(), SourceLocation loc = {}) {
        if (carryIn.isValid())
            subb<DT>(InstructionModifier(), dst, src0, src1, carryIn);
        else
            subb<DT>(InstructionModifier(), dst, src0, src1);
    }
    template <typename DT = void>
    void subb(const RegData &dst, const RegData &src0, const Immediate &src1, const FlagRegister &carryIn = FlagRegister(), SourceLocation loc = {}) {
        if (carryIn.isValid())
            subb<DT>(InstructionModifier(), dst, src0, src1, carryIn);
        else
            subb<DT>(InstructionModifier(), dst, src0, src1);
    }
    void tarb(const InstructionModifier &mod = InstructionModifier(), SourceLocation loc = {}) {
        opX(OpcodeClassXe4::tarb, DataType::invalid, mod, NoOperand(), NoOperand());
    }
#endif
    void wait(const InstructionModifier &mod, const RegData &nreg, SourceLocation loc = {}) {
        opX(Opcode::wait, mod, NoOperand(), nreg);
    }
    void while_(const InstructionModifier &mod, Label &jip, SourceLocation loc = {}) {
        (void) jip.getID(labelManager);
        opX(Opcode::while_, mod, jip);
    }
    template <typename DT = void>
    void xor_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            bfn2<DT>(mod, 0x66, dst, src0, src1);
        else
#endif
        opX(isGen12 ? Opcode::xor_gen12 : Opcode::xor_, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void xor_(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
#if XE4
        if (hardware >= HW::Xe4)
            bfn2<DT>(mod, 0x66, dst, src0, src1);
        else
#endif
        opX(isGen12 ? Opcode::xor_gen12 : Opcode::xor_, getDataType<DT>(), mod, dst, src0, src1);
    }
#if XE4
    void yield(const InstructionModifier &mod = InstructionModifier(), SourceLocation loc = {}) {
        opX(OpcodeClassXe4::yield, DataType::invalid, mod, NoOperand(), NoOperand());
    }
#endif

#if XE3P
    template <typename DT = void>
    void bdpas(const InstructionModifier &mod, uint8_t sdepth, uint8_t rcount, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, const RegData &src3, const RegData &src4, SourceLocation loc = {}) {
        auto emod = mod | defaultModifier;
        if (emod.isAutoSWSB()) {
            if (!src3.isARF()) wrdep(GRF(src3.getBase()));
            if (!src4.isARF()) wrdep(GRF(src4.getBase()));
        }
        opBdpas(Opcode::bdpas, getDataType<DT>(), mod, sdepth, rcount, dst, src0, src1, src2, src3, src4);
    }

    template <typename DT = void>
    void dnscl(const InstructionModifier &mod, uint8_t mode, RoundingType rnd, RegData dst, RegData src0, RegData src1, const RegData &src2, SourceLocation loc = {}) {
        auto ctrl = encodeDnsclCtrl(mode, rnd, dst, src0, src1);
        opX(Opcode::dnscl, getDataType<DT>(), mod, dst, src0, src1, src2, ctrl);
    }

private:
    struct LFSR {
        AsmCodeGenerator &parent;

        LFSR(AsmCodeGenerator *parent_) : parent(*parent_) {}

        void operator()(LFSRFunction fc, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
            parent.opX(Opcode::lfsr, DataType::invalid, mod, dst, src0, src1, NoOperand(), static_cast<uint8_t>(fc));
        }

        template <typename DT = void>
        void b32(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
            parent.opX(Opcode::lfsr, getDataType<DT>(), mod, dst, src0, src1, NoOperand(), static_cast<uint8_t>(LFSRFunction::b32));
        }
        template <typename DT = void>
        void b32(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
            parent.opX(Opcode::lfsr, getDataType<DT>(), mod, dst, src0, src1, NoOperand(), static_cast<uint8_t>(LFSRFunction::b32));
        }
        template <typename DT = void>
        void b16v2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
            parent.opX(Opcode::lfsr, getDataType<DT>(), mod, dst, src0, src1, NoOperand(), static_cast<uint8_t>(LFSRFunction::b16v2));
        }
        template <typename DT = void>
        void b16v2(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
            parent.opX(Opcode::lfsr, getDataType<DT>(), mod, dst, src0, src1, NoOperand(), static_cast<uint8_t>(LFSRFunction::b16v2));
        }
        template <typename DT = void>
        void b8v4(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1) {
            parent.opX(Opcode::lfsr, getDataType<DT>(), mod, dst, src0, src1, NoOperand(), static_cast<uint8_t>(LFSRFunction::b8v4));
        }
        template <typename DT = void>
        void b8v4(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1) {
            parent.opX(Opcode::lfsr, getDataType<DT>(), mod, dst, src0, src1, NoOperand(), static_cast<uint8_t>(LFSRFunction::b8v4));
        }
    };
public:
    LFSR lfsr;

private:
    struct Shfl {
        AsmCodeGenerator &parent;

        Shfl(AsmCodeGenerator *parent_) : parent(*parent_) {}

        void operator()(ShuffleFunction fc, const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
            parent.opX(Opcode::shfl, DataType::invalid, mod, dst, src0, src1, NoOperand(), static_cast<uint8_t>(fc));
        }

        template <typename DT = void>
        void idx4(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
            parent.opX(Opcode::shfl, getDataType<DT>(), mod, dst, src0, src1, NoOperand(), static_cast<uint8_t>(ShuffleFunction::idx4));
        }
    };
public:
    Shfl shfl;
#endif

#if XE4
    /* Scalar instructions */
    template <typename DT = void>
    void sadd(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::sadd_128A, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sadd(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::sadd_128A, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sasr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::sasr, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sasr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::sasr, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sbfn2(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::sbfn2, getDataType<DT>(), mod, dst, src0, src1, NoOperand(), ctrl);
    }
    template <typename DT = void>
    void sbfn2(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::sbfn2, getDataType<DT>(), mod, dst, src0, src1, NoOperand(), ctrl);
    }
    template <typename DT = void>
    void sbfn3(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::bfn2, getDataType<DT>(), mod, dst, src0, src1, src2, ctrl);
    }
    template <typename DT = void>
    void sbfn3(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::bfn2, getDataType<DT>(), mod, dst, src0, src1, src2, ctrl);
    }
    template <typename DT = void>
    void sbfn3(const InstructionModifier &mod, uint8_t ctrl, const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::bfn2, getDataType<DT>(), mod, dst, src0, src1, src2, ctrl);
    }
    template <typename DT = uint32_t>
    void sbfe(const InstructionModifier &mod, unsigned width, unsigned offset, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::sbfe, getDataType<DT>(), mod, dst, src0, NoOperand(), NoOperand(), width | (offset << 8));
    }
    template <typename DT = uint32_t>
    void sbfegen(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::sbfegen, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = uint32_t>
    void sbfi(const InstructionModifier &mod, unsigned width, unsigned offset, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::sbfi, getDataType<DT>(), mod, dst, src0, src1, NoOperand(), width | (offset << 8));
    }
    template <typename DT = uint32_t>
    void sbfi(const InstructionModifier &mod, unsigned width, unsigned offset, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::sbfi, getDataType<DT>(), mod, dst, src0, src1, NoOperand(), width | (offset << 8));
    }
    template <typename DT = uint32_t>
    void sbfia(const InstructionModifier &mod, unsigned width, unsigned offset, IndirectARF dst, IndirectARF src0, const RegData &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::sbfia, getDataType<DT>(), mod, dst, src0, src1, NoOperand(), width | (offset << 8));
    }
    template <typename DT = uint32_t>
    void sbfia(const InstructionModifier &mod, unsigned width, unsigned offset, IndirectARF dst, IndirectARF src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::sbfia, getDataType<DT>(), mod, dst, src0, src1, NoOperand(), width | (offset << 8));
    }
    template <typename DT = uint32_t>
    void sbfigen(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::sbfigen, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = uint32_t>
    void sbfigen(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::sbfigen, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void sbfrev(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::sbfrev, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void sbrepgen(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::sbrepgen, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sbrepgen(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::sbrepgen, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void scmp(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::scmp_128S, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void scmp(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::scmp_128S, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sfbh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::sfbh, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void sfbl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::sfbl, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = uint32_t>
    void sgeta(const InstructionModifier &mod, const RegData &dst, IndirectARF src0, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::sgeta, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void smad(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::smad_128A, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void smad(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::smad_128A, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void smad(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const RegData &src2, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::smad_128A, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void smad(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, const Immediate &src2, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::smad_128A, getDataType<DT>(), mod, dst, src0, src1, src2);
    }
    template <typename DT = void>
    void smov(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::smov_128R, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void smov(const InstructionModifier &mod, const RegData &dst, const Immediate &src0, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::smov_128R, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = uint32_t>
    void smsk(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::smsk, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = uint32_t>
    void smsk(const InstructionModifier &mod, const RegData &dst, const RegData &src0, uint32_t src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::smsk, getDataType<DT>(), mod, dst, src0, Immediate::ud(src1));
    }
    template <typename DT = uint32_t>
    void smsk(const InstructionModifier &mod, const RegData &dst, uint32_t src0, const RegData &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::smsk, getDataType<DT>(), mod, dst, Immediate::ud(src0), src1);
    }
    template <typename DT = uint32_t>
    void smsk(const InstructionModifier &mod, const RegData &dst, uint32_t src0, uint32_t src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::smsk,getDataType<DT>(),  mod, dst, Immediate::ud(src0), Immediate::ud(src1));
    }
    template <typename DT = void>
    void smul(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::smul_128A, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void smul(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::smul_128A, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void smullh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opMullh(OpcodeClassXe4::smullh, getDataType<DT>(), mod, dst, src0, src1, NoOperand());
    }
    template <typename DT = void>
    void smullh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opMullh(OpcodeClassXe4::smullh, getDataType<DT>(), mod, dst, src0, src1, NoOperand());
    }
    template <typename DT = uint32_t>
    void ssel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, FlagRegister predicate, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::ssel, getDataType<DT>(), mod, dst, src0, src1, predicate);
    }
    template <typename DT = uint32_t>
    void ssel(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, FlagRegister predicate, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::ssel, getDataType<DT>(), mod, dst, src0, src1, predicate);
    }
    template <typename DT = uint32_t>
    void sseta(const InstructionModifier &mod, IndirectARF dst, const RegData &src0, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::sseta, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = uint32_t>
    void sseta(const InstructionModifier &mod, IndirectARF dst, const Immediate &src0, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::sseta, getDataType<DT>(), mod, dst, src0);
    }
    template <typename DT = void>
    void sshl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::sshl, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sshl(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::sshl, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sshr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::sshr, getDataType<DT>(), mod, dst, src0, src1);
    }
    template <typename DT = void>
    void sshr(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
        opX(OpcodeClassXe4::sshr, getDataType<DT>(), mod, dst, src0, src1);
    }
#endif

private:
    struct Sync {
        AsmCodeGenerator &parent;

        Sync(AsmCodeGenerator *parent_) : parent(*parent_) {}

        void operator()(SyncFunction fc, const InstructionModifier &mod = InstructionModifier()) {
#if XE4
            if (parent.hardware >= HW::Xe4)
                parent.opSync(fc, mod, NoOperand());
            else
#endif
            parent.opSync(fc, mod, null);
        }
        void operator()(SyncFunction fc, const RegData &src0) {
            this->operator()(fc, InstructionModifier(), src0);
        }
        void operator()(SyncFunction fc, const InstructionModifier &mod, const RegData &src0) {
            parent.opSync(fc, mod, src0);
        }
        void operator()(SyncFunction fc, int src0) {
            this->operator()(fc, InstructionModifier(), src0);
        }
        void operator()(SyncFunction fc, const InstructionModifier &mod, int src0) {
            parent.opSync(fc, mod, Immediate::ud(src0));
        }
        void allrd() {
            allrd(null);
        }
        void allrd(const InstructionModifier &mod) {
            allrd(mod, null);
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
        void bar(const InstructionModifier &mod = InstructionModifier(), SourceLocation loc = {}) {
            this->operator()(SyncFunction::bar, mod);
        }
        void bar(const InstructionModifier &mod, uint32_t src0, SourceLocation loc = {}) {
            this->operator()(SyncFunction::bar, mod, src0);
        }
        void bar(const InstructionModifier &mod, const RegData &src0, SourceLocation loc = {}) {
#if XE4
            this->operator()((parent.hardware >= HW::Xe4) ? SyncFunction::barsrc : SyncFunction::bar, mod, src0);
#else
            this->operator()(SyncFunction::bar, mod, src0);
#endif
        }
        void bar(uint32_t src0, SourceLocation loc = {}) {
            this->operator()(SyncFunction::bar, InstructionModifier(), src0);
        }
        void bar(const RegData &src0, SourceLocation loc = {}) {
#if XE4
            this->operator()((parent.hardware >= HW::Xe4) ? SyncFunction::barsrc : SyncFunction::bar, InstructionModifier(), src0);
#else
            this->operator()(SyncFunction::bar, InstructionModifier(), src0);
#endif
        }
        void flush(const InstructionModifier &mod = InstructionModifier()) {
            this->operator()(SyncFunction::flush, InstructionModifier(), null);
        }
        void host(const InstructionModifier &mod = InstructionModifier()) {
            this->operator()(SyncFunction::host, mod);
        }
        void nop(const InstructionModifier &mod = InstructionModifier()) {
            this->operator()(SyncFunction::nop, mod);
        }
#if XE4
        void barflush(const InstructionModifier &mod = InstructionModifier()) {
            flush();
        }
        void barid(uint32_t src0, SourceLocation loc = {})                                         { bar(src0); }
        void barid(const InstructionModifier &mod, uint32_t src0, SourceLocation loc = {})         { bar(mod, src0); }
        void barsrc(const RegData &src0, SourceLocation loc = {})                                  { bar(src0); }
        void barsrc(const InstructionModifier &mod, const RegData &src0, SourceLocation loc = {})  { bar(src0); }
        void dstmsk(uint32_t src0, SourceLocation loc = {})                                        { allwr(src0); }
        void dstmsk(const InstructionModifier &mod, uint32_t src0, SourceLocation loc = {})        { allwr(src0); }
        void srcmsk(uint32_t src0, SourceLocation loc = {})                                        { allrd(src0); }
        void srcmsk(const InstructionModifier &mod, uint32_t src0, SourceLocation loc = {})        { allrd(src0); }
        void none(const InstructionModifier &mod = InstructionModifier(), SourceLocation loc = {}) {
            nop(mod);
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
            this->operator()(SyncFunction::nop, swsb4, SWSBItem::pack4({swsb0, swsb1, swsb2, swsb3}));
        }
#endif
    };
public:
    Sync sync;

    void ignoredep(Operand op) {
        if (hardware >= HW::Gen12LP)
            opDirective(static_cast<Directive>(op), NullRegister());
    }
    void subdep(Operand op, const GRFRange &r) {
        if (op == Operand::dst) {
#ifdef NGEN_SAFE
            if (r.getLen() > 32) throw invalid_directive_exception();
#endif
            opDirective(Directive::subdep_dst, r);
        } else {
            ignoredep(op);
            wrdep(r);
        }
    }
    void subdep(Operand op, const GRF &r) {
        subdep(op, r-r);
    }
    void wrdep(const GRFRange &r, SourceLocation loc = {}) {
#ifdef NGEN_SAFE
        if (hardware < HW::Gen12LP) throw unsupported_instruction();
#endif
        int len = r.getLen();
        for (int o = 0; o < len; o += 32) {
            int thisLen = std::min(len - o, 32);
            opDirective(Directive::wrdep, r[o] - r[o + thisLen - 1]);
        }
    }
    void wrdep(const GRF &r, SourceLocation loc = {}) {
        wrdep(r-r);
    }
    void fencedep(Label &fenceLocation, SourceLocation loc = {}) {
        (void) fenceLocation.getID(labelManager);
        opDirective(Directive::fencedep, fenceLocation);
    }

    void disablePVCWARWA(SourceLocation loc) {
        opDirective(Directive::pvcwarwa, NullRegister());
    }

    void mark(Label &label)            { streamStack.back()->mark(label, labelManager); }
    void markIfUndefined(Label &label) { if (!label.defined(labelManager)) mark(label); }

    using _self = AsmCodeGenerator;

#include "ngen_shortcuts.hpp"
#include "ngen_pseudo.hpp"
#ifndef NGEN_GLOBAL_REGS
#include "ngen_registers.hpp"
#endif
};


void AsmCodeGenerator::unsupported()
{
#ifdef NGEN_SAFE
    throw unsupported_instruction();
#endif
}

AsmCodeGenerator::InstructionStream *AsmCodeGenerator::popStream()
{
#ifdef NGEN_SAFE
    if (streamStack.size() <= 1) throw stream_stack_underflow();
#endif

    InstructionStream *result = streamStack.back();
    streamStack.pop_back();
    return result;
}

void AsmCodeGenerator::finalize()
{
#ifdef NGEN_SAFE
    if (streamStack.size() > 1) throw unfinished_stream_exception();
#endif
    auto &buffer = streamStack.back()->buffer;
    int inum = 0;
    for (auto &i : buffer)
        i.inum = inum++;
}

void AsmCodeGenerator::getCode(std::ostream &out)
{
    finalize();

    autoswsb::BasicBlockList analysis = autoswsb::autoSWSB(hardware, declaredGRFs, streamStack.back()->buffer, cancelAutoSWSB_);
    std::multimap<int32_t, autoswsb::SyncInsertion*> syncs;      // Syncs inserted by auto-SWSB.
    std::multimap<int32_t, autoswsb::DummyMovInsertion*> movs;   // Dummy moves inserted by auto-SWSB.

    for (auto &bb : analysis) {
        for (auto &sync: bb.syncs)
            syncs.insert(std::make_pair(sync.inum, &sync));
        for (auto &mov: bb.movs)
            movs.insert(std::make_pair(mov.inum, &mov));
    }

    auto nextSync = syncs.begin();
    auto nextMov = movs.begin();
    int lineNo = 0;

    for (auto &i : streamStack.back()->buffer) {
        while ((nextSync != syncs.end()) && (nextSync->second->inum == i.inum))
            outX(out, AsmInstruction(hardware, *(nextSync++)->second), lineNo);
        while ((nextMov != movs.end()) && (nextMov->second->inum == i.inum))
            outX(out, *(nextMov++)->second, lineNo);

        if (i.isLabel()) {
            i.dst.label.outputText(out, PrintDetail::full, labelManager);
            out << ':' << std::endl;
            if (i.dst.label == _interfaceLabels.localIDsLoaded)
                lineNo = 0;
        } else if (i.isComment())
            outComment(out, i);
        else if (!isDirective(i.op))
            outX(out, i, lineNo);
    }
}

#define NGEN_ASM_HAS_GET_PARTIAL_CODE
void AsmCodeGenerator::getPartialCode(std::ostream &out)
{
    int nstream = 0;
    int lineNo = 0;
    for (auto it = streamStack.rbegin(); it != streamStack.rend(); it++, nstream++) {
        out << "// Stream " << nstream << std::endl;
        for (auto &i: (*it)->buffer) {
            if (i.isLabel()) {
                i.dst.label.outputText(out, PrintDetail::full, labelManager);
                out << ':' << std::endl;
            } else if (i.isComment())
                outComment(out, i);
            else if (!isDirective(i.op))
                outX(out, i, lineNo);
        }
    }
}

#define NGEN_ASM_HAS_DISABLE_AUTOSWSB
void AsmCodeGenerator::disableAutoSWSB()
{
    for (auto &i: streamStack.back()->buffer)
        i.clearAutoSWSB();
}

template <typename D, typename S0, typename S1, typename S2>
void AsmCodeGenerator::opX(Opcode op, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2, uint16_t ext)
{
#if XE4
    if (hardware >= HW::Xe4) unsupported();
#endif

    bool is2Src = !S1::emptyOp;
    bool is3Src = !S2::emptyOp;
    int arity = 1 + is2Src + is3Src;

    InstructionModifier emod = mod | defaultModifier;
    auto esize = emod.getExecSize();

    if (is3Src && hardware < HW::Gen10)
        esize = std::min<int>(esize, 8);        // WA for IGA Align16 emulation issue

#ifdef NGEN_SAFE
    if (esize > 1 && dst.isScalar() && !std::is_base_of<NoOperand, D>::value)
        throw invalid_execution_size_exception();
#endif

    auto ewidth = getExecWidth({defaultType, dst.getType(), src0.getType(), src1.getType(), src2.getType()});
    dst.fixup(hardware,  esize, ewidth, defaultType, -1, arity);
    src0.fixup(hardware, esize, ewidth, defaultType, 0, arity);
    src1.fixup(hardware, esize, ewidth, defaultType, 1, arity);
    src2.fixup(hardware, esize, ewidth, defaultType, 2, arity);

    streamStack.back()->append(op, ext, emod, &labelManager, dst, src0, src1, src2);
}

static const char *getMnemonic(Opcode op, HW hw)
{
    static const char *names[0x80] = {
        "illegal", "sync", "sel", "movi", "not", "and", "or", "xor",
        "shr", "shl", "smov", "", "asr", "", "ror", "rol",
        "cmp", "cmpn", "csel", "", "", "", "", "bfrev",
        "bfe", "bfi1", "bfi2", "", "", "", "", "",
        "jmpi", "brd", "if", "brc", "else", "endif", "", "while",
        "break", "cont", "halt", "calla", "call", "ret", "goto", "join",
#if XE3P
        "wait", "send", "sendc", "sendg", "sendgc", "sendgx", "sendgxc", "",
        "math", "lfsr", "", "", "", "", "", "",
#else
        "wait", "send", "sendc", "sends", "sendsc", "", "", "",
        "math", "", "", "", "", "", "", "",
#endif
        "add", "mul", "avg", "frc", "rndu", "rndd", "rnde", "rndz",
        "mac", "mach", "lzd", "fbh", "fbl", "cbit", "addc", "subb",
#if XE3P
        "shfl", "sada2", "add3", "macl", "srnd", "dnscl", "dp3", "dp2",
        "dp4a", "dpas", "dpasw", "mad", "bdpas", "madm", "", "mullh",
#else
        "sad2", "sada2", "add3", "macl", "srnd", "dph", "dp3", "dp2",
        "dp4a", "dpas", "dpasw", "mad", "lrp", "madm", "", "",
#endif
        "nop", "mov", "sel", "movi", "not", "and", "or", "xor",
        "shr", "shl", "smov", "bfn", "asr", "", "ror", "rol",
        "cmp", "cmpn", "csel", "", "", "", "", "bfrev",
        "bfe", "bfi1", "bfi2", "", "", "", "nop", ""
    };

#if XE4
    if (isXe4(op)) {
        if (op == Opcode::goto__128B) return "goto";
        int iop = static_cast<int>(op) & 0x3FF;
#define NGEN_XE4_UNTYPED_OP(cls, enc, opcode) if (iop == opcode) return #cls;
#define NGEN_XE4_RAW_OP NGEN_XE4_OP
#define NGEN_XE4_OP(cls, enc, dt, opcode) if (iop == opcode) return #cls;
        NGEN_DEF_XE4_OPS
#undef NGEN_XE4_OP
#undef NGEN_XE4_RAW_OP
#undef NGEN_XE4_UNTYPED_OP
        return "";
    }
#endif

    const char *mnemonic = names[static_cast<int>(op) & 0x7F];

    if (hw < HW::Gen12LP) switch (op) {
#if XE3P
        case Opcode::sends:  mnemonic = "sends";  break;
        case Opcode::sendsc: mnemonic = "sendsc"; break;
        case Opcode::sad2:   mnemonic = "sad2";   break;
        case Opcode::dph:    mnemonic = "dph";    break;
        case Opcode::lrp:    mnemonic = "lrp";    break;
#endif
        case Opcode::mov:    mnemonic = "mov";    break;
        case Opcode::line:   mnemonic = "line";   break;
        case Opcode::pln:    mnemonic = "pln";    break;
        case Opcode::dp4:    mnemonic = "dp4";    break;
        default: break;
    }

    return mnemonic;
}

#if XE4
static inline void validateXe4(NoOperand) {}
static inline void validateXe4(RegisterRange) {}
template <> inline void processTypesXe4(DataType &type, NoOperand &o) {}
template <> inline void processTypesXe4(DataType &type, RegisterRange &rr) {}
static inline bool allowScalarization(RegisterRange) { return false; }

template <typename D, typename S0, typename S1, typename S2>
void AsmCodeGenerator::opX(OpcodeClassXe4 opclass, DataType defaultType, const InstructionModifier &mod, D dst, S0 src0, S1 src1, S2 src2, uint16_t ext)
{
    auto emod = mod | defaultModifier;
    if (allowScalarization(dst) && allowScalarization(src0) && allowScalarization(src1) && allowScalarization(src2))
        opclass = toScalar(opclass);
    validateXe4(emod, opclass);
    (opclass == OpcodeClassXe4::movs) ? validateIndXe4(dst)  : validateXe4(dst);
    (opclass == OpcodeClassXe4::movg) ? validateIndXe4(src0) : validateXe4(src0);
    validateXe4(src1);
    validateXe4(src2);
    processTypesXe4(defaultType, dst, src0, src1, src2);
    validateBaseXe4(dst, src0, src1, src2);
    if (hardware < HW::Xe4) unsupported();
    streamStack.back()->append(opcodeXe4(opclass, defaultType), ext, emod, &labelManager, dst, src0, src1, src2);
}

template <typename S1>
void AsmCodeGenerator::opCvt(OpcodeClassXe4 opclass, DataType defaultType, const InstructionModifier &mod, RegData dst, RegData src0, S1 src1)
{
    auto emod = mod | defaultModifier;
    auto srcType = DataType::invalid;
    validateXe4(emod, opclass);
    validateXe4(dst);
    validateXe4(src0);
    validateXe4(src1);
    processTypesXe4(srcType, src0, src1);
    processTypesXe4(defaultType, dst);
    validateBaseXe4(dst, src0, src1);
    if (hardware < HW::Xe4) unsupported();
    streamStack.back()->append(opcodeXe4(opclass, defaultType), 0, emod, &labelManager, dst, src0, src1);
}

template <typename S1, typename S2>
void AsmCodeGenerator::opDP4A(OpcodeClassXe4 opclass, DataType defaultType, const InstructionModifier &mod, RegData dst, RegData src0, S1 src1, S2 src2)
{
    auto emod = mod | defaultModifier;
    validateXe4(emod, opclass);
    validateXe4(dst);
    validateXe4(src0);
    validateXe4(src1);
    validateXe4(src2);
    processTypesXe4(defaultType, dst, src2);
    validateBaseXe4(dst, src0, src1, src2);
    if (hardware < HW::Xe4) unsupported();
    streamStack.back()->append(opcodeXe4(opclass, defaultType), 0, emod, &labelManager, dst, src0, src1, src2);
}

template <typename S0, typename S1>
void AsmCodeGenerator::opMovb(OpcodeClassXe4 opclass, DataType defaultType, const InstructionModifier &mod, RegData dst, S0 src0, S1 lanemask)
{
    auto emod = mod | defaultModifier;
    validateXe4(emod, opclass);
    validateXe4(dst);
    validateXe4(src0);
    validateXe4(lanemask);
    auto lmType = DataType::b32;
    processTypesXe4(defaultType, dst, src0);
    processTypesXe4(lmType, lanemask);
    validateBaseXe4(dst, src0, lanemask);
    int nr = getDwords(defaultType) * 32;
    if (hardware < HW::Xe4) unsupported();

    AsmOperand odst{dst}, osrc0{src0};
    for (auto *op: {&odst, &osrc0})
        if (op->type == AsmOperand::Type::reg && op->reg.isSRF())
            *op = RegisterRange(op->reg, nr);

    streamStack.back()->append(opcodeXe4(opclass, defaultType), 0, emod, &labelManager, odst, osrc0, lanemask);
}

template <typename S0, typename S1, typename S2>
void AsmCodeGenerator::opMullh(OpcodeClassXe4 opclass, DataType defaultType, const InstructionModifier &mod, RegData dst, S0 src0, S1 src1, S2 src2)
{
    auto emod = mod | defaultModifier;
    if (dst.isScalar() && src0.isScalar() && src1.isScalar() && src2.isScalar())
        opclass = toScalar(opclass);
    validateXe4(emod, opclass);
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
    if (hardware < HW::Xe4) unsupported();
    streamStack.back()->append(opcodeXe4(opclass, defaultType), 0, emod, &labelManager, dst, src0, src1, src2);
}

#endif

void AsmCodeGenerator::outComment(std::ostream &out, const AsmInstruction &i)
{
    bool newLine = true;
    for (auto c: i.comment) {
        if (newLine) out << "// ";
        out << c;
        newLine = (c == '\n');
    }
    if (!newLine) out << '\n';
}

void AsmCodeGenerator::outX(std::ostream &out, const AsmInstruction &i, int &lineNo)
{
#if XE4
    if (hardware >= HW::Xe4) return outXe4(out, i, lineNo);
#endif

    bool ternary = (i.src[2].type != AsmOperand::Type::none);
    PrintDetail ddst = PrintDetail::hs;
    PrintDetail dsrc01 = ternary ? PrintDetail::vs_hs : PrintDetail::full;
    PrintDetail dsrc[5] = {dsrc01, dsrc01, PrintDetail::hs, PrintDetail::base, PrintDetail::base};

    switch (i.op) {
        case Opcode::send:
        case Opcode::sends:
        case Opcode::sendc:
        case Opcode::sendsc:
#if XE3P
        case Opcode::sendgx:
        case Opcode::sendgxc:
#endif
            ddst = dsrc[0] = dsrc[1] = PrintDetail::base;
            dsrc[2] = dsrc[3] = PrintDetail::sub_no_type;
            break;
        case Opcode::brc:
        case Opcode::brd:
        case Opcode::call:
        case Opcode::calla:
            ddst = PrintDetail::sub;
            dsrc[0] = PrintDetail::sub_no_type;
            break;
        case Opcode::jmpi:
        case Opcode::ret:
            dsrc[0] = PrintDetail::sub_no_type;
            break;
#if XE3P
        case Opcode::bdpas:
            if (isGen12) dsrc[3] = dsrc[4] = PrintDetail::sub;
            /* fall through */
#endif
        case Opcode::dpas:
        case Opcode::dpasw:
            if (isGen12) ddst = dsrc[0] = dsrc[1] = dsrc[2] = PrintDetail::sub;
            break;
        case Opcode::sync:
            if (isGen12) {
                if (i.src[0].type == AsmOperand::Type::reg)
                    dsrc[0] = PrintDetail::sub;
                else
                    dsrc[0] = PrintDetail::sub_no_type;
            }
            break;
        default: break;
    }

    outMods(out, i.mod, i.op, ModPlacementType::Pre);

    out << getMnemonic(i.op, hardware);
    outExt(out, i);
    out << '\t';

    outMods(out, i.mod, i.op, ModPlacementType::Mid);

    i.dst.outputText(out, ddst, labelManager); out << '\t';
    for (int n = 0; n <= 4; n++) {
        i.src[n].outputText(out, dsrc[n], labelManager);
        bool showLen = false;
        if (i.ext & 0x80) {
            showLen |= (n == 1 && (i.op == Opcode::send || i.op == Opcode::sendc) && hardware >= HW::XeHPG);
#if XE3P
            showLen |= (n == 0 && (i.op == Opcode::sendg || i.op == Opcode::sendgc || i.op == Opcode::sendgx || i.op == Opcode::sendgxc));
#endif
        }

        if (showLen)
            out << ':' << (i.ext >> 8);
        out << '\t';
    }

    outMods(out, i.mod, i.op, ModPlacementType::Post);
    if (lineNumbers) {
        out << "\t// " << lineNo * 2;
        lineNo++;
    }
    out << std::endl;
}

#if XE4
inline bool hasSourceFormat(Opcode op)
{
    switch (op) {
        case Opcode::cmp_128S_b32:
        case Opcode::scmp_128S_b32:
        case Opcode::mulmx_128U_f32: return true;
        default:                     return isCvt(op);
    }
}

void AsmCodeGenerator::outXe4(std::ostream &out, const AsmInstruction &i, int &lineNo)
{
    bool withCMod = (i.mod.getCMod() != ConditionModifier::none);

    std::array<PrintDetail, 5> dsrc;
    dsrc.fill(PrintDetail::xe4);

    if (hasSourceFormat(i.op))
        dsrc[0] = PrintDetail::xe4_type;
    else if (i.op == Opcode::dp4a_128Q_u32 || i.op == Opcode::dp4a_128Q_s32)
        dsrc[0] = dsrc[1] = PrintDetail::xe4_type;
    else if (i.op == Opcode::sync_64E && static_cast<SyncFunction>(i.ext) == SyncFunction::none)
        dsrc[0] = PrintDetail::xe4_hide;

    auto haveDst = !i.dst.isEmptyOrNull();
    if (haveDst || withCMod) {
        if (haveDst)
            i.dst.outputText(out, PrintDetail::xe4_dst, labelManager);
        if (withCMod) {
            if (haveDst) out << ", ";
            out << 'p' << i.mod.getCFlag().getARFBase();
        }
        out << " = ";
    }

    outMods(out, i.mod, i.op, ModPlacementType::PreXe4);

    auto mnemonic = getMnemonic(i.op, hardware);
    out << mnemonic;

    outExt(out, i);

    auto type = dstDataType(i.op);
    if (type != DataType::invalid) {
        out << '.';
        if (isRaw(i.op))
            out << DataTypeRawXe4{type};
        else
            out << DataTypeXe4{type};
    }

    outMods(out, i.mod, i.op, ModPlacementType::MidXe4, i.ext);
    if (i.dst.type == AsmOperand::Type::ereg)
        out << "::mme" << int(i.dst.ereg.getMMENum());

    out << ' ';

    if (!outSpecialOps(out, i)) for (int n = 0; n <= 4; n++) if (i.src[n]) {
        if (n > 0) out << ", ";
        i.src[n].outputText(out, dsrc[n], labelManager);
    }

    uint32_t ext2 = static_cast<uint64_t>(i.src[0].imm);    /* sync.nop extra SWSB */
    outMods(out, i.mod, i.op, ModPlacementType::PostXe4, i.ext, ext2);

    if (lineNumbers) {
        out << "\t// " << lineNo;
        lineNo += isEncoding64(getEncodingXe4(i.op)) ? 1 : 2;
    }
    out << std::endl;
}
#endif

void AsmCodeGenerator::outExt(std::ostream &out, const AsmInstruction &i)
{
    switch (i.opcode()) {
        case Opcode::else_:
        case Opcode::goto_:
        case Opcode::if_:       if (i.ext) out << ".b";                         break;
        case Opcode::math:      out << '.' << static_cast<MathFunction>(i.ext); break;
        default: break;
    }

    if (isGen12) switch (i.opcode()) {
#if XE3P
        case Opcode::sendgx:
        case Opcode::sendgxc:
#endif
        case Opcode::send:
        case Opcode::sendc:
        case Opcode::sends:
        case Opcode::sendsc:    out << '.' << getMnemonic(static_cast<SharedFunction>(i.ext & 0xF), hardware); break;
        case Opcode::sync:      out << '.' << static_cast<SyncFunction>(i.ext);                                break;
        case Opcode::bfn:       out << ".0x" << std::hex << i.ext << std::dec;                                 break;
#if XE3P
        case Opcode::dnscl: {
            const char *sts[2] = {"hf", "bf"};
            const char *dts[4] = {"e3m0", "e2m1", "int4", ""};
            const char *rts[2] = {"srnd", "rne"};
            int dt = i.ext & 0x3, mode = i.ext >> 4;
            bool st = i.ext & 0x4, rt = i.ext & 0x8;
            out << '.' << sts[st] << "to" << dts[dt] << ".mode" << mode << '.' << rts[rt];
            break;
        }
        case Opcode::lfsr:      out << '.' << static_cast<LFSRFunction>(i.ext);                                break;
        case Opcode::shfl:      out << '.' << static_cast<ShuffleFunction>(i.ext);                             break;
        case Opcode::bdpas:
#endif
        case Opcode::dpas:
        case Opcode::dpasw: {
            int sdepth = i.ext >> 8;
            int rcount = i.ext & 0xFF;
            out << '.' << sdepth << 'x' << rcount;
        }
        default: break;
    }

#if XE4
    if (hardware >= HW::Xe4) switch (i.opcode()) {
        case Opcode::bfe_128G_b32:
        case Opcode::bfi_128G_b32:
        case Opcode::sbfia_128G_b32: {
            unsigned width = i.ext & 0xFF, offset = i.ext >> 8;
            out << ".(" << width << ',' << offset << ')';
            break;
        }
        case Opcode::bfn2_64D_b32:
        case Opcode::bfn3_128E_b32:
        case Opcode::sbfn2_64D_b32:
        case Opcode::sbfn3_128E_b32:
            if (i.ext == 0x55)      out << ".(~s0)";
            else if (i.ext == 0xEE) out << ".(s0|s1)";
            else if (i.ext == 0x88) out << ".(s0&s1)";
            else if (i.ext == 0x66) out << ".(s0^s1)";
            else
                out << ".(0x" << std::hex << i.ext << std::dec << ')';
            break;
        case Opcode::sync_64E:
            out << '.' << SyncFunctionXe4{static_cast<SyncFunction>(i.ext)};
            break;
        case Opcode::send_128C:
        case Opcode::sendc_128C:
        case Opcode::sendg_128C:
        case Opcode::sendcg_128C:
            outExtSendXe4(out, i);
            break;
        default: break;
    }
#endif
}

static const char *toText(PredCtrl ctrl, bool align16) {
    const char *names[2][16] = {{"", "", "anyv", "allv", "any2h", "all2h", "any4h", "all4h", "any8h", "all8h", "any16h", "all16h", "any32h", "all32h", "any", "all"},
                                {"", "", "x",    "y",    "z",     "w",     "",      "",      "",      "",      "",       "",       "",       "",       "",    ""}};
    return names[align16][static_cast<int>(ctrl) & 0xF];
}

void AsmCodeGenerator::outMods(std::ostream &out, const InstructionModifier &mod, Opcode op, AsmCodeGenerator::ModPlacementType location, uint16_t ext, uint32_t ext2)
{
    ConditionModifier cmod = mod.getCMod();
    PredCtrl ctrl = mod.getPredCtrl();
    bool wrEn = mod.isWrEn();
    bool havePred = (ctrl != PredCtrl::None) && (cmod != ConditionModifier::eo);

    bool havePostMod = false;
    auto startPostMod = [&](bool space = false) {
        out << (havePostMod ? "," :
                      space ? " {" : "{");
        havePostMod = true;
    };
    auto printPostMod = [&](const char *name) {
        startPostMod(); out << name;
    };

    switch (location) {
        case ModPlacementType::Pre:
            if (wrEn || havePred) {
                out << '(';
                if (wrEn) {
                    out << 'W';
                    if (havePred) out << '&';
                }
                if (havePred) {
                    if (mod.isPredInv()) out << '~';
                    mod.getFlagReg().outputText(out, PrintDetail::sub_no_type, labelManager);
                    if (ctrl != PredCtrl::Normal)
                        out << '.' << toText(ctrl, mod.isAlign16());
                }
                out << ')';
            }
            out << '\t';
            break;
        case ModPlacementType::Mid:
            out << '(' << mod.getExecSize() << "|M" << mod.getChannelOffset() << ')' << '\t';

            if (cmod != ConditionModifier::none) {
                out << '(' << cmod << ')';
                mod.getFlagReg().outputText(out, PrintDetail::sub_no_type, labelManager);
                out << '\t';
            }

            if (mod.isSaturate()) out << "(sat)";
            break;
        case ModPlacementType::Post:
        {
            auto swsb = mod.getSWSB();
            if (swsb[0] && swsb[1].isToken())
                std::swap(swsb[0], swsb[1]);
            for (auto item: swsb) {
                if (item.empty()) continue;
                startPostMod();
                if (item.isNoAccSBSet())
                    out << "NoAccSBSet";
                else if (item.isToken()) {
                    out << '$' << item.getToken();
                    if (item.token.src && !item.token.dst) out << ".src";
                    if (item.token.dst && !item.token.src) out << ".dst";
                } else {
                    if (hardware > HW::Gen12LP) {
                        if ((op == Opcode::send || op == Opcode::sendc) && item.getPipe() == Pipe::Default)
                            out << Pipe::A;
                        else
                            out << item.getPipe();
                    }
                    out << '@' << int(item.pipe.dist);
                }
            }

            if (mod.isAlign16())                                          printPostMod("Align16");
            if (mod.isNoDDClr())                                          printPostMod("NoDDClr");
            if (mod.isNoDDChk())                                          printPostMod("NoDDChk");
            if (mod.getThreadCtrl() == ThreadCtrl::Atomic)                printPostMod("Atomic");
            if (!isGen12 && mod.getThreadCtrl() == ThreadCtrl::Switch)    printPostMod("Switch");
            if (!isGen12 && mod.getThreadCtrl() == ThreadCtrl::NoPreempt) printPostMod("NoPreempt");
            if (mod.isAccWrEn() && hardware < HW::XeHPC)                  printPostMod("AccWrEn");
#if XE3P
            if (mod.isFwd() && hardware >= HW::XeHPC)                     printPostMod("Fwd");
#endif
            if (mod.isCompact())                                          printPostMod("Compact");
            if (mod.isBreakpoint())                                       printPostMod("Breakpoint");
            if (mod.isSerialized())                                       printPostMod("Serialize");
            if (mod.isEOT())                                              printPostMod("EOT");
            if (mod.isExBSO())                                            printPostMod("ExBSO");

            if (havePostMod) out << '}';
            break;
        }
#if XE4
        case ModPlacementType::PreXe4:
            if (wrEn || havePred) {
                out << '(';
                if (wrEn)
                    out << 'W';
                if (havePred) {
                    if (wrEn) out << '&';
                    if (mod.isPredInv()) out << '!';
                    out << 'p' << mod.getFlagReg().getARFBase();
                }
                out << ") ";
            }
            break;
        case ModPlacementType::MidXe4: {
            auto rmod = mod.getRounding();
            bool first = true;
            auto separate = [&] {
                out << (first ? "::" : ".");
                first = false;
            };
            switch (op) {
                case Opcode::brd_128B:
                case Opcode::goto__128B:
                    if (mod.getBranchCtrl()) { separate(); out << 'b'; }
                    break;
                case Opcode::cnvg_128L:
                    if (ext & 0x200) { separate(); out << (ext & 0xFF); }
                    if (ext & 0x100) { separate(); out << "exp"; }
                    break;
                default: break;
            }
            if (mod.isSaturate()) { separate(); out << (isCvt(op) ? "clmp" : "sat"); }
            if (rmod != RoundingOverride::none) {
                separate(); out << rmod;
            }
            if (cmod != ConditionModifier::none && cmod != ConditionModifier::eo) {
                separate(); out << ConditionModifierXe4{cmod};
            }
            break;
        }
        case ModPlacementType::PostXe4: {
            auto swsb = mod.getSWSB();
            std::array<SWSBItem, 4> extSWSB = {};
            if (op == Opcode::sync_64E && static_cast<SyncFunction>(ext) == SyncFunction::nop)
            extSWSB = SWSBItem::unpack4(ext2);
            std::array<SWSBItem, 6> allSWSB = {extSWSB[0], extSWSB[1], extSWSB[2], extSWSB[3], swsb[0], swsb[1]};

            for (auto item: allSWSB) {
                if (item.empty()) continue;
                startPostMod(true);
                if (item.isToken()) {
                    out << '$' << item.getToken();
                    if (item.token.src && !item.token.dst) out << ".src";
                    if (item.token.dst && !item.token.src) out << ".dst";
                } else {
                    out << PipeXe4{item.getPipe()};
                    if (item.getPipe() != Pipe::A)
                        out << '@' << int(item.pipe.dist);
                }
            }
            if (mod.isBreakpoint()) { startPostMod(true); out << "break"; }
#if NGEN_ASM_SHOW_FORMATS
            startPostMod(true);
            out << getEncodingXe4(op);
#endif
            if (havePostMod) out << '}';
            break;
        }
#endif
    }
}

#if XE4
void AsmCodeGenerator::outExtSendXe4(std::ostream &out, const AsmInstruction &i)
{
    auto desc = static_cast<SendgMessageDescriptor>(uint64_t(i.src[4].imm));

    bool first = true;
    auto separateOpt = [&] {
        out << (first ? "::" : ".");
        first = false;
    };

    auto asm_unsupported = [] {
#ifdef NGEN_SAFE
        throw asm_unsupported_message();
#endif
    };

    auto sfid = static_cast<SharedFunction>(i.ext);

    out << '.' << getMnemonic(sfid, hardware);

    switch (sfid) {
        case SharedFunction::gtwy:
            switch (desc.common.opcode) {
                case GatewayOpcode::eot:  out << ".eot"; break;
                case GatewayOpcode::eotr: out << ".eotr"; break;
                case GatewayOpcode::bar:  out << ".bar"; break;
                case GatewayOpcode::cbar: out << ".cbar"; break;
                case GatewayOpcode::abar_init: out << ".abar_init"; break;
                case GatewayOpcode::abar_expect: out << ".abar_expect"; break;
                case GatewayOpcode::abar_complete: out << ".abar_complete"; break;
                case GatewayOpcode::abar_arrive: out << ".abar_arrive"; break;
                case GatewayOpcode::abar_arrive_expect: out << ".abar_arrive_expect"; break;
                case GatewayOpcode::abar_try: out << ".abar_try"; break;
                case GatewayOpcode::abar_test_poll: out << ".abar_test_poll"; break;
                case GatewayOpcode::abar_inval: out << ".abar_inval"; break;
                /* The following opcodes are not yet supported by IGC, and asm syntax
                   is not defined for them: */
                case GatewayOpcode::save_bar: out << ".save_bar"; break;
                case GatewayOpcode::restore_bar: out << ".restore_bar"; break;
                case GatewayOpcode::sip_bar: out << ".sip_bar"; break;
                case GatewayOpcode::abar_save: out << ".abar_save"; break;
                case GatewayOpcode::abar_restore: out << ".abar_restore"; break;
                case GatewayOpcode::abar_query: out << ".abar_query"; break;
                case GatewayOpcode::async_mtp_fence: out << ".async_mtp_fence"; break;
                case GatewayOpcode::cbar_remote: out << ".cbar_remote"; break;
                case GatewayOpcode::cbar_wg_eot: out << ".cbar_wg_eot"; break;
                default: asm_unsupported();
            }
            switch (desc.common.opcode) {
                case GatewayOpcode::abar_init:
                    if (desc.abarInit.cfn) { separateOpt(); out << "cfn"; }
                    break;
                case GatewayOpcode::abar_expect:
                case GatewayOpcode::abar_complete:
                case GatewayOpcode::abar_arrive:
                case GatewayOpcode::abar_arrive_expect: {
                    const char *scopes[2] = {"wg", "cl"};
                    out << '.' << scopes[desc.abar.scope];
                    if (desc.abar.drop) { separateOpt(); out << "drop"; }
                    if (desc.abar.lmc)  { separateOpt(); out << "lmc"; }
                    break;
                }
                case GatewayOpcode::abar_try:
                    out << ".n" << desc.abarTest.nreg; break;
                default: break;
            }
            break;
        case SharedFunction::ugm:
        case SharedFunction::slm: {
            auto op = static_cast<LSCOpcode>(desc.common.opcode);
            bool write = true;
            bool matrix_access = false;
            switch (op) {
                case LSCOpcode::load_matrix:
                case LSCOpcode::store_matrix:
                case LSCOpcode::reduce_matrix:
                case LSCOpcode::load_matrix_unordered:
                case LSCOpcode::store_matrix_unordered:
                    matrix_access = true;
                default: break;
            }
            switch (op) {
                case LSCOpcode::load: out << ".ld"; write = false; break;
                case LSCOpcode::store: out << ".st"; break;
                case LSCOpcode::load_cmask: out << ".ldcm"; write = false; break;
                case LSCOpcode::store_cmask: out << ".stcm"; break;
                case LSCOpcode::load_status: out << ".ldst"; write = false; break;
                case LSCOpcode::atomic_inc: out << ".ainc"; break;
                case LSCOpcode::atomic_dec: out << ".adec"; break;
                case LSCOpcode::atomic_add: out << ".aadd"; break;
                case LSCOpcode::atomic_sub: out << ".asub"; break;
                case LSCOpcode::atomic_and: out << ".aand"; break;
                case LSCOpcode::atomic_load: out << ".ald"; break;
                case LSCOpcode::atomic_max: out << ".asmax"; break;
                case LSCOpcode::atomic_min: out << ".asmin"; break;
                case LSCOpcode::atomic_or: out << ".aor"; break;
                case LSCOpcode::atomic_store: out << ".ast"; break;
                case LSCOpcode::atomic_umax: out << ".aumax"; break;
                case LSCOpcode::atomic_umin: out << ".aumin"; break;
                case LSCOpcode::atomic_xor: out << ".axor"; break;
                case LSCOpcode::atomic_cmpxchg: out << ".acxg"; break;
                case LSCOpcode::atomic_fadd: out << ".afadd"; break;
                case LSCOpcode::atomic_fsub: out << ".afsub"; break;
                case LSCOpcode::atomic_fmax: out << ".afmax"; break;
                case LSCOpcode::atomic_fmin: out << ".afmin"; break;
                case LSCOpcode::atomic_fcmpxchg: out << ".afcxg"; break;
                case LSCOpcode::atomic_bfadd: out << ".abfadd"; break;
                case LSCOpcode::atomic_bfsub: out << ".abfsub"; break;
                case LSCOpcode::atomic_bfmax: out << ".abfmax"; break;
                case LSCOpcode::atomic_bfmin: out << ".abfmin"; break;
                case LSCOpcode::atomic_bfcmpxchg: out << ".abfcxg"; break;
                case LSCOpcode::fence: {
                    out << ".fence.";
                    const char *scopes[8] = {"tg", "local", "tile", "gpu", "all", "sysrel", "sysacq", ""};
                    auto scope = scopes[desc.fence.fenceScope];
                    if (!*scope) asm_unsupported();
                    out << scope;
                    if (desc.fence.flushType) {
                        separateOpt();
                        const char *flushes[8] = {"", "evict", "inval", "discard", "clean", "", "", ""};
                        auto flush = flushes[desc.fence.flushType];
                        if (!*flush) asm_unsupported();
                        out << flush;
                    }
                    return;
                }
                case LSCOpcode::load_matrix: out << ".ld_matrix"; break;
                case LSCOpcode::store_matrix: out << ".st_matrix"; break;
                case LSCOpcode::reduce_matrix: out << ".red_matrix"; break;
                case LSCOpcode::load_matrix_unordered: out << ".ld_matrix_unordered"; break;
                case LSCOpcode::store_matrix_unordered: out << ".st_matrix_unordered"; break;
                // TODO: ldp, stp, ac{add,st,sub}, a{dec,inc}w, ldcmst, ecc
                default: asm_unsupported();
            }

            if (op == LSCOpcode::load_cmask || op == LSCOpcode::store_cmask) {
                out << '.';
                if (desc.cmask.cmask & 0x1) out << 'x';
                if (desc.cmask.cmask & 0x2) out << 'y';
                if (desc.cmask.cmask & 0x4) out << 'z';
                if (desc.cmask.cmask & 0x8) out << 'w';
            }

            if (matrix_access) {
                out << ".";
                out << "a" << (1 << desc.matrix.alen);
                out << "x" << (1 << desc.matrix.astride);
                out << "." << (desc.matrix.aorient ? "acol" : "arow");
            }

            const char *ds = nullptr;
            if (matrix_access) {
                const char *matrixDataSizes[16] = {"", "", "", "4", "6", "8", "16", "32", "64"};
                ds = matrixDataSizes[desc.matrix.dataSize];
            } else {
                const char *dataSizes[8] = {"8", "16", "32", "64", "8u32", "16u32"};
                ds = dataSizes[desc.mem.dataSize];
            }
            if (!ds || !*ds) asm_unsupported();
            out << ".d" << ds;

            if (op == LSCOpcode::load || op == LSCOpcode::store || matrix_access) {
                int vlen = desc.vectorLength();
                if (vlen > 1) out << 'v' << vlen;
            }

            if (matrix_access) {
                const char *vorients[4] = {"vrow", "vcol", "coopcol", "cooprow"};
                out << "." << vorients[desc.matrix.vorient];
            }

            if (sfid != SharedFunction::slm) {
                const char *addrSizes[4] = {"a32u", "a32s", "a64", "sa32"};
                out << '.' << addrSizes[desc.mem.addrSize];
            }

            if (!matrix_access && desc.mem.cacheMode) {
                const char *cacheModes[2][16] = {{
                    "", "", "l1uc_l2uc_l3uc", "l1uc_l2uc_l3c",
                    "l1uc_l2c_l3uc", "l1uc_l2c_l3c", "l1c_l2uc_l3uc", "l1c_l2uc_l3c",
                    "l1c_l2c_l3uc", "l1c_l2c_l3c", "l1s_l2uc_l3uc", "l1s_l2uc_l3c",
                    "l1s_l2c_l3uc", "l1s_l2c_l3c", "l1i_l2i_l3i", ""
                }, {
                    "", "", "l1uc_l2uc_l3uc", "l1uc_l2uc_l3wb",
                    "l1uc_l2wb_l3uc", "l1uc_l2wb_l3wb", "l1wt_l2uc_l3uc", "l1wt_l2uc_l3wb",
                    "l1wt_l2wb_l3uc", "l1wt_l2wb_l3wb", "l1s_l2uc_l3uc", "l1s_l2uc_l3wb",
                    "l1s_l2wb_l3uc", "l1s_l2wb_l3wb", "l1wb_l2wb_l3uc", "l1wb_l2uc_l3wb"
                }};
                auto cacheMode = cacheModes[write][desc.mem.cacheMode];
                if (!*cacheMode) asm_unsupported();
                separateOpt(); out << cacheMode;
            }

            if (op == LSCOpcode::load || op == LSCOpcode::store) {
                if (desc.mem.overfetch) { separateOpt(); out << "of"; }
                if (desc.mem.transpose) { separateOpt(); out << 't'; }
            }
            break;
        }
        case SharedFunction::dma: {
            bool l2r = false, reduce = false, tensor = false, row = false, gwrite = false;
            bool needDS = false, needDT = false;
            switch (desc.common.opcode) {
                case ADMAOpcode::linear_l2r:        out << ".linear.copy_l2r"; l2r = true; break;
                case ADMAOpcode::linear_l2g:        out << ".linear.copy_l2g"; gwrite = true; break;
                case ADMAOpcode::linear_prefetch:   out << ".linear.prefetch"; break;
                case ADMAOpcode::linear_g2l:        out << ".linear.copy_g2l"; break;
                case ADMAOpcode::linear_reduce_l2r: out << ".linear.reduce_l2r"; l2r = reduce = true; break;
                case ADMAOpcode::linear_reduce_l2g: out << ".linear.reduce_l2g"; reduce = gwrite = true; break;
                case ADMAOpcode::tensor_l2g:        out << ".tensor.copy_l2g"; tensor = gwrite = true; break;
                case ADMAOpcode::tensor_prefetch:   out << ".tensor.prefetch"; tensor = true; break;
                case ADMAOpcode::tensor_g2l:        out << ".tensor.copy_g2l"; tensor = true; break;
                case ADMAOpcode::tensor_reduce_l2g: out << ".tensor.reduce_l2g"; reduce = tensor = gwrite = true; break;
                case ADMAOpcode::row_l2g:           out << ".row.copy_l2g"; row = gwrite = needDS = true; break;
                case ADMAOpcode::row_prefetch:      out << ".row.prefetch"; row = true; break;
                case ADMAOpcode::row_g2l:           out << ".row.copy_g2l"; row = needDT = needDS = true; break;
                case ADMAOpcode::row_reduce_l2g:    out << ".row.reduce_l2g"; reduce = row = gwrite = true; break;
                default: asm_unsupported();
            }

            if (tensor) out << '.' << desc.adma.dims + 1 << 'd';

            if (row) {
                const char *addrs[4] = {"a32s", "a32u", "a64", ""};
                auto addr = addrs[desc.adma.addrType];
                if (!*addr) asm_unsupported();
                out << '.' << addr;
            }

            if (reduce) {
                const char *rops[8] = {"incw", "decw", "add", "min", "max", "and", "or", "xor"};
                out << '.' << rops[desc.adma.reduction];
            }

            if (reduce || needDT) {
                const char *dtypes[4] = {"fp", "bf", "uint", "int"};
                out << '.' << dtypes[desc.adma.dataType];
            }

            if (tensor || reduce || needDS) {
                const char *dsizes[8] = {"d8", "d16", "d32", "d64", "d4", "d6", "", ""};
                auto ds = dsizes[desc.adma.dataSize];
                if (!*ds) asm_unsupported();
                out << '.' << ds;
            }

            if (row)
                out << '.' << (desc.adma.rowSize + 1) * 8;

            if (desc.adma.useCopySize) { separateOpt(); out << "ucs"; }
            if (desc.adma.multicast)   { separateOpt(); out << "mc"; }
            if (!l2r) {
                const char *cacheModes[2][4] = {{"l2c_l3c",   "l2c_l3uc",  "l2uc_l3c",  "l2uc_l3uc"},
                                                {"l2wb_l3wb", "l2wb_l3uc", "l2uc_l3wb", "l2uc_l3uc"}};
                separateOpt();
                out << cacheModes[gwrite][desc.adma.cache];
            }
            if (desc.adma.abar) { separateOpt(); out << "abar"; }
            break;
        }
        case SharedFunction::mma: {
            switch (desc.common.opcode) {
                case AMMAOpcode::dense_mma:      out << ".ns"; break;
                case AMMAOpcode::sparse_mma:     out << ".s";  break;
                case AMMAOpcode::fp_error_query: out << ".fp_error_query"; return;
                case AMMAOpcode::fp_error_clear: out << ".fp_error_clear"; return;
                default: asm_unsupported();
            }
            out << ".m" << (desc.amma.m + 1) * 32
                <<  "n" << (desc.amma.n + 1) * 32
                <<  "k" << (desc.amma.k + 1) * 32;
            const char *mtypes[16] = {"", "tf32", "f16",  "bf16", "e4m3", "e5m2", "e2m1", "e3m0",
                                      "", "",     "e2m3", "e3m2", "u4",   "s4",   "u8",   "s8"};
            const char *atypes[8]  = {"f32", "f16", "bf16", "", "", "", "", "s32"};
            out << '.' << atypes[desc.amma.dtype]
                << '_' << mtypes[desc.amma.atype]
                << '_' << mtypes[desc.amma.btype]
                << '_' << atypes[desc.amma.ctype];
            if (desc.amma.ascale)  { separateOpt(); out << "ascale"; }
            if (desc.amma.bscale)  { separateOpt(); out << "bscale"; }
            if (desc.amma.alayout) { separateOpt(); out << "am"; }
            if (desc.amma.blayout) { separateOpt(); out << "bk"; }
            if (desc.amma.atm)     { separateOpt(); out << "atm"; }
            if (desc.amma.btm)     { separateOpt(); out << "btm"; }
            if (desc.amma.dtm)     { separateOpt(); out << "dtm"; }
            if (desc.amma.areuse)  { separateOpt(); out << "a_reuse"; }
            break;
        }
        default: asm_unsupported();
    }
}

bool AsmCodeGenerator::outSpecialOps(std::ostream &out, const AsmInstruction &i)
{
    switch (i.op) {
        case Opcode::send_128C:
        case Opcode::sendc_128C:
        case Opcode::sendg_128C:
        case Opcode::sendcg_128C: break;
        default: return false;
    }

    auto defaultSend = [&] {
        if (!i.src[0].isEmptyOrNull())
            i.src[0].outputText(out, PrintDetail::xe4, labelManager);
        for (int n = 1; n <= 3; n++) if (!i.src[n].isEmptyOrNull()) {
            out << ", ";
            i.src[n].outputText(out, PrintDetail::xe4, labelManager);
        }
    };

    auto sfid = static_cast<SharedFunction>(i.ext);
    switch (sfid) {
        case SharedFunction::ugm:
        case SharedFunction::slm: break;
        default:                  defaultSend(); return true;
    }

    auto desc = static_cast<SendgMessageDescriptor>(uint64_t(i.src[4].imm));
    auto op = static_cast<LSCOpcode>(desc.common.opcode);

    switch (op) {
        case LSCOpcode::fence:
        case LSCOpcode::load_matrix:
        case LSCOpcode::store_matrix:
        case LSCOpcode::reduce_matrix:
        case LSCOpcode::load_matrix_unordered:
        case LSCOpcode::store_matrix_unordered: defaultSend(); return true;
        default: break;
    }

    auto &offset = i.src[0];
    RegisterRange roffset;
    if (offset.type == AsmOperand::Type::reg && offset.reg.isValid())
        roffset = GRFRange(offset.reg.getBase(), 1);
    else if (offset.type == AsmOperand::Type::range)
        roffset = offset.range;
    auto &base = i.src[2];
    bool surface = (desc.mem.addrSize == 0b11);
    auto immOffset = surface ? desc.surface.offset : desc.flat.offset;
    immOffset *= desc.elementBytesMem();

    int scale = 1;
    if (desc.mem.scale)
        scale = desc.vectorLength() * desc.elementBytesMem() << (desc.mem.scale - 1);

    bool haveBase = !base.isEmptyOrNull();
    bool haveOffset = roffset.isValid();

    if (surface) {
        out << '[' << base;
        if (desc.surface.ssIdx)
            out << '[' << desc.surface.ssIdx << ']';
        out << ']';
    }
    out << '[';
    bool first = false;
    if (!surface) {
        if (haveBase)
            base.outputText(out, PrintDetail::xe4, labelManager);
        else if (!haveOffset)
            out << "null";
        else
            first = true;
    }
    if (haveOffset) {
        if (!first) out << " + ";
        AsmOperand(roffset).outputText(out, PrintDetail::xe4, labelManager);
        out << '*' << scale;
    }
    if (immOffset) {
        out << ' ' << "+-"[immOffset < 0] << " 0x"
            << std::hex << std::abs(immOffset) << std::dec;
    }
    out << ']';

    if (!i.src[1].isEmptyOrNull()) {
        out << ", ";
        i.src[1].outputText(out, PrintDetail::xe4, labelManager);
    }
    return true;
}
#endif

} /* namespace NGEN_NAMESPACE */

#endif
#endif
