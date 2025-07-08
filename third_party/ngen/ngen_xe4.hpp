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

// Xe4 binary encoding.

#ifndef NGEN_XE4_HPP
#define NGEN_XE4_HPP

#include "ngen_auto_swsb.hpp"

namespace NGEN_NAMESPACE {

struct InstructionXe4 {
    union {
        struct {
            unsigned dbg : 1;
            unsigned op : 9;
            unsigned autoswsb : 1;  /* unused bit */
            unsigned : 21;
            unsigned : 32;
            unsigned : 32;
            unsigned : 32;
        } common;
        struct {
            uint64_t : 11;
            uint64_t sbid : 5;
            uint64_t sb0 : 7;
            uint64_t sb1 : 7;
            uint64_t : 11;
            uint64_t cmcf : 7;
            uint64_t ictrl : 2;
            uint64_t s2inv : 1;
            uint64_t s1inv : 1;
            uint64_t s0inv : 1;
            uint64_t dsat : 1;
            uint64_t ipred : 1;
            uint64_t pf : 4;
            uint64_t rmo : 3;
            uint64_t w : 1;
            uint64_t src2_0 : 1;

            uint64_t src2_1_10 : 10;
            uint64_t imm : 21;
            uint64_t src1 : 11;
            uint64_t src0 : 11;
            uint64_t dst : 11;
        } _128;
        struct {
            uint64_t : 32;
            uint64_t ipred : 1;
            uint64_t pf : 4;
            uint64_t brc : 1;
            uint64_t w : 1;
            uint64_t sctrl : 1;
            uint64_t uip0_23 : 24;

            uint64_t uip24_31 : 8;
            uint64_t jip : 32;
            uint64_t : 24;
        } _128B;
        struct {
            uint64_t : 23;
            uint64_t ipred : 1;
            uint64_t pf : 4;
            uint64_t w : 1;
            uint64_t eot : 1;
            uint64_t sfid : 4;
            uint64_t : 1;
            uint64_t msgd0_28 : 29;

            uint64_t msgd29_46 : 18;
            uint64_t ind : 1;
            uint64_t sgather : 1;
            uint64_t inda : 11;
            uint64_t : 33;          // -> 128A
        } _128C;
        struct {
            uint64_t : 14;
            uint64_t ictrl : 2;
            uint64_t : 7;
            uint64_t cmcf : 7;
            uint64_t w : 1;
            uint64_t src2 : 11;
            uint64_t imm0_21 : 22;

            uint64_t imm22_52 : 31;
            uint64_t src1 : 11;
            uint64_t : 22;
        } _128D;
        struct {
            uint64_t : 30;
            uint64_t bctrl : 8;
            uint64_t : 26;
            uint64_t : 64;
        } _128E;
        struct {
            uint64_t : 48;
            uint64_t sgran : 3;
            uint64_t : 8;
            uint64_t w : 1;
            uint64_t idx4 : 1;
            uint64_t ictrl : 2;
            uint64_t : 1;
            uint64_t : 64;
        } _128F;
        struct {
            uint64_t : 26;
            uint64_t ictrl : 1;
            uint64_t : 2;
            uint64_t s1inv : 1;
            uint64_t s0inv : 1;
            uint64_t : 1;
            uint64_t ipred : 1;
            uint64_t pf : 4;
            uint64_t : 3;
            uint64_t w : 1;
            uint64_t : 21;
            uint64_t bwidth0_1 : 2;

            uint64_t bwidth2_5 : 4;
            uint64_t boff : 5;
            uint64_t : 55;
        } _128G;
        struct {
            uint64_t : 26;
            uint64_t cf : 4;
            uint64_t dmme : 4;
            uint64_t s0mme : 4;
            uint64_t s1mme : 4;
            uint64_t s2mme : 4;
            uint64_t : 2;
            uint64_t ictrl : 2;
            uint64_t : 14;

            uint64_t : 64;
        } _128H;
        struct {
            uint64_t : 38;
            uint64_t dctrl : 1;
            uint64_t cmcf : 7;
            uint64_t ipred : 1;
            uint64_t pf : 4;
            uint64_t w : 1;
            uint64_t lctrl : 1;
            uint64_t src2 : 11;

            uint64_t imsk : 21;
            uint64_t dimm : 21;
            uint64_t : 22;
        } _128I;
        struct {
            uint64_t : 43;
            uint64_t spinv : 1;
            uint64_t spf : 4;
            uint64_t : 16;

            uint64_t : 64;
        } _128J;
        struct {
            uint64_t : 40;
            uint64_t spf : 4;
            uint64_t dpf : 4;
            uint64_t : 16;

            uint64_t : 64;
        } _128K;
        struct {
            uint64_t : 30;
            uint64_t ctype : 1;
            uint64_t cid : 8;
            uint64_t exp: 1;
            uint64_t : 24;

            uint64_t : 64;
        } _128L;
        struct {
            uint64_t : 30;
            uint64_t atype : 4;
            uint64_t btype : 4;
            uint64_t k : 3;
            uint64_t : 3;
            uint64_t n : 6;
            uint64_t top : 3;
            uint64_t : 11;

            uint64_t : 10;
            uint64_t src3 : 11;
            uint64_t amxeo : 2;
            uint64_t bmxeo : 2;
            uint64_t : 39;
        } _128M;
        struct {
            uint64_t : 30;
            uint64_t ttype : 4;
            uint64_t pack : 3;
            uint64_t mralg : 3;
            uint64_t mxdim : 2;
            uint64_t reddim : 2;
            uint64_t n : 6;
            uint64_t : 14;

            uint64_t : 21;
            uint64_t imm10 : 10;
            uint64_t : 33;
        } _128N;
        struct {
            uint64_t : 32;
            uint64_t sfmt : 5;
            uint64_t : 27;

            uint64_t : 64;
        } _128O;
        struct {
            uint64_t : 32;
            uint64_t ipred : 1;
            uint64_t pf : 4;
            uint64_t : 1;
            uint64_t w : 1;
            uint64_t : 1;
            uint64_t cip0_23 : 24;

            uint64_t cip24_63 : 40;
            uint64_t : 24;
        } _128P;
        struct {
            uint64_t : 30;
            uint64_t st0 : 4;
            uint64_t st1 : 4;
            uint64_t : 10;
            uint64_t ictrl : 2;
            uint64_t : 14;

            uint64_t : 64;
        } _128Q;
        struct {
            uint64_t : 32;
            uint64_t ipred : 1;
            uint64_t pf : 4;
            uint64_t w : 1;
            uint64_t : 10;
            uint64_t ictrl : 1;
            uint64_t : 4;
            uint64_t imm0_10 : 11;

            uint64_t imm11_52 : 42;
            uint64_t : 22;
        } _128R;
        struct {
            uint64_t : 12;
            uint64_t sb0 : 7;
            uint64_t : 2;
            uint64_t s1inv : 1;
            uint64_t s0inv : 1;
            uint64_t dsat : 1;
            uint64_t w : 1;
            uint64_t ictrl : 1;
            uint64_t imm : 5;
            uint64_t src1 : 11;
            uint64_t src0 : 11;
            uint64_t dst : 11;

            uint64_t : 64;
        } _64;
        struct {
            uint64_t : 19;
            uint64_t cmcf : 7;
            uint64_t ipred : 1;
            uint64_t pf : 4;
            uint64_t : 33;

            uint64_t : 64;
        } _64B;
        struct {
            uint64_t : 20;
            uint64_t src2 : 11;
            uint64_t : 33;

            uint64_t : 64;
        } _64C;
        struct {
            uint64_t : 23;
            uint64_t bctrl : 8;
            uint64_t : 33;

            uint64_t : 64;
        } _64D;
        struct {
            uint64_t : 13;
            uint64_t sctrl : 3;
            uint64_t : 1;
            uint64_t bar : 5;
            uint64_t sb0 : 7;
            uint64_t sb1 : 7;
            uint64_t sb2 : 7;
            uint64_t sb3 : 7;
            uint64_t sb4 : 7;
            uint64_t sb5 : 7;

            uint64_t : 64;
        } _64E;
        struct {
            uint32_t : 32;
            uint32_t imm;
            uint64_t : 64;
        } _64EImm;
        struct {
            uint64_t : 25;
            uint64_t ictrl0 : 1;
            uint64_t ictrl1 : 1;
            uint64_t : 37;

            uint64_t : 64;
        } _64F;
        struct {
            uint64_t : 26;
            uint64_t ipred : 1;
            uint64_t pf : 4;
            uint64_t w : 1;
            uint64_t jip : 32;

            uint64_t : 64;
        } _64G;    /* 64H is the same as 64G, with no JIP */
        struct {
            uint64_t : 26;
            uint64_t ipred : 1;
            uint64_t pf : 4;
            uint64_t : 6;
            uint64_t imm : 5;
            uint64_t : 22;

            uint64_t : 64;
        } _64I;
        uint64_t qword[2];
    };

    constexpr InstructionXe4() : qword{0,0} {};
    InstructionXe4(InstructionModifier mod, Opcode op) : InstructionXe4{} {
        common.dbg = mod.isBreakpoint();
        common.op = static_cast<int>(op);
        common.autoswsb = mod.isAutoSWSB();
    }

    // Decoding routines for auto-SWSB.
    bool autoSWSB() const          { return common.autoswsb; }
    inline SWSBInfo swsb() const;
    inline void setSWSB(SWSBInfo swsb);
    void clearAutoSWSB()           { common.autoswsb = 0; }
    Opcode opcode() const          { return static_cast<Opcode>(common.op | 0x8000); }
    inline SyncFunction syncFC() const;
    SharedFunction sfid() const    { return static_cast<SharedFunction>(_128C.sfid); }
    bool eot() const               { return isSend() && _128C.eot; }
    bool predicated() const        { return isSend() && (_128C.pf != 0xF); }
    static bool atomic()           { return false; }
    bool is64() const              { return isEncoding64(getEncodingXe4(opcode())); }
    static unsigned dstTypecode()  { return 0u; }
    static unsigned src0Typecode() { return 0u; }
    static unsigned src1Typecode() { return 0u; }

    void shiftJIP(int32_t shift)   { _128B.jip += shift; }
    void shiftUIP(int32_t shift)   {
        auto uip = (_128B.uip0_23 | (_128B.uip24_31 << 24)) + shift;
        _128B.uip0_23 = uip;
        _128B.uip24_31 = uip >> 24;
    }

    inline autoswsb::DestinationMask destinations(int &jip, int &uip) const;
    inline bool getOperandRegion(autoswsb::DependencyRegion &region, int opNum) const;
    inline bool getImm32(uint32_t &imm) const;
    static bool getSendDesc(MessageDescriptor &desc)           { return false; }
    static bool getARFType(ARFType &arfType, int opNum, HW hw) { return false; }
    inline int getFencedepJIP() const;
    inline SendgMessageDescriptor getSendgDesc() const {
        uint64_t desc = _128C.msgd0_28 | (_128C.msgd29_46 << 29);
        return static_cast<SendgMessageDescriptor>(desc);
    }

    bool isSend() const {
        switch (opcode()) {
            case Opcode::send_128C:
            case Opcode::sendc_128C:
            case Opcode::sendg_128C:
            case Opcode::sendcg_128C: return true;
            default:                  return false;
        }
    }

    static constexpr int directARFSize() { return 32; }
};

static_assert(sizeof(InstructionXe4) == 16, "Internal error: Instruction12 has been padded by the compiler.");

static inline int swsbCount(EncodingXe4 enc)
{
    switch (enc) {
        case EncodingXe4::_128A:
        case EncodingXe4::_128B:
        case EncodingXe4::_128E:
        case EncodingXe4::_128F:
        case EncodingXe4::_128J:
        case EncodingXe4::_128K:
        case EncodingXe4::_128O:
        case EncodingXe4::_128Q:
        case EncodingXe4::_128R: return 2;
        default: return 1;      /* 64E treated separately */
    }
}

// Encoding routines.
static inline unsigned encodeRegXe4(RegData rd)
{
    switch (rd.getRegFile()) {
        case RegFileARF: return static_cast<unsigned>(rd.getARFType()) & 0xF;
        case RegFileGRF: return InstructionXe4::directARFSize() + rd.getBase();
        case RegFileSRF: return 2047 - rd.getBase();
        default:         return 0; /* unreachable */
    }
}

static inline unsigned encodeRegXe4(IndirectARF iarf)
{
    return static_cast<unsigned>(iarf);
}

template <typename T>
static inline T encodeImmXe4(const Immediate &i)
{
    auto val = static_cast<uint64_t>(i);
    auto dt = i.getType();

    // Truncate high word repetitions for word-size immediates
    if (dt == DataType::s16 || dt == DataType::u16)
        val &= 0xFFFF;

    // Shift bits right for floating-point immediates
    if (dt == DataType::f && sizeof(T) < 4)
        val >>= (32 - std::min<int>(32, 8 * sizeof(T)));
    if (dt == DataType::df && sizeof(T) < 8)
        val >>= (64 - 8 * sizeof(T));

    return T(val);
}

template <typename T, bool extraHi>
static inline unsigned encodeRegOrImmXe4(RegData rd, uint64_t &immExtra)
{
    return encodeRegXe4(rd);
}

template <typename T, bool extraHi>
static inline unsigned encodeRegOrImmXe4(IndirectARF iarf, uint64_t &immExtra)
{
    return encodeRegXe4(iarf);
}

template <typename T, bool extraHi>
static inline unsigned encodeRegOrImmXe4(Immediate i, uint64_t &immExtra)
{
    auto val = encodeImmXe4<T>(i);
    if (extraHi) {
        immExtra = val >> 11;
        return val & 0x7FF;
    } else {
        immExtra = val;
        return val >> (8 * sizeof(T) - 11);
    }
}

static inline unsigned encodeCMCFXe4(InstructionModifier mod)     /* cm/cf fields */
{
    unsigned cmTable[16] = {0, 0, 1, 2, 4, 3, 5, 6,
                            7, 0, 0, 0, 0, 0, 0, 0};

    auto cf = mod.getCFlag().getARFBase();
    auto cmg = mod.getCMod();
    unsigned cm = cmTable[static_cast<uint8_t>(cmg) & 0xF];

    if (cmg == ConditionModifier::none)
        cf = 0xF;
    return cm | (cf << 3);
}

static inline unsigned encodePFXe4(InstructionModifier mod)
{
    return (mod.getPredCtrl() == PredCtrl::None) ? 0xF : mod.getFlagReg().getARFBase();
}

static inline unsigned encodeCvtSrcTypeXe4(DataType dt)
{
    unsigned table[32] = {6,  7,  8,  9,  10, 11, 0,  1,  4,  5,  3,  2,  31, 31, 31, 31,
                          31, 31, 31, 16, 31, 17, 13, 12, 31, 31, 31, 31, 31, 31, 31, 31};
    auto enc = table[static_cast<uint8_t>(dt) & 0x1F];

#ifdef NGEN_SAFE
    if (enc == 31) throw invalid_type_exception();
#endif

    return enc;
}

static inline unsigned encodeSWSBXe4(SWSBItem item)
{
    int sb = 0;
    if (item.isPipe()) {
        if (item.getPipe() == Pipe::A)
            sb = 0x1;
        else {
            unsigned table[8] = {0, 0, 2, 1, 3, 4, 5, 6};
            sb = table[item.pipe.pipe & 7] << 3;
            sb |= (item.pipe.dist - 1);
        }
    } else if (item.isToken()) {
        sb = item.token.dst ? 0x60 : 0x40;
        sb |= item.getToken();
    }
    return sb;
}

template <int nsb, bool withSBID = false>
static inline void encodeSWSBXe4(InstructionModifier mod, unsigned &sb0, unsigned &sb1, unsigned &sbid)
{
    sb0 = sb1 = 0;
    sbid = 0x20;

    for (auto item: mod.getSWSB()) {
        if (item.hasTokenSet()) {
#ifdef NGEN_SAFE
            if (!withSBID) throw invalid_modifiers_exception();
#endif
            sbid = item.getToken();
        } else
            ((sb0 == 0) ? sb0 : sb1) = encodeSWSBXe4(item);
    }

#ifdef NGEN_SAFE
    if (nsb < 2 && sb1 != 0) throw invalid_modifiers_exception();
#endif
}

template <int nsb, bool withSBID = false>
static inline void encodeSWSBXe4(InstructionXe4 &i, InstructionModifier mod)
{
    unsigned sb0, sb1, sbid;
    encodeSWSBXe4<nsb, withSBID>(mod, sb0, sb1, sbid);

    if (withSBID && sbid >= 0x20)
        sb0 = 7;        /* placeholder representing unassigned SBID */

    i._128.sb0 = sb0;
    i._128.sb1 = sb1;
    i._128.sbid = sbid;
}

static inline void encodeSWSB64Xe4(InstructionXe4 &i, InstructionModifier mod)
{
    unsigned sb0, sb1, sbid;
    encodeSWSBXe4<1, false>(mod, sb0, sb1, sbid);

    i._64.sb0 = sb0;
}

static inline unsigned encodeSyncFunctionXe4(SyncFunction fc)
{
    unsigned table[16] = {0, 0, 1, 2, 4, 0, 0, 0,
                          0, 0, 0, 0, 5, 0, 3, 6};
    return table[static_cast<uint8_t>(fc) & 0xF];
}

// Decoding routines.
static inline SWSBItem decodeSWSBXe4(unsigned sb)
{
    if (sb == 0 || sb == 7)
        return SWSBItem();
    else if (sb == 1)
        return SWSBItem(Pipe::A, 1);
    else if (sb & 0x40) {
        bool dst = (sb & 0x20);
        return SWSBItem(sb & 0x1F, !dst, dst);
    } else {
        Pipe table[8] = {Pipe::A, Pipe::I, Pipe::F, Pipe::L, Pipe::M, Pipe::S, Pipe::X, Pipe::I};
        return SWSBItem(table[(sb >> 3) & 7], (sb & 7) + 1);
    }
}

SWSBInfo InstructionXe4::swsb() const
{
    auto enc = getEncodingXe4(opcode());
    SWSBInfo swsb{};

    if (isEncoding64(enc)) {
        swsb[0] = decodeSWSBXe4(_64.sb0);
    } else {
        swsb[0] = decodeSWSBXe4(_128.sb0);
        if (swsbItems(enc) >= 2)
            swsb[1] = decodeSWSBXe4(_128.sb1);
        else if (enc == EncodingXe4::_128C && _128.sb0 != 7)
            swsb[1] = SBID(_128.sbid).set;
    }

    return swsb;
}

void InstructionXe4::setSWSB(SWSBInfo swsb)
{
    unsigned sb0, sb1, sbid;
    encodeSWSBXe4<2, true>(swsb, sb0, sb1, sbid);

    auto enc = getEncodingXe4(opcode());
    if (enc == EncodingXe4::_64E) {
        _64E.sb0 = sb0;
    } else if (isEncoding64(enc)) {
        _64.sb0 = sb0;
    } else {
        _128.sb0 = sb0;
        if (swsbItems(enc) >= 2)
            _128.sb1 = sb1;
        else if (enc == EncodingXe4::_128C)
            _128.sbid = sbid;
    }
}

SyncFunction InstructionXe4::syncFC() const
{
    using F = SyncFunction;
    F table[8] = {F::nop, F::srcmsk, F::dstmsk, F::barid, F::barsrc, F::barflush, F::host, F::nop};
    return table[_64E.sctrl];
}

autoswsb::DestinationMask InstructionXe4::destinations(int &jip, int &uip) const
{
    using namespace autoswsb;

    if (!isBranch(opcode())) {
        if (isSend() && _128C.eot && !predicated())
            return DestNone;
        return DestNextIP;
    }

    DestinationMask mask = DestNextIP;
    switch (opcode()) {
        case Opcode::call_128B:
        case Opcode::calld_128B:
        case Opcode::join_128B:
        case Opcode::jmpi_128B:
        case Opcode::brd_128B:
        case Opcode::ret_128B:
            mask = _128B.sctrl ? DestUnknown : (DestNextIP | DestJIP); break;
        case Opcode::calla_128P:
        case Opcode::callad_128P:
            mask = DestUnknown; break;
        case Opcode::goto__128B:
            mask = (DestNextIP | DestJIP | DestUIP); break;
        default: break;
    }

    if ((opcode() == Opcode::jmpi) && !predicated())
        mask &= ~DestNextIP;

    if (mask & DestJIP) jip = int32_t(_128B.jip) / sizeof(InstructionXe4);
    if (mask & DestUIP) uip = int32_t(_128B.uip0_23 | (_128B.uip24_31 << 24)) / sizeof(InstructionXe4);

    return mask;
}

bool InstructionXe4::getImm32(uint32_t &imm) const
{
    /* Only need to support sync.srcmsk/dstmsk */
    imm = _64EImm.imm;
    return true;
}

int InstructionXe4::getFencedepJIP() const
{
    return uint32_t(_128.imm | (_128.src1 << 21)) / sizeof(InstructionXe4);
}

bool InstructionXe4::getOperandRegion(autoswsb::DependencyRegion &region, int opNum) const
{
    using autoswsb::DependencyRegion;

    auto op = opcode();
    auto enc = getEncodingXe4(op);
    unsigned r = 0;
    int nr = getDwords(dstDataType(op));

    if (isEncoding64(enc)) {
        unsigned ictrl = 0;
        int nsrc = 2;

        switch (enc) {
            case EncodingXe4::_64A: ictrl = (_64.ictrl << 1); break;
            case EncodingXe4::_64D: break;
            case EncodingXe4::_64F: ictrl = _64F.ictrl0 | (_64F.ictrl1 << 1); break;
            case EncodingXe4::_64E:
            case EncodingXe4::_64G:
            case EncodingXe4::_64H: return false;
            default: throw std::runtime_error("unimplemented");
        }
        if (opNum >= nsrc) return false;
        if (opNum >= 0 && (ictrl & (1 << opNum))) return false;
        switch (opNum) {
            case -1: r = _64.dst; break;
            case  0: r = _64.src0; break;
            case  1: r = _64.src1; break;
            default: break;
        }
    } else {
        unsigned ictrl = 0;
        int nsrc = 3;
        switch (enc) {
            case EncodingXe4::_128E:
            case EncodingXe4::_128J:
            case EncodingXe4::_128K:
            case EncodingXe4::_128O:
            case EncodingXe4::_128Q:
            case EncodingXe4::_128A: ictrl = _128.ictrl << 1; break;
            case EncodingXe4::_128B: ictrl = _128B.sctrl; nsrc = 1; break;
            case EncodingXe4::_128D: ictrl = _128D.ictrl << 1; break;
            case EncodingXe4::_128F: ictrl = _128F.ictrl << 1; break;
            case EncodingXe4::_128G: ictrl = _128G.ictrl << 1; nsrc = 2; break;
            case EncodingXe4::_128H: ictrl = _128H.ictrl << 1; break;
            case EncodingXe4::_128R: ictrl = _128R.ictrl; nsrc = 1; break;
            case EncodingXe4::_128I: ictrl = (_128I.imsk | (_128I.lctrl << 1)); nsrc = 2; break;
            case EncodingXe4::_128L:
            case EncodingXe4::_128P: return false;
            case EncodingXe4::_128C: {
                uint64_t rdesc = _128C.msgd0_28 | (_128C.msgd29_46 << 29);
                SendgMessageDescriptor desc{rdesc};
                auto sfid = static_cast<SharedFunction>(_128C.sfid);
                nsrc = 2;
                bool dstSRF = (_128.dst >= 0x600);
                bool src1SRF = (_128.src1 >= 0x600);
                switch (opNum) {
                    case -1: nr = desc.dstLen(HW::Xe4, dstSRF ? 1 : 32, sfid); break;
                    case  0: nr = desc.src0Len(HW::Xe4, 32, sfid); break;
                    case  1: nr = desc.src1Len(HW::Xe4, src1SRF ? 1 : 32, sfid); break;
                    case  2: nr = 1 + _128C.ind; opNum = -2; break;
                    break;
                }
                break;
            }
            default: throw std::runtime_error("unimplemented");
        }
        if (opNum >= nsrc) return false;
        if (opNum >= 0 && (ictrl & (1 << opNum))) return false;
        if (opNum == 1 && enc == EncodingXe4::_128I)
            r = _128I.src2;
        else switch (opNum) {
            case -2: r = _128C.inda; break;
            case -1: r = _128.dst; break;
            case  0: r = _128.src0; break;
            case  1: r = _128.src1; break;
            case  2: r = _128.src2_0 | (_128.src2_1_10 << 1); break;
            default: break;
        }
    }

    if (r < directARFSize()) return false;      /* ARF */

    bool srf = (r >= 0x600);
    if (srf && enc == EncodingXe4::_128I) nr *= 32;

    if (nr < 0) {
        region = DependencyRegion{HW::Xe4};
        return true;
    }

    auto rr = srf ? RegisterRange(0x7FF - r, nr, true)
                  : RegisterRange(r - directARFSize(), nr, false);

    region = DependencyRegion{HW::Xe4, rr};
    return true;
}

} /* namespace */

#endif
