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

/*
 * Do not #include this file directly; ngen uses it internally.
 */


// Pseudo-instructions and macros.
template <typename DT = void>
void bfi(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, const RegData &src2, const RegData &src3, SourceLocation loc = {}) {
    bfi1<DT>(mod, dst, src0, src1, loc);
    bfi2<DT>(mod, dst, dst, src2, src3, loc);
}

// Brief compare instructions.
template <typename DT = void>
void cmp(const InstructionModifier &mod, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
    auto dt = getDataType<DT>();
    if (dt == DataType::invalid)
        dt = src0.getType();
    cmp<DT>(mod, null.retype(dt), src0, src1, loc);
}
template <typename DT = void>
void cmp(const InstructionModifier &mod, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
    auto dt = getDataType<DT>();
    if (dt == DataType::invalid)
        dt = src0.getType();
    cmp<DT>(mod, null.retype(dt), src0, src1, loc);
}
template <typename DT = void> void cmp(const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
    cmp<DT>(defaultMods(), src0, src1, loc);
}
template <typename DT = void> void cmp(const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
    cmp<DT>(defaultMods(), src0, src1, loc);
}
#if XE4
template <typename DT = void>
void scmp(const InstructionModifier &mod, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
    auto dt = getDataType<DT>();
    if (dt == DataType::invalid)
        dt = src0.getType();
    cmp<DT>(mod, null.retype(dt), src0, src1, loc);
}
template <typename DT = void>
void scmp(const InstructionModifier &mod, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
    auto dt = getDataType<DT>();
    if (dt == DataType::invalid)
        dt = src0.getType();
    scmp<DT>(mod, null.retype(dt), src0, src1, loc);
}
template <typename DT = void> void scmp(const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
    scmp<DT>(InstructionModifier(), src0, src1, loc);
}
template <typename DT = void> void scmp(const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
    scmp<DT>(InstructionModifier(), src0, src1, loc);
}
#endif

// Brief math instructions.
template <typename DT = void>
void cos(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::cos, dst, src0, loc);
}
template <typename DT = void>
void exp(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::exp, dst, src0, loc);
}
template <typename DT = void>
void fdiv(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::fdiv, dst, src0, src1, loc);
}
template <typename DT = void>
void fdiv(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::fdiv, dst, src0, src1, loc);
}
template <typename DT = void>
void idiv(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::idiv, dst, src0, src1, loc);
}
template <typename DT = void>
void idiv(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::idiv, dst, src0, src1, loc);
}
template <typename DT = void>
void inv(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::inv, dst, src0, loc);
}
template <typename DT = void>
void invm(const InstructionModifier &mod, const ExtendedReg &dst, const ExtendedReg &src0, const ExtendedReg &src1, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::invm, dst, src0, src1, loc);
}
template <typename DT = void>
void iqot(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::iqot, dst, src0, src1, loc);
}
template <typename DT = void>
void iqot(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::iqot, dst, src0, src1, loc);
}
template <typename DT = void>
void irem(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::irem, dst, src0, src1, loc);
}
template <typename DT = void>
void irem(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::irem, dst, src0, src1, loc);
}
template <typename DT = void>
void log(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::log, dst, src0, loc);
}
template <typename DT = void>
void pow(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const RegData &src1, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::pow, dst, src0, src1, loc);
}
template <typename DT = void>
void pow(const InstructionModifier &mod, const RegData &dst, const RegData &src0, const Immediate &src1, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::pow, dst, src0, src1, loc);
}
template <typename DT = void>
void rsqt(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::rsqt, dst, src0, loc);
}
template <typename DT = void>
void rsqtm(const InstructionModifier &mod, const ExtendedReg &dst, const ExtendedReg &src0, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::rsqtm, dst, src0, loc);
}
#if XE3P
template <typename DT = void>
void sigm(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::sigm, dst, src0, loc);
}
#endif
template <typename DT = void>
void sin(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::sin, dst, src0, loc);
}
template <typename DT = void>
void sqt(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::sqt, dst, src0, loc);
}
#if XE3P
template <typename DT = void>
void tanh(const InstructionModifier &mod, const RegData &dst, const RegData &src0, SourceLocation loc = {}) {
    math<DT>(mod, MathFunction::tanh, dst, src0, loc);
}
#endif

#define TMP(n) tmp[n].retype(dst.getType())

// IEEE 754-compliant divide math macro sequence.
//   Requires GRFs initialized with 0.0 and 1.0, as well as temporary GRFs (4 for single precision, 5 for double precision).
//   dst, num, denom must be distinct GRFs.
template <typename DT = void, typename A>
void fdiv_ieee(const InstructionModifier &mod, FlagRegister flag, RegData dst, RegData num, RegData denom,
               RegData zero, RegData one, const A &tmp, InstructionModifier cfmod = InstructionModifier(),
               SourceLocation loc = {})
{
    DataType dt = getDataType<DT>();
    if (dt == DataType::invalid)
        dt = dst.getType();
    if (cfmod.getExecSize() == 0)
        cfmod = mod;

    Label labelSkip;

    switch (dt) {
        case DataType::hf:
            fdiv<DT>(mod, dst, num, denom, loc);
            break;
        case DataType::f:
            invm<DT>(mod | eo | flag,         dst | mme0,      num | nomme,   denom | nomme, loc);
            if_(cfmod | ~flag, labelSkip, loc);

            madm<DT>(mod, TMP(0) | mme1,     zero | nomme,     num | nomme,     dst | mme0, loc);
            madm<DT>(mod, TMP(1) | mme2,      one | nomme,  -denom | nomme,     dst | mme0, loc);
            madm<DT>(mod, TMP(2) | mme3,      dst | mme0,   TMP(1) | mme2,      dst | mme0, loc);
            madm<DT>(mod, TMP(3) | mme4,      num | nomme,  -denom | nomme,  TMP(0) | mme1, loc);
            madm<DT>(mod, TMP(0) | mme5,   TMP(0) | mme1,   TMP(3) | mme4,   TMP(2) | mme3, loc);
            madm<DT>(mod, TMP(1) | mme6,      num | nomme,  -denom | nomme,  TMP(0) | mme5, loc);
            madm<DT>(mod,    dst | nomme,  TMP(0) | mme5,   TMP(1) | mme6,   TMP(2) | mme3, loc);

            mark(labelSkip);
            endif(cfmod, loc);
            break;
        case DataType::df:
            invm<DT>(mod | eo | flag,         dst | mme0,      num | nomme,   denom | nomme, loc);
            if_(cfmod | ~flag, labelSkip, loc);

            madm<DT>(mod, TMP(0) | mme1,     zero | nomme,     num | nomme,     dst | mme0, loc);
            madm<DT>(mod, TMP(1) | mme2,      one | nomme,  -denom | nomme,     dst | mme0, loc);
            madm<DT>(mod, TMP(2) | mme3,      num | nomme,  -denom | nomme,  TMP(0) | mme1, loc);
            madm<DT>(mod, TMP(3) | mme4,      dst | mme0,   TMP(1) | mme2,      dst | mme0, loc);
            madm<DT>(mod, TMP(4) | mme5,      one | nomme,  -denom | nomme,  TMP(3) | mme4, loc);
            madm<DT>(mod,    dst | mme6,      dst | mme0,   TMP(1) | mme2,   TMP(3) | mme4, loc);
            madm<DT>(mod, TMP(0) | mme7,   TMP(0) | mme1,   TMP(2) | mme3,   TMP(3) | mme4, loc);
            madm<DT>(mod, TMP(3) | mme0,   TMP(3) | mme4,      dst | mme6,   TMP(4) | mme5, loc);
            madm<DT>(mod, TMP(2) | mme1,      num | nomme,  -denom | nomme,  TMP(0) | mme7, loc);
            madm<DT>(mod,    dst | nomme,  TMP(0) | mme7,   TMP(2) | mme1,   TMP(3) | mme0, loc);

            mark(labelSkip);
            endif(cfmod, loc);
            break;
        default:
#ifdef NGEN_SAFE
            throw invalid_type_exception();
#endif
            break;
    }
}

// IEEE 754-compliant reciprocal math macro sequence.
//   Requires GRF initialized with 1.0, as well as 3 temporary GRFs.
//   dst and src must be distinct GRFs.
template <typename DT = void, typename A>
void inv_ieee(const InstructionModifier &mod, FlagRegister flag, RegData dst, RegData src, RegData one,
              const A &tmp, InstructionModifier cfmod = InstructionModifier(), SourceLocation loc = {})
{
    DataType dt = getDataType<DT>();
    if (dt == DataType::invalid)
        dt = dst.getType();
    if (cfmod.getExecSize() == 0)
        cfmod = mod;

    Label labelSkip;

    switch (dt) {
        case DataType::hf:
            inv<DT>(mod, dst, src, loc);
            break;
        case DataType::f:
            invm<DT>(mod | eo | flag,         dst | mme0,      one | nomme,     src | nomme, loc);
            if_(cfmod | ~flag, labelSkip, loc);

            madm<DT>(mod, TMP(1) | mme2,      one | nomme,    -src | nomme,     dst | mme0, loc);
            madm<DT>(mod, TMP(2) | mme3,      dst | mme0,   TMP(1) | mme2,      dst | mme0, loc);
            madm<DT>(mod, TMP(0) | mme5,      dst | mme0,   TMP(1) | mme2,   TMP(2) | mme3, loc);
            madm<DT>(mod, TMP(1) | mme6,      one | nomme,    -src | nomme,  TMP(0) | mme5, loc);
            madm<DT>(mod,    dst | nomme,  TMP(0) | mme5,   TMP(1) | mme6,   TMP(2) | mme3, loc);

            mark(labelSkip);
            endif(cfmod, loc);
            break;
        case DataType::df:
            invm<DT>(mod | eo | flag,        dst | mme0,      one | nomme,     src | nomme, loc);
            if_(cfmod | ~flag, labelSkip, loc);

            madm<DT>(mod, TMP(0) | mme2,     one | nomme,    -src | nomme,     dst | mme0, loc);
            madm<DT>(mod, TMP(1) | mme4,     dst | mme0,   TMP(0) | mme2,      dst | mme0, loc);
            madm<DT>(mod, TMP(2) | mme5,     one | nomme,    -src | nomme,  TMP(1) | mme4, loc);
            madm<DT>(mod,    dst | mme6,     dst | mme0,   TMP(0) | mme2,   TMP(1) | mme4, loc);
            madm<DT>(mod, TMP(1) | mme0,  TMP(1) | mme4,      dst | mme6,   TMP(2) | mme5, loc);
            madm<DT>(mod, TMP(0) | mme1,     one | nomme,    -src | nomme,     dst | mme6, loc);
            madm<DT>(mod,    dst | nomme,    dst | mme6,   TMP(0) | mme1,   TMP(1) | mme0, loc);

            mark(labelSkip);
            endif(cfmod, loc);
            break;
        default:
#ifdef NGEN_SAFE
            throw invalid_type_exception();
#endif
            break;
    }
}

// IEEE 754-compliant square root macro sequence.
//   Requires GRFs initialized with 0.0 and 0.5 (also 1.0 for double precision),
//     and temporary GRFs (3 for single precision, 4 for double precision).
//   dst and src must be distinct GRFs.
template <typename DT = void, typename A>
void sqt_ieee(const InstructionModifier &mod, FlagRegister flag, RegData dst, RegData src,
              RegData zero, RegData oneHalf, RegData one, const A &tmp, InstructionModifier cfmod = InstructionModifier(),
              SourceLocation loc = {})
{
    DataType dt = getDataType<DT>();
    if (dt == DataType::invalid)
        dt = dst.getType();
    if (cfmod.getExecSize() == 0)
        cfmod = mod;

    Label labelSkip;

    switch (dt) {
        case DataType::hf:
            sqt<DT>(mod, dst, src, loc);
            break;
        case DataType::f:
            rsqtm<DT>(mod | eo | flag,        dst | mme0,       src | nomme, loc);
            if_(cfmod | ~flag, labelSkip, loc);

            madm<DT>(mod, TMP(0) | mme1,     zero | nomme,  oneHalf | nomme,     dst | mme0, loc);
            madm<DT>(mod, TMP(1) | mme2,     zero | nomme,      src | nomme,     dst | mme0, loc);
            madm<DT>(mod, TMP(2) | mme3,  oneHalf | nomme,  -TMP(1) | mme2,   TMP(0) | mme1, loc);
            madm<DT>(mod, TMP(0) | mme4,   TMP(0) | mme1,    TMP(2) | mme3,   TMP(0) | mme1, loc);
            madm<DT>(mod,    dst | mme5,   TMP(1) | mme2,    TMP(2) | mme3,   TMP(1) | mme2, loc);
            madm<DT>(mod, TMP(2) | mme6,      src | nomme,     -dst | mme5,      dst | mme5, loc);
            madm<DT>(mod,    dst | nomme,     dst | mme5,    TMP(0) | mme4,   TMP(2) | mme6, loc);

            mark(labelSkip);
            endif(cfmod, loc);
            break;
        case DataType::df:
            rsqtm<DT>(mod | eo | flag,        dst | mme0,       src | nomme, loc);
            if_(cfmod | ~flag, labelSkip, loc);

            madm<DT>(mod, TMP(0) | mme1,     zero | mme0,   oneHalf | nomme,     dst | mme0, loc);
            madm<DT>(mod, TMP(1) | mme2,     zero | mme0,       src | nomme,     dst | mme0, loc);
            madm<DT>(mod, TMP(2) | mme3,  oneHalf | nomme,  -TMP(1) | mme2,   TMP(0) | mme1, loc);
            madm<DT>(mod, TMP(3) | mme4,      one | nomme,  oneHalf | nomme,     dst | nomme, loc);
            madm<DT>(mod, TMP(3) | mme5,      one | nomme,   TMP(3) | mme4,   TMP(2) | mme3, loc);
            madm<DT>(mod,    dst | mme6,     zero | mme0,    TMP(2) | mme3,   TMP(1) | mme2, loc);
            madm<DT>(mod, TMP(2) | mme7,     zero | mme0,    TMP(2) | mme3,   TMP(0) | mme1, loc);
            madm<DT>(mod,    dst | mme6,   TMP(1) | mme2,    TMP(3) | mme5,      dst | mme6, loc);
            madm<DT>(mod, TMP(3) | mme5,   TMP(0) | mme1,    TMP(3) | mme5,   TMP(2) | mme7, loc);
            madm<DT>(mod, TMP(0) | mme1,      src | nomme,     -dst | mme6,      dst | mme6, loc);
            madm<DT>(mod,    dst | nomme,     dst | mme6,    TMP(0) | mme1,   TMP(3) | mme5, loc);

            mark(labelSkip);
            endif(cfmod, loc);
            break;
        default:
#ifdef NGEN_SAFE
            throw invalid_type_exception();
#endif
            break;
    }
}

#undef TMP

// Thread spawner messages.
void threadend(const InstructionModifier &mod, RegData r0_info = {}, SourceLocation loc = {})
{
#if XE4
    if (hardware >= HW::Xe4) {
        if (r0_info.isInvalid()) r0_info = SRF(0);
        sendgx(1 | EOT | mod, SharedFunction::gtwy, null, RegisterRange(r0_info, 1), 0, loc);
    } else
#endif
#if XE3P
    if (useEfficient64Bit)
        sendgx(1 | EOT | mod | NoMask, SharedFunction::gtwy, null, RegisterRange(r0_info, 1), 0, loc);
    else
#endif
    {
        auto sf = (hardware <= HW::XeHP) ? SharedFunction::ts
                                         : SharedFunction::gtwy;
        uint32_t exdesc = 0x20 | (static_cast<int>(sf) & 0xF);
        send(8 | EOT | mod | NoMask, null, r0_info, exdesc, 0x2000010, loc);
    }
}

void threadend(const RegData &r0_info = {}, SourceLocation loc = {}) {
    threadend(InstructionModifier(), r0_info, loc);
}

// Gateway messages.
void barriermsg(const InstructionModifier &mod, Register header = {}, SourceLocation loc = {})
{
#if XE4
    if (hardware >= HW::Xe4) {
        if (header.isInvalid()) header = SRF(0);
        sendgx(mod, SharedFunction::gtwy, null, RegisterRange(header, 1), 4, loc);
    } else
#endif
#if XE3P
    if (useEfficient64Bit) {
        if (header.isInvalid()) header = GRF(0);
        sendgx(1 | mod | NoMask, SharedFunction::gtwy, null, RegisterRange(header, 1), 4, loc);
    } else
#endif
    {
        uint32_t exdesc = static_cast<int>(SharedFunction::gtwy) & 0xF;
        send(1 | mod | NoMask, null, header, exdesc, 0x2000004, loc);
    }
}

void barriermsg(Register header = {}, SourceLocation loc = {}) { barriermsg(InstructionModifier(), header, loc); }

// Prepare barrier header.
void barrierheader(const Register &header, Register r0_info = {}, SourceLocation loc = {})
{
#if XE4
    if (hardware >= HW::Xe4) {
        if (r0_info.isInvalid()) r0_info = SRF(0);
        mov<uint32_t>(1, header, r0_info);
        return;
    }
#endif
    if (r0_info.isInvalid()) r0_info = GRF(0);
#if XE3P
    if (useEfficient64Bit)
        mov<uint32_t>(1 | NoMask, header[2], r0_info[2], loc);
    else
#endif
    if (hardware >= HW::XeHPG) {
        mov(1 | NoMask, header.hf(4), Immediate::hf(0), loc);
        mov(2 | NoMask, header.ub(10)(1), r0_info.ub(11)(0), loc);
    } else
        and_(8 | NoMask, header.ud(), r0_info.ud(2), uint32_t((hardware >= HW::Gen11) ? 0x7F000000 : 0x8F000000), loc);
}

void barrierheader(const Register &header, uint32_t threadCount, Register r0_info = {}, SourceLocation loc = {})
{
#if XE4
    if (hardware >= HW::Xe4) {
        mov<uint32_t>(1, header, threadCount << 24, loc);
        return;
    }
#endif
    if (r0_info.isInvalid()) r0_info = GRF(0);
#if XE3P
    if (useEfficient64Bit)
        mov(1 | NoMask, header.ud(2), threadCount << 24, loc);
    else
#endif
    if (hardware >= HW::XeHPG)
        mov(1 | NoMask, header.ud(2), (threadCount << 24) | (threadCount << 16), loc);
    else {
        and_(8 | NoMask, header.ud(), r0_info.ud(2), uint32_t((hardware >= HW::Gen11) ? 0x7F000000 : 0x8F000000), loc);
        mov(1 | NoMask, header.ub(9), 0x80 | (threadCount & 0x7F), loc);
    }
}

void barriersignal(const InstructionModifier &mod = {}, const GRF &temp = {}, Register r0_info = {}, SourceLocation loc = {})
{
#if XE3P
    if (useEfficient64Bit)
        barriermsg(mod, r0_info, loc);
    else
#endif
    {
        barrierheader(temp, r0_info, loc);
        barriermsg(mod, temp, loc);
    }
}

void barriersignal(const InstructionModifier &mod, const GRF &temp, uint32_t threadCount, Register r0_info = {}, SourceLocation loc = {}) {
    barrierheader(temp, threadCount, r0_info, loc);
    barriermsg(mod, temp, loc);
}

void barriersignal(const GRF &temp, Register r0_info = {}, SourceLocation loc = {}) { barriersignal(InstructionModifier(), temp, r0_info, loc); }
void barriersignal(const GRF &temp, uint32_t threadCount, Register r0_info = {}, SourceLocation loc = {}) { barriersignal(InstructionModifier(), temp, threadCount, r0_info, loc); }

// Named barriers.
void nbarriermsg(const InstructionModifier &mod, const GRF &header, SourceLocation loc = {}) {
#if XE3P
    if (useEfficient64Bit)
        sendgx(1 | mod | NoMask, SharedFunction::gtwy, null, RegisterRange(header, 1), 5, loc);
    else
#endif
        barriermsg(mod, header, loc);
}

void nbarriermsg(const GRF &header, SourceLocation loc = {}) { nbarriermsg(InstructionModifier(), header, loc); }

void nbarriercheck() {
#ifdef NGEN_SAFE
    if (hardware < HW::XeHPC)
        throw unsupported_message();
#if XE4
    if (hardware >= HW::Xe4)
        throw unsupported_message();
#endif
#endif
}

void barriersignal(const InstructionModifier &mod, uint32_t barrierID, const GRF &temp, const GRF &r0_info = r0, SourceLocation loc = {}) {
    nbarriercheck();
    mov(1 | NoMask, temp.uw(4), uint8_t(barrierID), loc);
    mov(2 | NoMask, temp.ub(10)(1), r0_info.ub(11)(0), loc);
    nbarriermsg(mod, temp, loc);
}

void barriersignal(const InstructionModifier &mod, uint32_t barrierID, const GRF &temp, BarrierType barrierType, uint32_t producers, uint32_t consumers, SourceLocation loc = {}) {
    nbarriercheck();
    mov(1 | NoMask, temp.ud(2), (barrierID & 0xFF) | (static_cast<uint32_t>(barrierType) << 14) | ((producers & 0xFF) << 16) | ((consumers & 0xFF) << 24), loc);
    nbarriermsg(mod, temp, loc);
}

void barriersignal(uint32_t barrierID, const GRF &temp, const GRF &r0_info = r0, SourceLocation loc = {}) { barriersignal(InstructionModifier(), barrierID, temp, r0_info, loc); }
void barriersignal(uint32_t barrierID, const GRF &temp, BarrierType barrierType, uint32_t producers, uint32_t consumers, SourceLocation loc = {}) { barriersignal(InstructionModifier(), barrierID, temp, barrierType, producers, consumers, loc); }

void barrierwait(SourceLocation loc = {}) {
    if (isGen12)
        sync.bar(NoMask, loc);
    else
        wait(NoMask, n0[0], loc);
}

void barrier(const InstructionModifier &mod = {}, const GRF &temp = {}, Register r0_info = {},
             SourceLocation loc = {}) {
    barriersignal(mod, temp, r0_info, loc);
    barrierwait(loc);
}

void barrier(const InstructionModifier &mod, const GRF &temp, uint32_t threadCount,
             Register r0_info = {}, SourceLocation loc = {}) {
    barriersignal(mod, temp, threadCount, r0_info, loc);
    barrierwait(loc);
}

void barrier(const GRF &temp, Register r0_info = {}, SourceLocation loc = {}) {
    barriersignal(InstructionModifier(), temp, r0_info, loc);
    barrierwait(loc);
}

void barrier(const GRF &temp, uint32_t threadCount, Register r0_info = {},
             SourceLocation loc = {}) {
    barriersignal(temp, threadCount, r0_info, loc);
    barrierwait(loc);
}

void barrier(const InstructionModifier &mod, uint32_t barrierID,
             const GRF &temp, const GRF &r0_info = r0, SourceLocation loc = {}) {
    barriersignal(mod, barrierID, temp, r0_info, loc);
    barrierwait(loc);
}

void barrier(const InstructionModifier &mod, uint32_t barrierID,
             const GRF &temp, BarrierType barrierType, uint32_t producers,
             uint32_t consumers, SourceLocation loc = {}) {
    barriersignal(mod, barrierID, temp, barrierType, producers, consumers, loc);
    barrierwait(loc);
}

void barrier(uint32_t barrierID, const GRF &temp, const GRF &r0_info = r0,
             SourceLocation loc = {}) {
    barriersignal(barrierID, temp, r0_info, loc);
    barrierwait(loc);
}

void barrier(uint32_t barrierID, const GRF &temp, BarrierType barrierType,
             uint32_t producers, uint32_t consumers, SourceLocation loc = {}) {
    barriersignal(barrierID, temp, barrierType, producers, consumers, loc);
    barrierwait(loc);
}

#if XE4
void cbarriersignal(InstructionModifier mod = {}, SRF header = SRF(0), SourceLocation loc = {}) {
#ifdef NGEN_SAFE
    if (hardware < HW::Xe4) throw unsupported_message();
#endif
    sendgx(mod, SharedFunction::gtwy, null, RegisterRange(header, 2), GatewayOpcode::cbar, loc);
}

void cbarrierwait(InstructionModifier mod = {}, SourceLocation loc = {}) {
    sync.barid(mod, 1, loc);
}

void cbarrier(InstructionModifier mod = {}, SRF header = SRF(0), SourceLocation loc = {}) {
    cbarriersignal(mod, header, loc);
    cbarrierwait(mod, loc);
}

void abarrierinit(InstructionModifier mod, SRF addr, SRF arrivalCount, bool completionFunc = false, SourceLocation loc = {}) {
    uint64_t desc = GatewayOpcode::abar_init;
    if (completionFunc) desc |= 0x200;
    sendgx(mod, SharedFunction::gtwy, null, RegisterRange(addr, 1), RegisterRange(arrivalCount, 1), desc, loc);
}
void abarrierinit(SRF addr, SRF arrivalCount, bool completionFunc = false, SourceLocation loc = {}) {
    abarrierinit(InstructionModifier(), addr, arrivalCount, completionFunc, loc);
}

void abarrierarrive(InstructionModifier mod, RegData dst, SRF addr, SRF arrivalCount, bool drop = false, bool cluster = false, SourceLocation loc = {}) {
    uint64_t desc = GatewayOpcode::abar_arrive;
    if (drop) desc |= 0x80;
    if (cluster) desc |= 0x200;
    sendgx(mod, SharedFunction::gtwy, dst, RegisterRange(addr, 1), RegisterRange(arrivalCount, 1), desc, loc);
}
void abarrierarrive(RegData dst, SRF addr, SRF arrivalCount, bool drop = false, bool cluster = false, SourceLocation loc = {}) {
    abarrierarrive(InstructionModifier(), dst, addr, arrivalCount, drop, cluster, loc);
}

void abarrierexpect(InstructionModifier mod, SRF addr, SRF ops, bool cluster = false, SourceLocation loc = {}) {
    uint64_t desc = GatewayOpcode::abar_expect;
    if (cluster) desc |= 0x200;
    sendgx(mod, SharedFunction::gtwy, null, RegisterRange(addr, 1), RegisterRange(ops, 1), desc, loc);
}
void abarrierexpect(SRF addr, SRF ops, bool cluster = false, SourceLocation loc = {}) {
    abarrierexpect(InstructionModifier(), addr, ops, cluster, loc);
}

void abarrierarriveexp(InstructionModifier mod, RegData dst, SRF addr, SRF ops, bool drop = false, bool cluster = false, SourceLocation loc = {}) {
    uint64_t desc = GatewayOpcode::abar_arrive_expect;
    if (drop) desc |= 0x80;
    if (cluster) desc |= 0x200;
    sendgx(mod, SharedFunction::gtwy, dst, RegisterRange(addr, 1), RegisterRange(ops, 1), desc, loc);
}
void abarrierarriveexp(RegData dst, SRF addr, SRF ops, bool drop = false, bool cluster = false, SourceLocation loc = {}) {
    abarrierarriveexp(InstructionModifier(), dst, addr, ops, drop, cluster, loc);
}

void abarriercomplete(InstructionModifier mod, SRF addr, SRF ops, bool cluster = false, SourceLocation loc = {}) {
    uint64_t desc = GatewayOpcode::abar_complete;
    if (cluster) desc |= 0x200;
    sendgx(mod, SharedFunction::gtwy, null, RegisterRange(addr, 1), RegisterRange(ops, 1), desc, loc);
}
void abarriercomplete(SRF addr, SRF ops, bool cluster = false, SourceLocation loc = {}) {
    abarriercomplete(InstructionModifier(), addr, ops, cluster, loc);
}

void abarrierpoll(InstructionModifier mod, RegData dst, SRF addr, SRF phase, SourceLocation loc = {}) {
    sendgx(mod, SharedFunction::gtwy, dst, RegisterRange(addr, 1), RegisterRange(phase, 1), GatewayOpcode::abar_test_poll, loc);
}
void abarrierpoll(RegData dst, SRF addr, SRF phase, SourceLocation loc = {}) {
    abarrierpoll(InstructionModifier(), dst, addr, phase, loc);
}

void abarriertry(InstructionModifier mod, int notifyReg, SRF addr, SRF phase, SourceLocation loc = {}) {
    uint64_t desc = GatewayOpcode::abar_try | (notifyReg << 8);
    sendgx(mod, SharedFunction::gtwy, null, RegisterRange(addr, 1), RegisterRange(phase, 1), desc, loc);
}
void abarriertry(int notifyReg, SRF addr, SRF phase, SourceLocation loc = {}) {
    abarriertry(InstructionModifier(), notifyReg, addr, phase, loc);
}

void abarrierinval(InstructionModifier mod, SRF addr, SourceLocation loc = {}) {
    sendgx(mod, SharedFunction::gtwy, null, RegisterRange(addr, 1), GatewayOpcode::abar_inval, loc);
}
void abarrierinval(SRF addr, SourceLocation loc = {}) {
    abarrierinval(InstructionModifier(), addr, loc);
}

void abarrierquery(InstructionModifier mod, SRF dst, SRF addr, SourceLocation loc = {}) {
    sendgx(mod, SharedFunction::gtwy, dst, RegisterRange(addr, 1), GatewayOpcode::abar_query, loc);
}
void abarrierquery(SRF dst, SRF addr, SourceLocation loc = {}) {
    abarrierquery(InstructionModifier(), dst, addr, loc);
}

void abarrierwait(InstructionModifier mod, uint32_t notifyReg, SourceLocation loc = {}) {
    sync.barid(mod, 2 + notifyReg, loc);
}
void abarrierwait(uint32_t notifyReg, SourceLocation loc = {}) {
    abarrierwait(InstructionModifier(), notifyReg, loc);
}
#endif /* XE4 */

void registerfence(const RegData &dst, SourceLocation loc = {})
{
    _lastFenceDst = dst;
    if (isGen12) {
        _lastFenceLabel = Label();
        mark(_lastFenceLabel);
    }
}

// Global memory fence.
void memfence(const InstructionModifier &mod, FenceScopeLSC scope, FlushTypeLSC flushing, const RegData &dst = NullRegister(), const RegData &header = GRF(0), SourceLocation loc = {})
{
    registerfence(dst, loc);

#if XE3P
    if (useEfficient64Bit) {
        uint32_t desc = 0x1F;
        desc |= static_cast<uint32_t>(flushing) << 8;
        desc |= static_cast<uint32_t>(scope) << 11;
#if XE4
        if (hardware >= HW::Xe4)
            sendgx(mod, SharedFunction::ugm, null, null, 0, desc, loc);
        else
#endif
        sendgx(1 | mod | NoMask, SharedFunction::ugm, null, RegisterRange(header, 1), desc, loc);
    } else
#endif
    if (hardware >= HW::XeHPG) {
        if (flushing == FlushTypeLSC::None && hardware == HW::XeHPG && scope > FenceScopeLSC::Subslice)
            flushing = static_cast<FlushTypeLSC>(6);    /* workaround for DG2 bug */

        uint32_t desc = 0x0210011F;
        desc |= static_cast<uint32_t>(scope) << 9;
        desc |= static_cast<uint32_t>(flushing) << 12;
        send(1 | mod | NoMask, SharedFunction::ugm, dst, header, null, 0, desc, loc);
    } else {
        const uint32_t exdesc = static_cast<int>(SharedFunction::dc0) & 0xF;
        send(8 | mod | NoMask, dst, header, exdesc, 0x219E000, loc);
    }
}

void memfence(const InstructionModifier &mod, FenceScopeLSC scope, FlushTypeLSC flushing, const RegData &dst, SourceLocation loc = {}) {
    memfence(mod, scope, flushing, dst, GRF(0), loc);
}

void memfence(const InstructionModifier &mod, FenceScopeLSC scope, FlushTypeLSC flushing, SourceLocation loc = {}) {
    memfence(mod, scope, flushing, NullRegister(), GRF(0), loc);
}

void memfence(const InstructionModifier &mod, const RegData &dst, const RegData &header, SourceLocation loc = {}) {
    memfence(mod, FenceScopeLSC::GPU, FlushTypeLSC::None, dst, header, loc);
}

void memfence(const InstructionModifier &mod, const RegData &dst, SourceLocation loc = {}) {
    memfence(mod, FenceScopeLSC::GPU, FlushTypeLSC::None, dst, GRF(0), loc);
}

void memfence(const InstructionModifier &mod, SourceLocation loc = {}) {
    memfence(mod, FenceScopeLSC::GPU, FlushTypeLSC::None, NullRegister(), GRF(0), loc);
}

void memfence(FenceScopeLSC scope, FlushTypeLSC flushing, const RegData &dst, const RegData &header, SourceLocation loc = {}) {
    memfence(InstructionModifier(), scope, flushing, dst, header, loc);
}

void memfence(FenceScopeLSC scope, FlushTypeLSC flushing, const RegData &dst, SourceLocation loc = {}) {
    memfence(InstructionModifier(), scope, flushing, dst, GRF(0), loc);
}

void memfence(FenceScopeLSC scope, FlushTypeLSC flushing, SourceLocation loc = {}) {
    memfence(InstructionModifier(), scope, flushing, NullRegister(), GRF(0), loc);
}

void memfence(const RegData &dst, const RegData &header, SourceLocation loc = {}) {
    memfence(InstructionModifier(), dst, header, loc);
}

void memfence(const RegData &dst, SourceLocation loc = {}) {
    memfence(InstructionModifier(), dst, GRF(0), loc);
}

void memfence(SourceLocation loc = {}) {
    memfence(InstructionModifier(), NullRegister(), GRF(0), loc);
}

// SLM-only memory fence.
void slmfence(const InstructionModifier &mod, const RegData &dst, const RegData &header, SourceLocation loc = {})
{
    registerfence(dst, loc);

#if XE4
    if (hardware >= HW::Xe4)
        sendgx(mod, SharedFunction::slm, null, null, 0, 0x1F, loc);
    else
#endif
#if XE3P
    if (useEfficient64Bit)
        sendgx(1 | mod | NoMask, SharedFunction::slm, null, RegisterRange(header, 1), 0x1F, loc);
    else
#endif
    if (hardware >= HW::XeHPG)
        send(1 | mod | NoMask, SharedFunction::slm, dst, header, null, 0, 0x210011F, loc);
    else {
        const uint32_t exdesc = static_cast<int>(SharedFunction::dc0) & 0xF;
        send(8 | mod | NoMask, dst, header, exdesc, 0x219E0FE, loc);
    }
}
void slmfence(const InstructionModifier &mod, const RegData &dst, SourceLocation loc = {}) { slmfence(mod, dst, GRF(0), loc); }
void slmfence(const InstructionModifier &mod, SourceLocation loc = {})                     { slmfence(mod, NullRegister(), GRF(0), loc); }
void slmfence(const RegData &dst, const RegData &header, SourceLocation loc = {})          { slmfence(InstructionModifier(), dst, header, loc); }
void slmfence(const RegData &dst, SourceLocation loc = {})                                 { slmfence(InstructionModifier(), dst, GRF(0), loc); }
void slmfence(SourceLocation loc = {})                                                     { slmfence(InstructionModifier(), NullRegister(), GRF(0), loc); }

// Wait on the last global memory or SLM fence.
void fencewait(SourceLocation loc = {})
{
    if (isGen12)
        fencedep(_lastFenceLabel, loc);
    else
        mov<uint32_t>(8 | NoMask, null, _lastFenceDst, loc);
}

#if XE4
// Async DMA.
void admall2r(InstructionModifier mod, ADMAOptions opts, Register payload, SourceLocation loc = {}) {
    opts.setOpcode(ADMAOpcode::linear_l2r);
    sendgx(mod, SharedFunction::dma, null, RegisterRange(payload, 0), opts.desc.all, loc);
}
void admall2r(ADMAOptions opts, Register payload, SourceLocation loc = {}) {
    admall2r(InstructionModifier(), opts, payload, loc);
}

void admall2g(InstructionModifier mod, ADMAOptions opts, Register payload, Register baseAddr = {}, SourceLocation loc = {}) {
    opts.setOpcode(ADMAOpcode::linear_l2g);
    sendgx(mod, SharedFunction::dma, null, RegisterRange(payload, 0), baseAddr.uq(), opts.desc.all, loc);
}
void admall2g(ADMAOptions opts, Register payload, Register baseAddr = {}, SourceLocation loc = {}) {
    admall2g(InstructionModifier(), opts, payload, baseAddr, loc);
}

void admalpf(InstructionModifier mod, ADMAOptions opts, Register payload, Register baseAddr = {}, SourceLocation loc = {}) {
    opts.setOpcode(ADMAOpcode::linear_prefetch);
    sendgx(mod, SharedFunction::dma, null, RegisterRange(payload, 0), baseAddr.uq(), opts.desc.all, loc);
}
void admalpf(ADMAOptions opts, Register payload, Register baseAddr = {}, SourceLocation loc = {}) {
    admalpf(InstructionModifier(), opts, payload, baseAddr, loc);
}

void admalg2l(InstructionModifier mod, ADMAOptions opts, Register payload, Register baseAddr = {}, SourceLocation loc = {}) {
    opts.setOpcode(ADMAOpcode::linear_g2l);
    sendgx(mod, SharedFunction::dma, null, RegisterRange(payload, 0), baseAddr.uq(), opts.desc.all, loc);
}
void admalg2l(ADMAOptions opts, Register payload, Register baseAddr = {}, SourceLocation loc = {}) {
    admalg2l(InstructionModifier(), opts, payload, baseAddr, loc);
}

void admalrl2r(InstructionModifier mod, ADMAReduction rop, ADMAOptions opts, Register payload, SourceLocation loc = {}) {
    opts.setOpcode(ADMAOpcode::linear_reduce_l2r);
    opts.setReductionOp(rop);
    sendgx(mod, SharedFunction::dma, null, RegisterRange(payload, 0), opts.desc.all, loc);
}
void admalrl2r(ADMAReduction rop, ADMAOptions opts, Register payload, SourceLocation loc = {}) {
    admalrl2r(InstructionModifier(), rop, opts, payload, loc);
}

void admalrl2g(InstructionModifier mod, ADMAReduction rop, ADMAOptions opts, Register payload, Register baseAddr = {}, SourceLocation loc = {}) {
    opts.setOpcode(ADMAOpcode::linear_reduce_l2g);
    opts.setReductionOp(rop);
    sendgx(mod, SharedFunction::dma, null, RegisterRange(payload, 0), baseAddr.uq(), opts.desc.all, loc);
}
void admalrl2g(ADMAReduction rop, ADMAOptions opts, Register payload, Register baseAddr = {}, SourceLocation loc = {}) {
    admalrl2g(InstructionModifier(), rop, opts, payload, baseAddr, loc);
}

void admatl2g(InstructionModifier mod, ADMAOptions opts, Register payload, Register tdesc, Register baseAddr = {}, SourceLocation loc = {}) {
    opts.setOpcode(ADMAOpcode::tensor_l2g);
    sendgx(mod, SharedFunction::dma, null, RegisterRange(payload, 0), RegisterRange(tdesc, 0), baseAddr.uq(), opts.desc.all, loc);
}
void admatl2g(ADMAOptions opts, Register payload, Register tdesc, Register baseAddr = {}, SourceLocation loc = {}) {
    admatl2g(InstructionModifier(), opts, payload, tdesc, baseAddr, loc);
}

void admatpf(InstructionModifier mod, ADMAOptions opts, Register payload, Register tdesc, Register baseAddr = {}, SourceLocation loc = {}) {
    opts.setOpcode(ADMAOpcode::tensor_prefetch);
    sendgx(mod, SharedFunction::dma, null, RegisterRange(payload, 0), RegisterRange(tdesc, 0), baseAddr.uq(), opts.desc.all, loc);
}
void admatpf(ADMAOptions opts, Register payload, Register tdesc, Register baseAddr = {}, SourceLocation loc = {}) {
    admatpf(InstructionModifier(), opts, payload, tdesc, baseAddr, loc);
}

void admatg2l(InstructionModifier mod, ADMAOptions opts, Register payload, Register tdesc, Register baseAddr = {}, SourceLocation loc = {}) {
    opts.setOpcode(ADMAOpcode::tensor_g2l);
    sendgx(mod, SharedFunction::dma, null, RegisterRange(payload, 0), RegisterRange(tdesc, 0), baseAddr.uq(), opts.desc.all, loc);
}
void admatg2l(ADMAOptions opts, Register payload, Register tdesc, Register baseAddr = {}, SourceLocation loc = {}) {
    admatg2l(InstructionModifier(), opts, payload, tdesc, baseAddr, loc);
}

void admatrl2g(InstructionModifier mod, ADMAReduction rop, ADMAOptions opts, Register payload, Register tdesc, Register baseAddr = {}, SourceLocation loc = {}) {
    opts.setOpcode(ADMAOpcode::tensor_reduce_l2g);
    opts.setReductionOp(rop);
    sendgx(mod, SharedFunction::dma, null, RegisterRange(payload, 0), RegisterRange(tdesc, 0), baseAddr.uq(), opts.desc.all, loc);
}
void admatrl2g(ADMAReduction rop, ADMAOptions opts, Register payload, Register tdesc, Register baseAddr = {}, SourceLocation loc = {}) {
    admatrl2g(InstructionModifier(), rop, opts, payload, tdesc, baseAddr, loc);
}

void admarl2g(InstructionModifier mod, ADMAOptions opts, AddressBase base, Register perLane, Register payload, Register baseAddr = {}, SourceLocation loc = {}) {
    opts.setOpcode(ADMAOpcode::row_l2g);
    opts.setAddressing(base);
    sendgx(mod, SharedFunction::dma, null, RegisterRange(perLane, 0), RegisterRange(payload, 0), baseAddr.uq(), opts.desc.all, loc);
}
void admarl2g(ADMAOptions opts, AddressBase base, Register perLane, Register payload, Register baseAddr = {}, SourceLocation loc = {}) {
    admarl2g(InstructionModifier(), opts, base, perLane, payload, baseAddr, loc);
}

void admarpf(InstructionModifier mod, ADMAOptions opts, AddressBase base, Register perLane, Register payload, Register baseAddr = {}, SourceLocation loc = {}) {
    opts.setOpcode(ADMAOpcode::row_prefetch);
    opts.setAddressing(base);
    sendgx(mod, SharedFunction::dma, null, RegisterRange(perLane, 0), RegisterRange(payload, 0), baseAddr.uq(), opts.desc.all, loc);
}
void admarpf(ADMAOptions opts, AddressBase base, Register perLane, Register payload, Register baseAddr = {}, SourceLocation loc = {}) {
    admarpf(InstructionModifier(), opts, base, perLane, payload, baseAddr, loc);
}

void admarg2l(InstructionModifier mod, ADMAOptions opts, AddressBase base, Register perLane, Register payload, Register baseAddr = {}, SourceLocation loc = {}) {
    opts.setOpcode(ADMAOpcode::row_g2l);
    opts.setAddressing(base);
    sendgx(mod, SharedFunction::dma, null, RegisterRange(perLane, 0), RegisterRange(payload, 0), baseAddr.uq(), opts.desc.all, loc);
}
void admarg2l(ADMAOptions opts, AddressBase base, Register perLane, Register payload, Register baseAddr = {}, SourceLocation loc = {}) {
    admarg2l(InstructionModifier(), opts, base, perLane, payload, baseAddr, loc);
}

void admarrl2g(InstructionModifier mod, ADMAReduction rop, ADMAOptions opts, AddressBase base, Register perLane, Register payload, Register baseAddr = {}, SourceLocation loc = {}) {
    opts.setOpcode(ADMAOpcode::row_reduce_l2g);
    opts.setReductionOp(rop);
    opts.setAddressing(base);
    sendgx(mod, SharedFunction::dma, null, RegisterRange(perLane, 0), RegisterRange(payload, 0), baseAddr.uq(), opts.desc.all, loc);
}
void admarrl2g(ADMAReduction rop, ADMAOptions opts, AddressBase base, Register perLane, Register payload, Register baseAddr = {}, SourceLocation loc = {}) {
    admarrl2g(InstructionModifier(), rop, opts, base, perLane, payload, baseAddr, loc);
}

// Async MMA.
void amma(InstructionModifier mod, AMMAParams params, AMMAOptions opts,
          Register desc, Register barriers, Register flags, SourceLocation loc = {})
{
    auto encodeInputType = [](DataType dt) {
        uint8_t table[32] = {0, 0, 0, 0, 14, 15, 0, 0, 0, 0, 2, 3, 5,  0,  0, 0,
                             1, 4, 0, 0, 0,  0,  0, 0, 0, 0, 6, 7, 12, 13, 0, 0};
        return table[static_cast<uint8_t>(dt) & 0x1F];
    };
    auto encodeAccType = [](DataType dt) {
        uint8_t table[32] = {0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        return table[static_cast<uint8_t>(dt) & 0x1F];
    };

    auto &o = opts.desc.amma;
    opts.desc.common.opcode = params.sparse ? AMMAOpcode::sparse_mma : AMMAOpcode::dense_mma;
    o.m = (params.m >> 5) - 1;
    o.n = (params.n >> 5) - 1;
    o.k = (params.k >> 5) - 1;
    o.atype = encodeInputType(params.atype);
    o.btype = encodeInputType(params.btype);
    o.ctype = encodeAccType(params.ctype);
    o.dtype = encodeAccType(params.dtype);
    sendgx(mod, SharedFunction::mma, null, RegisterRange(desc, 0), RegisterRange(barriers, 0), flags.uq(), opts.desc.all, loc);
}

void amma(AMMAParams params, AMMAOptions opts,
          Register desc, Register barriers, Register flags, SourceLocation loc = {}) {
    amma(InstructionModifier(), params, opts, desc, barriers, flags, loc);
}

void ammaerrorclr(InstructionModifier mod = {}, SourceLocation loc = {}) {
    sendgx(mod, SharedFunction::mma, null, null, 0, (AMMAOpcode::fp_error_clear | 0x180), loc);
}

void ammaerrorquery(InstructionModifier mod, Register dst, SourceLocation loc = {}) {
    sendgx(mod, SharedFunction::mma, dst, null, 0, AMMAOpcode::fp_error_query, loc);
}
void ammaerrorquery(Register dst, SourceLocation loc = {}) {
    ammaerrorquery(InstructionModifier(), dst, loc);
}

// Matrix Access.
void checkMatrixAccess(SendgMessageDescriptor desc) {
#ifdef NGEN_SAFE
    switch (static_cast<LSCOpcode>(desc.common.opcode)) {
        case LSCOpcode::load_matrix_unordered:
        case LSCOpcode::store_matrix_unordered:
            if (desc.matrix.vorient != 3) {
                throw invalid_matrix_access_exception();
            }
            break;
        default: break;
    }
    if (desc.matrixElementBitsReg() * desc.matrix.vlen * desc.matrix.alen > 1024) {
        throw invalid_matrix_access_exception();
    }
    if (static_cast<LSCOpcode>(desc.common.opcode) == LSCOpcode::reduce_matrix) {
        bool d16 = (desc.matrix.dataSize == 6);
        bool d32 = (desc.matrix.dataSize == 7);
        switch (desc.matrix.reductionOp) {
            case MatrixReduction::bfadd:
            case MatrixReduction::bfmin:
            case MatrixReduction::bfmax:
                if (!d16) throw invalid_matrix_access_exception();
                break;
            case MatrixReduction::inc_wrap:
            case MatrixReduction::dec_wrap:
                if (!d32) throw invalid_matrix_access_exception();
                break;
            default:
                if (!d16 && !d32) throw invalid_matrix_access_exception();
        }
    }
#endif
}

void loadmatrix(InstructionModifier mod, MatrixAccessOptions opts, Register dst, Register coord, Register desc, SourceLocation loc = {}) {
    opts.setOpcode(opts.unordered ? LSCOpcode::load_matrix_unordered : LSCOpcode::load_matrix);
    checkMatrixAccess(opts.desc);
    sendgx(mod, SharedFunction::slm, dst, coord, desc, opts.desc.all, loc);
}

void loadmatrix(MatrixAccessOptions opts, Register dst, Register coord, Register desc, SourceLocation loc = {}) {
    loadmatrix(InstructionModifier(), opts, dst, coord, desc, loc);
}

void storematrix(InstructionModifier mod, MatrixAccessOptions opts, Register coord, Register desc, Register data, SourceLocation loc = {}) {
    opts.setOpcode(opts.unordered ? LSCOpcode::store_matrix_unordered : LSCOpcode::store_matrix);
    checkMatrixAccess(opts.desc);
    sendgx(mod, SharedFunction::slm, NullRegister(), coord, RegisterRange(data, 1), desc, opts.desc.all, loc);
}

void storematrix(MatrixAccessOptions opts, Register coord, Register desc, Register data, SourceLocation loc = {}) {
    storematrix(InstructionModifier(), opts, coord, desc, data, loc);
}

void reducematrix(InstructionModifier mod, MatrixReduction rop, MatrixAccessOptions opts, Register coord, Register desc, Register data, SourceLocation loc = {}) {
    opts.setOpcode(LSCOpcode::reduce_matrix);
    opts.setReductionOp(rop);
    checkMatrixAccess(opts.desc);
    sendgx(mod, SharedFunction::slm, NullRegister(), coord, RegisterRange(data, 1), desc, opts.desc.all, loc);
}

void reducematrix(MatrixReduction rop, MatrixAccessOptions opts, Register coord, Register desc, Register data, SourceLocation loc = {}) {
    reducematrix(InstructionModifier(), rop, opts, coord, desc, data, loc);
}
#endif

// XeHP+ prologues.
void loadlid(int argBytes, int dims = 3, int simd = 8, const GRF &temp = GRF(127), int paddedSize = 0, SourceLocation loc = {})
{
#if XE4
    if (hardware >= HW::Xe4) return;
#endif
    if (hardware < HW::XeHP) return;

    if (paddedSize < 0)
        paddedSize = 12*16;
    const int grfSize = GRF::bytes(hardware);
    const int grfOW = grfSize / 16;
    int simdGRFs = (simd > 16 && grfSize < 64) ? 2 : 1;
    int insns = 0;
    const bool lsc = (hardware >= HW::XeHPG);
    auto tempAddr = temp[lsc ? 0 : 2];

    if (dims > 0) {
        auto dmSave = defaultModifier;
        defaultModifier |= NoMask | AutoSWSB;


#if XE3P
        if (useEfficient64Bit) {        /* to do: SIMD1 */
            uint16_t stride = simdGRFs * grfSize;
            auto base = s0.uq(2);
            and_<uint32_t>(1, acc0[0], r0.uw(4), 0xFF, loc);
            mov<uint64_t>(1, base, r0.uq(7), loc);
            mad<uint32_t>(1, acc0[0], uint16_t(argBytes), acc0[0], uint16_t(3 * stride), loc);
            mov<uint32_t>(16, r4, r1, loc);
            markIfUndefined(_interfaceLabels.crossThreadPatches[0]);
            add<uint32_t>(1, temp[0], acc0[0], Immediate::ud(0), loc);   /* relocation */
            load(1, r1, D32T(std::min(dims, 2) * stride / 4) | L1C_L3CC, A64_A32U, temp + base, loc);
            insns = 6;
            if (dims == 3) {
                load(1, GRF(1 + 2 * simdGRFs), D32T(stride / 4) | L1C_L3CC, A64_A32U, temp + base + stride/2, loc);
                insns++;
            }
        } else
#endif
        {
            insns = lsc ? 5 : 6;
            if (!lsc)
                mov<uint32_t>(8, temp, uint16_t(0), loc);
            and_<uint32_t>(1, temp[2], r0[0], uint32_t(~0x1F), loc);
            and_<uint16_t>(1, temp[0], r0[4], uint16_t(0xFF), loc);
            add<uint32_t>(1, temp[2], temp[2], uint16_t(argBytes), loc);
            markIfUndefined(_interfaceLabels.crossThreadPatches[0]);
            add<uint32_t>(1, temp[2], temp[2], Immediate::ud(0), loc);  /* relocation */
            if (simd == 1) {
                mad<uint32_t>(1, tempAddr, temp[2], temp.uw(0), uint16_t(grfSize), loc);
                lsc ? load(1, r1, D32T(4) | L1C_L3C,      A32,   temp, loc)
                    : load(8, r1, aligned_block_oword(1), A32NC, temp, loc);
            } else {
                mad<uint32_t>(1, tempAddr, temp[2], temp.uw(0), uint16_t(3 * simdGRFs * grfSize), loc);
                lsc ? load(1, r1, D32T(simdGRFs * ((dims == 1) ? 1 : 2) * grfOW * 4) | L1C_L3C,  A32,   temp, loc)
                    : load(8, r1, aligned_block_oword(simdGRFs * ((dims == 1) ? 1 : 2) * grfOW), A32NC, temp, loc);
                if (dims == 3) {
                    add<uint32_t>(1, tempAddr, tempAddr, uint16_t(2 * simdGRFs * grfSize), loc);
                    lsc ? load(1, GRF(1 + 2 * simdGRFs), D32T(grfOW * 4 * simdGRFs) | L1C_L3C,  A32,   temp, loc)
                        : load(8, GRF(1 + 2 * simdGRFs), aligned_block_oword(grfOW * simdGRFs), A32NC, temp, loc);
                    insns += 2;
                }
            }
        }

        defaultModifier = dmSave;
    }

    if (paddedSize > 0) {
        int nops = (paddedSize >> 4) - insns;
#ifdef NGEN_SAFE
        if (paddedSize & 0xF) throw invalid_operand_exception();
        if (nops < 0)         throw invalid_operand_exception();
#endif
        for (int i = 0; i < nops; i++)
            nop(loc);
    }

    markIfUndefined(_interfaceLabels.localIDsLoaded);

#if XE3P
    /* Workaround for incorrect NEO/XeSim handling of crossthread entrance */
    if (useEfficient64Bit)
        for (int i = 0; i < 4; i++)
            nop(loc);
#endif
}

void loadlid(int argBytes, int dims, int simd, const GRF &temp, SourceLocation loc = {}) { loadlid(argBytes, dims, simd, temp,     0, loc); }
void loadlid(int argBytes, int dims, int simd,                  SourceLocation loc = {}) { loadlid(argBytes, dims, simd, GRF(127), 0, loc); }
void loadlid(int argBytes, int dims,                            SourceLocation loc = {}) { loadlid(argBytes, dims, 8,    GRF(127), 0, loc); }
void loadlid(int argBytes,                                      SourceLocation loc = {}) { loadlid(argBytes, 3,    8,    GRF(127), 0, loc); }

void loadargs(const Register &base, int argRegs, const GRF &temp, bool inPrologue, SourceLocation loc = {})
{
    if (hardware < HW::XeHP) return;

    if (argRegs > 0) {
        const bool lsc = (hardware >= HW::XeHPG);
        auto tempAddr = temp[lsc ? 0 : 2];
        auto dst = base;
        auto dmSave = defaultModifier;
        defaultModifier |= NoMask | AutoSWSB;

#if XE4
        if (hardware >= HW::Xe4) {
            defaultModifier = dmSave | AutoSWSB;
            int offset = 0;
            while (argRegs > 0) {
                int nload = utils::rounddown_pow2(std::max(std::min(argRegs, 64), 16));
                load(dst, D32T(nload) | L1C_L2C_L3C, A64, SRF(16).uq() + offset, loc);
                argRegs -= nload;
                dst += nload;
                offset += (nload << 2);
            }
        } else
#endif
#if XE3P
        if (useEfficient64Bit) {        /* to do: SIMD1 */
            int offset = 0;
            auto offsetRT = s0.uq(3);
            auto addr = inPrologue ? r4 : temp;
            if (!inPrologue)
                mov<uint64_t>(1, addr, r0[7], loc);
            markIfUndefined(_interfaceLabels.crossThreadPatches[1]);
            mov(1, offsetRT, Immediate::uq(0), loc);     /* relocation */
            while (argRegs > 0) {
                int nload = std::min(utils::rounddown_pow2(argRegs), 8);
                int loadBytes = nload * GRF::bytes(hardware);
                load(1, dst, D64T(loadBytes >> 3) | L1C_L3CC, A64, addr + offsetRT + offset, loc);
                argRegs -= nload;
                dst += nload;
                offset += loadBytes;
            }
        } else
#endif
        {
            if (!lsc)
                mov<uint32_t>(8, temp, uint16_t(0), loc);
            and_<uint32_t>(1, tempAddr, r0[0], uint32_t(~0x1F), loc);
            markIfUndefined(_interfaceLabels.crossThreadPatches[1]);
            add<uint32_t>(1, tempAddr, tempAddr, Immediate::ud(0), loc);  /* relocation */
            while (argRegs > 0) {
                int nload = std::min(utils::rounddown_pow2(argRegs), lsc ? 8 : 4);
                int loadBytes = nload * GRF::bytes(hardware);
                lsc ? load(1, dst, D64T(loadBytes >> 3) | L1C_L3C,      A32,   temp, loc)
                    : load(8, dst, aligned_block_oword(loadBytes >> 4), A32NC, temp, loc);
                argRegs -= nload;
                dst += nload;
                if (argRegs > 0)
                    add<uint32_t>(1, tempAddr, tempAddr, uint32_t(loadBytes), loc);
            }
        }

        defaultModifier = dmSave;
    }

    markIfUndefined(_interfaceLabels.argsLoaded);
}

void loadargs(const Register &base, int argRegs, const GRF &temp, SourceLocation loc = {}) { loadargs(base, argRegs, temp,     true, loc); }
void loadargs(const Register &base, int argRegs, SourceLocation loc = {})                  { loadargs(base, argRegs, GRF(127), true, loc); }

void epilogue(int GRFCount, bool hasSLM, const RegData &r0_info, SourceLocation loc = {})
{
#if XE4
    if (hardware >= HW::Xe4) {
        threadend({}, loc);
        return;
    }
#endif
    GRF tmp0(GRFCount - 3);
    GRF tmp1(GRFCount - 2);
    GRF r0_copy(GRFCount - 4);

    bool doMemFence = false;
    bool doSLMFence = false;
    bool setAccToZero = false;

    switch (hardware) {
        case HW::XeLP:
        case HW::XeHP:
            doMemFence = true;
            doSLMFence = true;
            setAccToZero = true;
            break;
        case HW::XeHPG:
            setAccToZero = true;
            break;
        default: break;
    }

    if (!hasSLM) doSLMFence = false;

    int dwordsPerReg = GRF::bytes(hardware) / sizeof(uint32_t);
    mov<uint32_t>(dwordsPerReg, r0_copy, r0_info, loc);

    if (doMemFence) memfence(tmp0, r0_info, loc);
    if (doSLMFence) slmfence(tmp1, r0_info, loc);

    if (setAccToZero) {
        mov(16, acc0.f(), 0.f, loc);
        if (hardware == HW::XeHP) mov(16, acc2.f(), 0.f, loc);
    }

    if (doMemFence) wrdep(tmp0, loc);
    if (doSLMFence) wrdep(tmp1, loc);

    threadend(r0_copy, loc);
}


private:

struct Load {
    _self &parent;

    Load(_self *parent_) : parent(*parent_) {}

    template <typename DataSpec>
    void operator()(const InstructionModifier &mod, const RegData &dst, const DataSpec &spec, AddressBase base, const RegData &addr, SourceLocation loc = {})
    {
        this->operator()(SharedFunction::automatic, mod, dst, spec, base, GRFDisp(addr), loc);
    }

    template <typename DataSpec>
    void operator()(const InstructionModifier &mod, const RegData &dst, const DataSpec &spec, AddressBase base, const GRFDisp &addr, SourceLocation loc = {})
    {
        this->operator()(SharedFunction::automatic, mod, dst, spec, base, addr, loc);
    }

    template <typename DataSpec> void operator()(const RegData &dst, const DataSpec &spec, AddressBase base, const RegData &addr, SourceLocation loc = {}) {
        this->operator()(parent.defaultMods(), dst, spec, base, GRFDisp(addr), loc);
    }

    template <typename DataSpec> void operator()(const RegData &dst, const DataSpec &spec, AddressBase base, const GRFDisp &addr, SourceLocation loc = {}) {
        this->operator()(parent.defaultMods(), dst, spec, base, addr, loc);
    }

    template <typename DataSpec>
    void operator()(SharedFunction sfid, const InstructionModifier &mod, const RegData &dst, const DataSpec &spec, AddressBase base, const GRFDisp &addr, SourceLocation loc = {})
    {
#if XE3P
        if (parent.useEfficient64Bit) {
            SendgMessageDescriptor desc;
            int dstLen, src0Len;
            encodeLoadDescriptor(parent.hardware, desc, sfid, dstLen, src0Len, mod, spec, base, addr);
#if XE4
            if (parent.hardware < HW::Xe4)
#endif
            if (!dst.isNull() && dstLen > 0)
                parent.subdep(Operand::dst, GRFRange(dst.getBase(), dstLen));
            parent.sendgx(mod, sfid, dst, RegisterRange(addr.getBase(), src0Len), addr.getInd0(), desc.all, loc);
        } else
#endif
        {
            MessageDescriptor desc;
            ExtendedMessageDescriptor exdesc;

            if (sfid != SharedFunction::automatic)
                exdesc.parts.sfid = static_cast<unsigned>(sfid);
            encodeLoadDescriptors(parent.hardware, desc, exdesc, mod, dst, spec, base, addr);
            if (sfid != SharedFunction::automatic)
                exdesc.parts.sfid = static_cast<unsigned>(sfid);
            parent.send(mod, dst, addr.getBase(), exdesc.all, desc.all, loc);
        }
    }

    void ugm(const InstructionModifier &mod, const RegData &dst, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, SourceLocation loc = {})
    {
        this->operator()(SharedFunction::ugm, mod, dst, spec, base, addr, loc);
    }
    void ugm(const RegData &dst, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, SourceLocation loc = {})
    {
        ugm(parent.defaultMods(), dst, spec, base, addr, loc);
    }
    void ugml(const InstructionModifier &mod, const RegData &dst, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, SourceLocation loc = {})
    {
        this->operator()(SharedFunction::ugml, mod, dst, spec, base, addr, loc);
    }
    void ugml(const RegData &dst, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, SourceLocation loc = {})
    {
        ugml(parent.defaultMods(), dst, spec, base, addr, loc);
    }
    void tgm(const InstructionModifier &mod, const RegData &dst, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, SourceLocation loc = {})
    {
        this->operator()(SharedFunction::tgm, mod, dst, spec, base, addr, loc);
    }
    void tgm(const RegData &dst, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, SourceLocation loc = {})
    {
        tgm(parent.defaultMods(), dst, spec, base, addr, loc);
    }
    void slm(const InstructionModifier &mod, const RegData &dst, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, SourceLocation loc = {})
    {
        this->operator()(SharedFunction::slm, mod, dst, spec, base, addr, loc);
    }
    void slm(const RegData &dst, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, SourceLocation loc = {})
    {
        slm(parent.defaultMods(), dst, spec, base, addr, loc);
    }
};

struct Store {
    _self &parent;

    Store(_self *parent_) : parent(*parent_) {}

    template <typename DataSpec>
    void operator()(const InstructionModifier &mod, const DataSpec &spec, AddressBase base, const RegData &addr, const RegData &data, SourceLocation loc = {})
    {
        this->operator()(SharedFunction::automatic, mod, spec, base, GRFDisp(addr), data, loc);
    }

    template <typename DataSpec>
    void operator()(const InstructionModifier &mod, const DataSpec &spec, AddressBase base, const GRFDisp &addr, const RegData &data, SourceLocation loc = {})
    {
        this->operator()(SharedFunction::automatic, mod, spec, base, addr, data, {});
    }

    template <typename DataSpec> void operator()(const DataSpec &spec, AddressBase base, const RegData &addr, const RegData &data, SourceLocation loc = {}) {
        this->operator()(parent.defaultMods(), spec, base, addr, data, loc);
    }

    template <typename DataSpec> void operator()(const DataSpec &spec, AddressBase base, const GRFDisp &addr, const RegData &data, SourceLocation loc = {}) {
        this->operator()(parent.defaultMods(), spec, base, addr, data, loc);
    }

    template <typename DataSpec>
    void operator()(SharedFunction sfid, const InstructionModifier &mod, const DataSpec &spec, AddressBase base, const GRFDisp &addr, const RegData &data, SourceLocation loc = {})
    {
#if XE3P
        if (parent.useEfficient64Bit) {
            SendgMessageDescriptor desc;
            int src0Len, src1Len;
            encodeStoreDescriptor(parent.hardware, desc, sfid, src0Len, src1Len, mod, spec, base, addr);
            parent.sendgx(mod, sfid, NullRegister(), RegisterRange(addr.getBase(), src0Len), RegisterRange(data, src1Len), addr.getInd0(), desc.all, loc);
        } else
#endif
        {
            MessageDescriptor desc;
            ExtendedMessageDescriptor exdesc;

            if (sfid != SharedFunction::automatic)
                exdesc.parts.sfid = static_cast<unsigned>(sfid);
            encodeStoreDescriptors(parent.hardware, desc, exdesc, mod, spec, base, addr);
            if (sfid != SharedFunction::automatic)
                exdesc.parts.sfid = static_cast<unsigned>(sfid);
            parent.sends(mod, NullRegister(), addr.getBase(), data, exdesc.all, desc.all, loc);
        }
    }

    void ugm(const InstructionModifier &mod, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data, SourceLocation loc = {})
    {
        this->operator()(SharedFunction::ugm, mod, spec, base, addr, data, loc);
    }
    void ugm(DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data, SourceLocation loc = {}) {
        ugm(parent.defaultMods(), spec, base, addr, data, loc);
    }
    void ugml(const InstructionModifier &mod, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data, SourceLocation loc = {})
    {
        this->operator()(SharedFunction::ugml, mod, spec, base, addr, data, loc);
    }
    void ugml(DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data, SourceLocation loc = {}) {
        ugml(parent.defaultMods(), spec, base, addr, data, loc);
    }
    void tgm(const InstructionModifier &mod, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data, SourceLocation loc = {})
    {
        this->operator()(SharedFunction::tgm, mod, spec, base, addr, data, loc);
    }
    void tgm(DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data, SourceLocation loc = {}) {
        tgm(parent.defaultMods(), spec, base, addr, data, loc);
    }
    void slm(const InstructionModifier &mod, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data, SourceLocation loc = {})
    {
        this->operator()(SharedFunction::slm, mod, spec, base, addr, data, loc);
    }
    void slm(DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data, SourceLocation loc = {}) {
        slm(parent.defaultMods(), spec, base, addr, data, loc);
    }
};

struct Atomic_ {
    _self &parent;

    Atomic_(_self *parent_) : parent(*parent_) {}

    template <typename DataSpec>
    void operator()(AtomicOp op, const InstructionModifier &mod, const RegData &dst, const DataSpec &spec, AddressBase base, const RegData &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        this->operator()(SharedFunction::automatic, op, mod, dst, spec, base, GRFDisp(addr), data, loc);
    }
    template <typename DataSpec>
    void operator()(AtomicOp op, const InstructionModifier &mod, const DataSpec &spec, AddressBase base, const RegData &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        this->operator()(SharedFunction::automatic, op, mod, NullRegister(), spec, base, GRFDisp(addr), data, loc);
    }

    template <typename DataSpec>
    void operator()(AtomicOp op, const InstructionModifier &mod, const RegData &dst, const DataSpec &spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        this->operator()(SharedFunction::automatic, op, mod, dst, spec, base, addr, data, loc);
    }
    template <typename DataSpec>
    void operator()(AtomicOp op, const InstructionModifier &mod, const DataSpec &spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        this->operator()(SharedFunction::automatic, op, mod, NullRegister(), spec, base, addr, data, loc);
    }

    template <typename DataSpec> void operator()(AtomicOp op, const RegData &dst, const DataSpec &spec, AddressBase base, const RegData &addr, const RegData &data = NullRegister(), SourceLocation loc = {}) {
        this->operator()(op, parent.defaultMods(), dst, spec, base, addr, data, loc);
    }
    template <typename DataSpec> void operator()(AtomicOp op, const DataSpec &spec, AddressBase base, const RegData &addr, const RegData &data = NullRegister(), SourceLocation loc = {}) {
        this->operator()(op, parent.defaultMods(), spec, base, addr, data, loc);
    }
    template <typename DataSpec> void operator()(AtomicOp op, const RegData &dst, const DataSpec &spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {}) {
        this->operator()(op, parent.defaultMods(), dst, spec, base, addr, data, loc);
    }
    template <typename DataSpec> void operator()(AtomicOp op, const DataSpec &spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {}) {
        this->operator()(op, parent.defaultMods(), spec, base, addr, data, loc);
    }

    template <typename DataSpec>
    void operator()(SharedFunction sfid, AtomicOp op, const InstructionModifier &mod, const RegData &dst, const DataSpec &spec, AddressBase base, const GRFDisp &addr, const RegData &data, SourceLocation loc = {})
    {
#if XE3P
        if (parent.useEfficient64Bit) {
            SendgMessageDescriptor desc;
            int src0Len, src1Len;
            encodeAtomicDescriptor(parent.hardware, desc, sfid, src0Len, src1Len, op, mod, spec, base, addr);
            if (data.isNull())
                parent.sendgx(mod, sfid, dst, RegisterRange(addr.getBase(), src0Len), addr.getInd0(), desc.all, loc);
            else
                parent.sendgx(mod, sfid, dst, RegisterRange(addr.getBase(), src0Len), RegisterRange(data, src1Len), addr.getInd0(), desc.all, loc);
        } else
#endif
        {
            MessageDescriptor desc;
            ExtendedMessageDescriptor exdesc;

            if (sfid != SharedFunction::automatic)
                exdesc.parts.sfid = static_cast<unsigned>(sfid);
            encodeAtomicDescriptors(parent.hardware, desc, exdesc, op, mod, dst, spec, base, addr);
            if (sfid != SharedFunction::automatic)
                exdesc.parts.sfid = static_cast<unsigned>(sfid);
            if (data.isNull())
                parent.send(mod, dst, addr.getBase(), exdesc.all, desc.all, loc);
            else
                parent.sends(mod, dst, addr.getBase(), data, exdesc.all, desc.all, loc);
        }
    }

    void ugm(AtomicOp op, const InstructionModifier &mod, const RegData &dst, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        this->operator()(SharedFunction::ugm, op, mod, dst, spec, base, addr, data, loc);
    }
    void ugm(AtomicOp op, const InstructionModifier &mod, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        this->operator()(SharedFunction::ugm, op, mod, NullRegister(), spec, base, addr, data, loc);
    }
    void ugm(AtomicOp op, const RegData &dst, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        ugm(op, parent.defaultMods(), dst, spec, base, addr, data, loc);
    }
    void ugm(AtomicOp op, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        ugm(op, parent.defaultMods(), spec, base, addr, data, loc);
    }
    void ugml(AtomicOp op, const InstructionModifier &mod, const RegData &dst, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        this->operator()(SharedFunction::ugml, op, mod, dst, spec, base, addr, data, loc);
    }
    void ugml(AtomicOp op, const InstructionModifier &mod, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        this->operator()(SharedFunction::ugml, op, mod, NullRegister(), spec, base, addr, data, loc);
    }
    void ugml(AtomicOp op, const RegData &dst, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        ugml(op, parent.defaultMods(), dst, spec, base, addr, data, loc);
    }
    void ugml(AtomicOp op, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        ugml(op, parent.defaultMods(), spec, base, addr, data, loc);
    }
    void tgm(AtomicOp op, const InstructionModifier &mod, const RegData &dst, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        this->operator()(SharedFunction::tgm, op, mod, dst, spec, base, addr, data, loc);
    }
    void tgm(AtomicOp op, const InstructionModifier &mod, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        this->operator()(SharedFunction::tgm, op, mod, NullRegister(), spec, base, addr, data, loc);
    }
    void tgm(AtomicOp op, const RegData &dst, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        tgm(op, parent.defaultMods(), dst, spec, base, addr, data, loc);
    }
    void tgm(AtomicOp op, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        tgm(op, parent.defaultMods(), spec, base, addr, data, loc);
    }
    void slm(AtomicOp op, const InstructionModifier &mod, const RegData &dst, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        this->operator()(SharedFunction::slm, op, mod, dst, spec, base, addr, data, loc);
    }
    void slm(AtomicOp op, const InstructionModifier &mod, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        this->operator()(SharedFunction::slm, op, mod, NullRegister(), spec, base, addr, data, loc);
    }
    void slm(AtomicOp op, const RegData &dst, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        slm(op, parent.defaultMods(), dst, spec, base, addr, data, loc);
    }
    void slm(AtomicOp op, DataSpecLSC spec, AddressBase base, const GRFDisp &addr, const RegData &data = NullRegister(), SourceLocation loc = {})
    {
        slm(op, parent.defaultMods(), spec, base, addr, data, loc);
    }
};

public:

Load load;
Store store;
Atomic_ atomic;
