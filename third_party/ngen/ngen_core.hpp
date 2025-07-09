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

#ifndef NGEN_CORE_HPP
#define NGEN_CORE_HPP

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-int-conversion"
#endif

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include "ngen_config_internal.hpp"

#include "ngen_utils.hpp"

#ifndef NGEN_NO_OP_NAMES
#if not +0
#error Compile with -fno-operator-names [Linux/OS X] or without /Za [Windows] if you want to use and(), or(), xor(), or define NGEN_NO_OP_NAMES and use and_(), or_(), xor_().
#endif
#endif

#ifdef NGEN_ASM
#include <ostream>
#endif

#ifdef NGEN_SAFE
#include <stdexcept>
#endif

/*
  Syntax
  ------

  Register Syntax Overview
    r17                 Plain register
    r17.f(4)            -> r17.4:f
                        In fact, r17.4<0;1,0>:f, as subregisters default to
                          being scalar
    r17.sub<float>(4)   Same as above, allowing for C++ templating.
    r17.f()             -> r17.0:f (defaults to offset 0)
    r17.sub<float>()    Same as above
    r17.df(3)(8,8,1)    Register regioning (vertical stride, width, horizontal stride)
    r17.df(3)(8,1)      (Width, horiz. stride): vertical stride is inferred
    r17.df(3)(1)        Horizontal stride only: width, vertical stride inferred from execution size.
    r[a0.w(8)].f(4,4,1) Indirect addressing: VxH (if NGEN_SHORT_NAMES defined otherwise use indirect[a0...])
    r[a0.w(8)].f(4,1)   Indirect addressing: Vx1
    -r17.q(1)           Source modifier: negation
    abs(r17)            Source modifier: absolute value. Note that abs is defined in namespace ngen.
    -abs(r3)
    ~r17                Alternative syntax to -r17 for logical operations.
    r17 + 3             ...is r20. Operators ++ and += are defined similarly.

  Command Syntax Overview
    add(8, r3.f(0)(8,8,1), r9.f(0)(8,8,1), r12.f(0)(0,1,0))         ->   add (8) r3.0<8;8,1>:f r9.0<8;8,1>:f r12.f<0;1,0>
    add(8, r3.f(), r9.f(), r12.f())                                 Same as above. Register regions default to unit stride.
    add<float>(8, r3, r9, r12)                                      A default operand data type can be provided.
    add<uint32_t>(8, r3, r9, r12.uw(8)(0,1,0))                      Default operand types can be overridden.
    add<float>(8, r3, r9, 3.14159f)                                 The data type of scalar immediate values is inferred.
    add<int32_t>(8, r3, r9, int16_t(12))                            Here an int16_t immediate is mapped to the :w data type.
    mul<float>(8, r3, r9, Immediate::vf(-1.0,1.0,-1.0,1.25))        Vector immediates require helper functions.
    mov(8, r2.d(), Immediate::uv(7,6,5,4,3,2,1,0))
    mov(8, r2.d(), Immediate::v(7,-6,5,-4,3,-2,1,0))

  All modifiers for an instruction go in the first parameter, OR'ed together.
    add(8 | M0, ...)
    add(8 | W | ~f0.w(0) | sat, ...)            Use NoMask instead of W if NGEN_SHORT_NAMES not defined.
    add(8 | lt | f1_0, ...)
    add(8 | ~any2h | f1, ...)
 */

namespace NGEN_NAMESPACE {

#ifdef NGEN_SAFE
static constexpr bool _safe_ = 1;
#else
static constexpr bool _safe_ = 0;
#endif

// Forward declarations.
class RegData;
class Register;
class GRFDisp;
class Offset2D;
class Subregister;
class RegisterRegion;
class NullRegister;
class InstructionModifier;
struct Instruction12;
enum class Opcode;
#if XE4
enum class IndirectARF : uint16_t;
#endif

struct EncodingTag12;
static inline void encodeCommon12(Instruction12 &i, Opcode opcode, const InstructionModifier &mod, const RegData &dst, EncodingTag12 tag);
struct EncodingTagXeHPC;
static inline void encodeCommon12(Instruction12 &i, Opcode opcode, const InstructionModifier &mod, const RegData &dst, EncodingTagXeHPC tag);

// Exceptions, used when NGEN_SAFE is defined.

#ifdef NGEN_SAFE
class invalid_type_exception : public std::runtime_error {
public:
    invalid_type_exception(SourceLocation loc = {}) : std::runtime_error("Instruction does not support this type or combination of types" + loc.str(" at ")) {}
};
class invalid_object_exception : public std::runtime_error {
public:
    invalid_object_exception(SourceLocation loc = {}) : std::runtime_error("Object is invalid" + loc.str(" at ")) {}
};
class invalid_immediate_exception : public std::runtime_error {
public:
    invalid_immediate_exception(SourceLocation loc = {}) : std::runtime_error("Invalid immediate value" + loc.str(" at ")) {}
};
class invalid_modifiers_exception : public std::runtime_error {
public:
    invalid_modifiers_exception(SourceLocation loc = {}) : std::runtime_error("Invalid or conflicting modifiers" + loc.str(" at ")) {}
};
class invalid_operand_exception : public std::runtime_error {
public:
    invalid_operand_exception(SourceLocation loc = {}) : std::runtime_error("Invalid operand to instruction" + loc.str(" at ")) {}
};
class invalid_operand_count_exception : public std::runtime_error {
public:
    invalid_operand_count_exception(SourceLocation loc = {}) : std::runtime_error("Invalid operand count" + loc.str(" at ")) {}
};
#if XE4
class unsupported_scalar_operation_exception : public std::runtime_error {
public:
    unsupported_scalar_operation_exception() : std::runtime_error("Instruction does not support scalar operation") {}
};
#endif
class invalid_directive_exception : public std::runtime_error {
public:
    invalid_directive_exception(SourceLocation loc = {}) : std::runtime_error("Invalid directive" + loc.str(" at ")) {}
};
class invalid_arf_exception : public std::runtime_error {
public:
    invalid_arf_exception(SourceLocation loc = {}) : std::runtime_error("Invalid ARF specified" + loc.str(" at ")) {}
};
class invalid_register_file_exception : public std::runtime_error {
public:
    invalid_register_file_exception() : std::runtime_error("Invalid register file specified") {}
};
class grf_expected_exception : public std::runtime_error {
public:
    grf_expected_exception(SourceLocation loc = {}) : std::runtime_error("GRF expected, but found an ARF" + loc.str(" at ")) {}
};
class invalid_model_exception : public std::runtime_error {
public:
    invalid_model_exception(SourceLocation loc = {}) : std::runtime_error("Invalid addressing model specified" + loc.str(" at ")) {}
};
class invalid_load_store_exception : public std::runtime_error {
public:
    invalid_load_store_exception(SourceLocation loc = {}) : std::runtime_error("Invalid operands for load/store/atomic" + loc.str(" at ")) {}
};
class invalid_range_exception : public std::runtime_error {
public:
    invalid_range_exception(SourceLocation loc = {}) : std::runtime_error("Invalid register range" + loc.str(" at ")) {}
};
class invalid_region_exception : public std::runtime_error {
public:
    invalid_region_exception(SourceLocation loc = {}) : std::runtime_error("Unsupported register region" + loc.str(" at ")) {}
};
class missing_type_exception : public std::runtime_error {
public:
    missing_type_exception(SourceLocation loc = {}) : std::runtime_error("Operand or instruction is missing its type" + loc.str(" at ")) {}
};
class missing_src1_length_exception : public std::runtime_error {
public:
    missing_src1_length_exception(SourceLocation loc = {}) : std::runtime_error("src1 length must be specified" + loc.str(" at ")) {}
};
class read_only_exception : public std::runtime_error {
public:
    read_only_exception(SourceLocation loc = {}) : std::runtime_error("Memory model is read-only" + loc.str(" at ")) {}
};
class stream_stack_underflow : public std::runtime_error {
public:
    stream_stack_underflow(SourceLocation loc = {}) : std::runtime_error("Stream stack underflow occurred" + loc.str(" at ")) {}
};
class unfinished_stream_exception : public std::runtime_error {
public:
    unfinished_stream_exception(SourceLocation loc = {}) : std::runtime_error("An unfinished instruction stream is still active" + loc.str(" at ")) {}
};
class dangling_label_exception : public std::runtime_error {
public:
    dangling_label_exception(SourceLocation loc = {}) : std::runtime_error("A label was referenced, but its location was not defined" + loc.str(" at ")) {}
};
class multiple_label_exception : public std::runtime_error {
public:
    multiple_label_exception(SourceLocation loc = {}) : std::runtime_error("Label already has a location" + loc.str(" at ")) {}
};
class unsupported_instruction : public std::runtime_error {
public:
    unsupported_instruction(SourceLocation loc = {}) : std::runtime_error("Instruction is not supported by the chosen hardware" + loc.str(" at ")) {}
};
class unsupported_message : public std::runtime_error {
public:
    unsupported_message(SourceLocation loc = {}) : std::runtime_error("Message is not supported by the chosen hardware" + loc.str(" at ")) {}
};
class asm_unsupported_message : public std::runtime_error {
public:
    asm_unsupported_message() : std::runtime_error("Cannot format this message as assembly text") {}
};
class iga_align16_exception : public std::runtime_error {
public:
    iga_align16_exception(SourceLocation loc = {}) : std::runtime_error("Align16 not supported by the IGA assembler; use binary output" + loc.str(" at ")) {}
};
class sfid_needed_exception : public std::runtime_error {
public:
    sfid_needed_exception(SourceLocation loc = {}) : std::runtime_error("SFID must be specified on Gen12+" + loc.str(" at ")) {}
};
class invalid_execution_size_exception : public std::runtime_error {
public:
    invalid_execution_size_exception(SourceLocation loc = {}) : std::runtime_error("Invalid execution size" + loc.str(" at ")) {}
};
class invalid_address_mode_exception : public std::runtime_error {
public:
    invalid_address_mode_exception(SourceLocation loc = {}) : std::runtime_error("Invalid address mode" + loc.str(" at ")) {}
};
class invalid_address_modifier_exception : public std::runtime_error {
public:
#if XE3P
    invalid_address_modifier_exception(SourceLocation loc = {}) : std::runtime_error("Invalid address offset or scaling factor" + loc.str(" at ")) {}
#else
    invalid_address_modifier_exception(SourceLocation loc = {}) : std::runtime_error("Invalid address offset" + loc.str(" at ")) {}
#endif
};
#if XE3P
class limited_to_256_grf_exception : public std::runtime_error {
public:
    limited_to_256_grf_exception(SourceLocation loc = {}) : std::runtime_error("This instruction only supports r0-r255" + loc.str(" at ")) {}
};
class r511_not_allowed_exception : public std::runtime_error {
public:
    r511_not_allowed_exception(SourceLocation loc = {}) : std::runtime_error("r511 cannot be used here" + loc.str(" at ")) {}
};
#endif
#if XE4
class invalid_64_bit_register_exception : public std::runtime_error {
public:
    invalid_64_bit_register_exception() : std::runtime_error("64-bit data types must start on an even register") {}
};
#endif
#endif

// Graphics core generations.
enum class Core {
    Unknown,
    Gen9,
    Gen10,
    Gen11,
    XeLP,
    Gen12LP = XeLP,
    XeHP,
    Gen12HP = XeHP,     /* Deprecated -- will be removed in the future */
    XeHPG,
    Gen12p7 = XeHPG,    /* Deprecated -- will be removed in the future */
    XeHPC,
    Gen12p8 = XeHPC,    /* Deprecated -- will be removed in the future */
    Xe2,
    Xe3,
#if XE3P
    Xe3p,
#endif
#if XE4
    Xe4,
#endif
};

typedef Core HW;

// Product and product families. Only product families with major EU differences are listed specifically.
// nGEN itself does not use this information currently, but users may query it
//   from the OpenCLCodeGenerator/LevelZeroCodeGenerator classes.
enum class ProductFamily : int {
    Unknown,
    GenericGen9,
    GenericGen10,
    GenericGen11,
    GenericXeLP,
    GenericGen12LP = GenericXeLP,
    GenericXeHP,
    GenericXeHPG,
    DG2,
    MTL,
    ARL,
    GenericXeHPC,
    PVC,
    PVCVG,
    GenericXe2,
    BMG,
    LNL,
    GenericXe3,
#if XE3P
    GenericXe3p,
#endif
#if XE4
    GenericXe4,
#endif
};

enum class PlatformType {Unknown, Integrated, Discrete};

struct Product {
    ProductFamily family;
    int stepping;
    PlatformType type;
};

static inline bool operator==(const Product &p1, const Product &p2) { return p1.family == p2.family && p1.stepping == p2.stepping && p1.type == p2.type; }
static inline bool operator!=(const Product &p1, const Product &p2) { return !(p1 == p2); }
static inline bool operator<(const Product &p1, const Product &p2) { return (p1.family < p2.family) || (p1.family == p2.family && p1.stepping < p2.stepping); }
static inline bool operator>(const Product &p1, const Product &p2) { return p2 < p1; }
static inline bool operator>=(const Product &p1, const Product &p2) { return !(p1 < p2); }
static inline bool operator<=(const Product &p1, const Product &p2) { return !(p2 < p1); }

static inline constexpr14 PlatformType getPlatformType(ProductFamily family) {
    switch (family) {
        // Guaranteed integrated
        case ProductFamily::GenericGen9:
        case ProductFamily::GenericGen10:
        case ProductFamily::GenericGen11:
        case ProductFamily::MTL:
        case ProductFamily::ARL:
        case ProductFamily::LNL:
            return PlatformType::Integrated;
        // Could be integrated or discrete
        case ProductFamily::GenericXeLP:
        case ProductFamily::GenericXeHPG:
        case ProductFamily::GenericXe2:
        case ProductFamily::GenericXe3:
#if XE3P
        case ProductFamily::GenericXe3p:
#endif
#if XE4
        case ProductFamily::GenericXe4:
#endif
            return PlatformType::Unknown;
        // Guaranteed discrete
        case ProductFamily::GenericXeHP:
        case ProductFamily::GenericXeHPC:
        case ProductFamily::DG2:
        case ProductFamily::PVC:
        case ProductFamily::PVCVG:
        case ProductFamily::BMG:
            return PlatformType::Discrete;
        case ProductFamily::Unknown:
            return PlatformType::Unknown;
    }
    return PlatformType::Unknown;
}

static inline constexpr14 ProductFamily genericProductFamily(HW hw)
{
    switch (hw) {
        case HW::Gen9:  return ProductFamily::GenericGen9;
        case HW::Gen10: return ProductFamily::GenericGen10;
        case HW::Gen11: return ProductFamily::GenericGen11;
        case HW::XeLP:  return ProductFamily::GenericXeLP;
        case HW::XeHP:  return ProductFamily::GenericXeHP;
        case HW::XeHPG: return ProductFamily::GenericXeHPG;
        case HW::XeHPC: return ProductFamily::GenericXeHPC;
        case HW::Xe2:   return ProductFamily::GenericXe2;
        case HW::Xe3:   return ProductFamily::GenericXe3;
#if XE3P
        case HW::Xe3p:  return ProductFamily::GenericXe3p;
#endif
#if XE4
        case HW::Xe4:   return ProductFamily::GenericXe4;
#endif
        default:        return ProductFamily::Unknown;
    }
}

static inline constexpr14 Core getCore(ProductFamily family)
{
#if XE4
    if (family >= ProductFamily::GenericXe4)   return Core::Xe4;
#endif
#if XE3P
    if (family >= ProductFamily::GenericXe3p)  return Core::Xe3p;
#endif
    if (family >= ProductFamily::GenericXe3)   return Core::Xe3;
    if (family >= ProductFamily::GenericXe2)   return Core::Xe2;
    if (family >= ProductFamily::GenericXeHPC) return Core::XeHPC;
    if (family >= ProductFamily::GenericXeHPG) return Core::XeHPG;
    if (family >= ProductFamily::GenericXeHP)  return Core::XeHP;
    if (family >= ProductFamily::GenericXeLP)  return Core::XeLP;
    if (family >= ProductFamily::GenericGen11) return Core::Gen11;
    if (family >= ProductFamily::GenericGen10) return Core::Gen10;
    if (family >= ProductFamily::GenericGen9)  return Core::Gen9;
    return Core::Unknown;
}

static inline constexpr14 bool hasSystolic(ProductFamily family)
{
    if (family == ProductFamily::MTL) return false;
    if (family == ProductFamily::PVCVG) return false;
    return (family >= ProductFamily::GenericXeHP);
}

// Stepping IDs.
enum {
    SteppingPVCXTA0 = 3,
    SteppingPVCXTB0 = 5,
    SteppingPVCXTB4 = 7,
};

// Data types. Bits[0:4] are the ID, bits[5:7] hold log2(width in bits).
enum class DataType : uint8_t {
    ud = 0xA0,
    d  = 0xA1,
    uw = 0x82,
    w  = 0x83,
    ub = 0x64,
    b  = 0x65,
    df = 0xC6,
    f  = 0xA7,
    uq = 0xC8,
    q  = 0xC9,
    hf = 0x8A,
    bf = 0x8B,
    uv = 0xAD,
    v  = 0xAE,
    vf = 0xAF,
    bf8 = 0x6C,
    tf32 = 0xB0,
    hf8 = 0x71,
    u4 = 0x5C,
    s4 = 0x5D,
    u2 = 0x3E,
    s2 = 0x3F,
#if XE3P
    e2m1 = 0x5A,
    e3m0 = 0x5B,
#endif
#if XE4
    e5m2 = bf8,
    e4m3 = hf8,
    s8 = b,
    u8 = ub,
    s16 = w,
    u16 = uw,
    f16 = hf,
    bf16 = bf,
    s32 = d,
    u32 = ud,
    b32 = ud,
    f32 = f,
    s64 = q,
    u64 = uq,
    b64 = uq,
    f64 = df,
    u16v2 = 0xB2,
    s16v2 = 0xB3,
    b16v2 = u16v2,
    u8v4 = 0xB4,
    b8v4 = u8v4,
    s8v4 = 0xB5,
    f16v2 = 0xB6,
    bf16v2 = 0xB7,
#endif
    invalid = 0x60
};

#ifdef NGEN_ASM
static inline std::ostream &operator<<(std::ostream &str, DataType type)
{
#if XE3P
    static const char *names[32] = {"ud",   "d",   "uw", "w", "ub", "b", "df", "f", "uq", "q", "hf",   "bf",   "bf8", "uv", "v",  "vf",
                                    "tf32", "hf8", "",   "",  "",   "",  "",   "",  "",   "",  "e2m1", "e3m0", "u4",  "s4", "u2", "s2"};
#else
    static const char *names[32] = {"ud",   "d",   "uw", "w", "ub", "b", "df", "f", "uq", "q", "hf", "bf", "bf8", "uv", "v",  "vf",
                                    "tf32", "hf8", "",   "",  "",   "",  "",   "",  "",   "",  "",   "",   "u4",  "s4", "u2", "s2"};
#endif
    str << names[static_cast<uint8_t>(type) & 0x1F];
    return str;
}

#if XE4
struct DataTypeXe4 {
    DataType dt;
};

static inline std::ostream &operator<<(std::ostream &str, DataTypeXe4 type)
{
    static const char *names[32] = {"u32",  "s32", "u16",   "s16",   "u8",   "s8",   "f64",   "f32",    "u64", "s64", "f16",  "bf16", "bf8", "", "", "",
                                    "tf32", "hf8", "u16v2", "s16v2", "u8v4", "s8v4", "f16v2", "bf16v2", "",    "",    "e2m1", "e3m0", "",    "", "", ""};
    str << names[static_cast<uint8_t>(type.dt) & 0x1F];
    return str;
}
#endif
#endif

static inline constexpr   int getLog2Bits(DataType type)               { return static_cast<int>(type) >> 5; }
static inline constexpr14 int getLog2Bytes(DataType type)              { return std::max<int>(getLog2Bits(type) - 3, 0); }
static inline constexpr14 int getLog2Dwords(DataType type)             { return std::max<int>(getLog2Bits(type) - 5, 0); }
static inline constexpr14 int log2ElementsPerByte(DataType type)       { return std::max<int>(3 - getLog2Bits(type), 0); }
static inline constexpr   int getBits(DataType type)                   { return 1 << getLog2Bits(type); }
static inline constexpr14 int getBytes(DataType type)                  { return 1 << getLog2Bytes(type); }
static inline constexpr14 int getDwords(DataType type)                 { return 1 << getLog2Dwords(type); }
static inline constexpr14 int elementsPerByte(DataType type)           { return 1 << log2ElementsPerByte(type); }

static inline constexpr bool isSigned(DataType type)
{
    return !(type == DataType::u2 || type == DataType::u4 || type == DataType::ub
          || type == DataType::uw || type == DataType::ud || type == DataType::uq);
}

template <typename T> static inline DataType getDataType() { return DataType::invalid; }

template <> inline DataType getDataType<uint64_t>() { return DataType::uq; }
template <> inline DataType getDataType<int64_t>()  { return DataType::q;  }
template <> inline DataType getDataType<uint32_t>() { return DataType::ud; }
template <> inline DataType getDataType<int32_t>()  { return DataType::d;  }
template <> inline DataType getDataType<uint16_t>() { return DataType::uw; }
template <> inline DataType getDataType<int16_t>()  { return DataType::w;  }
template <> inline DataType getDataType<uint8_t>()  { return DataType::ub; }
template <> inline DataType getDataType<int8_t>()   { return DataType::b;  }
template <> inline DataType getDataType<double>()   { return DataType::df; }
template <> inline DataType getDataType<float>()    { return DataType::f;  }
#ifdef NGEN_HALF_TYPE
template <> inline DataType getDataType<half>()     { return DataType::hf; }
#endif
#ifdef NGEN_BFLOAT16_TYPE
template <> inline DataType getDataType<bfloat16>() { return DataType::bf; }
#endif
#ifdef NGEN_BFLOAT8_TYPE
template <> inline DataType getDataType<bfloat8>() { return DataType::bf8; }
#endif
#ifdef NGEN_HFLOAT8_TYPE
template <> inline DataType getDataType<hfloat8>() { return DataType::hf8; }
#endif
#ifdef NGEN_TFLOAT32_TYPE
template <> inline DataType getDataType<tfloat32>() { return DataType::tf32; }
#endif
#ifdef NGEN_UINT4_TYPE
template <> inline DataType getDataType<uint4>() { return DataType::u4; }
#endif
#ifdef NGEN_INT4_TYPE
template <> inline DataType getDataType<int4>() { return DataType::s4; }
#endif
#ifdef NGEN_UINT2_TYPE
template <> inline DataType getDataType<uint2>() { return DataType::u2; }
#endif
#ifdef NGEN_INT2_TYPE
template <> inline DataType getDataType<int2>() { return DataType::s2; }
#endif
#if XE3P
#ifdef NGEN_E2M1_TYPE
template <> inline DataType getDataType<e2m1>() { return DataType::e2m1; }
#endif
#ifdef NGEN_E3M0_TYPE
template <> inline DataType getDataType<e3m0>() { return DataType::e3m0; }
#endif
#endif

#if XE4
static inline constexpr14 DataType rawType(DataType dt) {
    switch (getLog2Bits(dt)) {
        case 6:  return DataType::u64;
        case 5:  return DataType::u32;
        case 4:  return DataType::u16;
        case 3:  return DataType::u8;
        default: return dt;
    }
}
#endif

// Math function codes.
enum class MathFunction : uint8_t {
    inv   = 0x1,
    log   = 0x2,
    exp   = 0x3,
    sqt   = 0x4,
    rsqt  = 0x5,
    sin   = 0x6,
    cos   = 0x7,
    fdiv  = 0x9,
    pow   = 0xA,
    idiv  = 0xB,
    iqot  = 0xC,
    irem  = 0xD,
    invm  = 0xE,
    rsqtm = 0xF,
#if XE3P
    tanh  = 0x19,
    sigm  = 0x1A,
#endif

};

static inline int mathArgCount(HW hw, MathFunction func)
{
#if XE3P
    if (hw >= HW::Xe3p) {
        static const char argCounts[16] = {0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 2, 2, 2, 1};
        return argCounts[static_cast<uint8_t>(func) & 0xF];
    }
#endif
    static const char argCounts[16] = {0, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 1};
    return argCounts[static_cast<uint8_t>(func) & 0xF];
}

#ifdef NGEN_ASM
static inline std::ostream &operator<<(std::ostream &str, MathFunction func)
{
#if XE3P
    static const char *names[32] = {"", "inv", "log", "exp", "sqt", "rsqt", "sin", "cos", "", "fdiv", "pow",  "idiv", "iqot", "irem", "invm", "rsqtm",
                                    "", "",    "",    "",    "",    "",     "",    "",    "", "tanh", "sigm", "",     "",     "",     "",     ""};
    str << names[static_cast<uint8_t>(func) & 0x1F];
#else
    static const char *names[16] = {"", "inv", "log", "exp", "sqt", "rsqt", "sin", "cos", "", "fdiv", "pow", "idiv", "iqot", "irem", "invm", "rsqtm"};
    str << names[static_cast<uint8_t>(func) & 0xF];
#endif
    return str;
}
#endif

static inline bool hasIEEEMacro(HW hw) {
    if (hw == HW::Gen11) return false;
    if (hw == HW::Gen12LP) return false;
    if (hw == HW::XeHPG) return false;
    return true;
}

// Sync function codes.
enum class SyncFunction : uint8_t {
    nop   = 0,
    allrd = 2,
    allwr = 3,
    flush = 12,
    bar   = 14,
    host  = 15,
#if XE4
    none     = nop,
    srcmsk   = allrd,
    dstmsk   = allwr,
    barid    = bar,
    barsrc   = 4,
    barflush = flush,
#endif
};

#ifdef NGEN_ASM
static inline std::ostream &operator<<(std::ostream &str, SyncFunction func)
{
    static const char *names[16] = {"nop", "", "allrd", "allwr", "", "", "", "", "", "", "", "", "flush", "", "bar", "host"};
    str << names[static_cast<uint8_t>(func) & 0xF];
    return str;
}

struct SyncFunctionXe4 { SyncFunction fc; };

static inline std::ostream &operator<<(std::ostream &str, SyncFunctionXe4 func)
{
    static const char *names[16] = {"none", "", "srcmsk", "dstmsk", "barsrc", "", "", "", "", "", "", "", "barflush", "", "barid", "host"};
    str << names[static_cast<uint8_t>(func.fc) & 0xF];
    return str;
}
#endif

#if XE3P
// Rounding types for dnscl.
enum class RoundingType : uint8_t {
    rne = 0,
    srnd = 1,
};

// Shuffle function codes.
enum class ShuffleFunction : uint8_t {
    idx4 = 0x6,
};

#ifdef NGEN_ASM
static inline std::ostream &operator<<(std::ostream &str, ShuffleFunction func)
{
    static const char *names[16] = {"", "", "", "", "", "", "idx4", "", "", "", "", "", "", "", "", ""};
    str << names[static_cast<uint8_t>(func) & 0xF];
    return str;
}
#endif

// LFSR function codes.
enum class LFSRFunction : uint8_t {
    b32 = 0,
    b16v2 = 1,
    b8v4 = 2,
};

#ifdef NGEN_ASM
static inline std::ostream &operator<<(std::ostream &str, LFSRFunction func)
{
    static const char *names[4] = {"b32", "b16v2", "b8v4", ""};
    str << names[static_cast<uint8_t>(func) & 0x3];
    return str;
}
#endif
#endif /* XE3P */

// Shared function IDs (SFIDs).
enum class SharedFunction : uint8_t {
    null = 0x0,
    smpl = 0x2,
    gtwy = 0x3,
    dc2 = 0x4,
    rc = 0x5,
    urb = 0x6,
    ts = 0x7,
    vme = 0x8,
    dcro = 0x9,
    dc0 = 0xA,
    pixi = 0xB,
    dc1 = 0xC,
    cre = 0xD,
    btd = 0x7,
    rta = 0x8,
    ugml = 0x1,
    tgm = 0xD,
    slm = 0xE,
    ugm = 0xF,
#if XE4
    mma = 0xA,
    dma = 0xB,
#endif
    automatic = 0xFF,

    // alias
    sampler = smpl,
    gateway = gtwy,
    spawner = ts,
};

#ifdef NGEN_ASM
static inline const char *getMnemonic(SharedFunction sfid, HW hw)
{
#if XE4
    static const char *namesXe4[16] = {
        "null", "ugml", "smpl", "gtwy", "dc2", "rc" , "urb", "btd",
        "rta" , "dcro", "mma" , "dma",  "dc1", "tgm", "slm", "ugm",
    };
    if (hw >= HW::Xe4) return namesXe4[static_cast<uint8_t>(sfid) & 0xF];
#endif
    static const char *names[16] = {
        "null", ""    , "smpl", "gtwy", "dc2", "rc" , "urb", "ts" ,
        "vme" , "dcro", "dc0" , "pixi", "dc1", "cre", ""   , ""   ,
    };
    static const char *namesLSC[16] = {
        "null", "ugml", "smpl", "gtwy", "dc2", "rc" , "urb", "btd",
        "rta" , "dcro", "dc0" , "pixi", "dc1", "tgm", "slm", "ugm",
    };
    const auto &table = (hw >= HW::XeHPG) ? namesLSC : names;
    return table[static_cast<uint8_t>(sfid) & 0xF];
}
#endif

// ARFs: high nybble of register # specifies type
enum class ARFType : uint8_t {
    null = 0,
    a    = 1,
    acc  = 2,
    f    = 3,
    ce   = 4,
    msg  = 5,
    sp   = 6,
    s    = 0x16,
    sr   = 7,
    cr   = 8,
    n    = 9,
    ip   = 10,
    tdr  = 11,
    tm   = 12,
    fc   = 13,
    dbg  = 15,
#if XE4
    lid  = 0x12,
    alm  = 0x13,
#endif
};

#ifdef NGEN_ASM
static inline std::ostream &operator<<(std::ostream &str, ARFType type)
{
    static const char *names[32] = {"null", "a", "acc", "f", "ce", "msg", "sp", "sr", "cr", "n", "ip", "tdr", "tm", "fc", "", "dbg",
#if XE4
                                    "",   "",    "lid", "alm", "", "",    "s",  "",   "",   "",  "",   "",    "",   "",   "", ""};
#else
                                    "",    "" ,  "",    "",  "",   "",    "s",  "",   "",   "",  "",   "",    "",   "",   "", ""};
#endif
    str << names[static_cast<uint8_t>(type) & 0x1F];
    return str;
}

enum class PrintDetail {
    base = 0, sub_no_type = 1, sub = 2, hs = 3, vs_hs = 4, full = 5,
#if XE4
    xe4 = -1, xe4_dst = -2, xe4_type = -3, xe4_hide = -4
#endif
};
#endif

// Invalid singleton class. Can be assigned to nGEN objects to invalidate them.
static constexpr class Invalid {} invalid{};

class LabelManager {
protected:
    uint32_t nextID;
    std::vector<uint32_t> targets;

    enum TargetConstants : uint32_t {
        noTarget = uint32_t(-1),
    };

public:
    LabelManager() : nextID(0) {}

    uint32_t getNewID() {
        targets.push_back(TargetConstants::noTarget);
        return nextID++;
    }

    bool hasTarget(uint32_t id) const {
        return (targets[id] != TargetConstants::noTarget);
    }

    void setTarget(uint32_t id, uint32_t target) {
#ifdef NGEN_SAFE
        if (hasTarget(id)) throw multiple_label_exception();
#endif
        targets[id] = target;
    }

    void offsetTarget(uint32_t id, uint32_t offset) {
#ifdef NGEN_SAFE
        if (!hasTarget(id)) throw dangling_label_exception();
#endif
        targets[id] += offset;
    }

    uint32_t getTarget(uint32_t id) const {
#ifdef NGEN_SAFE
        if (!hasTarget(id)) throw dangling_label_exception();
#endif
        return targets[id];
    }
};

// An object representing a label.
class Label {
protected:
    unsigned id : 31;
    unsigned uninit : 1;

public:
    Label() : id(0), uninit(true) {}

    uint32_t getID(LabelManager &man) {
        if (uninit) {
            id = man.getNewID();
            uninit = false;
        }
        return id;
    }

    bool defined(const LabelManager &man) const {
        return !uninit && man.hasTarget(id);
    }

    /* for compatibility with RegData */
    void fixup(HW hw, int execSize, int execWidth, DataType defaultType, int srcN, int arity) {}
    constexpr DataType getType() const { return DataType::invalid; }
    constexpr bool isScalar() const { return false; }

#ifdef NGEN_ASM
    static const bool emptyOp = false;
    inline void outputText(std::ostream &str, PrintDetail detail, LabelManager &man);

    friend inline bool operator==(const Label &r1, const Label &r2) {
        return !std::memcmp(&r1, &r2, sizeof(Label));
    }
    friend inline bool operator!=(const Label &r1, const Label &r2) { return !(r1 == r2); }
#endif
};

static inline bool operator==(const RegData &r1, const RegData &r2);
static inline bool operator!=(const RegData &r1, const RegData &r2);

// Special set of labels used for prologues.
struct InterfaceLabels {
    Label localIDsLoaded;
    Label argsLoaded;
    Label crossThreadPatches[2];
};

enum RegFile : unsigned {
    RegFileARF = 0,
    RegFileGRF = 1,
#if XE4
    RegFileSRF = 2,
#endif
    RegFileIMM = 3,
};

enum RegFile8 : unsigned {
    RegFile8ARF = RegFileARF,
    RegFile8GRF = RegFileGRF,
};

// Superclass for registers, subregisters, and register regions, possibly
// with source modifiers.
class RegData {
protected:
    unsigned base : 9;
    unsigned rf : 2;
      signed off : 11;
    unsigned mods : 2;
    unsigned type : 8;
    unsigned indirect : 1;
    unsigned vs : 7;
    unsigned width : 5;
    unsigned hs : 6;
    unsigned _pad2 : 12;
    unsigned invalid : 1;

    constexpr RegData(int base_, int rf_, int off_, bool indirect_, DataType type_, int vs_, int width_, int hs_)
        : base(base_), rf(rf_), off(off_), mods(0), type(static_cast<int>(type_)), indirect(indirect_), vs(vs_), width(width_), hs(hs_), _pad2(0), invalid(0) {}

public:
#ifdef NGEN_ASM
    static const bool emptyOp = false;
#endif

    constexpr RegData()
        : base(0), rf(0), off(0), mods(0), type(0), indirect(0), vs(0), width(0), hs(0), _pad2(0), invalid(1) {}

    constexpr int getBase()            const { return base; }
    constexpr RegFile getRegFile()     const { return static_cast<RegFile>(rf); }
    constexpr14 RegFile8 getRegFile8()   const {
#ifdef NGEN_SAFE
        if (rf > 1) throw invalid_register_file_exception();
#endif
        return static_cast<RegFile8>(rf);
    }
    constexpr bool isARF()             const { return rf == RegFileARF; }
#if XE4
    constexpr bool isSRF()             const { return rf == RegFileSRF; }
#endif
    constexpr int getARFBase()         const { return base & 0xF; }
    constexpr ARFType getARFType()     const { return static_cast<ARFType>(base >> 4); }
    constexpr bool isIndirect()        const { return indirect; }
    constexpr bool isVxIndirect()      const { return indirect && (vs == 0x7F); }
    constexpr int getIndirectOff()     const { return base & 0xFF; }
    constexpr bool isNull()            const { return isARF() && (getARFType() == ARFType::null); }
    constexpr bool isInvalid()         const { return invalid; }
    constexpr bool isValid()           const { return !invalid; }
    constexpr int getOffset()          const { return off; }
    constexpr14 int getByteOffset()    const { return (off * getBits()) >> 3; }
    constexpr14 int getLogicalOffset() const { return off; }                /* Deprecated; use getOffset */
    constexpr DataType getType()       const { return static_cast<DataType>(type); }
    constexpr int getVS()              const { return vs; }
    constexpr int getWidth()           const { return width; }
    constexpr int getHS()              const { return hs; }
    constexpr bool getNeg()            const { return mods & 2; }
    constexpr bool getAbs()            const { return mods & 1; }
    constexpr int getMods()            const { return mods; }
    constexpr14 int getBits()          const { return NGEN_NAMESPACE::getBits(getType()); }
    constexpr14 int getBytes()         const { return NGEN_NAMESPACE::getBytes(getType()); }
    constexpr14 int getDwords()        const { return NGEN_NAMESPACE::getDwords(getType()); }
#if XE4
    constexpr bool isScalar()          const { return (hs == 0 && vs == 0 && width == 1) || (rf == RegFileSRF); }
#else
    constexpr bool isScalar()          const { return hs == 0 && vs == 0 && width == 1; }
#endif

    inline constexpr14 RegData getIndirectReg() const;
#if XE4
    inline constexpr14 RegData getIndirectRegXe4() const;
    inline constexpr14 RegData getIndirectBaseRegXe4() const;
    constexpr bool isLUOrUC()          const { return off & 0x200; }
    constexpr14 int getScalarIndex()   const { return isARF() ? getOffset() : getBase(); }
#else
    constexpr14 int getScalarIndex()   const { return getOffset(); }
#endif

    constexpr14 RegData &setBase(int base_)                      { base = base_; return *this; }
    constexpr14 RegData &setOffset(int off_)                     { off = off_; return *this; }
    constexpr14 RegData &setType(DataType newType)               { type = static_cast<unsigned>(newType); return *this; }
    constexpr14 RegData &setMods(int mods_)                      { mods = mods_; return *this; }
    constexpr14 RegData &setRegion(int vs_, int width_, int hs_) { vs = vs_; width = width_; hs = hs_; return *this; }
    constexpr14 RegData &setRegFile(int rf_)                     { rf = rf_; return *this; }

    void invalidate()                     { invalid = true; }
    RegData &operator=(const Invalid &i)  { this->invalidate(); return *this; }

    inline void fixup(HW hw, int execSize, int execWidth, DataType defaultType, int srcN, int arity);                    // Adjust automatically-computed strides given ESize.

    constexpr RegData operator+() const { return *this; }
    constexpr14 RegData operator-() const {
        auto result = *this;
        result.negate();
        return result;
    }
    constexpr14 RegData operator~() const { return -*this; }
    constexpr14 void negate()             { mods = mods ^ 2; }

    friend inline bool operator==(const RegData &r1, const RegData &r2);
    friend inline bool operator!=(const RegData &r1, const RegData &r2);

    friend inline RegData abs(const RegData &r);

#ifdef NGEN_ASM
    inline void outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const;
#endif
};

static_assert(sizeof(RegData) == 8, "RegData structure is not laid out correctly in memory.");

static inline bool operator==(const RegData &r1, const RegData &r2) {
    return !std::memcmp(&r1, &r2, sizeof(RegData));
}

static inline bool operator!=(const RegData &r1, const RegData &r2) {
    return !(r1 == r2);
}

inline RegData abs(const RegData &r)
{
    RegData result = r;
    return result.setMods(1);
}

inline void RegData::fixup(HW hw, int execSize, int execWidth, DataType defaultType, int srcN, int arity)
{
#ifdef NGEN_SAFE
    if (isInvalid()) throw invalid_object_exception();
#endif

    if (getType() == DataType::invalid) {
#ifdef NGEN_SAFE
        if (defaultType == DataType::invalid)
            throw missing_type_exception();
#endif
        setType(defaultType);
    }
    if (!isVxIndirect()) {
        if (execSize == 1) {
            vs = hs = 0;
            width = 1;
        } else if (width == 0) {
            int maxWidth = 32 / getBytes();
            width = (hs == 0) ? 1 : std::min<int>({int(maxWidth / hs), execSize, 16});
            vs = width * hs;
            if (arity == 3 && hw >= HW::Gen12LP && vs == 2 && srcN < 2) {
#ifdef NGEN_SAFE
                if (hs != 1) throw invalid_region_exception();
#endif
                vs = 1;
                hs = 0;
            }
        } else if (execSize == width)
            vs = width * hs;
        bool isDest = srcN < 0;
        if (isDest && hs == 0)
            hs = (execWidth > getBytes()) ? (execWidth / getBytes()) : 1;
    }
}

inline int getExecWidth(std::initializer_list<DataType> types)
{
    int ewidth = 1;
    for (auto dt: types) ewidth = std::max(ewidth, getBytes(dt));
    return ewidth;
}

// Operands for Align16 instructions
class Align16Operand {
protected:
    RegData rd;
    unsigned chanSel : 8;
    unsigned chanEn : 4;
    bool rep : 1;

public:
    constexpr Align16Operand(RegData rd_, int chanEn_) : rd(rd_), chanSel(0b11100100), chanEn(chanEn_), rep(false) {}
    constexpr Align16Operand(RegData rd_, int s0, int s1, int s2, int s3) : rd(rd_),
        chanSel((s0 & 3) | ((s1 & 3) << 2) | ((s2 & 3) << 4) | ((s3 & 3) << 6)), chanEn(0xF), rep(false) {}

    static constexpr14 Align16Operand createBroadcast(RegData rd_) {
        Align16Operand op{rd_, 0xF};
        op.rep = true;
        return op;
    }

    static constexpr14 Align16Operand createWithMME(RegData rd_, int mme) {
        Align16Operand op{rd_, mme};
        op.chanSel = mme;
        return op;
    }

    RegData &getReg()                           { return rd; }
    constexpr const RegData &getReg()     const { return rd; }
    constexpr uint8_t getChanSel()        const { return chanSel; }
    constexpr uint8_t getChanEn()         const { return chanEn; }
    constexpr bool isRep()                const { return rep; }

    constexpr bool isIndirect()           const { return rd.isIndirect(); }
    constexpr DataType getType()          const { return rd.getType(); }
    constexpr int getOffset()             const { return rd.getOffset(); }
    constexpr int getMods()               const { return rd.getMods(); }
    constexpr RegFile getRegFile()        const { return rd.getRegFile(); }
    constexpr bool isARF()                const { return rd.isARF(); }

    void invalidate() { rd.invalidate(); }
    Align16Operand &operator=(const Invalid &i) { this->invalidate(); return *this; }
    bool isInvalid()                      const { return rd.isInvalid(); }
    bool isValid()                        const { return !rd.isInvalid(); }
    constexpr bool isScalar()             const { return rd.isScalar(); }

    void fixup(HW hw, int execSize, int execWidth, DataType defaultType, int srcN, int arity) {
        rd.fixup(hw, execSize, execWidth, defaultType, srcN, arity);
    }

#ifdef NGEN_ASM
    inline void outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const;
    static const bool emptyOp = false;
#endif
};

// Register regions.
class RegisterRegion : public RegData
{
public:
    constexpr RegisterRegion() : RegData() {}
    constexpr14 RegisterRegion(RegData rdata_, int vs_, int width_, int hs_) {
        *static_cast<RegData *>(this) = rdata_;
        vs = vs_;
        width = width_;
        hs = hs_;
    }

    RegisterRegion &operator=(const Invalid &i) { this->invalidate(); return *this; }

    constexpr RegisterRegion operator+() const { return *this; }
    constexpr14 RegisterRegion operator-() const {
        auto result = *this;
        result.negate();
        return result;
    }
    constexpr14 RegisterRegion operator~() const { return -*this; }
};

// Subregister; always associated with a specific data type.
class Subregister : public RegData
{
protected:
    void checkGRF() const {
#ifdef NGEN_SAFE
        if (isARF()) throw grf_expected_exception();
#endif
    }

public:
    constexpr Subregister() : RegData() {}
    constexpr14 Subregister(RegData reg_, int offset_, DataType type_) {
        *static_cast<RegData *>(this) = reg_;
        off = offset_;
        type = static_cast<int>(type_);
        hs = vs = 0;
        width = 1;
    }
    constexpr14 Subregister(RegData reg_, DataType type_) {
        *static_cast<RegData *>(this) = reg_;
        off = 0;
        type = static_cast<int>(type_);
    }

    inline RegisterRegion operator()(int vs, int width, int hs) const;
    inline RegisterRegion operator()(int vs, int hs) const;
    inline RegisterRegion operator()(int hs) const;

    Subregister &operator=(const Invalid &i) { this->invalidate(); return *this; }

    constexpr Subregister operator+() const { return *this; }
    constexpr14 Subregister operator-() const {
        auto result = *this;
        result.negate();
        return result;
    }
    constexpr14 Subregister operator~() const { return -*this; }

    inline GRFDisp operator+(int offset) const;
    inline GRFDisp operator-(int offset) const;

    Align16Operand swizzle(int s0, int s1, int s2, int s3)    const { checkGRF(); return Align16Operand(*this, s0, s1, s2, s3); }
    Align16Operand broadcast()                                const { checkGRF(); return Align16Operand::createBroadcast(*this); }
    Align16Operand enable(bool c0, bool c1, bool c2, bool c3) const { checkGRF(); return Align16Operand(*this, (int(c3) << 3) | (int(c2) << 2) | (int(c1) << 1) | int(c0)); }
    Align16Operand noSwizzle()                                const { return swizzle(0, 1, 2, 3); }
    Align16Operand enableAll()                                const { return enable(true, true, true, true); }

    inline Subregister reinterpret(int offset, DataType type_) const;
    template <typename T> Subregister reinterpret(int offset = 0) const { return reinterpret(offset, getDataType<T>()); }

    inline Subregister offset(int off) const { return reinterpret(off, getType()); }

    Subregister   uq(int offset = 0) const { return reinterpret(offset, DataType::uq); }
    Subregister    q(int offset = 0) const { return reinterpret(offset, DataType::q);  }
    Subregister   ud(int offset = 0) const { return reinterpret(offset, DataType::ud); }
    Subregister    d(int offset = 0) const { return reinterpret(offset, DataType::d);  }
    Subregister   uw(int offset = 0) const { return reinterpret(offset, DataType::uw); }
    Subregister    w(int offset = 0) const { return reinterpret(offset, DataType::w);  }
    Subregister   ub(int offset = 0) const { return reinterpret(offset, DataType::ub); }
    Subregister    b(int offset = 0) const { return reinterpret(offset, DataType::b);  }
    Subregister   u4(int offset = 0) const { return reinterpret(offset, DataType::u4); }
    Subregister   s4(int offset = 0) const { return reinterpret(offset, DataType::s4); }
    Subregister   u2(int offset = 0) const { return reinterpret(offset, DataType::u2); }
    Subregister   s2(int offset = 0) const { return reinterpret(offset, DataType::s2); }
    Subregister   df(int offset = 0) const { return reinterpret(offset, DataType::df); }
    Subregister    f(int offset = 0) const { return reinterpret(offset, DataType::f);  }
    Subregister   hf(int offset = 0) const { return reinterpret(offset, DataType::hf); }
    Subregister   bf(int offset = 0) const { return reinterpret(offset, DataType::bf); }
    Subregister tf32(int offset = 0) const { return reinterpret(offset, DataType::tf32); }
    Subregister  bf8(int offset = 0) const { return reinterpret(offset, DataType::bf8); }
    Subregister  hf8(int offset = 0) const { return reinterpret(offset, DataType::hf8); }
#if XE3P
    Subregister e2m1(int offset = 0) const { return reinterpret(offset, DataType::e2m1); }
    Subregister e3m0(int offset = 0) const { return reinterpret(offset, DataType::e3m0); }
#endif
};

// Single register.
class Register : public RegData
{
public:
    constexpr Register() : RegData() {}
    constexpr Register(int reg_, RegFile rf_, DataType defaultType = DataType::invalid, int off_ = 0)
        : RegData(reg_, rf_, off_, false, defaultType, 0, 0, 1) {}

    constexpr Register operator+() const { return *this; }
    constexpr14 Register operator-() const {
        auto result = *this;
        result.negate();
        return result;
    }
    constexpr14 Register operator~() const { return -*this; }

    Register &operator+=(const int &inc) {
        base += inc;
        return *this;
    }

    Register operator++(int i) {
        auto old = *this;
        ++*this;
        return old;
    }

    Register &operator++() {
        *this += 1;
        return *this;
    }

    Register advance(int inc) {
        auto result = *this;
        result += inc;
        return result;
    }

    constexpr14 Subregister sub(int offset, DataType type_)        const { return Subregister(*this, offset, type_); }
    template <typename T> constexpr14 Subregister sub(int offset)  const { return sub(offset, getDataType<T>()); }

    constexpr14 Register retype(DataType type_)         const { auto clone = *this; clone.setType(type_); return clone; }
    template <typename T> constexpr14 Register retype() const { return retype(getDataType<T>()); }

    constexpr14 Subregister   uq(int offset) const { return sub(offset, DataType::uq); }
    constexpr14 Subregister    q(int offset) const { return sub(offset, DataType::q);  }
    constexpr14 Subregister   ud(int offset) const { return sub(offset, DataType::ud); }
    constexpr14 Subregister    d(int offset) const { return sub(offset, DataType::d);  }
    constexpr14 Subregister   uw(int offset) const { return sub(offset, DataType::uw); }
    constexpr14 Subregister    w(int offset) const { return sub(offset, DataType::w);  }
    constexpr14 Subregister   ub(int offset) const { return sub(offset, DataType::ub); }
    constexpr14 Subregister    b(int offset) const { return sub(offset, DataType::b);  }
    constexpr14 Subregister   u4(int offset) const { return sub(offset, DataType::u4); }
    constexpr14 Subregister   s4(int offset) const { return sub(offset, DataType::s4); }
    constexpr14 Subregister   u2(int offset) const { return sub(offset, DataType::u2); }
    constexpr14 Subregister   s2(int offset) const { return sub(offset, DataType::s2); }
    constexpr14 Subregister   df(int offset) const { return sub(offset, DataType::df); }
    constexpr14 Subregister    f(int offset) const { return sub(offset, DataType::f);  }
    constexpr14 Subregister   hf(int offset) const { return sub(offset, DataType::hf); }
    constexpr14 Subregister   bf(int offset) const { return sub(offset, DataType::bf); }
    constexpr14 Subregister tf32(int offset) const { return sub(offset, DataType::tf32); }
    constexpr14 Subregister  bf8(int offset) const { return sub(offset, DataType::bf8); }
    constexpr14 Subregister  hf8(int offset) const { return sub(offset, DataType::hf8); }
#if XE3P
    constexpr14 Subregister e2m1(int offset) const { return sub(offset, DataType::e2m1); }
    constexpr14 Subregister e3m0(int offset) const { return sub(offset, DataType::e3m0); }
#endif

    constexpr14 Register   uq() const { return retype(DataType::uq); }
    constexpr14 Register    q() const { return retype(DataType::q);  }
    constexpr14 Register   ud() const { return retype(DataType::ud); }
    constexpr14 Register    d() const { return retype(DataType::d);  }
    constexpr14 Register   uw() const { return retype(DataType::uw); }
    constexpr14 Register    w() const { return retype(DataType::w);  }
    constexpr14 Register   ub() const { return retype(DataType::ub); }
    constexpr14 Register    b() const { return retype(DataType::b);  }
    constexpr14 Register   u4() const { return retype(DataType::u4); }
    constexpr14 Register   s4() const { return retype(DataType::s4); }
    constexpr14 Register   u2() const { return retype(DataType::u2); }
    constexpr14 Register   s2() const { return retype(DataType::s2); }
    constexpr14 Register   df() const { return retype(DataType::df); }
    constexpr14 Register    f() const { return retype(DataType::f);  }
    constexpr14 Register   hf() const { return retype(DataType::hf); }
    constexpr14 Register   bf() const { return retype(DataType::bf); }
    constexpr14 Register tf32() const { return retype(DataType::tf32); }
    constexpr14 Register  bf8() const { return retype(DataType::bf8); }
    constexpr14 Register  hf8() const { return retype(DataType::hf8); }
#if XE3P
    constexpr14 Register e2m1() const { return retype(DataType::e2m1); }
    constexpr14 Register e3m0() const { return retype(DataType::e3m0); }
#endif

    constexpr14 Subregister operator[](int offset) const { return sub(offset, getType()); }

    Register &operator=(const Invalid &i) { this->invalidate(); return *this; }
};

class GRF : public Register
{
public:
    GRF() : Register() {}
    explicit constexpr GRF(int reg_) : Register(reg_, RegFileGRF) {}

    constexpr GRF operator+() const { return *this; }
    constexpr14 GRF operator-() const {
        auto result = *this;
        result.negate();
        return result;
    }
    constexpr14 GRF operator~() const { return -*this; }

    constexpr14 GRF retype(DataType type_)              const { auto clone = *this; clone.setType(type_); return clone; }
    template <typename T> constexpr14 Register retype() const { return retype(getDataType<T>()); }

    constexpr14 Subregister   uq(int offset) const { return sub(offset, DataType::uq); }
    constexpr14 Subregister    q(int offset) const { return sub(offset, DataType::q);  }
    constexpr14 Subregister   ud(int offset) const { return sub(offset, DataType::ud); }
    constexpr14 Subregister    d(int offset) const { return sub(offset, DataType::d);  }
    constexpr14 Subregister   uw(int offset) const { return sub(offset, DataType::uw); }
    constexpr14 Subregister    w(int offset) const { return sub(offset, DataType::w);  }
    constexpr14 Subregister   ub(int offset) const { return sub(offset, DataType::ub); }
    constexpr14 Subregister    b(int offset) const { return sub(offset, DataType::b);  }
    constexpr14 Subregister   u4(int offset) const { return sub(offset, DataType::u4); }
    constexpr14 Subregister   s4(int offset) const { return sub(offset, DataType::s4); }
    constexpr14 Subregister   u2(int offset) const { return sub(offset, DataType::u2); }
    constexpr14 Subregister   s2(int offset) const { return sub(offset, DataType::s2); }
    constexpr14 Subregister   df(int offset) const { return sub(offset, DataType::df); }
    constexpr14 Subregister    f(int offset) const { return sub(offset, DataType::f);  }
    constexpr14 Subregister   hf(int offset) const { return sub(offset, DataType::hf); }
    constexpr14 Subregister   bf(int offset) const { return sub(offset, DataType::bf); }
    constexpr14 Subregister tf32(int offset) const { return sub(offset, DataType::tf32); }
    constexpr14 Subregister  bf8(int offset) const { return sub(offset, DataType::bf8); }
    constexpr14 Subregister  hf8(int offset) const { return sub(offset, DataType::hf8); }
#if XE3P
    constexpr14 Subregister e2m1(int offset) const { return sub(offset, DataType::e2m1); }
    constexpr14 Subregister e3m0(int offset) const { return sub(offset, DataType::e3m0); }
#endif

    constexpr14 GRF   uq() const { return retype(DataType::uq); }
    constexpr14 GRF    q() const { return retype(DataType::q);  }
    constexpr14 GRF   ud() const { return retype(DataType::ud); }
    constexpr14 GRF    d() const { return retype(DataType::d);  }
    constexpr14 GRF   uw() const { return retype(DataType::uw); }
    constexpr14 GRF    w() const { return retype(DataType::w);  }
    constexpr14 GRF   ub() const { return retype(DataType::ub); }
    constexpr14 GRF    b() const { return retype(DataType::b);  }
    constexpr14 GRF   u4() const { return retype(DataType::u4); }
    constexpr14 GRF   s4() const { return retype(DataType::s4); }
    constexpr14 GRF   u2() const { return retype(DataType::u2); }
    constexpr14 GRF   s2() const { return retype(DataType::s2); }
    constexpr14 GRF   df() const { return retype(DataType::df); }
    constexpr14 GRF    f() const { return retype(DataType::f);  }
    constexpr14 GRF   hf() const { return retype(DataType::hf); }
    constexpr14 GRF   bf() const { return retype(DataType::bf); }
    constexpr14 GRF tf32() const { return retype(DataType::tf32); }
    constexpr14 GRF  bf8() const { return retype(DataType::bf8); }
    constexpr14 GRF  hf8() const { return retype(DataType::hf8); }
#if XE3P
    constexpr14 GRF e2m1() const { return retype(DataType::e2m1); }
    constexpr14 GRF e3m0() const { return retype(DataType::e3m0); }
#endif
#if XE4
    constexpr14 GRF    b64() const { return retype(DataType::b64);    }
    constexpr14 GRF    s64() const { return retype(DataType::s64);    }
    constexpr14 GRF    u64() const { return retype(DataType::u64);    }
    constexpr14 GRF    f64() const { return retype(DataType::f64);    }
    constexpr14 GRF    b32() const { return retype(DataType::b32);    }
    constexpr14 GRF    s32() const { return retype(DataType::s32);    }
    constexpr14 GRF    u32() const { return retype(DataType::u32);    }
    constexpr14 GRF    f32() const { return retype(DataType::f32);    }
    constexpr14 GRF    f16() const { return retype(DataType::f16);    }
    constexpr14 GRF   bf16() const { return retype(DataType::bf16);   }
    constexpr14 GRF    s16() const { return retype(DataType::s16);    }
    constexpr14 GRF    u16() const { return retype(DataType::u16);    }
    constexpr14 GRF  b16v2() const { return retype(DataType::b16v2);  }
    constexpr14 GRF  s16v2() const { return retype(DataType::s16v2);  }
    constexpr14 GRF  u16v2() const { return retype(DataType::u16v2);  }
    constexpr14 GRF  f16v2() const { return retype(DataType::f16v2);  }
    constexpr14 GRF bf16v2() const { return retype(DataType::bf16v2); }
    constexpr14 GRF     s8() const { return retype(DataType::s8);     }
    constexpr14 GRF     u8() const { return retype(DataType::u8);     }
    constexpr14 GRF   b8v4() const { return retype(DataType::b8v4);   }
    constexpr14 GRF   u8v4() const { return retype(DataType::u8v4);   }
    constexpr14 GRF   s8v4() const { return retype(DataType::s8v4);   }
#endif

    Align16Operand swizzle(int s0, int s1, int s2, int s3)    const { return Align16Operand(*this, s0, s1, s2, s3); }
    Align16Operand enable(bool c0, bool c1, bool c2, bool c3) const { return Align16Operand(*this, (int(c3) << 3) | (int(c2) << 2) | (int(c1) << 1) | int(c0)); }
    Align16Operand noSwizzle()                                const { return swizzle(0, 1, 2, 3); }
    Align16Operand enableAll()                                const { return enable(true, true, true, true); }

#if XE4
    constexpr14 GRF lu() const { auto clone = *this; clone.off |= 0x200; return clone; }
    constexpr14 GRF uc() const { return lu(); }
#endif

    GRF &operator=(const Invalid &i) { this->invalidate(); return *this; }

    GRF &operator+=(const int &inc) {
        base += inc;
        return *this;
    }

    GRF operator++(int i) {
        GRF old = *this;
        ++*this;
        return old;
    }

    GRF &operator++() {
        *this += 1;
        return *this;
    }

    GRF advance(int inc) {
        auto result = *this;
        result += inc;
        return result;
    }

    inline GRFDisp operator+(int offset) const;
    inline GRFDisp operator-(int offset) const;

    inline GRFDisp operator+(Offset2D offset) const;
    inline GRFDisp operator-(Offset2D offset) const;
#if XE3P
    inline GRFDisp operator*(int scale) const;
#endif

#if XE4
    static constexpr int log2Bytes(HW hw)                  { return (hw >= HW::Xe4)   ? 7 :
                                                                    (hw >= HW::XeHPC) ? 6 : 5;  }
#else
    static constexpr int log2Bytes(HW hw)                  { return (hw >= HW::XeHPC) ? 6 : 5;  }
#endif
    static constexpr int bytes(HW hw)                      { return (1 << log2Bytes(hw)); }
    static constexpr int bytesToGRFs(HW hw, unsigned x)    { return (x + bytes(hw) - 1) >> log2Bytes(hw); }

#if XE3P
    static constexpr int maxRegs()                         { return 512; }
#else
    static constexpr int maxRegs()                         { return 256; }
#endif
};

#if XE4
class SRF : public Register
{
public:
    SRF() : Register() {}
    explicit constexpr SRF(int reg_) : Register(reg_, RegFileSRF) {}

    constexpr SRF operator+() const { return *this; }
    constexpr14 SRF operator-() const {
        auto result = *this;
        result.negate();
        return result;
    }
    constexpr14 SRF operator~() const { return -*this; }

    constexpr14 SRF retype(DataType type_)              const { auto clone = *this; clone.setType(type_); return clone; }
    template <typename T> constexpr14 Register retype() const { return retype(getDataType<T>()); }

    constexpr14 SRF    b64() const { return retype(DataType::b64);    }
    constexpr14 SRF    s64() const { return retype(DataType::s64);    }
    constexpr14 SRF    u64() const { return retype(DataType::u64);    }
    constexpr14 SRF    f64() const { return retype(DataType::f64);    }
    constexpr14 SRF    b32() const { return retype(DataType::b32);    }
    constexpr14 SRF    s32() const { return retype(DataType::s32);    }
    constexpr14 SRF    u32() const { return retype(DataType::u32);    }
    constexpr14 SRF    f32() const { return retype(DataType::f32);    }
    constexpr14 SRF   tf32() const { return retype(DataType::tf32);   }
    constexpr14 SRF    f16() const { return retype(DataType::f16);    }
    constexpr14 SRF   bf16() const { return retype(DataType::bf16);   }
    constexpr14 SRF    s16() const { return retype(DataType::s16);    }
    constexpr14 SRF    u16() const { return retype(DataType::u16);    }
    constexpr14 SRF  b16v2() const { return retype(DataType::b16v2);  }
    constexpr14 SRF  s16v2() const { return retype(DataType::s16v2);  }
    constexpr14 SRF  u16v2() const { return retype(DataType::u16v2);  }
    constexpr14 SRF  f16v2() const { return retype(DataType::f16v2);  }
    constexpr14 SRF bf16v2() const { return retype(DataType::bf16v2); }
    constexpr14 SRF     s8() const { return retype(DataType::s8);     }
    constexpr14 SRF     u8() const { return retype(DataType::u8);     }
    constexpr14 SRF   b8v4() const { return retype(DataType::b8v4);   }
    constexpr14 SRF   u8v4() const { return retype(DataType::u8v4);   }
    constexpr14 SRF   s8v4() const { return retype(DataType::s8v4);   }
    constexpr14 SRF   e2m1() const { return retype(DataType::e2m1);   }
    constexpr14 SRF   e3m0() const { return retype(DataType::e3m0);   }

    /* Xe3-style names */
    constexpr14 SRF     uq() const { return retype(DataType::uq); }
    constexpr14 SRF      q() const { return retype(DataType::q);  }
    constexpr14 SRF     ud() const { return retype(DataType::ud); }
    constexpr14 SRF      d() const { return retype(DataType::d);  }
    constexpr14 SRF     uw() const { return retype(DataType::uw); }
    constexpr14 SRF      w() const { return retype(DataType::w);  }
    constexpr14 SRF     ub() const { return retype(DataType::ub); }
    constexpr14 SRF      b() const { return retype(DataType::b);  }
    constexpr14 SRF     df() const { return retype(DataType::df); }
    constexpr14 SRF      f() const { return retype(DataType::f);  }
    constexpr14 SRF     hf() const { return retype(DataType::hf); }
    constexpr14 SRF     bf() const { return retype(DataType::bf); }
    constexpr14 SRF    bf8() const { return retype(DataType::bf8); }
    constexpr14 SRF    hf8() const { return retype(DataType::hf8); }

    SRF &operator=(const Invalid &i) { this->invalidate(); return *this; }

    SRF &operator+=(const int &inc) {
        base += inc;
        return *this;
    }

    SRF operator++(int i) {
        SRF old = *this;
        ++*this;
        return old;
    }

    SRF &operator++() {
        *this += 1;
        return *this;
    }

    SRF advance(int inc) {
        auto result = *this;
        result += inc;
        return result;
    }

    inline GRFDisp operator+(int offset) const;
    inline GRFDisp operator-(int offset) const;

    static constexpr unsigned bytesToSRFs(unsigned bytes) { return (bytes + 3) >> 2; }
    static constexpr int maxRegs()                        { return 512; }
};

template <typename T>
static inline void canonicalizeSRF(T&) {}

inline void canonicalizeSRF(RegData &rd)
{
    if (rd.isARF() && rd.getARFType() == ARFType::s)
        rd = SRF{rd.getOffset()};
}
#endif

class ARF : public Register
{
public:
    constexpr ARF() : Register() {}
    constexpr ARF(ARFType type_, int reg_, DataType defaultType = DataType::invalid, int off_ = 0)
        : Register((static_cast<int>(type_) << 4) | (reg_ & 0xF), RegFileARF, defaultType, off_) {}

    ARF &operator=(const Invalid &i) { this->invalidate(); return *this; }
};

class NullRegister : public ARF
{
public:
    constexpr NullRegister() : ARF(ARFType::null, 0, DataType::ud) {}
};

class AddressRegister : public ARF
{
public:
    constexpr AddressRegister() : ARF() {}
    explicit constexpr AddressRegister(int reg_) : ARF(ARFType::a, reg_, DataType::uw) {}

    AddressRegister &operator=(const Invalid &i) { this->invalidate(); return *this; }
};

class AccumulatorRegister : public ARF
{
public:
    constexpr AccumulatorRegister() : ARF() {}
    explicit constexpr AccumulatorRegister(int reg_) : ARF(ARFType::acc, reg_) {}

    AccumulatorRegister &operator=(const Invalid &i) { this->invalidate(); return *this; }

    static constexpr14 int count(HW hw, DataType dt = DataType::invalid) {
#if XE4
        if (hw >= HW::Xe4) return 0;
#endif
        if (dt == DataType::df) {
            if (hw == HW::Gen9)  return 0;
            if (hw == HW::XeHPG) return 0;
            if (hw == HW::Xe2)   return 0;
            if (hw == HW::Xe3)   return 0;
        }
        if (hw >= HW::XeHP) return 4;
        return 2;
    }
    static constexpr14 int count(HW hw, int grfCount, DataType dt = DataType::invalid) {
        return count(hw, dt) * (grfCount == 256 ? 2 : 1);
    }
};

class SpecialAccumulatorRegister : public AccumulatorRegister
{
    uint8_t mmeNum;

public:
    constexpr SpecialAccumulatorRegister() : AccumulatorRegister(), mmeNum(0) {}
    constexpr SpecialAccumulatorRegister(int reg_, int mmeNum_) : AccumulatorRegister(reg_), mmeNum(mmeNum_) {}

    static constexpr SpecialAccumulatorRegister createNoMME() { return SpecialAccumulatorRegister(0, 8); }

    constexpr uint8_t getMME() const { return mmeNum; }

    SpecialAccumulatorRegister &operator=(const Invalid &i) { this->invalidate(); return *this; }
};

constexpr14 RegData RegData::getIndirectReg() const {
    auto type = (base & 0x100) ? ARFType::s : ARFType::a;
    return ARF(type, 0)[getIndirectOff()];
}

#if XE4
constexpr14 RegData RegData::getIndirectRegXe4() const {
    return _pad2 ? RegData(SRF(base)) : RegData(GRF(base));
}

constexpr14 RegData RegData::getIndirectBaseRegXe4() const {
    return isSRF() ? RegData(SRF(off)) : RegData(GRF(off));
}
#endif

// An "extended register" is a combination of a regular GRF and some extra accumulator bits, used for math macro operations.
class ExtendedReg {
    RegData base;
    uint8_t mmeNum;

public:
    constexpr ExtendedReg(RegData base_, uint8_t mmeNum_) : base(base_), mmeNum(mmeNum_) {}
    constexpr ExtendedReg(RegData base_, SpecialAccumulatorRegister acc) : base(base_), mmeNum(acc.getMME()) {}

    void fixup(HW hw, int execSize, int execWidth, DataType defaultType, int srcN, int arity) {
        base.fixup(hw, execSize, execWidth, defaultType, srcN, arity);
    }

    constexpr int getMods()         const { return base.getMods(); }
    constexpr DataType getType()    const { return base.getType(); }
    constexpr int getOffset()       const { return base.getOffset(); }
    constexpr bool isIndirect()     const { return base.isIndirect(); }
    constexpr bool isInvalid()      const { return base.isInvalid(); }
    constexpr bool isValid()        const { return !base.isInvalid(); }
    constexpr bool isScalar()       const { return base.isScalar(); }
    constexpr RegFile getRegFile()  const { return base.getRegFile(); }
    constexpr bool isARF()          const { return base.isARF(); }
    constexpr bool isNull()         const { return base.isNull(); }

    constexpr14 RegData &getBase()        { return base; }
    constexpr RegData getBase()     const { return base; }
    constexpr uint8_t getMMENum()   const { return mmeNum; }

    constexpr14 ExtendedReg &setType(DataType newType) { base.setType(newType); return *this; }

    ExtendedReg operator-() const {
        auto clone = *this;
        clone.base.negate();
        return clone;
    }

#ifdef NGEN_ASM
    inline void outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const;
    static const bool emptyOp = false;
#endif
};

static inline ExtendedReg operator|(const RegData &base, const SpecialAccumulatorRegister &acc)
{
    return ExtendedReg(base, acc);
}

class FlagRegister : public ARF
{
public:
    constexpr FlagRegister() : ARF() {}
    explicit constexpr FlagRegister(int reg_)  : ARF(ARFType::f, reg_, DataType::ud, 0) {}
    constexpr FlagRegister(int reg_, int off_) : ARF(ARFType::f, reg_, DataType::uw, off_) {}

    static FlagRegister createFromIndex(int index) {
        return FlagRegister(index >> 1, index & 1);
    }

    FlagRegister operator~() const {
        FlagRegister result = *this;
        result.mods = result.mods ^ 2;
        return result;
    }
    FlagRegister operator!() const { return ~*this; }

    FlagRegister &operator=(const Invalid &i) { this->invalidate(); return *this; }

    constexpr14 FlagRegister operator[](int offset) const {
        FlagRegister sub(getARFBase(), getOffset() + offset);
        sub.mods = mods;
        return sub;
    }

    int index() const { return (getARFBase() << 1) + getOffset(); }

    static inline constexpr14 int count(HW hw) {
#if XE4
        if (hw >= HW::Xe4) return 15;
#endif
        return (hw >= HW::XeHPC) ? 4 : 2;
    }
    static inline constexpr14 int subcount(HW hw) { return count(hw) * 2; }

#if XE4
    inline operator IndirectARF() const;
#endif
};

class ChannelEnableRegister : public ARF
{
public:
    explicit constexpr ChannelEnableRegister(int reg_ = 0) : ARF(ARFType::ce, reg_, DataType::ud) {}
};

class StackPointerRegister : public ARF
{
public:
    explicit constexpr StackPointerRegister(int reg_ = 0) : ARF(ARFType::sp, reg_, DataType::uq) {}
};

class ScalarRegister : public ARF
{
public:
    explicit constexpr ScalarRegister(int reg_, int off_ = 0, DataType type_ = DataType::ub) : ARF(ARFType::s, reg_, type_, off_) {}

    constexpr ScalarRegister operator[](int offset) const { return ScalarRegister(getARFBase(), getOffset() + offset); }
    constexpr14 ScalarRegister uq(int offset) const { return ScalarRegister(getARFBase(), (getByteOffset() >> 3) + offset, DataType::uq); }
    constexpr14 ScalarRegister  q(int offset) const { return ScalarRegister(getARFBase(), (getByteOffset() >> 3) + offset, DataType::q); }

    RegisterRegion operator()(int vs, int width, int hs) const { return reinterpret_cast<const Subregister &>(*this)(vs, width, hs); }
    RegisterRegion operator()(int vs, int hs) const            { return reinterpret_cast<const Subregister &>(*this)(vs, hs); }
    RegisterRegion operator()(int hs) const                    { return reinterpret_cast<const Subregister &>(*this)(vs); }

#if XE4
    operator SRF() const { return SRF{getARFBase()}; }      /* allow s0 to transparently convert to an SRF type */
#endif
};

class StateRegister : public ARF
{
public:
    explicit constexpr StateRegister(int reg_ = 0) : ARF(ARFType::sr, reg_, DataType::ud) {}
};

class ControlRegister : public ARF
{
public:
    explicit constexpr ControlRegister(int reg_ = 0) : ARF(ARFType::cr, reg_, DataType::ud) {}
};

class NotificationRegister : public ARF
{
public:
    explicit constexpr NotificationRegister(int reg_ = 0) : ARF(ARFType::n, reg_, DataType::ud) {}
};

class InstructionPointerRegister : public ARF
{
public:
    constexpr InstructionPointerRegister() : ARF(ARFType::ip, 0, DataType::ud) {}
};

class ThreadDependencyRegister : public ARF
{
public:
    explicit constexpr ThreadDependencyRegister(int reg_ = 0) : ARF(ARFType::tdr, reg_, DataType::uw) {}
};

class PerformanceRegister : public ARF
{
public:
    explicit constexpr PerformanceRegister(int reg_ = 0, int off_ = 0) : ARF(ARFType::tm, reg_, DataType::ud, off_) {}
};

class DebugRegister : public ARF
{
public:
    explicit constexpr DebugRegister(int reg_ = 0) : ARF(ARFType::dbg, reg_, DataType::ud) {}
};

class FlowControlRegister : public ARF
{
public:
    explicit constexpr FlowControlRegister(int reg_ = 0) : ARF(ARFType::fc, reg_, DataType::ud) {}
};

#if XE4
// Indirect ARF registers.
enum class IndirectARF : uint16_t {
    ts0 = 0, ts1 = 1,
    dmsk = 4, vmsk = 5,
    vrt = 8,
    tpst = 12,
    gctrl = 20, exctrl = 21, tpctrl = 22,
    tarb = 23,
    cctrl = 24, msgctrl = 28, sbctrl = 29, apctrl = 30,
    abar = 31,
    fs0 = 40, fs1 = 41, fs2 = 42, fs3 = 43,
    fs4 = 44, fs5 = 45, fs6 = 46, fs7 = 47,
    fs8 = 48, fs9 = 49, fs10 = 50, fs11 = 51,
    fs12 = 52, fs13 = 53, fs14 = 54,
    nbar = 64, nhost = 65, nflsh = 66,
    tsl = 72, tsh = 73,
    tme = 76,
    pse = 80,
    ctl = 84, cth = 85,
    fc00 = 96, fc01 = 97, fc02 = 98, fc03 = 99,
    fc04 = 100, fc05 = 101, fc06 = 102, fc07 = 103,
    fc08 = 104, fc09 = 105, fc010 = 106, fc011 = 107,
    fc012 = 108, fc013 = 109, fc014 = 110, fc015 = 111,
    fc016 = 112, fc017 = 113, fc018 = 114, fc019 = 115,
    fc020 = 116, fc021 = 117, fc022 = 118, fc023 = 119,
    fc024 = 120, fc025 = 121, fc026 = 122, fc027 = 123,
    fc028 = 124, fc029 = 125, fc030 = 126, fc031 = 127,
    cvid00 = 128, cvid01 = 129, cvid02 = 130, cvid03 = 131,
    cvid04 = 132, cvid05 = 133, cvid06 = 134, cvid07 = 135,
    cvid08 = 136, cvid09 = 137, cvid010 = 138, cvid011 = 139,
    cvid012 = 140, cvid013 = 141, cvid014 = 142, cvid015 = 143,
    cvid016 = 144, cvid017 = 145, cvid018 = 146, cvid019 = 147,
    cvid020 = 148, cvid021 = 149, cvid022 = 150, cvid023 = 151,
    cvid024 = 152, cvid025 = 153, cvid026 = 154, cvid027 = 155,
    cvid028 = 156, cvid029 = 157, cvid030 = 158, cvid031 = 159,
    enmsk = 160, sumsk = 161, camsk = 162, scmsk = 163, hcmsk = 164,
    rngs = 176, rngc = 177,
    euhst0 = 184, euhst1 = 185,
    dtmp0 = 192, dtmp1 = 193, dtmp2 = 194, dtmp3 = 195,
    dtmp4 = 196, dtmp5 = 197,
    kipl = 208, kipu = 209, aipl = 210, aipu = 211,
    exipl = 212, exipu = 213, sipl = 214, sipu = 215,
    iiem = 216, oobem = 217,
    fpel = 224, fpeu = 225, fpem = 226, fpdzm = 227,
    fpum = 228, fpom = 229, fpxm = 230, fpim = 231,
    sfet = 232, sfel = 233, sfeu = 234, sfem = 235,
    sfed = 236, sfedex = 237,
    tpel = 240, tpeu = 241, tpem = 242, tpee = 243,
    mme0 = 256, mme1 = 257, mme2 = 258, mme3 = 259,
    mme4 = 260, mme5 = 261, mme6 = 262, mme7 = 263,
    mme8 = 264, mme9 = 265, mme10 = 266, mme11 = 267,
    mme12 = 268, mme13 = 269, mme14 = 270, mme15 = 271,
};

static constexpr inline IndirectARF fs(int n)       { return static_cast<IndirectARF>(static_cast<int>(IndirectARF::fs0) + n); }
static constexpr inline IndirectARF fc0(int lane)   { return static_cast<IndirectARF>(static_cast<int>(IndirectARF::fc00) + lane); }
static constexpr inline IndirectARF cvid(int lane)  { return static_cast<IndirectARF>(static_cast<int>(IndirectARF::cvid00) + lane); }
static constexpr inline IndirectARF dtmp(int n)     { return static_cast<IndirectARF>(static_cast<int>(IndirectARF::dtmp0) + n); }
static constexpr inline IndirectARF mme(int n)      { return static_cast<IndirectARF>(static_cast<int>(IndirectARF::mme0) + n); }

static constexpr inline IndirectARF fs(FlagRegister f) { return fs(f.getARFBase()); }

FlagRegister::operator IndirectARF() const { return fs(*this); }

#ifdef NGEN_ASM
static inline std::ostream &operator<<(std::ostream &str, IndirectARF iarf)
{
    static const char *names[512] = {
        "ts0", "ts1", "", "", "dmsk", "vmsk", "", "",
        "vrt", "", "", "", "tpst", "", "", "",
        "", "", "", "", "gctrl", "exctrl", "tpctrl", "tarb",
        "cctrl", "", "", "", "msgctrl", "sbctrl", "apctrl", "abar",
        "", "", "", "", "", "", "", "",
        "fs0", "fs1", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7",
        "fs8", "fs9", "fs10", "fs11", "fs12", "fs13", "fs14", "",
        "", "", "", "", "", "", "", "",
        "nbar", "nhost", "nflsh", "", "", "", "", "",
        "tsl", "tsh", "", "", "tme", "", "", "",
        "pse", "", "", "", "ctl", "cth", "", "",
        "", "", "", "", "", "", "", "",
        "fc00", "fc01", "fc02", "fc03", "fc04", "fc05", "fc06", "fc07",
        "fc08", "fc09", "fc010", "fc011", "fc012", "fc013", "fc014", "fc015",
        "fc016", "fc017", "fc018", "fc019", "fc020", "fc021", "fc022", "fc023",
        "fc024", "fc025", "fc026", "fc027", "fc028", "fc029", "fc030", "fc031",
        "cvid00", "cvid01", "cvid02", "cvid03", "cvid04", "cvid05", "cvid06", "cvid07",
        "cvid08", "cvid09", "cvid010", "cvid011", "cvid012", "cvid013", "cvid014", "cvid015",
        "cvid016", "cvid017", "cvid018", "cvid019", "cvid020", "cvid021", "cvid022", "cvid023",
        "cvid024", "cvid025", "cvid026", "cvid027", "cvid028", "cvid029", "cvid030", "cvid031",
        "enmsk", "sumsk", "camsk", "scmsk", "hcmsk", "", "", "",
        "", "", "", "", "", "", "", "",
        "rngs", "rngc", "", "", "", "", "", "",
        "euhst0", "euhst1", "", "", "", "", "", "",
        "dtmp0", "dtmp1", "dtmp2", "dtmp3", "dtmp4", "dtmp5", "", "",
        "", "", "", "", "", "", "", "",
        "kipl", "kipu", "aipl", "aipu", "exipl", "exipu", "sipl", "sipu",
        "iiem", "oobem", "", "", "", "", "", "",
        "fpel", "fpeu", "fpem", "fpdzm", "fpum", "fpom", "fpxm", "fpim",
        "sfet", "sfel", "sfeu", "sfem", "sfed", "sfedex", "", "",
        "tpel", "tpeu", "tpem", "tpee", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "mme0", "mme1", "mme2", "mme3", "mme4", "mme5", "mme6", "mme7",
        "mme8", "mme9", "mme10", "mme11", "mme12", "mme13", "mme14", "mme15",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
        "", "", "", "", "", "", "", "",
    };
    str << names[static_cast<uint8_t>(iarf) & 0x1FF];
    return str;
}
#endif
#endif

class Offset2D {
public:
    int16_t x, y;

    constexpr Offset2D(int16_t x_, int16_t y_) : x(x_), y(y_) {}
    constexpr Offset2D operator-() const { return Offset2D(-x, -y); }
};

class GRFDisp {
protected:
    GRF base;
    int32_t disp;
#if XE3P
    uint16_t scale = 0;
    int16_t ind0SubReg = -1;
#endif

public:
    GRFDisp(const GRF &base_, int32_t disp_) : base(base_), disp(disp_) {}

    /* implicit */ GRFDisp(const RegData &rd) : disp(0) {
        switch (rd.getRegFile()) {
            case RegFileGRF: base = reinterpret_cast<const GRF &>(rd); return;
#if XE3P
            case RegFileARF:
                if (rd.getARFType() == ARFType::s) {
                    ind0SubReg = rd.getByteOffset();
                    return;
                }
#endif
#if XE4
            case RegFileSRF:
                ind0SubReg = rd.getBase();
                return;
#endif
            default: break;
        }
#ifdef NGEN_SAFE
        throw invalid_operand_exception();
#endif
    }

    GRFDisp(const GRF &base_, Offset2D offset) : base(base_), disp((uint32_t(uint16_t(offset.y)) << 16) | uint16_t(offset.x)) {}

#if XE3P
    GRFDisp(const GRF &base_, int32_t disp_, int scale_, int ind0SubReg_ = -1) : base(base_), disp(disp_), scale(scale_), ind0SubReg(ind0SubReg_) {}
    GRFDisp(const GRF &base_, int32_t disp_, int scale_, ScalarRegister ind0) : base(base_), disp(disp_), scale(scale_), ind0SubReg(ind0.getByteOffset()) {}
#endif

#if XE4
    GRFDisp(const GRF &base_, int32_t disp_, int scale_, SRF ind0) : base(base_), disp(disp_), scale(scale_), ind0SubReg(ind0.getBase()) {}
#endif

    constexpr GRF     getBase()  const { return base; }
    constexpr int32_t getDisp()  const { return disp; }

    constexpr int16_t getDispX() const { return disp & 0xFFFF; }
    constexpr int16_t getDispY() const { return disp >> 16; }

    void clearDisp()                   { disp = 0; }

#if XE3P
    constexpr int     getScale() const { return scale; }

    RegData getInd0() const {
        if (ind0SubReg >= 0)
            return ScalarRegister(0)[ind0SubReg];
        else
            return NullRegister();
    }

    GRFDisp operator+(int offset) const { return GRFDisp(base, disp + offset, scale, ind0SubReg); }
    GRFDisp operator-(int offset) const { return GRFDisp(base, disp - offset, scale, ind0SubReg); }
#else
    GRFDisp operator+(int offset) const { return GRFDisp(base, disp + offset); }
    GRFDisp operator-(int offset) const { return GRFDisp(base, disp - offset); }
#endif
};

GRFDisp GRF::operator+(int offset)      const { return GRFDisp(*this, offset); }
GRFDisp GRF::operator-(int offset)      const { return *this + (-offset); }

GRFDisp GRF::operator+(Offset2D offset) const { return GRFDisp(*this, offset); }
GRFDisp GRF::operator-(Offset2D offset) const { return *this + (-offset); }

#if XE3P
GRFDisp GRF::operator*(int scale)       const { return GRFDisp(*this, 0, scale); }

inline GRFDisp operator+(ScalarRegister s, GRF base) {
    return GRFDisp(base, 0, 0, s);
}
inline GRFDisp operator+(ScalarRegister s, GRFDisp addr) {
    return GRFDisp(addr.getBase(), addr.getDisp(), addr.getScale(), s);
}
inline GRFDisp operator+(GRF base,     ScalarRegister s) { return s + base; }
inline GRFDisp operator+(GRFDisp addr, ScalarRegister s) { return s + addr; }
#endif

#if XE4
GRFDisp SRF::operator+(int offset)      const { return GRFDisp(GRF(), offset, 0, *this); }
GRFDisp SRF::operator-(int offset)      const { return *this + (-offset); }

inline GRFDisp operator+(SRF s, GRF base) {
    return GRFDisp(base, 0, 0, s);
}
inline GRFDisp operator+(SRF s, GRFDisp addr) {
    return GRFDisp(addr.getBase(), addr.getDisp(), addr.getScale(), s);
}
inline GRFDisp operator+(GRF base,     SRF s) { return s + base; }
inline GRFDisp operator+(GRFDisp addr, SRF s) { return s + addr; }
#endif

GRFDisp Subregister::operator+(int offset) const
{
#if XE4
    if (isSRF())
        return reinterpret_cast<const SRF*>(this)->operator+(offset);
#endif
#ifdef NGEN_SAFE
    throw invalid_address_modifier_exception();
#endif
    return GRFDisp(GRF(), 0);
}
GRFDisp Subregister::operator-(int offset) const { return *this + (-offset); }

inline GRFDisp operator+(RegData s, GRF base)
{
#if XE4
    if (s.isSRF())
        return reinterpret_cast<SRF&>(s) + base;
#endif
#ifdef NGEN_SAFE
    throw invalid_address_modifier_exception();
#endif
    return GRFDisp(GRF(), 0);
}

inline GRFDisp operator+(RegData s, GRFDisp addr)
{
#if XE4
    if (s.isSRF())
        return reinterpret_cast<SRF&>(s) + addr;
#endif
#ifdef NGEN_SAFE
    throw invalid_address_modifier_exception();
#endif
    return GRFDisp(GRF(), 0);
}

inline GRFDisp operator+(GRF base,     RegData s) { return s + base; }
inline GRFDisp operator+(GRFDisp addr, RegData s) { return s + addr; }


inline RegisterRegion Subregister::operator()(int vs, int width, int hs) const
{
    RegisterRegion rr(*this, vs, width, hs);
    return rr;
}

inline RegisterRegion Subregister::operator()(int vs_or_width, int hs) const
{
    int vs, width;

    if (isIndirect()) {
        vs = -1;
        width = vs_or_width;
    } else {
        vs = vs_or_width;
        width = (hs == 0) ? ((vs == 0) ? 1 : vs) : vs / hs;
    }

    return operator()(vs, width, hs);
}

inline RegisterRegion Subregister::operator()(int hs) const
{
    return operator()(0, 0, hs);
}

inline Subregister Subregister::reinterpret(int offset, DataType type_) const
{
    Subregister r = *this;
    r.setType(type_);

    int o = getOffset();
    int oldbytes = getBits(), newbytes = r.getBits();
    int bitdiff = (oldbytes == 0) ? 0
                                  : (utils::log2(newbytes) - utils::log2(oldbytes));

    if (newbytes < oldbytes)
        r.setOffset((o << -bitdiff) + offset);
    else
        r.setOffset((o >>  bitdiff) + offset);

    return r;
}

// Indirect register and frames for making them.
class IndirectRegister : public Register {
protected:
    explicit constexpr14 IndirectRegister(const RegData &reg, RegFile rf, int offset = 0)
            : Register(reg.getScalarIndex(), rf)
    {
        if (reg.getARFType() == ARFType::s)
            base |= 0x100;
        indirect = true;
        off = offset;
#if XE4
        if (reg.isSRF()) _pad2 = 1;
#endif
    }
    template <RegFile rf> friend class IndirectRegisterFrame;

    IndirectRegister &operator=(const Invalid &i) { this->invalidate(); return *this; }
};

template <RegFile rf>
class IndirectRegisterFrame {
public:
    IndirectRegister operator[](const RegData &reg) const {
        return IndirectRegister(reg, rf);
    }
    IndirectRegister operator[](const GRFDisp &disp) const {
        return IndirectRegister(disp.getBase(), rf, disp.getDisp());
    }
};

// GRFRange represents a contiguous range of GRF registers.
class GRFRange {
protected:
#if XE3P
    uint16_t base;
    uint16_t len;

    static constexpr uint16_t invalidLen = 0xFFFF;
#else
    uint8_t base;
    uint8_t len;

    static constexpr uint8_t invalidLen = 0xFF;
#endif

public:
    GRFRange() : GRFRange(0, invalidLen) {}
    GRFRange(int base_, int len_) : base(base_), len(len_) {}
    GRFRange(RegData base_, int len_) : base(base_.getBase())
                                      , len(base_.isValid() ? len_ : invalidLen) {}

    int getBase()    const { return base; }
    int getLen()     const { return len; }
    bool isEmpty()   const { return len == 0; }
    bool isNull()    const { return false; }

    void invalidate()      { len = invalidLen; }
    bool isInvalid() const { return len == invalidLen; }
    bool isValid()   const { return !isInvalid(); }

    GRFRange &operator=(const Invalid &i) { this->invalidate(); return *this; }

    GRF operator[](int i) const {
#ifdef NGEN_SAFE
        if (isInvalid()) throw invalid_object_exception();
#endif
        return GRF(base + i);
    }

    operator GRF() const { return (*this)[0]; }

    inline Subregister sub(HW hw, int offset, DataType type) const;

    void fixup(HW hw, int execSize, int execWidth, DataType defaultType, int srcN, int arity) {}
    constexpr DataType getType() const { return DataType::invalid; }

#ifdef NGEN_ASM
    static const bool emptyOp = false;
    inline void outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const;
#endif
};

static inline GRFRange operator-(const GRF &reg1, const GRF &reg2)
{
    auto b1 = reg1.getBase(), b2 = reg2.getBase();
    int len = b2 + 1 - b1;

#ifdef NGEN_SAFE
    if (len < 0) throw invalid_range_exception();
#endif

    return GRFRange(reg1, len);
}

static inline bool operator==(const GRFRange &r1, const GRFRange &r2)
{
    return (r1.getBase() == r2.getBase()) && (r1.getLen() == r2.getLen());
}

static inline bool operator!=(const GRFRange &r1, const GRFRange &r2)
{
    return !(r1 == r2);
}

Subregister GRFRange::sub(HW hw, int offset, DataType type) const {
    const int lg2Len = GRF::log2Bytes(hw) - getLog2Bytes(type);
    return (*this)[offset >> lg2Len].sub(offset - ((offset >> lg2Len) << lg2Len), type);
}

#if XE4
// Contiguous range of GRF or SRF registers.
class RegisterRange {
protected:
    uint32_t base : 16;
    uint32_t len : 15;
    uint32_t srf : 1;

    static constexpr uint32_t invalidLen = 0x7FFF;

public:
    RegisterRange() : RegisterRange(0, invalidLen, false) {}
    RegisterRange(int base_, int len_, bool srf_) : base(base_), len(len_), srf(srf_) {}
    RegisterRange(RegData base_, int len_ = 1) {
        canonicalizeSRF(base_);
        base = base_.getBase();
        len = base_.isValid() ? len_ : invalidLen;
        srf = base_.isSRF();
    }
    RegisterRange(GRFRange range) : base(range.getBase()), len(range.getLen()), srf(false) {}

    int getBase()    const { return base; }
    int getLen()     const { return len; }
    bool isEmpty()   const { return len == 0; }
    bool isNull()    const { return false; }
    bool isSRF()     const { return srf; }

    void invalidate()      { len = invalidLen; }
    bool isInvalid() const { return len == invalidLen; }
    bool isValid()   const { return !isInvalid(); }

    RegisterRange &operator=(const Invalid &i) { this->invalidate(); return *this; }

    Register operator[](int i) const {
#ifdef NGEN_SAFE
        if (isInvalid()) throw invalid_object_exception();
#endif
        return srf ? Register(SRF(base + i)) : Register(GRF(base + i));
    }

    operator Register() const {
        if (isInvalid()) return Register();
        return (*this)[0];
    }

    void fixup(HW hw, int execSize, int execWidth, DataType defaultType, int srcN, int arity) {}
    constexpr DataType getType() const { return DataType::invalid; }

    GRFRange asGRFRange() const {
#ifdef NGEN_SAFE
        if (srf) throw invalid_register_file_exception();
#endif
        return GRFRange(base, len);
    }

#ifdef NGEN_ASM
    static const bool emptyOp = false;
    inline void outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const;
#endif
};

static inline RegisterRange operator-(const SRF &reg1, const SRF &reg2)
{
    auto b1 = reg1.getBase(), b2 = reg2.getBase();
    int len = b2 + 1 - b1;

#ifdef NGEN_SAFE
    if (len < 0) throw invalid_range_exception();
#endif

    return RegisterRange(reg1, len);
}
#else
using RegisterRange = GRFRange;
#endif

enum class ConditionModifier {
    none = 0,
    ze = 1,
    eq = 1,
    nz = 2,
    ne = 2,
    gt = 3,
    ge = 4,
    lt = 5,
    le = 6,
    ov = 8,
    un = 9,
    eo = 0xF
};

#ifdef NGEN_ASM
static inline std::ostream &operator<<(std::ostream &str, ConditionModifier cmod)
{
    static const char *names[16] = {"", "eq", "ne", "gt", "ge", "lt", "le", "", "ov", "un", "", "", "", "", "", "eo"};
    str << names[static_cast<uint8_t>(cmod) & 0xF];
    return str;
}

#if XE4
struct ConditionModifierXe4 {
    ConditionModifier cmod;
};

static inline std::ostream &operator<<(std::ostream &str, ConditionModifierXe4 cmod)
{
    static const char *names[16] = {"", "eq", "ne", "gt", "ge", "lt", "le", "", "ov", "nan", "", "", "", "", "", ""};
    str << names[static_cast<uint8_t>(cmod.cmod) & 0xF];
    return str;
}
#endif
#endif

enum class ChannelMask {
    rgba = 0,
    gba = 1,
    rba = 2,
    ba = 3,
    rga = 4,
    bga = 5,
    ga = 6,
    a = 7,
    rgb = 8,
    gb = 9,
    rb = 10,
    b = 11,
    rg = 12,
    g = 13,
    r = 14,
};

enum class PredCtrl {
    None = 0,
    Normal = 1,
    anyv = 2,
    allv = 3,
    any2h = 4,
    all2h = 5,
    any4h = 6,
    all4h = 7,
    any8h = 8,
    all8h = 9,
    any16h = 10,
    all16h = 11,
    any32h = 12,
    all32h = 13,
    any = 14,
    all = 15,
    x = 2,
    y = 3,
    z = 4,
    w = 5,
};

enum class ThreadCtrl {
    Normal = 0,
    Atomic = 1,
    Switch = 2,
    NoPreempt = 3
};

#if XE4
enum class RoundingOverride {
    none = 0,
    rne = 1,
    ru = 2,
    rd = 3,
    rtz = 4,
    rna = 5,
};

#ifdef NGEN_ASM
static inline std::ostream &operator<<(std::ostream &str, RoundingOverride rmo)
{
    static const char *names[8] = {"", "rne", "ru", "rd", "rtz", "rna", "", ""};
    str << names[static_cast<uint8_t>(rmo) & 0x7];
    return str;
}
#endif

/* Xe4 opcodes and type dispatching */
enum class OpcodeClassXe4 {
    abs, add_128A, add_64A, add_128D, add3, addc, asr, avg,
    bfe, bfegen, bfi, bfia, bfigen, bfn2, bfn3, bfrev, brepgen, brd,
    call, calla, callad, calld, cbit, cmp_128A, cmp_64B, cnvg, cvt, cvt2, dp4a,
    emcos, emexp2, eminv, eminvm, emlog2, emrsqt, emrsqtm, emsgmd, emsin, emsqt, emtanh,
    fbh, fbl, frc, geta, goto_, illegal, jmpi, join,
    mad_128A, mad_64C, mad_128D, madm, madlh, madc, max_, min_,
    mov_128R, mov_64I, movb, movg, movs, msk,
    mul_128A, mul_64A, mul_128D, mullh, nop128, nop64,
    redand, redfirst, redfirstidx, redmax, redmin, redor, redsum, redxor,
    ret, retd, rnd, rol, ror,
    sadd_128A, sadd_64A, sasr, sbfia, sbfn2, sbfn3, sbfrev, sbrepgen,
    scmp_128A, scmp_64B, sel, send, sendc, sendcg, sendg,
    seta, sgeta, shfld, shfli, shflsb, shflu, shflx, shl, shr, smad_128A, smad_64C,
    smov_128R, smov_64I, smul_128A, smul_64A, smullh, ssel, sseta,
    sshl, sshr, sbfi, sbfe, sfbh, sfbl, sbfegen, smsk, sbfigen,
    subb, sync, tarb, tmm, tmmd, tmmamx, tcvd, tcvdmx, tcvu, tcvumx, trng, tred, tmov,
    yield,

    abs_128A = abs,
    add3_128A = add3,
    addc_128K = addc,
    asr_128A = asr,
    avg_128A = avg,
    bfe_128G = bfe,
    bfegen_128A = bfegen,
    bfi_128G = bfi,
    bfia_128G = bfia,
    bfigen_128A = bfigen,
    bfn2_64D = bfn2,
    bfn3_128E = bfn3,
    bfrev_128G = bfrev,
    brd_128B = brd,
    brepgen_128A = brepgen,
    call_128B = call,
    calla_128P = calla,
    callad_128P = callad,
    calld_128B = calld,
    cbit_128A = cbit,
    cnvg_128L = cnvg,
    cvt_128O = cvt,
    cvt2_128O = cvt2,
    dp4a_128Q = dp4a,
    emcos_128A = emcos,
    emexp2_128A = emexp2,
    eminv_128A = eminv,
    eminvm_128H = eminvm,
    emlog2_128A = emlog2,
    emrsqt_128A = emrsqt,
    emrsqtm_128H = emrsqtm,
    emsgmd_128A = emsgmd,
    emsin_128A = emsin,
    emsqt_128A = emsqt,
    emtanh_128A = emtanh,
    fbh_128A = fbh,
    fbl_128A = fbl,
    frc_128A = frc,
    geta_128R = geta,
    goto__128B = goto_,
    illegal_128A = illegal,
    jmpi_128B = jmpi,
    join_128B = join,
    madc_128K = madc,
    madlh_128A = madlh,
    madm_128H = madm,
    max_128A = max_,
    min_128A = min_,
    movb_128I = movb,
    movg_128A = movg,
    movs_128A = movs,
    msk_64F = msk,
    mullh_128A = mullh,
    nop128_128A = nop128,
    nop64_64A = nop64,
    redand_128F = redand,
    redfirst_128F = redfirst,
    redfirstidx_128F = redfirstidx,
    redmax_128F = redmax,
    redmin_128F = redmin,
    redor_128F = redor,
    redsum_128F = redsum,
    redxor_128F = redxor,
    ret_128B = ret,
    retd_128B = retd,
    rnd_128A = rnd,
    rol_128A = rol,
    ror_128A = ror,
    sasr_128A = sasr,
    sbfe_128G = sbfe,
    sbfegen_128A = sbfegen,
    sbfi_128G = sbfi,
    sbfia_128G = sbfia,
    sbfigen_128A = sbfigen,
    sbfn2_64D = sbfn2,
    sbfn3_128E = sbfn3,
    sbfrev_128G = sbfrev,
    sbrepgen_128A = sbrepgen,
    sel_128J = sel,
    send_128C = send,
    sendc_128C = sendc,
    sendcg_128C = sendcg,
    sendg_128C = sendg,
    seta_128R = seta,
    sfbh_128A = sfbh,
    sfbl_128A = sfbl,
    sgeta_128R = sgeta,
    shfld_128F = shfld,
    shfli_128F = shfli,
    shflsb_128F = shflsb,
    shflu_128F = shflu,
    shflx_128F = shflx,
    shl_128A = shl,
    shr_128A = shr,
    smsk_64F = smsk,
    smullh_128A = smullh,
    ssel_128J = ssel,
    sseta_128R = sseta,
    sshl_128A = sshl,
    sshr_128A = sshr,
    subb_128K = subb,
    sync_64E = sync,
    tarb_64H = tarb,
    tcvd_128N = tcvd,
    tcvdmx_128N = tcvdmx,
    tcvu_128N = tcvu,
    tcvumx_128N = tcvumx,
    tmm_128M = tmm,
    tmmamx_128M = tmmamx,
    tmmd_128M = tmmd,
    tmov_128N = tmov,
    tred_128N = tred,
    trng_128N = trng,
    yield_64G = yield,
};

#define NGEN_DEF_XE4_SCALAR_OPCLASSES \
    NGEN_DEF_XE4_SCALAR_OPCLASS(add_128A) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(add_64A) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(asr) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(bfia) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(bfn2) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(bfn3) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(bfrev) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(brepgen) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(cmp_128A) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(cmp_64B) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(geta) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(mad_128A) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(mad_64C) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(mov_128R) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(mov_64I) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(mul_128A) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(mul_64A) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(mullh) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(sel) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(seta) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(shl) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(shr) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(bfi) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(bfe) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(fbh) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(fbl) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(bfegen) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(msk) \
    NGEN_DEF_XE4_SCALAR_OPCLASS(bfigen)

static inline bool isScalar(OpcodeClassXe4 opclass)
{
#define NGEN_DEF_XE4_SCALAR_OPCLASS(op) case OpcodeClassXe4::s##op:
    switch (opclass) {
        NGEN_DEF_XE4_SCALAR_OPCLASSES return true;
        case OpcodeClassXe4::jmpi_128B: return true;
        default: return false;
    }
#undef NGEN_DEF_XE4_SCALAR_OPCLASS
}

static inline OpcodeClassXe4 toScalar(OpcodeClassXe4 opclass)
{
#define NGEN_DEF_XE4_SCALAR_OPCLASS(op)                            \
        case OpcodeClassXe4::op:                                   \
        case OpcodeClassXe4::s##op: return OpcodeClassXe4::s##op;

    switch (opclass) {
        NGEN_DEF_XE4_SCALAR_OPCLASSES
        case OpcodeClassXe4::call:
        case OpcodeClassXe4::calla:
        case OpcodeClassXe4::calld:
        case OpcodeClassXe4::callad:
        case OpcodeClassXe4::cnvg:
        case OpcodeClassXe4::goto_:
        case OpcodeClassXe4::join:
        case OpcodeClassXe4::jmpi:
        case OpcodeClassXe4::nop64:
        case OpcodeClassXe4::nop128:
        case OpcodeClassXe4::tarb:
        case OpcodeClassXe4::yield:
            return opclass;
        default:
#ifdef NGEN_SAFE
            throw unsupported_scalar_operation_exception();
#endif
            return opclass;
    }
#undef NGEN_DEF_XE4_SCALAR_OPCLASS
}

#define NGEN_DEF_XE4_OPS \
NGEN_XE4_OP(abs, 128A, s32, 67) \
NGEN_XE4_OP(abs, 128A, s64, 83) \
NGEN_XE4_OP(abs, 128A, f32, 68) \
NGEN_XE4_OP(abs, 128A, f64, 69) \
NGEN_XE4_OP(abs, 128A, f16, 122) \
NGEN_XE4_OP(abs, 128A, bf16, 123) \
NGEN_XE4_OP(abs, 128A, f16v2, 70) \
NGEN_XE4_OP(abs, 128A, bf16v2, 71) \
NGEN_XE4_OP(add, 128A, u32, 74) \
NGEN_XE4_OP(add, 128A, s32, 75) \
NGEN_XE4_OP(add, 128A, u64, 76) \
NGEN_XE4_OP(add, 128A, s64, 77) \
NGEN_XE4_OP(add, 128A, u16v2, 45) \
NGEN_XE4_OP(add, 128A, s16v2, 78) \
NGEN_XE4_OP(add, 128A, s8v4, 79) \
NGEN_XE4_OP(add, 128A, f32, 80) \
NGEN_XE4_OP(add, 128A, f64, 268) \
NGEN_XE4_OP(add, 128A, f16, 124) \
NGEN_XE4_OP(add, 128A, bf16, 125) \
NGEN_XE4_OP(add, 128A, f16v2, 81) \
NGEN_XE4_OP(add, 128A, bf16v2, 82) \
NGEN_XE4_OP(add, 64A, u32, 85) \
NGEN_XE4_OP(add, 64A, s32, 86) \
NGEN_XE4_OP(add, 64A, u64, 87) \
NGEN_XE4_OP(add, 64A, s64, 88) \
NGEN_XE4_OP(add, 64A, u16v2, 46) \
NGEN_XE4_OP(add, 64A, s16v2, 89) \
NGEN_XE4_OP(add, 64A, s8v4, 90) \
NGEN_XE4_OP(add, 64A, f32, 91) \
NGEN_XE4_OP(add, 64A, f64, 315) \
NGEN_XE4_OP(add, 64A, f16, 126) \
NGEN_XE4_OP(add, 64A, bf16, 127) \
NGEN_XE4_OP(add, 64A, f16v2, 92) \
NGEN_XE4_OP(add, 64A, bf16v2, 93) \
NGEN_XE4_OP(add, 128D, u64, 96) \
NGEN_XE4_OP(add, 128D, s64, 97) \
NGEN_XE4_OP(add, 128D, f64, 316) \
NGEN_XE4_OP(add3, 128A, u32, 10) \
NGEN_XE4_OP(add3, 128A, s32, 241) \
NGEN_XE4_OP(add3, 128A, u64, 13) \
NGEN_XE4_OP(add3, 128A, s64, 242) \
NGEN_XE4_OP(addc, 128K, u32, 98) \
NGEN_XE4_OP(asr, 128A, s32, 26) \
NGEN_XE4_OP(asr, 128A, s64, 27) \
NGEN_XE4_OP(asr, 128A, s16v2, 347) \
NGEN_XE4_OP(asr, 128A, s8v4, 348) \
NGEN_XE4_OP(avg, 128A, s32, 99) \
NGEN_XE4_RAW_OP(bfe, 128G, b32, 28) \
NGEN_XE4_RAW_OP(bfegen, 128A, b32, 326) \
NGEN_XE4_RAW_OP(bfi, 128G, b32, 29) \
NGEN_XE4_RAW_OP(bfia, 128G, b32, 30) \
NGEN_XE4_RAW_OP(bfigen, 128A, b32, 305) \
NGEN_XE4_RAW_OP(bfn2, 64D, b32, 33) \
NGEN_XE4_RAW_OP(bfn3, 128E, b32, 306) \
NGEN_XE4_RAW_OP(bfrev, 128G, b32, 31) \
NGEN_XE4_RAW_OP(brepgen, 128A, b32, 325) \
NGEN_XE4_UNTYPED_OP(brd, 128B, 57) \
NGEN_XE4_UNTYPED_OP(call, 128B, 3) \
NGEN_XE4_UNTYPED_OP(calla, 128P, 4) \
NGEN_XE4_UNTYPED_OP(callad, 128P, 58) \
NGEN_XE4_UNTYPED_OP(calld, 128B, 59) \
NGEN_XE4_RAW_OP(cbit, 128A, b32, 34) \
NGEN_XE4_OP(cmp, 128A, u32, 100) \
NGEN_XE4_OP(cmp, 128A, s32, 101) \
NGEN_XE4_OP(cmp, 128A, u64, 102) \
NGEN_XE4_OP(cmp, 128A, s64, 103) \
NGEN_XE4_OP(cmp, 128A, u16v2, 47) \
NGEN_XE4_OP(cmp, 128A, s16v2, 104) \
NGEN_XE4_OP(cmp, 128A, s8v4, 105) \
NGEN_XE4_OP(cmp, 128A, f32, 106) \
NGEN_XE4_OP(cmp, 128A, f64, 107) \
NGEN_XE4_OP(cmp, 128A, f16, 233) \
NGEN_XE4_OP(cmp, 128A, bf16, 234) \
NGEN_XE4_OP(cmp, 128A, f16v2, 108) \
NGEN_XE4_OP(cmp, 128A, bf16v2, 109) \
NGEN_XE4_OP(cmp, 64B, u32, 112) \
NGEN_XE4_OP(cmp, 64B, s32, 113) \
NGEN_XE4_OP(cmp, 64B, u64, 114) \
NGEN_XE4_OP(cmp, 64B, s64, 115) \
NGEN_XE4_OP(cmp, 64B, u16v2, 48) \
NGEN_XE4_OP(cmp, 64B, s16v2, 116) \
NGEN_XE4_OP(cmp, 64B, s8v4, 117) \
NGEN_XE4_OP(cmp, 64B, f32, 118) \
NGEN_XE4_OP(cmp, 64B, f64, 119) \
NGEN_XE4_OP(cmp, 64B, f16, 235) \
NGEN_XE4_OP(cmp, 64B, bf16, 236) \
NGEN_XE4_OP(cmp, 64B, f16v2, 120) \
NGEN_XE4_OP(cmp, 64B, bf16v2, 121) \
NGEN_XE4_UNTYPED_OP(cnvg, 128L, 200) \
NGEN_XE4_OP(cvt, 128O, u8, 94) \
NGEN_XE4_OP(cvt, 128O, s8, 95) \
NGEN_XE4_OP(cvt, 128O, u16, 154) \
NGEN_XE4_OP(cvt, 128O, s16, 155) \
NGEN_XE4_OP(cvt, 128O, u32, 165) \
NGEN_XE4_OP(cvt, 128O, s32, 166) \
NGEN_XE4_OP(cvt, 128O, u64, 176) \
NGEN_XE4_OP(cvt, 128O, s64, 177) \
NGEN_XE4_OP(cvt, 128O, u16v2, 188) \
NGEN_XE4_OP(cvt, 128O, s16v2, 189) \
NGEN_XE4_OP(cvt, 128O, f32, 193) \
NGEN_XE4_OP(cvt, 128O, tf32, 194) \
NGEN_XE4_OP(cvt, 128O, f64, 201) \
NGEN_XE4_OP(cvt, 128O, f16, 204) \
NGEN_XE4_OP(cvt, 128O, bf16, 205) \
NGEN_XE4_OP(cvt, 128O, f16v2, 211) \
NGEN_XE4_OP(cvt, 128O, bf16v2, 212) \
NGEN_XE4_OP(cvt2, 128O, f16v2, 376) \
NGEN_XE4_OP(cvt2, 128O, bf16v2, 377) \
NGEN_XE4_OP(dp4a, 128Q, u32, 187) \
NGEN_XE4_OP(dp4a, 128Q, s32, 190) \
NGEN_XE4_OP(emcos, 128A, f32, 134) \
NGEN_XE4_OP(emcos, 128A, f16, 329) \
NGEN_XE4_OP(emcos, 128A, bf16, 338) \
NGEN_XE4_OP(emcos, 128A, f16v2, 213) \
NGEN_XE4_OP(emcos, 128A, bf16v2, 214) \
NGEN_XE4_OP(emexp2, 128A, f32, 135) \
NGEN_XE4_OP(emexp2, 128A, f16, 330) \
NGEN_XE4_OP(emexp2, 128A, bf16, 339) \
NGEN_XE4_OP(emexp2, 128A, f16v2, 219) \
NGEN_XE4_OP(emexp2, 128A, bf16v2, 220) \
NGEN_XE4_OP(eminv, 128A, f32, 136) \
NGEN_XE4_OP(eminv, 128A, f16, 331) \
NGEN_XE4_OP(eminv, 128A, bf16, 340) \
NGEN_XE4_OP(eminv, 128A, f16v2, 237) \
NGEN_XE4_OP(eminv, 128A, bf16v2, 238) \
NGEN_XE4_OP(eminvm, 128H, f32, 137) \
NGEN_XE4_OP(eminvm, 128H, f64, 327) \
NGEN_XE4_OP(emlog2, 128A, f32, 138) \
NGEN_XE4_OP(emlog2, 128A, f16, 332) \
NGEN_XE4_OP(emlog2, 128A, bf16, 341) \
NGEN_XE4_OP(emlog2, 128A, f16v2, 239) \
NGEN_XE4_OP(emlog2, 128A, bf16v2, 279) \
NGEN_XE4_OP(emrsqt, 128A, f32, 24) \
NGEN_XE4_OP(emrsqt, 128A, f16, 333) \
NGEN_XE4_OP(emrsqt, 128A, bf16, 342) \
NGEN_XE4_OP(emrsqt, 128A, f16v2, 283) \
NGEN_XE4_OP(emrsqt, 128A, bf16v2, 289) \
NGEN_XE4_OP(emrsqtm, 128H, f32, 139) \
NGEN_XE4_OP(emrsqtm, 128H, f64, 328) \
NGEN_XE4_OP(emsgmd, 128A, f32, 140) \
NGEN_XE4_OP(emsgmd, 128A, f16, 334) \
NGEN_XE4_OP(emsgmd, 128A, bf16, 343) \
NGEN_XE4_OP(emsgmd, 128A, f16v2, 308) \
NGEN_XE4_OP(emsgmd, 128A, bf16v2, 310) \
NGEN_XE4_OP(emsin, 128A, f32, 141) \
NGEN_XE4_OP(emsin, 128A, f16, 335) \
NGEN_XE4_OP(emsin, 128A, bf16, 344) \
NGEN_XE4_OP(emsin, 128A, f16v2, 362) \
NGEN_XE4_OP(emsin, 128A, bf16v2, 363) \
NGEN_XE4_OP(emsqt, 128A, f32, 142) \
NGEN_XE4_OP(emsqt, 128A, f16, 336) \
NGEN_XE4_OP(emsqt, 128A, bf16, 345) \
NGEN_XE4_OP(emsqt, 128A, f16v2, 364) \
NGEN_XE4_OP(emsqt, 128A, bf16v2, 365) \
NGEN_XE4_OP(emtanh, 128A, f32, 143) \
NGEN_XE4_OP(emtanh, 128A, f16, 337) \
NGEN_XE4_OP(emtanh, 128A, bf16, 346) \
NGEN_XE4_OP(emtanh, 128A, f16v2, 366) \
NGEN_XE4_OP(emtanh, 128A, bf16v2, 375) \
NGEN_XE4_RAW_OP(fbh, 128A, b32, 35) \
NGEN_XE4_RAW_OP(fbl, 128A, b32, 36) \
NGEN_XE4_OP(frc, 128A, f32, 144) \
NGEN_XE4_RAW_OP(geta, 128R, b32, 307) \
NGEN_XE4_UNTYPED_OP(goto_, 128B, 6) \
NGEN_XE4_UNTYPED_OP(illegal, 128A, 0) \
NGEN_XE4_UNTYPED_OP(jmpi, 128B, 8) \
NGEN_XE4_UNTYPED_OP(join, 128B, 9) \
NGEN_XE4_OP(mad, 128A, u32, 191) \
NGEN_XE4_OP(mad, 128A, s32, 192) \
NGEN_XE4_OP(mad, 128A, u16v2, 49) \
NGEN_XE4_OP(mad, 128A, s16v2, 195) \
NGEN_XE4_OP(mad, 128A, s8v4, 196) \
NGEN_XE4_OP(mad, 128A, f32, 197) \
NGEN_XE4_OP(mad, 128A, f64, 317) \
NGEN_XE4_OP(mad, 128A, f16, 221) \
NGEN_XE4_OP(mad, 128A, bf16, 222) \
NGEN_XE4_OP(mad, 128A, f16v2, 198) \
NGEN_XE4_OP(mad, 128A, bf16v2, 199) \
NGEN_XE4_OP(mad, 64C, u32, 202) \
NGEN_XE4_OP(mad, 64C, s32, 203) \
NGEN_XE4_OP(mad, 64C, u16v2, 50) \
NGEN_XE4_OP(mad, 64C, s16v2, 206) \
NGEN_XE4_OP(mad, 64C, s8v4, 207) \
NGEN_XE4_OP(mad, 64C, f32, 208) \
NGEN_XE4_OP(mad, 64C, f64, 318) \
NGEN_XE4_OP(mad, 64C, f16, 223) \
NGEN_XE4_OP(mad, 64C, bf16, 224) \
NGEN_XE4_OP(mad, 64C, f16v2, 209) \
NGEN_XE4_OP(mad, 64C, bf16v2, 210) \
NGEN_XE4_OP(mad, 128D, f32, 215) \
NGEN_XE4_OP(mad, 128D, f64, 319) \
NGEN_XE4_OP(madm, 128H, f32, 216) \
NGEN_XE4_OP(madm, 128H, f64, 169) \
NGEN_XE4_OP(madlh, 128A, u64, 170) \
NGEN_XE4_OP(madlh, 128A, s64, 180) \
NGEN_XE4_OP(madc, 128K, u64, 181) \
NGEN_XE4_OP(max, 128A, u32, 145) \
NGEN_XE4_OP(max, 128A, s32, 146) \
NGEN_XE4_OP(max, 128A, u64, 147) \
NGEN_XE4_OP(max, 128A, s64, 148) \
NGEN_XE4_OP(max, 128A, u16v2, 51) \
NGEN_XE4_OP(max, 128A, s16v2, 149) \
NGEN_XE4_OP(max, 128A, s8v4, 150) \
NGEN_XE4_OP(max, 128A, f32, 151) \
NGEN_XE4_OP(max, 128A, f64, 320) \
NGEN_XE4_OP(max, 128A, f16, 225) \
NGEN_XE4_OP(max, 128A, bf16, 226) \
NGEN_XE4_OP(max, 128A, f16v2, 152) \
NGEN_XE4_OP(max, 128A, bf16v2, 153) \
NGEN_XE4_OP(min, 128A, u32, 156) \
NGEN_XE4_OP(min, 128A, s32, 157) \
NGEN_XE4_OP(min, 128A, u64, 158) \
NGEN_XE4_OP(min, 128A, s64, 159) \
NGEN_XE4_OP(min, 128A, u16v2, 52) \
NGEN_XE4_OP(min, 128A, s16v2, 160) \
NGEN_XE4_OP(min, 128A, s8v4, 161) \
NGEN_XE4_OP(min, 128A, f32, 162) \
NGEN_XE4_OP(min, 128A, f64, 321) \
NGEN_XE4_OP(min, 128A, f16, 227) \
NGEN_XE4_OP(min, 128A, bf16, 228) \
NGEN_XE4_OP(min, 128A, f16v2, 163) \
NGEN_XE4_OP(min, 128A, bf16v2, 164) \
NGEN_XE4_RAW_OP(mov, 128R, b32, 17) \
NGEN_XE4_RAW_OP(mov, 128R, b64, 18) \
NGEN_XE4_RAW_OP(mov, 64I, b32, 19) \
NGEN_XE4_RAW_OP(mov, 64I, b64, 20) \
NGEN_XE4_RAW_OP(movb, 128I, b32, 25) \
NGEN_XE4_RAW_OP(movb, 128I, b64, 84) \
NGEN_XE4_RAW_OP(movg, 128A, b32, 22) \
NGEN_XE4_RAW_OP(movs, 128A, b32, 23) \
NGEN_XE4_RAW_OP(msk, 64F, b32, 282) \
NGEN_XE4_OP(mul, 128A, u32, 167) \
NGEN_XE4_OP(mul, 128A, s32, 168) \
NGEN_XE4_OP(mul, 128A, u16v2, 53) \
NGEN_XE4_OP(mul, 128A, s16v2, 171) \
NGEN_XE4_OP(mul, 128A, s8v4, 172) \
NGEN_XE4_OP(mul, 128A, f32, 173) \
NGEN_XE4_OP(mul, 128A, f64, 322) \
NGEN_XE4_OP(mul, 128A, f16, 229) \
NGEN_XE4_OP(mul, 128A, bf16, 230) \
NGEN_XE4_OP(mul, 128A, f16v2, 174) \
NGEN_XE4_OP(mul, 128A, bf16v2, 175) \
NGEN_XE4_OP(mul, 64A, u32, 178) \
NGEN_XE4_OP(mul, 64A, s32, 179) \
NGEN_XE4_OP(mul, 64A, u16v2, 54) \
NGEN_XE4_OP(mul, 64A, s16v2, 182) \
NGEN_XE4_OP(mul, 64A, s8v4, 183) \
NGEN_XE4_OP(mul, 64A, f32, 184) \
NGEN_XE4_OP(mul, 64A, f64, 323) \
NGEN_XE4_OP(mul, 64A, f16, 231) \
NGEN_XE4_OP(mul, 64A, bf16, 232) \
NGEN_XE4_OP(mul, 64A, f16v2, 185) \
NGEN_XE4_OP(mul, 64A, bf16v2, 186) \
NGEN_XE4_OP(mul, 128D, f64, 324) \
NGEN_XE4_OP(mullh, 128A, u64, 5) \
NGEN_XE4_OP(mullh, 128A, s64, 217) \
NGEN_XE4_UNTYPED_OP(nop128, 128A, 1) \
NGEN_XE4_UNTYPED_OP(nop64, 64A, 2) \
NGEN_XE4_RAW_OP(redand, 128F, b32, 302) \
NGEN_XE4_RAW_OP(redfirst, 128F, b32, 295) \
NGEN_XE4_OP(redfirstidx, 128F, u32, 299) \
NGEN_XE4_OP(redmax, 128F, u32, 291) \
NGEN_XE4_OP(redmax, 128F, s32, 292) \
NGEN_XE4_OP(redmax, 128F, f32, 290) \
NGEN_XE4_OP(redmax, 128F, f16, 367) \
NGEN_XE4_OP(redmax, 128F, bf16, 368) \
NGEN_XE4_OP(redmax, 128F, f16v2, 369) \
NGEN_XE4_OP(redmax, 128F, bf16v2, 370) \
NGEN_XE4_OP(redmin, 128F, u32, 300) \
NGEN_XE4_OP(redmin, 128F, s32, 301) \
NGEN_XE4_OP(redmin, 128F, f32, 296) \
NGEN_XE4_OP(redmin, 128F, f16, 371) \
NGEN_XE4_OP(redmin, 128F, bf16, 372) \
NGEN_XE4_OP(redmin, 128F, f16v2, 373) \
NGEN_XE4_OP(redmin, 128F, bf16v2, 374) \
NGEN_XE4_RAW_OP(redor, 128F, b32, 303) \
NGEN_XE4_OP(redsum, 128F, u32, 297) \
NGEN_XE4_OP(redsum, 128F, s32, 298) \
NGEN_XE4_RAW_OP(redxor, 128F, b32, 304) \
NGEN_XE4_UNTYPED_OP(ret, 128B, 11) \
NGEN_XE4_UNTYPED_OP(retd, 128B, 60) \
NGEN_XE4_OP(rnd, 128A, f32, 245) \
NGEN_XE4_OP(rnd, 128A, f64, 248) \
NGEN_XE4_OP(rnd, 128A, f16, 271) \
NGEN_XE4_OP(rnd, 128A, bf16, 274) \
NGEN_XE4_RAW_OP(rol, 128A, b32, 37) \
NGEN_XE4_RAW_OP(rol, 128A, b64, 38) \
NGEN_XE4_RAW_OP(rol, 128A, b16v2, 353) \
NGEN_XE4_RAW_OP(rol, 128A, b8v4, 354) \
NGEN_XE4_RAW_OP(ror, 128A, b32, 39) \
NGEN_XE4_RAW_OP(ror, 128A, b64, 40) \
NGEN_XE4_RAW_OP(ror, 128A, b16v2, 355) \
NGEN_XE4_RAW_OP(ror, 128A, b8v4, 356) \
NGEN_XE4_OP(sadd, 128A, u32, 243) \
NGEN_XE4_OP(sadd, 128A, s32, 244) \
NGEN_XE4_OP(sadd, 128A, u64, 258) \
NGEN_XE4_OP(sadd, 128A, s64, 259) \
NGEN_XE4_OP(sadd, 64A, u32, 246) \
NGEN_XE4_OP(sadd, 64A, s32, 247) \
NGEN_XE4_OP(sadd, 64A, u64, 264) \
NGEN_XE4_OP(sadd, 64A, s64, 265) \
NGEN_XE4_OP(sasr, 128A, s32, 249) \
NGEN_XE4_OP(sasr, 128A, s64, 250) \
NGEN_XE4_RAW_OP(sbfe, 128G, b32, 358) \
NGEN_XE4_RAW_OP(sbfegen, 128A, b32, 14) \
NGEN_XE4_RAW_OP(sbfi, 128G, b32, 357) \
NGEN_XE4_RAW_OP(sbfia, 128G, b32, 251) \
NGEN_XE4_RAW_OP(sbfigen, 128A, b32, 111) \
NGEN_XE4_RAW_OP(sbfn2, 64D, b32, 252) \
NGEN_XE4_RAW_OP(sbfn3, 128E, b32, 253) \
NGEN_XE4_RAW_OP(sbfrev, 128G, b32, 66) \
NGEN_XE4_RAW_OP(sbrepgen, 128A, b32, 65) \
NGEN_XE4_OP(scmp, 128A, u32, 254) \
NGEN_XE4_OP(scmp, 128A, s32, 255) \
NGEN_XE4_OP(scmp, 128A, u64, 256) \
NGEN_XE4_OP(scmp, 128A, s64, 257) \
NGEN_XE4_OP(scmp, 64B, u32, 260) \
NGEN_XE4_OP(scmp, 64B, s32, 261) \
NGEN_XE4_OP(scmp, 64B, u64, 262) \
NGEN_XE4_OP(scmp, 64B, s64, 263) \
NGEN_XE4_RAW_OP(sel, 128J, b32, 218) \
NGEN_XE4_UNTYPED_OP(send, 128C, 15) \
NGEN_XE4_UNTYPED_OP(sendc, 128C, 16) \
NGEN_XE4_UNTYPED_OP(sendcg, 128C, 73) \
NGEN_XE4_UNTYPED_OP(sendg, 128C, 72) \
NGEN_XE4_RAW_OP(seta, 128R, b32, 309) \
NGEN_XE4_RAW_OP(sfbh, 128A, b32, 359) \
NGEN_XE4_RAW_OP(sfbl, 128A, b32, 360) \
NGEN_XE4_RAW_OP(sgeta, 128R, b32, 313) \
NGEN_XE4_RAW_OP(shfld, 128F, b32, 293) \
NGEN_XE4_RAW_OP(shfli, 128F, b32, 287) \
NGEN_XE4_RAW_OP(shflsb, 128F, b32, 294) \
NGEN_XE4_RAW_OP(shflu, 128F, b32, 286) \
NGEN_XE4_RAW_OP(shflx, 128F, b32, 288) \
NGEN_XE4_RAW_OP(shl, 128A, b32, 41) \
NGEN_XE4_RAW_OP(shl, 128A, b64, 42) \
NGEN_XE4_RAW_OP(shl, 128A, b16v2, 349) \
NGEN_XE4_RAW_OP(shl, 128A, b8v4, 350) \
NGEN_XE4_RAW_OP(shr, 128A, b32, 43) \
NGEN_XE4_RAW_OP(shr, 128A, b64, 44) \
NGEN_XE4_RAW_OP(shr, 128A, b16v2, 351) \
NGEN_XE4_RAW_OP(shr, 128A, b8v4, 352) \
NGEN_XE4_OP(smad, 128A, u32, 277) \
NGEN_XE4_OP(smad, 128A, s32, 278) \
NGEN_XE4_OP(smad, 64C, u32, 280) \
NGEN_XE4_OP(smad, 64C, s32, 281) \
NGEN_XE4_RAW_OP(smov, 128R, b32, 266) \
NGEN_XE4_RAW_OP(smov, 128R, b64, 7) \
NGEN_XE4_RAW_OP(smov, 64I, b32, 267) \
NGEN_XE4_RAW_OP(smov, 64I, b64, 62) \
NGEN_XE4_RAW_OP(smsk, 64F, b32, 110) \
NGEN_XE4_OP(smul, 128A, u32, 269) \
NGEN_XE4_OP(smul, 128A, s32, 270) \
NGEN_XE4_OP(smul, 64A, u32, 272) \
NGEN_XE4_OP(smul, 64A, s32, 273) \
NGEN_XE4_OP(smullh, 128A, u64, 275) \
NGEN_XE4_OP(smullh, 128A, s64, 276) \
NGEN_XE4_RAW_OP(ssel, 128J, b32, 361) \
NGEN_XE4_RAW_OP(sseta, 128R, b32, 311) \
NGEN_XE4_RAW_OP(sshl, 128A, b32, 284) \
NGEN_XE4_RAW_OP(sshl, 128A, b64, 312) \
NGEN_XE4_RAW_OP(sshr, 128A, b32, 285) \
NGEN_XE4_RAW_OP(sshr, 128A, b64, 314) \
NGEN_XE4_OP(subb, 128K, u32, 240) \
NGEN_XE4_UNTYPED_OP(sync, 64E, 12) \
NGEN_XE4_UNTYPED_OP(tarb, 64H, 32) \
NGEN_XE4_OP(tmm, 128M, s32, 448) \
NGEN_XE4_OP(tmm, 128M, f32, 449) \
NGEN_XE4_OP(tmm, 128M, f16, 450) \
NGEN_XE4_OP(tmm, 128M, bf16, 451) \
NGEN_XE4_OP(tmmd, 128M, f16, 456) \
NGEN_XE4_OP(tmmd, 128M, bf16, 457) \
NGEN_XE4_OP(tmmamx, 128M, f32, 453) \
NGEN_XE4_OP(tcvd, 128N, f16, 488) \
NGEN_XE4_OP(tcvd, 128N, bf16, 489) \
NGEN_XE4_OP(tcvdmx, 128N, f16, 480) \
NGEN_XE4_OP(tcvdmx, 128N, bf16, 481) \
NGEN_XE4_OP(tcvu, 128N, f16, 504) \
NGEN_XE4_OP(tcvu, 128N, bf16, 505) \
NGEN_XE4_OP(tcvu, 128N, e5m2, 506) \
NGEN_XE4_OP(tcvu, 128N, e4m3, 507) \
NGEN_XE4_OP(tcvumx, 128N, f16, 496) \
NGEN_XE4_OP(tcvumx, 128N, bf16, 497) \
NGEN_XE4_RAW_OP(trng, 128N, b32, 464) \
NGEN_XE4_RAW_OP(trng, 128N, b16v2, 465) \
NGEN_XE4_RAW_OP(trng, 128N, b8v4, 466) \
NGEN_XE4_OP(tred, 128N, f16, 500) \
NGEN_XE4_OP(tred, 128N, bf16, 501) \
NGEN_XE4_UNTYPED_OP(tmov, 128N, 510) \
NGEN_XE4_UNTYPED_OP(yield, 64G, 56) \

#endif

enum class Opcode {
    illegal = 0x00,
    sync = 0x01,
    mov = 0x01,
    sel = 0x02,
    movi = 0x03,
    not_ = 0x04,
    and_ = 0x05,
    or_ = 0x06,
    xor_ = 0x07,
    shr = 0x08,
    shl = 0x09,
    smov = 0x0A,
    asr = 0x0C,
    ror = 0x0E,
    rol = 0x0F,
    cmp = 0x10,
    cmpn = 0x11,
    csel = 0x12,
    bfrev = 0x17,
    bfe = 0x18,
    bfi1 = 0x19,
    bfi2 = 0x1A,
    jmpi = 0x20,
    brd = 0x21,
    if_ = 0x22,
    brc = 0x23,
    else_ = 0x24,
    endif = 0x25,
    while_ = 0x27,
    break_ = 0x28,
    cont = 0x29,
    halt = 0x2A,
    calla = 0x2B,
    call = 0x2C,
    ret = 0x2D,
    goto_ = 0x2E,
    join = 0x2F,
    wait = 0x30,
    send = 0x31,
    sendc = 0x32,
    sends = 0x33,
    sendsc = 0x34,
#if XE3P
    sendg = 0x33,
    sendgc = 0x34,
    sendgx = 0x35,
    sendgxc = 0x36,
#endif
    math = 0x38,
#if XE3P
    lfsr = 0x39,
#endif
    add = 0x40,
    mul = 0x41,
    avg = 0x42,
    frc = 0x43,
    rndu = 0x44,
    rndd = 0x45,
    rnde = 0x46,
    rndz = 0x47,
    mac = 0x48,
    mach = 0x49,
    lzd = 0x4A,
    fbh = 0x4B,
    fbl = 0x4C,
    cbit = 0x4D,
    addc = 0x4E,
    subb = 0x4F,
    sad2 = 0x50,
#if XE3P
    shfl = 0x50,
#endif
    sada2 = 0x51,
    add3 = 0x52,
    macl = 0x53,
    srnd = 0x54,
    dp4 = 0x54,
    dph = 0x55,
#if XE3P
    dnscl = 0x55,
#endif
    dp3 = 0x56,
    dp2 = 0x57,
    dp4a = 0x58,
    line = 0x59,
    dpas = 0x59,
    pln = 0x5A,
    dpasw = 0x5A,
    mad = 0x5B,
    lrp = 0x5C,
#if XE3P
    bdpas = 0x5C,
#endif
    madm = 0x5D,
#if XE3P
    mullh = 0x5F,
#endif
    nop_gen12 = 0x60,
    mov_gen12 = 0x61,
    sel_gen12 = 0x62,
    movi_gen12 = 0x63,
    not_gen12 = 0x64,
    and_gen12 = 0x65,
    or_gen12 = 0x66,
    xor_gen12 = 0x67,
    shr_gen12 = 0x68,
    shl_gen12 = 0x69,
    smov_gen12 = 0x6A,
    bfn = 0x6B,
    asr_gen12 = 0x6C,
    ror_gen12 = 0x6E,
    rol_gen12 = 0x6F,
    cmp_gen12 = 0x70,
    cmpn_gen12 = 0x71,
    csel_gen12 = 0x72,
    bfrev_gen12 = 0x77,
    bfe_gen12 = 0x78,
    bfi1_gen12 = 0x79,
    bfi2_gen12 = 0x7A,
    nop = 0x7E,
    directive = 0x7F,   /* not a valid opcode; used internally by nGEN */
#if XE4
    directive_xe4 = 0x81FF,
#define NGEN_XE4_UNTYPED_OP(cls, enc, opcode) cls##_##enc = 0x8000 | opcode,
#define NGEN_XE4_RAW_OP NGEN_XE4_OP
#define NGEN_XE4_OP(cls, enc, dt, opcode) cls##_##enc##_##dt = 0x8000 | opcode,
    NGEN_DEF_XE4_OPS
#undef NGEN_XE4_OP
#undef NGEN_XE4_RAW_OP
#undef NGEN_XE4_UNTYPED_OP
#endif
};

#if XE4
static inline bool isXe4(Opcode op) { return (static_cast<int>(op) & 0x8000); }
#endif

enum class Operand {dst = 0, src0 = 1, src1 = 2, src2 = 3};

enum class Directive {
    ignoredep_dst = 0,
    ignoredep_src0 = 1,
    ignoredep_src1 = 2,
    ignoredep_src2 = 3,
    subdep_dst = 8,
    wrdep = 0x10,
    fencedep = 0x11,
    pvcwarwa = 0x20,
};

static inline bool isSend(Opcode op)
{
    switch (op) {
        case Opcode::send:
        case Opcode::sendc:
        case Opcode::sends:
        case Opcode::sendsc:
#if XE3P
        case Opcode::sendgx:
        case Opcode::sendgxc:
#endif
#if XE4
        case Opcode::send_128C:
        case Opcode::sendc_128C:
        case Opcode::sendg_128C:
        case Opcode::sendcg_128C:
#endif
            return true;
        default:
            return false;
    }
}

static inline bool trackedByToken(HW hw, Opcode op, unsigned dstTypecode)
{
    switch (op) {
        case Opcode::math:
            if (hw >= HW::XeHPC) return false;
            /* fall through */
        case Opcode::dpas:
        case Opcode::dpasw:
            return true;
#if XE3P
        case Opcode::bdpas:
            return (hw >= HW::Xe3p);
#endif
        default:
            if (isSend(op)) return true;
            if (hw == HW::XeHPG && dstTypecode == 0b1011 /* :df */) return true;
            return false;
    }
}

static inline bool isBranch(Opcode op)
{
#if XE4
    if (isXe4(op)) switch (op) {
        case Opcode::brd_128B:
        case Opcode::call_128B:
        case Opcode::calla_128P:
        case Opcode::calld_128B:
        case Opcode::callad_128P:
        case Opcode::goto__128B:
        case Opcode::jmpi_128B:
        case Opcode::join_128B:
        case Opcode::ret_128B:
        case Opcode::retd_128B:
            return true;
        default: return false;
    } else
#endif
    return (static_cast<int>(op) >> 4) == 2;
}

#if XE4
static inline bool isCvt(Opcode op)
{
    switch (op) {
        case Opcode::cvt_128O_u8:
        case Opcode::cvt_128O_s8:
        case Opcode::cvt_128O_u16:
        case Opcode::cvt_128O_s16:
        case Opcode::cvt_128O_u32:
        case Opcode::cvt_128O_s32:
        case Opcode::cvt_128O_u64:
        case Opcode::cvt_128O_s64:
        case Opcode::cvt_128O_u16v2:
        case Opcode::cvt_128O_s16v2:
        case Opcode::cvt_128O_f32:
        case Opcode::cvt_128O_tf32:
        case Opcode::cvt_128O_f64:
        case Opcode::cvt_128O_f16:
        case Opcode::cvt_128O_bf16:
        case Opcode::cvt_128O_f16v2:
        case Opcode::cvt_128O_bf16v2:
        case Opcode::cvt2_128O_f16v2:
        case Opcode::cvt2_128O_bf16v2: return true;
        default:                       return false;
    }
}

static constexpr14 inline Opcode opcodeXe4(OpcodeClassXe4 cls, DataType dt)
{
    auto rawDT = rawType(dt);

    // TODO: make this faster.
#define NGEN_XE4_UNTYPED_OP(cls_, enc_, opcode) \
    if (cls == OpcodeClassXe4::cls_##_##enc_) return Opcode::cls_##_##enc_;
#define NGEN_XE4_RAW_OP(cls_, enc_, dt_, opcode) \
    if (cls == OpcodeClassXe4::cls_##_##enc_ && rawDT == DataType::dt_) return Opcode::cls_##_##enc_##_##dt_;
#define NGEN_XE4_OP(cls_, enc_, dt_, opcode) \
    if (cls == OpcodeClassXe4::cls_##_##enc_ && dt == DataType::dt_) return Opcode::cls_##_##enc_##_##dt_;
NGEN_DEF_XE4_OPS
#undef NGEN_XE4_OP
#undef NGEN_XE4_RAW_OP
#undef NGEN_XE4_UNTYPED_OP

#ifdef NGEN_SAFE
    if (dt == DataType::invalid)
        throw missing_type_exception();
    else
        throw invalid_type_exception();
#endif
    return Opcode::illegal;
}

enum class EncodingXe4 {
    _128A, _128B, _128C, _128D, _128E, _128F, _128G, _128H, _128I, _128J, _128K, _128L, _128M, _128N, _128O, _128P, _128Q, _128R, _128S,
    _64A, _64B, _64C, _64D, _64E, _64F, _64G, _64H, _64I,
};


#ifdef NGEN_ASM
static inline std::ostream &operator<<(std::ostream &str, EncodingXe4 enc)
{
    if (enc >= EncodingXe4::_64A)
        str << "64" << char('A' + (int(enc) - int(EncodingXe4::_64A)));
    else
        str << "128" << char('A' + (int(enc) - int(EncodingXe4::_128A)));
    return str;
}
#endif

static inline EncodingXe4 getEncodingXe4(Opcode op)
{
#define NGEN_XE4_UNTYPED_OP(cls_, enc_, opcode) \
    if (op == Opcode::cls_##_##enc_) return EncodingXe4::_##enc_;
#define NGEN_XE4_RAW_OP(cls_, enc_, dt_, opcode) \
    if (op == Opcode::cls_##_##enc_##_##dt_) return EncodingXe4::_##enc_;
#define NGEN_XE4_OP(cls_, enc_, dt_, opcode) NGEN_XE4_RAW_OP(cls_, enc_, dt_, opcode)
NGEN_DEF_XE4_OPS
#undef NGEN_XE4_OP
#undef NGEN_XE4_RAW_OP
#undef NGEN_XE4_UNTYPED_OP
    return EncodingXe4::_128A;
}

static inline bool isEncoding64(EncodingXe4 enc)
{
    return (enc >= EncodingXe4::_64A);
}

static inline int swsbItems(EncodingXe4 enc)
{
    switch (enc) {
        case EncodingXe4::_128A:
        case EncodingXe4::_128B:
        case EncodingXe4::_128E:
        case EncodingXe4::_128F:
        case EncodingXe4::_128I:
        case EncodingXe4::_128J:
        case EncodingXe4::_128K:
        case EncodingXe4::_128L:
        case EncodingXe4::_128O:
        case EncodingXe4::_128P:
        case EncodingXe4::_128Q:
        case EncodingXe4::_128R:
        case EncodingXe4::_64E:
            return 2;
        case EncodingXe4::_128C:
        case EncodingXe4::_128D:
        case EncodingXe4::_128G:
        case EncodingXe4::_128H:
        case EncodingXe4::_64A:
        case EncodingXe4::_64B:
        case EncodingXe4::_64C:
        case EncodingXe4::_64D:
        case EncodingXe4::_64F:
        case EncodingXe4::_64G:
            return 1;
        default:
            return 0;
    }
}

static inline DataType dstDataType(Opcode op)
{
    if (!isXe4(op))
        return DataType::invalid;

    using DT = DataType;
    auto _ = DT::invalid;
    auto f64 = DT::f64, f32 = DT::f32;
    auto f16 = DT::f16, bf16 = DT::bf16, f16v2 = DT::f16v2, bf16v2 = DT::bf16v2;
    auto e4m3 = DT::e4m3, e5m2 = DT::e5m2;
    auto b64 = DT::b64, b32 = DT::b32, b16v2 = DT::b16v2, b8v4 = DT::b8v4;
    auto s64 = DT::s64, s32 = DT::s32, s16 = DT::s16, s16v2 = DT::s16v2, s8 = DT::s8, s8v4 = DT::s8v4;
    auto u64 = DT::u64, u32 = DT::u32, u16 = DT::u16, u16v2 = DT::u16v2, u8 = DT::u8;
    auto tf32 = DT::tf32;

    static const DataType dt[512] = {
        _, _, _, _, _, u64, _, b64,
        _, _, u32, _, _, u64, b32, _,
        _, b32, b64, b32, b64, b64, b32, b32,
        f32, b32, s32, s64, b32, b32, b32, b32,
        _, b32, b32, b32, b32, b32, b64, b32,
        b64, b32, b64, b32, b64, u16v2, u16v2, u16v2,
        u16v2, u16v2, u16v2, u16v2, u16v2, u16v2, u16v2, b64,
        _, _, _, _, _, _, b64, _,
        _, b32, b32, s32, f32, f64, f16v2, bf16v2,
        _, _, u32, s32, u64, s64, s16v2, s8v4,
        f32, f16v2, bf16v2, s64, b64, u32, s32, u64,
        s64, s16v2, s8v4, f32, f16v2, bf16v2, u8, s8,
        u64, s64, u32, s32, u32, s32, u64, s64,
        s16v2, s8v4, f32, f64, f16v2, bf16v2, b32, b32,
        u32, s32, u64, s64, s16v2, s8v4, f32, f64,
        f16v2, bf16v2, f16, bf16, f16, bf16, f16, bf16,
        _, _, _, _, _, _, f32, f32,
        f32, f32, f32, f32, f32, f32, f32, f32,
        f32, u32, s32, u64, s64, s16v2, s8v4, f32,
        f16v2, bf16v2, u16, s16, u32, s32, u64, s64,
        s16v2, s8v4, f32, f16v2, bf16v2, u32, s32, u32,
        s32, f64, u64, s16v2, s8v4, f32, f16v2, bf16v2,
        u64, s64, u32, s32, s64, u64, s16v2, s8v4,
        f32, f16v2, bf16v2, u32, u16v2, s16v2, s32, u32,
        s32, f32, tf32, s16v2, s8v4, f32, f16v2, bf16v2,
        _, f64, u32, s32, f16, bf16, s16v2, s8v4,
        f32, f16v2, bf16v2, f16v2, bf16v2, f16v2, bf16v2, f32,
        f32, s64, b32, f16v2, bf16v2, f16, bf16, f16,
        bf16, f16, bf16, f16, bf16, f16, bf16, f16,
        bf16, f16, bf16, f16, bf16, f16v2, bf16v2, f16v2,
        u32, s32, s64, u32, s32, f32, u32, s32,
        f64, s32, s64, b32, b32, b32, u32, s32,
        u64, s64, u64, s64, u32, s32, u64, s64,
        u64, s64, b32, b32, f64, u32, s32, f16,
        u32, s32, bf16, u64, s64, u32, s32, bf16v2,
        u32, s32, b32, f16v2, b32, b32, b32, b32,
        b32, bf16v2, f32, u32, s32, b32, b32, b32,
        f32, u32, s32, u32, u32, s32, b32, b32,
        b32, b32, b32, b32, f16v2, b32, bf16v2, b32,
        b64, b32, b64, f64, f64, f64, f64, f64,
        f64, f64, f64, f64, f64, b32, b32, f64,
        f64, f16, f16, f16, f16, f16, f16, f16,
        f16, f16, bf16, bf16, bf16, bf16, bf16, bf16,
        bf16, bf16, bf16, s16v2, s8v4, b16v2, b8v4, b16v2,
        b8v4, b16v2, b8v4, b16v2, b8v4, b32, b32, b32,
        b32, b32, f16v2, bf16v2, f16v2, bf16v2, f16v2, f16,
        bf16, f16v2, bf16v2, f16, bf16, f16v2, bf16v2, bf16v2,
        f16v2, bf16v2, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        s32, f32, f16, bf16, _, f32, _, _,
        f16, bf16, _, _, _, _, _, _,
        b32, b16v2, b8v4, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        f16, bf16, _, _, _, _, _, _,
        f16, bf16, _, _, _, _, _, _,
        f16, bf16, _, _, f16, bf16, _, _,
        f16, bf16, e5m2, e4m3, _, _, _, _,
    };

    return dt[static_cast<int>(op) & 0x1FF];
}
#endif

class AllPipes {};
enum class Pipe : uint8_t {
    Default = 0,
    A = 1, All = A,
    F = 2, Float = F,
    I = 3, Integer = I,
    L = 4, Long = L,
    M = 5, Math = M,
    S = 6, Scalar = S,
#if XE4
    X = 7, Shuffle = X, shfl = X,
    int_ = I,
    float_ = F,
    float64 = L,
    ext = M,
    scl = S,
#endif
};

#ifdef NGEN_ASM
static inline std::ostream &operator<<(std::ostream &str, Pipe pipe)
{
    static const char *names[8] = {"", "A", "F", "I", "L", "M", "S", ""};
    str << names[static_cast<uint8_t>(pipe) & 7];
    return str;
}

#if XE4
struct PipeXe4 {
    Pipe p;
};

static inline std::ostream &operator<<(std::ostream &str, PipeXe4 pipe)
{
    static const char *names[8] = {"", "all", "f32", "int", "f64", "em", "scl", "shfl"};
    str << names[static_cast<uint8_t>(pipe.p) & 7];
    return str;
}
#endif
#endif

class SWSBItem
{
    friend class InstructionModifier;

public:
    union {
        struct {
            uint8_t dist : 3;
            uint8_t pipe : 4;
            uint8_t isToken : 1;
        } pipe;
        struct {
            uint8_t token : 5;
            uint8_t dst : 1;
            uint8_t src : 1;
            uint8_t isToken : 1;
        } token;
        uint8_t all;
    };

    constexpr bool isToken() const       { return token.isToken; }
    constexpr bool hasTokenSet() const   { return token.isToken && token.src && token.dst; }
    constexpr int getToken() const       { return token.token; }
    constexpr unsigned tokenMode() const { return ((token.src << 1) | token.dst); }

    constexpr bool isPipe() const        { return !empty() && !pipe.isToken; }
    constexpr Pipe getPipe() const       { return static_cast<Pipe>(pipe.pipe); }
    void setPipe(Pipe pipe_)             { pipe.pipe = static_cast<unsigned>(pipe_); }

    constexpr bool empty() const         { return (all == 0); }

    explicit operator bool() const       { return !empty(); }
    bool operator!() const               { return empty(); }

protected:
    explicit constexpr SWSBItem(uint8_t all_) : all(all_) {}

public:
    constexpr SWSBItem() : all(0) {}
    constexpr SWSBItem(Pipe pipe_, int dist_) : all((dist_ & 0x7) | (static_cast<unsigned>(pipe_) << 3)) {}
    constexpr SWSBItem(int token_, bool src_, bool dst_) : all((token_ & 0x1F) | (uint16_t(src_) << 6) | (uint16_t(dst_) << 5) | 0x80) {}

    static constexpr SWSBItem createNoAccSBSet() { return SWSBItem(0x08); }
    constexpr bool isNoAccSBSet() const          { return (all == 0x08); }

    static uint32_t pack4(std::array<SWSBItem, 4> items) {
        return items[0].all | (items[1].all << 8) | (items[2].all << 16) | (items[3].all << 24);
    }
    static std::array<SWSBItem, 4> unpack4(uint32_t i) {
        std::array<SWSBItem, 4> result;
        result[0].all =  i        & 0xFF;
        result[1].all = (i >> 8)  & 0xFF;
        result[2].all = (i >> 16) & 0xFF;
        result[3].all = (i >> 24) & 0xFF;
        return result;
    }
};

static_assert(sizeof(SWSBItem) == 1, "SWSBItem has been padded by the compiler");

using SWSBInfo = std::array<SWSBItem, 2>;

template <typename T> static constexpr Pipe getPipe() { return (sizeof(T) == 8) ? Pipe::L : Pipe::I; }
template <> constexpr Pipe getPipe<float>()           { return Pipe::F; }
template <> constexpr Pipe getPipe<void>()            { return Pipe::Default; }
template <> constexpr Pipe getPipe<AllPipes>()        { return Pipe::A; }

constexpr SWSBItem SWSB(SWSBItem info)                              { return info; }
constexpr SWSBItem SWSB(Pipe pipe, int dist)                        { return SWSBItem(pipe, dist); }
template <typename T = void> constexpr SWSBItem SWSB(int dist = 1)  { return SWSB(getPipe<T>(), dist); }

template <typename T = void> constexpr InstructionModifier SWSB(SWSBItem item, int dist);

static inline void normalizeSWSB(SWSBInfo &info) {
    if (info[0].isPipe() || info[1].isToken())
        std::swap(info[0], info[1]);
}

// Token count.
constexpr inline int tokenCount(HW hw, int grfCount = 128)
{
    return (hw == HW::Xe2 && grfCount < 256) ? 16 :
                           (hw >= HW::XeHPC) ? 32 :
                         (hw >= HW::Gen12LP) ? 16
                                             : 0;
}

class SBID
{
public:
    SWSBItem set;
    SWSBItem src;
    SWSBItem dst;

    constexpr explicit SBID(int id) : set(id, true, true), src(id, true, false), dst(id, false, true) {}
    constexpr operator SWSBItem() const { return set; }

    constexpr int getID() const { return set.getToken(); }
};

class InstructionModifier {
protected:
    union {
        struct {
#if XE4
            unsigned cflag : 4;
            unsigned _pad1 : 4;
#else
            unsigned _pad1 : 8;
#endif
            unsigned accessMode : 1;        // From here on matches the low 64-bits of the binary format for Gen8-11
            unsigned noDDClr : 1;
            unsigned noDDChk : 1;
            unsigned chanOff : 3;
            unsigned threadCtrl : 2;
            unsigned predCtrl : 4;
            unsigned predInv : 1;
            unsigned log2ExecSize : 3;
            unsigned cmod : 4;              // Also stores channel mask temporarily for surface r/w
            unsigned accWrCtrl : 1;         // = noSrcDepSet for send, = branchCtrl for branch instructions
            unsigned cmptCtrl : 1;
            unsigned debugCtrl : 1;
            unsigned saturate : 1;
            unsigned flagSubRegNum : 1;
            unsigned flagRegNum : 1;
            unsigned maskCtrl : 1;
            unsigned exBSO : 1;
#if XE4
            unsigned rmo : 3;
            unsigned _pad2: 2;
            unsigned esizeSet : 1;
            unsigned flagRegNum1 : 3;
#else
            unsigned _pad2: 8;
            unsigned flagRegNum1 : 1;
#endif
            unsigned autoSWSB : 1;
            unsigned fusionCtrl : 1;        // Gen12
            unsigned eot : 1;
            unsigned swsb0 : 8;
            unsigned swsb1 : 8;
        } parts;
        uint64_t all;
    };

    constexpr InstructionModifier(uint64_t all_) : all(all_) {}

    friend inline void encodeCommon12(Instruction12 &i, Opcode opcode, const InstructionModifier &mod, const RegData &dst, EncodingTag12 tag);
    friend inline void encodeCommon12(Instruction12 &i, Opcode opcode, const InstructionModifier &mod, const RegData &dst, EncodingTagXeHPC tag);

public:
    constexpr int getExecSize()            const { return 1 << parts.log2ExecSize; }
    constexpr bool isAlign16()             const { return parts.accessMode; }
    constexpr bool isNoDDClr()             const { return parts.noDDClr; }
    constexpr bool isNoDDChk()             const { return parts.noDDChk; }
    constexpr int getChannelOffset()       const { return parts.chanOff << 2; }
    constexpr ThreadCtrl getThreadCtrl()   const { return static_cast<ThreadCtrl>(parts.threadCtrl); }
    constexpr bool isAtomic()              const { return getThreadCtrl() == ThreadCtrl::Atomic; }
    constexpr PredCtrl getPredCtrl()       const { return static_cast<PredCtrl>(parts.predCtrl); }
    constexpr bool isPredInv()             const { return parts.predInv; }
    constexpr ConditionModifier getCMod()  const { return static_cast<ConditionModifier>(parts.cmod); }
    constexpr bool isAccWrEn()             const { return parts.accWrCtrl; }
    constexpr bool getBranchCtrl()         const { return parts.accWrCtrl; }
#if XE3P
    constexpr bool isFwd()                 const { return parts.accWrCtrl; }
#endif
    constexpr bool isCompact()             const { return parts.cmptCtrl; }
    constexpr bool isBreakpoint()          const { return parts.debugCtrl; }
    constexpr bool isSaturate()            const { return parts.saturate; }
    constexpr14 FlagRegister getFlagReg()  const { return FlagRegister((parts.flagRegNum1 << 1) | parts.flagRegNum, parts.flagSubRegNum); }
#if XE4
    constexpr14 FlagRegister getCFlag()    const { return FlagRegister(parts.cflag); }
#endif
    constexpr bool isWrEn()                const { return parts.maskCtrl; }
    constexpr bool isExBSO()               const { return parts.exBSO; }
#if XE4
    constexpr RoundingOverride getRounding() const { return static_cast<RoundingOverride>(parts.rmo); }
    constexpr bool hasExecSize()           const { return parts.esizeSet; }
#endif
    constexpr bool isAutoSWSB()            const { return parts.autoSWSB; }
    constexpr bool isSerialized()          const { return parts.fusionCtrl; }
    constexpr bool isEOT()                 const { return parts.eot; }
    constexpr SWSBInfo getSWSB()           const { return {SWSBItem(parts.swsb0), SWSBItem(parts.swsb1)}; }
    constexpr uint64_t getAll()            const { return all; }

    constexpr14 void setExecSize(int execSize_)              { parts.log2ExecSize = utils::log2(execSize_); }
    constexpr14 void setPredCtrl(PredCtrl predCtrl_)         { parts.predCtrl = static_cast<unsigned>(predCtrl_); }
    constexpr14 void setPredInv(bool predInv_)               { parts.predInv = predInv_; }
    constexpr14 void setCMod(const ConditionModifier &cmod_) { parts.cmod = static_cast<unsigned>(cmod_); }
    constexpr14 void setBranchCtrl(bool branchCtrl)          { parts.accWrCtrl = branchCtrl; }
    constexpr14 void setFlagReg(FlagRegister &flag)          { parts.flagRegNum1 = flag.getBase() >> 1; parts.flagRegNum = flag.getBase() & 1; parts.flagSubRegNum = flag.getOffset(); }
#if XE4
    constexpr14 void setCFlag(FlagRegister &flag)            { parts.cflag = flag.getBase(); }
#endif
    constexpr14 void setWrEn(bool maskCtrl_)                 { parts.maskCtrl = maskCtrl_; }
    constexpr14 void setAutoSWSB(bool autoSWSB_)             { parts.autoSWSB = autoSWSB_; }
    constexpr14 void setSWSB(SWSBInfo swsb)                  { parts.swsb0 = swsb[0].all; parts.swsb1 = swsb[1].all; }

    constexpr InstructionModifier() : all(0) {}

    // Hardcoded shift counts are a workaround for MSVC v140 bug.
    constexpr14 /* implicit */ InstructionModifier(FlagRegister flag) : InstructionModifier() {
        *this |= flag;
    }
    constexpr /* implicit */ InstructionModifier(PredCtrl predCtrl_)
        : all{static_cast<uint64_t>(predCtrl_) << 16} {}

    constexpr /* implicit */ InstructionModifier(ThreadCtrl threadCtrl_)
        : all{static_cast<uint64_t>(threadCtrl_) << 14} {}

    constexpr /* implicit */ InstructionModifier(ConditionModifier cmod_)
        : all{static_cast<uint64_t>(cmod_) << 24} {}

#if XE4
    constexpr /* implicit */ InstructionModifier(RoundingOverride rmo_)
        : all{static_cast<uint64_t>(rmo_) << 36} {}
#endif

    constexpr14 /* implicit */ InstructionModifier(int execSize_) : InstructionModifier() {
        setExecSize(execSize_);
#if XE4
        parts.esizeSet = true;
#endif
    }
    constexpr14 /* implicit */ InstructionModifier(SWSBInfo swsb) : InstructionModifier() {
        parts.swsb0 = swsb[0].all;
        parts.swsb1 = swsb[1].all;
    }
    constexpr14 /* implicit */ InstructionModifier(SWSBItem swsb) : InstructionModifier() {
        parts.swsb0 = swsb.all;
    }
    constexpr14 /* implicit */ InstructionModifier(SBID sb) : InstructionModifier(SWSB(sb)) {}

protected:
    constexpr InstructionModifier(bool accessMode_, bool noDDClr_, bool noDDChk_, unsigned chanOff_, bool accWrCtrl_,
                                  bool debugCtrl_, bool saturate_, bool maskCtrl_, bool exBSO_, bool autoSWSB_, bool fusionCtrl_, bool eot_)
        : all{(uint64_t(accessMode_) << 8) | (uint64_t(noDDClr_) << 9) | (uint64_t(noDDChk_) << 10) | (uint64_t(chanOff_ >> 2) << 11)
            | (uint64_t(accWrCtrl_) << 28) | (uint64_t(debugCtrl_) << 30) | (uint64_t(saturate_) << 31) | (uint64_t(maskCtrl_) << 34)
            | (uint64_t(exBSO_) << 35) | (uint64_t(autoSWSB_) << 45) | (uint64_t(fusionCtrl_) << 46) | (uint64_t(eot_) << 47)} {}

public:
    static constexpr InstructionModifier createAccessMode(int accessMode_) {
        return InstructionModifier(accessMode_, false, false, 0, false, false, false, false, false, false, false, false);
    }
    static constexpr InstructionModifier createNoDDClr() {
        return InstructionModifier(false, true, false, 0, false, false, false, false, false, false, false, false);
    }
    static constexpr InstructionModifier createNoDDChk() {
        return InstructionModifier(false, false, true, 0, false, false, false, false, false, false, false, false);
    }
    static constexpr InstructionModifier createChanOff(int offset) {
        return InstructionModifier(false, false, false, offset, false, false, false, false, false, false, false, false);
    }
    static constexpr InstructionModifier createAccWrCtrl() {
        return InstructionModifier(false, false, false, 0, true, false, false, false, false, false, false, false);
    }
#if XE3P
    static constexpr InstructionModifier createFwd() {
        return createAccWrCtrl();
    }
#endif
    static constexpr InstructionModifier createDebugCtrl() {
        return InstructionModifier(false, false, false, 0, false, true, false, false, false, false, false, false);
    }
    static constexpr InstructionModifier createSaturate() {
        return InstructionModifier(false, false, false, 0, false, false, true, false, false, false, false, false);
    }
    static constexpr InstructionModifier createMaskCtrl(bool maskCtrl_) {
        return InstructionModifier(false, false, false, 0, false, false, false, maskCtrl_, false, false, false, false);
    }
    static constexpr InstructionModifier createExBSO() {
        return InstructionModifier(false, false, false, 0, false, false, false, false, true, false, false, false);
    }
    static constexpr InstructionModifier createAutoSWSB() {
        return InstructionModifier(false, false, false, 0, false, false, false, false, false, true, false, false);
    }
    static constexpr InstructionModifier createSerialized() {
        return InstructionModifier(false, false, false, 0, false, false, false, false, false, false, true, false);
    }
    static constexpr InstructionModifier createEOT() {
        return InstructionModifier(false, false, false, 0, false, false, false, false, false, false, false, true);
    }

    friend constexpr14 InstructionModifier operator|(const InstructionModifier &mod1, const InstructionModifier &mod2);
    friend constexpr14 InstructionModifier operator|(const InstructionModifier &mod1, const FlagRegister &mod2);
    friend constexpr14 InstructionModifier operator|(const InstructionModifier &mod1, const PredCtrl &mod2);
    friend constexpr14 InstructionModifier operator|(const InstructionModifier &mod1, const SWSBItem &mod2);
    friend constexpr14 InstructionModifier operator|(const InstructionModifier &mod1, const SBID &mod2);

    friend constexpr14 InstructionModifier operator^(const InstructionModifier &mod1, const InstructionModifier &mod2);

    constexpr14 InstructionModifier operator~() {
        InstructionModifier mod = *this;
        mod.parts.predInv = ~mod.parts.predInv;
        return mod;
    }

    template <typename T>
    constexpr14 InstructionModifier &operator|=(const T &mod) {
        *this = *this | mod;
        return *this;
    }

    constexpr14 InstructionModifier &operator^=(const InstructionModifier &mod) {
        *this = *this ^ mod;
        return *this;
    }
};

inline constexpr14 InstructionModifier operator|(const InstructionModifier &mod1, const InstructionModifier &mod2)
{
    return InstructionModifier(mod1.all | mod2.all);
}


inline constexpr14 InstructionModifier operator|(const InstructionModifier &mod1, const FlagRegister &flag)
{
    InstructionModifier mod = mod1;

    if (mod.parts.predCtrl == static_cast<int>(PredCtrl::None)) {
        mod.parts.flagRegNum1 = flag.getBase() >> 1;
        mod.parts.flagRegNum = flag.getBase() & 1;
        mod.parts.flagSubRegNum = flag.getOffset();
    }

    if (mod.getCMod() == ConditionModifier::none) {
        mod.parts.predInv = flag.getNeg();
        mod.parts.predCtrl = static_cast<int>(PredCtrl::Normal);
    }
#if XE4
    else
        mod.parts.cflag = flag.getBase();
#endif

    return mod;
}

inline constexpr14 InstructionModifier operator|(const InstructionModifier &mod1, const PredCtrl &mod2)
{
    InstructionModifier mod = mod1;
    mod.parts.predCtrl = static_cast<int>(mod2);
    return mod;
}

inline constexpr14 InstructionModifier operator|(const InstructionModifier &mod1, const SWSBItem &mod2)
{
    InstructionModifier mod = mod1;
    if (mod.parts.swsb0 == 0)
        mod.parts.swsb0 = mod2.all;
    else
        mod.parts.swsb1 = mod2.all;
    return mod;
}

inline constexpr14 InstructionModifier operator|(const InstructionModifier &mod1, const SBID &mod2)
{
    return mod1 | SWSBItem(mod2);
}

inline constexpr14 InstructionModifier operator^(const InstructionModifier &mod1, const InstructionModifier &mod2)
{
    return InstructionModifier(mod1.all ^ mod2.all);
}

template <typename T> constexpr InstructionModifier SWSB(SWSBItem item, int dist) {
    return InstructionModifier(item) | SWSB<T>(dist);
}

class Immediate {
protected:
    uint64_t payload;
    DataType type;
    bool hiddenType = false;

    Immediate(uint64_t payload_, DataType type_) : payload(payload_), type(type_) {}

    template <typename T> typename std::enable_if<sizeof(T) == 2>::type setPayload(T imm) {
        uint32_t ximm = utils::bitcast<T, uint16_t>(imm);
        payload = ximm | (ximm << 16);
    }
    template <typename T> typename std::enable_if<sizeof(T) == 4>::type setPayload(T imm) {
        payload = utils::bitcast<T, uint32_t>(imm);
    }
    template <typename T> typename std::enable_if<sizeof(T) == 8>::type setPayload(T imm) {
        payload = utils::bitcast<T, uint64_t>(imm);
    }

    template <typename T> void set(T imm) {
        setPayload<T>(imm);
        type = getDataType<T>();
    }

    template <typename T> void shrinkSigned(T imm) {
        if (imm == T(int16_t(imm)))       set(int16_t(imm));
        else if (imm == T(uint16_t(imm))) set(uint16_t(imm));
        else if (imm == T(int32_t(imm)))  set(int32_t(imm));
        else if (imm == T(uint32_t(imm))) set(uint32_t(imm));
        else                              set(imm);
    }

    template <typename T> void shrinkUnsigned(T imm) {
        if (imm == T(uint16_t(imm)))      set(uint16_t(imm));
        else if (imm == T(uint32_t(imm))) set(uint32_t(imm));
        else                              set(imm);
    }

public:
    Immediate() : payload(0), type(DataType::invalid) {}

#ifdef NGEN_ASM
    static const bool emptyOp = false;
#endif

    constexpr14 DataType getType()           const { return type; }
    explicit constexpr14 operator uint64_t() const { return payload; }
    constexpr14 int getMods()                const { return 0; }
    constexpr14 bool isARF()                 const { return false; }

    Immediate &setType(DataType type_)             { type = type_; return *this; }

    Immediate(uint16_t imm) { set(imm); }
    Immediate(int16_t  imm) { set(imm); }
    Immediate(uint32_t imm) { shrinkUnsigned(imm); }
    Immediate(int32_t  imm) { shrinkSigned(imm); }
    Immediate(uint64_t imm) { shrinkUnsigned(imm); }
    Immediate(int64_t  imm) { shrinkSigned(imm); }

    Immediate(float    imm) { set(imm); }
    Immediate(double   imm) { set(imm); }
#ifdef NGEN_HALF_TYPE
    Immediate(half     imm) { set(imm); }
#endif
#ifdef NGEN_BFLOAT16_TYPE
    Immediate(bfloat16 imm) { set(imm); }
#endif

    Immediate hideType() const {
        Immediate result = *this;
        result.hiddenType = true;
        return result;
    }

    static inline Immediate uw(uint16_t imm) { return Immediate(imm); }
    static inline Immediate  w(int16_t  imm) { return Immediate(imm); }
    static inline Immediate ud(uint32_t imm) { Immediate i; i.set(imm); return i; }
    static inline Immediate  d(int32_t  imm) { Immediate i; i.set(imm); return i; }
    static inline Immediate uq(uint64_t imm) { Immediate i; i.set(imm); return i; }
    static inline Immediate  q(int64_t  imm) { Immediate i; i.set(imm); return i; }
    static inline Immediate  f(float    imm) { return Immediate(imm); }
    static inline Immediate df(double   imm) { return Immediate(imm); }

    static inline Immediate hf(uint16_t f) {
        uint32_t fimm = f;
        fimm |= (fimm << 16);
        return Immediate(fimm, DataType::hf);
    }

    static inline Immediate bf(uint16_t f) {
        uint32_t fimm = f;
        fimm |= (fimm << 16);
        return Immediate(fimm, DataType::bf);
    }

protected:
    static inline uint32_t toUV(int8_t i) {
#ifdef NGEN_SAFE
        if (i & 0xF0) throw invalid_immediate_exception();
#endif
        return i;
    }

public:
    static inline Immediate uv(uint32_t i) {
        return Immediate(i, DataType::uv);
    }

    static inline Immediate uv(uint8_t i0, uint8_t i1, uint8_t i2, uint8_t i3, uint8_t i4, uint8_t i5, uint8_t i6, uint8_t i7) {
        uint32_t payload = (toUV(i0) << 0)
                         | (toUV(i1) << 4)
                         | (toUV(i2) << 8)
                         | (toUV(i3) << 12)
                         | (toUV(i4) << 16)
                         | (toUV(i5) << 20)
                         | (toUV(i6) << 24)
                         | (toUV(i7) << 28);
        return uv(payload);
    }

protected:
    static inline uint32_t toV(int8_t i) {
#ifdef NGEN_SAFE
        if (i < -8 || i > 7) throw invalid_immediate_exception();
#endif
        return (i & 0x7) | ((i >> 4) & 0x8);
    }

public:
    static inline Immediate v(uint32_t i) {
        return Immediate(i, DataType::v);
    }

    static inline Immediate v(int8_t i0, int8_t i1, int8_t i2, int8_t i3, int8_t i4, int8_t i5, int8_t i6, int8_t i7) {
        uint32_t payload = (toV(i0) << 0)
                         | (toV(i1) << 4)
                         | (toV(i2) << 8)
                         | (toV(i3) << 12)
                         | (toV(i4) << 16)
                         | (toV(i5) << 20)
                         | (toV(i6) << 24)
                         | (toV(i7) << 28);
        return v(payload);
    }

    static inline uint32_t toVF(float f) {
        uint32_t fi = utils::bitcast<float, uint32_t>(f);
        int exp = (fi >> 23) & 0xFF;
        int new_exp = exp - 127 + 3;

        if (f == 0.) new_exp = 0;

#ifdef NGEN_SAFE
        if ((new_exp & ~7) || (fi & 0x0007FFFF))
            throw invalid_immediate_exception();
#endif

        return ((fi >> 24) & 0x80)
             | ((new_exp & 0x7) << 4)
             | ((fi >> 19) & 0xF);
    }

    static inline Immediate vf(float f0, float f1, float f2, float f3) {
        uint32_t payload = (toVF(f0) << 0)
                         | (toVF(f1) << 8)
                         | (toVF(f2) << 16)
                         | (toVF(f3) << 24);

        return Immediate(payload, DataType::vf);
    }

    static Immediate zero(DataType dt) {
        return Immediate(0, dt);
    }

    void fixup(HW hw, int execSize, int execWidth, DataType defaultType, int srcN, int arity) const {
#ifdef NGEN_SAFE
        if (getBytes(type) > (16 >> arity))
            throw invalid_immediate_exception();
#endif
    }

    constexpr RegFile getRegFile() const { return RegFileIMM; }

    constexpr14 bool isScalar() const {
        switch (type) {
            case DataType::uv:
            case DataType::v:
            case DataType::vf:
                return false;
            default:
                return true;
        }
    }

    Immediate forceInt32() const {
        auto result = *this;
        if (result.type == DataType::uw)
            result.set(uint32_t(uint16_t(payload)));
        else if (result.type == DataType::w)
            result.set(int32_t(int16_t(payload)));
        return result;
    }

    Immediate cast(DataType newType) const {
        auto clone = *this;
        if (newType == type)
            return clone;

        auto isQ = [](DataType dt) { return (dt == DataType::uq) || (dt == DataType::q); };
        if (isQ(type) && isQ(newType)) {
            clone.type = newType;
            return clone;
        }

        double val = 0.;
        switch (type) {
            case DataType::uw: val = uint16_t(payload); break;
            case DataType::w:  val =  int16_t(payload); break;
            case DataType::ud: val = uint32_t(payload); break;
            case DataType::d:  val =  int32_t(payload); break;
            case DataType::uq: val = uint64_t(payload); break;
            case DataType::q:  val =  int64_t(payload); break;
            case DataType::f:  val = utils::bitcast<uint32_t,float>(uint32_t(payload)); break;
            case DataType::df: val = utils::bitcast<uint64_t,double>(payload); break;
#ifdef NGEN_HALF_TYPE
            case DataType::hf: val = float(half(utils::bitcast<uint16_t,half>(uint16_t(payload)))); break;
#endif
#ifdef NGEN_BFLOAT16_TYPE
            case DataType::bf: val = float(bfloat16(utils::bitcast<uint16_t,bfloat16>(uint16_t(payload)))); break;
#endif
            default:
#ifdef NGEN_SAFE
                throw invalid_type_exception();
#endif
                break;
        }

        switch (newType) {
            case DataType::uw: return Immediate::uw(uint16_t(val));
            case DataType::w:  return Immediate::w(int16_t(val));
            case DataType::ud: return Immediate::ud(uint32_t(val));
            case DataType::d:  return Immediate::d(int32_t(val));
            case DataType::uq: return Immediate::uq(uint64_t(val));
            case DataType::q:  return Immediate::q(int64_t(val));
            case DataType::f:  return Immediate::f(float(val));
            case DataType::df: return Immediate::df(val);
#ifdef NGEN_HALF_TYPE
            case DataType::hf: return Immediate::hf(utils::bitcast<half,uint16_t>(half(val)));
#endif
#ifdef NGEN_BFLOAT16_TYPE
            case DataType::bf: return Immediate::bf(utils::bitcast<bfloat16,uint16_t>(bfloat16(val)));
#endif
            default:
#ifdef NGEN_SAFE
                throw invalid_type_exception();
#endif
                break;
        }

        return clone;
    }

#ifdef NGEN_ASM
    inline void outputText(std::ostream &str, PrintDetail detail, LabelManager &man) const;
#endif
};

// Compute ctrl field for bfn instruction.
// e.g. ctrl = getBFNCtrl([](uint8_t a, uint8_t b, uint8_t c) { return (a & b) | (c & ~b); });
template <typename F>
inline uint8_t getBFNCtrl(F func) { return func(0xAA, 0xCC, 0xF0); }

enum class BarrierType : uint8_t {
    ProducerConsumer = 0,
    Producer = 1,
    Consumer = 2,
};

/********************************************************************/
/* HDC sends                                                        */
/********************************************************************/
union MessageDescriptor {
    uint32_t all;
    struct {
        unsigned funcCtrl : 19;     /* SF-dependent */
        unsigned header : 1;        /* is a header present? */
        unsigned responseLen : 5;   /* # of GRFs returned: valid range 0-16 */
        unsigned messageLen : 4;    /* # of GRFs sent in src0: valid range 1-15 */
        unsigned : 3;
    } parts;
    struct {
        unsigned index : 8;
        unsigned rest : 24;
    } bti;
    struct {
        unsigned index : 8;
        unsigned elements : 3;
        unsigned subtype : 2;
        unsigned subtype2 : 1;
        unsigned messageType : 5;
        unsigned header : 1;
        unsigned responseLen : 5;
        unsigned messageLen : 4;
        unsigned : 3;
    } block;
    struct {
        unsigned index : 8;
        unsigned simd16 : 1;
        unsigned legacySIMD : 1;
        unsigned elements : 2;
        unsigned : 1;
        unsigned : 1;
        unsigned messageType : 5;
        unsigned header : 1;
        unsigned responseLen : 5;
        unsigned messageLen : 4;
        unsigned : 3;
    } scattered;
    struct {
        unsigned index : 8;
        unsigned subtype : 2;
        unsigned elements : 2;
        unsigned simd16 : 1;
        unsigned : 1;
        unsigned messageType : 5;
        unsigned header : 1;
        unsigned responseLen : 5;
        unsigned messageLen : 4;
        unsigned : 3;
    } a64_scattered;
    struct {
        unsigned index : 8;
        unsigned atomicOp : 4;
        unsigned simd8 : 1;         // or data width.
        unsigned returnData : 1;
        unsigned messageType : 5;
        unsigned header : 1;
        unsigned responseLen : 5;
        unsigned messageLen : 4;
        unsigned : 3;
    } atomic;
    struct {
        unsigned index : 8;
        unsigned cmask : 4;
        unsigned simdMode : 2;
        unsigned messageType : 5;
        unsigned header : 1;
        unsigned responseLen : 5;
        unsigned messageLen : 4;
        unsigned : 3;
    } surface;
    struct {
        unsigned opcode : 6;
        unsigned : 1;
        unsigned addrSize : 2;
        unsigned dataSize : 3;
        unsigned vectSize : 3;
        unsigned transpose : 1;
        unsigned cache : 4;
        unsigned : 9;
        unsigned model : 2;
#if XE3P
        unsigned overfetch : 1;     /* storage location only, not supported in HW */
#else
        unsigned : 1;
#endif
    } standardLSC;
    struct {
        unsigned : 12;
        unsigned cmask : 4;
        unsigned : 16;
    } cmask;
    struct {
        unsigned : 7;
        unsigned vnni : 1;
        unsigned : 24;
    } block2D;

    MessageDescriptor() : all(0) {}
    explicit constexpr MessageDescriptor(uint32_t all_) : all(all_) {}
};

inline constexpr MessageDescriptor operator|(const MessageDescriptor &desc1, const MessageDescriptor &desc2) {
    return MessageDescriptor{desc1.all | desc2.all};
}

union ExtendedMessageDescriptor {
    uint32_t all;
    struct {
        unsigned sfid : 5;
        unsigned eot : 1;
        unsigned extMessageLen : 5;    /* # of GRFs sent in src1: valid range 0-15 (pre-Gen12) */
        unsigned : 1;
        unsigned : 4;                  /* Part of exFuncCtrl for non-immediate sends */
        unsigned exFuncCtrl : 16;
    } parts;
    struct {
        unsigned : 12;
        signed offset : 20;
    } flat;
    struct {
        unsigned : 12;
        signed offset : 12;
        unsigned index : 8;
    } bti;
    struct {
        unsigned : 6;
        unsigned index : 26;
    } surface;
    struct {
        unsigned : 12;
        signed xOffset : 10;
        signed yOffset : 10;
    } block2D;

    ExtendedMessageDescriptor() : all(0) {}
    ExtendedMessageDescriptor& operator=(SharedFunction sfid_) { parts.sfid = static_cast<int>(sfid_); return *this; }
};

#if XE3P
union SendgMessageDescriptor;
#endif

enum class AtomicOp : uint16_t {
    cmpwr_2w = 0x00,
    and_ = 0x1801,
    or_ = 0x1902,
    xor_ = 0x1A03,
    mov = 0x0B04,
    inc = 0x0805,
    dec = 0x0906,
    add = 0x0C07,
    sub = 0x0D08,
    revsub = 0x09,
    imax = 0x0F0A,
    imin = 0x0E0B,
    umax = 0x110C,
    umin = 0x100D,
    cmpwr = 0x120E,
    predec = 0x000F,
    fmax = 0x1611,
    fmin = 0x1512,
    fcmpwr = 0x1713,
    fadd = 0x1314,
    fsub = 0x1415,
    fadd_64b = 0x1316,
    fsub_64b = 0x1417,
    load = 0x0A00,
    store = mov,
    cmpxchg = cmpwr,
    fcmpxchg = fcmpwr,
#if XE3P
    bfadd = 0x21FF,
    bfsub = 0x22FF,
    bfmin = 0x23FF,
    bfmax = 0x24FF,
    bfcmpxchg = 0x25FF,
#endif
};

static inline int operandCount(AtomicOp op) {
    switch (op) {
    case AtomicOp::inc:
    case AtomicOp::dec:
    case AtomicOp::predec:
    case AtomicOp::load:
        return 1;
    case AtomicOp::cmpwr_2w:
    case AtomicOp::cmpwr:
    case AtomicOp::fcmpwr:
#if XE3P
    case AtomicOp::bfcmpxchg:
#endif
        return 3;
    default:
        return 2;
    }
}

static inline constexpr bool isFloatAtomicOp(AtomicOp op) {
    return static_cast<int>(op) & 0x10;
}

// Access types.
enum class Access {Read, Write, AtomicInteger, AtomicFloat};

// Address models.
enum AddressModel : uint8_t {
    ModelInvalid = 0,
    ModelBTS = 1,
    ModelA32 = 2,
    ModelA64 = 4,
    ModelSLM = 8,
    ModelCC = 0x10,
    ModelSC = 0x20,
    ModelScratch = 0x40,
    ModelSS = 0x80,
    ModelBSS = 0x81,
#if XE3P
    ModelA64A32U = 0xA4,
    ModelA64A32S = 0xB4,
#endif
};

class AddressBase {
protected:
    uint32_t index;
    AddressModel model;
    uint8_t pad0[3] = {};

    constexpr AddressBase(uint8_t index_, AddressModel model_) : index(index_), model(model_) {}

    static const uint8_t invalidIndex = 0xF0;

public:
    constexpr AddressBase() : AddressBase(invalidIndex, ModelInvalid) {}

    constexpr uint32_t getIndex()     const { return index; }
    constexpr AddressModel getModel() const { return model; }

    void setIndex(uint8_t newIndex)         { index = newIndex; }

    static constexpr AddressBase createBTS(uint8_t index) {
        return AddressBase(index, ModelBTS);
    }
    static constexpr AddressBase createA32(bool coherent) {
        return AddressBase(coherent ? 0xFF : 0xFD, ModelA32);
    }
    static constexpr AddressBase createA64(bool coherent) {
        return AddressBase(coherent ? 0xFF : 0xFD, ModelA64);
    }
#if XE3P
    static constexpr AddressBase createA64A32U() {
        return AddressBase(0, ModelA64A32U);
    }
    static constexpr AddressBase createA64A32S() {
        return AddressBase(0, ModelA64A32S);
    }
#endif
    static constexpr AddressBase createSLM() {
        return AddressBase(0xFE, ModelSLM);
    }
    static constexpr AddressBase createCC(uint8_t index) {
        return AddressBase(index, ModelCC);
    }
    static constexpr AddressBase createSC(uint8_t index) {
        return AddressBase(index, ModelSC);
    }
    static constexpr AddressBase createSS(uint32_t index) {
        return AddressBase(index, ModelSS);
    }
    static constexpr AddressBase createBSS(uint32_t index) {
        return AddressBase(index, ModelBSS);
    }

    inline constexpr bool isRO() const {
        return (getModel() == ModelSC || getModel() == ModelCC);
    }
    inline constexpr bool isStateless() const {
        return model & (ModelA32 | ModelA64);
    }
    inline constexpr bool isA64() const {
        return model & ModelA64;
    }

    void checkModel(uint8_t allowed) { checkModel(static_cast<AddressModel>(allowed)); }
    void checkModel(AddressModel allowed) {
#ifdef NGEN_SAFE
        if (!(model & allowed))
            throw invalid_model_exception();
#endif
    }
};

class hdc_base {
public:
#if XE3P
    template <Access access> inline void getDescriptor(HW hw, int esize, SharedFunction &sfid, AddressBase base, SendgMessageDescriptor &desc, int &addrLen, int &dataLen, const GRFDisp &addr) const {
#ifdef NGEN_SAFE
        throw unsupported_message();
#endif
    }
#endif
protected:
    void hwCheck(HW hw) const {
#ifdef NGEN_SAFE
        if (hw >= HW::Xe2) throw unsupported_message();
#endif
    }
};

class block_hword : public hdc_base {
protected:
    uint8_t count;

public:
    block_hword(int count_ = 1) : count(count_) {};

    template <Access access> void getDescriptors(HW hw, const InstructionModifier &mod, AddressBase base, MessageDescriptor &desc, ExtendedMessageDescriptor &exdesc, const GRFDisp &addr) const
    {
        hwCheck(hw);

        int dataGRFCount = count;
        if (GRF::bytes(hw) == 64) dataGRFCount = (dataGRFCount + 1) >> 1;

        base.checkModel(ModelA64 | ModelBTS | ModelA32 | ModelSLM);
        desc.all = 0;
        desc.bti.index = base.getIndex();
        desc.block.elements = 1 + utils::log2(count);
        desc.block.header = true;
        desc.block.messageLen = 1;
        desc.block.responseLen = dataGRFCount;

        if (base.getModel() == ModelA64) {
            exdesc = SharedFunction::dc1;
            desc.block.subtype = 0x3;
            desc.block.messageType = (access == Access::Write) ? 0x15 : 0x14;
        } else {
            exdesc = SharedFunction::dc0;
            desc.block.messageType = 0x1;
            desc.block.subtype2 = 1;
        }
    }
};

class block_oword : public hdc_base {
protected:
    uint8_t count;
    uint8_t highHalf;

    constexpr block_oword(uint8_t count_, bool highHalf_) : count(count_), highHalf(highHalf_) {}

public:
    block_oword(int count_ = 1) : count(count_), highHalf(false) {}
    static block_oword high() { return block_oword(1, true); }

    template <Access access> void getDescriptors(HW hw, const InstructionModifier &mod, AddressBase base, MessageDescriptor &desc, ExtendedMessageDescriptor &exdesc, const GRFDisp &addr) const
    {
        hwCheck(hw);

        int dataGRFCount = (GRF::bytes(hw) == 64) ? (count + 3) >> 2 : (count + 1) >> 1;

        base.checkModel(ModelA32 | ModelA64 | ModelBTS | ModelCC | ModelSLM);
        exdesc = (base.getModel() == ModelCC)  ? SharedFunction::dcro :
                 (base.getModel() == ModelA64) ? SharedFunction::dc1  :
                                                 SharedFunction::dc0;

        desc.all = 0;
        desc.bti.index = base.getIndex();
        desc.parts.header = true;
        desc.parts.messageLen = 1;
        desc.parts.responseLen = dataGRFCount;
        desc.block.elements = (count == 1) ? highHalf : (1 + utils::log2(count));

        if (base.getModel() == ModelA64)
            desc.block.messageType = (access == Access::Write) ? 0x15 : 0x14;
        else
            desc.block.messageType = (access == Access::Write) << 3;
    }
};

class aligned_block_oword : public hdc_base {
protected:
    uint8_t count;
    uint8_t highHalf;

    constexpr aligned_block_oword(uint8_t count_, bool highHalf_) : count(count_), highHalf(highHalf_) {}

public:
    aligned_block_oword(int count_ = 1) : count(count_), highHalf(false) {}
    static aligned_block_oword high() { return aligned_block_oword(1, true); }

    template <Access access> void getDescriptors(HW hw, const InstructionModifier &mod, AddressBase base, MessageDescriptor &desc, ExtendedMessageDescriptor &exdesc, const GRFDisp &addr) const
    {
        hwCheck(hw);

        int dataGRFCount = (GRF::bytes(hw) == 64) ? (count + 3) >> 2 : (count + 1) >> 1;

        base.checkModel(ModelA32 | ModelA64 | ModelBTS | ModelCC | ModelSLM | ModelSC);
        exdesc = (base.getModel() == ModelCC || base.getModel() == ModelSC) ? SharedFunction::dcro :
                                              (base.getModel() == ModelA64) ? SharedFunction::dc1 :
                                                                              SharedFunction::dc0;

        desc.all = 0;
        desc.bti.index = base.getIndex();
        desc.parts.header = true;
        desc.parts.messageLen = 1;
        desc.parts.responseLen = dataGRFCount;
        desc.block.elements = (count == 1) ? highHalf : (1 + utils::log2(count));

        if (base.getModel() == ModelA64) {
            desc.block.messageType = (access == Access::Write) ? 0x15 : 0x14;
            desc.block.subtype = 1;
        } else if (base.getModel() == ModelSC)
            desc.block.messageType = 4;
        else
            desc.block.messageType = ((access == Access::Write) << 3) + 1;
    }
};

class scattered_byte : public hdc_base {
protected:
    uint8_t count;

public:
    scattered_byte(int count_ = 1) : count(count_) {}

    template <Access access> void getDescriptors(HW hw, const InstructionModifier &mod, AddressBase base, MessageDescriptor &desc, ExtendedMessageDescriptor &exdesc, const GRFDisp &addr) const
    {
        hwCheck(hw);

        bool a64 = (base.getModel() == ModelA64);
        int simd16 = mod.getExecSize() >> 4;
        int dataGRFCount = 1 + simd16;
        int addrGRFCount = dataGRFCount << int(a64);
        if (GRF::bytes(hw) == 64) {
            dataGRFCount = 1;
            addrGRFCount = 1 << int(a64);
            simd16 = 1;
        }

        base.checkModel(ModelA32 | ModelA64 | ModelBTS | ModelSLM);
        desc.all = 0;
        desc.bti.index = base.getIndex();
        desc.parts.header = false;
        desc.parts.messageLen = addrGRFCount;
        desc.parts.responseLen = dataGRFCount;

        if (a64) {
            exdesc = SharedFunction::dc1;
            desc.a64_scattered.elements = utils::log2(count);
            desc.a64_scattered.simd16 = simd16;
            desc.a64_scattered.subtype = 0;
        } else {
            exdesc = SharedFunction::dc0;
            desc.scattered.elements = utils::log2(count);
            desc.scattered.simd16 = simd16;
        }

        if (access == Access::Write)
            desc.scattered.messageType = a64 ? 0x1A : 0xC;
        else
            desc.scattered.messageType = a64 ? 0x10 : 0x4;
    }
};

class scattered_atomic : public hdc_base {
public:
    void applyAtomicOp(AtomicOp op, const RegData &dst, MessageDescriptor &desc) const
    {
#if XE3P
#ifdef NGEN_SAFE
        if ((static_cast<int>(op) & 0xFF) == 0xFF)
            throw unsupported_message();
#endif
#endif
        desc.atomic.returnData = !dst.isNull();
        desc.atomic.atomicOp = static_cast<int>(op) & 0xF;
    }
#if XE3P
    inline void applyAtomicOp(AtomicOp op, SendgMessageDescriptor &desc) const {}
#endif
};

class scattered_word : public scattered_atomic {
public:
    template <Access access> void getDescriptors(HW hw, const InstructionModifier &mod, AddressBase base, MessageDescriptor &desc, ExtendedMessageDescriptor &exdesc, const GRFDisp &addr) const
    {
        hwCheck(hw);

        bool a64 = (base.getModel() == ModelA64);
        int simd16 = mod.getExecSize() >> 4;
        int addrGRFCount = (1 + simd16) << int(a64);
        int dataGRFCount = 1 + simd16;
        if (GRF::bytes(hw) == 64) {
            addrGRFCount = 1 << int(a64);
            dataGRFCount = 1;
            simd16 = 1;
        }

#ifdef NGEN_SAFE
        if (!(access == Access::AtomicInteger || access == Access::AtomicFloat))
            throw invalid_load_store_exception();
#endif
        base.checkModel(ModelA32 | ModelA64 | ModelBTS | ModelSLM);
        exdesc = SharedFunction::dc1;
        desc.all = 0;
        desc.bti.index = base.getIndex();
        desc.parts.header = false;
        desc.parts.messageLen = addrGRFCount;
        desc.parts.responseLen = dataGRFCount;

        if (access == Access::AtomicFloat)
            desc.atomic.messageType = a64 ? 0x1E : 0x1C;
        else
            desc.atomic.messageType = a64 ? 0x13 : 0x03;

        desc.atomic.simd8 = a64 ? 0 : !simd16;
    }
};

class scattered_dword : public scattered_atomic {
protected:
    uint8_t count;

public:
    scattered_dword(int count_ = 1) : count(count_) {}

    template <Access access> void getDescriptors(HW hw, const InstructionModifier &mod, AddressBase base, MessageDescriptor &desc, ExtendedMessageDescriptor &exdesc, const GRFDisp &addr) const
    {
        hwCheck(hw);

        bool a64 = (base.getModel() == ModelA64);
        int simd16 = mod.getExecSize() >> 4;
        int addrGRFCount = (1 + simd16) << int(a64);
        int dataGRFCount = count * (1 + simd16);
        if (GRF::bytes(hw) == 64) {
            addrGRFCount = 1 << int(a64);
            dataGRFCount = count;
            simd16 = 1;
        }

        desc.all = 0;
        desc.bti.index = base.getIndex();
        desc.parts.header = false;
        desc.parts.messageLen = addrGRFCount;
        desc.parts.responseLen = dataGRFCount;

        if (access == Access::AtomicInteger || access == Access::AtomicFloat) {
            base.checkModel(ModelA32 | ModelA64 | ModelBTS | ModelSLM);
            exdesc = SharedFunction::dc1;
            if (access == Access::AtomicFloat)
                desc.atomic.messageType = a64 ? 0x1D : 0x1B;
            else
                desc.atomic.messageType = a64 ? 0x12 : 0x02;
            desc.atomic.simd8 = a64 ? 0 : !simd16;
        } else if (a64) {
            exdesc = SharedFunction::dc1;
            desc.a64_scattered.elements = utils::log2(count);
            desc.a64_scattered.simd16 = simd16;
            desc.a64_scattered.subtype = 0x1;
            desc.a64_scattered.messageType = (access == Access::Write) ? 0x1A : 0x10;
        } else {
            base.checkModel(ModelA32 | ModelBTS | ModelCC);
            exdesc = (base.getModel() == ModelCC) ? SharedFunction::dcro : SharedFunction::dc0;
            desc.scattered.elements = utils::log2(count);
            desc.scattered.legacySIMD = 1;
            desc.scattered.simd16 = simd16;
            desc.scattered.messageType = (access == Access::Write) ? 0xB : 0x3;
        }
    }
};

class scattered_qword : public scattered_atomic {
protected:
    uint8_t count;

public:
    scattered_qword(int count_ = 1) : count(count_) {}

    template <Access access> void getDescriptors(HW hw, const InstructionModifier &mod, AddressBase base, MessageDescriptor &desc, ExtendedMessageDescriptor &exdesc, const GRFDisp &addr) const
    {
        hwCheck(hw);

        bool a64 = (base.getModel() == ModelA64);
        int simd16 = mod.getExecSize() >> 4;
        int addrGRFCount = (1 + simd16) << int(a64);
        int dataGRFCount = count * 2 * (1 + simd16);
        if (GRF::bytes(hw) == 64) {
            addrGRFCount = 1 << int(a64);
            dataGRFCount = count * 2;
            simd16 = 1;
        }

        base.checkModel(ModelA32 | ModelA64 | ModelBTS | ModelSLM);
        desc.all = 0;
        desc.bti.index = base.getIndex();
        desc.parts.header = false;
        desc.parts.messageLen = addrGRFCount;
        desc.parts.responseLen = dataGRFCount;

        if (access == Access::AtomicInteger || access == Access::AtomicFloat) {
            // Note: atomics have same encoding as scattered dword. The atomic operation type
            //   determines the length. The one exception is A64 atomic float.
            exdesc = SharedFunction::dc1;
            if (access == Access::AtomicFloat) {
                desc.atomic.messageType = a64 ? 0x1D : 0x1B;
                desc.atomic.simd8 = a64 ? 0 : !simd16;
            } else {
                desc.atomic.messageType = a64 ? 0x12 : 0x02;
                desc.atomic.simd8 = a64 ? 1 : !simd16;
            }
        } else if (a64) {
            exdesc = SharedFunction::dc1;
            desc.a64_scattered.elements = utils::log2(count);
            desc.a64_scattered.simd16 = simd16;
            desc.a64_scattered.subtype = 0x2;
            desc.a64_scattered.messageType = (access == Access::Write) ? 0x1A : 0x10;
        } else {
            exdesc = SharedFunction::dc0;
            desc.scattered.elements = utils::log2(count);
            desc.scattered.legacySIMD = 1;
            desc.scattered.simd16 = simd16;
            desc.scattered.messageType = (access == Access::Write) ? 0xD : 0x5;
        }
    }
};

class surface_dword : public hdc_base {
protected:
    ChannelMask cmask;
    bool structured;

public:
    surface_dword(ChannelMask cmask_ = ChannelMask::r, bool structured_ = false) : cmask(cmask_), structured(structured_) {}

    template <Access access> void getDescriptors(HW hw, const InstructionModifier &mod, AddressBase base, MessageDescriptor &desc, ExtendedMessageDescriptor &exdesc, const GRFDisp &addr) const
    {
        hwCheck(hw);

        int simd16 = mod.getExecSize() >> 4;
        if (GRF::bytes(hw) == 64) simd16 = 1;
        int nChannels = utils::popcnt(0xF ^ static_cast<int8_t>(cmask));
        bool isA64 = base.getModel() == ModelA64;
        int addrGRFCount = (1 + simd16) << int(isA64) << int(structured);
        int dataGRFCount = nChannels * (1 + simd16);
        if (GRF::bytes(hw) == 64) {
            addrGRFCount = (addrGRFCount + 1) >> 1;
            dataGRFCount = (dataGRFCount + 1) >> 1;
        }

        base.checkModel(ModelBTS | ModelA32 | ModelA64 | ModelSLM);

        exdesc = SharedFunction::dc1;

        desc.all = 0;
        desc.bti.index = base.getIndex();
        desc.parts.header = false;
        desc.parts.messageLen = addrGRFCount;
        desc.parts.responseLen = dataGRFCount;
        desc.surface.messageType = (isA64 << 4) | ((access == Access::Write) << 3) | 0x01;
        desc.surface.cmask = static_cast<int>(cmask);
        desc.surface.simdMode = 2 - simd16;
    }
};

class media_block : public hdc_base {
protected:
    bool vls_override;
    uint8_t vls_offset;
    uint8_t width;
    uint8_t height;

public:
    media_block(int width_, int height_) : vls_override(false), vls_offset(0),
        width(width_), height(height_) {}
    media_block(int width_, int height_, int vls_offset_) : vls_override(true),
        vls_offset(vls_offset_), width(width_), height(height_) {}
    media_block() : media_block(0, 0) {}

    template <Access access> void getDescriptors(HW hw, const InstructionModifier &mod, AddressBase base, MessageDescriptor &desc, ExtendedMessageDescriptor &exdesc, const GRFDisp &addr) const
    {
        hwCheck(hw);

        exdesc = SharedFunction::dc1;
        desc.all = 0;
        desc.bti.index = base.getIndex();
        desc.block.messageType = (base.getModel() == ModelSC) ? 0x05 :
                                    (access == Access::Write) ? 0x0A :
                                                                0x04;
        desc.block.elements = (vls_override << 2) | (vls_offset & 1);
        desc.block.header = true;

        int dataGRFCount = 0;
        if (width > 0) {
            int lg2_rows_per_2grf = std::min<int>(4, 6 - utils::bsr(width));
            dataGRFCount = utils::roundup_pow2((height + (1 << lg2_rows_per_2grf) - 1) >> lg2_rows_per_2grf);
        }

        desc.parts.responseLen = dataGRFCount;
        desc.parts.messageLen = 1;
    }
};

/********************************************************************/
/* New dataport messages.                                           */
/********************************************************************/
enum class LSCOpcode : uint8_t {
    load = 0,
    load_block = 1,
    load_cmask = 2,
    load_2dblock = 3,
    store = 4,
    store_block = 5,
    store_cmask = 6,
    store_2dblock = 7,
    atomic_inc = 8,
    atomic_dec = 9,
    atomic_load = 0xA,
    atomic_store = 0xB,
    atomic_add = 0xC,
    atomic_sub = 0xD,
    atomic_min = 0xE,
    atomic_max = 0xF,
    atomic_umin = 0x10,
    atomic_umax = 0x11,
    atomic_cmpxchg = 0x12,
    atomic_fadd = 0x13,
    atomic_fsub = 0x14,
    atomic_fmin = 0x15,
    atomic_fmax = 0x16,
    atomic_fcmpxchg = 0x17,
    atomic_and = 0x18,
    atomic_or = 0x19,
    atomic_xor = 0x1A,
    load_status = 0x1B,
    store_uncompressed = 0x1C,
    ccs_update = 0x1D,
    rsi = 0x1E,
    fence = 0x1F,
#if XE3P
    atomic_bfadd = 0x21,
    atomic_bfsub = 0x22,
    atomic_bfmin = 0x23,
    atomic_bfmax = 0x24,
    atomic_bfcmpxchg = 0x25,
#endif
};

enum class DataSizeLSC : uint16_t {
    D8 = 0x0100,
    D16 = 0x0201,
    D32 = 0x0402,
    D64 = 0x0803,
    D8U32 = 0x0404,
    D16U32 = 0x0405,
#if XE4
    D4 = 0x0004,    /* async DMA only */
    D6 = 0x0005,    /* async DMA only */
#endif
};

static inline constexpr unsigned getRegisterWidth(DataSizeLSC dsize) {
    return static_cast<uint16_t>(dsize) >> 8;
}

enum class CacheSettingsLSC : uint8_t {
    Default   = 0,
    L1UC_L3UC = 2,
    L1UC_L3C  = 4,    L1UC_L3WB = 4,
    L1C_L3UC  = 6,    L1WT_L3UC = 6,
    L1C_L3C   = 8,    L1WT_L3WB = 8,
    L1S_L3UC  = 10,
    L1S_L3C   = 12,   L1S_L3WB  = 12,
    L1IAR_L3C = 14,   L1WB_L3WB = 14,
    L1UC_L3CC = 5,
    L1C_L3CC  = 9,
#if XE3P
    L1UC_L2UC_L3UC = 2,
    L1UC_L2UC_L3C  = 3,     L1UC_L2UC_L3WB = 3,
    L1UC_L2C_L3UC  = 4,     L1UC_L2WB_L3UC = 4,
    L1UC_L2C_L3C   = 5,
    L1C_L2UC_L3UC  = 6,     L1WT_L2UC_L3UC = 6,
    L1C_L2UC_L3C   = 7,     L1WT_L2UC_L3WB = 7,
    L1C_L2C_L3UC   = 8,     L1WT_L2WB_L3UC = 8,
    L1C_L2C_L3C    = 9,
    L1S_L2UC_L3UC  = 10,
    L1S_L2UC_L3C   = 11,    L1S_L2UC_L3WB  = 11,
    L1S_L2C_L3UC   = 12,    L1S_L2WB_L3UC  = 12,
    L1S_L2C_L3C    = 13,    L1S_L2WB_L3WB  = 13,
    L1IAR_L2IAR_L3IAR = 14, L1WB_L2WB_L3UC = 14,
                            L1WB_L2UC_L3WB = 15,
#endif
#if XE4
    L2C_L3C = L1UC_L2C_L3C,
    L2C_L3UC = L1UC_L2C_L3UC,
    L2UC_L3C = L1UC_L2UC_L3C,
    L2UC_L3UC = L1UC_L2UC_L3UC,
#endif
};

enum FenceScopeLSC : uint8_t {
    ThreadGroup = 0,
    Subslice = 1,
    Tile = 2,
    GPU = 3,
    AllGPUs = 4,
    SystemRelease = 5,
    SystemAcquire = 6
};

enum FlushTypeLSC : uint8_t {
    None = 0,
    Evict = 1,
    Invalidate = 2,
    Discard = 3,
    Clean = 4,
    FlushL3 = 5,
};

struct DataSpecLSC {
    MessageDescriptor desc;
    uint16_t vcount = 0;
    uint8_t dbytes = 0;

    enum { AddrSize16 = 1, AddrSize32 = 2, AddrSize64 = 3 };
    enum { AddrFlat = 0, AddrSS = 1, AddrBSS = 2, AddrBTI = 3 };

    explicit constexpr DataSpecLSC(MessageDescriptor desc_, uint8_t vcount_ = 0, uint8_t dbytes_ = 0) : desc(desc_), vcount(vcount_), dbytes(dbytes_) {}
    /* implicit */ DataSpecLSC(ChannelMask m) {
        desc.standardLSC.opcode = static_cast<uint8_t>(LSCOpcode::load_cmask);
        desc.cmask.cmask = static_cast<uint8_t>(m) ^ 0xF;
        vcount = utils::popcnt(desc.cmask.cmask);
    }
    /* implicit */ DataSpecLSC(CacheSettingsLSC s) {
        desc.standardLSC.cache = static_cast<unsigned>(s);
    }
    /* implicit */ constexpr DataSpecLSC(DataSizeLSC d) : desc((static_cast<uint32_t>(d) & 0x7) << 9), dbytes(getRegisterWidth(d)) {}

    DataSpecLSC operator()(int vcount) const {
        auto vsEncoded = (vcount <= 4) ? (vcount - 1) : (utils::log2(vcount) + 1);
        return *this | createV(vcount, vsEncoded);
    }
    friend inline constexpr DataSpecLSC operator|(const DataSpecLSC &s1, const DataSpecLSC &s2);
    constexpr14 DataSpecLSC &operator|=(const DataSpecLSC &other) {
        *this = *this | other;
        return *this;
    }

    static constexpr DataSpecLSC createV(unsigned vcount, unsigned venc) { return DataSpecLSC{MessageDescriptor(venc << 12), uint8_t(vcount), 0}; }
    static constexpr DataSpecLSC createTranspose()                       { return DataSpecLSC{MessageDescriptor(1 << 15)}; }
    static constexpr DataSpecLSC createVNNI()                            { return DataSpecLSC{MessageDescriptor(1 << 7)}; }
#if XE3P
    static constexpr DataSpecLSC createOverfetch()                       { return DataSpecLSC{MessageDescriptor(1u << 31)}; }
#endif

    template <Access access> void getDescriptors(HW hw, const InstructionModifier &mod, AddressBase base, MessageDescriptor &desc, ExtendedMessageDescriptor &exdesc, const GRFDisp &addr) const
    {
        bool a64 = (base.getModel() == ModelA64);
        desc = this->desc;
        exdesc = (base.getModel() == ModelSLM) ? SharedFunction::slm : SharedFunction::ugm;

        desc.standardLSC.addrSize = a64 ? AddrSize64 : AddrSize32;
#if XE3P
        desc.standardLSC.overfetch = false;
#endif

        if (base.getModel() == ModelA32) base = AddressBase::createBTS(0xFF);

        switch (base.getModel()) {
            case ModelA64:
            case ModelSLM:
                desc.standardLSC.model = AddrFlat;
                exdesc.flat.offset = addr.getDisp();
#ifdef NGEN_SAFE
                if (exdesc.flat.offset != addr.getDisp())
                    throw invalid_address_modifier_exception();
#endif
                break;
            case ModelBTS:
                desc.standardLSC.model = AddrBTI;
                exdesc.bti.index = base.getIndex();
                exdesc.bti.offset = addr.getDisp();
#ifdef NGEN_SAFE
                if (exdesc.bti.offset != addr.getDisp())
                    throw invalid_address_modifier_exception();
#endif
                break;
            case ModelSS:
            case ModelBSS:
                desc.standardLSC.model = (base.getModel() == ModelSS ? AddrSS : AddrBSS);
                exdesc.surface.index = base.getIndex();
                break;
            default:
#ifdef NGEN_SAFE
                throw invalid_model_exception();
#endif
                break;
        }

        auto vc = std::max<unsigned>(vcount, 1);
        if (this->desc.standardLSC.transpose && !desc.standardLSC.opcode) {
            desc.parts.messageLen = 1;
            desc.parts.responseLen = GRF::bytesToGRFs(hw, dbytes * vc);
        } else {
            auto effSIMDGRFs = 1 + ((mod.getExecSize()) >> (GRF::log2Bytes(hw) - 1));
            desc.parts.messageLen = effSIMDGRFs * (a64 ? 2 : 1);
            desc.parts.responseLen = effSIMDGRFs * vc * (1 + (dbytes >> 3));
        }

        if (access == Access::Write)
            desc.standardLSC.opcode |= static_cast<uint8_t>(LSCOpcode::store);
    }

    void applyAtomicOp(AtomicOp op, const RegData &dst, MessageDescriptor &desc) const
    {
        desc.standardLSC.opcode = static_cast<uint16_t>(op) >> 8;
    }

#if XE3P
    template <Access access> inline void getDescriptor(HW hw, int esize, SharedFunction &sfid, AddressBase base, SendgMessageDescriptor &desc, int &addrLen, int &dataLen, const GRFDisp &addr) const;
    inline void applyAtomicOp(AtomicOp op, SendgMessageDescriptor &desc) const;
#endif
};

static inline DataSpecLSC scattered(const DataSpecLSC &dtype, int vsize = 1) { return dtype(vsize); }
static inline DataSpecLSC block(const DataSpecLSC &dtype, int vsize = 1) { return dtype(vsize) | DataSpecLSC::createTranspose(); }

inline constexpr DataSpecLSC operator|(const DataSpecLSC &s1, const DataSpecLSC &s2) {
    return DataSpecLSC{s1.desc | s2.desc, uint8_t(s1.vcount | s2.vcount), uint8_t(s1.dbytes | s2.dbytes)};
}

class block_2d : public DataSpecLSC {
protected:
    uint8_t width, height, count;

public:
    block_2d(const DataSpecLSC &dtype_, int width_, int height_, int count_ = 1) : DataSpecLSC(dtype_), width(width_), height(height_), count(count_) {}

    friend block_2d operator|(block_2d left, const DataSpecLSC &right) {
        left.DataSpecLSC::operator|=(right);
        return left;
    }

    template <Access access> void getDescriptors(HW hw, const InstructionModifier &mod, AddressBase base, MessageDescriptor &desc, ExtendedMessageDescriptor &exdesc, const GRFDisp &addr) const
    {
        base.checkModel(ModelA64);

        desc = this->desc;
#if XE3P
        desc.standardLSC.overfetch = false;
#endif

        desc.standardLSC.opcode = static_cast<uint8_t>((access == Access::Write) ? LSCOpcode::store_2dblock : LSCOpcode::load_2dblock);
        desc.standardLSC.model = AddrFlat;

        auto w = width, h = height;
        if (this->desc.standardLSC.transpose) std::swap(w, h);
        desc.parts.messageLen = 1;
        desc.parts.responseLen = std::min(count * GRF::bytesToGRFs(hw, utils::roundup_pow2(w) * h * this->dbytes), 31);

        exdesc = SharedFunction::ugm;

        exdesc.block2D.xOffset = addr.getDispX();
        exdesc.block2D.yOffset = addr.getDispY();
    }

#if XE3P
    template <Access access> inline void getDescriptor(HW hw, int esize, SharedFunction &sfid, AddressBase base, SendgMessageDescriptor &desc, int &addrLen, int &dataLen, const GRFDisp &addr) const;
#endif
};

// Generate descriptors for a load operation.
template <typename DataSpec, typename Addr>
static inline void encodeLoadDescriptors(HW hw, MessageDescriptor &desc, ExtendedMessageDescriptor &exdesc,
    const InstructionModifier &mod, const RegData &dst, const DataSpec &spec, AddressBase base, const Addr &addr)
{
    spec.template getDescriptors<Access::Read>(hw, mod, base, desc, exdesc, addr);
    if (dst.isNull())
        desc.parts.responseLen = 0;
}

// Generate descriptors for a store operation. Requires split send for pre-Gen12.
template <typename DataSpec, typename Addr>
static inline void encodeStoreDescriptors(HW hw, MessageDescriptor &desc, ExtendedMessageDescriptor &exdesc,
    const InstructionModifier &mod, const DataSpec &spec, AddressBase base, const Addr &addr)
{
#ifdef NGEN_SAFE
    if (base.isRO()) throw read_only_exception();
#endif

    spec.template getDescriptors<Access::Write>(hw, mod, base, desc, exdesc, addr);
    exdesc.parts.extMessageLen = desc.parts.responseLen;
    desc.parts.responseLen = 0;
}

// Generate descriptors for an atomic operation. Requires split send for binary and ternary atomics pre-Gen12.
template <typename DataSpec, typename Addr>
static inline void encodeAtomicDescriptors(HW hw, MessageDescriptor &desc, ExtendedMessageDescriptor &exdesc,
    AtomicOp op, const InstructionModifier &mod, const RegData &dst, const DataSpec &spec, AddressBase base, const Addr &addr)
{
    if (isFloatAtomicOp(op))
        spec.template getDescriptors<Access::AtomicFloat>(hw, mod, base, desc, exdesc, addr);
    else
        spec.template getDescriptors<Access::AtomicInteger>(hw, mod, base, desc, exdesc, addr);

    spec.applyAtomicOp(op, dst, desc);

    exdesc.parts.extMessageLen = desc.parts.responseLen * (operandCount(op) - 1);
    if (dst.isNull())
        desc.parts.responseLen = 0;
}


#if XE3P
/********************************************************************/
/* New send encoding and decoding.                                  */
/********************************************************************/
enum GatewayOpcode {
    eot = 0,
    bar = 4,
    nbar = 5,
    save_bar = 8,
    restore_bar = 9,
    eotr = 10,
    restore_btd_stack = 11,
    sip_bar = 12,
#if XE4
    cbar = 13,
    abar_init = 14,
    abar_expect = 15,
    abar_arrive = 16,
    abar_complete = 17,
    abar_arrive_expect = 18,
    abar_try = 19,
    abar_test_poll = 20,
    abar_inval = 21,
    abar_save = 22,
    abar_restore = 23,
    abar_query = 24,
    async_mtp_fence = 25,
    cbar_remote = 62,
    cbar_wg_eot = 63,
#endif
};

#if XE4
enum ADMAOpcode {
    /* l = SLM, r = remote SLM, g = global memory */
    linear_l2r = 0,
    linear_l2g = 1,
    linear_prefetch = 2,
    linear_g2l = 3,
    linear_reduce_l2r = 4,
    linear_reduce_l2g = 5,
    tensor_l2g = 6,
    tensor_prefetch = 7,
    tensor_g2l = 8,
    tensor_reduce_l2g = 9,
    row_l2g = 10,
    row_prefetch = 11,
    row_g2l = 12,
    row_reduce_l2g = 13,
};

enum class ADMAReduction {
    inc_wrap = 0,
    dec_wrap = 1,
    add = 2,
    min_ = 3,
    max_ = 4,
    and_ = 5,
    or_ = 6,
    xor_ = 7,
};

enum AMMAOpcode {
    dense_mma = 0,
    sparse_mma = 1,
    fp_error_query = 2,
    fp_error_clear = 3,
};
#endif

union SendgMessageDescriptor {
    uint64_t all;
    struct {
        uint64_t opcode : 6;
        uint64_t : 58;
    } common;
    struct {
        uint64_t : 7;
        uint64_t vlen : 3;
        uint64_t transpose : 1;
        uint64_t dataSize : 3;
        uint64_t addrSize : 2;
        uint64_t cacheMode : 4;
        uint64_t : 1;
        uint64_t overfetch : 1;
        uint64_t : 22;
        uint64_t scale : 2;
        uint64_t : 18;
    } mem;
    struct {
        uint64_t : 7;
        uint64_t cmask : 4;
        uint64_t : 53;
    } cmask;
    struct {
        uint64_t : 22;
         int64_t offset : 22;
        uint64_t : 20;
    } flat;
    struct {
        uint64_t : 22;
        uint64_t ssIdx : 5;
         int64_t offset : 17;
        uint64_t : 20;
    } surface;
    struct {
        uint64_t : 9;
        uint64_t vnni : 1;
        uint64_t transpose : 1;
        uint64_t : 11;
         int64_t xOffset : 12;
         int64_t yOffset : 12;
        uint64_t : 18;
    } block2D;
    struct {
        uint64_t : 8;
        uint64_t flushType : 3;
        uint64_t fenceScope : 3;
        uint64_t : 50;
    } fence;
    struct {
        uint64_t : 7;
        uint64_t activeOnly : 1;
        uint64_t legacy : 1;
        uint64_t : 55;
    } barrier;
    struct {
        uint64_t : 7;
        uint64_t replay : 2;
        uint64_t : 55;
    } eot;
#if XE4
    struct {
        uint64_t : 9;
        uint64_t cfn : 1;
        uint64_t : 54;
    } abarInit;
    struct {
        uint64_t : 7;
        uint64_t drop : 1;
        uint64_t lmc : 1;
        uint64_t scope : 1;
        uint64_t : 54;
    } abar;
    struct {
        uint64_t : 8;
        uint64_t nreg : 2;
        uint64_t : 54;
    } abarTest;
    struct {
        uint64_t : 7;
        uint64_t dims : 3;
        uint64_t abar : 1;
        uint64_t multicast : 1;
        uint64_t fillMode : 1;
        uint64_t useCopySize : 1;
        uint64_t memtype : 2;
        uint64_t cache : 4;
        uint64_t reduction : 3;
        uint64_t : 1;
        uint64_t dataSize : 3;
        uint64_t : 1;
        uint64_t dataType : 3;
        uint64_t : 1;
        uint64_t coreLayout : 2;
        uint64_t slmLayout : 1;
        uint64_t : 1;
        uint64_t addrType : 2;
        uint64_t rowSize : 8;
        uint64_t : 18;
    } adma;
    struct {
        uint64_t : 6;
        uint64_t m : 3;
        uint64_t : 2;
        uint64_t n : 4;
        uint64_t : 1;
        uint64_t k : 4;
        uint64_t dtype : 3;
        uint64_t ctype : 3;
        uint64_t atype : 4;
        uint64_t : 1;
        uint64_t btype : 4;
        uint64_t : 1;
        uint64_t ascale : 1;
        uint64_t bscale : 1;
        uint64_t alayout : 1;
        uint64_t blayout : 1;
        uint64_t atm : 1;
        uint64_t btm : 1;
        uint64_t dtm : 1;
        uint64_t areuse : 1;
        uint64_t : 20;
    } amma;
#endif

    constexpr SendgMessageDescriptor() : all(0) {}
    explicit constexpr SendgMessageDescriptor(uint64_t all_) : all(all_) {}

    int vectorLength() const {
        const int vlDecode[8] = {1, 2, 3, 4, 8, 16, 32, 64};
        return vlDecode[mem.vlen];
    }

    int log2ElementBytesMem() const {
        return mem.dataSize & 0x3;
    }

    int elementBytesMem() const {
        return 1 << (mem.dataSize & 0x3);
    }

    int elementBytesReg() const {
        const int dsDecode[8] = {1, 2, 4, 8, 4, 4, 0, 0};
        return dsDecode[mem.dataSize];
    }

    // Return # destination registers if known, and -1 if not.
    inline int dstLen(HW hw, int execSize, SharedFunction sfid) const
    {
        int effSIMDGRFs = 1 + (execSize >> (GRF::log2Bytes(hw) - 1));
#if XE4
        bool srf = (hw >= HW::Xe4) && (execSize == 1);
#endif

        switch (sfid) {
            case SharedFunction::ugm:
            case SharedFunction::tgm:
            case SharedFunction::slm:
            case SharedFunction::urb:
                switch (static_cast<LSCOpcode>(common.opcode)) {
                    case LSCOpcode::load: {
                        int vc = vectorLength();
                        int dbytes = elementBytesReg();
#if XE4
                        if (srf && mem.transpose)
                            return SRF::bytesToSRFs(dbytes * vc);
#endif
                        if (mem.transpose)
                            return GRF::bytesToGRFs(hw, dbytes * vc);
                        else
                            return effSIMDGRFs * vc * (1 + (dbytes >> 3));
                        break;
                    }
                    case LSCOpcode::load_cmask: {
                        int vc = utils::popcnt(cmask.cmask);
                        return effSIMDGRFs * vc;
                        break;
                    }
                    case LSCOpcode::load_2dblock:
                        return -1;      /* cannot determine from descriptor */
                    case LSCOpcode::fence:
                        return 1;
                    case LSCOpcode::atomic_inc:
                    case LSCOpcode::atomic_dec:
                    case LSCOpcode::atomic_load:
                    case LSCOpcode::atomic_add:
                    case LSCOpcode::atomic_sub:
                    case LSCOpcode::atomic_min:
                    case LSCOpcode::atomic_max:
                    case LSCOpcode::atomic_umin:
                    case LSCOpcode::atomic_umax:
                    case LSCOpcode::atomic_cmpxchg:
                    case LSCOpcode::atomic_fadd:
                    case LSCOpcode::atomic_fsub:
                    case LSCOpcode::atomic_fmin:
                    case LSCOpcode::atomic_fmax:
                    case LSCOpcode::atomic_fcmpxchg:
                    case LSCOpcode::atomic_and:
                    case LSCOpcode::atomic_or:
                    case LSCOpcode::atomic_xor:
                        return effSIMDGRFs * (1 + (elementBytesReg() >> 3));
                    default:
                        return 0;
                }
                break;
            case SharedFunction::gtwy:
                switch (static_cast<GatewayOpcode>(common.opcode)) {
                    case GatewayOpcode::sip_bar:
                    case GatewayOpcode::save_bar:
                        return 1;
#if XE4
                    case GatewayOpcode::abar_arrive:
                    case GatewayOpcode::abar_arrive_expect:
                    case GatewayOpcode::abar_test_poll:
                        return 1;
                    case GatewayOpcode::abar_query:
                        return 2;
#endif
                    default:
                        return 0;
                }
                break;
#if XE4
            case SharedFunction::dma: return 0;
            case SharedFunction::mma:
                switch (static_cast<AMMAOpcode>(common.opcode)) {
                    case AMMAOpcode::fp_error_query: return 1;
                    default: return 0;
                }
                break;
#endif
            default: break;
        }

        return -1;
    }

    inline int src0Len(HW hw, int execSize, SharedFunction sfid) const
    {
        switch (sfid) {
            case SharedFunction::slm:
            case SharedFunction::ugm:
            case SharedFunction::tgm:
            case SharedFunction::urb:
                if (static_cast<LSCOpcode>(common.opcode) == LSCOpcode::fence) return 0;
                if (sfid == SharedFunction::slm) return 1;
                if (mem.addrSize == 0b10) return GRF::bytesToGRFs(hw, execSize * 8);
                return 1;
            case SharedFunction::gtwy:
                switch (static_cast<GatewayOpcode>(common.opcode)) {
                    case GatewayOpcode::eot:
                    case GatewayOpcode::eotr:
                    case GatewayOpcode::bar:
                    case GatewayOpcode::sip_bar:
                    case GatewayOpcode::nbar:
                    case GatewayOpcode::restore_bar:
                        return 1;
#if XE4
                    case GatewayOpcode::cbar:
                        return 2;
                    case GatewayOpcode::abar_init:
                    case GatewayOpcode::abar_arrive:
                    case GatewayOpcode::abar_expect:
                    case GatewayOpcode::abar_arrive_expect:
                    case GatewayOpcode::abar_complete:
                    case GatewayOpcode::abar_test_poll:
                    case GatewayOpcode::abar_try:
                    case GatewayOpcode::abar_inval:
                    case GatewayOpcode::abar_query:
                        return 1;
#endif
                    case GatewayOpcode::save_bar:
#if XE4
                        return (hw >= HW::Xe4) ? 0 : 1;
#else
                        return 1;
#endif
                    default: return 0;
                }
                break;
#if XE4
            case SharedFunction::dma:
                switch (static_cast<ADMAOpcode>(common.opcode)) {
                    case ADMAOpcode::linear_l2r:
                        return 5;
                    case ADMAOpcode::linear_g2l:
                    case ADMAOpcode::linear_reduce_l2r:
                        return 4;
                    case ADMAOpcode::linear_l2g:
                    case ADMAOpcode::linear_reduce_l2g:
                        return 3;
                    case ADMAOpcode::linear_prefetch:
                        return 1;
                    case ADMAOpcode::row_g2l:
                    case ADMAOpcode::row_l2g:
                    case ADMAOpcode::row_reduce_l2g:
                    case ADMAOpcode::row_prefetch:
                        return ((adma.addrType == 2) ? 2 : 1) + adma.useCopySize;
                    case ADMAOpcode::tensor_g2l:
                        return 8;
                    case ADMAOpcode::tensor_prefetch:
                        return 5;
                    case ADMAOpcode::tensor_l2g:
                    case ADMAOpcode::tensor_reduce_l2g:
                        return 7;
                    default: break;
                }
                break;
            case SharedFunction::mma:
                switch (static_cast<AMMAOpcode>(common.opcode)) {
                    case AMMAOpcode::dense_mma:
                    case AMMAOpcode::sparse_mma:
                        return 16;
                    default: return 0;
                }
                break;
#endif
            default: break;
        }
        return -1;
    }

    inline int src1Len(HW hw, int execSize, SharedFunction sfid) const
    {
        int effSIMDGRFs = 1 + (execSize >> (GRF::log2Bytes(hw) - 1));
#if XE4
        bool srf = (hw >= HW::Xe4) && (execSize == 1);
#endif

        switch (sfid) {
            case SharedFunction::ugm:
            case SharedFunction::tgm:
            case SharedFunction::slm:
            case SharedFunction::urb:
                switch (static_cast<LSCOpcode>(common.opcode)) {
                    case LSCOpcode::store: {
                        int vc = vectorLength();
                        int dbytes = elementBytesReg();
#if XE4
                        if (srf && mem.transpose)
                            return SRF::bytesToSRFs(dbytes * vc);
#endif
                        if (mem.transpose)
                            return GRF::bytesToGRFs(hw, dbytes * vc);
                        else
                            return effSIMDGRFs * vc * (1 + (dbytes >> 3));
                        break;
                    }
                    case LSCOpcode::store_cmask: {
                        int vc = utils::popcnt(cmask.cmask);
                        return effSIMDGRFs * vc;
                        break;
                    }
                    case LSCOpcode::store_2dblock:
                        return -1;      /* cannot determine from descriptor */
                    case LSCOpcode::atomic_add:
                    case LSCOpcode::atomic_sub:
                    case LSCOpcode::atomic_min:
                    case LSCOpcode::atomic_max:
                    case LSCOpcode::atomic_umin:
                    case LSCOpcode::atomic_umax:
                    case LSCOpcode::atomic_fadd:
                    case LSCOpcode::atomic_fsub:
                    case LSCOpcode::atomic_fmin:
                    case LSCOpcode::atomic_fmax:
                    case LSCOpcode::atomic_and:
                    case LSCOpcode::atomic_or:
                    case LSCOpcode::atomic_xor:
                        return effSIMDGRFs * (1 + (elementBytesReg() >> 3));
                    case LSCOpcode::atomic_cmpxchg:
                    case LSCOpcode::atomic_fcmpxchg:
                        return 2 * effSIMDGRFs * (1 + (elementBytesReg() >> 3));
                    default: return 0;
                }
                break;
            case SharedFunction::gtwy:
#if XE4
                switch (static_cast<GatewayOpcode>(common.opcode)) {
                    case GatewayOpcode::abar_init:
                    case GatewayOpcode::abar_arrive:
                    case GatewayOpcode::abar_arrive_expect:
                    case GatewayOpcode::abar_expect:
                    case GatewayOpcode::abar_complete:
                    case GatewayOpcode::abar_test_poll:
                    case GatewayOpcode::abar_try:
                        return 1;
                    default: break;
                }
#endif
                return 0;
#if XE4
            case SharedFunction::dma:
                switch (static_cast<ADMAOpcode>(common.opcode)) {
                    case ADMAOpcode::linear_g2l:
                    case ADMAOpcode::linear_l2r:
                    case ADMAOpcode::linear_reduce_l2r:
                    case ADMAOpcode::linear_l2g:
                    case ADMAOpcode::linear_reduce_l2g:
                    case ADMAOpcode::linear_prefetch:
                        return 0;
                    case ADMAOpcode::row_g2l:
                    case ADMAOpcode::row_l2g:
                    case ADMAOpcode::row_reduce_l2g:
                    case ADMAOpcode::row_prefetch:
                        return 3;
                    case ADMAOpcode::tensor_g2l:
                    case ADMAOpcode::tensor_prefetch:
                    case ADMAOpcode::tensor_l2g:
                    case ADMAOpcode::tensor_reduce_l2g:
                        return 16;
                    default: break;
                }
                break;
            case SharedFunction::mma:
                switch (static_cast<AMMAOpcode>(common.opcode)) {
                    case AMMAOpcode::dense_mma:
                    case AMMAOpcode::sparse_mma:
                        return 10;
                    default: return 0;
                }
                break;
#endif
            default: break;
        }

        return -1;
    }
};

static_assert(sizeof(SendgMessageDescriptor) == 8, "SendgMessageDescriptor has been padded by compiler");

static inline unsigned encodeScaleLSC(int scale)
{
    if (scale <= 2) return scale;
    if (scale == 4) return 3;
#ifdef NGEN_SAFE
    throw invalid_address_modifier_exception();
#endif
    return 0;
}

template <Access access>
void DataSpecLSC::getDescriptor(HW hw, int execSize, SharedFunction &sfid, AddressBase base, SendgMessageDescriptor &desc, int &addrLen, int &dataLen, const GRFDisp &addr) const
{
    SharedFunction defaultSFID = SharedFunction::ugm;

    desc.common.opcode = this->desc.standardLSC.opcode;
    if (access == Access::Write)
        desc.common.opcode |= static_cast<uint8_t>(LSCOpcode::store);
    desc.cmask.cmask = this->desc.cmask.cmask;      /* or vlen + transpose */
    desc.mem.dataSize = this->desc.standardLSC.dataSize;
    desc.mem.cacheMode = this->desc.standardLSC.cache;
    desc.mem.scale = encodeScaleLSC(addr.getScale() / (std::max<int>(vcount, 1) * desc.elementBytesMem()));
    desc.mem.overfetch = this->desc.standardLSC.overfetch;

    bool flat = true;

    auto model = base.getModel();
    if (model == ModelA64) {
        auto ind0 = addr.getInd0();
        auto base = addr.getBase();
        if (!ind0.isNull() && base.isValid()) switch (base.getType()) {
            case DataType::ud: model = ModelA64A32U; break;
            case DataType::d:  model = ModelA64A32S; break;
            default: break;
        }
    }

    switch (model) {
        case ModelA64:          desc.mem.addrSize = 0b10; break;
        case ModelA64A32U:      desc.mem.addrSize = 0b00; break;
        case ModelA64A32S:      desc.mem.addrSize = 0b01; break;
        case ModelSLM:
            defaultSFID = SharedFunction::slm;
            desc.mem.addrSize = 0b00;
            break;
        case ModelSS:
        case ModelBSS:
            flat = false;
            desc.mem.addrSize = 0b11;
            desc.surface.ssIdx = base.getIndex();
            break;
        default:
#ifdef NGEN_SAFE
            throw invalid_model_exception();
#endif
            break;
    }

    int offsetShift = desc.log2ElementBytesMem();
    int sdisp = addr.getDisp() >> offsetShift;

    if (flat) {
        desc.flat.offset = sdisp;
#ifdef NGEN_SAFE
        if ((desc.flat.offset << offsetShift) != addr.getDisp())
            throw invalid_address_modifier_exception();
#endif
    } else {
        desc.surface.offset = sdisp;
#ifdef NGEN_SAFE
        if ((desc.surface.offset << offsetShift) != addr.getDisp())
            throw invalid_address_modifier_exception();
#endif
    }

    auto vc = std::max<unsigned>(vcount, 1);
    bool block = this->desc.standardLSC.transpose && this->desc.standardLSC.opcode == static_cast<uint8_t>(LSCOpcode::load);
    if (block) {
        addrLen = 1;
        dataLen = GRF::bytesToGRFs(hw, dbytes * vc);
    } else {
        auto effSIMDGRFs = 1 + (execSize >> (GRF::log2Bytes(hw) - 1));
        addrLen = effSIMDGRFs * (base.isA64() ? 2 : 1);
        dataLen = effSIMDGRFs * vc * (1 + (dbytes >> 3));
    }

    if (sfid == SharedFunction::automatic)
        sfid = defaultSFID;
}

void DataSpecLSC::applyAtomicOp(AtomicOp op, SendgMessageDescriptor &desc) const
{
    desc.common.opcode = static_cast<uint16_t>(op) >> 8;
}

template <Access access>
void block_2d::getDescriptor(HW hw, int execSize, SharedFunction &sfid, AddressBase base, SendgMessageDescriptor &desc, int &addrLen, int &dataLen, const GRFDisp &addr) const
{
    auto addrNoDisp = addr;
    addrNoDisp.clearDisp();

    DataSpecLSC::getDescriptor<access>(hw, execSize, sfid, base, desc, addrLen, dataLen, addrNoDisp);
    desc.common.opcode = static_cast<uint8_t>((access == Access::Write) ? LSCOpcode::store_2dblock : LSCOpcode::load_2dblock);
    desc.block2D.vnni = this->desc.block2D.vnni;
    desc.block2D.xOffset = addr.getDispX();
    desc.block2D.yOffset = addr.getDispY();

#ifdef NGEN_SAFE
    if (desc.block2D.xOffset != addr.getDispX() || desc.block2D.yOffset != addr.getDispY())
        throw invalid_address_modifier_exception();
#endif

    auto w = width, h = height;
    if (desc.mem.transpose) std::swap(w, h);

    addrLen = 1;
    dataLen = std::min(count * GRF::bytesToGRFs(hw, utils::roundup_pow2(w) * h * this->dbytes), 31);

    if (sfid == SharedFunction::automatic)
        sfid = SharedFunction::ugm;
}

template <typename DataSpec>
static inline void encodeLoadDescriptor(HW hw, SendgMessageDescriptor &desc, SharedFunction &sfid, int &dstLen, int &src0Len,
    const InstructionModifier &mod, const DataSpec &spec, AddressBase base, const GRFDisp &addr)
{
    spec.template getDescriptor<Access::Read>(hw, mod.getExecSize(), sfid, base, desc, src0Len, dstLen, addr);
}

template <typename DataSpec>
static inline void encodeStoreDescriptor(HW hw, SendgMessageDescriptor &desc, SharedFunction &sfid, int &src0Len, int &src1Len,
    const InstructionModifier &mod, const DataSpec &spec, AddressBase base, const GRFDisp &addr)
{
    spec.template getDescriptor<Access::Write>(hw, mod.getExecSize(), sfid, base, desc, src0Len, src1Len, addr);
}

template <typename DataSpec>
static inline void encodeAtomicDescriptor(HW hw, SendgMessageDescriptor &desc, SharedFunction &sfid, int &src0Len, int &src1Len,
    AtomicOp op, const InstructionModifier &mod, const DataSpec &spec, AddressBase base, const GRFDisp &addr)
{
    spec.template getDescriptor<Access::AtomicInteger>(hw, mod.getExecSize(), sfid, base, desc, src0Len, src1Len, addr);
    spec.applyAtomicOp(op, desc);
}
#endif /* XE3P */

#if XE4
/* Async DMA interface */
struct ADMAOptions
{
    SendgMessageDescriptor desc{};

    ADMAOptions() = default;
    explicit constexpr ADMAOptions(uint64_t raw) : desc(raw) {}

    ADMAOptions(CacheSettingsLSC cs) : ADMAOptions(DataSpecLSC(cs)) {}
    ADMAOptions(DataSpecLSC ds) {
        desc.adma.dataSize = ds.desc.standardLSC.dataSize;
        desc.adma.rowSize = std::max<int>(1, ds.vcount >> 3) - 1;
        if (ds.desc.standardLSC.cache)
            desc.adma.cache = (ds.desc.standardLSC.cache & 0x3) ^ 1;
    }
    ADMAOptions(DataType dt) {
        enum { fp = 0, bf = 1, uint = 2, sint = 3 } type = fp;
        switch (dt) {
            case DataType::e2m1: case DataType::hf8: case DataType::f16:  case DataType::f32: case DataType::f64: type = fp;   break;
            case DataType::e3m0: case DataType::bf8: case DataType::bf16: case DataType::tf32:                    type = bf;   break;
            case DataType::u4:   case DataType::u8:  case DataType::u16:  case DataType::u32: case DataType::u64: type = uint; break;
            case DataType::s4:   case DataType::s8:  case DataType::s16:  case DataType::s32: case DataType::s64: type = sint; break;
            default:
#ifdef NGEN_SAFE
                throw invalid_type_exception();
#endif
                break;
        }
        desc.adma.dataType = type;
        int lg2Bits = getLog2Bits(dt);
        desc.adma.dataSize = (lg2Bits == 2) ? 4 : lg2Bits - 3;
    }

    static constexpr ADMAOptions createABarrier()      { return ADMAOptions{1ull << 10}; }
    static constexpr ADMAOptions createMulticast()     { return ADMAOptions{1ull << 11}; }
    static constexpr ADMAOptions createNaNFill()       { return ADMAOptions{1ull << 12}; }
    static constexpr ADMAOptions createCopySize()      { return ADMAOptions{1ull << 13}; }
    static constexpr ADMAOptions createCoreType(int i) { return ADMAOptions{(uint64_t(i) << 32) | (1ull << 34)}; }
    static constexpr ADMAOptions createTensorDims(int ndims) { return ADMAOptions{uint64_t(ndims - 1) << 7}; }

    friend inline constexpr ADMAOptions operator|(const ADMAOptions &s1, const ADMAOptions &s2) {
        return ADMAOptions{s1.desc.all | s2.desc.all};
    }
    constexpr14 ADMAOptions &operator|=(const ADMAOptions &other) {
        *this = *this | other;
        return *this;
    }

    constexpr14 void setOpcode(ADMAOpcode op)         { desc.common.opcode  = static_cast<uint8_t>(op); }
    constexpr14 void setReductionOp(ADMAReduction op) { desc.adma.reduction = static_cast<uint8_t>(op); }

    void setAddressing(AddressBase base) {
        switch (base.getModel()) {
            case ModelA64:     desc.adma.addrType = 2; break;
            case ModelA64A32S: desc.adma.addrType = 0; break;
            case ModelA64A32U: desc.adma.addrType = 1; break;
            default:
#ifdef NGEN_SAFE
                throw invalid_model_exception();
#endif
                break;
        }
    }
};

static inline ADMAOptions operator|(DataType dt, CacheSettingsLSC cs) { return ADMAOptions(dt) | ADMAOptions(cs); }
static inline ADMAOptions operator|(CacheSettingsLSC cs, DataType dt) { return ADMAOptions(dt) | ADMAOptions(cs); }

/* Async MMA interface */
struct AMMAOptions
{
    SendgMessageDescriptor desc{};

    AMMAOptions() = default;
    explicit constexpr AMMAOptions(uint64_t raw) : desc(raw) {}

    static constexpr AMMAOptions createAScale()      { return AMMAOptions{1ull << 36}; }
    static constexpr AMMAOptions createBScale()      { return AMMAOptions{1ull << 37}; }
    static constexpr AMMAOptions createATranspose()  { return AMMAOptions{1ull << 38}; }
    static constexpr AMMAOptions createBTranspose()  { return AMMAOptions{1ull << 39}; }
    static constexpr AMMAOptions createATrack()      { return AMMAOptions{1ull << 40}; }
    static constexpr AMMAOptions createBTrack()      { return AMMAOptions{1ull << 41}; }
    static constexpr AMMAOptions createDTrack()      { return AMMAOptions{1ull << 42}; }
    static constexpr AMMAOptions createAReuse()      { return AMMAOptions{1ull << 43}; }

    friend inline constexpr AMMAOptions operator|(const AMMAOptions &s1, const AMMAOptions &s2) {
        return AMMAOptions{s1.desc.all | s2.desc.all};
    }
    constexpr14 AMMAOptions &operator|=(const AMMAOptions &other) {
        *this = *this | other;
        return *this;
    }
};
#endif

#if XE4
/* Xe4 common code generator logic */

template <typename O0>
static inline void processTypesXe4(DataType &type, O0 &o)
{
    if (type == DataType::invalid)
        type = o.getType();
#ifdef NGEN_SAFE
    else if (o.getType() == DataType::invalid)
        o.setType(type);
    else if (!o.isNull() && type != o.getType())
        throw invalid_type_exception();
#endif
}

template <>
inline void processTypesXe4(DataType &type, Immediate &o)
{
    if (type == DataType::invalid)
        type = o.getType();
    else if (type != o.getType())
        o = o.cast(type);
}

template <> inline void processTypesXe4(DataType &type, Label &o) {}
template <> inline void processTypesXe4(DataType &type, IndirectARF &o) {}
template <> inline void processTypesXe4(DataType &type, uint32_t &o) {}

template <typename O0, typename... Os>
static inline void processTypesXe4(DataType &type, O0 &o, Os &...others)
{
    processTypesXe4(type, o);
    processTypesXe4(type, others...);
}

static inline void validateXe4(InstructionModifier &mod, OpcodeClassXe4 opclass)
{
#ifdef NGEN_SAFE
    if (mod.isAlign16() || mod.isNoDDClr() || mod.isNoDDChk() || mod.getChannelOffset() || mod.getThreadCtrl() != ThreadCtrl::Normal
            || mod.isAtomic() || mod.isExBSO() || mod.isSerialized() || mod.isEOT())
        throw invalid_modifiers_exception();
    if (mod.hasExecSize()) {
        auto esize = mod.getExecSize();
        if (esize != (isScalar(opclass) ? 1 : 32))
            throw invalid_execution_size_exception();
    }
#endif
}

static inline void validateXe4(RegData &rd)
{
    canonicalizeSRF(rd);
#ifdef NGEN_SAFE
    if (rd.getAbs())
        throw invalid_modifiers_exception();
    if (rd.isARF()) {
        auto atype = rd.getARFType();
        if (atype != ARFType::null && atype != ARFType::lid && atype != ARFType::f)
            throw invalid_arf_exception();
    }
    bool regionOK = ((rd.getOffset() & 0xFF) == 0);
    auto w = rd.getWidth(), hs = rd.getHS(), vs = rd.getVS();
    if (w) {
        if (rd.isSRF())
            regionOK = (w == 1 && hs == 0 && vs == 0);
        else
            regionOK = (w == vs) && (hs == (w > 1) ? 1 : 0);
    }
    if (!regionOK)
        throw invalid_region_exception();
#endif
}

static inline void validateXe4(const Align16Operand &op)
{
#ifdef NGEN_SAFE
    throw invalid_region_exception();
#endif
}

static inline void validateXe4(ExtendedReg &ereg)
{
    validateXe4(ereg.getBase());
}

static inline void validateXe4(const Immediate &i)
{
#ifdef NGEN_SAFE
    switch (i.getType()) {
        case DataType::uv:
        case DataType::v:
        case DataType::vf: throw invalid_type_exception();
        default: break;
    }
#endif
}

static inline void validateXe4(const Label &) {}
static inline void validateXe4(IndirectARF) {}

template <typename T>
static inline void validateIndXe4(const T &t) {
#ifdef NGEN_SAFE
    throw invalid_operand_exception();
#endif
}
template <> inline void validateIndXe4(const RegData &rd) {
#ifdef NGEN_SAFE
    if (!rd.isIndirect()) throw invalid_operand_exception();
#endif
}

template <typename O>
static inline void validateBaseXe4(const O &o) {}
template <> inline void validateBaseXe4(const RegData &rd) {
#ifdef NGEN_SAFE
    if (rd.getDwords() == 2 && (rd.getBase() & 1))
        throw invalid_64_bit_register_exception();
#endif
}

template <typename O0, typename... Os>
static inline void validateBaseXe4(const O0 &o, const Os &...others)
{
    validateBaseXe4(o);
    validateBaseXe4(others...);
}

template <typename O>
static inline bool allowScalarization(O o)              { return o.isScalar(); }
static inline bool allowScalarization(RegData rd)       { return rd.isScalar() || rd.isNull(); }
static inline bool allowScalarization(NullRegister n)   { return true; }
static inline bool allowScalarization(IndirectARF iarf) { return true; }
static inline bool allowScalarization(uint32_t i)       { return true; }
#endif

} /* namespace NGEN_NAMESPACE */

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#endif /* header guard */
