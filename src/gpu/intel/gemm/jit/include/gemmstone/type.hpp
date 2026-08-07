/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef GEMMSTONE_INCLUDE_GEMMSTONE_TYPE_HPP
#define GEMMSTONE_INCLUDE_GEMMSTONE_TYPE_HPP

#include "gemmstone/config.hpp"

#include "internal/ngen_includes.hpp"
#include "internal/utils.hpp"

GEMMSTONE_NAMESPACE_START


// Enum-like class for data types.
class Type {
public:
    /*
    * Bitfield type ID, consists of following flags/ranges:
    * 0 - int/fp
    * 1 - (if int) u/s
    * 2 - byte/bit size
    * 3 - reserved
    * 4-7 - log2 of size in bytes / bit size (depends on flag in bit 2)
    * 8-13 - ngen reference table
    * 14-15 - reserved
    * 16 - real/complex
    * 17 - 1 if split complex
    * 18-19 - reserved
    * 20-23 - vector component number
    * 24-31 - reserved
    */
    enum _Type : uint32_t {
        invalid  = 0,
        f16      = 0x00100011,
        f32      = 0x00100121,
        f64      = 0x00100231,
        u2       = 0x00101024,
        s2       = 0x00101126,
        u4       = 0x00101244,
        s4       = 0x00101346,
        u8       = 0x00101400,
        s8       = 0x00101502,
        u16      = 0x00101610,
        s16      = 0x00101712,
        u32      = 0x00101820,
        s32      = 0x00101922,
        u64      = 0x00101A30,
        s64      = 0x00101B32,
        f4_e2m1  = 0x00100445,
        nf4      = 0x00100645,
        f8_e8m0  = 0x00100801,

        bf8      = 0x00100E01,
        hf8      = 0x00100F01,
        bf16     = 0x00100C11,
        tf32     = 0x00100D21,
    };

private:
    _Type val;

public:
    constexpr Type() : Type(f32) {}
    constexpr Type(_Type val_) : val(val_) {}
    constexpr operator _Type() const { return val; }

    constexpr Type real()             const { return *this; }
    constexpr bool isComplex()        const { return false; }
    constexpr int complexComponents() const { return 1; }
    constexpr int components()        const { return 1; }
    constexpr bool isFP()             const { return uint32_t(val) & 0x1; }
    constexpr bool isInteger()        const { return !isFP(); }
    constexpr bool isSubByteInt()     const { return isSubByte() && isInteger(); };
    constexpr bool isInt8()           const { return (val == Type::u8)  || (val == Type::s8);  }
    constexpr bool isInt16()          const { return (val == Type::u16) || (val == Type::s16); }
    constexpr bool isF8()             const { return (val == Type::bf8) || (val == Type::hf8) || (val == Type::f8_e8m0); }
    constexpr bool isF4()             const { return bits() == 4 && isFP(); }
    constexpr bool isSigned()         const { return (uint32_t(val) & 0x3) != 0x0; }
    constexpr int bits()              const { return isSubByte() ? (uint32_t(val) >> 4) & 0xF : paddedSize() * 8; }
    constexpr int paddedSize()        const { return isSubByte() ? 1 : 1 << ((uint32_t(val) >> 4) & 0xF); }
    int log2Size()                    const { subByteCheck(); return (uint32_t(val) >> 4) & 0xF; }
    int size()                        const { subByteCheck(); return paddedSize(); }
    constexpr int isSubByte()         const { return uint32_t(val) & 0x4; }
    constexpr int perByte()           const { return isSubByte() ? 8 / bits() : 1; }
    uint16_t logPerByte()             const { if (!isSubByte()) stub(); return (perByte() == 4) ? 2 : 1; }
    void subByteCheck()               const { if (isSubByte()) stub(); }

    constexpr Type arithmetic() const {
        return (val == tf32) ? Type(f32) : real();
    }
    constexpr Type asUnsigned() const {
        return static_cast<_Type>(uint32_t(val) & ~(isInteger() ? 0x102 : 0));
    }
    constexpr Type asSigned() const {
        return static_cast<_Type>(uint32_t(val) | (isInteger() ? 0x102 : 0));
    }
    constexpr Type baseType() const { return *this; }

    template <typename U> constexpr friend decltype(std::declval<U>()*1) operator*(U a, Type t) {
        return t.isSubByte() ? (a * t.bits() + (a >= 0 ? 7 : -7)) / 8 : a * int(1u << t.log2Size());
    }
    template <typename U> constexpr friend decltype(std::declval<U>()*1) operator*(Type t, U a) { return a * t; }
    template <typename U>           friend U operator*=(U &a, Type t) { a = a * t; return a; }
    template <typename U> constexpr friend decltype(std::declval<U>()/1) operator/(U a, Type t) {
        return t.isSubByte() ? a * 8 / t.bits() : a / int(1u << t.log2Size());
    }

    // Not valid nGEN DataTypes; for gemmstone internal use only
    static constexpr ngen::DataType ngen_nf4()  { return static_cast<ngen::DataType>(0x58); }
    static constexpr ngen::DataType ngen_e8m0() { return static_cast<ngen::DataType>(0x79); }

    ngen::DataType ngen() const
    {
        uint32_t index = (uint32_t(val) >> 8) & 0x3F;
        using DT = ngen::DataType;
        auto none = DT::invalid;
        static const DT table[64] = {
            DT::hf,      DT::f,       DT::df,      none,
            DT::e2m1,    none,        ngen_nf4(),  none,
            ngen_e8m0(), none,        none,        none,
            DT::bf,      DT::tf32,    DT::bf8,     DT::hf8,
            DT::u2,      DT::s2,      DT::u4,      DT::s4,
            DT::ub,      DT::b,       DT::uw,      DT::w,
            DT::ud,      DT::d,       DT::uq,      DT::q,
            none,        none,        none,        none,
            none,        none,        none,        none,
            none,        none,        none,        none,
            none,        none,        none,        none,
            none,        none,        none,        none,
            none,        none,        none,        none,
            none,        none,        none,        none,
            none,        none,        none,        none,
            none,        none,        none,        none
        };
        return table[index];
    }

    bool isSubsetOf(Type T) const
    {
        if (*this == T) return true;
        return (real().bits() < T.real().bits());
    }

    Type asSignedInt() const
    {
        switch (bits()) {
            case 8:  return Type::s8;
            case 16: return Type::s16;
            case 32: return Type::s32;
            default: return Type::invalid;
        }
    }
};

static_assert((-9 * Type(Type::s4) == -5) && (9 * Type(Type::s4) == 5), "Round away from zero is required non-integer type sizes");

inline char typeToChar(Type T)
{
    switch (T.baseType()) {
        case Type::bf8:     return 'Q';
        case Type::hf8:     return 'q';
        case Type::f16:     return 'H';
        case Type::f32:     return 'S';
        case Type::f64:     return 'D';
        case Type::u2:      return 'p';
        case Type::s2:      return 'P';
        case Type::u4:      return 'f';
        case Type::s4:      return 'F';
        case Type::u8:      return 'o';
        case Type::s8:      return 'O';
        case Type::u16:     return 'w';
        case Type::s16:     return 'W';
        case Type::u32:     return 'i';
        case Type::s32:     return 'I';
        case Type::u64:     return 'l';
        case Type::s64:     return 'L';
        case Type::bf16:    return 'B';
        case Type::tf32:    return 'T';
        case Type::f4_e2m1: return 'E';
        case Type::nf4:     return 'N';
        case Type::f8_e8m0: return 'X';
        default:            return '?';
    }
}

inline Type charToType(char c)
{
    switch (c) {
        case 'Q': return Type::bf8;
        case 'q': return Type::hf8;
        case 'H': return Type::f16;
        case 'S': return Type::f32;
        case 'D': return Type::f64;
        case 'p': return Type::u2;
        case 'P': return Type::s2;
        case 'f': return Type::u4;
        case 'F': return Type::s4;
        case 'o': return Type::u8;
        case 'O': return Type::s8;
        case 'w': return Type::u16;
        case 'W': return Type::s16;
        case 'i': return Type::u32;
        case 'I': return Type::s32;
        case 'B': return Type::bf16;
        case 'T': return Type::tf32;
        case 'E': return Type::f4_e2m1;
        case 'N': return Type::nf4;
        case 'X': return Type::f8_e8m0;
        default:  return Type::invalid;
    }
}

/****************************/
/* Old names -- deprecated. */
static inline char precisionChar(Type T) { return typeToChar(T); }
static inline Type charPrecision(char c) { return charToType(c); }
/****************************/

GEMMSTONE_NAMESPACE_END

#endif /* header guard */
