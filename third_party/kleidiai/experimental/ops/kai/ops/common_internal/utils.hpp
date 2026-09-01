//
// SPDX-FileCopyrightText: Copyright 2017-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kai/ops/gemm/kai_ops.hpp"

#include <cstddef>

// Macro for unreachable code (e.g. impossible default cases on switch)
#define UNREACHABLE(why)  __builtin_unreachable()

// Paranoid option for the above with assert
// #define UNREACHABLE(why)   assert(0 && why)

namespace kai {
namespace ops {

template<typename T>
std::string get_type_name() {
#ifdef __GNUC__
    std::string s = __PRETTY_FUNCTION__;

    auto start = s.find("cls_");

    if (start==std::string::npos) {
        return "(unknown)";
    }

    for(size_t x = start+4; x<s.size(); x++) {
        if (s[x] == ';' || s[x] == ']') {
            return s.substr(start+4, x-(start+4));
        }
    }

    return "(unknown)";
#else
    return "(unsupported)";
#endif
}

template<typename T>
inline T iceildiv(const T a, const T b) {
    return (a + b - 1) / b;
}

// Undefine any existing roundup macro from macOS headers
#if defined(__APPLE__) && defined(roundup)
#undef roundup
#endif

template <typename T>
inline T roundup(const T a, const T b) {
    T rem = a % b;

    if (rem) {
        return a + b - rem;
    } else {
        return a;
    }
}

enum class VLType {
    None,
    SVE,
    SME,
    SME2
};

template<typename T>
struct OutputArg {
    T *base = nullptr;
    size_t stride = 0;

    OutputArg(T *base, size_t stride) : base(base), stride(stride) {}

    OutputArg() = default;
};

// Check that the provided Requantize32 doesn't have a left shift.
inline bool quant_no_left_shift(const Requantize32 &qp) {
    if (qp.per_channel_requant) {
        return (qp.per_channel_left_shifts == nullptr);
    } else {
        return (qp.per_layer_left_shift == 0);
    }
}

// Check that the provided Requantize32 is compatible with the "symmetric" hybrid kernels.  These don't include row
// sums, so the 'b_offset' has to be zero.
inline bool quant_hybrid_symmetric(const Requantize32 &qp) {
    return quant_no_left_shift(qp) && qp.b_offset == 0;
}

// Check that the provided Requantize32 is compatible with the "asymmetric" hybrid kernels.  These don't support per
// channel quantization.  Technically b_offset==0 cases would work, but it is a waste to sum and then multiply by 0...
inline bool quant_hybrid_asymmetric(const Requantize32 &qp) {
    return quant_no_left_shift(qp) /*  && qp.b_offset != 0 */ && qp.per_channel_requant==false;
}

template<typename T>
struct IndirectInputArg {
    struct {
        const T *base;
        size_t   stride;
    } direct = {};
    struct {
        const T * const * const * ptr;
        unsigned int start_row;
        unsigned int start_col;
    } indirect = {};
    bool is_indirect;

    // Direct
    IndirectInputArg(const T *base, size_t stride) : is_indirect(false) {
        direct.base = base;
        direct.stride = stride;
    }

    // Indirect
    IndirectInputArg(const T * const * const *ptr, unsigned int start_row, unsigned int start_col) : is_indirect(true) {
        indirect.ptr = ptr;
        indirect.start_row = start_row;
        indirect.start_col = start_col;
    }

    IndirectInputArg() : is_indirect(false) {
        direct.base = nullptr;
        direct.stride = 0;
    }
};

namespace utils {

#ifdef __aarch64__
// get_vector_length(): Returns SVE vector length for type "T".
//
// It is required that this can be compiled by a compiler in non-SVE mode, but it must be prevented from running (at
// runtime) if SVE is not enabled.  Typically this is used by switchyard/driver code which is built in normal mode
// which then calls SVE kernels (compiled accordingly) iff SVE is detected at runtime.
template <typename T>
inline unsigned long get_vector_length() {
    uint64_t vl;

    __asm __volatile (
        ".inst 0x0420e3e0\n" // CNTB X0, ALL, MUL #1
        "mov %0, X0\n"
        : "=r" (vl)
        :
        : "x0"
    );

    return vl / sizeof(T);
}

namespace sme {

template <typename T>
inline unsigned long get_vector_length() {
    uint64_t raw_vector_length;

    __asm __volatile (
        ".inst 0x04bf5821\n" // RDSVL X1, #1
        "mov %0, X1\n"
        : "=r" (raw_vector_length)
        :
        : "x1"
    );

    return raw_vector_length / sizeof(T);
}

} // namespace sme
#endif // __aarch64__

// get_vector_length(VLType): Returns vector length for type "T".
//
// This has the same requirements and constraints as the SVE-only form above, so we call into that code for SVE.

template <typename T>
inline unsigned long get_vector_length(VLType vl_type) {
  switch (vl_type) {
#ifdef __aarch64__
    case VLType::SME:
      return sme::get_vector_length<T>();
    case VLType::SVE:
      return get_vector_length<T>();
#endif // __aarch64__
    default:
      return 16 / sizeof(T);
  }
}

}  // namespace utils
}  // namespace ops
}  // namespace kai

using namespace kai::ops::utils;
