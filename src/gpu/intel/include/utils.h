/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GPU_INTEL_INCLUDE_UTILS_H
#define GPU_INTEL_INCLUDE_UTILS_H

#include "gpu/intel/include/config.h"

#define CONCAt2(a, b) a##b
#define CONCAT2(a, b) CONCAt2(a, b)
#define CONCAT3(a, b, c) CONCAT2(CONCAT2(a, b), c)

// Type guard macros: GUARD_<type>(x) expands to x when the type is available,
// or to nothing when it is not. Used to conditionally emit definitions.
#define GUARD(dt) CONCAT2(GUARD_, dt)

// Always-available types
#define GUARD_float(x) x
#define GUARD_int(x) x
#define GUARD_char(x) x
#define GUARD_uchar(x) x
#define GUARD_undef_data(x) x

// Hardware extension types
// Define extension macros if the corresponding DT_* macros are set but the
// compiler hasn't defined the extension macro (e.g. cl_khr_fp16 is not
// automatically defined by the pragma).
#if !defined(cl_khr_fp16) \
        && (defined(DT_F16) || defined(SRC_DT_F16) || defined(SRC0_DT_F16) \
                || defined(SRC1_DT_F16) || defined(DST_DT_F16) \
                || defined(WEI_DT_F16) || defined(BIA_DT_F16) \
                || defined(ACC_DT_F16))
#define cl_khr_fp16 1
#endif

#if !defined(cl_khr_fp64) \
        && (defined(DT_F64) || defined(SRC_DT_F64) || defined(SRC0_DT_F64) \
                || defined(SRC1_DT_F64) || defined(DST_DT_F64) \
                || defined(WEI_DT_F64) || defined(BIA_DT_F64) \
                || defined(ACC_DT_F64))
#define cl_khr_fp64 1
#endif

#ifdef cl_khr_fp64
#define GUARD_double(x) x
#else
#define GUARD_double(x)
#endif

#ifdef cl_khr_fp16
#define GUARD_half(x) x
#else
#define GUARD_half(x)
#endif

// Custom struct types
#ifdef MATH_UTILS_DECLARE_BF16
#define GUARD_bf16(x) x
#else
#define GUARD_bf16(x)
#endif

#ifdef MATH_UTILS_DECLARE_BF8
#define GUARD_f8_e5m2(x) x
#else
#define GUARD_f8_e5m2(x)
#endif

#ifdef MATH_UTILS_DECLARE_HF8
#define GUARD_f8_e4m3(x) x
#else
#define GUARD_f8_e4m3(x)
#endif

#ifdef MATH_UTILS_DECLARE_F4_E2M1
#define GUARD_f4_e2m1(x) x
#else
#define GUARD_f4_e2m1(x)
#endif

#ifdef MATH_UTILS_DECLARE_F4_E3M0
#define GUARD_f4_e3m0(x) x
#else
#define GUARD_f4_e3m0(x)
#endif

#ifdef MATH_UTILS_DECLARE_S4
#define GUARD_s4(x) x
#else
#define GUARD_s4(x)
#endif

#ifdef MATH_UTILS_DECLARE_U4
#define GUARD_u4(x) x
#else
#define GUARD_u4(x)
#endif

#ifdef MATH_UTILS_DECLARE_E8M0
#define GUARD_e8m0(x) x
#else
#define GUARD_e8m0(x)
#endif

#if __OPENCL_C_VERSION__ >= CL_VERSION_2_0
#define ATOMICS_SUPPORTED 1
#else
#define ATOMICS_SUPPORTED 0
#endif

#if defined(cl_ext_float_atomics) && ATOMICS_SUPPORTED
#define ATOMIC_FLOAT_SUPPORTED 1
#else
#define ATOMIC_FLOAT_SUPPORTED 0
#endif

#ifdef OCL_DEBUG
#define DEBUG_PRINT(...) printf(__VA_ARGS__);
#else
#define DEBUG_PRINT(...)
#endif

#ifdef __has_builtin
#define HAS_BUILTIN(x) __has_builtin(x)
#else
#define HAS_BUILTIN(x) false
#endif

// Defines (for example) float_zero, float_one, float_min, and float_max
// Can be used in a data-type agnostic way with the SPECIAL macro below
#define DEF_special_vals(dt, zero_val, one_val, min_val, max_val) \
    dt CONCAT2(dt, _zero)() { \
        return zero_val; \
    } \
    dt CONCAT2(dt, _one)() { \
        return one_val; \
    } \
    dt CONCAT2(dt, _min)() { \
        return min_val; \
    } \
    dt CONCAT2(dt, _max)() { \
        return max_val; \
    }

DEF_special_vals(float, 0.0f, 1.0f, -FLT_MAX, FLT_MAX);
DEF_special_vals(int, 0, 1, INT_MIN, INT_MAX);
DEF_special_vals(char, 0, 1, CHAR_MIN, CHAR_MAX);
DEF_special_vals(uchar, 0, 1, 0, UCHAR_MAX);
GUARD_double(DEF_special_vals(double, 0.0, 1.0, -DBL_MAX, DBL_MAX));
GUARD_half(DEF_special_vals(half, 0.0h, 1.0h, -HALF_MAX, HALF_MAX));
GUARD_bf16(DEF_special_vals(bf16, as_bf16(0), as_bf16(0x3F80),
        as_bf16((short)0xFF7F), as_bf16(0x7F7F)));

#define SPECIAL(dt, val) CONCAT3(dt, _, val)()

#define DEF_overloadable_special_vals(dt) \
    dt __attribute__((overloadable)) zero_val(dt dummy) { \
        return CONCAT2(dt, _zero)(); \
    } \
    dt __attribute__((overloadable)) one_val(dt dummy) { \
        return CONCAT2(dt, _one)(); \
    } \
    dt __attribute__((overloadable)) min_val(dt dummy) { \
        return CONCAT2(dt, _min)(); \
    } \
    dt __attribute__((overloadable)) max_val(dt dummy) { \
        return CONCAT2(dt, _max)(); \
    }

DEF_overloadable_special_vals(float);
DEF_overloadable_special_vals(int);
DEF_overloadable_special_vals(char);
DEF_overloadable_special_vals(uchar);
GUARD_double(DEF_overloadable_special_vals(double));
GUARD_half(DEF_overloadable_special_vals(half));
GUARD_bf16(DEF_overloadable_special_vals(bf16));

GUARD_f8_e5m2(DEF_special_vals(f8_e5m2, as_f8_e5m2(0), as_f8_e5m2(0x3C),
        as_f8_e5m2((char)0xFB), as_f8_e5m2(0x7B)));
GUARD_f8_e5m2(DEF_overloadable_special_vals(f8_e5m2));

GUARD_f8_e4m3(DEF_special_vals(f8_e4m3, as_f8_e4m3(0), as_f8_e4m3(0x38),
        as_f8_e4m3((char)0xFE), as_f8_e4m3(0x7E)));
GUARD_f8_e4m3(DEF_overloadable_special_vals(f8_e4m3));

#ifdef ENABLE_CHECK_ASSUMPTIONS
// Don't actually inform the compiler about the assumption
#define ASSUME(x) \
    if (!(x)) { \
        printf("Error - GWS indices (%ld,%ld,%ld): Runtime assumption \"%s\" " \
               "violated\n", \
                get_global_id(0), get_global_id(1), get_global_id(2), #x); \
        return; \
    }
#elif HAS_BUILTIN(__builtin_assume)
#define ASSUME(x) __builtin_assume(x)
#else
#define ASSUME(x)
#endif

/* Conditionally insert text if enabled is true */
#define OPTIONAL(condition, text) CONCAT2(OPTIONAL_, condition)(text)
#define OPTIONAL_0(text)
#define OPTIONAL_1(text) , text

// Boolean OR preprocessor
#define OR(a, b) CONCAT2(OR_RESULT, CONCAT2(a, b))
#define OR_RESULT00 0
#define OR_RESULT01 1
#define OR_RESULT10 1
#define OR_RESULT11 1

// Boolean AND preprocessor
#define AND(a, b) CONCAT2(AND_RESULT, CONCAT2(a, b))
#define AND_RESULT00 0
#define AND_RESULT01 0
#define AND_RESULT10 0
#define AND_RESULT11 1

#endif // GPU_INTEL_INCLUDE_UTILS_H
