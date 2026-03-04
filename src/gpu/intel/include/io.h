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

#ifndef GPU_INTEL_INCLUDE_IO_H
#define GPU_INTEL_INCLUDE_IO_H

#include "gpu/intel/include/conversion.h"
#include "gpu/intel/include/custom_types.h"
#include "gpu/intel/include/utils.h"

//******* Always-available load/writes *********//

#define BLOCK_READ_FUNC_0 get_half_byte
#define BLOCK_READ_FUNC_1 intel_sub_group_block_read_uc
#define BLOCK_READ_FUNC_2 intel_sub_group_block_read_us
#define BLOCK_READ_FUNC_4 intel_sub_group_block_read
#define BLOCK_READ_FUNC_8 intel_sub_group_block_read_ul

#define BLOCK_WRITE_FUNC_0 intel_sub_group_block_write_uc
#define BLOCK_WRITE_FUNC_1 intel_sub_group_block_write_uc
#define BLOCK_WRITE_FUNC_2 intel_sub_group_block_write_us
#define BLOCK_WRITE_FUNC_4 intel_sub_group_block_write
#define BLOCK_WRITE_FUNC_8 intel_sub_group_block_write_ul

#define BLOCK_DT_0 uchar
#define BLOCK_DT_1 uchar
#define BLOCK_DT_2 ushort
#define BLOCK_DT_4 uint
#define BLOCK_DT_8 ulong

#define SIZE_undef_data 1
#define SIZE_char 1
#define SIZE_uchar 1
#define SIZE_f8_e5m2 1
#define SIZE_f8_e4m3 1
#define SIZE_f4_e2m1 0
#define SIZE_f4_e3m0 0
#define SIZE_e8m0 1
#define SIZE_bf16 2
#define SIZE_half 2
#define SIZE_int 4
#define SIZE_float 4
#define SIZE_double 8

#define SIZE(dt) CONCAT2(SIZE_, dt)
#define BLOCK_DT(dt) CONCAT2(BLOCK_DT_, SIZE(dt))
#define BLOCK_READ_FUNC(dt) CONCAT2(BLOCK_READ_FUNC_, SIZE(dt))
#define BLOCK_WRITE_FUNC(dt) CONCAT2(BLOCK_WRITE_FUNC_, SIZE(dt))

#define BLOCK_DT_N(dt, n) CONCAT2(BLOCK_DT(dt), n)
#define BLOCK_READ_FUNC_N(dt, n) CONCAT2(BLOCK_READ_FUNC(dt), n)
#define BLOCK_WRITE_FUNC_N(dt, n) CONCAT2(BLOCK_WRITE_FUNC(dt), n)

#define DECLARE_AS_BLOCK(t) \
    BLOCK_DT(t) __attribute__((overloadable)) as_block_data(t a) { \
        return CONCAT2(as_, BLOCK_DT(t))(a); \
    }

#define DECLARE_AS_STRUCT_BLOCK(t) \
    BLOCK_DT(t) __attribute__((overloadable)) as_block_data(t a) { \
        return CONCAT2(as_, BLOCK_DT(t))(a.data); \
    }

#ifdef MATH_UTILS_DECLARE_F4_E2M1
DECLARE_AS_STRUCT_BLOCK(f4_e2m1)
#endif
#ifdef MATH_UTILS_DECLARE_F4_E3M0
DECLARE_AS_STRUCT_BLOCK(f4_e3m0)
#endif
DECLARE_AS_BLOCK(char)
DECLARE_AS_BLOCK(uchar)
DECLARE_AS_STRUCT_BLOCK(f8_e5m2)
DECLARE_AS_STRUCT_BLOCK(f8_e4m3)
DECLARE_AS_STRUCT_BLOCK(e8m0)
DECLARE_AS_STRUCT_BLOCK(bf16)
DECLARE_AS_BLOCK(half)
DECLARE_AS_BLOCK(int)
DECLARE_AS_BLOCK(float)
#ifdef cl_khr_fp64
DECLARE_AS_BLOCK(double)
#endif

// Vector struct as_block_data
#define DECLARE_AS_STRUCT_BLOCK_VEC(t, n, block_vec_t) \
    block_vec_t __attribute__((overloadable)) \
            as_block_data(CONCAT2(t, CONCAT2(x, n)) a) { \
        return CONCAT2(as_, block_vec_t)(a.data); \
    }

DECLARE_AS_STRUCT_BLOCK_VEC(bf16, 2, ushort2)
DECLARE_AS_STRUCT_BLOCK_VEC(bf16, 4, ushort4)
DECLARE_AS_STRUCT_BLOCK_VEC(bf16, 8, ushort8)

DECLARE_AS_STRUCT_BLOCK_VEC(f8_e5m2, 2, uchar2)
DECLARE_AS_STRUCT_BLOCK_VEC(f8_e5m2, 4, uchar4)
DECLARE_AS_STRUCT_BLOCK_VEC(f8_e5m2, 8, uchar8)

DECLARE_AS_STRUCT_BLOCK_VEC(f8_e4m3, 2, uchar2)
DECLARE_AS_STRUCT_BLOCK_VEC(f8_e4m3, 4, uchar4)
DECLARE_AS_STRUCT_BLOCK_VEC(f8_e4m3, 8, uchar8)

DECLARE_AS_STRUCT_BLOCK_VEC(f4_e2m1, 2, uchar2)
DECLARE_AS_STRUCT_BLOCK_VEC(f4_e2m1, 4, uchar4)
DECLARE_AS_STRUCT_BLOCK_VEC(f4_e2m1, 8, uchar8)

DECLARE_AS_STRUCT_BLOCK_VEC(f4_e3m0, 2, uchar2)
DECLARE_AS_STRUCT_BLOCK_VEC(f4_e3m0, 4, uchar4)
DECLARE_AS_STRUCT_BLOCK_VEC(f4_e3m0, 8, uchar8)

#undef DECLARE_AS_BLOCK
#undef DECLARE_AS_STRUCT_BLOCK
#undef DECLARE_AS_STRUCT_BLOCK_VEC

#define DEF_load(dst_dt, src_dt) \
    void __attribute__((overloadable)) load( \
            __private dst_dt *dst, __global const src_dt *val) { \
        *dst = CONCAT2(into_, dst_dt)(*val); \
    } \
    dst_dt __attribute__((overloadable, warn_unused_result)) load( \
            dst_dt dst, __global const src_dt *val) { \
        return CONCAT2(into_, dst_dt)(*val); \
    } \
    void __attribute__((overloadable)) load( \
            __private dst_dt *dst, __global const src_dt *val, off_t off) { \
        *dst = CONCAT2(into_, dst_dt)(val[off]); \
    } \
    dst_dt __attribute__((overloadable, warn_unused_result)) load( \
            dst_dt dst, __global const src_dt *val, off_t off) { \
        return CONCAT2(into_, dst_dt)(val[off]); \
    } \
    void __attribute__((overloadable)) load( \
            __private dst_dt *dst, __private const src_dt *val) { \
        *dst = CONCAT2(into_, dst_dt)(*val); \
    } \
    dst_dt __attribute__((overloadable, warn_unused_result)) load( \
            dst_dt dst, __private const src_dt *val) { \
        return CONCAT2(into_, dst_dt)(*val); \
    } \
    __attribute__((overloadable)) void block_load( \
            __private dst_dt *dst, __global const src_dt *src, int n) { \
        __attribute__((opencl_unroll_hint)) while (n >= 8) { \
            __global BLOCK_DT(src_dt) *data \
                    = (__global BLOCK_DT(src_dt) *)(src); \
            BLOCK_DT_N(src_dt, 8) \
            block_val = BLOCK_READ_FUNC_N(src_dt, 8)(data); \
            for (int i = 0; i < 8; i++) { \
                src_dt src_val = CONCAT2(as_, src_dt)(block_val[i]); \
                dst[i] = CONCAT2(into_, dst_dt)(src_val); \
            } \
            dst += 8; \
            src += 8 * get_max_sub_group_size(); \
            n -= 8; \
        } \
        if (n >= 4) { \
            __global BLOCK_DT(src_dt) *data \
                    = (__global BLOCK_DT(src_dt) *)(src); \
            BLOCK_DT_N(src_dt, 4) \
            block_val = BLOCK_READ_FUNC_N(src_dt, 4)(data); \
            for (int i = 0; i < 4; i++) { \
                src_dt src_val = CONCAT2(as_, src_dt)(block_val[i]); \
                dst[i] = CONCAT2(into_, dst_dt)(src_val); \
            } \
            dst += 4; \
            src += 4 * get_max_sub_group_size(); \
            n -= 4; \
        } \
        if (n >= 2) { \
            __global BLOCK_DT(src_dt) *data \
                    = (__global BLOCK_DT(src_dt) *)(src); \
            BLOCK_DT_N(src_dt, 2) \
            block_val = BLOCK_READ_FUNC_N(src_dt, 2)(data); \
            for (int i = 0; i < 2; i++) { \
                src_dt src_val = CONCAT2(as_, src_dt)(block_val[i]); \
                dst[i] = CONCAT2(into_, dst_dt)(src_val); \
            } \
            dst += 2; \
            src += 2 * get_max_sub_group_size(); \
            n -= 2; \
        } \
        if (n >= 1) { \
            __global BLOCK_DT(src_dt) *data \
                    = (__global BLOCK_DT(src_dt) *)(src); \
            BLOCK_DT(src_dt) \
            block_val = BLOCK_READ_FUNC(src_dt)(data); \
            src_dt src_val = CONCAT2(as_, src_dt)(block_val); \
            *dst = CONCAT2(into_, dst_dt)(src_val); \
        } \
    } \
    __attribute__((overloadable)) void block_load( \
            __private dst_dt *dst, __global src_dt *src) { \
        __global BLOCK_DT(src_dt) *data = (__global BLOCK_DT(src_dt) *)(src); \
        BLOCK_DT(src_dt) block_val = BLOCK_READ_FUNC(src_dt)(data); \
        src_dt src_val = CONCAT2(as_, src_dt)(block_val); \
        *dst = CONCAT2(into_, dst_dt)(src_val); \
    } \
    __attribute__((overloadable, warn_unused_result)) dst_dt block_load( \
            dst_dt dst, __global src_dt *src) { \
        __global BLOCK_DT(src_dt) *data = (__global BLOCK_DT(src_dt) *)(src); \
        BLOCK_DT(src_dt) block_val = BLOCK_READ_FUNC(src_dt)(data); \
        src_dt src_val = CONCAT2(as_, src_dt)(block_val); \
        return CONCAT2(into_, dst_dt)(src_val); \
    }

#define DEF_load_half_byte(dst_dt, src_dt) \
    dst_dt __attribute__((overloadable, warn_unused_result)) load( \
            dst_dt dst, __global const src_dt *val, off_t off) { \
        src_dt data = CONCAT2(as_, src_dt)( \
                get_half_byte((__global const uchar *)val, off)); \
        return CONCAT2(into_, dst_dt)(data); \
    } \
    void __attribute__((overloadable)) load( \
            __private dst_dt *dst, __global const src_dt *val, off_t off) { \
        *dst = load(*dst, val, off); \
    }

#define DEF_write(dst_dt, src_dt) \
    void __attribute__((overloadable)) write( \
            __global dst_dt *dst, __private const src_dt *val) { \
        *dst = CONCAT2(into_, dst_dt)(*val); \
    } \
    void __attribute__((overloadable)) write( \
            __global dst_dt *dst, __private src_dt val) { \
        *dst = CONCAT2(into_, dst_dt)(val); \
    } \
    void __attribute__((overloadable)) write( \
            __private dst_dt *dst, __private const src_dt *val) { \
        *dst = CONCAT2(into_, dst_dt)(*val); \
    } \
    void __attribute__((overloadable)) write( \
            __private dst_dt *dst, __private src_dt val) { \
        *dst = CONCAT2(into_, dst_dt)(val); \
    } \
    __attribute__((overloadable)) void block_write( \
            __global dst_dt *dst, __private const src_dt *src, int n) { \
        __attribute__((opencl_unroll_hint)) while (n >= 8) { \
            BLOCK_DT_N(dst_dt, 8) block_val; \
            for (int i = 0; i < 8; i++) { \
                dst_dt val = CONCAT2(into_, dst_dt)(src[i]); \
                block_val[i] = as_block_data(val); \
            } \
            __global BLOCK_DT(dst_dt) *data \
                    = (__global BLOCK_DT(dst_dt) *)(dst); \
            BLOCK_WRITE_FUNC_N(dst_dt, 8)(data, block_val); \
            dst += 8 * get_max_sub_group_size(); \
            src += 8; \
            n -= 8; \
        } \
        if (n >= 4) { \
            BLOCK_DT_N(dst_dt, 4) block_val; \
            for (int i = 0; i < 4; i++) { \
                dst_dt val = CONCAT2(into_, dst_dt)(src[i]); \
                block_val[i] = as_block_data(val); \
            } \
            __global BLOCK_DT(dst_dt) *data \
                    = (__global BLOCK_DT(dst_dt) *)(dst); \
            BLOCK_WRITE_FUNC_N(dst_dt, 4)(data, block_val); \
            dst += 4 * get_max_sub_group_size(); \
            src += 4; \
            n -= 4; \
        } \
        if (n >= 2) { \
            BLOCK_DT_N(dst_dt, 2) block_val; \
            for (int i = 0; i < 2; i++) { \
                dst_dt val = CONCAT2(into_, dst_dt)(src[i]); \
                block_val[i] = as_block_data(val); \
            } \
            __global BLOCK_DT(dst_dt) *data \
                    = (__global BLOCK_DT(dst_dt) *)(dst); \
            BLOCK_WRITE_FUNC_N(dst_dt, 2)(data, block_val); \
            dst += 2 * get_max_sub_group_size(); \
            src += 2; \
            n -= 2; \
        } \
        if (n >= 1) { \
            BLOCK_DT(dst_dt) block_val; \
            dst_dt val = CONCAT2(into_, dst_dt)(*src); \
            block_val = as_block_data(val); \
            __global BLOCK_DT(dst_dt) *data \
                    = (__global BLOCK_DT(dst_dt) *)(dst); \
            BLOCK_WRITE_FUNC(dst_dt)(data, block_val); \
        } \
    } \
    __attribute__((overloadable)) void block_write( \
            __global dst_dt *dst, __private const src_dt *src) { \
        BLOCK_DT(dst_dt) block_val; \
        dst_dt val = CONCAT2(into_, dst_dt)(*src); \
        block_val = as_block_data(val); \
        __global BLOCK_DT(dst_dt) *data = (__global BLOCK_DT(dst_dt) *)(dst); \
        BLOCK_WRITE_FUNC(dst_dt)(data, block_val); \
    }

// Loads
DEF_load(float, int);
DEF_load(float, float);
DEF_load(float, char);
DEF_load(float, uchar);
DEF_load(int, char);
DEF_load(int, uchar);
DEF_load(int, int);
DEF_load(float, bf16);
DEF_load(int, bf16);

// Included for compile time compatibility
DEF_load(int, undef_data);
DEF_load(float, undef_data);

// char/uchar identity block I/O (workspace flags, etc.)
DEF_load(char, char);
DEF_write(char, char);

// Writes
DEF_write(char, float);
DEF_write(uchar, float);
DEF_write(bf16, float);
DEF_write(float, float);

DEF_write(char, int);
DEF_write(uchar, int);
DEF_write(bf16, int);
DEF_write(int, int);
DEF_write(float, int);
DEF_write(int, float);

//******* Conditionally-available load/writes *********//

#ifdef cl_khr_fp16
// Loads
DEF_load(half, half);
DEF_load(float, half);
DEF_load(half, char);
DEF_load(half, uchar);
DEF_load(half, int);
DEF_load(half, bf16);

// Writes
DEF_write(half, float);
DEF_write(half, int);
DEF_write(half, half);
DEF_write(char, half);
DEF_write(uchar, half);
DEF_write(bf16, half);
DEF_write(int, half);

#ifdef MATH_UTILS_DECLARE_BF8
// Loads
DEF_load(half, f8_e5m2);
DEF_load(float, f8_e5m2);

// Writes
DEF_write(f8_e5m2, half);
DEF_write(f8_e5m2, float);
DEF_write(f8_e5m2, int);
#endif // MATH_UTILS_DECLARE_BF8

#ifdef MATH_UTILS_DECLARE_HF8
// Loads
DEF_load(half, f8_e4m3);
DEF_load(float, f8_e4m3);

// Writes
DEF_write(f8_e4m3, half);
DEF_write(f8_e4m3, float);
DEF_write(f8_e4m3, int);

#endif // MATH_UTILS_DECLARE_HF8

#ifdef MATH_UTILS_DECLARE_F4_E2M1
// Loads
DEF_load_half_byte(half, f4_e2m1);
DEF_load_half_byte(float, f4_e2m1);

// Writes
DEF_write(f4_e2m1, half);
DEF_write(f4_e2m1, float);
DEF_write(f4_e2m1, int);

#endif // MATH_UTILS_DECLARE_F4_E2M1

#ifdef MATH_UTILS_DECLARE_F4_E3M0
// Loads
DEF_load_half_byte(half, f4_e3m0);
DEF_load_half_byte(float, f4_e3m0);

// Writes
DEF_write(f4_e3m0, half);
DEF_write(f4_e3m0, float);
DEF_write(f4_e3m0, int);

#endif // MATH_UTILS_DECLARE_F4_E3M0

#ifdef MATH_UTILS_DECLARE_E8M0
// Loads
DEF_load(float, e8m0);

#endif

#endif // cl_khr_fp16

#ifdef cl_khr_fp64
// Included for compile time compatibility
DEF_load(double, undef_data);

DEF_load(float, double); // Needed for src=f64, dst=f32

DEF_load(double, char);
DEF_load(double, uchar);
DEF_load(double, int);
DEF_load(double, bf16);
DEF_load(double, float);
DEF_load(double, double);

DEF_write(char, double);
DEF_write(uchar, double);
DEF_write(bf16, double);
DEF_write(float, double);
DEF_write(double, double);
DEF_write(double, float);
DEF_write(double, int);
DEF_write(int, double);
#endif

//******* Interactions between extended data types *********//

#if defined(cl_khr_fp16) && defined(cl_khr_fp64)

DEF_load(double, half);
DEF_write(half, double);

#ifdef MATH_UTILS_DECLARE_BF8
DEF_load(double, f8_e5m2);
DEF_write(f8_e5m2, double);
#endif // MATH_UTILS_DECLARE_BF8

#ifdef MATH_UTILS_DECLARE_HF8
DEF_load(double, f8_e4m3);
DEF_write(f8_e4m3, double);
#endif // MATH_UTILS_DECLARE_HF8
#endif

//******* Vector block_load / block_write *********//

// Macro for types with vector cvt_* builtins (no loop):
#define DEF_block_load_vec(acc_vec, src_dt, block_read_func, \
        block_dt_vec, cvt_func) \
    void __attribute__((overloadable)) block_load( \
            __private acc_vec *dst, __global const src_dt *src) { \
        block_dt_vec raw = block_read_func( \
                (const __global BLOCK_DT(src_dt) *)src); \
        *dst = cvt_func(raw); \
    } \
    acc_vec __attribute__((overloadable, warn_unused_result)) block_load( \
            acc_vec dst, __global const src_dt *src) { \
        block_dt_vec raw = block_read_func( \
                (const __global BLOCK_DT(src_dt) *)src); \
        return cvt_func(raw); \
    }

// Macro for types without vector builtins (scalar loop fallback):
#define DEF_block_load_vec_loop(acc_vec, src_dt, block_read_func, \
        block_dt_vec, as_scalar, scalar_fn, n) \
    void __attribute__((overloadable)) block_load( \
            __private acc_vec *dst, __global const src_dt *src) { \
        block_dt_vec raw = block_read_func( \
                (const __global BLOCK_DT(src_dt) *)src); \
        for (int i = 0; i < n; i++) \
            (*dst)[i] = scalar_fn(as_scalar(raw[i])); \
    } \
    acc_vec __attribute__((overloadable, warn_unused_result)) block_load( \
            acc_vec dst, __global const src_dt *src) { \
        block_dt_vec raw = block_read_func( \
                (const __global BLOCK_DT(src_dt) *)src); \
        for (int i = 0; i < n; i++) \
            dst[i] = scalar_fn(as_scalar(raw[i])); \
        return dst; \
    }

// Macro for types with vector cvt_* builtins (no loop):
#define DEF_block_write_vec(dst_dt, acc_vec, block_write_func, \
        block_dt_vec, cvt_func) \
    void __attribute__((overloadable)) block_write( \
            __global dst_dt *dst, __private const acc_vec *src) { \
        block_dt_vec raw = cvt_func(*src); \
        block_write_func((__global BLOCK_DT(dst_dt) *)dst, raw); \
    }

// Macro for types without vector builtins (scalar loop fallback):
#define DEF_block_write_vec_loop(dst_dt, acc_vec, block_write_func, \
        block_dt_vec, scalar_fn, n) \
    void __attribute__((overloadable)) block_write( \
            __global dst_dt *dst, __private const acc_vec *src) { \
        block_dt_vec raw; \
        for (int i = 0; i < n; i++) \
            raw[i] = as_block_data(scalar_fn((*src)[i])); \
        block_write_func((__global BLOCK_DT(dst_dt) *)dst, raw); \
    }

// Raw block_load/write for custom vector struct types (no conversion)
#define DEF_block_load_vec_raw(struct_vec, scalar_dt, block_read_func, \
        block_vec, raw_vec) \
    void __attribute__((overloadable)) block_load( \
            __private struct_vec *dst, __global const scalar_dt *src) { \
        block_vec raw = block_read_func( \
                (const __global BLOCK_DT(scalar_dt) *)src); \
        dst->data = CONCAT2(as_, raw_vec)(raw); \
    } \
    struct_vec __attribute__((overloadable, warn_unused_result)) block_load( \
            struct_vec dst, __global const scalar_dt *src) { \
        block_vec raw = block_read_func( \
                (const __global BLOCK_DT(scalar_dt) *)src); \
        dst.data = CONCAT2(as_, raw_vec)(raw); \
        return dst; \
    }

#define DEF_block_write_vec_raw(scalar_dt, struct_vec, block_write_func, \
        block_vec) \
    void __attribute__((overloadable)) block_write( \
            __global scalar_dt *dst, __private const struct_vec *src) { \
        block_write_func((__global BLOCK_DT(scalar_dt) *)dst, \
                CONCAT2(as_, block_vec)(src->data)); \
    }

// --- block_load instantiations ---

#if MATH_UTILS_DECLARE_BF16
// bf16 source (vector cvt_bf16_to_f32 exists)
DEF_block_load_vec(float8, bf16, intel_sub_group_block_read_us8, ushort8, cvt_bf16_to_f32)
DEF_block_load_vec(float4, bf16, intel_sub_group_block_read_us4, ushort4, cvt_bf16_to_f32)
DEF_block_load_vec(float2, bf16, intel_sub_group_block_read_us2, ushort2, cvt_bf16_to_f32)

// float source (as_float* reinterpret only)
DEF_block_load_vec(float8, float, intel_sub_group_block_read8, uint8, as_float8)
DEF_block_load_vec(float4, float, intel_sub_group_block_read4, uint4, as_float4)
DEF_block_load_vec(float2, float, intel_sub_group_block_read2, uint2, as_float2)

// char/uchar source (scalar loop, no vector cvt_*)
DEF_block_load_vec_loop(float8, char, intel_sub_group_block_read_uc8, uchar8, as_char, into_float, 8)
DEF_block_load_vec_loop(float8, uchar, intel_sub_group_block_read_uc8, uchar8, as_uchar, into_float, 8)
DEF_block_load_vec_loop(float2, char, intel_sub_group_block_read_uc2, uchar2, as_char, into_float, 2)
DEF_block_load_vec_loop(float2, uchar, intel_sub_group_block_read_uc2, uchar2, as_uchar, into_float, 2)

// --- block_write instantiations ---

// bf16 destination (vector cvt_f32_to_bf16 exists)
DEF_block_write_vec(bf16, float8, intel_sub_group_block_write_us8, ushort8, cvt_f32_to_bf16)
DEF_block_write_vec(bf16, float4, intel_sub_group_block_write_us4, ushort4, cvt_f32_to_bf16)
#endif // MATH_UTILS_DECLARE_BF16

// float destination (as_uint* reinterpret only)
DEF_block_write_vec(float, float8, intel_sub_group_block_write8, uint8, as_uint8)

// char/uchar destination (scalar loop)
DEF_block_write_vec_loop(char, float8, intel_sub_group_block_write_uc8, uchar8, into_char, 8)
DEF_block_write_vec_loop(uchar, float8, intel_sub_group_block_write_uc8, uchar8, into_uchar, 8)

// Raw char8 block I/O (no conversion)
void __attribute__((overloadable)) block_load(
        __private char8 *dst, __global const char *src) {
    uchar8 raw = intel_sub_group_block_read_uc8((const __global uchar *)src);
    *dst = as_char8(raw);
}
char8 __attribute__((overloadable, warn_unused_result)) block_load(
        char8 dst, __global const char *src) {
    uchar8 raw = intel_sub_group_block_read_uc8((const __global uchar *)src);
    return as_char8(raw);
}
void __attribute__((overloadable)) block_write(
        __global char *dst, __private const char8 *src) {
    intel_sub_group_block_write_uc8((__global uchar *)dst, as_uchar8(*src));
}

// Raw bf16 vector struct block_load/write
DEF_block_load_vec_raw(bf16x8, bf16, intel_sub_group_block_read_us8, ushort8, short8)
DEF_block_load_vec_raw(bf16x4, bf16, intel_sub_group_block_read_us4, ushort4, short4)
DEF_block_write_vec_raw(bf16, bf16x8, intel_sub_group_block_write_us8, ushort8)
DEF_block_write_vec_raw(bf16, bf16x4, intel_sub_group_block_write_us4, ushort4)

#ifdef cl_khr_fp16
// half source (scalar loop: reinterpret ushort as half, then convert to float)
DEF_block_load_vec_loop(float8, half, intel_sub_group_block_read_us8, ushort8, as_half, into_float, 8)
DEF_block_load_vec_loop(float4, half, intel_sub_group_block_read_us4, ushort4, as_half, into_float, 4)
DEF_block_load_vec_loop(float2, half, intel_sub_group_block_read_us2, ushort2, as_half, into_float, 2)

// half destination (scalar loop: convert float to half, then reinterpret as ushort)
DEF_block_write_vec_loop(half, float8, intel_sub_group_block_write_us8, ushort8, into_half, 8)

#ifdef MATH_UTILS_DECLARE_HF8
// f8_e4m3 block_load/write (scalar loop)
DEF_block_load_vec_loop(float8, f8_e4m3, intel_sub_group_block_read_uc8, uchar8, as_f8_e4m3, into_float, 8)
DEF_block_load_vec_loop(float4, f8_e4m3, intel_sub_group_block_read_uc4, uchar4, as_f8_e4m3, into_float, 4)
DEF_block_load_vec_loop(float2, f8_e4m3, intel_sub_group_block_read_uc2, uchar2, as_f8_e4m3, into_float, 2)
DEF_block_write_vec_loop(f8_e4m3, float8, intel_sub_group_block_write_uc8, uchar8, into_f8_e4m3, 8)
DEF_block_write_vec_loop(f8_e4m3, float4, intel_sub_group_block_write_uc4, uchar4, into_f8_e4m3, 4)

// Raw f8_e4m3 vector struct block_load/write
DEF_block_load_vec_raw(f8_e4m3x8, f8_e4m3, intel_sub_group_block_read_uc8, uchar8, char8)
DEF_block_write_vec_raw(f8_e4m3, f8_e4m3x8, intel_sub_group_block_write_uc8, uchar8)
#endif // MATH_UTILS_DECLARE_HF8

#ifdef MATH_UTILS_DECLARE_BF8
// f8_e5m2 block_load/write (scalar loop)
DEF_block_load_vec_loop(float8, f8_e5m2, intel_sub_group_block_read_uc8, uchar8, as_f8_e5m2, into_float, 8)
DEF_block_load_vec_loop(float2, f8_e5m2, intel_sub_group_block_read_uc2, uchar2, as_f8_e5m2, into_float, 2)
DEF_block_write_vec_loop(f8_e5m2, float8, intel_sub_group_block_write_uc8, uchar8, into_f8_e5m2, 8)

// Raw f8_e5m2 vector struct block_load/write
DEF_block_load_vec_raw(f8_e5m2x8, f8_e5m2, intel_sub_group_block_read_uc8, uchar8, char8)
DEF_block_write_vec_raw(f8_e5m2, f8_e5m2x8, intel_sub_group_block_write_uc8, uchar8)
#endif // MATH_UTILS_DECLARE_BF8
#endif // cl_khr_fp16

#endif
