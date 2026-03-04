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

#define BLOCK_READ_FUNC_1 intel_sub_group_block_read_uc
#define BLOCK_READ_FUNC_2 intel_sub_group_block_read_us
#define BLOCK_READ_FUNC_4 intel_sub_group_block_read
#define BLOCK_READ_FUNC_8 intel_sub_group_block_read_ul

#define BLOCK_WRITE_FUNC_1 intel_sub_group_block_write_uc
#define BLOCK_WRITE_FUNC_2 intel_sub_group_block_write_us
#define BLOCK_WRITE_FUNC_4 intel_sub_group_block_write
#define BLOCK_WRITE_FUNC_8 intel_sub_group_block_write_ul
// _0 variants: sub-byte types must not reach block_load/block_write.
// Expanding these produces an undefined identifier, causing a compile error.
#define BLOCK_READ_FUNC_0 \
    BLOCK_READ_WRITE_FUNC_0_is_not_supported_for_sub_byte_types
#define BLOCK_WRITE_FUNC_0 \
    BLOCK_READ_WRITE_FUNC_0_is_not_supported_for_sub_byte_types

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
// 4-bit types: size is 0 because they are accessed via GET_HALF_BYTE.
#define SIZE_s4 0
#define SIZE_u4 0
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

// as_block_data: reinterpret scalar/vector as block I/O data type

#define DECLARE_AS_BLOCK(t) \
    GUARD(t) \
    (BLOCK_DT(t) __attribute__((overloadable)) as_block_data( \
            t a) { return CONCAT2(as_, BLOCK_DT(t))(a); })

#define DECLARE_AS_STRUCT_BLOCK(t) \
    GUARD(t) \
    (BLOCK_DT(t) __attribute__((overloadable)) as_block_data( \
            t a) { return CONCAT2(as_, BLOCK_DT(t))(a.data); })

DECLARE_AS_BLOCK(char);
DECLARE_AS_BLOCK(uchar);
DECLARE_AS_STRUCT_BLOCK(f8_e5m2);
DECLARE_AS_STRUCT_BLOCK(f8_e4m3);
DECLARE_AS_STRUCT_BLOCK(f4_e2m1);
DECLARE_AS_STRUCT_BLOCK(f4_e3m0);
DECLARE_AS_STRUCT_BLOCK(e8m0);
DECLARE_AS_STRUCT_BLOCK(bf16);
DECLARE_AS_BLOCK(half);
DECLARE_AS_BLOCK(int);
DECLARE_AS_BLOCK(float);
DECLARE_AS_BLOCK(double);

// Vector struct as_block_data
#define DECLARE_AS_STRUCT_BLOCK_VEC_impl(t, n) \
    BLOCK_DT_N(t, n) \
    __attribute__((overloadable)) as_block_data(CONCAT2(t, CONCAT2(x, n)) a) { \
        return CONCAT2(as_, BLOCK_DT_N(t, n))(a.data); \
    }

#define DECLARE_AS_STRUCT_BLOCK_VECS(t) \
    GUARD(t) \
    (DECLARE_AS_STRUCT_BLOCK_VEC_impl(t, 8) DECLARE_AS_STRUCT_BLOCK_VEC_impl( \
            t, 4) DECLARE_AS_STRUCT_BLOCK_VEC_impl(t, 2))

DECLARE_AS_STRUCT_BLOCK_VECS(f8_e5m2);
DECLARE_AS_STRUCT_BLOCK_VECS(f8_e4m3);
DECLARE_AS_STRUCT_BLOCK_VECS(f4_e2m1);
DECLARE_AS_STRUCT_BLOCK_VECS(f4_e3m0);
DECLARE_AS_STRUCT_BLOCK_VECS(bf16);

// Vector native as_block_data
#define DECLARE_AS_BLOCK_VEC(t, n) \
    CONCAT2(BLOCK_DT(t), n) \
    __attribute__((overloadable)) as_block_data(CONCAT2(t, n) a) { \
        return CONCAT2(as_, CONCAT2(BLOCK_DT(t), n))(a); \
    }

DECLARE_AS_BLOCK_VEC(char, 2);
DECLARE_AS_BLOCK_VEC(char, 4);
DECLARE_AS_BLOCK_VEC(char, 8);
DECLARE_AS_BLOCK_VEC(uchar, 2);
DECLARE_AS_BLOCK_VEC(uchar, 4);
DECLARE_AS_BLOCK_VEC(uchar, 8);
GUARD_half(DECLARE_AS_BLOCK_VEC(half, 2));
GUARD_half(DECLARE_AS_BLOCK_VEC(half, 4));
GUARD_half(DECLARE_AS_BLOCK_VEC(half, 8));
DECLARE_AS_BLOCK_VEC(int, 2);
DECLARE_AS_BLOCK_VEC(int, 4);
DECLARE_AS_BLOCK_VEC(int, 8);
DECLARE_AS_BLOCK_VEC(float, 2);
DECLARE_AS_BLOCK_VEC(float, 4);
DECLARE_AS_BLOCK_VEC(float, 8);
GUARD_double(DECLARE_AS_BLOCK_VEC(double, 2));
GUARD_double(DECLARE_AS_BLOCK_VEC(double, 4));
GUARD_double(DECLARE_AS_BLOCK_VEC(double, 8));

// Struct vector reinterpret (block data -> struct vec)
#define DECLARE_AS_STRUCT_VEC(t, n, raw_vec_t) \
    GUARD(t) \
    (CONCAT2(t, CONCAT2(x, n)) __attribute__((overloadable)) CONCAT2( \
            as_, CONCAT2(t, CONCAT2(x, n)))(BLOCK_DT_N(t, n) a) { \
        CONCAT2(t, CONCAT2(x, n)) res; \
        res.data = CONCAT2(as_, raw_vec_t)(a); \
        return res; \
    })

DECLARE_AS_STRUCT_VEC(f8_e5m2, 2, char2);
DECLARE_AS_STRUCT_VEC(f8_e5m2, 4, char4);
DECLARE_AS_STRUCT_VEC(f8_e5m2, 8, char8);
DECLARE_AS_STRUCT_VEC(f8_e4m3, 2, char2);
DECLARE_AS_STRUCT_VEC(f8_e4m3, 4, char4);
DECLARE_AS_STRUCT_VEC(f8_e4m3, 8, char8);
DECLARE_AS_STRUCT_VEC(bf16, 2, short2);
DECLARE_AS_STRUCT_VEC(bf16, 4, short4);
DECLARE_AS_STRUCT_VEC(bf16, 8, short8);

#undef DECLARE_AS_BLOCK
#undef DECLARE_AS_STRUCT_BLOCK
#undef DECLARE_AS_STRUCT_BLOCK_VEC_impl
#undef DECLARE_AS_STRUCT_BLOCK_VECS
#undef DECLARE_AS_BLOCK_VEC
#undef DECLARE_AS_STRUCT_VEC

// Helpers for the 8→4→2 block size decomposition inside block_load/block_write.
// K: block width (8, 4, 2).  op: while for K=8 (loops), if for K=4,2.
// Requires in scope: dst, src, n (will be decremented), get_max_sub_group_size().
#define BLOCK_LOAD_CASE(K, dst_dt, src_dt, op) \
    op(n >= K) { \
        __global BLOCK_DT(src_dt) *data = (__global BLOCK_DT(src_dt) *)(src); \
        BLOCK_DT_N(src_dt, K) \
        block_val = BLOCK_READ_FUNC_N(src_dt, K)(data); \
        for (int i = 0; i < K; i++) { \
            src_dt src_val = CONCAT2(as_, src_dt)(block_val[i]); \
            dst[i] = CONCAT2(into_, dst_dt)(src_val); \
        } \
        dst += K; \
        src += K * get_max_sub_group_size(); \
        n -= K; \
    }

#define BLOCK_WRITE_CASE(K, dst_dt, src_dt, op) \
    op(n >= K) { \
        BLOCK_DT_N(dst_dt, K) block_val; \
        for (int i = 0; i < K; i++) { \
            dst_dt val = CONCAT2(into_, dst_dt)(src[i]); \
            block_val[i] = as_block_data(val); \
        } \
        __global BLOCK_DT(dst_dt) *data = (__global BLOCK_DT(dst_dt) *)(dst); \
        BLOCK_WRITE_FUNC_N(dst_dt, K)(data, block_val); \
        dst += K * get_max_sub_group_size(); \
        src += K; \
        n -= K; \
    }

//******* Scalar load/write implementation macros *********//

#define DEF_load_impl(dst_dt, src_dt) \
    /* load(&dst, ptr) */ \
    void __attribute__((overloadable)) load( \
            __private dst_dt *dst, __global const src_dt *val) { \
        *dst = CONCAT2(into_, dst_dt)(*val); \
    } \
    /* load(dst, ptr) */ \
    dst_dt __attribute__((overloadable, warn_unused_result)) load( \
            dst_dt dst, __global const src_dt *val) { \
        return CONCAT2(into_, dst_dt)(*val); \
    } \
    /* load(&dst, ptr, off) */ \
    void __attribute__((overloadable)) load( \
            __private dst_dt *dst, __global const src_dt *val, off_t off) { \
        *dst = CONCAT2(into_, dst_dt)(val[off]); \
    } \
    /* load(dst, ptr, off) */ \
    dst_dt __attribute__((overloadable, warn_unused_result)) load( \
            dst_dt dst, __global const src_dt *val, off_t off) { \
        return CONCAT2(into_, dst_dt)(val[off]); \
    } \
    /* load(&dst, &src) */ \
    void __attribute__((overloadable)) load( \
            __private dst_dt *dst, __private const src_dt *val) { \
        *dst = CONCAT2(into_, dst_dt)(*val); \
    } \
    /* load(dst, &src) */ \
    dst_dt __attribute__((overloadable, warn_unused_result)) load( \
            dst_dt dst, __private const src_dt *val) { \
        return CONCAT2(into_, dst_dt)(*val); \
    } \
    /* load(&dst, &src, off) */ \
    void __attribute__((overloadable)) load( \
            __private dst_dt *dst, __private const src_dt *val, off_t off) { \
        *dst = CONCAT2(into_, dst_dt)(val[off]); \
    } \
    /* load(dst, &src, off) */ \
    dst_dt __attribute__((overloadable, warn_unused_result)) load( \
            dst_dt dst, __private const src_dt *val, off_t off) { \
        return CONCAT2(into_, dst_dt)(val[off]); \
    } \
    /* block_load(&dst, ptr, n) */ \
    __attribute__((overloadable)) void block_load( \
            __private dst_dt *dst, __global const src_dt *src, int n) { \
        __attribute__((opencl_unroll_hint)) BLOCK_LOAD_CASE(8, dst_dt, src_dt, \
                while) BLOCK_LOAD_CASE(4, dst_dt, src_dt, if) \
                BLOCK_LOAD_CASE(2, dst_dt, src_dt, if) if (n >= 1) { \
            __global BLOCK_DT(src_dt) *data \
                    = (__global BLOCK_DT(src_dt) *)(src); \
            BLOCK_DT(src_dt) \
            block_val = BLOCK_READ_FUNC(src_dt)(data); \
            src_dt src_val = CONCAT2(as_, src_dt)(block_val); \
            *dst = CONCAT2(into_, dst_dt)(src_val); \
        } \
    } \
    /* block_load(&dst, ptr) */ \
    __attribute__((overloadable)) void block_load( \
            __private dst_dt *dst, __global const src_dt *src) { \
        __global BLOCK_DT(src_dt) *data = (__global BLOCK_DT(src_dt) *)(src); \
        BLOCK_DT(src_dt) block_val = BLOCK_READ_FUNC(src_dt)(data); \
        src_dt src_val = CONCAT2(as_, src_dt)(block_val); \
        *dst = CONCAT2(into_, dst_dt)(src_val); \
    } \
    /* block_load(dst, ptr) */ \
    __attribute__((overloadable, warn_unused_result)) dst_dt block_load( \
            dst_dt dst, __global const src_dt *src) { \
        __global BLOCK_DT(src_dt) *data = (__global BLOCK_DT(src_dt) *)(src); \
        BLOCK_DT(src_dt) block_val = BLOCK_READ_FUNC(src_dt)(data); \
        src_dt src_val = CONCAT2(as_, src_dt)(block_val); \
        return CONCAT2(into_, dst_dt)(src_val); \
    }

#define DEF_load_half_byte_impl(dst_dt, src_dt) \
    /* load(dst, ptr, off) */ \
    dst_dt __attribute__((overloadable, warn_unused_result)) load( \
            dst_dt dst, __global const src_dt *val, off_t off) { \
        src_dt data = get_half_byte(data, (__global const uchar *)val, off); \
        return CONCAT2(into_, dst_dt)(data); \
    } \
    /* load(&dst, ptr, off) */ \
    void __attribute__((overloadable)) load( \
            __private dst_dt *dst, __global const src_dt *val, off_t off) { \
        *dst = load(*dst, val, off); \
    }

// Write-only (no block_write): used for sub-byte types where block_write
// would write more bits than intended.
#define DEF_write_only_impl(dst_dt, src_dt) \
    /* write(ptr, &src) */ \
    void __attribute__((overloadable)) write( \
            __global dst_dt *dst, __private const src_dt *val) { \
        *dst = CONCAT2(into_, dst_dt)(*val); \
    } \
    /* write(ptr, src) */ \
    void __attribute__((overloadable)) write( \
            __global dst_dt *dst, __private src_dt val) { \
        *dst = CONCAT2(into_, dst_dt)(val); \
    } \
    /* write(&dst, &src) */ \
    void __attribute__((overloadable)) write( \
            __private dst_dt *dst, __private const src_dt *val) { \
        *dst = CONCAT2(into_, dst_dt)(*val); \
    } \
    /* write(&dst, src) */ \
    void __attribute__((overloadable)) write( \
            __private dst_dt *dst, __private src_dt val) { \
        *dst = CONCAT2(into_, dst_dt)(val); \
    }

#define DEF_write_impl(dst_dt, src_dt) \
    DEF_write_only_impl(dst_dt, src_dt) /* block_write(ptr, src_ptr, n) */ \
            __attribute__((overloadable)) void \
            block_write(__global dst_dt *dst, __private const src_dt *src, \
                    int n) { \
        __attribute__((opencl_unroll_hint)) BLOCK_WRITE_CASE(8, dst_dt, \
                src_dt, while) BLOCK_WRITE_CASE(4, dst_dt, src_dt, if) \
                BLOCK_WRITE_CASE(2, dst_dt, src_dt, if) if (n >= 1) { \
            BLOCK_DT(dst_dt) block_val; \
            dst_dt val = CONCAT2(into_, dst_dt)(*src); \
            block_val = as_block_data(val); \
            __global BLOCK_DT(dst_dt) *data \
                    = (__global BLOCK_DT(dst_dt) *)(dst); \
            BLOCK_WRITE_FUNC(dst_dt)(data, block_val); \
        } \
    } \
    /* block_write(ptr, src_ptr) */ \
    __attribute__((overloadable)) void block_write( \
            __global dst_dt *dst, __private const src_dt *src) { \
        BLOCK_DT(dst_dt) block_val; \
        dst_dt val = CONCAT2(into_, dst_dt)(*src); \
        block_val = as_block_data(val); \
        __global BLOCK_DT(dst_dt) *data = (__global BLOCK_DT(dst_dt) *)(dst); \
        BLOCK_WRITE_FUNC(dst_dt)(data, block_val); \
    } \
    /* block_write(ptr, src) */ \
    __attribute__((overloadable)) void block_write( \
            __global dst_dt *dst, src_dt src) { \
        block_write(dst, &src); \
    }

// Guarded wrappers: automatically elide when either type is unavailable.
#define DEF_load(dst_dt, src_dt) \
    GUARD(dst_dt)(GUARD(src_dt)(DEF_load_impl(dst_dt, src_dt)))
#define DEF_load_half_byte(dst_dt, src_dt) \
    GUARD(dst_dt)(GUARD(src_dt)(DEF_load_half_byte_impl(dst_dt, src_dt)))
#define DEF_write(dst_dt, src_dt) \
    GUARD(dst_dt)(GUARD(src_dt)(DEF_write_impl(dst_dt, src_dt)))
#define DEF_write_only(dst_dt, src_dt) \
    GUARD(dst_dt)(GUARD(src_dt)(DEF_write_only_impl(dst_dt, src_dt)))

// Loads
DEF_load(char, char);
DEF_load(bf16, bf16);
DEF_load(half, char);
DEF_load(half, uchar);
DEF_load(half, f8_e5m2);
DEF_load(half, f8_e4m3);
DEF_load_half_byte(half, s4);
DEF_load_half_byte(half, u4);
DEF_load_half_byte(half, f4_e2m1);
DEF_load_half_byte(half, f4_e3m0);
DEF_load(half, bf16);
DEF_load(half, half);
DEF_load(half, int);
DEF_load(half, float);
DEF_load(half, double);
DEF_load(int, undef_data);
DEF_load(int, char);
DEF_load(int, uchar);
DEF_load_half_byte(int, s4);
DEF_load_half_byte(int, u4);
DEF_load(int, bf16);
DEF_load(int, int);
DEF_load(float, undef_data);
DEF_load(float, char);
DEF_load(float, uchar);
DEF_load(float, f8_e5m2);
DEF_load(float, f8_e4m3);
DEF_load_half_byte(float, s4);
DEF_load_half_byte(float, u4);
DEF_load_half_byte(float, f4_e2m1);
DEF_load_half_byte(float, f4_e3m0);
DEF_load(float, e8m0);
DEF_load(float, bf16);
DEF_load(float, half);
DEF_load(float, int);
DEF_load(float, float);
DEF_load(float, double);
DEF_load(double, undef_data);
DEF_load(double, char);
DEF_load(double, uchar);
DEF_load(double, f8_e5m2);
DEF_load(double, f8_e4m3);
DEF_load(double, bf16);
DEF_load(double, half);
DEF_load(double, int);
DEF_load(double, float);
DEF_load(double, double);

// Writes
DEF_write(char, char);
DEF_write(char, half);
DEF_write(char, int);
DEF_write(char, float);
DEF_write(char, double);
DEF_write(uchar, half);
DEF_write(uchar, int);
DEF_write(uchar, float);
DEF_write(uchar, double);
DEF_write(f8_e5m2, half);
DEF_write(f8_e5m2, int);
DEF_write(f8_e5m2, float);
DEF_write(f8_e5m2, double);
DEF_write(f8_e4m3, half);
DEF_write(f8_e4m3, int);
DEF_write(f8_e4m3, float);
DEF_write(f8_e4m3, double);
DEF_write_only(f4_e2m1, half);
DEF_write_only(f4_e2m1, int);
DEF_write_only(f4_e2m1, float);
DEF_write_only(f4_e3m0, half);
DEF_write_only(f4_e3m0, int);
DEF_write_only(f4_e3m0, float);
DEF_write(e8m0, float);
DEF_write(bf16, bf16);
DEF_write(bf16, half);
DEF_write(bf16, int);
DEF_write(bf16, float);
DEF_write(bf16, double);
DEF_write(half, half);
DEF_write(half, int);
DEF_write(half, float);
DEF_write(half, double);
DEF_write(int, half);
DEF_write(int, int);
DEF_write(int, float);
DEF_write(int, double);
DEF_write(float, half);
DEF_write(float, int);
DEF_write(float, float);
DEF_write(float, double);
DEF_write(double, half);
DEF_write(double, int);
DEF_write(double, float);
DEF_write(double, double);

#undef BLOCK_LOAD_CASE
#undef BLOCK_WRITE_CASE

//******* Vector block_load / block_write *********//

// VEC(dt, n): construct vector type name from scalar type and width.
// Native types use dt##n (e.g. float8), struct types use dt##x##n (e.g. bf16x8).
#define VEC_float(n) CONCAT2(float, n)
#define VEC_double(n) CONCAT2(double, n)
#define VEC_half(n) CONCAT2(half, n)
#define VEC_char(n) CONCAT2(char, n)
#define VEC_uchar(n) CONCAT2(uchar, n)
#define VEC_int(n) CONCAT2(int, n)
#define VEC_bf16(n) CONCAT2(bf16x, n)
#define VEC_f8_e4m3(n) CONCAT2(f8_e4m3x, n)
#define VEC_f8_e5m2(n) CONCAT2(f8_e5m2x, n)
#define VEC_f4_e2m1(n) CONCAT2(f4_e2m1x, n)
#define VEC_f4_e3m0(n) CONCAT2(f4_e3m0x, n)
#define VEC(dt, n) CONCAT2(VEC_, dt)(n)

#define DEF_block_load_vec(dst_dt, src_dt, n) \
    /* block_load(&dst_vecN, ptr) */ \
    void __attribute__((overloadable)) block_load( \
            __private VEC(dst_dt, n) * dst, __global const src_dt *src) { \
        VEC(src_dt, n) \
        tmp = CONCAT2(as_, VEC(src_dt, n))(BLOCK_READ_FUNC_N(src_dt, n)( \
                (const __global BLOCK_DT(src_dt) *)src)); \
        *dst = CONCAT2(into_, VEC(dst_dt, n))(tmp); \
    } \
    /* block_load(dst_vecN, ptr) */ \
    VEC(dst_dt, n) \
    __attribute__((overloadable, warn_unused_result)) block_load( \
            VEC(dst_dt, n) dst, __global const src_dt *src) { \
        VEC(src_dt, n) \
        tmp = CONCAT2(as_, VEC(src_dt, n))(BLOCK_READ_FUNC_N(src_dt, n)( \
                (const __global BLOCK_DT(src_dt) *)src)); \
        return CONCAT2(into_, VEC(dst_dt, n))(tmp); \
    }

#define DEF_block_write_vec(dst_dt, src_dt, n) \
    /* block_write(ptr, &src_vecN) */ \
    void __attribute__((overloadable)) block_write( \
            __global dst_dt *dst, __private const VEC(src_dt, n) * src) { \
        VEC(dst_dt, n) tmp = CONCAT2(into_, VEC(dst_dt, n))(*src); \
        BLOCK_WRITE_FUNC_N(dst_dt, n) \
        ((__global BLOCK_DT(dst_dt) *)dst, as_block_data(tmp)); \
    } \
    /* block_write(ptr, src_vecN) */ \
    void __attribute__((overloadable)) block_write( \
            __global dst_dt *dst, VEC(src_dt, n) src) { \
        block_write(dst, &src); \
    }

#define DEF_block_load_vecs_impl(dst_dt, src_dt) \
    DEF_block_load_vec(dst_dt, src_dt, 8) DEF_block_load_vec( \
            dst_dt, src_dt, 4) DEF_block_load_vec(dst_dt, src_dt, 2)

#define DEF_block_write_vecs_impl(dst_dt, src_dt) \
    DEF_block_write_vec(dst_dt, src_dt, 8) DEF_block_write_vec( \
            dst_dt, src_dt, 4) DEF_block_write_vec(dst_dt, src_dt, 2)

// Guarded wrappers
#define DEF_block_load_vecs(dst_dt, src_dt) \
    GUARD(dst_dt)(GUARD(src_dt)(DEF_block_load_vecs_impl(dst_dt, src_dt)))
#define DEF_block_write_vecs(dst_dt, src_dt) \
    GUARD(dst_dt)(GUARD(src_dt)(DEF_block_write_vecs_impl(dst_dt, src_dt)))

// Block loads
DEF_block_load_vecs(char, char);
DEF_block_load_vecs(f8_e5m2, f8_e5m2);
DEF_block_load_vecs(f8_e4m3, f8_e4m3);
DEF_block_load_vecs(bf16, bf16);
DEF_block_load_vecs(half, half);
DEF_block_load_vecs(int, int);
DEF_block_load_vecs(float, char);
DEF_block_load_vecs(float, uchar);
DEF_block_load_vecs(float, f8_e5m2);
DEF_block_load_vecs(float, f8_e4m3);
DEF_block_load_vecs(float, bf16);
DEF_block_load_vecs(float, half);
DEF_block_load_vecs(float, int);
DEF_block_load_vecs(float, float);
DEF_block_load_vecs(float, double);
DEF_block_load_vecs(double, char);
DEF_block_load_vecs(double, uchar);
DEF_block_load_vecs(double, bf16);
DEF_block_load_vecs(double, half);
DEF_block_load_vecs(double, int);
DEF_block_load_vecs(double, float);
DEF_block_load_vecs(double, double);

// Block writes
DEF_block_write_vecs(char, char);
DEF_block_write_vecs(char, float);
DEF_block_write_vecs(char, double);
DEF_block_write_vecs(uchar, float);
DEF_block_write_vecs(uchar, double);
DEF_block_write_vecs(f8_e5m2, f8_e5m2);
DEF_block_write_vecs(f8_e5m2, float);
DEF_block_write_vecs(f8_e4m3, f8_e4m3);
DEF_block_write_vecs(f8_e4m3, float);
DEF_block_write_vecs(bf16, bf16);
DEF_block_write_vecs(bf16, float);
DEF_block_write_vecs(bf16, double);
DEF_block_write_vecs(half, half);
DEF_block_write_vecs(half, float);
DEF_block_write_vecs(half, double);
DEF_block_write_vecs(int, int);
DEF_block_write_vecs(int, float);
DEF_block_write_vecs(int, double);
DEF_block_write_vecs(float, float);
DEF_block_write_vecs(float, double);
DEF_block_write_vecs(double, float);
DEF_block_write_vecs(double, double);

#endif
