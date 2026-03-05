/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef GPU_INTEL_CONCAT_COMMON_H
#define GPU_INTEL_CONCAT_COMMON_H

#include "gpu/intel/include/io.h"

#define REDUCE_STAGE_0(cat, f)
#define REDUCE_STAGE_1(cat, f) f(0)
#define REDUCE_STAGE_2(cat, f) cat(REDUCE_STAGE_1(cat, f), f(1))
#define REDUCE_STAGE_3(cat, f) cat(REDUCE_STAGE_2(cat, f), f(2))
#define REDUCE_STAGE_4(cat, f) cat(REDUCE_STAGE_3(cat, f), f(3))
#define REDUCE_STAGE_5(cat, f) cat(REDUCE_STAGE_4(cat, f), f(4))
#define REDUCE_STAGE_6(cat, f) cat(REDUCE_STAGE_5(cat, f), f(5))
#define REDUCE_STAGE_7(cat, f) cat(REDUCE_STAGE_6(cat, f), f(6))
#define REDUCE_STAGE_8(cat, f) cat(REDUCE_STAGE_7(cat, f), f(7))
#define REDUCE_STAGE_9(cat, f) cat(REDUCE_STAGE_8(cat, f), f(8))
#define REDUCE_STAGE_10(cat, f) cat(REDUCE_STAGE_9(cat, f), f(9))
#define REDUCE_STAGE_11(cat, f) cat(REDUCE_STAGE_10(cat, f), f(10))
#define REDUCE_STAGE_12(cat, f) cat(REDUCE_STAGE_11(cat, f), f(11))
#define REDUCE_STAGE_13(cat, f) cat(REDUCE_STAGE_12(cat, f), f(12))
#define REDUCE_STAGE_14(cat, f) cat(REDUCE_STAGE_13(cat, f), f(13))
#define REDUCE_STAGE_15(cat, f) cat(REDUCE_STAGE_14(cat, f), f(14))
#define REDUCE_STAGE_16(cat, f) cat(REDUCE_STAGE_15(cat, f), f(15))
#define REDUCE_STAGE_17(cat, f) cat(REDUCE_STAGE_16(cat, f), f(16))
#define REDUCE_STAGE_18(cat, f) cat(REDUCE_STAGE_17(cat, f), f(17))
#define REDUCE_STAGE_19(cat, f) cat(REDUCE_STAGE_18(cat, f), f(18))
#define REDUCE_STAGE_20(cat, f) cat(REDUCE_STAGE_19(cat, f), f(19))
#define REDUCE_STAGE_21(cat, f) cat(REDUCE_STAGE_20(cat, f), f(20))
#define REDUCE_STAGE_22(cat, f) cat(REDUCE_STAGE_21(cat, f), f(21))
#define REDUCE_STAGE_23(cat, f) cat(REDUCE_STAGE_22(cat, f), f(22))
#define REDUCE_STAGE_24(cat, f) cat(REDUCE_STAGE_23(cat, f), f(23))
#define REDUCE_STAGE_25(cat, f) cat(REDUCE_STAGE_24(cat, f), f(24))
#define REDUCE_STAGE_26(cat, f) cat(REDUCE_STAGE_25(cat, f), f(25))
#define REDUCE_STAGE_27(cat, f) cat(REDUCE_STAGE_26(cat, f), f(26))
#define REDUCE_STAGE_28(cat, f) cat(REDUCE_STAGE_27(cat, f), f(27))
#define REDUCE_STAGE_29(cat, f) cat(REDUCE_STAGE_28(cat, f), f(28))
#define REDUCE_STAGE_30(cat, f) cat(REDUCE_STAGE_29(cat, f), f(29))
#define REDUCE_STAGE_31(cat, f) cat(REDUCE_STAGE_30(cat, f), f(30))
#define REDUCE_STAGE_32(cat, f) cat(REDUCE_STAGE_31(cat, f), f(31))
#define REDUCE_STAGE_33(cat, f) cat(REDUCE_STAGE_32(cat, f), f(32))
#define REDUCE_STAGE_34(cat, f) cat(REDUCE_STAGE_33(cat, f), f(33))
#define REDUCE_STAGE_35(cat, f) cat(REDUCE_STAGE_34(cat, f), f(34))
#define REDUCE_STAGE_36(cat, f) cat(REDUCE_STAGE_35(cat, f), f(35))
#define REDUCE_STAGE_37(cat, f) cat(REDUCE_STAGE_36(cat, f), f(36))
#define REDUCE_STAGE_38(cat, f) cat(REDUCE_STAGE_37(cat, f), f(37))
#define REDUCE_STAGE_39(cat, f) cat(REDUCE_STAGE_38(cat, f), f(38))
#define REDUCE_STAGE_40(cat, f) cat(REDUCE_STAGE_39(cat, f), f(39))
#define REDUCE_STAGE_41(cat, f) cat(REDUCE_STAGE_40(cat, f), f(40))
#define REDUCE_STAGE_42(cat, f) cat(REDUCE_STAGE_41(cat, f), f(41))
#define REDUCE_STAGE_43(cat, f) cat(REDUCE_STAGE_42(cat, f), f(42))
#define REDUCE_STAGE_44(cat, f) cat(REDUCE_STAGE_43(cat, f), f(43))
#define REDUCE_STAGE_45(cat, f) cat(REDUCE_STAGE_44(cat, f), f(44))
#define REDUCE_STAGE_46(cat, f) cat(REDUCE_STAGE_45(cat, f), f(45))
#define REDUCE_STAGE_47(cat, f) cat(REDUCE_STAGE_46(cat, f), f(46))
#define REDUCE_STAGE_48(cat, f) cat(REDUCE_STAGE_47(cat, f), f(47))
#define REDUCE_STAGE_49(cat, f) cat(REDUCE_STAGE_48(cat, f), f(48))
#define REDUCE_STAGE_50(cat, f) cat(REDUCE_STAGE_49(cat, f), f(49))
#define REDUCE_STAGE_51(cat, f) cat(REDUCE_STAGE_50(cat, f), f(50))
#define REDUCE_STAGE_52(cat, f) cat(REDUCE_STAGE_51(cat, f), f(51))
#define REDUCE_STAGE_53(cat, f) cat(REDUCE_STAGE_52(cat, f), f(52))
#define REDUCE_STAGE_54(cat, f) cat(REDUCE_STAGE_53(cat, f), f(53))
#define REDUCE_STAGE_55(cat, f) cat(REDUCE_STAGE_54(cat, f), f(54))
#define REDUCE_STAGE_56(cat, f) cat(REDUCE_STAGE_55(cat, f), f(55))
#define REDUCE_STAGE_57(cat, f) cat(REDUCE_STAGE_56(cat, f), f(56))
#define REDUCE_STAGE_58(cat, f) cat(REDUCE_STAGE_57(cat, f), f(57))
#define REDUCE_STAGE_59(cat, f) cat(REDUCE_STAGE_58(cat, f), f(58))
#define REDUCE_STAGE_60(cat, f) cat(REDUCE_STAGE_59(cat, f), f(59))
#define REDUCE_STAGE_61(cat, f) cat(REDUCE_STAGE_60(cat, f), f(60))
#define REDUCE_STAGE_62(cat, f) cat(REDUCE_STAGE_61(cat, f), f(61))
#define REDUCE_STAGE_63(cat, f) cat(REDUCE_STAGE_62(cat, f), f(62))
#define REDUCE_STAGE_64(cat, f) cat(REDUCE_STAGE_63(cat, f), f(63))
#define REDUCE2(n, cat, f) REDUCE_STAGE_##n(cat, f)
#define REDUCE(n, cat, f) REDUCE2(n, cat, f)

#define JOIN_COMMA(x, y) x, y
#define CS_PARAM(p0, p1, p2, p3, p4) \
    JOIN_COMMA(p0, JOIN_COMMA(p1, JOIN_COMMA(p2, JOIN_COMMA(p3, p4))))

#define DIV_UP(a, b) (((a) + ((b) - 1)) / (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define JOIN_ELSE(x, y) y else x
#define JOIN_SEMICOLON(x, y) \
    x; \
    y

#ifndef DATA_T

#if DATA_TYPE_SIZE == 8
#define DATA_T ulong
#elif DATA_TYPE_SIZE == 4
#define DATA_T uint
#elif DATA_TYPE_SIZE == 2
#define DATA_T ushort
#elif DATA_TYPE_SIZE == 1
#define DATA_T uchar
#endif

#define DATA2_T CONCAT2(DATA_T, 2)
#define DATA4_T CONCAT2(DATA_T, 4)
#define DATA8_T CONCAT2(DATA_T, 8)
#define DATA16_T CONCAT2(DATA_T, 16)

// Block I/O overloads for raw integer types (no fp64/fp16 dependency).
// Generates scalar + vec2/4/8 block_load and block_write matching the io.h
// interface, using the given intrinsic prefix (e.g. intel_sub_group_block_read_ul).
#define DEF_CONCAT_BLOCK_LOAD_VEC(T, n, rd) \
    void __attribute__((overloadable)) block_load( \
            __private CONCAT2(T, n) * dst, __global const T *src) { \
        *dst = CONCAT2(rd, n)(src); \
    } \
    CONCAT2(T, n) \
    __attribute__((overloadable, warn_unused_result)) block_load( \
            CONCAT2(T, n) dst, __global const T *src) { \
        return CONCAT2(rd, n)(src); \
    }

#define DEF_CONCAT_BLOCK_WRITE_VEC(T, n, wr) \
    void __attribute__((overloadable)) block_write( \
            __global T *dst, CONCAT2(T, n) src) { \
        CONCAT2(wr, n)(dst, src); \
    } \
    void __attribute__((overloadable)) block_write( \
            __global T *dst, __private const CONCAT2(T, n) * src) { \
        CONCAT2(wr, n)(dst, *src); \
    }

// block_write(T*, T) scalar value form is provided by math_utils.h for all
// sizes (uint via pre-existing DECLARE_BLOCK_WRITE; ulong/ushort/uchar added
// alongside it). Only block_load, pointer-form block_write, and vector
// overloads are defined here.
#define DEF_CONCAT_BLOCK_IO(T, rd, wr) \
    void __attribute__((overloadable)) block_load( \
            __private T *dst, __global const T *src) { \
        *dst = rd(src); \
    } \
    T __attribute__((overloadable, warn_unused_result)) block_load( \
            T dst, __global const T *src) { \
        return rd(src); \
    } \
    void __attribute__((overloadable)) block_write( \
            __global T *dst, __private const T *src) { \
        wr(dst, *src); \
    } \
    DEF_CONCAT_BLOCK_LOAD_VEC(T, 2, rd) \
    DEF_CONCAT_BLOCK_LOAD_VEC(T, 4, rd) \
    DEF_CONCAT_BLOCK_LOAD_VEC(T, 8, rd) \
    DEF_CONCAT_BLOCK_WRITE_VEC(T, 2, wr) \
    DEF_CONCAT_BLOCK_WRITE_VEC(T, 4, wr) \
    DEF_CONCAT_BLOCK_WRITE_VEC(T, 8, wr)

#if DATA_TYPE_SIZE == 8
DEF_CONCAT_BLOCK_IO(
        ulong, intel_sub_group_block_read_ul, intel_sub_group_block_write_ul)
#elif DATA_TYPE_SIZE == 4
DEF_CONCAT_BLOCK_IO(
        uint, intel_sub_group_block_read, intel_sub_group_block_write)
#elif DATA_TYPE_SIZE == 2
DEF_CONCAT_BLOCK_IO(
        ushort, intel_sub_group_block_read_us, intel_sub_group_block_write_us)
#elif DATA_TYPE_SIZE == 1
DEF_CONCAT_BLOCK_IO(
        uchar, intel_sub_group_block_read_uc, intel_sub_group_block_write_uc)
#endif

#undef DEF_CONCAT_BLOCK_IO
#undef DEF_CONCAT_BLOCK_LOAD_VEC
#undef DEF_CONCAT_BLOCK_WRITE_VEC

#define DATA1_T DATA_T
#define VECTOR(n) DATA##n##_T v##n[DIV_UP(READ_BLOCK, n)]
typedef union {
    VECTOR(16);
    VECTOR(8);
    VECTOR(4);
    VECTOR(2);
    VECTOR(1);
} buffer_t;
#undef VECTOR
#undef DATA1_T

DATA_T load_vec1(const buffer_t *buf, size_t offset) {
    return buf->v1[offset];
}

DATA2_T load_vec2(const buffer_t *buf, size_t offset) {
    return offset & 0x1
            ? (DATA2_T)(load_vec1(buf, offset), load_vec1(buf, offset + 1))
            : buf->v2[offset / 2];
}

// XXX: Consider handling the special cases
//   offset & 0x1 -> (DATA4_T)(load_vec1(buf, offset),
//          load_vec2(buf, offset + 1), load_vec1(buf, offset + 3))
// (and similar for vec8/16)
DATA4_T load_vec4(const buffer_t *buf, size_t offset) {
    return offset & 0x3
            ? (DATA4_T)(load_vec2(buf, offset), load_vec2(buf, offset + 2))
            : buf->v4[offset / 4];
}

DATA8_T load_vec8(const buffer_t *buf, size_t offset) {
    return offset & 0x7
            ? (DATA8_T)(load_vec4(buf, offset), load_vec4(buf, offset + 4))
            : buf->v8[offset / 8];
}

DATA16_T load_vec16(const buffer_t *buf, size_t offset) {
    return offset & 0xf
            ? (DATA16_T)(load_vec8(buf, offset), load_vec8(buf, offset + 8))
            : buf->v16[offset / 16];
}
#endif

#endif
