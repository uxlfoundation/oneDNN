//
// SPDX-FileCopyrightText: Copyright 2021, 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

#ifdef __aarch64__

namespace {

void sve_transpose_interleave_6VL_1x8(uint8_t *out, const uint8_t *in, size_t width, size_t in_stride, size_t height)
{
    uint8_t *pad_row = reinterpret_cast<uint8_t *>(alloca(width * sizeof(uint8_t)));

    if (height % 8) {
        memset(pad_row, 0, width * sizeof(uint8_t));
    }

    size_t out_stride = 6 * roundup<size_t>(height, 8) * get_vector_length<uint64_t>();

    __asm__ __volatile__(
      "ptrue p1.b\n"
      "mov x11, %x[height]\n"
      "cbz %x[height], 6f\n"
      "1:"  // Main row loop: Head
      "mov x10, %x[in]\n"
      "cmp x11, #0x7\n"
      "mov x9, %x[width]\n"
      "cntb x28, ALL, MUL #3\n"
      "mov x27, %x[out]\n"
      "add x26, x10, %x[in_stride]\n"
      "add x25, x26, %x[in_stride]\n"
      "add x24, x25, %x[in_stride]\n"
      "add x23, x24, %x[in_stride]\n"
      "add x22, x23, %x[in_stride]\n"
      "add x21, x22, %x[in_stride]\n"
      "add x20, x21, %x[in_stride]\n"
      "add %x[in], x20, %x[in_stride]\n"
      "csel %x[in], %x[in], x20, GT\n"
      "csel x20, x20, %x[pad_row], GT\n"
      "csel %x[in], %x[in], x21, GE\n"
      "csel x21, x21, %x[pad_row], GE\n"
      "cmp x11, #0x5\n"
      "csel %x[in], %x[in], x22, GT\n"
      "csel x22, x22, %x[pad_row], GT\n"
      "csel %x[in], %x[in], x23, GE\n"
      "csel x23, x23, %x[pad_row], GE\n"
      "cmp x11, #0x3\n"
      "csel %x[in], %x[in], x24, GT\n"
      "csel x24, x24, %x[pad_row], GT\n"
      "csel %x[in], %x[in], x25, GE\n"
      "csel x25, x25, %x[pad_row], GE\n"
      "cmp x11, #0x1\n"
      "sub x11, x11, #0x8\n"
      "csel %x[in], %x[in], x26, GT\n"
      "csel x26, x26, %x[pad_row], GT\n"
      "cmp x9, x28\n"
      "blt 3f\n"
      "2:"  // Main row loop: Unroll column loop
      "ld1b { z1.b }, p1/Z, [x10]\n"
      "ld1b { z0.b }, p1/Z, [x26]\n"
      "sub x9, x9, x28\n"
      "ld1b { z31.b }, p1/Z, [x25]\n"
      "ld1b { z28.b }, p1/Z, [x24]\n"
      "cmp x9, x28\n"
      "ld1b { z27.b }, p1/Z, [x23]\n"
      "ld1b { z26.b }, p1/Z, [x22]\n"
      "ld1b { z25.b }, p1/Z, [x21]\n"
      "ld1b { z24.b }, p1/Z, [x20]\n"
      "ld1b { z23.b }, p1/Z, [x10, #1, MUL VL]\n"
      "ld1b { z30.b }, p1/Z, [x26, #1, MUL VL]\n"
      "ld1b { z22.b }, p1/Z, [x25, #1, MUL VL]\n"
      "ld1b { z21.b }, p1/Z, [x24, #1, MUL VL]\n"
      "zip1 z15.b, z1.b, z27.b\n"
      "zip1 z9.b, z0.b, z26.b\n"
      "ld1b { z20.b }, p1/Z, [x23, #1, MUL VL]\n"
      "ld1b { z19.b }, p1/Z, [x22, #1, MUL VL]\n"
      "zip1 z18.b, z31.b, z25.b\n"
      "zip1 z29.b, z28.b, z24.b\n"
      "ld1b { z17.b }, p1/Z, [x21, #1, MUL VL]\n"
      "ld1b { z16.b }, p1/Z, [x20, #1, MUL VL]\n"
      "zip2 z14.b, z1.b, z27.b\n"
      "zip2 z13.b, z31.b, z25.b\n"
      "ld1b { z8.b }, p1/Z, [x10, #2, MUL VL]\n"
      "ld1b { z7.b }, p1/Z, [x26, #2, MUL VL]\n"
      "zip2 z12.b, z0.b, z26.b\n"
      "zip2 z6.b, z28.b, z24.b\n"
      "ld1b { z5.b }, p1/Z, [x25, #2, MUL VL]\n"
      "ld1b { z4.b }, p1/Z, [x24, #2, MUL VL]\n"
      "zip1 z3.b, z23.b, z20.b\n"
      "zip1 z11.b, z30.b, z19.b\n"
      "ld1b { z28.b }, p1/Z, [x23, #2, MUL VL]\n"
      "ld1b { z27.b }, p1/Z, [x22, #2, MUL VL]\n"
      "zip1 z2.b, z22.b, z17.b\n"
      "zip1 z1.b, z21.b, z16.b\n"
      "ld1b { z26.b }, p1/Z, [x21, #2, MUL VL]\n"
      "ld1b { z25.b }, p1/Z, [x20, #2, MUL VL]\n"
      "zip2 z24.b, z23.b, z20.b\n"
      "zip2 z23.b, z22.b, z17.b\n"
      "zip2 z22.b, z30.b, z19.b\n"
      "zip2 z21.b, z21.b, z16.b\n"
      "addvl x10, x10, #3\n"
      "addvl x26, x26, #3\n"
      "zip1 z0.b, z8.b, z28.b\n"
      "zip1 z10.b, z7.b, z27.b\n"
      "addvl x25, x25, #3\n"
      "addvl x24, x24, #3\n"
      "zip1 z31.b, z5.b, z26.b\n"
      "zip1 z30.b, z4.b, z25.b\n"
      "addvl x23, x23, #3\n"
      "addvl x22, x22, #3\n"
      "zip1 z20.b, z15.b, z18.b\n"
      "zip1 z19.b, z9.b, z29.b\n"
      "addvl x21, x21, #3\n"
      "addvl x20, x20, #3\n"
      "zip2 z18.b, z15.b, z18.b\n"
      "zip2 z16.b, z9.b, z29.b\n"
      "zip1 z29.b, z14.b, z13.b\n"
      "zip1 z17.b, z12.b, z6.b\n"
      "zip2 z9.b, z8.b, z28.b\n"
      "zip2 z28.b, z5.b, z26.b\n"
      "zip2 z8.b, z7.b, z27.b\n"
      "zip2 z27.b, z4.b, z25.b\n"
      "zip2 z7.b, z14.b, z13.b\n"
      "zip2 z6.b, z12.b, z6.b\n"
      "zip1 z5.b, z3.b, z2.b\n"
      "zip1 z4.b, z11.b, z1.b\n"
      "zip2 z3.b, z3.b, z2.b\n"
      "zip2 z2.b, z11.b, z1.b\n"
      "zip1 z26.b, z24.b, z23.b\n"
      "zip1 z25.b, z22.b, z21.b\n"
      "zip2 z24.b, z24.b, z23.b\n"
      "zip2 z23.b, z22.b, z21.b\n"
      "zip1 z1.b, z0.b, z31.b\n"
      "zip1 z22.b, z10.b, z30.b\n"
      "zip1 z21.b, z20.b, z19.b\n"
      "zip2 z20.b, z20.b, z19.b\n"
      "zip1 z19.b, z18.b, z16.b\n"
      "zip2 z18.b, z18.b, z16.b\n"
      "zip1 z16.b, z29.b, z17.b\n"
      "zip2 z17.b, z29.b, z17.b\n"
      "st1b { z21.b }, p1, [x27]\n"
      "st1b { z20.b }, p1, [x27, #1, MUL VL]\n"
      "zip2 z0.b, z0.b, z31.b\n"
      "zip2 z31.b, z10.b, z30.b\n"
      "st1b { z19.b }, p1, [x27, #2, MUL VL]\n"
      "zip1 z30.b, z9.b, z28.b\n"
      "zip1 z29.b, z8.b, z27.b\n"
      "st1b { z18.b }, p1, [x27, #3, MUL VL]\n"
      "zip2 z28.b, z9.b, z28.b\n"
      "zip2 z27.b, z8.b, z27.b\n"
      "st1b { z16.b }, p1, [x27, #4, MUL VL]\n"
      "zip1 z21.b, z7.b, z6.b\n"
      "zip2 z16.b, z7.b, z6.b\n"
      "st1b { z17.b }, p1, [x27, #5, MUL VL]\n"
      "add x27, x27, %x[out_stride]\n"
      "zip1 z20.b, z5.b, z4.b\n"
      "zip2 z19.b, z5.b, z4.b\n"
      "zip1 z18.b, z3.b, z2.b\n"
      "zip2 z17.b, z3.b, z2.b\n"
      "st1b { z21.b }, p1, [x27]\n"
      "st1b { z16.b }, p1, [x27, #1, MUL VL]\n"
      "zip1 z16.b, z26.b, z25.b\n"
      "zip2 z26.b, z26.b, z25.b\n"
      "st1b { z20.b }, p1, [x27, #2, MUL VL]\n"
      "zip1 z25.b, z24.b, z23.b\n"
      "zip2 z24.b, z24.b, z23.b\n"
      "st1b { z19.b }, p1, [x27, #3, MUL VL]\n"
      "zip1 z23.b, z1.b, z22.b\n"
      "zip2 z22.b, z1.b, z22.b\n"
      "st1b { z18.b }, p1, [x27, #4, MUL VL]\n"
      "zip1 z21.b, z0.b, z31.b\n"
      "zip2 z20.b, z0.b, z31.b\n"
      "st1b { z17.b }, p1, [x27, #5, MUL VL]\n"
      "add x27, x27, %x[out_stride]\n"
      "zip1 z19.b, z30.b, z29.b\n"
      "zip2 z18.b, z30.b, z29.b\n"
      "st1b { z16.b }, p1, [x27]\n"
      "zip1 z17.b, z28.b, z27.b\n"
      "zip2 z16.b, z28.b, z27.b\n"
      "st1b { z26.b }, p1, [x27, #1, MUL VL]\n"
      "st1b { z25.b }, p1, [x27, #2, MUL VL]\n"
      "st1b { z24.b }, p1, [x27, #3, MUL VL]\n"
      "st1b { z23.b }, p1, [x27, #4, MUL VL]\n"
      "st1b { z22.b }, p1, [x27, #5, MUL VL]\n"
      "add x27, x27, %x[out_stride]\n"
      "st1b { z21.b }, p1, [x27]\n"
      "st1b { z20.b }, p1, [x27, #1, MUL VL]\n"
      "st1b { z19.b }, p1, [x27, #2, MUL VL]\n"
      "st1b { z18.b }, p1, [x27, #3, MUL VL]\n"
      "st1b { z17.b }, p1, [x27, #4, MUL VL]\n"
      "st1b { z16.b }, p1, [x27, #5, MUL VL]\n"
      "add x27, x27, %x[out_stride]\n"
      "bge 2b\n"
      "3:"  // Main row loop: Unroll column loop skip
      "cbz x9, 5f\n"
      "4:"  // Main row loop: Column loop
      "whilelt p0.b, XZR, x9\n"
      "decd x9, ALL, MUL #6\n"
      "ld1b { z21.b }, p0/Z, [x10]\n"
      "incd x10, ALL, MUL #6\n"
      "ld1b { z26.b }, p0/Z, [x26]\n"
      "incd x26, ALL, MUL #6\n"
      "ld1b { z20.b }, p0/Z, [x25]\n"
      "incd x25, ALL, MUL #6\n"
      "ld1b { z25.b }, p0/Z, [x24]\n"
      "incd x24, ALL, MUL #6\n"
      "ld1b { z18.b }, p0/Z, [x23]\n"
      "ld1b { z19.b }, p0/Z, [x22]\n"
      "ld1b { z17.b }, p0/Z, [x21]\n"
      "cmp x9, #0\n"
      "incd x23, ALL, MUL #6\n"
      "ld1b { z16.b }, p0/Z, [x20]\n"
      "incd x22, ALL, MUL #6\n"
      "incd x21, ALL, MUL #6\n"
      "zip1 z23.b, z21.b, z18.b\n"
      "zip2 z24.b, z21.b, z18.b\n"
      "incd x20, ALL, MUL #6\n"
      "zip1 z22.b, z20.b, z17.b\n"
      "zip1 z21.b, z26.b, z19.b\n"
      "zip2 z18.b, z20.b, z17.b\n"
      "zip1 z17.b, z25.b, z16.b\n"
      "zip2 z20.b, z26.b, z19.b\n"
      "zip2 z16.b, z25.b, z16.b\n"
      "zip1 z19.b, z23.b, z22.b\n"
      "zip2 z23.b, z23.b, z22.b\n"
      "zip1 z22.b, z24.b, z18.b\n"
      "zip1 z18.b, z21.b, z17.b\n"
      "zip2 z17.b, z21.b, z17.b\n"
      "zip1 z16.b, z20.b, z16.b\n"
      "zip1 z21.b, z19.b, z18.b\n"
      "zip2 z20.b, z19.b, z18.b\n"
      "zip1 z19.b, z23.b, z17.b\n"
      "zip2 z18.b, z23.b, z17.b\n"
      "zip1 z17.b, z22.b, z16.b\n"
      "zip2 z16.b, z22.b, z16.b\n"
      "st1b { z21.b }, p1, [x27]\n"
      "st1b { z20.b }, p1, [x27, #1, MUL VL]\n"
      "st1b { z19.b }, p1, [x27, #2, MUL VL]\n"
      "st1b { z18.b }, p1, [x27, #3, MUL VL]\n"
      "st1b { z17.b }, p1, [x27, #4, MUL VL]\n"
      "st1b { z16.b }, p1, [x27, #5, MUL VL]\n"
      "add x27, x27, %x[out_stride]\n"
      "bgt 4b\n"
      "5:"  // Main row loop: Column loop skip
      "cmp x11, #0x1\n"
      "addvl %x[out], %x[out], #6\n"
      "bge 1b\n"
      "6:"  // Done
      : [in] "+&r" (in), [out] "+&r" (out)
      : [height] "r" (height), [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [pad_row] "r" (pad_row), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "x9", "x10", "x11", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

} // anonymous namespace

template<>
void Transform<6, 8, true, VLType::SVE>(
    uint8_t *out, const uint8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sve_transpose_interleave_6VL_1x8(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(uint8_t) / 1,
        stride * sizeof(uint8_t),
        (kmax-k0)
    );
}

template<>
void Transform<6, 8, true, VLType::SVE>(
    int8_t *out, const int8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sve_transpose_interleave_6VL_1x8(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(int8_t) / 1,
        stride * sizeof(int8_t),
        (kmax-k0)
    );
}

#endif // __aarch64__
