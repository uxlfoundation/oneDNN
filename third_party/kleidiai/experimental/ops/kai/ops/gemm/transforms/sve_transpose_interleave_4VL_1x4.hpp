//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

#ifdef __aarch64__

namespace {

void sve_transpose_interleave_4VL_1x4(uint8_t *out, const uint8_t *in, size_t width, size_t in_stride, size_t height)
{
    uint8_t *pad_row = reinterpret_cast<uint8_t *>(alloca(width * sizeof(uint8_t)));

    if (height % 4) {
        memset(pad_row, 0, width * sizeof(uint8_t));
    }

    size_t out_stride = 4 * roundup<size_t>(height, 4) * get_vector_length<uint32_t>();

    __asm__ __volatile__(
      "mov x11, %x[height]\n"
      "ptrue p1.b\n"
      "cmp x11, #0x8\n"
      "blt 6f\n"
      "1:"  // Main row loop: Head
      "mov x10, %x[in]\n"
      "mov x9, %x[width]\n"
      "cntb x28, ALL, MUL #2\n"
      "mov x27, %x[out]\n"
      "sub x11, x11, #0x8\n"
      "add x26, x10, %x[in_stride]\n"
      "add x25, x26, %x[in_stride]\n"
      "add x24, x25, %x[in_stride]\n"
      "add x23, x24, %x[in_stride]\n"
      "cmp x9, x28\n"
      "add x22, x23, %x[in_stride]\n"
      "add x21, x22, %x[in_stride]\n"
      "add x20, x21, %x[in_stride]\n"
      "add %x[in], x20, %x[in_stride]\n"
      "blt 3f\n"
      "2:"  // Main row loop: Unroll column loop
      "ld1b { z19.b }, p1/Z, [x10]\n"
      "ld1b { z18.b }, p1/Z, [x26]\n"
      "sub x9, x9, x28\n"
      "ld1b { z17.b }, p1/Z, [x25]\n"
      "ld1b { z16.b }, p1/Z, [x24]\n"
      "cmp x9, x28\n"
      "ld1b { z24.b }, p1/Z, [x23]\n"
      "ld1b { z23.b }, p1/Z, [x22]\n"
      "ld1b { z22.b }, p1/Z, [x21]\n"
      "ld1b { z21.b }, p1/Z, [x20]\n"
      "ld1b { z29.b }, p1/Z, [x10, #1, MUL VL]\n"
      "ld1b { z28.b }, p1/Z, [x26, #1, MUL VL]\n"
      "zip1 z4.b, z19.b, z17.b\n"
      "zip1 z3.b, z18.b, z16.b\n"
      "ld1b { z27.b }, p1/Z, [x25, #1, MUL VL]\n"
      "ld1b { z20.b }, p1/Z, [x24, #1, MUL VL]\n"
      "zip2 z2.b, z19.b, z17.b\n"
      "zip2 z1.b, z18.b, z16.b\n"
      "ld1b { z19.b }, p1/Z, [x23, #1, MUL VL]\n"
      "ld1b { z18.b }, p1/Z, [x22, #1, MUL VL]\n"
      "zip1 z26.b, z24.b, z22.b\n"
      "zip1 z25.b, z23.b, z21.b\n"
      "ld1b { z17.b }, p1/Z, [x21, #1, MUL VL]\n"
      "ld1b { z16.b }, p1/Z, [x20, #1, MUL VL]\n"
      "zip2 z24.b, z24.b, z22.b\n"
      "zip2 z23.b, z23.b, z21.b\n"
      "zip1 z22.b, z29.b, z27.b\n"
      "zip1 z21.b, z28.b, z20.b\n"
      "addvl x10, x10, #2\n"
      "addvl x26, x26, #2\n"
      "zip2 z0.b, z29.b, z27.b\n"
      "zip2 z31.b, z28.b, z20.b\n"
      "addvl x25, x25, #2\n"
      "addvl x24, x24, #2\n"
      "zip1 z30.b, z19.b, z17.b\n"
      "zip1 z29.b, z18.b, z16.b\n"
      "addvl x23, x23, #2\n"
      "addvl x22, x22, #2\n"
      "zip2 z28.b, z19.b, z17.b\n"
      "zip2 z27.b, z18.b, z16.b\n"
      "addvl x21, x21, #2\n"
      "addvl x20, x20, #2\n"
      "zip1 z20.b, z4.b, z3.b\n"
      "zip2 z19.b, z4.b, z3.b\n"
      "zip1 z18.b, z2.b, z1.b\n"
      "zip2 z17.b, z2.b, z1.b\n"
      "zip1 z16.b, z26.b, z25.b\n"
      "zip2 z26.b, z26.b, z25.b\n"
      "zip1 z25.b, z24.b, z23.b\n"
      "zip2 z24.b, z24.b, z23.b\n"
      "st1b { z20.b }, p1, [x27]\n"
      "st1b { z19.b }, p1, [x27, #1, MUL VL]\n"
      "zip1 z23.b, z22.b, z21.b\n"
      "zip2 z22.b, z22.b, z21.b\n"
      "st1b { z18.b }, p1, [x27, #2, MUL VL]\n"
      "zip1 z21.b, z0.b, z31.b\n"
      "zip2 z20.b, z0.b, z31.b\n"
      "st1b { z17.b }, p1, [x27, #3, MUL VL]\n"
      "zip1 z19.b, z30.b, z29.b\n"
      "zip2 z18.b, z30.b, z29.b\n"
      "st1b { z16.b }, p1, [x27, #4, MUL VL]\n"
      "zip1 z17.b, z28.b, z27.b\n"
      "zip2 z16.b, z28.b, z27.b\n"
      "st1b { z26.b }, p1, [x27, #5, MUL VL]\n"
      "st1b { z25.b }, p1, [x27, #6, MUL VL]\n"
      "st1b { z24.b }, p1, [x27, #7, MUL VL]\n"
      "add x27, x27, %x[out_stride]\n"
      "st1b { z23.b }, p1, [x27]\n"
      "st1b { z22.b }, p1, [x27, #1, MUL VL]\n"
      "st1b { z21.b }, p1, [x27, #2, MUL VL]\n"
      "st1b { z20.b }, p1, [x27, #3, MUL VL]\n"
      "st1b { z19.b }, p1, [x27, #4, MUL VL]\n"
      "st1b { z18.b }, p1, [x27, #5, MUL VL]\n"
      "st1b { z17.b }, p1, [x27, #6, MUL VL]\n"
      "st1b { z16.b }, p1, [x27, #7, MUL VL]\n"
      "add x27, x27, %x[out_stride]\n"
      "bge 2b\n"
      "3:"  // Main row loop: Unroll column loop skip
      "cbz x9, 5f\n"
      "4:"  // Main row loop: Column loop
      "whilelt p0.b, XZR, x9\n"
      "decw x9, ALL, MUL #4\n"
      "ld1b { z20.b }, p0/Z, [x10]\n"
      "addvl x10, x10, #1\n"
      "ld1b { z24.b }, p0/Z, [x26]\n"
      "addvl x26, x26, #1\n"
      "ld1b { z19.b }, p0/Z, [x25]\n"
      "addvl x25, x25, #1\n"
      "ld1b { z18.b }, p0/Z, [x24]\n"
      "addvl x24, x24, #1\n"
      "ld1b { z23.b }, p0/Z, [x23]\n"
      "ld1b { z22.b }, p0/Z, [x22]\n"
      "ld1b { z17.b }, p0/Z, [x21]\n"
      "cmp x9, #0\n"
      "addvl x23, x23, #1\n"
      "ld1b { z16.b }, p0/Z, [x20]\n"
      "zip1 z21.b, z20.b, z19.b\n"
      "zip2 z25.b, z20.b, z19.b\n"
      "addvl x22, x22, #1\n"
      "zip1 z20.b, z24.b, z18.b\n"
      "zip2 z19.b, z24.b, z18.b\n"
      "addvl x21, x21, #1\n"
      "addvl x20, x20, #1\n"
      "zip1 z18.b, z23.b, z17.b\n"
      "zip2 z24.b, z23.b, z17.b\n"
      "zip1 z17.b, z22.b, z16.b\n"
      "zip2 z16.b, z22.b, z16.b\n"
      "zip1 z23.b, z21.b, z20.b\n"
      "zip2 z22.b, z21.b, z20.b\n"
      "zip1 z21.b, z25.b, z19.b\n"
      "zip2 z20.b, z25.b, z19.b\n"
      "zip1 z19.b, z18.b, z17.b\n"
      "zip2 z18.b, z18.b, z17.b\n"
      "zip1 z17.b, z24.b, z16.b\n"
      "zip2 z16.b, z24.b, z16.b\n"
      "st1b { z23.b }, p1, [x27]\n"
      "st1b { z22.b }, p1, [x27, #1, MUL VL]\n"
      "st1b { z21.b }, p1, [x27, #2, MUL VL]\n"
      "st1b { z20.b }, p1, [x27, #3, MUL VL]\n"
      "st1b { z19.b }, p1, [x27, #4, MUL VL]\n"
      "st1b { z18.b }, p1, [x27, #5, MUL VL]\n"
      "st1b { z17.b }, p1, [x27, #6, MUL VL]\n"
      "st1b { z16.b }, p1, [x27, #7, MUL VL]\n"
      "add x27, x27, %x[out_stride]\n"
      "bgt 4b\n"
      "5:"  // Main row loop: Column loop skip
      "cmp x11, #0x8\n"
      "addvl %x[out], %x[out], #8\n"
      "bge 1b\n"
      "cbz x11, 12f\n"
      "6:"  // Main loop skip
      "7:"  // Tail row loop: Head
      "mov x10, %x[in]\n"
      "cmp x11, #0x3\n"
      "mov x21, %x[width]\n"
      "cntb x20, ALL, MUL #2\n"
      "mov x27, %x[out]\n"
      "add x26, x10, %x[in_stride]\n"
      "add x25, x26, %x[in_stride]\n"
      "add x24, x25, %x[in_stride]\n"
      "add %x[in], x24, %x[in_stride]\n"
      "csel %x[in], %x[in], x24, GT\n"
      "csel x24, x24, %x[pad_row], GT\n"
      "csel %x[in], %x[in], x25, GE\n"
      "csel x25, x25, %x[pad_row], GE\n"
      "cmp x11, #0x1\n"
      "sub x11, x11, #0x4\n"
      "csel %x[in], %x[in], x26, GT\n"
      "csel x26, x26, %x[pad_row], GT\n"
      "cmp x21, x20\n"
      "blt 9f\n"
      "8:"  // Tail row loop: Unroll column loop
      "ld1b { z20.b }, p1/Z, [x10]\n"
      "ld1b { z24.b }, p1/Z, [x26]\n"
      "sub x21, x21, x20\n"
      "ld1b { z19.b }, p1/Z, [x25]\n"
      "ld1b { z18.b }, p1/Z, [x24]\n"
      "cmp x21, x20\n"
      "ld1b { z23.b }, p1/Z, [x10, #1, MUL VL]\n"
      "ld1b { z25.b }, p1/Z, [x26, #1, MUL VL]\n"
      "addvl x10, x10, #2\n"
      "addvl x26, x26, #2\n"
      "ld1b { z22.b }, p1/Z, [x25, #1, MUL VL]\n"
      "ld1b { z17.b }, p1/Z, [x24, #1, MUL VL]\n"
      "addvl x25, x25, #2\n"
      "addvl x24, x24, #2\n"
      "zip1 z21.b, z20.b, z19.b\n"
      "zip1 z16.b, z24.b, z18.b\n"
      "zip2 z20.b, z20.b, z19.b\n"
      "zip2 z19.b, z24.b, z18.b\n"
      "zip1 z24.b, z23.b, z22.b\n"
      "zip1 z18.b, z25.b, z17.b\n"
      "zip2 z23.b, z23.b, z22.b\n"
      "zip2 z22.b, z25.b, z17.b\n"
      "zip1 z17.b, z21.b, z16.b\n"
      "zip2 z16.b, z21.b, z16.b\n"
      "zip1 z21.b, z20.b, z19.b\n"
      "zip2 z20.b, z20.b, z19.b\n"
      "zip1 z19.b, z24.b, z18.b\n"
      "zip2 z18.b, z24.b, z18.b\n"
      "st1b { z17.b }, p1, [x27]\n"
      "st1b { z16.b }, p1, [x27, #1, MUL VL]\n"
      "zip1 z17.b, z23.b, z22.b\n"
      "zip2 z16.b, z23.b, z22.b\n"
      "st1b { z21.b }, p1, [x27, #2, MUL VL]\n"
      "st1b { z20.b }, p1, [x27, #3, MUL VL]\n"
      "add x27, x27, %x[out_stride]\n"
      "st1b { z19.b }, p1, [x27]\n"
      "st1b { z18.b }, p1, [x27, #1, MUL VL]\n"
      "st1b { z17.b }, p1, [x27, #2, MUL VL]\n"
      "st1b { z16.b }, p1, [x27, #3, MUL VL]\n"
      "add x27, x27, %x[out_stride]\n"
      "bge 8b\n"
      "9:"  // Tail row loop: Unroll column loop skip
      "cbz x21, 11f\n"
      "10:"  // Tail row loop: Column loop
      "whilelt p0.b, XZR, x21\n"
      "decw x21, ALL, MUL #4\n"
      "ld1b { z20.b }, p0/Z, [x10]\n"
      "addvl x10, x10, #1\n"
      "ld1b { z19.b }, p0/Z, [x26]\n"
      "addvl x26, x26, #1\n"
      "ld1b { z17.b }, p0/Z, [x25]\n"
      "addvl x25, x25, #1\n"
      "ld1b { z16.b }, p0/Z, [x24]\n"
      "addvl x24, x24, #1\n"
      "cmp x21, #0\n"
      "zip1 z18.b, z20.b, z17.b\n"
      "zip2 z20.b, z20.b, z17.b\n"
      "zip1 z17.b, z19.b, z16.b\n"
      "zip2 z16.b, z19.b, z16.b\n"
      "zip1 z19.b, z18.b, z17.b\n"
      "zip2 z18.b, z18.b, z17.b\n"
      "zip1 z17.b, z20.b, z16.b\n"
      "zip2 z16.b, z20.b, z16.b\n"
      "st1b { z19.b }, p1, [x27]\n"
      "st1b { z18.b }, p1, [x27, #1, MUL VL]\n"
      "st1b { z17.b }, p1, [x27, #2, MUL VL]\n"
      "st1b { z16.b }, p1, [x27, #3, MUL VL]\n"
      "add x27, x27, %x[out_stride]\n"
      "bgt 10b\n"
      "11:"  // Tail row loop: Column loop skip
      "cmp x11, #0x1\n"
      "addvl %x[out], %x[out], #4\n"
      "bge 7b\n"
      "12:"  // Done
      : [in] "+&r" (in), [out] "+&r" (out)
      : [height] "r" (height), [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [pad_row] "r" (pad_row), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "x9", "x10", "x11", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

} // anonymous namespace

template<>
void Transform<4, 4, true, VLType::SVE>(
    uint8_t *out, const uint8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sve_transpose_interleave_4VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(uint8_t) / 1,
        stride * sizeof(uint8_t),
        (kmax-k0)
    );
}

template<>
void Transform<4, 4, true, VLType::SVE>(
    int8_t *out, const int8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sve_transpose_interleave_4VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(int8_t) / 1,
        stride * sizeof(int8_t),
        (kmax-k0)
    );
}

#endif // __aarch64__
