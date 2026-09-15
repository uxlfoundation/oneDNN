//
// SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

#ifdef __aarch64__

namespace {

void sme_transpose_interleave_16VL_1x4(uint8_t *out, const uint8_t *in, size_t width, size_t in_stride, size_t height)
{
    uint8_t *pad_row = reinterpret_cast<uint8_t *>(alloca(width * sizeof(uint8_t)));

    if (height % 4) {
        memset(pad_row, 0, width * sizeof(uint8_t));
    }

    size_t out_stride = 16 * roundup<size_t>(height, 4) * sme::get_vector_length<uint32_t>();

    __asm__ __volatile__(
      ".inst 0xd503477f  // SMSTART ZA\n"
      "mov x28, %x[height]\n"
      "ptrue p4.b\n"
      "cbz %x[height], 4f\n"
      "1:"  // Main row loop: Head
      "mov x27, %x[in]\n"
      "cmp x28, #0x3\n"
      "add x26, x27, %x[in_stride]\n"
      "mov x25, %x[out]\n"
      "add x24, x26, %x[in_stride]\n"
      "mov x23, %x[width]\n"
      "add x22, x24, %x[in_stride]\n"
      "add %x[in], x22, %x[in_stride]\n"
      "csel %x[in], %x[in], x22, GT\n"
      "csel x22, x22, %x[pad_row], GT\n"
      "csel %x[in], %x[in], x24, GE\n"
      "csel x24, x24, %x[pad_row], GE\n"
      "cmp x28, #0x1\n"
      "sub x28, x28, #0x4\n"
      "csel %x[in], %x[in], x26, GT\n"
      "csel x26, x26, %x[pad_row], GT\n"
      "2:"  // Main row loop: Column loop
      "mov x21, x23\n"
      "mov x20, x25\n"
      "whilelt p3.b, XZR, x21\n"
      "decb x21\n"
      "whilelt p2.b, XZR, x21\n"
      "decb x21\n"
      "ld1b { z21.b }, p3/Z, [x27]\n"
      "whilelt p1.b, XZR, x21\n"
      "decb x21\n"
      "ld1b { z24.b }, p2/Z, [x27, #1, MUL VL]\n"
      "whilelt p0.b, XZR, x21\n"
      "ld1b { z23.b }, p3/Z, [x26]\n"
      "decw x23, ALL, MUL #16\n"
      "ld1b { z20.b }, p2/Z, [x26, #1, MUL VL]\n"
      "cmp x23, #0\n"
      "add x25, x25, %x[out_stride]\n"
      "ld1b { z19.b }, p3/Z, [x24]\n"
      "ld1b { z17.b }, p2/Z, [x24, #1, MUL VL]\n"
      "ld1b { z16.b }, p3/Z, [x22]\n"
      "ld1b { z18.b }, p2/Z, [x22, #1, MUL VL]\n"
      "zip1 z22.b, z21.b, z19.b\n"
      "zip2 z21.b, z21.b, z19.b\n"
      "ld1b { z28.b }, p1/Z, [x27, #2, MUL VL]\n"
      "zip1 z1.b, z24.b, z17.b\n"
      "zip2 z0.b, z24.b, z17.b\n"
      "ld1b { z27.b }, p0/Z, [x27, #3, MUL VL]\n"
      "zip1 z17.b, z23.b, z16.b\n"
      "zip2 z16.b, z23.b, z16.b\n"
      "addvl x27, x27, #4\n"
      "ld1b { z26.b }, p1/Z, [x26, #2, MUL VL]\n"
      "zip1 z31.b, z20.b, z18.b\n"
      "zip2 z30.b, z20.b, z18.b\n"
      "ld1b { z25.b }, p0/Z, [x26, #3, MUL VL]\n"
      "addvl x26, x26, #4\n"
      "ld1b { z20.b }, p1/Z, [x24, #2, MUL VL]\n"
      "ld1b { z19.b }, p0/Z, [x24, #3, MUL VL]\n"
      "zip1 z18.b, z22.b, z17.b\n"
      "zip2 z24.b, z22.b, z17.b\n"
      "addvl x24, x24, #4\n"
      "ld1b { z17.b }, p1/Z, [x22, #2, MUL VL]\n"
      "zip1 z23.b, z21.b, z16.b\n"
      "zip2 z22.b, z21.b, z16.b\n"
      "ld1b { z16.b }, p0/Z, [x22, #3, MUL VL]\n"
      "zip1 z21.b, z28.b, z20.b\n"
      "zip2 z29.b, z28.b, z20.b\n"
      "addvl x22, x22, #4\n"
      "zip1 z28.b, z27.b, z19.b\n"
      "zip2 z27.b, z27.b, z19.b\n"
      "zip1 z20.b, z26.b, z17.b\n"
      "zip2 z19.b, z26.b, z17.b\n"
      "st1b { z18.b }, p4, [x20]\n"
      "zip1 z18.b, z25.b, z16.b\n"
      "zip2 z26.b, z25.b, z16.b\n"
      "st1b { z24.b }, p4, [x20, #1, MUL VL]\n"
      "zip1 z17.b, z1.b, z31.b\n"
      "zip2 z16.b, z1.b, z31.b\n"
      "st1b { z23.b }, p4, [x20, #2, MUL VL]\n"
      "zip1 z25.b, z0.b, z30.b\n"
      "zip2 z24.b, z0.b, z30.b\n"
      "st1b { z22.b }, p4, [x20, #3, MUL VL]\n"
      "zip1 z23.b, z21.b, z20.b\n"
      "zip2 z22.b, z21.b, z20.b\n"
      "zip1 z21.b, z29.b, z19.b\n"
      "zip2 z20.b, z29.b, z19.b\n"
      "st1b { z17.b }, p4, [x20, #4, MUL VL]\n"
      "zip1 z19.b, z28.b, z18.b\n"
      "zip2 z18.b, z28.b, z18.b\n"
      "st1b { z16.b }, p4, [x20, #5, MUL VL]\n"
      "zip1 z17.b, z27.b, z26.b\n"
      "zip2 z16.b, z27.b, z26.b\n"
      "st1b { z25.b }, p4, [x20, #6, MUL VL]\n"
      "st1b { z24.b }, p4, [x20, #7, MUL VL]\n"
      "addvl x20, x20, #16\n"
      "st1b { z23.b }, p4, [x20, #-8, MUL VL]\n"
      "st1b { z22.b }, p4, [x20, #-7, MUL VL]\n"
      "st1b { z21.b }, p4, [x20, #-6, MUL VL]\n"
      "st1b { z20.b }, p4, [x20, #-5, MUL VL]\n"
      "st1b { z19.b }, p4, [x20, #-4, MUL VL]\n"
      "st1b { z18.b }, p4, [x20, #-3, MUL VL]\n"
      "st1b { z17.b }, p4, [x20, #-2, MUL VL]\n"
      "st1b { z16.b }, p4, [x20, #-1, MUL VL]\n"
      "bgt 2b\n"
      "cmp x28, #0x1\n"
      "addvl %x[out], %x[out], #16\n"
      "bge 1b\n"
      "4:"  // Done
      ".inst 0xd503467f  // SMSTOP\n"
      : [in] "+&r" (in), [out] "+&r" (out)
      : [height] "r" (height), [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [pad_row] "r" (pad_row), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

} // anonymous namespace

template<>
void Transform<16, 4, true, VLType::SME>(
    uint8_t *out, const uint8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sme_transpose_interleave_16VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(uint8_t) / 1,
        stride * sizeof(uint8_t),
        (kmax-k0)
    );
}

template<>
void Transform<16, 4, true, VLType::SME>(
    int8_t *out, const int8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sme_transpose_interleave_16VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(int8_t) / 1,
        stride * sizeof(int8_t),
        (kmax-k0)
    );
}

#endif // __aarch64__
