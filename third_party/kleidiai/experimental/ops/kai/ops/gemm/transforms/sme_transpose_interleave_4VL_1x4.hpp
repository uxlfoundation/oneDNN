//
// SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

#ifdef __aarch64__

namespace {

void sme_transpose_interleave_4VL_1x4(uint8_t *out, const uint8_t *in, size_t width, size_t in_stride, size_t height)
{
    uint8_t *pad_row = reinterpret_cast<uint8_t *>(alloca(width * sizeof(uint8_t)));

    if (height % 4) {
        memset(pad_row, 0, width * sizeof(uint8_t));
    }

    size_t out_stride = 4 * roundup<size_t>(height, 4) * sme::get_vector_length<uint32_t>();

    __asm__ __volatile__(
      ".inst 0xd503477f  // SMSTART ZA\n"
      "mov x26, %x[height]\n"
      "ptrue p1.b\n"
      "cbz %x[height], 4f\n"
      "1:"  // Main row loop: Head
      "mov x25, %x[in]\n"
      "cmp x26, #0x3\n"
      "add x24, x25, %x[in_stride]\n"
      "mov x23, %x[out]\n"
      "add x22, x24, %x[in_stride]\n"
      "mov x21, %x[width]\n"
      "add x20, x22, %x[in_stride]\n"
      "add %x[in], x20, %x[in_stride]\n"
      "csel %x[in], %x[in], x20, GT\n"
      "csel x20, x20, %x[pad_row], GT\n"
      "csel %x[in], %x[in], x22, GE\n"
      "csel x22, x22, %x[pad_row], GE\n"
      "cmp x26, #0x1\n"
      "sub x26, x26, #0x4\n"
      "csel %x[in], %x[in], x24, GT\n"
      "csel x24, x24, %x[pad_row], GT\n"
      "2:"  // Main row loop: Column loop
      "whilelt p0.b, XZR, x21\n"
      "decw x21, ALL, MUL #4\n"
      "ld1b { z20.b }, p0/Z, [x25]\n"
      "cmp x21, #0\n"
      "addvl x25, x25, #1\n"
      "ld1b { z19.b }, p0/Z, [x24]\n"
      "addvl x24, x24, #1\n"
      "ld1b { z17.b }, p0/Z, [x22]\n"
      "addvl x22, x22, #1\n"
      "ld1b { z16.b }, p0/Z, [x20]\n"
      "addvl x20, x20, #1\n"
      "zip1 z18.b, z20.b, z17.b\n"
      "zip2 z20.b, z20.b, z17.b\n"
      "zip1 z17.b, z19.b, z16.b\n"
      "zip2 z16.b, z19.b, z16.b\n"
      "zip1 z19.b, z18.b, z17.b\n"
      "zip2 z18.b, z18.b, z17.b\n"
      "zip1 z17.b, z20.b, z16.b\n"
      "zip2 z16.b, z20.b, z16.b\n"
      "st1b { z19.b }, p1, [x23]\n"
      "st1b { z18.b }, p1, [x23, #1, MUL VL]\n"
      "st1b { z17.b }, p1, [x23, #2, MUL VL]\n"
      "st1b { z16.b }, p1, [x23, #3, MUL VL]\n"
      "add x23, x23, %x[out_stride]\n"
      "bgt 2b\n"
      "cmp x26, #0x1\n"
      "addvl %x[out], %x[out], #4\n"
      "bge 1b\n"
      "4:"  // Done
      ".inst 0xd503467f  // SMSTOP\n"
      : [in] "+&r" (in), [out] "+&r" (out)
      : [height] "r" (height), [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [pad_row] "r" (pad_row), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

} // anonymous namespace

template<>
void Transform<4, 4, true, VLType::SME>(
    uint8_t *out, const uint8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sme_transpose_interleave_4VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(uint8_t) / 1,
        stride * sizeof(uint8_t),
        (kmax-k0)
    );
}

template<>
void Transform<4, 4, true, VLType::SME>(
    int8_t *out, const int8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sme_transpose_interleave_4VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(int8_t) / 1,
        stride * sizeof(int8_t),
        (kmax-k0)
    );
}

#endif // __aarch64__
