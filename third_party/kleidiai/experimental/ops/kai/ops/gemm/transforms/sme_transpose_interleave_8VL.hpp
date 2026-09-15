//
// SPDX-FileCopyrightText: Copyright 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

#ifdef __aarch64__

namespace {

void sme_transpose_interleave_8VL(uint16_t *out, const uint16_t *in, size_t width, size_t in_stride, size_t height)
{
    size_t out_stride = 8 * height * sme::get_vector_length<uint8_t>();

    __asm__ __volatile__(
      "mov x26, %x[height]\n"
      ".inst 0xd503477f  // SMSTART ZA\n"
      "cmp x26, #0x2\n"
      "ptrue p7.b\n"
      "blt 4f\n"
      "1:"  // Main row loop: Head
      "mov x25, %x[in]\n"
      "mov x24, %x[out]\n"
      "add x23, x25, %x[in_stride]\n"
      "sub x26, x26, #0x2\n"
      "add %x[in], x23, %x[in_stride]\n"
      "mov x22, %x[width]\n"
      "2:"  // Main row loop: Column loop
      "mov x21, x22\n"
      "mov x20, x24\n"
      "whilelt p0.h, XZR, x21\n"
      "dech x21\n"
      "whilelt p6.h, XZR, x21\n"
      "dech x21\n"
      "ld1h { z31.h }, p0/Z, [x25]\n"
      "whilelt p5.h, XZR, x21\n"
      "dech x21\n"
      "ld1h { z30.h }, p6/Z, [x25, #1, MUL VL]\n"
      "whilelt p4.h, XZR, x21\n"
      "dech x21\n"
      "ld1h { z29.h }, p5/Z, [x25, #2, MUL VL]\n"
      "whilelt p3.h, XZR, x21\n"
      "dech x21\n"
      "ld1h { z28.h }, p4/Z, [x25, #3, MUL VL]\n"
      "whilelt p2.h, XZR, x21\n"
      "dech x21\n"
      "ld1h { z27.h }, p3/Z, [x25, #4, MUL VL]\n"
      "whilelt p1.h, XZR, x21\n"
      "dech x21\n"
      "ld1h { z26.h }, p2/Z, [x25, #5, MUL VL]\n"
      "dech x22, ALL, MUL #8\n"
      "ld1h { z25.h }, p1/Z, [x25, #6, MUL VL]\n"
      "add x24, x24, %x[out_stride]\n"
      "ld1h { z24.h }, p0/Z, [x23]\n"
      "whilelt p0.h, XZR, x21\n"
      "cmp x22, #0\n"
      "ld1h { z23.h }, p0/Z, [x25, #7, MUL VL]\n"
      "addvl x25, x25, #8\n"
      "ld1h { z22.h }, p6/Z, [x23, #1, MUL VL]\n"
      "ld1h { z21.h }, p5/Z, [x23, #2, MUL VL]\n"
      "ld1h { z20.h }, p4/Z, [x23, #3, MUL VL]\n"
      "ld1h { z19.h }, p3/Z, [x23, #4, MUL VL]\n"
      "ld1h { z18.h }, p2/Z, [x23, #5, MUL VL]\n"
      "ld1h { z17.h }, p1/Z, [x23, #6, MUL VL]\n"
      "ld1h { z16.h }, p0/Z, [x23, #7, MUL VL]\n"
      "st1h { z31.h }, p7, [x20]\n"
      "addvl x23, x23, #8\n"
      "st1h { z30.h }, p7, [x20, #1, MUL VL]\n"
      "st1h { z29.h }, p7, [x20, #2, MUL VL]\n"
      "st1h { z28.h }, p7, [x20, #3, MUL VL]\n"
      "st1h { z27.h }, p7, [x20, #4, MUL VL]\n"
      "st1h { z26.h }, p7, [x20, #5, MUL VL]\n"
      "st1h { z25.h }, p7, [x20, #6, MUL VL]\n"
      "st1h { z23.h }, p7, [x20, #7, MUL VL]\n"
      "addvl x20, x20, #16\n"
      "st1h { z24.h }, p7, [x20, #-8, MUL VL]\n"
      "st1h { z22.h }, p7, [x20, #-7, MUL VL]\n"
      "st1h { z21.h }, p7, [x20, #-6, MUL VL]\n"
      "st1h { z20.h }, p7, [x20, #-5, MUL VL]\n"
      "st1h { z19.h }, p7, [x20, #-4, MUL VL]\n"
      "st1h { z18.h }, p7, [x20, #-3, MUL VL]\n"
      "st1h { z17.h }, p7, [x20, #-2, MUL VL]\n"
      "st1h { z16.h }, p7, [x20, #-1, MUL VL]\n"
      "bgt 2b\n"
      "cmp x26, #0x2\n"
      "addvl %x[out], %x[out], #16\n"
      "bge 1b\n"
      "cbz x26, 8f\n"
      "4:"  // Main loop skip
      "5:"  // Tail row loop: Head
      "mov x25, %x[in]\n"
      "mov x24, %x[out]\n"
      "add %x[in], x25, %x[in_stride]\n"
      "sub x26, x26, #0x1\n"
      "mov x21, %x[width]\n"
      "6:"  // Tail row loop: Column loop
      "mov x20, x21\n"
      "dech x21, ALL, MUL #8\n"
      "whilelt p1.h, XZR, x20\n"
      "dech x20\n"
      "whilelt p0.h, XZR, x20\n"
      "dech x20\n"
      "ld1h { z23.h }, p1/Z, [x25]\n"
      "whilelt p1.h, XZR, x20\n"
      "dech x20\n"
      "ld1h { z22.h }, p0/Z, [x25, #1, MUL VL]\n"
      "whilelt p0.h, XZR, x20\n"
      "dech x20\n"
      "ld1h { z21.h }, p1/Z, [x25, #2, MUL VL]\n"
      "whilelt p1.h, XZR, x20\n"
      "dech x20\n"
      "ld1h { z20.h }, p0/Z, [x25, #3, MUL VL]\n"
      "whilelt p0.h, XZR, x20\n"
      "dech x20\n"
      "ld1h { z19.h }, p1/Z, [x25, #4, MUL VL]\n"
      "whilelt p1.h, XZR, x20\n"
      "dech x20\n"
      "ld1h { z18.h }, p0/Z, [x25, #5, MUL VL]\n"
      "whilelt p0.h, XZR, x20\n"
      "cmp x21, #0\n"
      "ld1h { z17.h }, p1/Z, [x25, #6, MUL VL]\n"
      "ld1h { z16.h }, p0/Z, [x25, #7, MUL VL]\n"
      "addvl x25, x25, #8\n"
      "st1h { z23.h }, p7, [x24]\n"
      "st1h { z22.h }, p7, [x24, #1, MUL VL]\n"
      "st1h { z21.h }, p7, [x24, #2, MUL VL]\n"
      "st1h { z20.h }, p7, [x24, #3, MUL VL]\n"
      "st1h { z19.h }, p7, [x24, #4, MUL VL]\n"
      "st1h { z18.h }, p7, [x24, #5, MUL VL]\n"
      "st1h { z17.h }, p7, [x24, #6, MUL VL]\n"
      "st1h { z16.h }, p7, [x24, #7, MUL VL]\n"
      "add x24, x24, %x[out_stride]\n"
      "bgt 6b\n"
      "cmp x26, #0x1\n"
      "addvl %x[out], %x[out], #8\n"
      "bge 5b\n"
      "8:"  // Done
      ".inst 0xd503467f  // SMSTOP\n"
      : [in] "+&r" (in), [out] "+&r" (out)
      : [height] "r" (height), [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

} // anonymous namespace

template<>
void Transform<8, 1, true, VLType::SME>(
    float *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sme_transpose_interleave_8VL(
        reinterpret_cast<uint16_t *>(out),
        reinterpret_cast<const uint16_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(float) / 2,
        stride * sizeof(float),
        (kmax-k0)
    );
}

template<>
void Transform<8, 1, true, VLType::SME>(
    __fp16 *out, const __fp16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sme_transpose_interleave_8VL(
        reinterpret_cast<uint16_t *>(out),
        reinterpret_cast<const uint16_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(__fp16) / 2,
        stride * sizeof(__fp16),
        (kmax-k0)
    );
}

#endif // __aarch64__
