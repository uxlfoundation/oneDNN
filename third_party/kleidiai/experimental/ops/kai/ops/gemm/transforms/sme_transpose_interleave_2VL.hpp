//
// SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

#ifdef __aarch64__

namespace {

void sme_transpose_interleave_2VL(uint16_t *out, const uint16_t *in, size_t width, size_t in_stride, size_t height)
{
    size_t out_stride = 2 * height * sme::get_vector_length<uint8_t>();

    __asm__ __volatile__(
      "mov x27, %x[height]\n"
      ".inst 0xd503477f  // SMSTART ZA\n"
      "cmp x27, #0x4\n"
      "ptrue p2.b\n"
      "blt 6f\n"
      "1:"  // Main row loop: Head
      "mov x26, %x[in]\n"
      "mov x25, %x[width]\n"
      "add x24, x26, %x[in_stride]\n"
      "cnth x20, ALL, MUL #4\n"
      "add x23, x24, %x[in_stride]\n"
      "cmp x25, x20\n"
      "add x21, x23, %x[in_stride]\n"
      "mov x22, %x[out]\n"
      "add %x[in], x21, %x[in_stride]\n"
      "sub x27, x27, #0x4\n"
      "blt 3f\n"
      "2:"  // Main row loop: Unroll column loop
      "sub x25, x25, x20\n"
      "ld1h { z31.h }, p2/Z, [x26]\n"
      "ld1h { z30.h }, p2/Z, [x26, #1, MUL VL]\n"
      "cmp x25, x20\n"
      "ld1h { z29.h }, p2/Z, [x26, #2, MUL VL]\n"
      "ld1h { z28.h }, p2/Z, [x26, #3, MUL VL]\n"
      "addvl x26, x26, #4\n"
      "ld1h { z27.h }, p2/Z, [x24]\n"
      "ld1h { z26.h }, p2/Z, [x24, #1, MUL VL]\n"
      "ld1h { z25.h }, p2/Z, [x24, #2, MUL VL]\n"
      "ld1h { z24.h }, p2/Z, [x24, #3, MUL VL]\n"
      "addvl x24, x24, #4\n"
      "ld1h { z23.h }, p2/Z, [x23]\n"
      "ld1h { z22.h }, p2/Z, [x23, #1, MUL VL]\n"
      "ld1h { z21.h }, p2/Z, [x23, #2, MUL VL]\n"
      "ld1h { z20.h }, p2/Z, [x23, #3, MUL VL]\n"
      "addvl x23, x23, #4\n"
      "ld1h { z19.h }, p2/Z, [x21]\n"
      "ld1h { z18.h }, p2/Z, [x21, #1, MUL VL]\n"
      "ld1h { z17.h }, p2/Z, [x21, #2, MUL VL]\n"
      "ld1h { z16.h }, p2/Z, [x21, #3, MUL VL]\n"
      "st1h { z31.h }, p2, [x22]\n"
      "addvl x21, x21, #4\n"
      "st1h { z30.h }, p2, [x22, #1, MUL VL]\n"
      "st1h { z27.h }, p2, [x22, #2, MUL VL]\n"
      "st1h { z26.h }, p2, [x22, #3, MUL VL]\n"
      "st1h { z23.h }, p2, [x22, #4, MUL VL]\n"
      "st1h { z22.h }, p2, [x22, #5, MUL VL]\n"
      "st1h { z19.h }, p2, [x22, #6, MUL VL]\n"
      "st1h { z18.h }, p2, [x22, #7, MUL VL]\n"
      "add x22, x22, %x[out_stride]\n"
      "st1h { z29.h }, p2, [x22]\n"
      "st1h { z28.h }, p2, [x22, #1, MUL VL]\n"
      "st1h { z25.h }, p2, [x22, #2, MUL VL]\n"
      "st1h { z24.h }, p2, [x22, #3, MUL VL]\n"
      "st1h { z21.h }, p2, [x22, #4, MUL VL]\n"
      "st1h { z20.h }, p2, [x22, #5, MUL VL]\n"
      "st1h { z17.h }, p2, [x22, #6, MUL VL]\n"
      "st1h { z16.h }, p2, [x22, #7, MUL VL]\n"
      "add x22, x22, %x[out_stride]\n"
      "bge 2b\n"
      "3:"  // Main row loop: Unroll column loop skip
      "cbz x25, 5f\n"
      "4:"  // Main row loop: Column loop
      "mov x20, x25\n"
      "dech x25, ALL, MUL #2\n"
      "whilelt p1.h, XZR, x20\n"
      "dech x20\n"
      "whilelt p0.h, XZR, x20\n"
      "cmp x25, #0\n"
      "ld1h { z23.h }, p1/Z, [x26]\n"
      "ld1h { z22.h }, p0/Z, [x26, #1, MUL VL]\n"
      "addvl x26, x26, #2\n"
      "ld1h { z21.h }, p1/Z, [x24]\n"
      "ld1h { z20.h }, p0/Z, [x24, #1, MUL VL]\n"
      "addvl x24, x24, #2\n"
      "ld1h { z19.h }, p1/Z, [x23]\n"
      "ld1h { z18.h }, p0/Z, [x23, #1, MUL VL]\n"
      "addvl x23, x23, #2\n"
      "ld1h { z17.h }, p1/Z, [x21]\n"
      "ld1h { z16.h }, p0/Z, [x21, #1, MUL VL]\n"
      "addvl x21, x21, #2\n"
      "st1h { z23.h }, p2, [x22]\n"
      "st1h { z22.h }, p2, [x22, #1, MUL VL]\n"
      "st1h { z21.h }, p2, [x22, #2, MUL VL]\n"
      "st1h { z20.h }, p2, [x22, #3, MUL VL]\n"
      "st1h { z19.h }, p2, [x22, #4, MUL VL]\n"
      "st1h { z18.h }, p2, [x22, #5, MUL VL]\n"
      "st1h { z17.h }, p2, [x22, #6, MUL VL]\n"
      "st1h { z16.h }, p2, [x22, #7, MUL VL]\n"
      "add x22, x22, %x[out_stride]\n"
      "bgt 4b\n"
      "5:"  // Main row loop: Column loop skip
      "cmp x27, #0x4\n"
      "addvl %x[out], %x[out], #8\n"
      "bge 1b\n"
      "cbz x27, 12f\n"
      "6:"  // Main loop skip
      "7:"  // Tail row loop: Head
      "mov x21, %x[width]\n"
      "cnth x20, ALL, MUL #4\n"
      "mov x26, %x[in]\n"
      "cmp x21, x20\n"
      "add %x[in], x26, %x[in_stride]\n"
      "mov x22, %x[out]\n"
      "sub x27, x27, #0x1\n"
      "blt 9f\n"
      "8:"  // Tail row loop: Unroll column loop
      "sub x21, x21, x20\n"
      "ld1h { z19.h }, p2/Z, [x26]\n"
      "ld1h { z18.h }, p2/Z, [x26, #1, MUL VL]\n"
      "cmp x21, x20\n"
      "ld1h { z17.h }, p2/Z, [x26, #2, MUL VL]\n"
      "ld1h { z16.h }, p2/Z, [x26, #3, MUL VL]\n"
      "st1h { z19.h }, p2, [x22]\n"
      "addvl x26, x26, #4\n"
      "st1h { z18.h }, p2, [x22, #1, MUL VL]\n"
      "add x22, x22, %x[out_stride]\n"
      "st1h { z17.h }, p2, [x22]\n"
      "st1h { z16.h }, p2, [x22, #1, MUL VL]\n"
      "add x22, x22, %x[out_stride]\n"
      "bge 8b\n"
      "9:"  // Tail row loop: Unroll column loop skip
      "cbz x21, 11f\n"
      "10:"  // Tail row loop: Column loop
      "mov x20, x21\n"
      "dech x21, ALL, MUL #2\n"
      "whilelt p1.h, XZR, x20\n"
      "dech x20\n"
      "whilelt p0.h, XZR, x20\n"
      "cmp x21, #0\n"
      "ld1h { z17.h }, p1/Z, [x26]\n"
      "ld1h { z16.h }, p0/Z, [x26, #1, MUL VL]\n"
      "addvl x26, x26, #2\n"
      "st1h { z17.h }, p2, [x22]\n"
      "st1h { z16.h }, p2, [x22, #1, MUL VL]\n"
      "add x22, x22, %x[out_stride]\n"
      "bgt 10b\n"
      "11:"  // Tail row loop: Column loop skip
      "cmp x27, #0x1\n"
      "addvl %x[out], %x[out], #2\n"
      "bge 7b\n"
      "12:"  // Done
      ".inst 0xd503467f  // SMSTOP\n"
      : [in] "+&r" (in), [out] "+&r" (out)
      : [height] "r" (height), [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

} // anonymous namespace

template<>
void Transform<2, 1, true, VLType::SME>(
    float *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sme_transpose_interleave_2VL(
        reinterpret_cast<uint16_t *>(out),
        reinterpret_cast<const uint16_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(float) / 2,
        stride * sizeof(float),
        (kmax-k0)
    );
}

template<>
void Transform<2, 1, true, VLType::SME>(
    bfloat16 *out, const bfloat16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sme_transpose_interleave_2VL(
        reinterpret_cast<uint16_t *>(out),
        reinterpret_cast<const uint16_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(bfloat16) / 2,
        stride * sizeof(bfloat16),
        (kmax-k0)
    );
}

template<>
void Transform<2, 1, true, VLType::SME>(
    __fp16 *out, const __fp16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sme_transpose_interleave_2VL(
        reinterpret_cast<uint16_t *>(out),
        reinterpret_cast<const uint16_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(__fp16) / 2,
        stride * sizeof(__fp16),
        (kmax-k0)
    );
}

#endif // __aarch64__
