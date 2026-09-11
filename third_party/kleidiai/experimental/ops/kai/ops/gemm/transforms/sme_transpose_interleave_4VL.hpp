//
// SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

#ifdef __aarch64__

namespace {

void sme_transpose_interleave_4VL(uint16_t *out, const uint16_t *in, size_t width, size_t in_stride, size_t height)
{
    size_t out_stride = 4 * height * sme::get_vector_length<uint8_t>();

    __asm__ __volatile__(
      "mov x28, %x[height]\n"
      ".inst 0xd503477f  // SMSTART ZA\n"
      "cmp x28, #0x4\n"
      "ptrue p4.b\n"
      "blt 4f\n"
      "1:"  // Main row loop: Head
      "mov x27, %x[in]\n"
      "mov x26, %x[out]\n"
      "add x25, x27, %x[in_stride]\n"
      "sub x28, x28, #0x4\n"
      "add x24, x25, %x[in_stride]\n"
      "mov x23, %x[width]\n"
      "add x22, x24, %x[in_stride]\n"
      "add %x[in], x22, %x[in_stride]\n"
      "2:"  // Main row loop: Column loop
      "mov x21, x23\n"
      "mov x20, x26\n"
      "whilelt p3.h, XZR, x21\n"
      "dech x21\n"
      "whilelt p2.h, XZR, x21\n"
      "dech x21\n"
      "ld1h { z31.h }, p3/Z, [x27]\n"
      "whilelt p1.h, XZR, x21\n"
      "dech x21\n"
      "ld1h { z30.h }, p2/Z, [x27, #1, MUL VL]\n"
      "whilelt p0.h, XZR, x21\n"
      "dech x23, ALL, MUL #4\n"
      "ld1h { z29.h }, p1/Z, [x27, #2, MUL VL]\n"
      "ld1h { z28.h }, p0/Z, [x27, #3, MUL VL]\n"
      "cmp x23, #0\n"
      "addvl x27, x27, #4\n"
      "ld1h { z27.h }, p3/Z, [x25]\n"
      "add x26, x26, %x[out_stride]\n"
      "ld1h { z26.h }, p2/Z, [x25, #1, MUL VL]\n"
      "ld1h { z25.h }, p1/Z, [x25, #2, MUL VL]\n"
      "ld1h { z24.h }, p0/Z, [x25, #3, MUL VL]\n"
      "addvl x25, x25, #4\n"
      "ld1h { z23.h }, p3/Z, [x24]\n"
      "ld1h { z22.h }, p2/Z, [x24, #1, MUL VL]\n"
      "ld1h { z21.h }, p1/Z, [x24, #2, MUL VL]\n"
      "ld1h { z20.h }, p0/Z, [x24, #3, MUL VL]\n"
      "addvl x24, x24, #4\n"
      "ld1h { z19.h }, p3/Z, [x22]\n"
      "ld1h { z18.h }, p2/Z, [x22, #1, MUL VL]\n"
      "ld1h { z17.h }, p1/Z, [x22, #2, MUL VL]\n"
      "ld1h { z16.h }, p0/Z, [x22, #3, MUL VL]\n"
      "st1h { z31.h }, p4, [x20]\n"
      "addvl x22, x22, #4\n"
      "st1h { z30.h }, p4, [x20, #1, MUL VL]\n"
      "st1h { z29.h }, p4, [x20, #2, MUL VL]\n"
      "st1h { z28.h }, p4, [x20, #3, MUL VL]\n"
      "st1h { z27.h }, p4, [x20, #4, MUL VL]\n"
      "st1h { z26.h }, p4, [x20, #5, MUL VL]\n"
      "st1h { z25.h }, p4, [x20, #6, MUL VL]\n"
      "st1h { z24.h }, p4, [x20, #7, MUL VL]\n"
      "addvl x20, x20, #16\n"
      "st1h { z23.h }, p4, [x20, #-8, MUL VL]\n"
      "st1h { z22.h }, p4, [x20, #-7, MUL VL]\n"
      "st1h { z21.h }, p4, [x20, #-6, MUL VL]\n"
      "st1h { z20.h }, p4, [x20, #-5, MUL VL]\n"
      "st1h { z19.h }, p4, [x20, #-4, MUL VL]\n"
      "st1h { z18.h }, p4, [x20, #-3, MUL VL]\n"
      "st1h { z17.h }, p4, [x20, #-2, MUL VL]\n"
      "st1h { z16.h }, p4, [x20, #-1, MUL VL]\n"
      "bgt 2b\n"
      "cmp x28, #0x4\n"
      "addvl %x[out], %x[out], #16\n"
      "bge 1b\n"
      "cbz x28, 8f\n"
      "4:"  // Main loop skip
      "5:"  // Tail row loop: Head
      "mov x27, %x[in]\n"
      "mov x26, %x[out]\n"
      "add %x[in], x27, %x[in_stride]\n"
      "sub x28, x28, #0x1\n"
      "mov x21, %x[width]\n"
      "6:"  // Tail row loop: Column loop
      "mov x20, x21\n"
      "dech x21, ALL, MUL #4\n"
      "whilelt p1.h, XZR, x20\n"
      "dech x20\n"
      "whilelt p0.h, XZR, x20\n"
      "dech x20\n"
      "ld1h { z19.h }, p1/Z, [x27]\n"
      "whilelt p1.h, XZR, x20\n"
      "dech x20\n"
      "ld1h { z18.h }, p0/Z, [x27, #1, MUL VL]\n"
      "whilelt p0.h, XZR, x20\n"
      "cmp x21, #0\n"
      "ld1h { z17.h }, p1/Z, [x27, #2, MUL VL]\n"
      "ld1h { z16.h }, p0/Z, [x27, #3, MUL VL]\n"
      "addvl x27, x27, #4\n"
      "st1h { z19.h }, p4, [x26]\n"
      "st1h { z18.h }, p4, [x26, #1, MUL VL]\n"
      "st1h { z17.h }, p4, [x26, #2, MUL VL]\n"
      "st1h { z16.h }, p4, [x26, #3, MUL VL]\n"
      "add x26, x26, %x[out_stride]\n"
      "bgt 6b\n"
      "cmp x28, #0x1\n"
      "addvl %x[out], %x[out], #4\n"
      "bge 5b\n"
      "8:"  // Done
      ".inst 0xd503467f  // SMSTOP\n"
      : [in] "+&r" (in), [out] "+&r" (out)
      : [height] "r" (height), [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

} // anonymous namespace

template<>
void Transform<4, 1, true, VLType::SME>(
    float *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sme_transpose_interleave_4VL(
        reinterpret_cast<uint16_t *>(out),
        reinterpret_cast<const uint16_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(float) / 2,
        stride * sizeof(float),
        (kmax-k0)
    );
}

#endif // __aarch64__
