//
// SPDX-FileCopyrightText: Copyright 2021, 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

#ifdef __aarch64__

namespace {

void sve_transpose_interleave_1VL(uint32_t *out, const uint32_t *in, size_t width, size_t in_stride, size_t height)
{
    size_t out_stride = 1 * height * get_vector_length<uint8_t>();

    __asm__ __volatile__(
      "mov x27, %x[height]\n"
      "ptrue p1.b\n"
      "cmp x27, #0x4\n"
      "blt 6f\n"
      "1:"  // Main row loop: Head
      "mov x26, %x[in]\n"
      "mov x25, %x[width]\n"
      "cntw x24, ALL, MUL #2\n"
      "mov x23, %x[out]\n"
      "sub x27, x27, #0x4\n"
      "add x22, x26, %x[in_stride]\n"
      "add x21, x22, %x[in_stride]\n"
      "add x20, x21, %x[in_stride]\n"
      "cmp x25, x24\n"
      "add %x[in], x20, %x[in_stride]\n"
      "blt 3f\n"
      "2:"  // Main row loop: Unroll column loop
      "sub x25, x25, x24\n"
      "ld1w { z23.s }, p1/Z, [x26]\n"
      "ld1w { z22.s }, p1/Z, [x26, #1, MUL VL]\n"
      "addvl x26, x26, #2\n"
      "ld1w { z21.s }, p1/Z, [x22]\n"
      "ld1w { z20.s }, p1/Z, [x22, #1, MUL VL]\n"
      "cmp x25, x24\n"
      "addvl x22, x22, #2\n"
      "ld1w { z19.s }, p1/Z, [x21]\n"
      "ld1w { z18.s }, p1/Z, [x21, #1, MUL VL]\n"
      "addvl x21, x21, #2\n"
      "ld1w { z17.s }, p1/Z, [x20]\n"
      "ld1w { z16.s }, p1/Z, [x20, #1, MUL VL]\n"
      "st1w { z23.s }, p1, [x23]\n"
      "addvl x20, x20, #2\n"
      "st1w { z21.s }, p1, [x23, #1, MUL VL]\n"
      "st1w { z19.s }, p1, [x23, #2, MUL VL]\n"
      "st1w { z17.s }, p1, [x23, #3, MUL VL]\n"
      "add x23, x23, %x[out_stride]\n"
      "st1w { z22.s }, p1, [x23]\n"
      "st1w { z20.s }, p1, [x23, #1, MUL VL]\n"
      "st1w { z18.s }, p1, [x23, #2, MUL VL]\n"
      "st1w { z16.s }, p1, [x23, #3, MUL VL]\n"
      "add x23, x23, %x[out_stride]\n"
      "bge 2b\n"
      "3:"  // Main row loop: Unroll column loop skip
      "cbz x25, 5f\n"
      "4:"  // Main row loop: Column loop
      "whilelt p0.s, XZR, x25\n"
      "decw x25\n"
      "cmp x25, #0\n"
      "ld1w { z19.s }, p0/Z, [x26]\n"
      "addvl x26, x26, #1\n"
      "ld1w { z18.s }, p0/Z, [x22]\n"
      "addvl x22, x22, #1\n"
      "ld1w { z17.s }, p0/Z, [x21]\n"
      "addvl x21, x21, #1\n"
      "ld1w { z16.s }, p0/Z, [x20]\n"
      "addvl x20, x20, #1\n"
      "st1w { z19.s }, p1, [x23]\n"
      "st1w { z18.s }, p1, [x23, #1, MUL VL]\n"
      "st1w { z17.s }, p1, [x23, #2, MUL VL]\n"
      "st1w { z16.s }, p1, [x23, #3, MUL VL]\n"
      "add x23, x23, %x[out_stride]\n"
      "bgt 4b\n"
      "5:"  // Main row loop: Column loop skip
      "cmp x27, #0x4\n"
      "addvl %x[out], %x[out], #4\n"
      "bge 1b\n"
      "cbz x27, 12f\n"
      "6:"  // Main loop skip
      "7:"  // Tail row loop: Head
      "mov x21, %x[width]\n"
      "cntw x20, ALL, MUL #2\n"
      "mov x26, %x[in]\n"
      "mov x23, %x[out]\n"
      "sub x27, x27, #0x1\n"
      "cmp x21, x20\n"
      "add %x[in], x26, %x[in_stride]\n"
      "blt 9f\n"
      "8:"  // Tail row loop: Unroll column loop
      "sub x21, x21, x20\n"
      "ld1w { z17.s }, p1/Z, [x26]\n"
      "ld1w { z16.s }, p1/Z, [x26, #1, MUL VL]\n"
      "addvl x26, x26, #2\n"
      "cmp x21, x20\n"
      "st1w { z17.s }, p1, [x23]\n"
      "add x23, x23, %x[out_stride]\n"
      "st1w { z16.s }, p1, [x23]\n"
      "add x23, x23, %x[out_stride]\n"
      "bge 8b\n"
      "9:"  // Tail row loop: Unroll column loop skip
      "cbz x21, 11f\n"
      "10:"  // Tail row loop: Column loop
      "whilelt p0.s, XZR, x21\n"
      "decw x21\n"
      "cmp x21, #0\n"
      "ld1w { z16.s }, p0/Z, [x26]\n"
      "addvl x26, x26, #1\n"
      "st1w { z16.s }, p1, [x23]\n"
      "add x23, x23, %x[out_stride]\n"
      "bgt 10b\n"
      "11:"  // Tail row loop: Column loop skip
      "cmp x27, #0x1\n"
      "addvl %x[out], %x[out], #1\n"
      "bge 7b\n"
      "12:"  // Done
      : [in] "+&r" (in), [out] "+&r" (out)
      : [height] "r" (height), [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23"
    );
}

} // anonymous namespace

template<>
void Transform<1, 1, true, VLType::SVE>(
    float *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sve_transpose_interleave_1VL(
        reinterpret_cast<uint32_t *>(out),
        reinterpret_cast<const uint32_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(float) / 4,
        stride * sizeof(float),
        (kmax-k0)
    );
}

#endif // __aarch64__
