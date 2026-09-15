//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

#ifdef __aarch64__

namespace {

void sve_transpose_interleave_2VL(uint16_t *out, const uint16_t *in, size_t width, size_t in_stride, size_t height)
{
    size_t out_stride = 2 * height * get_vector_length<uint8_t>();

    __asm__ __volatile__(
      "mov x27, %x[height]\n"
      "ptrue p2.b\n"
      "cmp x27, #0x4\n"
      "blt 4f\n"
      "1:"  // Main row loop: Head
      "mov x26, %x[in]\n"
      "mov x25, %x[out]\n"
      "sub x27, x27, #0x4\n"
      "mov x24, %x[width]\n"
      "add x23, x26, %x[in_stride]\n"
      "add x22, x23, %x[in_stride]\n"
      "add x21, x22, %x[in_stride]\n"
      "add %x[in], x21, %x[in_stride]\n"
      "2:"  // Main row loop: Column loop
      "mov x20, x24\n"
      "dech x24, ALL, MUL #2\n"
      "whilelt p1.h, XZR, x20\n"
      "dech x20\n"
      "whilelt p0.h, XZR, x20\n"
      "cmp x24, #0\n"
      "ld1h { z23.h }, p1/Z, [x26]\n"
      "ld1h { z22.h }, p1/Z, [x23]\n"
      "ld1h { z21.h }, p1/Z, [x22]\n"
      "ld1h { z20.h }, p1/Z, [x21]\n"
      "ld1h { z19.h }, p0/Z, [x26, #1, MUL VL]\n"
      "addvl x26, x26, #2\n"
      "ld1h { z18.h }, p0/Z, [x23, #1, MUL VL]\n"
      "addvl x23, x23, #2\n"
      "ld1h { z17.h }, p0/Z, [x22, #1, MUL VL]\n"
      "addvl x22, x22, #2\n"
      "ld1h { z16.h }, p0/Z, [x21, #1, MUL VL]\n"
      "addvl x21, x21, #2\n"
      "st1h { z23.h }, p2, [x25]\n"
      "st1h { z19.h }, p2, [x25, #1, MUL VL]\n"
      "st1h { z22.h }, p2, [x25, #2, MUL VL]\n"
      "st1h { z18.h }, p2, [x25, #3, MUL VL]\n"
      "st1h { z21.h }, p2, [x25, #4, MUL VL]\n"
      "st1h { z17.h }, p2, [x25, #5, MUL VL]\n"
      "st1h { z20.h }, p2, [x25, #6, MUL VL]\n"
      "st1h { z16.h }, p2, [x25, #7, MUL VL]\n"
      "add x25, x25, %x[out_stride]\n"
      "bgt 2b\n"
      "cmp x27, #0x4\n"
      "addvl %x[out], %x[out], #8\n"
      "bge 1b\n"
      "cbz x27, 8f\n"
      "4:"  // Main loop skip
      "5:"  // Tail row loop: Head
      "mov x26, %x[in]\n"
      "mov x25, %x[out]\n"
      "sub x27, x27, #0x1\n"
      "mov x21, %x[width]\n"
      "add %x[in], x26, %x[in_stride]\n"
      "6:"  // Tail row loop: Column loop
      "mov x20, x21\n"
      "dech x21, ALL, MUL #2\n"
      "whilelt p1.h, XZR, x20\n"
      "dech x20\n"
      "whilelt p0.h, XZR, x20\n"
      "cmp x21, #0\n"
      "ld1h { z17.h }, p1/Z, [x26]\n"
      "ld1h { z16.h }, p0/Z, [x26, #1, MUL VL]\n"
      "addvl x26, x26, #2\n"
      "st1h { z17.h }, p2, [x25]\n"
      "st1h { z16.h }, p2, [x25, #1, MUL VL]\n"
      "add x25, x25, %x[out_stride]\n"
      "bgt 6b\n"
      "cmp x27, #0x1\n"
      "addvl %x[out], %x[out], #2\n"
      "bge 5b\n"
      "8:"  // Done
      : [in] "+&r" (in), [out] "+&r" (out)
      : [height] "r" (height), [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23"
    );
}

} // anonymous namespace

template<>
void Transform<2, 1, true, VLType::SVE>(
    __fp16 *out, const __fp16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sve_transpose_interleave_2VL(
        reinterpret_cast<uint16_t *>(out),
        reinterpret_cast<const uint16_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(__fp16) / 2,
        stride * sizeof(__fp16),
        (kmax-k0)
    );
}

#endif // __aarch64__
