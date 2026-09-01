//
// SPDX-FileCopyrightText: Copyright 2021, 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

#ifdef __aarch64__

namespace {

void sve_transpose_interleave_6VL_2x4_fp32bf16(bfloat16 *out, const float *in, size_t width, size_t in_stride, size_t height)
{
    float *pad_row = reinterpret_cast<float *>(alloca(width * sizeof(float)));

    if (height % 4) {
        memset(pad_row, 0, width * sizeof(float));
    }

    size_t out_stride = 6 * roundup<size_t>(height, 4) * get_vector_length<uint32_t>();

    __asm__ __volatile__(
      "ptrue p2.b\n"
      "mov x27, %x[height]\n"
      "cbz %x[height], 6f\n"
      "1:"  // Main row loop: Head
      "mov x26, %x[in]\n"
      "cmp x27, #0x3\n"
      "mov x25, %x[width]\n"
      "cnth x20, ALL, MUL #3\n"
      "mov x24, %x[out]\n"
      "add x23, x26, %x[in_stride]\n"
      "add x22, x23, %x[in_stride]\n"
      "add x21, x22, %x[in_stride]\n"
      "add %x[in], x21, %x[in_stride]\n"
      "csel %x[in], %x[in], x21, GT\n"
      "csel x21, x21, %x[pad_row], GT\n"
      "csel %x[in], %x[in], x22, GE\n"
      "csel x22, x22, %x[pad_row], GE\n"
      "cmp x27, #0x1\n"
      "sub x27, x27, #0x4\n"
      "csel %x[in], %x[in], x23, GT\n"
      "csel x23, x23, %x[pad_row], GT\n"
      "cmp x25, x20\n"
      "blt 3f\n"
      "2:"  // Main row loop: Unroll column loop
      "ld1w { z17.s }, p2/Z, [x26]\n"
      "ld1w { z24.s }, p2/Z, [x26, #1, MUL VL]\n"
      "sub x25, x25, x20\n"
      "ld1w { z23.s }, p2/Z, [x26, #2, MUL VL]\n"
      "ld1w { z16.s }, p2/Z, [x22]\n"
      "cmp x25, x20\n"
      "ld1w { z22.s }, p2/Z, [x22, #1, MUL VL]\n"
      "ld1w { z21.s }, p2/Z, [x22, #2, MUL VL]\n"
      "ld1w { z20.s }, p2/Z, [x26, #3, MUL VL]\n"
      "ld1w { z19.s }, p2/Z, [x26, #4, MUL VL]\n"
      "ld1w { z5.s }, p2/Z, [x26, #5, MUL VL]\n"
      "ld1w { z18.s }, p2/Z, [x22, #3, MUL VL]\n"
      "zip1 z4.s, z17.s, z16.s\n"
      "zip2 z3.s, z17.s, z16.s\n"
      "ld1w { z17.s }, p2/Z, [x22, #4, MUL VL]\n"
      "ld1w { z16.s }, p2/Z, [x22, #5, MUL VL]\n"
      "zip1 z2.s, z24.s, z22.s\n"
      "zip2 z1.s, z24.s, z22.s\n"
      "ld1w { z0.s }, p2/Z, [x23]\n"
      "ld1w { z31.s }, p2/Z, [x23, #1, MUL VL]\n"
      "zip1 z30.s, z23.s, z21.s\n"
      "zip2 z29.s, z23.s, z21.s\n"
      "ld1w { z28.s }, p2/Z, [x23, #2, MUL VL]\n"
      "ld1w { z27.s }, p2/Z, [x21]\n"
      "zip1 z26.s, z20.s, z18.s\n"
      "zip2 z25.s, z20.s, z18.s\n"
      "ld1w { z24.s }, p2/Z, [x21, #1, MUL VL]\n"
      "ld1w { z23.s }, p2/Z, [x21, #2, MUL VL]\n"
      "zip1 z22.s, z19.s, z17.s\n"
      "zip2 z10.s, z19.s, z17.s\n"
      "ld1w { z21.s }, p2/Z, [x23, #3, MUL VL]\n"
      "ld1w { z20.s }, p2/Z, [x23, #4, MUL VL]\n"
      "zip1 z19.s, z5.s, z16.s\n"
      "zip2 z9.s, z5.s, z16.s\n"
      "ld1w { z8.s }, p2/Z, [x23, #5, MUL VL]\n"
      "ld1w { z18.s }, p2/Z, [x21, #3, MUL VL]\n"
      ".inst 0x658aa887  // bfcvt z7.h, p2/M, z4.s\n"
      "zip1 z6.s, z0.s, z27.s\n"
      "ld1w { z17.s }, p2/Z, [x21, #4, MUL VL]\n"
      "ld1w { z16.s }, p2/Z, [x21, #5, MUL VL]\n"
      ".inst 0x658aa865  // bfcvt z5.h, p2/M, z3.s\n"
      "zip2 z4.s, z0.s, z27.s\n"
      ".inst 0x658aa843  // bfcvt z3.h, p2/M, z2.s\n"
      "zip1 z2.s, z31.s, z24.s\n"
      "addvl x26, x26, #6\n"
      "addvl x23, x23, #6\n"
      ".inst 0x658aa821  // bfcvt z1.h, p2/M, z1.s\n"
      "zip2 z0.s, z31.s, z24.s\n"
      "addvl x22, x22, #6\n"
      "addvl x21, x21, #6\n"
      ".inst 0x658aabdf  // bfcvt z31.h, p2/M, z30.s\n"
      "zip1 z30.s, z28.s, z23.s\n"
      ".inst 0x658aabbd  // bfcvt z29.h, p2/M, z29.s\n"
      "zip2 z28.s, z28.s, z23.s\n"
      ".inst 0x658aab5b  // bfcvt z27.h, p2/M, z26.s\n"
      "zip1 z26.s, z21.s, z18.s\n"
      ".inst 0x658aab39  // bfcvt z25.h, p2/M, z25.s\n"
      "zip2 z24.s, z21.s, z18.s\n"
      ".inst 0x658aaad7  // bfcvt z23.h, p2/M, z22.s\n"
      "zip1 z22.s, z20.s, z17.s\n"
      ".inst 0x658aa955  // bfcvt z21.h, p2/M, z10.s\n"
      "zip2 z20.s, z20.s, z17.s\n"
      ".inst 0x658aaa73  // bfcvt z19.h, p2/M, z19.s\n"
      "zip1 z18.s, z8.s, z16.s\n"
      ".inst 0x658aa931  // bfcvt z17.h, p2/M, z9.s\n"
      "zip2 z16.s, z8.s, z16.s\n"
      ".inst 0x648aa8c7  // bfcvtnt z7.h, p2/M, z6.s\n"
      ".inst 0x648aa885  // bfcvtnt z5.h, p2/M, z4.s\n"
      ".inst 0x648aa843  // bfcvtnt z3.h, p2/M, z2.s\n"
      ".inst 0x648aa801  // bfcvtnt z1.h, p2/M, z0.s\n"
      ".inst 0x648aabdf  // bfcvtnt z31.h, p2/M, z30.s\n"
      ".inst 0x648aab9d  // bfcvtnt z29.h, p2/M, z28.s\n"
      "st1h { z7.h }, p2, [x24]\n"
      "st1h { z5.h }, p2, [x24, #1, MUL VL]\n"
      ".inst 0x648aab5b  // bfcvtnt z27.h, p2/M, z26.s\n"
      ".inst 0x648aab19  // bfcvtnt z25.h, p2/M, z24.s\n"
      "st1h { z3.h }, p2, [x24, #2, MUL VL]\n"
      ".inst 0x648aaad7  // bfcvtnt z23.h, p2/M, z22.s\n"
      ".inst 0x648aaa95  // bfcvtnt z21.h, p2/M, z20.s\n"
      "st1h { z1.h }, p2, [x24, #3, MUL VL]\n"
      ".inst 0x648aaa53  // bfcvtnt z19.h, p2/M, z18.s\n"
      ".inst 0x648aaa11  // bfcvtnt z17.h, p2/M, z16.s\n"
      "st1h { z31.h }, p2, [x24, #4, MUL VL]\n"
      "st1h { z29.h }, p2, [x24, #5, MUL VL]\n"
      "add x24, x24, %x[out_stride]\n"
      "st1h { z27.h }, p2, [x24]\n"
      "st1h { z25.h }, p2, [x24, #1, MUL VL]\n"
      "st1h { z23.h }, p2, [x24, #2, MUL VL]\n"
      "st1h { z21.h }, p2, [x24, #3, MUL VL]\n"
      "st1h { z19.h }, p2, [x24, #4, MUL VL]\n"
      "st1h { z17.h }, p2, [x24, #5, MUL VL]\n"
      "add x24, x24, %x[out_stride]\n"
      "bge 2b\n"
      "3:"  // Main row loop: Unroll column loop skip
      "cbz x25, 5f\n"
      "4:"  // Main row loop: Column loop
      "mov x20, x25\n"
      "decd x25, ALL, MUL #6\n"
      "whilelt p0.s, XZR, x20\n"
      "decw x20\n"
      "whilelt p1.s, XZR, x20\n"
      "decw x20\n"
      "ld1w { z17.s }, p0/Z, [x26]\n"
      "ld1w { z16.s }, p0/Z, [x22]\n"
      "ld1w { z23.s }, p0/Z, [x23]\n"
      "ld1w { z19.s }, p0/Z, [x21]\n"
      "whilelt p0.s, XZR, x20\n"
      "cmp x25, #0\n"
      "ld1w { z22.s }, p1/Z, [x26, #1, MUL VL]\n"
      "ld1w { z18.s }, p1/Z, [x22, #1, MUL VL]\n"
      "ld1w { z31.s }, p1/Z, [x23, #1, MUL VL]\n"
      "ld1w { z30.s }, p1/Z, [x21, #1, MUL VL]\n"
      "zip1 z21.s, z17.s, z16.s\n"
      "zip2 z17.s, z17.s, z16.s\n"
      "ld1w { z20.s }, p0/Z, [x26, #2, MUL VL]\n"
      "ld1w { z16.s }, p0/Z, [x22, #2, MUL VL]\n"
      "zip1 z29.s, z23.s, z19.s\n"
      "zip2 z28.s, z23.s, z19.s\n"
      "ld1w { z27.s }, p0/Z, [x23, #2, MUL VL]\n"
      "ld1w { z26.s }, p0/Z, [x21, #2, MUL VL]\n"
      "zip1 z19.s, z22.s, z18.s\n"
      "zip2 z18.s, z22.s, z18.s\n"
      ".inst 0x658aaab9  // bfcvt z25.h, p2/M, z21.s\n"
      ".inst 0x658aaa38  // bfcvt z24.h, p2/M, z17.s\n"
      "addvl x26, x26, #3\n"
      "addvl x23, x23, #3\n"
      "zip1 z17.s, z20.s, z16.s\n"
      "zip2 z16.s, z20.s, z16.s\n"
      "addvl x22, x22, #3\n"
      "addvl x21, x21, #3\n"
      ".inst 0x658aaa77  // bfcvt z23.h, p2/M, z19.s\n"
      "zip1 z22.s, z31.s, z30.s\n"
      ".inst 0x658aaa55  // bfcvt z21.h, p2/M, z18.s\n"
      "zip2 z20.s, z31.s, z30.s\n"
      ".inst 0x658aaa33  // bfcvt z19.h, p2/M, z17.s\n"
      "zip1 z18.s, z27.s, z26.s\n"
      ".inst 0x658aaa11  // bfcvt z17.h, p2/M, z16.s\n"
      "zip2 z16.s, z27.s, z26.s\n"
      ".inst 0x648aabb9  // bfcvtnt z25.h, p2/M, z29.s\n"
      ".inst 0x648aab98  // bfcvtnt z24.h, p2/M, z28.s\n"
      ".inst 0x648aaad7  // bfcvtnt z23.h, p2/M, z22.s\n"
      ".inst 0x648aaa95  // bfcvtnt z21.h, p2/M, z20.s\n"
      ".inst 0x648aaa53  // bfcvtnt z19.h, p2/M, z18.s\n"
      ".inst 0x648aaa11  // bfcvtnt z17.h, p2/M, z16.s\n"
      "st1h { z25.h }, p2, [x24]\n"
      "st1h { z24.h }, p2, [x24, #1, MUL VL]\n"
      "st1h { z23.h }, p2, [x24, #2, MUL VL]\n"
      "st1h { z21.h }, p2, [x24, #3, MUL VL]\n"
      "st1h { z19.h }, p2, [x24, #4, MUL VL]\n"
      "st1h { z17.h }, p2, [x24, #5, MUL VL]\n"
      "add x24, x24, %x[out_stride]\n"
      "bgt 4b\n"
      "5:"  // Main row loop: Column loop skip
      "cmp x27, #0x1\n"
      "addvl %x[out], %x[out], #6\n"
      "bge 1b\n"
      "6:"  // Done
      : [in] "+&r" (in), [out] "+&r" (out)
      : [height] "r" (height), [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [pad_row] "r" (pad_row), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

} // anonymous namespace

template<>
void Transform<6, 4, true, VLType::SVE>(
    bfloat16 *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sve_transpose_interleave_6VL_2x4_fp32bf16(
        out,
        in + k0 * stride + x0,
        (xmax-x0),
        stride * sizeof(float),
        (kmax-k0)
    );
}

#endif // __aarch64__
