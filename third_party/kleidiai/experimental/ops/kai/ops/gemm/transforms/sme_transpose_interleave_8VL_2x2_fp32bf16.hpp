//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

#ifdef __aarch64__

namespace {

void sme_transpose_interleave_8VL_2x2_fp32bf16(bfloat16 *out, const float *in, size_t width, size_t in_stride, size_t height)
{
    float *pad_row = reinterpret_cast<float *>(alloca(width * sizeof(float)));

    if (height % 2) {
        memset(pad_row, 0, width * sizeof(float));
    }

    size_t out_stride = 8 * roundup<size_t>(height, 2) * sme::get_vector_length<uint16_t>();

    __asm__ __volatile__(
      ".inst 0xd503477f  // SMSTART ZA\n"
      "mov x25, %x[height]\n"
      "ptrue p7.b\n"
      "cbz %x[height], 4f\n"
      "1:"  // Main row loop: Head
      "mov x24, %x[in]\n"
      "cmp x25, #0x1\n"
      "add x23, x24, %x[in_stride]\n"
      "mov x22, %x[out]\n"
      "add %x[in], x23, %x[in_stride]\n"
      "sub x25, x25, #0x2\n"
      "csel %x[in], %x[in], x23, GT\n"
      "csel x23, x23, %x[pad_row], GT\n"
      "mov x21, %x[width]\n"
      "2:"  // Main row loop: Column loop
      "mov x20, x21\n"
      "decw x21, ALL, MUL #8\n"
      "whilelt p0.s, XZR, x20\n"
      "decw x20\n"
      "whilelt p6.s, XZR, x20\n"
      "decw x20\n"
      "ld1w { z16.s }, p0/Z, [x24]\n"
      "whilelt p5.s, XZR, x20\n"
      "decw x20\n"
      "ld1w { z18.s }, p6/Z, [x24, #1, MUL VL]\n"
      "whilelt p4.s, XZR, x20\n"
      "decw x20\n"
      "ld1w { z17.s }, p5/Z, [x24, #2, MUL VL]\n"
      "whilelt p3.s, XZR, x20\n"
      "decw x20\n"
      "ld1w { z19.s }, p4/Z, [x24, #3, MUL VL]\n"
      ".inst 0x658abe1b  // bfcvt z27.h, p7/M, z16.s\n"
      "whilelt p2.s, XZR, x20\n"
      "decw x20\n"
      "ld1w { z16.s }, p3/Z, [x24, #4, MUL VL]\n"
      ".inst 0x658abe5a  // bfcvt z26.h, p7/M, z18.s\n"
      "whilelt p1.s, XZR, x20\n"
      "decw x20\n"
      "ld1w { z18.s }, p2/Z, [x24, #5, MUL VL]\n"
      ".inst 0x658abe39  // bfcvt z25.h, p7/M, z17.s\n"
      "ld1w { z17.s }, p1/Z, [x24, #6, MUL VL]\n"
      ".inst 0x658abe78  // bfcvt z24.h, p7/M, z19.s\n"
      ".inst 0x658abe17  // bfcvt z23.h, p7/M, z16.s\n"
      "ld1w { z19.s }, p0/Z, [x23]\n"
      "whilelt p0.s, XZR, x20\n"
      "cmp x21, #0\n"
      "ld1w { z16.s }, p0/Z, [x24, #7, MUL VL]\n"
      ".inst 0x658abe56  // bfcvt z22.h, p7/M, z18.s\n"
      "addvl x24, x24, #8\n"
      ".inst 0x658abe35  // bfcvt z21.h, p7/M, z17.s\n"
      "ld1w { z18.s }, p6/Z, [x23, #1, MUL VL]\n"
      "ld1w { z17.s }, p5/Z, [x23, #2, MUL VL]\n"
      ".inst 0x648abe7b  // bfcvtnt z27.h, p7/M, z19.s\n"
      ".inst 0x658abe14  // bfcvt z20.h, p7/M, z16.s\n"
      "ld1w { z16.s }, p4/Z, [x23, #3, MUL VL]\n"
      "ld1w { z19.s }, p3/Z, [x23, #4, MUL VL]\n"
      ".inst 0x648abe5a  // bfcvtnt z26.h, p7/M, z18.s\n"
      "ld1w { z18.s }, p2/Z, [x23, #5, MUL VL]\n"
      ".inst 0x648abe39  // bfcvtnt z25.h, p7/M, z17.s\n"
      "ld1w { z17.s }, p1/Z, [x23, #6, MUL VL]\n"
      ".inst 0x648abe18  // bfcvtnt z24.h, p7/M, z16.s\n"
      "ld1w { z16.s }, p0/Z, [x23, #7, MUL VL]\n"
      "addvl x23, x23, #8\n"
      ".inst 0x648abe77  // bfcvtnt z23.h, p7/M, z19.s\n"
      "st1h { z27.h }, p7, [x22]\n"
      ".inst 0x648abe56  // bfcvtnt z22.h, p7/M, z18.s\n"
      "st1h { z26.h }, p7, [x22, #1, MUL VL]\n"
      ".inst 0x648abe35  // bfcvtnt z21.h, p7/M, z17.s\n"
      "st1h { z25.h }, p7, [x22, #2, MUL VL]\n"
      ".inst 0x648abe14  // bfcvtnt z20.h, p7/M, z16.s\n"
      "st1h { z24.h }, p7, [x22, #3, MUL VL]\n"
      "st1h { z23.h }, p7, [x22, #4, MUL VL]\n"
      "st1h { z22.h }, p7, [x22, #5, MUL VL]\n"
      "st1h { z21.h }, p7, [x22, #6, MUL VL]\n"
      "st1h { z20.h }, p7, [x22, #7, MUL VL]\n"
      "add x22, x22, %x[out_stride]\n"
      "bgt 2b\n"
      "cmp x25, #0x1\n"
      "addvl %x[out], %x[out], #8\n"
      "bge 1b\n"
      "4:"  // Done
      ".inst 0xd503467f  // SMSTOP\n"
      : [in] "+&r" (in), [out] "+&r" (out)
      : [height] "r" (height), [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [pad_row] "r" (pad_row), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x20", "x21", "x22", "x23", "x24", "x25", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

} // anonymous namespace

template<>
void Transform<8, 2, true, VLType::SME>(
    bfloat16 *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    sme_transpose_interleave_8VL_2x2_fp32bf16(
        out,
        in + k0 * stride + x0,
        (xmax-x0),
        stride * sizeof(float),
        (kmax-k0)
    );
}

#endif // __aarch64__
