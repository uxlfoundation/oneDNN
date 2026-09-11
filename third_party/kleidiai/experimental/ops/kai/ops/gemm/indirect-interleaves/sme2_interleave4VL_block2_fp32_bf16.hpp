//
// SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#ifdef __aarch64__

template <>
void interleave_block<4, 2, VLType::SME, false>(
  bfloat16 * &out, const float * const *in,
  size_t width, size_t height, size_t row_offset, bool, int32_t
)
{
  __asm__ __volatile__(
      ".inst 0xd503477f  // SMSTART ZA\n"
      "sub x14, %x[width], #0x1\n"
      "mov x13, %x[in]\n"
      "cntw x23, ALL, MUL #2\n"
      "cntw x11\n"
      "cntw x22, ALL, MUL #2\n"
      "cntw x20, ALL, MUL #3\n"
      "sub x21, x23, #0x1\n"
      ".inst 0x25207817  // ptrue pn15.b\n"
      "whilelt p2.s, XZR, %x[height]\n"
      "whilelt p1.s, x11, %x[height]\n"
      "whilelt p14.s, x22, %x[height]\n"
      "whilelt p13.s, x20, %x[height]\n"
      "cntw x20, ALL, MUL #2\n"
      "ands x10, %x[width], x21\n"
      "add x14, x14, x20\n"
      "csel x10, x10, x23, NE\n"
      "add x9, x13, x11, LSL #3\n"
      "mov x28, #0\n"
      "udiv x14, x14, x20\n"
      "add x10, x10, #0x1\n"
      "mov x20, %x[width]\n"
      "add x27, x9, x11, LSL #3\n"
      "ptrue p0.b\n"
      "mov x26, %x[outptr_raw]\n"
      "mov x25, %x[row_offset]\n"
      "sub x14, x14, #0x1\n"
      "lsr x10, x10, #0x1\n"
      "mov x12, #0\n"
      ".inst 0x25b44794  // whilelt pn12.s, x28, x20, VLx2\n"
      "add x24, x27, x11, LSL #3\n"
      "1:"  // Width loop: Preamble: Loop
      "ldr x23, [x13], #0x8\n"
      ".inst 0x2530704b  // psel p11.s, p12.s/Z, p2.s[w12]\n"
      ".inst 0x2530702a  // psel p10.s, p12.s/Z, p1.s[w12]\n"
      "ldr x22, [x9], #0x8\n"
      ".inst 0x253071c9  // psel p9.s, p12.s/Z, p14.s[w12]\n"
      ".inst 0x253071a8  // psel p8.s, p12.s/Z, p13.s[w12]\n"
      "ldr x21, [x27], #0x8\n"
      "ldr x20, [x24], #0x8\n"
      ".inst 0xa0194eea  // ld1w { z10.s-z11.s }, pn11.s/Z, [x23, x25, LSL #2]\n"
      ".inst 0xa0194ada  // ld1w { z26.s-z27.s }, pn10.s/Z, [x22, x25, LSL #2]\n"
      ".inst 0xa01946be  // ld1w { z30.s-z31.s }, pn9.s/Z, [x21, x25, LSL #2]\n"
      ".inst 0xa019428c  // ld1w { z12.s-z13.s }, pn8.s/Z, [x20, x25, LSL #2]\n"
      ".inst 0xc160e14a  // bfcvt z10.h, { z10.s-z11.s }\n"
      ".inst 0xc160e35a  // bfcvt z26.h, { z26.s-z27.s }\n"
      ".inst 0xc0800140  // mova za0h.s[x12], p0/M, z10.s\n"
      ".inst 0xc160e3de  // bfcvt z30.h, { z30.s-z31.s }\n"
      ".inst 0xc0800344  // mova za1h.s[x12], p0/M, z26.s\n"
      ".inst 0xc160e18c  // bfcvt z12.h, { z12.s-z13.s }\n"
      ".inst 0xc08003c8  // mova za2h.s[x12], p0/M, z30.s\n"
      ".inst 0xc080018c  // mova za3h.s[x12], p0/M, z12.s\n"
      "add x12, x12, #0x1\n"
      "cmp x12, x11\n"
      "blt 1b\n"
      "incw x25, ALL, MUL #2\n"
      "incw x28, ALL, MUL #2\n"
      "cbz x14, 5f\n"
      "2:"  // Width loop
      "mov x12, #0\n"
      "3:"  // Width loop: Store: Loop
      ".inst 0xc0828011  // mova z17.s, p0/M, za0v.s[x12]\n"
      ".inst 0xc0828095  // mova z21.s, p0/M, za1v.s[x12]\n"
      ".inst 0xc0828119  // mova z25.s, p0/M, za2v.s[x12]\n"
      ".inst 0xc082819d  // mova z29.s, p0/M, za3v.s[x12]\n"
      "add x12, x12, #0x1\n"
      "cmp x12, x11\n"
      ".inst 0xa160df51  // st1w { z17.s, z21.s, z25.s, z29.s }, pn15.b, [x26]\n"
      "addvl x26, x26, #4\n"
      "blt 3b\n"
      "mov x13, %x[in]\n"
      "mov x20, %x[width]\n"
      "add x9, x13, x11, LSL #3\n"
      "mov x12, #0\n"
      "add x27, x9, x11, LSL #3\n"
      ".inst 0x25b44794  // whilelt pn12.s, x28, x20, VLx2\n"
      "add x24, x27, x11, LSL #3\n"
      "4:"  // Width loop: Load: Loop
      "ldr x23, [x13], #0x8\n"
      ".inst 0x2530704b  // psel p11.s, p12.s/Z, p2.s[w12]\n"
      ".inst 0x2530702a  // psel p10.s, p12.s/Z, p1.s[w12]\n"
      "ldr x22, [x9], #0x8\n"
      ".inst 0x253071c9  // psel p9.s, p12.s/Z, p14.s[w12]\n"
      ".inst 0x253071a8  // psel p8.s, p12.s/Z, p13.s[w12]\n"
      "ldr x21, [x27], #0x8\n"
      "ldr x20, [x24], #0x8\n"
      ".inst 0xa0194eec  // ld1w { z12.s-z13.s }, pn11.s/Z, [x23, x25, LSL #2]\n"
      ".inst 0xa0194ace  // ld1w { z14.s-z15.s }, pn10.s/Z, [x22, x25, LSL #2]\n"
      ".inst 0xa01946b2  // ld1w { z18.s-z19.s }, pn9.s/Z, [x21, x25, LSL #2]\n"
      ".inst 0xa019429e  // ld1w { z30.s-z31.s }, pn8.s/Z, [x20, x25, LSL #2]\n"
      ".inst 0xc160e18c  // bfcvt z12.h, { z12.s-z13.s }\n"
      ".inst 0xc160e1ce  // bfcvt z14.h, { z14.s-z15.s }\n"
      ".inst 0xc0800180  // mova za0h.s[x12], p0/M, z12.s\n"
      ".inst 0xc160e252  // bfcvt z18.h, { z18.s-z19.s }\n"
      ".inst 0xc08001c4  // mova za1h.s[x12], p0/M, z14.s\n"
      ".inst 0xc160e3de  // bfcvt z30.h, { z30.s-z31.s }\n"
      ".inst 0xc0800248  // mova za2h.s[x12], p0/M, z18.s\n"
      ".inst 0xc08003cc  // mova za3h.s[x12], p0/M, z30.s\n"
      "add x12, x12, #0x1\n"
      "cmp x12, x11\n"
      "blt 4b\n"
      "subs x14, x14, #0x1\n"
      "incw x25, ALL, MUL #2\n"
      "incw x28, ALL, MUL #2\n"
      "bgt 2b\n"
      "5:"  // Width loop: Tails
      "mov x12, #0\n"
      "6:"  // Width loop: Tails: Loop
      ".inst 0xc0828011  // mova z17.s, p0/M, za0v.s[x12]\n"
      ".inst 0xc0828095  // mova z21.s, p0/M, za1v.s[x12]\n"
      ".inst 0xc0828119  // mova z25.s, p0/M, za2v.s[x12]\n"
      ".inst 0xc082819d  // mova z29.s, p0/M, za3v.s[x12]\n"
      "add x12, x12, #0x1\n"
      "cmp x12, x10\n"
      ".inst 0xa160df51  // st1w { z17.s, z21.s, z25.s, z29.s }, pn15.b, [x26]\n"
      "addvl x26, x26, #4\n"
      "blt 6b\n"
      "mov %x[outptr_raw], x26\n"
      ".inst 0xd503467f  // SMSTOP\n"
      : [outptr_raw] "+&r" (out)
      : [height] "r" (height), [in] "r" (in), [row_offset] "r" (row_offset), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

#endif // __aarch64__
