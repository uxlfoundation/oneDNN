//
// SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#ifdef __aarch64__

template <>
void interleave_block<2, 2, VLType::SME, false>(
  bfloat16 * &out, const float * const *in,
  size_t width, size_t height, size_t row_offset, bool, int32_t
)
{
  __asm__ __volatile__(
      ".inst 0xd503477f  // SMSTART ZA\n"
      "sub x10, %x[width], #0x1\n"
      "mov x9, #0\n"
      "cntw x22, ALL, MUL #2\n"
      "cntw x28\n"
      "cntw x21, ALL, MUL #2\n"
      "sub x20, x22, #0x1\n"
      ".inst 0x25207815  // ptrue pn13.b\n"
      "whilelt p12.s, XZR, %x[height]\n"
      "whilelt p11.s, x28, %x[height]\n"
      "add x10, x10, x21\n"
      "ands x27, %x[width], x20\n"
      "udiv x10, x10, x21\n"
      "csel x27, x27, x22, NE\n"
      "and x26, x10, #0x1\n"
      "sub x10, x10, #0x1\n"
      "add x27, x27, #0x1\n"
      "mov x20, %x[width]\n"
      "mov x25, %x[in]\n"
      "ptrue p0.b\n"
      "mov x24, %x[outptr_raw]\n"
      "mov x23, %x[row_offset]\n"
      "lsr x10, x10, #0x1\n"
      "lsr x27, x27, #0x1\n"
      "mov x12, #0\n"
      ".inst 0x25b44532  // whilelt pn10.s, x9, x20, VLx2\n"
      "add x22, x25, x28, LSL #3\n"
      "1:"  // Width loop: Preamble: Loop
      "ldr x21, [x25], #0x8\n"
      ".inst 0x25306989  // psel p9.s, p10.s/Z, p12.s[w12]\n"
      ".inst 0x25306968  // psel p8.s, p10.s/Z, p11.s[w12]\n"
      "ldr x20, [x22], #0x8\n"
      ".inst 0xa01746b4  // ld1w { z20.s-z21.s }, pn9.s/Z, [x21, x23, LSL #2]\n"
      ".inst 0xa017428c  // ld1w { z12.s-z13.s }, pn8.s/Z, [x20, x23, LSL #2]\n"
      ".inst 0xc160e294  // bfcvt z20.h, { z20.s-z21.s }\n"
      ".inst 0xc160e18c  // bfcvt z12.h, { z12.s-z13.s }\n"
      ".inst 0xc0800280  // mova za0h.s[x12], p0/M, z20.s\n"
      ".inst 0xc0800184  // mova za1h.s[x12], p0/M, z12.s\n"
      "add x12, x12, #0x1\n"
      "cmp x12, x28\n"
      "blt 1b\n"
      "incw x23, ALL, MUL #2\n"
      "incw x9, ALL, MUL #2\n"
      "cbz x10, 5f\n"
      "2:"  // Width loop
      "mov x20, %x[width]\n"
      "mov x25, %x[in]\n"
      "mov x12, #0\n"
      ".inst 0x25b44532  // whilelt pn10.s, x9, x20, VLx2\n"
      "add x22, x25, x28, LSL #3\n"
      "3:"  // Width loop: Odd: Loop
      "ldr x21, [x25], #0x8\n"
      ".inst 0x25306989  // psel p9.s, p10.s/Z, p12.s[w12]\n"
      ".inst 0x25306968  // psel p8.s, p10.s/Z, p11.s[w12]\n"
      ".inst 0xc0828007  // mova z7.s, p0/M, za0v.s[x12]\n"
      "ldr x20, [x22], #0x8\n"
      ".inst 0xc082808f  // mova z15.s, p0/M, za1v.s[x12]\n"
      ".inst 0xa01746b6  // ld1w { z22.s-z23.s }, pn9.s/Z, [x21, x23, LSL #2]\n"
      ".inst 0xa017429a  // ld1w { z26.s-z27.s }, pn8.s/Z, [x20, x23, LSL #2]\n"
      ".inst 0xa1605707  // st1w { z7.s, z15.s }, pn13.b, [x24]\n"
      "addvl x24, x24, #2\n"
      ".inst 0xc160e2d6  // bfcvt z22.h, { z22.s-z23.s }\n"
      ".inst 0xc160e35a  // bfcvt z26.h, { z26.s-z27.s }\n"
      ".inst 0xc08002c8  // mova za2h.s[x12], p0/M, z22.s\n"
      ".inst 0xc080034c  // mova za3h.s[x12], p0/M, z26.s\n"
      "add x12, x12, #0x1\n"
      "cmp x12, x28\n"
      "blt 3b\n"
      "incw x9, ALL, MUL #2\n"
      "mov x20, %x[width]\n"
      "mov x25, %x[in]\n"
      "incw x23, ALL, MUL #2\n"
      "mov x12, #0\n"
      ".inst 0x25b44532  // whilelt pn10.s, x9, x20, VLx2\n"
      "add x22, x25, x28, LSL #3\n"
      "4:"  // Width loop: Even: Loop
      "ldr x21, [x25], #0x8\n"
      ".inst 0x25306989  // psel p9.s, p10.s/Z, p12.s[w12]\n"
      ".inst 0x25306968  // psel p8.s, p10.s/Z, p11.s[w12]\n"
      ".inst 0xc0828108  // mova z8.s, p0/M, za2v.s[x12]\n"
      "ldr x20, [x22], #0x8\n"
      ".inst 0xc0828189  // mova z9.s, p0/M, za3v.s[x12]\n"
      ".inst 0xa01746ae  // ld1w { z14.s-z15.s }, pn9.s/Z, [x21, x23, LSL #2]\n"
      ".inst 0xa017428c  // ld1w { z12.s-z13.s }, pn8.s/Z, [x20, x23, LSL #2]\n"
      ".inst 0xa0605708  // st1w { z8.s-z9.s }, pn13.b, [x24]\n"
      "addvl x24, x24, #2\n"
      ".inst 0xc160e1ce  // bfcvt z14.h, { z14.s-z15.s }\n"
      ".inst 0xc160e18c  // bfcvt z12.h, { z12.s-z13.s }\n"
      ".inst 0xc08001c0  // mova za0h.s[x12], p0/M, z14.s\n"
      ".inst 0xc0800184  // mova za1h.s[x12], p0/M, z12.s\n"
      "add x12, x12, #0x1\n"
      "cmp x12, x28\n"
      "blt 4b\n"
      "subs x10, x10, #0x1\n"
      "incw x23, ALL, MUL #2\n"
      "incw x9, ALL, MUL #2\n"
      "bgt 2b\n"
      "5:"  // Width loop: Tails
      "cbnz x26, 8f\n"
      "mov x20, %x[width]\n"
      "mov x25, %x[in]\n"
      "mov x12, #0\n"
      ".inst 0x25b44532  // whilelt pn10.s, x9, x20, VLx2\n"
      "add x22, x25, x28, LSL #3\n"
      "6:"  // Width loop: Tails: Even: Odd: Loop
      "ldr x21, [x25], #0x8\n"
      ".inst 0x25306989  // psel p9.s, p10.s/Z, p12.s[w12]\n"
      ".inst 0x25306968  // psel p8.s, p10.s/Z, p11.s[w12]\n"
      ".inst 0xc0828003  // mova z3.s, p0/M, za0v.s[x12]\n"
      "ldr x20, [x22], #0x8\n"
      ".inst 0xc082808b  // mova z11.s, p0/M, za1v.s[x12]\n"
      ".inst 0xa01746ac  // ld1w { z12.s-z13.s }, pn9.s/Z, [x21, x23, LSL #2]\n"
      ".inst 0xa017428e  // ld1w { z14.s-z15.s }, pn8.s/Z, [x20, x23, LSL #2]\n"
      ".inst 0xa1605703  // st1w { z3.s, z11.s }, pn13.b, [x24]\n"
      "addvl x24, x24, #2\n"
      ".inst 0xc160e18c  // bfcvt z12.h, { z12.s-z13.s }\n"
      ".inst 0xc160e1ce  // bfcvt z14.h, { z14.s-z15.s }\n"
      ".inst 0xc0800188  // mova za2h.s[x12], p0/M, z12.s\n"
      ".inst 0xc08001cc  // mova za3h.s[x12], p0/M, z14.s\n"
      "add x12, x12, #0x1\n"
      "cmp x12, x28\n"
      "blt 6b\n"
      "mov x12, #0\n"
      "7:"  // Width loop: Tails: Even: Even: Loop
      ".inst 0xc082810e  // mova z14.s, p0/M, za2v.s[x12]\n"
      ".inst 0xc082818f  // mova z15.s, p0/M, za3v.s[x12]\n"
      "add x12, x12, #0x1\n"
      "cmp x12, x27\n"
      ".inst 0xa060570e  // st1w { z14.s-z15.s }, pn13.b, [x24]\n"
      "addvl x24, x24, #2\n"
      "blt 7b\n"
      "b 10f\n"
      "8:"  // Width loop: Tails: Odd
      "mov x12, #0\n"
      "9:"  // Width loop: Tails: Odd: Loop
      ".inst 0xc0828014  // mova z20.s, p0/M, za0v.s[x12]\n"
      ".inst 0xc0828095  // mova z21.s, p0/M, za1v.s[x12]\n"
      "add x12, x12, #0x1\n"
      "cmp x12, x27\n"
      ".inst 0xa0605714  // st1w { z20.s-z21.s }, pn13.b, [x24]\n"
      "addvl x24, x24, #2\n"
      "blt 9b\n"
      "10:"  // End
      "mov %x[outptr_raw], x24\n"
      ".inst 0xd503467f  // SMSTOP\n"
      : [outptr_raw] "+&r" (out)
      : [height] "r" (height), [in] "r" (in), [row_offset] "r" (row_offset), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x12", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

#endif // __aarch64__
