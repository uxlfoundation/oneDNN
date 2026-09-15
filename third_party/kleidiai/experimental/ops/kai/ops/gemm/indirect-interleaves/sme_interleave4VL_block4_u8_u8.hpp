//
// SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#ifdef __aarch64__

template <>
void interleave_block<4, 4, VLType::SME, false>(
  uint8_t * &out, const uint8_t * const *in,
  size_t width, size_t height, size_t row_offset, bool, int32_t
)
{
  __asm__ __volatile__(
      ".inst 0xd503477f  // SMSTART ZA\n"
      "mov x17, #0\n"
      "mov x16, %x[row_offset]\n"
      "cntw x15\n"
      "cntw x14\n"
      "cntw x11, ALL, MUL #2\n"
      "cntw x10, ALL, MUL #3\n"
      "cmp %x[height], x15\n"
      "cntb x9\n"
      "csel x15, %x[height], x15, LT\n"
      "whilelt p12.b, XZR, %x[height]\n"
      "whilelt p10.b, x14, %x[height]\n"
      "whilelt p9.b, x11, %x[height]\n"
      "whilelt p8.b, x10, %x[height]\n"
      "zip1 p12.b, p12.b, p9.b\n"
      "zip1 p10.b, p10.b, p8.b\n"
      "ptrue p11.b\n"
      "sub x15, x15, #0x1\n"
      "zip1 p10.b, p12.b, p10.b\n"
      "mov x28, %x[out]\n"
      "whilelt p9.b, x17, %x[width]\n"
      "whilelt p8.b, x17, %x[width]\n"
      "1:"  // Width loop
      "add x27, %x[in], XZR, LSL #3\n"
      "add x26, %x[in], x14, LSL #3\n"
      "add x25, %x[in], x11, LSL #3\n"
      "add x20, %x[in], x10, LSL #3\n"
      "ldr x24, [x27], #0x8\n"
      "mov x13, #0\n"
      "ldr x23, [x26], #0x8\n"
      "ldr x22, [x25], #0x8\n"
      "ldr x21, [x20], #0x8\n"
      "cbz x15, 3f\n"
      "2:"  // Loads: Loop
      ".inst 0x25256140  // psel p0.b, p8.b/Z, p10.b[w13]\n"
      ".inst 0x252d6142  // psel p2.b, p8.b/Z, p10.b[w13, #1]\n"
      ".inst 0x25356141  // psel p1.b, p8.b/Z, p10.b[w13, #2]\n"
      ".inst 0xe0102300  // ld1b { za0h.b[x13] }, p0/Z, [x24, x16]\n"
      ".inst 0x253d6140  // psel p0.b, p8.b/Z, p10.b[w13, #3]\n"
      "ldr x24, [x27], #0x8\n"
      ".inst 0xe0102ae1  // ld1b { za0h.b[x13, #1] }, p2/Z, [x23, x16]\n"
      "ldr x23, [x26], #0x8\n"
      ".inst 0xe01026c2  // ld1b { za0h.b[x13, #2] }, p1/Z, [x22, x16]\n"
      "ldr x22, [x25], #0x8\n"
      ".inst 0xe01022a3  // ld1b { za0h.b[x13, #3] }, p0/Z, [x21, x16]\n"
      "add x13, x13, #0x4\n"
      "ldr x21, [x20], #0x8\n"
      "cmp x13, x15, LSL #2\n"
      "blt 2b\n"
      "3:"  // Loads: Tail
      ".inst 0x25256140  // psel p0.b, p8.b/Z, p10.b[w13]\n"
      ".inst 0x252d6142  // psel p2.b, p8.b/Z, p10.b[w13, #1]\n"
      ".inst 0x25356141  // psel p1.b, p8.b/Z, p10.b[w13, #2]\n"
      "sub x20, %x[width], x17\n"
      "cmp x20, x9\n"
      "mov x12, #0\n"
      ".inst 0xe0102300  // ld1b { za0h.b[x13] }, p0/Z, [x24, x16]\n"
      ".inst 0x253d6140  // psel p0.b, p8.b/Z, p10.b[w13, #3]\n"
      "csel x20, x20, x9, LT\n"
      ".inst 0xe0102ae1  // ld1b { za0h.b[x13, #1] }, p2/Z, [x23, x16]\n"
      "add x20, x20, #0x3\n"
      ".inst 0xe01026c2  // ld1b { za0h.b[x13, #2] }, p1/Z, [x22, x16]\n"
      "lsr x20, x20, #0x2\n"
      ".inst 0xe01022a3  // ld1b { za0h.b[x13, #3] }, p0/Z, [x21, x16]\n"
      "4:"  // Stores: Loop
      ".inst 0x25306d20  // psel p0.s, p11.s/Z, p9.s[w12]\n"
      ".inst 0x25306d22  // psel p2.s, p11.s/Z, p9.s[w12]\n"
      ".inst 0x25306d21  // psel p1.s, p11.s/Z, p9.s[w12]\n"
      ".inst 0xe0bf8380  // st1w { za0v.s[x12] }, p0/Z, [x28, XZR, LSL #2]\n"
      ".inst 0x25306d20  // psel p0.s, p11.s/Z, p9.s[w12]\n"
      ".inst 0xe0ae8b84  // st1w { za1v.s[x12] }, p2/Z, [x28, x14, LSL #2]\n"
      ".inst 0xe0ab8788  // st1w { za2v.s[x12] }, p1/Z, [x28, x11, LSL #2]\n"
      ".inst 0xe0aa838c  // st1w { za3v.s[x12] }, p0/Z, [x28, x10, LSL #2]\n"
      "add x12, x12, #0x1\n"
      "addvl x28, x28, #4\n"
      "cmp x12, x20\n"
      "blt 4b\n"
      "incb x17\n"
      "incb x16\n"
      "whilelt p9.b, x17, %x[width]\n"
      "whilelt p8.b, x17, %x[width]\n"
      "b.ne 1b\n"
      "mov %x[out], x28\n"
      ".inst 0xd503467f  // SMSTOP\n"
      : [out] "+&r" (out)
      : [height] "r" (height), [in] "r" (in), [row_offset] "r" (row_offset), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

#endif // __aarch64__
