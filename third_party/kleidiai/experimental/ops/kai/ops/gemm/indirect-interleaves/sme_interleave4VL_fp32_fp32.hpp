//
// SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#ifdef __aarch64__

template <>
void interleave_block<4, 1, VLType::SME, false>(
  float * &out, const float * const *in,
  size_t width, size_t height, size_t row_offset, bool, int32_t
)
{
  __asm__ __volatile__(
      ".inst 0xd503477f  // SMSTART ZA\n"
      "mov x16, #0\n"
      "mov x15, %x[row_offset]\n"
      "cntw x14\n"
      "cntw x11\n"
      "cmp %x[height], x14\n"
      "cntw x10, ALL, MUL #2\n"
      "cntw x9, ALL, MUL #3\n"
      "csel x14, %x[height], x14, LT\n"
      "ptrue p4.b\n"
      "sub x14, x14, #0x1\n"
      "whilelt p3.s, XZR, %x[height]\n"
      "whilelt p15.s, x11, %x[height]\n"
      "whilelt p14.s, x10, %x[height]\n"
      "whilelt p13.s, x9, %x[height]\n"
      "mov x28, %x[out]\n"
      "whilelt p12.s, x16, %x[width]\n"
      "whilelt p11.s, x16, %x[width]\n"
      "whilelt p10.s, x16, %x[width]\n"
      "whilelt p9.s, x16, %x[width]\n"
      "whilelt p8.s, x16, %x[width]\n"
      "1:"  // Width loop
      "add x27, %x[in], XZR, LSL #3\n"
      "add x26, %x[in], x11, LSL #3\n"
      "add x25, %x[in], x10, LSL #3\n"
      "add x20, %x[in], x9, LSL #3\n"
      "ldr x24, [x27], #0x8\n"
      "mov x13, #0\n"
      "ldr x23, [x26], #0x8\n"
      "ldr x22, [x25], #0x8\n"
      "ldr x21, [x20], #0x8\n"
      "cbz x14, 3f\n"
      "2:"  // Loads: Loop
      ".inst 0x25316c60  // psel p0.s, p11.s/Z, p3.s[w13]\n"
      ".inst 0x253169e2  // psel p2.s, p10.s/Z, p15.s[w13]\n"
      ".inst 0x253165c1  // psel p1.s, p9.s/Z, p14.s[w13]\n"
      ".inst 0xe08f2300  // ld1w { za0h.s[x13] }, p0/Z, [x24, x15, LSL #2]\n"
      ".inst 0x253161a0  // psel p0.s, p8.s/Z, p13.s[w13]\n"
      "ldr x24, [x27], #0x8\n"
      ".inst 0xe08f2ae4  // ld1w { za1h.s[x13] }, p2/Z, [x23, x15, LSL #2]\n"
      "ldr x23, [x26], #0x8\n"
      ".inst 0xe08f26c8  // ld1w { za2h.s[x13] }, p1/Z, [x22, x15, LSL #2]\n"
      "ldr x22, [x25], #0x8\n"
      ".inst 0xe08f22ac  // ld1w { za3h.s[x13] }, p0/Z, [x21, x15, LSL #2]\n"
      "add x13, x13, #0x1\n"
      "ldr x21, [x20], #0x8\n"
      "cmp x13, x14\n"
      "blt 2b\n"
      "3:"  // Loads: Tail
      ".inst 0x25316c60  // psel p0.s, p11.s/Z, p3.s[w13]\n"
      ".inst 0x253169e2  // psel p2.s, p10.s/Z, p15.s[w13]\n"
      ".inst 0x253165c1  // psel p1.s, p9.s/Z, p14.s[w13]\n"
      "sub x20, %x[width], x16\n"
      "cmp x20, x11\n"
      "mov x12, #0\n"
      ".inst 0xe08f2300  // ld1w { za0h.s[x13] }, p0/Z, [x24, x15, LSL #2]\n"
      ".inst 0x253161a0  // psel p0.s, p8.s/Z, p13.s[w13]\n"
      "csel x20, x20, x11, LT\n"
      ".inst 0xe08f2ae4  // ld1w { za1h.s[x13] }, p2/Z, [x23, x15, LSL #2]\n"
      ".inst 0xe08f26c8  // ld1w { za2h.s[x13] }, p1/Z, [x22, x15, LSL #2]\n"
      ".inst 0xe08f22ac  // ld1w { za3h.s[x13] }, p0/Z, [x21, x15, LSL #2]\n"
      "4:"  // Stores: Loop
      ".inst 0x25305180  // psel p0.s, p4.s/Z, p12.s[w12]\n"
      ".inst 0x25305182  // psel p2.s, p4.s/Z, p12.s[w12]\n"
      ".inst 0x25305181  // psel p1.s, p4.s/Z, p12.s[w12]\n"
      ".inst 0xe0bf8380  // st1w { za0v.s[x12] }, p0/Z, [x28, XZR, LSL #2]\n"
      ".inst 0x25305180  // psel p0.s, p4.s/Z, p12.s[w12]\n"
      ".inst 0xe0ab8b84  // st1w { za1v.s[x12] }, p2/Z, [x28, x11, LSL #2]\n"
      ".inst 0xe0aa8788  // st1w { za2v.s[x12] }, p1/Z, [x28, x10, LSL #2]\n"
      ".inst 0xe0a9838c  // st1w { za3v.s[x12] }, p0/Z, [x28, x9, LSL #2]\n"
      "add x12, x12, #0x1\n"
      "addvl x28, x28, #4\n"
      "cmp x12, x20\n"
      "blt 4b\n"
      "incw x16\n"
      "incw x15\n"
      "whilelt p12.s, x16, %x[width]\n"
      "whilelt p11.s, x16, %x[width]\n"
      "whilelt p10.s, x16, %x[width]\n"
      "whilelt p9.s, x16, %x[width]\n"
      "whilelt p8.s, x16, %x[width]\n"
      "b.ne 1b\n"
      "mov %x[out], x28\n"
      ".inst 0xd503467f  // SMSTOP\n"
      : [out] "+&r" (out)
      : [height] "r" (height), [in] "r" (in), [row_offset] "r" (row_offset), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

#endif // __aarch64__
