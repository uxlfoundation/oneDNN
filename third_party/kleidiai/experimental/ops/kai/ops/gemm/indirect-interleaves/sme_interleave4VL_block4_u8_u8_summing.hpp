//
// SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#ifdef __aarch64__

template <>
void interleave_block<4, 4, VLType::SME, true>(
  uint8_t * &out, const uint8_t * const *in,
  size_t width, size_t height, size_t row_offset, bool first, int32_t multiplier
)
{
  __asm__ __volatile__(
      ".inst 0xd503477f  // SMSTART ZA\n"
      "mov x17, %x[row_offset]\n"
      "mov x16, %x[out]\n"
      "cntw x15\n"
      "cntw x14\n"
      "mov z25.b, #0x1\n"
      "mov z24.s, #0\n"
      "cntw x11, ALL, MUL #2\n"
      "cntw x10, ALL, MUL #3\n"
      "mov z23.s, #0\n"
      "mov z22.s, #0\n"
      "cmp %x[height], x15\n"
      "ptrue p3.b\n"
      "mov z21.s, #0\n"
      "csel x15, %x[height], x15, LT\n"
      "whilelt p12.b, XZR, %x[height]\n"
      "ld1rw { z20.s }, p3/Z, [%x[mul_ptr]]\n"
      "whilelt p10.b, x14, %x[height]\n"
      "whilelt p9.b, x11, %x[height]\n"
      "whilelt p8.b, x10, %x[height]\n"
      "zip1 p12.b, p12.b, p9.b\n"
      "zip1 p10.b, p10.b, p8.b\n"
      "cntb x9\n"
      "ptrue p11.b\n"
      "sub x15, x15, #0x1\n"
      "zip1 p10.b, p12.b, p10.b\n"
      "cbnz %x[first], 1f\n"
      "addvl x16, x16, #-4\n"
      "ld1w { z24.s }, p3/Z, [x16]\n"
      "ld1w { z23.s }, p3/Z, [x16, #1, MUL VL]\n"
      "ld1w { z22.s }, p3/Z, [x16, #2, MUL VL]\n"
      "ld1w { z21.s }, p3/Z, [x16, #3, MUL VL]\n"
      "1:"  // Initialise row sums: End
      "mov x28, #0\n"
      "whilelt p9.b, x28, %x[width]\n"
      "whilelt p8.b, x28, %x[width]\n"
      "2:"  // Width loop
      "add x27, %x[in], XZR, LSL #3\n"
      "add x26, %x[in], x14, LSL #3\n"
      "add x25, %x[in], x11, LSL #3\n"
      "add x20, %x[in], x10, LSL #3\n"
      "ldr x24, [x27], #0x8\n"
      "mov x13, #0\n"
      "ldr x23, [x26], #0x8\n"
      "ldr x22, [x25], #0x8\n"
      "ldr x21, [x20], #0x8\n"
      "cbz x15, 4f\n"
      "3:"  // Loads: Loop
      ".inst 0x25256140  // psel p0.b, p8.b/Z, p10.b[w13]\n"
      ".inst 0x252d6142  // psel p2.b, p8.b/Z, p10.b[w13, #1]\n"
      ".inst 0x25356141  // psel p1.b, p8.b/Z, p10.b[w13, #2]\n"
      ".inst 0xe0112300  // ld1b { za0h.b[x13] }, p0/Z, [x24, x17]\n"
      ".inst 0x253d6140  // psel p0.b, p8.b/Z, p10.b[w13, #3]\n"
      "ldr x24, [x27], #0x8\n"
      ".inst 0xe0112ae1  // ld1b { za0h.b[x13, #1] }, p2/Z, [x23, x17]\n"
      "ldr x23, [x26], #0x8\n"
      ".inst 0xe01126c2  // ld1b { za0h.b[x13, #2] }, p1/Z, [x22, x17]\n"
      "ldr x22, [x25], #0x8\n"
      ".inst 0xe01122a3  // ld1b { za0h.b[x13, #3] }, p0/Z, [x21, x17]\n"
      "add x13, x13, #0x4\n"
      "ldr x21, [x20], #0x8\n"
      "cmp x13, x15, LSL #2\n"
      "blt 3b\n"
      "4:"  // Loads: Tail
      ".inst 0x25256140  // psel p0.b, p8.b/Z, p10.b[w13]\n"
      ".inst 0x252d6142  // psel p2.b, p8.b/Z, p10.b[w13, #1]\n"
      ".inst 0x25356141  // psel p1.b, p8.b/Z, p10.b[w13, #2]\n"
      "sub x20, %x[width], x28\n"
      "cmp x20, x9\n"
      "mov x12, #0\n"
      ".inst 0xe0112300  // ld1b { za0h.b[x13] }, p0/Z, [x24, x17]\n"
      ".inst 0x253d6140  // psel p0.b, p8.b/Z, p10.b[w13, #3]\n"
      "csel x20, x20, x9, LT\n"
      ".inst 0xe0112ae1  // ld1b { za0h.b[x13, #1] }, p2/Z, [x23, x17]\n"
      "add x20, x20, #0x3\n"
      ".inst 0xe01126c2  // ld1b { za0h.b[x13, #2] }, p1/Z, [x22, x17]\n"
      "lsr x20, x20, #0x2\n"
      ".inst 0xe01122a3  // ld1b { za0h.b[x13, #3] }, p0/Z, [x21, x17]\n"
      "5:"  // Stores: Loop
      ".inst 0x25306d20  // psel p0.s, p11.s/Z, p9.s[w12]\n"
      ".inst 0x25306d22  // psel p2.s, p11.s/Z, p9.s[w12]\n"
      ".inst 0xc0828c13  // mova z19.s, p3/M, za0v.s[x12]\n"
      ".inst 0x25306d21  // psel p1.s, p11.s/Z, p9.s[w12]\n"
      ".inst 0xc0828c91  // mova z17.s, p3/M, za1v.s[x12]\n"
      "udot z24.s, z19.b, z25.b\n"
      ".inst 0xc0828d10  // mova z16.s, p3/M, za2v.s[x12]\n"
      "udot z23.s, z17.b, z25.b\n"
      ".inst 0xe0bf8200  // st1w { za0v.s[x12] }, p0/Z, [x16, XZR, LSL #2]\n"
      ".inst 0x25306d20  // psel p0.s, p11.s/Z, p9.s[w12]\n"
      ".inst 0xc0828d92  // mova z18.s, p3/M, za3v.s[x12]\n"
      "udot z22.s, z16.b, z25.b\n"
      ".inst 0xe0ae8a04  // st1w { za1v.s[x12] }, p2/Z, [x16, x14, LSL #2]\n"
      "udot z21.s, z18.b, z25.b\n"
      ".inst 0xe0ab8608  // st1w { za2v.s[x12] }, p1/Z, [x16, x11, LSL #2]\n"
      ".inst 0xe0aa820c  // st1w { za3v.s[x12] }, p0/Z, [x16, x10, LSL #2]\n"
      "add x12, x12, #0x1\n"
      "addvl x16, x16, #4\n"
      "cmp x12, x20\n"
      "blt 5b\n"
      "incb x28\n"
      "incb x17\n"
      "whilelt p9.b, x28, %x[width]\n"
      "whilelt p8.b, x28, %x[width]\n"
      "b.ne 2b\n"
      "mul z24.s, p3/M, z24.s, z20.s\n"
      "mul z23.s, p3/M, z23.s, z20.s\n"
      "mul z22.s, p3/M, z22.s, z20.s\n"
      "mul z21.s, p3/M, z21.s, z20.s\n"
      "st1w { z24.s }, p3, [x16]\n"
      "st1w { z23.s }, p3, [x16, #1, MUL VL]\n"
      "st1w { z22.s }, p3, [x16, #2, MUL VL]\n"
      "st1w { z21.s }, p3, [x16, #3, MUL VL]\n"
      "addvl x16, x16, #4\n"
      ".inst 0xd503467f  // SMSTOP\n"
      "mov %x[out], x16\n"
      : [out] "+&r" (out)
      : [first] "r" (first), [height] "r" (height), [in] "r" (in), [mul_ptr] "r" (&multiplier), [row_offset] "r" (row_offset), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

#endif // __aarch64__
