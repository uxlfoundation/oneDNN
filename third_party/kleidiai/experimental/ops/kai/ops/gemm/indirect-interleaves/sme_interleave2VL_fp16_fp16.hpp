//
// SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#ifdef __aarch64__

template <>
void interleave_block<2, 1, VLType::SME, false>(
  __fp16 * &out, const __fp16 * const *in,
  size_t width, size_t height, size_t row_offset, bool, int32_t
)
{
  __asm__ __volatile__(
      ".inst 0xd503477f  // SMSTART ZA\n"
      "mov x28, #0\n"
      "mov x27, %x[row_offset]\n"
      "cnth x26\n"
      "cnth x25\n"
      "cmp %x[height], x26\n"
      "ptrue p13.b\n"
      "csel x26, %x[height], x26, LT\n"
      "whilelt p12.h, XZR, %x[height]\n"
      "sub x26, x26, #0x1\n"
      "whilelt p11.h, x25, %x[height]\n"
      "mov x24, %x[out]\n"
      "whilelt p10.h, x28, %x[width]\n"
      "whilelt p9.h, x28, %x[width]\n"
      "whilelt p8.h, x28, %x[width]\n"
      "1:"  // Width loop
      "add x23, %x[in], XZR, LSL #3\n"
      "add x20, %x[in], x25, LSL #3\n"
      "mov x13, #0\n"
      "ldr x22, [x23], #0x8\n"
      "ldr x21, [x20], #0x8\n"
      "cbz x26, 3f\n"
      "2:"  // Loads: Loop
      ".inst 0x25296581  // psel p1.h, p9.h/Z, p12.h[w13]\n"
      ".inst 0x25296160  // psel p0.h, p8.h/Z, p11.h[w13]\n"
      ".inst 0xe05b26c0  // ld1h { za0h.h[x13] }, p1/Z, [x22, x27, LSL #1]\n"
      "ldr x22, [x23], #0x8\n"
      ".inst 0xe05b22a8  // ld1h { za1h.h[x13] }, p0/Z, [x21, x27, LSL #1]\n"
      "add x13, x13, #0x1\n"
      "ldr x21, [x20], #0x8\n"
      "cmp x13, x26\n"
      "blt 2b\n"
      "3:"  // Loads: Tail
      ".inst 0x25296581  // psel p1.h, p9.h/Z, p12.h[w13]\n"
      ".inst 0x25296160  // psel p0.h, p8.h/Z, p11.h[w13]\n"
      "sub x20, %x[width], x28\n"
      "mov x12, #0\n"
      "cmp x20, x25\n"
      ".inst 0xe05b26c0  // ld1h { za0h.h[x13] }, p1/Z, [x22, x27, LSL #1]\n"
      "csel x20, x20, x25, LT\n"
      ".inst 0xe05b22a8  // ld1h { za1h.h[x13] }, p0/Z, [x21, x27, LSL #1]\n"
      "4:"  // Stores: Loop
      ".inst 0x25287541  // psel p1.h, p13.h/Z, p10.h[w12]\n"
      ".inst 0x25287540  // psel p0.h, p13.h/Z, p10.h[w12]\n"
      ".inst 0xe07f8700  // st1h { za0v.h[x12] }, p1/Z, [x24, XZR, LSL #1]\n"
      ".inst 0xe0798308  // st1h { za1v.h[x12] }, p0/Z, [x24, x25, LSL #1]\n"
      "add x12, x12, #0x1\n"
      "addvl x24, x24, #2\n"
      "cmp x12, x20\n"
      "blt 4b\n"
      "inch x28\n"
      "inch x27\n"
      "whilelt p10.h, x28, %x[width]\n"
      "whilelt p9.h, x28, %x[width]\n"
      "whilelt p8.h, x28, %x[width]\n"
      "b.ne 1b\n"
      "mov %x[out], x24\n"
      ".inst 0xd503467f  // SMSTOP\n"
      : [out] "+&r" (out)
      : [height] "r" (height), [in] "r" (in), [row_offset] "r" (row_offset), [width] "r" (width)
      : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x12", "x13", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

#endif // __aarch64__
