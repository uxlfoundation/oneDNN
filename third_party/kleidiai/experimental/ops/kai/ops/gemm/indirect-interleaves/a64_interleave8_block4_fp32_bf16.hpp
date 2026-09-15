//
// SPDX-FileCopyrightText: Copyright 2019-2021, 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#ifdef __aarch64__

template<>
void interleave_block<8, 4, VLType::None, false>(
  bfloat16 * &out_ptr, const float * const * in, size_t width, size_t height,
  size_t row_offset, bool, int32_t
)
{
  __asm__ __volatile__(
      "ldr x28, [%x[in], #0]\n"
      "ldr x27, [%x[in], #0x8]\n"
      "cmp %x[height], #0x8\n"
      "ldr x26, [%x[in], #0x10]\n"
      "ldr x25, [%x[in], #0x18]\n"
      "ldr x24, [%x[in], #0x20]\n"
      "ldr x23, [%x[in], #0x28]\n"
      "ldr x22, [%x[in], #0x30]\n"
      "ldr x21, [%x[in], #0x38]\n"
      "add x28, x28, %x[row_offset], LSL #2\n"
      "add x27, x27, %x[row_offset], LSL #2\n"
      "add x26, x26, %x[row_offset], LSL #2\n"
      "add x25, x25, %x[row_offset], LSL #2\n"
      "add x24, x24, %x[row_offset], LSL #2\n"
      "add x23, x23, %x[row_offset], LSL #2\n"
      "add x22, x22, %x[row_offset], LSL #2\n"
      "add x21, x21, %x[row_offset], LSL #2\n"
      "beq 1f\n"
      "cmp %x[height], #0x2\n"
      "mov x21, x28\n"
      "csel x27, x27, x28, GE\n"
      "csel x26, x26, x28, GT\n"
      "cmp %x[height], #0x4\n"
      "csel x25, x25, x28, GE\n"
      "csel x24, x24, x28, GT\n"
      "cmp %x[height], #0x6\n"
      "csel x23, x23, x28, GE\n"
      "csel x22, x22, x28, GT\n"
      "1:"  // no_pointer_adj
      "cmp %x[width], #0x4\n"
      "prfm pldl1keep, [x28, #0]\n"
      "prfm pldl1keep, [x27, #0]\n"
      "prfm pldl1keep, [x26, #0]\n"
      "prfm pldl1keep, [x25, #0]\n"
      "prfm pldl1keep, [x24, #0]\n"
      "prfm pldl1keep, [x23, #0]\n"
      "prfm pldl1keep, [x22, #0]\n"
      "prfm pldl1keep, [x21, #0]\n"
      "prfm pldl1keep, [x28, #0x40]\n"
      "prfm pldl1keep, [x27, #0x40]\n"
      "prfm pldl1keep, [x26, #0x40]\n"
      "prfm pldl1keep, [x25, #0x40]\n"
      "prfm pldl1keep, [x24, #0x40]\n"
      "prfm pldl1keep, [x23, #0x40]\n"
      "prfm pldl1keep, [x22, #0x40]\n"
      "prfm pldl1keep, [x21, #0x40]\n"
      "blt 3f\n"
      "2:"  // Main loop head
      "ldr q19, [x28], #0x10\n"
      "ldr q18, [x26], #0x10\n"
      "subs %x[width], %x[width], #0x4\n"
      "ldr q17, [x24], #0x10\n"
      "ldr q16, [x22], #0x10\n"
      "cmp %x[width], #0x4\n"
      "ldr q23, [x27], #0x10\n"
      "ldr q22, [x25], #0x10\n"
      "ldr q21, [x23], #0x10\n"
      "ldr q20, [x21], #0x10\n"
      ".inst 0x0ea16a73  // bfcvtn v19.4h, v19.4s\n"
      ".inst 0x0ea16a52  // bfcvtn v18.4h, v18.4s\n"
      ".inst 0x0ea16a31  // bfcvtn v17.4h, v17.4s\n"
      ".inst 0x0ea16a10  // bfcvtn v16.4h, v16.4s\n"
      "prfm pldl1keep, [x28, #0x70]\n"
      "prfm pldl1keep, [x27, #0x70]\n"
      "prfm pldl1keep, [x26, #0x70]\n"
      "prfm pldl1keep, [x25, #0x70]\n"
      "prfm pldl1keep, [x24, #0x70]\n"
      "prfm pldl1keep, [x23, #0x70]\n"
      ".inst 0x4ea16af3  // bfcvtn2 v19.8h, v23.4s\n"
      ".inst 0x4ea16ad2  // bfcvtn2 v18.8h, v22.4s\n"
      "prfm pldl1keep, [x22, #0x70]\n"
      "prfm pldl1keep, [x21, #0x70]\n"
      ".inst 0x4ea16ab1  // bfcvtn2 v17.8h, v21.4s\n"
      ".inst 0x4ea16a90  // bfcvtn2 v16.8h, v20.4s\n"
      "str q19, [%x[out_ptr], #0]\n"
      "str q18, [%x[out_ptr], #0x10]\n"
      "str q17, [%x[out_ptr], #0x20]\n"
      "str q16, [%x[out_ptr], #0x30]\n"
      "add %x[out_ptr], %x[out_ptr], #0x40\n"
      "bge 2b\n"
      "3:"  // Main loop skip
      "cbz %x[width], 6f\n"
      "tbz %x[width], #1, 4f\n"
      "ldr d19, [x28], #0x8\n"
      "ldr d23, [x27], #0x8\n"
      "mov x20, #0x1\n"
      "ldr d18, [x26], #0x8\n"
      "ldr d22, [x25], #0x8\n"
      "ldr d17, [x24], #0x8\n"
      "ldr d21, [x23], #0x8\n"
      "ldr d16, [x22], #0x8\n"
      "ldr d20, [x21], #0x8\n"
      "tbz %x[width], #0, 5f\n"
      "ld1 { v19.s }[2], [x28]\n"
      "ld1 { v23.s }[2], [x27]\n"
      "ld1 { v18.s }[2], [x26]\n"
      "ld1 { v22.s }[2], [x25]\n"
      "ld1 { v17.s }[2], [x24]\n"
      "ld1 { v21.s }[2], [x23]\n"
      "ld1 { v16.s }[2], [x22]\n"
      "ld1 { v20.s }[2], [x21]\n"
      "b 5f\n"
      "4:"  // odd_loads_1_0
      "ldr s19, [x28, #0]\n"
      "ldr s23, [x27, #0]\n"
      "mov x20, #0x1\n"
      "ldr s18, [x26, #0]\n"
      "ldr s22, [x25, #0]\n"
      "ldr s17, [x24, #0]\n"
      "ldr s21, [x23, #0]\n"
      "ldr s16, [x22, #0]\n"
      "ldr s20, [x21, #0]\n"
      "5:"  // Odd load end
      ".inst 0x0ea16a73  // bfcvtn v19.4h, v19.4s\n"
      ".inst 0x0ea16a52  // bfcvtn v18.4h, v18.4s\n"
      ".inst 0x0ea16a31  // bfcvtn v17.4h, v17.4s\n"
      ".inst 0x0ea16a10  // bfcvtn v16.4h, v16.4s\n"
      ".inst 0x4ea16af3  // bfcvtn2 v19.8h, v23.4s\n"
      ".inst 0x4ea16ad2  // bfcvtn2 v18.8h, v22.4s\n"
      ".inst 0x4ea16ab1  // bfcvtn2 v17.8h, v21.4s\n"
      ".inst 0x4ea16a90  // bfcvtn2 v16.8h, v20.4s\n"
      "str q19, [%x[out_ptr], #0]\n"
      "str q18, [%x[out_ptr], #0x10]\n"
      "str q17, [%x[out_ptr], #0x20]\n"
      "str q16, [%x[out_ptr], #0x30]\n"
      "add %x[out_ptr], %x[out_ptr], #0x40\n"
      "6:"  // Odds skip
      : [out_ptr] "+&r" (out_ptr), [width] "+&r" (width)
      : [height] "r" (height), [in] "r" (in), [row_offset] "r" (row_offset)
      : "cc", "memory", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
    );
}


#endif // __aarch64__
