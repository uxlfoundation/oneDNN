//
// SPDX-FileCopyrightText: Copyright 2021, 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

#ifdef __aarch64__

namespace {

void a64_transpose_interleave_12_s8s16(int16_t *out, const int8_t *in, size_t width, size_t in_stride, size_t height)
{
    size_t out_stride = 12 * height * sizeof(int16_t);

    __asm__ __volatile__(
      "mov x26, %x[height]\n"
      "cmp x26, #0x4\n"
      "blt 11f\n"
      "1:"  // Main row loop: Head
      "mov x25, %x[in]\n"
      "mov x24, %x[width]\n"
      "mov x23, %x[out]\n"
      "sub x26, x26, #0x4\n"
      "add x22, x25, %x[in_stride]\n"
      "add x21, x22, %x[in_stride]\n"
      "add x20, x21, %x[in_stride]\n"
      "cmp x24, #0x18\n"
      "add %x[in], x20, %x[in_stride]\n"
      "blt 3f\n"
      "2:"  // Main row loop: Unroll column loop
      "ldr q3, [x25], #0x10\n"
      "ldr q21, [x22], #0x10\n"
      "sub x24, x24, #0x18\n"
      "ldr q2, [x21], #0x10\n"
      "ldr q20, [x20], #0x10\n"
      "cmp x24, #0x18\n"
      "ldr d19, [x25], #0x8\n"
      "ldr d17, [x21], #0x8\n"
      "ldr d1, [x22], #0x8\n"
      "sshll2 v18.8h, v3.16b, #0\n"
      "sshll v0.8h, v21.8b, #0\n"
      "ldr d31, [x20], #0x8\n"
      "sshll2 v16.8h, v2.16b, #0\n"
      "sshll v30.8h, v20.8b, #0\n"
      "sshll v29.8h, v19.8b, #0\n"
      "sshll v28.8h, v17.8b, #0\n"
      "sshll2 v27.8h, v21.16b, #0\n"
      "sshll2 v26.8h, v20.16b, #0\n"
      "dup v17.2d, v18.d[0]\n"
      "dup v25.2d, v0.d[1]\n"
      "dup v24.2d, v16.d[0]\n"
      "dup v23.2d, v30.d[1]\n"
      "dup v22.2d, v18.d[1]\n"
      "dup v21.2d, v29.d[1]\n"
      "dup v20.2d, v16.d[1]\n"
      "dup v19.2d, v28.d[1]\n"
      "sshll v16.8h, v3.8b, #0\n"
      "sshll v18.8h, v2.8b, #0\n"
      "mov v17.d[1], v0.d[0]\n"
      "mov v25.d[1], v27.d[0]\n"
      "mov v24.d[1], v30.d[0]\n"
      "mov v23.d[1], v26.d[0]\n"
      "str q16, [x23, #0]\n"
      "str q17, [x23, #0x10]\n"
      "sshll v17.8h, v1.8b, #0\n"
      "sshll v16.8h, v31.8b, #0\n"
      "str q25, [x23, #0x20]\n"
      "mov v22.d[1], v29.d[0]\n"
      "mov v21.d[1], v27.d[1]\n"
      "str q18, [x23, #0x30]\n"
      "mov v20.d[1], v28.d[0]\n"
      "mov v19.d[1], v26.d[1]\n"
      "str q24, [x23, #0x40]\n"
      "str q23, [x23, #0x50]\n"
      "add x23, x23, %x[out_stride]\n"
      "str q22, [x23, #0]\n"
      "str q21, [x23, #0x10]\n"
      "str q17, [x23, #0x20]\n"
      "str q20, [x23, #0x30]\n"
      "str q19, [x23, #0x40]\n"
      "str q16, [x23, #0x50]\n"
      "add x23, x23, %x[out_stride]\n"
      "bge 2b\n"
      "3:"  // Main row loop: Unroll column loop skip
      "cmp x24, #0xc\n"
      "blt 5f\n"
      "4:"  // Main row loop: Column loop
      "ldr d19, [x22], #0x8\n"
      "ldr d18, [x20], #0x8\n"
      "sub x24, x24, #0xc\n"
      "ld1 { v19.s }[2], [x22], #0x4\n"
      "ldr d17, [x25], #0x8\n"
      "cmp x24, #0xc\n"
      "ld1 { v18.s }[2], [x20], #0x4\n"
      "ldr d16, [x21], #0x8\n"
      "ld1 { v17.s }[2], [x25], #0x4\n"
      "ld1 { v16.s }[2], [x21], #0x4\n"
      "sshll v25.8h, v19.8b, #0\n"
      "sshll2 v24.8h, v19.16b, #0\n"
      "sshll v23.8h, v18.8b, #0\n"
      "sshll2 v22.8h, v18.16b, #0\n"
      "sshll2 v21.8h, v17.16b, #0\n"
      "sshll2 v20.8h, v16.16b, #0\n"
      "dup v19.2d, v25.d[1]\n"
      "sshll v18.8h, v17.8b, #0\n"
      "dup v17.2d, v23.d[1]\n"
      "sshll v16.8h, v16.8b, #0\n"
      "mov v21.d[1], v25.d[0]\n"
      "mov v19.d[1], v24.d[0]\n"
      "mov v20.d[1], v23.d[0]\n"
      "mov v17.d[1], v22.d[0]\n"
      "str q18, [x23, #0]\n"
      "str q21, [x23, #0x10]\n"
      "str q19, [x23, #0x20]\n"
      "str q16, [x23, #0x30]\n"
      "str q20, [x23, #0x40]\n"
      "str q17, [x23, #0x50]\n"
      "add x23, x23, %x[out_stride]\n"
      "bge 4b\n"
      "5:"  // Main row loop: Column loop skip
      "cbz x24, 10f\n"
      "cmp x24, #0x4\n"
      "movi v16.8h, #0\n"
      "str q16, [x23, #0]\n"
      "str q16, [x23, #0x10]\n"
      "str q16, [x23, #0x20]\n"
      "str q16, [x23, #0x30]\n"
      "str q16, [x23, #0x40]\n"
      "str q16, [x23, #0x50]\n"
      "blt 7f\n"
      "6:"  // Main row loop: width 4 loop: loop
      "ldr s19, [x25], #0x4\n"
      "ldr s18, [x22], #0x4\n"
      "sub x24, x24, #0x4\n"
      "ldr s17, [x21], #0x4\n"
      "ldr s16, [x20], #0x4\n"
      "cmp x24, #0x4\n"
      "sshll v19.8h, v19.8b, #0\n"
      "sshll v18.8h, v18.8b, #0\n"
      "sshll v17.8h, v17.8b, #0\n"
      "sshll v16.8h, v16.8b, #0\n"
      "str d19, [x23, #0]\n"
      "str d18, [x23, #0x18]\n"
      "str d17, [x23, #0x30]\n"
      "str d16, [x23, #0x48]\n"
      "add x23, x23, #0x8\n"
      "bge 6b\n"
      "7:"  // Main row loop: width 4 loop: skip
      "cmp x24, #0x1\n"
      "blt 9f\n"
      "8:"  // Main row loop: width 1 loop: loop
      "ldr b19, [x25], #0x1\n"
      "ldr b18, [x22], #0x1\n"
      "sub x24, x24, #0x1\n"
      "ldr b17, [x21], #0x1\n"
      "ldr b16, [x20], #0x1\n"
      "cmp x24, #0x1\n"
      "sshll v19.8h, v19.8b, #0\n"
      "sshll v18.8h, v18.8b, #0\n"
      "sshll v17.8h, v17.8b, #0\n"
      "sshll v16.8h, v16.8b, #0\n"
      "str h19, [x23, #0]\n"
      "str h18, [x23, #0x18]\n"
      "str h17, [x23, #0x30]\n"
      "str h16, [x23, #0x48]\n"
      "add x23, x23, #0x2\n"
      "bge 8b\n"
      "9:"  // Main row loop: width 1 loop: skip
      "10:"  // Main row loop: odd col skip
      "cmp x26, #0x4\n"
      "add %x[out], %x[out], #0x60\n"
      "bge 1b\n"
      "cbz x26, 22f\n"
      "11:"  // Main loop skip
      "12:"  // Tail row loop: Head
      "mov x20, %x[width]\n"
      "mov x25, %x[in]\n"
      "mov x23, %x[out]\n"
      "sub x26, x26, #0x1\n"
      "cmp x20, #0x18\n"
      "add %x[in], x25, %x[in_stride]\n"
      "blt 14f\n"
      "13:"  // Tail row loop: Unroll column loop
      "ldr q18, [x25], #0x10\n"
      "sub x20, x20, #0x18\n"
      "ldr d16, [x25], #0x8\n"
      "cmp x20, #0x18\n"
      "sshll2 v17.8h, v18.16b, #0\n"
      "sshll v16.8h, v16.8b, #0\n"
      "sshll v19.8h, v18.8b, #0\n"
      "dup v18.2d, v17.d[1]\n"
      "dup v17.2d, v17.d[0]\n"
      "mov v18.d[1], v16.d[0]\n"
      "dup v16.2d, v16.d[1]\n"
      "str q19, [x23, #0]\n"
      "str d17, [x23, #0x10]\n"
      "add x23, x23, %x[out_stride]\n"
      "str q18, [x23, #0]\n"
      "str d16, [x23, #0x10]\n"
      "add x23, x23, %x[out_stride]\n"
      "bge 13b\n"
      "14:"  // Tail row loop: Unroll column loop skip
      "cmp x20, #0xc\n"
      "blt 16f\n"
      "15:"  // Tail row loop: Column loop
      "ldr d16, [x25], #0x8\n"
      "sub x20, x20, #0xc\n"
      "ld1 { v16.s }[2], [x25], #0x4\n"
      "cmp x20, #0xc\n"
      "sshll v17.8h, v16.8b, #0\n"
      "sshll2 v16.8h, v16.16b, #0\n"
      "str q17, [x23, #0]\n"
      "str d16, [x23, #0x10]\n"
      "add x23, x23, %x[out_stride]\n"
      "bge 15b\n"
      "16:"  // Tail row loop: Column loop skip
      "cbz x20, 21f\n"
      "cmp x20, #0x4\n"
      "movi v16.8h, #0\n"
      "str q16, [x23, #0]\n"
      "str d16, [x23, #0x10]\n"
      "blt 18f\n"
      "17:"  // Tail row loop: width 4 loop: loop
      "ldr s16, [x25], #0x4\n"
      "sub x20, x20, #0x4\n"
      "cmp x20, #0x4\n"
      "sshll v16.8h, v16.8b, #0\n"
      "str d16, [x23, #0]\n"
      "add x23, x23, #0x8\n"
      "bge 17b\n"
      "18:"  // Tail row loop: width 4 loop: skip
      "cmp x20, #0x1\n"
      "blt 20f\n"
      "19:"  // Tail row loop: width 1 loop: loop
      "ldr b16, [x25], #0x1\n"
      "sub x20, x20, #0x1\n"
      "cmp x20, #0x1\n"
      "sshll v16.8h, v16.8b, #0\n"
      "str h16, [x23, #0]\n"
      "add x23, x23, #0x2\n"
      "bge 19b\n"
      "20:"  // Tail row loop: width 1 loop: skip
      "21:"  // Tail row loop: odd col skip
      "cmp x26, #0x1\n"
      "add %x[out], %x[out], #0x18\n"
      "bge 12b\n"
      "22:"  // Done
      : [in] "+&r" (in), [out] "+&r" (out)
      : [height] "r" (height), [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [width] "r" (width)
      : "cc", "memory", "v0", "v1", "v2", "v3", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x20", "x21", "x22", "x23", "x24", "x25", "x26"
    );
}

} // anonymous namespace

template<>
void Transform<12, 1, true, VLType::None>(
    int16_t *out, const int8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    a64_transpose_interleave_12_s8s16(
        out,
        in + k0 * stride + x0,
        (xmax-x0),
        stride * sizeof(int8_t),
        (kmax-k0)
    );
}

#endif // __aarch64__
