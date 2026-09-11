//
// SPDX-FileCopyrightText: Copyright 2021, 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

#ifdef __aarch64__

namespace {

void a64_transpose_interleave_96(uint32_t *out, const uint32_t *in, size_t width, size_t in_stride, size_t height)
{
    size_t out_stride = 24 * height * sizeof(uint32_t);

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
      "2:"  // Main row loop: Column loop
      "ldr q7, [x25], #0x10\n"
      "ldr q6, [x22], #0x10\n"
      "sub x24, x24, #0x18\n"
      "ldr q5, [x21], #0x10\n"
      "ldr q4, [x20], #0x10\n"
      "cmp x24, #0x18\n"
      "ldr q3, [x25], #0x10\n"
      "ldr q2, [x22], #0x10\n"
      "ldr q1, [x21], #0x10\n"
      "ldr q0, [x20], #0x10\n"
      "ldr q31, [x25], #0x10\n"
      "ldr q30, [x22], #0x10\n"
      "ldr q29, [x21], #0x10\n"
      "ldr q28, [x20], #0x10\n"
      "ldr q27, [x25], #0x10\n"
      "ldr q26, [x22], #0x10\n"
      "ldr q25, [x21], #0x10\n"
      "ldr q24, [x20], #0x10\n"
      "ldr q23, [x25], #0x10\n"
      "ldr q22, [x22], #0x10\n"
      "ldr q21, [x21], #0x10\n"
      "ldr q20, [x20], #0x10\n"
      "ldr q19, [x25], #0x10\n"
      "ldr q18, [x22], #0x10\n"
      "ldr q17, [x21], #0x10\n"
      "ldr q16, [x20], #0x10\n"
      "str q7, [x23, #0]\n"
      "str q3, [x23, #0x10]\n"
      "str q31, [x23, #0x20]\n"
      "str q27, [x23, #0x30]\n"
      "str q23, [x23, #0x40]\n"
      "str q19, [x23, #0x50]\n"
      "str q6, [x23, #0x60]\n"
      "str q2, [x23, #0x70]\n"
      "str q30, [x23, #0x80]\n"
      "str q26, [x23, #0x90]\n"
      "str q22, [x23, #0xa0]\n"
      "str q18, [x23, #0xb0]\n"
      "str q5, [x23, #0xc0]\n"
      "str q1, [x23, #0xd0]\n"
      "str q29, [x23, #0xe0]\n"
      "str q25, [x23, #0xf0]\n"
      "str q21, [x23, #0x100]\n"
      "str q17, [x23, #0x110]\n"
      "str q4, [x23, #0x120]\n"
      "str q0, [x23, #0x130]\n"
      "str q28, [x23, #0x140]\n"
      "str q24, [x23, #0x150]\n"
      "str q20, [x23, #0x160]\n"
      "str q16, [x23, #0x170]\n"
      "add x23, x23, %x[out_stride]\n"
      "bge 2b\n"
      "3:"  // Main row loop: Column loop skip
      "cbz x24, 10f\n"
      "cmp x24, #0x10\n"
      "movi v16.4s, #0\n"
      "str q16, [x23, #0]\n"
      "str q16, [x23, #0x10]\n"
      "str q16, [x23, #0x20]\n"
      "str q16, [x23, #0x30]\n"
      "str q16, [x23, #0x40]\n"
      "str q16, [x23, #0x50]\n"
      "str q16, [x23, #0x60]\n"
      "str q16, [x23, #0x70]\n"
      "str q16, [x23, #0x80]\n"
      "str q16, [x23, #0x90]\n"
      "str q16, [x23, #0xa0]\n"
      "str q16, [x23, #0xb0]\n"
      "str q16, [x23, #0xc0]\n"
      "str q16, [x23, #0xd0]\n"
      "str q16, [x23, #0xe0]\n"
      "str q16, [x23, #0xf0]\n"
      "str q16, [x23, #0x100]\n"
      "str q16, [x23, #0x110]\n"
      "str q16, [x23, #0x120]\n"
      "str q16, [x23, #0x130]\n"
      "str q16, [x23, #0x140]\n"
      "str q16, [x23, #0x150]\n"
      "str q16, [x23, #0x160]\n"
      "str q16, [x23, #0x170]\n"
      "blt 5f\n"
      "4:"  // Main row loop: width 16 loop: loop
      "ldr q31, [x25], #0x10\n"
      "ldr q30, [x22], #0x10\n"
      "sub x24, x24, #0x10\n"
      "ldr q29, [x21], #0x10\n"
      "ldr q28, [x20], #0x10\n"
      "cmp x24, #0x10\n"
      "ldr q27, [x25], #0x10\n"
      "ldr q26, [x22], #0x10\n"
      "ldr q25, [x21], #0x10\n"
      "ldr q24, [x20], #0x10\n"
      "ldr q23, [x25], #0x10\n"
      "ldr q22, [x22], #0x10\n"
      "ldr q21, [x21], #0x10\n"
      "ldr q20, [x20], #0x10\n"
      "ldr q19, [x25], #0x10\n"
      "ldr q18, [x22], #0x10\n"
      "ldr q17, [x21], #0x10\n"
      "ldr q16, [x20], #0x10\n"
      "str q31, [x23, #0]\n"
      "str q27, [x23, #0x10]\n"
      "str q23, [x23, #0x20]\n"
      "str q19, [x23, #0x30]\n"
      "str q30, [x23, #0x60]\n"
      "str q26, [x23, #0x70]\n"
      "str q22, [x23, #0x80]\n"
      "str q18, [x23, #0x90]\n"
      "str q29, [x23, #0xc0]\n"
      "str q25, [x23, #0xd0]\n"
      "str q21, [x23, #0xe0]\n"
      "str q17, [x23, #0xf0]\n"
      "str q28, [x23, #0x120]\n"
      "str q24, [x23, #0x130]\n"
      "str q20, [x23, #0x140]\n"
      "str q16, [x23, #0x150]\n"
      "add x23, x23, #0x40\n"
      "bge 4b\n"
      "5:"  // Main row loop: width 16 loop: skip
      "cmp x24, #0x4\n"
      "blt 7f\n"
      "6:"  // Main row loop: width 4 loop: loop
      "ldr q19, [x25], #0x10\n"
      "ldr q18, [x22], #0x10\n"
      "sub x24, x24, #0x4\n"
      "ldr q17, [x21], #0x10\n"
      "ldr q16, [x20], #0x10\n"
      "cmp x24, #0x4\n"
      "str q19, [x23, #0]\n"
      "str q18, [x23, #0x60]\n"
      "str q17, [x23, #0xc0]\n"
      "str q16, [x23, #0x120]\n"
      "add x23, x23, #0x10\n"
      "bge 6b\n"
      "7:"  // Main row loop: width 4 loop: skip
      "cmp x24, #0x1\n"
      "blt 9f\n"
      "8:"  // Main row loop: width 1 loop: loop
      "ldr s19, [x25], #0x4\n"
      "ldr s18, [x22], #0x4\n"
      "sub x24, x24, #0x1\n"
      "ldr s17, [x21], #0x4\n"
      "ldr s16, [x20], #0x4\n"
      "cmp x24, #0x1\n"
      "str s19, [x23, #0]\n"
      "str s18, [x23, #0x60]\n"
      "str s17, [x23, #0xc0]\n"
      "str s16, [x23, #0x120]\n"
      "add x23, x23, #0x4\n"
      "bge 8b\n"
      "9:"  // Main row loop: width 1 loop: skip
      "10:"  // Main row loop: odd col skip
      "cmp x26, #0x4\n"
      "add %x[out], %x[out], #0x180\n"
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
      "13:"  // Tail row loop: Column loop
      "ldr q21, [x25], #0x10\n"
      "sub x20, x20, #0x18\n"
      "ldr q20, [x25], #0x10\n"
      "ldr q19, [x25], #0x10\n"
      "cmp x20, #0x18\n"
      "ldr q18, [x25], #0x10\n"
      "ldr q17, [x25], #0x10\n"
      "ldr q16, [x25], #0x10\n"
      "str q21, [x23, #0]\n"
      "str q20, [x23, #0x10]\n"
      "str q19, [x23, #0x20]\n"
      "str q18, [x23, #0x30]\n"
      "str q17, [x23, #0x40]\n"
      "str q16, [x23, #0x50]\n"
      "add x23, x23, %x[out_stride]\n"
      "bge 13b\n"
      "14:"  // Tail row loop: Column loop skip
      "cbz x20, 21f\n"
      "cmp x20, #0x10\n"
      "movi v16.4s, #0\n"
      "str q16, [x23, #0]\n"
      "str q16, [x23, #0x10]\n"
      "str q16, [x23, #0x20]\n"
      "str q16, [x23, #0x30]\n"
      "str q16, [x23, #0x40]\n"
      "str q16, [x23, #0x50]\n"
      "blt 16f\n"
      "15:"  // Tail row loop: width 16 loop: loop
      "ldr q19, [x25], #0x10\n"
      "sub x20, x20, #0x10\n"
      "ldr q18, [x25], #0x10\n"
      "ldr q17, [x25], #0x10\n"
      "cmp x20, #0x10\n"
      "ldr q16, [x25], #0x10\n"
      "str q19, [x23, #0]\n"
      "str q18, [x23, #0x10]\n"
      "str q17, [x23, #0x20]\n"
      "str q16, [x23, #0x30]\n"
      "add x23, x23, #0x40\n"
      "bge 15b\n"
      "16:"  // Tail row loop: width 16 loop: skip
      "cmp x20, #0x4\n"
      "blt 18f\n"
      "17:"  // Tail row loop: width 4 loop: loop
      "ldr q16, [x25], #0x10\n"
      "sub x20, x20, #0x4\n"
      "cmp x20, #0x4\n"
      "str q16, [x23, #0]\n"
      "add x23, x23, #0x10\n"
      "bge 17b\n"
      "18:"  // Tail row loop: width 4 loop: skip
      "cmp x20, #0x1\n"
      "blt 20f\n"
      "19:"  // Tail row loop: width 1 loop: loop
      "ldr s16, [x25], #0x4\n"
      "sub x20, x20, #0x1\n"
      "cmp x20, #0x1\n"
      "str s16, [x23, #0]\n"
      "add x23, x23, #0x4\n"
      "bge 19b\n"
      "20:"  // Tail row loop: width 1 loop: skip
      "21:"  // Tail row loop: odd col skip
      "cmp x26, #0x1\n"
      "add %x[out], %x[out], #0x60\n"
      "bge 12b\n"
      "22:"  // Done
      : [in] "+&r" (in), [out] "+&r" (out)
      : [height] "r" (height), [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [width] "r" (width)
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x20", "x21", "x22", "x23", "x24", "x25", "x26"
    );
}

} // anonymous namespace

template<>
void Transform<24, 1, true, VLType::None>(
    float *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    a64_transpose_interleave_96(
        reinterpret_cast<uint32_t *>(out),
        reinterpret_cast<const uint32_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(float) / 4,
        stride * sizeof(float),
        (kmax-k0)
    );
}

#endif // __aarch64__
