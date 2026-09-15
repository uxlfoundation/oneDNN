//
// SPDX-FileCopyrightText: Copyright 2021-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

#ifdef __aarch64__

namespace {

void a64_transpose_interleave_16(uint32_t *out, const uint32_t *in, size_t width, size_t in_stride, size_t height)
{
    size_t out_stride = 4 * height * sizeof(uint32_t);

    __asm__ __volatile__(
      "mov x26, %x[height]\n"
      "cmp x26, #0x4\n"
      "blt 6f\n"
      "1:"  // Main row loop: Head
      "mov x25, %x[in]\n"
      "mov x24, %x[width]\n"
      "mov x23, %x[out]\n"
      "sub x26, x26, #0x4\n"
      "add x22, x25, %x[in_stride]\n"
      "add x21, x22, %x[in_stride]\n"
      "add x20, x21, %x[in_stride]\n"
      "cmp x24, #0x4\n"
      "add %x[in], x20, %x[in_stride]\n"
      "blt 3f\n"
      "2:"  // Main row loop: Column loop
      "ldr q19, [x25], #0x10\n"
      "ldr q18, [x22], #0x10\n"
      "sub x24, x24, #0x4\n"
      "ldr q17, [x21], #0x10\n"
      "ldr q16, [x20], #0x10\n"
      "cmp x24, #0x4\n"
      "str q19, [x23, #0]\n"
      "str q18, [x23, #0x10]\n"
      "str q17, [x23, #0x20]\n"
      "str q16, [x23, #0x30]\n"
      "add x23, x23, %x[out_stride]\n"
      "bge 2b\n"
      "3:"  // Main row loop: Column loop skip
      "cbz x24, 5f\n"
      "movi v16.4s, #0\n"
      "str q16, [x23, #0]\n"
      "str q16, [x23, #0x10]\n"
      "str q16, [x23, #0x20]\n"
      "str q16, [x23, #0x30]\n"
      "4:"  // Main row loop: width 1 loop: loop
      "ldr s19, [x25], #0x4\n"
      "ldr s18, [x22], #0x4\n"
      "sub x24, x24, #0x1\n"
      "ldr s17, [x21], #0x4\n"
      "ldr s16, [x20], #0x4\n"
      "cmp x24, #0x1\n"
      "str s19, [x23, #0]\n"
      "str s18, [x23, #0x10]\n"
      "str s17, [x23, #0x20]\n"
      "str s16, [x23, #0x30]\n"
      "add x23, x23, #0x4\n"
      "bge 4b\n"
      "5:"  // Main row loop: odd col skip
      "cmp x26, #0x4\n"
      "add %x[out], %x[out], #0x40\n"
      "bge 1b\n"
      "cbz x26, 12f\n"
      "6:"  // Main loop skip
      "7:"  // Tail row loop: Head
      "mov x20, %x[width]\n"
      "mov x25, %x[in]\n"
      "mov x23, %x[out]\n"
      "sub x26, x26, #0x1\n"
      "cmp x20, #0x4\n"
      "add %x[in], x25, %x[in_stride]\n"
      "blt 9f\n"
      "8:"  // Tail row loop: Column loop
      "ldr q16, [x25], #0x10\n"
      "sub x20, x20, #0x4\n"
      "cmp x20, #0x4\n"
      "str q16, [x23, #0]\n"
      "add x23, x23, %x[out_stride]\n"
      "bge 8b\n"
      "9:"  // Tail row loop: Column loop skip
      "cbz x20, 11f\n"
      "movi v16.4s, #0\n"
      "str q16, [x23, #0]\n"
      "10:"  // Tail row loop: width 1 loop: loop
      "ldr s16, [x25], #0x4\n"
      "sub x20, x20, #0x1\n"
      "cmp x20, #0x1\n"
      "str s16, [x23, #0]\n"
      "add x23, x23, #0x4\n"
      "bge 10b\n"
      "11:"  // Tail row loop: odd col skip
      "cmp x26, #0x1\n"
      "add %x[out], %x[out], #0x10\n"
      "bge 7b\n"
      "12:"  // Done
      : [in] "+&r" (in), [out] "+&r" (out)
      : [height] "r" (height), [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [width] "r" (width)
      : "cc", "memory", "v16", "v17", "v18", "v19", "x20", "x21", "x22", "x23", "x24", "x25", "x26"
    );
}

} // anonymous namespace

template<>
void Transform<4, 1, true, VLType::None>(
    float *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    a64_transpose_interleave_16(
        reinterpret_cast<uint32_t *>(out),
        reinterpret_cast<const uint32_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(float) / 4,
        stride * sizeof(float),
        (kmax-k0)
    );
}

#endif // __aarch64__
