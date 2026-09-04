//
// SPDX-FileCopyrightText: Copyright 2021, 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

#ifdef __aarch64__

namespace {

void a64_transpose_interleave_16_2x2(uint16_t *out, const uint16_t *in, size_t width, size_t in_stride, size_t height)
{
    uint16_t *pad_row = reinterpret_cast<uint16_t *>(alloca(width * sizeof(uint16_t)));

    if (height % 2) {
        memset(pad_row, 0, width * sizeof(uint16_t));
    }

    size_t out_stride = 16 * roundup<size_t>(height, 2) * sizeof(uint16_t);

    __asm__ __volatile__(
      "mov x10, %x[height]\n"
      "cmp x10, #0x8\n"
      "blt 9f\n"
      "1:"  // Main row loop: Head
      "mov x9, %x[in]\n"
      "mov x28, %x[width]\n"
      "mov x27, %x[out]\n"
      "sub x10, x10, #0x8\n"
      "add x26, x9, %x[in_stride]\n"
      "add x25, x26, %x[in_stride]\n"
      "add x24, x25, %x[in_stride]\n"
      "cmp x28, #0x10\n"
      "add x23, x24, %x[in_stride]\n"
      "add x22, x23, %x[in_stride]\n"
      "add x21, x22, %x[in_stride]\n"
      "add x20, x21, %x[in_stride]\n"
      "add %x[in], x20, %x[in_stride]\n"
      "blt 3f\n"
      "2:"  // Main row loop: Column loop
      "ldr q22, [x9], #0x10\n"
      "ldr q21, [x26], #0x10\n"
      "sub x28, x28, #0x10\n"
      "ldr q20, [x25], #0x10\n"
      "ldr q19, [x24], #0x10\n"
      "cmp x28, #0x10\n"
      "ldr q18, [x23], #0x10\n"
      "ldr q17, [x22], #0x10\n"
      "ldr q23, [x21], #0x10\n"
      "ldr q16, [x20], #0x10\n"
      "zip1 v0.8h, v22.8h, v21.8h\n"
      "zip2 v31.8h, v22.8h, v21.8h\n"
      "ldr q22, [x9], #0x10\n"
      "ldr q21, [x26], #0x10\n"
      "zip1 v30.8h, v20.8h, v19.8h\n"
      "zip2 v29.8h, v20.8h, v19.8h\n"
      "ldr q20, [x25], #0x10\n"
      "ldr q19, [x24], #0x10\n"
      "zip1 v28.8h, v18.8h, v17.8h\n"
      "zip2 v27.8h, v18.8h, v17.8h\n"
      "ldr q18, [x23], #0x10\n"
      "ldr q17, [x22], #0x10\n"
      "zip1 v26.8h, v23.8h, v16.8h\n"
      "zip2 v25.8h, v23.8h, v16.8h\n"
      "ldr q24, [x21], #0x10\n"
      "ldr q16, [x20], #0x10\n"
      "zip1 v23.8h, v22.8h, v21.8h\n"
      "zip2 v22.8h, v22.8h, v21.8h\n"
      "zip1 v21.8h, v20.8h, v19.8h\n"
      "zip2 v20.8h, v20.8h, v19.8h\n"
      "str q0, [x27, #0]\n"
      "zip1 v19.8h, v18.8h, v17.8h\n"
      "zip2 v18.8h, v18.8h, v17.8h\n"
      "str q31, [x27, #0x10]\n"
      "zip1 v17.8h, v24.8h, v16.8h\n"
      "zip2 v16.8h, v24.8h, v16.8h\n"
      "str q23, [x27, #0x20]\n"
      "str q22, [x27, #0x30]\n"
      "str q30, [x27, #0x40]\n"
      "str q29, [x27, #0x50]\n"
      "str q21, [x27, #0x60]\n"
      "str q20, [x27, #0x70]\n"
      "str q28, [x27, #0x80]\n"
      "str q27, [x27, #0x90]\n"
      "str q19, [x27, #0xa0]\n"
      "str q18, [x27, #0xb0]\n"
      "str q26, [x27, #0xc0]\n"
      "str q25, [x27, #0xd0]\n"
      "str q17, [x27, #0xe0]\n"
      "str q16, [x27, #0xf0]\n"
      "add x27, x27, %x[out_stride]\n"
      "bge 2b\n"
      "3:"  // Main row loop: Column loop skip
      "cbz x28, 8f\n"
      "cmp x28, #0x4\n"
      "movi v16.8h, #0\n"
      "str q16, [x27, #0]\n"
      "str q16, [x27, #0x10]\n"
      "str q16, [x27, #0x20]\n"
      "str q16, [x27, #0x30]\n"
      "str q16, [x27, #0x40]\n"
      "str q16, [x27, #0x50]\n"
      "str q16, [x27, #0x60]\n"
      "str q16, [x27, #0x70]\n"
      "str q16, [x27, #0x80]\n"
      "str q16, [x27, #0x90]\n"
      "str q16, [x27, #0xa0]\n"
      "str q16, [x27, #0xb0]\n"
      "str q16, [x27, #0xc0]\n"
      "str q16, [x27, #0xd0]\n"
      "str q16, [x27, #0xe0]\n"
      "str q16, [x27, #0xf0]\n"
      "blt 5f\n"
      "4:"  // Main row loop: width 4 loop: loop
      "ldr d23, [x9], #0x8\n"
      "ldr d18, [x26], #0x8\n"
      "sub x28, x28, #0x4\n"
      "ldr d22, [x25], #0x8\n"
      "ldr d16, [x24], #0x8\n"
      "cmp x28, #0x4\n"
      "ldr d21, [x23], #0x8\n"
      "ldr d17, [x22], #0x8\n"
      "ldr d20, [x21], #0x8\n"
      "ldr d19, [x20], #0x8\n"
      "zip1 v18.8h, v23.8h, v18.8h\n"
      "zip1 v16.8h, v22.8h, v16.8h\n"
      "zip1 v17.8h, v21.8h, v17.8h\n"
      "str q18, [x27, #0]\n"
      "str q16, [x27, #0x40]\n"
      "zip1 v16.8h, v20.8h, v19.8h\n"
      "str q17, [x27, #0x80]\n"
      "str q16, [x27, #0xc0]\n"
      "add x27, x27, #0x10\n"
      "bge 4b\n"
      "5:"  // Main row loop: width 4 loop: skip
      "cmp x28, #0x1\n"
      "blt 7f\n"
      "6:"  // Main row loop: width 1 loop: loop
      "ldr h23, [x9], #0x2\n"
      "ldr h18, [x26], #0x2\n"
      "sub x28, x28, #0x1\n"
      "ldr h22, [x25], #0x2\n"
      "ldr h16, [x24], #0x2\n"
      "cmp x28, #0x1\n"
      "ldr h21, [x23], #0x2\n"
      "ldr h17, [x22], #0x2\n"
      "ldr h20, [x21], #0x2\n"
      "ldr h19, [x20], #0x2\n"
      "zip1 v18.8h, v23.8h, v18.8h\n"
      "zip1 v16.8h, v22.8h, v16.8h\n"
      "zip1 v17.8h, v21.8h, v17.8h\n"
      "str s18, [x27, #0]\n"
      "str s16, [x27, #0x40]\n"
      "zip1 v16.8h, v20.8h, v19.8h\n"
      "str s17, [x27, #0x80]\n"
      "str s16, [x27, #0xc0]\n"
      "add x27, x27, #0x4\n"
      "bge 6b\n"
      "7:"  // Main row loop: width 1 loop: skip
      "8:"  // Main row loop: odd col skip
      "cmp x10, #0x8\n"
      "add %x[out], %x[out], #0x100\n"
      "bge 1b\n"
      "cbz x10, 18f\n"
      "9:"  // Main loop skip
      "10:"  // Tail row loop: Head
      "mov x9, %x[in]\n"
      "mov x20, %x[width]\n"
      "cmp x10, #0x1\n"
      "mov x27, %x[out]\n"
      "sub x10, x10, #0x2\n"
      "add x26, x9, %x[in_stride]\n"
      "add %x[in], x26, %x[in_stride]\n"
      "csel %x[in], %x[in], x26, GT\n"
      "csel x26, x26, %x[pad_row], GT\n"
      "cmp x20, #0x10\n"
      "blt 12f\n"
      "11:"  // Tail row loop: Column loop
      "ldr q18, [x9], #0x10\n"
      "ldr q17, [x26], #0x10\n"
      "sub x20, x20, #0x10\n"
      "ldr q20, [x9], #0x10\n"
      "cmp x20, #0x10\n"
      "ldr q16, [x26], #0x10\n"
      "zip1 v19.8h, v18.8h, v17.8h\n"
      "zip2 v18.8h, v18.8h, v17.8h\n"
      "zip1 v17.8h, v20.8h, v16.8h\n"
      "zip2 v16.8h, v20.8h, v16.8h\n"
      "str q19, [x27, #0]\n"
      "str q18, [x27, #0x10]\n"
      "str q17, [x27, #0x20]\n"
      "str q16, [x27, #0x30]\n"
      "add x27, x27, %x[out_stride]\n"
      "bge 11b\n"
      "12:"  // Tail row loop: Column loop skip
      "cbz x20, 17f\n"
      "cmp x20, #0x4\n"
      "movi v16.8h, #0\n"
      "str q16, [x27, #0]\n"
      "str q16, [x27, #0x10]\n"
      "str q16, [x27, #0x20]\n"
      "str q16, [x27, #0x30]\n"
      "blt 14f\n"
      "13:"  // Tail row loop: width 4 loop: loop
      "ldr d17, [x9], #0x8\n"
      "ldr d16, [x26], #0x8\n"
      "sub x20, x20, #0x4\n"
      "cmp x20, #0x4\n"
      "zip1 v16.8h, v17.8h, v16.8h\n"
      "str q16, [x27, #0]\n"
      "add x27, x27, #0x10\n"
      "bge 13b\n"
      "14:"  // Tail row loop: width 4 loop: skip
      "cmp x20, #0x1\n"
      "blt 16f\n"
      "15:"  // Tail row loop: width 1 loop: loop
      "ldr h17, [x9], #0x2\n"
      "ldr h16, [x26], #0x2\n"
      "sub x20, x20, #0x1\n"
      "cmp x20, #0x1\n"
      "zip1 v16.8h, v17.8h, v16.8h\n"
      "str s16, [x27, #0]\n"
      "add x27, x27, #0x4\n"
      "bge 15b\n"
      "16:"  // Tail row loop: width 1 loop: skip
      "17:"  // Tail row loop: odd col skip
      "cmp x10, #0x1\n"
      "add %x[out], %x[out], #0x40\n"
      "bge 10b\n"
      "18:"  // Done
      : [in] "+&r" (in), [out] "+&r" (out)
      : [height] "r" (height), [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [pad_row] "r" (pad_row), [width] "r" (width)
      : "cc", "memory", "v0", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x10", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
    );
}

} // anonymous namespace

template<>
void Transform<16, 2, true, VLType::None>(
    bfloat16 *out, const bfloat16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    a64_transpose_interleave_16_2x2(
        reinterpret_cast<uint16_t *>(out),
        reinterpret_cast<const uint16_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(bfloat16) / 2,
        stride * sizeof(bfloat16),
        (kmax-k0)
    );
}

#endif // __aarch64__
