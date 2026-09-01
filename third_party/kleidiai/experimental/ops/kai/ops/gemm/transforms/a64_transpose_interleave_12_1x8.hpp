//
// SPDX-FileCopyrightText: Copyright 2021, 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

#ifdef __aarch64__

namespace {

void a64_transpose_interleave_12_1x8(uint8_t *out, const uint8_t *in, size_t width, size_t in_stride, size_t height)
{
    uint8_t *pad_row = reinterpret_cast<uint8_t *>(alloca(width * sizeof(uint8_t)));

    if (height % 8) {
        memset(pad_row, 0, width * sizeof(uint8_t));
    }

    size_t out_stride = 12 * roundup<size_t>(height, 8) * sizeof(uint8_t);

    __asm__ __volatile__(
      "mov x10, %x[height]\n"
      "cbz %x[height], 11f\n"
      "1:"  // Main row loop: Head
      "mov x9, %x[in]\n"
      "cmp x10, #0x7\n"
      "mov x28, %x[width]\n"
      "mov x27, %x[out]\n"
      "add x26, x9, %x[in_stride]\n"
      "add x25, x26, %x[in_stride]\n"
      "add x24, x25, %x[in_stride]\n"
      "add x23, x24, %x[in_stride]\n"
      "add x22, x23, %x[in_stride]\n"
      "add x21, x22, %x[in_stride]\n"
      "add x20, x21, %x[in_stride]\n"
      "add %x[in], x20, %x[in_stride]\n"
      "csel %x[in], %x[in], x20, GT\n"
      "csel x20, x20, %x[pad_row], GT\n"
      "csel %x[in], %x[in], x21, GE\n"
      "csel x21, x21, %x[pad_row], GE\n"
      "cmp x10, #0x5\n"
      "csel %x[in], %x[in], x22, GT\n"
      "csel x22, x22, %x[pad_row], GT\n"
      "csel %x[in], %x[in], x23, GE\n"
      "csel x23, x23, %x[pad_row], GE\n"
      "cmp x10, #0x3\n"
      "csel %x[in], %x[in], x24, GT\n"
      "csel x24, x24, %x[pad_row], GT\n"
      "csel %x[in], %x[in], x25, GE\n"
      "csel x25, x25, %x[pad_row], GE\n"
      "cmp x10, #0x1\n"
      "sub x10, x10, #0x8\n"
      "csel %x[in], %x[in], x26, GT\n"
      "csel x26, x26, %x[pad_row], GT\n"
      "cmp x28, #0x30\n"
      "blt 3f\n"
      "2:"  // Main row loop: Unroll column loop
      "ldr q1, [x9], #0x10\n"
      "ldr q0, [x26], #0x10\n"
      "sub x28, x28, #0x30\n"
      "ldr q31, [x25], #0x10\n"
      "ldr q28, [x24], #0x10\n"
      "cmp x28, #0x30\n"
      "ldr q27, [x23], #0x10\n"
      "ldr q26, [x22], #0x10\n"
      "ldr q25, [x21], #0x10\n"
      "ldr q24, [x20], #0x10\n"
      "ldr q23, [x9], #0x10\n"
      "ldr q30, [x26], #0x10\n"
      "ldr q22, [x25], #0x10\n"
      "ldr q21, [x24], #0x10\n"
      "zip1 v15.16b, v1.16b, v27.16b\n"
      "zip1 v9.16b, v0.16b, v26.16b\n"
      "ldr q20, [x23], #0x10\n"
      "ldr q19, [x22], #0x10\n"
      "zip1 v18.16b, v31.16b, v25.16b\n"
      "zip1 v29.16b, v28.16b, v24.16b\n"
      "ldr q17, [x21], #0x10\n"
      "ldr q16, [x20], #0x10\n"
      "zip2 v14.16b, v1.16b, v27.16b\n"
      "zip2 v13.16b, v31.16b, v25.16b\n"
      "ldr q8, [x9], #0x10\n"
      "ldr q7, [x26], #0x10\n"
      "zip2 v12.16b, v0.16b, v26.16b\n"
      "zip2 v6.16b, v28.16b, v24.16b\n"
      "ldr q5, [x25], #0x10\n"
      "ldr q4, [x24], #0x10\n"
      "zip1 v3.16b, v23.16b, v20.16b\n"
      "zip1 v11.16b, v30.16b, v19.16b\n"
      "ldr q28, [x23], #0x10\n"
      "ldr q27, [x22], #0x10\n"
      "zip1 v2.16b, v22.16b, v17.16b\n"
      "zip1 v1.16b, v21.16b, v16.16b\n"
      "ldr q26, [x21], #0x10\n"
      "ldr q25, [x20], #0x10\n"
      "zip2 v24.16b, v23.16b, v20.16b\n"
      "zip2 v23.16b, v22.16b, v17.16b\n"
      "zip2 v22.16b, v30.16b, v19.16b\n"
      "zip2 v21.16b, v21.16b, v16.16b\n"
      "zip1 v0.16b, v8.16b, v28.16b\n"
      "zip1 v10.16b, v7.16b, v27.16b\n"
      "zip1 v31.16b, v5.16b, v26.16b\n"
      "zip1 v30.16b, v4.16b, v25.16b\n"
      "zip1 v20.16b, v15.16b, v18.16b\n"
      "zip1 v19.16b, v9.16b, v29.16b\n"
      "zip2 v18.16b, v15.16b, v18.16b\n"
      "zip2 v16.16b, v9.16b, v29.16b\n"
      "zip1 v29.16b, v14.16b, v13.16b\n"
      "zip1 v17.16b, v12.16b, v6.16b\n"
      "zip2 v9.16b, v8.16b, v28.16b\n"
      "zip2 v28.16b, v5.16b, v26.16b\n"
      "zip2 v8.16b, v7.16b, v27.16b\n"
      "zip2 v27.16b, v4.16b, v25.16b\n"
      "zip2 v7.16b, v14.16b, v13.16b\n"
      "zip2 v6.16b, v12.16b, v6.16b\n"
      "zip1 v5.16b, v3.16b, v2.16b\n"
      "zip1 v4.16b, v11.16b, v1.16b\n"
      "zip2 v3.16b, v3.16b, v2.16b\n"
      "zip2 v2.16b, v11.16b, v1.16b\n"
      "zip1 v26.16b, v24.16b, v23.16b\n"
      "zip1 v25.16b, v22.16b, v21.16b\n"
      "zip2 v24.16b, v24.16b, v23.16b\n"
      "zip2 v23.16b, v22.16b, v21.16b\n"
      "zip1 v1.16b, v0.16b, v31.16b\n"
      "zip1 v22.16b, v10.16b, v30.16b\n"
      "zip1 v21.16b, v20.16b, v19.16b\n"
      "zip2 v20.16b, v20.16b, v19.16b\n"
      "zip1 v19.16b, v18.16b, v16.16b\n"
      "zip2 v18.16b, v18.16b, v16.16b\n"
      "zip1 v16.16b, v29.16b, v17.16b\n"
      "zip2 v17.16b, v29.16b, v17.16b\n"
      "str q21, [x27, #0]\n"
      "str q20, [x27, #0x10]\n"
      "zip2 v0.16b, v0.16b, v31.16b\n"
      "zip2 v31.16b, v10.16b, v30.16b\n"
      "str q19, [x27, #0x20]\n"
      "zip1 v30.16b, v9.16b, v28.16b\n"
      "zip1 v29.16b, v8.16b, v27.16b\n"
      "str q18, [x27, #0x30]\n"
      "zip2 v28.16b, v9.16b, v28.16b\n"
      "zip2 v27.16b, v8.16b, v27.16b\n"
      "str q16, [x27, #0x40]\n"
      "zip1 v21.16b, v7.16b, v6.16b\n"
      "zip2 v16.16b, v7.16b, v6.16b\n"
      "str q17, [x27, #0x50]\n"
      "add x27, x27, %x[out_stride]\n"
      "zip1 v20.16b, v5.16b, v4.16b\n"
      "zip2 v19.16b, v5.16b, v4.16b\n"
      "zip1 v18.16b, v3.16b, v2.16b\n"
      "zip2 v17.16b, v3.16b, v2.16b\n"
      "str q21, [x27, #0]\n"
      "str q16, [x27, #0x10]\n"
      "zip1 v16.16b, v26.16b, v25.16b\n"
      "zip2 v26.16b, v26.16b, v25.16b\n"
      "str q20, [x27, #0x20]\n"
      "zip1 v25.16b, v24.16b, v23.16b\n"
      "zip2 v24.16b, v24.16b, v23.16b\n"
      "str q19, [x27, #0x30]\n"
      "zip1 v23.16b, v1.16b, v22.16b\n"
      "zip2 v22.16b, v1.16b, v22.16b\n"
      "str q18, [x27, #0x40]\n"
      "zip1 v21.16b, v0.16b, v31.16b\n"
      "zip2 v20.16b, v0.16b, v31.16b\n"
      "str q17, [x27, #0x50]\n"
      "add x27, x27, %x[out_stride]\n"
      "zip1 v19.16b, v30.16b, v29.16b\n"
      "zip2 v18.16b, v30.16b, v29.16b\n"
      "str q16, [x27, #0]\n"
      "zip1 v17.16b, v28.16b, v27.16b\n"
      "zip2 v16.16b, v28.16b, v27.16b\n"
      "str q26, [x27, #0x10]\n"
      "str q25, [x27, #0x20]\n"
      "str q24, [x27, #0x30]\n"
      "str q23, [x27, #0x40]\n"
      "str q22, [x27, #0x50]\n"
      "add x27, x27, %x[out_stride]\n"
      "str q21, [x27, #0]\n"
      "str q20, [x27, #0x10]\n"
      "str q19, [x27, #0x20]\n"
      "str q18, [x27, #0x30]\n"
      "str q17, [x27, #0x40]\n"
      "str q16, [x27, #0x50]\n"
      "add x27, x27, %x[out_stride]\n"
      "bge 2b\n"
      "3:"  // Main row loop: Unroll column loop skip
      "cmp x28, #0xc\n"
      "blt 5f\n"
      "4:"  // Main row loop: Column loop
      "ldr d22, [x9], #0x8\n"
      "ldr d23, [x26], #0x8\n"
      "sub x28, x28, #0xc\n"
      "ldr d21, [x25], #0x8\n"
      "ldr d20, [x24], #0x8\n"
      "cmp x28, #0xc\n"
      "ldr d18, [x23], #0x8\n"
      "ldr d17, [x22], #0x8\n"
      "ldr d19, [x21], #0x8\n"
      "ldr d16, [x20], #0x8\n"
      "ld1 { v22.s }[2], [x9], #0x4\n"
      "ld1 { v23.s }[2], [x26], #0x4\n"
      "ld1 { v21.s }[2], [x25], #0x4\n"
      "ld1 { v20.s }[2], [x24], #0x4\n"
      "ld1 { v18.s }[2], [x23], #0x4\n"
      "ld1 { v17.s }[2], [x22], #0x4\n"
      "ld1 { v19.s }[2], [x21], #0x4\n"
      "ld1 { v16.s }[2], [x20], #0x4\n"
      "zip1 v25.16b, v22.16b, v18.16b\n"
      "zip1 v24.16b, v23.16b, v17.16b\n"
      "zip2 v22.16b, v22.16b, v18.16b\n"
      "zip2 v23.16b, v23.16b, v17.16b\n"
      "zip1 v18.16b, v21.16b, v19.16b\n"
      "zip1 v17.16b, v20.16b, v16.16b\n"
      "zip2 v21.16b, v21.16b, v19.16b\n"
      "zip2 v16.16b, v20.16b, v16.16b\n"
      "zip1 v20.16b, v25.16b, v18.16b\n"
      "zip1 v19.16b, v24.16b, v17.16b\n"
      "zip2 v18.16b, v25.16b, v18.16b\n"
      "zip2 v17.16b, v24.16b, v17.16b\n"
      "zip1 v22.16b, v22.16b, v21.16b\n"
      "zip1 v16.16b, v23.16b, v16.16b\n"
      "zip1 v21.16b, v20.16b, v19.16b\n"
      "zip2 v20.16b, v20.16b, v19.16b\n"
      "zip1 v19.16b, v18.16b, v17.16b\n"
      "zip2 v18.16b, v18.16b, v17.16b\n"
      "zip1 v17.16b, v22.16b, v16.16b\n"
      "zip2 v16.16b, v22.16b, v16.16b\n"
      "str q21, [x27, #0]\n"
      "str q20, [x27, #0x10]\n"
      "str q19, [x27, #0x20]\n"
      "str q18, [x27, #0x30]\n"
      "str q17, [x27, #0x40]\n"
      "str q16, [x27, #0x50]\n"
      "add x27, x27, %x[out_stride]\n"
      "bge 4b\n"
      "5:"  // Main row loop: Column loop skip
      "cbz x28, 10f\n"
      "cmp x28, #0x4\n"
      "movi v16.16b, #0\n"
      "str q16, [x27, #0]\n"
      "str q16, [x27, #0x10]\n"
      "str q16, [x27, #0x20]\n"
      "str q16, [x27, #0x30]\n"
      "str q16, [x27, #0x40]\n"
      "str q16, [x27, #0x50]\n"
      "blt 7f\n"
      "6:"  // Main row loop: width 4 loop: loop
      "ldr s23, [x9], #0x4\n"
      "ldr s22, [x26], #0x4\n"
      "sub x28, x28, #0x4\n"
      "ldr s21, [x25], #0x4\n"
      "ldr s20, [x24], #0x4\n"
      "cmp x28, #0x4\n"
      "ldr s18, [x23], #0x4\n"
      "ldr s19, [x22], #0x4\n"
      "ldr s17, [x21], #0x4\n"
      "ldr s16, [x20], #0x4\n"
      "zip1 v18.16b, v23.16b, v18.16b\n"
      "zip1 v19.16b, v22.16b, v19.16b\n"
      "zip1 v17.16b, v21.16b, v17.16b\n"
      "zip1 v16.16b, v20.16b, v16.16b\n"
      "zip1 v18.16b, v18.16b, v17.16b\n"
      "zip1 v16.16b, v19.16b, v16.16b\n"
      "zip1 v17.16b, v18.16b, v16.16b\n"
      "zip2 v16.16b, v18.16b, v16.16b\n"
      "str q17, [x27, #0]\n"
      "str q16, [x27, #0x10]\n"
      "add x27, x27, #0x20\n"
      "bge 6b\n"
      "7:"  // Main row loop: width 4 loop: skip
      "cmp x28, #0x1\n"
      "blt 9f\n"
      "8:"  // Main row loop: width 1 loop: loop
      "ldr b23, [x9], #0x1\n"
      "ldr b22, [x26], #0x1\n"
      "sub x28, x28, #0x1\n"
      "ldr b21, [x25], #0x1\n"
      "ldr b20, [x24], #0x1\n"
      "cmp x28, #0x1\n"
      "ldr b19, [x23], #0x1\n"
      "ldr b18, [x22], #0x1\n"
      "ldr b17, [x21], #0x1\n"
      "ldr b16, [x20], #0x1\n"
      "zip1 v19.16b, v23.16b, v19.16b\n"
      "zip1 v18.16b, v22.16b, v18.16b\n"
      "zip1 v17.16b, v21.16b, v17.16b\n"
      "zip1 v16.16b, v20.16b, v16.16b\n"
      "zip1 v17.16b, v19.16b, v17.16b\n"
      "zip1 v16.16b, v18.16b, v16.16b\n"
      "zip1 v16.16b, v17.16b, v16.16b\n"
      "str d16, [x27, #0]\n"
      "add x27, x27, #0x8\n"
      "bge 8b\n"
      "9:"  // Main row loop: width 1 loop: skip
      "10:"  // Main row loop: odd col skip
      "cmp x10, #0x1\n"
      "add %x[out], %x[out], #0x60\n"
      "bge 1b\n"
      "11:"  // Done
      : [in] "+&r" (in), [out] "+&r" (out)
      : [height] "r" (height), [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [pad_row] "r" (pad_row), [width] "r" (width)
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x10", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
    );
}

} // anonymous namespace

template<>
void Transform<12, 8, true, VLType::None>(
    uint8_t *out, const uint8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    a64_transpose_interleave_12_1x8(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(uint8_t) / 1,
        stride * sizeof(uint8_t),
        (kmax-k0)
    );
}

template<>
void Transform<12, 8, true, VLType::None>(
    int8_t *out, const int8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    a64_transpose_interleave_12_1x8(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(int8_t) / 1,
        stride * sizeof(int8_t),
        (kmax-k0)
    );
}

#endif // __aarch64__
