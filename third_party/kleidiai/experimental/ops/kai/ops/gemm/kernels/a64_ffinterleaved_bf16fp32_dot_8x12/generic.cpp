//
// SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#ifdef __aarch64__

#include <cstddef>
#include "kai/ops/bfloat.hpp"

namespace kai {
namespace ops {

void a64_ffinterleaved_bf16fp32_dot_8x12(
    const bfloat16 *Apanel,
    const bfloat16 *Bpanel,
    size_t B_stride,
    float *Cpanel,
    int ablocks,
    size_t N,
    int K)
{

    struct KernelArgs {
        size_t K = {};
        const bfloat16 *Bpanel = {};
        size_t N = {};
        size_t B_stride = {};
        const bfloat16 *cur_B_ptr = {};
    } ka;

    ka.K = (K/2) - 1;
    ka.Bpanel = Bpanel;
    ka.N = N;
    ka.B_stride = B_stride;

    __asm__ __volatile__(
      "cbz %x[ablocks], 7f\n"
      "1:"  // Height loop
      "ldr x20, [%x[args_ptr], %[offsetof_Bpanel]]\n"
      "ldr x25, [%x[args_ptr], %[offsetof_N]]\n"
      "mov x24, %x[Apanel]\n"
      "str x20, [%x[args_ptr], %[offsetof_cur_B_ptr]]\n"
      "2:"  // Width loop
      "ldr x23, [%x[args_ptr], %[offsetof_cur_B_ptr]]\n"
      "ldr x20, [%x[args_ptr], %[offsetof_B_stride]]\n"
      "cmp x25, #0x8\n"
      "mov %x[Apanel], x24\n"
      "add x22, x23, x20, LSL #1\n"
      "add x21, x22, x20, LSL #1\n"
      "add x20, x21, x20, LSL #1\n"
      "str x20, [%x[args_ptr], %[offsetof_cur_B_ptr]]\n"
      "bgt 3f\n"
      "cmp x25, #0x4\n"
      "mov x21, x23\n"
      "bgt 3f\n"
      "mov x22, x23\n"
      "3:"  // B setup done
      "ldr q0, [%x[Apanel], #0]\n"
      "ldr q1, [%x[Apanel], #0x10]\n"
      "movi v8.16b, #0\n"
      "ldr q4, [x23, #0]\n"
      "ldr q5, [x22, #0]\n"
      "movi v9.16b, #0\n"
      "ldr q6, [x21, #0]\n"
      "ldr x20, [%x[args_ptr], %[offsetof_K]]\n"
      "movi v10.16b, #0\n"
      "movi v11.16b, #0\n"
      "movi v12.16b, #0\n"
      "movi v13.16b, #0\n"
      "movi v14.16b, #0\n"
      "cmp x20, #0x2\n"
      "movi v15.16b, #0\n"
      "movi v16.16b, #0\n"
      "movi v17.16b, #0\n"
      "movi v18.16b, #0\n"
      "movi v19.16b, #0\n"
      "movi v20.16b, #0\n"
      "movi v21.16b, #0\n"
      "movi v22.16b, #0\n"
      "movi v23.16b, #0\n"
      "movi v24.16b, #0\n"
      "movi v25.16b, #0\n"
      "movi v26.16b, #0\n"
      "movi v27.16b, #0\n"
      "movi v28.16b, #0\n"
      "movi v29.16b, #0\n"
      "movi v30.16b, #0\n"
      "movi v31.16b, #0\n"
      "blt 5f\n"
      "4:"  // main loop head
      "ldr q2, [%x[Apanel], #0x20]\n"
      "ldr q3, [%x[Apanel], #0x30]\n"
      ".inst 0x4f40f088  // bfdot v8.4s, v4.8h, v0.h[0]\n"
      ".inst 0x4f60f08b  // bfdot v11.4s, v4.8h, v0.h[1]\n"
      ".inst 0x4f40f88e  // bfdot v14.4s, v4.8h, v0.h[2]\n"
      "sub x20, x20, #0x2\n"
      ".inst 0x4f60f891  // bfdot v17.4s, v4.8h, v0.h[3]\n"
      ".inst 0x4f41f094  // bfdot v20.4s, v4.8h, v1.h[0]\n"
      "cmp x20, #0x2\n"
      ".inst 0x4f61f097  // bfdot v23.4s, v4.8h, v1.h[1]\n"
      ".inst 0x4f41f89a  // bfdot v26.4s, v4.8h, v1.h[2]\n"
      "add %x[Apanel], %x[Apanel], #0x40\n"
      ".inst 0x4f61f89d  // bfdot v29.4s, v4.8h, v1.h[3]\n"
      "ldr q4, [x23, #0x10]\n"
      ".inst 0x4f40f0a9  // bfdot v9.4s, v5.8h, v0.h[0]\n"
      ".inst 0x4f60f0ac  // bfdot v12.4s, v5.8h, v0.h[1]\n"
      ".inst 0x4f40f8af  // bfdot v15.4s, v5.8h, v0.h[2]\n"
      "add x23, x23, #0x20\n"
      ".inst 0x4f60f8b2  // bfdot v18.4s, v5.8h, v0.h[3]\n"
      ".inst 0x4f41f0b5  // bfdot v21.4s, v5.8h, v1.h[0]\n"
      ".inst 0x4f61f0b8  // bfdot v24.4s, v5.8h, v1.h[1]\n"
      ".inst 0x4f41f8bb  // bfdot v27.4s, v5.8h, v1.h[2]\n"
      ".inst 0x4f61f8be  // bfdot v30.4s, v5.8h, v1.h[3]\n"
      "ldr q5, [x22, #0x10]\n"
      ".inst 0x4f40f0ca  // bfdot v10.4s, v6.8h, v0.h[0]\n"
      ".inst 0x4f60f0cd  // bfdot v13.4s, v6.8h, v0.h[1]\n"
      ".inst 0x4f40f8d0  // bfdot v16.4s, v6.8h, v0.h[2]\n"
      "add x22, x22, #0x20\n"
      ".inst 0x4f60f8d3  // bfdot v19.4s, v6.8h, v0.h[3]\n"
      "ldr q0, [%x[Apanel], #0]\n"
      ".inst 0x4f41f0d6  // bfdot v22.4s, v6.8h, v1.h[0]\n"
      ".inst 0x4f61f0d9  // bfdot v25.4s, v6.8h, v1.h[1]\n"
      ".inst 0x4f41f8dc  // bfdot v28.4s, v6.8h, v1.h[2]\n"
      ".inst 0x4f61f8df  // bfdot v31.4s, v6.8h, v1.h[3]\n"
      "ldr q6, [x21, #0x10]\n"
      "ldr q1, [%x[Apanel], #0x10]\n"
      "add x21, x21, #0x20\n"
      ".inst 0x4f42f088  // bfdot v8.4s, v4.8h, v2.h[0]\n"
      ".inst 0x4f62f08b  // bfdot v11.4s, v4.8h, v2.h[1]\n"
      ".inst 0x4f42f88e  // bfdot v14.4s, v4.8h, v2.h[2]\n"
      ".inst 0x4f62f891  // bfdot v17.4s, v4.8h, v2.h[3]\n"
      ".inst 0x4f43f094  // bfdot v20.4s, v4.8h, v3.h[0]\n"
      ".inst 0x4f63f097  // bfdot v23.4s, v4.8h, v3.h[1]\n"
      ".inst 0x4f43f89a  // bfdot v26.4s, v4.8h, v3.h[2]\n"
      ".inst 0x4f63f89d  // bfdot v29.4s, v4.8h, v3.h[3]\n"
      "ldr q4, [x23, #0]\n"
      ".inst 0x4f42f0a9  // bfdot v9.4s, v5.8h, v2.h[0]\n"
      ".inst 0x4f62f0ac  // bfdot v12.4s, v5.8h, v2.h[1]\n"
      ".inst 0x4f42f8af  // bfdot v15.4s, v5.8h, v2.h[2]\n"
      ".inst 0x4f62f8b2  // bfdot v18.4s, v5.8h, v2.h[3]\n"
      ".inst 0x4f43f0b5  // bfdot v21.4s, v5.8h, v3.h[0]\n"
      ".inst 0x4f63f0b8  // bfdot v24.4s, v5.8h, v3.h[1]\n"
      ".inst 0x4f43f8bb  // bfdot v27.4s, v5.8h, v3.h[2]\n"
      ".inst 0x4f63f8be  // bfdot v30.4s, v5.8h, v3.h[3]\n"
      "ldr q5, [x22, #0]\n"
      ".inst 0x4f42f0ca  // bfdot v10.4s, v6.8h, v2.h[0]\n"
      ".inst 0x4f62f0cd  // bfdot v13.4s, v6.8h, v2.h[1]\n"
      ".inst 0x4f42f8d0  // bfdot v16.4s, v6.8h, v2.h[2]\n"
      ".inst 0x4f62f8d3  // bfdot v19.4s, v6.8h, v2.h[3]\n"
      ".inst 0x4f43f0d6  // bfdot v22.4s, v6.8h, v3.h[0]\n"
      ".inst 0x4f63f0d9  // bfdot v25.4s, v6.8h, v3.h[1]\n"
      ".inst 0x4f43f8dc  // bfdot v28.4s, v6.8h, v3.h[2]\n"
      ".inst 0x4f63f8df  // bfdot v31.4s, v6.8h, v3.h[3]\n"
      "ldr q6, [x21, #0]\n"
      "bge 4b\n"
      "5:"  // main loop skip
      "add %x[Apanel], %x[Apanel], #0x20\n"
      "add x23, x23, #0x10\n"
      ".inst 0x4f40f088  // bfdot v8.4s, v4.8h, v0.h[0]\n"
      "add x22, x22, #0x10\n"
      "add x21, x21, #0x10\n"
      ".inst 0x4f60f08b  // bfdot v11.4s, v4.8h, v0.h[1]\n"
      ".inst 0x4f40f88e  // bfdot v14.4s, v4.8h, v0.h[2]\n"
      ".inst 0x4f60f891  // bfdot v17.4s, v4.8h, v0.h[3]\n"
      ".inst 0x4f41f094  // bfdot v20.4s, v4.8h, v1.h[0]\n"
      ".inst 0x4f61f097  // bfdot v23.4s, v4.8h, v1.h[1]\n"
      ".inst 0x4f41f89a  // bfdot v26.4s, v4.8h, v1.h[2]\n"
      ".inst 0x4f61f89d  // bfdot v29.4s, v4.8h, v1.h[3]\n"
      ".inst 0x4f40f0a9  // bfdot v9.4s, v5.8h, v0.h[0]\n"
      ".inst 0x4f60f0ac  // bfdot v12.4s, v5.8h, v0.h[1]\n"
      ".inst 0x4f40f8af  // bfdot v15.4s, v5.8h, v0.h[2]\n"
      ".inst 0x4f60f8b2  // bfdot v18.4s, v5.8h, v0.h[3]\n"
      ".inst 0x4f41f0b5  // bfdot v21.4s, v5.8h, v1.h[0]\n"
      ".inst 0x4f61f0b8  // bfdot v24.4s, v5.8h, v1.h[1]\n"
      ".inst 0x4f41f8bb  // bfdot v27.4s, v5.8h, v1.h[2]\n"
      ".inst 0x4f61f8be  // bfdot v30.4s, v5.8h, v1.h[3]\n"
      ".inst 0x4f40f0ca  // bfdot v10.4s, v6.8h, v0.h[0]\n"
      ".inst 0x4f60f0cd  // bfdot v13.4s, v6.8h, v0.h[1]\n"
      ".inst 0x4f40f8d0  // bfdot v16.4s, v6.8h, v0.h[2]\n"
      ".inst 0x4f60f8d3  // bfdot v19.4s, v6.8h, v0.h[3]\n"
      ".inst 0x4f41f0d6  // bfdot v22.4s, v6.8h, v1.h[0]\n"
      ".inst 0x4f61f0d9  // bfdot v25.4s, v6.8h, v1.h[1]\n"
      ".inst 0x4f41f8dc  // bfdot v28.4s, v6.8h, v1.h[2]\n"
      ".inst 0x4f61f8df  // bfdot v31.4s, v6.8h, v1.h[3]\n"
      "cbz x20, 6f\n"
      "ldr q0, [%x[Apanel], #0]\n"
      "ldr q1, [%x[Apanel], #0x10]\n"
      "add %x[Apanel], %x[Apanel], #0x20\n"
      "ldr q7, [x23, #0]\n"
      "ldr q4, [x22, #0]\n"
      "ldr q5, [x21, #0]\n"
      ".inst 0x4f40f0e8  // bfdot v8.4s, v7.8h, v0.h[0]\n"
      ".inst 0x4f60f0eb  // bfdot v11.4s, v7.8h, v0.h[1]\n"
      ".inst 0x4f40f8ee  // bfdot v14.4s, v7.8h, v0.h[2]\n"
      ".inst 0x4f60f8f1  // bfdot v17.4s, v7.8h, v0.h[3]\n"
      ".inst 0x4f41f0f4  // bfdot v20.4s, v7.8h, v1.h[0]\n"
      ".inst 0x4f61f0f7  // bfdot v23.4s, v7.8h, v1.h[1]\n"
      ".inst 0x4f41f8fa  // bfdot v26.4s, v7.8h, v1.h[2]\n"
      ".inst 0x4f61f8fd  // bfdot v29.4s, v7.8h, v1.h[3]\n"
      ".inst 0x4f40f089  // bfdot v9.4s, v4.8h, v0.h[0]\n"
      ".inst 0x4f60f08c  // bfdot v12.4s, v4.8h, v0.h[1]\n"
      ".inst 0x4f40f88f  // bfdot v15.4s, v4.8h, v0.h[2]\n"
      ".inst 0x4f60f892  // bfdot v18.4s, v4.8h, v0.h[3]\n"
      ".inst 0x4f41f095  // bfdot v21.4s, v4.8h, v1.h[0]\n"
      ".inst 0x4f61f098  // bfdot v24.4s, v4.8h, v1.h[1]\n"
      ".inst 0x4f41f89b  // bfdot v27.4s, v4.8h, v1.h[2]\n"
      ".inst 0x4f61f89e  // bfdot v30.4s, v4.8h, v1.h[3]\n"
      ".inst 0x4f40f0aa  // bfdot v10.4s, v5.8h, v0.h[0]\n"
      ".inst 0x4f60f0ad  // bfdot v13.4s, v5.8h, v0.h[1]\n"
      ".inst 0x4f40f8b0  // bfdot v16.4s, v5.8h, v0.h[2]\n"
      ".inst 0x4f60f8b3  // bfdot v19.4s, v5.8h, v0.h[3]\n"
      ".inst 0x4f41f0b6  // bfdot v22.4s, v5.8h, v1.h[0]\n"
      ".inst 0x4f61f0b9  // bfdot v25.4s, v5.8h, v1.h[1]\n"
      ".inst 0x4f41f8bc  // bfdot v28.4s, v5.8h, v1.h[2]\n"
      ".inst 0x4f61f8bf  // bfdot v31.4s, v5.8h, v1.h[3]\n"
      "6:"  // multiply loop done
      "subs x25, x25, #0xc\n"
      "str q8, [%x[Cpanel], #0]\n"
      "str q9, [%x[Cpanel], #0x10]\n"
      "str q10, [%x[Cpanel], #0x20]\n"
      "str q11, [%x[Cpanel], #0x30]\n"
      "str q12, [%x[Cpanel], #0x40]\n"
      "str q13, [%x[Cpanel], #0x50]\n"
      "str q14, [%x[Cpanel], #0x60]\n"
      "str q15, [%x[Cpanel], #0x70]\n"
      "str q16, [%x[Cpanel], #0x80]\n"
      "str q17, [%x[Cpanel], #0x90]\n"
      "str q18, [%x[Cpanel], #0xa0]\n"
      "str q19, [%x[Cpanel], #0xb0]\n"
      "str q20, [%x[Cpanel], #0xc0]\n"
      "str q21, [%x[Cpanel], #0xd0]\n"
      "str q22, [%x[Cpanel], #0xe0]\n"
      "str q23, [%x[Cpanel], #0xf0]\n"
      "str q24, [%x[Cpanel], #0x100]\n"
      "str q25, [%x[Cpanel], #0x110]\n"
      "str q26, [%x[Cpanel], #0x120]\n"
      "str q27, [%x[Cpanel], #0x130]\n"
      "str q28, [%x[Cpanel], #0x140]\n"
      "str q29, [%x[Cpanel], #0x150]\n"
      "str q30, [%x[Cpanel], #0x160]\n"
      "str q31, [%x[Cpanel], #0x170]\n"
      "add %x[Cpanel], %x[Cpanel], #0x180\n"
      "bgt 2b\n"
      "subs %x[ablocks], %x[ablocks], #0x1\n"
      "bgt 1b\n"
      "7:"  // Exit
      : [Apanel] "+&r" (Apanel), [Cpanel] "+&r" (Cpanel), [ablocks] "+&r" (ablocks)
      : [args_ptr] "r" (&ka), [offsetof_B_stride] "I" (offsetof(KernelArgs, B_stride)), [offsetof_Bpanel] "I" (offsetof(KernelArgs, Bpanel)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_cur_B_ptr] "I" (offsetof(KernelArgs, cur_B_ptr))
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x20", "x21", "x22", "x23", "x24", "x25"
    );
}

} // namespace ops
} // namespace kai

#endif // __aarch64__
