//
// SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#ifdef __aarch64__

#include <cstddef>

namespace kai {
namespace ops {

void sve_ffinterleaved_fp32_mla_8x3VL(
    const float *Apanel,
    const float *Bpanel,
    size_t B_stride,
    float *Cpanel,
    int ablocks,
    size_t N,
    int K)
{

    struct KernelArgs {
        size_t K = {};
        const float *Bpanel = {};
        size_t N = {};
        size_t B_stride = {};
        const float *cur_B_ptr = {};
    } ka;

    ka.K = (K/1) - 1;
    ka.Bpanel = Bpanel;
    ka.N = N;
    ka.B_stride = B_stride;

    __asm__ __volatile__(
      "ptrue p0.b\n"
      "cbz %x[ablocks], 7f\n"
      "1:"  // Height loop
      "ldr x20, [%x[args_ptr], %[offsetof_Bpanel]]\n"
      "ldr x26, [%x[args_ptr], %[offsetof_N]]\n"
      "mov x25, %x[Apanel]\n"
      "str x20, [%x[args_ptr], %[offsetof_cur_B_ptr]]\n"
      "2:"  // Width loop
      "ldr x24, [%x[args_ptr], %[offsetof_cur_B_ptr]]\n"
      "ldr x20, [%x[args_ptr], %[offsetof_B_stride]]\n"
      "cntw x23, ALL, MUL #2\n"
      "mov %x[Apanel], x25\n"
      "add x22, x24, x20, LSL #2\n"
      "cmp x26, x23\n"
      "add x21, x22, x20, LSL #2\n"
      "add x20, x21, x20, LSL #2\n"
      "str x20, [%x[args_ptr], %[offsetof_cur_B_ptr]]\n"
      "bgt 3f\n"
      "decw x23\n"
      "mov x21, x24\n"
      "cmp x26, x23\n"
      "bgt 3f\n"
      "mov x22, x24\n"
      "3:"  // B setup done
      "ldr x20, [%x[args_ptr], %[offsetof_K]]\n"
      "mov z8.b, #0\n"
      "mov z9.b, #0\n"
      "mov z10.b, #0\n"
      "mov z11.b, #0\n"
      "ld1rqw { z0.s }, p0/Z, [%x[Apanel]]\n"
      "mov z12.b, #0\n"
      "mov z13.b, #0\n"
      "ld1rqw { z1.s }, p0/Z, [%x[Apanel], #16]\n"
      "cmp x20, #0x2\n"
      "mov z14.b, #0\n"
      "mov z15.b, #0\n"
      "mov z16.b, #0\n"
      "mov z17.b, #0\n"
      "ld1w { z4.s }, p0/Z, [x24]\n"
      "mov z18.b, #0\n"
      "mov z19.b, #0\n"
      "ld1w { z5.s }, p0/Z, [x22]\n"
      "mov z20.b, #0\n"
      "mov z21.b, #0\n"
      "ld1w { z6.s }, p0/Z, [x21]\n"
      "mov z22.b, #0\n"
      "mov z23.b, #0\n"
      "mov z24.b, #0\n"
      "mov z25.b, #0\n"
      "mov z26.b, #0\n"
      "mov z27.b, #0\n"
      "mov z28.b, #0\n"
      "mov z29.b, #0\n"
      "mov z30.b, #0\n"
      "mov z31.b, #0\n"
      "blt 5f\n"
      "4:"  // main loop head
      "fmla z8.s, z4.s, z0.s[0]\n"
      "fmla z11.s, z4.s, z0.s[1]\n"
      "ld1rqw { z2.s }, p0/Z, [%x[Apanel], #32]\n"
      "fmla z14.s, z4.s, z0.s[2]\n"
      "fmla z17.s, z4.s, z0.s[3]\n"
      "ld1rqw { z3.s }, p0/Z, [%x[Apanel], #48]\n"
      "fmla z20.s, z4.s, z1.s[0]\n"
      "fmla z23.s, z4.s, z1.s[1]\n"
      "sub x20, x20, #0x2\n"
      "fmla z26.s, z4.s, z1.s[2]\n"
      "fmla z29.s, z4.s, z1.s[3]\n"
      "ld1w { z4.s }, p0/Z, [x24, #1, MUL VL]\n"
      "fmla z9.s, z5.s, z0.s[0]\n"
      "fmla z12.s, z5.s, z0.s[1]\n"
      "addvl x24, x24, #2\n"
      "fmla z15.s, z5.s, z0.s[2]\n"
      "fmla z18.s, z5.s, z0.s[3]\n"
      "cmp x20, #0x2\n"
      "fmla z21.s, z5.s, z1.s[0]\n"
      "fmla z24.s, z5.s, z1.s[1]\n"
      "add %x[Apanel], %x[Apanel], #0x40\n"
      "fmla z27.s, z5.s, z1.s[2]\n"
      "fmla z30.s, z5.s, z1.s[3]\n"
      "ld1w { z5.s }, p0/Z, [x22, #1, MUL VL]\n"
      "fmla z10.s, z6.s, z0.s[0]\n"
      "fmla z13.s, z6.s, z0.s[1]\n"
      "addvl x22, x22, #2\n"
      "fmla z16.s, z6.s, z0.s[2]\n"
      "fmla z19.s, z6.s, z0.s[3]\n"
      "ld1rqw { z0.s }, p0/Z, [%x[Apanel]]\n"
      "fmla z22.s, z6.s, z1.s[0]\n"
      "fmla z25.s, z6.s, z1.s[1]\n"
      "fmla z28.s, z6.s, z1.s[2]\n"
      "fmla z31.s, z6.s, z1.s[3]\n"
      "ld1w { z6.s }, p0/Z, [x21, #1, MUL VL]\n"
      "addvl x21, x21, #2\n"
      "fmla z8.s, z4.s, z2.s[0]\n"
      "fmla z11.s, z4.s, z2.s[1]\n"
      "fmla z14.s, z4.s, z2.s[2]\n"
      "fmla z17.s, z4.s, z2.s[3]\n"
      "ld1rqw { z1.s }, p0/Z, [%x[Apanel], #16]\n"
      "fmla z20.s, z4.s, z3.s[0]\n"
      "fmla z23.s, z4.s, z3.s[1]\n"
      "fmla z26.s, z4.s, z3.s[2]\n"
      "fmla z29.s, z4.s, z3.s[3]\n"
      "ld1w { z4.s }, p0/Z, [x24]\n"
      "fmla z9.s, z5.s, z2.s[0]\n"
      "fmla z12.s, z5.s, z2.s[1]\n"
      "fmla z15.s, z5.s, z2.s[2]\n"
      "fmla z18.s, z5.s, z2.s[3]\n"
      "fmla z21.s, z5.s, z3.s[0]\n"
      "fmla z24.s, z5.s, z3.s[1]\n"
      "fmla z27.s, z5.s, z3.s[2]\n"
      "fmla z30.s, z5.s, z3.s[3]\n"
      "ld1w { z5.s }, p0/Z, [x22]\n"
      "fmla z10.s, z6.s, z2.s[0]\n"
      "fmla z13.s, z6.s, z2.s[1]\n"
      "fmla z16.s, z6.s, z2.s[2]\n"
      "fmla z19.s, z6.s, z2.s[3]\n"
      "fmla z22.s, z6.s, z3.s[0]\n"
      "fmla z25.s, z6.s, z3.s[1]\n"
      "fmla z28.s, z6.s, z3.s[2]\n"
      "fmla z31.s, z6.s, z3.s[3]\n"
      "ld1w { z6.s }, p0/Z, [x21]\n"
      "bge 4b\n"
      "5:"  // main loop skip
      "add %x[Apanel], %x[Apanel], #0x20\n"
      "addvl x24, x24, #1\n"
      "fmla z8.s, z4.s, z0.s[0]\n"
      "addvl x22, x22, #1\n"
      "addvl x21, x21, #1\n"
      "fmla z11.s, z4.s, z0.s[1]\n"
      "fmla z14.s, z4.s, z0.s[2]\n"
      "fmla z17.s, z4.s, z0.s[3]\n"
      "fmla z20.s, z4.s, z1.s[0]\n"
      "fmla z23.s, z4.s, z1.s[1]\n"
      "fmla z26.s, z4.s, z1.s[2]\n"
      "fmla z29.s, z4.s, z1.s[3]\n"
      "fmla z9.s, z5.s, z0.s[0]\n"
      "fmla z12.s, z5.s, z0.s[1]\n"
      "fmla z15.s, z5.s, z0.s[2]\n"
      "fmla z18.s, z5.s, z0.s[3]\n"
      "fmla z21.s, z5.s, z1.s[0]\n"
      "fmla z24.s, z5.s, z1.s[1]\n"
      "fmla z27.s, z5.s, z1.s[2]\n"
      "fmla z30.s, z5.s, z1.s[3]\n"
      "fmla z10.s, z6.s, z0.s[0]\n"
      "fmla z13.s, z6.s, z0.s[1]\n"
      "fmla z16.s, z6.s, z0.s[2]\n"
      "fmla z19.s, z6.s, z0.s[3]\n"
      "fmla z22.s, z6.s, z1.s[0]\n"
      "fmla z25.s, z6.s, z1.s[1]\n"
      "fmla z28.s, z6.s, z1.s[2]\n"
      "fmla z31.s, z6.s, z1.s[3]\n"
      "cbz x20, 6f\n"
      "ld1rqw { z0.s }, p0/Z, [%x[Apanel]]\n"
      "ld1rqw { z1.s }, p0/Z, [%x[Apanel], #16]\n"
      "add %x[Apanel], %x[Apanel], #0x20\n"
      "ld1w { z7.s }, p0/Z, [x24]\n"
      "ld1w { z4.s }, p0/Z, [x22]\n"
      "ld1w { z5.s }, p0/Z, [x21]\n"
      "fmla z8.s, z7.s, z0.s[0]\n"
      "fmla z11.s, z7.s, z0.s[1]\n"
      "fmla z14.s, z7.s, z0.s[2]\n"
      "fmla z17.s, z7.s, z0.s[3]\n"
      "fmla z20.s, z7.s, z1.s[0]\n"
      "fmla z23.s, z7.s, z1.s[1]\n"
      "fmla z26.s, z7.s, z1.s[2]\n"
      "fmla z29.s, z7.s, z1.s[3]\n"
      "fmla z9.s, z4.s, z0.s[0]\n"
      "fmla z12.s, z4.s, z0.s[1]\n"
      "fmla z15.s, z4.s, z0.s[2]\n"
      "fmla z18.s, z4.s, z0.s[3]\n"
      "fmla z21.s, z4.s, z1.s[0]\n"
      "fmla z24.s, z4.s, z1.s[1]\n"
      "fmla z27.s, z4.s, z1.s[2]\n"
      "fmla z30.s, z4.s, z1.s[3]\n"
      "fmla z10.s, z5.s, z0.s[0]\n"
      "fmla z13.s, z5.s, z0.s[1]\n"
      "fmla z16.s, z5.s, z0.s[2]\n"
      "fmla z19.s, z5.s, z0.s[3]\n"
      "fmla z22.s, z5.s, z1.s[0]\n"
      "fmla z25.s, z5.s, z1.s[1]\n"
      "fmla z28.s, z5.s, z1.s[2]\n"
      "fmla z31.s, z5.s, z1.s[3]\n"
      "6:"  // multiply loop done
      "decw x26, ALL, MUL #3\n"
      "st1w { z8.s }, p0, [%x[Cpanel]]\n"
      "st1w { z9.s }, p0, [%x[Cpanel], #1, MUL VL]\n"
      "st1w { z10.s }, p0, [%x[Cpanel], #2, MUL VL]\n"
      "st1w { z11.s }, p0, [%x[Cpanel], #3, MUL VL]\n"
      "cmp x26, XZR\n"
      "st1w { z12.s }, p0, [%x[Cpanel], #4, MUL VL]\n"
      "st1w { z13.s }, p0, [%x[Cpanel], #5, MUL VL]\n"
      "st1w { z14.s }, p0, [%x[Cpanel], #6, MUL VL]\n"
      "st1w { z15.s }, p0, [%x[Cpanel], #7, MUL VL]\n"
      "addvl %x[Cpanel], %x[Cpanel], #16\n"
      "st1w { z16.s }, p0, [%x[Cpanel], #-8, MUL VL]\n"
      "st1w { z17.s }, p0, [%x[Cpanel], #-7, MUL VL]\n"
      "st1w { z18.s }, p0, [%x[Cpanel], #-6, MUL VL]\n"
      "st1w { z19.s }, p0, [%x[Cpanel], #-5, MUL VL]\n"
      "st1w { z20.s }, p0, [%x[Cpanel], #-4, MUL VL]\n"
      "st1w { z21.s }, p0, [%x[Cpanel], #-3, MUL VL]\n"
      "st1w { z22.s }, p0, [%x[Cpanel], #-2, MUL VL]\n"
      "st1w { z23.s }, p0, [%x[Cpanel], #-1, MUL VL]\n"
      "st1w { z24.s }, p0, [%x[Cpanel]]\n"
      "st1w { z25.s }, p0, [%x[Cpanel], #1, MUL VL]\n"
      "st1w { z26.s }, p0, [%x[Cpanel], #2, MUL VL]\n"
      "st1w { z27.s }, p0, [%x[Cpanel], #3, MUL VL]\n"
      "st1w { z28.s }, p0, [%x[Cpanel], #4, MUL VL]\n"
      "st1w { z29.s }, p0, [%x[Cpanel], #5, MUL VL]\n"
      "st1w { z30.s }, p0, [%x[Cpanel], #6, MUL VL]\n"
      "st1w { z31.s }, p0, [%x[Cpanel], #7, MUL VL]\n"
      "addvl %x[Cpanel], %x[Cpanel], #8\n"
      "bgt 2b\n"
      "subs %x[ablocks], %x[ablocks], #0x1\n"
      "bgt 1b\n"
      "7:"  // Exit
      : [Apanel] "+&r" (Apanel), [Cpanel] "+&r" (Cpanel), [ablocks] "+&r" (ablocks)
      : [args_ptr] "r" (&ka), [offsetof_B_stride] "I" (offsetof(KernelArgs, B_stride)), [offsetof_Bpanel] "I" (offsetof(KernelArgs, Bpanel)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_N] "I" (offsetof(KernelArgs, N)), [offsetof_cur_B_ptr] "I" (offsetof(KernelArgs, cur_B_ptr))
      : "cc", "memory", "p0", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

} // namespace ops
} // namespace kai

#endif // __aarch64__
