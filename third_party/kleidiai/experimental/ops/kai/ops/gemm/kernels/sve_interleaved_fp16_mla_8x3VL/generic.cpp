//
// SPDX-FileCopyrightText: Copyright 2019-2021, 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#ifdef __aarch64__

#include <cstddef>

namespace kai {
namespace ops {

void sve_interleaved_fp16_mla_8x3VL(
    const __fp16 *Apanel,
    const __fp16 *Bpanel,
    __fp16 *Cpanel,
    int ablocks,
    int bblocks,
    int K)
{

    struct KernelArgs {
        size_t K = {};
        const __fp16 *Bpanel = {};
        size_t bblocks = {};
    } ka;

    ka.K = (K/1) - 1;
    ka.Bpanel = Bpanel;
    ka.bblocks = bblocks;

    __asm__ __volatile__(
      "ptrue p0.b\n"
      "cbz %x[ablocks], 6f\n"
      "1:"  // Height loop
      "ldr x23, [%x[args_ptr], %[offsetof_bblocks]]\n"
      "ldr x22, [%x[args_ptr], %[offsetof_Bpanel]]\n"
      "mov x21, %x[Apanel]\n"
      "2:"  // Width loop
      "ldr x20, [%x[args_ptr], %[offsetof_K]]\n"
      "mov %x[Apanel], x21\n"
      "mov z8.b, #0\n"
      "mov z9.b, #0\n"
      "mov z10.b, #0\n"
      "ld1h { z2.h }, p0/Z, [x22]\n"
      "mov z11.b, #0\n"
      "mov z12.b, #0\n"
      "ld1h { z3.h }, p0/Z, [x22, #1, MUL VL]\n"
      "cmp x20, #0x2\n"
      "mov z13.b, #0\n"
      "mov z14.b, #0\n"
      "mov z15.b, #0\n"
      "mov z16.b, #0\n"
      "ld1rqh { z0.h }, p0/Z, [%x[Apanel]]\n"
      "mov z17.b, #0\n"
      "mov z18.b, #0\n"
      "ld1h { z4.h }, p0/Z, [x22, #2, MUL VL]\n"
      "mov z19.b, #0\n"
      "mov z20.b, #0\n"
      "mov z21.b, #0\n"
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
      "blt 4f\n"
      "3:"  // main loop head
      "fmla z8.h, z2.h, z0.h[0]\n"
      "fmla z11.h, z2.h, z0.h[1]\n"
      "ld1rqh { z1.h }, p0/Z, [%x[Apanel], #16]\n"
      "fmla z14.h, z2.h, z0.h[2]\n"
      "fmla z17.h, z2.h, z0.h[3]\n"
      "ld1h { z5.h }, p0/Z, [x22, #3, MUL VL]\n"
      "fmla z20.h, z2.h, z0.h[4]\n"
      "fmla z23.h, z2.h, z0.h[5]\n"
      "ld1h { z6.h }, p0/Z, [x22, #4, MUL VL]\n"
      "fmla z26.h, z2.h, z0.h[6]\n"
      "fmla z29.h, z2.h, z0.h[7]\n"
      "ld1h { z7.h }, p0/Z, [x22, #5, MUL VL]\n"
      "fmla z9.h, z3.h, z0.h[0]\n"
      "fmla z12.h, z3.h, z0.h[1]\n"
      "addvl x22, x22, #6\n"
      "fmla z15.h, z3.h, z0.h[2]\n"
      "fmla z18.h, z3.h, z0.h[3]\n"
      "sub x20, x20, #0x2\n"
      "fmla z21.h, z3.h, z0.h[4]\n"
      "fmla z24.h, z3.h, z0.h[5]\n"
      "cmp x20, #0x2\n"
      "fmla z27.h, z3.h, z0.h[6]\n"
      "fmla z30.h, z3.h, z0.h[7]\n"
      "add %x[Apanel], %x[Apanel], #0x20\n"
      "fmla z10.h, z4.h, z0.h[0]\n"
      "fmla z13.h, z4.h, z0.h[1]\n"
      "ld1h { z2.h }, p0/Z, [x22]\n"
      "fmla z16.h, z4.h, z0.h[2]\n"
      "fmla z19.h, z4.h, z0.h[3]\n"
      "ld1h { z3.h }, p0/Z, [x22, #1, MUL VL]\n"
      "fmla z22.h, z4.h, z0.h[4]\n"
      "fmla z25.h, z4.h, z0.h[5]\n"
      "fmla z28.h, z4.h, z0.h[6]\n"
      "fmla z31.h, z4.h, z0.h[7]\n"
      "ld1rqh { z0.h }, p0/Z, [%x[Apanel]]\n"
      "fmla z8.h, z5.h, z1.h[0]\n"
      "fmla z11.h, z5.h, z1.h[1]\n"
      "ld1h { z4.h }, p0/Z, [x22, #2, MUL VL]\n"
      "fmla z14.h, z5.h, z1.h[2]\n"
      "fmla z17.h, z5.h, z1.h[3]\n"
      "fmla z20.h, z5.h, z1.h[4]\n"
      "fmla z23.h, z5.h, z1.h[5]\n"
      "fmla z26.h, z5.h, z1.h[6]\n"
      "fmla z29.h, z5.h, z1.h[7]\n"
      "fmla z9.h, z6.h, z1.h[0]\n"
      "fmla z12.h, z6.h, z1.h[1]\n"
      "fmla z15.h, z6.h, z1.h[2]\n"
      "fmla z18.h, z6.h, z1.h[3]\n"
      "fmla z21.h, z6.h, z1.h[4]\n"
      "fmla z24.h, z6.h, z1.h[5]\n"
      "fmla z27.h, z6.h, z1.h[6]\n"
      "fmla z30.h, z6.h, z1.h[7]\n"
      "fmla z10.h, z7.h, z1.h[0]\n"
      "fmla z13.h, z7.h, z1.h[1]\n"
      "fmla z16.h, z7.h, z1.h[2]\n"
      "fmla z19.h, z7.h, z1.h[3]\n"
      "fmla z22.h, z7.h, z1.h[4]\n"
      "fmla z25.h, z7.h, z1.h[5]\n"
      "fmla z28.h, z7.h, z1.h[6]\n"
      "fmla z31.h, z7.h, z1.h[7]\n"
      "bge 3b\n"
      "4:"  // main loop skip
      "add %x[Apanel], %x[Apanel], #0x10\n"
      "addvl x22, x22, #3\n"
      "fmla z8.h, z2.h, z0.h[0]\n"
      "fmla z11.h, z2.h, z0.h[1]\n"
      "fmla z14.h, z2.h, z0.h[2]\n"
      "fmla z17.h, z2.h, z0.h[3]\n"
      "fmla z20.h, z2.h, z0.h[4]\n"
      "fmla z23.h, z2.h, z0.h[5]\n"
      "fmla z26.h, z2.h, z0.h[6]\n"
      "fmla z29.h, z2.h, z0.h[7]\n"
      "fmla z9.h, z3.h, z0.h[0]\n"
      "fmla z12.h, z3.h, z0.h[1]\n"
      "fmla z15.h, z3.h, z0.h[2]\n"
      "fmla z18.h, z3.h, z0.h[3]\n"
      "fmla z21.h, z3.h, z0.h[4]\n"
      "fmla z24.h, z3.h, z0.h[5]\n"
      "fmla z27.h, z3.h, z0.h[6]\n"
      "fmla z30.h, z3.h, z0.h[7]\n"
      "fmla z10.h, z4.h, z0.h[0]\n"
      "fmla z13.h, z4.h, z0.h[1]\n"
      "fmla z16.h, z4.h, z0.h[2]\n"
      "fmla z19.h, z4.h, z0.h[3]\n"
      "fmla z22.h, z4.h, z0.h[4]\n"
      "fmla z25.h, z4.h, z0.h[5]\n"
      "fmla z28.h, z4.h, z0.h[6]\n"
      "fmla z31.h, z4.h, z0.h[7]\n"
      "cbz x20, 5f\n"
      "ld1rqh { z0.h }, p0/Z, [%x[Apanel]]\n"
      "ld1h { z5.h }, p0/Z, [x22]\n"
      "add %x[Apanel], %x[Apanel], #0x10\n"
      "ld1h { z6.h }, p0/Z, [x22, #1, MUL VL]\n"
      "ld1h { z7.h }, p0/Z, [x22, #2, MUL VL]\n"
      "addvl x22, x22, #3\n"
      "fmla z8.h, z5.h, z0.h[0]\n"
      "fmla z11.h, z5.h, z0.h[1]\n"
      "fmla z14.h, z5.h, z0.h[2]\n"
      "fmla z17.h, z5.h, z0.h[3]\n"
      "fmla z20.h, z5.h, z0.h[4]\n"
      "fmla z23.h, z5.h, z0.h[5]\n"
      "fmla z26.h, z5.h, z0.h[6]\n"
      "fmla z29.h, z5.h, z0.h[7]\n"
      "fmla z9.h, z6.h, z0.h[0]\n"
      "fmla z12.h, z6.h, z0.h[1]\n"
      "fmla z15.h, z6.h, z0.h[2]\n"
      "fmla z18.h, z6.h, z0.h[3]\n"
      "fmla z21.h, z6.h, z0.h[4]\n"
      "fmla z24.h, z6.h, z0.h[5]\n"
      "fmla z27.h, z6.h, z0.h[6]\n"
      "fmla z30.h, z6.h, z0.h[7]\n"
      "fmla z10.h, z7.h, z0.h[0]\n"
      "fmla z13.h, z7.h, z0.h[1]\n"
      "fmla z16.h, z7.h, z0.h[2]\n"
      "fmla z19.h, z7.h, z0.h[3]\n"
      "fmla z22.h, z7.h, z0.h[4]\n"
      "fmla z25.h, z7.h, z0.h[5]\n"
      "fmla z28.h, z7.h, z0.h[6]\n"
      "fmla z31.h, z7.h, z0.h[7]\n"
      "5:"  // multiply loop done
      "st1h { z8.h }, p0, [%x[Cpanel]]\n"
      "subs x23, x23, #0x1\n"
      "st1h { z9.h }, p0, [%x[Cpanel], #1, MUL VL]\n"
      "st1h { z10.h }, p0, [%x[Cpanel], #2, MUL VL]\n"
      "st1h { z11.h }, p0, [%x[Cpanel], #3, MUL VL]\n"
      "st1h { z12.h }, p0, [%x[Cpanel], #4, MUL VL]\n"
      "st1h { z13.h }, p0, [%x[Cpanel], #5, MUL VL]\n"
      "st1h { z14.h }, p0, [%x[Cpanel], #6, MUL VL]\n"
      "st1h { z15.h }, p0, [%x[Cpanel], #7, MUL VL]\n"
      "addvl %x[Cpanel], %x[Cpanel], #16\n"
      "st1h { z16.h }, p0, [%x[Cpanel], #-8, MUL VL]\n"
      "st1h { z17.h }, p0, [%x[Cpanel], #-7, MUL VL]\n"
      "st1h { z18.h }, p0, [%x[Cpanel], #-6, MUL VL]\n"
      "st1h { z19.h }, p0, [%x[Cpanel], #-5, MUL VL]\n"
      "st1h { z20.h }, p0, [%x[Cpanel], #-4, MUL VL]\n"
      "st1h { z21.h }, p0, [%x[Cpanel], #-3, MUL VL]\n"
      "st1h { z22.h }, p0, [%x[Cpanel], #-2, MUL VL]\n"
      "st1h { z23.h }, p0, [%x[Cpanel], #-1, MUL VL]\n"
      "st1h { z24.h }, p0, [%x[Cpanel]]\n"
      "st1h { z25.h }, p0, [%x[Cpanel], #1, MUL VL]\n"
      "st1h { z26.h }, p0, [%x[Cpanel], #2, MUL VL]\n"
      "st1h { z27.h }, p0, [%x[Cpanel], #3, MUL VL]\n"
      "st1h { z28.h }, p0, [%x[Cpanel], #4, MUL VL]\n"
      "st1h { z29.h }, p0, [%x[Cpanel], #5, MUL VL]\n"
      "st1h { z30.h }, p0, [%x[Cpanel], #6, MUL VL]\n"
      "st1h { z31.h }, p0, [%x[Cpanel], #7, MUL VL]\n"
      "addvl %x[Cpanel], %x[Cpanel], #8\n"
      "bgt 2b\n"
      "subs %x[ablocks], %x[ablocks], #0x1\n"
      "bgt 1b\n"
      "6:"  // Exit
      : [Apanel] "+&r" (Apanel), [Cpanel] "+&r" (Cpanel), [ablocks] "+&r" (ablocks)
      : [args_ptr] "r" (&ka), [offsetof_Bpanel] "I" (offsetof(KernelArgs, Bpanel)), [offsetof_K] "I" (offsetof(KernelArgs, K)), [offsetof_bblocks] "I" (offsetof(KernelArgs, bblocks))
      : "cc", "memory", "p0", "x20", "x21", "x22", "x23", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
    );
}

} // namespace ops
} // namespace kai

#endif // __aarch64__
