//
// SPDX-FileCopyrightText: Copyright 2017-2018, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#ifdef __aarch64__

#include <arm_neon.h>

#include "asmlib.hpp"

namespace kai {
namespace ops {

void a64_gemm_u16_asimd_8x12(const uint16_t *Apanel, const uint16_t *Bpanel, uint32_t *Cpanel, int ablocks, int bblocks, int K)
{
  const uint16_t *a_ptr = Apanel;
  uint32_t *c_ptr = Cpanel;

  for (int yb = 0; yb < ablocks; yb++)
  {
    const uint16_t *a_ptr0 = a_ptr;
    const uint16_t *b_ptr = Bpanel;

    for (int xb = 0; xb < bblocks; xb++)
    {
      a_ptr = a_ptr0;
      const bool odd_k = K & 0x1;
      int k = (K+1)/2 - 1;

      register uint16x8_t aa asm("v0");
      register uint16x8_t ab asm("v1");
      register uint16x8_t b0 asm("v2");
      register uint16x8_t b1 asm("v3");
      register uint16x8_t b2 asm("v4");

      __asm __volatile (
        "ldr %d[aa], [%x[a_ptr]]\n"  // Load A[A].lower
        "movi v5.4s, #0\n"
        "ldr x20, [%x[a_ptr], #0x08]\n"  // Load A[A].upper
        "movi v6.4s, #0\n"
        "ldr %d[b0], [%x[b_ptr]]\n"  // Load B[0].lower
        "ins %[aa].d[1], x20\n"  // Merge A[A].lower and upper
        "movi v7.4s, #0\n"
        ASM_PREFETCH("[%[a_ptr], #64]")
        "movi v8.4s, #0\n"
        "ldr x20, [%x[b_ptr], #0x08]\n"  // Load B[0].upper
        "movi v9.4s, #0\n"
        ASM_PREFETCH("[%[b_ptr], #64]")
        "movi v10.4s, #0\n"
        "ldr %d[b1], [%x[b_ptr], #0x10]\n"  // Load B[1].lower
        "ins %[b0].d[1], x20\n"  // Merge B[0].lower and upper
        "movi v11.4s, #0\n"
        ASM_PREFETCH("[%[a_ptr], #96]")
        "movi v12.4s, #0\n"
        "movi v13.4s, #0\n"
        ASM_PREFETCH("[%[b_ptr], #96]")
        "movi v14.4s, #0\n"
        "movi v15.4s, #0\n"
        ASM_PREFETCH("[%[a_ptr], #128]")
        "movi v16.4s, #0\n"
        "movi v17.4s, #0\n"
        ASM_PREFETCH("[%[b_ptr], #128]")
        "movi v18.4s, #0\n"
        "movi v19.4s, #0\n"
        ASM_PREFETCH("[%[a_ptr], #160]")
        "movi v20.4s, #0\n"
        "movi v21.4s, #0\n"
        ASM_PREFETCH("[%[b_ptr], #160]")
        "movi v22.4s, #0\n"
        "movi v23.4s, #0\n"
        ASM_PREFETCH("[%[a_ptr], #192]")
        "movi v24.4s, #0\n"
        "add %x[a_ptr], %x[a_ptr], #0x10\n"
        "movi v25.4s, #0\n"
        ASM_PREFETCH("[%[b_ptr], #192]")
        "movi v26.4s, #0\n"
        "add %x[b_ptr], %x[b_ptr], #0x18\n"
        "movi v27.4s, #0\n"
        "movi v28.4s, #0\n"

        "cbz %x[k], 2f\n"  // Skip the loop if doing zero iterations.

        "1:\n"  // Main loop
          // First unroll
          "umlal v5.4s, %[b0].4h, %[aa].h[0]\n"
          "ldr x20, [%x[b_ptr]]\n"  // Load B[1].upper
          "umlal v6.4s, %[b0].4h, %[aa].h[1]\n"
          "umlal v7.4s, %[b0].4h, %[aa].h[2]\n"
          "ldr %d[ab], [%x[a_ptr]]\n"  // Load A[B].lower
          "ins %[b1].d[1], x20\n"  // Merge B[1].lower and .upper
          "umlal v8.4s, %[b0].4h, %[aa].h[3]\n"
          "umlal v9.4s, %[b0].4h, %[aa].h[4]\n"
          "ldr x20, [%x[a_ptr], #0x8]\n"  // Load A[B].upper
          "umlal v10.4s, %[b0].4h, %[aa].h[5]\n"
          "umlal v11.4s, %[b0].4h, %[aa].h[6]\n"
          "ldr %d[b2], [%x[b_ptr], #0x8]\n"  // Load B[2].lower
          "ins %[ab].d[1], x20\n"  // Merge A[B].lower and .upper
          "umlal v12.4s, %[b0].4h, %[aa].h[7]\n"
          "umlal2 v13.4s, %[b0].8h, %[aa].h[0]\n"
          "ldr x20, [%x[b_ptr], #0x10]\n"  // Load B[2].upper
          "umlal2 v14.4s, %[b0].8h, %[aa].h[1]\n"
          "umlal2 v15.4s, %[b0].8h, %[aa].h[2]\n"
          "umlal2 v16.4s, %[b0].8h, %[aa].h[3]\n"
          "umlal2 v17.4s, %[b0].8h, %[aa].h[4]\n"
          "umlal2 v18.4s, %[b0].8h, %[aa].h[5]\n"
          "umlal2 v19.4s, %[b0].8h, %[aa].h[6]\n"
          "umlal2 v20.4s, %[b0].8h, %[aa].h[7]\n"
          "ldr %d[b0], [%x[b_ptr], #0x18]\n"  // Load B[0].lower
          "ins %[b2].d[1], x20\n"  // Merge B[2].lower and .upper
          "umlal v21.4s, %[b1].4h, %[aa].h[0]\n"
          "umlal v22.4s, %[b1].4h, %[aa].h[1]\n"
          "ldr x20, [%x[b_ptr], #0x20]\n"  // Load B[0].upper
          "umlal v23.4s, %[b1].4h, %[aa].h[2]\n"
          "umlal v24.4s, %[b1].4h, %[aa].h[3]\n"
          "umlal v25.4s, %[b1].4h, %[aa].h[4]\n"
          "umlal v26.4s, %[b1].4h, %[aa].h[5]\n"
          "umlal v27.4s, %[b1].4h, %[aa].h[6]\n"
          "umlal v28.4s, %[b1].4h, %[aa].h[7]\n"

          // Second unroll
          "umlal2 v5.4s, %[b1].8h, %[ab].h[0]\n"
          "ldr %d[aa], [%x[a_ptr], #0x10]\n"  // Load A[A].lower
          "ins %[b0].d[1], x20\n"  // Merge B[0].lower and .upper
          "umlal2 v6.4s, %[b1].8h, %[ab].h[1]\n"
          "umlal2 v7.4s, %[b1].8h, %[ab].h[2]\n"
          "ldr x20, [%x[a_ptr], #0x18]\n"  // Load A[A].upper
          "umlal2 v8.4s, %[b1].8h, %[ab].h[3]\n"
          "umlal2 v9.4s, %[b1].8h, %[ab].h[4]\n"
          "umlal2 v10.4s, %[b1].8h, %[ab].h[5]\n"
          "umlal2 v11.4s, %[b1].8h, %[ab].h[6]\n"
          "add %x[a_ptr], %x[a_ptr], #0x20\n"
          "umlal2 v12.4s, %[b1].8h, %[ab].h[7]\n"
          "umlal v13.4s, %[b2].4h, %[ab].h[0]\n"
          ASM_PREFETCH("[%[b_ptr], #320]")
          "umlal v14.4s, %[b2].4h, %[ab].h[1]\n"
          "umlal v15.4s, %[b2].4h, %[ab].h[2]\n"
          ASM_PREFETCH("[%[a_ptr], #320]")
          "umlal v16.4s, %[b2].4h, %[ab].h[3]\n"
          "umlal v17.4s, %[b2].4h, %[ab].h[4]\n"
          ASM_PREFETCH("[%[b_ptr], #448]")
          "umlal v18.4s, %[b2].4h, %[ab].h[5]\n"
          "umlal v19.4s, %[b2].4h, %[ab].h[6]\n"
          "umlal v20.4s, %[b2].4h, %[ab].h[7]\n"
          "umlal2 v21.4s, %[b2].8h, %[ab].h[0]\n"
          "umlal2 v22.4s, %[b2].8h, %[ab].h[1]\n"
          "subs %x[k], %x[k], #0x1\n"
          "umlal2 v23.4s, %[b2].8h, %[ab].h[2]\n"
          "umlal2 v24.4s, %[b2].8h, %[ab].h[3]\n"
          "ldr %d[b1], [%x[b_ptr], #0x28]\n"  // Load B[1].lower
          "ins %[aa].d[1], x20\n"  // Merge A[A].lower and .upper
          "umlal2 v25.4s, %[b2].8h, %[ab].h[4]\n"
          "umlal2 v26.4s, %[b2].8h, %[ab].h[5]\n"
          "add %x[b_ptr], %x[b_ptr], #0x30\n"
          "umlal2 v27.4s, %[b2].8h, %[ab].h[6]\n"
          "umlal2 v28.4s, %[b2].8h, %[ab].h[7]\n"
          "bne 1b\n"

        "2:\n"  // Even tail
          "cbnz %x[odd_k], 3f\n"

          "umlal v5.4s, %[b0].4h, %[aa].h[0]\n"
          "ldr x20, [%x[b_ptr]]\n"  // Load B[1].upper
          "umlal v6.4s, %[b0].4h, %[aa].h[1]\n"
          "umlal v7.4s, %[b0].4h, %[aa].h[2]\n"
          "ldr %d[ab], [%x[a_ptr]]\n"  // Load A[B].lower
          "ins %[b1].d[1], x20\n"  // Merge B[1].lower and .upper
          "umlal v8.4s, %[b0].4h, %[aa].h[3]\n"
          "umlal v9.4s, %[b0].4h, %[aa].h[4]\n"
          "ldr x20, [%x[a_ptr], #0x8]\n"  // Load A[B].upper
          "umlal v10.4s, %[b0].4h, %[aa].h[5]\n"
          "umlal v11.4s, %[b0].4h, %[aa].h[6]\n"
          "ldr %d[b2], [%x[b_ptr], #0x8]\n"  // Load B[2].lower
          "ins %[ab].d[1], x20\n"  // Merge A[B].lower and .upper
          "umlal v12.4s, %[b0].4h, %[aa].h[7]\n"
          "umlal2 v13.4s, %[b0].8h, %[aa].h[0]\n"
          "ldr x20, [%x[b_ptr], #0x10]\n"  // Load B[2].upper
          "umlal2 v14.4s, %[b0].8h, %[aa].h[1]\n"
          "umlal2 v15.4s, %[b0].8h, %[aa].h[2]\n"
          "umlal2 v16.4s, %[b0].8h, %[aa].h[3]\n"
          "add %[a_ptr], %[a_ptr], #0x10\n"
          "umlal2 v17.4s, %[b0].8h, %[aa].h[4]\n"
          "add %[b_ptr], %[b_ptr], #0x18\n"
          "umlal2 v18.4s, %[b0].8h, %[aa].h[5]\n"
          "umlal2 v19.4s, %[b0].8h, %[aa].h[6]\n"
          "umlal2 v20.4s, %[b0].8h, %[aa].h[7]\n"
          "ins %[b2].d[1], x20\n"  // Merge B[2].lower and .upper
          "umlal v21.4s, %[b1].4h, %[aa].h[0]\n"
          "umlal v22.4s, %[b1].4h, %[aa].h[1]\n"
          "umlal v23.4s, %[b1].4h, %[aa].h[2]\n"
          "umlal v24.4s, %[b1].4h, %[aa].h[3]\n"
          "umlal v25.4s, %[b1].4h, %[aa].h[4]\n"
          "umlal v26.4s, %[b1].4h, %[aa].h[5]\n"
          "umlal v27.4s, %[b1].4h, %[aa].h[6]\n"
          "umlal v28.4s, %[b1].4h, %[aa].h[7]\n"

          "umlal2 v5.4s, %[b1].8h, %[ab].h[0]\n"
          "umlal v13.4s, %[b2].4h, %[ab].h[0]\n"
          "umlal2 v21.4s, %[b2].8h, %[ab].h[0]\n"
          "umlal2 v6.4s, %[b1].8h, %[ab].h[1]\n"
          "umlal v14.4s, %[b2].4h, %[ab].h[1]\n"
          "str q5, [%x[c_ptr]]\n"
          "umlal2 v22.4s, %[b2].8h, %[ab].h[1]\n"
          "str q13, [%x[c_ptr], #0x10]\n"
          "umlal2 v7.4s, %[b1].8h, %[ab].h[2]\n"
          "str q21, [%x[c_ptr], #0x20]\n"
          "umlal v15.4s, %[b2].4h, %[ab].h[2]\n"
          "str q6, [%x[c_ptr], #0x30]\n"
          "umlal2 v23.4s, %[b2].8h, %[ab].h[2]\n"
          "str q14, [%x[c_ptr], #0x40]\n"
          "umlal2 v8.4s, %[b1].8h, %[ab].h[3]\n"
          "str q22, [%x[c_ptr], #0x50]\n"
          "umlal v16.4s, %[b2].4h, %[ab].h[3]\n"
          "str q7, [%x[c_ptr], #0x60]\n"
          "umlal2 v24.4s, %[b2].8h, %[ab].h[3]\n"
          "str q15, [%x[c_ptr], #0x70]\n"
          "umlal2 v9.4s, %[b1].8h, %[ab].h[4]\n"
          "str q23, [%x[c_ptr], #0x80]\n"
          "umlal v17.4s, %[b2].4h, %[ab].h[4]\n"
          "str q8, [%x[c_ptr], #0x90]\n"
          "umlal2 v25.4s, %[b2].8h, %[ab].h[4]\n"
          "str q16, [%x[c_ptr], #0xa0]\n"
          "umlal2 v10.4s, %[b1].8h, %[ab].h[5]\n"
          "str q24, [%x[c_ptr], #0xb0]\n"
          "umlal v18.4s, %[b2].4h, %[ab].h[5]\n"
          "str q9, [%x[c_ptr], #0xc0]\n"
          "umlal2 v26.4s, %[b2].8h, %[ab].h[5]\n"
          "str q17, [%x[c_ptr], #0xd0]\n"
          "umlal2 v11.4s, %[b1].8h, %[ab].h[6]\n"
          "str q25, [%x[c_ptr], #0xe0]\n"
          "umlal v19.4s, %[b2].4h, %[ab].h[6]\n"
          "str q10, [%x[c_ptr], #0xf0]\n"
          "umlal2 v27.4s, %[b2].8h, %[ab].h[6]\n"
          "str q18, [%x[c_ptr], #0x100]\n"
          "umlal2 v12.4s, %[b1].8h, %[ab].h[7]\n"
          "str q26, [%x[c_ptr], #0x110]\n"
          "umlal v20.4s, %[b2].4h, %[ab].h[7]\n"
          "str q11, [%x[c_ptr], #0x120]\n"
          "umlal2 v28.4s, %[b2].8h, %[ab].h[7]\n"
          "str q19, [%x[c_ptr], #0x130]\n"
          "b 4f\n"  // Complete write out

        "3:\n"  // Odd tail
          "umlal v5.4s, %[b0].4h, %[aa].h[0]\n"
          "umlal2 v13.4s, %[b0].8h, %[aa].h[0]\n"
          "umlal v21.4s, %[b1].4h, %[aa].h[0]\n"
          "umlal v6.4s, %[b0].4h, %[aa].h[1]\n"
          "umlal2 v14.4s, %[b0].8h, %[aa].h[1]\n"
          "umlal v22.4s, %[b1].4h, %[aa].h[1]\n"
          "str q5, [%x[c_ptr]]\n"
          "umlal v7.4s, %[b0].4h, %[aa].h[2]\n"
          "str q13, [%x[c_ptr], #0x10]\n"
          "umlal2 v15.4s, %[b0].8h, %[aa].h[2]\n"
          "str q21, [%x[c_ptr], #0x20]\n"
          "umlal v23.4s, %[b1].4h, %[aa].h[2]\n"
          "str q6, [%x[c_ptr], #0x30]\n"
          "umlal v8.4s, %[b0].4h, %[aa].h[3]\n"
          "str q14, [%x[c_ptr], #0x40]\n"
          "umlal2 v16.4s, %[b0].8h, %[aa].h[3]\n"
          "str q22, [%x[c_ptr], #0x50]\n"
          "umlal v24.4s, %[b1].4h, %[aa].h[3]\n"
          "str q7, [%x[c_ptr], #0x60]\n"
          "umlal v9.4s, %[b0].4h, %[aa].h[4]\n"
          "str q15, [%x[c_ptr], #0x70]\n"
          "umlal2 v17.4s, %[b0].8h, %[aa].h[4]\n"
          "str q23, [%x[c_ptr], #0x80]\n"
          "umlal v25.4s, %[b1].4h, %[aa].h[4]\n"
          "str q8, [%x[c_ptr], #0x90]\n"
          "umlal v10.4s, %[b0].4h, %[aa].h[5]\n"
          "str q16, [%x[c_ptr], #0xa0]\n"
          "umlal2 v18.4s, %[b0].8h, %[aa].h[5]\n"
          "str q24, [%x[c_ptr], #0xb0]\n"
          "umlal v26.4s, %[b1].4h, %[aa].h[5]\n"
          "str q9, [%x[c_ptr], #0xc0]\n"
          "umlal v11.4s, %[b0].4h, %[aa].h[6]\n"
          "str q17, [%x[c_ptr], #0xd0]\n"
          "umlal2 v19.4s, %[b0].8h, %[aa].h[6]\n"
          "str q25, [%x[c_ptr], #0xe0]\n"
          "umlal v27.4s, %[b1].4h, %[aa].h[6]\n"
          "str q10, [%x[c_ptr], #0xf0]\n"
          "umlal v12.4s, %[b0].4h, %[aa].h[7]\n"
          "str q18, [%x[c_ptr], #0x100]\n"
          "umlal2 v20.4s, %[b0].8h, %[aa].h[7]\n"
          "str q26, [%x[c_ptr], #0x110]\n"
          "umlal v28.4s, %[b1].4h, %[aa].h[7]\n"
          "str q11, [%x[c_ptr], #0x120]\n"

        "4:\n"  // End of function
          "str q19, [%x[c_ptr], #0x130]\n"
          "str q27, [%x[c_ptr], #0x140]\n"
          "str q12, [%x[c_ptr], #0x150]\n"
          "str q20, [%x[c_ptr], #0x160]\n"
          "str q28, [%x[c_ptr], #0x170]\n"
          "add %x[c_ptr], %x[c_ptr], #0x180\n"
        : [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [c_ptr] "+r" (c_ptr), [k] "+r" (k),
          [aa] "+w" (aa), [ab] "+w" (ab), [b0] "+w" (b0), [b1] "+w" (b1), [b2] "+w" (b2)
        : [odd_k] "r" (odd_k)
        : "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "x20", "cc", "memory"
      );
    }
  }
}

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
