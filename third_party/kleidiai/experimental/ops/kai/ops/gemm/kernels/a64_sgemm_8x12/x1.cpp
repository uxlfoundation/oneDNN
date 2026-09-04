//
// SPDX-FileCopyrightText: Copyright 2017-2020, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#ifdef __aarch64__

#include <arm_neon.h>

#include "asmlib.hpp"

// Kernel implementation.
//
// Assume that "Apanel" points to a chunk of A blocks (each size 8xK) in read-order.
// Assume that "Bpanel" points to a chunk of B blocks (each size 12xK) in read-order.
// Assume that "Cpanel" points to a chunk of C output blocks (each size
// 12x8), the chunks being arranged in a row major fashion.
//
// Note that the intent of this is that either ablocks or bblocks will be 1
// - this construction allows the output loop to proceed in either order.

namespace kai {
namespace ops {

void a64_sgemm_asimd_8x12_x1(const float *Apanel, const float *Bpanel, float *Cpanel, int ablocks, int bblocks, int K) {
    const float *a_ptr = Apanel;
    float *c_ptr = Cpanel;

    for (int yb=0; yb<ablocks; yb++) {
        const float *a_ptr0 = a_ptr;
        const float *b_ptr = Bpanel;

        for (int xb=0; xb<bblocks; xb++) {
            a_ptr = a_ptr0;
            // Fix up for odd lengths - set a flag if K is odd, but make
            // sure we round up the iteration count.
            int oddk = (K & 1);
            int k = ((K+1)/2) - 1;

            register float32x4_t a0  asm("v0");
            register float32x4_t a1  asm("v1");
            register float32x4_t b0  asm("v2");
            register float32x4_t b1  asm("v3");
            register float32x4_t b2  asm("v4");

            __asm __volatile (
                // Initialize result registers, load initial operands, prime prefetches.
                "movi    v8.4s, #0x0\n"
                "ldr    %q[a0], [%[a_ptr]]\n"
                "movi    v9.4s, #0x0\n"
                "ldr    %q[b0], [%[b_ptr]]\n"
                "movi    v10.4s, #0x0\n"
                "ldr    %q[a1], [%[a_ptr], #16]\n"
                "movi    v11.4s, #0x0\n"
                "ldr    %q[b1], [%[b_ptr], #16]\n"
                "movi    v12.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #64]")
                "movi    v13.4s, #0x0\n"
                ASM_PREFETCH("[%[a_ptr], #64]")
                "movi    v14.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #128]")
                "movi    v15.4s, #0x0\n"
                ASM_PREFETCH("[%[a_ptr], #128]")
                "movi    v16.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #192]")
                "movi    v17.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #256]")
                "movi    v18.4s, #0x0\n"
                ASM_PREFETCH("[%[a_ptr], #192]")
                "movi    v19.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #320]")
                "movi    v20.4s, #0x0\n"
                ASM_PREFETCH("[%[a_ptr], #256]")
                "movi    v21.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #384]")
                "movi    v22.4s, #0x0\n"
                "movi    v23.4s, #0x0\n"
                "movi    v24.4s, #0x0\n"
                "movi    v25.4s, #0x0\n"
                "movi    v26.4s, #0x0\n"
                "movi    v27.4s, #0x0\n"
                "movi    v28.4s, #0x0\n"
                "movi    v29.4s, #0x0\n"
                "movi    v30.4s, #0x0\n"
                "movi    v31.4s, #0x0\n"

                // Skip loop if we are doing zero iterations of it.
                "cbz    %w[k], 4f\n"

                // Loop proper
                "1:\n"
                "fmla     v8.4s , %[b0].4s, %[a0].s[0]\n"
                "fmla      v9.4s , %[b0].4s, %[a0].s[1]\n"
                "ldr    %q[b2], [%[b_ptr], #32]\n"
                "fmla    v10.4s, %[b0].4s, %[a0].s[2]\n"
                "fmla    v11.4s, %[b0].4s, %[a0].s[3]\n"
                "fmla     v12.4s, %[b0].4s, %[a1].s[0]\n"
                "fmla    v13.4s, %[b0].4s, %[a1].s[1]\n"
                "fmla    v14.4s, %[b0].4s, %[a1].s[2]\n"
                "fmla    v15.4s, %[b0].4s, %[a1].s[3]\n"
                "ldr    %q[b0], [%[b_ptr], #48]\n"

                "fmla    v16.4s, %[b1].4s, %[a0].s[0]\n"
                "fmla    v17.4s, %[b1].4s, %[a0].s[1]\n"
                ASM_PREFETCH("[%[a_ptr], #320]")
                "fmla    v18.4s, %[b1].4s, %[a0].s[2]\n"
                "fmla    v19.4s, %[b1].4s, %[a0].s[3]\n"
                "fmla    v20.4s, %[b1].4s, %[a1].s[0]\n"
                "fmla    v21.4s, %[b1].4s, %[a1].s[1]\n"
                "fmla    v22.4s, %[b1].4s, %[a1].s[2]\n"
                "fmla    v23.4s, %[b1].4s, %[a1].s[3]\n"
                "ldr    %q[b1], [%[b_ptr], #64]\n"

                "fmla    v24.4s, %[b2].4s, %[a0].s[0]\n"
                "fmla    v25.4s, %[b2].4s, %[a0].s[1]\n"
                ASM_PREFETCH("[%[b_ptr], #448]")
                "fmla    v26.4s, %[b2].4s, %[a0].s[2]\n"
                "fmla    v27.4s, %[b2].4s, %[a0].s[3]\n"
                "ldr    %q[a0], [%[a_ptr], #32]\n"
                "fmla    v28.4s, %[b2].4s, %[a1].s[0]\n"
                "fmla    v29.4s, %[b2].4s, %[a1].s[1]\n"
                "fmla    v30.4s, %[b2].4s, %[a1].s[2]\n"
                "fmla    v31.4s, %[b2].4s, %[a1].s[3]\n"
                "ldr    %q[a1], [%[a_ptr], #48]\n"

                "fmla     v8.4s , %[b0].4s, %[a0].s[0]\n"
                "fmla    v9.4s , %[b0].4s, %[a0].s[1]\n"
                "ldr    %q[b2], [%[b_ptr], #80]\n"
                "fmla    v10.4s, %[b0].4s, %[a0].s[2]\n"
                "fmla    v11.4s, %[b0].4s, %[a0].s[3]\n"
                "fmla     v12.4s, %[b0].4s, %[a1].s[0]\n"
                "fmla   v13.4s, %[b0].4s, %[a1].s[1]\n"
                "fmla    v14.4s, %[b0].4s, %[a1].s[2]\n"
                "fmla    v15.4s, %[b0].4s, %[a1].s[3]\n"
                "ldr    %q[b0], [%[b_ptr], #96]\n"

                "fmla    v16.4s, %[b1].4s, %[a0].s[0]\n"
                "fmla    v17.4s, %[b1].4s, %[a0].s[1]\n"
                "fmla    v18.4s, %[b1].4s, %[a0].s[2]\n"
                "fmla    v19.4s, %[b1].4s, %[a0].s[3]\n"
                ASM_PREFETCH("[%[b_ptr], #512]")
                "fmla    v20.4s, %[b1].4s, %[a1].s[0]\n"
                "fmla    v21.4s, %[b1].4s, %[a1].s[1]\n"
                "subs    %w[k], %w[k], #1\n"
                "fmla    v22.4s, %[b1].4s, %[a1].s[2]\n"
                "fmla    v23.4s, %[b1].4s, %[a1].s[3]\n"
                "ldr    %q[b1], [%[b_ptr], #112]\n"

                "fmla    v24.4s, %[b2].4s, %[a0].s[0]\n"
                "fmla    v25.4s, %[b2].4s, %[a0].s[1]\n"
                "add    %[a_ptr], %[a_ptr], #64\n"
                "fmla    v26.4s, %[b2].4s, %[a0].s[2]\n"
                "fmla    v27.4s, %[b2].4s, %[a0].s[3]\n"
                "ldr    %q[a0], [%[a_ptr]]\n"
                "fmla    v28.4s, %[b2].4s, %[a1].s[0]\n"
                "fmla    v29.4s, %[b2].4s, %[a1].s[1]\n"
                "add    %[b_ptr], %[b_ptr], #96\n"
                "fmla    v30.4s, %[b2].4s, %[a1].s[2]\n"
                "fmla    v31.4s, %[b2].4s, %[a1].s[3]\n"
                "ldr    %q[a1], [%[a_ptr], #16]\n"
                "bne    1b\n"

                // Target to use when K is 1 or 2 (i.e. zero iterations of main loop)
                "4:\n"

                // Branch to alternative tail for odd K
                "cbnz    %w[oddk], 2f\n"

                // Detached final iteration (even K)
                "fmla     v8.4s , %[b0].4s, %[a0].s[0]\n"
                "fmla   v9.4s , %[b0].4s, %[a0].s[1]\n"
                "ldr    %q[b2], [%[b_ptr], #32]\n"
                "fmla    v10.4s, %[b0].4s, %[a0].s[2]\n"
                "fmla    v11.4s, %[b0].4s, %[a0].s[3]\n"
                "fmla     v12.4s, %[b0].4s, %[a1].s[0]\n"
                "fmla   v13.4s, %[b0].4s, %[a1].s[1]\n"
                "fmla    v14.4s, %[b0].4s, %[a1].s[2]\n"
                "fmla    v15.4s, %[b0].4s, %[a1].s[3]\n"
                "ldr    %q[b0], [%[b_ptr], #48]\n"

                "fmla    v16.4s, %[b1].4s, %[a0].s[0]\n"
                "fmla    v17.4s, %[b1].4s, %[a0].s[1]\n"
                "fmla    v18.4s, %[b1].4s, %[a0].s[2]\n"
                "fmla    v19.4s, %[b1].4s, %[a0].s[3]\n"
                "fmla    v20.4s, %[b1].4s, %[a1].s[0]\n"
                "fmla    v21.4s, %[b1].4s, %[a1].s[1]\n"
                "fmla    v22.4s, %[b1].4s, %[a1].s[2]\n"
                "fmla    v23.4s, %[b1].4s, %[a1].s[3]\n"
                "ldr    %q[b1], [%[b_ptr], #64]\n"

                "fmla    v24.4s, %[b2].4s, %[a0].s[0]\n"
                "fmla    v25.4s, %[b2].4s, %[a0].s[1]\n"
                "add    %[a_ptr], %[a_ptr], #64\n"
                "fmla    v26.4s, %[b2].4s, %[a0].s[2]\n"
                "fmla    v27.4s, %[b2].4s, %[a0].s[3]\n"
                "ldr    %q[a0], [%[a_ptr], #-32]\n"
                "fmla    v28.4s, %[b2].4s, %[a1].s[0]\n"
                "fmla    v29.4s, %[b2].4s, %[a1].s[1]\n"
                "add    %[b_ptr], %[b_ptr], #96\n"
                "fmla    v30.4s, %[b2].4s, %[a1].s[2]\n"
                "fmla    v31.4s, %[b2].4s, %[a1].s[3]\n"
                "ldr    %q[a1], [%[a_ptr], #-16]\n"

                "fmla     v8.4s , %[b0].4s, %[a0].s[0]\n"
                "fmla    v16.4s, %[b1].4s, %[a0].s[0]\n"
                "ldr    %q[b2], [%[b_ptr], #-16]\n"
                "fmla   v9.4s , %[b0].4s, %[a0].s[1]\n"
                "str    q8, [%[c_ptr], #0]\n"
                "fmla    v17.4s, %[b1].4s, %[a0].s[1]\n"
                "str    q16, [%[c_ptr], #16]\n"
                "fmla    v24.4s, %[b2].4s, %[a0].s[0]\n"
                "str    q24, [%[c_ptr], #32]\n"

                "fmla    v25.4s, %[b2].4s, %[a0].s[1]\n"
                "str    q9, [%[c_ptr], #48]\n"
                "fmla    v10.4s, %[b0].4s, %[a0].s[2]\n"
                "str    q17, [%[c_ptr], #64]\n"
                "fmla    v18.4s, %[b1].4s, %[a0].s[2]\n"
                "str    q25, [%[c_ptr], #80]\n"
                "fmla    v26.4s, %[b2].4s, %[a0].s[2]\n"
                "str    q10, [%[c_ptr], #96]\n"

                "fmla    v11.4s, %[b0].4s, %[a0].s[3]\n"
                "str    q18, [%[c_ptr], #112]\n"
                "fmla    v19.4s, %[b1].4s, %[a0].s[3]\n"
                "str    q26, [%[c_ptr], #128]\n"
                "fmla    v27.4s, %[b2].4s, %[a0].s[3]\n"
                "str    q11, [%[c_ptr], #144]\n"

                "fmla     v12.4s, %[b0].4s, %[a1].s[0]\n"
                "str    q19, [%[c_ptr], #160]\n"
                "fmla    v20.4s, %[b1].4s, %[a1].s[0]\n"
                "str    q27, [%[c_ptr], #176]\n"
                "fmla    v28.4s, %[b2].4s, %[a1].s[0]\n"
                "str    q12, [%[c_ptr], #192]\n"

                "fmla   v13.4s, %[b0].4s, %[a1].s[1]\n"
                "str    q20, [%[c_ptr], #208]\n"
                "fmla    v21.4s, %[b1].4s, %[a1].s[1]\n"
                "str    q28, [%[c_ptr], #224]\n"
                "fmla    v29.4s, %[b2].4s, %[a1].s[1]\n"
                "str    q13, [%[c_ptr], #240]\n"

                "fmla    v14.4s, %[b0].4s, %[a1].s[2]\n"
                "str    q21, [%[c_ptr], #256]\n"
                "fmla    v22.4s, %[b1].4s, %[a1].s[2]\n"
                "str    q29, [%[c_ptr], #272]\n"
                "fmla    v30.4s, %[b2].4s, %[a1].s[2]\n"
                "str    q14, [%[c_ptr], #288]\n"

                "fmla    v15.4s, %[b0].4s, %[a1].s[3]\n"
                "str    q22, [%[c_ptr], #304]\n"
                "fmla    v23.4s, %[b1].4s, %[a1].s[3]\n"
                "str    q30, [%[c_ptr], #320]\n"
                "fmla    v31.4s, %[b2].4s, %[a1].s[3]\n"
                "str    q15, [%[c_ptr], #336]\n"

                "b    3f\n"

                // Detached final iteration (odd K)
                "2:\n"
                "fmla     v8.4s , %[b0].4s, %[a0].s[0]\n"
                "ldr    %q[b2], [%[b_ptr], #32]\n"
                "fmla    v16.4s, %[b1].4s, %[a0].s[0]\n"
                "fmla   v9.4s , %[b0].4s, %[a0].s[1]\n"
                "str    q8, [%[c_ptr], #0]\n"
                "fmla    v17.4s, %[b1].4s, %[a0].s[1]\n"
                "str    q16, [%[c_ptr], #16]\n"
                "fmla    v24.4s, %[b2].4s, %[a0].s[0]\n"
                "add    %[b_ptr], %[b_ptr], #48\n"
                "add    %[a_ptr], %[a_ptr], #32\n"
                "str    q24, [%[c_ptr], #32]\n"
                "fmla    v25.4s, %[b2].4s, %[a0].s[1]\n"
                "str    q9, [%[c_ptr], #48]\n"

                "fmla    v10.4s, %[b0].4s, %[a0].s[2]\n"
                "str    q17, [%[c_ptr], #64]\n"
                "fmla    v18.4s, %[b1].4s, %[a0].s[2]\n"
                "str    q25, [%[c_ptr], #80]\n"
                "fmla    v26.4s, %[b2].4s, %[a0].s[2]\n"
                "str    q10, [%[c_ptr], #96]\n"

                "fmla    v11.4s, %[b0].4s, %[a0].s[3]\n"
                "str    q18, [%[c_ptr], #112]\n"
                "fmla    v19.4s, %[b1].4s, %[a0].s[3]\n"
                "str    q26, [%[c_ptr], #128]\n"
                "fmla    v27.4s, %[b2].4s, %[a0].s[3]\n"
                "str    q11, [%[c_ptr], #144]\n"

                "fmla     v12.4s, %[b0].4s, %[a1].s[0]\n"
                "str    q19, [%[c_ptr], #160]\n"
                "fmla    v20.4s, %[b1].4s, %[a1].s[0]\n"
                "str    q27, [%[c_ptr], #176]\n"
                "fmla    v28.4s, %[b2].4s, %[a1].s[0]\n"
                "str    q12, [%[c_ptr], #192]\n"

                "fmla   v13.4s, %[b0].4s, %[a1].s[1]\n"
                "str    q20, [%[c_ptr], #208]\n"
                "fmla    v21.4s, %[b1].4s, %[a1].s[1]\n"
                "str    q28, [%[c_ptr], #224]\n"
                "fmla    v29.4s, %[b2].4s, %[a1].s[1]\n"
                "str    q13, [%[c_ptr], #240]\n"

                "fmla    v14.4s, %[b0].4s, %[a1].s[2]\n"
                "str    q21, [%[c_ptr], #256]\n"
                "fmla    v22.4s, %[b1].4s, %[a1].s[2]\n"
                "str    q29, [%[c_ptr], #272]\n"
                "fmla    v30.4s, %[b2].4s, %[a1].s[2]\n"
                "str    q14, [%[c_ptr], #288]\n"

                "fmla    v15.4s, %[b0].4s, %[a1].s[3]\n"
                "str    q22, [%[c_ptr], #304]\n"
                "fmla    v23.4s, %[b1].4s, %[a1].s[3]\n"
                "str    q30, [%[c_ptr], #320]\n"
                "fmla    v31.4s, %[b2].4s, %[a1].s[3]\n"
                "str    q15, [%[c_ptr], #336]\n"

                // Common tail
                "3:\n"
                "str    q23, [%[c_ptr], #352]\n"
                "str    q31, [%[c_ptr], #368]\n"
                "add    %[c_ptr], %[c_ptr], #384\n"
            :
              [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [c_ptr] "+r" (c_ptr),
              [a0] "+w" (a0), [a1] "+w" (a1),
              [b0] "+w" (b0), [b1] "+w" (b1), [b2] "+w" (b2), [k] "+r" (k)
            : [oddk] "r" (oddk)
            : "x20", "x21", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
              "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc"
            );
        }
    }
}

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
