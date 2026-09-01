//
// SPDX-FileCopyrightText: Copyright 2017-2018, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Build on AArch64 where either ENABLE_FP16_KERNELS is set or FP16 is explicitly supported.
#ifdef __aarch64__

#include <arm_neon.h>

#include "asmlib.hpp"

// Kernel implementation.
//
// Assume that "Apanel" points to a chunk of A blocks (each size 8xK) in read-order.
// Assume that "Bpanel" points to a chunk of B blocks (each size 24xK) in read-order.
// Assume that "Cpanel" points to a chunk of C output blocks (each size
// 8x24), the chunks being arranged in a row major fashion.
//
// Note that the intent of this is that either ablocks or bblocks will be 1
// - this construction allows the output loop to proceed in either order.

namespace kai {
namespace ops {

void a64_hgemm_asimd_8x24(const __fp16 *Apanel, const __fp16 *Bpanel, __fp16 *Cpanel, int ablocks, int bblocks, int K) {
    const __fp16 *a_ptr = Apanel;
    __fp16 *c_ptr = Cpanel;

    for (int yb=0; yb<ablocks; yb++) {
        const __fp16 *a_ptr0 = a_ptr;
        const __fp16 *b_ptr = Bpanel;

        for (int xb=0; xb<bblocks; xb++) {
            a_ptr = a_ptr0;
            // Fix up for odd lengths - set a flag if K is odd, but make
            // sure we round up the iteration count.
            int oddk = (K & 1);
            int k = ((K+1)/2) - 1;

            register float16x8_t a0  asm("v0");
            register float16x8_t a0a asm("v1");
            register float16x8_t b0  asm("v2");
            register float16x8_t b1  asm("v3");
            register float16x8_t b2  asm("v4");
            register float16x8_t b0a asm("v5");
            register float16x8_t b1a asm("v6");
            register float16x8_t b2a asm("v7");

            __asm __volatile (
                // Initialize result registers, load initial operands, prime prefetches.
                "movi    v8.8h, #0x0\n"
                "ldr    %q[a0], [%[a_ptr]]\n"
                "movi    v9.8h, #0x0\n"
                "ldr    %q[b0], [%[b_ptr]]\n"
                "movi    v10.8h, #0x0\n"
                "ldr    %q[b1], [%[b_ptr], #16]\n"
                "movi    v11.8h, #0x0\n"
                "ldr    %q[b2], [%[b_ptr], #32]\n"
                "movi    v12.8h, #0x0\n"
                "movi    v13.8h, #0x0\n"
                "movi    v14.8h, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #64]")
                "movi    v15.8h, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #128]")
                "movi    v16.8h, #0x0\n"
                ASM_PREFETCH("[%[a_ptr], #64]")
                "movi    v17.8h, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #192]")
                "movi    v18.8h, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #256]")
                "movi    v19.8h, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #320]")
                "movi    v20.8h, #0x0\n"
                "movi    v21.8h, #0x0\n"
                "movi    v22.8h, #0x0\n"
                "movi    v23.8h, #0x0\n"
                "movi    v24.8h, #0x0\n"
                "movi    v25.8h, #0x0\n"
                "movi    v26.8h, #0x0\n"
                "movi    v27.8h, #0x0\n"
                "movi    v28.8h, #0x0\n"
                "movi    v29.8h, #0x0\n"
                "movi    v30.8h, #0x0\n"
                "movi    v31.8h, #0x0\n"

                // Skip loop if we are doing zero iterations of it.
                "cbz    %w[k], 4f\n"

                "1:\n"
                "fmla     v8.8h , %[b0].8h, %[a0].h[0]\n"
                "fmla     v9.8h , %[b0].8h, %[a0].h[1]\n"
                "ldr    %q[a0a], [%[a_ptr], #16]\n"
                "fmla    v10.8h, %[b0].8h, %[a0].h[2]\n"
                "fmla    v11.8h, %[b0].8h, %[a0].h[3]\n"
                "ldr    %q[b0a], [%[b_ptr], #48]\n"
                "fmla     v12.8h, %[b0].8h, %[a0].h[4]\n"
                "fmla    v13.8h, %[b0].8h, %[a0].h[5]\n"
                "ldr    %q[b1a], [%[b_ptr], #64]\n"
                "fmla    v14.8h, %[b0].8h, %[a0].h[6]\n"
                "fmla    v15.8h, %[b0].8h, %[a0].h[7]\n"
                "ldr    %q[b0], [%[b_ptr], #96]\n"

                "fmla    v16.8h, %[b1].8h, %[a0].h[0]\n"
                "fmla    v17.8h, %[b1].8h, %[a0].h[1]\n"
                "ldr    %q[b2a], [%[b_ptr], #80]\n"
                "fmla    v18.8h, %[b1].8h, %[a0].h[2]\n"
                "fmla    v19.8h, %[b1].8h, %[a0].h[3]\n"
                "add    %[b_ptr], %[b_ptr], #96\n"
                "fmla    v20.8h, %[b1].8h, %[a0].h[4]\n"
                "fmla    v21.8h, %[b1].8h, %[a0].h[5]\n"
                ASM_PREFETCH("[%[a_ptr], #128]")
                "fmla    v22.8h, %[b1].8h, %[a0].h[6]\n"
                "fmla    v23.8h, %[b1].8h, %[a0].h[7]\n"
                "ldr    %q[b1], [%[b_ptr], #16]\n"

                "fmla    v24.8h, %[b2].8h, %[a0].h[0]\n"
                "fmla    v25.8h, %[b2].8h, %[a0].h[1]\n"
                ASM_PREFETCH("[%[b_ptr], #288]")
                "fmla    v26.8h, %[b2].8h, %[a0].h[2]\n"
                "fmla    v27.8h, %[b2].8h, %[a0].h[3]\n"
                "fmla    v28.8h, %[b2].8h, %[a0].h[4]\n"
                "fmla    v29.8h, %[b2].8h, %[a0].h[5]\n"
                "fmla    v30.8h, %[b2].8h, %[a0].h[6]\n"
                "fmla    v31.8h, %[b2].8h, %[a0].h[7]\n"
                "ldr    %q[a0], [%[a_ptr], #32]\n"

                "fmla     v8.8h , %[b0a].8h, %[a0a].h[0]\n"
                "fmla    v9.8h , %[b0a].8h, %[a0a].h[1]\n"
                "ldr    %q[b2], [%[b_ptr], #32]\n"
                "fmla    v10.8h, %[b0a].8h, %[a0a].h[2]\n"
                "fmla    v11.8h, %[b0a].8h, %[a0a].h[3]\n"
                "fmla     v12.8h, %[b0a].8h, %[a0a].h[4]\n"
                "fmla    v13.8h, %[b0a].8h, %[a0a].h[5]\n"
                "fmla    v14.8h, %[b0a].8h, %[a0a].h[6]\n"
                "fmla    v15.8h, %[b0a].8h, %[a0a].h[7]\n"

                "fmla    v16.8h, %[b1a].8h, %[a0a].h[0]\n"
                "fmla    v17.8h, %[b1a].8h, %[a0a].h[1]\n"
                ASM_PREFETCH("[%[b_ptr], #352]")
                "fmla    v18.8h, %[b1a].8h, %[a0a].h[2]\n"
                "fmla    v19.8h, %[b1a].8h, %[a0a].h[3]\n"
                "fmla    v20.8h, %[b1a].8h, %[a0a].h[4]\n"
                "fmla    v21.8h, %[b1a].8h, %[a0a].h[5]\n"
                "fmla    v22.8h, %[b1a].8h, %[a0a].h[6]\n"
                "fmla    v23.8h, %[b1a].8h, %[a0a].h[7]\n"

                "fmla    v24.8h, %[b2a].8h, %[a0a].h[0]\n"
                "fmla    v25.8h, %[b2a].8h, %[a0a].h[1]\n"
                "add    %[a_ptr], %[a_ptr], #32\n"
                "fmla    v26.8h, %[b2a].8h, %[a0a].h[2]\n"
                "fmla    v27.8h, %[b2a].8h, %[a0a].h[3]\n"
                "fmla    v28.8h, %[b2a].8h, %[a0a].h[4]\n"
                "fmla    v29.8h, %[b2a].8h, %[a0a].h[5]\n"
                "subs    %w[k], %w[k], #1\n"
                "fmla    v30.8h, %[b2a].8h, %[a0a].h[6]\n"
                "fmla    v31.8h, %[b2a].8h, %[a0a].h[7]\n"

                "bne    1b\n"
                "4:\n"

                // Jump to odd tail if necessary.
                "cbnz    %w[oddk], 2f\n"

                // Even tail.
                "fmla     v8.8h , %[b0].8h, %[a0].h[0]\n"
                "fmla   v9.8h , %[b0].8h, %[a0].h[1]\n"
                "ldr    %q[a0a], [%[a_ptr], #16]\n"
                "fmla    v10.8h, %[b0].8h, %[a0].h[2]\n"
                "fmla    v11.8h, %[b0].8h, %[a0].h[3]\n"
                "ldr    %q[b0a], [%[b_ptr], #48]\n"
                "fmla     v12.8h, %[b0].8h, %[a0].h[4]\n"
                "fmla   v13.8h, %[b0].8h, %[a0].h[5]\n"
                "ldr    %q[b1a], [%[b_ptr], #64]\n"
                "fmla    v14.8h, %[b0].8h, %[a0].h[6]\n"
                "fmla    v15.8h, %[b0].8h, %[a0].h[7]\n"
                "ldr    %q[b2a], [%[b_ptr], #80]\n"

                "fmla    v16.8h, %[b1].8h, %[a0].h[0]\n"
                "fmla    v17.8h, %[b1].8h, %[a0].h[1]\n"
                "add    %[b_ptr], %[b_ptr], #96\n"
                "fmla    v18.8h, %[b1].8h, %[a0].h[2]\n"
                "fmla    v19.8h, %[b1].8h, %[a0].h[3]\n"
                "fmla    v20.8h, %[b1].8h, %[a0].h[4]\n"
                "fmla    v21.8h, %[b1].8h, %[a0].h[5]\n"
                "add    %[a_ptr], %[a_ptr], #32\n"
                "fmla    v22.8h, %[b1].8h, %[a0].h[6]\n"
                "fmla    v23.8h, %[b1].8h, %[a0].h[7]\n"

                "fmla    v24.8h, %[b2].8h, %[a0].h[0]\n"
                "fmla    v25.8h, %[b2].8h, %[a0].h[1]\n"
                "fmla    v26.8h, %[b2].8h, %[a0].h[2]\n"
                "fmla    v27.8h, %[b2].8h, %[a0].h[3]\n"
                "fmla    v28.8h, %[b2].8h, %[a0].h[4]\n"
                "fmla    v29.8h, %[b2].8h, %[a0].h[5]\n"
                "fmla    v30.8h, %[b2].8h, %[a0].h[6]\n"
                "fmla    v31.8h, %[b2].8h, %[a0].h[7]\n"

                "fmla     v8.8h , %[b0a].8h, %[a0a].h[0]\n"
                "fmla    v16.8h, %[b1a].8h, %[a0a].h[0]\n"
                "str    q8, [%[c_ptr]]\n"
                "fmla    v24.8h, %[b2a].8h, %[a0a].h[0]\n"
                "str    q16, [%[c_ptr], #16]\n"

                "fmla      v9.8h , %[b0a].8h, %[a0a].h[1]\n"
                "str    q24, [%[c_ptr], #32]\n"
                "fmla    v17.8h, %[b1a].8h, %[a0a].h[1]\n"
                "str    q9, [%[c_ptr], #48]\n"
                "fmla    v25.8h, %[b2a].8h, %[a0a].h[1]\n"
                "str    q17, [%[c_ptr], #64]\n"

                "fmla    v10.8h, %[b0a].8h, %[a0a].h[2]\n"
                "str    q25, [%[c_ptr], #80]\n"
                "fmla    v18.8h, %[b1a].8h, %[a0a].h[2]\n"
                "str    q10, [%[c_ptr], #96]\n"
                "fmla    v26.8h, %[b2a].8h, %[a0a].h[2]\n"
                "str    q18, [%[c_ptr], #112]\n"

                "fmla    v11.8h, %[b0a].8h, %[a0a].h[3]\n"
                "str    q26, [%[c_ptr], #128]\n"
                "fmla    v19.8h, %[b1a].8h, %[a0a].h[3]\n"
                "str    q11, [%[c_ptr], #144]\n"
                "fmla    v27.8h, %[b2a].8h, %[a0a].h[3]\n"
                "str    q19, [%[c_ptr], #160]\n"

                "fmla     v12.8h, %[b0a].8h, %[a0a].h[4]\n"
                "str    q27, [%[c_ptr], #176]\n"
                "fmla    v20.8h, %[b1a].8h, %[a0a].h[4]\n"
                "str    q12, [%[c_ptr], #192]\n"
                "fmla    v28.8h, %[b2a].8h, %[a0a].h[4]\n"
                "str    q20, [%[c_ptr], #208]\n"

                "fmla      v13.8h, %[b0a].8h, %[a0a].h[5]\n"
                "str    q28, [%[c_ptr], #224]\n"
                "fmla    v21.8h, %[b1a].8h, %[a0a].h[5]\n"
                "str    q13, [%[c_ptr], #240]\n"
                "fmla    v29.8h, %[b2a].8h, %[a0a].h[5]\n"
                "str    q21, [%[c_ptr], #256]\n"

                "fmla    v14.8h, %[b0a].8h, %[a0a].h[6]\n"
                "str    q29, [%[c_ptr], #272]\n"
                "fmla    v22.8h, %[b1a].8h, %[a0a].h[6]\n"
                "str    q14, [%[c_ptr], #288]\n"
                "fmla    v30.8h, %[b2a].8h, %[a0a].h[6]\n"
                "str    q22, [%[c_ptr], #304]\n"

                "fmla    v15.8h, %[b0a].8h, %[a0a].h[7]\n"
                "str    q30, [%[c_ptr], #320]\n"
                "fmla    v23.8h, %[b1a].8h, %[a0a].h[7]\n"
                "str    q15, [%[c_ptr], #336]\n"
                "fmla    v31.8h, %[b2a].8h, %[a0a].h[7]\n"
                "b    3f\n"

                // Odd tail
                "2:\n"
                "fmla     v8.8h , %[b0].8h, %[a0].h[0]\n"
                "add    %[b_ptr], %[b_ptr], #48\n"
                "fmla    v16.8h, %[b1].8h, %[a0].h[0]\n"
                "add    %[a_ptr], %[a_ptr], #16\n"
                "str    q8, [%[c_ptr]]\n"
                "fmla    v24.8h, %[b2].8h, %[a0].h[0]\n"
                "str    q16, [%[c_ptr], #16]\n"

                "fmla      v9.8h , %[b0].8h, %[a0].h[1]\n"
                "str    q24, [%[c_ptr], #32]\n"
                "fmla    v17.8h, %[b1].8h, %[a0].h[1]\n"
                "str    q9, [%[c_ptr], #48]\n"
                "fmla    v25.8h, %[b2].8h, %[a0].h[1]\n"
                "str    q17, [%[c_ptr], #64]\n"

                "fmla    v10.8h, %[b0].8h, %[a0].h[2]\n"
                "str    q25, [%[c_ptr], #80]\n"
                "fmla    v18.8h, %[b1].8h, %[a0].h[2]\n"
                "str    q10, [%[c_ptr], #96]\n"
                "fmla    v26.8h, %[b2].8h, %[a0].h[2]\n"
                "str    q18, [%[c_ptr], #112]\n"

                "fmla    v11.8h, %[b0].8h, %[a0].h[3]\n"
                "str    q26, [%[c_ptr], #128]\n"
                "fmla    v19.8h, %[b1].8h, %[a0].h[3]\n"
                "str    q11, [%[c_ptr], #144]\n"
                "fmla    v27.8h, %[b2].8h, %[a0].h[3]\n"
                "str    q19, [%[c_ptr], #160]\n"

                "fmla     v12.8h, %[b0].8h, %[a0].h[4]\n"
                "str    q27, [%[c_ptr], #176]\n"
                "fmla    v20.8h, %[b1].8h, %[a0].h[4]\n"
                "str    q12, [%[c_ptr], #192]\n"
                "fmla    v28.8h, %[b2].8h, %[a0].h[4]\n"
                "str    q20, [%[c_ptr], #208]\n"

                "fmla      v13.8h, %[b0].8h, %[a0].h[5]\n"
                "str    q28, [%[c_ptr], #224]\n"
                "fmla    v21.8h, %[b1].8h, %[a0].h[5]\n"
                "str    q13, [%[c_ptr], #240]\n"
                "fmla    v29.8h, %[b2].8h, %[a0].h[5]\n"
                "str    q21, [%[c_ptr], #256]\n"

                "fmla    v14.8h, %[b0].8h, %[a0].h[6]\n"
                "str    q29, [%[c_ptr], #272]\n"
                "fmla    v22.8h, %[b1].8h, %[a0].h[6]\n"
                "str    q14, [%[c_ptr], #288]\n"
                "fmla    v30.8h, %[b2].8h, %[a0].h[6]\n"
                "str    q22, [%[c_ptr], #304]\n"

                "fmla    v15.8h, %[b0].8h, %[a0].h[7]\n"
                "str    q30, [%[c_ptr], #320]\n"
                "fmla    v23.8h, %[b1].8h, %[a0].h[7]\n"
                "str    q15, [%[c_ptr], #336]\n"
                "fmla    v31.8h, %[b2].8h, %[a0].h[7]\n"

                "3:\n"
                "str    q23, [%[c_ptr], #352]\n"
                "str    q31, [%[c_ptr], #368]\n"
                "add    %[c_ptr], %[c_ptr], #384\n"
            :
              [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [c_ptr] "+r" (c_ptr),
              [a0] "+w" (a0), [a0a] "+w" (a0a),
              [b0] "+w" (b0), [b1] "+w" (b1), [b2] "+w" (b2), [k] "+r" (k),
              [b0a] "+w" (b0a), [b1a] "+w" (b1a), [b2a] "+w" (b2a)
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
