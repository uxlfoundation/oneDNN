//
// SPDX-FileCopyrightText: Copyright 2017-2021, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

void a64_sgemm_asimd_8x6(const float *Apanel, const float *Bpanel, float *Cpanel, int ablocks, int bblocks, int K) {
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
            register float32x4_t a2  asm("v2");
            register float32x4_t a3  asm("v3");
            register float32x4_t b0  asm("v4");
            register float32x4_t b1  asm("v5");
            register float32x4_t b2  asm("v6");

            __asm __volatile (
                // Initialize result registers, load initial operands, prime prefetches.
                "movi    v8.2s, #0x0\n"
                "ld1r    { %[a0].2s }, [%[a_ptr]], #4\n"
                "movi    v9.2s, #0x0\n"
                "movi    v10.2s, #0x0\n"
                "ld1r    { %[a1].2s }, [%[a_ptr]], #4\n"
                "movi    v11.2s, #0x0\n"
                "movi    v12.2s, #0x0\n"
                "movi    v13.2s, #0x0\n"
                "movi    v14.2s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #64]")
                ASM_PREFETCHU("[%[a_ptr], #52]")
                ASM_PREFETCHU("[%[a_ptr], #116]")
                ASM_PREFETCH("[%[b_ptr], #128]")
                "movi    v15.2s, #0x0\n"
                "movi    v16.2s, #0x0\n"
                "movi    v17.2s, #0x0\n"
                "movi    v18.2s, #0x0\n"
                "movi    v19.2s, #0x0\n"
                "movi    v20.2s, #0x0\n"
                "movi    v21.2s, #0x0\n"
                "movi    v22.2s, #0x0\n"
                "movi    v23.2s, #0x0\n"
                "movi    v24.2s, #0x0\n"
                "movi    v25.2s, #0x0\n"
                "movi    v26.2s, #0x0\n"
                "movi    v27.2s, #0x0\n"
                "movi    v28.2s, #0x0\n"
                "movi    v29.2s, #0x0\n"
                "movi    v30.2s, #0x0\n"
                "movi    v31.2s, #0x0\n"

                // Skip loop if we are doing zero iterations of it.
                "cbz    %w[k], 4f\n"

                // Loop proper
                "1:\n"
                "ldr    %d[b0], [%[b_ptr], #0]\n"
                "ld1r    { %[a2].2s }, [%[a_ptr]], #4\n"
                "ldr    %d[b1], [%[b_ptr], #8]\n"
                "fmla     v8.2s , %[b0].2s, %[a0].2s\n"
                "fmla      v9.2s , %[b0].2s, %[a1].2s\n"
                "fmla    v10.2s, %[b0].2s, %[a2].2s\n"

                "ld1r    { %[a3].2s }, [%[a_ptr]], #4\n"
                "fmla    v16.2s, %[b1].2s, %[a0].2s\n"
                "fmla    v17.2s, %[b1].2s, %[a1].2s\n"
                "fmla    v11.2s, %[b0].2s, %[a3].2s\n"

                "ldr    %d[b2], [%[b_ptr], #16]\n"
                "fmla    v18.2s, %[b1].2s, %[a2].2s\n"
                "fmla    v19.2s, %[b1].2s, %[a3].2s\n"
                "fmla    v24.2s, %[b2].2s, %[a0].2s\n"

                "ld1r    { %[a0].2s }, [%[a_ptr]], #4\n"
                "fmla    v25.2s, %[b2].2s, %[a1].2s\n"
                "fmla    v26.2s, %[b2].2s, %[a2].2s\n"
                "fmla    v27.2s, %[b2].2s, %[a3].2s\n"

                "ld1r    { %[a1].2s }, [%[a_ptr]], #4\n"
                "fmla     v12.2s, %[b0].2s, %[a0].2s\n"
                "fmla    v20.2s, %[b1].2s, %[a0].2s\n"
                "fmla    v28.2s, %[b2].2s, %[a0].2s\n"

                "ld1r    { %[a2].2s }, [%[a_ptr]], #4\n"
                "fmla    v13.2s, %[b0].2s, %[a1].2s\n"
                "fmla    v21.2s, %[b1].2s, %[a1].2s\n"
                "fmla    v29.2s, %[b2].2s, %[a1].2s\n"

                "ld1r    { %[a3].2s }, [%[a_ptr]], #4\n"
                "fmla    v14.2s, %[b0].2s, %[a2].2s\n"
                "fmla    v22.2s, %[b1].2s, %[a2].2s\n"
                "fmla    v30.2s, %[b2].2s, %[a2].2s\n"

                "ld1r    { %[a0].2s }, [%[a_ptr]], #4\n"
                "fmla    v15.2s, %[b0].2s, %[a3].2s\n"
                "fmla    v23.2s, %[b1].2s, %[a3].2s\n"
                "fmla    v31.2s, %[b2].2s, %[a3].2s\n"

                "ld1r    { %[a1].2s }, [%[a_ptr]], #4\n"
                 ASM_PREFETCH("[%[b_ptr], #128]")
                "subs    %w[k], %w[k], #1\n"
                 ASM_PREFETCHU("[%[a_ptr], #156]")
                "ldr    %d[b0], [%[b_ptr], #24]\n"
                "ld1r    { %[a2].2s }, [%[a_ptr]], #4\n"

                "ldr    %d[b1], [%[b_ptr], #32]\n"
                "fmla     v8.2s , %[b0].2s, %[a0].2s\n"
                "fmla      v9.2s , %[b0].2s, %[a1].2s\n"
                "fmla    v10.2s, %[b0].2s, %[a2].2s\n"

                "ld1r    { %[a3].2s }, [%[a_ptr]], #4\n"
                "fmla    v16.2s, %[b1].2s, %[a0].2s\n"
                "fmla    v17.2s, %[b1].2s, %[a1].2s\n"
                "fmla    v11.2s, %[b0].2s, %[a3].2s\n"

                "ldr    %d[b2], [%[b_ptr], #40]\n"
                "fmla    v18.2s, %[b1].2s, %[a2].2s\n"
                "fmla    v19.2s, %[b1].2s, %[a3].2s\n"
                "fmla    v24.2s, %[b2].2s, %[a0].2s\n"

                "ld1r    { %[a0].2s }, [%[a_ptr]], #4\n"
                "fmla    v25.2s, %[b2].2s, %[a1].2s\n"
                "fmla    v26.2s, %[b2].2s, %[a2].2s\n"
                "fmla    v27.2s, %[b2].2s, %[a3].2s\n"

                "ld1r    { %[a1].2s }, [%[a_ptr]], #4\n"
                "fmla     v12.2s, %[b0].2s, %[a0].2s\n"
                "fmla    v20.2s, %[b1].2s, %[a0].2s\n"
                "fmla    v28.2s, %[b2].2s, %[a0].2s\n"

                "ld1r    { %[a2].2s }, [%[a_ptr]], #4\n"
                "fmla    v13.2s, %[b0].2s, %[a1].2s\n"
                "fmla    v21.2s, %[b1].2s, %[a1].2s\n"
                "fmla    v29.2s, %[b2].2s, %[a1].2s\n"

                "ld1r    { %[a3].2s }, [%[a_ptr]], #4\n"
                "fmla    v14.2s, %[b0].2s, %[a2].2s\n"
                "fmla    v22.2s, %[b1].2s, %[a2].2s\n"
                "fmla    v30.2s, %[b2].2s, %[a2].2s\n"

                "ld1r    { %[a0].2s }, [%[a_ptr]], #4\n"
                "fmla    v15.2s, %[b0].2s, %[a3].2s\n"
                "fmla    v23.2s, %[b1].2s, %[a3].2s\n"
                "fmla    v31.2s, %[b2].2s, %[a3].2s\n"

                "ld1r    { %[a1].2s }, [%[a_ptr]], #4\n"
                "add    %[b_ptr], %[b_ptr], #48\n"
                 ASM_PREFETCHU("[%[a_ptr], #188]")
                "bne    1b\n"

                // Target to use when K is 1 or 2 (i.e. zero iterations of main loop)
                "4:\n"
                ASM_PREFETCH("[%[c_ptr]]")
                ASM_PREFETCH("[%[c_ptr], #64]")

                "ldr    %d[b0], [%[b_ptr]]\n"
                "ld1r    { %[a2].2s }, [%[a_ptr]], #4\n"

                // Branch to alternative tail for odd K
                "cbnz    %w[oddk], 2f\n"

                // Detached final iteration (even K)
                "ldr    %d[b1], [%[b_ptr], #8]\n"
                "fmla     v8.2s , %[b0].2s, %[a0].2s\n"
                "fmla      v9.2s , %[b0].2s, %[a1].2s\n"
                "fmla    v10.2s, %[b0].2s, %[a2].2s\n"

                "ld1r    { %[a3].2s }, [%[a_ptr]], #4\n"
                "fmla    v16.2s, %[b1].2s, %[a0].2s\n"
                "fmla    v17.2s, %[b1].2s, %[a1].2s\n"
                "fmla    v11.2s, %[b0].2s, %[a3].2s\n"

                "ldr    %d[b2], [%[b_ptr], #16]\n"
                "fmla    v18.2s, %[b1].2s, %[a2].2s\n"
                "fmla    v19.2s, %[b1].2s, %[a3].2s\n"
                "fmla    v24.2s, %[b2].2s, %[a0].2s\n"

                "ld1r    { %[a0].2s }, [%[a_ptr]], #4\n"
                "fmla    v25.2s, %[b2].2s, %[a1].2s\n"
                "fmla    v26.2s, %[b2].2s, %[a2].2s\n"
                "fmla    v27.2s, %[b2].2s, %[a3].2s\n"

                "ld1r    { %[a1].2s }, [%[a_ptr]], #4\n"
                "fmla     v12.2s, %[b0].2s, %[a0].2s\n"
                "fmla    v20.2s, %[b1].2s, %[a0].2s\n"
                "fmla    v28.2s, %[b2].2s, %[a0].2s\n"

                "ld1r    { %[a2].2s }, [%[a_ptr]], #4\n"
                "fmla    v13.2s, %[b0].2s, %[a1].2s\n"
                "fmla    v21.2s, %[b1].2s, %[a1].2s\n"
                "fmla    v29.2s, %[b2].2s, %[a1].2s\n"

                "ld1r    { %[a3].2s }, [%[a_ptr]], #4\n"
                "fmla    v14.2s, %[b0].2s, %[a2].2s\n"
                "fmla    v22.2s, %[b1].2s, %[a2].2s\n"
                "fmla    v30.2s, %[b2].2s, %[a2].2s\n"

                "ld1r    { %[a0].2s }, [%[a_ptr]], #4\n"
                "fmla    v15.2s, %[b0].2s, %[a3].2s\n"
                "fmla    v23.2s, %[b1].2s, %[a3].2s\n"
                "fmla    v31.2s, %[b2].2s, %[a3].2s\n"

                "ldr    %d[b0], [%[b_ptr], #24]\n"
                "add    %[b_ptr], %[b_ptr], #48\n"
                 ASM_PREFETCH("[%[b_ptr], #128]")
                "ld1r    { %[a1].2s }, [%[a_ptr]], #4\n"
                "ld1r    { %[a2].2s }, [%[a_ptr]], #4\n"

                "ldr    %d[b1], [%[b_ptr], #-16]\n"
                "fmla     v8.2s , %[b0].2s, %[a0].2s\n"
                "fmla      v9.2s , %[b0].2s, %[a1].2s\n"
                "fmla    v10.2s, %[b0].2s, %[a2].2s\n"

                "ld1r    { %[a3].2s }, [%[a_ptr]], #4\n"
                "fmla    v16.2s, %[b1].2s, %[a0].2s\n"
                "fmla    v17.2s, %[b1].2s, %[a1].2s\n"
                "fmla    v11.2s, %[b0].2s, %[a3].2s\n"

                "ldr    %d[b2], [%[b_ptr], #-8]\n"
                "fmla    v18.2s, %[b1].2s, %[a2].2s\n"
                "fmla    v19.2s, %[b1].2s, %[a3].2s\n"
                "fmla    v24.2s, %[b2].2s, %[a0].2s\n"

                "ld1r    { %[a0].2s }, [%[a_ptr]], #4\n"
                "fmla    v25.2s, %[b2].2s, %[a1].2s\n"
                "fmla    v26.2s, %[b2].2s, %[a2].2s\n"
                "fmla    v27.2s, %[b2].2s, %[a3].2s\n"

                "ld1r    { %[a1].2s }, [%[a_ptr]], #4\n"
                "fmla     v12.2s, %[b0].2s, %[a0].2s\n"
                "fmla    v20.2s, %[b1].2s, %[a0].2s\n"
                "fmla    v28.2s, %[b2].2s, %[a0].2s\n"

                "ld1r    { %[a2].2s }, [%[a_ptr]], #4\n"
                "fmla    v13.2s, %[b0].2s, %[a1].2s\n"
                "fmla    v21.2s, %[b1].2s, %[a1].2s\n"
                "fmla    v29.2s, %[b2].2s, %[a1].2s\n"

                "ld1r    { %[a3].2s }, [%[a_ptr]], #4\n"
                "fmla    v14.2s, %[b0].2s, %[a2].2s\n"
                "fmla    v22.2s, %[b1].2s, %[a2].2s\n"
                "fmla    v30.2s, %[b2].2s, %[a2].2s\n"

                "fmla    v15.2s, %[b0].2s, %[a3].2s\n"
                "fmla    v23.2s, %[b1].2s, %[a3].2s\n"
                "fmla    v31.2s, %[b2].2s, %[a3].2s\n"

                "b    3f\n"

                // Detached final iteration (odd K)
                "2:\n"
                "ldr    %d[b1], [%[b_ptr], #8]\n"
                "fmla     v8.2s , %[b0].2s, %[a0].2s\n"
                "fmla      v9.2s , %[b0].2s, %[a1].2s\n"
                "fmla    v10.2s, %[b0].2s, %[a2].2s\n"

                "ld1r    { %[a3].2s }, [%[a_ptr]], #4\n"
                "fmla    v16.2s, %[b1].2s, %[a0].2s\n"
                "fmla    v17.2s, %[b1].2s, %[a1].2s\n"
                "fmla    v11.2s, %[b0].2s, %[a3].2s\n"

                "ldr    %d[b2], [%[b_ptr], #16]\n"
                "fmla    v18.2s, %[b1].2s, %[a2].2s\n"
                "fmla    v19.2s, %[b1].2s, %[a3].2s\n"
                "fmla    v24.2s, %[b2].2s, %[a0].2s\n"

                "ld1r    { %[a0].2s }, [%[a_ptr]], #4\n"
                "fmla    v25.2s, %[b2].2s, %[a1].2s\n"
                "fmla    v26.2s, %[b2].2s, %[a2].2s\n"
                "fmla    v27.2s, %[b2].2s, %[a3].2s\n"

                "ld1r    { %[a1].2s }, [%[a_ptr]], #4\n"
                "fmla     v12.2s, %[b0].2s, %[a0].2s\n"
                "fmla    v20.2s, %[b1].2s, %[a0].2s\n"
                "fmla    v28.2s, %[b2].2s, %[a0].2s\n"

                "ld1r    { %[a2].2s }, [%[a_ptr]], #4\n"
                "fmla    v13.2s, %[b0].2s, %[a1].2s\n"
                "fmla    v21.2s, %[b1].2s, %[a1].2s\n"
                "fmla    v29.2s, %[b2].2s, %[a1].2s\n"

                "ld1r    { %[a3].2s }, [%[a_ptr]], #4\n"
                "fmla    v14.2s, %[b0].2s, %[a2].2s\n"
                "fmla    v22.2s, %[b1].2s, %[a2].2s\n"
                "fmla    v30.2s, %[b2].2s, %[a2].2s\n"

                "fmla    v15.2s, %[b0].2s, %[a3].2s\n"
                "fmla    v23.2s, %[b1].2s, %[a3].2s\n"
                "fmla    v31.2s, %[b2].2s, %[a3].2s\n"

                "add    %[b_ptr], %[b_ptr], #24\n"

                // Common tail
                "3:\n"
                "str    d8, [%[c_ptr], #0]\n"
                "str    d16, [%[c_ptr], #8]\n"
                "str    d24, [%[c_ptr], #16]\n"
                "str    d9, [%[c_ptr], #24]\n"
                "str    d17, [%[c_ptr], #32]\n"
                "str    d25, [%[c_ptr], #40]\n"
                "str    d10, [%[c_ptr], #48]\n"
                "str    d18, [%[c_ptr], #56]\n"
                "str    d26, [%[c_ptr], #64]\n"
                "str    d11, [%[c_ptr], #72]\n"
                "str    d19, [%[c_ptr], #80]\n"
                "str    d27, [%[c_ptr], #88]\n"
                "str    d12, [%[c_ptr], #96]\n"
                "str    d20, [%[c_ptr], #104]\n"
                "str    d28, [%[c_ptr], #112]\n"
                "str    d13, [%[c_ptr], #120]\n"
                "str    d21, [%[c_ptr], #128]\n"
                "str    d29, [%[c_ptr], #136]\n"
                "str    d14, [%[c_ptr], #144]\n"
                "str    d22, [%[c_ptr], #152]\n"
                "str    d30, [%[c_ptr], #160]\n"
                "str    d15, [%[c_ptr], #168]\n"
                "str    d23, [%[c_ptr], #176]\n"
                "str    d31, [%[c_ptr], #184]\n"
                "add    %[c_ptr], %[c_ptr], #192\n"

            :
              [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [c_ptr] "+r" (c_ptr),
              [a0] "+w" (a0), [a1] "+w" (a1), [a2] "+w" (a2), [a3] "+w" (a3),
              [b0] "+w" (b0), [b1] "+w" (b1), [b2] "+w" (b2), [k] "+r" (k)
            : [oddk] "r" (oddk)
            : "x20", "x21", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
              "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc", "memory"
            );
        }
    }
}

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
