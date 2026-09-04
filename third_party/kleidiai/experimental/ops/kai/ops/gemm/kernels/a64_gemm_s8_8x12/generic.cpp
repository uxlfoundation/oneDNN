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

void a64_gemm_s8_8x12(const int8_t *Apanel, const int8_t *Bpanel, int32_t *Cpanel, int ablocks, int bblocks, int K) {
    const int8_t *a_ptr = Apanel;
    int32_t *c_ptr = Cpanel;
    // We divide K by 4 because the sdot instruction processes 4 elements at a time.
    const int W = K/4;
    // Fix up for odd lengths - set a flag if K is odd, but make
    // sure we round up the iteration count.
    const int oddk = (W & 1);
    const int init_value_k = ((W+1)/2) - 1;
    for (int yb=0; yb<ablocks; yb++) {
        const int8_t *a_ptr0 = a_ptr;
        const int8_t *b_ptr = Bpanel;
        for (int xb=0; xb<bblocks; xb++) {
            a_ptr = a_ptr0;
            int k = init_value_k;
            register int32x4_t a0  asm("v0");
            register int32x4_t a1  asm("v1");
            register int32x4_t b0  asm("v2");
            register int32x4_t b1  asm("v3");
            register int32x4_t b2  asm("v4");
            register int32x4_t a0a asm("v5");
            register int32x4_t a1a asm("v6");
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
                ".inst 0x4f80e048 // sdot v8.4s , %[b0].16b, %[a0].4b[0]\n"
                ".inst 0x4fa0e049 // sdot v9.4s , %[b0].16b, %[a0].4b[1]\n"

                "ldr    %q[b2], [%[b_ptr], #32]\n"
                ".inst 0x4f80e84a // sdot v10.4s, %[b0].16b, %[a0].4b[2]\n"
                ".inst 0x4fa0e84b // sdot v11.4s, %[b0].16b, %[a0].4b[3]\n"
                "ldr    %q[a0a], [%[a_ptr], #32]\n"
                ".inst 0x4f81e04c // sdot v12.4s, %[b0].16b, %[a1].4b[0]\n"
                ".inst 0x4fa1e04d // sdot v13.4s, %[b0].16b, %[a1].4b[1]\n"
                "ldr    %q[a1a], [%[a_ptr], #48]\n"
                ".inst 0x4f81e84e // sdot v14.4s, %[b0].16b, %[a1].4b[2]\n"
                ".inst 0x4fa1e84f // sdot v15.4s, %[b0].16b, %[a1].4b[3]\n"
                "ldr    %q[b0], [%[b_ptr], #48]\n"

                ".inst 0x4f80e070 // sdot v16.4s, %[b1].16b, %[a0].4b[0]\n"
                ".inst 0x4fa0e071 // sdot v17.4s, %[b1].16b, %[a0].4b[1]\n"
                ASM_PREFETCH("[%[a_ptr], #320]")
                ".inst 0x4f80e872 // sdot v18.4s, %[b1].16b, %[a0].4b[2]\n"
                ".inst 0x4fa0e873 // sdot v19.4s, %[b1].16b, %[a0].4b[3]\n"
                ".inst 0x4f81e074 // sdot v20.4s, %[b1].16b, %[a1].4b[0]\n"
                ".inst 0x4fa1e075 // sdot v21.4s, %[b1].16b, %[a1].4b[1]\n"
                ".inst 0x4f81e876 // sdot v22.4s, %[b1].16b, %[a1].4b[2]\n"
                ".inst 0x4fa1e877 // sdot v23.4s, %[b1].16b, %[a1].4b[3]\n"
                "ldr    %q[b1], [%[b_ptr], #64]\n"

                ".inst 0x4f80e098 // sdot v24.4s, %[b2].16b, %[a0].4b[0]\n"
                ".inst 0x4fa0e099 // sdot v25.4s, %[b2].16b, %[a0].4b[1]\n"
                ASM_PREFETCH("[%[b_ptr], #448]")
                ".inst 0x4f80e89a // sdot v26.4s, %[b2].16b, %[a0].4b[2]\n"
                ".inst 0x4fa0e89b // sdot v27.4s, %[b2].16b, %[a0].4b[3]\n"
                ".inst 0x4f81e09c // sdot v28.4s, %[b2].16b, %[a1].4b[0]\n"
                ".inst 0x4fa1e09d // sdot v29.4s, %[b2].16b, %[a1].4b[1]\n"
                ".inst 0x4f81e89e // sdot v30.4s, %[b2].16b, %[a1].4b[2]\n"
                ".inst 0x4fa1e89f // sdot v31.4s, %[b2].16b, %[a1].4b[3]\n"
                "ldr    %q[b2], [%[b_ptr], #80]\n"

                ".inst 0x4f85e048 // sdot v8.4s , %[b0].16b, %[a0a].4b[0]\n"
                ".inst 0x4fa5e049 // sdot v9.4s , %[b0].16b, %[a0a].4b[1]\n"
                "ldr    %q[a0], [%[a_ptr], #64]\n"
                ".inst 0x4f85e84a // sdot v10.4s, %[b0].16b, %[a0a].4b[2]\n"
                ".inst 0x4fa5e84b // sdot v11.4s, %[b0].16b, %[a0a].4b[3]\n"
                ".inst 0x4f86e04c // sdot v12.4s, %[b0].16b, %[a1a].4b[0]\n"
                "ldr    %q[a1], [%[a_ptr], #80]\n"
                ".inst 0x4fa6e04d // sdot v13.4s, %[b0].16b, %[a1a].4b[1]\n"
                ".inst 0x4f86e84e // sdot v14.4s, %[b0].16b, %[a1a].4b[2]\n"
                ".inst 0x4fa6e84f // sdot v15.4s, %[b0].16b, %[a1a].4b[3]\n"
                "ldr    %q[b0], [%[b_ptr], #96]\n"

                ".inst 0x4f85e070 // sdot v16.4s, %[b1].16b, %[a0a].4b[0]\n"
                ".inst 0x4fa5e071 // sdot v17.4s, %[b1].16b, %[a0a].4b[1]\n"
                ASM_PREFETCH("[%[b_ptr], #512]")
                ".inst 0x4f85e872 // sdot v18.4s, %[b1].16b, %[a0a].4b[2]\n"
                ".inst 0x4fa5e873 // sdot v19.4s, %[b1].16b, %[a0a].4b[3]\n"
                ".inst 0x4f86e074 // sdot v20.4s, %[b1].16b, %[a1a].4b[0]\n"
                ".inst 0x4fa6e075 // sdot v21.4s, %[b1].16b, %[a1a].4b[1]\n"
                ".inst 0x4f86e876 // sdot v22.4s, %[b1].16b, %[a1a].4b[2]\n"
                ".inst 0x4fa6e877 // sdot v23.4s, %[b1].16b, %[a1a].4b[3]\n"
                "ldr    %q[b1], [%[b_ptr], #112]\n"

                ".inst 0x4f85e098 // sdot v24.4s, %[b2].16b, %[a0a].4b[0]\n"
                ".inst 0x4fa5e099 // sdot v25.4s, %[b2].16b, %[a0a].4b[1]\n"
                "add    %[a_ptr], %[a_ptr], #64\n"
                ".inst 0x4f85e89a // sdot v26.4s, %[b2].16b, %[a0a].4b[2]\n"
                ".inst 0x4fa5e89b // sdot v27.4s, %[b2].16b, %[a0a].4b[3]\n"
                "add    %[b_ptr], %[b_ptr], #96\n"
                ".inst 0x4f86e09c // sdot v28.4s, %[b2].16b, %[a1a].4b[0]\n"
                ".inst 0x4fa6e09d // sdot v29.4s, %[b2].16b, %[a1a].4b[1]\n"
                "subs    %w[k], %w[k], #1\n"
                ".inst 0x4f86e89e // sdot v30.4s, %[b2].16b, %[a1a].4b[2]\n"
                ".inst 0x4fa6e89f // sdot v31.4s, %[b2].16b, %[a1a].4b[3]\n"
                "bne    1b\n"

                // Target to use when K is 1 or 2 (i.e. zero iterations of main loop)
                "4:\n"

                // Branch to alternative tail for odd K
                "cbnz    %w[oddk], 2f\n"

                // Detached final iteration (even K)
                ".inst 0x4f80e048 // sdot v8.4s , %[b0].16b, %[a0].4b[0]\n"
                ".inst 0x4fa0e049 // sdot v9.4s , %[b0].16b, %[a0].4b[1]\n"
                "ldr    %q[b2], [%[b_ptr], #32]\n"
                ".inst 0x4f80e84a // sdot v10.4s, %[b0].16b, %[a0].4b[2]\n"
                ".inst 0x4fa0e84b // sdot v11.4s, %[b0].16b, %[a0].4b[3]\n"
                "ldr    %q[a0a], [%[a_ptr], #32]\n"
                ".inst 0x4f81e04c // sdot v12.4s, %[b0].16b, %[a1].4b[0]\n"
                ".inst 0x4fa1e04d // sdot v13.4s, %[b0].16b, %[a1].4b[1]\n"
                "ldr    %q[a1a], [%[a_ptr], #48]\n"
                ".inst 0x4f81e84e // sdot v14.4s, %[b0].16b, %[a1].4b[2]\n"
                ".inst 0x4fa1e84f // sdot v15.4s, %[b0].16b, %[a1].4b[3]\n"
                "ldr    %q[b0], [%[b_ptr], #48]\n"

                ".inst 0x4f80e070 // sdot v16.4s, %[b1].16b, %[a0].4b[0]\n"
                ".inst 0x4fa0e071 // sdot v17.4s, %[b1].16b, %[a0].4b[1]\n"
                ".inst 0x4f80e872 // sdot v18.4s, %[b1].16b, %[a0].4b[2]\n"
                ".inst 0x4fa0e873 // sdot v19.4s, %[b1].16b, %[a0].4b[3]\n"
                ".inst 0x4f81e074 // sdot v20.4s, %[b1].16b, %[a1].4b[0]\n"
                ".inst 0x4fa1e075 // sdot v21.4s, %[b1].16b, %[a1].4b[1]\n"
                ".inst 0x4f81e876 // sdot v22.4s, %[b1].16b, %[a1].4b[2]\n"
                ".inst 0x4fa1e877 // sdot v23.4s, %[b1].16b, %[a1].4b[3]\n"
                "ldr    %q[b1], [%[b_ptr], #64]\n"

                ".inst 0x4f80e098 // sdot v24.4s, %[b2].16b, %[a0].4b[0]\n"
                ".inst 0x4fa0e099 // sdot v25.4s, %[b2].16b, %[a0].4b[1]\n"
                "add    %[a_ptr], %[a_ptr], #64\n"
                ".inst 0x4f80e89a // sdot v26.4s, %[b2].16b, %[a0].4b[2]\n"
                ".inst 0x4fa0e89b // sdot v27.4s, %[b2].16b, %[a0].4b[3]\n"
                ".inst 0x4f81e09c // sdot v28.4s, %[b2].16b, %[a1].4b[0]\n"
                ".inst 0x4fa1e09d // sdot v29.4s, %[b2].16b, %[a1].4b[1]\n"
                ".inst 0x4f81e89e // sdot v30.4s, %[b2].16b, %[a1].4b[2]\n"
                ".inst 0x4fa1e89f // sdot v31.4s, %[b2].16b, %[a1].4b[3]\n"
                "ldr    %q[b2], [%[b_ptr], #80]\n"

                ".inst 0x4f85e048 // sdot v8.4s , %[b0].16b, %[a0a].4b[0]\n"

                ".inst 0x4f85e070 // sdot v16.4s, %[b1].16b, %[a0a].4b[0]\n"
                "add    %[b_ptr], %[b_ptr], #96\n"
                ".inst 0x4fa5e049 // sdot v9.4s , %[b0].16b, %[a0a].4b[1]\n"
                "str    q8, [%[c_ptr], #0]\n"
                ".inst 0x4fa5e071 // sdot v17.4s, %[b1].16b, %[a0a].4b[1]\n"
                "str    q16, [%[c_ptr], #16]\n"
                ".inst 0x4f85e098 // sdot v24.4s, %[b2].16b, %[a0a].4b[0]\n"
                "str    q24, [%[c_ptr], #32]\n"

                ".inst 0x4fa5e099 // sdot v25.4s, %[b2].16b, %[a0a].4b[1]\n"
                "str    q9, [%[c_ptr], #48]\n"
                ".inst 0x4f85e84a // sdot v10.4s, %[b0].16b, %[a0a].4b[2]\n"
                "str    q17, [%[c_ptr], #64]\n"
                ".inst 0x4f85e872 // sdot v18.4s, %[b1].16b, %[a0a].4b[2]\n"
                "str    q25, [%[c_ptr], #80]\n"
                ".inst 0x4f85e89a // sdot v26.4s, %[b2].16b, %[a0a].4b[2]\n"
                "str    q10, [%[c_ptr], #96]\n"

                ".inst 0x4fa5e84b // sdot v11.4s, %[b0].16b, %[a0a].4b[3]\n"
                "str    q18, [%[c_ptr], #112]\n"
                ".inst 0x4fa5e873 // sdot v19.4s, %[b1].16b, %[a0a].4b[3]\n"
                "str    q26, [%[c_ptr], #128]\n"
                ".inst 0x4fa5e89b // sdot v27.4s, %[b2].16b, %[a0a].4b[3]\n"
                "str    q11, [%[c_ptr], #144]\n"

                ".inst 0x4f86e04c // sdot v12.4s, %[b0].16b, %[a1a].4b[0]\n"
                "str    q19, [%[c_ptr], #160]\n"
                ".inst 0x4f86e074 // sdot v20.4s, %[b1].16b, %[a1a].4b[0]\n"
                "str    q27, [%[c_ptr], #176]\n"
                ".inst 0x4f86e09c // sdot v28.4s, %[b2].16b, %[a1a].4b[0]\n"
                "str    q12, [%[c_ptr], #192]\n"

                ".inst 0x4fa6e04d // sdot v13.4s, %[b0].16b, %[a1a].4b[1]\n"
                "str    q20, [%[c_ptr], #208]\n"
                ".inst 0x4fa6e075 // sdot v21.4s, %[b1].16b, %[a1a].4b[1]\n"
                "str    q28, [%[c_ptr], #224]\n"
                ".inst 0x4fa6e09d // sdot v29.4s, %[b2].16b, %[a1a].4b[1]\n"
                "str    q13, [%[c_ptr], #240]\n"

                ".inst 0x4f86e84e // sdot v14.4s, %[b0].16b, %[a1a].4b[2]\n"
                "str    q21, [%[c_ptr], #256]\n"
                ".inst 0x4f86e876 // sdot v22.4s, %[b1].16b, %[a1a].4b[2]\n"
                "str    q29, [%[c_ptr], #272]\n"
                ".inst 0x4f86e89e // sdot v30.4s, %[b2].16b, %[a1a].4b[2]\n"
                "str    q14, [%[c_ptr], #288]\n"

                ".inst 0x4fa6e84f // sdot v15.4s, %[b0].16b, %[a1a].4b[3]\n"
                "str    q22, [%[c_ptr], #304]\n"
                ".inst 0x4fa6e877 // sdot v23.4s, %[b1].16b, %[a1a].4b[3]\n"
                "str    q30, [%[c_ptr], #320]\n"
                ".inst 0x4fa6e89f // sdot v31.4s, %[b2].16b, %[a1a].4b[3]\n"
                "str    q15, [%[c_ptr], #336]\n"

                "b    3f\n"

                // Detached final iteration (odd K)
                "2:\n"
                ".inst 0x4f80e048 // sdot v8.4s , %[b0].16b, %[a0].4b[0]\n"
                "ldr    %q[b2], [%[b_ptr], #32]\n"
                ".inst 0x4f80e070 // sdot v16.4s, %[b1].16b, %[a0].4b[0]\n"
                ".inst 0x4fa0e049 // sdot v9.4s , %[b0].16b, %[a0].4b[1]\n"
                "str    q8, [%[c_ptr], #0]\n"
                ".inst 0x4fa0e071 // sdot v17.4s, %[b1].16b, %[a0].4b[1]\n"
                "str    q16, [%[c_ptr], #16]\n"
                ".inst 0x4f80e098 // sdot v24.4s, %[b2].16b, %[a0].4b[0]\n"
                "add    %[b_ptr], %[b_ptr], #48\n"
                "add    %[a_ptr], %[a_ptr], #32\n"
                "str    q24, [%[c_ptr], #32]\n"
                ".inst 0x4fa0e099 // sdot v25.4s, %[b2].16b, %[a0].4b[1]\n"
                "str    q9, [%[c_ptr], #48]\n"

                ".inst 0x4f80e84a // sdot v10.4s, %[b0].16b, %[a0].4b[2]\n"
                "str    q17, [%[c_ptr], #64]\n"
                ".inst 0x4f80e872 // sdot v18.4s, %[b1].16b, %[a0].4b[2]\n"
                "str    q25, [%[c_ptr], #80]\n"
                ".inst 0x4f80e89a // sdot v26.4s, %[b2].16b, %[a0].4b[2]\n"
                "str    q10, [%[c_ptr], #96]\n"

                ".inst 0x4fa0e84b // sdot v11.4s, %[b0].16b, %[a0].4b[3]\n"
                "str    q18, [%[c_ptr], #112]\n"
                ".inst 0x4fa0e873 // sdot v19.4s, %[b1].16b, %[a0].4b[3]\n"
                "str    q26, [%[c_ptr], #128]\n"
                ".inst 0x4fa0e89b // sdot v27.4s, %[b2].16b, %[a0].4b[3]\n"
                "str    q11, [%[c_ptr], #144]\n"

                ".inst 0x4f81e04c // sdot v12.4s, %[b0].16b, %[a1].4b[0]\n"
                "str    q19, [%[c_ptr], #160]\n"
                ".inst 0x4f81e074 // sdot v20.4s, %[b1].16b, %[a1].4b[0]\n"
                "str    q27, [%[c_ptr], #176]\n"
                ".inst 0x4f81e09c // sdot v28.4s, %[b2].16b, %[a1].4b[0]\n"
                "str    q12, [%[c_ptr], #192]\n"

                ".inst 0x4fa1e04d // sdot v13.4s, %[b0].16b, %[a1].4b[1]\n"
                "str    q20, [%[c_ptr], #208]\n"
                ".inst 0x4fa1e075 // sdot v21.4s, %[b1].16b, %[a1].4b[1]\n"
                "str    q28, [%[c_ptr], #224]\n"
                ".inst 0x4fa1e09d // sdot v29.4s, %[b2].16b, %[a1].4b[1]\n"
                "str    q13, [%[c_ptr], #240]\n"

                ".inst 0x4f81e84e // sdot v14.4s, %[b0].16b, %[a1].4b[2]\n"
                "str    q21, [%[c_ptr], #256]\n"
                ".inst 0x4f81e876 // sdot v22.4s, %[b1].16b, %[a1].4b[2]\n"
                "str    q29, [%[c_ptr], #272]\n"
                ".inst 0x4f81e89e // sdot v30.4s, %[b2].16b, %[a1].4b[2]\n"
                "str    q14, [%[c_ptr], #288]\n"

                ".inst 0x4fa1e84f // sdot v15.4s, %[b0].16b, %[a1].4b[3]\n"
                "str    q22, [%[c_ptr], #304]\n"
                ".inst 0x4fa1e877 // sdot v23.4s, %[b1].16b, %[a1].4b[3]\n"
                "str    q30, [%[c_ptr], #320]\n"
                ".inst 0x4fa1e89f // sdot v31.4s, %[b2].16b, %[a1].4b[3]\n"
                "str    q15, [%[c_ptr], #336]\n"


                // Common tail
                "3:\n"
                "str    q23, [%[c_ptr], #352]\n"
                "str    q31, [%[c_ptr], #368]\n"
                "add    %[c_ptr], %[c_ptr], #384\n"

            :
              [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [c_ptr] "+r" (c_ptr),
              [a0] "+w" (a0), [a1] "+w" (a1), [a0a] "+w" (a0a), [a1a] "+w" (a1a),
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
