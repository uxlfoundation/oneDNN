//
// SPDX-FileCopyrightText: Copyright 2017, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#ifdef __aarch64__

#include <arm_neon.h>

#include "asmlib.hpp"

namespace kai {
namespace ops {

void a64_gemm_s8_4x4(const int8_t *Apanel, const int8_t *Bpanel, int32_t *Cpanel, int ablocks, int bblocks, int K) {
    const int8_t *a_ptr = Apanel;
    int32_t *c_ptr = Cpanel;

    K /= 16;
    int oddk = (K & 1);

    for (int yb=0; yb<ablocks; yb++) {
        const int8_t *a_ptr0 = a_ptr;
        const int8_t *b_ptr = Bpanel;

        for (int xb=0; xb<bblocks; xb++) {
            a_ptr = a_ptr0;

            int k = ((K+1)/2)-1;

            register int8x16_t b0  asm("v4");
            register int8x16_t b1  asm("v5");
            register int8x16_t b2  asm("v6");
            register int8x16_t b3  asm("v7");
            register int8x16_t b0a asm("v8");
            register int8x16_t b1a asm("v9");
            register int8x16_t b2a asm("v10");
            register int8x16_t b3a asm("v11");

            __asm __volatile (
                "movi    v16.4s, #0x0\n"
                "ldr    q0, [%[a_ptr]]\n"
                "movi    v17.4s, #0x0\n"
                "ldr    %q[b0], [%[b_ptr]]\n"
                "movi    v18.4s, #0x0\n"
                "ldr    %q[b1], [%[b_ptr], #16]\n"
                "movi    v19.4s, #0x0\n"
                "ldr    %q[b2], [%[b_ptr], #32]\n"
                "movi    v20.4s, #0x0\n"
                "ldr    %q[b3], [%[b_ptr], #48]\n"
                "movi    v21.4s, #0x0\n"
                "ldr    q1, [%[a_ptr], #16]\n"
                "movi    v22.4s, #0x0\n"
                "ldr    q2, [%[a_ptr], #32]\n"
                "movi    v23.4s, #0x0\n"
                "ldr    q3, [%[a_ptr], #48]\n"
                "movi    v24.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #64]")
                "movi    v25.4s, #0x0\n"
                ASM_PREFETCH("[%[a_ptr], #64]")
                "movi    v26.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #128]")
                "movi    v27.4s, #0x0\n"
                ASM_PREFETCH("[%[a_ptr], #128]")
                "movi    v28.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #192]")
                "movi    v29.4s, #0x0\n"
                ASM_PREFETCH("[%[a_ptr], #192]")
                "movi    v30.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #256]")
                "movi    v31.4s, #0x0\n"
                ASM_PREFETCH("[%[a_ptr], #256]")

                // Start of unroll 0 (first iteration)
                "smull    v12.8h, v0.8b, %[b0].8b\n"
                "smull    v13.8h, v0.8b, %[b1].8b\n"

                // Skip loop if we are doing zero iterations of it.
                "cbz    %w[k], 4f\n"

                // Unroll 0 continuation (branch target)
                "1:\n"
                "smull    v14.8h, v0.8b, %[b2].8b\n"
                "subs    %w[k], %w[k], #1\n"
                "smull    v15.8h, v0.8b, %[b3].8b\n"
                "ldr    %q[b0a], [%[b_ptr], #64]\n"
                "smlal2    v12.8h, v0.16b, %[b0].16b\n"
                "smlal2    v13.8h, v0.16b, %[b1].16b\n"
                "ldr    %q[b1a], [%[b_ptr], #80]\n"
                "smlal2    v14.8h, v0.16b, %[b2].16b\n"
                "smlal2    v15.8h, v0.16b, %[b3].16b\n"
                "ldr     q0, [%[a_ptr], #64]\n"

                "sadalp    v16.4s, v12.8h\n"
                "smull    v12.8h, v1.8b, %[b0].8b\n"
                "sadalp    v17.4s, v13.8h\n"
                "sadalp    v18.4s, v14.8h\n"
                "smull    v13.8h, v1.8b, %[b1].8b\n"
                "sadalp    v19.4s, v15.8h\n"
                "smull    v14.8h, v1.8b, %[b2].8b\n"
                "ldr    %q[b2a], [%[b_ptr], #96]\n"
                "smull    v15.8h, v1.8b, %[b3].8b\n"
                "smlal2    v12.8h, v1.16b, %[b0].16b\n"
                "ldr    %q[b3a], [%[b_ptr], #112]\n"
                "smlal2    v13.8h, v1.16b, %[b1].16b\n"
                "add    %[b_ptr], %[b_ptr], #128\n"
                "smlal2    v14.8h, v1.16b, %[b2].16b\n"
                "smlal2    v15.8h, v1.16b, %[b3].16b\n"
                "ldr     q1, [%[a_ptr], #80]\n"

                "sadalp    v20.4s, v12.8h\n"
                "smull    v12.8h, v2.8b, %[b0].8b\n"
                "sadalp    v21.4s, v13.8h\n"
                "sadalp    v22.4s, v14.8h\n"
                "smull    v13.8h, v2.8b, %[b1].8b\n"
                "sadalp    v23.4s, v15.8h\n"
                "smull    v14.8h, v2.8b, %[b2].8b\n"
                "smull    v15.8h, v2.8b, %[b3].8b\n"
                "smlal2    v12.8h, v2.16b, %[b0].16b\n"
                ASM_PREFETCH("[%[b_ptr], #192]")
                "smlal2    v13.8h, v2.16b, %[b1].16b\n"
                "smlal2    v14.8h, v2.16b, %[b2].16b\n"
                ASM_PREFETCH("[%[a_ptr], #320]")
                "smlal2    v15.8h, v2.16b, %[b3].16b\n"
                "ldr     q2, [%[a_ptr], #96]\n"

                "sadalp    v24.4s, v12.8h\n"
                "smull    v12.8h, v3.8b, %[b0].8b\n"
                "sadalp    v25.4s, v13.8h\n"
                "sadalp    v26.4s, v14.8h\n"
                "smull    v13.8h, v3.8b, %[b1].8b\n"
                "sadalp    v27.4s, v15.8h\n"
                "smull    v14.8h, v3.8b, %[b2].8b\n"
                "smull    v15.8h, v3.8b, %[b3].8b\n"
                "smlal2    v12.8h, v3.16b, %[b0].16b\n"
                "ldr     %q[b0], [%[b_ptr], #0]\n"
                "smlal2    v13.8h, v3.16b, %[b1].16b\n"
                "smlal2    v14.8h, v3.16b, %[b2].16b\n"
                "smlal2    v15.8h, v3.16b, %[b3].16b\n"
                "ldr     q3, [%[a_ptr], #112]\n"

                // Unroll 1
                "sadalp    v28.4s, v12.8h\n"
                "smull    v12.8h, v0.8b, %[b0a].8b\n"
                "sadalp    v29.4s, v13.8h\n"
                "sadalp    v30.4s, v14.8h\n"
                "smull    v13.8h, v0.8b, %[b1a].8b\n"
                "sadalp    v31.4s, v15.8h\n"
                "smull    v14.8h, v0.8b, %[b2a].8b\n"
                "smull    v15.8h, v0.8b, %[b3a].8b\n"
                "ldr     %q[b1], [%[b_ptr], #16]\n"
                "smlal2    v12.8h, v0.16b, %[b0a].16b\n"
                "smlal2    v13.8h, v0.16b, %[b1a].16b\n"
                "ldr     %q[b2], [%[b_ptr], #32]\n"
                "smlal2    v14.8h, v0.16b, %[b2a].16b\n"
                "smlal2    v15.8h, v0.16b, %[b3a].16b\n"
                "ldr     q0, [%[a_ptr], #128]\n"

                "sadalp    v16.4s, v12.8h\n"
                "smull    v12.8h, v1.8b, %[b0a].8b\n"
                "sadalp    v17.4s, v13.8h\n"
                "sadalp    v18.4s, v14.8h\n"
                "smull    v13.8h, v1.8b, %[b1a].8b\n"
                "sadalp    v19.4s, v15.8h\n"
                "add    %[a_ptr], %[a_ptr], #128\n"
                "smull    v14.8h, v1.8b, %[b2a].8b\n"
                "smull    v15.8h, v1.8b, %[b3a].8b\n"
                "ldr     %q[b3], [%[b_ptr], #48]\n"
                "smlal2    v12.8h, v1.16b, %[b0a].16b\n"
                "smlal2    v13.8h, v1.16b, %[b1a].16b\n"
                "smlal2    v14.8h, v1.16b, %[b2a].16b\n"
                "smlal2    v15.8h, v1.16b, %[b3a].16b\n"
                "ldr     q1, [%[a_ptr], #16]\n"

                "sadalp    v20.4s, v12.8h\n"
                "smull    v12.8h, v2.8b, %[b0a].8b\n"
                "sadalp    v21.4s, v13.8h\n"
                "sadalp    v22.4s, v14.8h\n"
                "smull    v13.8h, v2.8b, %[b1a].8b\n"
                "sadalp    v23.4s, v15.8h\n"
                "smull    v14.8h, v2.8b, %[b2a].8b\n"
                "smull    v15.8h, v2.8b, %[b3a].8b\n"
                "smlal2    v12.8h, v2.16b, %[b0a].16b\n"
                ASM_PREFETCH("[%[b_ptr], #256]")
                "smlal2    v13.8h, v2.16b, %[b1a].16b\n"
                "smlal2    v14.8h, v2.16b, %[b2a].16b\n"
                ASM_PREFETCH("[%[a_ptr], #256]")
                "smlal2    v15.8h, v2.16b, %[b3a].16b\n"
                "ldr     q2, [%[a_ptr], #32]\n"

                "sadalp    v24.4s, v12.8h\n"
                "smull    v12.8h, v3.8b, %[b0a].8b\n"
                "sadalp    v25.4s, v13.8h\n"
                "sadalp    v26.4s, v14.8h\n"
                "smull    v13.8h, v3.8b, %[b1a].8b\n"
                "sadalp    v27.4s, v15.8h\n"
                "smull    v14.8h, v3.8b, %[b2a].8b\n"
                "smull    v15.8h, v3.8b, %[b3a].8b\n"
                "smlal2    v12.8h, v3.16b, %[b0a].16b\n"
                "smlal2    v13.8h, v3.16b, %[b1a].16b\n"
                "smlal2    v14.8h, v3.16b, %[b2a].16b\n"
                "smlal2    v15.8h, v3.16b, %[b3a].16b\n"
                "ldr     q3, [%[a_ptr], #48]\n"

                // Start of unroll 0 for next iteration.
                "sadalp    v28.4s, v12.8h\n"
                "smull    v12.8h, v0.8b, %[b0].8b\n"
                "sadalp    v29.4s, v13.8h\n"
                "sadalp    v30.4s, v14.8h\n"
                "smull    v13.8h, v0.8b, %[b1].8b\n"
                "sadalp    v31.4s, v15.8h\n"
                "bne    1b\n"

                // Target to use when K=1 or 2 (i.e. zero iterations of main loop)
                "4:\n"

                // Branch to alternative tail for odd K
                "cbnz    %w[oddk], 2f\n"

                // Detached final iteration (even K)
                "smull    v14.8h, v0.8b, %[b2].8b\n"
                "smull    v15.8h, v0.8b, %[b3].8b\n"
                "ldr    %q[b0a], [%[b_ptr], #64]\n"
                "smlal2    v12.8h, v0.16b, %[b0].16b\n"
                "smlal2    v13.8h, v0.16b, %[b1].16b\n"
                "ldr    %q[b1a], [%[b_ptr], #80]\n"
                "smlal2    v14.8h, v0.16b, %[b2].16b\n"
                "smlal2    v15.8h, v0.16b, %[b3].16b\n"
                "ldr     q0, [%[a_ptr], #64]\n"

                "sadalp    v16.4s, v12.8h\n"
                "smull    v12.8h, v1.8b, %[b0].8b\n"
                "sadalp    v17.4s, v13.8h\n"
                "sadalp    v18.4s, v14.8h\n"
                "smull    v13.8h, v1.8b, %[b1].8b\n"
                "sadalp    v19.4s, v15.8h\n"
                "smull    v14.8h, v1.8b, %[b2].8b\n"
                "ldr    %q[b2a], [%[b_ptr], #96]\n"
                "smull    v15.8h, v1.8b, %[b3].8b\n"
                "smlal2    v12.8h, v1.16b, %[b0].16b\n"
                "ldr    %q[b3a], [%[b_ptr], #112]\n"
                "smlal2    v13.8h, v1.16b, %[b1].16b\n"
                "add    %[b_ptr], %[b_ptr], #128\n"
                "smlal2    v14.8h, v1.16b, %[b2].16b\n"
                "smlal2    v15.8h, v1.16b, %[b3].16b\n"
                "ldr     q1, [%[a_ptr], #80]\n"

                "sadalp    v20.4s, v12.8h\n"
                "smull    v12.8h, v2.8b, %[b0].8b\n"
                "sadalp    v21.4s, v13.8h\n"
                "sadalp    v22.4s, v14.8h\n"
                "smull    v13.8h, v2.8b, %[b1].8b\n"
                "sadalp    v23.4s, v15.8h\n"
                "smull    v14.8h, v2.8b, %[b2].8b\n"
                "smull    v15.8h, v2.8b, %[b3].8b\n"
                "smlal2    v12.8h, v2.16b, %[b0].16b\n"
                "smlal2    v13.8h, v2.16b, %[b1].16b\n"
                "smlal2    v14.8h, v2.16b, %[b2].16b\n"
                "smlal2    v15.8h, v2.16b, %[b3].16b\n"
                "ldr     q2, [%[a_ptr], #96]\n"

                "sadalp    v24.4s, v12.8h\n"
                "smull    v12.8h, v3.8b, %[b0].8b\n"
                "sadalp    v25.4s, v13.8h\n"
                "sadalp    v26.4s, v14.8h\n"
                "smull    v13.8h, v3.8b, %[b1].8b\n"
                "sadalp    v27.4s, v15.8h\n"
                "smull    v14.8h, v3.8b, %[b2].8b\n"
                "smull    v15.8h, v3.8b, %[b3].8b\n"
                "smlal2    v12.8h, v3.16b, %[b0].16b\n"
                "smlal2    v13.8h, v3.16b, %[b1].16b\n"
                "smlal2    v14.8h, v3.16b, %[b2].16b\n"
                "smlal2    v15.8h, v3.16b, %[b3].16b\n"
                "ldr     q3, [%[a_ptr], #112]\n"

                // Unroll 1
                "sadalp    v28.4s, v12.8h\n"
                "smull    v12.8h, v0.8b, %[b0a].8b\n"
                "sadalp    v29.4s, v13.8h\n"
                "sadalp    v30.4s, v14.8h\n"
                "smull    v13.8h, v0.8b, %[b1a].8b\n"
                "sadalp    v31.4s, v15.8h\n"
                "smull    v14.8h, v0.8b, %[b2a].8b\n"
                "add    %[a_ptr], %[a_ptr], #128\n"
                "smull    v15.8h, v0.8b, %[b3a].8b\n"
                "smlal2    v12.8h, v0.16b, %[b0a].16b\n"
                "smlal2    v13.8h, v0.16b, %[b1a].16b\n"
                "smlal2    v14.8h, v0.16b, %[b2a].16b\n"
                "smlal2    v15.8h, v0.16b, %[b3a].16b\n"

                "sadalp    v16.4s, v12.8h\n"
                "smull    v12.8h, v1.8b, %[b0a].8b\n"
                "sadalp    v17.4s, v13.8h\n"
                "sadalp    v18.4s, v14.8h\n"
                "smull    v13.8h, v1.8b, %[b1a].8b\n"
                "sadalp    v19.4s, v15.8h\n"
                "smull    v14.8h, v1.8b, %[b2a].8b\n"
                "smull    v15.8h, v1.8b, %[b3a].8b\n"
                "smlal2    v12.8h, v1.16b, %[b0a].16b\n"
                "addp    v16.4s, v16.4s, v17.4s\n"
                "smlal2    v13.8h, v1.16b, %[b1a].16b\n"
                "addp    v17.4s, v18.4s, v19.4s\n"
                "smlal2    v14.8h, v1.16b, %[b2a].16b\n"
                "smlal2    v15.8h, v1.16b, %[b3a].16b\n"

                "sadalp    v20.4s, v12.8h\n"
                "smull    v12.8h, v2.8b, %[b0a].8b\n"
                "sadalp    v21.4s, v13.8h\n"
                "sadalp    v22.4s, v14.8h\n"
                "smull    v13.8h, v2.8b, %[b1a].8b\n"
                "sadalp    v23.4s, v15.8h\n"
                "addp    v16.4s, v16.4s, v17.4s\n"
                "smull    v14.8h, v2.8b, %[b2a].8b\n"
                "addp    v18.4s, v20.4s, v21.4s\n"
                "addp    v19.4s, v22.4s, v23.4s\n"
                "smull    v15.8h, v2.8b, %[b3a].8b\n"
                "smlal2    v12.8h, v2.16b, %[b0a].16b\n"
                "str    q16, [%[c_ptr]]\n"
                "smlal2    v13.8h, v2.16b, %[b1a].16b\n"
                "smlal2    v14.8h, v2.16b, %[b2a].16b\n"
                "smlal2    v15.8h, v2.16b, %[b3a].16b\n"

                "sadalp    v24.4s, v12.8h\n"
                "smull    v12.8h, v3.8b, %[b0a].8b\n"
                "sadalp    v25.4s, v13.8h\n"
                "sadalp    v26.4s, v14.8h\n"
                "smull    v13.8h, v3.8b, %[b1a].8b\n"
                "sadalp    v27.4s, v15.8h\n"
                "addp    v17.4s, v18.4s, v19.4s\n"
                "smull    v14.8h, v3.8b, %[b2a].8b\n"
                "addp    v20.4s, v24.4s, v25.4s\n"
                "addp    v21.4s, v26.4s, v27.4s\n"
                "smull    v15.8h, v3.8b, %[b3a].8b\n"
                "smlal2    v12.8h, v3.16b, %[b0a].16b\n"
                "str    q17, [%[c_ptr], #16]\n"
                "smlal2    v13.8h, v3.16b, %[b1a].16b\n"
                "smlal2    v14.8h, v3.16b, %[b2a].16b\n"
                "addp    v18.4s, v20.4s, v21.4s\n"
                "smlal2    v15.8h, v3.16b, %[b3a].16b\n"
                "b    3f\n"

                // Detached final iteration (odd K)
                "2:\n"
                "smull    v14.8h, v0.8b, %[b2].8b\n"
                "add    %[a_ptr], %[a_ptr], #64\n"
                "smull    v15.8h, v0.8b, %[b3].8b\n"
                "add    %[b_ptr], %[b_ptr], #64\n"
                "smlal2    v12.8h, v0.16b, %[b0].16b\n"
                "smlal2    v13.8h, v0.16b, %[b1].16b\n"
                "smlal2    v14.8h, v0.16b, %[b2].16b\n"
                "smlal2    v15.8h, v0.16b, %[b3].16b\n"

                "sadalp    v16.4s, v12.8h\n"
                "smull    v12.8h, v1.8b, %[b0].8b\n"
                "sadalp    v17.4s, v13.8h\n"
                "sadalp    v18.4s, v14.8h\n"
                "smull    v13.8h, v1.8b, %[b1].8b\n"
                "sadalp    v19.4s, v15.8h\n"
                "smull    v14.8h, v1.8b, %[b2].8b\n"
                "smull    v15.8h, v1.8b, %[b3].8b\n"
                "smlal2    v12.8h, v1.16b, %[b0].16b\n"
                "addp    v16.4s, v16.4s, v17.4s\n"
                "smlal2    v13.8h, v1.16b, %[b1].16b\n"
                "addp    v17.4s, v18.4s, v19.4s\n"
                "smlal2    v14.8h, v1.16b, %[b2].16b\n"
                "smlal2    v15.8h, v1.16b, %[b3].16b\n"

                "sadalp    v20.4s, v12.8h\n"
                "smull    v12.8h, v2.8b, %[b0].8b\n"
                "sadalp    v21.4s, v13.8h\n"
                "sadalp    v22.4s, v14.8h\n"
                "smull    v13.8h, v2.8b, %[b1].8b\n"
                "sadalp    v23.4s, v15.8h\n"
                "addp    v16.4s, v16.4s, v17.4s\n"
                "smull    v14.8h, v2.8b, %[b2].8b\n"
                "addp    v18.4s, v20.4s, v21.4s\n"
                "addp    v19.4s, v22.4s, v23.4s\n"
                "smull    v15.8h, v2.8b, %[b3].8b\n"
                "smlal2    v12.8h, v2.16b, %[b0].16b\n"
                "str    q16, [%[c_ptr]]\n"
                "smlal2    v13.8h, v2.16b, %[b1].16b\n"
                "smlal2    v14.8h, v2.16b, %[b2].16b\n"
                "smlal2    v15.8h, v2.16b, %[b3].16b\n"

                "sadalp    v24.4s, v12.8h\n"
                "smull    v12.8h, v3.8b, %[b0].8b\n"
                "sadalp    v25.4s, v13.8h\n"
                "sadalp    v26.4s, v14.8h\n"
                "smull    v13.8h, v3.8b, %[b1].8b\n"
                "sadalp    v27.4s, v15.8h\n"
                "addp    v17.4s, v18.4s, v19.4s\n"
                "smull    v14.8h, v3.8b, %[b2].8b\n"
                "addp    v20.4s, v24.4s, v25.4s\n"
                "addp    v21.4s, v26.4s, v27.4s\n"
                "smull    v15.8h, v3.8b, %[b3].8b\n"
                "smlal2    v12.8h, v3.16b, %[b0].16b\n"
                "str    q17, [%[c_ptr], #16]\n"
                "smlal2    v13.8h, v3.16b, %[b1].16b\n"
                "smlal2    v14.8h, v3.16b, %[b2].16b\n"
                "addp    v18.4s, v20.4s, v21.4s\n"
                "smlal2    v15.8h, v3.16b, %[b3].16b\n"

                "3:\n"

                // Final additions
                "sadalp    v28.4s, v12.8h\n"
                "str    q18, [%[c_ptr], #32]\n"
                "sadalp    v29.4s, v13.8h\n"
                "sadalp    v30.4s, v14.8h\n"
                "sadalp    v31.4s, v15.8h\n"

                // Horizontal reduction, phase 1
                "addp    v22.4s, v28.4s, v29.4s\n"
                "addp    v23.4s, v30.4s, v31.4s\n"

                // Horizontal reduction, phase 2
                "addp    v19.4s, v22.4s, v23.4s\n"
                "str    q19, [%[c_ptr], #48]\n"
                "add    %[c_ptr], %[c_ptr], #64\n"

            :
              [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [c_ptr] "+r" (c_ptr),
              [b0] "+w" (b0), [b1] "+w" (b1), [b2] "+w" (b2), [b3] "+w" (b3),
              [b0a] "+w" (b0a), [b1a] "+w" (b1a), [b2a] "+w" (b2a), [b3a] "+w" (b3a),
              [k] "+r" (k)
            : [oddk] "r" (oddk)
            : "x20", "x21", "v0","v1","v2","v3","v12","v13","v14","v15","v16","v17","v18","v19",
              "v20","v21","v22","v23","v24","v25","v26","v27","v28","v29","v30","v31", "cc");
        }
    }
}

}  // namespace ops
}  // namespace kai

#endif // __aarch64__
