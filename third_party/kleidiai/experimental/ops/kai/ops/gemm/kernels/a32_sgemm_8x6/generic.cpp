//
// SPDX-FileCopyrightText: Copyright 2017-2018, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#ifdef __arm__

#include <arm_neon.h>

#include "asmlib.hpp"

// Kernel implementation.
//
// Assume that "Apanel" points to a chunk of A blocks (each size 6xK) in read-order.
// Assume that "Bpanel" points to a chunk of B blocks (each size 8xK) in read-order.
// Assume that "Cpanel" points to a chunk of C output blocks (each size
// 8x6), the chunks being arranged in a row major fashion.
//
// Note that the intent of this is that either ablocks or bblocks will be 1
// - this construction allows the output loop to proceed in either order.

namespace kai {
namespace ops {

void a32_sgemm_8x6(const float *Apanel, const float *Bpanel, float *Cpanel, int ablocks, int bblocks, int K) {
    const float *a_ptr = Apanel;
    float *c_ptr = Cpanel;

    for (int yb=0; yb<ablocks; yb++) {
        const float *a_ptr0 = a_ptr;
        const float *b_ptr = Bpanel;

        for (int xb=0; xb<bblocks; xb++) {
            a_ptr = a_ptr0;
            int tails = (K & 3);
            if (tails == 0) {
                tails = 4;
            }
            int k = ((K+3)/4) - 1;

            __asm __volatile (
                "vmov.i32    q4, #0\n"
                "vld1.32    {d0-d1}, [%[a_ptr] :64]!\n"
                "vmov.i32    q5, #0\n"
                "vld1.32    {d4-d5}, [%[b_ptr] :128]!\n"
                "vmov.i32    q6, #0\n"
                ASM_PREFETCH("[%[a_ptr], #48]")
                "vmov.i32    q7, #0\n"
                ASM_PREFETCH("[%[b_ptr], #48]")
                "vmov.i32    q8, #0\n"
                ASM_PREFETCH("[%[a_ptr], #112]")
                "vmov.i32    q9, #0\n"
                ASM_PREFETCH("[%[b_ptr], #112]")
                "vmov.i32    q10, #0\n"
                "vmov.i32    q11, #0\n"
                "vmov.i32    q12, #0\n"
                "vmov.i32    q13, #0\n"
                ASM_PREFETCH("[%[a_ptr], #176]")
                "vmov.i32    q14, #0\n"
                ASM_PREFETCH("[%[b_ptr], #176]")
                "vmov.i32    q15, #0\n"

                "cmp        %[k], #0\n"
                "beq        6f\n"

                "1:\n"
                // Unroll 0
                "vmla.f32    q4, q2, d0[0]\n"
                "vld1.32    {d2-d3}, [%[a_ptr] :64]!\n"
                "vmla.f32    q5, q2, d0[1]\n"
                "vmla.f32    q6, q2, d1[0]\n"
                "vld1.32    {d6-d7}, [%[b_ptr] :128]!\n"
                "vmla.f32    q7, q2, d1[1]\n"
                "vmla.f32    q8, q2, d2[0]\n"
                "vmla.f32    q9, q2, d2[1]\n"
                "vld1.32    {d4-d5}, [%[b_ptr] :128]!\n"

                "vmla.f32    q10, q3, d0[0]\n"
                "vmla.f32    q11, q3, d0[1]\n"
                "vmla.f32    q12, q3, d1[0]\n"
                "vmla.f32    q13, q3, d1[1]\n"
                "vld1.32    {d0-d1}, [%[a_ptr] :64]!\n"
                "vmla.f32    q14, q3, d2[0]\n"
                "vmla.f32    q15, q3, d2[1]\n"
                "vld1.32    {d6-d7}, [%[b_ptr] :128]!\n"

                // Unroll 1
                "vmla.f32    q4, q2, d3[0]\n"
                "subs        %[k], %[k], #1\n"
                "vmla.f32    q5, q2, d3[1]\n"
                ASM_PREFETCH("[%[a_ptr], #208]")
                "vmla.f32    q6, q2, d0[0]\n"
                "vmla.f32    q7, q2, d0[1]\n"
                ASM_PREFETCH("[%[b_ptr], #192]")
                "vmla.f32    q8, q2, d1[0]\n"
                "vmla.f32    q9, q2, d1[1]\n"
                "vld1.32    {d4-d5}, [%[b_ptr] :128]!\n"

                "vmla.f32    q10, q3, d3[0]\n"
                "vmla.f32    q11, q3, d3[1]\n"
                "vld1.32    {d2-d3}, [%[a_ptr] :64]!\n"
                "vmla.f32    q12, q3, d0[0]\n"
                "vmla.f32    q13, q3, d0[1]\n"
                "vmla.f32    q14, q3, d1[0]\n"
                "vmla.f32    q15, q3, d1[1]\n"
                "vld1.32    {d0-d1}, [%[a_ptr] :64]!\n"

                // Unroll 2
                "vmla.f32    q4, q2, d2[0]\n"
                "vmla.f32    q5, q2, d2[1]\n"
                "vld1.32    {d6-d7}, [%[b_ptr] :128]!\n"
                "vmla.f32    q6, q2, d3[0]\n"
                "vmla.f32    q7, q2, d3[1]\n"
                ASM_PREFETCH("[%[a_ptr], #240]")
                "vmla.f32    q8, q2, d0[0]\n"
                "vmla.f32    q9, q2, d0[1]\n"
                "vld1.32    {d4-d5}, [%[b_ptr] :128]!\n"

                "vmla.f32    q10, q3, d2[0]\n"
                "vmla.f32    q11, q3, d2[1]\n"
                ASM_PREFETCH("[%[b_ptr], #208]")
                "vmla.f32    q12, q3, d3[0]\n"
                "vmla.f32    q13, q3, d3[1]\n"
                "vld1.32    {d2-d3}, [%[a_ptr] :64]!\n"
                "vmla.f32    q14, q3, d0[0]\n"
                "vmla.f32    q15, q3, d0[1]\n"
                "vld1.32    {d6-d7}, [%[b_ptr] :128]!\n"

                // Unroll 3
                "vmla.f32    q4, q2, d1[0]\n"
                "vmla.f32    q5, q2, d1[1]\n"
                "vmla.f32    q6, q2, d2[0]\n"
                "vmla.f32    q7, q2, d2[1]\n"
                "vmla.f32    q8, q2, d3[0]\n"
                "vmla.f32    q9, q2, d3[1]\n"
                "vld1.32    {d4-d5}, [%[b_ptr] :128]!\n"

                "vmla.f32    q10, q3, d1[0]\n"
                "vmla.f32    q11, q3, d1[1]\n"
                "vld1.32    {d0-d1}, [%[a_ptr] :64]!\n"
                "vmla.f32    q12, q3, d2[0]\n"
                "vmla.f32    q13, q3, d2[1]\n"
                "vmla.f32    q14, q3, d3[0]\n"
                "vmla.f32    q15, q3, d3[1]\n"
                "bne        1b\n"

                // Branch here if we never execute main loop.
                "6:\n"

                // "Tails" shows how many multiply blocks are needed at the
                // end, must be 1-4 inclusive.  Bail out to alternative tail
                // immediately if it's 1.
                "subs        %[tails], %[tails], #1\n"
                "beq        3f\n"

                // Detached final iteration
                // Unroll 0
                "vmla.f32    q4, q2, d0[0]\n"
                "vld1.32    {d2-d3}, [%[a_ptr] :64]!\n"
                "vmla.f32    q5, q2, d0[1]\n"
                "vmla.f32    q6, q2, d1[0]\n"
                "vld1.32    {d6-d7}, [%[b_ptr] :128]!\n"
                "vmla.f32    q7, q2, d1[1]\n"
                "vmla.f32    q8, q2, d2[0]\n"
                "subs        %[tails], %[tails], #1\n"
                "vmla.f32    q9, q2, d2[1]\n"
                "vld1.32    {d4-d5}, [%[b_ptr] :128]!\n"

                "vmla.f32    q10, q3, d0[0]\n"
                "vmla.f32    q11, q3, d0[1]\n"
                "vmla.f32    q12, q3, d1[0]\n"
                "vmla.f32    q13, q3, d1[1]\n"
                "vld1.32    {d0-d1}, [%[a_ptr] :64]!\n"
                "vmla.f32    q14, q3, d2[0]\n"
                "vmla.f32    q15, q3, d2[1]\n"
                "vld1.32    {d6-d7}, [%[b_ptr] :128]!\n"
                "beq        4f\n"

                // Unroll 1
                "vmla.f32    q4, q2, d3[0]\n"
                "vmla.f32    q5, q2, d3[1]\n"
                "subs        %[tails], %[tails], #1\n"
                "vmla.f32    q6, q2, d0[0]\n"
                "vmla.f32    q7, q2, d0[1]\n"
                "vmla.f32    q8, q2, d1[0]\n"
                "vmla.f32    q9, q2, d1[1]\n"
                "vld1.32    {d4-d5}, [%[b_ptr] :128]!\n"

                "vmla.f32    q10, q3, d3[0]\n"
                "vmla.f32    q11, q3, d3[1]\n"
                "vld1.32    {d2-d3}, [%[a_ptr] :64]!\n"
                "vmla.f32    q12, q3, d0[0]\n"
                "vmla.f32    q13, q3, d0[1]\n"
                "vmla.f32    q14, q3, d1[0]\n"
                "vmla.f32    q15, q3, d1[1]\n"
                "vld1.32    {d6-d7}, [%[b_ptr] :128]!\n"
                "beq        5f\n"

                // Unroll 2
                "vld1.32    {d0-d1}, [%[a_ptr] :64]!\n"
                "vmla.f32    q4, q2, d2[0]\n"
                "vmla.f32    q5, q2, d2[1]\n"
                "vmla.f32    q6, q2, d3[0]\n"
                "vmla.f32    q7, q2, d3[1]\n"
                "vmla.f32    q8, q2, d0[0]\n"
                "vmla.f32    q9, q2, d0[1]\n"
                "vld1.32    {d4-d5}, [%[b_ptr] :128]!\n"

                "vmla.f32    q10, q3, d2[0]\n"
                "vmla.f32    q11, q3, d2[1]\n"
                "vmla.f32    q12, q3, d3[0]\n"
                "vmla.f32    q13, q3, d3[1]\n"
                "vld1.32    {d2-d3}, [%[a_ptr] :64]!\n"
                "vmla.f32    q14, q3, d0[0]\n"
                "vmla.f32    q15, q3, d0[1]\n"
                "vld1.32    {d6-d7}, [%[b_ptr] :128]!\n"

                // Unroll 3
                "vmla.f32    q4, q2, d1[0]\n"
                "vmla.f32    q10, q3, d1[0]\n"
                "vst1.32    {d8-d9}, [%[c_ptr] :128]!\n"
                "vmla.f32    q5, q2, d1[1]\n"
                "vst1.32    {d20-d21}, [%[c_ptr] :128]!\n"
                "vmla.f32    q11, q3, d1[1]\n"
                "vst1.32    {d10-d11}, [%[c_ptr] :128]!\n"
                "vmla.f32    q6, q2, d2[0]\n"
                "vst1.32    {d22-d23}, [%[c_ptr] :128]!\n"
                "vmla.f32    q12, q3, d2[0]\n"
                "vst1.32    {d12-d13}, [%[c_ptr] :128]!\n"
                "vmla.f32    q7, q2, d2[1]\n"
                "vst1.32    {d24-d25}, [%[c_ptr] :128]!\n"
                "vmla.f32    q13, q3, d2[1]\n"
                "vst1.32    {d14-d15}, [%[c_ptr] :128]!\n"
                "vmla.f32    q8, q2, d3[0]\n"
                "vst1.32    {d26-d27}, [%[c_ptr] :128]!\n"
                "vmla.f32    q14, q3, d3[0]\n"
                "vst1.32    {d16-d17}, [%[c_ptr] :128]!\n"
                "vmla.f32    q9, q2, d3[1]\n"
                "vst1.32    {d28-d29}, [%[c_ptr] :128]!\n"
                "vmla.f32    q15, q3, d3[1]\n"
                "vst1.32    {d18-d19}, [%[c_ptr] :128]!\n"
                "b        2f\n"

                // tails==1 final tail
                "3:\n"
                "vmla.f32    q4, q2, d0[0]\n"
                "vld1.32    {d2}, [%[a_ptr] :64]!\n"
                "vmla.f32    q5, q2, d0[1]\n"
                "vld1.32    {d6-d7}, [%[b_ptr] :128]!\n"
                "vmla.f32    q6, q2, d1[0]\n"
                "vst1.32    {d8-d9}, [%[c_ptr] :128]!\n"
                "vmla.f32    q10, q3, d0[0]\n"
                "vst1.32    {d20-d21}, [%[c_ptr] :128]!\n"
                "vmla.f32    q11, q3, d0[1]\n"
                "vst1.32    {d10-d11}, [%[c_ptr] :128]!\n"
                "vmla.f32    q12, q3, d1[0]\n"
                "vst1.32    {d22-d23}, [%[c_ptr] :128]!\n"
                "vmla.f32    q7, q2, d1[1]\n"
                "vst1.32    {d12-d13}, [%[c_ptr] :128]!\n"
                "vmla.f32    q13, q3, d1[1]\n"
                "vst1.32    {d24-d25}, [%[c_ptr] :128]!\n"
                "vmla.f32    q8, q2, d2[0]\n"
                "vst1.32    {d14-d15}, [%[c_ptr] :128]!\n"
                "vmla.f32    q14, q3, d2[0]\n"
                "vst1.32    {d26-d27}, [%[c_ptr] :128]!\n"
                "vmla.f32    q9, q2, d2[1]\n"
                "vst1.32    {d16-d17}, [%[c_ptr] :128]!\n"
                "vmla.f32    q15, q3, d2[1]\n"
                "vst1.32    {d28-d29}, [%[c_ptr] :128]!\n"
                "vst1.32    {d18-d19}, [%[c_ptr] :128]!\n"
                "b        2f\n"

                // tails==2 final tail
                "4:\n"
                "vmla.f32    q4, q2, d3[0]\n"
                "vmla.f32    q10, q3, d3[0]\n"
                "vst1.32    {d8-d9}, [%[c_ptr] :128]!\n"
                "vmla.f32    q5, q2, d3[1]\n"
                "vst1.32    {d20-d21}, [%[c_ptr] :128]!\n"
                "vmla.f32    q11, q3, d3[1]\n"
                "vst1.32    {d10-d11}, [%[c_ptr] :128]!\n"
                "vmla.f32    q6, q2, d0[0]\n"
                "vst1.32    {d22-d23}, [%[c_ptr] :128]!\n"
                "vmla.f32    q12, q3, d0[0]\n"
                "vst1.32    {d12-d13}, [%[c_ptr] :128]!\n"
                "vmla.f32    q7, q2, d0[1]\n"
                "vst1.32    {d24-d25}, [%[c_ptr] :128]!\n"
                "vmla.f32    q13, q3, d0[1]\n"
                "vst1.32    {d14-d15}, [%[c_ptr] :128]!\n"
                "vmla.f32    q8, q2, d1[0]\n"
                "vst1.32    {d26-d27}, [%[c_ptr] :128]!\n"
                "vmla.f32    q14, q3, d1[0]\n"
                "vst1.32    {d16-d17}, [%[c_ptr] :128]!\n"
                "vmla.f32    q9, q2, d1[1]\n"
                "vst1.32    {d28-d29}, [%[c_ptr] :128]!\n"
                "vmla.f32    q15, q3, d1[1]\n"
                "vst1.32    {d18-d19}, [%[c_ptr] :128]!\n"
                "b        2f\n"

                // tails==3 final tail
                "5:\n"
                "vmla.f32    q4, q2, d2[0]\n"
                "vld1.32    {d0}, [%[a_ptr] :64]!\n"
                "vmla.f32    q5, q2, d2[1]\n"
                "vmla.f32    q6, q2, d3[0]\n"
                "vst1.32    {d8-d9}, [%[c_ptr] :128]!\n"
                "vmla.f32    q10, q3, d2[0]\n"
                "vst1.32    {d20-d21}, [%[c_ptr] :128]!\n"
                "vmla.f32    q11, q3, d2[1]\n"
                "vst1.32    {d10-d11}, [%[c_ptr] :128]!\n"
                "vmla.f32    q12, q3, d3[0]\n"
                "vst1.32    {d22-d23}, [%[c_ptr] :128]!\n"
                "vmla.f32    q7, q2, d3[1]\n"
                "vst1.32    {d12-d13}, [%[c_ptr] :128]!\n"
                "vmla.f32    q13, q3, d3[1]\n"
                "vst1.32    {d24-d25}, [%[c_ptr] :128]!\n"
                "vmla.f32    q8, q2, d0[0]\n"
                "vst1.32    {d14-d15}, [%[c_ptr] :128]!\n"
                "vmla.f32    q14, q3, d0[0]\n"
                "vst1.32    {d26-d27}, [%[c_ptr] :128]!\n"
                "vmla.f32    q9, q2, d0[1]\n"
                "vst1.32    {d16-d17}, [%[c_ptr] :128]!\n"
                "vmla.f32    q15, q3, d0[1]\n"
                "vst1.32    {d28-d29}, [%[c_ptr] :128]!\n"
                "vst1.32    {d18-d19}, [%[c_ptr] :128]!\n"

                "2:\n"
                "vst1.32    {d30-d31}, [%[c_ptr] :128]!\n"
            : [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [c_ptr] "+r" (c_ptr), [k] "+r" (k), [tails] "+r" (tails)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15",
              "cc", "memory"
            );
        }
    }
}

}  // namespace ops
}  // namespace kai

#endif // __arm__
