//
// SPDX-FileCopyrightText: Copyright 2017-2018, 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef __arm__

#include <arm_neon.h>

#include "asmlib.hpp"

template<>
void interleave_block<6, 1, VLType::None, false>(
    float * &outptr, const float * const * in, size_t width, size_t height,
    size_t row_offset, bool, int32_t
)
{
    const float *inptr0 = in[0] + row_offset;
    const float *inptr1 = in[1] + row_offset;
    const float *inptr2 = in[2] + row_offset;
    const float *inptr3 = in[3] + row_offset;
    const float *inptr4 = in[4] + row_offset;
    const float *inptr5 = in[5] + row_offset;

    // Cope with ragged cases by aliasing the first row (which is always valid).
    // The nonsense output produced will be suppressed later anyway.
    switch (height) {
        case 1:
            inptr1 = inptr0;
            // fall through
        case 2:
            inptr2 = inptr0;
            // fall through
        case 3:
            inptr3 = inptr0;
            // fall through
        case 4:
            inptr4 = inptr0;
            // fall through
        case 5:
            inptr5 = inptr0;
            // fall through
        default:
        case 6:
            break;
    }

    //prefetch_2x(inptr0);
    //prefetch_2x(inptr1);
    //prefetch_2x(inptr2);
    //prefetch_2x(inptr3);
    //prefetch_2x(inptr4);
    //prefetch_2x(inptr5);

    for (;width>7;width-=8) {
        __asm __volatile (
            // Load up 8 elements (2 vectors) from each of 8 sources.
            "VLD1.32    {d0-d3}, [%[inptr0]]!\n"   // q0=A0A1A2A3
            "VLD1.32    {d4-d7}, [%[inptr1]]!\n"   // q2=B0B1B2B3
            "VLD1.32    {d8-d11}, [%[inptr2]]!\n"  // q4=C0C1C2C3
            "VZIP.32    q0, q4\n"     // q0=A0C0A1C1, q4 = A2C2A3C3
            "VLD1.32    {d12-d15}, [%[inptr3]]!\n" // q6=D0D1D2D3
            "VZIP.32    q2, q6\n"     // q2=B0D0B1D1, q6 = B2D2B3D3
            "VLD1.32    {d16-d19}, [%[inptr4]]!\n"
            "VLD1.32    {d20-d23}, [%[inptr5]]!\n"
            "VZIP.32    q8, q10\n"    // q8=E0F0E1F1, q10 = E2F2E3F3
            ASM_PREFETCH("[%[inptr0], #128]")
            "VZIP.32    q0, q2\n"    // q0 = A0B0C0D0, q2 = A1B1C1D1

            // Store first elements
            "VST1.32    {d0-d1}, [%[outptr]]!\n"
            "VST1.32    {d16}, [%[outptr]]!\n"

            "VZIP.32    q4, q6\n"    // q4 = A2B2C2D2, q6 = A3B3C3D3

            // Store second elements
            "VST1.32    {d4-d5}, [%[outptr]]!\n"
            "VZIP.32    q1, q5\n"
            ASM_PREFETCH("[%[inptr1], #128]")
            "VST1.32    {d17}, [%[outptr]]!\n"
            "VZIP.32    q3, q7\n"

            // Store third elements
            "VZIP.32    q9, q11\n"
            "VST1.32    {d8-d9}, [%[outptr]]!\n"
            "VZIP.32    q1, q3\n"
            ASM_PREFETCH("[%[inptr2], #128]")
            "VST1.32    {d20}, [%[outptr]]!\n"

            // Store fourth elements
            "VZIP.32    q5, q7\n"
            "VST1.32    {d12-d13}, [%[outptr]]!\n"
            ASM_PREFETCH("[%[inptr3], #128]")
            "VST1.32    {d21}, [%[outptr]]!\n"

            // Fifth
            "VST1.32    {d2-d3}, [%[outptr]]!\n"
            ASM_PREFETCH("[%[inptr4], #128]")
            "VST1.32    {d18}, [%[outptr]]!\n"

            // Sixth
            "VST1.32    {d6-d7}, [%[outptr]]!\n"
            ASM_PREFETCH("[%[inptr5], #128]")
            "VST1.32    {d19}, [%[outptr]]!\n"

            // Seventh
            "VST1.32    {d10-d11}, [%[outptr]]!\n"
            "VST1.32    {d22}, [%[outptr]]!\n"

            // Eighth
            "VST1.32    {d14-d15}, [%[outptr]]!\n"
            "VST1.32    {d23}, [%[outptr]]!\n"

            : [inptr0] "+r" (inptr0), [inptr1] "+r" (inptr1), [inptr2] "+r" (inptr2), [inptr3] "+r" (inptr3),
              [inptr4] "+r" (inptr4), [inptr5] "+r" (inptr5), [outptr] "+r" (outptr)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "memory"
        );
    }

    for (;width>0;width--) {
        *outptr++ = *inptr0++;
        *outptr++ = *inptr1++;
        *outptr++ = *inptr2++;
        *outptr++ = *inptr3++;
        *outptr++ = *inptr4++;
        *outptr++ = *inptr5++;
    }
}

#endif // __arm__
