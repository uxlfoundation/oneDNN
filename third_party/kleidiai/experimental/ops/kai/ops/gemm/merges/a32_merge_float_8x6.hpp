//
// SPDX-FileCopyrightText: Copyright 2017-2018, 2023, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#ifdef __arm__

#include <arm_neon.h>

template<>
void MergeResults<8, 6, false>(float *out, const float *in, const int ldout, const int y0, const int ymax, const int x0, const int xmax, const float *bias, Activation act, bool append) {
    const float *inptr = in;
    prefetch_6x(inptr);
    prefetch_6x(inptr + 96);

    float nullbias[8];
    float minval = - std::numeric_limits<float>::infinity();
    float maxval =   std::numeric_limits<float>::infinity();

    switch(act.type)
    {
        default:
        case Activation::Type::None:
            break;
        case Activation::Type::BoundedReLU:
            maxval = static_cast<float>(act.param1);
            /* fall through */
        case Activation::Type::ReLU:
            minval = 0.0f;
            break;
    }

    float32x4_t minv = vdupq_n_f32(minval);
    float32x4_t maxv = vdupq_n_f32(maxval);

    if (!append && !bias)
    {
        memset(nullbias, 0, (8 * sizeof(float)));
    }

    for (int y=y0; y<ymax; y+=8) {
        float *outptr0 = out + (y * ldout) + x0;
        float *outptr1 = outptr0 + ldout;
        float *outptr2 = outptr1 + ldout;
        float *outptr3 = outptr2 + ldout;
        float *outptr4 = outptr3 + ldout;
        float *outptr5 = outptr4 + ldout;

        prefetch_2x(outptr0);
        prefetch_2x(outptr1);
        prefetch_2x(outptr2);
        prefetch_2x(outptr3);
        prefetch_2x(outptr4);
        prefetch_2x(outptr5);

        for (int i=x0; i<xmax; i+=8) {
            float dummyres[8];

            /* Make sure we throw away results if Y isn't a multiple of 8.
             * We do this by pointing the result pointer at a dummy buffer
             * we later discard.  */
            if ((y+5) >= ymax) {
                switch ((y + 5) - ymax) {
                    case 4:
                        outptr1 = dummyres;
                        /* fall through */
                    case 3:
                        outptr2 = dummyres;
                        /* fall through */
                    case 2:
                        outptr3 = dummyres;
                        /* fall through */
                    case 1:
                        outptr4 = dummyres;
                        /* fall through */
                    case 0:
                        outptr5 = dummyres;
                        break;

                    default:
                        UNREACHABLE("Impossible.");
                }
            }

            if (append) {
               /* Append mode: Read, activate, write. */

                /* For ragged X, manually copy over the valid results. */
                if ((i+7) >= xmax) {
                    for (int xi=0; xi<8; xi++) {
                        if ((i+xi) < xmax) {
                            *outptr0 = std::min(std::max(minval, inptr[xi] + *outptr0), maxval);
                            outptr0++;
                            *outptr1 = std::min(std::max(minval, inptr[xi + 8] + *outptr1), maxval);
                            outptr1++;
                            *outptr2 = std::min(std::max(minval, inptr[xi + 16] + *outptr2), maxval);
                            outptr2++;
                            *outptr3 = std::min(std::max(minval, inptr[xi + 24] + *outptr3), maxval);
                            outptr3++;
                            *outptr4 = std::min(std::max(minval, inptr[xi + 32] + *outptr4), maxval);
                            outptr4++;
                            *outptr5 = std::min(std::max(minval, inptr[xi + 40] + *outptr5), maxval);
                            outptr5++;
                        }
                    }
                    inptr += 48;
                } else {
                    /* Optimized routine to copy an entire block */
                    __asm __volatile (
                        // Rows 0-1
                        "VLD1.32	{d0-d3},   [%[inptr]]!\n"
                        "VLD1.32	{d8-d11},  [%[outptr0]]\n"
                        "VLD1.32	{d4-d7},   [%[inptr]]!\n"
                        "VLD1.32	{d12-d15}, [%[outptr1]]\n"

                        "VADD.f32	q4, q4, q0\n"
                        ASM_PREFETCH("[%[inptr], #352]")
                        "VADD.f32	q5, q5, q1\n"
                        "VADD.f32	q6, q6, q2\n"
                        "VADD.f32	q7, q7, q3\n"
                        ASM_PREFETCH("[%[inptr], #416]")
                        "VMAX.f32	q4, q4, %q[minv]\n"
                        "VMAX.f32	q5, q5, %q[minv]\n"
                        "VMAX.f32	q6, q6, %q[minv]\n"
                        ASM_PREFETCH("[%[inptr], #480]")
                        "VMAX.f32	q7, q7, %q[minv]\n"
                        "VMIN.f32	q4, q4, %q[maxv]\n"
                        "VMIN.f32	q5, q5, %q[maxv]\n"
                        "VST1.32	{d8-d11}, [%[outptr0]]!\n"
                        "VMIN.f32	q6, q6, %q[maxv]\n"
                        "VMIN.f32	q7, q7, %q[maxv]\n"
                        "VST1.32	{d12-d15}, [%[outptr1]]!\n"

                        // Rows 2-3
                        "VLD1.32	{d0-d3},   [%[inptr]]!\n"
                        "VLD1.32	{d8-d11},  [%[outptr2]]\n"
                        "VLD1.32	{d4-d7},   [%[inptr]]!\n"
                        "VLD1.32	{d12-d15}, [%[outptr3]]\n"

                        "VADD.f32	q4, q4, q0\n"
                        ASM_PREFETCH("[%[outptr0], #96]")
                        "VADD.f32	q5, q5, q1\n"
                        "VADD.f32	q6, q6, q2\n"
                        "VADD.f32	q7, q7, q3\n"
                        ASM_PREFETCH("[%[outptr1], #96]")
                        "VMAX.f32	q4, q4, %q[minv]\n"
                        "VMAX.f32	q5, q5, %q[minv]\n"
                        "VMAX.f32	q6, q6, %q[minv]\n"
                        ASM_PREFETCH("[%[outptr2], #128]")
                        "VMAX.f32	q7, q7, %q[minv]\n"
                        "VMIN.f32	q4, q4, %q[maxv]\n"
                        "VMIN.f32	q5, q5, %q[maxv]\n"
                        "VST1.32	{d8-d11}, [%[outptr2]]!\n"
                        "VMIN.f32	q6, q6, %q[maxv]\n"
                        "VMIN.f32	q7, q7, %q[maxv]\n"
                        "VST1.32	{d12-d15}, [%[outptr3]]!\n"

                        // Rows 4-5
                        "VLD1.32	{d0-d3},   [%[inptr]]!\n"
                        "VLD1.32	{d8-d11},  [%[outptr4]]\n"
                        "VLD1.32	{d4-d7},   [%[inptr]]!\n"
                        "VLD1.32	{d12-d15}, [%[outptr5]]\n"

                        "VADD.f32	q4, q4, q0\n"
                        ASM_PREFETCH("[%[outptr3], #96]")
                        "VADD.f32	q5, q5, q1\n"
                        "VADD.f32	q6, q6, q2\n"
                        "VADD.f32	q7, q7, q3\n"
                        ASM_PREFETCH("[%[outptr4], #128]")
                        "VMAX.f32	q4, q4, %q[minv]\n"
                        "VMAX.f32	q5, q5, %q[minv]\n"
                        "VMAX.f32	q6, q6, %q[minv]\n"
                        ASM_PREFETCH("[%[outptr5], #128]")
                        "VMAX.f32	q7, q7, %q[minv]\n"
                        "VMIN.f32	q4, q4, %q[maxv]\n"
                        "VMIN.f32	q5, q5, %q[maxv]\n"
                        "VST1.32	{d8-d11}, [%[outptr4]]!\n"
                        "VMIN.f32	q6, q6, %q[maxv]\n"
                        "VMIN.f32	q7, q7, %q[maxv]\n"
                        "VST1.32	{d12-d15}, [%[outptr5]]!\n"
                    : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3),
                      [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [inptr] "+r" (inptr)
                    : [minv] "w" (minv), [maxv] "w" (maxv)
                    : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "memory"
                    );
                }
            } else {
                /* Bias mode: Add bias to everything, then min/max/write as before. */
                const float *biasptr = bias ? bias + i : nullbias;

                /* For ragged X, manually copy over the valid results. */
                if ((i+7) >= xmax) {
                    for (int xi=0; xi<7; xi++) {
                        if ((i+xi) < xmax) {
                            *outptr0 = std::min(std::max(minval, inptr[xi] + biasptr[xi]), maxval);
                            outptr0++;
                            *outptr1 = std::min(std::max(minval, inptr[xi + 8] + biasptr[xi]), maxval);
                            outptr1++;
                            *outptr2 = std::min(std::max(minval, inptr[xi + 16] + biasptr[xi]), maxval);
                            outptr2++;
                            *outptr3 = std::min(std::max(minval, inptr[xi + 24] + biasptr[xi]), maxval);
                            outptr3++;
                            *outptr4 = std::min(std::max(minval, inptr[xi + 32] + biasptr[xi]), maxval);
                            outptr4++;
                            *outptr5 = std::min(std::max(minval, inptr[xi + 40] + biasptr[xi]), maxval);
                            outptr5++;
                        }
                    }
                    inptr += 48;
                } else {
                    /* Optimized routine to copy an entire block */
                    __asm __volatile (
                        // Rows 0-1
                        "VLD1.32	{d8-d11},   [%[inptr]]!\n"
                        "VLD1.32	{d0-d3},   [%[biasptr]]\n"
                        "VLD1.32	{d12-d15},  [%[inptr]]!\n"

                        "VADD.f32	q4, q4, q0\n"
                        ASM_PREFETCH("[%[inptr], #352]")
                        "VADD.f32	q5, q5, q1\n"
                        "VADD.f32	q6, q6, q0\n"
                        "VADD.f32	q7, q7, q1\n"
                        ASM_PREFETCH("[%[inptr], #416]")
                        "VMAX.f32	q4, q4, %q[minv]\n"
                        "VMAX.f32	q5, q5, %q[minv]\n"
                        "VMAX.f32	q6, q6, %q[minv]\n"
                        ASM_PREFETCH("[%[inptr], #480]")
                        "VMAX.f32	q7, q7, %q[minv]\n"
                        "VMIN.f32	q4, q4, %q[maxv]\n"
                        "VMIN.f32	q5, q5, %q[maxv]\n"
                        "VST1.32	{d8-d11}, [%[outptr0]]!\n"
                        "VMIN.f32	q6, q6, %q[maxv]\n"
                        "VMIN.f32	q7, q7, %q[maxv]\n"
                        "VST1.32	{d12-d15}, [%[outptr1]]!\n"

                        // Rows 2-3
                        "VLD1.32	{d8-d11},   [%[inptr]]!\n"
                        "VLD1.32	{d12-d15},  [%[inptr]]!\n"

                        "VADD.f32	q4, q4, q0\n"
                        ASM_PREFETCH("[%[outptr0], #96]")
                        "VADD.f32	q5, q5, q1\n"
                        "VADD.f32	q6, q6, q0\n"
                        "VADD.f32	q7, q7, q1\n"
                        ASM_PREFETCH("[%[outptr1], #96]")
                        "VMAX.f32	q4, q4, %q[minv]\n"
                        "VMAX.f32	q5, q5, %q[minv]\n"
                        "VMAX.f32	q6, q6, %q[minv]\n"
                        ASM_PREFETCH("[%[outptr2], #128]")
                        "VMAX.f32	q7, q7, %q[minv]\n"
                        "VMIN.f32	q4, q4, %q[maxv]\n"
                        "VMIN.f32	q5, q5, %q[maxv]\n"
                        "VST1.32	{d8-d11}, [%[outptr2]]!\n"
                        "VMIN.f32	q6, q6, %q[maxv]\n"
                        "VMIN.f32	q7, q7, %q[maxv]\n"
                        "VST1.32	{d12-d15}, [%[outptr3]]!\n"
                    : [outptr0] "+r" (outptr0), [outptr1] "+r" (outptr1), [outptr2] "+r" (outptr2), [outptr3] "+r" (outptr3),
                      [inptr] "+r" (inptr)
                    : [minv] "w" (minv), [maxv] "w" (maxv), [biasptr] "r" (biasptr)
                    : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "memory"
                    );

                    __asm __volatile (
                        // Rows 4-5
                        "VLD1.32	{d8-d11},   [%[inptr]]!\n"
                        "VLD1.32	{d12-d15},  [%[inptr]]!\n"

                        "VADD.f32	q4, q4, q0\n"
                        ASM_PREFETCH("[%[outptr3], #96]")
                        "VADD.f32	q5, q5, q1\n"
                        "VADD.f32	q6, q6, q0\n"
                        "VADD.f32	q7, q7, q1\n"
                        ASM_PREFETCH("[%[outptr4], #128]")
                        "VMAX.f32	q4, q4, %q[minv]\n"
                        "VMAX.f32	q5, q5, %q[minv]\n"
                        "VMAX.f32	q6, q6, %q[minv]\n"
                        ASM_PREFETCH("[%[outptr5], #128]")
                        "VMAX.f32	q7, q7, %q[minv]\n"
                        "VMIN.f32	q4, q4, %q[maxv]\n"
                        "VMIN.f32	q5, q5, %q[maxv]\n"
                        "VST1.32	{d8-d11}, [%[outptr4]]!\n"
                        "VMIN.f32	q6, q6, %q[maxv]\n"
                        "VMIN.f32	q7, q7, %q[maxv]\n"
                        "VST1.32	{d12-d15}, [%[outptr5]]!\n"
                    : [outptr3] "+r" (outptr3),
                      [outptr4] "+r" (outptr4), [outptr5] "+r" (outptr5), [inptr] "+r" (inptr)
                    : [minv] "w" (minv), [maxv] "w" (maxv), [biasptr] "r" (biasptr)
                    : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "memory"
                    );
                }
            }
        }
    }
}

#endif // __arm__
