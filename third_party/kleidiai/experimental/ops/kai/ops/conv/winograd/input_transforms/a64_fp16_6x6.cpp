//
// SPDX-FileCopyrightText: Copyright 2022, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

#include <arm_neon.h>
#include <cstddef>

namespace kai {
namespace ops {
namespace winograd {
namespace input_transform {

void a64_fp16_6x6(
    const unsigned int n_channels,
    const __fp16* const input_base,
    const size_t input_row_stride,
    const size_t input_col_stride,
    __fp16* outptr,
    const size_t matrix_stride
)
{
    constexpr int inner_tile_rows = 6;
    constexpr int inner_tile_cols = 6;

    // Get pointers into the input tile
    const __fp16 *x_ptrs[inner_tile_rows][inner_tile_cols];
    for (int i = 0, xi = 0; i < inner_tile_rows; i++, xi++)
    {
        // Get a pointer into the row
        const __fp16* const row_ptr = input_base + xi*input_row_stride;

        for (int j = 0, xj = 0; j < inner_tile_cols; j++, xj++)
        {
            x_ptrs[i][j] = row_ptr + xj*input_col_stride;
        }
    }

    // Matrices used/computed in this kernel.
    __fp16 x[inner_tile_rows][inner_tile_cols];
    __fp16 XTx[inner_tile_rows][inner_tile_cols];
    __fp16 U[inner_tile_rows][inner_tile_cols];
    for (int i = 0; i < inner_tile_rows; i++)
    {
        for (int j = 0; j < inner_tile_cols; j++)
        {
            x[i][j] = XTx[i][j] = 0.0f;
        }
    }

    // Perform the Winograd input transformation for each channel in the input
    // tensor.
    int channels_remaining = n_channels;
    for (; channels_remaining >= 8; channels_remaining -= 8)
    {
        // Matrices used/computed in this kernel
        float16x8_t x[inner_tile_rows][inner_tile_cols];
        float16x8_t XTx[inner_tile_rows][inner_tile_cols];
        float16x8_t U[inner_tile_rows][inner_tile_cols];
        for (int i = 0; i < inner_tile_rows; i++)
        {
            for (int j = 0; j < inner_tile_cols; j++)
            {
                x[i][j] = vdupq_n_f16(0.0f);
                XTx[i][j] = vdupq_n_f16(0.0f);
            }
        }

        // Read a 6x6 tile in the Winograd domain
        for (int i = 0; i < inner_tile_rows; i++)
        {
            for (int j = 0; j < inner_tile_cols; j++)
            {
                x[i][j] = vld1q_f16(x_ptrs[i][j]);
                x_ptrs[i][j] += 8;
            }
        }

        // Compute XT . x
        const auto _1over4q = vdupq_n_f16(1.0f/4.0f);
        const auto _1over8q = vdupq_n_f16(1.0f/8.0f);
        const auto _1over16q = vdupq_n_f16(1.0f/16.0f);
        const auto _3over16q = vdupq_n_f16(3.0f/16.0f);
        const auto _5over16q = vdupq_n_f16(5.0f/16.0f);

        for (int j = 0; j < inner_tile_cols; j++)
        {
            // XTx[0][j] = (1/8)*(x[0][j] + x[4][j]) + (3/16)*(x[1][j] - x[3][j]) - (1/4)*x[2][j]
            auto tmp1 = vmulq_f16(vaddq_f16(x[0][j], x[4][j]), _1over8q);
            auto tmp2 = vmulq_f16(vsubq_f16(x[1][j], x[3][j]), _3over16q);
            XTx[0][j] = vsubq_f16(vaddq_f16(tmp1, tmp2), vmulq_f16(x[2][j], _1over4q));

            // XTx[1][j] = (1/8)*(x[1][j] + x[4][j]) + (1/16)*x[2][j] - (5/16)*x[3][j]
            tmp1 = vmulq_f16(vaddq_f16(x[1][j], x[4][j]), _1over8q);
            tmp2 = vsubq_f16( vmulq_f16(x[2][j], _1over16q), vmulq_f16(x[3][j], _5over16q) );
            XTx[1][j] = vaddq_f16(tmp1, tmp2);

            // XTx[2][j] = (1/8)*(x[4][j] - x[1][j]) - (5/16)*x[2][j] - (1/16)*x[3][j]
            tmp1 = vmulq_f16(vsubq_f16(x[4][j], x[1][j]), _1over8q);
            tmp2 = vaddq_f16(vmulq_f16(x[2][j], _5over16q), vmulq_f16(x[3][j], _1over16q));
            XTx[2][j] = vsubq_f16(tmp1, tmp2);

            // XTx[3][j] = (1/4)*(x[1][j] - x[3][j]) + (1/8)*(x[4][j] - x[2][j])
            tmp1 = vmulq_f16(vsubq_f16(x[1][j], x[3][j]), _1over4q);
            tmp2 = vmulq_f16(vsubq_f16(x[4][j], x[2][j]), _1over8q);
            XTx[3][j] = vaddq_f16(tmp1, tmp2);

            // XTx[4][j] = (1/8)*(x[3][j] - x[1][j]) + (1/4)*(x[4][j] - x[2][j])
            tmp1 = vmulq_f16(vsubq_f16(x[3][j], x[1][j]), _1over8q);
            tmp2 = vmulq_f16(vsubq_f16(x[4][j], x[2][j]), _1over4q);
            XTx[4][j] = vaddq_f16(tmp1, tmp2);

            // XTx[5][j] = (1/8)*(x[1][j] + x[5][j]) + (3/16)*(x[2][j] - x[4][j]) - (1/4)*x[3][j]
            tmp1 = vmulq_f16(vaddq_f16(x[1][j], x[5][j]), _1over8q);
            tmp2 = vmulq_f16(vsubq_f16(x[2][j], x[4][j]), _3over16q);
            XTx[5][j] = vsubq_f16(vaddq_f16(tmp1, tmp2), vmulq_f16(x[3][j], _1over4q));
        }

        // Compute U = XT . x . X
        for (int i = 0; i < inner_tile_rows; i++)
        {
            // U[i][0] = (1/8)*(XTx[i][0] + XTx[i][4]) + (3/16)*(XTx[i][1] - XTx[i][3]) - (1/4)*XTx[i][2]
            auto tmp1 = vmulq_f16(vaddq_f16(XTx[i][0], XTx[i][4]), _1over8q);
            auto tmp2 = vmulq_f16(vsubq_f16(XTx[i][1], XTx[i][3]), _3over16q);
            U[i][0] = vsubq_f16(vaddq_f16(tmp1, tmp2), vmulq_f16(XTx[i][2], _1over4q));

            // U[i][1] = (1/8)*(XTx[i][1] + XTx[i][4]) + (1/16)*XTx[i][2] - (5/16)*XTx[i][3]
            tmp1 = vmulq_f16(vaddq_f16(XTx[i][1], XTx[i][4]), _1over8q);
            tmp2 = vsubq_f16(vmulq_f16(XTx[i][2], _1over16q), vmulq_f16(XTx[i][3], _5over16q));
            U[i][1] = vaddq_f16(tmp1, tmp2);

            // U[i][2] = (1/8)*(XTx[i][4] - XTx[i][1]) - (5/16)*XTx[i][2] - (1/16)*XTx[i][3]
            tmp1 = vmulq_f16(vsubq_f16(XTx[i][4], XTx[i][1]), _1over8q);
            tmp2 = vaddq_f16(vmulq_f16(XTx[i][2], _5over16q), vmulq_f16(XTx[i][3], _1over16q));
            U[i][2] = vsubq_f16(tmp1, tmp2);

            // U[i][3] = (1/4)*(XTx[i][1] - XTx[i][3]) + (1/8)*(XTx[i][4] - XTx[i][2])
            tmp1 = vmulq_f16(vsubq_f16(XTx[i][1], XTx[i][3]), _1over4q);
            tmp2 = vmulq_f16(vsubq_f16(XTx[i][4], XTx[i][2]), _1over8q);
            U[i][3] = vaddq_f16(tmp1, tmp2);

            // U[i][4] = (1/8)*(XTx[i][3] - XTx[i][1]) + (1/4)*(XTx[i][4] - XTx[i][2])
            tmp1 = vmulq_f16(vsubq_f16(XTx[i][3], XTx[i][1]), _1over8q);
            tmp2 = vmulq_f16(vsubq_f16(XTx[i][4], XTx[i][2]), _1over4q);
            U[i][4] = vaddq_f16(tmp1, tmp2);

            // U[i][5] = (1/8)*(XTx[i][1] + XTx[i][5]) + (3/16)*(XTx[i][2] - XTx[i][4]) - (1/4)*XTx[i][3]
            tmp1 = vmulq_f16(vaddq_f16(XTx[i][1], XTx[i][5]), _1over8q);
            tmp2 = vmulq_f16(vsubq_f16(XTx[i][2], XTx[i][4]), _3over16q);
            U[i][5] = vsubq_f16(vaddq_f16(tmp1, tmp2), vmulq_f16(XTx[i][3], _1over4q));
        }

        // Store the transformed matrix
        for (int i = 0, m = 0; i < inner_tile_rows; i++)
        {
            for (int j = 0; j < inner_tile_cols; j++, m++)
            {
                vst1q_f16(outptr + m*matrix_stride, U[i][j]);
            }
        }
        outptr += 8;
    }
    for (; channels_remaining >= 4; channels_remaining -= 4)
    {
        // Matrices used/computed in this kernel
        float16x4_t x[inner_tile_rows][inner_tile_cols];
        float16x4_t XTx[inner_tile_rows][inner_tile_cols];
        float16x4_t U[inner_tile_rows][inner_tile_cols];
        for (int i = 0; i < inner_tile_rows; i++)
        {
            for (int j = 0; j < inner_tile_cols; j++)
            {
                x[i][j] = vdup_n_f16(0.0f);
                XTx[i][j] = vdup_n_f16(0.0f);
            }
        }

        // Read a 6x6 tile in the Winograd domain
        for (int i = 0; i < inner_tile_rows; i++)
        {
            for (int j = 0; j < inner_tile_cols; j++)
            {
                x[i][j] = vld1_f16(x_ptrs[i][j]);
                x_ptrs[i][j] += 4;
            }
        }

        // Compute XT . x
        const auto _1over4 = vdup_n_f16(1.0f/4.0f);
        const auto _1over8 = vdup_n_f16(1.0f/8.0f);
        const auto _1over16 = vdup_n_f16(1.0f/16.0f);
        const auto _3over16 = vdup_n_f16(3.0f/16.0f);
        const auto _5over16 = vdup_n_f16(5.0f/16.0f);

        for (int j = 0; j < inner_tile_cols; j++)
        {
            // XTx[0][j] = (1/8)*(x[0][j] + x[4][j]) + (3/16)*(x[1][j] - x[3][j]) - (1/4)*x[2][j]
            auto tmp1 = vmul_f16(vadd_f16(x[0][j], x[4][j]), _1over8);
            auto tmp2 = vmul_f16(vsub_f16(x[1][j], x[3][j]), _3over16);
            XTx[0][j] = vsub_f16(vadd_f16(tmp1, tmp2), vmul_f16(x[2][j], _1over4));

            // XTx[1][j] = (1/8)*(x[1][j] + x[4][j]) + (1/16)*x[2][j] - (5/16)*x[3][j]
            tmp1 = vmul_f16(vadd_f16(x[1][j], x[4][j]), _1over8);
            tmp2 = vsub_f16(vmul_f16(x[2][j], _1over16), vmul_f16(x[3][j], _5over16));
            XTx[1][j] = vadd_f16(tmp1, tmp2);

            // XTx[2][j] = (1/8)*(x[4][j] - x[1][j]) - (5/16)*x[2][j] - (1/16)*x[3][j]
            tmp1 = vmul_f16(vsub_f16(x[4][j], x[1][j]), _1over8);
            tmp2 = vadd_f16(vmul_f16(x[2][j], _5over16), vmul_f16(x[3][j], _1over16));
            XTx[2][j] = vsub_f16(tmp1, tmp2);

            // XTx[3][j] = (1/4)*(x[1][j] - x[3][j]) + (1/8)*(x[4][j] - x[2][j])
            tmp1 = vmul_f16(vsub_f16(x[1][j], x[3][j]), _1over4);
            tmp2 = vmul_f16(vsub_f16(x[4][j], x[2][j]), _1over8);
            XTx[3][j] = vadd_f16(tmp1, tmp2);

            // XTx[4][j] = (1/8)*(x[3][j] - x[1][j]) + (1/4)*(x[4][j] - x[2][j])
            tmp1 = vmul_f16(vsub_f16(x[3][j], x[1][j]), _1over8);
            tmp2 = vmul_f16(vsub_f16(x[4][j], x[2][j]), _1over4);
            XTx[4][j] = vadd_f16(tmp1, tmp2);

            // XTx[5][j] = (1/8)*(x[1][j] + x[5][j]) + (3/16)*(x[2][j] - x[4][j]) - (1/4)*x[3][j]
            tmp1 = vmul_f16(vadd_f16(x[1][j], x[5][j]), _1over8);
            tmp2 = vmul_f16(vsub_f16(x[2][j], x[4][j]), _3over16);
            XTx[5][j] = vsub_f16(vadd_f16(tmp1, tmp2), vmul_f16(x[3][j], _1over4));
        }

        // Compute U = XT . x . X
        for (int i = 0; i < inner_tile_rows; i++)
        {
            // U[i][0] = (1/8)*(XTx[i][0] + XTx[i][4]) + (3/16)*(XTx[i][1] - XTx[i][3]) - (1/4)*XTx[i][2]
            auto tmp1 = vmul_f16(vadd_f16(XTx[i][0], XTx[i][4]), _1over8);
            auto tmp2 = vmul_f16(vsub_f16(XTx[i][1], XTx[i][3]), _3over16);
            U[i][0] = vsub_f16(vadd_f16(tmp1, tmp2), vmul_f16(XTx[i][2], _1over4));

            // U[i][1] = (1/8)*(XTx[i][1] + XTx[i][4]) + (1/16)*XTx[i][2] - (5/16)*XTx[i][3]
            tmp1 = vmul_f16(vadd_f16(XTx[i][1], XTx[i][4]), _1over8);
            tmp2 = vsub_f16(vmul_f16(XTx[i][2], _1over16), vmul_f16(XTx[i][3], _5over16));
            U[i][1] = vadd_f16(tmp1, tmp2);

            // U[i][2] = (1/8)*(XTx[i][4] - XTx[i][1]) - (5/16)*XTx[i][2] - (1/16)*XTx[i][3]
            tmp1 = vmul_f16(vsub_f16(XTx[i][4], XTx[i][1]), _1over8);
            tmp2 = vadd_f16(vmul_f16(XTx[i][2], _5over16), vmul_f16(XTx[i][3], _1over16));
            U[i][2] = vsub_f16(tmp1, tmp2);

            // U[i][3] = (1/4)*(XTx[i][1] - XTx[i][3]) + (1/8)*(XTx[i][4] - XTx[i][2])
            tmp1 = vmul_f16(vsub_f16(XTx[i][1], XTx[i][3]), _1over4);
            tmp2 = vmul_f16(vsub_f16(XTx[i][4], XTx[i][2]), _1over8);
            U[i][3] = vadd_f16(tmp1, tmp2);

            // U[i][4] = (1/8)*(XTx[i][3] - XTx[i][1]) + (1/4)*(XTx[i][4] - XTx[i][2])
            tmp1 = vmul_f16(vsub_f16(XTx[i][3], XTx[i][1]), _1over8);
            tmp2 = vmul_f16(vsub_f16(XTx[i][4], XTx[i][2]), _1over4);
            U[i][4] = vadd_f16(tmp1, tmp2);

            // U[i][5] = (1/8)*(XTx[i][1] + XTx[i][5]) + (3/16)*(XTx[i][2] - XTx[i][4]) - (1/4)*XTx[i][3]
            tmp1 = vmul_f16(vadd_f16(XTx[i][1], XTx[i][5]), _1over8);
            tmp2 = vmul_f16(vsub_f16(XTx[i][2], XTx[i][4]), _3over16);
            U[i][5] = vsub_f16(vadd_f16(tmp1, tmp2), vmul_f16(XTx[i][3], _1over4));
        }

        // Store the transformed matrix
        for (int i = 0, m = 0; i < inner_tile_rows; i++)
        {
            for (int j = 0; j < inner_tile_cols; j++, m++)
            {
                vst1_f16(outptr + m*matrix_stride, U[i][j]);
            }
        }
        outptr += 4;
    }
    for (; channels_remaining; channels_remaining--)
    {
        // Load x
        for (int i = 0; i < inner_tile_rows; i++)
        {
            for (int j = 0; j < inner_tile_cols; j++)
            {
                x[i][j] = *(x_ptrs[i][j]++);
            }
        }

        // Compute XT . x
        for (int j = 0; j < inner_tile_cols; j++)
        {
            XTx[0][j] = ((x[0][j] + x[4][j]) * (1.0f / 8.0f) + (x[1][j] - x[3][j]) * (3.0f / 16.0f)) - x[2][j] * (1.0f / 4.0f);
            XTx[1][j] = (x[1][j] + x[4][j]) * (1.0f / 8.0f) + (x[2][j] * (1.0f / 16.0f) - x[3][j] * (5.0f / 16.0f));
            XTx[2][j] = (x[4][j] - x[1][j]) * (1.0f / 8.0f) - (x[2][j] * (5.0f / 16.0f) + x[3][j] * (1.0f / 16.0f));
            XTx[3][j] = (x[1][j] - x[3][j]) * (1.0f / 4.0f) + (x[4][j] - x[2][j]) * (1.0f / 8.0f);
            XTx[4][j] = (x[3][j] - x[1][j]) * (1.0f / 8.0f) + (x[4][j] - x[2][j]) * (1.0f / 4.0f);
            XTx[5][j] = ((x[1][j] + x[5][j]) * (1.0f / 8.0f) + (x[2][j] - x[4][j]) * (3.0f / 16.0f)) - x[3][j] * (1.0f / 4.0f);
        }

        // Compute U = XT . x . X
        for (int i = 0; i < inner_tile_rows; i++)
        {
            U[i][0] = ((XTx[i][0] + XTx[i][4]) * (1.0f / 8.0f) + (XTx[i][1] - XTx[i][3]) * (3.0f / 16.0f)) - XTx[i][2] * (1.0f / 4.0f);
            U[i][1] = (XTx[i][1] + XTx[i][4]) * (1.0f / 8.0f) + (XTx[i][2] * (1.0f / 16.0f) - XTx[i][3] * (5.0f / 16.0f));
            U[i][2] = (XTx[i][4] - XTx[i][1]) * (1.0f / 8.0f) - (XTx[i][2] * (5.0f / 16.0f) + XTx[i][3] * (1.0f / 16.0f));
            U[i][3] = (XTx[i][1] - XTx[i][3]) * (1.0f / 4.0f) + (XTx[i][4] - XTx[i][2]) * (1.0f / 8.0f);
            U[i][4] = (XTx[i][3] - XTx[i][1]) * (1.0f / 8.0f) + (XTx[i][4] - XTx[i][2]) * (1.0f / 4.0f);
            U[i][5] = ((XTx[i][1] + XTx[i][5]) * (1.0f / 8.0f) + (XTx[i][2] - XTx[i][4]) * (3.0f / 16.0f)) - XTx[i][3] * (1.0f / 4.0f);
        }

        // Store the transformed matrix
        for (int i = 0, m = 0; i < inner_tile_rows; i++)
        {
            for (int j = 0; j < inner_tile_cols; j++, m++)
            {
                *(outptr + m*matrix_stride) = U[i][j];
            }
        }
        outptr++;
    }
}

}  // namespace input_transform
}  // namespace winograd
}  // namespace ops
}  // namespace kai

#endif // defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
