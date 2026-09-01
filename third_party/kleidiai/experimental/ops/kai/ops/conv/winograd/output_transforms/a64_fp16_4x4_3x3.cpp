//
// SPDX-FileCopyrightText: Copyright 2022, 2024, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#include <algorithm>
#include <arm_neon.h>
#include <cstddef>

namespace kai {
namespace ops {
namespace winograd {
namespace output_transform {

void a64_fp16_4x4_3x3(
    unsigned int n_channels,
    const __fp16* inptr,
    const size_t matrix_stride,
    const __fp16* bptr,
    __fp16* const output,
    const size_t output_row_stride,
    const size_t output_col_stride,
    const __fp16 output_min,
    const __fp16 output_max
)
{
    constexpr int output_tile_rows = 4, output_tile_cols = 4;

    // Construct a map to the output cells
    __fp16 *outptrs[output_tile_rows][output_tile_cols];
    for (int i = 0; i < output_tile_rows; i++)
    {
        for (int j = 0; j < output_tile_cols; j++)
        {
            outptrs[i][j] = output + i*output_row_stride + j*output_col_stride;
        }
    }

    // For each channel of the output
    int channels_remaining = n_channels;
    const __fp16 scale_factor = 16.0f;

#ifdef __aarch64__
    for (; channels_remaining >= 8; channels_remaining -= 8)
  {
    // Matrices used and computed during this transform
    float16x8_t F[6][6], FZ[6][4], f[4][4], b;

    // Read a 6x6 tile in the Winograd domain
    for (int i = 0, m = 0; i < 6; i++)
    {
      for (int j = 0; j < 6; j++, m++)
      {
        F[i][j] = vld1q_f16(inptr + m*matrix_stride);
      }
    }
    inptr += 8;

    const auto _1over2q = vdupq_n_f16(1.0f/2.0f);
    const auto _1over4q = vdupq_n_f16(1.0f/4.0f);
    const auto _1over8q = vdupq_n_f16(1.0f/8.0f);

    // Compute the matrix F Z
    for (int i = 0; i < 6; i++)
    {
      // FZ[i][0] = 16 * (0.5*(F[i][0] + F[i][1] + F[i][2]) + F[i][3] + 0.125*F[i][4])
      auto tmp1 = vmulq_f16(vaddq_f16(vaddq_f16(F[i][0], F[i][1]), F[i][2]), _1over2q);
      auto tmp2 = vaddq_f16(F[i][3], vmulq_f16(F[i][4], _1over8q));
      FZ[i][0] = vmulq_n_f16(vaddq_f16(tmp1, tmp2), scale_factor);

      // FZ[i][1] = 16 * (-0.5*(F[i][1] + F[i][3] - F[i][2]) + 0.25*F[i][4])
      tmp1 = vmulq_f16(vsubq_f16(F[i][2], vaddq_f16(F[i][1], F[i][3])), _1over2q);
      FZ[i][1] = vmulq_n_f16(vaddq_f16(tmp1, vmulq_f16(F[i][4], _1over4q)), scale_factor);

      // FZ[i][2] = 16 * (0.5*(F[i][1] + F[i][2] + F[i][4]) + 0.25*F[i][3])
      tmp1 = vmulq_f16(vaddq_f16(vaddq_f16(F[i][1], F[i][2]), F[i][4]), _1over2q);
      FZ[i][2] = vmulq_n_f16(vaddq_f16(tmp1, vmulq_f16(F[i][3], _1over4q)), scale_factor);

      // FZ[i][3] = 16 * (0.5*(F[i][5] + F[i][2] - F[i][1]) + (F[i][4] - 0.125*F[i][3]))
      tmp1 = vmulq_f16(vsubq_f16(vaddq_f16(F[i][5], F[i][2]), F[i][1]), _1over2q);
      tmp2 = vsubq_f16(F[i][4], vmulq_f16(F[i][3], _1over8q));
      FZ[i][3] = vmulq_n_f16(vaddq_f16(tmp1, tmp2), scale_factor);
    }

    // Compute the output tile f = ZT F Z
    for (int j = 0; j < 4; j++)
    {
        // f[0][j] = 16 * (0.5*(FZ[0][j] + FZ[1][j] + FZ[2][j]) + FZ[3][j] + 0.125*FZ[4][j])
        auto tmp1 = vmulq_f16(vaddq_f16(vaddq_f16(FZ[0][j], FZ[1][j]), FZ[2][j]), _1over2q);
        auto tmp2 = vaddq_f16(FZ[3][j], vmulq_f16(FZ[4][j], _1over8q));
        f[0][j] = vmulq_n_f16(vaddq_f16(tmp1, tmp2), scale_factor);

        // f[1][j] = 16 * (-0.5*(FZ[1][j] + FZ[3][j] - FZ[2][j]) + 0.25*FZ[4][j])
        tmp1 = vmulq_f16(vsubq_f16(FZ[2][j], vaddq_f16(FZ[1][j], FZ[3][j])), _1over2q);
        f[1][j] = vmulq_n_f16(vaddq_f16(tmp1, vmulq_f16(FZ[4][j], _1over4q)), scale_factor);

        // f[2][j] = 16 * (0.5*(FZ[1][j] + FZ[2][j] + FZ[4][j]) + 0.25*FZ[3][j])
        tmp1 = vmulq_f16(vaddq_f16(vaddq_f16(FZ[1][j], FZ[2][j]), FZ[4][j]), _1over2q);
        f[2][j] = vmulq_n_f16(vaddq_f16(tmp1, vmulq_f16(FZ[3][j], _1over4q)), scale_factor);

        // f[3][j] = 16 * (0.5*(FZ[5][j] + FZ[2][j] - FZ[1][j]) + (FZ[4][j] - 0.125*FZ[3][j]))
        tmp1 = vmulq_f16(vsubq_f16(vaddq_f16(FZ[5][j], FZ[2][j]), FZ[1][j]), _1over2q);
        tmp2 = vsubq_f16(FZ[4][j], vmulq_f16(FZ[3][j], _1over8q));
        f[3][j] = vmulq_n_f16(vaddq_f16(tmp1, tmp2), scale_factor);
    }

    // Write out the output tile
    if (bptr != nullptr)
    {
      b = vld1q_f16(bptr);
      bptr += 8;
    }
    else
    {
      b = vdupq_n_f16(0.0f);
    }
    for (int i = 0; i < output_tile_rows; i++)
    {
      for (int j = 0; j < output_tile_cols; j++)
      {
        const auto y =
            vmaxq_f16(vminq_f16(vaddq_f16(f[i][j], b), vdupq_n_f16(output_max)),
                     vdupq_n_f16(output_min));
        vst1q_f16(outptrs[i][j], y);
        outptrs[i][j] += 8;
      }
    }
  }
#endif  // __aarch64__
    for (; channels_remaining >= 4; channels_remaining -= 4)
  {
    // Matrices used and computed during this transform
    float16x4_t F[6][6], FZ[6][4], f[4][4], b;

    // Read a 6x6 tile in the Winograd domain
    for (int i = 0, m = 0; i < 6; i++)
    {
      for (int j = 0; j < 6; j++, m++)
      {
        F[i][j] = vld1_f16(inptr + m*matrix_stride);
      }
    }
    inptr += 4;

    const auto _1over2 = vdup_n_f16(1.0f/2.0f);
    const auto _1over4 = vdup_n_f16(1.0f/4.0f);
    const auto _1over8 = vdup_n_f16(1.0f/8.0f);

    // Compute the matrix F Z
    for (int i = 0; i < 6; i++)
    {
        // FZ[i][0] = 16 * (0.5*(F[i][0] + F[i][1] + F[i][2]) + F[i][3] + 0.125*F[i][4])
        auto tmp1 = vmul_f16(vadd_f16(vadd_f16(F[i][0], F[i][1]), F[i][2]), _1over2);
        auto tmp2 = vadd_f16(F[i][3], vmul_f16(F[i][4], _1over8));
        FZ[i][0] = vmul_n_f16(vadd_f16(tmp1, tmp2), scale_factor);

        // FZ[i][1] = 16 * (-0.5*(F[i][1] + F[i][3] - F[i][2]) + 0.25*F[i][4])
        tmp1 = vmul_f16(vsub_f16(F[i][2], vadd_f16(F[i][1], F[i][3])), _1over2);
        FZ[i][1] = vmul_n_f16(vadd_f16(tmp1, vmul_f16(F[i][4], _1over4)), scale_factor);

        // FZ[i][2] = 16 * (0.5*(F[i][1] + F[i][2] + F[i][4]) + 0.25*F[i][3])
        tmp1 = vmul_f16(vadd_f16(vadd_f16(F[i][1], F[i][2]), F[i][4]), _1over2);
        FZ[i][2] = vmul_n_f16(vadd_f16(tmp1, vmul_f16(F[i][3], _1over4)), scale_factor);

        // FZ[i][3] = 16 * (0.5*(F[i][5] + F[i][2] - F[i][1]) + (F[i][4] - 0.125*F[i][3]))
        tmp1 = vmul_f16(vsub_f16(vadd_f16(F[i][5], F[i][2]), F[i][1]), _1over2);
        tmp2 = vsub_f16(F[i][4], vmul_f16(F[i][3], _1over8));
        FZ[i][3] = vmul_n_f16(vadd_f16(tmp1, tmp2), scale_factor);
    }

    // Compute the output tile f = ZT F Z
    for (int j = 0; j < 4; j++)
    {
        // f[0][j] = 16 * (0.5*(FZ[0][j] + FZ[1][j] + FZ[2][j]) + FZ[3][j] + 0.125*FZ[4][j])
        auto tmp1 = vmul_f16(vadd_f16(vadd_f16(FZ[0][j], FZ[1][j]), FZ[2][j]), _1over2);
        auto tmp2 = vadd_f16(FZ[3][j], vmul_f16(FZ[4][j], _1over8));
        f[0][j] = vmul_n_f16(vadd_f16(tmp1, tmp2), scale_factor);

        // f[1][j] = 16 * (-0.5*(FZ[1][j] + FZ[3][j] - FZ[2][j]) + 0.25*FZ[4][j])
        tmp1 = vmul_f16(vsub_f16(FZ[2][j], vadd_f16(FZ[1][j], FZ[3][j])), _1over2);
        f[1][j] = vmul_n_f16(vadd_f16(tmp1, vmul_f16(FZ[4][j], _1over4)), scale_factor);

        // f[2][j] = 16 * (0.5*(FZ[1][j] + FZ[2][j] + FZ[4][j]) + 0.25*FZ[3][j])
        tmp1 = vmul_f16(vadd_f16(vadd_f16(FZ[1][j], FZ[2][j]), FZ[4][j]), _1over2);
        f[2][j] = vmul_n_f16(vadd_f16(tmp1, vmul_f16(FZ[3][j], _1over4)), scale_factor);

        // f[3][j] = 16 * (0.5*(FZ[5][j] + FZ[2][j] - FZ[1][j]) + (FZ[4][j] - 0.125*FZ[3][j]))
        tmp1 = vmul_f16(vsub_f16(vadd_f16(FZ[5][j], FZ[2][j]), FZ[1][j]), _1over2);
        tmp2 = vsub_f16(FZ[4][j], vmul_f16(FZ[3][j], _1over8));
        f[3][j] = vmul_n_f16(vadd_f16(tmp1, tmp2), scale_factor);
    }

    // Write out the output tile
    if (bptr != nullptr)
    {
      b = vld1_f16(bptr);
      bptr += 4;
    }
    else
    {
      b = vdup_n_f16(0.0f);
    }
    for (int i = 0; i < output_tile_rows; i++)
    {
      for (int j = 0; j < output_tile_cols; j++)
      {
        const auto y =
            vmax_f16(vmin_f16(vadd_f16(f[i][j], b), vdup_n_f16(output_max)),
                     vdup_n_f16(output_min));
        vst1_f16(outptrs[i][j], y);
        outptrs[i][j] += 4;
      }
    }
  }
    for (; channels_remaining; channels_remaining--)
    {
        // Matrices used and computed during this transform
        __fp16 F[6][6], FZ[6][4], f[4][4], b;

        // Read a 6x6 tile in the Winograd domain
        for (int i = 0, m = 0; i < 6; i++)
        {
            for (int j = 0; j < 6; j++, m++)
            {
                F[i][j] = *(inptr + m*matrix_stride);
            }
        }
        inptr++;

        // Compute the matrix F Z
        for (int i = 0; i < 6; i++)
        {
            FZ[i][0] = scale_factor * (
                0.5f * (F[i][0] + F[i][1] + F[i][2]) +
                F[i][3] + 0.125f * F[i][4]
            );

            FZ[i][1] = scale_factor * (
                0.5f * (F[i][2] - (F[i][1] + F[i][3])) +
                0.25f * F[i][4]
            );

            FZ[i][2] = scale_factor * (
                0.5f * (F[i][1] + F[i][2] + F[i][4]) +
                0.25f * F[i][3]
            );

            FZ[i][3] = scale_factor * (
                0.5f * (F[i][5] + F[i][2] - F[i][1]) +
                (F[i][4] - 0.125f * F[i][3])
            );
        }

        // Compute the output tile f = ZT F Z
        for (int j = 0; j < 4; j++)
        {
            f[0][j] = scale_factor * (
                0.5f * (FZ[0][j] + FZ[1][j] + FZ[2][j]) +
                FZ[3][j] + 0.125f * FZ[4][j]
            );

            f[1][j] = scale_factor * (
                0.5f * (FZ[2][j] - (FZ[1][j] + FZ[3][j])) +
                0.25f * FZ[4][j]
            );

            f[2][j] = scale_factor * (
                0.5f * (FZ[1][j] + FZ[2][j] + FZ[4][j]) +
                0.25f * FZ[3][j]
            );

            f[3][j] = scale_factor * (
                0.5f * (FZ[5][j] + FZ[2][j] - FZ[1][j]) +
                (FZ[4][j] - 0.125f * FZ[3][j])
            );
        }

        // Write out the output tile
        if (bptr != nullptr)
        {
            b = *(bptr++);
        }
        else
        {
            b = 0.0f;
        }
        for (int i = 0; i < output_tile_rows; i++)
        {
            for (int j = 0; j < output_tile_cols; j++)
            {
                const auto y = std::max(std::min<__fp16>(f[i][j] + b, output_max), output_min);
                *(outptrs[i][j]++) = y;
            }
        }
    }
}

} // namespace output_transform
} // namespace winograd
}  // namespace ops
}  // namespace kai

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
