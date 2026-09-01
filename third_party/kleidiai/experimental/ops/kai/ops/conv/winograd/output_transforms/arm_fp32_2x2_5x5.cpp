//
// SPDX-FileCopyrightText: Copyright 2022, 2024, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstddef>
#include <arm_neon.h>

namespace kai {
namespace ops {
namespace winograd {
namespace output_transform {

void arm_fp32_2x2_5x5(
  unsigned int n_channels,
  const float* inptr,
  const size_t matrix_stride,
  const float* bptr,
  float *outptr,
  const size_t output_row_stride,
  const size_t output_col_stride,
  const float output_min,
  const float output_max
)
{
  constexpr auto output_tile_rows = 2u, output_tile_cols = 2u;

  // For each channel of the output
  for (; n_channels >= 4; n_channels -= 4)
  {
    // Matrices used and computed during this transform
    float32x4_t F[6][6], FZ[6][2], f[2][2], b;

    // Read a 6x6 tile in the Winograd domain
    for (auto i = 0u, m = 0u; i < 6; i++)
    {
      for (auto j = 0u; j < 6; j++, m++)
      {
        F[i][j] = vld1q_f32(inptr + m*matrix_stride);
      }
    }
    inptr += 4;

    // Compute the matrix F Z
    for (auto i = 0u; i < 6; i++)
    {
      // FZ[i][0] =  1*F[i][0] +  1*F[i][1] +  1*F[i][2] +  1*F[i][3] +  1*F[i][4];
      FZ[i][0] = vaddq_f32(vaddq_f32(vaddq_f32(F[i][0], F[i][1]), vaddq_f32(F[i][2], F[i][3])), F[i][4]);

      // FZ[i][1] =               1*F[i][1] + -1*F[i][2] +  2*F[i][3] + -2*F[i][4] +  1*F[i][5];
      FZ[i][1] = vaddq_f32(vmlaq_n_f32(vsubq_f32(F[i][1], F[i][2]), vsubq_f32(F[i][3], F[i][4]), 2.0f), F[i][5]);
    }

    // Compute the output tile f = ZT F Z
    for (auto j = 0u; j < 2; j++)
    {
      // f[0][j] =  1*FZ[0][j] +  1*FZ[1][j] +  1*FZ[2][j] +  1*FZ[3][j] +  1*FZ[4][j];
      f[0][j] = vaddq_f32(vaddq_f32(vaddq_f32(FZ[0][j], FZ[1][j]), vaddq_f32(FZ[2][j], FZ[3][j])), FZ[4][j]);

      // f[1][j] =               1*FZ[1][j] + -1*FZ[2][j] +  2*FZ[3][j] + -2*FZ[4][j] +  1*FZ[5][j];
      f[1][j] = vaddq_f32(vmlaq_n_f32(vsubq_f32(FZ[1][j], FZ[2][j]), vsubq_f32(FZ[3][j], FZ[4][j]), 2.0f), FZ[5][j]);
    }

    // Write out the output tile
    if (bptr != nullptr)
    {
      b = vld1q_f32(bptr);
      bptr += 4;
    }
    else
    {
      b = vdupq_n_f32(0.0f);
    }
    for (auto i = 0u; i < output_tile_rows; i++)
    {
      for (auto j = 0u; j < output_tile_cols; j++)
      {
        const auto y =
            vmaxq_f32(vminq_f32(vaddq_f32(f[i][j], b), vdupq_n_f32(output_max)),
                      vdupq_n_f32(output_min));
        vst1q_f32(outptr + i*output_row_stride + j*output_col_stride, y);
      }
    }
    outptr += 4;
  }
  for (; n_channels >= 2; n_channels -= 2)
  {
    // Matrices used and computed during this transform
    float32x2_t F[6][6], FZ[6][2], f[2][2], b;

    // Read a 6x6 tile in the Winograd domain
    for (auto i = 0u, m = 0u; i < 6; i++)
    {
      for (auto j = 0u; j < 6; j++, m++)
      {
        F[i][j] = vld1_f32(inptr + m*matrix_stride);
      }
    }
    inptr += 2;

    // Compute the matrix F Z
    for (auto i = 0u; i < 6; i++)
    {
      // FZ[i][0] =  1*F[i][0] +  1*F[i][1] +  1*F[i][2] +  1*F[i][3] +  1*F[i][4];
      FZ[i][0] = vadd_f32(vadd_f32(vadd_f32(F[i][0], F[i][1]), vadd_f32(F[i][2], F[i][3])), F[i][4]);

      // FZ[i][1] =               1*F[i][1] + -1*F[i][2] +  2*F[i][3] + -2*F[i][4] +  1*F[i][5];
      FZ[i][1] = vadd_f32(vmla_n_f32(vsub_f32(F[i][1], F[i][2]), vsub_f32(F[i][3], F[i][4]), 2.0f), F[i][5]);
    }

    // Compute the output tile f = ZT F Z
    for (auto j = 0u; j < 2; j++)
    {
      // f[0][j] =  1*FZ[0][j] +  1*FZ[1][j] +  1*FZ[2][j] +  1*FZ[3][j] +  1*FZ[4][j];
      f[0][j] = vadd_f32(vadd_f32(vadd_f32(FZ[0][j], FZ[1][j]), vadd_f32(FZ[2][j], FZ[3][j])), FZ[4][j]);

      // f[1][j] =               1*FZ[1][j] + -1*FZ[2][j] +  2*FZ[3][j] + -2*FZ[4][j] +  1*FZ[5][j];
      f[1][j] = vadd_f32(vmla_n_f32(vsub_f32(FZ[1][j], FZ[2][j]), vsub_f32(FZ[3][j], FZ[4][j]), 2.0f), FZ[5][j]);
    }

    // Write out the output tile
    if (bptr != nullptr)
    {
      b = vld1_f32(bptr);
      bptr += 2;
    }
    else
    {
      b = vdup_n_f32(0.0f);
    }
    for (auto i = 0u; i < output_tile_rows; i++)
    {
      for (auto j = 0u; j < output_tile_cols; j++)
      {
        const auto y =
            vmax_f32(vmin_f32(vadd_f32(f[i][j], b), vdup_n_f32(output_max)),
                     vdup_n_f32(output_min));
        vst1_f32(outptr + i*output_row_stride + j*output_col_stride, y);
      }
    }
    outptr += 2;
  }
  if (n_channels)
  {
    // Matrices used and computed during this transform
    float F[6][6], FZ[6][2], f[2][2], b;

    // Read a 6x6 tile in the Winograd domain
    for (auto i = 0u, m = 0u; i < 6; i++)
    {
      for (auto j = 0u; j < 6; j++, m++)
      {
        F[i][j] = *(inptr + m*matrix_stride);
      }
    }

    // Compute the matrix F Z
    for (auto i = 0u; i < 6; i++)
    {
      FZ[i][0] =  1*F[i][0] +  1*F[i][1] +  1*F[i][2] +  1*F[i][3] +  1*F[i][4];
      FZ[i][1] =               1*F[i][1] + -1*F[i][2] +  2*F[i][3] + -2*F[i][4] +  1*F[i][5];
    }

    // Compute the output tile f = ZT F Z
    for (auto j = 0u; j < 2; j++)
    {
      f[0][j] =  1*FZ[0][j] +  1*FZ[1][j] +  1*FZ[2][j] +  1*FZ[3][j] +  1*FZ[4][j];
      f[1][j] =                1*FZ[1][j] + -1*FZ[2][j] +  2*FZ[3][j] + -2*FZ[4][j] +  1*FZ[5][j];
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
    for (auto i = 0u; i < output_tile_rows; i++)
    {
      for (auto j = 0u; j < output_tile_cols; j++)
      {
        const auto y = std::max(std::min(f[i][j] + b, output_max), output_min);
        *(outptr + i*output_row_stride + j*output_col_stride) = y;
      }
    }
  }
}

}  // namespace output_transform
}  // namespace winograd
}  // namespace ops
}  // namespace kai
