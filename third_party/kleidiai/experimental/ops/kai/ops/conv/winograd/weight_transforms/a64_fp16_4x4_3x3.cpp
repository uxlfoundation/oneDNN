//
// SPDX-FileCopyrightText: Copyright 2022, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

#include <cstddef>
#include <arm_neon.h>

namespace kai {
namespace ops {
namespace winograd {
namespace weight_transform {

void a64_fp16_4x4_3x3(
    unsigned int n_channels,
    const __fp16* inptr,  // NOTE: Data in HWIO order
    const size_t ld_weight_row,
    const size_t ld_weight_col,
    __fp16* outptr,
    const size_t matrix_stride
)
{
    for (; n_channels >= 8; n_channels -= 8)
    {
      // Matrices used and computed in this kernel
      float16x8_t w[3][3], Ww[6][3], V[6][6];

      // Read weights
      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j < 3; j++)
        {
          w[i][j] = vld1q_f16(inptr + i*ld_weight_row + j*ld_weight_col);
        }
      }

      const auto _1over3q = vdupq_n_f16(1.0f/3.0f);
      const auto minus_1over3q = vdupq_n_f16(-1.0f/3.0f);

      // Compute the matrix W w
      for (int j = 0; j < 3; j++)
      {
        // Ww[0][j] = 1*w[0][j];
        Ww[0][j] = w[0][j];

        // Ww[1][j] = (w[0][j] - w[1][j] + w[2][j]) * (1/3)
        Ww[1][j] = vmulq_f16(vsubq_f16(vaddq_f16(w[0][j], w[2][j]), w[1][j]), _1over3q);

        // Ww[2][j] = (-w[0][j] - w[1][j] - w[2][j]) * (1/3)
        Ww[2][j] = vmulq_f16(vaddq_f16(vaddq_f16(w[0][j], w[1][j]), w[2][j]), minus_1over3q);

        // Ww[3][j] = -8/15*w[0][j] + 4/15*w[1][j] - 2/15*w[2][j]
        auto tmp1 = vmulq_n_f16(w[1][j], 4.0f/15.0f);
        auto tmp2 = vaddq_f16( vmulq_n_f16(w[0][j], 8.0f/15.0f), vmulq_n_f16(w[2][j], 2.0f/15.0f));
        Ww[3][j] = vsubq_f16(tmp1, tmp2);

        // Ww[4][j] = 2/15*w[0][j] + 4/15*w[1][j] + 8/15*w[2][j]
        tmp1 = vmulq_n_f16(w[0][j], 2.0f/15.0f);
        tmp2 = vaddq_f16( vmulq_n_f16(w[1][j], 4.0f/15.0f), vmulq_n_f16(w[2][j], 8.0f/15.0f));
        Ww[4][j] = vaddq_f16(tmp1, tmp2);

        // Ww[5][j] = 1*w[2][j];
        Ww[5][j] = w[2][j];
      }

      // Compute V = W w WT
      for (int i = 0; i < 6; i++)
      {
        // V[i][0] = 1*Ww[i][0];
        V[i][0] = Ww[i][0];

        // V[i][1] = (Ww[i][0] - Ww[i][1] + Ww[i][2]) * (1/3)
        V[i][1] = vmulq_f16(vsubq_f16(vaddq_f16(Ww[i][0], Ww[i][2]), Ww[i][1]), _1over3q);

        // V[i][2] = (-Ww[i][0] - Ww[i][1] - Ww[i][2]) * (1/3)
        V[i][2] = vmulq_f16(vaddq_f16(vaddq_f16(Ww[i][0], Ww[i][1]), Ww[i][2]), minus_1over3q);

        // V[i][3] = -8/15*Ww[i][0] + 4/15*Ww[i][1] - 2/15*Ww[i][2]
        auto tmp1 = vmulq_n_f16(Ww[i][1], 4.0f/15.0f);
        auto tmp2 = vaddq_f16( vmulq_n_f16(Ww[i][0], 8.0f/15.0f), vmulq_n_f16(Ww[i][2], 2.0f/15.0f));
        V[i][3] = vsubq_f16(tmp1, tmp2);

        // V[i][4] = 2/15*Ww[i][0] + 4/15*Ww[i][1] + 8/15*Ww[i][2]
        tmp1 = vmulq_n_f16(Ww[i][0], 2.0f/15.0f);
        tmp2 = vaddq_f16( vmulq_n_f16(Ww[i][1], 4.0f/15.0f), vmulq_n_f16(Ww[i][2], 8.0f/15.0f));
        V[i][4] = vaddq_f16(tmp1, tmp2);

        // V[i][5] = 1*Ww[i][2];
        V[i][5] = Ww[i][2];
      }

      // Store the transformed weights
      for (int i = 0, m = 0; i < 6; i++)
      {
        for (int j = 0; j < 6; j++, m++)
        {
          vst1q_f16(outptr + m*matrix_stride, V[i][j]);
        }
      }
      inptr += 8;
      outptr += 8;
    }
    for (; n_channels >= 4; n_channels -= 4)
    {
      const auto _1over3 = vdup_n_f16(1.0f/3.0f);
      const auto minus_1over3 = vdup_n_f16(-1.0f/3.0f);

      // Matrices used and computed in this kernel
      float16x4_t w[3][3], Ww[6][3], V[6][6];

      // Read weights
      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j < 3; j++)
        {
          w[i][j] = vld1_f16(inptr + i*ld_weight_row + j*ld_weight_col);
        }
      }

      // Compute the matrix W w
      for (int j = 0; j < 3; j++)
      {
        // Ww[0][j] = 1*w[0][j];
        Ww[0][j] = w[0][j];

        // Ww[1][j] = (w[0][j] - w[1][j] + w[2][j]) * (1/3)
        Ww[1][j] = vmul_f16(vsub_f16(vadd_f16(w[0][j], w[2][j]), w[1][j]), _1over3);

        // Ww[2][j] = (-w[0][j] - w[1][j] - w[2][j]) * (1/3)
        Ww[2][j] = vmul_f16(vadd_f16(vadd_f16(w[0][j], w[1][j]), w[2][j]), minus_1over3);

        // Ww[3][j] = -8/15*w[0][j] + 4/15*w[1][j] - 2/15*w[2][j]
        auto tmp1 = vmul_n_f16(w[1][j], 4.0f/15.0f);
        auto tmp2 = vadd_f16( vmul_n_f16(w[0][j], 8.0f/15.0f), vmul_n_f16(w[2][j], 2.0f/15.0f));
        Ww[3][j] = vsub_f16(tmp1, tmp2);

        // Ww[4][j] = 2/15*w[0][j] + 4/15*w[1][j] + 8/15*w[2][j]
        tmp1 = vmul_n_f16(w[0][j], 2.0f/15.0f);
        tmp2 = vadd_f16( vmul_n_f16(w[1][j], 4.0f/15.0f), vmul_n_f16(w[2][j], 8.0f/15.0f));
        Ww[4][j] = vadd_f16(tmp1, tmp2);

        // Ww[5][j] = 1*w[2][j];
        Ww[5][j] = w[2][j];
      }

      // Compute V = W w WT
      for (int i = 0; i < 6; i++)
      {
        // V[i][0] = 1 * Ww[i][0];
        V[i][0] = Ww[i][0];

        // V[i][1] = (Ww[i][0] - Ww[i][1] + Ww[i][2]) * (1/3)
        V[i][1] = vmul_f16(vsub_f16(vadd_f16(Ww[i][0], Ww[i][2]), Ww[i][1]), _1over3);

        // V[i][2] = (-Ww[i][0] - Ww[i][1] - Ww[i][2]) * (1/3)
        V[i][2] = vmul_f16(vadd_f16(vadd_f16(Ww[i][0], Ww[i][1]), Ww[i][2]), minus_1over3);

        // V[i][3] = -8/15*Ww[i][0] + 4/15*Ww[i][1] - 2/15*Ww[i][2]
        auto tmp1 = vmul_n_f16(Ww[i][1], 4.0f/15.0f);
        auto tmp2 = vadd_f16( vmul_n_f16(Ww[i][0], 8.0f/15.0f), vmul_n_f16(Ww[i][2], 2.0f/15.0f));
        V[i][3] = vsub_f16(tmp1, tmp2);

        // V[i][4] = 2/15*Ww[i][0] + 4/15*Ww[i][1] + 8/15*Ww[i][2]
        tmp1 = vmul_n_f16(Ww[i][0], 2.0f/15.0f);
        tmp2 = vadd_f16( vmul_n_f16(Ww[i][1], 4.0f/15.0f), vmul_n_f16(Ww[i][2], 8.0f/15.0f));
        V[i][4] = vadd_f16(tmp1, tmp2);

        // V[i][5] = 1 * Ww[i][2];
        V[i][5] = Ww[i][2];
      }

      // Store the transformed weights
      for (int i = 0, m = 0; i < 6; i++)
      {
        for (int j = 0; j < 6; j++, m++)
        {
          vst1_f16(outptr + m*matrix_stride, V[i][j]);
        }
      }
      inptr += 4;
      outptr += 4;
    }
    for (; n_channels; n_channels--)
    {
      // Matrices used and computed in this kernel
      __fp16 w[3][3], Ww[6][3], V[6][6];

      // Read weights
      for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j < 3; j++)
        {
          w[i][j] = *(inptr + i*ld_weight_row + j*ld_weight_col);
        }
      }

      // Compute the matrix W w
      for (int j = 0; j < 3; j++)
      {
        Ww[0][j] = w[0][j];
        Ww[1][j] = (w[0][j] + w[2][j] - w[1][j]) * (1.0f / 3.0f);
        Ww[2][j] = -(w[0][j] + w[1][j] + w[2][j]) * (1.0f / 3.0f);
        Ww[3][j] = (4.0f / 15.0f) * w[1][j] - ((8.0f / 15.0f) * w[0][j] + (2.0f / 15.0f) * w[2][j]);
        Ww[4][j] = (2.0f / 15.0f) * w[0][j] + (4.0f / 15.0f) * w[1][j] + (8.0f / 15.0f) * w[2][j];
        Ww[5][j] = w[2][j];
      }

      // Compute V = W w WT
      for (int i = 0; i < 6; i++)
      {
        V[i][0] = Ww[i][0];
        V[i][1] = (Ww[i][0] + Ww[i][2] - Ww[i][1]) * (1.0f / 3.0f);
        V[i][2] = -(Ww[i][0] + Ww[i][1] + Ww[i][2]) * (1.0f / 3.0f);
        V[i][3] = (4.0f / 15.0f) * Ww[i][1] - ((8.0f / 15.0f) * Ww[i][0] + (2.0f / 15.0f) * Ww[i][2]);
        V[i][4] = (2.0f / 15.0f) * Ww[i][0] + (4.0f / 15.0f) * Ww[i][1] + (8.0f / 15.0f) * Ww[i][2];
        V[i][5] = Ww[i][2];
      }

      // Store the transformed weights
      for (int i = 0, m = 0; i < 6; i++)
      {
        for (int j = 0; j < 6; j++, m++)
        {
          *(outptr + m*matrix_stride) = V[i][j];
        }
      }

      inptr++;
      outptr++;
    }
}

}  // namespace weight_transform
}  // namespace winograd
}  // namespace ops
}  // namespace kai

#endif // defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
