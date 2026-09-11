//
// SPDX-FileCopyrightText: Copyright 2022, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <arm_neon.h>

namespace kai {
namespace ops {
namespace winograd {
namespace weight_transform {

void arm_fp32_4x4_3x3(
  unsigned int n_channels,
  const float *inptr, const size_t ld_weight_row, const size_t ld_weight_col,
  float *outptr, const size_t matrix_stride
)
{
#ifdef __aarch64__
  for (; n_channels >= 4; n_channels -= 4)
  {
    // Matrices used and computed in this kernel
    float32x4_t w[3][3], Ww[6][3], V[6][6];

    // Read weights
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        w[i][j] = vld1q_f32(inptr + i*ld_weight_row + j*ld_weight_col);
      }
    }

    // Compute the matrix W w
    for (int j = 0; j < 3; j++)
    {
      // Ww[0][j] =  6*w[0][j];
      Ww[0][j] = vmulq_n_f32(w[0][j], 6.0);

      // Ww[1][j] = -4*w[0][j] + -4*w[1][j] + -4*w[2][j];
      Ww[1][j] = vmulq_n_f32(vaddq_f32(vaddq_f32(w[0][j], w[1][j]), w[2][j]), -4.0);

      // Ww[2][j] = -4*w[0][j] +  4*w[1][j] + -4*w[2][j];
      Ww[2][j] = vmulq_n_f32(vsubq_f32(vsubq_f32(w[1][j], w[0][j]), w[2][j]), 4.0);

      // Ww[3][j] =  1*w[0][j] +  2*w[1][j] +  4*w[2][j];
      Ww[3][j] = vmlaq_n_f32(vmlaq_n_f32(w[0][j], w[1][j], 2.0f), w[2][j], 4.0f);

      // Ww[4][j] =  1*w[0][j] + -2*w[1][j] +  4*w[2][j];
      Ww[4][j] = vmlaq_n_f32(vmlsq_n_f32(w[0][j], w[1][j], 2.0f), w[2][j], 4.0f);

      // Ww[5][j] = 24*w[2][j];
      Ww[5][j] = vmulq_n_f32(w[2][j], 24.0f);
    }

    // Compute V = W w WT
    for (int i = 0; i < 6; i++)
    {
      const float recip576 = 1.0f / 576.0f;

      // V[i][0] =  6*Ww[i][0];
      V[i][0] = vmulq_n_f32(vmulq_n_f32(Ww[i][0], 6.0), recip576);

      // V[i][1] = -4*Ww[i][0] + -4*Ww[i][1] + -4*Ww[i][2];
      V[i][1] = vmulq_n_f32(vmulq_n_f32(vaddq_f32(vaddq_f32(Ww[i][0], Ww[i][1]), Ww[i][2]), -4.0), recip576);

      // V[i][2] = -4*Ww[i][0] +  4*Ww[i][1] + -4*Ww[i][2];
      V[i][2] = vmulq_n_f32(vmulq_n_f32(vsubq_f32(vsubq_f32(Ww[i][1], Ww[i][0]), Ww[i][2]), 4.0), recip576);

      // V[i][3] =  1*Ww[i][0] +  2*Ww[i][1] +  4*Ww[i][2];
      V[i][3] = vmulq_n_f32(vmlaq_n_f32(vmlaq_n_f32(Ww[i][0], Ww[i][1], 2.0f), Ww[i][2], 4.0f), recip576);

      // V[i][4] =  1*Ww[i][0] + -2*Ww[i][1] +  4*Ww[i][2];
      V[i][4] = vmulq_n_f32(vmlaq_n_f32(vmlsq_n_f32(Ww[i][0], Ww[i][1], 2.0f), Ww[i][2], 4.0f), recip576);

      // V[i][5] = 24*Ww[i][2];
      V[i][5] = vmulq_n_f32(vmulq_n_f32(Ww[i][2], 24.0f), recip576);
    }

    // Store the transformed weights
    for (int i = 0, m = 0; i < 6; i++)
    {
      for (int j = 0; j < 6; j++, m++)
      {
        vst1q_f32(outptr + m*matrix_stride, V[i][j]);
      }
    }

    inptr += 4;
    outptr += 4;
  }
#endif // __aarch64__
  for (; n_channels >= 2; n_channels -= 2)
  {
    // Matrices used and computed in this kernel
    float32x2_t w[3][3], Ww[6][3], V[6][6];

    // Read weights
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        w[i][j] = vld1_f32(inptr + i*ld_weight_row + j*ld_weight_col);
      }
    }

    // Compute the matrix W w
    for (int j = 0; j < 3; j++)
    {
      // Ww[0][j] =  6*w[0][j];
      Ww[0][j] = vmul_n_f32(w[0][j], 6.0);

      // Ww[1][j] = -4*w[0][j] + -4*w[1][j] + -4*w[2][j];
      Ww[1][j] = vmul_n_f32(vadd_f32(vadd_f32(w[0][j], w[1][j]), w[2][j]), -4.0);

      // Ww[2][j] = -4*w[0][j] +  4*w[1][j] + -4*w[2][j];
      Ww[2][j] = vmul_n_f32(vsub_f32(vsub_f32(w[1][j], w[0][j]), w[2][j]), 4.0);

      // Ww[3][j] =  1*w[0][j] +  2*w[1][j] +  4*w[2][j];
      Ww[3][j] = vmla_n_f32(vmla_n_f32(w[0][j], w[1][j], 2.0f), w[2][j], 4.0f);

      // Ww[4][j] =  1*w[0][j] + -2*w[1][j] +  4*w[2][j];
      Ww[4][j] = vmla_n_f32(vmls_n_f32(w[0][j], w[1][j], 2.0f), w[2][j], 4.0f);

      // Ww[5][j] = 24*w[2][j];
      Ww[5][j] = vmul_n_f32(w[2][j], 24.0f);
    }

    // Compute V = W w WT
    for (int i = 0; i < 6; i++)
    {
      const float recip576 = 1.0f / 576.0f;

      // V[i][0] =  6*Ww[i][0];
      V[i][0] = vmul_n_f32(vmul_n_f32(Ww[i][0], 6.0), recip576);

      // V[i][1] = -4*Ww[i][0] + -4*Ww[i][1] + -4*Ww[i][2];
      V[i][1] = vmul_n_f32(vmul_n_f32(vadd_f32(vadd_f32(Ww[i][0], Ww[i][1]), Ww[i][2]), -4.0), recip576);

      // V[i][2] = -4*Ww[i][0] +  4*Ww[i][1] + -4*Ww[i][2];
      V[i][2] = vmul_n_f32(vmul_n_f32(vsub_f32(vsub_f32(Ww[i][1], Ww[i][0]), Ww[i][2]), 4.0), recip576);

      // V[i][3] =  1*Ww[i][0] +  2*Ww[i][1] +  4*Ww[i][2];
      V[i][3] = vmul_n_f32(vmla_n_f32(vmla_n_f32(Ww[i][0], Ww[i][1], 2.0f), Ww[i][2], 4.0f), recip576);

      // V[i][4] =  1*Ww[i][0] + -2*Ww[i][1] +  4*Ww[i][2];
      V[i][4] = vmul_n_f32(vmla_n_f32(vmls_n_f32(Ww[i][0], Ww[i][1], 2.0f), Ww[i][2], 4.0f), recip576);

      // V[i][5] = 24*Ww[i][2];
      V[i][5] = vmul_n_f32(vmul_n_f32(Ww[i][2], 24.0f), recip576);
    }

    // Store the transformed weights
    for (int i = 0, m = 0; i < 6; i++)
    {
      for (int j = 0; j < 6; j++, m++)
      {
        vst1_f32(outptr + m*matrix_stride, V[i][j]);
      }
    }

    inptr += 2;
    outptr += 2;
  }
  for (; n_channels; n_channels--)
  {
    // Matrices used and computed in this kernel
    float w[3][3], Ww[6][3], V[6][6];

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
      Ww[0][j] =  6*w[0][j];
      Ww[1][j] = -4*w[0][j] + -4*w[1][j] + -4*w[2][j];
      Ww[2][j] = -4*w[0][j] +  4*w[1][j] + -4*w[2][j];
      Ww[3][j] =  1*w[0][j] +  2*w[1][j] +  4*w[2][j];
      Ww[4][j] =  1*w[0][j] + -2*w[1][j] +  4*w[2][j];
      Ww[5][j] = 24*w[2][j];
    }

    // Compute V = W w WT
    for (int i = 0; i < 6; i++)
    {
      V[i][0] = ( 6*Ww[i][0]) / 576.0;
      V[i][1] = (-4*Ww[i][0] + -4*Ww[i][1] + -4*Ww[i][2]) / 576.0;
      V[i][2] = (-4*Ww[i][0] +  4*Ww[i][1] + -4*Ww[i][2]) / 576.0;
      V[i][3] = ( 1*Ww[i][0] +  2*Ww[i][1] +  4*Ww[i][2]) / 576.0;
      V[i][4] = ( 1*Ww[i][0] + -2*Ww[i][1] +  4*Ww[i][2]) / 576.0;
      V[i][5] = (24*Ww[i][2]) / 576.0;
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
