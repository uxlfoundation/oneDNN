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

void arm_fp32_2x2_5x5(
  unsigned int n_channels,
  const float *inptr, const size_t ld_weight_row, const size_t ld_weight_col,
  float *outptr, const size_t matrix_stride
)
{
#ifdef __aarch64__
  // For each output channel
  for (; n_channels >= 4; n_channels -= 4)
  {
    // Matrices used and computed in this kernel
    float32x4_t w[5][5], Ww[6][5], V[6][6];

    // Read weights
    for (int i = 0; i < 5; i++)
    {
      for (int j = 0; j < 5; j++)
      {
        w[i][j] = vld1q_f32(inptr + i*ld_weight_row + j*ld_weight_col);
      }
    }

    // Compute the matrix W w
    for (int j = 0; j < 5; j++)
    {
      // Ww[0][j] = w[0][j]/4.0f;
      Ww[0][j] = vmulq_n_f32(w[0][j], 1.0f/4.0f);

      // Ww[1][j] = -( w[0][j] + w[1][j] + w[2][j] + w[3][j] + w[4][j])/6.0f;
      Ww[1][j] = vmulq_n_f32(
        vaddq_f32(
          vaddq_f32(
            vaddq_f32(w[1][j], w[0][j]),
            vaddq_f32(w[3][j], w[2][j])
          ),
          w[4][j]
        ),
        -1.0f/6.0f
      );

      // Ww[2][j] = +(-w[0][j] + w[1][j] - w[2][j] + w[3][j] - w[4][j])/6.0f;
      // Ww[2][j] = ((w[1][j] - w[0][j]) + (w[3][j] - w[2][j]) - w[4][j])/6.0f;
      Ww[2][j] = vmulq_n_f32(
        vsubq_f32(
          vaddq_f32(
            vsubq_f32(w[1][j], w[0][j]),
            vsubq_f32(w[3][j], w[2][j])
          ),
          w[4][j]
        ),
        1.0f/6.0f
      );

      // Ww[3][j] = (w[0][j]/8.0f + w[1][j]/4.0f + w[2][j]/2.0f + w[3][j] + 2*w[4][j])/3.0f;
      Ww[3][j] = vmulq_n_f32(
        vmlaq_n_f32(
          vaddq_f32(
            vaddq_f32(vmulq_n_f32(w[0][j], 1.0f/8.0f), vmulq_n_f32(w[1][j], 1.0f/4.0f)),
            vaddq_f32(vmulq_n_f32(w[2][j], 1.0f/2.0f), w[3][j])
          ),
          w[4][j], 2.0f
        ),
        1.0f/3.0f
      );

      // Ww[4][j] = (w[0][j]/8.0f - w[1][j]/4.0f + w[2][j]/2.0f - w[3][j] + 2*w[4][j])/3.0f;
      Ww[4][j] = vmulq_n_f32(
        vmlaq_n_f32(
          vaddq_f32(
            vsubq_f32(vmulq_n_f32(w[0][j], 1.0f/8.0f), vmulq_n_f32(w[1][j], 1.0f/4.0f)),
            vsubq_f32(vmulq_n_f32(w[2][j], 1.0f/2.0f), w[3][j])
          ),
          w[4][j], 2.0f
        ),
        1.0f/3.0f
      );

      // Ww[5][j] = w[4][j];
      Ww[5][j] = w[4][j];
    }

    // Compute V = W w WT
    for (int i = 0; i < 6; i++)
    {
      // V[i][0] = Ww[i][0]/4.0f;
      V[i][0] = vmulq_n_f32(Ww[i][0], 1.0f/4.0f);

      // V[i][1] = -( Ww[i][0] + Ww[i][1] + Ww[i][2] + Ww[i][3] + Ww[i][4])/6.0f;
      V[i][1] = vmulq_n_f32(
        vaddq_f32(
          vaddq_f32(
            vaddq_f32(Ww[i][1], Ww[i][0]),
            vaddq_f32(Ww[i][3], Ww[i][2])
          ),
          Ww[i][4]
        ),
        -1.0f/6.0f
      );

      // V[i][2] = +(-Ww[i][0] + Ww[i][1] - Ww[i][2] + Ww[i][3] - Ww[i][4])/6.0f;
      // V[i][2] = ((Ww[i][1] - Ww[i][0]) + (Ww[i][3] - Ww[i][2]) - Ww[i][4])/6.0f;
      V[i][2] = vmulq_n_f32(
        vsubq_f32(
          vaddq_f32(
            vsubq_f32(Ww[i][1], Ww[i][0]),
            vsubq_f32(Ww[i][3], Ww[i][2])
          ),
          Ww[i][4]
        ),
        1.0f/6.0f
      );

      // V[i][3] = (Ww[i][0]/8.0f + Ww[i][1]/4.0f + Ww[i][2]/2.0f + Ww[i][3] + 2*Ww[i][4])/3.0f;
      V[i][3] = vmulq_n_f32(
        vmlaq_n_f32(
          vaddq_f32(
            vaddq_f32(vmulq_n_f32(Ww[i][0], 1.0f/8.0f), vmulq_n_f32(Ww[i][1], 1.0f/4.0f)),
            vaddq_f32(vmulq_n_f32(Ww[i][2], 1.0f/2.0f), Ww[i][3])
          ),
          Ww[i][4], 2.0f
        ),
        1.0f/3.0f
      );

      // V[i][4] = (Ww[i][0]/8.0f - Ww[i][1]/4.0f + Ww[i][2]/2.0f - Ww[i][3] + 2*Ww[i][4])/3.0f;
      V[i][4] = vmulq_n_f32(
        vmlaq_n_f32(
          vaddq_f32(
            vsubq_f32(vmulq_n_f32(Ww[i][0], 1.0f/8.0f), vmulq_n_f32(Ww[i][1], 1.0f/4.0f)),
            vsubq_f32(vmulq_n_f32(Ww[i][2], 1.0f/2.0f), Ww[i][3])
          ),
          Ww[i][4], 2.0f
        ),
        1.0f/3.0f
      );

      // V[i][5] = Ww[i][4];
      V[i][5] = Ww[i][4];
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
    float32x2_t w[5][5], Ww[6][5], V[6][6];

    // Read weights
    for (int i = 0; i < 5; i++)
    {
      for (int j = 0; j < 5; j++)
      {
        w[i][j] = vld1_f32(inptr + i*ld_weight_row + j*ld_weight_col);
      }
    }

    // Compute the matrix W w
    for (int j = 0; j < 5; j++)
    {
      // Ww[0][j] = w[0][j]/4.0f;
      Ww[0][j] = vmul_n_f32(w[0][j], 1.0f/4.0f);

      // Ww[1][j] = -( w[0][j] + w[1][j] + w[2][j] + w[3][j] + w[4][j])/6.0f;
      Ww[1][j] = vmul_n_f32(
        vadd_f32(
          vadd_f32(
            vadd_f32(w[1][j], w[0][j]),
            vadd_f32(w[3][j], w[2][j])
          ),
          w[4][j]
        ),
        -1.0f/6.0f
      );

      // Ww[2][j] = +(-w[0][j] + w[1][j] - w[2][j] + w[3][j] - w[4][j])/6.0f;
      // Ww[2][j] = ((w[1][j] - w[0][j]) + (w[3][j] - w[2][j]) - w[4][j])/6.0f;
      Ww[2][j] = vmul_n_f32(
        vsub_f32(
          vadd_f32(
            vsub_f32(w[1][j], w[0][j]),
            vsub_f32(w[3][j], w[2][j])
          ),
          w[4][j]
        ),
        1.0f/6.0f
      );

      // Ww[3][j] = (w[0][j]/8.0f + w[1][j]/4.0f + w[2][j]/2.0f + w[3][j] + 2*w[4][j])/3.0f;
      Ww[3][j] = vmul_n_f32(
        vmla_n_f32(
          vadd_f32(
            vadd_f32(vmul_n_f32(w[0][j], 1.0f/8.0f), vmul_n_f32(w[1][j], 1.0f/4.0f)),
            vadd_f32(vmul_n_f32(w[2][j], 1.0f/2.0f), w[3][j])
          ),
          w[4][j], 2.0f
        ),
        1.0f/3.0f
      );

      // Ww[4][j] = (w[0][j]/8.0f - w[1][j]/4.0f + w[2][j]/2.0f - w[3][j] + 2*w[4][j])/3.0f;
      Ww[4][j] = vmul_n_f32(
        vmla_n_f32(
          vadd_f32(
            vsub_f32(vmul_n_f32(w[0][j], 1.0f/8.0f), vmul_n_f32(w[1][j], 1.0f/4.0f)),
            vsub_f32(vmul_n_f32(w[2][j], 1.0f/2.0f), w[3][j])
          ),
          w[4][j], 2.0f
        ),
        1.0f/3.0f
      );

      // Ww[5][j] = w[4][j];
      Ww[5][j] = w[4][j];
    }

    // Compute V = W w WT
    for (int i = 0; i < 6; i++)
    {
      // V[i][0] = Ww[i][0]/4.0f;
      V[i][0] = vmul_n_f32(Ww[i][0], 1.0f/4.0f);

      // V[i][1] = -( Ww[i][0] + Ww[i][1] + Ww[i][2] + Ww[i][3] + Ww[i][4])/6.0f;
      V[i][1] = vmul_n_f32(
        vadd_f32(
          vadd_f32(
            vadd_f32(Ww[i][1], Ww[i][0]),
            vadd_f32(Ww[i][3], Ww[i][2])
          ),
          Ww[i][4]
        ),
        -1.0f/6.0f
      );

      // V[i][2] = +(-Ww[i][0] + Ww[i][1] - Ww[i][2] + Ww[i][3] - Ww[i][4])/6.0f;
      // V[i][2] = ((Ww[i][1] - Ww[i][0]) + (Ww[i][3] - Ww[i][2]) - Ww[i][4])/6.0f;
      V[i][2] = vmul_n_f32(
        vsub_f32(
          vadd_f32(
            vsub_f32(Ww[i][1], Ww[i][0]),
            vsub_f32(Ww[i][3], Ww[i][2])
          ),
          Ww[i][4]
        ),
        1.0f/6.0f
      );

      // V[i][3] = (Ww[i][0]/8.0f + Ww[i][1]/4.0f + Ww[i][2]/2.0f + Ww[i][3] + 2*Ww[i][4])/3.0f;
      V[i][3] = vmul_n_f32(
        vmla_n_f32(
          vadd_f32(
            vadd_f32(vmul_n_f32(Ww[i][0], 1.0f/8.0f), vmul_n_f32(Ww[i][1], 1.0f/4.0f)),
            vadd_f32(vmul_n_f32(Ww[i][2], 1.0f/2.0f), Ww[i][3])
          ),
          Ww[i][4], 2.0f
        ),
        1.0f/3.0f
      );

      // V[i][4] = (Ww[i][0]/8.0f - Ww[i][1]/4.0f + Ww[i][2]/2.0f - Ww[i][3] + 2*Ww[i][4])/3.0f;
      V[i][4] = vmul_n_f32(
        vmla_n_f32(
          vadd_f32(
            vsub_f32(vmul_n_f32(Ww[i][0], 1.0f/8.0f), vmul_n_f32(Ww[i][1], 1.0f/4.0f)),
            vsub_f32(vmul_n_f32(Ww[i][2], 1.0f/2.0f), Ww[i][3])
          ),
          Ww[i][4], 2.0f
        ),
        1.0f/3.0f
      );

      // V[i][5] = Ww[i][4];
      V[i][5] = Ww[i][4];
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
    float w[5][5], Ww[6][5], V[6][6];

    // Read weights
    for (int i = 0; i < 5; i++)
    {
      for (int j = 0; j < 5; j++)
      {
        w[i][j] = *(inptr + i*ld_weight_row + j*ld_weight_col);
      }
    }

    // Compute the matrix W w
    for (int j = 0; j < 5; j++)
    {
      Ww[0][j] = w[0][j]/4.0f;
      Ww[1][j] = -( w[0][j] + w[1][j] + w[2][j] + w[3][j] + w[4][j])/6.0f;
      Ww[2][j] = +(-w[0][j] + w[1][j] - w[2][j] + w[3][j] - w[4][j])/6.0f;
      Ww[3][j] = (w[0][j]/8.0f + w[1][j]/4.0f + w[2][j]/2.0f + w[3][j] + 2*w[4][j])/3.0f;
      Ww[4][j] = (w[0][j]/8.0f - w[1][j]/4.0f + w[2][j]/2.0f - w[3][j] + 2*w[4][j])/3.0f;
      Ww[5][j] = w[4][j];
    }

    // Compute V = W w WT
    for (int i = 0; i < 6; i++)
    {
      V[i][0] = Ww[i][0]/4.0f;
      V[i][1] = -( Ww[i][0] + Ww[i][1] + Ww[i][2] + Ww[i][3] + Ww[i][4])/6.0f;
      V[i][2] = +(-Ww[i][0] + Ww[i][1] - Ww[i][2] + Ww[i][3] - Ww[i][4])/6.0f;
      V[i][3] = (Ww[i][0]/8.0f + Ww[i][1]/4.0f + Ww[i][2]/2.0f + Ww[i][3] + 2*Ww[i][4])/3.0f;
      V[i][4] = (Ww[i][0]/8.0f - Ww[i][1]/4.0f + Ww[i][2]/2.0f - Ww[i][3] + 2*Ww[i][4])/3.0f;
      V[i][5] = Ww[i][4];
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
