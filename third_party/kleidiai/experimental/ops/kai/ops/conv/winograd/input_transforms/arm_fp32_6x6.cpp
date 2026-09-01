//
// SPDX-FileCopyrightText: Copyright 2022, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __aarch64__

#include <arm_neon.h>
#include <cstddef>

namespace kai {
namespace ops {
namespace winograd {
namespace input_transform {

void arm_fp32_6x6(
  unsigned int n_channels,
  const float* const input_base,
  const size_t input_row_stride,
  const size_t input_col_stride,
  float* outptr,
  const size_t matrix_stride
)
{
  constexpr int inner_tile_rows = 6;
  constexpr int inner_tile_cols = 6;

  // Get pointers into the input tile
  const float *x_ptrs[inner_tile_rows][inner_tile_cols];
  for (int i = 0, xi = 0; i < inner_tile_rows; i++, xi++)
  {
    // Get a pointer into the row
    const float* const row_ptr = input_base + xi*input_row_stride;

    for (int j = 0, xj = 0; j < inner_tile_cols; j++, xj++)
    {
      x_ptrs[i][j] = row_ptr + xj*input_col_stride;
    }
  }

  // Matrices used/computed in this kernel.
  float x[inner_tile_rows][inner_tile_cols];
  float XTx[inner_tile_rows][inner_tile_cols];
  float U[inner_tile_rows][inner_tile_cols];
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
  for (; channels_remaining >= 2; channels_remaining -= 2)
  {
    // Matrices used/computed in this kernel
    float32x2_t x[inner_tile_rows][inner_tile_cols];
    float32x2_t XTx[inner_tile_rows][inner_tile_cols];
    float32x2_t U[inner_tile_rows][inner_tile_cols];
    for (int i = 0; i < inner_tile_rows; i++)
    {
      for (int j = 0; j < inner_tile_cols; j++)
      {
        x[i][j] = vdup_n_f32(0.0f);
        XTx[i][j] = vdup_n_f32(0.0f);
      }
    }

    // Read a 6x6 tile in the Winograd domain
    for (int i = 0; i < inner_tile_rows; i++)
    {
      for (int j = 0; j < inner_tile_cols; j++)
      {
        x[i][j] = vld1_f32(x_ptrs[i][j]);
        x_ptrs[i][j] += 2;
      }
    }

    // Compute XT . x
    for (int j = 0; j < inner_tile_cols; j++)
    {
      // XTx[0][j] =  4*x[0][j] + -5*x[2][j] +  1*x[4][j];
      XTx[0][j] = vmls_n_f32(vmla_n_f32(x[4][j], x[0][j], 4.0f), x[2][j], 5.0f);

      // XTx[1][j] = -4*x[1][j] + -4*x[2][j] +  1*x[3][j] +  1*x[4][j];
      XTx[1][j] = vmls_n_f32(vadd_f32(x[3][j], x[4][j]), vadd_f32(x[1][j], x[2][j]), 4.0f);

      // XTx[2][j] =  4*x[1][j] + -4*x[2][j] + -1*x[3][j] +  1*x[4][j];
      XTx[2][j] = vmla_n_f32(vsub_f32(x[4][j], x[3][j]), vsub_f32(x[1][j], x[2][j]), 4.0f);

      // XTx[3][j] = -2*x[1][j] + -1*x[2][j] +  2*x[3][j] +  1*x[4][j];
      XTx[3][j] = vmla_n_f32(vsub_f32(x[4][j], x[2][j]), vsub_f32(x[3][j], x[1][j]), 2.0f);

      // XTx[4][j] =  2*x[1][j] + -1*x[2][j] + -2*x[3][j] +  1*x[4][j];
      XTx[4][j] = vmla_n_f32(vsub_f32(x[4][j], x[2][j]), vsub_f32(x[1][j], x[3][j]), 2.0f);

      // XTx[5][j] =  4*x[1][j] + -5*x[3][j] +  1*x[5][j];
      XTx[5][j] = vmls_n_f32(vmla_n_f32(x[5][j], x[1][j], 4.0f), x[3][j], 5.0f);
    }

    // Compute U = XT . x . X
    for (int i = 0; i < inner_tile_rows; i++)
    {
      // U[i][0] =  4*XTx[i][0] + -5*XTx[i][2] +  1*XTx[i][4];
      U[i][0] = vmls_n_f32(vmla_n_f32(XTx[i][4], XTx[i][0], 4.0f), XTx[i][2], 5.0f);

      // U[i][1] = -4*XTx[i][1] + -4*XTx[i][2] +  1*XTx[i][3] +  1*XTx[i][4];
      U[i][1] = vmls_n_f32(vadd_f32(XTx[i][3], XTx[i][4]), vadd_f32(XTx[i][1], XTx[i][2]), 4.0f);

      // U[i][2] =  4*XTx[i][1] + -4*XTx[i][2] + -1*XTx[i][3] +  1*XTx[i][4];
      U[i][2] = vmla_n_f32(vsub_f32(XTx[i][4], XTx[i][3]), vsub_f32(XTx[i][1], XTx[i][2]), 4.0f);

      // U[i][3] = -2*XTx[i][1] + -1*XTx[i][2] +  2*XTx[i][3] +  1*XTx[i][4];
      U[i][3] = vmla_n_f32(vsub_f32(XTx[i][4], XTx[i][2]), vsub_f32(XTx[i][3], XTx[i][1]), 2.0f);

      // U[i][4] =  2*XTx[i][1] + -1*XTx[i][2] + -2*XTx[i][3] +  1*XTx[i][4];
      U[i][4] = vmla_n_f32(vsub_f32(XTx[i][4], XTx[i][2]), vsub_f32(XTx[i][1], XTx[i][3]), 2.0f);

      // U[i][5] =  4*XTx[i][1] + -5*XTx[i][3] +  1*XTx[i][5];
      U[i][5] = vmls_n_f32(vmla_n_f32(XTx[i][5], XTx[i][1], 4.0f), XTx[i][3], 5.0f);
    }

    // Store the transformed matrix
    for (int i = 0, m = 0; i < inner_tile_rows; i++)
    {
      for (int j = 0; j < inner_tile_cols; j++, m++)
      {
        vst1_f32(outptr + m*matrix_stride, U[i][j]);
      }
    }
    outptr += 2;
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
      XTx[0][j] =  4*x[0][j] + -5*x[2][j] +  1*x[4][j];
      XTx[1][j] = -4*x[1][j] + -4*x[2][j] +  1*x[3][j] +  1*x[4][j];
      XTx[2][j] =  4*x[1][j] + -4*x[2][j] + -1*x[3][j] +  1*x[4][j];
      XTx[3][j] = -2*x[1][j] + -1*x[2][j] +  2*x[3][j] +  1*x[4][j];
      XTx[4][j] =  2*x[1][j] + -1*x[2][j] + -2*x[3][j] +  1*x[4][j];
      XTx[5][j] =  4*x[1][j] + -5*x[3][j] +  1*x[5][j];
    }

    // Compute U = XT . x . X
    for (int i = 0; i < inner_tile_rows; i++)
    {
      U[i][0] =  4*XTx[i][0] + -5*XTx[i][2] +  1*XTx[i][4];
      U[i][1] = -4*XTx[i][1] + -4*XTx[i][2] +  1*XTx[i][3] +  1*XTx[i][4];
      U[i][2] =  4*XTx[i][1] + -4*XTx[i][2] + -1*XTx[i][3] +  1*XTx[i][4];
      U[i][3] = -2*XTx[i][1] + -1*XTx[i][2] +  2*XTx[i][3] +  1*XTx[i][4];
      U[i][4] =  2*XTx[i][1] + -1*XTx[i][2] + -2*XTx[i][3] +  1*XTx[i][4];
      U[i][5] =  4*XTx[i][1] + -5*XTx[i][3] +  1*XTx[i][5];
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

#endif // ! __aarch64__
