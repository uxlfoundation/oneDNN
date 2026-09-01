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
namespace input_transform {

void arm_fp32_4x4(
  const unsigned int n_channels,
  const float *input_base,
  const size_t input_row_stride,
  const size_t input_col_stride,
  float *outptr,
  const size_t matrix_stride
)
{
  constexpr int inner_tile_rows = 4, inner_tile_cols = 4;

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
  for (; channels_remaining >= 4; channels_remaining -= 4)
  {
    // Matrices used/computed in this kernel.
    float32x4_t x[inner_tile_rows][inner_tile_cols];
    float32x4_t XTx[inner_tile_rows][inner_tile_cols];
    float32x4_t U[inner_tile_rows][inner_tile_cols];

    for (int i = 0; i < inner_tile_rows; i++)
    {
      for (int j = 0; j < inner_tile_cols; j++)
      {
        x[i][j] = vdupq_n_f32(0.0f);
        XTx[i][j] = vdupq_n_f32(0.0f);
      }
    }

    // Load x
    for (int i = 0; i < inner_tile_rows; i++)
    {
      for (int j = 0; j < inner_tile_cols; j++)
      {
        x[i][j] = vld1q_f32(x_ptrs[i][j]);
        x_ptrs[i][j] += 4;
      }
    }

    // Compute XT . x
    for (int j = 0; j < inner_tile_cols; j++)
    {
      // XTx[0][j] = x[0][j] - x[2][j];
      XTx[0][j] = vsubq_f32(x[0][j], x[2][j]);

      // XTx[1][j] = x[1][j] + x[2][j];
      XTx[1][j] = vaddq_f32(x[1][j], x[2][j]);

      // XTx[2][j] = x[2][j] - x[1][j];
      XTx[2][j] = vsubq_f32(x[2][j], x[1][j]);

      // XTx[3][j] = x[1][j] - x[3][j];
      XTx[3][j] = vsubq_f32(x[1][j], x[3][j]);
    }

    // Compute U = XT . x . X
    for (int i = 0; i < inner_tile_rows; i++)
    {
      // U[i][0] = XTx[i][0] - XTx[i][2];
      U[i][0] = vsubq_f32(XTx[i][0], XTx[i][2]);

      // U[i][1] = XTx[i][1] + XTx[i][2];
      U[i][1] = vaddq_f32(XTx[i][1], XTx[i][2]);

      // U[i][2] = XTx[i][2] - XTx[i][1];
      U[i][2] = vsubq_f32(XTx[i][2], XTx[i][1]);

      // U[i][3] = XTx[i][1] - XTx[i][3];
      U[i][3] = vsubq_f32(XTx[i][1], XTx[i][3]);
    }

    // Store the transformed matrix
    for (int i = 0, m = 0; i < inner_tile_rows; i++)
    {
      for (int j = 0; j < inner_tile_cols; j++, m++)
      {
        vst1q_f32(outptr + m*matrix_stride, U[i][j]);
      }
    }
    outptr += 4;
  }
  for (; channels_remaining >= 2; channels_remaining -= 2)
  {
    // Matrices used/computed in this kernel.
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

    // Load x
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
      // XTx[0][j] = x[0][j] - x[2][j];
      XTx[0][j] = vsub_f32(x[0][j], x[2][j]);

      // XTx[1][j] = x[1][j] + x[2][j];
      XTx[1][j] = vadd_f32(x[1][j], x[2][j]);

      // XTx[2][j] = x[2][j] - x[1][j];
      XTx[2][j] = vsub_f32(x[2][j], x[1][j]);

      // XTx[3][j] = x[1][j] - x[3][j];
      XTx[3][j] = vsub_f32(x[1][j], x[3][j]);
    }

    // Compute U = XT . x . X
    for (int i = 0; i < inner_tile_rows; i++)
    {
      // U[i][0] = XTx[i][0] - XTx[i][2];
      U[i][0] = vsub_f32(XTx[i][0], XTx[i][2]);

      // U[i][1] = XTx[i][1] + XTx[i][2];
      U[i][1] = vadd_f32(XTx[i][1], XTx[i][2]);

      // U[i][2] = XTx[i][2] - XTx[i][1];
      U[i][2] = vsub_f32(XTx[i][2], XTx[i][1]);

      // U[i][3] = XTx[i][1] - XTx[i][3];
      U[i][3] = vsub_f32(XTx[i][1], XTx[i][3]);
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
      XTx[0][j] = x[0][j] - x[2][j];
      XTx[1][j] = x[1][j] + x[2][j];
      XTx[2][j] = x[2][j] - x[1][j];
      XTx[3][j] = x[1][j] - x[3][j];
    }

    // Compute U = XT . x . X
    for (int i = 0; i < inner_tile_rows; i++)
    {
      U[i][0] = XTx[i][0] - XTx[i][2];
      U[i][1] = XTx[i][1] + XTx[i][2];
      U[i][2] = XTx[i][2] - XTx[i][1];
      U[i][3] = XTx[i][1] - XTx[i][3];
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
