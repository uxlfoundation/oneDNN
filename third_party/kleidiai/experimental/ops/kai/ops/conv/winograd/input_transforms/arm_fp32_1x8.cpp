//
// SPDX-FileCopyrightText: Copyright 2022-2024, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <arm_neon.h>

namespace kai {
namespace ops {
namespace winograd {
namespace input_transform {

void arm_fp32_1x8(
  const unsigned int n_channels,
  const float *const input_base,
  size_t,  // We don't need to stride over rows
  const size_t input_col_stride,
  float *outptr,
  const size_t matrix_stride
)
{
  constexpr int inner_tile_cols = 8;

  // Get pointers into the input tile
  const float *x_ptrs[inner_tile_cols];
  for (int j = 0, xj = 0; j < inner_tile_cols; j++, xj++)
  {
    x_ptrs[j] = input_base + xj*input_col_stride;
  }

  // Vectors used/computed in this kernel.
  float x[inner_tile_cols];
  float U[inner_tile_cols];

  for (int j = 0; j < inner_tile_cols; j++)
  {
    x[j] = 0.0f;
  }

  // Perform the Winograd input transformation for each channel in the input
  // tensor.
  int channels_remaining = n_channels;
  for (; channels_remaining >= 4; channels_remaining -= 4)
  {
    float32x4_t x[inner_tile_cols], U[inner_tile_cols];
    for (int j = 0; j < inner_tile_cols; j++)
    {
      x[j] = vdupq_n_f32(0.0f);
    }

    // Load x
    for (int j = 0; j < inner_tile_cols; j++)
    {
      x[j] = vld1q_f32(x_ptrs[j]);
      x_ptrs[j] += 4;
    }

    // Compute U = x . X
    U[0] = vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(x[6], 1), x[2], 49), x[4], -14), x[0], -36);
    U[1] = vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(x[6], 1), x[2], 36), x[3], 13), x[4], -13), x[1], -36), x[5], -1);
    U[2] = vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(x[6], 1), x[5], 1), x[2], 36), x[1], 36), x[4], -13), x[3], -13);
    U[3] = vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(x[6], 1), x[3], 20), x[2], 9), x[5], -2), x[4], -10), x[1], -18);
    U[4] = vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(x[6], 1), x[1], 18), x[2], 9), x[5], 2), x[4], -10), x[3], -20);
    U[5] = vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(x[6], 1), x[3], 15), x[2], 4), x[5], -3), x[4], -5), x[1], -12);
    U[6] = vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(x[6], 1), x[1], 12), x[2], 4), x[5], 3), x[4], -5), x[3], -15);
    U[7] = vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(x[7], 1), x[3], 49), x[5], -14), x[1], -36);

    // Store the transformed vector
    for (int j = 0; j < inner_tile_cols; j++)
    {
      vst1q_f32(outptr + j*matrix_stride, U[j]);
    }
    outptr += 4;
  }
  for (; channels_remaining >= 2; channels_remaining -= 2)
  {
    float32x2_t x[inner_tile_cols], U[inner_tile_cols];
    for (int j = 0; j < inner_tile_cols; j++)
    {
      x[j] = vdup_n_f32(0.0f);
    }

    // Load x
    for (int j = 0; j < inner_tile_cols; j++)
    {
      x[j] = vld1_f32(x_ptrs[j]);
      x_ptrs[j] += 2;
    }

    // Compute U = x . X
    U[0] = vmla_n_f32(vmla_n_f32(vmla_n_f32(vmul_n_f32(x[6], 1), x[2], 49), x[4], -14), x[0], -36);
    U[1] = vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmul_n_f32(x[6], 1), x[2], 36), x[3], 13), x[4], -13), x[1], -36), x[5], -1);
    U[2] = vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmul_n_f32(x[6], 1), x[5], 1), x[2], 36), x[1], 36), x[4], -13), x[3], -13);
    U[3] = vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmul_n_f32(x[6], 1), x[3], 20), x[2], 9), x[5], -2), x[4], -10), x[1], -18);
    U[4] = vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmul_n_f32(x[6], 1), x[1], 18), x[2], 9), x[5], 2), x[4], -10), x[3], -20);
    U[5] = vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmul_n_f32(x[6], 1), x[3], 15), x[2], 4), x[5], -3), x[4], -5), x[1], -12);
    U[6] = vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmul_n_f32(x[6], 1), x[1], 12), x[2], 4), x[5], 3), x[4], -5), x[3], -15);
    U[7] = vmla_n_f32(vmla_n_f32(vmla_n_f32(vmul_n_f32(x[7], 1), x[3], 49), x[5], -14), x[1], -36);

    // Store the transformed vector
    for (int j = 0; j < inner_tile_cols; j++)
    {
      vst1_f32(outptr + j*matrix_stride, U[j]);
    }
    outptr += 2;
  }
  for (; channels_remaining; channels_remaining--)
  {
    // Load x
    for (int j = 0; j < inner_tile_cols; j++)
    {
      x[j] = *(x_ptrs[j]++);
    }

    // Compute U = x . X
    U[0] = x[0]*-36 + x[4]*-14 + x[2]*49 + x[6]*1;
    U[1] = x[5]*-1 + x[1]*-36 + x[4]*-13 + x[3]*13 + x[2]*36 + x[6]*1;
    U[2] = x[3]*-13 + x[4]*-13 + x[1]*36 + x[2]*36 + x[5]*1 + x[6]*1;
    U[3] = x[1]*-18 + x[4]*-10 + x[5]*-2 + x[2]*9 + x[3]*20 + x[6]*1;
    U[4] = x[3]*-20 + x[4]*-10 + x[5]*2 + x[2]*9 + x[1]*18 + x[6]*1;
    U[5] = x[1]*-12 + x[4]*-5 + x[5]*-3 + x[2]*4 + x[3]*15 + x[6]*1;
    U[6] = x[3]*-15 + x[4]*-5 + x[5]*3 + x[2]*4 + x[1]*12 + x[6]*1;
    U[7] = x[1]*-36 + x[5]*-14 + x[3]*49 + x[7]*1;

    // Store the transformed vector
    for (int j = 0; j < inner_tile_cols; j++)
    {
      *(outptr + j*matrix_stride) = U[j];
    }
    outptr++;
  }
}

}  // namespace input_transform
}  // namespace winograd
}  // namespace ops
}  // namespace kai
