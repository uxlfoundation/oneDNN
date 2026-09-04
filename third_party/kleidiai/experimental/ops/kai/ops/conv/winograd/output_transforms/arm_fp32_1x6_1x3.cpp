//
// SPDX-FileCopyrightText: Copyright 2022-2024, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

void arm_fp32_1x6_1x3(
  unsigned int n_channels,
  const float* inptr,
  const size_t matrix_stride,
  const float* bptr,
  float *outptr,
  size_t,  // No need to stride across rows
  const size_t output_col_stride,
  const float output_min,
  const float output_max
)
{
  constexpr unsigned int inner_tile_cols = 8, output_tile_cols = 6;

  // For each channel of the output
  for (; n_channels >= 4; n_channels -= 4)
  {
    // Matrices used and computed during this transform
    float32x4_t F[inner_tile_cols], f[output_tile_cols], b = vdupq_n_f32(0.0f);

    // Read a 1x8 tile in the Winograd domain
    for (auto j = 0u; j < inner_tile_cols; j++)
    {
      F[j] = vld1q_f32(inptr + j*matrix_stride);
    }
    inptr += 4;

    f[0] = vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(F[6], 1), F[5], 1), F[4], 1), F[3], 1), F[2], 1), F[1], 1), F[0], 1);
    f[1] = vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(F[2], 1), F[6], 3), F[4], 2), F[3], -2), F[5], -3), F[1], -1);
    f[2] = vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(F[2], 1), F[1], 1), F[6], 9), F[5], 9), F[4], 4), F[3], 4);
    f[3] = vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(F[2], 1), F[6], 27), F[4], 8), F[3], -8), F[5], -27), F[1], -1);
    f[4] = vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(F[2], 1), F[1], 1), F[6], 81), F[5], 81), F[4], 16), F[3], 16);
    f[5] = vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmlaq_n_f32(vmulq_n_f32(F[7], 1), F[2], 1), F[6], 243), F[4], 32), F[3], -32), F[5], -243), F[1], -1);

    // Write out the output tile
    if (bptr != 0)
    {
      b = vld1q_f32(bptr);
      bptr += 4;
    }
    for (auto j = 0u; j < output_tile_cols; j++)
    {
      const auto y = vminq_f32(vmaxq_f32(f[j] + b, vdupq_n_f32(output_min)),
                               vdupq_n_f32(output_max));
      vst1q_f32(outptr + j*output_col_stride, y);
    }
    outptr += 4;
  }
  for (; n_channels >= 2; n_channels -= 2)
  {
    // Matrices used and computed during this transform
    float32x2_t F[inner_tile_cols], f[output_tile_cols], b = vdup_n_f32(0.0f);

    // Read a 1x8 tile in the Winograd domain
    for (auto j = 0u; j < inner_tile_cols; j++)
    {
      F[j] = vld1_f32(inptr + j*matrix_stride);
    }
    inptr += 2;

    f[0] = vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmul_n_f32(F[6], 1), F[5], 1), F[4], 1), F[3], 1), F[2], 1), F[1], 1), F[0], 1);
    f[1] = vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmul_n_f32(F[2], 1), F[6], 3), F[4], 2), F[3], -2), F[5], -3), F[1], -1);
    f[2] = vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmul_n_f32(F[2], 1), F[1], 1), F[6], 9), F[5], 9), F[4], 4), F[3], 4);
    f[3] = vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmul_n_f32(F[2], 1), F[6], 27), F[4], 8), F[3], -8), F[5], -27), F[1], -1);
    f[4] = vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmul_n_f32(F[2], 1), F[1], 1), F[6], 81), F[5], 81), F[4], 16), F[3], 16);
    f[5] = vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmla_n_f32(vmul_n_f32(F[7], 1), F[2], 1), F[6], 243), F[4], 32), F[3], -32), F[5], -243), F[1], -1);

    // Write out the output tile
    if (bptr != 0)
    {
      b = vld1_f32(bptr);
      bptr += 2;
    }
    for (auto j = 0u; j < output_tile_cols; j++)
    {
      const auto y = vmin_f32(vmax_f32(f[j] + b, vdup_n_f32(output_min)),
                              vdup_n_f32(output_max));
      vst1_f32(outptr + j*output_col_stride, y);
    }
    outptr += 2;
  }
  for (; n_channels; n_channels--)
  {
    // Matrices used and computed during this transform
    float F[inner_tile_cols], f[output_tile_cols], b = 0.0f;

    // Read a 1x8 tile in the Winograd domain
    for (auto j = 0u; j < inner_tile_cols; j++)
    {
      F[j] = *(inptr + j*matrix_stride);
    }
    inptr++;

    f[0] = F[0]*1 + F[1]*1 + F[2]*1 + F[3]*1 + F[4]*1 + F[5]*1 + F[6]*1;
    f[1] = F[1]*-1 + F[5]*-3 + F[3]*-2 + F[4]*2 + F[6]*3 + F[2]*1;
    f[2] = F[3]*4 + F[4]*4 + F[5]*9 + F[6]*9 + F[1]*1 + F[2]*1;
    f[3] = F[1]*-1 + F[5]*-27 + F[3]*-8 + F[4]*8 + F[6]*27 + F[2]*1;
    f[4] = F[3]*16 + F[4]*16 + F[5]*81 + F[6]*81 + F[1]*1 + F[2]*1;
    f[5] = F[1]*-1 + F[5]*-243 + F[3]*-32 + F[4]*32 + F[6]*243 + F[2]*1 + F[7]*1;

    // Write out the output tile
    if (bptr != 0)
    {
      b = *(bptr++);
    }
    for (auto j = 0u; j < output_tile_cols; j++)
    {
      *(outptr + j*output_col_stride) = std::max(std::min(f[j] + b, output_max), output_min);
    }
    outptr++;
  }
}

}  // namespace output_transform
}  // namespace winograd
}  // namespace ops
}  // namespace kai
