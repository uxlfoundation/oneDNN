//
// SPDX-FileCopyrightText: Copyright 2022, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>

namespace kai {
namespace ops {
namespace winograd {
namespace weight_transform {

void cpp_fp32_1x2_1x7(
  unsigned int n_channels,
  const float* inptr, size_t, size_t ld_weight_col,
  float *outptr, size_t matrix_stride
)
{
  for (; n_channels; n_channels--)
  {
    // Matrices used and computed in this kernel
    float w[7], V[8];

    // Read weights
    for (int j = 0; j < 7; j++)
    {
      w[j] = *(inptr + j*ld_weight_col);
    }

    // Compute V = w WT
    V[0] = (w[0]*-1) / 36.0f;
    V[1] = (w[1]*-1 + w[3]*-1 + w[5]*-1 + w[0]*1 + w[2]*1 + w[4]*1 + w[6]*1) / 48.0f;
    V[2] = (w[0]*1 + w[1]*1 + w[2]*1 + w[3]*1 + w[4]*1 + w[5]*1 + w[6]*1) / 48.0f;
    V[3] = (w[0]*-1 + w[6]*-64 + w[4]*-16 + w[2]*-4 + w[1]*2 + w[3]*8 + w[5]*32) / 120.0f;
    V[4] = (w[0]*-1 + w[6]*-64 + w[5]*-32 + w[4]*-16 + w[3]*-8 + w[2]*-4 + w[1]*-2) / 120.0f;
    V[5] = (w[5]*-243 + w[3]*-27 + w[1]*-3 + w[2]*9 + w[4]*81 + w[6]*729 + w[0]*1) / 720.0f;
    V[6] = (w[1]*3 + w[2]*9 + w[3]*27 + w[4]*81 + w[5]*243 + w[6]*729 + w[0]*1) / 720.0f;
    V[7] = (w[6]*1) / 1.0f;

    // Store the transformed weights
    for (int j = 0; j < 8; j++)
    {
      *(outptr + j*matrix_stride) = V[j];
    }

    inptr++;
    outptr++;
  }
}

}  // namespace output_transform
}  // namespace winograd
}  // namespace ops
}  // namespace kai
