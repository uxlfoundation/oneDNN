//
// SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

extern "C" void kai_ops_sme_interleave2VL_fp32_fp32(
  float **out_ptr, const float * const * in, size_t width, size_t height,
  size_t row_offset, bool first, int32_t multiplier
);

template<>
void interleave_block<2, 1, VLType::SME, false>(
  float **out_ptr, const float * const * in, size_t width, size_t height,
  size_t row_offset, bool first, int32_t multiplier
)
{
    kai_ops_sme_interleave2VL_fp32_fp32(out_ptr, in, width, height, row_offset, first, multiplier);
}

