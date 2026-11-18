//
// SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

extern "C" void kai_ops_sme_interleave2VL_block4_s8_s8_summing(
  int8_t **out_ptr, const int8_t * const * in, size_t width, size_t height,
  size_t row_offset, bool first, int32_t multiplier
);

template<>
void interleave_block<2, 4, VLType::SME, true>(
  int8_t **out_ptr, const int8_t * const * in, size_t width, size_t height,
  size_t row_offset, bool first, int32_t multiplier
)
{
    kai_ops_sme_interleave2VL_block4_s8_s8_summing(out_ptr, in, width, height, row_offset, first, multiplier);
}

