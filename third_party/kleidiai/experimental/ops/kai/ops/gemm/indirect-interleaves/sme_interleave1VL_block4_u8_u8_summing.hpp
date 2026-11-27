//
// SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

extern "C" void kai_ops_sme_interleave1VL_block4_u8_u8_summing(
  uint8_t **out_ptr, const uint8_t * const * in, size_t width, size_t height,
  size_t row_offset, bool first, int32_t multiplier
);

template<>
void interleave_block<1, 4, VLType::SME, true>(
  uint8_t **out_ptr, const uint8_t * const * in, size_t width, size_t height,
  size_t row_offset, bool first, int32_t multiplier
)
{
    kai_ops_sme_interleave1VL_block4_u8_u8_summing(out_ptr, in, width, height, row_offset, first, multiplier);
}

