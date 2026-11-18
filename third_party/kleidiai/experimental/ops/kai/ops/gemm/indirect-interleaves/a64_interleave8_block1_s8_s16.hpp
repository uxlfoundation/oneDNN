//
// SPDX-FileCopyrightText: Copyright 2019-2021, 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

extern "C" void kai_ops_a64_interleave8_block1_s8_s16(
  int16_t **out_ptr, const int8_t * const * in, size_t width, size_t height,
  size_t row_offset, bool first, int32_t multiplier
);

template<>
void interleave_block<8, 1, VLType::None, false>(
  int16_t **out_ptr, const int8_t * const * in, size_t width, size_t height,
  size_t row_offset, bool first, int32_t multiplier
)
{
    kai_ops_a64_interleave8_block1_s8_s16(out_ptr, in, width, height, row_offset, first, multiplier);
}

