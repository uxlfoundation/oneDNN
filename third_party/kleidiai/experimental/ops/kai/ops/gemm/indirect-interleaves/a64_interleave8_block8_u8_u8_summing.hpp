//
// SPDX-FileCopyrightText: Copyright 2019-2021, 2023-2024, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

extern "C" void kai_ops_a64_interleave8_block8_u8_u8_summing(
  uint8_t **out_ptr, const uint8_t * const * in, size_t width, size_t height,
  size_t row_offset, bool first, int32_t multiplier
);

template<>
void interleave_block<8, 8, VLType::None, true>(
  uint8_t **out_ptr, const uint8_t * const * in, size_t width, size_t height,
  size_t row_offset, bool first, int32_t multiplier
)
{
    kai_ops_a64_interleave8_block8_u8_u8_summing(out_ptr, in, width, height, row_offset, first, multiplier);
}

