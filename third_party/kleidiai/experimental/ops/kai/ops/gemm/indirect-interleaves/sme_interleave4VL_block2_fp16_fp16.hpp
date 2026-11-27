//
// SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

extern "C" void kai_ops_sme_interleave4VL_block2_fp16_fp16(
  __fp16 **out_ptr, const __fp16 * const * in, size_t width, size_t height,
  size_t row_offset, bool first, int32_t multiplier
);

template<>
void interleave_block<4, 2, VLType::SME, false>(
  __fp16 **out_ptr, const __fp16 * const * in, size_t width, size_t height,
  size_t row_offset, bool first, int32_t multiplier
)
{
    kai_ops_sme_interleave4VL_block2_fp16_fp16(out_ptr, in, width, height, row_offset, first, multiplier);
}

