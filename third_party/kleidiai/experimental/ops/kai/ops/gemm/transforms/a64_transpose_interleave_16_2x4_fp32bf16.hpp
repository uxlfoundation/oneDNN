//
// SPDX-FileCopyrightText: Copyright 2021, 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

extern "C" void kai_ops_a64_transpose_interleave_16_2x4_fp32bf16(bfloat16 *out, const float *in, size_t width, size_t in_stride, size_t height);


template<>
void Transform<16, 4, true, VLType::None>(
    bfloat16 *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_transpose_interleave_16_2x4_fp32bf16(
        out,
        in + k0 * stride + x0,
        (xmax-x0),
        stride * sizeof(float),
        (kmax-k0)
    );
}
