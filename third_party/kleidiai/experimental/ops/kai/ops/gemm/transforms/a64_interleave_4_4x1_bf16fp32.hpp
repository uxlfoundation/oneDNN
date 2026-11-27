//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

extern "C" void kai_ops_a64_interleave_4_4x1_bf16fp32(float *out, const bfloat16 *in, size_t width, size_t height, size_t in_stride, size_t num_blocks);


template<>
void Transform<12, 1, false, VLType::None>(
    float *out, const bfloat16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_4_4x1_bf16fp32(
        out,
        in + x0 * stride + k0,
        (kmax-k0),
        (xmax-x0),
        stride * sizeof(bfloat16),
        3
    );
}
