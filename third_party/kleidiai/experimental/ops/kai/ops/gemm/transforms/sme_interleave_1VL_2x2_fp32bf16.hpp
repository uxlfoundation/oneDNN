//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

extern "C" void kai_ops_sme_interleave_1VL_2x2_fp32bf16(bfloat16 *out, const float *in, size_t height, size_t width, size_t in_stride, size_t num_blocks);


template<>
void Transform<1, 2, false, VLType::SME>(
    bfloat16 *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_interleave_1VL_2x2_fp32bf16(
        out,
        in + x0 * stride + k0,
        (kmax-k0),
        (xmax-x0),
        stride * sizeof(float),
        1
    );
}
