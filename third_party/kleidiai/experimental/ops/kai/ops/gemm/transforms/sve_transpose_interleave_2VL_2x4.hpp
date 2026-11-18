//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

extern "C" void kai_ops_sve_transpose_interleave_2VL_2x4(uint16_t *out, const uint16_t *in, size_t width, size_t in_stride, size_t height);


template<>
void Transform<2, 4, true, VLType::SVE>(
    bfloat16 *out, const bfloat16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sve_transpose_interleave_2VL_2x4(
        reinterpret_cast<uint16_t *>(out),
        reinterpret_cast<const uint16_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(bfloat16) / 2,
        stride * sizeof(bfloat16),
        (kmax-k0)
    );
}
