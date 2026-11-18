//
// SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

extern "C" void kai_ops_sme_transpose_interleave_4VL(uint16_t *out, const uint16_t *in, size_t width, size_t in_stride, size_t height);


template<>
void Transform<4, 1, true, VLType::SME>(
    float *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_transpose_interleave_4VL(
        reinterpret_cast<uint16_t *>(out),
        reinterpret_cast<const uint16_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(float) / 2,
        stride * sizeof(float),
        (kmax-k0)
    );
}
