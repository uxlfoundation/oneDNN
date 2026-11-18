//
// SPDX-FileCopyrightText: Copyright 2021, 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

extern "C" void kai_ops_a64_transpose_interleave_24(uint16_t *out, const uint16_t *in, size_t width, size_t in_stride, size_t height);


template<>
void Transform<12, 1, true, VLType::None>(
    int16_t *out, const int16_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_transpose_interleave_24(
        reinterpret_cast<uint16_t *>(out),
        reinterpret_cast<const uint16_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(int16_t) / 2,
        stride * sizeof(int16_t),
        (kmax-k0)
    );
}

template<>
void Transform<12, 1, true, VLType::None>(
    uint16_t *out, const uint16_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_transpose_interleave_24(
        reinterpret_cast<uint16_t *>(out),
        reinterpret_cast<const uint16_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(uint16_t) / 2,
        stride * sizeof(uint16_t),
        (kmax-k0)
    );
}
