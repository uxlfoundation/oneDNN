//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

extern "C" void kai_ops_a64_transpose_interleave_24_u8u16(uint16_t *out, const uint8_t *in, size_t width, size_t in_stride, size_t height);


template<>
void Transform<12, 1, true, VLType::None>(
    uint16_t *out, const uint8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_transpose_interleave_24_u8u16(
        out,
        in + k0 * stride + x0,
        (xmax-x0),
        stride * sizeof(uint8_t),
        (kmax-k0)
    );
}
