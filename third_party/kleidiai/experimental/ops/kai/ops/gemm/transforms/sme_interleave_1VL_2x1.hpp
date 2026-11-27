//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

extern "C" void kai_ops_sme_interleave_1VL_2x1(__fp16 *out, const __fp16 *in, size_t height, size_t width, size_t in_stride, size_t num_blocks);


template<>
void Transform<1, 1, false, VLType::SME>(
    __fp16 *out, const __fp16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_interleave_1VL_2x1(
        reinterpret_cast<__fp16 *>(out),
        reinterpret_cast<const __fp16 *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(__fp16) / 2,
        (xmax-x0),
        stride * sizeof(__fp16),
        1
    );
}

template<>
void Transform<2, 1, false, VLType::SME>(
    __fp16 *out, const __fp16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_interleave_1VL_2x1(
        reinterpret_cast<__fp16 *>(out),
        reinterpret_cast<const __fp16 *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(__fp16) / 2,
        (xmax-x0),
        stride * sizeof(__fp16),
        2
    );
}

template<>
void Transform<8, 1, false, VLType::SME>(
    __fp16 *out, const __fp16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_interleave_1VL_2x1(
        reinterpret_cast<__fp16 *>(out),
        reinterpret_cast<const __fp16 *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(__fp16) / 2,
        (xmax-x0),
        stride * sizeof(__fp16),
        8
    );
}

template<>
void Transform<16, 1, false, VLType::SME>(
    __fp16 *out, const __fp16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_interleave_1VL_2x1(
        reinterpret_cast<__fp16 *>(out),
        reinterpret_cast<const __fp16 *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(__fp16) / 2,
        (xmax-x0),
        stride * sizeof(__fp16),
        16
    );
}

template<>
void Transform<1, 1, false, VLType::SME>(
    bfloat16 *out, const bfloat16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_interleave_1VL_2x1(
        reinterpret_cast<__fp16 *>(out),
        reinterpret_cast<const __fp16 *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(bfloat16) / 2,
        (xmax-x0),
        stride * sizeof(bfloat16),
        1
    );
}

template<>
void Transform<2, 1, false, VLType::SME>(
    bfloat16 *out, const bfloat16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_interleave_1VL_2x1(
        reinterpret_cast<__fp16 *>(out),
        reinterpret_cast<const __fp16 *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(bfloat16) / 2,
        (xmax-x0),
        stride * sizeof(bfloat16),
        2
    );
}

template<>
void Transform<16, 1, false, VLType::SME>(
    bfloat16 *out, const bfloat16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_interleave_1VL_2x1(
        reinterpret_cast<__fp16 *>(out),
        reinterpret_cast<const __fp16 *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(bfloat16) / 2,
        (xmax-x0),
        stride * sizeof(bfloat16),
        16
    );
}
