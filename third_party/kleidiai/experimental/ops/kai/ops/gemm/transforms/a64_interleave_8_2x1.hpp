//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

extern "C" void kai_ops_a64_interleave_8_2x1(__fp16 *out, const __fp16 *in, size_t width, size_t height, size_t in_stride, size_t num_blocks);


template<>
void Transform<8, 1, false, VLType::None>(
    __fp16 *out, const __fp16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_8_2x1(
        reinterpret_cast<__fp16 *>(out),
        reinterpret_cast<const __fp16 *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(__fp16) / 2,
        (xmax-x0),
        stride * sizeof(__fp16),
        1
    );
}

template<>
void Transform<1, 1, false, VLType::SVE>(
    __fp16 *out, const __fp16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_8_2x1(
        reinterpret_cast<__fp16 *>(out),
        reinterpret_cast<const __fp16 *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(__fp16) / 2,
        (xmax-x0),
        stride * sizeof(__fp16),
        get_vector_length<__fp16>() * 1 / 8
    );
}

template<>
void Transform<16, 1, false, VLType::None>(
    __fp16 *out, const __fp16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_8_2x1(
        reinterpret_cast<__fp16 *>(out),
        reinterpret_cast<const __fp16 *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(__fp16) / 2,
        (xmax-x0),
        stride * sizeof(__fp16),
        2
    );
}

template<>
void Transform<2, 1, false, VLType::SVE>(
    __fp16 *out, const __fp16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_8_2x1(
        reinterpret_cast<__fp16 *>(out),
        reinterpret_cast<const __fp16 *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(__fp16) / 2,
        (xmax-x0),
        stride * sizeof(__fp16),
        get_vector_length<__fp16>() * 2 / 8
    );
}

template<>
void Transform<24, 1, false, VLType::None>(
    __fp16 *out, const __fp16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_8_2x1(
        reinterpret_cast<__fp16 *>(out),
        reinterpret_cast<const __fp16 *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(__fp16) / 2,
        (xmax-x0),
        stride * sizeof(__fp16),
        3
    );
}

template<>
void Transform<3, 1, false, VLType::SVE>(
    __fp16 *out, const __fp16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_8_2x1(
        reinterpret_cast<__fp16 *>(out),
        reinterpret_cast<const __fp16 *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(__fp16) / 2,
        (xmax-x0),
        stride * sizeof(__fp16),
        get_vector_length<__fp16>() * 3 / 8
    );
}

template<>
void Transform<32, 1, false, VLType::None>(
    __fp16 *out, const __fp16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_8_2x1(
        reinterpret_cast<__fp16 *>(out),
        reinterpret_cast<const __fp16 *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(__fp16) / 2,
        (xmax-x0),
        stride * sizeof(__fp16),
        4
    );
}

template<>
void Transform<4, 1, false, VLType::SVE>(
    __fp16 *out, const __fp16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_8_2x1(
        reinterpret_cast<__fp16 *>(out),
        reinterpret_cast<const __fp16 *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(__fp16) / 2,
        (xmax-x0),
        stride * sizeof(__fp16),
        get_vector_length<__fp16>() * 4 / 8
    );
}
