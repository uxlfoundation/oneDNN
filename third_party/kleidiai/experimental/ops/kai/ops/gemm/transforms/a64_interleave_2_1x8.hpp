//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

extern "C" void kai_ops_a64_interleave_2_1x8(uint8_t *out, const uint8_t *in, size_t width, size_t height, size_t in_stride, size_t num_blocks);


template<>
void Transform<12, 8, false, VLType::None>(
    uint8_t *out, const uint8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_2_1x8(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(uint8_t) / 1,
        (xmax-x0),
        stride * sizeof(uint8_t),
        6
    );
}

template<>
void Transform<6, 8, false, VLType::SVE>(
    uint8_t *out, const uint8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_2_1x8(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(uint8_t) / 1,
        (xmax-x0),
        stride * sizeof(uint8_t),
        get_vector_length<uint8_t>() * 6 / 16
    );
}

template<>
void Transform<16, 8, false, VLType::None>(
    uint8_t *out, const uint8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_2_1x8(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(uint8_t) / 1,
        (xmax-x0),
        stride * sizeof(uint8_t),
        8
    );
}

template<>
void Transform<8, 8, false, VLType::SVE>(
    uint8_t *out, const uint8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_2_1x8(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(uint8_t) / 1,
        (xmax-x0),
        stride * sizeof(uint8_t),
        get_vector_length<uint8_t>() * 8 / 16
    );
}

template<>
void Transform<12, 8, false, VLType::None>(
    int8_t *out, const int8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_2_1x8(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(int8_t) / 1,
        (xmax-x0),
        stride * sizeof(int8_t),
        6
    );
}

template<>
void Transform<6, 8, false, VLType::SVE>(
    int8_t *out, const int8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_2_1x8(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(int8_t) / 1,
        (xmax-x0),
        stride * sizeof(int8_t),
        get_vector_length<int8_t>() * 6 / 16
    );
}

template<>
void Transform<16, 8, false, VLType::None>(
    int8_t *out, const int8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_2_1x8(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(int8_t) / 1,
        (xmax-x0),
        stride * sizeof(int8_t),
        8
    );
}

template<>
void Transform<8, 8, false, VLType::SVE>(
    int8_t *out, const int8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_2_1x8(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(int8_t) / 1,
        (xmax-x0),
        stride * sizeof(int8_t),
        get_vector_length<int8_t>() * 8 / 16
    );
}

template<>
void Transform<4, 4, false, VLType::None>(
    bfloat16 *out, const bfloat16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_2_1x8(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(bfloat16) / 1,
        (xmax-x0),
        stride * sizeof(bfloat16),
        2
    );
}

template<>
void Transform<2, 4, false, VLType::SVE>(
    bfloat16 *out, const bfloat16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_2_1x8(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(bfloat16) / 1,
        (xmax-x0),
        stride * sizeof(bfloat16),
        get_vector_length<bfloat16>() * 2 / 8
    );
}

template<>
void Transform<12, 4, false, VLType::None>(
    bfloat16 *out, const bfloat16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_2_1x8(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(bfloat16) / 1,
        (xmax-x0),
        stride * sizeof(bfloat16),
        6
    );
}

template<>
void Transform<6, 4, false, VLType::SVE>(
    bfloat16 *out, const bfloat16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_2_1x8(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(bfloat16) / 1,
        (xmax-x0),
        stride * sizeof(bfloat16),
        get_vector_length<bfloat16>() * 6 / 8
    );
}

template<>
void Transform<16, 4, false, VLType::None>(
    bfloat16 *out, const bfloat16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_2_1x8(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(bfloat16) / 1,
        (xmax-x0),
        stride * sizeof(bfloat16),
        8
    );
}

template<>
void Transform<8, 4, false, VLType::SVE>(
    bfloat16 *out, const bfloat16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_2_1x8(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(bfloat16) / 1,
        (xmax-x0),
        stride * sizeof(bfloat16),
        get_vector_length<bfloat16>() * 8 / 8
    );
}

template<>
void Transform<6, 2, false, VLType::SVE>(
    float *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_2_1x8(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(float) / 1,
        (xmax-x0),
        stride * sizeof(float),
        get_vector_length<float>() * 6 / 4
    );
}
