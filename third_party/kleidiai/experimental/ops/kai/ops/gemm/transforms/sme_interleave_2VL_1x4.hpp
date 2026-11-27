//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

extern "C" void kai_ops_sme_interleave_2VL_1x4(uint8_t *out, const uint8_t *in, size_t height, size_t width, size_t in_stride, size_t num_blocks);


template<>
void Transform<2, 1, false, VLType::SME>(
    float *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_interleave_2VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(float) / 1,
        (xmax-x0),
        stride * sizeof(float),
        1
    );
}

template<>
void Transform<4, 1, false, VLType::SME>(
    float *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_interleave_2VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(float) / 1,
        (xmax-x0),
        stride * sizeof(float),
        2
    );
}

template<>
void Transform<8, 1, false, VLType::SME>(
    float *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_interleave_2VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(float) / 1,
        (xmax-x0),
        stride * sizeof(float),
        4
    );
}

template<>
void Transform<16, 1, false, VLType::SME>(
    float *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_interleave_2VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(float) / 1,
        (xmax-x0),
        stride * sizeof(float),
        8
    );
}

template<>
void Transform<2, 2, false, VLType::SME>(
    __fp16 *out, const __fp16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_interleave_2VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(__fp16) / 1,
        (xmax-x0),
        stride * sizeof(__fp16),
        1
    );
}

template<>
void Transform<4, 2, false, VLType::SME>(
    __fp16 *out, const __fp16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_interleave_2VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(__fp16) / 1,
        (xmax-x0),
        stride * sizeof(__fp16),
        2
    );
}

template<>
void Transform<16, 2, false, VLType::SME>(
    __fp16 *out, const __fp16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_interleave_2VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(__fp16) / 1,
        (xmax-x0),
        stride * sizeof(__fp16),
        8
    );
}

template<>
void Transform<2, 2, false, VLType::SME>(
    bfloat16 *out, const bfloat16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_interleave_2VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(bfloat16) / 1,
        (xmax-x0),
        stride * sizeof(bfloat16),
        1
    );
}

template<>
void Transform<4, 2, false, VLType::SME>(
    bfloat16 *out, const bfloat16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_interleave_2VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(bfloat16) / 1,
        (xmax-x0),
        stride * sizeof(bfloat16),
        2
    );
}

template<>
void Transform<8, 2, false, VLType::SME>(
    bfloat16 *out, const bfloat16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_interleave_2VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(bfloat16) / 1,
        (xmax-x0),
        stride * sizeof(bfloat16),
        4
    );
}

template<>
void Transform<16, 2, false, VLType::SME>(
    bfloat16 *out, const bfloat16 *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_interleave_2VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(bfloat16) / 1,
        (xmax-x0),
        stride * sizeof(bfloat16),
        8
    );
}

template<>
void Transform<2, 4, false, VLType::SME>(
    int8_t *out, const int8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_interleave_2VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(int8_t) / 1,
        (xmax-x0),
        stride * sizeof(int8_t),
        1
    );
}

template<>
void Transform<4, 4, false, VLType::SME>(
    int8_t *out, const int8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_interleave_2VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(int8_t) / 1,
        (xmax-x0),
        stride * sizeof(int8_t),
        2
    );
}

template<>
void Transform<8, 4, false, VLType::SME>(
    int8_t *out, const int8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_interleave_2VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(int8_t) / 1,
        (xmax-x0),
        stride * sizeof(int8_t),
        4
    );
}

template<>
void Transform<16, 4, false, VLType::SME>(
    int8_t *out, const int8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_interleave_2VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(int8_t) / 1,
        (xmax-x0),
        stride * sizeof(int8_t),
        8
    );
}

template<>
void Transform<2, 4, false, VLType::SME>(
    uint8_t *out, const uint8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_interleave_2VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(uint8_t) / 1,
        (xmax-x0),
        stride * sizeof(uint8_t),
        1
    );
}

template<>
void Transform<4, 4, false, VLType::SME>(
    uint8_t *out, const uint8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_interleave_2VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(uint8_t) / 1,
        (xmax-x0),
        stride * sizeof(uint8_t),
        2
    );
}

template<>
void Transform<8, 4, false, VLType::SME>(
    uint8_t *out, const uint8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_interleave_2VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(uint8_t) / 1,
        (xmax-x0),
        stride * sizeof(uint8_t),
        4
    );
}

template<>
void Transform<16, 4, false, VLType::SME>(
    uint8_t *out, const uint8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_sme_interleave_2VL_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + x0 * stride + k0),
        (kmax-k0) * sizeof(uint8_t) / 1,
        (xmax-x0),
        stride * sizeof(uint8_t),
        8
    );
}
