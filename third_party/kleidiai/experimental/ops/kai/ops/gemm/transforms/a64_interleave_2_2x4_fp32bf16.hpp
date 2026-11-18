//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off

#pragma once

extern "C" void kai_ops_a64_interleave_2_2x4_fp32bf16(bfloat16 *out, const float *in, size_t width, size_t height, size_t in_stride, size_t num_blocks);


template<>
void Transform<4, 4, false, VLType::None>(
    bfloat16 *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_2_2x4_fp32bf16(
        out,
        in + x0 * stride + k0,
        (kmax-k0),
        (xmax-x0),
        stride * sizeof(float),
        2
    );
}

template<>
void Transform<12, 4, false, VLType::None>(
    bfloat16 *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_2_2x4_fp32bf16(
        out,
        in + x0 * stride + k0,
        (kmax-k0),
        (xmax-x0),
        stride * sizeof(float),
        6
    );
}

template<>
void Transform<16, 4, false, VLType::None>(
    bfloat16 *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_2_2x4_fp32bf16(
        out,
        in + x0 * stride + k0,
        (kmax-k0),
        (xmax-x0),
        stride * sizeof(float),
        8
    );
}

template<>
void Transform<24, 4, false, VLType::None>(
    bfloat16 *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_2_2x4_fp32bf16(
        out,
        in + x0 * stride + k0,
        (kmax-k0),
        (xmax-x0),
        stride * sizeof(float),
        12
    );
}

template<>
void Transform<2, 4, false, VLType::SVE>(
    bfloat16 *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_2_2x4_fp32bf16(
        out,
        in + x0 * stride + k0,
        (kmax-k0),
        (xmax-x0),
        stride * sizeof(float),
        get_vector_length<bfloat16>() * 2 / 8
    );
}

template<>
void Transform<6, 4, false, VLType::SVE>(
    bfloat16 *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_2_2x4_fp32bf16(
        out,
        in + x0 * stride + k0,
        (kmax-k0),
        (xmax-x0),
        stride * sizeof(float),
        get_vector_length<bfloat16>() * 6 / 8
    );
}

template<>
void Transform<8, 4, false, VLType::SVE>(
    bfloat16 *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_2_2x4_fp32bf16(
        out,
        in + x0 * stride + k0,
        (kmax-k0),
        (xmax-x0),
        stride * sizeof(float),
        get_vector_length<bfloat16>() * 8 / 8
    );
}

template<>
void Transform<12, 4, false, VLType::SVE>(
    bfloat16 *out, const float *in, int stride, int x0, int xmax, int k0, int kmax)
{
    kai_ops_a64_interleave_2_2x4_fp32bf16(
        out,
        in + x0 * stride + k0,
        (kmax-k0),
        (xmax-x0),
        stride * sizeof(float),
        get_vector_length<bfloat16>() * 12 / 8
    );
}
