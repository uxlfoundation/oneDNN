//
// SPDX-FileCopyrightText: Copyright 2018-2021, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>

namespace kai {
namespace ops {
/*
 * Parameter set for "convolution" type GEMM.
 *
 * For a "convolution" GEMM, the GEMM parameters (M, K) are specified as if
 * an im2row had been performed on the input tensor to generate the operand
 * matrix, but instead this structure describes the convolution parameters
 * such that this can be done on the fly.
 *
 * The parameters describe the convolution details - the notional shape of
 * the input and output tensors, whether padding is to be applied, the size
 * of the kernel and a constant value to be used for padding (needed for
 * quantized tensors).
 *
 * The second part describes the layout of the input tensor in memory, which
 * is assumed to be in NHWC format.  This consists of a base pointer and
 * strides for columns, rows and batches.  'multis' are not supported for
 * convolution type GEMMs.
 */
struct ConvolutionParameters
{
    int64_t input_width;
    int64_t input_height;
    int64_t input_channels;
    int64_t kernel_width;
    int64_t kernel_height;
    int64_t output_width;
    int64_t output_height;
    int64_t output_stride_w;
    int64_t output_stride_h;
    //          output_channels not included as they do not affect the input.
    int64_t dilation_w;
    int64_t dilation_h;
    int64_t padding_top;
    int64_t padding_left;
    float   padding_value;
};

}  // namespace ops
}  // namespace kai
