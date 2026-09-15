//
// SPDX-FileCopyrightText: Copyright 2019, 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "common_internal/utils.hpp" // IndirectInputArg

namespace kai {
namespace ops {

template<typename Tin, typename Tout>
void requantize_block_32(const Requantize32 &qp, unsigned int width, unsigned int height,
                         const Tin *input, unsigned int in_stride, Tout *output, unsigned int out_stride,
                         const int32_t *row_bias, const int32_t *col_bias, unsigned int start_col);

template<typename T>
void compute_row_sums(const Requantize32 &qp, unsigned int width, unsigned int height,
                      const T *input, unsigned int in_stride, int32_t *row_bias);

template<typename T>
void compute_col_sums(const Requantize32 &qp, unsigned int width, unsigned int height,
                      const T *input, unsigned int in_stride, int32_t *col_bias, unsigned int depth,
                      unsigned int multi, unsigned int first_col);

template<typename T>
void row_sums_indirect(size_t num_strings, const unsigned int *string_lengths, IndirectInputArg<T> A_arg,
                       size_t M, int32_t *output_ptr, const Requantize32 *qp);

template<typename T>
void dequantize_block_32(const DequantizeFloat &qp, unsigned int width, unsigned int height,
                         const int32_t* input, unsigned int in_stride, T *output, unsigned int out_stride,
                         const T *row_bias, bool not_first_pass, const Activation &act);

}  // namespace ops
}  // namespace kai
