//
// SPDX-FileCopyrightText: Copyright 2019, 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "common_internal/kernel_args.hpp"
#include "common_internal/utils.hpp"

namespace kai {
namespace ops {

template<typename Tin, typename Tout>
void requantize_block_32(const Requantize32 &qp, unsigned int width, unsigned int height,
                         const Tin *input, unsigned int in_stride, Tout *output, unsigned int out_stride,
                         const int32_t *row_bias, const int32_t *col_bias, unsigned int start_col);

template<typename T>
void compute_row_sums(const Requantize32 &qp, unsigned int width, unsigned int height,
                      const T *input, unsigned int in_stride, int32_t *row_bias);

// Compute -offset times each column sum of a direct K x N RHS matrix.
// Uses rhs, rhs_stride (bytes), K and N; writes N contiguous int32_t values.
template<typename T>
void compute_col_sums(const KernelArgs *ka, int32_t *out_ptr, int32_t offset);

template<typename T>
void row_sums_indirect(const KernelArgs *ka, int32_t *out_ptr, int32_t rhs_offset);

template<typename T>
void dequantize_block_32(const DequantizeFloat &qp, unsigned int width, unsigned int height,
                         const int32_t* input, unsigned int in_stride, T *output, unsigned int out_stride,
                         const T *row_bias, bool not_first_pass, const Activation &act);

}  // namespace ops
}  // namespace kai
