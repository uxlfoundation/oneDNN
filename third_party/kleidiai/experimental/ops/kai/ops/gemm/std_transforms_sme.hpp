//
// SPDX-FileCopyrightText: Copyright 2022-2024, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "interleave_indirect.hpp"
#include "transform.hpp"

namespace kai {
namespace ops {

/*
 * Define "standard" transforms for the blocked GEMMs for SVE.
 *
 * This assumes that A is interleaved 'height' ways, B is interleaved
 * 'width'xVL ways and transposed, and that the merge needs to work in
 * 'height' x 'width'xVL blocks.
 *
 * The optional 'block' parameter is for kernels using dot-product type
 * instructions like UDOT and SDOT.
 */
template<typename TOperand, typename TResult, unsigned int height_vectors, unsigned int width_vectors, unsigned int block=1, bool integrate_sums=false>
class StdTransformsSME
{
public:
    template<typename TIn>
    void PrepareA(TOperand *out, const TIn *in, const int stride, const int y0,
                  const int ymax, const int k0, const int kmax, int32_t row_sum_multiplier) {
        Interleave<height_vectors, block, VLType::SME>(out, in, stride, y0, ymax, k0, kmax, integrate_sums, row_sum_multiplier);
    }

    template<typename TIn>
    void PrepareA_indirect(TOperand *out, const TIn * const * const *ptr, size_t stringlen, size_t rounded_stringlen, const int y0,
                           const int ymax, const int k0, const int kmax, int32_t row_sum_multiplier) {
        IndirectInterleave<height_vectors, block, VLType::SME>(out, ptr, stringlen, rounded_stringlen, y0, ymax, k0, kmax, integrate_sums, row_sum_multiplier);
    }

    template<typename TIn>
    void PrepareA_convolution(TOperand *out, const TIn *ptr, size_t stride, const convolver<TIn> &conv, size_t rounded_stringlen,
                              const int y0, const int ymax, const int k0, const int kmax, int32_t row_sum_multiplier) {
        ConvolutionInterleave<height_vectors, block, VLType::SME>(out, ptr, stride, conv, rounded_stringlen, y0, ymax, k0, kmax, integrate_sums, row_sum_multiplier);
    }

    bool PrepareB_supports_transpose() const {
        return false;
    }

    template<typename TIn>
    void PrepareB(TOperand *out, const TIn *in, const int stride, const int x0,
                  const int xmax, const int k0, const int kmax, bool transposed) {
        assert (!transposed);
        Transform<width_vectors, block,  true, VLType::SME>(out, in, stride, x0, xmax, k0, kmax);
    }

    template<typename TOut>
    void Merge(TOut *, const TResult *, int, int, int, int, int, const TOut *, const Activation, bool) {
        // Separate merge not supported for SME.
    }
};

}  // namespace ops
}  // namespace kai
