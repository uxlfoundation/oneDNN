//
// SPDX-FileCopyrightText: Copyright 2017-2018, 2023-2024, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "convolver.hpp"
#include "interleave_indirect.hpp"
#include "mergeresults.hpp"
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
template<typename TInput, typename TWeight, typename TResult, unsigned int height, unsigned int width_vectors, unsigned int block=1, unsigned int mmla=1, bool integrate_sums=false>
class StdTransformsSVE
{
public:
    template<typename TIn>
    void PrepareA(TInput *out, const TIn *in, const int stride, const int y0,
                  const int ymax, const int k0, const int kmax, int32_t row_sum_multiplier) {
        Interleave<height, block, VLType::None>(out, in, stride, y0, ymax, k0, kmax, integrate_sums, row_sum_multiplier);
    }

    template<typename TIn>
    void PrepareA_indirect(TInput *out, const TIn * const * const *ptr, size_t stringlen, size_t rounded_stringlen, const int y0,
                           const int ymax, const int k0, const int kmax, int32_t row_sum_multiplier) {
        IndirectInterleave<height, block, VLType::None>(out, ptr, stringlen, rounded_stringlen, y0, ymax, k0, kmax, integrate_sums, row_sum_multiplier);
    }

    template<typename TIn>
    void PrepareA_convolution(TInput *out, const TIn *ptr, size_t stride, const convolver<TIn> &conv, size_t rounded_stringlen,
                              const int y0, const int ymax, const int k0, const int kmax, int32_t row_sum_multiplier) {
        ConvolutionInterleave<height, block, VLType::None>(out, ptr, stride, conv, rounded_stringlen, y0, ymax, k0, kmax, integrate_sums, row_sum_multiplier);
    }

    bool PrepareB_supports_transpose() const {
        return false;
    }

    template<typename TIn>
    void PrepareB(TWeight *out, const TIn *in, const int stride, const int x0,
                  const int xmax, const int k0, const int kmax, bool transposed) {
        assert (!transposed);
        Transform<width_vectors, block,  true, VLType::SVE>(out, in, stride, x0, xmax, k0, kmax);
    }

    template<typename TOut>
    void Merge(TOut *out, const TResult *in, int stride, int y0, int ymax, int x0, int xmax, const TOut *bias, const Activation act, bool append) {
        MergeResults<width_vectors / mmla, height, true>(out, in, stride, y0, ymax, x0, xmax, bias, act, append);
    }
};

}  // namespace ops
}  // namespace kai
