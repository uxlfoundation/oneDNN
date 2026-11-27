//
// SPDX-FileCopyrightText: Copyright 2018-2020, 2024, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "convolver.hpp"
#include "mergeresults.hpp"
#include "transform.hpp"
#include "interleave_indirect.hpp"

namespace kai {
namespace ops {

/*
 * Define "standard" transforms for the blocked GEMMs with fixed vector
 * length.
 *
 * This assumes that A is interleaved 'height' ways, B is interleaved
 * 'transform_width' ways and transposed, and that the merge needs to work
 * in 'height' x 'result_width' blocks.
 *
 * The optional 'block' parameter is for kernels using dot-product type
 * instructions like UDOT and SDOT.
 */
template<typename TInput, typename TWeight, typename TResult, unsigned int height, unsigned int transform_width, unsigned int result_width=transform_width, unsigned int block=1, bool integrate_sums=false>
class StdTransformsFixed
{
public:
    template<typename TIn>
    void PrepareA(TInput *out, const TIn *in, const int stride, const int y0,
                  const int ymax, const int k0, const int kmax, int32_t row_sum_multiplier) const {
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
        return true;
    }

    template<typename TIn>
    void PrepareB(TWeight *out, const TIn *in, const int stride, const int x0,
                  const int xmax, const int k0, const int kmax, bool transposed) const {
        if (transposed) {
            Transform<transform_width, block, false>(out, in, stride, x0, xmax, k0, kmax);
        } else {
            Transform<transform_width, block, true>(out, in, stride, x0, xmax, k0, kmax);
        }
    }

    template<typename TOut>
    void Merge(const KernelArgs *args) const {
        MergeResults<result_width, height, false, TResult, TOut>(args);
    }
};

}  // namespace ops
}  // namespace kai
