//
// SPDX-FileCopyrightText: Copyright 2020, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <cstdint>

#include "kai/ops/gemm/convolution_parameters.hpp"
#include "convolver.hpp"
#include "common_internal/utils.hpp"

namespace kai {
namespace ops {

template<unsigned int height_vectors, unsigned int block, VLType vlt, typename TIn, typename TOut>
void IndirectInterleave(TOut *out, const TIn * const * const *ptr, unsigned int stringlen, unsigned int rounded_stringlen, unsigned int y0, unsigned int ymax, unsigned int k0, unsigned int kmax, bool, int32_t);

template<unsigned int height_vectors, unsigned int block, VLType vlt, typename TIn, typename TOut>
void ConvolutionInterleave(TOut *out, const TIn *in, size_t in_stride, const convolver<TIn> &conv, const unsigned int rounded_stringlen, const unsigned int y0, const unsigned int ymax, const unsigned int k0, const unsigned int kmax, bool, int32_t);

template<unsigned int height_vectors, unsigned int block, VLType vlt, typename TIn, typename TOut>
void Interleave(TOut *out, const TIn *in, size_t in_stride, const unsigned int y0, const unsigned int ymax, const unsigned int k0, const unsigned int kmax, bool, int32_t);

}  // namespace ops
}  // namespace kai
