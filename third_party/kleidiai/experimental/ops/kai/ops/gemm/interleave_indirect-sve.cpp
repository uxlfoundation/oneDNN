//
// SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "asmlib.hpp"
#include "kai/ops/gemm/convolution_parameters.hpp"
#include "convolver.hpp"
#include "interleave_indirect.hpp"
#include "kai/ops/bfloat.hpp"
#include <alloca.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <tuple>
#include <type_traits>
#include <vector>

#include <arm_neon.h>

#include "common_internal/utils.hpp"

namespace kai {
namespace ops {

#include "interleave_indirect_impl.hpp"
#include "indirect-interleaves/list-sve.hpp"

/**** Instantiate needed implementations ****/
#ifdef __aarch64__
template void IndirectInterleave<1, 1, VLType::SME>(__fp16 *, const __fp16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<1, 1, VLType::SME>(__fp16 *, const __fp16 *, size_t, const convolver<__fp16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<1, 1, VLType::SME>(__fp16 *, const __fp16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<2, 1, VLType::SME>(__fp16 *, const __fp16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<2, 1, VLType::SME>(__fp16 *, const __fp16 *, size_t, const convolver<__fp16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<2, 1, VLType::SME>(__fp16 *, const __fp16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<4, 2, VLType::SME>(__fp16 *, const __fp16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<4, 2, VLType::SME>(__fp16 *, const __fp16 *, size_t, const convolver<__fp16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<4, 2, VLType::SME>(__fp16 *, const __fp16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<1, 2, VLType::SME>(__fp16 *, const __fp16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<1, 2, VLType::SME>(__fp16 *, const __fp16 *, size_t, const convolver<__fp16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<1, 2, VLType::SME>(__fp16 *, const __fp16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<2, 2, VLType::SME>(__fp16 *, const __fp16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<2, 2, VLType::SME>(__fp16 *, const __fp16 *, size_t, const convolver<__fp16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<2, 2, VLType::SME>(__fp16 *, const __fp16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<1, 4, VLType::SME>(uint8_t *, const uint8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<1, 4, VLType::SME>(uint8_t *, const uint8_t *, size_t, const convolver<uint8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<1, 4, VLType::SME>(uint8_t *, const uint8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<4, 4, VLType::SME>(uint8_t *, const uint8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<4, 4, VLType::SME>(uint8_t *, const uint8_t *, size_t, const convolver<uint8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<4, 4, VLType::SME>(uint8_t *, const uint8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<2, 4, VLType::SME>(uint8_t *, const uint8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<2, 4, VLType::SME>(uint8_t *, const uint8_t *, size_t, const convolver<uint8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<2, 4, VLType::SME>(uint8_t *, const uint8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<1, 4, VLType::SME>(int8_t *, const int8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<1, 4, VLType::SME>(int8_t *, const int8_t *, size_t, const convolver<int8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<1, 4, VLType::SME>(int8_t *, const int8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<4, 4, VLType::SME>(int8_t *, const int8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<4, 4, VLType::SME>(int8_t *, const int8_t *, size_t, const convolver<int8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<4, 4, VLType::SME>(int8_t *, const int8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<2, 4, VLType::SME>(int8_t *, const int8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<2, 4, VLType::SME>(int8_t *, const int8_t *, size_t, const convolver<int8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<2, 4, VLType::SME>(int8_t *, const int8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<1, 2, VLType::SME>(bfloat16 *, const bfloat16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<1, 2, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, const convolver<bfloat16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<1, 2, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<4, 2, VLType::SME>(bfloat16 *, const bfloat16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<4, 2, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, const convolver<bfloat16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<4, 2, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<2, 2, VLType::SME>(bfloat16 *, const bfloat16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<2, 2, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, const convolver<bfloat16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<2, 2, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<1, 2, VLType::SME>(bfloat16 *, const float * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<1, 2, VLType::SME>(bfloat16 *, const float *, size_t, const convolver<float> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<1, 2, VLType::SME>(bfloat16 *, const float *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<1, 1, VLType::SME>(float *, const float * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<1, 1, VLType::SME>(float *, const float *, size_t, const convolver<float> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<1, 1, VLType::SME>(float *, const float *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<4, 2, VLType::SME>(bfloat16 *, const float * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<4, 2, VLType::SME>(bfloat16 *, const float *, size_t, const convolver<float> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<4, 2, VLType::SME>(bfloat16 *, const float *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<4, 1, VLType::SME>(float *, const float * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<4, 1, VLType::SME>(float *, const float *, size_t, const convolver<float> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<4, 1, VLType::SME>(float *, const float *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<2, 2, VLType::SME>(bfloat16 *, const float * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<2, 2, VLType::SME>(bfloat16 *, const float *, size_t, const convolver<float> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<2, 2, VLType::SME>(bfloat16 *, const float *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<2, 1, VLType::SME>(float *, const float * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<2, 1, VLType::SME>(float *, const float *, size_t, const convolver<float> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<2, 1, VLType::SME>(float *, const float *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<1, 1, VLType::SME>(bfloat16 *, const bfloat16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<1, 1, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, const convolver<bfloat16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<1, 1, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<2, 1, VLType::SME>(bfloat16 *, const bfloat16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<2, 1, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, const convolver<bfloat16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<2, 1, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
#endif // __aarch64__

}  // namespace ops
}  // namespace kai
