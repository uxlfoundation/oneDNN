//
// SPDX-FileCopyrightText: Copyright 2020-2022, 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "asmlib.hpp"
#include "kai/ops/gemm/convolution_parameters.hpp"
#include "convolver.hpp"
#include "interleave_indirect.hpp"
#include "kai/ops/bfloat.hpp"
#include "kai/ops/float16.hpp"

#if !defined(_WIN64) && !defined(__OpenBSD__)
#include <alloca.h>
#endif /* !defined(_WIN64) && !defined(__OpenBSD__) */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <tuple>
#include <type_traits>
#include <vector>

#include "common_internal/utils.hpp"

namespace kai {
namespace ops {

#include "interleave_indirect_impl.hpp"
#include "indirect-interleaves/list.hpp"

/**** Instantiate needed implementations ****/

template void IndirectInterleave<1, 1, VLType::SME>(__fp16 *, const __fp16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<1, 1, VLType::SME>(__fp16 *, const __fp16 *, size_t, const convolver<__fp16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<1, 1, VLType::SME>(__fp16 *, const __fp16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<2, 1, VLType::SME>(__fp16 *, const __fp16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<2, 1, VLType::SME>(__fp16 *, const __fp16 *, size_t, const convolver<__fp16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<2, 1, VLType::SME>(__fp16 *, const __fp16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<1, 2, VLType::SME>(__fp16 *, const __fp16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<1, 2, VLType::SME>(__fp16 *, const __fp16 *, size_t, const convolver<__fp16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<1, 2, VLType::SME>(__fp16 *, const __fp16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<4, 2, VLType::SME>(__fp16 *, const __fp16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<4, 2, VLType::SME>(__fp16 *, const __fp16 *, size_t, const convolver<__fp16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<4, 2, VLType::SME>(__fp16 *, const __fp16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<2, 2, VLType::SME>(__fp16 *, const __fp16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<2, 2, VLType::SME>(__fp16 *, const __fp16 *, size_t, const convolver<__fp16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<2, 2, VLType::SME>(__fp16 *, const __fp16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<8, 1, VLType::None>(__fp16 *, const __fp16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 1, VLType::None>(__fp16 *, const __fp16 *, size_t, const convolver<__fp16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 1, VLType::None>(__fp16 *, const __fp16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<8, 1, VLType::None>(float *, const __fp16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 1, VLType::None>(float *, const __fp16 *, size_t, const convolver<__fp16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 1, VLType::None>(float *, const __fp16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<8, 8, VLType::None>(uint8_t *, const uint8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 8, VLType::None>(uint8_t *, const uint8_t *, size_t, const convolver<uint8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 8, VLType::None>(uint8_t *, const uint8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<1, 4, VLType::SME>(uint8_t *, const uint8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<1, 4, VLType::SME>(uint8_t *, const uint8_t *, size_t, const convolver<uint8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<1, 4, VLType::SME>(uint8_t *, const uint8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<4, 4, VLType::SME>(uint8_t *, const uint8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<4, 4, VLType::SME>(uint8_t *, const uint8_t *, size_t, const convolver<uint8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<4, 4, VLType::SME>(uint8_t *, const uint8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<2, 4, VLType::SME>(uint8_t *, const uint8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<2, 4, VLType::SME>(uint8_t *, const uint8_t *, size_t, const convolver<uint8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<2, 4, VLType::SME>(uint8_t *, const uint8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<8, 4, VLType::None>(uint8_t *, const uint8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 4, VLType::None>(uint8_t *, const uint8_t *, size_t, const convolver<uint8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 4, VLType::None>(uint8_t *, const uint8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<8, 1, VLType::None>(uint16_t *, const uint8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 1, VLType::None>(uint16_t *, const uint8_t *, size_t, const convolver<uint8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 1, VLType::None>(uint16_t *, const uint8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<1, 4, VLType::SME>(int8_t *, const int8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<1, 4, VLType::SME>(int8_t *, const int8_t *, size_t, const convolver<int8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<1, 4, VLType::SME>(int8_t *, const int8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<4, 4, VLType::SME>(int8_t *, const int8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<4, 4, VLType::SME>(int8_t *, const int8_t *, size_t, const convolver<int8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<4, 4, VLType::SME>(int8_t *, const int8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<2, 4, VLType::SME>(int8_t *, const int8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<2, 4, VLType::SME>(int8_t *, const int8_t *, size_t, const convolver<int8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<2, 4, VLType::SME>(int8_t *, const int8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<8, 8, VLType::None>(int8_t *, const int8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 8, VLType::None>(int8_t *, const int8_t *, size_t, const convolver<int8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 8, VLType::None>(int8_t *, const int8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<8, 4, VLType::None>(int8_t *, const int8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 4, VLType::None>(int8_t *, const int8_t *, size_t, const convolver<int8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 4, VLType::None>(int8_t *, const int8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<8, 1, VLType::None>(int16_t *, const int8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 1, VLType::None>(int16_t *, const int8_t *, size_t, const convolver<int8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 1, VLType::None>(int16_t *, const int8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<8, 1, VLType::None>(uint16_t *, const uint16_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 1, VLType::None>(uint16_t *, const uint16_t *, size_t, const convolver<uint16_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 1, VLType::None>(uint16_t *, const uint16_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<1, 2, VLType::SME>(bfloat16 *, const bfloat16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<1, 2, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, const convolver<bfloat16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<1, 2, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<2, 2, VLType::SME>(bfloat16 *, const bfloat16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<2, 2, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, const convolver<bfloat16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<2, 2, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<8, 4, VLType::None>(bfloat16 *, const bfloat16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 4, VLType::None>(bfloat16 *, const bfloat16 *, size_t, const convolver<bfloat16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 4, VLType::None>(bfloat16 *, const bfloat16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<8, 2, VLType::None>(bfloat16 *, const bfloat16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 2, VLType::None>(bfloat16 *, const bfloat16 *, size_t, const convolver<bfloat16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 2, VLType::None>(bfloat16 *, const bfloat16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<8, 1, VLType::None>(float *, const bfloat16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 1, VLType::None>(float *, const bfloat16 *, size_t, const convolver<bfloat16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 1, VLType::None>(float *, const bfloat16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<8, 4, VLType::None>(bfloat16 *, const float * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 4, VLType::None>(bfloat16 *, const float *, size_t, const convolver<float> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 4, VLType::None>(bfloat16 *, const float *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<1, 2, VLType::SME>(bfloat16 *, const float * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<1, 2, VLType::SME>(bfloat16 *, const float *, size_t, const convolver<float> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<1, 2, VLType::SME>(bfloat16 *, const float *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<2, 2, VLType::SME>(bfloat16 *, const float * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<2, 2, VLType::SME>(bfloat16 *, const float *, size_t, const convolver<float> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<2, 2, VLType::SME>(bfloat16 *, const float *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<1, 1, VLType::SME>(float *, const float * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<1, 1, VLType::SME>(float *, const float *, size_t, const convolver<float> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<1, 1, VLType::SME>(float *, const float *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<4, 1, VLType::SME>(float *, const float * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<4, 1, VLType::SME>(float *, const float *, size_t, const convolver<float> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<4, 1, VLType::SME>(float *, const float *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<2, 1, VLType::SME>(float *, const float * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<2, 1, VLType::SME>(float *, const float *, size_t, const convolver<float> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<2, 1, VLType::SME>(float *, const float *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<8, 2, VLType::None>(float *, const float * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 2, VLType::None>(float *, const float *, size_t, const convolver<float> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 2, VLType::None>(float *, const float *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<8, 1, VLType::None>(float *, const float * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 1, VLType::None>(float *, const float *, size_t, const convolver<float> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 1, VLType::None>(float *, const float *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<8, 1, VLType::None>(int16_t *, const int16_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 1, VLType::None>(int16_t *, const int16_t *, size_t, const convolver<int16_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 1, VLType::None>(int16_t *, const int16_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<1, 1, VLType::SME>(bfloat16 *, const bfloat16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<1, 1, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, const convolver<bfloat16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<1, 1, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<2, 1, VLType::SME>(bfloat16 *, const bfloat16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<2, 1, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, const convolver<bfloat16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<2, 1, VLType::SME>(bfloat16 *, const bfloat16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);


}  // namespace ops
}  // namespace kai
