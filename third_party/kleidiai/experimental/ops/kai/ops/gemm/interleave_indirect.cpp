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

#include <arm_neon.h>

#include "common_internal/utils.hpp"

namespace kai {
namespace ops {

#include "interleave_indirect_impl.hpp"
#include "indirect-interleaves/list.hpp"

/**** Instantiate needed implementations ****/
#ifdef __aarch64__
template void IndirectInterleave<8, 1, VLType::None>(__fp16 *, const __fp16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 1, VLType::None>(__fp16 *, const __fp16 *, size_t, const convolver<__fp16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 1, VLType::None>(__fp16 *, const __fp16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<8, 1, VLType::None>(float *, const __fp16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 1, VLType::None>(float *, const __fp16 *, size_t, const convolver<__fp16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 1, VLType::None>(float *, const __fp16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<8, 8, VLType::None>(uint8_t *, const uint8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 8, VLType::None>(uint8_t *, const uint8_t *, size_t, const convolver<uint8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 8, VLType::None>(uint8_t *, const uint8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<8, 4, VLType::None>(uint8_t *, const uint8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 4, VLType::None>(uint8_t *, const uint8_t *, size_t, const convolver<uint8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 4, VLType::None>(uint8_t *, const uint8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<8, 1, VLType::None>(uint16_t *, const uint8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 1, VLType::None>(uint16_t *, const uint8_t *, size_t, const convolver<uint8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 1, VLType::None>(uint16_t *, const uint8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<4, 16, VLType::None>(uint8_t *, const uint8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<4, 16, VLType::None>(uint8_t *, const uint8_t *, size_t, const convolver<uint8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<4, 16, VLType::None>(uint8_t *, const uint8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<8, 8, VLType::None>(int8_t *, const int8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 8, VLType::None>(int8_t *, const int8_t *, size_t, const convolver<int8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 8, VLType::None>(int8_t *, const int8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<8, 4, VLType::None>(int8_t *, const int8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 4, VLType::None>(int8_t *, const int8_t *, size_t, const convolver<int8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 4, VLType::None>(int8_t *, const int8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<8, 1, VLType::None>(int16_t *, const int8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 1, VLType::None>(int16_t *, const int8_t *, size_t, const convolver<int8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 1, VLType::None>(int16_t *, const int8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<4, 16, VLType::None>(int8_t *, const int8_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<4, 16, VLType::None>(int8_t *, const int8_t *, size_t, const convolver<int8_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<4, 16, VLType::None>(int8_t *, const int8_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<8, 1, VLType::None>(uint16_t *, const uint16_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 1, VLType::None>(uint16_t *, const uint16_t *, size_t, const convolver<uint16_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 1, VLType::None>(uint16_t *, const uint16_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

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

template void IndirectInterleave<8, 2, VLType::None>(float *, const float * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 2, VLType::None>(float *, const float *, size_t, const convolver<float> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 2, VLType::None>(float *, const float *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<8, 1, VLType::None>(float *, const float * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 1, VLType::None>(float *, const float *, size_t, const convolver<float> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 1, VLType::None>(float *, const float *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<8, 1, VLType::None>(int16_t *, const int16_t * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<8, 1, VLType::None>(int16_t *, const int16_t *, size_t, const convolver<int16_t> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<8, 1, VLType::None>(int16_t *, const int16_t *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
#endif // __aarch64__

#ifdef __arm__
template void IndirectInterleave<6, 1, VLType::None>(float *, const bfloat16 * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<6, 1, VLType::None>(float *, const bfloat16 *, size_t, const convolver<bfloat16> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<6, 1, VLType::None>(float *, const bfloat16 *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);

template void IndirectInterleave<6, 1, VLType::None>(float *, const float * const * const *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void ConvolutionInterleave<6, 1, VLType::None>(float *, const float *, size_t, const convolver<float> &, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
template void Interleave<6, 1, VLType::None>(float *, const float *, size_t, unsigned int, unsigned int, unsigned int, unsigned int, bool, int32_t);
#endif // __arm__

}  // namespace ops
}  // namespace kai
