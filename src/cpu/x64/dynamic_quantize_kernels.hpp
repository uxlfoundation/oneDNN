/*******************************************************************************
* Copyright 2026 Advanced Micro Devices, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef CPU_X64_DYNAMIC_QUANTIZE_KERNELS_HPP
#define CPU_X64_DYNAMIC_QUANTIZE_KERNELS_HPP

#include <cstdint>

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace dynamic_quantize_kernels {

// Fused per-token (per-row) AVX-512 dynamic-quant kernels ported from ZenDNN.
// Scale layout is [M, 1]; each row is scanned for absmax and quantized to s8.
void dynamic_per_token_quant_bf16_s8_native(
        const uint16_t *src, int8_t *dst, float *scales, int64_t M, int64_t N);
void dynamic_per_token_quant_f32_s8_native(
        const float *src, int8_t *dst, float *scales, int64_t M, int64_t N);
void dynamic_per_token_quant_f16_s8_native(
        const uint16_t *src, int8_t *dst, float *scales, int64_t M, int64_t N);
// Fused per-group (per-group-col) AVX-512 dynamic-quant kernels ported from
// ZenDNN. Scale layout is [M, G]; the K axis is split into G contiguous groups
// of size K / G (G must divide K). Each (row, group) is scanned and quantized
// in one pass.
void dynamic_per_group_quant_f32_s8_native(const float *src, int8_t *dst,
        float *scales, int64_t M, int64_t K, int64_t G);
void dynamic_per_group_quant_bf16_s8_native(const uint16_t *src, int8_t *dst,
        float *scales, int64_t M, int64_t K, int64_t G);
void dynamic_per_group_quant_f16_s8_native(const uint16_t *src, int8_t *dst,
        float *scales, int64_t M, int64_t K, int64_t G);
} // namespace dynamic_quantize_kernels
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
