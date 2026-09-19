/*******************************************************************************
* Copyright 2024 Intel Corporation
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


#ifndef GEMMSTONE_GENERATOR_PIECES_QUANTIZATION_HPP
#define GEMMSTONE_GENERATOR_PIECES_QUANTIZATION_HPP

#include "internal/ngen_includes.hpp"
#include "gemmstone/type.hpp"
#include "register_layout.hpp"

GEMMSTONE_NAMESPACE_START

// Check if the optimized int4 dequantization sequence (dequantizeInt4) can be used.
bool canDequantizeInt4(const RegisterLayout &layoutSrc, const RegisterLayout &layoutDst,
                       const RegisterLayout &layoutOffset, const RegisterLayout &layoutScale);

// Check if dequantizeInt4's int4/int3 -> f16 copy leaves its 2^10 bias in place (see CopyPlan::keepSubByteBias).
bool dequantizeInt4KeepsBias(ngen::HW hw, Type Tsrc);

// Total bias to remove after dequantizeInt4's copy, including the s4 -> u4 shift.
int dequantizeInt4Bias(ngen::HW hw, Type Tsrc);

// Check if repacked offsets of the given (external) type absorb that bias. bias + offset must be exact in f16.
bool int4OffsetsCarryBias(Type Txo);

// f16 encoding of a small integer (|value| < 2048).
uint16_t f16Bits(int value);

GEMMSTONE_NAMESPACE_END

#endif /* header guard */
