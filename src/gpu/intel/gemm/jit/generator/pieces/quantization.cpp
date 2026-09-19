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

#include <cstdlib>

#include "internal/utils.hpp"
#include "layout_utils.hpp"
#include "quantization.hpp"

using namespace ngen;
using std::vector;

GEMMSTONE_NAMESPACE_START

bool canDequantizeInt4(const RegisterLayout &layoutSrc, const RegisterLayout &layoutDst,
                       const RegisterLayout &layoutOffset, const RegisterLayout &layoutScale)
{
    auto Tsrc = layoutSrc.type(), Tdst = layoutDst.type();
    if (!Tsrc.isIntSubByte() || !one_of(Tdst, {Type::f16, Type::f32}))
        return false;

    if (layoutOffset.empty() || layoutScale.empty())
        if (layoutSrc.rows() < layoutDst.rows() || layoutSrc.cols() < layoutDst.cols())
            return false;

    return true;
}

bool dequantizeInt4KeepsBias(ngen::HW hw, Type Tsrc)
{
    // Xe3p converts int4 directly with shfl.
    return !(hw == ngen::HW::Xe3p && Tsrc.isInt4());
}

int dequantizeInt4Bias(ngen::HW hw, Type Tsrc)
{
    return (dequantizeInt4KeepsBias(hw, Tsrc) ? 1024 : 0) + (Tsrc == Type::s4 ? 8 : 0);
}

bool int4OffsetsCarryBias(Type Txo)
{
    return Txo.isInteger() && Txo.paddedSize() <= 1;
}

uint16_t f16Bits(int value)
{
    uint16_t sign = (value < 0) ? 0x8000 : 0;
    int v = std::abs(value);
    if (v >= 2048) stub();
    if (v == 0) return sign;
    int e = ilog2(v);
    return sign | ((e + 15) << 10) | ((v - (1 << e)) << (10 - e));
}

GEMMSTONE_NAMESPACE_END
