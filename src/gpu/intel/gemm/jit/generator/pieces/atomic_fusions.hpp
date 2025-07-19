/*******************************************************************************
* Copyright 2025 Intel Corporation
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


#ifndef GEMMSTONE_GUARD_ATOMIC_FUSIONS_HPP
#define GEMMSTONE_GUARD_ATOMIC_FUSIONS_HPP

#include "gemmstone/problem.hpp"
#include "gemmstone/strategy.hpp"

GEMMSTONE_NAMESPACE_START

// Calculate per-thread stride within temporary C memory.
inline int tempCThreadStride(const GEMMProblem &problem, const GEMMStrategy &strategy)
{
    int stride = strategy.unroll[LoopM] * strategy.unroll[LoopN];
    if (problem.sumA) stride += strategy.unroll[LoopM];
    if (problem.sumB) stride += strategy.unroll[LoopN];
    stride *= problem.Tc;
    stride = align_up(stride, 64);
    return stride;
}

// Calculate per-workgroup stride within temporary C memory.
inline int tempCWGStride(const GEMMProblem &problem, const GEMMStrategy &strategy) {
    return tempCThreadStride(problem, strategy) * strategy.wg[LoopM] * strategy.wg[LoopN];
}

GEMMSTONE_NAMESPACE_END

#endif /* header guard */
