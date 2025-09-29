/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#ifndef GEMMSTONE_GUARD_MICROKERNEL_PROVIDER_HPP
#define GEMMSTONE_GUARD_MICROKERNEL_PROVIDER_HPP

#include "gemmstone/config.hpp"
#include "gemmstone/kernel_selector.hpp"
#include "gemmstone/kernel_evaluator.hpp"

GEMMSTONE_NAMESPACE_START

/* Hardware information for microkernel provider */
struct HWInformation {
    uint32_t gmdid;
    int euCount;
    bool systolicAvailable;
};

/* Main entrypoint for microkernel auto-selection */
micro::Package selectGEMMMicrokernel(micro::GEMMProtocol protocol, HWInformation hwInfo, SizeParams sizes, const GEMMProblem &problem,
                                     const std::vector<StrategyRequirement> &reqs = std::vector<StrategyRequirement>(),
                                     void (*strategyAdjuster)(GEMMStrategy &strategy) = nullptr);

/* Helpers */
static inline int alignmentForLD(int ld)
{
    for (int x = 1; x <= 64; x <<= 1)
        if (ld & x) return x;
    return 128;
};

GEMMSTONE_NAMESPACE_END

#endif /* header guard */
