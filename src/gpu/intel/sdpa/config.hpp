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

#ifndef GPU_INTEL_SDPA_CONFIG_HPP
#define GPU_INTEL_SDPA_CONFIG_HPP

#include "common/sdpa_pd.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sdpa {

using pd_t = sdpa_pd_t;


// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#define  GET_KERNEL_PRINT \
    kernel_ctx.define_int("KERNEL_PRINT", getenv_int("KERNEL_PRINT", 0)); \
    int g0 = getenv_int("G0", 0); \
    int g1 = getenv_int("G1", 0); \
    int g2 = getenv_int("G2", 0); \
    kernel_ctx.define_int("G0", g0); \
    kernel_ctx.define_int("G1", g1); \
    kernel_ctx.define_int("G2", g2);

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

} // namespace sdpa
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
