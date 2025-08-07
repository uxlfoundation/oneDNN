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

#ifndef GPU_INTEL_SOFTMAX_CONFIG_HPP
#define GPU_INTEL_SOFTMAX_CONFIG_HPP

#include "gpu/gpu_softmax_pd.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace softmax {

using fwd_pd_t = gpu_softmax_fwd_pd_t;
using bwd_pd_t = gpu_softmax_bwd_pd_t;

} // namespace softmax
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
