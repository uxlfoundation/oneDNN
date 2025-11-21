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

#ifndef GPU_INTEL_JIT_CODEGEN_ALLOCATION_SIZE_HPP
#define GPU_INTEL_JIT_CODEGEN_ALLOCATION_SIZE_HPP

#include "gpu/intel/jit/ir/legacy.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

static const int ngen_alloc_granularity = 4;
inline int register_alloc_size(const alloc_t &obj, int grf_size) {
    return (obj.kind == alloc_kind_t::grf)
            ? into<int>(utils::rnd_up(obj.size, grf_size))
            : 0;
}

inline int register_alloc_size(const let_t &obj) {
    // Empty objects are allocated in reserved space
    // nGEN only claims subregisters at dword granularity
    if (obj.value.is_empty()) return 0;
    return utils::rnd_up(obj.var.type().size(), ngen_alloc_granularity);
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
