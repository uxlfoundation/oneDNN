/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#ifndef CPU_X64_PLATFORM_HPP
#define CPU_X64_PLATFORM_HPP

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace platform {

// Returns true if the CPU reports hybrid core types (P-cores and E-cores).
// May return false on platforms where hybrid-core topology detection is unavailable.
bool is_hybrid();

// Returns true if a low-power E-core island (LP E-cores) is present,
// i.e. Efficient cores with no L3 cache. Meaningful only on hybrid CPUs.
bool has_lpe_core();

// Returns the per-core cache size in bytes for the given level
// (1=L1d, 2=L2, 3=L3).
// On hybrid systems, returns a conservative value based on the smallest
// per-core cache among present core types.
unsigned get_per_core_cache_size(int level);

} // namespace platform
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
