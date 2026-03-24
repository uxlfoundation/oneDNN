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

#include "cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace platform {

// returns true if the CPU is a hybrid CPU
// (e.g. Meteor Lake, Alder Lake, Raptor Lake, Lunar Lake)
bool is_hybrid();

// Get the Xbyak::util::CoreType of the core the calling thread is running on.
// To get the core type of a specific core, set thread affinity
// to that core or use the xbyak_util::CpuTopology methods to query
// core types directly.
// Returns Xbyak::util::Performance (P-core) by default on non-hybrid CPUs.
Xbyak::util::CoreType get_core_type();

enum class behavior_t {
    p_core, // Performance core
    lp_core, // Efficiency core (E-core with shared L3)
    lpe_core, // Low-power efficiency core no L3 cache (e.g. Meteor Lake's SoC tile E-core island).
    current, // Current core
    min, // (default) used to select the smallest value for all the cores
    max, // used to select the largest value for all the cores
    legacy // legacy get_per_core_cache_size behavior (uses CPUID doesn't consider hybrid)
};

// Returns true if this hybrid CPU has a low-power E-core island (LP E-cores),
// i.e. Efficient cores that have no L3 cache. Only meaningful on hybrid CPUs.
bool has_lpe_core();

// Use Xbyak_utils cache topology methods to determine the per-core cache size.
//
// This avoids using older CPUID-based methods which can result in inaccurate
// values on hybrid CPUs.
//
// The behavior_t argument specifies the behavior of the query on hybrid CPUs.
// On non-hybrid CPUs, the behavior_t argument is ignored and the function
// returns the per-core cache size as normal.
//
// - if behavior_t is p_core/lp_core/lpe_core, the function returns the per-core cache
//   size for that core type.
// - if behavior_t is min/max, the function returns the min/max per-core cache
//   size among all cores
// - if behavior_t is current, the function returns the cache size of the core
//   the calling thread is running on.
// - if behavior_t is legacy, the function behaves like the legacy
//   get_per_core_cache_size(level) function using CPUID with no consideration of
//   hybrid CPUs.
//
// Assumption each core type on a system is homogeneous in terms of cache
// topology e.g. all P-cores have the same cache topology, all LP-cores have the
// same cache topology, all LPE-cores have the same cache topology. The LPE-core
// type is a subset of the E-core type, so the presence of LPE-cores is determined
// by checking for E-cores with no L3 cache.
unsigned get_per_core_cache_size(int level, behavior_t btype = behavior_t::min);

} // namespace platform
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
