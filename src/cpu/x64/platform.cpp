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

#include <algorithm>
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/platform.hpp"
#include "xbyak/xbyak_util.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace platform {

namespace {

#ifndef __APPLE__
// Global CPU topology instance, initialized on first use
// This is thread-safe due to C++11 magic statics
// NOTE: xbyak::util::CpuTopology is not supported on macOS.
struct cpu_topology_cach_t {
    Xbyak::util::CpuTopology topology;
    cpu_topology_cach_t() : topology(cpu()) {}
};

// Thread-safe lazy initialization using C++11 magic statics
cpu_topology_cach_t &get_topology_cache() {
    static cpu_topology_cach_t cache;
    return cache;
}

// Map platform cache level (1=L1d, 2=L2, 3=L3) to xbyak CacheType
Xbyak::util::CacheType convert_cache_level(int level) {
    switch (level) {
        case 1: return Xbyak::util::L1d;
        case 2: return Xbyak::util::L2;
        case 3: return Xbyak::util::L3;
        default: return Xbyak::util::CACHE_UNKNOWN;
    }
}

// Find a representative logical CPU of the specified core type
// Returns the first CPU found with the specified type, or 0 if not found
size_t find_representative_cpu(Xbyak::util::CoreType target_type) {
    const auto &topo = get_topology_cache().topology;
    size_t num_cpus = topo.getLogicalCpuNum();

    for (size_t i = 0; i < num_cpus; i++) {
        const auto &logi = topo.getLogicalCpu(i);
        if (logi.coreType == target_type) { return i; }
    }

    // If not found, return 0 (first CPU)
    return 0;
}

// Calculate per-core cache size for a specific cache level and CPU index
// Accounts for cache sharing among logical cores
uint32_t calculate_per_core_cache(size_t cpu_index, int level) {
    const auto &topo = get_topology_cache().topology;

    Xbyak::util::CacheType cache_type = convert_cache_level(level);
    if (cache_type == Xbyak::util::CACHE_UNKNOWN) { return 0; }

    const auto &cache = topo.getCache(cpu_index, cache_type);
    if (cache.size == 0) { return 0; }

    // Get number of CPUs sharing this cache
    size_t sharing_cpus = cache.getSharedCpuNum();
    if (sharing_cpus == 0) {
        sharing_cpus = 1; // Assume private cache
    }

    return cache.size / sharing_cpus;
}
#endif // !__APPLE__

} // anonymous namespace

bool is_hybrid() {
#ifdef __APPLE__
    // xbyak::util::CpuTopology is not supported on macOS; assume non-hybrid.
    return false;
#else
    return get_topology_cache().topology.isHybrid();
#endif
}

Xbyak::util::CoreType get_core_type() {
    // These correspond to values returned in CPUID leaf 0x1A core-type field.
    constexpr uint32_t CPUID_CORE_TYPE_ATOM = 0x20; // Intel Atom / E-core
    constexpr uint32_t CPUID_CORE_TYPE_CORE = 0x40; // Intel Core / P-core
    uint32_t regs[4] = {0};

    // Get max basic CPUID leaf
    Xbyak::util::Cpu::getCpuidEx(0x0, 0, regs);
    uint32_t max_basic_leaf = regs[0];
    // If 0x1A is not supported, default to Performance (P-core)
    if (max_basic_leaf < 0x1A) return Xbyak::util::Performance;

    Xbyak::util::Cpu::getCpuidEx(0x1A, 0, regs);
    uint32_t core_type_field = (regs[0] >> 24) & 0xFF;
    switch (core_type_field) {
        case CPUID_CORE_TYPE_ATOM: return Xbyak::util::Efficient;
        case CPUID_CORE_TYPE_CORE: return Xbyak::util::Performance;
        default: return Xbyak::util::Performance;
    }
}

// Legacy implementation using Xbyak::util::Cpu CPUID methods
// This matches the original behavior without OS-specific topology
unsigned get_per_core_cache_size_legacy(int level) {
    if (level > 0 && (unsigned)level <= cpu().getDataCacheLevels()) {
        unsigned l = level - 1;
        return cpu().getDataCacheSize(l) / cpu().getCoresSharingDataCache(l);
    }
    return 0;
}

unsigned get_per_core_cache_size(int level, behavior_t btype) {
    // Validate level
    if (level < 1 || level > 3) { return 0; }

#ifdef __APPLE__
    // xbyak::util::CpuTopology is not supported on macOS.
    // Always fall back to legacy CPUID-based behavior.
    return get_per_core_cache_size_legacy(level);
#else
    // Handle legacy behavior
    if (btype == behavior_t::legacy) {
        return get_per_core_cache_size_legacy(level);
    }

    const auto &topo = get_topology_cache().topology;

    // For non-hybrid systems, all cores are the same
    if (!topo.isHybrid()) {
        // Use first CPU as representative
        return calculate_per_core_cache(0, level);
    }

    // For hybrid systems, find representative CPUs for each core type
    size_t pcore_cpu = find_representative_cpu(Xbyak::util::Performance);
    size_t ecore_cpu = find_representative_cpu(Xbyak::util::Efficient);

    uint32_t pcore_size = calculate_per_core_cache(pcore_cpu, level);
    uint32_t ecore_size = calculate_per_core_cache(ecore_cpu, level);

    switch (btype) {
        case behavior_t::p_core: return pcore_size;

        case behavior_t::e_core: return ecore_size;

        case behavior_t::current: {
            // Determine current core type and return appropriate size
            Xbyak::util::CoreType current_ctype = get_core_type();
            return (current_ctype == Xbyak::util::Performance) ? pcore_size
                                                               : ecore_size;
        }

        case behavior_t::min: return (std::min)(pcore_size, ecore_size);

        case behavior_t::max: return (std::max)(pcore_size, ecore_size);

        default: return get_per_core_cache_size_legacy(level);
    }
#endif // __APPLE__
}

} // namespace platform
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
