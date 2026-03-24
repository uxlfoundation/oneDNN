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

// Filter used when searching for Efficient cores to distinguish between
// regular E-cores (those sharing an L3) and LP E-cores (no L3 present).
// For non-Efficient core types this filter is ignored.
enum class l3_filter_t {
    any, // no L3 filtering
    with_l3, // only cores that have a non-zero L3 (regular E-cores)
    without_l3, // only cores with L3 size == 0 (LP E-cores)
};

// Find the index of a representative logical CPU of the given core type.
// For Efficient cores, l3_filter optionally restricts matches by L3 presence.
// Returns the first matching CPU index, or SIZE_MAX if none is found.
size_t find_representative_cpu(Xbyak::util::CoreType target_type,
        l3_filter_t l3_filter = l3_filter_t::any) {
    const auto &topo = get_topology_cache().topology;
    size_t num_cpus = topo.getLogicalCpuNum();

    for (size_t i = 0; i < num_cpus; i++) {
        const auto &logi = topo.getLogicalCpu(i);
        if (logi.coreType != target_type) continue;

        // Apply L3 filter for Efficient cores only
        if (target_type == Xbyak::util::Efficient
                && l3_filter != l3_filter_t::any) {
            const auto &l3 = topo.getCache(i, Xbyak::util::L3);
            bool has_l3 = (l3.size > 0);
            if (l3_filter == l3_filter_t::with_l3 && !has_l3) continue;
            if (l3_filter == l3_filter_t::without_l3 && has_l3) continue;
        }
        return i;
    }
    return SIZE_MAX; // not found
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

// Probe whether the currently executing CPU has an L3 cache via CPUID leaf 0x4.
// CPUID.04H enumerates cache levels via ECX subleaf; EAX[4:0]==0 signals the
// end of the list, EAX[7:5] holds the cache level (3 == L3).
// This is safe to call on any x86 CPU that supports leaf 0x4 (all modern Intel).
bool current_cpu_has_l3() {
    for (uint32_t subleaf = 0; subleaf < 8; subleaf++) {
        uint32_t regs[4] = {0};
        Xbyak::util::Cpu::getCpuidEx(0x4, subleaf, regs);
        uint32_t cache_type = regs[0] & 0x1F; // EAX[4:0]: 0 = no more caches
        if (cache_type == 0) break;
        uint32_t cache_level = (regs[0] >> 5) & 0x7; // EAX[7:5]
        if (cache_level == 3) return true;
    }
    return false;
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

    // For hybrid systems, find representative CPUs for each core type.
    // LP E-cores (no L3) are distinguished from regular E-cores (with L3)
    // purely by L3 presence since xbyak reports both as Efficient.
    size_t pcore_cpu = find_representative_cpu(Xbyak::util::Performance);
    size_t lp_core_cpu = find_representative_cpu(
            Xbyak::util::Efficient, l3_filter_t::with_l3);
    size_t lpe_core_cpu = find_representative_cpu(
            Xbyak::util::Efficient, l3_filter_t::without_l3);

    // Fall back to the first CPU when a core type is not present
    if (pcore_cpu == SIZE_MAX) pcore_cpu = 0;
    if (lp_core_cpu == SIZE_MAX) lp_core_cpu = 0;
    // lpe_core_cpu stays SIZE_MAX when there are no LP E-cores — callers
    // that request lpe_core will get 0 in that case (no such cache level).

    uint32_t pcore_size = calculate_per_core_cache(pcore_cpu, level);
    uint32_t lp_core_size = calculate_per_core_cache(lp_core_cpu, level);
    uint32_t lpe_core_size = (lpe_core_cpu != SIZE_MAX)
            ? calculate_per_core_cache(lpe_core_cpu, level)
            : 0;

    switch (btype) {
        case behavior_t::p_core: return pcore_size;

        case behavior_t::lp_core: return lp_core_size;

        // Returns 0 for any cache level that LP E-cores lack (typically L3)
        case behavior_t::lpe_core: return lpe_core_size;

        case behavior_t::current: {
            // Determine current core type and return appropriate size.
            // LP E-cores report as Efficient in CPUID leaf 0x1A; distinguish
            // them from regular E-cores by probing the current CPU's L3 via
            // CPUID leaf 0x4, which is per-core on Intel hybrid CPUs.
            Xbyak::util::CoreType current_ctype = get_core_type();
            if (current_ctype == Xbyak::util::Performance) return pcore_size;
            if (lpe_core_cpu != SIZE_MAX && !current_cpu_has_l3())
                return lpe_core_size;
            return lp_core_size;
        }

        case behavior_t::min: {
            uint32_t m = (std::min)(pcore_size, lp_core_size);
            if (lpe_core_cpu != SIZE_MAX && lpe_core_size > 0)
                m = (std::min)(m, lpe_core_size);
            return m;
        }

        case behavior_t::max: {
            uint32_t m = (std::max)(pcore_size, lp_core_size);
            if (lpe_core_cpu != SIZE_MAX && lpe_core_size > 0)
                m = (std::max)(m, lpe_core_size);
            return m;
        }

        default: return get_per_core_cache_size_legacy(level);
    }
#endif // __APPLE__
}

bool has_lpe_core() {
#ifdef __APPLE__
    return false;
#else
    if (!get_topology_cache().topology.isHybrid()) return false;
    return find_representative_cpu(
                   Xbyak::util::Efficient, l3_filter_t::without_l3)
            != SIZE_MAX;
#endif
}

} // namespace platform
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
