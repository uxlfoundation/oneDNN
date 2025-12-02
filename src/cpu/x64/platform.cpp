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

#include <algorithm>
#include <set>
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/platform.hpp"

// if this is defined, the num_sharing_cores field in cache_info_t
// will count only physical cores, not logical cores (SMT)
#define COUNT_ONLY_PHYSICAL_CORES 1

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace platform {
bool is_hybrid() {
    // currently only Intel is using hybrid architecture
    if (!x64::cpu().has(Xbyak::util::Cpu::tINTEL)) return false;
    // check CPUID.07H:EDX[15] (Hybrid bit)
    uint32_t data_7[4] = {0};
    Xbyak::util::Cpu::getCpuidEx(7, 0, data_7);
    const bool hybrid_flag = (data_7[3] & (1u << 15)) != 0;
    return hybrid_flag;
}

core_type get_core_type() {
    // These correspond to values returned in CPUID leaf 0x1A core-type field.
    constexpr uint32_t CPUID_CORE_TYPE_ATOM = 0x20; // Intel Atom / E-core
    constexpr uint32_t CPUID_CORE_TYPE_CORE = 0x40; // Intel Core / P-core
    uint32_t regs[4] = {0};

    // Get max basic CPUID leaf
    Xbyak::util::Cpu::getCpuidEx(0x0, 0, regs);
    uint32_t max_basic_leaf = regs[0];
    // If 0x1A is not supported, default to p_core
    if (max_basic_leaf < 0x1A) return core_type::p_core;

    Xbyak::util::Cpu::getCpuidEx(0x1A, 0, regs);
    uint32_t core_type_field = (regs[0] >> 24) & 0xFF;
    switch (core_type_field) {
        case CPUID_CORE_TYPE_ATOM: return core_type::e_core;
        case CPUID_CORE_TYPE_CORE: return core_type::p_core;
        default: return core_type::p_core;
    }
}

// Assumption each core type on a system is homogeneous in terms of cache
// topology e.g. all P-cores have the same cache topology, all E-cores have the
// same cache topology this is true for all Intel hybrid CPUs so far
// (Alder Lake, Raptor Lake, Lunar Lake)
struct cache_info_t {
    uint8_t level; // cache level (0 - L1i, 1 - L1d, 2 - L2, 3 - L3, etc)
    uint32_t size; // cache size in bytes
    uint32_t num_sharing_cores; // number of cores sharing this cache
    core_type ctype; // core type (used for hybrid CPUs)
};

// Cache topology for hybrid CPUs
// For non-hybrid CPUs, the entries only the first entries should be used
//    e.g. caches[0..3] - L1i, L1d, L2, L3
//    caches[4..7] expected to be a copy of caches[0..3] but not guaranteed
// For hybrid CPUs with 2 core types (e.g. Alder Lake) the entries are as follows:
//    caches[0..3] - L1i, L1d, L2, L3 P-core
//    caches[4..7] - L1i, L1d, L2, L3 E-core
//
// Currently the L1 instruction cache (L1i) entries are not populated by any
// of the platform initialization functions, thus, caches[0] and caches[4]
// will have size 0.
// TODO: consider populating L1i cache or remove it from the topology making
// the 0 entry the L1d cache.
struct cache_topology_t {
    static constexpr size_t max_cache_levels = 4;
    static constexpr size_t max_core_types = 2;
    cache_info_t caches[max_cache_levels * max_core_types];
    bool is_hybrid;
    const cache_info_t &get_cache(
            int level, core_type ctype = core_type::default_core) const {
        size_t type_idx = (ctype == core_type::p_core) ? 0
                : (ctype == core_type::e_core)         ? 1
                                                       : 0;
        // Validate level and computed index to avoid out-of-bounds access.
        const size_t lvl = static_cast<size_t>(level);
        const size_t total = max_cache_levels * max_core_types;
        size_t idx = type_idx * max_cache_levels + lvl;
        if (lvl >= max_cache_levels || idx >= total) {
            // Fallback to a safe, well-defined element
            // (L1i of default core) on invalid input.
            // Currently the L1i cache entry is not populated,
            // so this will return an element with cache size 0.
            return caches[0];
        }
        return caches[idx];
    }
};

// If this is not a hybrid system, and the current core is a P-core,
// copy P-core cache info to E-core slots
// else if current core is E-core, copy E-core info to P-core slots
// This is to make sure that non-hybrid systems have both core types populated
// with the same cache info.
void copy_cache_topology_non_hybrid(cache_topology_t &cache_topology) {
    core_type ctype = get_core_type();
    if (ctype == core_type::p_core) {
        for (size_t level = 0; level < cache_topology_t::max_cache_levels;
                level++) {
            size_t p_idx = 0 * cache_topology_t::max_cache_levels + level;
            size_t e_idx = 1 * cache_topology_t::max_cache_levels + level;
            if (cache_topology.caches[p_idx].level > 0) {
                cache_topology.caches[e_idx] = cache_topology.caches[p_idx];
                cache_topology.caches[e_idx].ctype = core_type::e_core;
            }
        }
    } else {
        for (size_t level = 0; level < cache_topology_t::max_cache_levels;
                level++) {
            size_t e_idx = 1 * cache_topology_t::max_cache_levels + level;
            size_t p_idx = 0 * cache_topology_t::max_cache_levels + level;
            if (cache_topology.caches[e_idx].level > 0) {
                cache_topology.caches[p_idx] = cache_topology.caches[e_idx];
                cache_topology.caches[p_idx].ctype = core_type::p_core;
            }
        }
    }
}

//------------------Start CPUID cache topology code-----------------------------

void populate_cache_topology_from_cpuid(cache_topology_t &cache_topology) {
    static constexpr uint32_t MAX_SUBLEAF_GUARD = 16;
    static constexpr uint32_t CACHE_PARAMETERS_LEAF = 0x04;
    uint32_t regs[4] = {0};

    core_type ctype = get_core_type();
    for (uint32_t subleaf = 0; subleaf < MAX_SUBLEAF_GUARD; ++subleaf) {
        Xbyak::util::Cpu::getCpuidEx(CACHE_PARAMETERS_LEAF, subleaf, regs);
        // EAX[4:0] Cache Type Field
        uint32_t cache_type = regs[0] & 0x1F;
        if (cache_type == 0) break; // no more caches

        if (cache_type != 1 && cache_type != 3)
            continue; // skip non-data/unified caches
        // EAX[7:5] Cache Level
        int level = (regs[0] >> 5) & 0x7;

        // calculate cache size
        // EBX[31:22] + 1 Ways of associativity
        uint32_t ways = ((regs[1] >> 22) & 0x3FF) + 1;
        // EBX[21:12] + 1 Physical Line partitions
        uint32_t partitions = ((regs[1] >> 12) & 0x3FF) + 1;
        // EBX[11:0] + 1 System Coherency line size
        uint32_t line_size = (regs[1] & 0xFFF) + 1;
        // EBX[31:00] + 1 Number of Sets
        uint32_t sets = regs[2] + 1;
        uint32_t cache_size = ways * partitions * line_size * sets;

        // number of sharing cores
        // EAX[25:14] + 1 Max number of sharing cores (typically larger than actual count)
        uint32_t max_cores_sharing = ((regs[0] >> 14) & 0xFFF) + 1;

        int type_idx = (ctype == core_type::p_core) ? 0 : 1;
        size_t idx = type_idx * cache_topology_t::max_cache_levels
                + static_cast<size_t>(level);
        const size_t total_slots = cache_topology_t::max_cache_levels
                * cache_topology_t::max_core_types;
        if (idx >= total_slots) continue;

        if (cache_topology.caches[idx].size == 0) {
            cache_info_t info;
            info.level = static_cast<uint8_t>(level);
            info.size = cache_size;
            info.num_sharing_cores = max_cores_sharing;
            info.ctype = ctype;
            cache_topology.caches[idx] = info;
        }
    }

    // refine sharing counts using topology leaf if available
    // Get Basic CPUID Information leaf
    Xbyak::util::Cpu::getCpuidEx(0x0, 0x0, regs);
    uint32_t max_basic_leaf = regs[0];
    uint32_t topo_leaf = 0;
    // TODO: Since we can can use either leaf 0x1F or 0x0B
    // should we just default to 0x0B? Since the only levels we care about
    // are SMT (level 1) and Core (level 3) which are supported in 0x0B
    // and 0x1F would it make since to just use 0x0B?
    // Assumption was that V2 of the topology leaf would be more accurate
    // but both leaves should be equivalent for our purposes.
    static constexpr uint32_t EXT_TOPOLOGY_LEAF_V1 = 0x0B;
    static constexpr uint32_t EXT_TOPOLOGY_LEAF_V2 = 0x1F;
    if (max_basic_leaf >= EXT_TOPOLOGY_LEAF_V2) {
        topo_leaf = EXT_TOPOLOGY_LEAF_V2;
    } else if (max_basic_leaf >= EXT_TOPOLOGY_LEAF_V1) {
        topo_leaf = EXT_TOPOLOGY_LEAF_V1;
    }

    // Use Extended Topology Enumeration leaf
    // to refine number of sharing cores if available
#if COUNT_ONLY_PHYSICAL_CORES
    // First pass: determine threads per core from SMT level
    uint32_t threads_per_core = 1;
    if (topo_leaf != 0) {
        for (uint32_t subleaf = 0; subleaf < MAX_SUBLEAF_GUARD; ++subleaf) {
            Xbyak::util::Cpu::getCpuidEx(topo_leaf, subleaf, regs);
            uint32_t level_type = (regs[2] >> 8) & 0xFF;
            if (level_type == 0) break;
            if (level_type == 1) { // SMT level
                threads_per_core = regs[1] & 0xFFFF; // EBX[15:0]
                if (threads_per_core == 0) threads_per_core = 1;
                break;
            }
        }
    }
#endif

    if (topo_leaf != 0) {
        for (uint32_t subleaf = 0; subleaf < MAX_SUBLEAF_GUARD; ++subleaf) {
            Xbyak::util::Cpu::getCpuidEx(topo_leaf, subleaf, regs);
            uint32_t level_type
                    = (regs[2] >> 8) & 0xFF; // CPUID(0x1F).ECX[15:8]
            if (level_type == 0) break; // no more levels
            if (level_type == 1 || level_type == 2) {
                int level = 1; // assume SMT level corresponds to L1d
                if (level_type == 2) level = 3; // Core level corresponds to L3
                int type_idx = (ctype == core_type::p_core) ? 0 : 1;
                size_t idx = type_idx * cache_topology_t::max_cache_levels
                        + static_cast<size_t>(level);
                const size_t total_slots = cache_topology_t::max_cache_levels
                        * cache_topology_t::max_core_types;
                if (idx >= total_slots) continue;

                if (cache_topology.caches[idx].size != 0) {
                    uint32_t logical_processors_at_level = regs[1]; // EBX
                    uint32_t existing
                            = cache_topology.caches[idx].num_sharing_cores;
                    uint32_t newval
                            = (std::min)(existing, logical_processors_at_level);
#if COUNT_ONLY_PHYSICAL_CORES
                    // Divide by threads per core to get physical core count
                    if (threads_per_core > 1) {
                        newval = (newval + threads_per_core - 1) / threads_per_core;
                    }
#endif
                    if (newval != existing)
                        cache_topology.caches[idx].num_sharing_cores = newval;
                }
            }
        }
    }
}

// The init_cache_topology_cpuid function is a fallback method to initialize
// the cache topology using only CPUID instructions. This method is less
// accurate than the Windows API or Linux methods, but can be used on
// systems where those methods are not available.
//
// The function works as follows:
// 1. Check if the system is hybrid using CPUID.07H:EDX[15] see is_hybrid()
// 2. Use OS specific methods to set thread affinity to each logical core:
// 3. populate the cache_map using the CPUID for each core: see populate_cache_map_cpuid()
//   3a. Use CPUID(0x04) to get the cache information for each core
//   3b. Use CPUID(0x1A) to get the core type of the current core see get_core_type()
//   3c. if available Use CPUID(0x0B) or CPUID(0x1F) to refine the number of sharing cores
// 4. If no caches were detected, populate with guessed defaults
// 5. If the system is not hybrid, copy the P-core cache information to the E-core entries
//
// This function is considered the fallback method and may give incorrect
// results. The actual cache topology should be obtained through the OS interfaces.
void init_cache_topology_cpuid(cache_topology_t &cache_topology) {
    auto guess = [](int level) {
        switch (level) {
            case 0: return 32U * 1024; // L1 instruction cache
            case 1: return 32U * 1024; // L1 data cache
            case 2: return 512U * 1024; // L2 cache
            case 3: return 1024U * 1024; // L3 cache
            default: return 0U;
        }
    };

    cache_topology.is_hybrid = is_hybrid();
    if (cache_topology.is_hybrid) {
#if defined(_WIN32)
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);

        DWORD num_processors = sysinfo.dwNumberOfProcessors;
        for (DWORD cpu = 0; cpu < num_processors; cpu++) {
            HANDLE current_thread = GetCurrentThread();
            DWORD_PTR cpu_mask = 1ULL << cpu;
            DWORD_PTR oldMask = SetThreadAffinityMask(current_thread, cpu_mask);

            if (oldMask != 0) {
                populate_cache_topology_from_cpuid(cache_topology);
            }
            SetThreadAffinityMask(current_thread, oldMask);
        }
#elif defined(__linux__)
        int num_processors = (int)sysconf(_SC_NPROCESSORS_CONF);
        if (num_processors < 1) num_processors = 1;

        cpu_set_t original_mask;
        CPU_ZERO(&original_mask);
        sched_getaffinity(0, sizeof(cpu_set_t), &original_mask);

        for (int cpu = 0; cpu < num_processors; cpu++) {
            cpu_set_t cpu_mask;
            CPU_ZERO(&cpu_mask);
            CPU_SET(cpu, &cpu_mask);
            if (sched_setaffinity(0, sizeof(cpu_set_t), &cpu_mask) == 0) {
                populate_cache_topology_from_cpuid(cache_topology);
            }
        }
        // Restore original affinity mask
        sched_setaffinity(0, sizeof(cpu_set_t), &original_mask);
#endif
    } else {
        populate_cache_topology_from_cpuid(cache_topology);
    }

    // Check whether we managed to populate any entries
    bool any_detected = false;
    for (size_t i = 0; i < cache_topology_t::max_cache_levels
                    * cache_topology_t::max_core_types;
            ++i) {
        if (cache_topology.caches[i].size != 0) {
            any_detected = true;
            break;
        }
    }

    // If we couldn't detect any caches, populate with defaults from guess function
    if (!any_detected) {
        for (int type_idx = 0; type_idx < (int)cache_topology_t::max_core_types;
                ++type_idx) {
            for (size_t level = 0; level < cache_topology_t::max_cache_levels;
                    ++level) {
                size_t idx
                        = type_idx * cache_topology_t::max_cache_levels + level;
                cache_info_t info;
                info.level = static_cast<uint8_t>(level);
                info.size = guess(static_cast<int>(level));
                info.num_sharing_cores
                        = 1; // assume 1 cache per-core by default
                info.ctype = (type_idx == 0) ? core_type::p_core
                                             : core_type::e_core;
                cache_topology.caches[idx] = info;
            }
        }
    }

    // If this is not a hybrid system, and the current core is a P-core,
    // copy P-core cache info to E-core slots
    // else if current core is E-core, copy E-core info to P-core slots
    if (!cache_topology.is_hybrid) {
        copy_cache_topology_non_hybrid(cache_topology);
    }
}
//------------------End CPUID cache topology code-------------------------------
//------------------Start windows specific cache topology code------------------
#if defined(_WIN32)
// inline helper to count bits in KAFFINITY processor mask
inline unsigned count_bits_in_mask(KAFFINITY mask) {
#if defined(_MSC_VER)
    return static_cast<unsigned>(__popcnt64(mask));
#else
    unsigned count = 0;
    while (mask) {
        count += (mask & 1);
        mask >>= 1;
    }
    return count;
#endif
}

// Helper function to determine core type from processor info
// Returns p_core for performance cores, e_core for efficiency cores
//
// **Helper function assumes that is_hybrid() has already returned true**
//
// If not hybrid, this function is unreliable it can return either core type
// depending on if the EfficiencyClass field is present or not.
// TODO: revisit this helper function considering that EfficiencyClass can be
// `0` for for P-cores when there are only P-cores present in the system.
core_type get_core_type_from_processor_info(
        const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *info) {
    // For hybrid CPUs, we need to distinguish between P-cores and E-cores
    // On Intel 12th gen+, EfficiencyClass in PROCESSOR_RELATIONSHIP indicates core type:
    //   0 = E-core (efficiency)
    //   1 = P-core (performance)
    if (info->Relationship == RelationProcessorCore) {
        const auto &proc = info->Processor;
        // EfficiencyClass is only available on Windows 10 1903+ / Windows 11
#if defined(NTDDI_WIN10_VB) && NTDDI_VERSION >= NTDDI_WIN10_VB
        if (proc.EfficiencyClass == 0)
            return core_type::e_core;
        else if (proc.EfficiencyClass == 1)
            return core_type::p_core;
#endif
    }
    // Default to p_core if we can't determine the core_type
    return core_type::p_core;
}

// Function to initialize cache topology using Windows APIs
// 1. Query CPUID to check if system is hybrid
// 2. Use GetLogicalProcessorInformationEx to enumerate all processor cores
//    and find their core type (P-core or E-core)
// 3. Use GetLogicalProcessorInformationEx to enumerate all caches
// 4. For each cache, determine which core types it is associated with
//    by checking the cache's processor affinity mask against each core's mask
// 5. Populate cache_map with cache sizes and sharing info for each core type
// 6. Populate cache_topology_t structure from the cache_map.
// 7. If not hybrid, copy P-core cache info to E-core slots
void init_cache_topology_windows(cache_topology_t &cache_topology) {
    // Determine if system is hybrid
    cache_topology.is_hybrid = is_hybrid();

    // Query buffer size
    DWORD bufferSize = 0;
    GetLogicalProcessorInformationEx(RelationAll, nullptr, &bufferSize);
    if (bufferSize == 0) {
        // Fallback to CPUID-based method if Windows API fails
        init_cache_topology_cpuid(cache_topology);
        return;
    }

    // Allocate buffer
    std::vector<BYTE> buffer(bufferSize);
    auto *info_base_ptr
            = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
                    buffer.data());

    if (!GetLogicalProcessorInformationEx(
                RelationAll, info_base_ptr, &bufferSize)) {
        // Fallback to CPUID-based method if Windows API fails
        init_cache_topology_cpuid(cache_topology);
        return;
    }

    // First pass: collect all processor cores with their types and masks
    // this will create a list of all cores with their core type and group affinity mask
    std::vector<std::pair<core_type, GROUP_AFFINITY>> core_info;

    DWORD offset = 0;
    while (offset < bufferSize) {
        auto *info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
                reinterpret_cast<BYTE *>(info_base_ptr) + offset);
        if (info->Relationship == RelationProcessorCore) {
            core_type ctype = core_type::p_core;
            if (cache_topology.is_hybrid)
                ctype = get_core_type_from_processor_info(info);

            // Store each group mask for this core type
            // Note: Each RelationProcessorCore entry represents one physical core,
            // and its mask may contain multiple bits if hyperthreading is enabled
            for (WORD i = 0; i < info->Processor.GroupCount; i++) {
                core_info.push_back({ctype, info->Processor.GroupMask[i]});
            }
        }
        offset += info->Size;
    }

    // Second pass: for each cache, determine which core types use which caches.
    // This is a little backwards since we have to check each cache against all cores
    // to see which core types intersect with the cache's processor affinity mask.
    // This this will map each cache to the core_type that uses it.
    offset = 0;
    while (offset < bufferSize) {
        auto *info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
                buffer.data() + offset);
        if (info->Relationship == RelationCache) {
            const auto &cache = info->Cache;
            if (cache.Type == CacheData || cache.Type == CacheUnified) {
                // Check which core types intersect with this cache's mask
                // Note: Considered using a std::set but the number of core types
                // is small enough that simple bool flags are sufficient.
                bool has_p_core = false;
                bool has_e_core = false;

                for (size_t i = 0; i < core_info.size(); ++i) {
                    core_type ctype = core_info[i].first;
                    GROUP_AFFINITY core_mask = core_info[i].second;
                    // Check if this cache intersects with this core's processors
                    if (cache.GroupMask.Group == core_mask.Group
                            && (cache.GroupMask.Mask & core_mask.Mask) != 0) {
                        if (ctype == core_type::p_core)
                            has_p_core = true;
                        else if (ctype == core_type::e_core)
                            has_e_core = true;
                    }
                }
                // Normalize level and validate.
                int level = static_cast<int>(cache.Level);
                if (level < 0
                        || level > (int)cache_topology_t::max_cache_levels) {
                    offset += info->Size;
                    continue;
                }

                // Count sharing cores for this cache descriptor.
#if COUNT_ONLY_PHYSICAL_CORES
                // Count physical cores: each entry in core_info represents
                // one physical core. Check how many physical cores intersect
                // with this cache's mask.
                unsigned sharing_cores = 0;
                for (size_t i = 0; i < core_info.size(); ++i) {
                    if (cache.GroupMask.Group == core_info[i].second.Group
                            && (cache.GroupMask.Mask & core_info[i].second.Mask)
                                    != 0) {
                        sharing_cores++;
                    }
                }
#else
                // Count logical processors (includes hyperthreads)
                unsigned sharing_cores
                        = count_bits_in_mask(cache.GroupMask.Mask);
#endif

                // Helper to set or merge an entry in cache_topology.
                auto set_cache_topology_entry = [&](core_type ctype) {
                    int type_idx = (ctype == core_type::p_core) ? 0 : 1;
                    size_t idx = type_idx * cache_topology_t::max_cache_levels
                            + static_cast<size_t>(level);
                    const size_t total_slots
                            = cache_topology_t::max_cache_levels
                            * cache_topology_t::max_core_types;
                    if (idx >= total_slots) return;

                    // If slot not yet populated (size == 0), set it.
                    if (cache_topology.caches[idx].size == 0) {
                        cache_info_t info;
                        info.level = static_cast<uint8_t>(level);
                        info.size = cache.CacheSize;
                        info.num_sharing_cores = sharing_cores;
                        info.ctype = ctype;
                        cache_topology.caches[idx] = info;
                    } else {
                        // Merge sharing information conservatively.
                        cache_topology.caches[idx].num_sharing_cores
                                = (std::max)(cache_topology.caches[idx]
                                                     .num_sharing_cores,
                                        sharing_cores);
                    }
                };

                // Assign to p_core/e_core based on which core types are present.
                if (has_p_core) set_cache_topology_entry(core_type::p_core);
                if (has_e_core) set_cache_topology_entry(core_type::e_core);
                // If neither was set, this is likely a shared cache: populate both
                // core slots unless already present.
                if (!has_p_core && !has_e_core) {
                    set_cache_topology_entry(core_type::p_core);
                    set_cache_topology_entry(core_type::e_core);
                }
            }
        }
        offset += info->Size;
    }

    // For non-hybrid systems the cache enumeration populates only the P-core
    // slots (all detected cores are treated as p_core see first pass above).
    // Copy those P-core entries into the E-core slots so callers asking for
    // e_core cache info still receive sensible values. If no P-core data was
    // populated, E-core slots are left unchanged.
    if (!cache_topology.is_hybrid) {
        for (size_t level = 0; level < cache_topology_t::max_cache_levels;
                level++) {
            size_t p_idx = 0 * cache_topology_t::max_cache_levels + level;
            size_t e_idx = 1 * cache_topology_t::max_cache_levels + level;
            if (cache_topology.caches[p_idx].level > 0) {
                cache_topology.caches[e_idx] = cache_topology.caches[p_idx];
                cache_topology.caches[e_idx].ctype = core_type::e_core;
            }
        }
    }
}
//------------------End windows specific cache topology code--------------------
//------------------Start linux specific cache topology code--------------------
#elif defined(__linux__)
// Helper function to set CPU affinity on Linux
bool set_cpu_affinity_linux(int cpu) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    return sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) == 0;
#else
    return false;
#endif
}

// Helper function to restore original CPU affinity
void restore_cpu_affinity_linux(const cpu_set_t *original_affinity_mask) {
#ifdef __linux__
    sched_setaffinity(0, sizeof(cpu_set_t), original_affinity_mask);
#endif
}

// Function to initialize cache topology on Linux
// 1. Check if system is hybrid
// 2. For each CPU core, set affinity and run CPUID to determine core type
//   2a. Restore original affinity after checking all cores
// 3. Read cache information from /sys/devices/system/cpu/cpu*/cache
//   3a. get cache size, from /sys/devices/system/cpu/cpu*/cache/index*/size
//   3b. get cache level, from /sys/devices/system/cpu/cpu*/cache/index*/level
//   3c. get number of sharing cores, from /sys/devices/system/cpu/cpu*/cache/index*/shared_cpu_list
//       (parse the list to count number of cores sharing this cache)
//   3d. populate cache_info structure and store in map keyed by (core_type, level)
// 4. Populate cache_topology structure
// 5. If not hybrid, copy P-core cache info to E-core slots
// Note: this function assumes a maximum of 256 CPUs
void init_cache_topology_linux(cache_topology_t &cache_topology) {
    cache_topology.is_hybrid = is_hybrid();

    std::map<std::pair<core_type, int>, cache_info_t> cache_map;

    // Save original CPU affinity
    cpu_set_t original_affinity_mask;
    sched_getaffinity(0, sizeof(cpu_set_t), &original_affinity_mask);

    // First pass: identify all CPUs and their core types using CPUID
    std::map<int, core_type> cpu_core_types;

#if COUNT_ONLY_PHYSICAL_CORES
    // Build mapping from logical CPU to physical core ID using thread_siblings_list
    // The physical core ID is represented by the lowest CPU number in the sibling list
    std::map<int, int> logical_to_physical_core;
#endif

    // All char arrays are sized much larger than needed to avoid overflow.
    // TODO: investigate reducing size of path char buffers. for example
    //       `cpu_path` is 256 bytes but only needs to be about 30 bytes.
    //       The `online_path` only needs to be about 40 bytes, but it was
    //       increased to 512 bytes.  Similar larger numbers were used for
    //       other paths associated with cache information.

    // Read CPU information from /sys/devices/system/cpu/
    for (int cpu = 0; cpu < 256; cpu++) { // Check up to 256 CPUs
        char cpu_path[256];
        snprintf(cpu_path, sizeof(cpu_path), "/sys/devices/system/cpu/cpu%d",
                cpu);

        // Check if CPU directory exists by trying to access it
        char online_path[512];
        snprintf(online_path, sizeof(online_path), "%s/online", cpu_path);
        FILE *online_file = fopen(online_path, "r");
        if (!online_file) {
            // Try checking the cache directory instead for CPU0 (which might not have online file)
            char cache_check[512];
            snprintf(cache_check, sizeof(cache_check), "%s/cache", cpu_path);
            FILE *cache_dir = fopen(cache_check, "r");
            if (!cache_dir && cpu > 0) {
                break; // No more CPUs
            }
            if (cache_dir) fclose(cache_dir);
        } else {
            fclose(online_file);
        }

        core_type ctype = core_type::p_core; // Default assumption

        // For hybrid CPUs, detect core type using CPUID on each core
        if (cache_topology.is_hybrid) {
            // Set affinity to this specific CPU to run CPUID on it
            if (set_cpu_affinity_linux(cpu)) { ctype = get_core_type(); }
        }

        cpu_core_types[cpu] = ctype;

#if COUNT_ONLY_PHYSICAL_CORES
        // Read thread_siblings_list to identify which logical CPUs share a physical core
        char siblings_path[512];
        snprintf(siblings_path, sizeof(siblings_path),
                "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list",
                cpu);
        FILE *siblings_file = fopen(siblings_path, "r");
        if (siblings_file) {
            char siblings_str[256];
            if (fgets(siblings_str, sizeof(siblings_str), siblings_file)) {
                // Parse sibling list to find the minimum CPU ID (physical core ID)
                int min_cpu = cpu;
                char siblings_copy[256];
                strcpy(siblings_copy, siblings_str);
                char *token = strtok(siblings_copy, ",\n");
                while (token) {
                    while (*token == ' ') token++;
                    if (strchr(token, '-')) {
                        int start, end;
                        if (sscanf(token, "%d-%d", &start, &end) == 2) {
                            if (start < min_cpu) min_cpu = start;
                        }
                    } else {
                        int cpu_id;
                        if (sscanf(token, "%d", &cpu_id) == 1) {
                            if (cpu_id < min_cpu) min_cpu = cpu_id;
                        }
                    }
                    token = strtok(nullptr, ",\n");
                }
                logical_to_physical_core[cpu] = min_cpu;
            }
            fclose(siblings_file);
        } else {
            // If we can't read siblings, assume each logical CPU is its own physical core
            logical_to_physical_core[cpu] = cpu;
        }
#endif
    }

    if (cpu_core_types.empty()) {
        // Failed to detect any CPUs, fallback to CPUID-based method
        init_cache_topology_cpuid(cache_topology);
        return;
    }

    // Restore original CPU affinity
    restore_cpu_affinity_linux(&original_affinity_mask);

    // Second pass: read cache information for each CPU with detected core types
    for (std::map<int, core_type>::const_iterator it = cpu_core_types.begin();
            it != cpu_core_types.end(); ++it) {
        int cpu = it->first;
        core_type ctype = it->second;

        char cache_base_path[256];
        snprintf(cache_base_path, sizeof(cache_base_path),
                "/sys/devices/system/cpu/cpu%d/cache", cpu);

        // Check cache levels (typically index0=L1i, index1=L1d, index2=L2, index3=L3)
        for (int cache_idx = 0; cache_idx < 10; cache_idx++) {
            char cache_path[512];
            snprintf(cache_path, sizeof(cache_path), "%s/index%d",
                    cache_base_path, cache_idx);

            // Check if this cache index exists
            char size_path[768];
            snprintf(size_path, sizeof(size_path), "%s/size", cache_path);

            FILE *size_file = fopen(size_path, "r");
            if (!size_file) {
                continue; // This cache index doesn't exist
            }

            // Read cache size
            char size_str[64];
            if (!fgets(size_str, sizeof(size_str), size_file)) {
                fclose(size_file);
                continue;
            }
            fclose(size_file);

            // Parse size (format like "32K", "256K", "8M")
            uint32_t cache_size = 0;
            int size_val;
            char size_unit;
            if (sscanf(size_str, "%d%c", &size_val, &size_unit) >= 1) {
                cache_size = size_val;
                if (size_unit == 'K' || size_unit == 'k') {
                    cache_size *= 1024;
                } else if (size_unit == 'M' || size_unit == 'm') {
                    cache_size *= 1024 * 1024;
                }
            }

            if (cache_size == 0) continue;

            // Read cache level
            char level_path[768];
            snprintf(level_path, sizeof(level_path), "%s/level", cache_path);

            FILE *level_file = fopen(level_path, "r");
            int cache_level = 0;
            if (level_file) {
                if (fscanf(level_file, "%d", &cache_level) != 1) {
                    // Failed to read cache level, use default 0
                    cache_level = 0;
                }
                fclose(level_file);
            }

            // Validate cache level
            if (cache_level < 0
                    || cache_level > (int)cache_topology_t::max_cache_levels) {
                continue;
            }

            // Read cache type to filter for data/unified caches
            char type_path[768];
            snprintf(type_path, sizeof(type_path), "%s/type", cache_path);

            FILE *type_file = fopen(type_path, "r");
            char cache_type_str[64] = {0};
            if (type_file) {
                if (fgets(cache_type_str, sizeof(cache_type_str), type_file)
                        != nullptr) {
                    // Remove newline
                    char *newline = strchr(cache_type_str, '\n');
                    if (newline) *newline = '\0';
                } else {
                    // Failed to read cache type, default to empty string.
                    cache_type_str[0] = '\0';
                }
                fclose(type_file);
            }

            // Only process Data and Unified caches (skip Instruction)
            if (strstr(cache_type_str, "Instruction") != nullptr) { continue; }

            // Read shared CPU list to determine sharing
            char shared_path[768];
            snprintf(shared_path, sizeof(shared_path), "%s/shared_cpu_list",
                    cache_path);

            uint32_t num_sharing_cores = 1; // Default to 1
            FILE *shared_file = fopen(shared_path, "r");
            if (shared_file) {
                char shared_str[256];
                if (fgets(shared_str, sizeof(shared_str), shared_file)) {
                    // Parse shared_cpu_list (format like "0-3", "0,2,4")
#if COUNT_ONLY_PHYSICAL_CORES
                    // Count unique physical cores sharing this cache
                    std::set<int> physical_cores_sharing;
                    char shared_str_copy[256];
                    strcpy(shared_str_copy, shared_str);
                    char *token = strtok(shared_str_copy, ",\n");
                    while (token) {
                        while (*token == ' ') token++;
                        // Range format like "0-3"
                        if (strchr(token, '-')) {
                            int start, end;
                            if (sscanf(token, "%d-%d", &start, &end) == 2) {
                                for (int logical_cpu = start; logical_cpu <= end;
                                        ++logical_cpu) {
                                    auto it = logical_to_physical_core.find(
                                            logical_cpu);
                                    if (it != logical_to_physical_core.end()) {
                                        physical_cores_sharing.insert(it->second);
                                    }
                                }
                            }
                        } else {
                            // Single CPU
                            int logical_cpu;
                            if (sscanf(token, "%d", &logical_cpu) == 1) {
                                auto it = logical_to_physical_core.find(
                                        logical_cpu);
                                if (it != logical_to_physical_core.end()) {
                                    physical_cores_sharing.insert(it->second);
                                }
                            }
                        }
                        token = strtok(nullptr, ",\n");
                    }
                    num_sharing_cores = physical_cores_sharing.size();
                    if (num_sharing_cores == 0) num_sharing_cores = 1;
#else
                    // Count logical processors (includes hyperthreads)
                    num_sharing_cores = 0;
                    char shared_str_copy[256];
                    strcpy(shared_str_copy, shared_str);
                    char *token = strtok(shared_str_copy, ",\n");
                    while (token) {
                        // Trim whitespace
                        while (*token == ' ')
                            token++;
                        if (strchr(token, '-')) {
                            // Range format like "0-3"
                            int start, end;
                            if (sscanf(token, "%d-%d", &start, &end) == 2) {
                                num_sharing_cores += (end - start + 1);
                            }
                        } else {
                            // Single CPU
                            num_sharing_cores++;
                        }
                        token = strtok(nullptr, ",\n");
                    }
#endif
                }
                fclose(shared_file);
            }

            // Directly write into cache_topology if size == 0, otherwise merge
            int type_idx = (ctype == core_type::p_core) ? 0 : 1;
            size_t idx = type_idx * cache_topology_t::max_cache_levels
                    + static_cast<size_t>(cache_level);
            const size_t total_slots = cache_topology_t::max_cache_levels
                    * cache_topology_t::max_core_types;
            if (idx >= total_slots) continue;

            if (cache_topology.caches[idx].size == 0) {
                cache_info_t info;
                info.level = static_cast<uint8_t>(cache_level);
                info.size = cache_size;
                info.num_sharing_cores = num_sharing_cores;
                info.ctype = ctype;
                cache_topology.caches[idx] = info;
            } else {
                // Merge sharing information conservatively.
                cache_topology.caches[idx].num_sharing_cores = (std::max)(
                        cache_topology.caches[idx].num_sharing_cores,
                        num_sharing_cores);
            }
        }
    }

    // TODO do we need a fallback? It is unlikely that no caches were detected
    // Check whether we managed to populate any entries
    bool any_detected = false;
    for (size_t i = 0; i < cache_topology_t::max_cache_levels
                    * cache_topology_t::max_core_types;
            ++i) {
        if (cache_topology.caches[i].size != 0) {
            any_detected = true;
            break;
        }
    }

    if (!any_detected) {
        // Failed to detect any caches, fallback to CPUID-based method
        init_cache_topology_cpuid(cache_topology);
        return;
    }

    // If not hybrid, and core_type is P-core copy P-core data to E-core slots
    // if not hybrid, and core_type is E-core copy E-core data to P-core slots
    if (!cache_topology.is_hybrid) {
        copy_cache_topology_non_hybrid(cache_topology);
    }
}
#endif
//------------------End linux specific cache topology code----------------------

// Lazy initialization cache topology
// This function is thread-safe and will only initialize the cache_topology
// once, on the first call. Subsequent calls will return immediately.
// The functions init_cache_topology_windows/linux/cpuid have no thread-safety
// guarantees and are intended to only be called once from this function.
static std::once_flag g_cache_topology_once_flag;
void init_cache_topology(cache_topology_t &cache_topology) {
    std::call_once(g_cache_topology_once_flag, [&cache_topology]() {
        // Initialize global cache topology to safe defaults:
        // - non-hybrid
        // - all cache entries zeroed
        // - default core type
        cache_topology.is_hybrid = false;
        size_t max_entries = cache_topology_t::max_cache_levels
                * cache_topology_t::max_core_types;
        for (size_t i = 0; i < max_entries; ++i) {
            cache_topology.caches[i].level = 0;
            cache_topology.caches[i].size = 0;
            cache_topology.caches[i].num_sharing_cores = 0;
            cache_topology.caches[i].ctype = core_type::default_core;
        }
#if defined(_WIN32)
        init_cache_topology_windows(cache_topology);
#elif defined(__linux__)
        init_cache_topology_linux(cache_topology);
#else
        // Fallback: use CPUID-based method
        init_cache_topology_cpuid(cache_topology);
#endif
// Enable debug printout
//TODO: George remove before final PR
#define DEBUG_PRINT_LEGACY_CACHE_SIZE 0
#define DEBUG_PRINT_CACHE_TOPOLOGY 0
#if DEBUG_PRINT_LEGACY_CACHE_SIZE
        {
            printf("Legacy per-core cache sizes:\n");
            // Print what the legacy per-core cache sizes would be for comparison.
            for (int lvl = 1; lvl < static_cast<int>(cache_topology_t::max_cache_levels); ++lvl) {
                printf("level=%u size=%u shared=%u\n",
                        (unsigned)lvl,
                        cpu().getDataCacheSize(lvl - 1),
                        cpu().getCoresSharingDataCache(lvl - 1)
                );
            }
        }
#endif // DEBUB_PRINT_LEGACY_CACHE_SIZE
#if DEBUG_PRINT_CACHE_TOPOLOGY
        {
            // Debug printout of the discovered global cache topology
            printf("Global cache topology initialized: is_hybrid=%d\n",
                    cache_topology.is_hybrid ? 1 : 0);

            for (size_t type_idx = 0;
                    type_idx < cache_topology_t::max_core_types; ++type_idx) {
                for (size_t level_slot = 0;
                        level_slot < cache_topology_t::max_cache_levels;
                        ++level_slot) {
                    size_t idx = type_idx * cache_topology_t::max_cache_levels
                            + level_slot;
                    const auto &ci = cache_topology.caches[idx];
                    if (ci.size == 0) continue; // skip empty slots

                    const char *ctype_str = (ci.ctype == core_type::p_core)
                            ? "p_core"
                            : (ci.ctype == core_type::e_core) ? "e_core"
                                                              : "default_core";

                    printf("cache[%zu] type_idx=%zu slot=%zu: level=%u size=%u "
                           "shared=%u ctype=%s\n",
                            idx, type_idx, level_slot, (unsigned)ci.level,
                            (unsigned)ci.size, (unsigned)ci.num_sharing_cores,
                            ctype_str);
                }
            }
        }
#endif  // DEBUG_PRINT_CACHE_TOPOLOGY
    });
}

// This is the legacy implementation of get_per_core_cache_size
// which does not consider hybrid architectures. An only uses
// cpuid to get the cache size per core. Which can be inaccurate
// due to the 0x1A leaf often reporting the maximum number of
// cores sharing a cache instead of the actual number of cores
// sharing a cache. This is compensated by using the topology
// leaf (0x0B or 0x1F) to refine the number of sharing cores.
// That is limited to L1d and L3 caches only.
// The huge advantage of this legacy method is that it is simple
// and works on all x86 systems without requiring OS specific
// APIs or setting thread affinity.
static inline unsigned get_per_core_cache_size_legacy(int level) {
    //return get_per_core_cache_size(level);
    auto guess = [](int level) {
        switch (level) {
            case 1: return 32U * 1024;
            case 2: return 512U * 1024;
            case 3: return 1024U * 1024;
            default: return 0U;
        }
    };
    if (cpu().getDataCacheLevels() == 0) return guess(level);

    if (level > 0 && (unsigned)level <= cpu().getDataCacheLevels()) {
        unsigned l = level - 1;
        return cpu().getDataCacheSize(l) / cpu().getCoresSharingDataCache(l);
    } else
        return 0;
}

unsigned get_per_core_cache_size(int level, behavior_t btype) {
    // Lazy-initialization used to initialize the cache_topology
    // the first time get_per_core_cache_size is called.
    static cache_topology_t cache_topology;
    init_cache_topology(cache_topology);

    if (level < 1 || level > (int)cache_topology_t::max_cache_levels) {
        return 0;
    }

    auto pcore_cache = cache_topology.get_cache(level, core_type::p_core);

    // Calculate effective cache size per core (considering sharing)
    auto pcore_size = pcore_cache.size;
    if (pcore_cache.num_sharing_cores > 0) {
        pcore_size /= pcore_cache.num_sharing_cores;
    }

    if (!cache_topology.is_hybrid) {
        // For Non-hybrid system, all cores are assumed to be p-core
        return pcore_size;
    }

    auto ecore_cache = cache_topology.get_cache(level, core_type::e_core);

    auto ecore_size = ecore_cache.size;
    if (ecore_cache.num_sharing_cores > 0) {
        ecore_size /= ecore_cache.num_sharing_cores;
    }

    switch (btype) {
        case behavior_t::p_core: return pcore_size; break;
        case behavior_t::e_core: return ecore_size; break;
        case behavior_t::current:
            // Detect current core type using CPUID 0x1AH
            {
                core_type current_ctype = get_core_type();
                if (current_ctype == core_type::p_core) {
                    return pcore_size;
                } else {
                    return ecore_size;
                }
            }
        case behavior_t::min: return (std::min)(pcore_size, ecore_size); break;
        case behavior_t::max: return (std::max)(pcore_size, ecore_size); break;
        case behavior_t::legacy:
            // [[fallthrough]]; // fallthrough attribute is a C++17 feature
        default:
            // unknown behavior, fallback to non-hybrid behavior
            return get_per_core_cache_size_legacy(level);
            break;
    }
    return get_per_core_cache_size_legacy(level);
}

} // namespace platform
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
