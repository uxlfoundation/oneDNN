/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
* Copyright 2020-2024 FUJITSU LIMITED
* Copyright 2022-2024 Arm Ltd. and affiliates
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

#include <thread>

#include "cpu/platform.hpp"

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include <algorithm>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__GLIBC__)
#include <sched.h>
#endif
#endif

#if DNNL_X64
#include "cpu/x64/cpu_isa_traits.hpp"
#elif DNNL_AARCH64
#include "cpu/aarch64/cpu_isa_traits.hpp"
#if defined(DNNL_AARCH64_USE_ACL)
// For checking if fp16 isa is supported on the platform
#include "arm_compute/core/CPP/CPPTypes.h"
#endif
#endif

// For DNNL_X64 build we compute the timestamp using rdtsc. Use std::chrono for
// other builds.
#if !DNNL_X64
#include <chrono>
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace platform {

const char *get_isa_info() {
#if DNNL_X64
    return x64::get_isa_info();
#elif DNNL_AARCH64
    return aarch64::get_isa_info();
#else
    return "Generic";
#endif
}

dnnl_cpu_isa_t get_effective_cpu_isa() {
#if DNNL_X64
    return x64::get_effective_cpu_isa();
#elif DNNL_AARCH64
    return aarch64::get_effective_cpu_isa();
#else
    return dnnl_cpu_isa_default;
#endif
}

status_t set_max_cpu_isa(dnnl_cpu_isa_t isa) {
#if DNNL_X64
    return x64::set_max_cpu_isa(isa);
#else
    return status::unimplemented;
#endif
}

status_t set_cpu_isa_hints(dnnl_cpu_isa_hints_t isa_hints) {
#if DNNL_X64
    return x64::set_cpu_isa_hints(isa_hints);
#elif DNNL_AARCH64
    return status::success;
#else
    return status::unimplemented;
#endif
}

dnnl_cpu_isa_hints_t get_cpu_isa_hints() {
#if DNNL_X64
    return x64::get_cpu_isa_hints();
#else
    return dnnl_cpu_isa_no_hints;
#endif
}

bool prefer_ymm_requested() {
#if DNNL_X64
    const bool prefer_ymm = x64::get_cpu_isa_hints() == dnnl_cpu_isa_prefer_ymm;
    return prefer_ymm;
#else
    return false;
#endif
}

bool has_data_type_support(data_type_t data_type) {
    // Notice: see notes in header
    switch (data_type) {
        case data_type::bf16:
#if DNNL_X64
            return x64::mayiuse(x64::avx512_core)
                    || x64::mayiuse(x64::avx2_vnni_2);
#elif DNNL_PPC64
#if defined(USE_CBLAS) && defined(BLAS_HAS_SBGEMM) && defined(__MMA__)
            return true;
#endif
#elif DNNL_AARCH64
            return aarch64::mayiuse_bf16();
#else
            return false;
#endif
        case data_type::f16:
#if DNNL_X64
            return x64::mayiuse(x64::avx512_core_fp16)
                    || x64::mayiuse(x64::avx2_vnni_2);
#elif defined(DNNL_AARCH64_USE_ACL)
            return arm_compute::CPUInfo::get().has_fp16();
#else
            return false;
#endif
        case data_type::f8_e5m2:
        case data_type::f8_e4m3:
#if DNNL_X64
            return x64::mayiuse(x64::avx512_core_fp16);
#else
            return false;
#endif
        case data_type::f4_e3m0:
        case data_type::f4_e2m1: return false;
        default: return true;
    }
}

bool has_training_support(data_type_t data_type) {
    // TODO: maybe return false for int8, but some primitives like prelu
    // have training support
    switch (data_type) {
        case data_type::bf16:
#if DNNL_X64
            return x64::mayiuse(x64::avx512_core);
#elif DNNL_PPC64
#if defined(USE_CBLAS) && defined(BLAS_HAS_SBGEMM) && defined(__MMA__)
            return true;
#endif
#elif defined(DNNL_AARCH64_USE_ACL)
            return arm_compute::CPUInfo::get().has_bf16();
#else
            return false;
#endif
        case data_type::f16:
#if DNNL_X64
            return x64::mayiuse(x64::avx512_core_fp16);
#elif defined(DNNL_AARCH64_USE_ACL)
            return arm_compute::CPUInfo::get().has_fp16();
#else
            return false;
#endif
        default: return true;
    }
}

float s8s8_weights_scale_factor() {
#if DNNL_X64
    return x64::mayiuse(x64::avx512_core_vnni) || x64::mayiuse(x64::avx2_vnni)
            ? 1.0f
            : 0.5f;
#else
    return 1.0f;
#endif
}
uint32_t get_num_sets_in_cache(int level) {
    // if level 1 is requested we map it to the closest cache which is level 0
    // level 1 is actually instruction cache
    if (level <= 1) { level = level - 1; }

    auto guess = [](int level) {
        switch (level) {
            case 1: return 64u;
            case 2: return 2U * 1024;
            case 3: return 114688u;
            default: return 0U;
        }
    };

#if DNNL_X64
    using namespace x64;

    if (cpu().getDataCacheLevels() == 0) return guess(level);

    if (level >= 0 && (unsigned)level <= cpu().getDataCacheLevels()) {
        uint32_t data[4] = {0};
        Xbyak::util::Cpu::getCpuidEx(4, level, data);
        uint32_t num_sets = data[2] + 1;
        return num_sets;
    } else
        return 0;
#else
    return guess(level);
#endif
}
uint32_t get_num_ways_in_cache(int level) {
    // if level 1 is requested we map it to the closest cache which is level 0
    // level 1 is actually instruction cache
    if (level <= 1) { level = level - 1; }

    auto guess = [](int level) {
        switch (level) {
            case 1: return 12u;
            case 2: return 16u;
            case 3: return 15u;
            default: return 0U;
        }
    };

#if DNNL_X64

    using namespace x64;
    if (cpu().getDataCacheLevels() == 0) return guess(level);

    if (level >= 0 && (unsigned)level <= cpu().getDataCacheLevels()) {
        uint32_t data[4] = {0};
        Xbyak::util::Cpu::getCpuidEx(4, level, data);
        uint32_t num_ways = ((data[1] & 0xFFC00000) >> 22) + 1;
        return num_ways;
    } else
        return 0;

#else
    return guess(level);
#endif
}

unsigned get_per_core_cache_size(int level) {
    auto guess = [](int level) {
        switch (level) {
            case 1: return 32U * 1024;
            case 2: return 512U * 1024;
            case 3: return 1024U * 1024;
            default: return 0U;
        }
    };

#if DNNL_X64
    using namespace x64;
    if (cpu().getDataCacheLevels() == 0) return guess(level);
    
    if (level > 0 && (unsigned)level <= cpu().getDataCacheLevels()) {
        unsigned l = level - 1;
        return cpu().getDataCacheSize(l) / cpu().getCoresSharingDataCache(l);
    } else
        return 0;
#else
    return guess(level);
#endif
}

unsigned get_num_cores() {
#if DNNL_X64
    return x64::cpu().getNumCores(Xbyak::util::CoreLevel);
#elif defined(DNNL_AARCH64_USE_ACL)
    return aarch64::cpu().getNumCores(Xbyak_aarch64::util::CoreLevel);
#else
    return 1;
#endif
}

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
// The purpose of this function is to return the potential maximum number of
// threads in user's threadpool. It is assumed that the number of threads in an
// actual threadpool will not exceed the number cores in a socket reported by
// the OS, which may or may not be equal to the number of total physical cores
// in a socket depending on the OS configuration (read -- VM environment). In
// order to simulate the number of cores available in such environment, this
// function supports process affinity.
unsigned get_max_threads_to_use() {
    // TODO: the logic below should involve number of sockets to provide exact
    // number of cores on 2+ socket systems.
    int num_cores_per_socket = (int)dnnl::impl::cpu::platform::get_num_cores();
    // It may happen that XByak doesn't get num of threads identified, e.g. for
    // AMD. In order to make threadpool working, we supply an additional
    // condition to have some reasonable number of threads available at
    // primitive descriptor creation time.
    if (num_cores_per_socket == 0)
        num_cores_per_socket = std::thread::hardware_concurrency();

#if defined(_WIN32)
    DWORD_PTR proc_affinity_mask;
    DWORD_PTR sys_affinity_mask;
    if (GetProcessAffinityMask(
                GetCurrentProcess(), &proc_affinity_mask, &sys_affinity_mask)) {
        int masked_nthr = 0;
        for (int i = 0; i < CHAR_BIT * sizeof(proc_affinity_mask);
                i++, proc_affinity_mask >>= 1)
            masked_nthr += proc_affinity_mask & 1;
        return std::min(masked_nthr, num_cores_per_socket);
    }
#elif defined(__GLIBC__)
    cpu_set_t cpu_set;
    // Check if the affinity of the process has been set using, e.g.,
    // numactl.
    if (::sched_getaffinity(0, sizeof(cpu_set_t), &cpu_set) == 0)
        return std::min(CPU_COUNT(&cpu_set), num_cores_per_socket);
#endif
    return num_cores_per_socket;
}
#endif

int get_vector_register_size() {
#if DNNL_X64
    using namespace x64;
    if (mayiuse(avx512_core)) return cpu_isa_traits_t<avx512_core>::vlen;
    if (mayiuse(avx)) return cpu_isa_traits_t<avx>::vlen;
    if (mayiuse(sse41)) return cpu_isa_traits_t<sse41>::vlen;
#elif DNNL_AARCH64
    using namespace aarch64;
    if (mayiuse(asimd)) return cpu_isa_traits<asimd>::vlen;
    if (mayiuse(sve_512)) return cpu_isa_traits<sve_512>::vlen;
    if (mayiuse(sve_256)) return cpu_isa_traits<sve_256>::vlen;
#endif
    return 0;
}

/* The purpose of this function is to provide a very efficient timestamp
 * calculation (used primarily for primitive cache). For DNNL_X64, this can be
 * accomplished using *rdtsc* since it provides a timestamp value that (i) is
 * independent for each core, and (ii) is synchronized across cores in multiple
 * sockets.
 * TODO: For now, use std::chrono::steady_clock for other builds, however
 * another more optimized function may be called here.
 */
size_t get_timestamp() {
#if DNNL_X64
    return static_cast<size_t>(Xbyak::util::Clock::getRdtsc());
#else
    return static_cast<size_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
#endif
}

#if DNNL_X64
bool is_hybrid() {
#if DNNL_X64
    // currently only Intel is using hybrid architecture
    if (!x64::cpu().has(Xbyak::util::Cpu::tINTEL)) return false;
    // check CPUID.07H:EDX[15] (Hybrid bit)
    uint32_t data_7[4] = {0};
    x64::cpu().getCpuidEx(7, 0, data_7);
    const bool hybrid_flag = (data_7[3] & (1u << 15)) != 0;
    return hybrid_flag;
#else
return false;
#endif
}

// Global cache topology (lazy-initialized)
cache_topology_t global_cache_topology = {0};
bool global_cache_topology_initialized = false;

core_type get_current_core_type() {
#if DNNL_X64
    uint32_t data[4] = {0};
    x64::cpu().getCpuidEx(0x1A, 0, data);
    uint32_t core_type_field = (data[0] >> 24) & 0xFF;
    switch (core_type_field) {
        case 0x20: // Intel Atom
            return core_type::e_core;
        case 0x40: // Intel Core
            return core_type::p_core;
        default:
            return core_type::p_core;
    }
#endif
    return core_type::default;
}

void init_cache_topology_cpuid(cache_topology_t &cache_topology) {

}
//------------------Start windows specific cache topology code------------------
#if defined(_WIN32)
// inline helper to count bits in KAFFINITY processor mask
inline unsigned count_bits_in_mask(KAFFINITY mask) {
    unsigned count = 0;
    while (mask) {
        count += (mask & 1);
        mask >>= 1;
    }
    return count;
}

// Helper function to determine core type from processor info
// Returns p_core for performance cores, e_core for efficiency cores
core_type get_core_type_from_processor_info(
        const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *info) {
    // For hybrid CPUs, we need to distinguish between P-cores and E-cores
    // On Intel 12th gen+, EfficiencyClass in PROCESSOR_RELATIONSHIP indicates core type:
    //   1 = P-core (performance)
    //   0 = E-core (efficiency)
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
    // Default to p_core for non-hybrid or if we can't determine
    return core_type::p_core;
}

// Structure to track cache information during enumeration
struct cache_accumulator_t {
    // Track which caches we've seen for each core type
    // [core_type_idx][level] -> cache info
    std::map<std::pair<int, int>, cache_info_t> cache_map;
    
    void add_cache(const CACHE_RELATIONSHIP &cache, core_type ctype) {
        int level = static_cast<int>(cache.Level);
        if (level < 1 || level > (int)cache_topology_t::max_cache_levels)
            return;
        
        // Only process data or unified caches
        if (cache.Type != CacheData && cache.Type != CacheUnified)
            return;
        
        int type_idx = (ctype == core_type::p_core) ? 0 : 1;
        auto key = std::make_pair(type_idx, level);
        
        // Count sharing cores from GroupMask
        unsigned sharing_cores = count_bits_in_mask(cache.GroupMask.Mask);
        
        // If we've seen this cache before, take the maximum sharing count
        // (different cache descriptors may report different views)
        auto it = cache_map.find(key);
        if (it != cache_map.end()) {
            it->second.num_sharing_cores = 
                std::max(it->second.num_sharing_cores, sharing_cores);
        } else {
            cache_info_t info;
            info.level = level;
            info.size = cache.CacheSize;
            info.num_sharing_cores = sharing_cores;
            info.ctype = ctype;
            cache_map[key] = info;
        }
    }
};

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
    auto *info_base_ptr = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
            buffer.data());

    if (!GetLogicalProcessorInformationEx(RelationAll, info_base_ptr, &bufferSize)) {
        // Fallback to CPUID-based method if Windows API fails
        init_cache_topology_cpuid(cache_topology);
        return;
    }

    // First pass: collect all processor cores with their types and masks
    std::vector<std::pair<core_type, GROUP_AFFINITY>> core_info;

    DWORD offset = 0;
    while (offset < bufferSize) {
        auto *info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
            reinterpret_cast<BYTE*>(info_base_ptr) + offset);
        if (info->Relationship == RelationProcessorCore) {
            core_type ctype = get_core_type_from_processor_info(info);
            // Store each group mask for this core type
            for (WORD i = 0; i < info->Processor.GroupCount; i++) {
                core_info.push_back({ctype, info->Processor.GroupMask[i]});
            }
        }
        offset += info->Size;
    }

    // Second pass: for each cache, determine which core types use it
    cache_accumulator_t accumulator;

    offset = 0;
    while (offset < bufferSize) {
        auto *info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
                buffer.data() + offset);
        if (info->Relationship == RelationCache) {
            const auto &cache = info->Cache;
            if (cache.Type == CacheData || cache.Type == CacheUnified) {
                // Check which core types intersect with this cache's mask
                bool has_p_core = false;
                bool has_e_core = false;


                for (size_t i = 0; i < core_info.size(); ++i) {
                    core_type ctype = core_info[i].first;
                    GROUP_AFFINITY core_mask = core_info[i].second;
                    // Check if this cache intersects with this core's processors
                    if (cache.GroupMask.Group == core_mask.Group &&
                        (cache.GroupMask.Mask & core_mask.Mask) != 0) {
                        if (ctype == core_type::p_core)
                            has_p_core = true;
                        else if (ctype == core_type::e_core)
                            has_e_core = true;
                    }
                }
                // Add cache entry for each core type that uses it
                if (has_p_core) {
                    accumulator.add_cache(cache, core_type::p_core);
                }
                if (has_e_core) {
                    accumulator.add_cache(cache, core_type::e_core);
                }
                }
                // If no specific assignment, this might be a shared cache
                if (!has_p_core && !has_e_core) {
                    accumulator.add_cache(cache, core_type::p_core);
                    accumulator.add_cache(cache, core_type::e_core);
                }
            }
        }
        
        offset += info->Size;
    }
    
    // Third pass: populate cache_topology structure from accumulator
    for (const auto &entry : accumulator.cache_map) {
        int type_idx = entry.first.first;
        int level = entry.first.second;
        const auto &cache_info = entry.second;
        
        size_t idx = type_idx * cache_topology_t::max_cache_levels + (level);
        if (idx < cache_topology_t::max_cache_levels * cache_topology_t::max_core_types) {
            cache_topology.caches[idx] = cache_info;
        }
    }
    
    // If this is not a hybrid system, copy P-core data to E-core slots
    // so that queries work regardless of which core type is specified
    if (!cache_topology.is_hybrid) {
        for (size_t level = 0; level < cache_topology_t::max_cache_levels; level++) {
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
void init_cache_topology_linux(cache_topology_t &cache_topology) {

}
#endif
//------------------End linux specific cache topology code----------------------

void init_cache_topology() {
    if (global_cache_topology_initialized) return;

#if defined(_WIN32)
    init_cache_topology_windows(global_cache_topology);
#elif defined(__linux__)
    init_cache_topology_linux(global_cache_topology);
#else
    // Fallback: use CPUID-based method
    init_cache_topology_cpuid(global_cache_topology);
#endif
    global_cache_topology_initialized = true;
}
#endif // DNNL_X64
unsigned get_per_core_cache_size(int level, behavior_t btype) {
#if DNNL_X64
    init_cache_topology();
    
    // Convert 1-based level to 0-based for array access
    if (level < 1 || level > (int)cache_topology_t::max_cache_levels) {
        return 0;
    }
    
    auto pcore_cache = global_cache_topology.get_cache(level, core_type::p_core);
    auto ecore_cache = global_cache_topology.get_cache(level, core_type::e_core);
    
    // Calculate effective cache size per core (considering sharing)
    auto pcore_size = pcore_cache.size;
    if (pcore_cache.num_sharing_cores > 0) {
        pcore_size /= pcore_cache.num_sharing_cores;
    }
    
    auto ecore_size = ecore_cache.size;
    if (ecore_cache.num_sharing_cores > 0) {
        ecore_size /= ecore_cache.num_sharing_cores;
    }
    switch (btype)
    {
    case behavior_t::p_core:
        return pcore_size;
        break;
    case behavior_t::e_core:
        return ecore_size;
        break;
    case behavior_t::current:
        // Detect current core type using CPUID 0x1AH
        {
            core_type current_ctype = get_current_core_type();
            if (current_ctype == core_type::p_core) {
                return pcore_size;
            } else {
                return ecore_size;
            }
        }
    case behavior_t::min:
        return std::min(pcore_size, ecore_size);
        break;
    case behavior_t::max:
        return std::max(pcore_size, ecore_size);
        break;
    case behavior_t::unknown:// [[fallthrough]]; // fallthrough attribute is a C++17 feature
    default:
        // unknown behavior, fallback to non-hybrid behavior
        return get_per_core_cache_size(level);
        break;
    }
#endif // DNNL_X64
    return get_per_core_cache_size(level);
}

} // namespace platform
} // namespace cpu
} // namespace impl
} // namespace dnnl
