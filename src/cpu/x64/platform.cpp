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
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "common/nstl.hpp"
#include "common/verbose.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/platform.hpp"
#include "xbyak/xbyak_util.h"

#if defined(_WIN32)
#include <vector>
#include <windows.h>
#endif

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
struct cpu_topology_cache_t {
    Xbyak::util::CpuTopology topology;
    cpu_topology_cache_t() : topology(cpu()) {}
};

// Thread-safe lazy initialization using C++11 magic statics
cpu_topology_cache_t &get_topology_cache() {
    static cpu_topology_cache_t cache;
    return cache;
}

// Map platform cache level (1=L1d, 2=L2, 3=L3) to Xbyak cache type.
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
// For Efficient cores, l3_filter distinguishes regular E-cores (with L3)
// from LP E-cores (without L3) and optionally restricts matches by L3 presence.
// Returns the first matching CPU index, or SIZE_MAX if none is found.
size_t find_representative_cpu(Xbyak::util::CoreType target_type,
        l3_filter_t l3_filter = l3_filter_t::any) {
    const auto &topo = get_topology_cache().topology;
    size_t num_cpus = topo.getLogicalCpuNum();

    for (size_t i = 0; i < num_cpus; i++) {
        const auto &logical_cpu = topo.getLogicalCpu(i);
        if (logical_cpu.coreType != target_type) continue;

        if (target_type == Xbyak::util::Efficient
                && l3_filter != l3_filter_t::any) {
            const auto &l3 = topo.getCache(i, Xbyak::util::L3);
            bool has_l3 = (l3.size > 0);
            if (l3_filter == l3_filter_t::with_l3 && !has_l3) continue;
            if (l3_filter == l3_filter_t::without_l3 && has_l3) continue;
        }
        return i;
    }
    return SIZE_MAX;
}

// Calculate per-core cache size for a specific cache level and CPU index.
// Matches the legacy getCoresSharingDataCache semantics: divides by the number
// of PHYSICAL cores sharing the cache, not logical CPUs.  The legacy Xbyak Cpu
// path (CPUID leaf 4) uses L1d sharing count as the SMT width and divides it
// out of every level's logical-CPU sharing count.  We replicate that here so
// that non-hybrid results are identical to the legacy path.
uint32_t calculate_per_core_cache(size_t cpu_index, int level) {
    const auto &topo = get_topology_cache().topology;

    Xbyak::util::CacheType cache_type = convert_cache_level(level);
    if (cache_type == Xbyak::util::CACHE_UNKNOWN) { return 0; }

    const auto &cache = topo.getCache(cpu_index, cache_type);
    if (cache.size == 0) { return 0; }

    // Number of logical CPUs (threads) sharing this cache instance.
    size_t sharing_logical = cache.getSharedCpuNum();
    if (sharing_logical == 0) sharing_logical = 1;

    // SMT width = logical CPUs sharing L1d (L1 is always private to one
    // physical core, so this count equals the number of HT threads per core).
    size_t smt_width
            = topo.getCache(cpu_index, Xbyak::util::L1d).getSharedCpuNum();
    if (smt_width == 0) smt_width = 1;

    // Physical cores sharing this cache (mirrors legacy smt_width division).
    size_t sharing_cores = std::max(sharing_logical / smt_width, size_t(1));

    return static_cast<uint32_t>(cache.size / sharing_cores);
}

struct cache_level_info_t {
    uint32_t total_kb; // total physical cache capacity in KB
    size_t sharing_cores; // number of physical cores sharing this instance
    size_t smt_width; // logical threads per physical core (HT width)
};

cache_level_info_t get_cache_level_info(size_t cpu_index, int level) {
    const auto &topo = get_topology_cache().topology;
    Xbyak::util::CacheType cache_type = convert_cache_level(level);
    if (cache_type == Xbyak::util::CACHE_UNKNOWN) return {0, 1, 1};
    const auto &cache = topo.getCache(cpu_index, cache_type);
    if (cache.size == 0) return {0, 1, 1};
    size_t sharing_logical = cache.getSharedCpuNum();
    if (sharing_logical == 0) sharing_logical = 1;
    size_t smt = topo.getCache(cpu_index, Xbyak::util::L1d).getSharedCpuNum();
    if (smt == 0) smt = 1;
    size_t sharing_cores = std::max(sharing_logical / smt, size_t(1));
    return {cache.size / 1024, sharing_cores, smt};
}

// Print hybrid per-core cache topology once per process at debuginfo=1 or higher.
// One line per core type showing physical cache sizes.
// Shared levels annotated: L{N}:{total}KB({N}cores,{per_core}KB/core)
// Private levels:          L{N}:{total}KB
// smt:{N} at end of each line: logical threads per physical core (HT width).
// Final line: per-core cache size used, which is the minimum across all present core types.
void print_hybrid_cache_debuginfo_once(
        size_t pcore_cpu, size_t lp_core_cpu, size_t lpe_core_cpu) {
    if (get_verbose(verbose_t::debuginfo) < 1) return;
    static std::atomic_flag printed = ATOMIC_FLAG_INIT;
    if (printed.test_and_set()) return;

    auto print_core_line
            = [](const char *tag, size_t cpu_idx, bool include_l3) {
        auto l1 = get_cache_level_info(cpu_idx, 1);
        auto l2 = get_cache_level_info(cpu_idx, 2);
        char buf[256];
        int offset = 0;
        offset += snprintf(buf + offset, (int)sizeof(buf) - offset,
                "cpu,debuginfo,platform,%s,L1d:%uKB", tag, l1.total_kb);
        if (l2.sharing_cores > 1)
            offset += snprintf(buf + offset, (int)sizeof(buf) - offset,
                    ",L2:%uKB(%zucores,%uKB/core)", l2.total_kb,
                    l2.sharing_cores, l2.total_kb / (uint32_t)l2.sharing_cores);
        else
            offset += snprintf(buf + offset, (int)sizeof(buf) - offset,
                    ",L2:%uKB", l2.total_kb);
        if (include_l3) {
            auto l3 = get_cache_level_info(cpu_idx, 3);
            if (l3.total_kb > 0) {
                if (l3.sharing_cores > 1)
                    offset += snprintf(buf + offset, (int)sizeof(buf) - offset,
                            ",L3:%uMB(%zucores,%uKB/core)", l3.total_kb / 1024,
                            l3.sharing_cores,
                            l3.total_kb / (uint32_t)l3.sharing_cores);
                else
                    offset += snprintf(buf + offset, (int)sizeof(buf) - offset,
                            ",L3:%uMB", l3.total_kb / 1024);
            }
        }
        snprintf(buf + offset, (int)sizeof(buf) - offset, ",smt:%zu\n",
                l1.smt_width);
        verbose_printf(verbose_t::debuginfo, "%s", buf);
    };

    print_core_line("pcore_cache", pcore_cpu, true);
    print_core_line("lp_core_cache", lp_core_cpu, true);
    if (lpe_core_cpu != SIZE_MAX)
        print_core_line("lpe_core_cache", lpe_core_cpu, false);

    uint32_t l1_used = (std::min)(calculate_per_core_cache(pcore_cpu, 1),
            calculate_per_core_cache(lp_core_cpu, 1));
    uint32_t l2_used = (std::min)(calculate_per_core_cache(pcore_cpu, 2),
            calculate_per_core_cache(lp_core_cpu, 2));
    uint32_t l3_used = (std::min)(calculate_per_core_cache(pcore_cpu, 3),
            calculate_per_core_cache(lp_core_cpu, 3));
    if (lpe_core_cpu != SIZE_MAX) {
        l1_used = (std::min)(
                l1_used, calculate_per_core_cache(lpe_core_cpu, 1));
        l2_used = (std::min)(
                l2_used, calculate_per_core_cache(lpe_core_cpu, 2));
        // lpe_core has no L3 -- excluded to avoid zeroing L3 budget
    }
    verbose_printf(verbose_t::debuginfo,
            "cpu,debuginfo,platform,per_core_cache(min),L1d:%uKB,L2:%uKB,L3:%"
            "uKB\n",
            l1_used / 1024, l2_used / 1024, l3_used / 1024);
}

// Cached per-core cache sizes for all core types and levels on hybrid systems.
// Computed once at first use; all fields are bytes (not KB).
// levels[0]=L1, levels[1]=L2, levels[2]=L3 (0 if not present).
// lpe_core_cpu == SIZE_MAX means no LP E-core present; lpe_core[] is all zero.
struct hybrid_core_cache_sizes_t {
    size_t pcore_cpu;
    size_t lp_core_cpu;
    size_t lpe_core_cpu;
    uint32_t pcore[3];
    uint32_t lp_core[3];
    uint32_t lpe_core[3];
};

hybrid_core_cache_sizes_t &get_hybrid_core_cache_sizes() {
    static hybrid_core_cache_sizes_t result = []() {
        hybrid_core_cache_sizes_t cache_sizes {};
        cache_sizes.pcore_cpu
                = find_representative_cpu(Xbyak::util::Performance);
        cache_sizes.lp_core_cpu = find_representative_cpu(
                Xbyak::util::Efficient, l3_filter_t::with_l3);
        cache_sizes.lpe_core_cpu = find_representative_cpu(
                Xbyak::util::Efficient, l3_filter_t::without_l3);

        // If no P-core is found, use CPU 0 as a conservative fallback.
        if (cache_sizes.pcore_cpu == SIZE_MAX) cache_sizes.pcore_cpu = 0;
        // If no E-core with L3 is found, reuse P-core values for lp_core.
        if (cache_sizes.lp_core_cpu == SIZE_MAX)
            cache_sizes.lp_core_cpu = cache_sizes.pcore_cpu;
        // LP E-core is optional: keep lpe_core_cpu as SIZE_MAX when absent.

        for (int lvl = 1; lvl <= 3; lvl++) {
            cache_sizes.pcore[lvl - 1]
                    = calculate_per_core_cache(cache_sizes.pcore_cpu, lvl);
            cache_sizes.lp_core[lvl - 1]
                    = calculate_per_core_cache(cache_sizes.lp_core_cpu, lvl);
            cache_sizes.lpe_core[lvl - 1]
                    = (cache_sizes.lpe_core_cpu != SIZE_MAX)
                    ? calculate_per_core_cache(cache_sizes.lpe_core_cpu, lvl)
                    : 0;
        }
        return cache_sizes;
    }();
    return result;
}

#if defined(__linux__)
// Number of logical CPUs listed in a sysfs cpu list such as "0-1" or "0,192".
// Returns 0 (unavailable) if the value is truncated by the buffer: sysfs
// terminates every value with '\n', so a missing newline means more data.
unsigned count_sysfs_cpu_list(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char buf[1024] = {};
    const bool ok = fgets(buf, sizeof(buf), f) != nullptr;
    fclose(f);
    if (!ok || strchr(buf, '\n') == nullptr) return 0;

    unsigned count = 0;
    const char *p = buf;
    while (*p) {
        char *end = nullptr;
        const long lo = strtol(p, &end, 10);
        if (end == p) break;
        long hi = lo;
        if (*end == '-') {
            p = end + 1;
            hi = strtol(p, &end, 10);
            if (end == p) break;
        }
        if (hi >= lo) count += (unsigned)(hi - lo + 1);
        p = (*end == ',') ? end + 1 : end;
    }
    return count;
}

// Actual number of physical cores sharing L1d/L2/L3, as enumerated by the
// kernel. counts[0]=L1d, counts[1]=L2, counts[2]=L3; 0 means unavailable.
// Not built on Xbyak::util::CpuTopology: that enumerates every logical CPU
// (probing core type on hybrid parts), far more work than this needs.
struct os_cache_sharing_t {
    unsigned counts[3];
};

// Formats "/sys/devices/system/cpu/cpu0/cache/index<idx>/<suffix>" into buf.
void format_cache_index_path(
        char *buf, size_t bufsize, int idx, const char *suffix) {
    snprintf(buf, bufsize, "/sys/devices/system/cpu/cpu0/cache/index%d/%s", idx,
            suffix);
}

os_cache_sharing_t get_os_cores_sharing_cache_all() {
    os_cache_sharing_t result {{0, 0, 0}};
    char path[128];
    // SMT width: L1d is private to a physical core, so the number of logical
    // CPUs sharing it is the number of hardware threads per core.
    unsigned smt = 0;
    unsigned logical[3] = {0, 0, 0};

    for (int idx = 0; idx < 8; idx++) {
        format_cache_index_path(path, sizeof(path), idx, "level");
        FILE *f = fopen(path, "r");
        if (!f) break;
        int lvl = 0;
        const bool got_level = fscanf(f, "%d", &lvl) == 1;
        fclose(f);
        if (!got_level || lvl < 1 || lvl > 3) continue;

        format_cache_index_path(path, sizeof(path), idx, "type");
        f = fopen(path, "r");
        if (!f) continue;
        char type[32] = {};
        const bool got_type = fgets(type, sizeof(type), f) != nullptr;
        fclose(f);
        // Skip instruction caches; data and unified caches both count.
        if (!got_type || type[0] == 'I') continue;

        format_cache_index_path(path, sizeof(path), idx, "shared_cpu_list");
        const unsigned shared = count_sysfs_cpu_list(path);
        if (shared == 0) continue;

        if (lvl == 1) smt = shared;
        logical[lvl - 1] = shared;
    }

    if (smt == 0) return result;
    // shared_cpu_list counts logical CPUs; convert to physical cores.
    for (int l = 0; l < 3; l++) {
        if (logical[l] == 0) continue;
        const unsigned cores = logical[l] / smt;
        result.counts[l] = cores > 0 ? cores : 1;
    }
    return result;
}
#elif defined(_WIN32)
// GroupMasks[]/GroupCount on CACHE_RELATIONSHIP are absent from older SDKs.
// Gate on NTDDI_WIN10_NI, the same threshold Xbyak uses, and treat them as
// unavailable when NTDDI_VERSION isn't defined.
#if defined(NTDDI_VERSION) && defined(NTDDI_WIN10_NI) \
        && NTDDI_VERSION >= NTDDI_WIN10_NI
#define ONEDNN_WINSDK_HAS_CACHE_RELATIONSHIP_GROUPMASKS 1
#else
#define ONEDNN_WINSDK_HAS_CACHE_RELATIONSHIP_GROUPMASKS 0
#endif

// Actual number of physical cores sharing L1d/L2/L3, as enumerated by the OS.
// GetLogicalProcessorInformationEx(RelationCache, ...) reports the exact
// GROUP_AFFINITY mask(s) of logical processors sharing each cache instance.
struct os_cache_sharing_t {
    unsigned counts[3];
};

os_cache_sharing_t get_os_cores_sharing_cache_all() {
    os_cache_sharing_t result {{0, 0, 0}};

    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationCache, nullptr, &len);
    if (len == 0) return result;

    std::vector<char> buf(len);
    auto *base = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(
            buf.data());
    if (!GetLogicalProcessorInformationEx(RelationCache, base, &len))
        return result;

    // SMT width: L1d is private to a physical core, so the number of logical
    // CPUs sharing it is the number of hardware threads per core.
    unsigned smt = 0;
    unsigned logical[3] = {0, 0, 0};

    for (DWORD off = 0; off < len;) {
        auto *entry
                = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(
                        buf.data() + off);
        off += entry->Size;
        if (entry->Relationship != RelationCache) continue;

        const auto &cache = entry->Cache;
        // Skip instruction caches; data and unified caches both count.
        if (cache.Type == CacheInstruction) continue;
        if (cache.Level < 1 || cache.Level > 3) continue;

        unsigned shared = 0;
#if ONEDNN_WINSDK_HAS_CACHE_RELATIONSHIP_GROUPMASKS
        for (WORD gi = 0; gi < cache.GroupCount; gi++) {
            for (KAFFINITY m = cache.GroupMasks[gi].Mask; m; m &= m - 1)
                shared++;
        }
#else
        for (KAFFINITY m = cache.GroupMask.Mask; m; m &= m - 1)
            shared++;
#endif
        if (shared == 0) continue;

        if (cache.Level == 1) smt = smt > shared ? smt : shared;
        logical[cache.Level - 1] = logical[cache.Level - 1] > shared
                ? logical[cache.Level - 1]
                : shared;
    }

    if (smt == 0) return result;
    // The mask counts logical CPUs; convert to physical cores.
    for (int l = 0; l < 3; l++) {
        if (logical[l] == 0) continue;
        const unsigned cores = logical[l] / smt;
        result.counts[l] = cores > 0 ? cores : 1;
    }
    return result;
}
#undef ONEDNN_WINSDK_HAS_CACHE_RELATIONSHIP_GROUPMASKS
#endif

// Smallest `width` such that `(1u << width) >= n`, for n > 0.
unsigned ceil_log2(unsigned n) {
    unsigned width = 0;
    while ((1u << width) < n)
        width++;
    return width;
}

// CPUID leaf 4 cache-sharing mask width, in x2APIC ID bits, for platform
// cache level `level` (1=L1d, 2=L2, 3=L3). Returns -1 if not present.
int get_cache_mask_width(unsigned level) {
    uint32_t data[4];
    for (uint32_t sub = 0;; sub++) {
        Xbyak::util::Cpu::getCpuidEx(0x4, sub, data);
        const uint32_t cache_type = data[0] & 0x1f;
        if (cache_type == 0) break; // no more cache entries
        if (cache_type == 2) continue; // instruction cache
        const uint32_t cache_level = (data[0] >> 5) & 0x7;
        if (cache_level != level) continue;
        const uint32_t max_ids_sharing = ((data[0] >> 14) & 0xfff) + 1;
        return (int)ceil_log2(max_ids_sharing);
    }
    return -1;
}

struct topology_level_t {
    unsigned shift; // cumulative x2APIC ID shift width for this domain
    // 1=SMT, 2=Core, 3=Module, 4=Tile, 5=Die, 6=DieGrp (SDM Vol 2A); leaf 0xB
    // defines only 1 and 2.
    unsigned type;
    unsigned logical_count; // actual (not rounded) logical processors in it
};

// Enumerates topology levels via CPUID `leaf`/subleaf. Returns the number of
// levels found, or 0 if the leaf yields none (e.g. present but unpopulated
// under some hypervisors).
unsigned enumerate_topology_leaf(
        uint32_t leaf, topology_level_t *levels, unsigned max_levels) {
    uint32_t data[4];
    unsigned n = 0;
    for (uint32_t sub = 0; n < max_levels; sub++) {
        Xbyak::util::Cpu::getCpuidEx(leaf, sub, data);
        const uint32_t level_type = (data[2] >> 8) & 0xff;
        if (level_type == 0) break; // no more levels
        levels[n].shift = data[0] & 0x1f;
        levels[n].type = level_type;
        levels[n].logical_count = data[1] & 0xffff;
        n++;
    }
    return n;
}

// Walks CPUID leaf 0x1F, falling back to leaf 0xB if it yields no levels.
// Returns the number of topology levels found, or 0 if neither leaf is usable.
unsigned get_topology_levels(topology_level_t *levels, unsigned max_levels) {
    uint32_t data[4];
    Xbyak::util::Cpu::getCpuid(0x0, data);
    const uint32_t max_leaf = data[0];

    if (max_leaf >= 0x1f) {
        const unsigned n = enumerate_topology_leaf(0x1f, levels, max_levels);
        if (n > 0) return n;
    }
    if (max_leaf < 0xb) return 0;
    return enumerate_topology_leaf(0xb, levels, max_levels);
}

// Number of physical cores sharing platform cache level `level` (1-based),
// derived from CPUID leaf 0x1F/0xB topology: each level's EAX[4:0] shift
// width is matched against leaf 4's cache-sharing mask width, and the
// matching level's EBX[15:0] logical-processor count is divided by the SMT
// level's count.
unsigned get_topology_cores_sharing_cache(unsigned level) {
    const int mask_width = get_cache_mask_width(level);
    if (mask_width < 0) return 0;
    // One reserved ID: the cache is private to a single logical processor.
    if (mask_width == 0) return 1;

    topology_level_t levels[8];
    const unsigned n = get_topology_levels(levels, 8);
    if (n == 0) return 0;

    unsigned smt_count = 0;
    for (unsigned i = 0; i < n; i++)
        if (levels[i].type == 1) smt_count = levels[i].logical_count;
    if (smt_count == 0) return 0;

    for (unsigned i = 0; i < n; i++) {
        if ((int)levels[i].shift != mask_width) continue;
        return nstl::max(1u, levels[i].logical_count / smt_count);
    }
    return 0; // no topology level maps exactly onto the cache's ID width
}

// Number of physical cores sharing the data/unified cache at 0-based index `l`.
// Leaf 4's EAX[25:14]+1 is an ID reservation that can overstate the true
// sharing count; prefer leaf 0x1F/0xB, then the OS-enumerated topology
// (sysfs/GetLogicalProcessorInformationEx), and fall back to raw CPUID.
unsigned compute_cores_sharing_cache(unsigned l) {
    // Absent level (e.g. no L3 exposed under a hypervisor): Xbyak's getter
    // would record a sticky ERR_BAD_PARAMETER that fails every later JIT.
    if (l >= cpu().getDataCacheLevels()) return 1;
    const unsigned topo_sharing = get_topology_cores_sharing_cache(l + 1);
    if (topo_sharing > 0) return topo_sharing;

#if defined(__linux__) || defined(_WIN32)
    static const os_cache_sharing_t os_sharing
            = get_os_cores_sharing_cache_all();
    if (os_sharing.counts[l] > 0) return os_sharing.counts[l];
#endif
    const unsigned sharing = cpu().getCoresSharingDataCache(l);
    return sharing > 0 ? sharing : 1;
}

// Cached: the CPUID walk above is a dozen serializing instructions (each a
// VM exit under a hypervisor) and the result never changes.
unsigned cores_sharing_cache(unsigned l) {
    if (l >= 3) {
        const unsigned sharing = cpu().getCoresSharingDataCache(l);
        return sharing > 0 ? sharing : 1;
    }
    static const unsigned sharing[3] = {compute_cores_sharing_cache(0),
            compute_cores_sharing_cache(1), compute_cores_sharing_cache(2)};
    return sharing[l];
}

// Print per-core cache sizes once (CPUID path, no CpuTopology init).
// Format mirrors the hybrid path: shared levels show total, sharing count, and
// per-core budget; private levels show just the total. smt field is appended.
void print_cache_debuginfo_once() {
    if (get_verbose(verbose_t::debuginfo) < 1) return;
    static std::atomic_flag printed = ATOMIC_FLAG_INIT;
    if (printed.test_and_set()) return;

    // SMT width = L1d sharing count, same source (topology-first) as the
    // per-level sharing counts printed below.
    uint32_t smt = cores_sharing_cache(0);
    if (smt == 0) smt = 1;

    char buf[256];
    int offset = 0;
    unsigned nlevels = cpu().getDataCacheLevels();
    offset += snprintf(buf + offset, (int)sizeof(buf) - offset,
            "cpu,debuginfo,platform,cache");
    for (unsigned li = 0; li < nlevels && li < 3; li++) {
        uint32_t total_kb = cpu().getDataCacheSize(li) / 1024;
        uint32_t sharing = cores_sharing_cache(li);
        if (sharing == 0) sharing = 1;
        uint32_t per_core_kb = total_kb / sharing;
        const char *label = (li == 0) ? "L1d" : (li == 1) ? "L2" : "L3";
        if (li == 2) {
            // L3: show total in MB, per-core in KB
            if (sharing > 1)
                offset += snprintf(buf + offset, (int)sizeof(buf) - offset,
                        ",%s:%uMB(%ucores,%uKB/core)", label, total_kb / 1024,
                        sharing, per_core_kb);
            else
                offset += snprintf(buf + offset, (int)sizeof(buf) - offset,
                        ",%s:%uMB", label, total_kb / 1024);
        } else {
            if (sharing > 1)
                offset += snprintf(buf + offset, (int)sizeof(buf) - offset,
                        ",%s:%uKB(%ucores,%uKB/core)", label, total_kb, sharing,
                        per_core_kb);
            else
                offset += snprintf(buf + offset, (int)sizeof(buf) - offset,
                        ",%s:%uKB", label, total_kb);
        }
    }
    snprintf(buf + offset, (int)sizeof(buf) - offset, ",smt:%u\n", smt);
    verbose_printf(verbose_t::debuginfo, "%s", buf);
}

#endif // !__APPLE__

} // anonymous namespace

bool is_hybrid() {
#ifdef __APPLE__
    // xbyak::util::CpuTopology is not supported on macOS; assume non-hybrid.
    return false;
#else
    // Use the HYBRID bit from the already-cached Xbyak::Cpu instance
    // (CPUID leaf 7 EDX[15]) to avoid triggering CpuTopology init.
    return cpu().has(Xbyak::util::Cpu::tHYBRID);
#endif
}

// CPUID-based implementation using Xbyak::util::Cpu (leaf 4).
// Used on non-hybrid systems to avoid the expensive CpuTopology init.
unsigned get_per_core_cache_size_cpuid(int level) {
    if (level > 0 && (unsigned)level <= cpu().getDataCacheLevels()) {
        unsigned l = level - 1;
#ifdef __APPLE__
        // cores_sharing_cache() (leaf 0x1F/0xB + OS fallback) is not compiled
        // on macOS; use the legacy leaf-4-only count directly.
        return cpu().getDataCacheSize(l) / cpu().getCoresSharingDataCache(l);
#else
        return cpu().getDataCacheSize(l) / cores_sharing_cache(l);
#endif
    }
    return 0;
}

unsigned get_per_core_cache_size(int level) {
    if (level < 1 || level > 3) { return 0; }

#ifdef __APPLE__
    return get_per_core_cache_size_cpuid(level);
#else
    // Fast path: on non-hybrid systems avoid the expensive CpuTopology init.
    if (!is_hybrid()) {
        print_cache_debuginfo_once();
        return get_per_core_cache_size_cpuid(level);
    }

    const auto &cache_sizes = get_hybrid_core_cache_sizes();
    const int li = level - 1; // 0-indexed

    print_hybrid_cache_debuginfo_once(cache_sizes.pcore_cpu,
            cache_sizes.lp_core_cpu, cache_sizes.lpe_core_cpu);

    uint32_t min_cache_bytes
            = (std::min)(cache_sizes.pcore[li], cache_sizes.lp_core[li]);
    if (cache_sizes.lpe_core_cpu != SIZE_MAX && cache_sizes.lpe_core[li] > 0)
        min_cache_bytes = (std::min)(min_cache_bytes, cache_sizes.lpe_core[li]);
    return min_cache_bytes;
#endif
}

bool has_lpe_core() {
#ifdef __APPLE__
    return false;
#else
    if (!is_hybrid()) return false;
    return get_hybrid_core_cache_sizes().lpe_core_cpu != SIZE_MAX;
#endif
}

} // namespace platform
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
