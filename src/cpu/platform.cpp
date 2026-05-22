/*******************************************************************************
* Copyright 2020 Intel Corporation
* Copyright 2020-2024 FUJITSU LIMITED
* Copyright 2022-2024, 2026 Arm Ltd. and affiliates
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

#include <algorithm>
#include <cctype>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <dirent.h>
#include <sys/types.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL

#if defined(_WIN32)
// windows.h already included above for NUMA query.
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
#elif DNNL_RV64
#include "cpu/rv64/cpu_isa_traits.hpp"
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
        case data_type::f64: return false;
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
#elif DNNL_RV64
            return rv64::mayiuse(rv64::zvfh);
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
        case data_type::f32: return true;
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
#elif DNNL_RV64
            return rv64::mayiuse(rv64::zvfh);
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
        default: return false;
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
#elif DNNL_AARCH64
    const auto num_caches
            = static_cast<int>(aarch64::cpu().getLastDataCacheLevel());

    if (num_caches == 0) { return guess(level); }

    if (level > 0 && level <= num_caches) {
        const auto &cache_level
                = static_cast<Xbyak_aarch64::util::Arm64CacheLevel>(level);

        return aarch64::cpu().getDataCacheSize(cache_level)
                / aarch64::cpu().getCoresSharingDataCache(cache_level);
    } else {
        return 0;
    }
#else
    return guess(level);
#endif
}

unsigned get_num_cores() {
#if DNNL_X64
    return x64::cpu().getNumCores(Xbyak::util::CoreLevel);
#elif DNNL_AARCH64
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

std::vector<std::vector<int>> numa_topology_t::query_numa_node_cpus() {
    std::vector<std::vector<int>> nodes;
#if defined(__linux__)
    // Parse strings of the form "0-31,192-223" produced by the kernel in
    // /sys/devices/system/node/nodeN/cpulist.
    auto parse_cpulist = [](const std::string &s) {
        std::vector<int> cpus;
        size_t i = 0;
        while (i < s.size()) {
            while (i < s.size() && std::isspace((unsigned char)s[i]))
                ++i;
            if (i >= s.size()) break;
            int a = 0;
            bool have_a = false;
            while (i < s.size() && std::isdigit((unsigned char)s[i])) {
                a = a * 10 + (s[i] - '0');
                ++i;
                have_a = true;
            }
            if (!have_a) break;
            int b = a;
            if (i < s.size() && s[i] == '-') {
                ++i;
                b = 0;
                while (i < s.size() && std::isdigit((unsigned char)s[i])) {
                    b = b * 10 + (s[i] - '0');
                    ++i;
                }
            }
            for (int c = a; c <= b; ++c)
                cpus.push_back(c);
            if (i < s.size() && s[i] == ',') ++i;
        }
        return cpus;
    };

    const char *base = "/sys/devices/system/node";
    DIR *dir = ::opendir(base);
    if (!dir) return nodes;

    std::vector<std::pair<int, std::vector<int>>> entries;
    int max_id = -1;
    for (struct dirent *ent = ::readdir(dir); ent != nullptr;
            ent = ::readdir(dir)) {
        const char *name = ent->d_name;
        if (name[0] != 'n' || name[1] != 'o' || name[2] != 'd'
                || name[3] != 'e')
            continue;
        const char *p = name + 4;
        if (*p == '\0') continue;
        int id = 0;
        bool ok = true;
        for (; *p; ++p) {
            if (!std::isdigit((unsigned char)*p)) {
                ok = false;
                break;
            }
            id = id * 10 + (*p - '0');
        }
        if (!ok) continue;

        std::string path = std::string(base) + "/" + name + "/cpulist";
        std::ifstream ifs(path);
        if (!ifs.is_open()) continue;
        std::string line;
        std::getline(ifs, line);
        entries.emplace_back(id, parse_cpulist(line));
        if (id > max_id) max_id = id;
    }
    ::closedir(dir);

    if (max_id >= 0) {
        nodes.resize(max_id + 1);
        for (auto &e : entries)
            nodes[e.first] = std::move(e.second);
    }
#elif defined(_WIN32)
    // Use GetLogicalProcessorInformationEx(RelationNumaNode, ...) to enumerate
    // NUMA nodes and their owning CPU masks. Each NUMA_NODE_RELATIONSHIP carries
    // a single GROUP_AFFINITY (one Windows processor group, up to 64 CPUs).
    // For systems with more than one processor group, multiple records can
    // share the same NodeNumber; we merge them. Logical CPU ids are encoded as
    //     cpu_id = group_index * 64 + bit_index
    // which matches the "global processor index" convention also used by tools
    // like Process Explorer.
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationNumaNode, nullptr, &len);
    if (len == 0) return nodes;

    std::vector<char> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationNumaNode,
                reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(
                        buf.data()),
                &len))
        return nodes;

    int max_id = -1;
    std::vector<std::pair<int, std::vector<int>>> entries;
    const char *p = buf.data();
    const char *end = p + len;
    while (p < end) {
        const auto &info = *reinterpret_cast<
                const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(p);
        p += info.Size;
        if (info.Relationship != RelationNumaNode) continue;

        const int id = static_cast<int>(info.NumaNode.NodeNumber);
        const GROUP_AFFINITY &ga = info.NumaNode.GroupMask;
        const int group_base = static_cast<int>(ga.Group) * 64;

        std::vector<int> cpus;
        KAFFINITY mask = ga.Mask;
        for (int bit = 0; bit < 64; ++bit) {
            if (mask & (KAFFINITY {1} << bit)) cpus.push_back(group_base + bit);
        }
        entries.emplace_back(id, std::move(cpus));
        if (id > max_id) max_id = id;
    }

    if (max_id >= 0) {
        nodes.resize(max_id + 1);
        for (auto &e : entries) {
            // Merge if multiple records map to the same NUMA node id (one per
            // processor group).
            auto &dst = nodes[e.first];
            dst.insert(dst.end(), e.second.begin(), e.second.end());
        }
        // Keep CPUs sorted so that range collapsing produces tight ranges.
        for (auto &v : nodes)
            std::sort(v.begin(), v.end());
    }
#endif
    return nodes;
}

namespace {

// Collapse a sorted CPU list into contiguous ranges.
std::vector<numa_topology_t::cpu_range_t> encode_cpu_ranges(
        const std::vector<int> &cpus) {
    std::vector<numa_topology_t::cpu_range_t> ranges;
    for (size_t i = 0; i < cpus.size();) {
        size_t j = i;
        while (j + 1 < cpus.size() && cpus[j + 1] == cpus[j] + 1)
            ++j;
        ranges.push_back({cpus[i], cpus[j]});
        i = j + 1;
    }
    return ranges;
}

} // namespace

numa_topology_t::numa_topology_t() {
    auto raw = query_numa_node_cpus();
    if (raw.empty()) return;

    nodes_.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        if (raw[i].empty()) continue;
        node_info_t info;
        info.id = static_cast<int>(i);
        info.cpu_ranges = encode_cpu_ranges(raw[i]);
        nodes_.push_back(std::move(info));
    }
    num_nodes_ = nodes_.size();
}

const numa_topology_t &numa_topology_t::instance() {
    static const numa_topology_t inst;
    return inst;
}

const numa_topology_t::node_info_t *numa_topology_t::node_by_id(int id) const {
    for (const auto &n : nodes_)
        if (n.id == id) return &n;
    return nullptr;
}

int numa_topology_t::node_of_cpu(int cpu) const {
    for (const auto &n : nodes_)
        for (const auto &r : n.cpu_ranges)
            if (cpu >= r.first && cpu <= r.last) return n.id;
    return -1;
}

int numa_topology_t::num_nodes_for_threads(int nthr) const {
    if (nthr <= 0 || num_nodes_ == 0) return 0;

    // Threads are assumed to occupy a contiguous range of logical CPU ids
    // [0, nthr-1]. Count how many NUMA nodes contain at least one CPU in
    // that range.
    int nodes_used = 0;
    for (const auto &n : nodes_) {
        bool intersects = false;
        for (const auto &r : n.cpu_ranges) {
            // Range [r.first, r.last] intersects [0, nthr-1] iff
            //   r.first <= nthr-1  &&  r.last >= 0.
            if (r.first <= nthr - 1 && r.last >= 0) {
                intersects = true;
                break;
            }
        }
        if (intersects) ++nodes_used;
    }
    return nodes_used;
}

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

} // namespace platform
} // namespace cpu
} // namespace impl
} // namespace dnnl
