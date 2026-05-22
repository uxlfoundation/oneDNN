/*******************************************************************************
* Copyright 2020 Intel Corporation
* Copyright 2020 Arm Ltd. and affiliates
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

#ifndef CPU_PLATFORM_HPP
#define CPU_PLATFORM_HPP

#include "oneapi/dnnl/dnnl_config.h"

#include <vector>

#include "common/c_types_map.hpp"
#include "common/impl_registration.hpp"
#include "common/z_magic.hpp"

// Possible architectures:
// - DNNL_X64
// - DNNL_AARCH64
// - DNNL_PPC64
// - DNNL_S390X
// - DNNL_RV64
// - DNNL_ARCH_GENERIC
// Target architecture macro is set to 1, others to 0. All macros are defined.

#if defined(DNNL_X64) + defined(DNNL_AARCH64) + defined(DNNL_PPC64) \
                + defined(DNNL_S390X) + defined(DNNL_RV64) \
                + defined(DNNL_ARCH_GENERIC) \
        == 0
#if defined(__x86_64__) || defined(_M_X64)
#define DNNL_X64 1
#elif defined(__aarch64__)
#define DNNL_AARCH64 1
#elif defined(__powerpc64__) || defined(__PPC64__) || defined(_ARCH_PPC64)
#define DNNL_PPC64 1
#elif defined(__s390x__)
#define DNNL_S390X 1
#elif defined(__riscv)
#define DNNL_RV64 1
#else
#define DNNL_ARCH_GENERIC 1
#endif
#endif // defined(DNNL_X64) + ... == 0

#if defined(DNNL_X64) + defined(DNNL_AARCH64) + defined(DNNL_PPC64) \
                + defined(DNNL_S390X) + defined(DNNL_RV64) \
                + defined(DNNL_ARCH_GENERIC) \
        != 1
#error One and only one architecture should be defined at a time
#endif

#if !defined(DNNL_X64)
#define DNNL_X64 0
#endif
#if !defined(DNNL_AARCH64)
#define DNNL_AARCH64 0
#endif
#if !defined(DNNL_PPC64)
#define DNNL_PPC64 0
#endif
#if !defined(DNNL_S390X)
#define DNNL_S390X 0
#endif
#if !defined(DNNL_RV64)
#define DNNL_RV64 0
#endif
#if !defined(DNNL_ARCH_GENERIC)
#define DNNL_ARCH_GENERIC 0
#endif

// Helper macros: expand the parameters only on the corresponding architecture.
// Equivalent to: #if DNNL_$ARCH ... #endif
#define DNNL_X64_ONLY(...) Z_CONDITIONAL_DO(DNNL_X64, __VA_ARGS__)
#define DNNL_PPC64_ONLY(...) Z_CONDITIONAL_DO(DNNL_PPC64, __VA_ARGS__)
#define DNNL_S390X_ONLY(...) Z_CONDITIONAL_DO(DNNL_S390X_ONLY, __VA_ARGS__)
#define DNNL_AARCH64_ONLY(...) Z_CONDITIONAL_DO(DNNL_AARCH64, __VA_ARGS__)

// Using RISC-V implementations optimized with RVV Intrinsics is optional for RISC-V builds
// and can be enabled with DNNL_ARCH_OPT_FLAGS="-march=<ISA-string>" option, where <ISA-string>
// contains V extension. If disabled, generic reference implementations will be used.
#if defined(DNNL_RV64) && defined(DNNL_RISCV_USE_RVV_INTRINSICS)
#define DNNL_RV64GCV_ONLY(...) __VA_ARGS__
#else
#define DNNL_RV64GCV_ONLY(...)
#endif

// Zvfh intrinsics are enabled if V extension is enabled and Zvfh is supported by the toolchain.
#if defined(DNNL_RV64) && defined(DNNL_RISCV_USE_ZVFH_INTRINSICS)
#define DNNL_RV64GCV_ZVFH_ONLY(...) __VA_ARGS__
#else
#define DNNL_RV64GCV_ZVFH_ONLY(...)
#endif

// Negation of the helper macros above
#define DNNL_NON_X64_ONLY(...) Z_CONDITIONAL_DO(Z_NOT(DNNL_X64), __VA_ARGS__)

// Using Arm Compute Library kernels is optional for AArch64 builds
// and can be enabled with the DNNL_AARCH64_USE_ACL CMake option
#if defined(DNNL_AARCH64) && defined(DNNL_AARCH64_USE_ACL)
#define DNNL_AARCH64_ACL_ONLY(...) __VA_ARGS__
#else
#define DNNL_AARCH64_ACL_ONLY(...)
#endif

// Primitive ISA section for configuring knobs.
// Note: MSVC preprocessor by some reason "eats" symbols it's not supposed to
// if __VA_ARGS__ is passed as empty. Then things happen like this for non-x64:
// impl0, AMX(X64_impl1), impl2, ... -> impl0   impl2, ...
// resulting in compilation error. Such problem happens for lists interleaving
// X64 impls and non-X64 for non-X64 build.
#if DNNL_X64
// Note: unlike workload or primitive set, these macros will work with impl
// items directly, thus, just make an item disappear, no empty lists.
#define __BUILD_AMX BUILD_PRIMITIVE_CPU_ISA_ALL || BUILD_AMX
#define __BUILD_AVX512 __BUILD_AMX || BUILD_AVX512
#define __BUILD_AVX2 __BUILD_AVX512 || BUILD_AVX2
#define __BUILD_SSE41 __BUILD_AVX2 || BUILD_SSE41
#else
#define __BUILD_AMX 0
#define __BUILD_AVX512 0
#define __BUILD_AVX2 0
#define __BUILD_SSE41 0
#endif

#if __BUILD_AMX
#define REG_AMX_ISA(...) __VA_ARGS__
#else
#define REG_AMX_ISA(...)
#endif

#if __BUILD_AVX512
#define REG_AVX512_ISA(...) __VA_ARGS__
#else
#define REG_AVX512_ISA(...)
#endif

#if __BUILD_AVX2
#define REG_AVX2_ISA(...) __VA_ARGS__
#else
#define REG_AVX2_ISA(...)
#endif

#if __BUILD_SSE41
#define REG_SSE41_ISA(...) __VA_ARGS__
#else
#define REG_SSE41_ISA(...)
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace platform {

const char *get_isa_info();
dnnl_cpu_isa_t get_effective_cpu_isa();
status_t set_max_cpu_isa(dnnl_cpu_isa_t isa);
status_t set_cpu_isa_hints(dnnl_cpu_isa_hints_t isa_hints);
dnnl_cpu_isa_hints_t get_cpu_isa_hints();

bool DNNL_API prefer_ymm_requested();
// This call is limited to performing checks on plain C-code implementations
// (e.g. 'ref' and 'simple_primitive') and should avoid any x64 JIT
// implementations since these require specific code-path updates.
bool DNNL_API has_data_type_support(data_type_t data_type);
bool DNNL_API has_training_support(data_type_t data_type);
float DNNL_API s8s8_weights_scale_factor();

unsigned DNNL_API get_per_core_cache_size(int level);
uint32_t get_num_ways_in_cache(int level);
uint32_t get_num_sets_in_cache(int level);
unsigned DNNL_API get_num_cores();
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
unsigned DNNL_API get_max_threads_to_use();
#endif

// Encapsulates NUMA (incl. SubNUMA-cluster) topology of the system.
//
// Topology is queried from the OS only once, in the constructor. After that
// all accessors are cheap, read-only lookups. Use `instance()` to get a
// process-wide cached object, or construct your own if you need a fresh
// snapshot (e.g. for tests).
//
// On Linux the data is parsed from /sys/devices/system/node/nodeN/cpulist.
// On other platforms the topology is reported as empty / not available.
class DNNL_API numa_topology_t {
public:
    // Inclusive numeric range of contiguous CPU ids: [first, last].
    struct cpu_range_t {
        int first;
        int last;

        int size() const { return last - first + 1; }
    };

    // Compact description of a single NUMA node.
    struct node_info_t {
        int id; // NUMA node id as reported by the OS
        // CPU ids that belong to this node, collapsed into contiguous
        // ranges (matches the "0-31,192-223" style produced by lscpu, but
        // as numeric pairs). A node may have multiple ranges (e.g. when
        // SMT pairs the two threads of a core into very different CPU id
        // spaces).
        std::vector<cpu_range_t> cpu_ranges;

        // Total number of CPUs in this node (sum of all range sizes).
        int num_cpus() const {
            int n = 0;
            for (const auto &r : cpu_ranges)
                n += r.size();
            return n;
        }
    };

    numa_topology_t();

    // Process-wide cached instance (lazily initialized, thread-safe).
    static const numa_topology_t &instance();

    // True if the OS reported at least one NUMA node.
    bool is_available() const { return num_nodes_ > 0; }

    // Number of NUMA nodes actually present.
    size_t num_nodes() const { return num_nodes_; }

    // All present nodes (entries with id >= 0), in id order.
    const std::vector<node_info_t> &nodes() const { return nodes_; }

    // Look up a node by its OS-reported id. Returns nullptr if absent.
    const node_info_t *node_by_id(int id) const;

    // For a logical CPU id, return the owning node's id, or -1 if unknown.
    int node_of_cpu(int cpu) const;

    // Returns the number of NUMA nodes that contain at least one CPU from
    // the contiguous range [0, nthr-1]. Use this to estimate how many NUMA
    // nodes a thread pool of size `nthr` would touch when its threads are
    // pinned to logical CPUs starting from CPU 0. Returns 0 if topology is
    // not available or nthr <= 0.
    int num_nodes_for_threads(int nthr) const;

private:
    // Queries NUMA topology from the OS.
    // On Linux this is parsed from /sys/devices/system/node/nodeN/cpulist.
    // The outer vector is indexed by NUMA node id; each inner vector
    // contains the logical CPU ids that belong to that node (sorted,
    // expanded from the "0-31,192-223" style ranges). On unsupported
    // platforms the result is empty.
    static std::vector<std::vector<int>> query_numa_node_cpus();

    std::vector<node_info_t> nodes_;
    size_t num_nodes_ = 0;
};

constexpr int get_cache_line_size() {
    return 64;
}

int get_vector_register_size();

// Helper to avoid #ifdefs for DNNL_PPC64
static constexpr bool is_ppc64() {
#if DNNL_PPC64
    return true;
#else
    return false;
#endif
}

size_t get_timestamp();

} // namespace platform

// XXX: find a better place for these values?
enum {
    PAGE_4K = 4096,
    PAGE_2M = 2097152,
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
