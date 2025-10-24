/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

// P-core: Performance core (high performance, high power consumption)
// E-core: Efficiency core (low performance, low power consumption)
// However the naming in the Intel SDM is different:
//  - core for P-core
//  - atom for E-core
enum class core_type : int {
    p_core = 0, // Performance core
    e_core = 1, // Efficiency core
    default_core = p_core // Default core (used for non-hybrid CPUs)
};

// returns true if the CPU is a hybrid CPU
// (e.g. Alder Lake, Raptor Lake, Lunar Lake)
bool is_hybrid();

enum class behavior_t {
    p_core, // Performance core
    e_core, // Efficiency core
    current, // Current core
    min, // used to select the smallest value for all the cores
    max, // used to select the largest value for all the cores
    unknown
};

// get the core_type of the core the calling thread is running on
// to get the core_type of a specific core, set thread affinity to that core
core_type get_core_type();

// Use OS specific methods to determine the per-core cache size.
//
// This is avoids using CPUID-based methods which can result in inaccurate values
//
// Expected to be called in place of get_per_core_cache_size(level) the
// behavior_t argument specifies the behavior of the query on hybrid CPUs.
//
// - if behavior_t is p_core/e_core, the function returns the per-core cache
//   size for that core type.
// - if behavior_t is min/max, the function returns the min/max per-core cache
//   size among all cores
// - if behavior_t is current, the function returns the cache size of the core
//   the calling thread is running on.
// - if behavior_t is unknown, the function behaves like legacy
//   get_per_core_cache_size(level) function.
//
// Examples: (showing KB and MB values for clarity actual function returns bytes)
// for a hybrid CPU with (e.g. Alder Lake)
//   48KB L1d cache on P-cores (with hyperthreading) and
//   32KB L1d cache on E-cores
// get_per_core_cache_size(1, core_type::min)
//   returns 24KB (48KB/2) (P-core).
// get_per_core_cache_size(1, core_type::max)
//   returns 32KB (E-core).
// get_per_core_cache_size(1, core_type::p_core)
//   returns 24KB
// get_per_core_cache_size(1, core_type::e_core)
//   returns 32KB
// get_per_core_cache_size(1, core_type::current)
//   returns 24KB or 32KB depending on the core the thread is running on.
//
// for a hybrid CPU (4p+4e) with (e.g.Lunar Lake)
//   3MB L2 cache on P-cores (no hyperthreading) and
//   4MB L2 cache on E-cores (shared in 4-core clusters)
// get_per_core_cache_size(2, core_type::max)
//   returns 3MB
// get_per_core_cache_size(2, core_type::min)
//   returns 1MB (4MB/4)
// get_per_core_cache_size(2, core_type::p_core)
//   returns 3MB
// get_per_core_cache_size(2, core_type::e_core)
//   returns 1MB (4MB/4)
// get_per_core_cache_size(2, core_type::current)
//   returns 3MB or 1MB depending on the core the thread is running on
// get_per_core_cache_size(3, core_type::unknown)
//   uses the get_per_core_cache_size(int level) function.
//
// TODO: Test behavior on non-hybrid CPUs.
unsigned DNNL_API get_per_core_cache_size(int level, behavior_t btype);

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
