#ifndef CPU_X64_PLATFORM_HPP
#define CPU_X64_PLATFORM_HPP

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace platform {

// returns true if the CPU is a hybrid CPU
// (e.g. Alder Lake, Raptor Lake, Lunar Lake)
bool is_hybrid();

// p_core: Performance core (high performance, high power consumption)
// e_core: Efficiency core (low performance, low power consumption)
enum class core_type : int {
    p_core = 0, // Performance core
    e_core = 1, // Efficiency core
    default_core = p_core // Default core (used for non-hybrid CPUs)
};


// get the core_type of the core the calling thread is running on
// to get the core_type of a specific core, set thread affinity
// to that core
core_type get_core_type();

enum class behavior_t {
    p_core, // Performance core
    e_core, // Efficiency core
    current, // Current core
    min, // used to select the smallest value for all the cores
    max, // used to select the largest value for all the cores
    unknown
};

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
// TODO: George test on systems that only have E-cores
unsigned get_per_core_cache_size(int level, behavior_t btype);

} // namespace dnnl::impl::cpu::x64::platform
} // namespace dnnl::impl::cpu::x64
} // namespace dnnl::impl::cpu
} // namespace dnnl::impl
} // namespace dnnl

#endif