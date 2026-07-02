/*******************************************************************************
* Copyright 2026 Barcelona Supercomputing Center
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

#ifndef CPU_RV64_RVJIT_RVJIT_RVV_HPP
#define CPU_RV64_RVJIT_RVJIT_RVV_HPP

#include <algorithm>

#include "cpu/rv64/rvjit/rvjit_types.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace rvjit {

/// Obtains the RVV LMUL value for a given group size
/// @details defaults to m1 for group_sizes other than 2, 4, and 8.
inline LMUL lmul_for(const int group_size) {
    switch (group_size) {
        case 1: return LMUL::m1;
        case 2: return LMUL::m2;
        case 4: return LMUL::m4;
        case 8: return LMUL::m8;
        default: return LMUL::m1;
    }
}

/// Obtain the number of vector registers in a group for a given lmul setting
inline int vgroup_size(const LMUL &m) {
    switch (m) {
        case LMUL::m1: return 1;
        case LMUL::m2: return 2;
        case LMUL::m4: return 4;
        case LMUL::m8: return 8;
        default: return 1;
    }
}

/// Obtain the number of vector register groups for a given lmul setting
inline int get_nvgroups(const LMUL &m) {
    switch (m) {
        case LMUL::m1: return 32;
        case LMUL::m2: return 16;
        case LMUL::m4: return 8;
        case LMUL::m8: return 4;
        default: return 1;
    }
}

/// Checks if LMUL A generates vectors larger than or equal to B
inline bool is_lmul_gte(const LMUL &A, const LMUL &B) {
    return vgroup_size(A) >= vgroup_size(B);
}

/// Single-element-width size in bytes for a given sew setting
inline int sewb(const SEW &s) {
    switch (s) {
        case SEW::e8: return 1;
        case SEW::e16: return 2;
        case SEW::e32: return 4;
        case SEW::e64: return 8;
        default: return 0;
    }
}

/// Obtains the RVV LMUL value for a base group size and in/out SEW
inline LMUL lmul_for(LMUL base, SEW in, SEW out) {
    return lmul_for(vgroup_size(base) * (sewb(in) / sewb(out)));
}

/// Computes the maximum vector length given a certain VPU configuration
inline int get_maximum_vector_length(const int vlenb, const SEW &sew = SEW::e8,
        const LMUL &lmul = LMUL::m1) {
    return vlenb * vgroup_size(lmul) / sewb(sew);
}

/// Hardware model driving arithmetic micro-kernel unrolling decisions
struct rvv_t {
    struct vpu_t {
        int vlenb = 0;
        int vlaneb = 0;
        LMUL max_lmul = LMUL::m8;
        LMUL lmul_preference = LMUL::m1;

        vpu_t() = default;
        vpu_t(int vlenb, int vlaneb, LMUL max_lmul, LMUL lmul_preference)
            : vlenb(vlenb)
            , vlaneb(vlaneb)
            , max_lmul(max_lmul)
            , lmul_preference(lmul_preference) {}

        int vector_latency() const { return vlaneb > 0 ? vlenb / vlaneb : 1; }

        /// Accumulator group ceiling at lmul_preference
        ///
        /// @note Reserves 1 (mixed precision, or m8) or 2 (otherwise) groups
        ///     for pipeline buffers, then caps the result at 16
        int max_n_accumulators(SEW sew_inp, SEW sew_acc) const {
            const int n = get_nvgroups(lmul_preference);
            const int reserved
                    = (sew_inp != sew_acc || lmul_preference == LMUL::m8) ? 1
                                                                          : 2;
            return std::min(16, std::max(1, n - reserved));
        }
    };

    // Cache level feeding the vector loads/stores (not necessarily L1)
    struct memory_t {
        int cache_size = 0;
        int cache_line_size = 0;
        int cache_latency = 0;

        memory_t() = default;
        memory_t(int cache_size, int cache_line_size, int cache_latency)
            : cache_size(cache_size)
            , cache_line_size(cache_line_size)
            , cache_latency(cache_latency) {}

        int memory_latency() const { return cache_latency; }
    };

    vpu_t vpu;
    memory_t memory;

    rvv_t() = default;
    rvv_t(const vpu_t &vpu, const memory_t &memory)
        : vpu(vpu), memory(memory) {}

    int vector_latency() const { return vpu.vector_latency(); }
    int memory_latency() const { return memory.memory_latency(); }

    /// Builds a model from explicit hardware parameters
    ///
    /// @details `vlen`/`cache_size`/`cache_line_size`/`cache_latency` are
    ///     plain numbers the caller has already obtained (typically by
    ///     querying the real hardware) — this method is pure arithmetic on
    ///     top of them, with no hardware-querying of its own
    static rvv_t from_params(
            int vlen, int cache_size, int cache_line_size, int cache_latency) {
        // Long vector architecture vlen
        static constexpr int vlong = 2048;

        // Vector architectural storage
        const int vlenb = vlen / 8;

        const auto mem = memory_t(cache_size, cache_line_size, cache_latency);

        // Guess the vector lane width from vlen: short vectors use a
        // pipelined 64-bit SIMD unit, mid ones widen to control latency,
        // and long ones match the cache-line size.
        const int vlaneb = vlen <= 256 ? 8
                : vlen <= 1024         ? 16
                                       : cache_line_size;

        // Limit vector grouping to the limits of vlong
        LMUL max_lmul = LMUL::m8;
        for (LMUL m : {LMUL::m1, LMUL::m2, LMUL::m4, LMUL::m8}) {
            if (vlen * vgroup_size(m) >= vlong) {
                max_lmul = m;
                break;
            }
        }

        // Set a preferred LMUL setting with enough registers to mask latency.
        // m8 is deliberately excluded: its 4 register groups leave too few
        // accumulators after reserving buffers, so m4 is the widest a
        // preference ever settles on; only max_lmul (above) can force this
        // narrower still, for long-vector architectures.
        LMUL lmul_preference = LMUL::m1;
        const int lat = vlaneb > 0 ? vlenb / vlaneb : 0;
        for (LMUL m : {LMUL::m4, LMUL::m2, LMUL::m1}) {
            if (is_lmul_gte(max_lmul, m) && lat <= get_nvgroups(m)) {
                lmul_preference = m;
                break;
            }
        }

        return rvv_t(vpu_t(vlenb, vlaneb, max_lmul, lmul_preference), mem);
    }
};

} // namespace rvjit
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
