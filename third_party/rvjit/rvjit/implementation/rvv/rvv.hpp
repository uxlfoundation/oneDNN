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

#ifndef RVJIT_IMPLEMENTATION_RVV_RVV_HPP
#define RVJIT_IMPLEMENTATION_RVV_RVV_HPP

#include <algorithm>

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace rvv {

/// The SEW constant associated to an operand type
///
/// @param dt The operand type
inline SEW sew_for(const type_t &dt) {
    switch (rvjit::sewb(dt)) {
        case 8: return SEW::e64;
        case 4: return SEW::e32;
        case 2: return SEW::e16;
        default: return SEW::e8;
    }
}

/// Obtains the LMUL of a group of `num / den` registers, whole or fractional
///
/// @param num The number of registers in a group, or the numerator of a
///     fraction of one
/// @param den The denominator of the register count
///
/// @note Defaults to m1 for counts other than 1/8 up to 8 in powers of two
inline LMUL lmul_for(const int num, const int den = 1) {
    if (num <= 0 || den <= 0) return LMUL::m1;
    if (num < den) {
        switch (den / num) {
            case 2: return LMUL::mf2;
            case 4: return LMUL::mf4;
            case 8: return LMUL::mf8;
            default: return LMUL::m1;
        }
    }
    switch (num / den) {
        case 1: return LMUL::m1;
        case 2: return LMUL::m2;
        case 4: return LMUL::m4;
        case 8: return LMUL::m8;
        default: return LMUL::m1;
    }
}

/// Obtain the number of vector register groups for a given LMUL setting
///
/// @param m The LMUL setting
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
///
/// @param A The LMUL setting compared
/// @param B The LMUL setting compared against
inline bool is_lmul_gte(const LMUL &A, const LMUL &B) {
    return vgroup_size(A) >= vgroup_size(B);
}

/// Get the single-element-width size, in bytes, associated to a SEW setting
///
/// @param s The SEW setting
inline int sewb(const SEW &s) {
    switch (s) {
        case SEW::e8: return 1;
        case SEW::e16: return 2;
        case SEW::e32: return 4;
        case SEW::e64: return 8;
        default: return 0;
    }
}

/// Obtains the LMUL representation for a base LMUL and in/out SEW settings
///
/// @param base The LMUL setting of the input, whole or fractional
/// @param in The SEW setting of the input
/// @param out The SEW setting of the output
///
/// @details Scales the group by the ratio of the widths, so that both settings
///     hold the same number of elements
inline LMUL lmul_for(LMUL base, SEW in, SEW out) {
    return lmul_for(
            vgroup_size(base) * sewb(out), lmul_fraction(base) * sewb(in));
}

/// Computes the maximum vector length given a certain vector type settings
///
/// @param vlenb The vector register length in bytes
/// @param sew The SEW setting
/// @param lmul The LMUL setting, whole or fractional
inline int get_maximum_vector_length(const int vlenb, const SEW &sew = SEW::e8,
        const LMUL &lmul = LMUL::m1) {
    return vlenb * vgroup_size(lmul) / lmul_fraction(lmul) / sewb(sew);
}

/// Hardware model driving arithmetic micro-kernel unrolling decisions
struct vpu_t {
    int vlenb = 0;
    int vlaneb = 0;

    // Cache level feeding the vector loads/stores (not necessarily L1)
    int cache_size = 0;
    int cache_line_size = 0;
    int cache_latency = 0;

    vpu_t() = default;
    virtual ~vpu_t() = default;

    /// Creates a model from its derived fields
    ///
    /// @param vlenb The vector register length in bytes
    /// @param vlaneb The vector lane width in bytes
    /// @param cache_size The size of the cache level feeding the vector unit
    /// @param cache_line_size The line size of that cache level
    /// @param cache_latency The latency of that cache level
    vpu_t(int vlenb, int vlaneb, int cache_size, int cache_line_size,
            int cache_latency)
        : vlenb(vlenb)
        , vlaneb(vlaneb)
        , cache_size(cache_size)
        , cache_line_size(cache_line_size)
        , cache_latency(cache_latency) {}

    /// The latency of a vector instruction, in lane passes
    int vector_latency() const { return vlaneb > 0 ? vlenb / vlaneb : 1; }

    /// The latency of a vector memory instruction
    int memory_latency() const { return cache_latency; }

    /// Fills the model in from explicit hardware parameters
    ///
    /// @param vlen The vector register length in bits
    /// @param cache_size The size of the cache level feeding the vector unit
    /// @param cache_line_size The line size of that cache level
    /// @param cache_latency The latency of that cache level
    ///
    /// @note The parameters are plain numbers the caller has already
    ///     obtained; this method performs no hardware querying of its own
    virtual void from_params(
            int vlen, int cache_size, int cache_line_size, int cache_latency) {
        // Vector architectural storage
        vlenb = vlen / 8;

        // Guess the vector lane width from vlen: short vectors use a
        // pipelined 64-bit SIMD unit, mid ones widen to control latency,
        // and long ones match the cache-line size.
        vlaneb = vlen <= 256 ? 8 : vlen <= 1024 ? 16 : cache_line_size;

        this->cache_size = cache_size;
        this->cache_line_size = cache_line_size;
        this->cache_latency = cache_latency;
    }
};

// Internal linkage avoiding an out-of-line definition
static const Reg ignore_out(0); // Ignore `set_vtype`'s own output

/// Vector type settings (SEW, LMUL, policies) of a micro-kernel
///
/// @details Input and output views of SEW/LMUL support mixed precision
struct vtype_t {
    /// Selects which SEW/LMUL pair to apply on a vtype setter operation
    ///
    /// @note `output` is the accumulators'
    enum class cfg_t { input, output };

    SEW sewi = SEW::e8; // ESEW for input operands
    SEW sewo = SEW::e8; // ESEW for output operands
    LMUL lmuli = LMUL::m1; // ELMUL for input operands
    LMUL lmulo = LMUL::m1; // ELMUL for output operands
    VTA vt = VTA::ta; // Vector tail policy, uses agnostic as default
    VMA vm = VMA::ma; // Vector mask policy, uses agnostic as default
    Reg avl {}; // AVL register consumed directly by set_vtype()'s vsetvli

    /// Configures vtype for an uniform-precision code segment
    ///
    /// @param sew The SEW setting of inputs and outputs
    /// @param lmul The LMUL setting of inputs and outputs
    ///
    /// @note Does not validate `lmul` against any hardware ceiling; the
    ///     caller is responsible for passing an `lmul` its target supports
    void for_uniform(SEW sew, LMUL lmul) {
        sewi = sewo = sew;
        lmuli = lmulo = lmul;
    }

    /// Configures vtype for a widening mixed-precision code segment
    ///
    /// @param sewi The SEW setting of the inputs
    /// @param sewo The SEW setting of the outputs
    /// @param lmul The LMUL setting of the inputs, from which the LMUL of the
    ///     outputs is derived
    ///
    /// @note Does not validate the sewi/sewo ratio or lmul/the derived
    ///     lmulo against any hardware ceiling
    void for_mixed(SEW sewi, SEW sewo, LMUL lmul) {
        this->sewi = sewi;
        this->sewo = sewo;
        lmuli = lmul;
        lmulo = lmul_for(lmul, sewi, sewo);
    }

    /// Whether input and output share the same SEW/LMUL settings
    bool is_uniform() const { return sewi == sewo && lmuli == lmulo; }

    /// Whether input and output have different SEW/LMUL settings
    bool is_mixed() const { return !is_uniform(); }
};

/// The LMUL that keeps the output configuration's vector length at `sew`
///
/// @param vt The vector type settings, whose output SEW and LMUL are scaled
/// @param sew The element width switched to
/// @param lmul The LMUL found
///
/// @details The LMUL scales the output's by the ratio of the widths. A width
///     narrower than a whole register takes the fractional LMUL, which keeps
///     the SEW/LMUL ratio, and so the vector length, of the output
///     configuration
///
/// @note Returns false, leaving `lmul` alone, when that LMUL is beyond m8 or
///     below the SEW/ELEN floor of a 64-bit ELEN
inline bool keep_vl_lmul(const vtype_t &vt, SEW sew, LMUL &lmul) {
    const int num = vgroup_size(vt.lmulo) * sewb(sew);
    const int den = lmul_fraction(vt.lmulo) * sewb(vt.sewo);
    const bool whole = num >= den;
    const int scaled = whole ? num / den : den / num;
    static constexpr int elen_bytes = 8;
    const bool fits = (num % den == 0 || den % num == 0)
            && (whole ? scaled <= 8 : scaled * sewb(sew) <= elen_bytes);
    if (!fits) return false;
    lmul = lmul_for(num, den);
    return true;
}

} // namespace rvv
} // namespace rvjit

#endif
