/*******************************************************************************
* Copyright 2018 Intel Corporation
* Copyright 2025 Institute of Software, Chinese Academy of Sciences
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

#ifndef CPU_RV64_CPU_ISA_TRAITS_HPP
#define CPU_RV64_CPU_ISA_TRAITS_HPP

#include <type_traits>

#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "dnnl_types.h"

#ifndef XBYAK_RISCV_V
#define XBYAK_RISCV_V 1
#endif

#include "xbyak_riscv/xbyak_riscv_util.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

enum cpu_isa_bit_t : unsigned {
    v_bit = 1u << 0,
    zvfh_bit = 1u << 1,
    zvfbfmin_bit = 1u << 2,
    zvfbfwma_bit = 1u << 3,
};

enum cpu_isa_t : unsigned {
    isa_undef = 0u,
    v = v_bit,
    zvfh = zvfh_bit | v,
    zvfbfmin = zvfbfmin_bit | v,
    zvfbfwma = zvfbfwma_bit | zvfbfmin,
    isa_all = ~0u,
};

struct Riscv64Cpu {
public:
    static Riscv64Cpu &getInstance() {
        static Riscv64Cpu instance;
        return instance;
    }

    bool get_has_v() const { return has_v; }
    bool get_has_zvfh() const { return has_zvfh; }
    uint32_t get_vlen() const { return vlen; }

private:
    bool has_v = false;
    bool has_zvfh = false;
    uint32_t vlen = 0;

    Riscv64Cpu() {
        const auto &xbyak_cpu = Xbyak_riscv::CPU::getInstance();

        has_v = xbyak_cpu.hasExtension(Xbyak_riscv::RISCVExtension::V);
        vlen = xbyak_cpu.getVlen();

        if (has_v) {
            has_zvfh
                    = xbyak_cpu.hasExtension(Xbyak_riscv::RISCVExtension::Zvfh);
        } else {
            has_zvfh = false;
        }
    }
};

// Zvfbfmin / Zvfbfwma are gated at build time by the compiler's own
// extension macros. The cmake march test pins the build flag based on
// the running toolchain + CPU; the binary contains the bf16 path only
// when -march already advertises it, so we trust the build flag as the
// runtime answer too. This keeps the binary self-consistent in exchange
// for assuming build target == run target.
inline bool mayiuse(const cpu_isa_t cpu_isa, bool soft = false) {
    MAYBE_UNUSED(soft);
    const Riscv64Cpu &cpu = Riscv64Cpu::getInstance();

    switch (cpu_isa) {
        case v: return cpu.get_has_v();
        case zvfh: return cpu.get_has_v() && cpu.get_has_zvfh();
        case zvfbfmin:
#ifdef __riscv_zvfbfmin
            return cpu.get_has_v();
#else
            return false;
#endif
        case zvfbfwma:
#ifdef __riscv_zvfbfwma
            return cpu.get_has_v();
#else
            return false;
#endif
        case isa_undef: return true;
        case isa_all: return false;
    }
    return false;
}

cpu_isa_t get_max_cpu_isa();

inline uint32_t get_platform_vlen() {
    const Riscv64Cpu &cpu = Riscv64Cpu::getInstance();
    return cpu.get_vlen();
}

/// Returns an index derived from a given vector length in bits (vlen).
/// @details Computes log2(vlen) relative to the minimum supported value,
/// producing a zero-based index. The smallest vlen (128-bits) maps to 0,
/// and each subsequent power-of-two step maps to the next integer.
inline int get_vlen_implementation_id(int vlen) {
    static constexpr int VLEN_MIN = 128;
    if (math::is_pow2(vlen) && vlen >= VLEN_MIN) {
        return math::ilog2q(vlen) - math::ilog2q(VLEN_MIN);
    } else {
        return -1;
    }
}

#include "common/z_magic.hpp"
/* clang-format off */
#define JIT_IMPL_NAME_HELPER(prefix, isa, suffix_if_any) \
    ((isa) == isa_undef ? prefix STRINGIFY(any) : \
    ((isa) == v ? prefix STRINGIFY(rvv) : \
    ((isa) == zvfh ? prefix STRINGIFY(rvv_zvfh) : \
    ((isa) == zvfbfmin ? prefix STRINGIFY(rvv_zvfbfmin) : \
    ((isa) == zvfbfwma ? prefix STRINGIFY(rvv_zvfbfwma) : \
    prefix suffix_if_any)))))
/* clang-format on */

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
