/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include <fstream>
#include <string>
#include <sys/auxv.h>
#include <type_traits>

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
};

enum cpu_isa_t : unsigned {
    isa_undef = 0u,
    v = v_bit,
    zvfh = zvfh_bit | v,
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

private:
    bool has_v = false;
    bool has_zvfh = false;

    // Parse /proc/cpuinfo to find Zvfh
    bool detect_zvfh_procfs() {
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;

        if (!cpuinfo.is_open()) { return false; }

        while (std::getline(cpuinfo, line)) {
            if (line.rfind("isa", 0) == 0) {
                // Find "_zvfh"
                if (line.find("_zvfh") != std::string::npos) {
                    cpuinfo.close();
                    return true;
                }
                // Only need to check the isa line for the first core
                break;
            }
        }
        cpuinfo.close();
        return false;
    }

    Riscv64Cpu() {
        // Use xbyak_riscv for V extension detection
        const auto &xbyak_cpu = Xbyak_riscv::CPU::getInstance();
        has_v = xbyak_cpu.hasExtension(Xbyak_riscv::RISCVExtension::V);

        // Zvfh detection: xbyak_riscv doesn't support half-precision yet,
        // so continue using procfs method
        if (has_v) {
            // TODO: Prioritize riscv_hwprobe when available (Linux 6.4+)
            has_zvfh = detect_zvfh_procfs();
        } else {
            has_zvfh = false;
        }
    }
};

inline bool mayiuse(const cpu_isa_t cpu_isa, bool soft = false) {
    MAYBE_UNUSED(soft);
    const Riscv64Cpu &cpu = Riscv64Cpu::getInstance();

    switch (cpu_isa) {
        case v: return cpu.get_has_v();
        case zvfh: return cpu.get_has_v() && cpu.get_has_zvfh();
        case isa_undef: return true;
        case isa_all: return false;
    }
    return false;
}

cpu_isa_t get_max_cpu_isa();

#include "common/z_magic.hpp"
/* clang-format off */
#define JIT_IMPL_NAME_HELPER(prefix, isa, suffix_if_any) \
    ((isa) == isa_undef ? prefix STRINGIFY(any) : \
    ((isa) == v ? prefix STRINGIFY(rvv) : \
    ((isa) == zvfh ? prefix STRINGIFY(rvv_zvfh) : \
    prefix suffix_if_any)))
/* clang-format on */

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
