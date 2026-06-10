/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "cpu/rv64/cpu_isa_traits.hpp"
#include "cpu/platform.hpp"

#if defined(__linux__) && defined(__riscv) && (__riscv_xlen == 64)
#include <csetjmp>
#include <csignal>
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

#if defined(__linux__) && defined(__riscv) && (__riscv_xlen == 64)

namespace {

// SIGILL-trap probe for Zvfbfwma. Linux HWPROBE may not surface the
// extension yet on a given chip, so we try to execute one
// vfwmaccbf16.vf in a sigaction-protected block. The probe runs once
// from the Riscv64Cpu singleton.

sigjmp_buf bf16_probe_jmp;

void bf16_probe_sigill_handler(int) {
    siglongjmp(bf16_probe_jmp, 1);
}

bool probe_zvfbfwma_impl() {
    struct sigaction sa {};
    sa.sa_handler = bf16_probe_sigill_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    struct sigaction old_sa {};
    if (sigaction(SIGILL, &sa, &old_sa) != 0) return false;

    bool ok = false;
    if (sigsetjmp(bf16_probe_jmp, 1) == 0) {
        // vsetivli zero, 4, e16, m1, ta, ma   (0xcc807057)
        // vfwmaccbf16.vf v2, fa0, v6          (0xee655157)
        // Raw bytes because older binutils may lack the mnemonics.
        asm volatile(
                ".4byte 0xcc807057\n"
                ".4byte 0xee655157\n"
                :
                :
                : "memory");
        ok = true;
    }
    sigaction(SIGILL, &old_sa, nullptr);
    return ok;
}

} // namespace

bool probe_zvfbfwma() {
    return probe_zvfbfwma_impl();
}

#else

bool probe_zvfbfwma() {
    return false;
}

#endif

struct isa_info_t {
    isa_info_t(cpu_isa_t aisa) : isa(aisa) {}
    cpu_isa_t isa;
};

static isa_info_t get_isa_info_t(void) {
    if (mayiuse(zvfh)) return isa_info_t(zvfh);
    if (mayiuse(v)) return isa_info_t(v);
    return isa_info_t(isa_undef);
}

cpu_isa_t get_max_cpu_isa() {
    return get_isa_info_t().isa;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
