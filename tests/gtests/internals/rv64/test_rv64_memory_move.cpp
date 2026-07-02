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

#include <cstdint>

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "cpu/rv64/rvjit/rvjit.hpp"
#include "rv64_jit_test_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace Xbyak_riscv;
using namespace rvjit;

// Invalid combinations: all emit nothing, leaving only ret().

struct move_noop_probe_t : public rv64_jit_test_generator_t {
protected:
    void generate() override {
        rvjit_t m(*this);
        auto &mem = m.memory_move();

        // None of these should emit anything
        mem.fload(ft0, t0, SEW::e8, 0); // fload has no e8 case
        mem.xload(t0, t0, SEW::e16, true, 0); // xload has no e16 case

        ret();
    }
};

HANDLE_EXCEPTIONS_FOR_TEST(memory_move, invalid_parameters) {
    move_noop_probe_t k;
    ASSERT_TRUE(k.create_kernel());
    ASSERT_EQ(k.getSize(), sizeof(uint32_t));
}

// Valid combinations: probe and reference instructions interleaved

struct move_probe_t : public rv64_jit_test_generator_t {
protected:
    void generate() override {
        rvjit_t m(*this);
        auto &mem = m.memory_move();

        // Float loads
        mem.fload(ft0, t0, SEW::e16, 0);
        flh(ft0, t0, 0);
        mem.fload(ft0, t0, SEW::e32, 0);
        flw(ft0, t0, 0);
        mem.fload(ft0, t0, SEW::e64, 0);
        fld(ft0, t0, 0);

        // Integer loads
        mem.xload(t0, t0, SEW::e8, /*is_signed=*/false, 0);
        lbu(t0, t0, 0);
        mem.xload(t0, t0, SEW::e8, /*is_signed=*/true, 0);
        lb(t0, t0, 0);
        mem.xload(t0, t0, SEW::e32, /*is_signed=*/true, 0);
        lw(t0, t0, 0);
        mem.xload(t0, t0, SEW::e64, /*is_signed=*/true, 0);
        ld(t0, t0, 0);

        // Vector unit-stride loads
        mem.vload(v0, vaddr_t::unit(t0), SEW::e8);
        vle8_v(v0, t0);
        mem.vload(v0, vaddr_t::unit(t0), SEW::e16);
        vle16_v(v0, t0);
        mem.vload(v0, vaddr_t::unit(t0), SEW::e32);
        vle32_v(v0, t0);
        mem.vload(v0, vaddr_t::unit(t0), SEW::e64);
        vle64_v(v0, t0);

        // Vector unit-stride stores
        mem.vstore(v0, vaddr_t::unit(t0), SEW::e8);
        vse8_v(v0, t0);
        mem.vstore(v0, vaddr_t::unit(t0), SEW::e16);
        vse16_v(v0, t0);
        mem.vstore(v0, vaddr_t::unit(t0), SEW::e32);
        vse32_v(v0, t0);
        mem.vstore(v0, vaddr_t::unit(t0), SEW::e64);
        vse64_v(v0, t0);

        // Vector constant stride loads
        mem.vload(v0, vaddr_t::strided(t0, t1), SEW::e8);
        vlse8_v(v0, t0, t1);
        mem.vload(v0, vaddr_t::strided(t0, t1), SEW::e16);
        vlse16_v(v0, t0, t1);
        mem.vload(v0, vaddr_t::strided(t0, t1), SEW::e32);
        vlse32_v(v0, t0, t1);
        mem.vload(v0, vaddr_t::strided(t0, t1), SEW::e64);
        vlse64_v(v0, t0, t1);

        // Vector constant stride stores
        mem.vstore(v0, vaddr_t::strided(t0, t1), SEW::e8);
        vsse8_v(v0, t0, t1);
        mem.vstore(v0, vaddr_t::strided(t0, t1), SEW::e16);
        vsse16_v(v0, t0, t1);
        mem.vstore(v0, vaddr_t::strided(t0, t1), SEW::e32);
        vsse32_v(v0, t0, t1);
        mem.vstore(v0, vaddr_t::strided(t0, t1), SEW::e64);
        vsse64_v(v0, t0, t1);

        ret();
    }
};

HANDLE_EXCEPTIONS_FOR_TEST(memory_move, expected_instructions) {
    move_probe_t k;
    ASSERT_TRUE(k.create_kernel());
    constexpr int n_pairs = 3;
    const auto *code = k.getCode<const uint32_t *>();
    ASSERT_GT(k.getSize(), n_pairs * 2 * sizeof(uint32_t));
    for (int i = 0; i < n_pairs; ++i)
        EXPECT_EQ(code[2 * i], code[2 * i + 1]) << "mismatch at pair " << i;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
