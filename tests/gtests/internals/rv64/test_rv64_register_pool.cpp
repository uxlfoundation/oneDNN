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
#include <utility>
#include <vector>

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

// Allocation advances the cursor by the right amount

TEST(register_pool_partition, single_alloc) {
    register_pool_t::x_partition_t p({a0, a1, a2, a3});
    ASSERT_EQ(p.allocated(), 0);
    p.allocate();
    ASSERT_EQ(p.allocated(), 1);
}

TEST(register_pool_partition, panel_alloc) {
    register_pool_t::x_partition_t p({a0, a1, a2, a3});
    p.allocate(3);
    ASSERT_EQ(p.allocated(), 3);
}

TEST(register_pool_partition, block_alloc) {
    register_pool_t::x_partition_t p({a0, a1, a2, a3, a5, a6, a7, s0, s1, s2});
    p.allocate(2, 3);
    ASSERT_EQ(p.allocated(), 6);
}

TEST(register_pool_partition, mixed_alloc) {
    register_pool_t::x_partition_t p({a0, a1, a2, a3});
    p.allocate(); // 1
    p.allocate(2); // +2 = 3
    ASSERT_EQ(p.allocated(), 3);
}

// partition_t: failed allocations leave the cursor and block intact

TEST(register_pool_partition, failed_single_alloc) {
    register_pool_t::x_partition_t p({a0});
    p.allocate(); // consume the only register
    p.allocate(); // should fail silently
    ASSERT_EQ(p.allocated(), 1);
}

TEST(register_pool_partition, failed_multi_alloc) {
    register_pool_t::x_partition_t p({a0, a1});
    const auto blk = p.allocate(1, 3); // needs 3, only 2 available
    ASSERT_EQ(blk.size(), 0);
    ASSERT_EQ(p.allocated(), 0);
}

// preserve()/restore() emits the right sd/ld count and offsets

namespace {

// Emits only int_register_file_excluding + new_int(n_alloc) +
// preserve() + restore() + ret(), so the sd/ld pairs are unambiguous.
struct preserve_restore_probe_t : public rv64_jit_test_generator_t {
    preserve_restore_probe_t(std::vector<Reg> excl, int n_alloc)
        : excl_(std::move(excl)), n_alloc_(n_alloc) {}

protected:
    void generate() override {
        rvjit_t m(*this);
        auto &pool = m.register_pool();
        pool.int_register_file_excluding(excl_.begin(), excl_.end());
        pool.new_int(n_alloc_);
        pool.preserve();
        pool.restore();
        ret();
    }

private:
    std::vector<Reg> excl_;
    int n_alloc_;
};

// RV64 S-type (SD) with rs1 == sp (x2): bits[6:0]=0x23, bits[14:12]=3
static bool is_sd_sp(uint32_t instr) {
    return (instr & 0x707fu) == 0x3023u && ((instr >> 15) & 0x1fu) == 2u;
}

// RV64 I-type (LD) with rs1 == sp (x2): bits[6:0]=0x03, bits[14:12]=3
static bool is_ld_sp(uint32_t instr) {
    return (instr & 0x707fu) == 0x3003u && ((instr >> 15) & 0x1fu) == 2u;
}

// S-type immediate: imm[4:0] = bits[11:7], imm[11:5] = bits[31:25]
static int sd_imm(uint32_t instr) {
    return static_cast<int>(((instr >> 7) & 0x1fu) | ((instr >> 25) << 5));
}

// S-type rs2 (data register): bits[24:20]
static int sd_rs2(uint32_t instr) {
    return static_cast<int>((instr >> 20) & 0x1fu);
}

// I-type immediate: bits[31:20]
static int ld_imm(uint32_t instr) {
    return static_cast<int>((instr >> 20) & 0xfffu);
}

// I-type rd (destination register): bits[11:7]
static int ld_rd(uint32_t instr) {
    return static_cast<int>((instr >> 7) & 0x1fu);
}

} // namespace

// Excluding a0..a7 leaves the pool ordered t0..t6, s0..s11.
// Allocating 10 GPRs yields 7 caller-saved (t0..t6) and 3 callee-saved
// (s0..s2).  preserve() must emit exactly 3 sd-with-sp instructions, and
// restore() must emit exactly 3 ld-with-sp instructions.  Each sd/ld pair
// for the same slot must reference the same register index and the same
// stack offset.
TEST(register_pool_jit, preserve_restore_pairs) {
    const int n_alloc = 10;
    const int n_callee_saved = 3; // n_alloc - 7 caller-saved (t0..t6)

    preserve_restore_probe_t k({a0, a1, a2, a3, a4, a5, a6, a7}, n_alloc);
    ASSERT_TRUE(k.create_kernel());

    const auto *code = k.getCode<const uint32_t *>();
    const int n_insn = static_cast<int>(k.getSize() / sizeof(uint32_t));

    std::vector<std::pair<int, int>> stores; // (reg_idx, stack_offset)
    std::vector<std::pair<int, int>> loads;
    for (int i = 0; i < n_insn; ++i) {
        if (is_sd_sp(code[i]))
            stores.push_back({sd_rs2(code[i]), sd_imm(code[i])});
        if (is_ld_sp(code[i]))
            loads.push_back({ld_rd(code[i]), ld_imm(code[i])});
    }

    ASSERT_EQ(static_cast<int>(stores.size()), n_callee_saved);
    ASSERT_EQ(static_cast<int>(loads.size()), n_callee_saved);

    for (int i = 0; i < n_callee_saved; ++i) {
        EXPECT_EQ(stores[i].first, loads[i].first)
                << "register index mismatch at save slot " << i;
        EXPECT_EQ(stores[i].second, loads[i].second)
                << "stack offset mismatch at save slot " << i;
    }
}

// Allocating only caller-saved registers (≤ 7 after excluding a0..a7) must
// leave preserve()/restore() as no-ops: no sd or ld with sp emitted.
TEST(register_pool_jit, preserve_noop_caller_saved_only) {
    const int n_alloc = 7; // exactly t0..t6, all caller-saved

    preserve_restore_probe_t k({a0, a1, a2, a3, a4, a5, a6, a7}, n_alloc);
    ASSERT_TRUE(k.create_kernel());

    const auto *code = k.getCode<const uint32_t *>();
    const int n_insn = static_cast<int>(k.getSize() / sizeof(uint32_t));

    int n_stores = 0, n_loads = 0;
    for (int i = 0; i < n_insn; ++i) {
        if (is_sd_sp(code[i])) ++n_stores;
        if (is_ld_sp(code[i])) ++n_loads;
    }

    EXPECT_EQ(n_stores, 0);
    EXPECT_EQ(n_loads, 0);
}

// vector ctx invalidation: allocation results are captured inside generate

namespace {

struct acc_blocks_inp_probe_t : public rv64_jit_test_generator_t {
    acc_blocks_inp_probe_t() {}
    int acc0_idx = -1;
    int inp0_idx = -1;

protected:
    void generate() override {
        rvjit_t m(*this);
        auto &pool = m.register_pool();
        pool.vector_register_file(LMUL::m4, LMUL::m8);
        // v0 at m8 occupies physical regs 0-7: blocks m4 groups v0 and v4
        acc0_idx = pool.new_vector_accumulator().getIdx();
        inp0_idx = pool.new_vector().getIdx();
        ret();
    }
};

struct inp_blocks_acc_probe_t : public rv64_jit_test_generator_t {
    inp_blocks_acc_probe_t() {}
    int inp0_idx = -1;
    int acc1_idx = -1;

protected:
    void generate() override {
        rvjit_t m(*this);
        auto &pool = m.register_pool();
        pool.vector_register_file(LMUL::m4, LMUL::m8);
        pool.new_vector_accumulator(); // v0 — consume first acc slot
        // v8 at m4 overlaps v8 at m8
        inp0_idx = pool.new_vector().getIdx();
        acc1_idx = pool.new_vector_accumulator().getIdx();
        ret();
    }
};

struct single_pool_probe_t : public rv64_jit_test_generator_t {
    single_pool_probe_t() {}
    static constexpr int N = 4;
    int acc_idx[N] = {};
    int inp_idx[N] = {};

protected:
    void generate() override {
        // Two independent rvjit_t (and thus two separate pools) sharing the
        // same code generator so we can compare their allocation sequences.
        rvjit_t ma(*this), mb(*this);
        auto &pool_a = ma.register_pool();
        auto &pool_b = mb.register_pool();
        pool_a.vector_register_file(LMUL::m4);
        pool_b.vector_register_file(LMUL::m4);
        const v_block_t acc = pool_a.new_vector_accumulator(N);
        const v_block_t inp = pool_b.new_vector(N);
        for (int i = 0; i < N; ++i) {
            acc_idx[i] = acc(i).getIdx();
            inp_idx[i] = inp(i).getIdx();
        }
        ret();
    }
};

} // namespace

// Allocating an m8 accumulator must block the two m4 input candidates that
// fall within its physical range (v0 m8 → v0+v4 m4 both unavailable).
HANDLE_EXCEPTIONS_FOR_TEST(register_pool_vector_split, acc_blocks_inp) {
    acc_blocks_inp_probe_t k;
    ASSERT_TRUE(k.create_kernel());
    EXPECT_EQ(k.acc0_idx, 0); // v0 — first m8 candidate
    EXPECT_EQ(k.inp0_idx, 8); // v8 — v0 and v4 blocked by v0 m8
}

// Allocating an m4 input must block the m8 accumulator that overlaps it
// (inp v8 m4 → bits 8-11 used → acc v8 m8 needs bits 8-15, blocked).
HANDLE_EXCEPTIONS_FOR_TEST(register_pool_vector_split, inp_blocks_acc) {
    inp_blocks_acc_probe_t k;
    ASSERT_TRUE(k.create_kernel());
    EXPECT_EQ(k.inp0_idx, 8); // v8 — first free m4 after acc v0 was taken
    EXPECT_EQ(k.acc1_idx, 16); // v16 — v8 acc blocked by inp v8
}

// In single-pool mode, new_vector_accumulator falls back to the inp pool
// and produces the same register sequence as new_vector would.
HANDLE_EXCEPTIONS_FOR_TEST(register_pool_vector_split, single_pool_fallback) {
    single_pool_probe_t k;
    ASSERT_TRUE(k.create_kernel());
    for (int i = 0; i < single_pool_probe_t::N; ++i)
        EXPECT_EQ(k.acc_idx[i], k.inp_idx[i]) << "mismatch at i=" << i;
}

// new_loop: allocation count and struct properties

namespace {

struct eager_unrolled_loop_probe_t : public rv64_jit_test_generator_t {
    eager_unrolled_loop_probe_t() {}
    int alloc_delta = -1;
    loop_t result;

protected:
    void generate() override {
        rvjit_t m(*this);
        auto &pool = m.register_pool();

        pool.int_register_file();

        // Eager plans require a caller-supplied limit register
        const Reg limit = pool.new_int();
        const int before = pool.x_available();

        result = loop_t::unroll(4).limit(limit);
        pool.new_loop(result);
        alloc_delta = before - pool.x_available();

        ret();
    }
};

struct lazy_unrolled_loop_probe_t : public rv64_jit_test_generator_t {
    lazy_unrolled_loop_probe_t() {}
    int alloc_delta = -1;
    loop_t result;

protected:
    void generate() override {
        rvjit_t m(*this);
        auto &pool = m.register_pool();

        pool.int_register_file();

        const int before = pool.x_available();

        result = loop_t::unroll(4).limit([](const Reg &) {});
        pool.new_loop(result);

        alloc_delta = before - pool.x_available();

        ret();
    }
};

// Lazy unrolled loop probe with a non-power-of-two unroll factor.
// This test stresses the const_folding_t::round_down() requirement that
// registers cannot be aliased when the constant is an immediate.
struct lazy_unrolled_loop_nonpow2_probe_t : public rv64_jit_test_generator_t {
    lazy_unrolled_loop_nonpow2_probe_t() {}
    int alloc_delta = -1;
    loop_t result;

protected:
    void generate() override {
        rvjit_t m(*this);
        auto &pool = m.register_pool();

        pool.int_register_file();

        const int before = pool.x_available();

        result = loop_t::unroll(6).limit([](const Reg &) {});
        pool.new_loop(result);

        alloc_delta = before - pool.x_available();

        ret();
    }
};

} // namespace

// Eager mode allocates iter and a separate tmp for the user-supplied limit
HANDLE_EXCEPTIONS_FOR_TEST(register_pool_jit, new_loop_eager) {
    eager_unrolled_loop_probe_t k;
    ASSERT_TRUE(k.create_kernel());
    EXPECT_EQ(k.alloc_delta, 2);
}

// Eager mode iter, limit and tmp must all be distinct registers
HANDLE_EXCEPTIONS_FOR_TEST(register_pool_jit, new_loop_eager_registers) {
    eager_unrolled_loop_probe_t k;
    ASSERT_TRUE(k.create_kernel());
    const auto &r = k.result;
    EXPECT_TRUE(r.is_eager() && r.is_valid());
    EXPECT_NE(r.iter().getIdx(), r.limit().getIdx());
    EXPECT_NE(r.limit().getIdx(), r.tmp().getIdx());
    EXPECT_NE(r.iter().getIdx(), r.tmp().getIdx());
}

// Lazy mode reuses the limit register as tmp
HANDLE_EXCEPTIONS_FOR_TEST(register_pool_jit, new_loop_lazy) {
    lazy_unrolled_loop_probe_t k;
    ASSERT_TRUE(k.create_kernel());
    EXPECT_EQ(k.alloc_delta, 2);
}

// Lazy mode tmp aliases limit
HANDLE_EXCEPTIONS_FOR_TEST(register_pool_jit, new_loop_lazy_alias) {
    lazy_unrolled_loop_probe_t k;
    ASSERT_TRUE(k.create_kernel());
    const auto &r = k.result;
    EXPECT_TRUE(r.is_lazy() && r.is_valid());
    EXPECT_EQ(r.tmp().getIdx(), r.limit().getIdx());
    EXPECT_NE(r.iter().getIdx(), r.limit().getIdx());
}

// Lazy mode with a non-power-of-two unroll factor allocates a distinct tmp
HANDLE_EXCEPTIONS_FOR_TEST(register_pool_jit, new_loop_lazy_nonpow2_regs) {
    lazy_unrolled_loop_nonpow2_probe_t k;
    ASSERT_TRUE(k.create_kernel());
    EXPECT_EQ(k.alloc_delta, 3);
}

// Lazy mode with a non-power-of-two unroll factor allocate distinct registers
HANDLE_EXCEPTIONS_FOR_TEST(register_pool_jit, new_loop_lazy_nonpow2_names) {
    lazy_unrolled_loop_nonpow2_probe_t k;
    ASSERT_TRUE(k.create_kernel());
    const auto &r = k.result;
    EXPECT_TRUE(r.is_lazy() && r.is_valid());
    EXPECT_NE(r.tmp().getIdx(), r.limit().getIdx());
    EXPECT_NE(r.iter().getIdx(), r.limit().getIdx());
    EXPECT_NE(r.iter().getIdx(), r.tmp().getIdx());
}

// new_const / new_runtime_value

namespace {

struct new_const_and_runtime_value_probe_t : public rv64_jit_test_generator_t {
    new_const_and_runtime_value_probe_t() {}

    int const_small_delta = -1;
    bool const_small_is_imm = false;
    int const_large_delta = -1;
    bool const_large_has_reg = false;

    int rv_eager_delta = -1;
    bool rv_eager_reg_same = false;

    int rv_lazy_delta = -1;
    bool rv_lazy_reg_set = false;
    bool rv_lazy_get_preserved = false;

protected:
    void generate() override {
        rvjit_t m(*this);
        auto &pool = m.register_pool();
        pool.int_register_file();

        {
            const int before = pool.x_available();
            const const_t c = pool.new_const(42);
            const_small_delta = before - pool.x_available();
            const_small_is_imm = c.is_imm();
        }
        {
            const int before = pool.x_available();
            const const_t c = pool.new_const(1 << 20);
            const_large_delta = before - pool.x_available();
            const_large_has_reg = c.has_reg();
        }
        {
            const Reg r = pool.new_int();
            const int before = pool.x_available();
            const runtime_value_t rv
                    = pool.new_runtime_value(runtime_value_t(r));
            rv_eager_delta = before - pool.x_available();
            rv_eager_reg_same = rv.reg == r;
        }
        {
            const int before = pool.x_available();
            const runtime_value_t rv = pool.new_runtime_value(
                    runtime_value_t([](const Reg &) {}));
            rv_lazy_delta = before - pool.x_available();
            rv_lazy_reg_set = is_not_zero(rv.reg);
            rv_lazy_get_preserved = rv.is_lazy();
        }

        ret();
    }
};

} // namespace

// A value that fits a 12-bit immediate takes no allocation
HANDLE_EXCEPTIONS_FOR_TEST(register_pool_jit, new_const_immediate) {
    new_const_and_runtime_value_probe_t k;
    ASSERT_TRUE(k.create_kernel());
    EXPECT_EQ(k.const_small_delta, 0);
    EXPECT_TRUE(k.const_small_is_imm);
}

// A value that doesn't fit a 12-bit immediate allocates one register
HANDLE_EXCEPTIONS_FOR_TEST(register_pool_jit, new_const_overflow) {
    new_const_and_runtime_value_probe_t k;
    ASSERT_TRUE(k.create_kernel());
    EXPECT_EQ(k.const_large_delta, 1);
    EXPECT_TRUE(k.const_large_has_reg);
}

// An already-materialized runtime_value_t passes through unchanged
HANDLE_EXCEPTIONS_FOR_TEST(
        register_pool_jit, new_runtime_value_eager_passthrough) {
    new_const_and_runtime_value_probe_t k;
    ASSERT_TRUE(k.create_kernel());
    EXPECT_EQ(k.rv_eager_delta, 0);
    EXPECT_TRUE(k.rv_eager_reg_same);
}

// A lazy runtime_value_t with reg == 0 gets a register, keeping its loader
HANDLE_EXCEPTIONS_FOR_TEST(
        register_pool_jit, new_runtime_value_lazy_materializes) {
    new_const_and_runtime_value_probe_t k;
    ASSERT_TRUE(k.create_kernel());
    EXPECT_EQ(k.rv_lazy_delta, 1);
    EXPECT_TRUE(k.rv_lazy_reg_set);
    EXPECT_TRUE(k.rv_lazy_get_preserved);
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
