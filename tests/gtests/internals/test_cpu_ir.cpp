/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include <algorithm>
#include <limits>
#include <vector>

#include "gtest/gtest.h"

#include "common/c_types_map.hpp"

#include "cpu/x64/ir/ir.hpp"
#include "cpu/x64/ir/reg_alloc.hpp"

namespace dnnl {

using namespace impl;
using namespace impl::cpu::x64;
using namespace impl::cpu::x64::ir;

// Helpers shared by the tests

// Build a register pool with `n_gpr` GPRs plus all available vector registers.
reg_pools_t make_pools(int n_gpr) {
    reg_pools_t pools {};

    reg_file_t gpr_file {};
    // Spill slot size for GPR in bytes.
    gpr_file.slot_size = 8;
    for (int i = 0; i < n_gpr; i++)
        gpr_file.regs.push_back(i);

    reg_file_t vec_file {};
    // Spill slot size for vector registers in bytes. AVX2 only.
    vec_file.slot_size = 32;
    for (int i = 0; i < 16; i++)
        vec_file.regs.push_back(i);

    pools.files = {gpr_file, vec_file};
    // reg_kind_t order is { gpr, vec, mask }. A mask and vec kinds share the
    // same register file on AVX2.
    pools.kind_to_file = {/* gpr */ 0, /*vec*/ 1, /*mask*/ 1};

    return pools;
}

// A virtual register's (vreg) live interval. Defined as [start, end]
// operation indices in IR.
struct interval_t {
    int start = std::numeric_limits<int>::max();
    int end = -1;
    bool used() const { return end >= 0; }
};

// Reconstruct each vreg's live interval for branch-free IR.
//
// In linear code, a value is considered live from the first time it is
// used until the last time it is used. So, we define its live interval using
// the recorded reads and writes, without running liveness analysis again. This
// allows the allocator tests to check for interference using the same approach
// as the allocator itself.
std::vector<interval_t> linear_code_intervals(const ir_t &ir) {
    std::vector<interval_t> iv(ir.n_vregs());
    std::vector<int> defs, uses;

    for (int i = 0; i < ir.n_ops(); i++) {
        ir.def_use(ir.ops()[i], defs, uses);

        auto update_interval = [&](int v) {
            if (v < 0) return;
            iv[v].start = std::min(iv[v].start, i);
            iv[v].end = std::max(iv[v].end, i);
        };

        for (int v : defs)
            update_interval(v);
        for (int v : uses)
            update_interval(v);
    }
    return iv;
}

// Check the allocator's main rule. Two overlapping vregs targeting the same
// register file must not be assigned the same physical register.
// Spilled values are stored on the stack, so this rule does not apply to them.
void expect_no_reg_conflicts(const ir_t &ir, const reg_pools_t &pools,
        const reg_alloc_result_t &res) {
    const auto iv = linear_code_intervals(ir);
    const int n = ir.n_vregs();

    for (int a = 0; a < n; a++) {
        if (!iv[a].used()) continue;
        for (int b = a + 1; b < n; b++) {
            if (!iv[b].used()) continue;

            const int file_a = pools.kind_to_file[(int)ir.vreg_info()[a].kind];
            const int file_b = pools.kind_to_file[(int)ir.vreg_info()[b].kind];
            if (file_a != file_b) continue;

            const bool overlap
                    = iv[a].start <= iv[b].end && iv[b].start <= iv[a].end;
            if (!overlap) continue;

            const assignment_t &aa = res.assignments[a];
            const assignment_t &ab = res.assignments[b];

            if (!aa.spilled && !ab.spilled) {
                EXPECT_NE(aa.phys, ab.phys)
                        << "vregs " << a << " and " << b
                        << " overlap but share physical register " << aa.phys;
            }
        }
    }
}

// Build an IR where all GPRs in the live set are live at the same time.
// Each register is initialized first, then they are all used together by adding
// them into an accumulator. The live set size controls the register pressure to
// exercise register allocation and spilling.
ir_t build_gpr_live_set(int live_set_size) {
    ir_t ir;
    const int acc = ir.new_gpr();
    ir.mov_imm(acc, 0);

    std::vector<int> live_set(live_set_size);
    for (int i = 0; i < live_set_size; i++) {
        live_set[i] = ir.new_gpr();
        ir.mov_imm(live_set[i], i + 1);
    }
    for (int i = 0; i < live_set_size; i++)
        ir.add_reg(acc, live_set[i]);

    return ir;
}

// Find the index of the first operation with the given opcode. Otherwise
// return -1.
int find_op(const ir_t &ir, op_kind_t op) {
    for (int i = 0; i < ir.n_ops(); i++)
        if (ir.ops()[i].kind == op) return i;
    return -1;
}

// IR builder tests
//
// Checks that the builder records an IR correctly. Operations stay in the right
// order. Each virtual register has the correct kind and data type. Every
// operation has the right inputs and outputs. The fused multiply add is an
// important example. It must treat its accumulator as both read and written or
// the system would incorrectly think a value is no longer needed even though it
// is still in use.
TEST(IRBuilderTests, OperationOrderMetadataAndDefUse) {
    ir_t ir {};
    const int ptr = ir.new_gpr();
    ir.load_param(ptr, 0);

    const int acc = ir.new_vec(data_type::f32);
    ir.vzero(acc);

    const int a = ir.new_vec(data_type::f32);
    ir.vload(a, ptr, 0);

    const int b = ir.new_vec(data_type::f32);
    // AVX2 only.
    ir.vload(b, ptr, simd_w * (dim_t)sizeof(float));

    ir.vfma(acc, a, b);

    // Instructions appear in the exact order they were built.
    ASSERT_EQ(ir.n_ops(), 5);
    EXPECT_EQ(ir.ops()[0].kind, op_kind_t::load);
    EXPECT_EQ(ir.ops()[1].kind, op_kind_t::vzero);
    EXPECT_EQ(ir.ops()[2].kind, op_kind_t::vload);
    EXPECT_EQ(ir.ops()[3].kind, op_kind_t::vload);
    EXPECT_EQ(ir.ops()[4].kind, op_kind_t::vfma);

    // Register kinds and data type are recorded for the allocator and emitter.
    EXPECT_EQ(ir.vreg_info()[ptr].kind, reg_kind_t::gpr);
    EXPECT_EQ(ir.vreg_info()[ptr].dt, data_type::undef);

    EXPECT_EQ(ir.vreg_info()[a].kind, reg_kind_t::vec);
    EXPECT_EQ(ir.vreg_info()[a].dt, data_type::f32);

    EXPECT_EQ(ir.vreg_info()[b].kind, reg_kind_t::vec);
    EXPECT_EQ(ir.vreg_info()[b].dt, data_type::f32);

    EXPECT_EQ(ir.vreg_info()[acc].kind, reg_kind_t::vec);
    EXPECT_EQ(ir.vreg_info()[acc].dt, data_type::f32);

    std::vector<int> defs, uses;

    // vzero writes its destination and reads nothing.
    ir.def_use(ir.ops()[1], defs, uses);
    EXPECT_EQ(defs, std::vector<int>({acc}));
    EXPECT_TRUE(uses.empty());

    // vload reads the base pointer and writes the loaded vector.
    ir.def_use(ir.ops()[2], defs, uses);
    EXPECT_EQ(defs, std::vector<int>({a}));
    EXPECT_EQ(uses, std::vector<int>({ptr}));

    // vfma accumulates in place so the destination is read and written, both
    // sources are read.
    ir.def_use(ir.ops()[4], defs, uses);
    EXPECT_EQ(defs, std::vector<int>({acc}));
    ASSERT_EQ(uses.size(), 3u);
    EXPECT_NE(std::find(uses.begin(), uses.end(), a), uses.end());
    EXPECT_NE(std::find(uses.begin(), uses.end(), b), uses.end());
    EXPECT_NE(std::find(uses.begin(), uses.end(), acc), uses.end());
}

// Validates loop construction. A real loop links its end back to its begin and
// shares one counter register, while a loop that would run only once is inlined
// rather than emitted as a branch that is never taken.
TEST(IRBuilderTests, LoopLinkageAndSingleIterationInlining) {
    {
        ir_t ir {};
        const int acc = ir.new_gpr();
        ir.mov_imm(acc, 0);
        emit_loop_imm(ir, 4, [&]() { ir.add_imm(acc, 1); });

        const int begin = find_op(ir, op_kind_t::loop_begin);
        const int end = find_op(ir, op_kind_t::loop_end);
        ASSERT_NE(begin, -1);
        ASSERT_NE(end, -1);

        // The back-edge is linked and the body sits between the two markers.
        EXPECT_EQ(ir.ops()[end].match, begin);
        EXPECT_LT(begin, end);
        // loop_begin and loop_end operate on the same counter register.
        EXPECT_EQ(ir.ops()[begin].dst, ir.ops()[end].dst);
        // The counter is a general-purpose register created for the loop.
        EXPECT_EQ(ir.vreg_info()[ir.ops()[begin].dst].kind, reg_kind_t::gpr);
    }

    {
        // A single-iteration loop is inlined so no loop markers are emitted.
        ir_t ir {};
        const int acc = ir.new_gpr();
        ir.mov_imm(acc, 0);
        emit_loop_imm(ir, 1, [&]() { ir.add_imm(acc, 1); });

        EXPECT_EQ(find_op(ir, op_kind_t::loop_begin), -1);
        EXPECT_EQ(find_op(ir, op_kind_t::loop_end), -1);
        EXPECT_EQ(ir.n_ops(), 2);
    }
}

// Validates that an if/else is well-formed. Labels are distinct and bound, and
// each jump targets the right label. This proves the forward-edge control flow
// is constructed correctly.
TEST(IRBuilderTests, ForwardEdgeControlFlow) {
    ir_t ir {};

    const int cond = ir.new_gpr();
    ir.load_param(cond, 0);

    const int a = ir.new_vec(data_type::f32);
    const int acc = ir.new_vec(data_type::f32);

    const int base = ir.new_gpr();

    ir.load_param(base, sizeof(int));
    ir.vload(a, base, 0);

    const int lbl_else = ir.new_label();
    const int lbl_end = ir.new_label();

    // Build an IR for the following control flow:
    //
    //     if (cond != 0) { acc += a * a; }  // then block
    //     else           { acc = 0; }       // else block
    //
    // which lowers to a forward-branch skeleton:
    //
    //     jz cond -> else      ; fall through to 'then' when cond != 0
    //   then:
    //     ...work...
    //     jmp -> end           ; skip the else block
    //   else:
    //     ...
    //   end:

    ir.jz(cond, lbl_else);

    // then:
    ir.vzero(acc);
    ir.vfma(acc, a, a); // arbitrary work, just to populate the block
    ir.jmp(lbl_end);

    // else:
    ir.label(lbl_else);
    ir.vzero(acc);

    // end:
    ir.label(lbl_end);

    // The two labels have distinct ids.
    EXPECT_NE(lbl_else, lbl_end);
    EXPECT_EQ(ir.n_labels(), 2);

    // Every branch target points to a label that exists in the IR.
    std::vector<int> bound(ir.n_labels(), -1);
    for (int i = 0; i < ir.n_ops(); i++) {
        if (ir.ops()[i].kind == op_kind_t::label) {
            // Save the position of the label (`i`).
            bound[(int)ir.ops()[i].imm] = i;
        }
    }
    // >= 0 means it's bound.
    EXPECT_GE(bound[lbl_else], -1);
    EXPECT_GE(bound[lbl_end], -1);

    const int jz_idx = find_op(ir, op_kind_t::jz);
    const int jmp_idx = find_op(ir, op_kind_t::jmp);
    ASSERT_NE(jz_idx, -1);
    ASSERT_NE(jmp_idx, -1);

    // The conditional jump targets the else label. The unconditional jump jumps
    // over the else block to the end label.
    EXPECT_EQ(ir.ops()[jz_idx].imm, lbl_else);
    EXPECT_EQ(ir.ops()[jmp_idx].imm, lbl_end);

    // The then block's jmp precedes the else label it jumps over.
    EXPECT_LT(jmp_idx, bound[lbl_else]);

    // The IR that contains branches can be handled by the allocator.
    // It assigns a location to every value that is read somewhere.
    const reg_pools_t pools = make_pools(/*n_gpr=*/8);
    const reg_alloc_result_t res = allocate_registers(ir, pools);
    // All virtual registers have been assigned.
    ASSERT_EQ((int)res.assignments.size(), ir.n_vregs());

    for (int v : {cond, base, a, acc}) {
        const assignment_t &as = res.assignments[v];
        // Each value is either spilled or has a physical register assigned.
        EXPECT_TRUE(as.spilled || as.phys >= 0);
    }
}

// Allocator tests
//
// Checks that there are no register collisions. If all values are live at the
// same time and there are enough registers, each value gets its own register
// and no values are spilled.
TEST(AllocatorTests, DoesNotDoubleAllocateSimultaneouslyLiveValues) {
    const int live_set_size = 6;
    // Create an IR that uses `live_set_size` GPRs for values plus 1 additional
    // GPR for an accumulator.
    const ir_t ir = build_gpr_live_set(live_set_size);

    // Create a pool for the exact number of required GPRs.
    const reg_pools_t pools = make_pools(/*n_gpr=*/live_set_size + 1);

    const reg_alloc_result_t res = allocate_registers(ir, pools);

    EXPECT_FALSE(res.any_spill);
    EXPECT_EQ(res.frame_bytes, 0u);
    expect_no_reg_conflicts(ir, pools, res);
}

// Checks that registers are reused correctly. A temporary that is no longer
// needed gives up its register, which is then used by a later temporary.
// A value that is still in use keeps its own register.
TEST(AllocatorTests, ReusesRegisterOfExpiredTemporary) {
    ir_t ir {};

    const int acc = ir.new_gpr();
    ir.mov_imm(acc, 0);

    const int t0 = ir.new_gpr();
    ir.mov_imm(t0, 5);
    ir.add_reg(acc, t0); // t0 dies here

    const int t1 = ir.new_gpr();
    ir.mov_imm(t1, 7);
    ir.add_reg(acc, t1); // t1 is used only after t0 is dead

    // Two registers are sufficient. One for the accumulator and one reused by
    // t0 then t1.
    const reg_pools_t pools = make_pools(/*n_gpr=*/2);
    const reg_alloc_result_t res = allocate_registers(ir, pools);

    EXPECT_FALSE(res.any_spill);
    EXPECT_FALSE(res.assignments[t0].spilled);
    EXPECT_FALSE(res.assignments[t1].spilled);
    // t0 and t1 reuse the same physical register.
    EXPECT_EQ(res.assignments[t0].phys, res.assignments[t1].phys);
    // The accumulator, live across both, does not collide with them.
    EXPECT_NE(res.assignments[acc].phys, res.assignments[t0].phys);
}

// Checks allocator behavior under register pressure. If the pool doesn't have
// enough registers for all active values, the allocator spills some values to
// the stack. It also reserves frame space, keeps the remaining values from
// overlapping, and assigns each spilled value its own slot offset.
TEST(AllocatorTests, SpillsUnderRegisterPressure) {
    const int live_set_size = 6;
    // Create an IR that uses `live_set_size` GPRs for values plus 1 additional
    // GPR for an accumulator.
    const ir_t ir = build_gpr_live_set(live_set_size);

    // Create pool that has fewer registers than live values.
    const int gpr_pool_size = 4;
    const reg_pools_t pools = make_pools(/*n_gpr=*/gpr_pool_size);

    const reg_alloc_result_t res = allocate_registers(ir, pools);

    // Expect some spills and non-zero stack frame size.
    EXPECT_TRUE(res.any_spill);
    EXPECT_GT(res.frame_bytes, 0u);

    // Check that non-spilled registers do not share the same register.
    expect_no_reg_conflicts(ir, pools, res);

    // Collect spill slots and check that they are unique, slot-aligned, and
    // inside the reserved frame.
    std::vector<size_t> slots;
    for (const auto &as : res.assignments) {
        if (as.spilled) {
            // Check alignment.
            EXPECT_EQ(as.slot % pools.files[0].slot_size, 0u);
            // The slot is within the frame.
            EXPECT_LT(as.slot, res.frame_bytes);
            slots.push_back(as.slot);
        }
    }

    ASSERT_FALSE(slots.empty());

    // Check that each spilled register has a unique slot.
    std::sort(slots.begin(), slots.end());
    EXPECT_EQ(std::unique(slots.begin(), slots.end()), slots.end())
            << "two spilled values were given the same stack slot";
}

// Checks that allocation depends only on its inputs. The same IR and register
// pool produces identical register assignments, spill decisions, and stack
// size, which makes the emitted code reproducible.
TEST(AllocatorTests, AllocatesDeterministically) {
    const ir_t ir = build_gpr_live_set(6);
    // Create smaller GPR register pool to put pressure.
    const reg_pools_t pools = make_pools(4);

    const reg_alloc_result_t a = allocate_registers(ir, pools);
    const reg_alloc_result_t b = allocate_registers(ir, pools);

    ASSERT_EQ(a.assignments.size(), b.assignments.size());
    EXPECT_EQ(a.frame_bytes, b.frame_bytes);
    EXPECT_EQ(a.any_spill, b.any_spill);

    for (size_t v = 0; v < a.assignments.size(); v++) {
        EXPECT_EQ(a.assignments[v].spilled, b.assignments[v].spilled);
        EXPECT_EQ(a.assignments[v].phys, b.assignments[v].phys);
        EXPECT_EQ(a.assignments[v].slot, b.assignments[v].slot);
    }
}

} // namespace dnnl
