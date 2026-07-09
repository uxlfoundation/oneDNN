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

#include "cpu/x64/ir/emitter/emitter.hpp"
#include "cpu/x64/ir/ir.hpp"
#include "cpu/x64/ir/reg_alloc.hpp"
#include "cpu/x64/ir/reg_config.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {

using namespace impl;
using namespace impl::cpu::x64;
using namespace impl::cpu::x64::ir;

// Tests that require generating a kernel require AVX2.
#define SKIP_IF_NO_AVX2() \
    do { \
        if (!mayiuse(avx2)) GTEST_SKIP() << "IR emitter require AVX2"; \
    } while (0)

// Helpers shared by the tests

constexpr int simd_w = 8;

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

// Reference dot product over `n` f32 elements.
float ref_dot(const float *a, const float *b, int n) {
    float acc = 0.f;
    for (int i = 0; i < n; i++)
        acc += a[i] * b[i];
    return acc;
}

// IR-based kernel.
class ir_kernel_t : public impl::cpu::x64::jit_generator_t {
public:
    ir_kernel_t(ir_t ir, int vec_regs_limit = -1)
        // Only AVX2 is currently supported.
        : jit_generator_t("ir_run_kernel", cpu::x64::avx2)
        , ir_(std::move(ir))
        , vec_regs_limit_(vec_regs_limit) {}

    const char *name() const override { return "ir_kernel"; }
    const char *source_file() const override { return __FILE__; }

    // Build and finalize the code. Return false on any Xbyak error.
    bool run_ir_pipeline() {
        generate();
        this->ready();
        return Xbyak::GetError() == Xbyak::ERR_NONE;
    }

    // Get generated code.
    const uint8_t *code_ptr() { return this->Xbyak::CodeGenerator::getCode(); }
    // Get generated code size.
    size_t code_size() { return this->Xbyak::CodeGenerator::getSize(); }

    // Calls the generated kernel.
    template <typename arg_t>
    void run(const arg_t *args) {
        auto fn = reinterpret_cast<void (*)(const arg_t *)>(
                const_cast<uint8_t *>(code_ptr()));
        fn(args);
    }

    // Allocation outcome. Becomes valid after `run_ir_pipeline()`.
    //
    // True if the generated kernel has any spills.
    bool spilled() const { return spilled_; }
    // The stack size required by the allocator for spilling.
    size_t stack_size() const { return stack_size_; }

protected:
    void generate() override {
        const int rsp_idx = Xbyak::Operand::RSP;
        const int param_idx = abi_param1.getIdx();

        // Scratch registers the emitter reserves for spill handling. They are
        // not part of the register pool. The indices of the registers are
        // irrelevant. Should work fine for AVX2 and AVX-512.
        const int gpr_scratch0 = 10, gpr_scratch1 = 11;
        const int vec_scratch0 = 13, vec_scratch1 = 14, vec_scratch2 = 15;

        reg_config_t reg_cfg = make_reg_config(avx2, param_idx, rsp_idx,
                {gpr_scratch0, gpr_scratch1},
                {vec_scratch0, vec_scratch1, vec_scratch2});

        // Shrink the vector register pool to force spills when requested.
        if (vec_regs_limit_ >= 0) {
            const int vec_reg_pool_idx = 1;
            auto &vec_regs = reg_cfg.pools.files[vec_reg_pool_idx].regs;
            if ((int)vec_regs.size() > vec_regs_limit_)
                vec_regs.resize(vec_regs_limit_);
        }

        // Run allocator.
        const reg_alloc_result_t alloc = allocate_registers(ir_, reg_cfg.pools);
        spilled_ = alloc.any_spill;
        stack_size_ = alloc.frame_bytes;

        preamble();

        const int frame = (int)utils::rnd_up(alloc.frame_bytes, 16);
        if (frame > 0) sub(rsp, frame);

        data_section_t data;
        emit(*this, ir_, alloc, reg_cfg, data);

        if (frame > 0) add(rsp, frame);

        postamble();

        emit_data_section(*this, data);
    }

private:
    ir_t ir_ {};
    int vec_regs_limit_ = 0;
    bool spilled_ = false;
    size_t stack_size_ = 0;
};

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
            bound[ir.ops()[i].label_id] = i;
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
    EXPECT_EQ(ir.ops()[jz_idx].label_id, lbl_else);
    EXPECT_EQ(ir.ops()[jmp_idx].label_id, lbl_end);

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

// Emitter tests
//
// Builds a small reduction IR (dot product of one vector pair) used by the
// emitter and integration tests.
ir_t build_dot_ir() {
    ir_t ir {};

    // Load pointers for a, b, c.
    const int a_ptr = ir.new_gpr();
    ir.load_param(a_ptr, 0);

    const int b_ptr = ir.new_gpr();
    ir.load_param(b_ptr, sizeof(void *));

    const int c_ptr = ir.new_gpr();
    ir.load_param(c_ptr, 2 * sizeof(void *));

    const int acc = ir.new_vec(data_type::f32);
    ir.vzero(acc);

    const int a = ir.new_vec(data_type::f32);
    ir.vload(a, a_ptr, 0);

    const int b = ir.new_vec(data_type::f32);
    ir.vload(b, b_ptr, 0);

    ir.vfma(acc, a, b);

    const int ws = ir.new_vec(data_type::f32);
    ir.vhreduce(acc, ws);

    // store the reduced scalar
    ir.vstore_masked(c_ptr, 0, acc, -1, 1);

    return ir;
}

// Validates emitter determinism. Emitting the same program twice produces
// byte-for-byte identical machine code. The encoding is position independent
// (relative branches, rip-relative constants), so equal bytes is a valid check.
TEST(EmitterTests, EmitsDeterministicCodeForIdenticalIr) {
    SKIP_IF_NO_AVX2();
    ir_kernel_t k1(build_dot_ir());
    ir_kernel_t k2(build_dot_ir());

    ASSERT_TRUE(k1.run_ir_pipeline());
    ASSERT_TRUE(k2.run_ir_pipeline());

    ASSERT_GT(k1.code_size(), 0u);
    ASSERT_EQ(k1.code_size(), k2.code_size());
    EXPECT_EQ(0, std::memcmp(k1.code_ptr(), k2.code_ptr(), k1.code_size()));
}

// Validates the emitter's spill path. When the allocation spills, the
// reload-compute-store sequence must lower with no assembler error, and a stack
// frame must be reserved for the spilled values.
TEST(EmitterTests, EmitsValidCodeForSpilledAllocation) {
    SKIP_IF_NO_AVX2();

    // Six independent accumulators plus temporaries exceed a four-register
    // vector file, so the allocator must spill.
    ir_t ir {};

    const int a_ptr = ir.new_gpr();
    ir.load_param(a_ptr, 0);

    const int b_ptr = ir.new_gpr();
    ir.load_param(b_ptr, sizeof(void *));

    const int b = ir.new_vec(data_type::f32);
    ir.vload(b, b_ptr, 0);

    std::vector<int> acc(6);
    for (int r = 0; r < 6; r++) {
        acc[r] = ir.new_vec(data_type::f32);
        ir.vzero(acc[r]);
    }

    for (int r = 0; r < 6; r++) {
        const int a = ir.new_vec(data_type::f32);
        ir.vload(a, a_ptr, r * simd_w * (dim_t)sizeof(float));
        ir.vfma(acc[r], a, b);
    }

    ir_kernel_t k(ir, /*vec_regs_limit=*/4);
    ASSERT_TRUE(k.run_ir_pipeline());
    EXPECT_TRUE(k.spilled());
    EXPECT_GT(k.stack_size(), 0u);
    EXPECT_GT(k.code_size(), 0u);
}

// Integration tests

// Arguments for the dot-product kernels.
struct dot_args_t {
    const float *a;
    const float *b;
    float *c;
};

// Pipeline test. A dot product over sixteen elements, expressed as a
// two-iteration loop, is built, allocated, emitted, run, and checked against
// a reference. Passing it means the whole pipeline computes the right number,
// including loop control flow and values kept live across the back-edge.
TEST(IntegrationTests, BuildsLoopReduction) {
    SKIP_IF_NO_AVX2();

    constexpr int k_blocks = 2;
    constexpr int k = k_blocks * simd_w; // 16 elements

    ir_t ir {};

    const int a_ptr = ir.new_gpr();
    ir.load_param(a_ptr, offsetof(dot_args_t, a));

    const int b_ptr = ir.new_gpr();
    ir.load_param(b_ptr, offsetof(dot_args_t, b));

    const int c_ptr = ir.new_gpr();
    ir.load_param(c_ptr, offsetof(dot_args_t, c));

    const int acc = ir.new_vec(data_type::f32);
    ir.vzero(acc);

    // Reduce one simd_w-wide chunk per iteration and advance the pointers.
    emit_loop_imm(ir, k_blocks, [&]() {
        const int a = ir.new_vec(data_type::f32);
        ir.vload(a, a_ptr, 0);
        const int b = ir.new_vec(data_type::f32);
        ir.vload(b, b_ptr, 0);
        ir.vfma(acc, a, b);
    }, [&]() {
        ir.add_imm(a_ptr, simd_w * (dim_t)sizeof(float));
        ir.add_imm(b_ptr, simd_w * (dim_t)sizeof(float));
    });

    const int ws = ir.new_vec(data_type::f32);
    ir.vhreduce(acc, ws);
    ir.vstore_masked(c_ptr, 0, acc, -1, 1);

    ir_kernel_t kernel(ir);
    ASSERT_TRUE(kernel.run_ir_pipeline());

    std::vector<float> a(k), b(k);
    for (int i = 0; i < k; i++) {
        a[i] = (float)(i + 1);
        b[i] = (float)(2 * i - 3);
    }

    float c = -12345.f;
    dot_args_t args {a.data(), b.data(), &c};
    kernel.run(&args);

    EXPECT_FLOAT_EQ(c, ref_dot(a.data(), b.data(), k));
}

// Computes a dot product where one vector is multiplied by n vectors into
// n independent accumulators, each of which is reduced and stored.
ir_t build_shared_vector_dot_ir(int n) {
    ir_t ir {};

    const int a_ptr = ir.new_gpr();
    ir.load_param(a_ptr, offsetof(dot_args_t, a));

    const int b_ptr = ir.new_gpr();
    ir.load_param(b_ptr, offsetof(dot_args_t, b));

    const int c_ptr = ir.new_gpr();
    ir.load_param(c_ptr, offsetof(dot_args_t, c));

    const int b = ir.new_vec(data_type::f32);
    // Load shared vector.
    ir.vload(b, b_ptr, 0);

    std::vector<int> acc(n);
    for (int r = 0; r < n; r++) {
        acc[r] = ir.new_vec(data_type::f32);
        ir.vzero(acc[r]);

        const int a = ir.new_vec(data_type::f32);
        ir.vload(a, a_ptr, r * simd_w * (dim_t)sizeof(float));

        ir.vfma(acc[r], a, b);
    }

    const int ws = ir.new_vec(data_type::f32);
    for (int r = 0; r < n; r++)
        ir.vhreduce(acc[r], ws);

    for (int r = 0; r < n; r++)
        ir.vstore_masked(c_ptr, r * (dim_t)sizeof(float), acc[r], -1, 1);

    return ir;
}

// Validates that register allocator decisions never change results. The same
// computation is run with a full register file and with one too small to avoid
// spills. Both must produce identical results.
TEST(IntegrationTests, SpillProducesEquivalentResults) {
    SKIP_IF_NO_AVX2();

    constexpr int n = 6;

    ir_kernel_t full(build_shared_vector_dot_ir(n));
    ir_kernel_t limited(build_shared_vector_dot_ir(n), /*vec_regs_cap=*/4);

    ASSERT_TRUE(full.run_ir_pipeline());
    ASSERT_TRUE(limited.run_ir_pipeline());

    // The full file fits everything. The limited file must spill.
    EXPECT_FALSE(full.spilled());
    EXPECT_TRUE(limited.spilled());

    std::vector<float> a(n * simd_w), b(simd_w);
    for (int i = 0; i < n * simd_w; i++)
        a[i] = (float)(i % 7) - 3.f;

    for (int i = 0; i < simd_w; i++)
        b[i] = (float)(i - 2);

    std::vector<float> c_full(n, 0.f), c_limited(n, 0.f);

    dot_args_t args_full {a.data(), b.data(), c_full.data()};
    dot_args_t args_limited {a.data(), b.data(), c_limited.data()};

    full.run(&args_full);
    limited.run(&args_limited);

    for (int r = 0; r < n; r++) {
        const float expected = ref_dot(&a[r * simd_w], b.data(), simd_w);
        EXPECT_FLOAT_EQ(c_full[r], expected) << "row " << r;
        EXPECT_FLOAT_EQ(c_limited[r], expected) << "row " << r;
    }
}

// Arguments for the branch-selection kernel. A runtime flag plus two candidate
// input vectors and one output vector.
struct select_args_t {
    int64_t cond;
    const float *a;
    const float *b;
    float *c;
};

// Validates forward branches end to end. A runtime flag selects which of two
// inputs to store, and each flag value produces the expected output. This check
// that emitted control flow works and both candidates stayed live across the
// branching.
TEST(IntegrationTests, BranchSelectsCorrectValue) {
    SKIP_IF_NO_AVX2();

    ir_t ir {};

    const int cond = ir.new_gpr();
    ir.load_param(cond, offsetof(select_args_t, cond));

    const int a_ptr = ir.new_gpr();
    ir.load_param(a_ptr, offsetof(select_args_t, a));

    const int b_ptr = ir.new_gpr();
    ir.load_param(b_ptr, offsetof(select_args_t, b));

    const int c_ptr = ir.new_gpr();
    ir.load_param(c_ptr, offsetof(select_args_t, c));

    const int a = ir.new_vec(data_type::f32);
    ir.vload(a, a_ptr, 0);

    const int b = ir.new_vec(data_type::f32);
    ir.vload(b, b_ptr, 0);

    const int lbl_else = ir.new_label();
    const int lbl_end = ir.new_label();

    //     if (cond != 0) { c = a; }  // then block
    //     else           { c = b; }  // else block
    //
    // which lowers to a forward-branch skeleton:
    //
    //     jz cond -> else      ; fall through to 'then' when cond != 0
    //   then:
    //     store a -> c
    //     jmp -> end           ; skip the else block
    //   else:
    //     store b -> c
    //   end:
    ir.jz(cond, lbl_else);
    ir.vstore_masked(c_ptr, 0, a, -1, simd_w); // then: c = a
    ir.jmp(lbl_end);
    ir.label(lbl_else);
    ir.vstore_masked(c_ptr, 0, b, -1, simd_w); // else: c = b
    ir.label(lbl_end);

    ir_kernel_t kernel(ir);
    ASSERT_TRUE(kernel.run_ir_pipeline());

    std::vector<float> a_data(simd_w), b_data(simd_w), c_data(simd_w, 0.f);
    for (int i = 0; i < simd_w; i++) {
        a_data[i] = (float)(10 + i);
        b_data[i] = (float)(100 + i);
    }

    // cond != 0 selects a.
    select_args_t args_a {1, a_data.data(), b_data.data(), c_data.data()};
    kernel.run(&args_a);
    for (int i = 0; i < simd_w; i++)
        EXPECT_FLOAT_EQ(c_data[i], a_data[i]) << "lane " << i;

    // cond == 0 selects b.
    std::fill(c_data.begin(), c_data.end(), 0.f);
    select_args_t args_b {0, a_data.data(), b_data.data(), c_data.data()};
    kernel.run(&args_b);
    for (int i = 0; i < simd_w; i++)
        EXPECT_FLOAT_EQ(c_data[i], b_data[i]) << "lane " << i;
}

} // namespace dnnl
