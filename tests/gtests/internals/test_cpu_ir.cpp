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
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

#include "common/c_types_map.hpp"

#include "cpu/x64/ir/eltwise_injector.hpp"
#include "cpu/x64/ir/emitter/emitter.hpp"
#include "cpu/x64/ir/ir.hpp"
#include "cpu/x64/ir/postops_injector.hpp"
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
    const vreg_t acc = ir.new_gpr();
    ir.mov_imm(acc, 0);

    std::vector<vreg_t> live_set(live_set_size, vreg_t::none);
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

    // Optional post-ops injector setup. When `post_ops` is set, generate()
    // creates the JIT post-ops injector and drives the `inject_postops` op
    // through it, the same way a real IR kernel does.
    //   rhs_arg_offset  - byte offset in the argument struct of the binary
    //                     right-hand-side pointer array
    //   dst_orig_offset - byte offset in the argument struct of the
    //                     destination origin pointer
    //   tail_elems      - right-hand-side elements a partial (tail) load reads
    struct postops_cfg_t {
        const impl::post_ops_t *post_ops = nullptr;
        const impl::memory_desc_t *dst_md = nullptr;
        size_t rhs_arg_offset = 0;
        size_t dst_orig_offset = 0;
        int tail_elems = 0;
    };
    void set_postops(const postops_cfg_t &cfg) { postops_cfg_ = cfg; }

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

        // Create the post-ops injector and the callback the emitter uses to
        // lower the `inject_postops` op, the same as a real IR kernel's
        // generate(). The injector saves and restores every register it borrows,
        // so it takes no part in the IR register allocation.
        std::unique_ptr<postops_injector_t> injector;
        inject_postops_fn_t emit_injector;
        if (postops_cfg_.post_ops) {
            injector.reset(new postops_injector_t(*this, avx2,
                    *postops_cfg_.post_ops, *postops_cfg_.dst_md, abi_param1,
                    postops_cfg_.rhs_arg_offset, postops_cfg_.dst_orig_offset,
                    postops_cfg_.tail_elems));
            emit_injector = [&](const std::vector<int> &acc_phys, int base_phys,
                                    const std::vector<dim_t> &out_byte_off,
                                    int mask_phys, int elems) {
                injector->apply(
                        acc_phys, base_phys, out_byte_off, mask_phys, elems);
            };
        }

        // Create an eltwise injector for each algorithm the IR uses, discovered
        // by scanning the `veltwise` ops, the same way a real IR kernel would.
        // Like the post-ops injector, each one saves and restores every register
        // it borrows, so it takes no part in the IR register allocation. One
        // generic callback dispatches by algorithm, so a new algorithm needs no
        // new wiring here.
        std::map<alg_kind_t, std::unique_ptr<eltwise_injector_t>>
                eltwise_injectors;
        for (const auto &op : ir_.ops()) {
            if (op.kind != op_kind_t::veltwise) continue;
            const auto alg = (alg_kind_t)op.imm;
            if (eltwise_injectors.count(alg)) continue;
            eltwise_injectors.emplace(alg,
                    std::unique_ptr<eltwise_injector_t>(
                            new eltwise_injector_t(*this, avx2, alg,
                                    /* alpha = */ 0.f, /* beta = */ 0.f,
                                    /* scale = */ 1.f)));
        }
        eltwise_fn_t emit_eltwise;
        if (!eltwise_injectors.empty()) {
            emit_eltwise = [&](alg_kind_t alg, int vec_phys) {
                eltwise_injectors.at(alg)->apply(vec_phys);
            };
        }

        preamble();

        const int frame = (int)utils::rnd_up(alloc.frame_bytes, 16);
        if (frame > 0) sub(rsp, frame);

        data_section_t data;
        emit(*this, ir_, alloc, reg_cfg, data, emit_injector, emit_eltwise);

        if (frame > 0) add(rsp, frame);

        postamble();

        emit_data_section(*this, data);

        // The injector's constant table follows the postamble. A no-op unless
        // the chain has an eltwise post-op.
        if (injector) injector->maybe_prepare_table();

        // Each eltwise injector's constant table also follows the postamble.
        for (auto &kv : eltwise_injectors)
            kv.second->prepare_table();
    }

private:
    ir_t ir_ {};
    int vec_regs_limit_ = 0;
    bool spilled_ = false;
    size_t stack_size_ = 0;
    postops_cfg_t postops_cfg_ {};
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
    ir_t ir;
    const vreg_t ptr = ir.new_gpr();
    ir.load_param(ptr, 0);

    const vreg_t acc = ir.new_vec(data_type::f32);
    ir.vzero(acc);

    const vreg_t a = ir.new_vec(data_type::f32);
    ir.vload(a, ptr, 0);

    const vreg_t b = ir.new_vec(data_type::f32);
    // AVX2 only.
    ir.vload(b, ptr, simd_w * (dim_t)sizeof(float));

    ir.vdot(acc, a, b);

    // Instructions appear in the exact order they were built.
    ASSERT_EQ(ir.n_ops(), 5);
    EXPECT_EQ(ir.ops()[0].kind, op_kind_t::load);
    EXPECT_EQ(ir.ops()[1].kind, op_kind_t::vzero);
    EXPECT_EQ(ir.ops()[2].kind, op_kind_t::vload);
    EXPECT_EQ(ir.ops()[3].kind, op_kind_t::vload);
    EXPECT_EQ(ir.ops()[4].kind, op_kind_t::vdot);

    // Register kinds and data type are recorded for the allocator and emitter.
    EXPECT_EQ(ir.vreg_info()[(int)ptr].kind, reg_kind_t::gpr);
    EXPECT_EQ(ir.vreg_info()[(int)ptr].dt, data_type::undef);

    EXPECT_EQ(ir.vreg_info()[(int)a].kind, reg_kind_t::vec);
    EXPECT_EQ(ir.vreg_info()[(int)a].dt, data_type::f32);

    EXPECT_EQ(ir.vreg_info()[(int)b].kind, reg_kind_t::vec);
    EXPECT_EQ(ir.vreg_info()[(int)b].dt, data_type::f32);

    EXPECT_EQ(ir.vreg_info()[(int)acc].kind, reg_kind_t::vec);
    EXPECT_EQ(ir.vreg_info()[(int)acc].dt, data_type::f32);

    std::vector<int> defs, uses;

    // vzero writes its destination and reads nothing.
    ir.def_use(ir.ops()[1], defs, uses);
    EXPECT_EQ(defs, std::vector<int>({(int)acc}));
    EXPECT_TRUE(uses.empty());

    // vload reads the base pointer and writes the loaded vector.
    ir.def_use(ir.ops()[2], defs, uses);
    EXPECT_EQ(defs, std::vector<int>({(int)a}));
    EXPECT_EQ(uses, std::vector<int>({(int)ptr}));

    // vdot accumulates in place so the destination is read and written, both
    // sources are read.
    ir.def_use(ir.ops()[4], defs, uses);
    EXPECT_EQ(defs, std::vector<int>({(int)acc}));
    ASSERT_EQ(uses.size(), 3u);
    EXPECT_NE(std::find(uses.begin(), uses.end(), (int)a), uses.end());
    EXPECT_NE(std::find(uses.begin(), uses.end(), (int)b), uses.end());
    EXPECT_NE(std::find(uses.begin(), uses.end(), (int)acc), uses.end());
}

// Validates loop construction. A real loop links its end back to its begin and
// shares one counter register, while a loop that would run only once is inlined
// rather than emitted as a branch that is never taken.
TEST(IRBuilderTests, LoopLinkageAndSingleIterationInlining) {
    {
        ir_t ir;
        const vreg_t acc = ir.new_gpr();
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
        EXPECT_EQ(
                ir.vreg_info()[(int)ir.ops()[begin].dst].kind, reg_kind_t::gpr);
    }

    {
        // A single-iteration loop is inlined so no loop markers are emitted.
        ir_t ir;
        const vreg_t acc = ir.new_gpr();
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
    ir_t ir;

    const vreg_t cond = ir.new_gpr();
    ir.load_param(cond, 0);

    const vreg_t a = ir.new_vec(data_type::f32);
    const vreg_t acc = ir.new_vec(data_type::f32);

    const vreg_t base = ir.new_gpr();

    ir.load_param(base, sizeof(int));
    ir.vload(a, base, 0);

    const label_t lbl_else = ir.new_label();
    const label_t lbl_end = ir.new_label();

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
    ir.vdot(acc, a, a); // arbitrary work, just to populate the block
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
            bound[(int)ir.ops()[i].label_id] = i;
        }
    }

    // != -1 means it's bound.
    EXPECT_NE(bound[(int)lbl_else], -1);
    EXPECT_NE(bound[(int)lbl_end], -1);

    const int jz_idx = find_op(ir, op_kind_t::jz);
    const int jmp_idx = find_op(ir, op_kind_t::jmp);
    ASSERT_NE(jz_idx, -1);
    ASSERT_NE(jmp_idx, -1);

    // The conditional jump targets the else label. The unconditional jump jumps
    // over the else block to the end label.
    EXPECT_EQ(ir.ops()[jz_idx].label_id, lbl_else);
    EXPECT_EQ(ir.ops()[jmp_idx].label_id, lbl_end);

    // The then block's jmp precedes the else label it jumps over.
    EXPECT_LT(jmp_idx, bound[(int)lbl_else]);

    // The IR that contains branches can be handled by the allocator.
    // It assigns a location to every value that is read somewhere.
    const reg_pools_t pools = make_pools(/*n_gpr=*/8);
    const reg_alloc_result_t res = allocate_registers(ir, pools);
    // All virtual registers have been assigned.
    ASSERT_EQ((int)res.assignments.size(), ir.n_vregs());

    for (vreg_t v : {cond, base, a, acc}) {
        const assignment_t &as = res.assignments[(int)v];
        // Each value is either spilled or has a physical register assigned.
        EXPECT_TRUE(as.spilled || as.phys >= 0);
    }
}

// Validates the inject_postops builder and its liveness. The operation stores
// its variable-length operands in a side table and keeps only the table index,
// so the test checks that the side table holds exactly what the builder was
// given. def_use must report each accumulator as read and written in place, and
// the base pointer and mask as read, or liveness across the injected post-ops
// would be wrong.
TEST(IRBuilderTests, InjectPostopsRecordsArgsAndDefUse) {
    ir_t ir;

    const vreg_t base = ir.new_gpr();
    ir.load_param(base, 0);

    const vreg_t mask = ir.new_mask();
    ir.set_mask_imm(mask, simd_w - 1);

    constexpr int n = 3;
    std::vector<vreg_t> acc(n, vreg_t::none);
    for (int r = 0; r < n; r++) {
        acc[r] = ir.new_vec(data_type::f32);
        ir.vzero(acc[r]);
    }

    std::vector<dim_t> out_byte_off(n);
    for (int r = 0; r < n; r++)
        out_byte_off[r] = r * (dim_t)sizeof(float);

    ir.inject_postops(acc, base, out_byte_off, mask, /*elems=*/simd_w - 1);

    const int idx = find_op(ir, op_kind_t::inject_postops);
    ASSERT_NE(idx, -1);

    // The operation carries only an index into the side table, which holds the
    // full argument set.
    ASSERT_EQ(ir.inject_postops_args().size(), 1u);
    const auto &args = ir.inject_postops_args()[(int)ir.ops()[idx].imm];
    EXPECT_EQ(args.acc, acc);
    EXPECT_EQ(args.base_ptr, base);
    EXPECT_EQ(args.out_byte_off, out_byte_off);
    EXPECT_EQ(args.mask, mask);
    EXPECT_EQ(args.elems, simd_w - 1);

    // def_use reports every accumulator as read and written, and the base
    // pointer and mask as read.
    std::vector<int> defs, uses;
    ir.def_use(ir.ops()[idx], defs, uses);

    for (int r = 0; r < n; r++) {
        EXPECT_NE(std::find(defs.begin(), defs.end(), (int)acc[r]), defs.end())
                << "acc " << r << " not written";
        EXPECT_NE(std::find(uses.begin(), uses.end(), (int)acc[r]), uses.end())
                << "acc " << r << " not read";
    }
    EXPECT_NE(std::find(uses.begin(), uses.end(), (int)base), uses.end());
    EXPECT_NE(std::find(uses.begin(), uses.end(), (int)mask), uses.end());
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
    ir_t ir;

    const vreg_t acc = ir.new_gpr();
    ir.mov_imm(acc, 0);

    const vreg_t t0 = ir.new_gpr();
    ir.mov_imm(t0, 5);
    ir.add_reg(acc, t0); // t0 dies here

    const vreg_t t1 = ir.new_gpr();
    ir.mov_imm(t1, 7);
    ir.add_reg(acc, t1); // t1 is used only after t0 is dead

    // Two registers are sufficient. One for the accumulator and one reused by
    // t0 then t1.
    const reg_pools_t pools = make_pools(/*n_gpr=*/2);
    const reg_alloc_result_t res = allocate_registers(ir, pools);

    EXPECT_FALSE(res.any_spill);
    EXPECT_FALSE(res.assignments[(int)t0].spilled);
    EXPECT_FALSE(res.assignments[(int)t1].spilled);
    // t0 and t1 reuse the same physical register.
    EXPECT_EQ(res.assignments[(int)t0].phys, res.assignments[(int)t1].phys);
    // The accumulator, live across both, does not collide with them.
    EXPECT_NE(res.assignments[(int)acc].phys, res.assignments[(int)t0].phys);
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
    ir_t ir;

    // Load pointers for a, b, c.
    const vreg_t a_ptr = ir.new_gpr();
    ir.load_param(a_ptr, 0);

    const vreg_t b_ptr = ir.new_gpr();
    ir.load_param(b_ptr, sizeof(void *));

    const vreg_t c_ptr = ir.new_gpr();
    ir.load_param(c_ptr, 2 * sizeof(void *));

    const vreg_t acc = ir.new_vec(data_type::f32);
    ir.vzero(acc);

    const vreg_t a = ir.new_vec(data_type::f32);
    ir.vload(a, a_ptr, 0);

    const vreg_t b = ir.new_vec(data_type::f32);
    ir.vload(b, b_ptr, 0);

    ir.vdot(acc, a, b);

    const vreg_t ws = ir.new_vec(data_type::f32);
    ir.vhreduce(acc, ws);

    // store the reduced scalar
    ir.vstore_masked(c_ptr, 0, acc, vreg_t::none, 1);

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
    ir_t ir;

    const vreg_t a_ptr = ir.new_gpr();
    ir.load_param(a_ptr, 0);

    const vreg_t b_ptr = ir.new_gpr();
    ir.load_param(b_ptr, sizeof(void *));

    const vreg_t b = ir.new_vec(data_type::f32);
    ir.vload(b, b_ptr, 0);

    std::vector<vreg_t> acc(6, vreg_t::none);
    for (int r = 0; r < 6; r++) {
        acc[r] = ir.new_vec(data_type::f32);
        ir.vzero(acc[r]);
    }

    for (int r = 0; r < 6; r++) {
        const vreg_t a = ir.new_vec(data_type::f32);
        ir.vload(a, a_ptr, r * simd_w * (dim_t)sizeof(float));
        ir.vdot(acc[r], a, b);
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

    ir_t ir;

    const vreg_t a_ptr = ir.new_gpr();
    ir.load_param(a_ptr, offsetof(dot_args_t, a));

    const vreg_t b_ptr = ir.new_gpr();
    ir.load_param(b_ptr, offsetof(dot_args_t, b));

    const vreg_t c_ptr = ir.new_gpr();
    ir.load_param(c_ptr, offsetof(dot_args_t, c));

    const vreg_t acc = ir.new_vec(data_type::f32);
    ir.vzero(acc);

    // Reduce one simd_w-wide chunk per iteration and advance the pointers.
    emit_loop_imm(ir, k_blocks, [&]() {
        const vreg_t a = ir.new_vec(data_type::f32);
        ir.vload(a, a_ptr, 0);
        const vreg_t b = ir.new_vec(data_type::f32);
        ir.vload(b, b_ptr, 0);
        ir.vdot(acc, a, b);
    }, [&]() {
        ir.add_imm(a_ptr, simd_w * (dim_t)sizeof(float));
        ir.add_imm(b_ptr, simd_w * (dim_t)sizeof(float));
    });

    const vreg_t ws = ir.new_vec(data_type::f32);
    ir.vhreduce(acc, ws);
    ir.vstore_masked(c_ptr, 0, acc, vreg_t::none, 1);

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
    ir_t ir;

    const vreg_t a_ptr = ir.new_gpr();
    ir.load_param(a_ptr, offsetof(dot_args_t, a));

    const vreg_t b_ptr = ir.new_gpr();
    ir.load_param(b_ptr, offsetof(dot_args_t, b));

    const vreg_t c_ptr = ir.new_gpr();
    ir.load_param(c_ptr, offsetof(dot_args_t, c));

    const vreg_t b = ir.new_vec(data_type::f32);
    // Load shared vector.
    ir.vload(b, b_ptr, 0);

    std::vector<vreg_t> acc(n, vreg_t::none);
    for (int r = 0; r < n; r++) {
        acc[r] = ir.new_vec(data_type::f32);
        ir.vzero(acc[r]);

        const vreg_t a = ir.new_vec(data_type::f32);
        ir.vload(a, a_ptr, r * simd_w * (dim_t)sizeof(float));

        ir.vdot(acc[r], a, b);
    }

    const vreg_t ws = ir.new_vec(data_type::f32);
    for (int r = 0; r < n; r++)
        ir.vhreduce(acc[r], ws);

    for (int r = 0; r < n; r++)
        ir.vstore_masked(
                c_ptr, r * (dim_t)sizeof(float), acc[r], vreg_t::none, 1);

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

    ir_t ir;

    const vreg_t cond = ir.new_gpr();
    ir.load_param(cond, offsetof(select_args_t, cond));

    const vreg_t a_ptr = ir.new_gpr();
    ir.load_param(a_ptr, offsetof(select_args_t, a));

    const vreg_t b_ptr = ir.new_gpr();
    ir.load_param(b_ptr, offsetof(select_args_t, b));

    const vreg_t c_ptr = ir.new_gpr();
    ir.load_param(c_ptr, offsetof(select_args_t, c));

    const vreg_t a = ir.new_vec(data_type::f32);
    ir.vload(a, a_ptr, 0);

    const vreg_t b = ir.new_vec(data_type::f32);
    ir.vload(b, b_ptr, 0);

    const label_t lbl_else = ir.new_label();
    const label_t lbl_end = ir.new_label();

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
    ir.vstore_masked(c_ptr, 0, a, vreg_t::none, simd_w); // then: c = a
    ir.jmp(lbl_end);
    ir.label(lbl_else);
    ir.vstore_masked(c_ptr, 0, b, vreg_t::none, simd_w); // else: c = b
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

// Arguments for the binary post-op kernel. `binary_rhs` points to the array of
// right-hand-side base pointers the injector indexes, and `dst_orig` is the
// destination origin it subtracts to locate each accumulator's output element.
struct binary_args_t {
    const float *a;
    const float *b;
    float *c;
    const void **binary_rhs;
    const void *dst_orig;
};

// Validates the binary post-op path end to end. It computes n independent dot
// products, adds a per-element right-hand-side through the JIT post-ops
// injector, and checks each result against a reference. This exercises the
// inject_postops op, the postops_injector_t driver, and the injector's
// destination-relative right-hand-side addressing with a one-element
// (scalar accumulator) tail load.
TEST(IntegrationTests, BinaryPostOpAddsPerElementRhs) {
    SKIP_IF_NO_AVX2();

    constexpr int n = 4;

    // Destination and right-hand-side share the same n x 1 shape, which selects
    // the injector's no-broadcast (per-element) strategy. The descriptors and
    // post-ops are built with the public API, then handed to the injector as the
    // internal types it takes.
    using dt = dnnl::memory::data_type;
    using tag = dnnl::memory::format_tag;
    const dnnl::memory::desc dst_desc({n, 1}, dt::f32, tag::ab);
    const dnnl::memory::desc rhs_desc({n, 1}, dt::f32, tag::ab);

    dnnl::post_ops po;
    po.append_binary(dnnl::algorithm::binary_add, rhs_desc);

    const impl::memory_desc_t &dst_md = *dst_desc.get();
    const impl::post_ops_t &post_ops = *po.get();

    // Build the dot-product IR that feeds the injector: one accumulator per
    // output element, each reduced to a scalar.
    ir_t ir;

    const vreg_t a_ptr = ir.new_gpr();
    ir.load_param(a_ptr, offsetof(binary_args_t, a));

    const vreg_t b_ptr = ir.new_gpr();
    ir.load_param(b_ptr, offsetof(binary_args_t, b));

    const vreg_t c_ptr = ir.new_gpr();
    ir.load_param(c_ptr, offsetof(binary_args_t, c));

    const vreg_t b = ir.new_vec(data_type::f32);
    ir.vload(b, b_ptr, 0);

    std::vector<vreg_t> acc(n, vreg_t::none);
    for (int r = 0; r < n; r++) {
        acc[r] = ir.new_vec(data_type::f32);
        ir.vzero(acc[r]);

        const vreg_t a = ir.new_vec(data_type::f32);
        ir.vload(a, a_ptr, r * simd_w * (dim_t)sizeof(float));
        ir.vdot(acc[r], a, b);
    }

    const vreg_t ws = ir.new_vec(data_type::f32);
    for (int r = 0; r < n; r++)
        ir.vhreduce(acc[r], ws);

    // Each accumulator is a single scalar. Its output offset locates it in the
    // destination so the injector reads the matching right-hand-side element.
    std::vector<dim_t> out_byte_off(n);
    for (int r = 0; r < n; r++)
        out_byte_off[r] = r * (dim_t)sizeof(float);
    ir.inject_postops(acc, c_ptr, out_byte_off, vreg_t::none, /*elems=*/1);

    for (int r = 0; r < n; r++)
        ir.vstore_masked(
                c_ptr, r * (dim_t)sizeof(float), acc[r], vreg_t::none, 1);

    ir_kernel_t kernel(ir);
    ir_kernel_t::postops_cfg_t cfg;
    cfg.post_ops = &post_ops;
    cfg.dst_md = &dst_md;
    cfg.rhs_arg_offset = offsetof(binary_args_t, binary_rhs);
    cfg.dst_orig_offset = offsetof(binary_args_t, dst_orig);
    cfg.tail_elems = 1;
    kernel.set_postops(cfg);

    ASSERT_TRUE(kernel.run_ir_pipeline());

    std::vector<float> a(n * simd_w), b_data(simd_w), rhs(n), c(n, 0.f);
    for (int i = 0; i < n * simd_w; i++)
        a[i] = (float)(i % 5) - 2.f;
    for (int i = 0; i < simd_w; i++)
        b_data[i] = (float)(i - 3);
    for (int r = 0; r < n; r++)
        rhs[r] = (float)(10 * (r + 1));

    const void *rhs_ptrs[1] = {rhs.data()};
    binary_args_t args {a.data(), b_data.data(), c.data(), rhs_ptrs, c.data()};
    kernel.run(&args);

    for (int r = 0; r < n; r++) {
        const float expected
                = ref_dot(&a[r * simd_w], b_data.data(), simd_w) + rhs[r];
        EXPECT_FLOAT_EQ(c[r], expected) << "row " << r;
    }
}

// Validates def_use for the elementwise/reduction ops added for the softmax
// epilogue. The rmw ops read dst and s0 and write dst; vbcast overwrites dst
// and only reads s0; vhreduce_max reads and writes both dst and its scratch.
TEST(IRBuilderTests, NewVectorOpsDefUse) {
    ir_t ir;
    const vreg_t ptr = ir.new_gpr();
    ir.load_param(ptr, 0);
    const vreg_t x = ir.new_vec(data_type::f32);
    ir.vload(x, ptr, 0);
    const vreg_t y = ir.new_vec(data_type::f32);
    ir.vload(y, ptr, simd_w * (dim_t)sizeof(float));

    const int i_sub = ir.n_ops();
    ir.vsub(x, y);
    const int i_div = ir.n_ops();
    ir.vdiv(x, y);
    const int i_max = ir.n_ops();
    ir.vmax(x, y);
    const vreg_t mask = ir.new_mask();
    ir.set_mask_imm(mask, simd_w - 1);
    const int i_blend = ir.n_ops();
    ir.vblend(x, y, mask);
    const vreg_t bc = ir.new_vec(data_type::f32);
    const int i_bc = ir.n_ops();
    ir.vbcast(bc, x);
    const vreg_t ws = ir.new_vec(data_type::f32);
    const int i_hm = ir.n_ops();
    ir.vhreduce_max(x, ws);

    const int i_exp = ir.n_ops();
    ir.vexp(x);

    std::vector<int> defs, uses;
    // rmw ops read dst and s0 and write dst.
    for (int idx : {i_sub, i_div, i_max}) {
        ir.def_use(ir.ops()[idx], defs, uses);
        EXPECT_EQ(defs, std::vector<int>({(int)x}));
        ASSERT_EQ(uses.size(), 2u);
        EXPECT_NE(std::find(uses.begin(), uses.end(), (int)x), uses.end());
        EXPECT_NE(std::find(uses.begin(), uses.end(), (int)y), uses.end());
    }
    // vbcast overwrites dst and reads s0.
    ir.def_use(ir.ops()[i_bc], defs, uses);
    EXPECT_EQ(defs, std::vector<int>({(int)bc}));
    EXPECT_EQ(uses, std::vector<int>({(int)x}));
    // vblend reads dst, s0 and the mask, and writes dst.
    ir.def_use(ir.ops()[i_blend], defs, uses);
    EXPECT_EQ(defs, std::vector<int>({(int)x}));
    ASSERT_EQ(uses.size(), 3u);
    EXPECT_NE(std::find(uses.begin(), uses.end(), (int)x), uses.end());
    EXPECT_NE(std::find(uses.begin(), uses.end(), (int)y), uses.end());
    EXPECT_NE(std::find(uses.begin(), uses.end(), (int)mask), uses.end());
    // vhreduce_max reads and writes both dst and workspace.
    ir.def_use(ir.ops()[i_hm], defs, uses);
    ASSERT_EQ(defs.size(), 2u);
    EXPECT_NE(std::find(defs.begin(), defs.end(), (int)x), defs.end());
    EXPECT_NE(std::find(defs.begin(), defs.end(), (int)ws), defs.end());
    ASSERT_EQ(uses.size(), 2u);
    EXPECT_NE(std::find(uses.begin(), uses.end(), (int)x), uses.end());
    EXPECT_NE(std::find(uses.begin(), uses.end(), (int)ws), uses.end());
    // vexp reads and writes dst in place.
    ir.def_use(ir.ops()[i_exp], defs, uses);
    EXPECT_EQ(defs, std::vector<int>({(int)x}));
    EXPECT_EQ(uses, std::vector<int>({(int)x}));
}

// Builds a kernel that applies one elementwise op to two loaded vectors and
// stores the full result. `kind` selects vsub/vmul/vdiv/vmax.
ir_t build_elementwise_ir(op_kind_t kind) {
    ir_t ir;

    const vreg_t a_ptr = ir.new_gpr();
    ir.load_param(a_ptr, offsetof(dot_args_t, a));
    const vreg_t b_ptr = ir.new_gpr();
    ir.load_param(b_ptr, offsetof(dot_args_t, b));
    const vreg_t c_ptr = ir.new_gpr();
    ir.load_param(c_ptr, offsetof(dot_args_t, c));

    const vreg_t acc = ir.new_vec(data_type::f32);
    ir.vload(acc, a_ptr, 0);
    const vreg_t b = ir.new_vec(data_type::f32);
    ir.vload(b, b_ptr, 0);

    switch (kind) {
        case op_kind_t::vsub: ir.vsub(acc, b); break;
        case op_kind_t::vmul: ir.vmul(acc, b); break;
        case op_kind_t::vdiv: ir.vdiv(acc, b); break;
        case op_kind_t::vmax: ir.vmax(acc, b); break;
        default: break;
    }

    ir.vstore_masked(c_ptr, 0, acc, vreg_t::none, simd_w);
    return ir;
}

// Validates the elementwise vector ops end to end against a scalar reference.
TEST(IntegrationTests, ElementwiseVectorOps) {
    SKIP_IF_NO_AVX2();

    std::vector<float> a(simd_w), b(simd_w);
    for (int i = 0; i < simd_w; i++) {
        a[i] = (float)(i - 3) * 1.5f + 0.5f;
        b[i] = (float)(i % 4) + 1.f; // nonzero for division
    }

    auto run = [&](op_kind_t kind) {
        ir_kernel_t k(build_elementwise_ir(kind));
        EXPECT_TRUE(k.run_ir_pipeline());
        std::vector<float> c(simd_w, -12345.f);
        dot_args_t args {a.data(), b.data(), c.data()};
        k.run(&args);
        return c;
    };

    {
        const auto c = run(op_kind_t::vsub);
        for (int i = 0; i < simd_w; i++)
            EXPECT_FLOAT_EQ(c[i], a[i] - b[i]) << "lane " << i;
    }
    {
        const auto c = run(op_kind_t::vmax);
        for (int i = 0; i < simd_w; i++)
            EXPECT_FLOAT_EQ(c[i], std::max(a[i], b[i])) << "lane " << i;
    }
    {
        const auto c = run(op_kind_t::vdiv);
        for (int i = 0; i < simd_w; i++)
            EXPECT_FLOAT_EQ(c[i], a[i] / b[i]) << "lane " << i;
    }
}

// Builds a kernel that reduces a vector to its horizontal max and broadcasts
// that scalar back across all lanes.
ir_t build_hmax_bcast_ir() {
    ir_t ir;

    const vreg_t a_ptr = ir.new_gpr();
    ir.load_param(a_ptr, offsetof(dot_args_t, a));
    const vreg_t c_ptr = ir.new_gpr();
    ir.load_param(c_ptr, offsetof(dot_args_t, c));

    const vreg_t acc = ir.new_vec(data_type::f32);
    ir.vload(acc, a_ptr, 0);

    const vreg_t ws = ir.new_vec(data_type::f32);
    ir.vhreduce_max(acc, ws);

    const vreg_t out = ir.new_vec(data_type::f32);
    ir.vbcast(out, acc);

    ir.vstore_masked(c_ptr, 0, out, vreg_t::none, simd_w);
    return ir;
}

// Validates horizontal-max reduction plus broadcast: every output lane must
// equal the maximum input element. This is the row-max step of online softmax.
TEST(IntegrationTests, HorizontalMaxBroadcast) {
    SKIP_IF_NO_AVX2();

    ir_kernel_t kernel(build_hmax_bcast_ir());
    ASSERT_TRUE(kernel.run_ir_pipeline());

    std::vector<float> a(simd_w), c(simd_w, -12345.f);
    for (int i = 0; i < simd_w; i++)
        a[i] = (float)((i * 3) % 7) - 2.f;

    dot_args_t args {a.data(), nullptr, c.data()};
    kernel.run(&args);

    const float m = *std::max_element(a.begin(), a.end());
    for (int i = 0; i < simd_w; i++)
        EXPECT_FLOAT_EQ(c[i], m) << "lane " << i;
}

// Builds a kernel that applies exp element-wise to a loaded vector, in place,
// and stores it. This is the P = exp(S - m) step of online softmax.
ir_t build_exp_ir() {
    ir_t ir;

    const vreg_t a_ptr = ir.new_gpr();
    ir.load_param(a_ptr, offsetof(dot_args_t, a));
    const vreg_t c_ptr = ir.new_gpr();
    ir.load_param(c_ptr, offsetof(dot_args_t, c));

    const vreg_t acc = ir.new_vec(data_type::f32);
    ir.vload(acc, a_ptr, 0);
    ir.vexp(acc);
    ir.vstore_masked(c_ptr, 0, acc, vreg_t::none, simd_w);
    return ir;
}

// Validates the `vexp` op end to end. The eltwise injector's exp is a
// polynomial approximation, so compare with a relative tolerance rather than
// exact equality.
TEST(IntegrationTests, ExpVector) {
    SKIP_IF_NO_AVX2();

    ir_kernel_t kernel(build_exp_ir());
    ASSERT_TRUE(kernel.run_ir_pipeline());

    std::vector<float> a(simd_w), c(simd_w, -12345.f);
    for (int i = 0; i < simd_w; i++)
        a[i] = (float)(i - 4) * 0.75f; // spans negative and positive

    dot_args_t args {a.data(), nullptr, c.data()};
    kernel.run(&args);

    for (int i = 0; i < simd_w; i++) {
        const float expected = std::exp(a[i]);
        EXPECT_NEAR(c[i], expected, 1e-5f * std::abs(expected) + 1e-6f)
                << "lane " << i;
    }
}

// Builds a kernel that applies two different eltwise algorithms in the same
// kernel: exp to the first input, tanh to the second, storing both. This shows
// that distinct `veltwise` ops carry their own algorithm (in `op.imm`) and are
// dispatched to the matching injector, with no per-algorithm wiring.
ir_t build_two_eltwise_ir() {
    ir_t ir;

    const vreg_t a_ptr = ir.new_gpr();
    ir.load_param(a_ptr, offsetof(dot_args_t, a));
    const vreg_t b_ptr = ir.new_gpr();
    ir.load_param(b_ptr, offsetof(dot_args_t, b));
    const vreg_t c_ptr = ir.new_gpr();
    ir.load_param(c_ptr, offsetof(dot_args_t, c));

    const vreg_t x = ir.new_vec(data_type::f32);
    ir.vload(x, a_ptr, 0);
    ir.veltwise(alg_kind::eltwise_exp, x);
    ir.vstore_masked(c_ptr, 0, x, vreg_t::none, simd_w);

    const vreg_t y = ir.new_vec(data_type::f32);
    ir.vload(y, b_ptr, 0);
    ir.veltwise(alg_kind::eltwise_tanh, y);
    ir.vstore_masked(
            c_ptr, simd_w * (dim_t)sizeof(float), y, vreg_t::none, simd_w);
    return ir;
}

// Validates that two different eltwise algorithms coexist in one kernel and are
// each applied correctly. If the algorithm were not distinguished per op, the
// tanh lanes would receive exp (and fail).
TEST(IntegrationTests, TwoEltwiseAlgorithms) {
    SKIP_IF_NO_AVX2();

    ir_kernel_t kernel(build_two_eltwise_ir());
    ASSERT_TRUE(kernel.run_ir_pipeline());

    std::vector<float> a(simd_w), b(simd_w), c(2 * simd_w, -12345.f);
    for (int i = 0; i < simd_w; i++) {
        a[i] = (float)(i - 4) * 0.75f;
        b[i] = (float)(i - 4) * 0.5f;
    }

    dot_args_t args {a.data(), b.data(), c.data()};
    kernel.run(&args);

    for (int i = 0; i < simd_w; i++) {
        const float exp_ref = std::exp(a[i]);
        EXPECT_NEAR(c[i], exp_ref, 1e-5f * std::abs(exp_ref) + 1e-6f)
                << "exp lane " << i;
        const float tanh_ref = std::tanh(b[i]);
        EXPECT_NEAR(c[simd_w + i], tanh_ref, 1e-5f * std::abs(tanh_ref) + 1e-6f)
                << "tanh lane " << i;
    }
}

// Arguments for the online-softmax tile epilogue kernel. A tile of `seq_q`
// score rows of `w` elements each is updated in place; per row, the running
// max/denominator and the tile's renormalization coefficient are read and
// written through scalar pointers, matching the per-row state the fused SDPA
// kernel carries across KV tiles. scores holds seq_q*w floats (row i at i*w);
// m/l/old_coef hold seq_q floats (row i at i); scale is shared by all rows.
struct softmax_row_args_t {
    float *scores; // in: raw scores rows; out: normalized probabilities P
    const float *scale; // scalar softmax scale (shared by all rows)
    float *m; // in: running row max m_old; out: m_new (one per row)
    float *l; // in: running denominator l_old; out: l_new (one per row)
    float *old_coef; // out: corr*l_old/l_new per row (renormalizes running acc)
};

// Builds the online-softmax epilogue for a tile of `seq_q` score rows, each of
// width `w` (any w >= 1; the ragged tail beyond the last full simd_w block is
// handled with masked loads/stores). Mirrors the per-row scalar epilogue in
// sdp_fused_brgemm.cpp for a single KV tile, minus the select mask (deferred).
// The rows are processed by a runtime loop over seq_q (inlined when seq_q == 1);
// the score and per-row state pointers advance one row per iteration. Per row
// the op chain is: scale -> running row max -> exp(scaled - m_new) -> running
// denominator -> divide by l_new. The scalar running state is kept in broadcast
// vectors so every step is one of the vector ops added for this framework, with
// no float-immediate op needed (old_coef and the normalization use vdiv). The
// tail's unused lanes are neutralized before each reduction with `vblend`:
// seeded with m_old for the max (never beats the running max) and 0 for the
// sum. The first KV tile (m_old == -inf, l_old == 0) needs no explicit guard:
// the eltwise exp saturates -inf to exactly 0, so corr == 0 and clo ==
// l_old*corr == 0 fall out, and m_new is just the tile max. The scale pointer
// and tail mask are loop invariant, set up once; the mask vreg stays live
// across all rows (masked ops assert it is never spilled).
ir_t build_softmax_tile_ir(int seq_q, int w) {
    const int n_blk = w / simd_w;
    const int tail = w % simd_w;
    const dim_t vbytes = simd_w * (dim_t)sizeof(float);
    const dim_t tail_off = n_blk * vbytes;
    const dim_t fsz = (dim_t)sizeof(float);

    ir_t ir;

    // Row pointers: advanced one row per loop iteration.
    const vreg_t sc_ptr = ir.new_gpr();
    ir.load_param(sc_ptr, offsetof(softmax_row_args_t, scores));
    const vreg_t m_ptr = ir.new_gpr();
    ir.load_param(m_ptr, offsetof(softmax_row_args_t, m));
    const vreg_t l_ptr = ir.new_gpr();
    ir.load_param(l_ptr, offsetof(softmax_row_args_t, l));
    const vreg_t oc_ptr = ir.new_gpr();
    ir.load_param(oc_ptr, offsetof(softmax_row_args_t, old_coef));
    // Loop invariant: the scale is shared by every row.
    const vreg_t scale_ptr = ir.new_gpr();
    ir.load_param(scale_ptr, offsetof(softmax_row_args_t, scale));

    // One mask, reused by every masked op and every row, active for `tail`.
    vreg_t mask = vreg_t::none;
    if (tail) {
        mask = ir.new_mask();
        ir.set_mask_imm(mask, tail);
    }

    // Online-softmax epilogue for the single row at the current pointers.
    auto row_body = [&]() {
        // Scratch shared by both horizontal reductions (overwritten each time).
        const vreg_t ws = ir.new_vec(data_type::f32);

        // Broadcast the scalar inputs so scalar arithmetic reuses vector ops.
        const vreg_t scale_bc = ir.new_vec(data_type::f32);
        ir.vload_masked(scale_bc, scale_ptr, 0, vreg_t::none, 1);
        ir.vbcast(scale_bc, scale_bc);

        const vreg_t m_old = ir.new_vec(data_type::f32);
        ir.vload_masked(m_old, m_ptr, 0, vreg_t::none, 1);
        ir.vbcast(m_old, m_old);

        const vreg_t l_old = ir.new_vec(data_type::f32);
        ir.vload_masked(l_old, l_ptr, 0, vreg_t::none, 1);
        ir.vbcast(l_old, l_old);

        // Pass 1: scale each block, store it back, and fold it into the running
        // max (seeded with m_old so the reduction yields m_new directly).
        const vreg_t rmax = ir.new_vec(data_type::f32);
        ir.vbcast(rmax, m_old);
        for (int b = 0; b < n_blk; b++) {
            const vreg_t blk = ir.new_vec(data_type::f32);
            ir.vload(blk, sc_ptr, b * vbytes);
            ir.vmul(blk, scale_bc);
            ir.vstore_masked(sc_ptr, b * vbytes, blk, vreg_t::none, simd_w);
            ir.vmax(rmax, blk);
        }
        if (tail) {
            const vreg_t blk = ir.new_vec(data_type::f32);
            ir.vload_masked(blk, sc_ptr, tail_off, mask, tail);
            ir.vmul(blk, scale_bc);
            ir.vstore_masked(sc_ptr, tail_off, blk, mask, tail);
            // Unused lanes take m_old so they never win the max.
            const vreg_t tmax = ir.new_vec(data_type::f32);
            ir.vbcast(tmax, m_old);
            ir.vblend(tmax, blk, mask);
            ir.vmax(rmax, tmax);
        }
        ir.vhreduce_max(rmax, ws); // rmax lane 0 = m_new
        const vreg_t m_new = ir.new_vec(data_type::f32);
        ir.vbcast(m_new, rmax);

        // corr = exp(m_old - m_new): rescales previous tiles' contributions.
        const vreg_t corr = ir.new_vec(data_type::f32);
        ir.vbcast(corr, m_old);
        ir.vsub(corr, m_new);
        ir.vexp(corr);

        // Pass 2: P_unnorm = exp(scaled - m_new); accumulate the tile denom.
        const vreg_t rsum = ir.new_vec(data_type::f32);
        ir.vzero(rsum);
        for (int b = 0; b < n_blk; b++) {
            const vreg_t blk = ir.new_vec(data_type::f32);
            ir.vload(blk, sc_ptr, b * vbytes);
            ir.vsub(blk, m_new);
            ir.vexp(blk);
            ir.vstore_masked(sc_ptr, b * vbytes, blk, vreg_t::none, simd_w);
            ir.vadd(rsum, blk);
        }
        if (tail) {
            const vreg_t blk = ir.new_vec(data_type::f32);
            ir.vload_masked(blk, sc_ptr, tail_off, mask, tail);
            ir.vsub(blk, m_new);
            ir.vexp(blk);
            ir.vstore_masked(sc_ptr, tail_off, blk, mask, tail);
            // Unused lanes take 0 so they add nothing to the denominator.
            const vreg_t tsum = ir.new_vec(data_type::f32);
            ir.vzero(tsum);
            ir.vblend(tsum, blk, mask);
            ir.vadd(rsum, tsum);
        }
        ir.vhreduce(rsum, ws); // rsum lane 0 = tile_sum
        const vreg_t tile_sum = ir.new_vec(data_type::f32);
        ir.vbcast(tile_sum, rsum);

        // clo = l_old*corr; l_new = clo + tile_sum; old_coef = clo / l_new.
        const vreg_t clo = ir.new_vec(data_type::f32);
        ir.vbcast(clo, l_old);
        ir.vmul(clo, corr);

        const vreg_t l_new = ir.new_vec(data_type::f32);
        ir.vbcast(l_new, clo);
        ir.vadd(l_new, tile_sum);

        const vreg_t old_coef = ir.new_vec(data_type::f32);
        ir.vbcast(old_coef, clo);
        ir.vdiv(old_coef, l_new);

        // Pass 3: normalize P by the running denominator.
        for (int b = 0; b < n_blk; b++) {
            const vreg_t blk = ir.new_vec(data_type::f32);
            ir.vload(blk, sc_ptr, b * vbytes);
            ir.vdiv(blk, l_new);
            ir.vstore_masked(sc_ptr, b * vbytes, blk, vreg_t::none, simd_w);
        }
        if (tail) {
            const vreg_t blk = ir.new_vec(data_type::f32);
            ir.vload_masked(blk, sc_ptr, tail_off, mask, tail);
            ir.vdiv(blk, l_new);
            ir.vstore_masked(sc_ptr, tail_off, blk, mask, tail);
        }

        // Write back this row's scalar running state (lane 0).
        ir.vstore_masked(m_ptr, 0, m_new, vreg_t::none, 1);
        ir.vstore_masked(l_ptr, 0, l_new, vreg_t::none, 1);
        ir.vstore_masked(oc_ptr, 0, old_coef, vreg_t::none, 1);
    };

    // Advance every row pointer to the next row.
    auto advance_row = [&]() {
        ir.add_imm(sc_ptr, w * fsz);
        ir.add_imm(m_ptr, fsz);
        ir.add_imm(l_ptr, fsz);
        ir.add_imm(oc_ptr, fsz);
    };

    emit_loop_imm(ir, seq_q, row_body, advance_row);

    return ir;
}

// Scalar reference for one online-softmax row update, matching one row of
// build_softmax_tile_ir (no select mask).
void ref_softmax_row(std::vector<float> &scores, float scale, float &m,
        float &l, float &old_coef) {
    const int w = (int)scores.size();
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const float m_old = m, l_old = l;
    float m_new = m_old;
    for (int j = 0; j < w; j++) {
        scores[j] *= scale;
        m_new = std::max(m_new, scores[j]);
    }
    // corr is 0 for the first tile (m_old == -inf), as in the fused kernel.
    const float corr = m_old == neg_inf ? 0.f : std::exp(m_old - m_new);
    float tile_sum = 0.f;
    for (int j = 0; j < w; j++) {
        const float e = std::exp(scores[j] - m_new);
        scores[j] = e;
        tile_sum += e;
    }
    const float l_new = l_old * corr + tile_sum;
    for (int j = 0; j < w; j++)
        scores[j] /= l_new;
    m = m_new;
    l = l_new;
    old_coef = (l_old * corr) / l_new;
}

// Validates the online-softmax tile epilogue end to end: the JIT builder's row
// update must match the scalar reference across several tile widths. This is
// the first step-3 milestone, proving the whole builder -> alloc -> emit -> run
// pipeline computes the real softmax math using the ops from steps 1 and 2. The
// eltwise exp is a polynomial approximation, so denominator-dependent outputs
// use a relative tolerance; m_new is a plain max and stays exact.
TEST(IntegrationTests, SoftmaxOnlineTileRow) {
    SKIP_IF_NO_AVX2();

    // Widths span pure tails (< simd_w), exact multiples, and multiples plus a
    // ragged tail, so the masked-tail path and its lane neutralization run.
    for (int w : {1, 5, 7, simd_w, 9, 15, 2 * simd_w, 17, 23, 4 * simd_w - 1,
                 4 * simd_w}) {
        ir_kernel_t kernel(build_softmax_tile_ir(1, w));
        ASSERT_TRUE(kernel.run_ir_pipeline()) << "w=" << w;

        std::vector<float> scores(w), ref(w);
        for (int j = 0; j < w; j++) {
            scores[j] = (float)((j * 5) % 13) * 0.5f - 3.f;
            ref[j] = scores[j];
        }
        const float scale = 0.125f;
        // Finite running state (a later, non-first KV tile).
        float m = -1.5f, l = 2.0f;
        float ref_m = m, ref_l = l, ref_oc = 0.f;
        ref_softmax_row(ref, scale, ref_m, ref_l, ref_oc);

        float oc = -12345.f;
        softmax_row_args_t args {scores.data(), &scale, &m, &l, &oc};
        kernel.run(&args);

        EXPECT_NEAR(m, ref_m, 1e-5f * std::abs(ref_m) + 1e-6f) << "w=" << w;
        EXPECT_NEAR(l, ref_l, 1e-4f * std::abs(ref_l) + 1e-6f) << "w=" << w;
        EXPECT_NEAR(oc, ref_oc, 1e-4f * std::abs(ref_oc) + 1e-6f) << "w=" << w;
        for (int j = 0; j < w; j++)
            EXPECT_NEAR(scores[j], ref[j], 1e-4f * std::abs(ref[j]) + 1e-6f)
                    << "w=" << w << " j=" << j;
    }
}

// Validates the first-tile path (m_old == -inf, l_old == 0) and the running
// state carried across tiles: one per-row state is threaded through two
// successive KV tiles, matching two scalar-reference updates. The first tile
// exercises the -inf seed (the max reduction and exp both saturate correctly so
// corr == 0); the second tile then consumes the finite state it produced.
TEST(IntegrationTests, SoftmaxOnlineFirstTile) {
    SKIP_IF_NO_AVX2();

    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int w : {1, 5, 7, simd_w, 9, 15, 2 * simd_w, 17, 23, 4 * simd_w - 1,
                 4 * simd_w}) {
        ir_kernel_t kernel(build_softmax_tile_ir(1, w));
        ASSERT_TRUE(kernel.run_ir_pipeline()) << "w=" << w;

        const float scale = 0.125f;
        // Two KV tiles of `w` scores, threaded through one running state that
        // starts at the first-tile seed.
        float m = neg_inf, l = 0.f, oc = -12345.f;
        float ref_m = neg_inf, ref_l = 0.f, ref_oc = -12345.f;

        for (int tile = 0; tile < 2; tile++) {
            std::vector<float> scores(w), ref(w);
            for (int j = 0; j < w; j++) {
                scores[j] = (float)((j * 7 + tile * 3) % 13) * 0.5f - 3.f;
                ref[j] = scores[j];
            }
            ref_softmax_row(ref, scale, ref_m, ref_l, ref_oc);

            softmax_row_args_t args {scores.data(), &scale, &m, &l, &oc};
            kernel.run(&args);

            EXPECT_NEAR(m, ref_m, 1e-5f * std::abs(ref_m) + 1e-6f)
                    << "w=" << w << " tile=" << tile;
            EXPECT_NEAR(l, ref_l, 1e-4f * std::abs(ref_l) + 1e-6f)
                    << "w=" << w << " tile=" << tile;
            EXPECT_NEAR(oc, ref_oc, 1e-4f * std::abs(ref_oc) + 1e-6f)
                    << "w=" << w << " tile=" << tile;
            for (int j = 0; j < w; j++)
                EXPECT_NEAR(scores[j], ref[j], 1e-4f * std::abs(ref[j]) + 1e-6f)
                        << "w=" << w << " tile=" << tile << " j=" << j;
        }
    }
}

// Validates the multi-row tile epilogue: one kernel processes seq_q score rows
// with a runtime loop, advancing the score and per-row state pointers each
// iteration. Every row carries its own finite running state and distinct data,
// so a wrong stride would bleed rows into each other. Each row must match an
// independent scalar-reference update.
TEST(IntegrationTests, SoftmaxOnlineTileMultiRow) {
    SKIP_IF_NO_AVX2();

    const float scale = 0.125f;
    for (int seq_q : {2, 3, 5}) {
        for (int w : {1, 7, simd_w, 9, 17, 4 * simd_w - 1, 4 * simd_w}) {
            ir_kernel_t kernel(build_softmax_tile_ir(seq_q, w));
            ASSERT_TRUE(kernel.run_ir_pipeline())
                    << "seq_q=" << seq_q << " w=" << w;

            std::vector<float> scores((size_t)seq_q * w);
            std::vector<float> m(seq_q), l(seq_q), oc(seq_q, -12345.f);
            std::vector<std::vector<float>> ref(seq_q, std::vector<float>(w));
            std::vector<float> ref_m(seq_q), ref_l(seq_q), ref_oc(seq_q);
            for (int i = 0; i < seq_q; i++) {
                for (int j = 0; j < w; j++) {
                    const float v
                            = (float)(((i + 1) * j * 5 + i * 3) % 13) * 0.5f
                            - 3.f;
                    scores[(size_t)i * w + j] = v;
                    ref[i][j] = v;
                }
                // Distinct finite per-row running state.
                m[i] = -1.5f + 0.25f * i;
                l[i] = 2.0f + 0.5f * i;
                ref_m[i] = m[i];
                ref_l[i] = l[i];
                ref_oc[i] = -12345.f;
                ref_softmax_row(ref[i], scale, ref_m[i], ref_l[i], ref_oc[i]);
            }

            softmax_row_args_t args {
                    scores.data(), &scale, m.data(), l.data(), oc.data()};
            kernel.run(&args);

            for (int i = 0; i < seq_q; i++) {
                EXPECT_NEAR(m[i], ref_m[i], 1e-5f * std::abs(ref_m[i]) + 1e-6f)
                        << "seq_q=" << seq_q << " w=" << w << " i=" << i;
                EXPECT_NEAR(l[i], ref_l[i], 1e-4f * std::abs(ref_l[i]) + 1e-6f)
                        << "seq_q=" << seq_q << " w=" << w << " i=" << i;
                EXPECT_NEAR(
                        oc[i], ref_oc[i], 1e-4f * std::abs(ref_oc[i]) + 1e-6f)
                        << "seq_q=" << seq_q << " w=" << w << " i=" << i;
                for (int j = 0; j < w; j++)
                    EXPECT_NEAR(scores[(size_t)i * w + j], ref[i][j],
                            1e-4f * std::abs(ref[i][j]) + 1e-6f)
                            << "seq_q=" << seq_q << " w=" << w << " i=" << i
                            << " j=" << j;
            }
        }
    }
}

// Arguments for the accumulator renormalization epilogue. After the softmax
// tile update produces old_coef per row, the running output accumulator is
// rescaled and the new tile's P*V contribution is added in one pass:
// acc = old_coef*acc + pv, over seq_q rows of hs_v head-size columns. acc and
// pv hold seq_q*hs floats (row i at i*hs); old_coef holds seq_q floats.
struct acc_renorm_args_t {
    float *acc; // in: running output acc; out: renormalized acc
    const float *pv; // in: this tile's P*V contribution
    const float *old_coef; // in: per-row renorm coefficient (corr*l_old/l_new)
};

// Builds the accumulator renormalization for a tile of `seq_q` rows, each of
// head size `hs` (any hs >= 1; the ragged tail beyond the last full simd_w
// block uses masked loads/stores). Mirrors the acc rescale in the fused SDPA
// kernel that follows the softmax epilogue: acc = old_coef*acc + pv. Rows are
// processed by a runtime loop over seq_q (inlined when seq_q == 1); the acc, pv
// and old_coef pointers advance one row per iteration. old_coef is broadcast so
// the per-row scalar reuses the vector ops; no reduction is needed, so the tail
// needs no lane neutralization (masked ld/st touch only the active columns).
ir_t build_acc_renorm_ir(int seq_q, int hs) {
    const int n_blk = hs / simd_w;
    const int tail = hs % simd_w;
    const dim_t vbytes = simd_w * (dim_t)sizeof(float);
    const dim_t tail_off = n_blk * vbytes;
    const dim_t fsz = (dim_t)sizeof(float);

    ir_t ir;

    // Row pointers: advanced one row per loop iteration.
    const vreg_t acc_ptr = ir.new_gpr();
    ir.load_param(acc_ptr, offsetof(acc_renorm_args_t, acc));
    const vreg_t pv_ptr = ir.new_gpr();
    ir.load_param(pv_ptr, offsetof(acc_renorm_args_t, pv));
    const vreg_t oc_ptr = ir.new_gpr();
    ir.load_param(oc_ptr, offsetof(acc_renorm_args_t, old_coef));

    // One mask, reused by every masked op and every row, active for `tail`.
    vreg_t mask = vreg_t::none;
    if (tail) {
        mask = ir.new_mask();
        ir.set_mask_imm(mask, tail);
    }

    // acc = old_coef*acc + pv for the single row at the current pointers.
    auto row_body = [&]() {
        const vreg_t oc_bc = ir.new_vec(data_type::f32);
        ir.vload_masked(oc_bc, oc_ptr, 0, vreg_t::none, 1);
        ir.vbcast(oc_bc, oc_bc);

        for (int b = 0; b < n_blk; b++) {
            const vreg_t acc = ir.new_vec(data_type::f32);
            ir.vload(acc, acc_ptr, b * vbytes);
            ir.vmul(acc, oc_bc);
            const vreg_t pv = ir.new_vec(data_type::f32);
            ir.vload(pv, pv_ptr, b * vbytes);
            ir.vadd(acc, pv);
            ir.vstore_masked(acc_ptr, b * vbytes, acc, vreg_t::none, simd_w);
        }
        if (tail) {
            const vreg_t acc = ir.new_vec(data_type::f32);
            ir.vload_masked(acc, acc_ptr, tail_off, mask, tail);
            ir.vmul(acc, oc_bc);
            const vreg_t pv = ir.new_vec(data_type::f32);
            ir.vload_masked(pv, pv_ptr, tail_off, mask, tail);
            ir.vadd(acc, pv);
            ir.vstore_masked(acc_ptr, tail_off, acc, mask, tail);
        }
    };

    // Advance every row pointer to the next row.
    auto advance_row = [&]() {
        ir.add_imm(acc_ptr, hs * fsz);
        ir.add_imm(pv_ptr, hs * fsz);
        ir.add_imm(oc_ptr, fsz);
    };

    emit_loop_imm(ir, seq_q, row_body, advance_row);

    return ir;
}

// Scalar reference for one accumulator renormalization row, matching one row of
// build_acc_renorm_ir.
void ref_acc_renorm(
        std::vector<float> &acc, const std::vector<float> &pv, float old_coef) {
    for (size_t d = 0; d < acc.size(); d++)
        acc[d] = old_coef * acc[d] + pv[d];
}

// Validates the accumulator renormalization tile epilogue: one kernel rescales
// seq_q accumulator rows (acc = old_coef*acc + pv) with a runtime loop,
// advancing the acc/pv/old_coef pointers each iteration. Every row carries a
// distinct old_coef and distinct data, so a wrong stride would bleed rows into
// each other. Each row must match an independent scalar-reference update.
TEST(IntegrationTests, AccRenormTile) {
    SKIP_IF_NO_AVX2();

    for (int seq_q : {1, 2, 3, 5}) {
        // Head sizes span pure tails, exact multiples, and multiples plus a
        // ragged tail, so the masked-tail path runs.
        for (int hs : {1, 7, simd_w, 9, 17, 4 * simd_w - 1, 4 * simd_w}) {
            ir_kernel_t kernel(build_acc_renorm_ir(seq_q, hs));
            ASSERT_TRUE(kernel.run_ir_pipeline())
                    << "seq_q=" << seq_q << " hs=" << hs;

            std::vector<float> acc((size_t)seq_q * hs);
            std::vector<float> pv((size_t)seq_q * hs);
            std::vector<float> oc(seq_q);
            std::vector<std::vector<float>> ref(seq_q);
            for (int i = 0; i < seq_q; i++) {
                ref[i].resize(hs);
                for (int j = 0; j < hs; j++) {
                    const float a
                            = (float)(((i + 1) * j * 3 + i * 2) % 11) * 0.5f
                            - 2.f;
                    const float p
                            = (float)(((i + 2) * j * 7 + i) % 13) * 0.25f - 1.f;
                    acc[(size_t)i * hs + j] = a;
                    pv[(size_t)i * hs + j] = p;
                    ref[i][j] = a;
                }
                // Distinct per-row renorm coefficient.
                oc[i] = 0.3f + 0.2f * i;
                std::vector<float> pv_row(pv.begin() + (size_t)i * hs,
                        pv.begin() + (size_t)(i + 1) * hs);
                ref_acc_renorm(ref[i], pv_row, oc[i]);
            }

            acc_renorm_args_t args {acc.data(), pv.data(), oc.data()};
            kernel.run(&args);

            for (int i = 0; i < seq_q; i++)
                for (int j = 0; j < hs; j++)
                    EXPECT_NEAR(acc[(size_t)i * hs + j], ref[i][j],
                            1e-5f * std::abs(ref[i][j]) + 1e-6f)
                            << "seq_q=" << seq_q << " hs=" << hs << " i=" << i
                            << " j=" << j;
        }
    }
}

} // namespace dnnl
