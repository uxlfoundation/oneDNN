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
#include <vector>

#include "gtest/gtest.h"

#include "common/c_types_map.hpp"

#include "cpu/x64/ir/ir.hpp"

namespace dnnl {

using namespace impl;
using namespace impl::cpu::x64;
using namespace impl::cpu::x64::ir;

// Helpers shared by the tests

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
}

} // namespace dnnl
