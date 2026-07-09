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

#include "cpu/x64/ir/ir.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

vreg_t ir_t::new_vreg(reg_kind_t k, data_type_t dt) {
    vreg_info_t info {};
    info.kind = k;
    info.dt = dt;
    vreg_info_.push_back(info);
    return static_cast<vreg_t>((int)vreg_info_.size() - 1);
}

int ir_t::mov_imm(vreg_t dst, dim_t imm) {
    op_t op;
    op.kind = op_kind_t::mov_imm;
    op.dst = dst;
    op.imm = imm;
    ops_.push_back(op);
    return (int)ops_.size() - 1;
}

void ir_t::mov_reg(vreg_t dst, vreg_t src) {
    op_t op;
    op.kind = op_kind_t::mov_reg;
    op.dst = dst;
    op.s0 = src;
    ops_.push_back(op);
}

void ir_t::add_imm(vreg_t dst, dim_t imm) {
    op_t op;
    op.kind = op_kind_t::add_imm;
    op.dst = dst;
    op.imm = imm;
    ops_.push_back(op);
}

void ir_t::add_reg(vreg_t dst, vreg_t src) {
    op_t op;
    op.kind = op_kind_t::add_reg;
    op.dst = dst;
    op.s0 = src;
    ops_.push_back(op);
}

void ir_t::load_param(vreg_t dst, dim_t disp) {
    op_t op;
    op.kind = op_kind_t::load;
    op.dst = dst;
    op.mem.is_param = true;
    op.mem.disp = disp;
    ops_.push_back(op);
}

void ir_t::load(vreg_t dst, vreg_t base, dim_t disp) {
    op_t op;
    op.kind = op_kind_t::load;
    op.dst = dst;
    op.mem.base = base;
    op.mem.disp = disp;
    ops_.push_back(op);
}

void ir_t::vzero(vreg_t dst) {
    op_t op;
    op.kind = op_kind_t::vzero;
    op.dst = dst;
    ops_.push_back(op);
}

void ir_t::vload(vreg_t dst, vreg_t base, dim_t disp) {
    op_t op;
    op.kind = op_kind_t::vload;
    op.dst = dst;
    op.mem.base = base;
    op.mem.disp = disp;
    ops_.push_back(op);
}

void ir_t::vfma(vreg_t dst, vreg_t a, vreg_t b) {
    op_t op;
    op.kind = op_kind_t::vfma;
    op.dst = dst;
    op.s0 = a;
    op.s1 = b;
    ops_.push_back(op);
}

void ir_t::vhreduce(vreg_t dst, vreg_t workspace) {
    op_t op;
    op.kind = op_kind_t::vhreduce;
    op.dst = dst;
    op.s0 = workspace;
    ops_.push_back(op);
}

void ir_t::set_mask_imm(vreg_t mask, int n_elems) {
    op_t op;
    op.kind = op_kind_t::set_mask_imm;
    op.dst = mask;
    op.imm = n_elems;
    ops_.push_back(op);
}

void ir_t::vload_masked(
        vreg_t dst, vreg_t base, dim_t disp, vreg_t mask, int elems) {
    op_t op;
    op.kind = op_kind_t::vload_masked;
    op.dst = dst;
    // `none` when no mask register is needed
    op.s1 = mask;
    op.imm = elems;
    op.mem.base = base;
    op.mem.disp = disp;
    ops_.push_back(op);
}

void ir_t::vstore_masked(
        vreg_t base, dim_t disp, vreg_t src, vreg_t mask, int elems) {
    op_t op;
    op.kind = op_kind_t::vstore_masked;
    op.s0 = src;
    // `none` when no mask register is needed
    op.s1 = mask;
    op.imm = elems;
    op.mem.base = base;
    op.mem.disp = disp;
    ops_.push_back(op);
}

int ir_t::loop_begin_imm(vreg_t counter, dim_t count) {
    op_t op;
    op.kind = op_kind_t::loop_begin;
    op.dst = counter;
    op.imm = count;
    ops_.push_back(op);
    return (int)ops_.size() - 1;
}

int ir_t::loop_begin_reg(vreg_t counter, vreg_t init) {
    op_t op;
    op.kind = op_kind_t::loop_begin;
    op.dst = counter;
    op.s0 = init;
    op.init_is_reg = true;
    ops_.push_back(op);
    return (int)ops_.size() - 1;
}

void ir_t::loop_end(vreg_t counter, int begin_idx) {
    op_t op;
    op.kind = op_kind_t::loop_end;
    op.dst = counter;
    op.match = begin_idx;
    ops_.push_back(op);
}

void ir_t::label(label_t label_id) {
    op_t op;
    op.kind = op_kind_t::label;
    op.label_id = label_id;
    ops_.push_back(op);
}

void ir_t::jmp(label_t label_id) {
    op_t op;
    op.kind = op_kind_t::jmp;
    op.label_id = label_id;
    ops_.push_back(op);
}

void ir_t::jz(vreg_t cond, label_t label_id) {
    op_t op;
    op.kind = op_kind_t::jz;
    op.s0 = cond;
    op.label_id = label_id;
    ops_.push_back(op);
}

// For each operation, we record which virtual registers it reads (uses)
// and which ones it writes (defs). This info is the basis for liveness
// analysis, so it must accurately reflect what the operation really does.
//
// Some tricky cases:
// - Operations that both read and write the same register (like add_imm,
//   vfma) count as both a use and a def, because they read the old value
//   and then overwrite it.
// - vhreduce uses its temporary register (s0) as both read and written,
//   so the register allocator keeps it separate from the accumulator.
// - A base register used for memory access counts as a read (use),
//   unless it's a fixed parameter register.
// - Control operations like loop_end both read and write the loop counter.
void ir_t::def_use(
        const op_t &op, std::vector<int> &defs, std::vector<int> &uses) const {
    defs.clear();
    uses.clear();

    auto u = [&](vreg_t v) {
        if (v != vreg_t::none) uses.push_back((int)v);
    };
    auto d = [&](vreg_t v) {
        if (v != vreg_t::none) defs.push_back((int)v);
    };

    switch (op.kind) {
        case op_kind_t::mov_imm: d(op.dst); break;
        case op_kind_t::mov_reg:
            d(op.dst);
            u(op.s0);
            break;
        case op_kind_t::add_imm: // read-modify-write
            u(op.dst);
            d(op.dst);
            break;
        case op_kind_t::add_reg:
            u(op.dst);
            u(op.s0);
            d(op.dst);
            break;
        case op_kind_t::load:
            if (!op.mem.is_param) u(op.mem.base);
            d(op.dst);
            break;
        case op_kind_t::vzero: d(op.dst); break;
        case op_kind_t::vload:
            u(op.mem.base);
            d(op.dst);
            break;
        case op_kind_t::vfma:
            u(op.dst);
            u(op.s0);
            u(op.s1);
            d(op.dst);
            break;
        case op_kind_t::vhreduce: // dst and workspace are both read and written
            u(op.dst);
            u(op.s0);
            d(op.dst);
            d(op.s0);
            break;
        case op_kind_t::set_mask_imm: d(op.dst); break;
        case op_kind_t::vload_masked:
            u(op.mem.base);
            u(op.s1); // mask (-1 -> not counted, dropped by u())
            d(op.dst);
            break;
        case op_kind_t::vstore_masked:
            u(op.s0);
            u(op.s1); // mask (-1 -> not counted, dropped by u())
            u(op.mem.base);
            break;
        case op_kind_t::loop_begin:
            if (op.init_is_reg) u(op.s0);
            d(op.dst);
            break;
        case op_kind_t::loop_end:
            u(op.dst);
            d(op.dst);
            break;
        case op_kind_t::label:
        case op_kind_t::jmp: break;
        case op_kind_t::jz: u(op.s0); break;
    }
}

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
