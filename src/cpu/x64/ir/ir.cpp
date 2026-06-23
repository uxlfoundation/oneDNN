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

int ir_t::new_vreg(reg_kind_t k, data_type_t dt) {
    vreg_info_t info {};
    info.kind = k;
    info.dt = dt;
    vreg_info.push_back(info);
    return (int)vreg_info.size() - 1;
}

int ir_t::mov_imm(int dst, dim_t imm) {
    instr_t in;
    in.op = op_kind_t::mov_imm;
    in.dst = dst;
    in.imm = imm;
    instrs.push_back(in);
    return (int)instrs.size() - 1;
}

void ir_t::mov_reg(int dst, int src) {
    instr_t in;
    in.op = op_kind_t::mov_reg;
    in.dst = dst;
    in.s0 = src;
    instrs.push_back(in);
}

void ir_t::add_imm(int dst, dim_t imm) {
    instr_t in;
    in.op = op_kind_t::add_imm;
    in.dst = dst;
    in.imm = imm;
    instrs.push_back(in);
}

void ir_t::add_reg(int dst, int src) {
    instr_t in;
    in.op = op_kind_t::add_reg;
    in.dst = dst;
    in.s0 = src;
    instrs.push_back(in);
}

void ir_t::load_param(int dst, dim_t disp) {
    instr_t in;
    in.op = op_kind_t::load;
    in.dst = dst;
    in.mem.is_param = true;
    in.mem.disp = disp;
    instrs.push_back(in);
}

void ir_t::load(int dst, int base, dim_t disp) {
    instr_t in;
    in.op = op_kind_t::load;
    in.dst = dst;
    in.mem.base = base;
    in.mem.disp = disp;
    instrs.push_back(in);
}

void ir_t::vzero(int dst) {
    instr_t in;
    in.op = op_kind_t::vzero;
    in.dst = dst;
    instrs.push_back(in);
}

void ir_t::vload(int dst, int base, dim_t disp) {
    instr_t in;
    in.op = op_kind_t::vload;
    in.dst = dst;
    in.mem.base = base;
    in.mem.disp = disp;
    instrs.push_back(in);
}

void ir_t::vfma(int dst, int a, int b) {
    instr_t in;
    in.op = op_kind_t::vfma;
    in.dst = dst;
    in.s0 = a;
    in.s1 = b;
    instrs.push_back(in);
}

void ir_t::vhreduce(int dst, int workspace) {
    instr_t in;
    in.op = op_kind_t::vhreduce;
    in.dst = dst;
    in.s0 = workspace;
    instrs.push_back(in);
}

void ir_t::set_mask_imm(int mask, int n_elems) {
    instr_t in;
    in.op = op_kind_t::set_mask_imm;
    in.dst = mask;
    in.imm = n_elems;
    instrs.push_back(in);
}

void ir_t::vload_masked(int dst, int base, dim_t disp, int mask, int elems) {
    instr_t in;
    in.op = op_kind_t::vload_masked;
    in.dst = dst;
    // -1 when no mask register is needed
    in.s1 = mask;
    in.imm = elems;
    in.mem.base = base;
    in.mem.disp = disp;
    instrs.push_back(in);
}

void ir_t::vstore_masked(int base, dim_t disp, int src, int mask, int elems) {
    instr_t in;
    in.op = op_kind_t::vstore_masked;
    in.s0 = src;
    // -1 when no mask register is needed
    in.s1 = mask;
    in.imm = elems;
    in.mem.base = base;
    in.mem.disp = disp;
    instrs.push_back(in);
}

int ir_t::loop_begin_imm(int counter, dim_t count) {
    instr_t in;
    in.op = op_kind_t::loop_begin;
    in.dst = counter;
    in.imm = count;
    instrs.push_back(in);
    return (int)instrs.size() - 1;
}

int ir_t::loop_begin_reg(int counter, int init) {
    instr_t in;
    in.op = op_kind_t::loop_begin;
    in.dst = counter;
    in.s0 = init;
    in.init_is_reg = true;
    instrs.push_back(in);
    return (int)instrs.size() - 1;
}

void ir_t::loop_end(int counter, int begin_idx) {
    instr_t in;
    in.op = op_kind_t::loop_end;
    in.dst = counter;
    in.match = begin_idx;
    instrs.push_back(in);
}

void ir_t::jmp(int label_id) {
    instr_t in;
    in.op = op_kind_t::jmp;
    in.imm = label_id;
    instrs.push_back(in);
}

void ir_t::jz(int cond, int label_id) {
    instr_t in;
    in.op = op_kind_t::jz;
    in.s0 = cond;
    in.imm = label_id;
    instrs.push_back(in);
}

// For each instruction, we record which virtual registers it reads (uses)
// and which ones it writes (defs). This info is the basis for liveness
// analysis, so it must accurately reflect what the instruction really does.
//
// Some tricky cases:
// - Instructions that both read and write the same register (like add_imm,
//   vfma) count as both a use and a def, because they read the old value
//   and then overwrite it.
// - vhreduce uses its temporary register (s0) as both read and written,
//   so the register allocator keeps it separate from the accumulator.
// - A base register used for memory access counts as a read (use),
//   unless it's a fixed parameter register.
// - Control instructions like loop_end both read and write the loop counter.
void ir_t::def_use(const instr_t &in, std::vector<int> &defs,
        std::vector<int> &uses) const {
    defs.clear();
    uses.clear();

    auto u = [&](int v) {
        if (v >= 0) uses.push_back(v);
    };
    auto d = [&](int v) {
        if (v >= 0) defs.push_back(v);
    };

    switch (in.op) {
        case op_kind_t::mov_imm: d(in.dst); break;
        case op_kind_t::mov_reg:
            d(in.dst);
            u(in.s0);
            break;
        case op_kind_t::add_imm: // read-modify-write
            u(in.dst);
            d(in.dst);
            break;
        case op_kind_t::add_reg:
            u(in.dst);
            u(in.s0);
            d(in.dst);
            break;
        case op_kind_t::load:
            if (!in.mem.is_param) u(in.mem.base);
            d(in.dst);
            break;
        case op_kind_t::vzero: d(in.dst); break;
        case op_kind_t::vload:
            u(in.mem.base);
            d(in.dst);
            break;
        case op_kind_t::vfma:
            u(in.dst);
            u(in.s0);
            u(in.s1);
            d(in.dst);
            break;
        case op_kind_t::vhreduce: // dst and workspace are both read and written
            u(in.dst);
            u(in.s0);
            d(in.dst);
            d(in.s0);
            break;
        case op_kind_t::set_mask_imm: d(in.dst); break;
        case op_kind_t::vload_masked:
            u(in.mem.base);
            u(in.s1); // mask (-1 -> not counted, dropped by u())
            d(in.dst);
            break;
        case op_kind_t::vstore_masked:
            u(in.s0);
            u(in.s1); // mask (-1 -> not counted, dropped by u())
            u(in.mem.base);
            break;
        case op_kind_t::loop_begin:
            if (in.init_is_reg) u(in.s0);
            d(in.dst);
            break;
        case op_kind_t::loop_end:
            u(in.dst);
            d(in.dst);
            break;
        case op_kind_t::label:
        case op_kind_t::jmp: break;
        case op_kind_t::jz: u(in.s0); break;
    }
}

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
