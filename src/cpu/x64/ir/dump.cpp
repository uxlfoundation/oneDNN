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
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <vector>

#include "oneapi/dnnl/dnnl_debug.h"

#include "common/utils.hpp"

#include "cpu/x64/ir/dump.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

namespace {

// Returns the name of an operation kind, which is its `op_kind_t` enumerator
// name. The switch has no `default`, so a new kind triggers a `-Wswitch`
// warning until it is handled here.
const char *kind_name(op_kind_t kind) {
    switch (kind) {
        case op_kind_t::mov_imm: return "mov_imm";
        case op_kind_t::mov_reg: return "mov_reg";
        case op_kind_t::add_imm: return "add_imm";
        case op_kind_t::add_reg: return "add_reg";
        case op_kind_t::load: return "load";
        case op_kind_t::vzero: return "vzero";
        case op_kind_t::vload: return "vload";
        case op_kind_t::vstore: return "vstore";
        case op_kind_t::vload_scalar: return "vload_scalar";
        case op_kind_t::vstore_scalar: return "vstore_scalar";
        case op_kind_t::vload_bcast: return "vload_bcast";
        case op_kind_t::vdot: return "vdot";
        case op_kind_t::vadd: return "vadd";
        case op_kind_t::vmul: return "vmul";
        case op_kind_t::vhreduce: return "vhreduce";
        case op_kind_t::inject_postops: return "inject_postops";
        case op_kind_t::set_mask_imm: return "set_mask_imm";
        case op_kind_t::vload_masked: return "vload_masked";
        case op_kind_t::vstore_masked: return "vstore_masked";
        case op_kind_t::prefetch: return "prefetch";
        case op_kind_t::loop_begin: return "loop_begin";
        case op_kind_t::loop_end: return "loop_end";
        case op_kind_t::label: return "label";
        case op_kind_t::jmp: return "jmp";
        case op_kind_t::jz: return "jz";
    }
    assert(!"unknown op kind");
    return "?";
}

bool is_valid(const ir_t &ir, vreg_t v) {
    return (int)v >= 0 && (int)v < ir.n_vregs();
}

// Returns `r<id>`, `<dt>:v<id>`, or `m<id>` by the kind of `v`. A vec vreg
// carries its data type, so the reader does not have to look it up. An id
// outside the IR prints as `?<id>` rather than failing, because the IR dump is
// most useful on an IR that is suspected to be wrong.
std::string vreg_str(const ir_t &ir, vreg_t v) {
    if (v == vreg_t::none) return "none";
    const std::string id = std::to_string((int)v);
    if (!is_valid(ir, v)) return "?" + id;
    switch (ir.vreg_info()[(int)v].kind) {
        case reg_kind_t::gpr: return "r" + id;
        case reg_kind_t::vec:
            return std::string(dnnl_dt2str(ir.vreg_info()[(int)v].dt)) + ":v"
                    + id;
        case reg_kind_t::mask: return "m" + id;
    }
    assert(!"unknown reg kind");
    return "?" + id;
}

std::string label_str(label_t l) {
    return "L" + std::to_string((int)l);
}

// Returns the memory operand of `op`. `data` is the vec vreg the access moves,
// or `none` for an access that moves no vec value (`load`, `prefetch`). A
// vector access is prefixed with the data type in memory, the same way a vec
// vreg is prefixed with its data type. A converting access therefore shows two
// different types (`vload f32:v3, bf16:[r0+0]`).
std::string mem_str(const ir_t &ir, const op_t &op, vreg_t data) {
    std::string s;
    if (data != vreg_t::none && op.mem_dt != data_type::undef) {
        s += dnnl_dt2str(op.mem_dt);
        s += ":";
    }
    s += "[";
    s += op.mem.is_param ? "param" : vreg_str(ir, op.mem.base);
    if (op.mem.disp >= 0) s += "+";
    s += std::to_string(op.mem.disp);
    s += "]";
    return s;
}

// Returns the operands of an `inject_postops` op from its side-table entry:
// the accumulators, then `base=` unless the base pointer is `none`, then
// `off=` unless the offsets are empty.
std::string postops_str(const ir_t &ir, const op_t &op) {
    const auto &table = ir.inject_postops_args();
    if (op.imm < 0 || op.imm >= (dim_t)table.size())
        return "args=?" + std::to_string(op.imm);
    const inject_postops_args_t &args = table[(int)op.imm];

    std::string s;
    for (size_t i = 0; i < args.acc.size(); i++) {
        if (i > 0) s += ", ";
        s += vreg_str(ir, args.acc[i]);
    }
    if (args.base_ptr != vreg_t::none)
        s += ", base=" + vreg_str(ir, args.base_ptr);
    if (!args.out_byte_off.empty()) {
        s += ", off=[";
        for (size_t i = 0; i < args.out_byte_off.size(); i++) {
            if (i > 0) s += ", ";
            s += std::to_string(args.out_byte_off[i]);
        }
        s += "]";
    }
    return s;
}

// Returns the text of one operation, without index or indentation. Every kind
// prints as its name followed by its operands, except the control-flow
// markers. A loop prints as an opening and a closing brace, and a label prints
// as `L<id>:`.
std::string op_str(const ir_t &ir, const op_t &op) {
    const auto r = [&](vreg_t v) { return vreg_str(ir, v); };
    const auto mem = [&](vreg_t data) { return mem_str(ir, op, data); };
    const std::string k = std::string(kind_name(op.kind)) + " ";

    switch (op.kind) {
        case op_kind_t::mov_imm:
        case op_kind_t::add_imm:
        case op_kind_t::set_mask_imm:
            return k + r(op.dst) + ", " + std::to_string(op.imm);
        case op_kind_t::mov_reg:
        case op_kind_t::add_reg:
        case op_kind_t::vadd:
        case op_kind_t::vmul:
        case op_kind_t::vhreduce: return k + r(op.dst) + ", " + r(op.s0);
        case op_kind_t::load: return k + r(op.dst) + ", " + mem(vreg_t::none);
        case op_kind_t::vzero: return k + r(op.dst);
        case op_kind_t::vload:
        case op_kind_t::vload_scalar:
        case op_kind_t::vload_bcast: return k + r(op.dst) + ", " + mem(op.dst);
        case op_kind_t::vstore:
        case op_kind_t::vstore_scalar: return k + mem(op.s0) + ", " + r(op.s0);
        case op_kind_t::vdot:
            return k + r(op.dst) + ", " + r(op.s0) + ", " + r(op.s1);
        case op_kind_t::inject_postops: return k + postops_str(ir, op);
        case op_kind_t::vload_masked:
            return k + r(op.dst) + ", " + mem(op.dst) + ", " + r(op.s1);
        case op_kind_t::vstore_masked:
            return k + mem(op.s0) + ", " + r(op.s0) + ", " + r(op.s1);
        case op_kind_t::prefetch: return k + mem(vreg_t::none);
        case op_kind_t::loop_begin:
            return "loop " + r(op.dst) + " = "
                    + (op.init_is_reg ? r(op.s0) : std::to_string(op.imm))
                    + " {";
        case op_kind_t::loop_end:
            return "} // " + r(op.dst) + " -= 1, repeat while > 0";
        case op_kind_t::label: return label_str(op.label_id) + ":";
        case op_kind_t::jmp: return k + label_str(op.label_id);
        case op_kind_t::jz: return k + r(op.s0) + ", " + label_str(op.label_id);
    }
    assert(!"unknown op kind");
    return k;
}

// Returns the line with the IR counts. The nesting depth is the largest number
// of loops around one operation, counted the same way as `compute_loop_depth()`
// in the allocator. Branches are the `jz` and `jmp` operations. Loops are not
// counted as branches.
std::string counts_line(const ir_t &ir) {
    int n_kind[3] = {0, 0, 0};
    for (const vreg_info_t &info : ir.vreg_info())
        n_kind[(int)info.kind]++;

    int n_loops = 0, n_branches = 0, depth = 0, max_depth = 0;
    for (const op_t &op : ir.ops()) {
        if (op.kind == op_kind_t::loop_begin) {
            n_loops++;
            max_depth = std::max(max_depth, ++depth);
        } else if (op.kind == op_kind_t::loop_end) {
            depth--;
        } else if (op.kind == op_kind_t::jz || op.kind == op_kind_t::jmp) {
            n_branches++;
        }
    }

    ostringstream_t ss;
    ss << "ir: " << ir.n_ops() << " ops, " << ir.n_vregs() << " vregs (gpr "
       << n_kind[(int)reg_kind_t::gpr] << ", vec "
       << n_kind[(int)reg_kind_t::vec] << ", mask "
       << n_kind[(int)reg_kind_t::mask] << "), " << n_loops
       << " loops, nesting depth " << max_depth << ", " << n_branches
       << " branches\n";
    return ss.str();
}

} // namespace

int parse_x64ir_level(const std::string &verbose_value) {
    // Tokens are split on `,` with no trimming, the same way the rest of
    // `ONEDNN_VERBOSE` is parsed.
    const char key[] = "x64ir=";
    const size_t key_len = sizeof(key) - 1;

    int level = 0;
    size_t pos = 0;
    while (true) {
        const size_t end = verbose_value.find(',', pos);
        const std::string tok = verbose_value.substr(pos, end - pos);
        if (tok.rfind(key, 0) == 0)
            level = std::max(0, std::atoi(tok.c_str() + key_len));
        if (end == std::string::npos) break;
        pos = end + 1;
    }
    return level;
}

int verbose_level() {
    if (!is_dev_mode()) return 0;
    // `getenv_string_user()` lowercases the value, so `X64IR=` works too.
    static const int level = parse_x64ir_level(getenv_string_user("VERBOSE"));
    return level;
}

std::string to_string(const ir_t &ir) {
    std::string s;

    // Indentation follows the loop nesting. A loop's closing line is indented
    // like its opening line, so the body stands out between the two.
    int depth = 0;
    for (int i = 0; i < ir.n_ops(); i++) {
        const op_t &op = ir.ops()[i];
        if (op.kind == op_kind_t::loop_end) depth--;

        ostringstream_t ss;
        ss << std::setw(5) << i << " | ";
        s += ss.str();
        s += std::string(2 * std::max(depth, 0), ' ');
        s += op_str(ir, op);
        s += "\n";

        if (op.kind == op_kind_t::loop_begin) depth++;
    }
    return s;
}

std::string format_kernel_dump(
        int level, int seq, const kernel_info_t &info, const ir_t &ir) {
    ostringstream_t ss;
    ss << "begin x64ir #" << seq << " " << info.name
       << " isa=" << isa2str(info.isa) << "\n";
    ss << counts_line(ir);
    ss << "code: " << info.code_size << " bytes (instructions "
       << info.code_size - info.data_size << ", static data " << info.data_size
       << ")\n";
    if (level >= 3) ss << to_string(ir);
    ss << "end x64ir #" << seq << "\n";
    return ss.str();
}

void print_kernel_dump(const jit_generator_t &gen, const ir_t &ir,
        const data_section_t &data) {
    const int level = verbose_level();
    if (level == 0) return;

    kernel_info_t info;
    info.name = gen.name();
    info.isa = gen.max_cpu_isa();
    info.code_size = gen.getSize();
    info.data_size = info.code_size - data.begin_offset;

    static std::atomic<int> seq {0};
    const std::string s = format_kernel_dump(level, ++seq, info, ir);
    // One stdio call per kernel. POSIX locks the stream for the call, so the
    // output of kernels created at the same time does not mix.
    printf("%s", s.c_str());
    fflush(stdout);
}

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
