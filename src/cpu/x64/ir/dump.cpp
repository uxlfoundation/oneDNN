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

const char *reg_kind_name(reg_kind_t kind) {
    switch (kind) {
        case reg_kind_t::gpr: return "gpr";
        case reg_kind_t::vec: return "vec";
        case reg_kind_t::mask: return "mask";
    }
    assert(!"unknown reg kind");
    return "?";
}

bool is_valid(const ir_t &ir, vreg_t v) {
    return (int)v >= 0 && (int)v < ir.n_vregs();
}

// Returns the name of the physical register `phys` for a vreg of kind `kind`.
// The register type is the one the emitter uses. The AVX-512 backend keeps a
// vec in a `zmm` and a mask in a `k` register. The AVX2 backend keeps both in a
// `ymm`.
std::string phys_str(cpu_isa_t isa, reg_kind_t kind, int phys) {
    const bool is_avx512 = is_superset(isa, avx512_core);
    switch (kind) {
        case reg_kind_t::gpr: return Xbyak::Reg64(phys).toString();
        case reg_kind_t::vec:
            return is_avx512 ? Xbyak::Zmm(phys).toString()
                             : Xbyak::Ymm(phys).toString();
        case reg_kind_t::mask:
            return is_avx512 ? Xbyak::Opmask(phys).toString()
                             : Xbyak::Ymm(phys).toString();
    }
    assert(!"unknown reg kind");
    return "?";
}

// Returns where the allocation in `info` keeps `v`: a physical register, or a
// stack slot `[rsp+<off>]`, the address that the emitter uses.
std::string location_str(const ir_t &ir, const kernel_info_t &info, vreg_t v) {
    const assignment_t &a = info.alloc->assignments[(int)v];
    if (a.spilled) return "[rsp+" + std::to_string(a.slot) + "]";
    if (a.phys < 0) return "?";
    return phys_str(info.isa, ir.vreg_info()[(int)v].kind, a.phys);
}

// Returns `r<id>`, `<dt>:v<id>`, or `m<id>` by the kind of `v`. A vec vreg
// carries its data type, so the reader does not have to look it up. An id
// outside the IR prints as `?<id>` rather than failing, because the IR dump is
// most useful on an IR that is suspected to be wrong. When `info` is not null,
// the location of `v` follows after `@`.
std::string vreg_str(
        const ir_t &ir, vreg_t v, const kernel_info_t *info = nullptr) {
    if (v == vreg_t::none) return "none";
    const std::string id = std::to_string((int)v);
    if (!is_valid(ir, v)) return "?" + id;
    const std::string at = info ? "@" + location_str(ir, *info, v) : "";
    switch (ir.vreg_info()[(int)v].kind) {
        case reg_kind_t::gpr: return "r" + id + at;
        case reg_kind_t::vec:
            return std::string(dnnl_dt2str(ir.vreg_info()[(int)v].dt)) + ":v"
                    + id + at;
        case reg_kind_t::mask: return "m" + id + at;
    }
    assert(!"unknown reg kind");
    return "?" + id;
}

// Returns the name of register file `f`: the kinds that allocate from it,
// joined with `+`, for example, `gpr` or `vec+mask`.
std::string file_name(const reg_pools_t &pools, int f) {
    std::string s;
    for (int k = 0; k < (int)pools.kind_to_file.size(); k++) {
        if (pools.kind_to_file[k] != f) continue;
        if (!s.empty()) s += "+";
        s += reg_kind_name((reg_kind_t)k);
    }
    return s.empty() ? "file" + std::to_string(f) : s;
}

std::string label_str(label_t l) {
    return "L" + std::to_string((int)l);
}

// Returns the memory operand of `op`. `data` is the vec vreg the access moves,
// or `none` for an access that moves no vec value (`load`, `prefetch`). A
// vector access is prefixed with the data type in memory, the same way a vec
// vreg is prefixed with its data type. A converting access therefore shows two
// different types (`vload f32:v3, bf16:[r0+0]`).
std::string mem_str(const ir_t &ir, const op_t &op, vreg_t data,
        const kernel_info_t *info) {
    std::string s;
    if (data != vreg_t::none && op.mem_dt != data_type::undef) {
        s += dnnl_dt2str(op.mem_dt);
        s += ":";
    }
    s += "[";
    s += op.mem.is_param ? "param" : vreg_str(ir, op.mem.base, info);
    if (op.mem.disp >= 0) s += "+";
    s += std::to_string(op.mem.disp);
    s += "]";
    return s;
}

// Returns the operands of an `inject_postops` op from its side-table entry:
// the accumulators, then `base=` unless the base pointer is `none`, then
// `off=` unless the offsets are empty.
std::string postops_str(
        const ir_t &ir, const op_t &op, const kernel_info_t *info) {
    const auto &table = ir.inject_postops_args();
    if (op.imm < 0 || op.imm >= (dim_t)table.size())
        return "args=?" + std::to_string(op.imm);
    const inject_postops_args_t &args = table[(int)op.imm];

    std::string s;
    for (size_t i = 0; i < args.acc.size(); i++) {
        if (i > 0) s += ", ";
        s += vreg_str(ir, args.acc[i], info);
    }
    if (args.base_ptr != vreg_t::none)
        s += ", base=" + vreg_str(ir, args.base_ptr, info);
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
// as `L<id>:`. When `info` is not null, each vreg shows its location.
std::string op_str(const ir_t &ir, const op_t &op, const kernel_info_t *info) {
    const auto r = [&](vreg_t v) { return vreg_str(ir, v, info); };
    const auto mem = [&](vreg_t data) { return mem_str(ir, op, data, info); };
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
        case op_kind_t::inject_postops: return k + postops_str(ir, op, info);
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

// Returns `<value> at op <index>` for the first largest value in `per_op`, or
// `0` when every value is 0.
std::string max_at_str(const std::vector<int> &per_op) {
    const auto it = std::max_element(per_op.begin(), per_op.end());
    if (it == per_op.end() || *it == 0) return "0";
    return std::to_string(*it) + " at op "
            + std::to_string((int)(it - per_op.begin()));
}

// Returns the number of scratch registers that the emitter reserves in file
// `f` for spill code. The gpr scratch registers belong to the file of the gpr
// kind, and the vec scratch registers belong to the file of the vec kind.
int n_scratch(const reg_config_t &rc, int f) {
    int n = 0;
    if (rc.pools.kind_to_file[(int)reg_kind_t::gpr] == f)
        n += (int)rc.gpr_scratch.size();
    if (rc.pools.kind_to_file[(int)reg_kind_t::vec] == f)
        n += (int)rc.vec_scratch.size();
    return n;
}

// Returns the vregs that allocate from file `f`.
std::vector<int> file_vregs(const ir_t &ir, const reg_pools_t &pools, int f) {
    std::vector<int> vregs;
    for (int v = 0; v < ir.n_vregs(); v++)
        if (pools.kind_to_file[(int)ir.vreg_info()[v].kind] == f)
            vregs.push_back(v);
    return vregs;
}

// Returns the level-1 lines of the register allocation: one line per register
// file, then one line for the whole allocation.
//
// The peak is the largest register pressure of the file. The scratch registers
// are used only by spill code, so they are unused when nothing in the file
// spills.
std::string alloc_lines(const ir_t &ir, const kernel_info_t &info) {
    const reg_pools_t &pools = info.reg_cfg->pools;
    ostringstream_t ss;
    for (int f = 0; f < (int)pools.files.size(); f++) {
        const std::vector<int> vregs = file_vregs(ir, pools, f);
        int n_spilled = 0;
        for (int v : vregs)
            if (info.alloc->assignments[v].spilled) n_spilled++;

        ss << "alloc " << file_name(pools, f) << ": pool "
           << pools.files[f].regs.size() << ", vregs " << vregs.size()
           << ", peak " << max_at_str(info.ra_stats->pressure[f])
           << ", spilled " << n_spilled;
        const int n_scr = n_scratch(*info.reg_cfg, f);
        if (n_scr > 0) {
            ss << ", scratch " << n_scr;
            if (n_spilled == 0) ss << " unused";
        }
        ss << "\n";
    }
    ss << "alloc: frame " << info.alloc->frame_bytes
       << " bytes, liveness passes " << info.ra_stats->liveness_passes << "\n";
    return ss.str();
}

// Returns the references of `v` (each read and each write) grouped by the loop
// depth of the operation, for example, `2 at depth 0, 1 at depth 2`. The spill
// weight of `v` counts the same references.
std::string refs_str(const ir_t &ir, const reg_alloc_stats_t &stats, int v) {
    std::vector<int> refs;
    std::vector<int> def_vregs, use_vregs;
    for (int i = 0; i < ir.n_ops(); i++) {
        ir.def_use(ir.ops()[i], def_vregs, use_vregs);
        const int n = (int)std::count(def_vregs.begin(), def_vregs.end(), v)
                + (int)std::count(use_vregs.begin(), use_vregs.end(), v);
        if (n == 0) continue;
        const int d = stats.loop_depth[i];
        if ((int)refs.size() <= d) refs.resize(d + 1, 0);
        refs[d] += n;
    }

    std::string s;
    for (int d = 0; d < (int)refs.size(); d++) {
        if (refs[d] == 0) continue;
        if (!s.empty()) s += ", ";
        s += std::to_string(refs[d]) + " at depth " + std::to_string(d);
    }
    return s;
}

// Returns the level-2 lines of the register allocation. For each register
// file with spilled vregs, a hint line tells why the file spills, and one line
// per spilled vreg follows.
//
// The scan spills in a file only when more intervals overlap than the pool
// holds (see `reg_alloc_stats_t`). If the peak still fits the pool, some
// intervals contain a dead gap, where the vreg holds no value that is read
// later. The allocator keeps one interval per vreg, so the vreg keeps its
// register through the gap. Otherwise the pool is too small.
std::string spill_lines(const ir_t &ir, const kernel_info_t &info) {
    const reg_pools_t &pools = info.reg_cfg->pools;
    const reg_alloc_stats_t &stats = *info.ra_stats;
    ostringstream_t ss;
    for (int f = 0; f < (int)pools.files.size(); f++) {
        std::vector<int> spilled;
        for (int v : file_vregs(ir, pools, f))
            if (info.alloc->assignments[v].spilled) spilled.push_back(v);
        if (spilled.empty()) continue;

        const std::string name = file_name(pools, f);
        const int pool = (int)pools.files[f].regs.size();
        const std::vector<int> &pressure = stats.pressure[f];
        const int peak = *std::max_element(pressure.begin(), pressure.end());
        if (peak <= pool) {
            ss << "hint " << name << ": overlap "
               << max_at_str(stats.overlap[f]) << " > pool " << pool
               << ", peak " << peak
               << " <= pool: the spills come from dead gaps in intervals\n";
        } else {
            ss << "hint " << name << ": peak " << max_at_str(pressure)
               << " > pool " << pool
               << ": the pool is too small for the live vregs\n";
        }

        for (int v : spilled)
            ss << "spill " << vreg_str(ir, (vreg_t)v, &info) << ": weight "
               << stats.weight[v] << ", interval " << stats.start[v] << ".."
               << stats.end[v] << ", refs " << refs_str(ir, stats, v) << "\n";
    }
    return ss.str();
}

// Returns the IR dump, with the register allocation when `info` is not null
// (see `to_string()` in `dump.hpp`).
std::string ir_dump(const ir_t &ir, const kernel_info_t *info) {
    std::vector<std::string> names;
    if (info)
        for (int f = 0; f < (int)info->reg_cfg->pools.files.size(); f++)
            names.push_back(file_name(info->reg_cfg->pools, f));

    ostringstream_t ss;
    if (info) {
        ss << "index |";
        for (const std::string &name : names)
            ss << " " << name;
        ss << " | operation\n";
    }

    // Indentation follows the loop nesting. A loop's closing line is indented
    // like its opening line, so the body stands out between the two.
    int depth = 0;
    for (int i = 0; i < ir.n_ops(); i++) {
        const op_t &op = ir.ops()[i];
        if (op.kind == op_kind_t::loop_end) depth--;

        ss << std::setw(5) << i << " |";
        if (info) {
            // Each number is right-aligned under the name of its file.
            for (int f = 0; f < (int)names.size(); f++)
                ss << " " << std::setw((int)names[f].size())
                   << info->ra_stats->pressure[f][i];
            ss << " |";
        }
        ss << " " << std::string(2 * std::max(depth, 0), ' ')
           << op_str(ir, op, info) << "\n";

        if (op.kind == op_kind_t::loop_begin) depth++;
    }
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
    return ir_dump(ir, nullptr);
}

std::string to_string(const ir_t &ir, const kernel_info_t &info) {
    assert(info.reg_cfg && info.alloc && info.ra_stats);
    return ir_dump(ir, &info);
}

std::string format_kernel_dump(
        int level, int seq, const kernel_info_t &info, const ir_t &ir) {
    assert(info.reg_cfg && info.alloc && info.ra_stats);
    ostringstream_t ss;
    ss << "begin x64ir #" << seq << " " << info.name
       << " isa=" << isa2str(info.isa) << "\n";
    ss << counts_line(ir);
    ss << "code: " << info.code_size << " bytes (instructions "
       << info.code_size - info.data_size << ", static data " << info.data_size
       << ")\n";
    ss << alloc_lines(ir, info);
    if (level >= 2) ss << spill_lines(ir, info);
    if (level >= 3) ss << ir_dump(ir, &info);
    ss << "end x64ir #" << seq << "\n";
    return ss.str();
}

void print_kernel_dump(const jit_generator_t &gen, const ir_t &ir,
        const data_section_t &data, const reg_config_t &reg_cfg,
        const reg_alloc_result_t &alloc, const reg_alloc_stats_t &ra_stats) {
    const int level = verbose_level();
    if (level == 0) return;

    kernel_info_t info;
    info.name = gen.name();
    info.isa = gen.max_cpu_isa();
    info.code_size = gen.getSize();
    info.data_size = info.code_size - data.begin_offset;
    info.reg_cfg = &reg_cfg;
    info.alloc = &alloc;
    info.ra_stats = &ra_stats;

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
