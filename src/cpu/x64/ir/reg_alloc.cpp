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

// The register allocator.
//
// Turns the IR's unlimited virtual registers into the CPU's real ones, spilling
// to the stack when more values are live at once than there are registers. It
// replaces the original brgemm kernel's hand-managed register numbering. The
// allocator is responsible for deciding WHO to spill, which relies on three
// ideas:
//   1. liveness
//   2. control-flow graph
//   3. linear scan
//
// It is designed to be generic. It knows only register kinds and control
// flow, nothing about the computation (brgemm, brgemv, copy, etc.) or the ISA.
//
// 1. Liveness
//
// A value is `live` at a point if it will be read again before being
// overwritten. Two values that are live at the same point interfere. They
// cannot share a register. So the allocator's input is each value's live span,
// and its job is to pack non-interfering spans onto the same register.
//
// 2. The Control-Flow Graph
//
// To know what is read later we must know what runs after what. In this IR
// that is nearly trivial. Each operation flows to the next, with three
// exceptions:
//   1. loop_end has two successors. It can fall through, or jump back to the
//      body start (`back-edge`).
//   2. jz has two successors. It can fall through, or jump to its label
//      (`forward-edge`).
//   3. jmp has one successor. It redirects to its label (`forward edge`).
//
//     Back-edge example:
//
//     0  loop_begin      <- back-edge returns here
//     1    v = load [ptr]
//     2    acc += v * x
//     3    ptr += stride
//     4  loop_end        -> 5, or jump back to 0
//     5  store acc
//
// So both branches and the back-edge depart from a straight list. The back-edge
// is special because it forms a cycle. A cycle is the one thing that forces the
// liveness pass to repeat, as the next section explains. Forward branches do
// not. There are no basic blocks. Every operation is a node.
//
// Backward Liveness Analysis
//
// Liveness is computed backwards and iterated. Needed here depends on uses
// later, so we go through the list back to front. Each operation keeps alive
// whatever its successors need, minus what it overwrites, plus what it reads.
//
// The back-edge makes this take more than one pass. Iterate to a fixed point
// is a static analysis of the program text. We don't need to know the loop
// counter values. We re-iterate over the operation list until a pass changes
// nothing. That stable result is the `fixed point`.
//
// Why a second pass is ever needed: a `ptr` read at the top of the body
// (see the back-edge example) and advanced at the bottom is, through the
// back-edge, the value the next iteration reads at the top. So it must be live
// across the whole body. The first pass does not yet know the loop start needs
// it. The next pass propagates that, and it appears. This settles in a handful
// of passes. The count scales with loop nesting depth, not with how many times
// the loop runs. The result is liveness valid for every possible execution at
// once.
//
// 3. Linear Scan
//
// Collapse each value's liveness to one [start, end] interval, ignoring holes.
// That simplification is what keeps this cheap. If we need, we can enable
// intervals split to improve register allocation under pressure.
// Per register kind, sort by start and go through them, keeping the live
// (`active`) intervals:
//
//     v0 |=================|   (0..9)
//     v1   |===|               (1..3)
//     v2         |=======|     (4..8)
//     v3           |=====|     (5..8)
//        0 1 2 3 4 5 6 7 8 9
//
// Free a register when its interval ends. When none is free, spill the interval
// whose end is furthest away. In the picture, at t=5 both registers are taken
// by v0 and v2. v0 ends at 9 and v2 ends at 8, so spill v0 and give its
// register to the newcomer. If the newcomer itself ends furthest, it is the one
// spilled. Furthest-end frees a register for the longest stretch. A heavier
// kernel will want to weight this by loop depth so hot values stay put. The
// mechanism stays the same.
//
// TODO: Enable weights based on the loop depth.

#include <algorithm>
#include <climits>

#include "cpu/x64/ir/reg_alloc.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

namespace {

// Compute liveness for each operation `i`.
//
// A value is `live` at operation `i` if some future operation may still
// read it before it is overwritten. In other words, the value must be kept
// available because it might be needed later.
//
// Two values that are live at the same operation cannot share a register.
//
// Backward data-flow to a fixed point. For each operation `i`:
//
//   1. Computing `live_in` at operation `i`:
//      A variable is live before `i` if `i` uses it, or if it is needed
//      later and not overwritten by `i`.
//      Formula: live_in[i] = use[i] U (live_out[i] - def[i])
//
//   2. Computing `live_out` at operation `i`:
//      A variable is live after `i` if any successor may need it.
//      Formula: live_out[i] = union of live_in over all successors of `i`
//
// Scan each `i` from last to first, and repeat the whole scan until nothing
// changes (the fixed point). Successors are `i+1` for a plain operation, the
// label for jmp/jz, and `i+1` plus the loop body start for loop_end
// (the back-edge).
//
// Small example: `p` is read at the top of a loop body and rewritten at the
// bottom:
//
//   0  loop_begin
//   1    v = load [p]      (p read)
//   2    p = p + stride    (p rewritten)
//   3  loop_end            (back-edge to 0)
//
// `p` must be live across the whole body, because the value written at 2 is
// read at 1 on the next turn. One backward pass finds most of it. The entry to
// loop_end needs the back-edge, so it appears only on the second pass. A third
// pass changes nothing, which is the fixed point.
void compute_liveness(
        const ir_t &ir, std::vector<std::vector<int8_t>> &live_in) {
    const int n_ops = ir.n_ops();
    const int n_vregs = ir.n_vregs();

    // Phase 1: build per-operation def/use sets and the control-flow graph.
    //
    // def_at[i][v] is 1 if operation `i` defines (writes) vreg `v`.
    // use_at[i][v] is 1 if operation `i` uses (reads) vreg `v`.
    // successors[i] lists all operations that may execute immediately after i.
    std::vector<std::vector<int8_t>> def_at(
            n_ops, std::vector<int8_t>(n_vregs, 0));
    std::vector<std::vector<int8_t>> use_at(
            n_ops, std::vector<int8_t>(n_vregs, 0));

    std::vector<std::vector<int>> successors(n_ops);
    std::vector<int> def_vregs, use_vregs;

    // Map label IDs to operation indices so jmp/jz can resolve their target
    // locations when building the control-flow graph.
    std::vector<int> label_to_op(ir.n_labels(), -1);
    for (int i = 0; i < n_ops; i++)
        if (ir.ops()[i].kind == op_kind_t::label)
            label_to_op[(int)ir.ops()[i].imm] = i;

    for (int i = 0; i < n_ops; i++) {
        const op_t &op = ir.ops()[i];
        ir.def_use(op, def_vregs, use_vregs);

        for (int v : def_vregs)
            def_at[i][v] = 1;
        for (int v : use_vregs)
            use_at[i][v] = 1;

        // Successor edges include fall-through to `i+1` and any explicit
        // control-flow transfers (branches and loop back-edges).
        if (op.kind == op_kind_t::loop_end) {
            if (i + 1 < n_ops) successors[i].push_back(i + 1);
            successors[i].push_back(op.match + 1);
        } else if (op.kind == op_kind_t::jmp) {
            successors[i].push_back(label_to_op[(int)op.imm]);
        } else if (op.kind == op_kind_t::jz) {
            if (i + 1 < n_ops) successors[i].push_back(i + 1);
            successors[i].push_back(label_to_op[(int)op.imm]);
        } else if (i + 1 < n_ops) {
            successors[i].push_back(i + 1);
        }
    }

    // Phase 2: compute `live_in` and `live_out` using backward dataflow
    // analysis. Iterate until reaching a fixed point (no changes in a full
    // pass).
    live_in.assign(n_ops, std::vector<int8_t>(n_vregs, 0));
    std::vector<std::vector<int8_t>> live_out(
            n_ops, std::vector<int8_t>(n_vregs, 0));

    bool changed = true;
    while (changed) {
        changed = false;
        for (int i = n_ops - 1; i >= 0; i--) {
            // live_out[i] = OR over successors `s` of live_in[s]
            for (int v = 0; v < n_vregs; v++) {
                int8_t live_out_v = 0;
                for (int s : successors[i])
                    live_out_v |= live_in[s][v];
                if (live_out_v != live_out[i][v]) {
                    live_out[i][v] = live_out_v;
                    changed = true;
                }
            }
            // live_in[i] = use[i] OR (live_out[i] AND NOT def[i])
            for (int v = 0; v < n_vregs; v++) {
                const char live_in_v
                        = use_at[i][v] || (live_out[i][v] && !def_at[i][v]);
                if (live_in_v != live_in[i][v]) {
                    live_in[i][v] = live_in_v;
                    changed = true;
                }
            }
        }
    }
}

// Assign physical registers to virtual registers within a single physical
// register file using linear-scan register allocation. A file may serve more
// than one register kind (e.g. vec and mask on AVX2*).
//
// Each virtual register is represented as an interval [start, end].
// Intervals are sorted by start and processed left to right, maintaining an
// `active` set of intervals currently holding registers.
//
// For each interval `v`:
//   1. Expire old intervals:
//      Remove any active interval `a` where end[a] < start[v], freeing its
//      register.
//
//   2. Allocate register:
//      If a register is free, assign it to `v`.
//
//   3. Spill if necessary:
//      Otherwise select the active interval with the latest end
//      (the `farthest live` interval).
//      If end[victim] > end[v], spill victim and reuse its register for `v`,
//      otherwise spill `v`.
//
// Spilled values are assigned stack slots starting at `frame`, increasing by
// `slot_size` per spill.
//
// Example (2 registers: r0, r1):
//
//   v0 [0..9]   v1 [1..3]   v2 [4..8]   v3 [5..8]
//
//   v0 -> r0
//   v1 -> r1
//
//   v2: v1 expires (3 < 4), r1 is free, so v2 -> r1
//
//   v3: no free registers.
//       active ends: v0=9, v2=8 -> victim = v0
//       since 9 > 8, spill v0 and assign r0 to v3
void alloc_file(const ir_t &ir, int file_idx,
        const std::vector<int> &kind_to_file, const std::vector<int> &pool,
        const std::vector<int> &start, const std::vector<int> &end,
        size_t slot_size, reg_alloc_result_t &res, size_t &frame) {

    const int n_vregs = ir.n_vregs();

    // Collect intervals belonging to this register file
    // (end[v] >= 0 means v is used) and sort by start time, since linear scan
    // processes intervals in increasing start order.
    std::vector<int> scan_order;

    for (int v = 0; v < n_vregs; v++) {
        if (kind_to_file[(int)ir.vreg_info()[v].kind] == file_idx
                && end[v] >= 0)
            scan_order.push_back(v);
    }

    std::sort(scan_order.begin(), scan_order.end(),
            [&](int lhs, int rhs) { return start[lhs] < start[rhs]; });

    // free_regs: pool of available physical registers.
    // active: intervals currently holding registers, sorted by increasing end.
    // still_active: intervals that survive expiry each iteration.
    std::vector<int> free_regs(pool.rbegin(), pool.rend());
    std::vector<int> active;
    std::vector<int> still_active;

    for (int v : scan_order) {
        // Expire intervals that end before the current interval starts,
        // returning their registers to the free pool.
        still_active.clear();
        for (int a : active) {
            if (end[a] < start[v]) {
                if (!res.assignments[a].spilled)
                    free_regs.push_back(res.assignments[a].phys);
            } else {
                still_active.push_back(a);
            }
        }
        active.swap(still_active);

        if (!free_regs.empty()) {
            // A register is available so assign it.
            res.assignments[v].phys = free_regs.back();
            free_regs.pop_back();
        } else {
            // No free register so choose a spill candidate
            // (farthest-live interval).
            int victim = -1;
            for (int a : active) {
                if (!res.assignments[a].spilled
                        && (victim < 0 || end[a] > end[victim])) {
                    victim = a;
                }
            }

            if (victim >= 0 && end[victim] > end[v]) {
                // Victim outlives `v` so spill victim and reuse its register.
                res.assignments[v].phys = res.assignments[victim].phys;
                res.assignments[victim].spilled = true;
                res.assignments[victim].slot = frame;
                frame += slot_size;
                res.any_spill = true;
            } else {
                // `v` itself ends furthest so spill `v`. It lives on the stack
                // and is not added to the active set.
                res.assignments[v].spilled = true;
                res.assignments[v].slot = frame;
                frame += slot_size;
                res.any_spill = true;
                continue;
            }
        }
        // `v` now holds a register. Insert it into the active set, keeping the
        // set sorted by increasing end.
        const auto pos = std::lower_bound(active.begin(), active.end(), v,
                [&](int lhs, int rhs) { return end[lhs] < end[rhs]; });
        active.insert(pos, v);
    }
}

} // namespace

// Run full register allocation pipeline.
// Returns, for each virtual register, either a physical register or a spill
// slot.
//
// The pipeline:
//   1. Compute liveness (live_in for each operation).
//   2. Convert liveness into a single interval per virtual register:
//        start[v] = first operation where `v` is defined, used, or live_in
//        end[v]   = last such operation
//   3. Run linear-scan allocation per physical register file, sharing a single
//      stack frame across all files.
//
// Interval construction is intentionally conservative. If a value is live in
// disjoint regions (e.g. 3-7 and 20-25), it is merged into [3-25]. This
// ignores holes but guarantees correctness and keeps the algorithm simple.
//
// This approximation is usually acceptable because the most important values
// are hot loop-invariant pointers (e.g. base pointers for C and batch data),
// which are intended to remain in registers for the entire kernel anyway.
// For these cases, merging gaps has no practical cost. The only downside
// appears when a value has a true dead region while registers are scarce,
// where the gap could otherwise be reused.
//
// TODO: improve allocation quality in two stages (in order):
//
//   1. Add loop-depth spill weighting.
//      The current `farthest end wins` rule may incorrectly spill hot values
//      (e.g. loop-invariant pointers) simply because they live long.
//      Weighting by loop nesting depth biases spills toward cold values.
//      This is a local change to alloc_kind.
//
//   2. Add hole-aware live-range splitting.
//      Required only under register pressure when intervals cannot all fit.
//      Splits must respect spill weights and must not break hot loop-carried
//      values inside loops.
reg_alloc_result_t allocate_registers(
        const ir_t &ir, const reg_pools_t &pools) {
    const int n_ops = ir.n_ops();
    const int n_vregs = ir.n_vregs();

    // Step 1: compute operation-level liveness.
    // live_in[i][v] is 1 when virtual register `v` is needed on entry to
    // operation `i`.
    std::vector<std::vector<int8_t>> live_in;
    compute_liveness(ir, live_in);

    // Step 2: build a single live interval per virtual register.
    // Each interval [start[v], end[v]] approximates all points where `v` is
    // live.
    //
    // extend_interval(v) expands the interval to include operation `i`
    // whenever `v` is defined, used, or live on entry to `i`.
    std::vector<int> start(n_vregs, INT_MAX), end(n_vregs, -1);
    std::vector<int> def_vregs, use_vregs;

    for (int i = 0; i < n_ops; i++) {
        auto extend_interval = [&](int v) {
            if (v < 0) return;
            start[v] = std::min(start[v], i);
            end[v] = std::max(end[v], i);
        };

        ir.def_use(ir.ops()[i], def_vregs, use_vregs);
        for (int v : def_vregs)
            extend_interval(v);
        for (int v : use_vregs)
            extend_interval(v);
        for (int v = 0; v < n_vregs; v++)
            if (live_in[i][v]) extend_interval(v);
    }

    // Step 3: run linear-scan register allocation per physical register file,
    // sharing a single stack frame so spill slots do not overlap across files.
    reg_alloc_result_t res;
    res.assignments.assign(n_vregs, assignment_t());

    size_t frame = 0;
    for (int f = 0; f < (int)pools.files.size(); f++) {
        alloc_file(ir, f, pools.kind_to_file, pools.files[f].regs, start, end,
                pools.files[f].slot_size, res, frame);
    }

    constexpr size_t stack_alignment = 16;
    res.frame_bytes = utils::rnd_up(frame, stack_alignment);

    return res;
}

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
