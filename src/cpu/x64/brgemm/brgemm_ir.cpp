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

// The IR-based BRGEMM kernel converts a brgemm descriptor into IR, then runs
// the register allocator and code emitter.
//
// This kernel uses the classic kernels as a reference: `jit_brgemm_kernel_t`
// for the vector compute path and `jit_brgemm_amx_uker_t` for the TMUL one. It
// is comparable with them instruction for instruction when it comes to compute
// structure.
//
// `init_conf()` is the only code that reads the descriptor for the builder. It
// converts the descriptor into `conf_t`, and the builder reads only `conf_t`.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "cpu/platform.hpp"
#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/brgemm/brgemm_ir.hpp"
#include "cpu/x64/brgemm/brgemm_ir_compute_path.hpp"
#include "cpu/x64/brgemm/brgemm_ir_conf.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/ir/codegen.hpp"
#include "cpu/x64/ir/ir.hpp"
#include "cpu/x64/ir/reg_config.hpp"
#include "cpu/x64/jit_generator.hpp"

#define GET_OFF(field) offsetof(brgemm_kernel_params_t, field)
#define GET_OFF_BATCH_ELEMENT(field) offsetof(brgemm_batch_element_t, field)

#define VCONDCHECK_BRGEMM_IR(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, brgemm_ir, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace brgemm_ir {

namespace {

// M-loop input register classification
//
// The same split the GEMV IR kernel uses (see `brgemv_ir.cpp`), by whether a
// value advances across M-loop iterations. The one difference is that this
// loop nest has two levels over the output, M and N, so the level a register
// advances at is stated per field instead of shared by the struct.
//
// Design rule:
// - A field's vreg ID never changes after assignment.
// - Because of this, these structs are passed by const reference.
//
// Note:
// - Registers created inside a loop body (the accumulators, the B vectors, the
//   A broadcast, the batch pointer, and the A and B pointers of each batch
//   element) are local temporaries and are not included here.

// Registers that advance by a fixed byte offset as the nest walks the output.
// Only the looped levels advance them.
struct advancing_regs_t {
    // Current C pointer. Of the M and N loops, the inner one advances it by
    // one block per iteration, then the outer one rewinds it and steps to its
    // next block with one constant. The classic kernel keeps a second base
    // pointer for this instead. One running pointer saves the general-purpose
    // register. `none` when the blocks are stored to D and beta is 0, since C
    // is never read then.
    ir::vreg_t c_ptr = ir::vreg_t::none;
    // Pointer the finished block is stored through. It is `c_ptr` when the
    // blocks are stored to C. When they are stored to D, it is a register of
    // its own that holds the `ptr_D` kernel argument and advances like `c_ptr`
    // by `d_off()`.
    ir::vreg_t store_ptr = ir::vreg_t::none;
    // Byte offset into A of the current M block. Starts at 0 and moves like
    // `c_ptr`, but only along M. `none` when the M level is unrolled.
    ir::vreg_t a_off = ir::vreg_t::none;
    // Byte offset into B of the current N block. Starts at 0 and moves like
    // `c_ptr`, but only along N. `none` when the N level is unrolled.
    ir::vreg_t b_off = ir::vreg_t::none;
    // Current bias pointer. The bias is indexed by the load dimension only, so
    // it moves like `b_off`. `none` without the `bias` stage.
    ir::vreg_t bias_ptr = ir::vreg_t::none;
    // Current weights-scale pointer. A per-N scale moves like `bias_ptr`. A
    // common one stays at its only element. `none` without a weights scale
    // stage.
    ir::vreg_t wei_scale_ptr = ir::vreg_t::none;
};

// Registers that hold the same value for the entire M loop.
struct invariant_regs_t {
    // Base pointer of the batch-element array.
    ir::vreg_t batch = ir::vreg_t::none;
    // Batch size loop count. `none` when `max_bs == 1` (single batch element)
    // or without `var_bs`.
    ir::vreg_t bs = ir::vreg_t::none;
    // A and B base pointers, the `ptr_A` and `ptr_B` kernel arguments, which
    // `brgemm_offs` adds the byte offsets of each batch element to. `none` for
    // `brgemm_addr`, where each element holds its own pointers.
    ir::vreg_t a_base = ir::vreg_t::none;
    ir::vreg_t b_base = ir::vreg_t::none;
    // A and B pointers of the only batch element, set once for the kernel when
    // no level moves them (see `init_m_loop_input_regs()`). `none` otherwise,
    // and `emit_batch_level()` creates them for each batch element.
    ir::vreg_t a_ptr = ir::vreg_t::none;
    ir::vreg_t b_ptr = ir::vreg_t::none;
    // Common source and destination scale pointers.
    // `none` without the `src_scale` and `dst_scale` stages respectively.
    ir::vreg_t src_scale_ptr = ir::vreg_t::none;
    ir::vreg_t dst_scale_ptr = ir::vreg_t::none;
};

// Complete input register set for the M loop, partitioned by whether values
// advance across iterations.
struct m_loop_input_regs_t {
    advancing_regs_t advancing;
    invariant_regs_t invariant;
};

// Sets `a_ptr` and `b_ptr` to the A and B of the batch element at
// `batch_ptr + disp`, at the current M and N block. Mirrors
// `set_A_B_matrices()` in `jit_brgemm_kernel.cpp`.
void emit_batch_element_ptrs(ir::ir_t &ir, const conf_t &cfg,
        const m_loop_input_regs_t &regs, ir::vreg_t batch_ptr, dim_t disp,
        ir::vreg_t a_ptr, ir::vreg_t b_ptr) {

    switch (cfg.compute_path.batch_kind) {
        case brgemm_addr:
            ir.load(a_ptr, batch_ptr, disp + GET_OFF_BATCH_ELEMENT(ptr.A));
            ir.load(b_ptr, batch_ptr, disp + GET_OFF_BATCH_ELEMENT(ptr.B));
            break;
        case brgemm_offs:
            ir.load(a_ptr, batch_ptr, disp + GET_OFF_BATCH_ELEMENT(offset.A));
            ir.add_reg(a_ptr, regs.invariant.a_base);
            ir.load(b_ptr, batch_ptr, disp + GET_OFF_BATCH_ELEMENT(offset.B));
            ir.add_reg(b_ptr, regs.invariant.b_base);
            break;
        default: assert(!"unsupported batch kind"); break;
    }

    // Shift to the current M and N block.
    if (regs.advancing.a_off != ir::vreg_t::none)
        ir.add_reg(a_ptr, regs.advancing.a_off);
    if (regs.advancing.b_off != ir::vreg_t::none)
        ir.add_reg(b_ptr, regs.advancing.b_off);
}

// Sets up the M-loop input registers of the loop nest of `nest_cfg`.
//
// Loads the kernel argument pointers and the batch count, zeroes the running A
// and B offsets, and sets the A and B pointers when no level moves them.
m_loop_input_regs_t init_m_loop_input_regs(
        ir::ir_t &ir, const conf_t &cfg, const nest_conf_t &nest_cfg) {

    m_loop_input_regs_t regs;
    const blocking_conf_t &blocking = cfg.blocking;
    const store_conf_t &store_cfg = nest_cfg.store;

    if (!store_cfg.to_d || cfg.output.beta != 0.0f) {
        regs.advancing.c_ptr = ir.new_gpr();
        ir.load_param(regs.advancing.c_ptr, GET_OFF(ptr_C));
    }

    regs.invariant.batch = ir.new_gpr();
    ir.load_param(regs.invariant.batch, GET_OFF(batch));

    if (blocking.var_bs && blocking.max_bs > 1) {
        regs.invariant.bs = ir.new_gpr();
        ir.load_param(regs.invariant.bs, GET_OFF(BS));
    }

    if (cfg.compute_path.batch_kind == brgemm_offs) {
        regs.invariant.a_base = ir.new_gpr();
        ir.load_param(regs.invariant.a_base, GET_OFF(ptr_A));
        regs.invariant.b_base = ir.new_gpr();
        ir.load_param(regs.invariant.b_base, GET_OFF(ptr_B));
    }

    if (nest_cfg.m_form == level_form_t::looped) {
        regs.advancing.a_off = ir.new_gpr();
        ir.mov_imm(regs.advancing.a_off, 0);
    }

    if (nest_cfg.n_form == level_form_t::looped) {
        regs.advancing.b_off = ir.new_gpr();
        ir.mov_imm(regs.advancing.b_off, 0);
    }

    // With one batch element, only a looped M, N, or K level moves the A and B
    // pointers. When none of them is looped, the pointers are set once for the
    // kernel.
    if (blocking.max_bs == 1
            && utils::everyone_is(level_form_t::unrolled, nest_cfg.m_form,
                    nest_cfg.n_form, nest_cfg.k_form)) {
        regs.invariant.a_ptr = ir.new_gpr();
        regs.invariant.b_ptr = ir.new_gpr();

        emit_batch_element_ptrs(ir, cfg, regs, regs.invariant.batch, 0,
                regs.invariant.a_ptr, regs.invariant.b_ptr);
    }

    if (store_cfg.has_stage(output_stage_t::bias)) {
        regs.advancing.bias_ptr = ir.new_gpr();
        ir.load_param(regs.advancing.bias_ptr, GET_OFF(ptr_bias));
    }

    if (store_cfg.has_stage(output_stage_t::src_scale)) {
        regs.invariant.src_scale_ptr = ir.new_gpr();
        ir.load_param(regs.invariant.src_scale_ptr, GET_OFF(ptr_src_scales));
    }

    if (store_cfg.has_stage(output_stage_t::wei_scale)
            || store_cfg.has_stage(output_stage_t::wei_scale_per_n)) {
        regs.advancing.wei_scale_ptr = ir.new_gpr();
        ir.load_param(regs.advancing.wei_scale_ptr, GET_OFF(ptr_wei_scales));
    }

    if (store_cfg.has_stage(output_stage_t::dst_scale)) {
        regs.invariant.dst_scale_ptr = ir.new_gpr();
        ir.load_param(regs.invariant.dst_scale_ptr, GET_OFF(ptr_dst_scales));
    }

    // With `beta == 1`, the blocks read C for the accumulators, so D needs a
    // register of its own.
    if (store_cfg.to_d) {
        regs.advancing.store_ptr = ir.new_gpr();
        ir.load_param(regs.advancing.store_ptr, GET_OFF(ptr_D));
    } else {
        regs.advancing.store_ptr = regs.advancing.c_ptr;
    }

    return regs;
}

// Iteration lists
//
// Each level of the loop nest (M, N, K, and the batch) has a list of its
// blocks, stored as runs of blocks with the same size. The code of a block
// depends only on its size and its position, so a looped level emits one body
// per run, and an unrolled level one body per block. With `var_bs`, the batch
// does not use its list: it is a runtime loop over `BS` elements.

// Consecutive blocks of one loop level with the same size.
//
//   size  - elements each block covers along the level's dimension
//   count - number of blocks, 0 if the level has no such blocks
struct block_run_t {
    dim_t size;
    dim_t count;
};

// Runs of the M, N, and K levels, in the order `jit_brgemm_kernel_t` walks
// them: the full blocks, then each tail. N has two tails: `ldb2_tail` full
// vector registers or tiles, then the partial one of `ldb_tail` columns.
// `n_tail_grouping` gives each its own block or puts both in one block.
std::vector<block_run_t> m_runs(const blocking_conf_t &blocking) {
    return {{blocking.bd_block, blocking.bdb},
            {blocking.bdb_tail, blocking.bdb_tail > 0 ? 1 : 0}};
}

std::vector<block_run_t> n_runs(const blocking_conf_t &blocking) {
    const dim_t full_size = (dim_t)blocking.ld_block2 * blocking.ld_block;
    const dim_t ldb2_tail_size = (dim_t)blocking.ldb2_tail * blocking.ld_block;

    switch (blocking.n_tail_grouping) {
        case tail_grouping_t::separate:
            return {{full_size, blocking.ldb2},
                    {ldb2_tail_size, blocking.ldb2_tail > 0 ? 1 : 0},
                    {blocking.ldb_tail, blocking.ldb_tail > 0 ? 1 : 0}};
        case tail_grouping_t::grouped: {
            const dim_t tail_size = ldb2_tail_size + blocking.ldb_tail;
            return {{full_size, blocking.ldb2},
                    {tail_size, tail_size > 0 ? 1 : 0}};
        }
    }
    assert(!"unknown tail grouping");
    return {};
}

std::vector<block_run_t> k_runs(const blocking_conf_t &blocking) {
    return {{blocking.rd_block, blocking.rdb},
            {blocking.rdb_tail, blocking.rdb_tail > 0 ? 1 : 0}};
}

// Runs of the batch: `max_bs` elements, each a block of size 1.
std::vector<block_run_t> bs_runs(const blocking_conf_t &blocking) {
    return {{1, blocking.max_bs}};
}

// Emits one level of the loop nest from its runs, looped.
//
// A run of more than one block becomes a runtime loop whose step moves the
// pointers by one block. A run of one block is emitted once, without a step
// (see `emit_loop_imm()`). Its accesses add its position to their
// displacements, so the pointers do not move to it.
//
// The level inside this one (N inside M, or M inside N) moves its own pointers
// as it runs, and they stay moved after each block of this level. The step of
// this level moves them back to where they stood before the block. `inner_pos`
// tracks where they stand.
//
// Example: N inside M, and N is one run of 3 blocks of 16 elements. After each
// M block the N pointers stand at 48. The M step moves the M pointers by one
// block and the N pointers by -48.
//
//   pos       - where the pointers of this level stand, in elements from its
//               first block. The level updates it to where it leaves them.
//   inner_pos - where the pointers of the level inside stand, in elements from
//               its first block. `body` updates it.
//   body      - `body(size, pos)` emits a block of `size` elements at `pos`
//               elements from the pointers
//   step      - `step(n, inner_n)` moves the pointers of this level by `n`
//               elements and the pointers of the level inside by `inner_n`
//               elements
void emit_looped_level(ir::ir_t &ir, const std::vector<block_run_t> &runs,
        dim_t &pos, dim_t &inner_pos,
        const std::function<void(dim_t, dim_t)> &body,
        const std::function<void(dim_t, dim_t)> &step) {

    dim_t start = 0;
    for (const block_run_t &run : runs) {
        const dim_t run_pos = start - pos;
        const dim_t inner_start = inner_pos;

        ir::emit_loop_imm(ir, run.count, [&]() { body(run.size, run_pos); },
                [&]() { step(run.size, inner_start - inner_pos); });

        // A runtime loop runs its step after every iteration, including the
        // last one, so the pointers of the level inside are back where they
        // stood before the run. After a run of one block, they stay where
        // `body` left them.
        if (run.count > 1) {
            pos += run.count * run.size;
            inner_pos = inner_start;
        }
        start += run.count * run.size;
    }
}

// Returns an upper bound, in elements, on the distance between the pointers
// of a looped level and any element it accesses or any position the pointers
// move to. It walks the runs as `emit_looped_level()` does.
//
// The bound also covers where the level leaves the pointers, which N needs.
// If the first M block has no step, it leaves the N pointers at the end of
// the N level, and the next M block reaches its N blocks at negative
// positions from there.
dim_t max_looped_pos(const std::vector<block_run_t> &runs) {
    dim_t max_pos = 0, start = 0, ptr_pos = 0;
    for (const block_run_t &run : runs) {
        if (run.count == 0) continue;
        max_pos = std::max(max_pos, start - ptr_pos + run.size);
        if (run.count > 1) ptr_pos += run.count * run.size;
        start += run.count * run.size;
    }
    return std::max(max_pos, ptr_pos);
}

// Emits one level of the loop nest from its runs, unrolled.
//
// Each block is emitted at its position, and the pointers never move.
//
//   pos  - where the pointers stand, in elements from the first block
//   body - `body(size, pos)` emits a block of `size` elements at `pos`
//          elements from the pointers
void emit_unrolled_level(const std::vector<block_run_t> &runs, dim_t pos,
        const std::function<void(dim_t, dim_t)> &body) {

    dim_t start = 0;
    for (const block_run_t &run : runs) {
        for (dim_t i = 0; i < run.count; i++) {
            body(run.size, start - pos);
            start += run.size;
        }
    }
}

// Returns the number of elements a level covers. It bounds every position an
// unrolled level addresses.
dim_t level_size(const std::vector<block_run_t> &runs) {
    dim_t size = 0;
    for (const block_run_t &run : runs) {
        size += run.count * run.size;
    }
    return size;
}

// Returns an upper bound, in elements, on the positions a level in `form`
// addresses and moves its pointers to.
dim_t max_level_pos(const std::vector<block_run_t> &runs, level_form_t form) {
    return form == level_form_t::looped ? max_looped_pos(runs)
                                        : level_size(runs);
}

// Emits one level of the loop nest from its runs in `form`, with the
// parameters of `emit_looped_level()`. An unrolled level has no step, and the
// pointers of the level inside stay where each block leaves them.
void emit_level(ir::ir_t &ir, const std::vector<block_run_t> &runs,
        level_form_t form, dim_t &pos, dim_t &inner_pos,
        const std::function<void(dim_t, dim_t)> &body,
        const std::function<void(dim_t, dim_t)> &step) {

    switch (form) {
        case level_form_t::unrolled:
            emit_unrolled_level(runs, pos, body);
            break;
        case level_form_t::looped:
            emit_looped_level(ir, runs, pos, inner_pos, body, step);
            break;
    }
}

// Like the overload above, for a level whose step does not move the pointers of
// the level inside. `step(n)` moves the pointers of the level by `n` elements.
void emit_level(ir::ir_t &ir, const std::vector<block_run_t> &runs,
        level_form_t form, dim_t &pos,
        const std::function<void(dim_t, dim_t)> &body,
        const std::function<void(dim_t)> &step) {

    dim_t inner_pos = 0;
    emit_level(ir, runs, form, pos, inner_pos, body,
            [&](dim_t n, dim_t) { step(n); });
}

// True if the batch and K levels are unrolled and the batch is not empty. Each
// K step of a block is then its own call of `emit_microkernel()`, so it can
// emit its own stores and prefetches (`k_step_interleave_t`).
bool has_unrolled_k_steps(const conf_t &cfg, const nest_conf_t &nest_cfg) {
    return nest_cfg.bs_form == level_form_t::unrolled
            && nest_cfg.k_form == level_form_t::unrolled
            && cfg.blocking.max_bs > 0;
}

// Returns the part of `interleave` of the batch element or K block that starts
// at `pos` and has `size` elements, in a level of `level_size` elements. The
// first one keeps `prev`, the last one keeps `acc_stores`, and each one gets
// the part of each list of `prefetches` in proportion to its position and size
// (each of 4 K blocks gets 4 of the 16 prefetches of a tile).
//
// `pos` counts from the first element of the level, as in an unrolled level.
// When `has_unrolled_k_steps()` is false, `interleave` has no stores or
// prefetches, so `pos` does not matter.
k_step_interleave_t k_step_interleave_at(const k_step_interleave_t &interleave,
        dim_t pos, dim_t size, dim_t level_size) {

    k_step_interleave_t part = interleave;
    if (pos != 0) part.prev = {};
    if (pos + size != level_size) part.acc_stores = acc_stores_t::none;

    for (int i = 0; i < (int)interleave.prefetches.size(); i++) {
        const std::vector<prefetch_t> &acc_prefetches
                = interleave.prefetches[i];
        const dim_t n = (dim_t)acc_prefetches.size();
        const auto first = acc_prefetches.begin();
        part.prefetches[i].assign(first + n * pos / level_size,
                first + n * (pos + size) / level_size);
    }
    return part;
}

// The K level of one batch element. Reduces K into `acc` from `a_ptr` and
// `b_ptr`, which a looped K level moves. Each K step gets its part of
// `interleave` (`k_step_interleave_at()`).
void emit_k_level(ir::ir_t &ir, const conf_t &cfg, const nest_conf_t &nest_cfg,
        const compute_path_t &cp, const out_block_t &blk,
        const std::vector<ir::vreg_t> &acc, ir::vreg_t a_ptr, ir::vreg_t b_ptr,
        const k_step_interleave_t &interleave) {

    // Advance the A and B pointers by `rd` elements of K.
    auto step = [&](dim_t rd) {
        ir.add_imm(a_ptr, cfg.a_off(0, rd));
        ir.add_imm(b_ptr, cfg.b_off(rd, 0));
    };

    const std::vector<block_run_t> runs = k_runs(cfg.blocking);
    const dim_t k_size = level_size(runs);

    // K tail needs no mask. A reduction step reads a full B vector along N
    // and one element of A, so a short tail is just fewer steps.
    dim_t k_pos = 0;
    emit_level(ir, runs, nest_cfg.k_form, k_pos,
            [&](dim_t rd_loop, dim_t rd_start) {
        const k_step_interleave_t step_interleave
                = k_step_interleave_at(interleave, rd_start, rd_loop, k_size);
        cp.emit_microkernel(ir, cfg, blk, acc, a_ptr, b_ptr, rd_start,
                (int)rd_loop, step_interleave);
    }, step);
}

// The batch level of one block. Reduces every batch element into `acc`. Each
// K step gets its part of `interleave` (`k_step_interleave_at()`).
//
// With `var_bs` it is a runtime loop over `BS` elements. Otherwise it is the
// `max_bs` elements of `bs_runs()` in `bs_form`.
void emit_batch_level(ir::ir_t &ir, const conf_t &cfg,
        const nest_conf_t &nest_cfg, const compute_path_t &cp,
        const m_loop_input_regs_t &regs, const out_block_t &blk,
        const std::vector<ir::vreg_t> &acc,
        const k_step_interleave_t &interleave) {

    const blocking_conf_t &blocking = cfg.blocking;

    // The only batch element, whose A and B pointers are set for the kernel.
    if (regs.invariant.a_ptr != ir::vreg_t::none) {
        emit_k_level(ir, cfg, nest_cfg, cp, blk, acc, regs.invariant.a_ptr,
                regs.invariant.b_ptr, interleave);
        return;
    }

    if (blocking.var_bs) {
        // `BS` is known only at run time, so the batch level cannot be unrolled.
        assert(nest_cfg.bs_form == level_form_t::looped);

        const ir::vreg_t batch_ptr = ir.new_gpr();
        ir.mov_reg(batch_ptr, regs.invariant.batch);

        auto element = [&]() {
            const ir::vreg_t a_ptr = ir.new_gpr();
            const ir::vreg_t b_ptr = ir.new_gpr();

            emit_batch_element_ptrs(ir, cfg, regs, batch_ptr, 0, a_ptr, b_ptr);

            // Advance to the next batch element. A single batch element has no
            // next one, so nothing is emitted then.
            if (blocking.max_bs > 1) {
                ir.add_imm(batch_ptr, sizeof(brgemm_batch_element_t));

                // The classic kernel prefetches the next element for the
                // address kind only.
                if (cfg.compute_path.batch_kind == brgemm_addr)
                    ir.prefetch(batch_ptr, 0, ir::prefetch_kind_t::t0);
            }

            emit_k_level(
                    ir, cfg, nest_cfg, cp, blk, acc, a_ptr, b_ptr, interleave);
        };

        if (blocking.max_bs > 1)
            ir::emit_loop_reg(ir, regs.invariant.bs, element);
        else
            ir::emit_loop_imm(ir, 1, element);

        return;
    }

    // An unrolled batch addresses every element from the `batch` argument. A
    // looped one moves its pointer, so it uses a copy of `batch`.
    ir::vreg_t batch_ptr = regs.invariant.batch;
    if (nest_cfg.bs_form == level_form_t::looped) {
        batch_ptr = ir.new_gpr();
        ir.mov_reg(batch_ptr, regs.invariant.batch);
    }

    const std::vector<block_run_t> runs = bs_runs(blocking);
    const dim_t element_size = sizeof(brgemm_batch_element_t);
    dim_t bs_pos = 0;

    emit_level(ir, runs, nest_cfg.bs_form, bs_pos, [&](dim_t, dim_t bs) {
        const ir::vreg_t a_ptr = ir.new_gpr();
        const ir::vreg_t b_ptr = ir.new_gpr();

        emit_batch_element_ptrs(
                ir, cfg, regs, batch_ptr, bs * element_size, a_ptr, b_ptr);
        emit_k_level(ir, cfg, nest_cfg, cp, blk, acc, a_ptr, b_ptr,
                k_step_interleave_at(interleave, bs, 1, blocking.max_bs));
    }, [&](dim_t bs) { ir.add_imm(batch_ptr, bs * element_size); });
}

// Broadcasts the common value at `ptr` to an f32 register and returns it once
// for each N register of `blk`, so a stage reads a common value as it reads a
// per-N one. `mem_dt` is the data type of the value in memory.
std::vector<ir::vreg_t> load_common(ir::ir_t &ir, const out_block_t &blk,
        ir::vreg_t ptr, data_type_t mem_dt) {

    const ir::vreg_t v = ir.new_vec(data_type::f32);
    ir.vload_bcast(v, ptr, 0, mem_dt);
    return std::vector<ir::vreg_t>(blk.ld_block2, v);
}

// Loads the per-N values of N register `ld` of `blk` from `ptr` at byte offset
// `off` into an f32 register. `mem_dt` is their data type in memory. Only the
// last N register is partial when `blk` has `ld_tail_mask`
// (`make_out_block()`).
ir::vreg_t load_per_n(ir::ir_t &ir, const out_block_t &blk, int ld,
        ir::vreg_t ptr, dim_t off, data_type_t mem_dt) {

    const ir::vreg_t v = ir.new_vec(data_type::f32);
    const bool is_tail
            = blk.ld_tail_mask != ir::vreg_t::none && ld == blk.ld_block2 - 1;

    if (is_tail) {
        ir.vload_masked(v, ptr, off, blk.ld_tail_mask, mem_dt);
    } else {
        ir.vload(v, ptr, off, mem_dt);
    }
    return v;
}

// Loads what `stage` reads for the rows of `blk`, one f32 register for each N
// register of `blk`. A common value is the same register for each. Returns no
// register for `injector_postops`, which loads its own arguments.
std::vector<ir::vreg_t> load_stage_regs(ir::ir_t &ir, const conf_t &cfg,
        const m_loop_input_regs_t &regs, const out_block_t &blk,
        output_stage_t stage) {

    const output_conf_t &output = cfg.output;
    std::vector<ir::vreg_t> stage_regs;

    switch (stage) {
        case output_stage_t::src_scale:
            stage_regs = load_common(ir, blk, regs.invariant.src_scale_ptr,
                    output.dt_src_scales);
            break;
        case output_stage_t::wei_scale:
            stage_regs = load_common(ir, blk, regs.advancing.wei_scale_ptr,
                    output.dt_wei_scales);
            break;
        case output_stage_t::wei_scale_per_n:
            for (int ld = 0; ld < blk.ld_block2; ld++) {
                const dim_t off = cfg.wei_scale_off(cfg.ld_pos(blk, ld));
                stage_regs.push_back(
                        load_per_n(ir, blk, ld, regs.advancing.wei_scale_ptr,
                                off, output.dt_wei_scales));
            }
            break;
        case output_stage_t::bias:
            for (int ld = 0; ld < blk.ld_block2; ld++) {
                const dim_t off = cfg.bias_off(cfg.ld_pos(blk, ld));
                stage_regs.push_back(load_per_n(ir, blk, ld,
                        regs.advancing.bias_ptr, off, output.dt_bias));
            }
            break;
        case output_stage_t::injector_postops: break;
        case output_stage_t::dst_scale:
            // The caller passes the destination scale as f32.
            stage_regs = load_common(
                    ir, blk, regs.invariant.dst_scale_ptr, data_type::f32);
            break;
    }
    return stage_regs;
}

// One stage of a post-ops call, applied to the rows of `blk` in `acc`.
// `stage_regs` holds what the stage reads for each N register of `blk`
// (`load_stage_regs()`).
void emit_post_ops_stage(ir::ir_t &ir, const conf_t &cfg,
        const store_conf_t &store_cfg, const m_loop_input_regs_t &regs,
        const out_block_t &blk, const std::vector<ir::vreg_t> &acc,
        output_stage_t stage, const std::vector<ir::vreg_t> &stage_regs) {

    switch (stage) {
        case output_stage_t::src_scale:
        case output_stage_t::wei_scale:
        case output_stage_t::wei_scale_per_n:
        case output_stage_t::dst_scale:
            // The caller passes the inverted destination scale, so every scale
            // is a multiply.
            for (int bd = 0; bd < blk.bd_block; bd++) {
                for (int ld = 0; ld < blk.ld_block2; ld++) {
                    ir.vmul(acc[bd * blk.ld_block2 + ld], stage_regs[ld]);
                }
            }
            break;
        case output_stage_t::bias:
            for (int bd = 0; bd < blk.bd_block; bd++) {
                for (int ld = 0; ld < blk.ld_block2; ld++) {
                    ir.vadd(acc[bd * blk.ld_block2 + ld], stage_regs[ld]);
                }
            }
            break;
        case output_stage_t::injector_postops: {
            // The injector gets the address of the block and the offset of
            // each accumulator in the block, as `jit_brgemm_kernel_t` passes
            // `reg_aux_D` and `D_offset()`. A binary post-op computes the
            // output coordinates from the address at run time and from the
            // offset at JIT time (`calculate_*_partial()`), and gets them
            // wrong for an offset outside the block. So a block that is not
            // at the pointers gets its address in its own register. A sum
            // post-op reads the previous value from the same address.
            ir::vreg_t blk_ptr = regs.advancing.store_ptr;
            if (blk.bd_start != 0 || blk.ld_start != 0) {
                blk_ptr = ir.new_gpr();
                ir.mov_reg(blk_ptr, regs.advancing.store_ptr);
                ir.add_imm(blk_ptr,
                        cfg.store_off(store_cfg, blk.bd_start, blk.ld_start));
            }

            std::vector<dim_t> out_byte_off(acc.size());
            for (int bd = 0; bd < blk.bd_block; bd++)
                for (int ld = 0; ld < blk.ld_block2; ld++)
                    out_byte_off[bd * blk.ld_block2 + ld] = cfg.store_off(
                            store_cfg, bd, (dim_t)ld * cfg.blocking.ld_block);

            // A block is either all full registers or the one masked
            // register of the N tail, which holds `ldb_tail` elements.
            const int n_elems = blk.ld_tail_mask != ir::vreg_t::none
                    ? cfg.blocking.ldb_tail
                    : cfg.blocking.ld_block;
            ir.inject_postops(acc, blk_ptr, out_byte_off, n_elems);
            break;
        }
    }
}

// The output path of the block `blk`, a whole output block or a part of one
// (`pass_accumulators_to_output_path()`). Adds C when beta is 1, applies the
// output stages of `store_cfg`, and stores the block to C or D as `store_cfg`
// says.
//
// It works on the rows of the output in `acc`, one register per (M row, N
// register).
//
// For a part of a block passed in parts, `blk_stage_regs` holds what each
// stage of `store_cfg.stages` reads for `blk`, loaded before the first part.
// Otherwise it is empty, and each stage loads what it reads right before it is
// applied.
void emit_output_path(ir::ir_t &ir, const conf_t &cfg,
        const store_conf_t &store_cfg, const m_loop_input_regs_t &regs,
        const out_block_t &blk, const std::vector<ir::vreg_t> &acc,
        const std::vector<std::vector<ir::vreg_t>> &blk_stage_regs) {

    // The kernel takes only 0 and 1 for beta. On the output path the
    // accumulators start at zero (`emit_block()`), so C is added here.
    if (cfg.output.beta != 0.0f) {
        for (int bd = 0; bd < blk.bd_block; bd++) {
            for (int ld = 0; ld < blk.ld_block2; ld++) {
                const ir::vreg_t c = ir.new_vec(cfg.output.dt_c);
                if (blk.ld_tail_mask == ir::vreg_t::none)
                    ir.vload(c, regs.advancing.c_ptr,
                            cfg.c_off(blk.bd_start + bd, cfg.ld_pos(blk, ld)),
                            cfg.output.dt_c);
                else
                    ir.vload_masked(c, regs.advancing.c_ptr,
                            cfg.c_off(blk.bd_start + bd, cfg.ld_pos(blk, ld)),
                            blk.ld_tail_mask, cfg.output.dt_c);
                ir.vadd(acc[bd * blk.ld_block2 + ld], c);
            }
        }
    }

    for (size_t i = 0; i < store_cfg.stages.size(); i++) {
        const output_stage_t stage = store_cfg.stages[i];
        const std::vector<ir::vreg_t> stage_regs = blk_stage_regs.empty()
                ? load_stage_regs(ir, cfg, regs, blk, stage)
                : blk_stage_regs[i];
        emit_post_ops_stage(
                ir, cfg, store_cfg, regs, blk, acc, stage, stage_regs);
    }

    for (int bd = 0; bd < blk.bd_block; bd++) {
        for (int ld = 0; ld < blk.ld_block2; ld++) {
            const ir::vreg_t src = acc[bd * blk.ld_block2 + ld];
            const dim_t off = cfg.store_off(
                    store_cfg, blk.bd_start + bd, cfg.ld_pos(blk, ld));

            if (blk.ld_tail_mask == ir::vreg_t::none) {
                ir.vstore(regs.advancing.store_ptr, off, src,
                        cfg.store_dt(store_cfg));
            } else {
                ir.vstore_masked(regs.advancing.store_ptr, off, src,
                        blk.ld_tail_mask, cfg.store_dt(store_cfg));
            }
        }
    }
}

// True if `emit_output_path()` stores the blocks instead of
// `store_accumulators()`. Only the output path applies output stages and
// converts the accumulators.
bool uses_output_path(const compute_path_t &cp, const store_conf_t &store_cfg) {
    return !cp.stores_accumulators() || !store_cfg.stages.empty()
            || store_cfg.convert_acc;
}

// Returns what each output stage reads for the rows `rows_blk` of `blk`, out of
// `blk_stage_regs`, which holds what it reads for all of `blk`. Empty when
// `blk_stage_regs` is.
std::vector<std::vector<ir::vreg_t>> rows_stage_regs(const conf_t &cfg,
        const out_block_t &blk, const out_block_t &rows_blk,
        const std::vector<std::vector<ir::vreg_t>> &blk_stage_regs) {

    // The rows start at N register `ld_first` of the block.
    const int ld_first
            = (int)((rows_blk.ld_start - blk.ld_start) / cfg.blocking.ld_block);
    std::vector<std::vector<ir::vreg_t>> stage_regs(blk_stage_regs.size());

    for (size_t i = 0; i < blk_stage_regs.size(); i++) {
        // `injector_postops` reads no register.
        if (blk_stage_regs[i].empty()) continue;

        const auto first = blk_stage_regs[i].begin() + ld_first;
        stage_regs[i].assign(first, first + rows_blk.ld_block2);
    }
    return stage_regs;
}

// One output block of the loop nest of `nest_cfg`.
//
// Creates the accumulators of the block with the compute path and reduces them
// over the batch. Then they are stored as the store configuration of the nest
// says. When `uses_output_path()` is true, the output path stores them.
// Otherwise the compute path stores them to C, in the K steps when
// `acc_stores` is not `none` and after them when it is.
//
// With `store_cfg.prefetch`, the block prefetches the lines it stores, in its K
// steps when `has_unrolled_k_steps()` is true and before the batch level
// otherwise.
//
// `prev` is the previous block. The first K step of this block stores the rest
// of the accumulators of `prev`. Returns this block when `acc_stores` is
// `next_block`, and an empty `prev_block_t` otherwise.
prev_block_t emit_block(ir::ir_t &ir, const conf_t &cfg,
        const nest_conf_t &nest_cfg, const compute_path_t &cp,
        const m_loop_input_regs_t &regs, const out_block_t &blk,
        acc_stores_t acc_stores, const prev_block_t &prev) {

    const store_conf_t &store_cfg = nest_cfg.store;

    // On the output path C is added to the rows, which is faster than
    // starting the accumulators from C.
    const bool to_output_path = uses_output_path(cp, store_cfg);
    const bool acc_from_c = cfg.output.beta != 0.0f && cp.loads_accumulators()
            && !to_output_path;
    const std::vector<ir::vreg_t> acc = acc_from_c
            ? cp.load_accumulators(ir, cfg, blk, regs.advancing.c_ptr, prev)
            : cp.init_accumulators(ir, cfg, blk, prev);

    std::vector<std::vector<prefetch_t>> prefetches;
    if (store_cfg.prefetch)
        prefetches = cp.store_prefetches(
                cfg, nest_cfg, blk, regs.advancing.store_ptr);

    k_step_interleave_t interleave;
    interleave.prev = prev;
    interleave.acc_stores = acc_stores;
    interleave.c_ptr = acc_from_c ? regs.advancing.c_ptr : ir::vreg_t::none;
    interleave.store_ptr = regs.advancing.store_ptr;

    // A K step in a loop would issue its prefetches on every iteration, and an
    // empty batch has no K step.
    if (has_unrolled_k_steps(cfg, nest_cfg)) {
        interleave.prefetches = std::move(prefetches);
    } else {
        for (const std::vector<prefetch_t> &acc_prefetches : prefetches) {
            for (const prefetch_t &p : acc_prefetches) {
                ir.prefetch(p.ptr, p.off, p.kind);
            }
        }
    }

    emit_batch_level(ir, cfg, nest_cfg, cp, regs, blk, acc, interleave);

    if (to_output_path) {
        // For a block passed in parts, what the stages read is loaded once,
        // before the first part. For a block passed in one call, the output
        // path loads what each stage reads right before the stage, since on the
        // vector compute path the accumulators take most vector registers
        // (`calculate_max_bcast_block()`).
        std::vector<std::vector<ir::vreg_t>> blk_stage_regs;
        if (cp.passes_block_in_parts()) {
            for (const output_stage_t stage : store_cfg.stages) {
                blk_stage_regs.push_back(
                        load_stage_regs(ir, cfg, regs, blk, stage));
            }
        }

        cp.pass_accumulators_to_output_path(ir, cfg, blk, acc,
                [&](const out_block_t &rows_blk,
                        const std::vector<ir::vreg_t> &row_regs) {
            emit_output_path(ir, cfg, store_cfg, regs, rows_blk, row_regs,
                    rows_stage_regs(cfg, blk, rows_blk, blk_stage_regs));
        });
    } else if (acc_stores == acc_stores_t::none) {
        cp.store_accumulators(ir, cfg, blk, acc, regs.advancing.store_ptr);
    }

    if (acc_stores != acc_stores_t::next_block) return {};
    return {blk, acc};
}

// Returns the block of `bd_block` rows at `bd_start` and `ld_elems` elements of
// N at `ld_start`. The last vector register or tile is partial when `ld_elems`
// is not a multiple of `ld_block`, and the block then gets `ld_tail_mask`.
out_block_t make_out_block(const conf_t &cfg, dim_t bd_block, dim_t bd_start,
        dim_t ld_elems, dim_t ld_start, ir::vreg_t ld_tail_mask) {

    const bool is_ld_tail = ld_elems % cfg.blocking.ld_block != 0;
    return {(int)bd_block, (int)utils::div_up(ld_elems, cfg.blocking.ld_block),
            bd_start, ld_start, is_ld_tail ? ld_tail_mask : ir::vreg_t::none};
}

// Moves the pointers of the M and N levels by `bd` rows and `ld` elements of
// N. C and D move along M and N with one `add` each. The A offset depends only
// on M, and the B offset, the bias, and a per-N weights scale only on N.
void emit_mn_step(ir::ir_t &ir, const conf_t &cfg,
        const store_conf_t &store_cfg, const advancing_regs_t &regs, dim_t bd,
        dim_t ld) {

    if (bd != 0) ir.add_imm(regs.a_off, cfg.a_off(bd, 0));
    if (ld != 0) ir.add_imm(regs.b_off, cfg.b_off(0, ld));

    if (regs.c_ptr != ir::vreg_t::none)
        ir.add_imm(regs.c_ptr, cfg.c_off(bd, ld));
    if (store_cfg.to_d) ir.add_imm(regs.store_ptr, cfg.d_off(bd, ld));

    if (ld != 0) {
        if (store_cfg.has_stage(output_stage_t::bias))
            ir.add_imm(regs.bias_ptr, cfg.bias_off(ld));
        if (store_cfg.has_stage(output_stage_t::wei_scale_per_n))
            ir.add_imm(regs.wei_scale_ptr, cfg.wei_scale_off(ld));
    }
}

// Returns what the last K step of the block that ends at row `bd_end` and N
// element `ld_end` stores, see `acc_stores_t`.
//
// It is `none` without `nest_cfg.store.interleave_acc_stores` or when
// `has_unrolled_k_steps()` is false. Otherwise it is `next_block` when the
// next block follows with no loop between them, and `block` otherwise. With N
// inside M, the next block is the next N block of an unrolled N level, or the
// first N block of the next M block when both levels are unrolled. The same
// holds with M inside N. Positions are compared only for unrolled levels,
// which count them from their first block.
acc_stores_t block_acc_stores(const conf_t &cfg, const nest_conf_t &nest_cfg,
        dim_t bd_end, dim_t ld_end) {

    if (!nest_cfg.store.interleave_acc_stores
            || !has_unrolled_k_steps(cfg, nest_cfg)) {
        return acc_stores_t::none;
    }

    const bool is_m_unrolled = nest_cfg.m_form == level_form_t::unrolled;
    const bool is_n_unrolled = nest_cfg.n_form == level_form_t::unrolled;
    const bool has_m_next = bd_end < level_size(m_runs(cfg.blocking));
    const bool has_n_next = ld_end < level_size(n_runs(cfg.blocking));

    const bool has_next_block
            = nest_cfg.innermost_loop == brgemm_ld_loop_innermost
            ? is_n_unrolled && (has_n_next || (is_m_unrolled && has_m_next))
            : is_m_unrolled && (has_m_next || (is_n_unrolled && has_n_next));
    return has_next_block ? acc_stores_t::next_block : acc_stores_t::block;
}

// Emits the loop nest of `nest_cfg` over the output blocks.
//
// `innermost_loop` decides which of the M and N levels is inside the other, as
// in `top_loop()` of the uker.
void emit_nest(ir::ir_t &ir, const conf_t &cfg, const nest_conf_t &nest_cfg,
        const compute_path_t &cp) {

    const m_loop_input_regs_t regs = init_m_loop_input_regs(ir, cfg, nest_cfg);
    const blocking_conf_t &blocking = cfg.blocking;
    const store_conf_t &store_cfg = nest_cfg.store;

    // The partial vector register or tile of every block has `ldb_tail`
    // columns, so one mask serves the whole kernel.
    ir::vreg_t ld_tail_mask = ir::vreg_t::none;

    if (blocking.ldb_tail > 0) {
        ld_tail_mask = ir.new_mask();
        ir.set_mask_imm(ld_tail_mask, blocking.ldb_tail);
    }

    const std::vector<block_run_t> m = m_runs(blocking);
    const std::vector<block_run_t> n = n_runs(blocking);

    // Where the M and N pointers stand, in elements from the first block.
    dim_t m_pos = 0, n_pos = 0;

    // The last block emitted, if some of its accumulators are still to be
    // stored.
    prev_block_t prev;

    if (nest_cfg.innermost_loop == brgemm_ld_loop_innermost) {
        emit_level(ir, m, nest_cfg.m_form, m_pos, n_pos,
                [&](dim_t bd_block, dim_t bd_start) {
            emit_level(ir, n, nest_cfg.n_form, n_pos,
                    [&](dim_t ld_elems, dim_t ld_start) {
                const out_block_t blk = make_out_block(cfg, bd_block, bd_start,
                        ld_elems, ld_start, ld_tail_mask);
                const acc_stores_t acc_stores = block_acc_stores(cfg, nest_cfg,
                        bd_start + bd_block, ld_start + ld_elems);
                prev = emit_block(
                        ir, cfg, nest_cfg, cp, regs, blk, acc_stores, prev);
            }, [&](dim_t ld) {
                emit_mn_step(ir, cfg, store_cfg, regs.advancing, 0, ld);
            });
        }, [&](dim_t bd, dim_t ld) {
            emit_mn_step(ir, cfg, store_cfg, regs.advancing, bd, ld);
        });
    } else {
        emit_level(ir, n, nest_cfg.n_form, n_pos, m_pos,
                [&](dim_t ld_elems, dim_t ld_start) {
            emit_level(ir, m, nest_cfg.m_form, m_pos,
                    [&](dim_t bd_block, dim_t bd_start) {
                const out_block_t blk = make_out_block(cfg, bd_block, bd_start,
                        ld_elems, ld_start, ld_tail_mask);
                const acc_stores_t acc_stores = block_acc_stores(cfg, nest_cfg,
                        bd_start + bd_block, ld_start + ld_elems);
                prev = emit_block(
                        ir, cfg, nest_cfg, cp, regs, blk, acc_stores, prev);
            }, [&](dim_t bd) {
                emit_mn_step(ir, cfg, store_cfg, regs.advancing, bd, 0);
            });
        }, [&](dim_t ld, dim_t bd) {
            emit_mn_step(ir, cfg, store_cfg, regs.advancing, bd, ld);
        });
    }

    // The last block never gets `next_block`, so no stores are left.
    assert(prev.acc.empty());
}

// Builds IR for BRGEMM.
//
// Computes:
//   C[i][j] = beta * C[i][j] + sum_bs sum_k A[i][k] * B[k][j]
//   (m = `bcast_dim`, n = `load_dim`, k = `reduce_dim` of the descriptor)
//
// A call that asks for post-ops applies the scales, bias and attribute
// post-ops, and stores to D instead of C:
//   D[i][j] = post_ops(C[i][j] * src_scale * wei_scale[j] + bias[j])
//             * dst_scale
//
// The output is partitioned into M blocks of `bd_block` rows, each split into
// N blocks of `ld_block2 * ld_block` columns. Each block keeps its result in
// registers across the whole batch and K reduction, then stores it once. The
// `level_form_t` of each level decides how the loop nest walks the blocks.
//
// When post-ops are applicable, the kernel has one loop nest per value of
// `do_post_ops` and picks one on entry. Each nest is emitted from its own
// configuration, so no code in it depends on the kind of call.
void build_brgemm(const conf_t &cfg, const compute_path_t &cp, ir::ir_t &ir) {
    if (!cfg.are_post_ops_applicable()) {
        emit_nest(ir, cfg, cfg.nest_without_post_ops, cp);
        return;
    }

    const ir::vreg_t do_post_ops = ir.new_gpr();
    ir.load_param(do_post_ops, GET_OFF(do_post_ops));

    const int if_post_ops = ir.if_begin(do_post_ops, ir::cmp_t::ne, 0);
    emit_nest(ir, cfg, cfg.nest_with_post_ops, cp);
    ir._else(if_post_ops);
    emit_nest(ir, cfg, cfg.nest_without_post_ops, cp);
    ir.if_end(if_post_ops);
}

// Creates the compute path of `cfg` for building `ir`.
std::unique_ptr<compute_path_t> make_compute_path(
        ir::ir_t &ir, const conf_t &cfg) {
    switch (cfg.compute_path.kind) {
        case compute_path_kind_t::vector: return make_vector_compute_path();
        case compute_path_kind_t::tmul: return make_tmul_compute_path(ir, cfg);
    }
    assert(!"unknown compute path");
    return nullptr;
}

// Returns true if `runs` has a run of more than one block, which a looped
// level emits as a loop.
bool has_loop(const std::vector<block_run_t> &runs) {
    for (const block_run_t &run : runs) {
        if (run.count > 1) return true;
    }
    return false;
}

// Returns the number of levels that have a loop.
int n_levels_with_loop(const blocking_conf_t &blocking) {
    int n = 0;
    for (const std::vector<block_run_t> &runs : {m_runs(blocking),
                 n_runs(blocking), bs_runs(blocking), k_runs(blocking)}) {
        if (has_loop(runs)) n++;
    }
    return n;
}

// Returns the `level_form_t` of a level with `runs` when the next `n_unrolled`
// levels that have a loop, from this one outward, are unrolled. Decrements
// `n_unrolled` if the level is one of them. A level without a loop is
// unrolled. Looped, it would emit the same blocks and could add pointer setup.
level_form_t next_level_form(
        const std::vector<block_run_t> &runs, int &n_unrolled) {

    if (!has_loop(runs)) return level_form_t::unrolled;
    if (n_unrolled == 0) return level_form_t::looped;
    n_unrolled--;
    return level_form_t::unrolled;
}

// Unrolls the `n` innermost levels of the nest of `nest_cfg` that have a loop
// and loops the other levels that have one. From the inside, the levels are K,
// the batch, then N and M if N is inside M (`brgemm_ld_loop_innermost`), M and
// N otherwise.
void set_level_forms(
        nest_conf_t &nest_cfg, const blocking_conf_t &blocking, int n) {

    nest_cfg.k_form = next_level_form(k_runs(blocking), n);
    nest_cfg.bs_form = next_level_form(bs_runs(blocking), n);

    if (nest_cfg.innermost_loop == brgemm_ld_loop_innermost) {
        nest_cfg.n_form = next_level_form(n_runs(blocking), n);
        nest_cfg.m_form = next_level_form(m_runs(blocking), n);
    } else {
        nest_cfg.m_form = next_level_form(m_runs(blocking), n);
        nest_cfg.n_form = next_level_form(n_runs(blocking), n);
    }
}

// Returns how many of the innermost levels that have a loop the nest of
// `nest_cfg` unrolls within half of L1i, and builds into `looped_ir` the IR of
// the nest alone with no loop unrolled.
//
// From `looped_ir`, `ir::n_ops_unrolled()` predicts the op count with the `n`
// innermost levels that have a loop unrolled, and `n` grows while that count
// times `code_bytes_per_op()` fits in half of L1i. The prediction still counts
// the pointer steps of the levels it unrolls, so the nest built with `n` levels
// unrolled is never larger than predicted.
int n_levels_to_unroll(
        const conf_t &cfg, nest_conf_t nest_cfg, ir::ir_t &looped_ir) {

    set_level_forms(nest_cfg, cfg.blocking, 0);
    const std::unique_ptr<compute_path_t> cp
            = make_compute_path(looped_ir, cfg);
    emit_nest(looped_ir, cfg, nest_cfg, *cp);

    const unsigned l1i_size = platform::get_per_core_cache_size(
            1, platform::cache_type_t::instruction);
    // TODO: tune the budget. Half of L1i is a safe starting point.
    const double budget = l1i_size / 2.0;
    const double bytes_per_op = cp->code_bytes_per_op();
    const int n_loop_levels = n_levels_with_loop(cfg.blocking);

    int n = 0;
    while (n < n_loop_levels) {
        const dim_t n_ops = ir::n_ops_unrolled(looped_ir, n + 1);
        if ((double)n_ops * bytes_per_op > budget) break;
        n++;
    }
    return n;
}

// Builds the IR of `cfg` with as many innermost levels of each nest unrolled
// as fit in half of L1i.
//
// A call runs one nest, so `n_levels_to_unroll()` sizes each nest on its own,
// and each nest can use the whole budget. The kernel is then built with the
// level forms of each nest. A kernel with one nest that unrolls no level is
// the IR that sized its nest, and is not built again.
void build_unrolled_ir(conf_t cfg, ir::ir_t &ir) {
    ir::ir_t looped_ir_without_post_ops;
    const int n_without_post_ops = n_levels_to_unroll(
            cfg, cfg.nest_without_post_ops, looped_ir_without_post_ops);

    if (!cfg.are_post_ops_applicable() && n_without_post_ops == 0) {
        ir = std::move(looped_ir_without_post_ops);
        return;
    }
    set_level_forms(
            cfg.nest_without_post_ops, cfg.blocking, n_without_post_ops);

    if (cfg.are_post_ops_applicable()) {
        ir::ir_t looped_ir_with_post_ops;
        const int n_with_post_ops = n_levels_to_unroll(
                cfg, cfg.nest_with_post_ops, looped_ir_with_post_ops);
        set_level_forms(cfg.nest_with_post_ops, cfg.blocking, n_with_post_ops);
    }

    const std::unique_ptr<compute_path_t> cp = make_compute_path(ir, cfg);
    build_brgemm(cfg, *cp, ir);
}

// Picks the compute path the caller chose when it created the descriptor, and
// fills the fields of the compute path configuration that do not depend on the
// compute path: the batch kind and the leading dimensions of A and B.
//
// The finalized descriptor already holds the choice (`is_ace()`, `is_tmm`), so
// it is read from there. The ACE compute path does not exist yet and is
// rejected.
status_t init_compute_path(
        compute_path_conf_t &compute_path, const brgemm_desc_t &brg) {
    using namespace utils;

    VCONDCHECK_BRGEMM_IR(!brg.is_ace(), VERBOSE_UNSUPPORTED_FEATURE, "ACE");
    VCONDCHECK_BRGEMM_IR(one_of(brg.type, brgemm_addr, brgemm_offs),
            VERBOSE_UNSUPPORTED_FEATURE, "batch kind");
    VCONDCHECK_BRGEMM_IR(brg.layout == brgemm_row_major,
            VERBOSE_UNSUPPORTED_FEATURE, "column-major layout");
    VCONDCHECK_BRGEMM_IR(
            !brg.is_blocked, VERBOSE_UNSUPPORTED_FEATURE, "blocked layout");
    VCONDCHECK_BRGEMM_IR(
            everyone_is(false, brg.is_runtime_lda, brg.is_runtime_ldb),
            VERBOSE_UNSUPPORTED_FEATURE, "runtime leading dimension");

    compute_path.kind = brg.is_tmm ? compute_path_kind_t::tmul
                                   : compute_path_kind_t::vector;
    compute_path.batch_kind = brg.type;
    compute_path.lda = brg.LDA;
    compute_path.ldb = brg.LDB;
    return status::success;
}

// Fills the blocking configuration.
//
// A tile descriptor counts its blocks differently from a vector one. `bdb` and
// `ldb` include the tail block, and the uker groups `bd_block2` M blocks and
// `ld_block2` N blocks into one block of the loop nest, with the tail block in
// the last group of its level (`fill_imap()`).
status_t init_blocking_conf(blocking_conf_t &blocking, compute_path_kind_t kind,
        const brgemm_desc_t &brg) {
    using namespace utils;
    const brgemm_attr_t &brgattr = brg.brgattr;

    VCONDCHECK_BRGEMM_IR(!brgattr.var_bs, VERBOSE_UNSUPPORTED_FEATURE,
            "variable batch size");
    VCONDCHECK_BRGEMM_IR(
            everyone_is(0, brgattr.max_top_vpad, brgattr.max_bottom_vpad),
            VERBOSE_UNSUPPORTED_PAD_FEATURE, "virtual padding");
    VCONDCHECK_BRGEMM_IR(
            everyone_is(0, brgattr.max_top_bpad, brgattr.max_bottom_bpad),
            VERBOSE_UNSUPPORTED_PAD_FEATURE, "batch padding");
    VCONDCHECK_BRGEMM_IR(
            brgattr.bd_mask_level == 0, VERBOSE_UNSUPPORTED_FEATURE, "bd_mask");

    // When `can_dispatch_uker()` is true, `use_uker` is set and `max_bs` is the
    // only batch size of a call. Otherwise the kernel reads `BS`.
    blocking.var_bs = !brg.can_dispatch_uker();

    const bool is_tmul = kind == compute_path_kind_t::tmul;
    VCONDCHECK_BRGEMM_IR(IMPLICATION(is_tmul, !blocking.var_bs),
            VERBOSE_UNSUPPORTED_FEATURE, "variable batch size with tiles");

    blocking.max_bs = brgattr.max_bs;
    blocking.ld_block = brg.ld_block;
    blocking.ld_block2 = brg.ld_block2;
    blocking.rd_block = brg.rd_block;
    blocking.rdb = brg.rdb;
    blocking.rdb_tail = brg.rdb_tail;

    switch (kind) {
        case compute_path_kind_t::vector:
            blocking.bd_block = brg.bd_block;
            blocking.bdb = brg.bdb;
            blocking.bdb_tail = brg.bdb_tail;
            blocking.ldb2 = brg.ldb2;
            blocking.ldb2_tail = brg.ldb2_tail;
            blocking.ldb_tail = brg.ldb_tail;
            // The vector microkernel and the output path treat every vector
            // register of a block that has `ld_tail_mask` as partial, so the
            // partial vector register is a block of its own.
            blocking.n_tail_grouping = tail_grouping_t::separate;
            break;
        case compute_path_kind_t::tmul: {
            VCONDCHECK_BRGEMM_IR(brg.rdb_tail == 0, VERBOSE_UNSUPPORTED_FEATURE,
                    "K tail with tiles");

            // The last block of the level holds the full tiles left over and
            // the tail tile, so `bdb_tail` counts the rows of both.
            const dim_t full_m_tiles = brg.bdb - (brg.bdb_tail != 0 ? 1 : 0);
            const dim_t full_n_tiles = brg.ldb - (brg.ldb_tail != 0 ? 1 : 0);

            blocking.bd_block = brg.bd_block2 * brg.bd_block;
            blocking.bdb = full_m_tiles / brg.bd_block2;
            blocking.bdb_tail
                    = (int)(brg.bcast_dim - blocking.bdb * blocking.bd_block);
            blocking.ldb2 = full_n_tiles / brg.ld_block2;
            blocking.ldb2_tail = (int)(full_n_tiles % brg.ld_block2);
            blocking.ldb_tail = brg.ldb_tail;
            // A separate block for the partial tile would load the A tiles of
            // the M block again at every K step.
            blocking.n_tail_grouping = tail_grouping_t::grouped;
            break;
        }
    }
    return status::success;
}

// Fills the configuration of the vector compute path.
status_t init_vector_conf(
        compute_path_conf_t &compute_path, const brgemm_desc_t &brg) {
    using namespace data_type;

    // `avx2` and `avx512_core` compute f32 directly. The other ISAs receive
    // inputs already converted to f32.
    VCONDCHECK_BRGEMM_IR(utils::one_of(brg.isa_impl, avx2, avx512_core,
                                 avx512_core_fp16, avx10_2),
            VERBOSE_UNSUPPORTED_ISA);
    VCONDCHECK_BRGEMM_IR(utils::everyone_is(f32, brg.dt_a, brg.dt_b),
            VERBOSE_UNSUPPORTED_DT);
    VCONDCHECK_BRGEMM_IR(!brg.n_bcast_1_load, VERBOSE_UNSUPPORTED_FEATURE,
            "one-load microkernel");
    VCONDCHECK_BRGEMM_IR(!brg.embd_bcst, VERBOSE_UNSUPPORTED_FEATURE,
            "embedded broadcast microkernel");

    compute_path.dt_a = brg.dt_a;
    compute_path.dt_b = brg.dt_b;
    compute_path.dt_acc = f32;
    compute_path.vnni_granularity = brg.rd_step;
    // A is broadcast with no hint, as in `jit_brgemm_kernel_t`.
    if (brg.read_shared_B())
        compute_path.b_load_hint |= ir::mem_hint_t::read_shared;
    return status::success;
}

// Fills the configuration of the TMUL compute path.
//
// The caller loads the tile configuration from `brgemm_init_tiles()` before it
// calls the kernel. The kernel converts the same configuration into `tiles`
// for code generation and takes the shapes of the operand tiles from it.
status_t init_tmul_conf(compute_path_conf_t &compute_path,
        ir::tile_config_t &tiles, const brgemm_desc_t &brg) {
    using namespace data_type;

    VCONDCHECK_BRGEMM_IR(
            is_superset(brg.isa_impl, amx_bf16), VERBOSE_UNSUPPORTED_ISA);
    VCONDCHECK_BRGEMM_IR(utils::everyone_is(bf16, brg.dt_a, brg.dt_b),
            VERBOSE_UNSUPPORTED_DT);
    VCONDCHECK_BRGEMM_IR(
            !brg.fused_copy_a, VERBOSE_UNSUPPORTED_FEATURE, "fused copy A");

    palette_config_t palette;
    const status_t st = brgemm_init_tiles(brg, (char *)&palette);
    VCONDCHECK_BRGEMM_IR(st == status::success, VERBOSE_UNSUPPORTED_FEATURE,
            "tile configuration");

    std::vector<ir::tile_shape_t> &shapes = tiles.shapes;
    for (int t = 0; t < brgemm_desc_t::AMX_TILES_NUM; t++) {
        shapes.push_back({palette.rows[t], palette.cols[t]});
    }

    compute_path.dt_a = brg.dt_a;
    compute_path.dt_b = brg.dt_b;
    compute_path.dt_acc = f32;
    compute_path.vnni_granularity = brg.rd_step;

    for (const bool is_m_tail : {false, true}) {
        compute_path.a_tile[is_m_tail] = shapes[brg.get_A_tensor(0, is_m_tail)];
        for (const bool is_n_tail : {false, true}) {
            compute_path.c_tile[is_m_tail][is_n_tail]
                    = shapes[brg.get_C_tensor(0, 0, is_m_tail, is_n_tail)];
        }
    }
    for (const bool is_n_tail : {false, true}) {
        compute_path.b_tile[is_n_tail] = shapes[brg.get_B_tensor(0, is_n_tail)];
    }

    if (brg.load_nt_A) compute_path.a_load_hint |= ir::mem_hint_t::non_temporal;
    if (brg.load_nt_B) compute_path.b_load_hint |= ir::mem_hint_t::non_temporal;

    if (brg.read_shared_A())
        compute_path.a_load_hint |= ir::mem_hint_t::read_shared;
    if (brg.read_shared_B())
        compute_path.b_load_hint |= ir::mem_hint_t::read_shared;
    return status::success;
}

// Fills the output path configuration.
status_t init_output_conf(output_conf_t &output, const brgemm_desc_t &brg) {
    using namespace data_type;
    using namespace utils;
    const brgemm_attr_t &brgattr = brg.brgattr;

    VCONDCHECK_BRGEMM_IR(brg.dt_c == f32, VERBOSE_UNSUPPORTED_DT);
    VCONDCHECK_BRGEMM_IR(one_of(brg.dt_d, f32, bf16), VERBOSE_UNSUPPORTED_DT);
    VCONDCHECK_BRGEMM_IR(
            brg.alpha == 1.0f, VERBOSE_UNSUPPORTED_FEATURE, "alpha != 1");
    VCONDCHECK_BRGEMM_IR(brg.beta == 0.0f || brg.beta == 1.0f,
            VERBOSE_UNSUPPORTED_FEATURE, "beta != 0 && beta != 1");

    VCONDCHECK_BRGEMM_IR(!brg.req_s8s8_compensation,
            VERBOSE_UNSUPPORTED_FEATURE, "s8s8 compensation");
    VCONDCHECK_BRGEMM_IR(!brgattr.use_intermediate_c_buffer,
            VERBOSE_UNSUPPORTED_FEATURE, "intermediate C buffer");
    VCONDCHECK_BRGEMM_IR(everyone_is(brgemm_broadcast_t::none, brg.zp_type_a,
                                 brg.zp_type_b, brg.zp_type_c),
            VERBOSE_UNSUPPORTED_ZP_CFG);
    VCONDCHECK_BRGEMM_IR(!brgattr.generate_skip_accumulation,
            VERBOSE_UNSUPPORTED_FEATURE, "skip accumulation");
    VCONDCHECK_BRGEMM_IR(brgattr.hint_prefetchw == brgemm_prfw_default,
            VERBOSE_UNSUPPORTED_FEATURE, "store prefetch hint");

    // Post-ops go through the JIT injector. Accept eltwise, binary and sum, and
    // only binary arguments the injector can handle on this ISA.
    if (brg.attr()) {
        const memory_desc_wrapper dst_d(brg.dst_md());
        const std::vector<injector::post_op_type> accepted
                = {injector::eltwise, injector::binary, injector::sum};
        VCONDCHECK_BRGEMM_IR(
                injector::post_ops_ok({brg.isa_impl, accepted,
                        brg.attr()->post_ops_, &dst_d,
                        /*sum_at_pos_0_only=*/false,
                        /*sum_requires_scale_one=*/false,
                        /*sum_requires_zp_zero=*/false,
                        /*sum_requires_same_params=*/false,
                        binary_injector::
                                get_all_strategies_supported_by_injector()}),
                VERBOSE_UNSUPPORTED_POSTOP);

        // The binary injector loads an fp8 argument through fp8 converters,
        // which `ir::postops_injector_t` does not create.
        const post_ops_t &post_ops = brg.attr()->post_ops_;
        for (int i = 0; i < post_ops.len(); i++) {
            const auto &e = post_ops.entry_[i];
            VCONDCHECK_BRGEMM_IR(!e.is_binary()
                            || !one_of(e.binary.src1_desc.data_type, f8_e5m2,
                                    f8_e4m3),
                    VERBOSE_UNSUPPORTED_POSTOP);
        }
    }

    // Scales are f32 values the kernel broadcasts or loads per N register:
    // src and dst scales are common, the weights scale common or per-N.
    VCONDCHECK_BRGEMM_IR(
            !brg.has_per_call_acc(), VERBOSE_UNSUPPORTED_SCALES_CFG);
    VCONDCHECK_BRGEMM_IR(
            IMPLICATION(brg.with_src_scales, brg.dt_src_scales == f32),
            VERBOSE_UNSUPPORTED_SCALES_CFG);
    VCONDCHECK_BRGEMM_IR(
            IMPLICATION(brg.with_wei_scales, brg.dt_wei_scales == f32),
            VERBOSE_UNSUPPORTED_SCALES_CFG);
    VCONDCHECK_BRGEMM_IR(!brg.with_weights_scale_adjust,
            VERBOSE_UNSUPPORTED_FEATURE, "weights scale adjust");

    VCONDCHECK_BRGEMM_IR(
            IMPLICATION(brg.with_bias, one_of(brg.dt_bias, f32, bf16)),
            VERBOSE_UNSUPPORTED_BIAS_CFG);
    VCONDCHECK_BRGEMM_IR(
            everyone_is(false, brg.is_runtime_ldc, brg.is_runtime_ldd),
            VERBOSE_UNSUPPORTED_FEATURE, "runtime leading dimension");

    output.beta = brg.beta;
    output.dt_c = brg.dt_c;
    output.dt_d = brg.dt_d;
    output.ldc = brg.LDC;
    output.ldd = brg.LDD;
    output.dt_bias = brg.dt_bias;
    output.dt_src_scales = brg.dt_src_scales;
    output.dt_wei_scales = brg.dt_wei_scales;

    if (brg.with_eltwise || brg.with_binary || brg.with_sum) {
        output.post_ops = brg.attr()->post_ops_;
        output.dst_md = *brg.dst_md();
    }
    return status::success;
}

// Fills the nest configurations, one per value of `do_post_ops`. Both nests
// start with the loop order and the level forms of the classic kernel. A call
// with post-ops applies the output stages and stores to D. A call without them
// stores to C.
status_t init_nest_confs(
        conf_t &conf, compute_path_kind_t kind, const brgemm_desc_t &brg) {
    const brgemm_attr_t &brgattr = brg.brgattr;
    const bool is_tmul = kind == compute_path_kind_t::tmul;
    const data_type_t dt_acc = conf.compute_path.dt_acc;
    const bool can_dispatch_uker = brg.can_dispatch_uker();

    // Reject the uker features that the kernel does not have yet, under the
    // conditions in which the uker uses them.
    if (can_dispatch_uker) {
        auto prefetches = [](const brgemm_prf_t &prf) {
            return prf.sprinkled || prf.dist0 >= 0 || prf.dist1 >= 0
                    || prf.dist2 >= 0 || prf.distNTA >= 0;
        };

        VCONDCHECK_BRGEMM_IR(!prefetches(brg.prfA), VERBOSE_UNSUPPORTED_FEATURE,
                "A prefetch");
        VCONDCHECK_BRGEMM_IR(!prefetches(brg.prfB), VERBOSE_UNSUPPORTED_FEATURE,
                "B prefetch");
        // A `prfC.dist0` above 0 prefetches the lines of a later block in the
        // loop order of the uker.
        VCONDCHECK_BRGEMM_IR(brg.prfC.dist0 <= 0, VERBOSE_UNSUPPORTED_FEATURE,
                "C prefetch of a later block");
    }

    // Every level is unrolled when the classic kernels use the uker, and looped
    // when they use `jit_brgemm_kernel_t`. Without the uker, N is inside M, as
    // in `jit_brgemm_kernel_t`.
    const level_form_t form
            = can_dispatch_uker ? level_form_t::unrolled : level_form_t::looped;
    conf.nest_without_post_ops.innermost_loop
            = can_dispatch_uker ? brg.innermost_loop : brgemm_ld_loop_innermost;
    conf.nest_without_post_ops.m_form = form;
    conf.nest_without_post_ops.n_form = form;
    conf.nest_without_post_ops.bs_form = form;
    conf.nest_without_post_ops.k_form = form;
    conf.nest_with_post_ops = conf.nest_without_post_ops;

    // `use_interleave_stores` allows the accumulators of a block to be stored
    // in its K steps and in the first K step of the next block.
    // TODO: interleave blockings other than 4 C tiles once they are measured.
    conf.nest_without_post_ops.store.interleave_acc_stores = is_tmul
            && brgattr.use_interleave_stores
            && brg.bd_block2 * brg.ld_block2 == 4;
    conf.nest_without_post_ops.store.convert_acc = brg.dt_c != dt_acc;

    conf.nest_with_post_ops.store.to_d = true;
    conf.nest_with_post_ops.store.convert_acc = brg.dt_d != dt_acc;

    // The order is the one oneDNN defines and the classic kernel follows:
    //   src scale -> weights scale -> bias -> post-ops -> dst scale.
    // The scales are applied one at a time, as in the classic kernel, so the
    // result rounds the same way.
    std::vector<output_stage_t> &stages = conf.nest_with_post_ops.store.stages;
    if (brg.with_src_scales) stages.push_back(output_stage_t::src_scale);
    if (brg.with_wei_scales)
        stages.push_back(brg.is_single_wei_scale
                        ? output_stage_t::wei_scale
                        : output_stage_t::wei_scale_per_n);
    if (brg.with_bias) stages.push_back(output_stage_t::bias);
    if (brg.with_eltwise || brg.with_binary || brg.with_sum)
        stages.push_back(output_stage_t::injector_postops);
    if (brg.with_dst_scales) stages.push_back(output_stage_t::dst_scale);

    // `jit_brgemm_kernel_t` ignores `prfC`, and the uker emits nothing for its
    // distances other than `dist0`. In a kernel with a post-ops call, the uker
    // prefetches nothing on a call without post-ops.
    const bool prefetch = can_dispatch_uker && brg.prfC.dist0 == 0;
    conf.nest_with_post_ops.store.prefetch = prefetch;
    conf.nest_without_post_ops.store.prefetch
            = prefetch && !conf.are_post_ops_applicable();

    // Only the TMUL compute path converts so far. Its ISA always has the
    // conversion to bf16, unlike `avx2` and `avx512_core`.
    VCONDCHECK_BRGEMM_IR(
            IMPLICATION(!is_tmul, !conf.nest_with_post_ops.store.convert_acc),
            VERBOSE_UNSUPPORTED_DT);

    // Only the TMUL compute path takes a bf16 bias so far.
    VCONDCHECK_BRGEMM_IR(IMPLICATION(!is_tmul && brg.with_bias,
                                 brg.dt_bias == data_type::f32),
            VERBOSE_UNSUPPORTED_BIAS_CFG);

    // TODO: accept the other output stages on the TMUL compute path after
    // testing them on the rows of its tiles.
    for (const output_stage_t stage : stages) {
        VCONDCHECK_BRGEMM_IR(
                IMPLICATION(is_tmul, stage == output_stage_t::bias),
                VERBOSE_UNSUPPORTED_FEATURE,
                "output stages other than the bias on tiles");
    }
    return status::success;
}

// Rejects a loop nest whose addresses the builder cannot encode.
//
// Every address the builder emits is a base register plus a build-time
// displacement encoded in the instruction, and every pointer move is a
// build-time immediate, so both have to fit in int32. `max_level_pos()` bounds
// the positions and moves along each level for its `level_form_t`. Checking
// the offset at the bounds of all levels covers every displacement and move,
// including the step of the outer one of the M and N loops, which moves C
// along M and N with one `add`. The B prefetch is not checked here, see
// `emit_microkernel()`. `build_unrolled_ir()` only loops levels that are
// unrolled here. A looped level stays within the bounds of an unrolled one, as
// `max_looped_pos()` is at most `level_size()`.
status_t check_displacements(const conf_t &conf, const nest_conf_t &nest_cfg) {
    auto fits = [](dim_t v) { return jit_generator_t::imm_fits_int32(v); };

    const blocking_conf_t &blocking = conf.blocking;
    const dim_t m = max_level_pos(m_runs(blocking), nest_cfg.m_form);
    const dim_t n = max_level_pos(n_runs(blocking), nest_cfg.n_form);
    const dim_t k = max_level_pos(k_runs(blocking), nest_cfg.k_form);

    VCONDCHECK_BRGEMM_IR(fits(conf.a_off(m, k)), VERBOSE_UNSUPPORTED_FEATURE,
            "A displacement overflows int32");
    VCONDCHECK_BRGEMM_IR(fits(conf.b_off(k, n)), VERBOSE_UNSUPPORTED_FEATURE,
            "B displacement overflows int32");
    VCONDCHECK_BRGEMM_IR(fits(conf.c_off(m, n)), VERBOSE_UNSUPPORTED_FEATURE,
            "C displacement overflows int32");
    VCONDCHECK_BRGEMM_IR(
            IMPLICATION(nest_cfg.store.to_d, fits(conf.d_off(m, n))),
            VERBOSE_UNSUPPORTED_FEATURE, "D displacement overflows int32");
    VCONDCHECK_BRGEMM_IR(fits(conf.bias_off(n)), VERBOSE_UNSUPPORTED_FEATURE,
            "bias displacement overflows int32");
    VCONDCHECK_BRGEMM_IR(fits(conf.wei_scale_off(n)),
            VERBOSE_UNSUPPORTED_FEATURE,
            "weights scale displacement overflows int32");

    // Without `var_bs`, the batch elements are addressed with displacements
    // from the batch pointer.
    const dim_t bs = max_level_pos(bs_runs(blocking), nest_cfg.bs_form);
    const dim_t batch_off = bs * (dim_t)sizeof(brgemm_batch_element_t);
    VCONDCHECK_BRGEMM_IR(IMPLICATION(!blocking.var_bs, fits(batch_off)),
            VERBOSE_UNSUPPORTED_FEATURE, "batch displacement overflows int32");

    return status::success;
}

// Reads the descriptor into `conf` and `tiles`.
//
// Returns `status::success` if the kernel supports the descriptor, otherwise
// `status::unimplemented`. This is the only code that reads the descriptor for
// the builder. Each part of `conf` has its own function, which rejects
// what its module does not support. A descriptor the kernel rejects falls back
// to the classic kernel.
//
// `tiles` is the tile configuration, which only code generation reads. It is
// kept out of `conf`, so the builder never receives it. It stays empty for the
// vector compute path.
status_t init_conf(
        conf_t &conf, ir::tile_config_t &tiles, const brgemm_desc_t &brg) {
    // GEMV and DGMM have kernels of their own.
    VCONDCHECK_BRGEMM_IR(!brg.is_gemv, VERBOSE_UNSUPPORTED_FEATURE, "gemv");
    VCONDCHECK_BRGEMM_IR(!brg.is_dgmm, VERBOSE_UNSUPPORTED_FEATURE, "dgmm");

    CHECK(init_compute_path(conf.compute_path, brg));
    const compute_path_kind_t kind = conf.compute_path.kind;
    CHECK(init_blocking_conf(conf.blocking, kind, brg));
    switch (kind) {
        case compute_path_kind_t::vector:
            CHECK(init_vector_conf(conf.compute_path, brg));
            break;
        case compute_path_kind_t::tmul:
            CHECK(init_tmul_conf(conf.compute_path, tiles, brg));
            break;
    }
    CHECK(init_output_conf(conf.output, brg));
    CHECK(init_nest_confs(conf, kind, brg));

    // `build_brgemm()` emits `nest_with_post_ops` only when post-ops are
    // applicable.
    CHECK(check_displacements(conf, conf.nest_without_post_ops));
    if (conf.are_post_ops_applicable())
        CHECK(check_displacements(conf, conf.nest_with_post_ops));
    return status::success;
}

// IR-based BRGEMM kernel.
struct jit_brgemm_ir_kernel_t : public brgemm_kernel_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_ir_kernel_t)

    jit_brgemm_ir_kernel_t(
            cpu_isa_t isa, const conf_t &conf, const ir::tile_config_t &tiles)
        : brgemm_kernel_t(jit_name(), isa), conf_(conf), tiles_(tiles) {}

    void generate() override {
        ir::ir_t ir;

        // TMUL kernels pick the `level_form_t` of their levels to fit in L1i.
        // The others keep the ones that `init_conf()` sets.
        // TODO: give vector kernels a code size budget too, after measuring
        // them. Their compute path then needs `code_bytes_per_op()`.
        if (conf_.compute_path.kind == compute_path_kind_t::tmul) {
            build_unrolled_ir(conf_, ir);
        } else {
            const std::unique_ptr<compute_path_t> cp
                    = make_compute_path(ir, conf_);
            build_brgemm(conf_, *cp, ir);
        }

        ir::codegen_config_t codegen_cfg;
        if (conf_.nest_with_post_ops.store.has_stage(
                    output_stage_t::injector_postops)) {
            ir::postops_config_t &postops = codegen_cfg.postops;
            postops.post_ops = &conf_.output.post_ops;
            postops.dst_md = &conf_.output.dst_md;
            postops.rhs_arg_offset = GET_OFF(post_ops_binary_rhs_arg_vec);
            postops.dst_orig_off = GET_OFF(data_C_ptr_);
        }
        codegen_cfg.tiles = tiles_;

        ir::generate_kernel(*this, ir, codegen_cfg);
    }

private:
    conf_t conf_;
    ir::tile_config_t tiles_;
};

} // namespace
} // namespace brgemm_ir

brgemm_kernel_t *create_brgemm_ir_kernel(const brgemm_desc_t &brg) {
    brgemm_ir::conf_t conf;
    ir::tile_config_t tiles;
    if (brgemm_ir::init_conf(conf, tiles, brg) != status::success)
        return nullptr;
    return new brgemm_ir::jit_brgemm_ir_kernel_t(brg.isa_impl, conf, tiles);
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
