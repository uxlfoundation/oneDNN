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
// This kernel uses `jit_brgemm_kernel_t` as a reference and is comparable with
// the classic kernel instruction for instruction when it comes to compute
// structure.

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "cpu/x64/brgemm/brgemm_ir.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/ir/emitter/emitter.hpp"
#include "cpu/x64/ir/ir.hpp"
#include "cpu/x64/ir/reg_alloc.hpp"
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

namespace {

// The block of the output one N-block emission computes.
//
//   bd_block     - rows of the broadcast dimension the block covers
//   ld_block2    - vector registers of the load dimension it covers
//   ld_start     - load-dimension element the block starts at, counted from
//                  where the B and C pointers stand. B and C convert it with
//                  their own element size, so one count serves both.
//   ld_tail_mask - live elements of the last register, `none` when every
//                  register of the block is full
//
// `ld_start` is what lets the N tail blocks reach their columns without a
// pointer of their own. It is 0 for a block the N loop walks to.
struct out_block_t {
    int bd_block;
    int ld_block2;
    dim_t ld_start;
    ir::vreg_t ld_tail_mask;
};

// Fixed configuration used during IR generation for the BRGEMM builder.
//
// Every field is a build-time constant taken from the descriptor.
//
//   lda/ldb/ldc  - leading dimensions of A, B, and C, in elements
//   max_bs       - maximum batch size known at IR generation time
//   batch_kind   - how a batch element gives the address of its A and B
//   bd_block     - M rows per block
//   ld_block     - N elements per vector register
//   ld_block2    - vector registers of N per block
//   ldb_ld_elems - N elements one N block covers, `ld_block2 * ld_block`
//   rd_block     - K elements reduced per K block
//   dt_sz_a/b/c  - element size in bytes of A, B, and C
//   dt_a/b/c     - element data type of A, B, and C in memory
//   dt_acc       - accumulation data type
//   beta         - output scaling: 0 overwrites C, 1 accumulates into C
//   bdb          - number of full M blocks
//   ldb2         - number of full N blocks
//   rdb          - number of full K blocks
//   bdb_tail     - M rows left over after the full M blocks
//   ldb2_tail    - full N registers left over after the full N blocks
//   ldb_tail     - N columns left over after every full N register
//   rdb_tail     - K elements left over after the full K blocks
//   rdb_a_off    - byte offset to advance A between K blocks
//   rdb_b_off    - byte offset to advance B between K blocks. This is also the
//                  distance a B load prefetches ahead, one K block of B rows
//   ldb_b_off    - byte offset to advance B between N blocks
//   ldb_c_off    - byte offset to advance C between N blocks
//   bdb_a_off    - byte offset to advance A between M blocks
//   bdb_c_off    - byte offset to advance C between M blocks
struct brgemm_ir_conf_t {
    brgemm_ir_conf_t(const brgemm_desc_t &brg)
        : lda(brg.LDA)
        , ldb(brg.LDB)
        , ldc(brg.LDC)
        , max_bs(brg.brgattr.max_bs)
        , batch_kind(brg.type)
        , bd_block(brg.bd_block)
        , ld_block(brg.ld_block)
        , ld_block2(brg.ld_block2)
        , ldb_ld_elems(ld_block2 * ld_block)
        , rd_block(brg.rd_block)
        , dt_sz_a(brg.typesize_A)
        , dt_sz_b(brg.typesize_B)
        , dt_sz_c(brg.typesize_C)
        , dt_a(brg.dt_a)
        , dt_b(brg.dt_b)
        , dt_c(brg.dt_c)
        , dt_acc(data_type::f32)
        , beta(brg.beta)
        , bdb(brg.bdb)
        , ldb2(brg.ldb2)
        , rdb(brg.rdb)
        , bdb_tail(brg.bdb_tail)
        , ldb2_tail(brg.ldb2_tail)
        , ldb_tail(brg.ldb_tail)
        , rdb_tail(brg.rdb_tail)
        , rdb_a_off(dt_sz_a * rd_block)
        , rdb_b_off(dt_sz_b * rd_block * ldb)
        , ldb_b_off(dt_sz_b * ldb_ld_elems)
        , ldb_c_off(dt_sz_c * ldb_ld_elems)
        , bdb_a_off(dt_sz_a * bd_block * lda)
        , bdb_c_off(dt_sz_c * bd_block * ldc) {}

    const dim_t lda, ldb, ldc;
    const dim_t max_bs;
    const brgemm_batch_kind_t batch_kind;
    const int bd_block, ld_block, ld_block2, ldb_ld_elems, rd_block;
    const int dt_sz_a, dt_sz_b, dt_sz_c;
    const data_type_t dt_a, dt_b, dt_c, dt_acc;
    const float beta;
    const dim_t bdb, ldb2, rdb;
    const int bdb_tail, ldb2_tail, ldb_tail, rdb_tail;
    const dim_t rdb_a_off, rdb_b_off;
    const dim_t ldb_b_off, ldb_c_off;
    const dim_t bdb_a_off, bdb_c_off;

    // Displacements of one element of A, one vector of B, and one accumulator
    // of C from the start of the current block.
    //
    // These are the constants the builder encodes into the operation, the
    // `disp` of `ir::mem_t`. They are functions rather than expressions written
    // at each use because `brgemm_ir_supported()` range-checks the largest of
    // them against int32, and the check is only meaningful if there is
    // consistency between the check and the builder.

    // A is read one element at a time and broadcast, so `rd` is an element
    // index.
    dim_t a_off(int bd, int rd) const {
        return dt_sz_a * ((dim_t)bd * lda + rd);
    }

    // B is read one full vector per `ld`, so `rd` is a row index and `ld`
    // picks the vector within the N block.
    dim_t b_off(const out_block_t &blk, int ld, int rd) const {
        return dt_sz_b * ((dim_t)rd * ldb + ld_pos(blk, ld));
    }

    dim_t c_off(const out_block_t &blk, int bd, int ld) const {
        return dt_sz_c * ((dim_t)bd * ldc + ld_pos(blk, ld));
    }

    // Load-dimension element the vector `ld` of the block starts at, counted
    // from where the B and C pointers stand. B and C differ only in the
    // element size they scale it by.
    dim_t ld_pos(const out_block_t &blk, int ld) const {
        return blk.ld_start + (dim_t)ld * ld_block;
    }

    // Load-dimension elements the N loop leaves on the B offset and the C
    // pointer, which the M loop rewinds. `emit_loop_imm()` inlines a single
    // iteration without its step, so a one-block N loop advances nothing.
    dim_t ldb_loop_ld_adv() const { return ldb2 > 1 ? ldb2 * ldb_ld_elems : 0; }
    dim_t ldb_loop_c_adv() const { return dt_sz_c * ldb_loop_ld_adv(); }

    // First load-dimension element of each N tail block, counted from where
    // the N loop left the pointers.
    //
    // Both blocks are emitted once, never in a loop, so they reach their
    // columns through the displacement of each access rather than through a
    // pointer of their own. `ldb2 * ldb_ld_elems` is where the tails start and
    // `ldb_loop_ld_adv()` is where the pointers stand.
    dim_t ldb2_tail_ld_start() const {
        return ldb2 * ldb_ld_elems - ldb_loop_ld_adv();
    }

    dim_t ldb_tail_ld_start() const {
        return ldb2_tail_ld_start() + (dim_t)ldb2_tail * ld_block;
    }
};

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
//   A broadcast, and the batch, A, and B pointers) are local temporaries and
//   are not included here.

// Registers that advance by a fixed byte offset as the nest walks the output.
struct advancing_regs_t {
    // Current C pointer. The N loop advances it by `ldb_c_off` per block, then
    // the M loop rewinds it and steps to the next M block with one constant,
    // `bdb_c_off - ldb_loop_c_adv()`. The classic kernel keeps a second base
    // pointer for this instead. One running pointer saves the general-purpose
    // register.
    ir::vreg_t c_ptr = ir::vreg_t::none;
    // Byte offset into A of the current M block. Starts at 0 and advances by
    // `bdb_a_off` per M block.
    ir::vreg_t a_off = ir::vreg_t::none;
    // Byte offset into B of the current N block. Advances by `ldb_b_off` per N
    // block and returns to 0 for the next M block.
    ir::vreg_t b_off = ir::vreg_t::none;
};

// Registers that hold the same value for the entire M loop.
struct invariant_regs_t {
    // Base pointer of the batch-element array.
    ir::vreg_t batch = ir::vreg_t::none;
    // Batch size loop count. `none` when max_bs == 1 (single batch element).
    ir::vreg_t bs = ir::vreg_t::none;
    // A and B base pointers, the `ptr_A` and `ptr_B` kernel arguments, which
    // `brgemm_offs` adds the byte offsets of each batch element to. `none` for
    // `brgemm_addr`, where each element holds its own pointers.
    ir::vreg_t a_base = ir::vreg_t::none;
    ir::vreg_t b_base = ir::vreg_t::none;
};

// Complete input register set for the M loop, partitioned by whether values
// advance across iterations.
struct m_loop_input_regs_t {
    advancing_regs_t advancing;
    invariant_regs_t invariant;
};

// Sets up the M-loop input registers.
//
// Loads the kernel argument pointers and the batch count, and zeroes the
// running A and B offsets.
m_loop_input_regs_t init_m_loop_input_regs(
        ir::ir_t &ir, const brgemm_ir_conf_t &cfg) {
    m_loop_input_regs_t regs;

    regs.advancing.c_ptr = ir.new_gpr();
    ir.load_param(regs.advancing.c_ptr, GET_OFF(ptr_C));

    regs.invariant.batch = ir.new_gpr();
    ir.load_param(regs.invariant.batch, GET_OFF(batch));

    if (cfg.max_bs > 1) {
        regs.invariant.bs = ir.new_gpr();
        ir.load_param(regs.invariant.bs, GET_OFF(BS));
    }

    if (cfg.batch_kind == brgemm_offs) {
        regs.invariant.a_base = ir.new_gpr();
        ir.load_param(regs.invariant.a_base, GET_OFF(ptr_A));
        regs.invariant.b_base = ir.new_gpr();
        ir.load_param(regs.invariant.b_base, GET_OFF(ptr_B));
    }

    regs.advancing.a_off = ir.new_gpr();
    ir.mov_imm(regs.advancing.a_off, 0);

    regs.advancing.b_off = ir.new_gpr();
    ir.mov_imm(regs.advancing.b_off, 0);

    return regs;
}

// Innermost reduction step, over `rd_loop` elements of K.
//
// Loads one B vector per N register, then for each M row broadcasts one A
// element and multiply-adds it into that row's accumulators. This is the
// `!n_bcast_1_load` shape of `gemm_microkernel()` in `jit_brgemm_kernel.cpp`.
void emit_microkernel(ir::ir_t &ir, const brgemm_ir_conf_t &cfg,
        const out_block_t &blk, const std::vector<ir::vreg_t> &acc,
        const std::vector<ir::vreg_t> &b, ir::vreg_t a, ir::vreg_t a_ptr,
        ir::vreg_t b_ptr, int rd_loop) {

    for (int rd = 0; rd < rd_loop; rd++) {
        // Only the last register of a block can be partial, and a block that
        // has one holds nothing else, so the mask covers every B load here.
        for (int ld = 0; ld < blk.ld_block2; ld++) {
            if (blk.ld_tail_mask == ir::vreg_t::none)
                ir.vload(b[ld], b_ptr, cfg.b_off(blk, ld, rd), cfg.dt_b);
            else
                ir.vload_masked(b[ld], b_ptr, cfg.b_off(blk, ld, rd),
                        blk.ld_tail_mask, cfg.dt_b);
        }

        // One prefetch per B register, issued on the first `ld_block2` rows, so
        // each reduction step fetches the next K block of B exactly once.
        int n_pf_b = 0;
        for (int bd = 0; bd < blk.bd_block; bd++) {
            ir.vload_bcast(a, a_ptr, cfg.a_off(bd, rd), cfg.dt_a);
            if (n_pf_b < blk.ld_block2) {
                ir.prefetch(b_ptr, cfg.b_off(blk, n_pf_b, rd) + cfg.rdb_b_off);
                n_pf_b++;
            }

            for (int ld = 0; ld < blk.ld_block2; ld++)
                ir.vdot(acc[bd * blk.ld_block2 + ld], b[ld], a);
        }
    }
}

// One batch element.
//
// Derives the A and B pointers of the element, shifts them to the current M and
// N block, then runs the reduction loop over K.
void emit_bs_body(ir::ir_t &ir, const brgemm_ir_conf_t &cfg,
        const m_loop_input_regs_t &regs, const out_block_t &blk,
        const std::vector<ir::vreg_t> &acc, const std::vector<ir::vreg_t> &b,
        ir::vreg_t a, ir::vreg_t batch_ptr) {

    const ir::vreg_t a_ptr = ir.new_gpr();
    const ir::vreg_t b_ptr = ir.new_gpr();

    // Where this element's A and B start. Mirrors `set_A_B_matrices()` in
    // `jit_brgemm_kernel.cpp`.
    switch (cfg.batch_kind) {
        case brgemm_addr:
            ir.load(a_ptr, batch_ptr, GET_OFF_BATCH_ELEMENT(ptr.A));
            ir.load(b_ptr, batch_ptr, GET_OFF_BATCH_ELEMENT(ptr.B));
            break;
        case brgemm_offs:
            ir.load(a_ptr, batch_ptr, GET_OFF_BATCH_ELEMENT(offset.A));
            ir.add_reg(a_ptr, regs.invariant.a_base);
            ir.load(b_ptr, batch_ptr, GET_OFF_BATCH_ELEMENT(offset.B));
            ir.add_reg(b_ptr, regs.invariant.b_base);
            break;
        default: assert(!"unsupported batch kind"); break;
    }

    // Shift to the current M and N block.
    ir.add_reg(a_ptr, regs.advancing.a_off);
    ir.add_reg(b_ptr, regs.advancing.b_off);

    // Advance to the next batch element. A single batch element has no next
    // one, so nothing is emitted then.
    if (cfg.max_bs > 1) {
        ir.add_imm(batch_ptr, sizeof(brgemm_batch_element_t));
        // The classic kernel prefetches the next element for the address kind
        // only.
        if (cfg.batch_kind == brgemm_addr) ir.prefetch(batch_ptr, 0);
    }

    // Advance the A and B pointers by one K block.
    auto advance_ptrs = [&]() {
        ir.add_imm(a_ptr, cfg.rdb_a_off);
        ir.add_imm(b_ptr, cfg.rdb_b_off);
    };

    // Reduce the full K blocks, then the tail if any. Cases by `rdb`:
    //   *  == 0  the whole reduction is the tail
    //   *  == 1  one block, advance by hand only if a tail follows
    //   *  >= 2  loop, advancing per iteration
    if (cfg.rdb >= 2) {
        ir::emit_loop_imm(ir, cfg.rdb, [&]() {
            emit_microkernel(
                    ir, cfg, blk, acc, b, a, a_ptr, b_ptr, cfg.rd_block);
        }, advance_ptrs);
    } else if (cfg.rdb == 1) {
        emit_microkernel(ir, cfg, blk, acc, b, a, a_ptr, b_ptr, cfg.rd_block);
        if (cfg.rdb_tail > 0) advance_ptrs();
    }

    // K tail needs no mask. A reduction step reads a full B vector along N
    // and one element of A, so a short tail is just fewer steps.
    if (cfg.rdb_tail > 0)
        emit_microkernel(ir, cfg, blk, acc, b, a, a_ptr, b_ptr, cfg.rdb_tail);
}

// One N block.
//
// Holds one accumulator per (M row, N register), reduces them over the batch,
// and stores them to C.
void emit_n_block(ir::ir_t &ir, const brgemm_ir_conf_t &cfg,
        const m_loop_input_regs_t &regs, const out_block_t &blk) {

    std::vector<ir::vreg_t> acc(blk.bd_block * blk.ld_block2, ir::vreg_t::none);

    for (int bd = 0; bd < blk.bd_block; bd++) {
        for (int ld = 0; ld < blk.ld_block2; ld++) {
            const int i = bd * blk.ld_block2 + ld;
            acc[i] = ir.new_vec(cfg.dt_acc);

            // The kernel takes only 0 and 1 for beta.
            if (cfg.beta == 0.0f)
                ir.vzero(acc[i]);
            else if (blk.ld_tail_mask == ir::vreg_t::none)
                ir.vload(acc[i], regs.advancing.c_ptr, cfg.c_off(blk, bd, ld),
                        cfg.dt_c);
            else
                ir.vload_masked(acc[i], regs.advancing.c_ptr,
                        cfg.c_off(blk, bd, ld), blk.ld_tail_mask, cfg.dt_c);
        }
    }

    std::vector<ir::vreg_t> b(blk.ld_block2, ir::vreg_t::none);
    for (int ld = 0; ld < blk.ld_block2; ld++)
        b[ld] = ir.new_vec(cfg.dt_b);

    // Batch reduction over the bs dimension.
    const ir::vreg_t batch_ptr = ir.new_gpr();
    ir.mov_reg(batch_ptr, regs.invariant.batch);

    const ir::vreg_t a = ir.new_vec(cfg.dt_a);

    auto bs_body
            = [&]() { emit_bs_body(ir, cfg, regs, blk, acc, b, a, batch_ptr); };

    if (cfg.max_bs > 1)
        ir::emit_loop_reg(ir, regs.invariant.bs, bs_body);
    else
        ir::emit_loop_imm(ir, 1, bs_body);

    for (int bd = 0; bd < blk.bd_block; bd++) {
        for (int ld = 0; ld < blk.ld_block2; ld++) {
            const ir::vreg_t src = acc[bd * blk.ld_block2 + ld];

            if (blk.ld_tail_mask == ir::vreg_t::none) {
                ir.vstore(regs.advancing.c_ptr, cfg.c_off(blk, bd, ld), src,
                        cfg.dt_c);
            } else {
                ir.vstore_masked(regs.advancing.c_ptr, cfg.c_off(blk, bd, ld),
                        src, blk.ld_tail_mask, cfg.dt_c);
            }
        }
    }
}

// One M block, which is the N loop over `ldb2` full blocks followed by the two
// N tails.
//
// The tails are emitted once each, so they address their columns through
// `ld_start`.
void emit_m_block(ir::ir_t &ir, const brgemm_ir_conf_t &cfg,
        const m_loop_input_regs_t &regs, int bd_block,
        ir::vreg_t ld_tail_mask) {

    auto advance_ptrs = [&]() {
        ir.add_imm(regs.advancing.b_off, cfg.ldb_b_off);
        ir.add_imm(regs.advancing.c_ptr, cfg.ldb_c_off);
    };
    const out_block_t full {bd_block, cfg.ld_block2, 0, ir::vreg_t::none};
    ir::emit_loop_imm(ir, cfg.ldb2,
            [&]() { emit_n_block(ir, cfg, regs, full); }, advance_ptrs);

    // Full registers left over after the N loop.
    if (cfg.ldb2_tail > 0) {
        const out_block_t blk {bd_block, cfg.ldb2_tail,
                cfg.ldb2_tail_ld_start(), ir::vreg_t::none};
        emit_n_block(ir, cfg, regs, blk);
    }

    // Columns left over after every full register, in one masked register.
    if (cfg.ldb_tail > 0) {
        const out_block_t blk {
                bd_block, 1, cfg.ldb_tail_ld_start(), ld_tail_mask};
        emit_n_block(ir, cfg, regs, blk);
    }
}

// Builds IR for BRGEMM.
//
// Computes:
//   C[i][j] = beta * C[i][j] + sum_bs sum_k A[i][k] * B[k][j]
//   (m = brg.bcast_dim, n = brg.load_dim, k = brg.reduce_dim)
//
// The output is partitioned into M blocks of `bd_block` rows, each split into
// N blocks of `ld_block2 * ld_block` columns. Each block keeps its result in
// registers across the whole batch and K reduction, then stores it once.
void build_brgemm(const brgemm_desc_t &brg, ir::ir_t &ir) {
    const brgemm_ir_conf_t cfg(brg);
    const m_loop_input_regs_t regs = init_m_loop_input_regs(ir, cfg);

    // Every partial N block is the same width, so one mask serves the whole
    // kernel.
    ir::vreg_t ld_tail_mask = ir::vreg_t::none;

    if (cfg.ldb_tail > 0) {
        ld_tail_mask = ir.new_mask();
        ir.set_mask_imm(ld_tail_mask, cfg.ldb_tail);
    }

    auto advance_ptrs = [&]() {
        ir.add_imm(regs.advancing.a_off, cfg.bdb_a_off);
        ir.add_imm(regs.advancing.c_ptr, cfg.bdb_c_off - cfg.ldb_loop_c_adv());
        if (cfg.ldb_loop_ld_adv() != 0) ir.mov_imm(regs.advancing.b_off, 0);
    };
    ir::emit_loop_imm(ir, cfg.bdb, [&]() {
        emit_m_block(ir, cfg, regs, cfg.bd_block, ld_tail_mask);
    }, advance_ptrs);

    // Rows left over after the M loop. They form a block with fewer
    // accumulators and are otherwise a block like any other, so the pointers
    // have to arrive the way a loop iteration would leave them.
    if (cfg.bdb_tail > 0) {
        if (cfg.bdb == 1) advance_ptrs();
        emit_m_block(ir, cfg, regs, cfg.bdb_tail, ld_tail_mask);
    }
}

#ifndef NDEBUG
bool any_vector_spill(const ir::ir_t &ir, const ir::reg_alloc_result_t &alloc) {
    for (int v = 0; v < ir.n_vregs(); v++) {
        if (ir.vreg_info()[v].kind == ir::reg_kind_t::gpr) continue;
        if (alloc.assignments[v].spilled) return true;
    }
    return false;
}
#endif

} // namespace

// generate() runs the full IR pipeline:
//
// - Build IR for the given `brgemm_desc_t` descriptor
// - Allocate registers
// - Emit code
// - Wrap in standard preamble, stack frame, and postamble
//
// TODO: Generalize the IR pipeline runner so it is shared across all kernels,
// while allowing different builder implementations to plug into the same
// fixed sequence:
// IR build -> register allocation -> preamble -> codegen -> postamble).
struct jit_brgemm_ir_kernel_t : public jit_base_brgemm_kernel_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_ir_kernel_t)

    jit_brgemm_ir_kernel_t(const brgemm_desc_t &abrg)
        : jit_base_brgemm_kernel_t(jit_name(), abrg.isa_impl), brg_(abrg) {}

    const brgemm_desc_t &get_brg() const override { return brg_; }

    void generate() override {
        ir::ir_t ir;
        build_brgemm(brg_, ir);

        const int rsp_idx = Xbyak::Operand::RSP;
        const int param_idx = abi_param1.getIdx();

        // Scratch registers (2 gpr + 3 vec) reserved for spill code.
        const int gpr_scratch0 = 10, gpr_scratch1 = 11;
        const int vec_scratch0 = 13, vec_scratch1 = 14, vec_scratch2 = 15;

        // Build register configuration for code emission.
        //
        // No post-ops reach this kernel, so no opmask is handed out and the
        // allocator gets the whole mask file.
        const ir::reg_config_t reg_cfg = ir::make_reg_config(brg_.isa_impl,
                param_idx, rsp_idx, {gpr_scratch0, gpr_scratch1},
                {vec_scratch0, vec_scratch1, vec_scratch2},
                /*mask_scratch=*/ {});

        const ir::reg_alloc_result_t alloc
                = allocate_registers(ir, reg_cfg.pools);

        // `brgemm_ir_supported()` allows only a blocking that fits the vector
        // pool, so a vector spill means that check and the builder disagree
        // about how many registers the kernel needs. The code stays correct,
        // but it is slower than the kernel it replaced.
        assert(!any_vector_spill(ir, alloc)
                && "brgemm_ir: unexpected vector spill");

        preamble();

        if (alloc.frame_bytes > 0) sub(rsp, (uint32_t)alloc.frame_bytes);

        ir::data_section_t data;
        ir::emit(*this, ir, alloc, reg_cfg, data, /*postops=*/nullptr);

        if (alloc.frame_bytes > 0) add(rsp, (uint32_t)alloc.frame_bytes);

        postamble();

        ir::emit_data_section(*this, data);
    }

private:
    brgemm_desc_t brg_;
};

// TODO: Reorganize `brgemm_kernel_t` to avoid redundant inheritance.
struct brgemm_ir_kernel_t : public brgemm_kernel_t {
    brgemm_ir_kernel_t(const brgemm_desc_t &abrd)
        : kernel_(new jit_brgemm_ir_kernel_t(abrd)) {}
    ~brgemm_ir_kernel_t() override = default;

    status_t create_kernel() override {
        if (!kernel_) return status::out_of_memory;
        return kernel_->create_kernel();
    }

    void operator()(const brgemm_kernel_params_t *params) const override {
        (*kernel_)(params);
    }

    const jit_generator_t *get_jit_generator() const override {
        return kernel_.get();
    }

    const brgemm_desc_t &get_brg() const override { return kernel_->get_brg(); }

private:
    std::unique_ptr<jit_brgemm_ir_kernel_t> kernel_;
    DNNL_DISALLOW_COPY_AND_ASSIGN(brgemm_ir_kernel_t);
};

// Returns `status::success` if the descriptor is supported by the GEMM IR
// kernel, otherwise `status::unimplemented`.
status_t brgemm_ir_supported(const brgemm_desc_t &brg) {
    using namespace data_type;
    using namespace utils;

    // This kernel supports only vector version of BRGEMM.
    VCONDCHECK_BRGEMM_IR(!brg.is_gemv, VERBOSE_UNSUPPORTED_FEATURE, "gemv");
    VCONDCHECK_BRGEMM_IR(!brg.is_dgmm, VERBOSE_UNSUPPORTED_FEATURE, "dgmm");
    VCONDCHECK_BRGEMM_IR(!brg.is_tmm && !brg.is_ace(),
            VERBOSE_UNSUPPORTED_FEATURE, "tile accumulation");

    // Allow the following ISAs. `avx512_core` is used to compute f32 directly.
    // For the other ISAs we receive already upconverted to f32 inputs.
    VCONDCHECK_BRGEMM_IR(
            one_of(brg.isa_impl, avx512_core, avx512_core_fp16, avx10_2),
            VERBOSE_UNSUPPORTED_ISA);
    VCONDCHECK_BRGEMM_IR(
            everyone_is(f32, brg.dt_a, brg.dt_b, brg.dt_c, brg.dt_d),
            VERBOSE_UNSUPPORTED_DT);

    VCONDCHECK_BRGEMM_IR(one_of(brg.type, brgemm_addr, brgemm_offs),
            VERBOSE_UNSUPPORTED_FEATURE, "batch kind");
    VCONDCHECK_BRGEMM_IR(brg.layout == brgemm_row_major,
            VERBOSE_UNSUPPORTED_FEATURE, "column-major layout");

    VCONDCHECK_BRGEMM_IR(
            brg.alpha == 1.0f, VERBOSE_UNSUPPORTED_FEATURE, "alpha != 1");
    VCONDCHECK_BRGEMM_IR(brg.beta == 0.0f || brg.beta == 1.0f,
            VERBOSE_UNSUPPORTED_FEATURE, "beta != 0 && beta != 1");

    VCONDCHECK_BRGEMM_IR(
            !brg.are_post_ops_applicable(), VERBOSE_UNSUPPORTED_POSTOP);

    VCONDCHECK_BRGEMM_IR(brg.brgattr.hint_prefetchw == brgemm_prfw_default,
            VERBOSE_UNSUPPORTED_FEATURE, "store prefetch hint");

    VCONDCHECK_BRGEMM_IR(
            everyone_is(false, brg.is_runtime_lda, brg.is_runtime_ldb,
                    brg.is_runtime_ldc, brg.is_runtime_ldd),
            VERBOSE_UNSUPPORTED_FEATURE, "runtime leading dimension");

    VCONDCHECK_BRGEMM_IR(!brg.brgattr.generate_skip_accumulation,
            VERBOSE_UNSUPPORTED_FEATURE, "skip accumulation");

    VCONDCHECK_BRGEMM_IR(!brg.n_bcast_1_load, VERBOSE_UNSUPPORTED_FEATURE,
            "one-load microkernel");
    VCONDCHECK_BRGEMM_IR(!brg.embd_bcst, VERBOSE_UNSUPPORTED_FEATURE,
            "embedded broadcast microkernel");

    // Below is a set of checks to check whether the problem fits the register
    // budget. The check will eventually go away once the allocator is optimized
    // and scratch registers are removed.
    const brgemm_ir_conf_t cfg(brg);

    // The builder holds every accumulator, every B vector, and the A broadcast
    // in a register for the whole block. A blocking that needs more registers
    // than the pool holds is still correct, because the allocator spills, but
    // it is slower than the classic kernel, so refuse it instead.
    //
    // Only the widest block matters. A tail block is narrower or shorter than
    // the block it follows, and every block frees its registers before the
    // next one starts.
    const int max_bd_block = cfg.bdb > 0 ? cfg.bd_block : cfg.bdb_tail;
    const int max_ld_block2 = cfg.ldb2 > 0
            ? cfg.ld_block2
            : (cfg.ldb2_tail > 0 ? cfg.ldb2_tail : 1);
    const int n_vregs = max_bd_block * max_ld_block2 + max_ld_block2 + 1;
    // 3 vector register are scratch.
    const int vec_pool_size = isa_num_vregs(brg.isa_impl) - 3;

    VCONDCHECK_BRGEMM_IR(n_vregs <= vec_pool_size, VERBOSE_UNSUPPORTED_FEATURE,
            "blocking exceeds the vector register file");

    // Every address the builder emits is a base register plus a build-time
    // displacement encoded in the instruction, so each displacement has to fit
    // in int32. These are the largest of them.
    auto fits = [](dim_t v) { return v <= INT32_MAX && v >= INT32_MIN; };
    const int rd_last = (cfg.rdb > 0 ? cfg.rd_block : cfg.rdb_tail) - 1;

    // Upper bound for the B and C displacements.
    const out_block_t last_col {
            max_bd_block, 1, brg.load_dim - 1, ir::vreg_t::none};

    VCONDCHECK_BRGEMM_IR(fits(cfg.a_off(max_bd_block - 1, rd_last)),
            VERBOSE_UNSUPPORTED_FEATURE, "A displacement overflows int32");
    VCONDCHECK_BRGEMM_IR(fits(cfg.b_off(last_col, 0, rd_last) + cfg.rdb_b_off),
            VERBOSE_UNSUPPORTED_FEATURE, "B displacement overflows int32");
    VCONDCHECK_BRGEMM_IR(fits(cfg.c_off(last_col, max_bd_block - 1, 0)),
            VERBOSE_UNSUPPORTED_FEATURE, "C displacement overflows int32");
    VCONDCHECK_BRGEMM_IR(fits(cfg.bdb_c_off - cfg.ldb_loop_c_adv()),
            VERBOSE_UNSUPPORTED_FEATURE, "C rewind overflows int32");
    VCONDCHECK_BRGEMM_IR(fits(cfg.bdb_a_off) && fits(cfg.rdb_a_off)
                    && fits(cfg.rdb_b_off) && fits(cfg.ldb_b_off)
                    && fits(cfg.ldb_c_off),
            VERBOSE_UNSUPPORTED_FEATURE, "pointer advance overflows int32");

    return status::success;
}

brgemm_kernel_t *create_brgemm_ir_kernel(const brgemm_desc_t &brg) {
    if (brgemm_ir_supported(brg) != status::success) return nullptr;
    return new brgemm_ir_kernel_t(brg);
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
