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
//
// `init_conf()` is the only code that reads the descriptor for the builder. It
// converts the descriptor into `conf_t`, and the builder reads only `conf_t`.

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "cpu/x64/brgemm/brgemm_ir.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/ir/codegen.hpp"
#include "cpu/x64/ir/ir.hpp"
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

// Builder configuration
//
// `conf_t` holds one configuration per module of the builder:
//   loop        - the loop nest: the batch, the leading dimensions, and the
//                 blocking
//   microkernel - the compute path and the data types of A, B, and the
//                 accumulators
//   output      - the output path: beta, the bias, the scales, and the
//                 attribute post-ops
//
// Every field is a build-time constant.

// How the product is computed. The caller chooses it when it creates the
// descriptor, and the kernel never changes it.
//
//   vector - broadcasts one A element and multiply-adds it with B vectors into
//            vector accumulators
enum class compute_path_t { vector };

// Loop nest configuration.
//
//   batch_kind  - how a batch element gives the address of its A and B
//   max_bs      - maximum batch size known at IR generation time
//   load_dim    - N, the columns of C
//   lda/ldb/ldc - leading dimensions of A, B, and C, in elements
//   bd_block    - M rows per block
//   ld_block    - N elements per vector register
//   ld_block2   - vector registers of N per block
//   rd_block    - K elements reduced per K block
//   bdb         - number of full M blocks
//   ldb2        - number of full N blocks
//   rdb         - number of full K blocks
//   bdb_tail    - M rows left over after the full M blocks
//   ldb2_tail   - full N registers left over after the full N blocks
//   ldb_tail    - N columns left over after every full N register
//   rdb_tail    - K elements left over after the full K blocks
struct loop_conf_t {
    brgemm_batch_kind_t batch_kind = brgemm_addr;
    dim_t max_bs = 0;
    dim_t load_dim = 0;
    dim_t lda = 0, ldb = 0, ldc = 0;
    int bd_block = 0, ld_block = 0, ld_block2 = 0, rd_block = 0;
    dim_t bdb = 0, ldb2 = 0, rdb = 0;
    int bdb_tail = 0, ldb2_tail = 0, ldb_tail = 0, rdb_tail = 0;
};

// Microkernel configuration.
//
//   compute_path - how the product is computed
//   dt_a/dt_b    - element data type of A and B in memory
//   dt_acc       - accumulation data type
struct microkernel_conf_t {
    compute_path_t compute_path = compute_path_t::vector;
    data_type_t dt_a = data_type::undef;
    data_type_t dt_b = data_type::undef;
    data_type_t dt_acc = data_type::undef;
};

// Output path configuration.
//
//   beta      - output scaling: 0 overwrites C, 1 accumulates into C
//   with_bias - whether a bias is added to the output
//   dt_c      - element data type of C and D in memory
//   dt_bias   - element data type of the bias
//   with_injector_postops - whether attribute post-ops (eltwise, binary, sum)
//               are requested
//   post_ops  - the attribute post-ops, applied through the post-ops injector
//   dst_md    - the destination memory descriptor, which a binary post-op
//               needs to locate its right-hand-side argument
//   with_src_scales  - whether a common source scale multiplies the output
//   with_wei_scales  - whether a weights scale multiplies the output
//   per_n_wei_scales - whether the weights scale has one value per output
//               column. Otherwise it is one common value
//   with_dst_scales  - whether a common destination scale multiplies the
//               output
//   dt_src_scales / dt_wei_scales - element data type of each scale
struct output_conf_t {
    float beta = 0.0f;
    bool with_bias = false;
    data_type_t dt_c = data_type::undef;
    data_type_t dt_bias = data_type::undef;
    bool with_injector_postops = false;
    post_ops_t post_ops;
    memory_desc_t dst_md;
    bool with_src_scales = false;
    bool with_wei_scales = false;
    bool per_n_wei_scales = false;
    bool with_dst_scales = false;
    data_type_t dt_src_scales = data_type::undef;
    data_type_t dt_wei_scales = data_type::undef;

    // True if the kernel has anything to apply on a post-ops call: the bias, a
    // scale, or an attribute post-op.
    bool has_post_ops() const {
        return with_bias || with_injector_postops || with_src_scales
                || with_wei_scales || with_dst_scales;
    }
};

// Builder configuration.
//
//   loop        - loop nest configuration
//   microkernel - microkernel configuration
//   output      - output path configuration
//
// The member functions derive the addressing of the builder from the module
// configurations.
struct conf_t {
    loop_conf_t loop;
    microkernel_conf_t microkernel;
    output_conf_t output;

    // Element sizes in bytes of A, B, C, and the bias. The bias size is 0
    // without a bias.
    int dt_sz_a() const { return (int)types::data_type_size(microkernel.dt_a); }
    int dt_sz_b() const { return (int)types::data_type_size(microkernel.dt_b); }
    int dt_sz_c() const { return (int)types::data_type_size(output.dt_c); }
    int dt_sz_bias() const {
        return output.with_bias ? (int)types::data_type_size(output.dt_bias)
                                : 0;
    }
    // Element size in bytes of a per-N weights scale, 0 for a common one,
    // which is never indexed.
    int dt_sz_wei_scales() const {
        return output.per_n_wei_scales
                ? (int)types::data_type_size(output.dt_wei_scales)
                : 0;
    }

    // N elements one N block covers.
    int ldb_ld_elems() const { return loop.ld_block2 * loop.ld_block; }

    // Byte offsets that advance the pointers from one block to the next:
    //   rdb_a_off    - A between K blocks
    //   rdb_b_off    - B between K blocks. This is also the distance a B load
    //                  prefetches ahead, one K block of B rows.
    //   ldb_b_off    - B between N blocks
    //   ldb_c_off    - C between N blocks
    //   ldb_bias_off - the bias between N blocks
    //   ldb_wei_scale_off - the weights scale between N blocks
    //   bdb_a_off    - A between M blocks
    //   bdb_c_off    - C between M blocks
    dim_t rdb_a_off() const { return dt_sz_a() * loop.rd_block; }
    dim_t rdb_b_off() const { return dt_sz_b() * loop.rd_block * loop.ldb; }
    dim_t ldb_b_off() const { return dt_sz_b() * ldb_ld_elems(); }
    dim_t ldb_c_off() const { return dt_sz_c() * ldb_ld_elems(); }
    dim_t ldb_bias_off() const { return dt_sz_bias() * ldb_ld_elems(); }
    dim_t ldb_wei_scale_off() const {
        return dt_sz_wei_scales() * ldb_ld_elems();
    }
    dim_t bdb_a_off() const { return dt_sz_a() * loop.bd_block * loop.lda; }
    dim_t bdb_c_off() const { return dt_sz_c() * loop.bd_block * loop.ldc; }

    // Displacements of one element of A, one vector of B, and one accumulator
    // of C from the start of the current block.
    //
    // These are the constants the builder encodes into the operation, the
    // `disp` of `ir::mem_t`. They are functions rather than expressions written
    // at each use because `init_conf()` range-checks the largest of them
    // against int32, and the check is only meaningful if there is consistency
    // between the check and the builder.

    // A is read one element at a time and broadcast, so `rd` is an element
    // index.
    dim_t a_off(int bd, int rd) const {
        return dt_sz_a() * ((dim_t)bd * loop.lda + rd);
    }

    // B is read one full vector per `ld`, so `rd` is a row index and `ld`
    // picks the vector within the N block.
    dim_t b_off(const out_block_t &blk, int ld, int rd) const {
        return dt_sz_b() * ((dim_t)rd * loop.ldb + ld_pos(blk, ld));
    }

    dim_t c_off(const out_block_t &blk, int bd, int ld) const {
        return dt_sz_c() * ((dim_t)bd * loop.ldc + ld_pos(blk, ld));
    }

    // One bias value per output column, so the offset has no row term.
    dim_t bias_off(const out_block_t &blk, int ld) const {
        return dt_sz_bias() * ld_pos(blk, ld);
    }

    // A per-N weights scale is indexed like the bias.
    dim_t wei_scale_off(const out_block_t &blk, int ld) const {
        return dt_sz_wei_scales() * ld_pos(blk, ld);
    }

    // Load-dimension element the vector `ld` of the block starts at, counted
    // from where the B and C pointers stand. B and C differ only in the
    // element size they scale it by.
    dim_t ld_pos(const out_block_t &blk, int ld) const {
        return blk.ld_start + (dim_t)ld * loop.ld_block;
    }

    // Load-dimension elements the N loop leaves on the B offset and the C
    // pointer, which the M loop rewinds. `emit_loop_imm()` inlines a single
    // iteration without its step, so a one-block N loop advances nothing.
    dim_t ldb_loop_ld_adv() const {
        return loop.ldb2 > 1 ? loop.ldb2 * ldb_ld_elems() : 0;
    }
    dim_t ldb_loop_c_adv() const { return dt_sz_c() * ldb_loop_ld_adv(); }
    dim_t ldb_loop_bias_adv() const { return dt_sz_bias() * ldb_loop_ld_adv(); }
    dim_t ldb_loop_wei_scale_adv() const {
        return dt_sz_wei_scales() * ldb_loop_ld_adv();
    }

    // First load-dimension element of each N tail block, counted from where
    // the N loop left the pointers.
    //
    // Both blocks are emitted once, never in a loop, so they reach their
    // columns through the displacement of each access rather than through a
    // pointer of their own. `ldb2 * ldb_ld_elems()` is where the tails start
    // and `ldb_loop_ld_adv()` is where the pointers stand.
    dim_t ldb2_tail_ld_start() const {
        return loop.ldb2 * ldb_ld_elems() - ldb_loop_ld_adv();
    }

    dim_t ldb_tail_ld_start() const {
        return ldb2_tail_ld_start() + (dim_t)loop.ldb2_tail * loop.ld_block;
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
    // Current C pointer. The N loop advances it by `ldb_c_off()` per block,
    // then the M loop rewinds it and steps to the next M block with one
    // constant, `bdb_c_off() - ldb_loop_c_adv()`. The classic kernel keeps a
    // second base pointer for this instead. One running pointer saves the
    // general-purpose register.
    ir::vreg_t c_ptr = ir::vreg_t::none;
    // Pointer the finished block is stored through. It holds the `ptr_D`
    // kernel argument on a post-ops call and `ptr_C` otherwise, and can be the
    // same register as `c_ptr` (see `init_m_loop_input_regs()`). C and D have
    // the same data type and leading dimension (`init_output_conf()`), so it
    // advances like `c_ptr`.
    ir::vreg_t store_ptr = ir::vreg_t::none;
    // Byte offset into A of the current M block. Starts at 0 and advances by
    // `bdb_a_off()` per M block.
    ir::vreg_t a_off = ir::vreg_t::none;
    // Byte offset into B of the current N block. Advances by `ldb_b_off()` per
    // N block and returns to 0 for the next M block.
    ir::vreg_t b_off = ir::vreg_t::none;
    // Current bias pointer. The bias is indexed by the load dimension only, so
    // it advances by `ldb_bias_off()` per N block and rewinds for the next M
    // block.
    // `none` unless `with_bias`.
    ir::vreg_t bias_ptr = ir::vreg_t::none;
    // Current weights-scale pointer. A per-N scale advances and rewinds like
    // `bias_ptr`, a common one stays at its only element.
    // `none` unless `with_wei_scales`.
    ir::vreg_t wei_scale_ptr = ir::vreg_t::none;
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
    // The `do_post_ops` kernel argument. Non-zero applies the post-ops and
    // stores to D, zero stores to C. `none` when the kernel has no post-op.
    ir::vreg_t do_post_ops = ir::vreg_t::none;
    // Common source and destination scale pointers.
    // `none` unless `with_src_scales` and `with_dst_scales` respectively.
    ir::vreg_t src_scale_ptr = ir::vreg_t::none;
    ir::vreg_t dst_scale_ptr = ir::vreg_t::none;
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
m_loop_input_regs_t init_m_loop_input_regs(ir::ir_t &ir, const conf_t &cfg) {
    m_loop_input_regs_t regs;

    regs.advancing.c_ptr = ir.new_gpr();
    ir.load_param(regs.advancing.c_ptr, GET_OFF(ptr_C));

    regs.invariant.batch = ir.new_gpr();
    ir.load_param(regs.invariant.batch, GET_OFF(batch));

    if (cfg.loop.max_bs > 1) {
        regs.invariant.bs = ir.new_gpr();
        ir.load_param(regs.invariant.bs, GET_OFF(BS));
    }

    if (cfg.loop.batch_kind == brgemm_offs) {
        regs.invariant.a_base = ir.new_gpr();
        ir.load_param(regs.invariant.a_base, GET_OFF(ptr_A));
        regs.invariant.b_base = ir.new_gpr();
        ir.load_param(regs.invariant.b_base, GET_OFF(ptr_B));
    }

    regs.advancing.a_off = ir.new_gpr();
    ir.mov_imm(regs.advancing.a_off, 0);

    regs.advancing.b_off = ir.new_gpr();
    ir.mov_imm(regs.advancing.b_off, 0);

    if (cfg.output.with_bias) {
        regs.advancing.bias_ptr = ir.new_gpr();
        ir.load_param(regs.advancing.bias_ptr, GET_OFF(ptr_bias));
    }

    if (cfg.output.with_src_scales) {
        regs.invariant.src_scale_ptr = ir.new_gpr();
        ir.load_param(regs.invariant.src_scale_ptr, GET_OFF(ptr_src_scales));
    }

    if (cfg.output.with_wei_scales) {
        regs.advancing.wei_scale_ptr = ir.new_gpr();
        ir.load_param(regs.advancing.wei_scale_ptr, GET_OFF(ptr_wei_scales));
    }

    if (cfg.output.with_dst_scales) {
        regs.invariant.dst_scale_ptr = ir.new_gpr();
        ir.load_param(regs.invariant.dst_scale_ptr, GET_OFF(ptr_dst_scales));
    }

    // The block goes to `ptr_D` when `do_post_ops` is set, and to `ptr_C`
    // otherwise. Resolve that once here so that the stores do not branch.
    //
    // A post-ops call with `beta == 1` reads C for the accumulators and writes
    // D, so the two pointers need separate registers. C is never read when
    // `beta == 0`, so one register is enough and it holds `ptr_D` on a post-ops
    // call.
    if (cfg.output.has_post_ops()) {
        regs.invariant.do_post_ops = ir.new_gpr();
        ir.load_param(regs.invariant.do_post_ops, GET_OFF(do_post_ops));

        if (cfg.output.beta == 0.0f) {
            regs.advancing.store_ptr = regs.advancing.c_ptr;
        } else {
            regs.advancing.store_ptr = ir.new_gpr();
            ir.mov_reg(regs.advancing.store_ptr, regs.advancing.c_ptr);
        }

        const int if_post_ops
                = ir.if_begin(regs.invariant.do_post_ops, ir::cmp_t::ne, 0);
        ir.load_param(regs.advancing.store_ptr, GET_OFF(ptr_D));
        ir.if_end(if_post_ops);
    } else {
        regs.advancing.store_ptr = regs.advancing.c_ptr;
    }

    return regs;
}

// Innermost reduction step, over `rd_loop` elements of K.
//
// Loads one B vector per N register, then for each M row broadcasts one A
// element and multiply-adds it into that row's accumulators. This is the
// `!n_bcast_1_load` variant of `gemm_microkernel()` in `jit_brgemm_kernel.cpp`.
void emit_microkernel(ir::ir_t &ir, const conf_t &cfg, const out_block_t &blk,
        const std::vector<ir::vreg_t> &acc, const std::vector<ir::vreg_t> &b,
        ir::vreg_t a, ir::vreg_t a_ptr, ir::vreg_t b_ptr, int rd_loop) {

    for (int rd = 0; rd < rd_loop; rd++) {
        // Only the last register of a block can be partial, and a block that
        // has one holds nothing else, so the mask covers every B load here.
        for (int ld = 0; ld < blk.ld_block2; ld++) {
            if (blk.ld_tail_mask == ir::vreg_t::none)
                ir.vload(b[ld], b_ptr, cfg.b_off(blk, ld, rd),
                        cfg.microkernel.dt_b);
            else
                ir.vload_masked(b[ld], b_ptr, cfg.b_off(blk, ld, rd),
                        blk.ld_tail_mask, cfg.microkernel.dt_b);
        }

        // One prefetch per B register, issued on the first `ld_block2` rows, so
        // each reduction step fetches the next K block of B exactly once.
        int n_pf_b = 0;
        for (int bd = 0; bd < blk.bd_block; bd++) {
            ir.vload_bcast(a, a_ptr, cfg.a_off(bd, rd), cfg.microkernel.dt_a);
            if (n_pf_b < blk.ld_block2) {
                ir.prefetch(
                        b_ptr, cfg.b_off(blk, n_pf_b, rd) + cfg.rdb_b_off());
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
void emit_bs_body(ir::ir_t &ir, const conf_t &cfg,
        const m_loop_input_regs_t &regs, const out_block_t &blk,
        const std::vector<ir::vreg_t> &acc, const std::vector<ir::vreg_t> &b,
        ir::vreg_t a, ir::vreg_t batch_ptr) {

    const ir::vreg_t a_ptr = ir.new_gpr();
    const ir::vreg_t b_ptr = ir.new_gpr();

    // Where this element's A and B start. Mirrors `set_A_B_matrices()` in
    // `jit_brgemm_kernel.cpp`.
    switch (cfg.loop.batch_kind) {
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
    if (cfg.loop.max_bs > 1) {
        ir.add_imm(batch_ptr, sizeof(brgemm_batch_element_t));
        // The classic kernel prefetches the next element for the address kind
        // only.
        if (cfg.loop.batch_kind == brgemm_addr) ir.prefetch(batch_ptr, 0);
    }

    // Advance the A and B pointers by one K block.
    auto advance_ptrs = [&]() {
        ir.add_imm(a_ptr, cfg.rdb_a_off());
        ir.add_imm(b_ptr, cfg.rdb_b_off());
    };

    // Reduce the full K blocks, then the tail if any. Cases by `rdb`:
    //   *  == 0  the whole reduction is the tail
    //   *  == 1  one block, advance by hand only if a tail follows
    //   *  >= 2  loop, advancing per iteration
    if (cfg.loop.rdb >= 2) {
        ir::emit_loop_imm(ir, cfg.loop.rdb, [&]() {
            emit_microkernel(
                    ir, cfg, blk, acc, b, a, a_ptr, b_ptr, cfg.loop.rd_block);
        }, advance_ptrs);
    } else if (cfg.loop.rdb == 1) {
        emit_microkernel(
                ir, cfg, blk, acc, b, a, a_ptr, b_ptr, cfg.loop.rd_block);
        if (cfg.loop.rdb_tail > 0) advance_ptrs();
    }

    // K tail needs no mask. A reduction step reads a full B vector along N
    // and one element of A, so a short tail is just fewer steps.
    if (cfg.loop.rdb_tail > 0)
        emit_microkernel(
                ir, cfg, blk, acc, b, a, a_ptr, b_ptr, cfg.loop.rdb_tail);
}

// One N block.
//
// Holds one accumulator per (M row, N register), reduces them over the batch,
// and stores them to C, or to D on a post-ops call.
void emit_n_block(ir::ir_t &ir, const conf_t &cfg,
        const m_loop_input_regs_t &regs, const out_block_t &blk) {

    std::vector<ir::vreg_t> acc(blk.bd_block * blk.ld_block2, ir::vreg_t::none);

    for (int bd = 0; bd < blk.bd_block; bd++) {
        for (int ld = 0; ld < blk.ld_block2; ld++) {
            const int i = bd * blk.ld_block2 + ld;
            acc[i] = ir.new_vec(cfg.microkernel.dt_acc);

            // The kernel takes only 0 and 1 for beta.
            if (cfg.output.beta == 0.0f)
                ir.vzero(acc[i]);
            else if (blk.ld_tail_mask == ir::vreg_t::none)
                ir.vload(acc[i], regs.advancing.c_ptr, cfg.c_off(blk, bd, ld),
                        cfg.output.dt_c);
            else
                ir.vload_masked(acc[i], regs.advancing.c_ptr,
                        cfg.c_off(blk, bd, ld), blk.ld_tail_mask,
                        cfg.output.dt_c);
        }
    }

    std::vector<ir::vreg_t> b(blk.ld_block2, ir::vreg_t::none);
    for (int ld = 0; ld < blk.ld_block2; ld++)
        b[ld] = ir.new_vec(cfg.microkernel.dt_b);

    // Batch reduction over the bs dimension.
    const ir::vreg_t batch_ptr = ir.new_gpr();
    ir.mov_reg(batch_ptr, regs.invariant.batch);

    const ir::vreg_t a = ir.new_vec(cfg.microkernel.dt_a);

    auto bs_body
            = [&]() { emit_bs_body(ir, cfg, regs, blk, acc, b, a, batch_ptr); };

    if (cfg.loop.max_bs > 1)
        ir::emit_loop_reg(ir, regs.invariant.bs, bs_body);
    else
        ir::emit_loop_imm(ir, 1, bs_body);

    if (cfg.output.has_post_ops()) {
        // The order is the one oneDNN defines and the classic kernel follows:
        //   src scale -> weights scale -> bias -> post-ops -> dst scale.
        // The scales are applied one at a time, as in the classic kernel, so
        // the result rounds the same way.
        const int if_post_ops
                = ir.if_begin(regs.invariant.do_post_ops, ir::cmp_t::ne, 0);

        if (cfg.output.with_src_scales) {
            const ir::vreg_t sc = ir.new_vec(cfg.output.dt_src_scales);
            ir.vload_bcast(sc, regs.invariant.src_scale_ptr, 0,
                    cfg.output.dt_src_scales);
            for (const ir::vreg_t &v : acc)
                ir.vmul(v, sc);
        }

        if (cfg.output.with_wei_scales && !cfg.output.per_n_wei_scales) {
            const ir::vreg_t sc = ir.new_vec(cfg.output.dt_wei_scales);
            ir.vload_bcast(sc, regs.advancing.wei_scale_ptr, 0,
                    cfg.output.dt_wei_scales);
            for (const ir::vreg_t &v : acc)
                ir.vmul(v, sc);
        }

        if (cfg.output.per_n_wei_scales) {
            // One scale vector per N register, shared by every M row of the
            // block.
            for (int ld = 0; ld < blk.ld_block2; ld++) {
                const ir::vreg_t sc = ir.new_vec(cfg.output.dt_wei_scales);
                if (blk.ld_tail_mask == ir::vreg_t::none)
                    ir.vload(sc, regs.advancing.wei_scale_ptr,
                            cfg.wei_scale_off(blk, ld),
                            cfg.output.dt_wei_scales);
                else
                    ir.vload_masked(sc, regs.advancing.wei_scale_ptr,
                            cfg.wei_scale_off(blk, ld), blk.ld_tail_mask,
                            cfg.output.dt_wei_scales);

                for (int bd = 0; bd < blk.bd_block; bd++)
                    ir.vmul(acc[bd * blk.ld_block2 + ld], sc);
            }
        }

        if (cfg.output.with_bias) {
            // One bias vector per N register, shared by every M row of the
            // block.
            for (int ld = 0; ld < blk.ld_block2; ld++) {
                const ir::vreg_t bias = ir.new_vec(cfg.output.dt_bias);
                if (blk.ld_tail_mask == ir::vreg_t::none)
                    ir.vload(bias, regs.advancing.bias_ptr,
                            cfg.bias_off(blk, ld), cfg.output.dt_bias);
                else
                    ir.vload_masked(bias, regs.advancing.bias_ptr,
                            cfg.bias_off(blk, ld), blk.ld_tail_mask,
                            cfg.output.dt_bias);

                for (int bd = 0; bd < blk.bd_block; bd++)
                    ir.vadd(acc[bd * blk.ld_block2 + ld], bias);
            }
        }

        if (cfg.output.with_injector_postops) {
            // The output offset of each accumulator is its store displacement
            // below. A binary post-op finds its right-hand-side element through
            // it, and a sum post-op reads the previous value from where the
            // result is written. A block is either all full registers or the
            // one masked register of the N tail, which holds `ldb_tail`
            // elements.
            std::vector<dim_t> out_byte_off(acc.size());
            for (int bd = 0; bd < blk.bd_block; bd++)
                for (int ld = 0; ld < blk.ld_block2; ld++)
                    out_byte_off[bd * blk.ld_block2 + ld]
                            = cfg.c_off(blk, bd, ld);

            const int n_elems = blk.ld_tail_mask != ir::vreg_t::none
                    ? cfg.loop.ldb_tail
                    : cfg.loop.ld_block;
            ir.inject_postops(
                    acc, regs.advancing.store_ptr, out_byte_off, n_elems);
        }

        if (cfg.output.with_dst_scales) {
            // The caller passes the inverted f32 destination scale, so it is a
            // multiply.
            const ir::vreg_t sc = ir.new_vec(data_type::f32);
            ir.vload_bcast(sc, regs.invariant.dst_scale_ptr, 0, data_type::f32);
            for (const ir::vreg_t &v : acc)
                ir.vmul(v, sc);
        }

        ir.if_end(if_post_ops);
    }

    // `store_ptr` already points at C or D, so the stores do not branch. Their
    // memory data type is fixed when the IR is built, which is why C and D must
    // share one.
    for (int bd = 0; bd < blk.bd_block; bd++) {
        for (int ld = 0; ld < blk.ld_block2; ld++) {
            const ir::vreg_t src = acc[bd * blk.ld_block2 + ld];

            if (blk.ld_tail_mask == ir::vreg_t::none) {
                ir.vstore(regs.advancing.store_ptr, cfg.c_off(blk, bd, ld), src,
                        cfg.output.dt_c);
            } else {
                ir.vstore_masked(regs.advancing.store_ptr,
                        cfg.c_off(blk, bd, ld), src, blk.ld_tail_mask,
                        cfg.output.dt_c);
            }
        }
    }
}

// One M block, which is the N loop over `ldb2` full blocks followed by the two
// N tails.
//
// The tails are emitted once each, so they address their columns through
// `ld_start`.
void emit_m_block(ir::ir_t &ir, const conf_t &cfg,
        const m_loop_input_regs_t &regs, int bd_block,
        ir::vreg_t ld_tail_mask) {

    auto advance_ptrs = [&]() {
        ir.add_imm(regs.advancing.b_off, cfg.ldb_b_off());
        ir.add_imm(regs.advancing.c_ptr, cfg.ldb_c_off());
        if (regs.advancing.store_ptr != regs.advancing.c_ptr)
            ir.add_imm(regs.advancing.store_ptr, cfg.ldb_c_off());
        if (cfg.output.with_bias)
            ir.add_imm(regs.advancing.bias_ptr, cfg.ldb_bias_off());
        if (cfg.output.per_n_wei_scales)
            ir.add_imm(regs.advancing.wei_scale_ptr, cfg.ldb_wei_scale_off());
    };
    const out_block_t full {bd_block, cfg.loop.ld_block2, 0, ir::vreg_t::none};
    ir::emit_loop_imm(ir, cfg.loop.ldb2,
            [&]() { emit_n_block(ir, cfg, regs, full); }, advance_ptrs);

    // Full registers left over after the N loop.
    if (cfg.loop.ldb2_tail > 0) {
        const out_block_t blk {bd_block, cfg.loop.ldb2_tail,
                cfg.ldb2_tail_ld_start(), ir::vreg_t::none};
        emit_n_block(ir, cfg, regs, blk);
    }

    // Columns left over after every full register, in one masked register.
    if (cfg.loop.ldb_tail > 0) {
        const out_block_t blk {
                bd_block, 1, cfg.ldb_tail_ld_start(), ld_tail_mask};
        emit_n_block(ir, cfg, regs, blk);
    }
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
// registers across the whole batch and K reduction, then stores it once.
void build_brgemm(const conf_t &cfg, ir::ir_t &ir) {
    const m_loop_input_regs_t regs = init_m_loop_input_regs(ir, cfg);

    // Every partial N block is the same width, so one mask serves the whole
    // kernel.
    ir::vreg_t ld_tail_mask = ir::vreg_t::none;

    if (cfg.loop.ldb_tail > 0) {
        ld_tail_mask = ir.new_mask();
        ir.set_mask_imm(ld_tail_mask, cfg.loop.ldb_tail);
    }

    // `store_ptr` is a separate register only when `beta == 1` and the kernel
    // has a post-op, so it needs its own advance only then. The bias and a
    // per-N weights scale do not depend on M, so they only rewind what the N
    // loop added.
    auto advance_ptrs = [&]() {
        ir.add_imm(regs.advancing.a_off, cfg.bdb_a_off());
        ir.add_imm(
                regs.advancing.c_ptr, cfg.bdb_c_off() - cfg.ldb_loop_c_adv());
        if (regs.advancing.store_ptr != regs.advancing.c_ptr)
            ir.add_imm(regs.advancing.store_ptr,
                    cfg.bdb_c_off() - cfg.ldb_loop_c_adv());
        if (cfg.output.with_bias && cfg.ldb_loop_bias_adv() != 0)
            ir.add_imm(regs.advancing.bias_ptr, -cfg.ldb_loop_bias_adv());
        if (cfg.output.per_n_wei_scales && cfg.ldb_loop_wei_scale_adv() != 0)
            ir.add_imm(regs.advancing.wei_scale_ptr,
                    -cfg.ldb_loop_wei_scale_adv());
        if (cfg.ldb_loop_ld_adv() != 0) ir.mov_imm(regs.advancing.b_off, 0);
    };
    ir::emit_loop_imm(ir, cfg.loop.bdb, [&]() {
        emit_m_block(ir, cfg, regs, cfg.loop.bd_block, ld_tail_mask);
    }, advance_ptrs);

    // Rows left over after the M loop. They form a block with fewer
    // accumulators and are otherwise a block like any other, so the pointers
    // have to arrive the way a loop iteration would leave them.
    if (cfg.loop.bdb_tail > 0) {
        if (cfg.loop.bdb == 1) advance_ptrs();
        emit_m_block(ir, cfg, regs, cfg.loop.bdb_tail, ld_tail_mask);
    }
}

// Picks the compute path the caller chose when it created the descriptor.
//
// The finalized descriptor already holds the choice (`is_ace()`, `is_tmm`), so
// it is read from there. Only the vector compute path exists so far, and the
// tile compute paths are rejected.
status_t init_compute_path(
        compute_path_t &compute_path, const brgemm_desc_t &brg) {
    VCONDCHECK_BRGEMM_IR(!brg.is_tmm && !brg.is_ace(),
            VERBOSE_UNSUPPORTED_FEATURE, "tile accumulation");

    compute_path = compute_path_t::vector;
    return status::success;
}

// Fills the loop nest configuration.
status_t init_loop_conf(loop_conf_t &loop, const brgemm_desc_t &brg) {
    using namespace utils;
    const brgemm_attr_t &brgattr = brg.brgattr;

    VCONDCHECK_BRGEMM_IR(one_of(brg.type, brgemm_addr, brgemm_offs),
            VERBOSE_UNSUPPORTED_FEATURE, "batch kind");
    VCONDCHECK_BRGEMM_IR(!brgattr.var_bs, VERBOSE_UNSUPPORTED_FEATURE,
            "variable batch size");
    VCONDCHECK_BRGEMM_IR(brg.layout == brgemm_row_major,
            VERBOSE_UNSUPPORTED_FEATURE, "column-major layout");
    VCONDCHECK_BRGEMM_IR(
            !brg.is_blocked, VERBOSE_UNSUPPORTED_FEATURE, "blocked layout");
    VCONDCHECK_BRGEMM_IR(everyone_is(false, brg.is_runtime_lda,
                                 brg.is_runtime_ldb, brg.is_runtime_ldc),
            VERBOSE_UNSUPPORTED_FEATURE, "runtime leading dimension");
    VCONDCHECK_BRGEMM_IR(
            everyone_is(0, brgattr.max_top_vpad, brgattr.max_bottom_vpad),
            VERBOSE_UNSUPPORTED_PAD_FEATURE, "virtual padding");
    VCONDCHECK_BRGEMM_IR(
            everyone_is(0, brgattr.max_top_bpad, brgattr.max_bottom_bpad),
            VERBOSE_UNSUPPORTED_PAD_FEATURE, "batch padding");
    VCONDCHECK_BRGEMM_IR(
            brgattr.bd_mask_level == 0, VERBOSE_UNSUPPORTED_FEATURE, "bd_mask");

    loop.batch_kind = brg.type;
    loop.max_bs = brgattr.max_bs;
    loop.load_dim = brg.load_dim;
    loop.lda = brg.LDA;
    loop.ldb = brg.LDB;
    loop.ldc = brg.LDC;
    loop.bd_block = brg.bd_block;
    loop.ld_block = brg.ld_block;
    loop.ld_block2 = brg.ld_block2;
    loop.rd_block = brg.rd_block;
    loop.bdb = brg.bdb;
    loop.ldb2 = brg.ldb2;
    loop.rdb = brg.rdb;
    loop.bdb_tail = brg.bdb_tail;
    loop.ldb2_tail = brg.ldb2_tail;
    loop.ldb_tail = brg.ldb_tail;
    loop.rdb_tail = brg.rdb_tail;
    return status::success;
}

// Fills the microkernel configuration of the vector compute path.
status_t init_microkernel_conf(
        microkernel_conf_t &microkernel, const brgemm_desc_t &brg) {
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

    microkernel.dt_a = brg.dt_a;
    microkernel.dt_b = brg.dt_b;
    microkernel.dt_acc = f32;
    return status::success;
}

// Fills the output path configuration.
status_t init_output_conf(output_conf_t &output, const brgemm_desc_t &brg) {
    using namespace data_type;
    using namespace utils;
    const brgemm_attr_t &brgattr = brg.brgattr;

    VCONDCHECK_BRGEMM_IR(
            everyone_is(f32, brg.dt_c, brg.dt_d), VERBOSE_UNSUPPORTED_DT);
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
    }

    // Scales are f32 values the kernel broadcasts or loads per N register:
    // src and dst scales are common, the weights scale common or per-N.
    VCONDCHECK_BRGEMM_IR(
            !brg.has_per_k_scales(), VERBOSE_UNSUPPORTED_SCALES_CFG);
    VCONDCHECK_BRGEMM_IR(
            IMPLICATION(brg.with_src_scales, brg.dt_src_scales == f32),
            VERBOSE_UNSUPPORTED_SCALES_CFG);
    VCONDCHECK_BRGEMM_IR(
            IMPLICATION(brg.with_wei_scales, brg.dt_wei_scales == f32),
            VERBOSE_UNSUPPORTED_SCALES_CFG);
    VCONDCHECK_BRGEMM_IR(!brg.with_weights_scale_adjust,
            VERBOSE_UNSUPPORTED_FEATURE, "weights scale adjust");

    VCONDCHECK_BRGEMM_IR(IMPLICATION(brg.with_bias, brg.dt_bias == f32),
            VERBOSE_UNSUPPORTED_BIAS_CFG);
    VCONDCHECK_BRGEMM_IR(!brg.is_runtime_ldd, VERBOSE_UNSUPPORTED_FEATURE,
            "runtime leading dimension");

    output.beta = brg.beta;
    output.with_bias = brg.with_bias;
    output.dt_c = brg.dt_c;
    output.dt_bias = brg.dt_bias;
    output.with_injector_postops
            = brg.with_eltwise || brg.with_binary || brg.with_sum;
    if (output.with_injector_postops) {
        output.post_ops = brg.attr()->post_ops_;
        output.dst_md = *brg.dst_md();
    }
    output.with_src_scales = brg.with_src_scales;
    output.with_wei_scales = brg.with_wei_scales;
    output.per_n_wei_scales = brg.with_wei_scales && !brg.is_single_wei_scale;
    output.with_dst_scales = brg.with_dst_scales;
    output.dt_src_scales = brg.dt_src_scales;
    output.dt_wei_scales = brg.dt_wei_scales;

    // A post-ops call stores to D through the C displacements.
    VCONDCHECK_BRGEMM_IR(IMPLICATION(output.has_post_ops(), brg.LDC == brg.LDD),
            VERBOSE_UNSUPPORTED_FEATURE, "LDC != LDD");
    return status::success;
}

// Rejects a configuration whose addresses the builder cannot encode.
//
// Every address the builder emits is a base register plus a build-time
// displacement encoded in the instruction, so each displacement has to fit in
// int32. These are the largest of them.
status_t check_displacements(const conf_t &conf) {
    const loop_conf_t &loop = conf.loop;
    auto fits = [](dim_t v) { return v <= INT32_MAX && v >= INT32_MIN; };
    const int rd_last = (loop.rdb > 0 ? loop.rd_block : loop.rdb_tail) - 1;

    // Rows of the tallest M block. The M tail is shorter than a full block.
    const int max_bd_block = loop.bdb > 0 ? loop.bd_block : loop.bdb_tail;

    // Upper bound for the B and C displacements.
    const out_block_t last_col {
            max_bd_block, 1, loop.load_dim - 1, ir::vreg_t::none};

    VCONDCHECK_BRGEMM_IR(fits(conf.a_off(max_bd_block - 1, rd_last)),
            VERBOSE_UNSUPPORTED_FEATURE, "A displacement overflows int32");
    VCONDCHECK_BRGEMM_IR(
            fits(conf.b_off(last_col, 0, rd_last) + conf.rdb_b_off()),
            VERBOSE_UNSUPPORTED_FEATURE, "B displacement overflows int32");
    VCONDCHECK_BRGEMM_IR(fits(conf.c_off(last_col, max_bd_block - 1, 0)),
            VERBOSE_UNSUPPORTED_FEATURE, "C displacement overflows int32");
    VCONDCHECK_BRGEMM_IR(fits(conf.bias_off(last_col, 0)),
            VERBOSE_UNSUPPORTED_FEATURE, "bias displacement overflows int32");
    VCONDCHECK_BRGEMM_IR(fits(conf.wei_scale_off(last_col, 0)),
            VERBOSE_UNSUPPORTED_FEATURE,
            "weights scale displacement overflows int32");
    VCONDCHECK_BRGEMM_IR(fits(conf.bdb_c_off() - conf.ldb_loop_c_adv()),
            VERBOSE_UNSUPPORTED_FEATURE, "C rewind overflows int32");
    VCONDCHECK_BRGEMM_IR(fits(conf.ldb_loop_bias_adv()),
            VERBOSE_UNSUPPORTED_FEATURE, "bias rewind overflows int32");
    VCONDCHECK_BRGEMM_IR(fits(conf.ldb_loop_wei_scale_adv()),
            VERBOSE_UNSUPPORTED_FEATURE,
            "weights scale rewind overflows int32");
    VCONDCHECK_BRGEMM_IR(fits(conf.bdb_a_off()) && fits(conf.rdb_a_off())
                    && fits(conf.rdb_b_off()) && fits(conf.ldb_b_off())
                    && fits(conf.ldb_c_off()) && fits(conf.ldb_bias_off())
                    && fits(conf.ldb_wei_scale_off()),
            VERBOSE_UNSUPPORTED_FEATURE, "pointer advance overflows int32");

    return status::success;
}

// Reads the descriptor into `conf`.
//
// Returns `status::success` if the kernel supports the descriptor, otherwise
// `status::unimplemented`. This is the only code that reads the descriptor for
// the builder. Each part of `conf` has its own function, which rejects
// what its module does not support. A descriptor the kernel rejects falls back
// to the classic kernel.
//
// TODO: reject `mem_advice` or implement it. For the vector compute path the
// classic kernel loads B with `vmovrsd` when it is set, and matmul sets it on
// avx10_2.
status_t init_conf(conf_t &conf, const brgemm_desc_t &brg) {
    // GEMV and DGMM have kernels of their own.
    VCONDCHECK_BRGEMM_IR(!brg.is_gemv, VERBOSE_UNSUPPORTED_FEATURE, "gemv");
    VCONDCHECK_BRGEMM_IR(!brg.is_dgmm, VERBOSE_UNSUPPORTED_FEATURE, "dgmm");

    CHECK(init_compute_path(conf.microkernel.compute_path, brg));
    CHECK(init_loop_conf(conf.loop, brg));
    CHECK(init_microkernel_conf(conf.microkernel, brg));
    CHECK(init_output_conf(conf.output, brg));
    return check_displacements(conf);
}

// IR-based BRGEMM kernel.
struct jit_brgemm_ir_kernel_t : public brgemm_kernel_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_ir_kernel_t)

    jit_brgemm_ir_kernel_t(cpu_isa_t isa, const conf_t &conf)
        : brgemm_kernel_t(jit_name(), isa), conf_(conf) {}

    void generate() override {
        ir::ir_t ir;
        build_brgemm(conf_, ir);

        ir::postops_config_t postops;
        if (conf_.output.with_injector_postops) {
            postops.post_ops = &conf_.output.post_ops;
            postops.dst_md = &conf_.output.dst_md;
            postops.rhs_arg_offset = GET_OFF(post_ops_binary_rhs_arg_vec);
            postops.dst_orig_off = GET_OFF(data_C_ptr_);
        }

        ir::generate_kernel(*this, ir, postops);
    }

private:
    conf_t conf_;
};

} // namespace

brgemm_kernel_t *create_brgemm_ir_kernel(const brgemm_desc_t &brg) {
    conf_t conf;
    if (init_conf(conf, brg) != status::success) return nullptr;
    return new jit_brgemm_ir_kernel_t(brg.isa_impl, conf);
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
