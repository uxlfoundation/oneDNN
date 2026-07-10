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

// The IR-based GEMV kernel converts a brgemm descriptor into IR, then runs the
// register allocator and code emitter.
//
// This is the only file that knows about GEMV-specific math and data layout.
// Everything in IR is generic infrastructure that can be reused for other
// problems.
//
// The brgemv_ir_supported() function checks whether a problem matches the
// current requirements.

#include <cstdint>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/x64/brgemm/brgemv_ir.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/ir/emitter/emitter.hpp"
#include "cpu/x64/ir/ir.hpp"
#include "cpu/x64/ir/reg_alloc.hpp"
#include "cpu/x64/jit_generator.hpp"

#define GET_OFF(field) offsetof(brgemm_kernel_params_t, field)
#define GET_OFF_BATCH_ELEMENT(field) offsetof(brgemm_batch_element_t, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace {

// Fixed configuration used during IR generation for the GEMV builder.
//
//   m / k        - dimensions of matrix A (GEMV always has n = 1)
//   lda / incy   - leading dimensions of A and stride between elements of `y`
//   max_bs       - maximum batch size known at IR generation time
//   m_block      - M rows per full block
//   k_block      - K elements reduced per K block
//   dt_sz_a/x/y  - element size in bytes of A, x, and y
//   dt_a/x/y     - element data type of A, x, and y. Tags the vec vregs so the
//                  emitter lowers each op to its dtype-specific instruction.
//   dt_acc       - accumulation data type
//   m_blocks     - number of full M blocks
//   m_tail       - remaining M rows after the full blocks
//   k_blocks     - number of full K blocks
//   k_tail       - remaining K elements after the full blocks
//   mblk_*_off   - byte offset to advance A/y pointers between M blocks
//   kblk_*_off   - byte offset to advance A/x pointers between K blocks
struct brgemv_ir_conf_t {
    brgemv_ir_conf_t(const brgemm_desc_t &brg)
        : m(brg.bcast_dim)
        , k(brg.reduce_dim)
        , lda(brg.LDA)
        , incy(brg.LDC)
        , max_bs(brg.brgattr.max_bs)
        , m_block(brg.gemv_bd_block())
        , k_block(brg.rd_block)
        , dt_sz_a(brg.typesize_A)
        , dt_sz_x(brg.typesize_B)
        , dt_sz_y(brg.typesize_C)
        , dt_a(brg.dt_a)
        , dt_x(brg.dt_b)
        , dt_y(brg.dt_c)
        , dt_acc(data_type::f32)
        , m_blocks(m / m_block)
        , m_tail(m % m_block)
        , k_blocks(k / k_block)
        , k_tail(k % k_block)
        , mblk_a_off(dt_sz_a * m_block * lda)
        , mblk_y_off(dt_sz_y * m_block * incy)
        , kblk_a_off(dt_sz_a * k_block)
        , kblk_x_off(dt_sz_x * k_block) {}

    const dim_t m, k, lda, incy;
    const int max_bs;
    const int m_block, k_block;
    const int dt_sz_a, dt_sz_x, dt_sz_y;
    const data_type_t dt_a, dt_x, dt_y, dt_acc;
    const dim_t m_blocks, m_tail, k_blocks, k_tail;
    const dim_t mblk_a_off, mblk_y_off;
    const dim_t kblk_a_off, kblk_x_off;
};

// M-loop input register classification
//
// These structs describe registers used by the M-block loop, split by
// whether their values advance each iteration or remain constant.
//
// Design rule:
// - A field's vreg ID never changes after assignment.
// - Because of this, these structs are passed by const reference.
//
// Note:
// - Registers created inside the M-block loop (acc, batch_ptr, ws, a_ptr,
//   x_ptr) are local temporaries and are not included here.

// Registers that advance each M-block iteration by a fixed byte offset.
struct advancing_regs_t {
    // Current output pointer. Advances by `mblk_y_off` per M-block.
    ir::vreg_t y_ptr = ir::vreg_t::none;
    // Byte offset into A for the current M-block. Starts at 0 and advances by
    // `mblk_a_off` each iteration.
    ir::vreg_t a_off = ir::vreg_t::none;
};

// Registers that remain constant for the entire M-loop.
struct invariant_regs_t {
    // Base pointer of the batch-element array.
    ir::vreg_t batch = ir::vreg_t::none;
    // Batch size loop count. `none` when max_bs == 1 (single batch element).
    ir::vreg_t bs = ir::vreg_t::none;
    // K-tail mask, shared by every masked tail load. It's only required when
    // `k_tail` is greater than 1.
    ir::vreg_t k_tail_mask = ir::vreg_t::none;
};

// Complete input register set for the M-loop, partitioned by whether values
// advance across iterations.
struct m_loop_input_regs_t {
    advancing_regs_t advancing;
    invariant_regs_t invariant;
};

// Sets up the M-loop input registers.
//
// Loads kernel argument pointers and the batch count, and initializes the
// running A offset to zero.
//
// Optional arguments are only initialized when the configuration specifies
// their presence. Otherwise, they remain `none`.
m_loop_input_regs_t init_m_loop_input_regs(
        ir::ir_t &ir, const brgemv_ir_conf_t &cfg) {
    m_loop_input_regs_t regs;

    regs.advancing.y_ptr = ir.new_gpr();
    ir.load_param(regs.advancing.y_ptr, GET_OFF(ptr_C));

    regs.invariant.batch = ir.new_gpr();
    ir.load_param(regs.invariant.batch, GET_OFF(batch));

    if (cfg.max_bs > 1) {
        regs.invariant.bs = ir.new_gpr();
        ir.load_param(regs.invariant.bs, GET_OFF(BS));
    }

    regs.advancing.a_off = ir.new_gpr();
    ir.mov_imm(regs.advancing.a_off, 0);

    if (cfg.k_tail > 1) {
        // We only need to set the mask once per kernel. It's lifetime is
        // managed automatically by the allocator.
        regs.invariant.k_tail_mask = ir.new_mask();
        ir.set_mask_imm(regs.invariant.k_tail_mask, (int)cfg.k_tail);
    }

    return regs;
}

} // namespace

namespace nontrans {

// Innermost reduction step.
//
// Loads one k_block-wide chunk of `x`, then performs a multiply-add into each
// accumulator in `acc` using the corresponding chunk of its A row.
//
// Each accumulator holds a partial dot product for one output. The number of
// accumulators is the number of rows in the current M block or M tail.
void emit_microkernel(ir::ir_t &ir, const brgemv_ir_conf_t &cfg,
        const std::vector<ir::vreg_t> &acc, ir::vreg_t a_ptr,
        ir::vreg_t x_ptr) {
    const ir::vreg_t x = ir.new_vec(cfg.dt_x);
    const ir::vreg_t a = ir.new_vec(cfg.dt_a);
    ir.vload(x, x_ptr, 0);
    for (int i = 0; i < (int)acc.size(); i++) {
        ir.vload(a, a_ptr, cfg.dt_sz_a * (dim_t)i * cfg.lda);
        ir.vdot(acc[i], a, x);
    }
}

// Innermost reduction step for K tail.
//
// Same shape as emit_microkernel, but uses masked loads.
void emit_microkernel_tail(ir::ir_t &ir, const brgemv_ir_conf_t &cfg,
        const std::vector<ir::vreg_t> &acc, ir::vreg_t a_ptr, ir::vreg_t x_ptr,
        ir::vreg_t mask) {
    const ir::vreg_t x = ir.new_vec(cfg.dt_x);
    const ir::vreg_t a = ir.new_vec(cfg.dt_a);
    ir.vload_masked(x, x_ptr, 0, mask, (int)cfg.k_tail);
    for (int i = 0; i < (int)acc.size(); i++) {
        ir.vload_masked(a, a_ptr, cfg.dt_sz_a * (dim_t)i * cfg.lda, mask,
                (int)cfg.k_tail);
        ir.vdot(acc[i], a, x);
    }
}

// One batch element.
//
// Loads A and x pointers, then runs the reduction loop over k.
// Each iteration processes one `k_block` chunk and advances the pointers.
void emit_bs_body(ir::ir_t &ir, const brgemv_ir_conf_t &cfg,
        const std::vector<ir::vreg_t> &acc, ir::vreg_t batch_ptr,
        ir::vreg_t a_off, ir::vreg_t k_tail_mask) {
    const ir::vreg_t a_ptr = ir.new_gpr();
    const ir::vreg_t x_ptr = ir.new_gpr();

    ir.load(a_ptr, batch_ptr, GET_OFF_BATCH_ELEMENT(ptr.A));
    ir.add_reg(a_ptr, a_off);
    ir.load(x_ptr, batch_ptr, GET_OFF_BATCH_ELEMENT(ptr.B));
    ir.add_imm(batch_ptr, sizeof(brgemm_batch_element_t));

    // Advance the A and x pointers by one K block.
    auto advance_ptrs = [&]() {
        ir.add_imm(a_ptr, cfg.kblk_a_off);
        ir.add_imm(x_ptr, cfg.kblk_x_off);
    };

    // Reduce the full K blocks, then the tail if any. Cases by `k_blocks`.
    // k > 0 guarantees k_blocks and k_tail are never both zero.
    //   *  == 0  whole reduction is the tail
    //   *  == 1  one block, advance by hand only if a tail follows
    //   *  >= 2  loop, advancing per iteration
    if (cfg.k_blocks >= 2) {
        ir::emit_loop_imm(ir, cfg.k_blocks, [&]() {
            emit_microkernel(ir, cfg, acc, a_ptr, x_ptr);
        }, advance_ptrs);
    } else if (cfg.k_blocks == 1) {
        emit_microkernel(ir, cfg, acc, a_ptr, x_ptr);
        if (cfg.k_tail > 0) advance_ptrs();
    }
    if (cfg.k_tail > 0)
        emit_microkernel_tail(ir, cfg, acc, a_ptr, x_ptr, k_tail_mask);
}

// One M block.
//
// Contains `m_block` independent accumulators, reduced across the batch.
// They are then horizontally reduced to a single scalar each and stored.
//
// Finally, advances the A and x pointers to the next M block.
void emit_m_block(ir::ir_t &ir, const brgemv_ir_conf_t &cfg,
        const m_loop_input_regs_t &regs, int m_block) {
    std::vector<ir::vreg_t> acc(m_block, ir::vreg_t::none);
    for (int r = 0; r < m_block; r++) {
        acc[r] = ir.new_vec(cfg.dt_acc);
        ir.vzero(acc[r]);
    }

    // Batch reduction over bs dimension
    const ir::vreg_t batch_ptr = ir.new_gpr();
    ir.mov_reg(batch_ptr, regs.invariant.batch);

    if (cfg.max_bs > 1) {
        ir::emit_loop_reg(ir, regs.invariant.bs, [&]() {
            emit_bs_body(ir, cfg, acc, batch_ptr, regs.advancing.a_off,
                    regs.invariant.k_tail_mask);
        });
    } else {
        ir::emit_loop_imm(ir, 1, [&]() {
            emit_bs_body(ir, cfg, acc, batch_ptr, regs.advancing.a_off,
                    regs.invariant.k_tail_mask);
        });
    }

    // Horizontal reduction + store
    const ir::vreg_t ws = ir.new_vec(cfg.dt_acc);
    for (int r = 0; r < m_block; r++)
        ir.vhreduce(acc[r], ws);

    for (int r = 0; r < m_block; r++)
        ir.vstore_masked(regs.advancing.y_ptr,
                cfg.dt_sz_y * (dim_t)r * cfg.incy, acc[r], ir::vreg_t::none, 1);

    // Advance to next M block
    ir.add_imm(regs.advancing.a_off, cfg.mblk_a_off);
    ir.add_imm(regs.advancing.y_ptr, cfg.mblk_y_off);
}

// Builds IR for GEMV with a non-transposed A matrix.
//
// Computes:
//   y[i] = sum_k A[i][k] * x[k]
//   (m = brg.bcast_dim, k = brg.reduce_dim, n = 1)
//
// The output is partitioned into full M blocks of `m_block` rows each, with a
// final partial block of `m_tail` rows when m is not a multiple of m_block.
//
// Each M block:
// - Maintains one independent accumulator per row
// - Accumulates over k in k_block-wide chunks across the batch
// - Horizontally reduces each accumulator to a scalar output value
//   and stores it to memory
void build_gemv(const brgemm_desc_t &brg, ir::ir_t &ir) {
    const brgemv_ir_conf_t cfg(brg);
    const m_loop_input_regs_t regs = init_m_loop_input_regs(ir, cfg);

    if (cfg.m_blocks > 0)
        ir::emit_loop_imm(ir, cfg.m_blocks,
                [&]() { emit_m_block(ir, cfg, regs, cfg.m_block); });

    if (cfg.m_tail > 0) emit_m_block(ir, cfg, regs, (int)cfg.m_tail);
}

} // namespace nontrans

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
struct jit_brgemv_ir_kernel_t : public jit_base_brgemm_kernel_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemv_ir_kernel_t)

    jit_brgemv_ir_kernel_t(const brgemm_desc_t &abrg)
        : jit_base_brgemm_kernel_t(jit_name(), abrg.isa_impl), brg_(abrg) {}

    const brgemm_desc_t &get_brg() const override { return brg_; }

    void generate() override {
        // Build IR for non-transposed GEMV kernel
        ir::ir_t ir {};
        nontrans::build_gemv(brg_, ir);

        // Scratch registers (2 gpr + 3 vec) reserved for spill code.
        const int gpr_scratch0 = 10, gpr_scratch1 = 11;
        const int vec_scratch0 = 13, vec_scratch1 = 14, vec_scratch2 = 15;

        const int rsp_idx = Xbyak::Operand::RSP;
        const int param_idx = abi_param1.getIdx();

        // Build register configuration for code emission
        const ir::reg_config_t reg_cfg = ir::make_reg_config(brg_.isa_impl,
                param_idx, rsp_idx, {gpr_scratch0, gpr_scratch1},
                {vec_scratch0, vec_scratch1, vec_scratch2});

        // Register allocation
        ir::reg_alloc_result_t alloc = allocate_registers(ir, reg_cfg.pools);

        preamble();

        // Stack frame setup
        if (alloc.frame_bytes > 0) sub(rsp, (uint32_t)alloc.frame_bytes);

        // Code generation. `ir::emit` dispatches to the ISA-specific emitter
        // based on `brg_.isa_impl`.
        // The emitter may accumulate static data (e.g. the mask table) that we
        // need to write down after the postamble.
        ir::data_section_t data;
        ir::emit(*this, ir, alloc, reg_cfg, data);

        // Stack cleanup
        if (alloc.frame_bytes > 0) add(rsp, (uint32_t)alloc.frame_bytes);

        postamble();

        // Emit any static data the emitter accumulated.
        ir::emit_data_section(*this, data);
    }

private:
    brgemm_desc_t brg_;
};

// TODO: Reorganize `brgemm_kernel_t` to avoid redundant inheritance.
struct brgemv_ir_kernel_t : public brgemm_kernel_t {
    brgemv_ir_kernel_t(const brgemm_desc_t &abrd)
        : kernel_(new jit_brgemv_ir_kernel_t(abrd)) {}
    ~brgemv_ir_kernel_t() override = default;

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
    std::unique_ptr<jit_brgemv_ir_kernel_t> kernel_;
    DNNL_DISALLOW_COPY_AND_ASSIGN(brgemv_ir_kernel_t);
};

// Returns true if the descriptor is supported by the GEMV IR kernel.
bool brgemv_ir_supported(const brgemm_desc_t &brg) {
    using namespace data_type;

    if (!utils::everyone_is(f32, brg.dt_a, brg.dt_b, brg.dt_c)) return false;
    if (brg.isa_impl != avx2) return false;
    if (brg.transA) return false;

    if (brg.are_post_ops_applicable()) return false;
    if (brg.alpha != 1.0f) return false;
    if (brg.beta != 0.0f) return false;

    if (brg.is_runtime_lda || brg.is_runtime_ldc) return false;

    const int m_block = brg.gemv_bd_block();
    const int dt_sz_a = brg.typesize_A;
    const int dt_sz_y = brg.typesize_C;
    const int incy = brg.LDC;

    // Ensure indexed displacements fit in 32-bit
    auto fits = [](dim_t v) { return v <= INT32_MAX && v >= INT32_MIN; };

    if (!fits(dt_sz_a * (dim_t)m_block * brg.LDA)) return false;
    if (!fits(dt_sz_y * (dim_t)m_block * incy)) return false;

    return true;
}

brgemm_kernel_t *create_brgemv_ir_kernel(const brgemm_desc_t &brg) {
    if (!brgemv_ir_supported(brg)) return nullptr;
    return new brgemv_ir_kernel_t(brg);
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
