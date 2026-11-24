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

#ifndef CPU_X64_BRGEMM_BRGEMM_IR_COMPUTE_PATH_HPP
#define CPU_X64_BRGEMM_BRGEMM_IR_COMPUTE_PATH_HPP

// The interface between the builder of the IR-based BRGEMM kernel
// (`brgemm_ir.cpp`) and its compute paths.

#include <cassert>
#include <functional>
#include <memory>
#include <vector>

#include "common/c_types_map.hpp"

#include "cpu/x64/brgemm/brgemm_ir_conf.hpp"
#include "cpu/x64/ir/ir.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace brgemm_ir {

// What the last K step of a block stores, with
// `store_conf_t::interleave_acc_stores`. The other K steps get `none`. A K
// step is one call of `emit_microkernel()`.
//
//   none       - nothing
//   block      - every accumulator of the block
//   next_block - some accumulators of the block. The first K step of the next
//                block stores the rest.
enum class acc_stores_t { none, block, next_block };

// The previous block and its accumulators, when the first K step of the
// current block stores some of them. `acc` is empty otherwise.
struct prev_block_t {
    out_block_t blk;
    std::vector<ir::vreg_t> acc;
};

// A prefetch of the line at byte offset `off` from `ptr`.
struct prefetch_t {
    ir::vreg_t ptr;
    dim_t off;
    ir::prefetch_kind_t kind;
};

// What a K step emits between the operations of its microkernel: the stores,
// loads, and zeroes of accumulators with `store_conf_t::interleave_acc_stores`,
// and the prefetches with `store_conf_t::prefetch`.
//
//   prev       - the previous block, for the first K step of a block only.
//                That K step stores the rest of the accumulators of `prev`,
//                then loads or zeroes some accumulators of the current block.
//   acc_stores - what the K step stores, see `acc_stores_t`
//   c_ptr      - where the K step loads accumulators from C, `none` when it
//                zeroes them
//   store_ptr  - where the accumulators are stored, C or D
//   prefetches - the prefetches the K step issues, one list per accumulator
struct k_step_interleave_t {
    prev_block_t prev;
    acc_stores_t acc_stores = acc_stores_t::none;
    ir::vreg_t c_ptr = ir::vreg_t::none;
    ir::vreg_t store_ptr = ir::vreg_t::none;
    std::vector<std::vector<prefetch_t>> prefetches;
};

// One compute path of the builder: the accumulators of an output block and
// the microkernel that reduces into them.
//
// For each output block the builder calls `init_accumulators()` before the
// batch loop, then `emit_microkernel()` for each K block of each batch element.
// A block on the output path (`uses_output_path()`) ends with
// `pass_accumulators_to_output_path()`. Any other block calls
// `load_accumulators()` instead of `init_accumulators()` when beta is 1 and
// `loads_accumulators()` is true, and ends with `store_accumulators()` unless
// the K steps store it (`k_step_interleave_t`). With `store_conf_t::prefetch`,
// it gets the prefetches of the block from `store_prefetches()`. The compute
// path chooses the layout of the accumulators.
struct compute_path_t {
    virtual ~compute_path_t() = default;

    // True if the accumulators start from C when beta is 1 and the block is
    // not on the output path. Otherwise the output path adds C.
    virtual bool loads_accumulators() const { return false; }

    // True if `store_accumulators()` ends a block with no output stages and
    // no conversion. Otherwise the output path stores the block.
    virtual bool stores_accumulators() const { return false; }

    // Average size in bytes of the code of one IR operation of a kernel on
    // this compute path. The kernel estimates its code size from it.
    virtual double code_bytes_per_op() const {
        assert(!"code_bytes_per_op: not supported by the compute path");
        return 0;
    }

    // Creates the accumulators of `blk` and zeroes them. A compute path may
    // zero some of them in the first K step instead, after the stores of
    // `prev`.
    virtual std::vector<ir::vreg_t> init_accumulators(ir::ir_t &ir,
            const conf_t &cfg, const out_block_t &blk,
            const prev_block_t &prev) const
            = 0;

    // Creates the accumulators of `blk` and loads them from C at `c_ptr`. A
    // compute path may load some of them in the first K step instead, after
    // the stores of `prev`.
    virtual std::vector<ir::vreg_t> load_accumulators(ir::ir_t &ir,
            const conf_t &cfg, const out_block_t &blk, ir::vreg_t c_ptr,
            const prev_block_t &prev) const {
        assert(!"load_accumulators: not supported by the compute path");
        return {};
    }

    // Stores the accumulators of `blk` to C at `c_ptr` unchanged.
    virtual void store_accumulators(ir::ir_t &ir, const conf_t &cfg,
            const out_block_t &blk, const std::vector<ir::vreg_t> &acc,
            ir::vreg_t c_ptr) const {
        assert(!"store_accumulators: not supported by the compute path");
    }

    // Passes the accumulators of `blk` to `output_path` as rows in vector
    // registers, in one or more calls:
    //   rows_blk - the part of `blk` the call covers
    //   row_regs - its rows, one register per (M row, N register), in row order
    // By default the whole block goes in one call, since the vector compute
    // path keeps its accumulators as rows.
    virtual void pass_accumulators_to_output_path(ir::ir_t &ir,
            const conf_t &cfg, const out_block_t &blk,
            const std::vector<ir::vreg_t> &acc,
            const std::function<void(const out_block_t &rows_blk,
                    const std::vector<ir::vreg_t> &row_regs)> &output_path)
            const {
        output_path(blk, acc);
    }

    // Returns the write prefetches of the lines at `store_ptr` that the
    // accumulators of `blk` are stored to, in C or D as the store
    // configuration of `nest_cfg` says, one list per accumulator.
    virtual std::vector<std::vector<prefetch_t>> store_prefetches(
            const conf_t &cfg, const nest_conf_t &nest_cfg,
            const out_block_t &blk, ir::vreg_t store_ptr) const {
        assert(!"store_prefetches: not supported by the compute path");
        return {};
    }

    // Reduces `rd_loop` elements of K into `acc`, starting at K index
    // `rd_start`, counted from where `a_ptr` and `b_ptr` stand. It also emits
    // what `interleave` holds between its operations, see
    // `k_step_interleave_t`.
    virtual void emit_microkernel(ir::ir_t &ir, const conf_t &cfg,
            const out_block_t &blk, const std::vector<ir::vreg_t> &acc,
            ir::vreg_t a_ptr, ir::vreg_t b_ptr, dim_t rd_start, int rd_loop,
            const k_step_interleave_t &interleave) const
            = 0;
};

std::unique_ptr<compute_path_t> make_vector_compute_path();

// The TMUL compute path sets the row strides of its tiles once, at the end of
// `ir`, so it is created before the loop nest is built.
std::unique_ptr<compute_path_t> make_tmul_compute_path(
        ir::ir_t &ir, const conf_t &cfg);

} // namespace brgemm_ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
