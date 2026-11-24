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

// The TMUL compute path of the IR-based BRGEMM kernel.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/x64/brgemm/brgemm_ir_compute_path.hpp"
#include "cpu/x64/ir/ir.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace brgemm_ir {

namespace {

struct tmul_compute_path_t : public compute_path_t {
    tmul_compute_path_t(ir::ir_t &ir, const conf_t &cfg)
        : a_stride_(ir.new_gpr())
        , b_stride_(ir.new_gpr())
        , c_stride_(ir.new_gpr()) {
        ir.mov_imm(a_stride_, cfg.a_off(1, 0));
        ir.mov_imm(b_stride_, cfg.b_off(cfg.compute_path.vnni_granularity, 0));
        ir.mov_imm(c_stride_, cfg.c_off(1, 0));

        // Only the post-ops nest passes its accumulators to the output path.
        if (cfg.are_post_ops_applicable()) {
            buf_ptr_ = ir.new_gpr();
            ir.load_param(buf_ptr_, offsetof(brgemm_kernel_params_t, ptr_buf));
            buf_stride_ = ir.new_gpr();
            ir.mov_imm(buf_stride_, buf_row_size(cfg));
        }
    }

    bool loads_accumulators() const override { return true; }
    bool stores_accumulators() const override { return true; }

    // Average code size over `n_ops()` of built unrolled TMUL kernels.
    double code_bytes_per_op() const override { return 7.5; }

    // One tile per (M tile, N tile) of the block, in row order.
    std::vector<ir::vreg_t> init_accumulators(ir::ir_t &ir, const conf_t &cfg,
            const out_block_t &blk, const prev_block_t &prev) const override {
        return create_accumulators(ir, cfg, blk, ir::vreg_t::none, prev);
    }

    std::vector<ir::vreg_t> load_accumulators(ir::ir_t &ir, const conf_t &cfg,
            const out_block_t &blk, ir::vreg_t c_ptr,
            const prev_block_t &prev) const override {
        return create_accumulators(ir, cfg, blk, c_ptr, prev);
    }

    void store_accumulators(ir::ir_t &ir, const conf_t &cfg,
            const out_block_t &blk, const std::vector<ir::vreg_t> &acc,
            ir::vreg_t c_ptr) const override {

        for (int i = 0; i < (int)acc.size(); i++) {
            store_accumulator(ir, cfg, blk, acc, i, c_ptr);
        }
    }

    // Stores each tile to the start of `ptr_buf` and passes it to
    // `output_path` `store_bd_step` rows at a time, before the next tile is
    // stored. One place for every tile is faster than a place per tile.
    void pass_accumulators_to_output_path(ir::ir_t &ir, const conf_t &cfg,
            const out_block_t &blk, const std::vector<ir::vreg_t> &acc,
            const std::function<void(const out_block_t &rows_blk,
                    const std::vector<ir::vreg_t> &row_regs)> &output_path)
            const override {

        const data_type_t dt_acc = cfg.compute_path.dt_acc;
        const int n_m = m_tiles(cfg, blk);
        const int n_n = blk.ld_block2;

        for (int m = 0; m < n_m; m++) {
            for (int n = 0; n < n_n; n++) {
                const int i = m * n_n + n;
                ir.tstore(buf_ptr_, 0, acc[i], buf_stride_);

                // The tile store writes only the columns of the tile, so the
                // load of a row of the N tail zeroes the rest.
                const ir::vreg_t ld_tail_mask = is_n_tail(blk, n)
                        ? blk.ld_tail_mask
                        : ir::vreg_t::none;
                const int rows = c_tile_shape(cfg, blk, m, n).rows;

                for (int bd = 0; bd < rows; bd += store_bd_step) {
                    const int bd_block = std::min(store_bd_step, rows - bd);
                    std::vector<ir::vreg_t> row_regs(bd_block);

                    for (int r = 0; r < bd_block; r++) {
                        const dim_t off = (bd + r) * buf_row_size(cfg);
                        row_regs[r] = ir.new_vec(dt_acc);
                        if (ld_tail_mask == ir::vreg_t::none) {
                            ir.vload(row_regs[r], buf_ptr_, off, dt_acc);
                        } else {
                            ir.vload_masked(row_regs[r], buf_ptr_, off,
                                    ld_tail_mask, dt_acc);
                        }
                    }

                    const dim_t bd_start
                            = blk.bd_start + m * m_tile_rows(cfg) + bd;
                    const out_block_t rows_blk = {bd_block, 1, bd_start,
                            cfg.ld_pos(blk, n), ld_tail_mask};
                    output_path(rows_blk, row_regs);
                }
            }
        }
    }

    // One prefetch per row of each accumulator tile, the lines in C or D that
    // the row is stored to. A row shorter than a line shares it with the next
    // N tiles, so only every `tiles_per_line`-th N tile prefetches. A looped N
    // level counts the tiles from the first one of the block, so two blocks
    // can prefetch the same line.
    std::vector<std::vector<prefetch_t>> store_prefetches(const conf_t &cfg,
            const store_conf_t &store_cfg, const out_block_t &blk,
            ir::vreg_t store_ptr) const override {

        const dim_t row_size = cfg.loop.ld_block
                * (dim_t)types::data_type_size(cfg.store_dt(store_cfg));
        const dim_t tiles_per_line = std::max<dim_t>(
                1, platform::get_cache_line_size() / row_size);
        // An unrolled level counts `ld_start` from its first block.
        const dim_t first_n_tile = cfg.loop.n_form == level_form_t::unrolled
                ? blk.ld_start / cfg.loop.ld_block
                : 0;

        const int n_m = m_tiles(cfg, blk);
        const int n_n = blk.ld_block2;
        std::vector<std::vector<prefetch_t>> prefetches(n_m * n_n);

        for (int m = 0; m < n_m; m++) {
            for (int n = 0; n < n_n; n++) {
                if ((first_n_tile + n) % tiles_per_line != 0) continue;

                const int i = m * n_n + n;
                const dim_t bd = blk.bd_start + m * m_tile_rows(cfg);
                const dim_t off
                        = cfg.store_off(store_cfg, bd, cfg.ld_pos(blk, n));
                const int rows = c_tile_shape(cfg, blk, m, n).rows;
                prefetches[i].reserve(rows);
                for (int r = 0; r < rows; r++) {
                    const dim_t row_off
                            = off + r * cfg.store_off(store_cfg, 1, 0);
                    prefetches[i].push_back(
                            {store_ptr, row_off, ir::prefetch_kind_t::write});
                }
            }
        }
        return prefetches;
    }

    // Loads an A tile for each M tile and a B tile for each N tile, and
    // multiplies each pair into its accumulator. The order follows
    // `gemm_microkernel_amx()` in `jit_brgemm_amx_uker.cpp`, which issues each
    // multiplication after the next tile load. For 2x2 tiles the order is A0 B0
    // B1 T00 A1 T01 T10 T11, where Tmn multiplies into the accumulator
    // (`m`, `n`). The `tmma` into accumulator `i` is the (`i` + 1)-th one.
    //
    // The prefetches of accumulator `i` in `interleave` go before the `tmma`
    // into it. Every store of the accumulator follows its `tmma` in the last K
    // step of the block, so each prefetch precedes the store of its line.
    void emit_microkernel(ir::ir_t &ir, const conf_t &cfg,
            const out_block_t &blk, const std::vector<ir::vreg_t> &acc,
            ir::vreg_t a_ptr, ir::vreg_t b_ptr, dim_t rd_start, int rd_loop,
            const k_step_interleave_t &interleave) const override {

        const compute_path_conf_t &compute_path = cfg.compute_path;

        // A row of an A tile holds the whole K block.
        assert(rd_loop * cfg.dt_sz_a() == compute_path.a_tile[0].colsb);

        const int n_m = m_tiles(cfg, blk);
        const int n_n = blk.ld_block2;
        const int n_acc = (int)acc.size();
        std::vector<ir::vreg_t> a(n_m, ir::vreg_t::none);
        std::vector<ir::vreg_t> b(n_n, ir::vreg_t::none);

        for (int m = 0; m < n_m; m++) {
            const dim_t bd = blk.bd_start + m * m_tile_rows(cfg);
            a[m] = ir.new_tile(compute_path.dt_a,
                    compute_path.a_tile[is_m_tail(cfg, blk, m)]);
            ir.tload(a[m], a_ptr, cfg.a_off(bd, rd_start), a_stride_,
                    compute_path.a_load_hint);

            for (int n = 0; n < n_n; n++) {
                if (m == 0) {
                    const dim_t ld = cfg.ld_pos(blk, n);
                    b[n] = ir.new_tile(compute_path.dt_b,
                            compute_path.b_tile[is_n_tail(blk, n)]);
                    ir.tload(b[n], b_ptr, cfg.b_off(rd_start, ld), b_stride_,
                            compute_path.b_load_hint);
                }

                // Multiply the previous pair: (`m`, `n - 1`), or the last pair
                // of the previous M tile when `n` is 0.
                if (n > 0) {
                    const int i = m * n_n + n - 1;
                    emit_prefetches_before_tmma(ir, interleave, i);
                    ir.tmma(acc[i], a[m], b[n - 1]);
                    emit_stores_after_tmma(
                            ir, cfg, blk, acc, interleave, i + 1);
                } else if (m > 0) {
                    const int i = m * n_n - 1;
                    emit_prefetches_before_tmma(ir, interleave, i);
                    ir.tmma(acc[i], a[m - 1], b[n_n - 1]);
                    emit_stores_after_tmma(
                            ir, cfg, blk, acc, interleave, i + 1);
                }
            }
        }
        emit_prefetches_before_tmma(ir, interleave, n_acc - 1);
        ir.tmma(acc.back(), a.back(), b.back());
        emit_stores_after_tmma(ir, cfg, blk, acc, interleave, n_acc);
    }

private:
    // Number of `tmma` after the last `tmma` into an accumulator before its
    // store, when another block follows (`acc_stores_t::next_block`).
    static constexpr int store_delay = 2;

    // Rows of a tile per output path call (the uker's `store_bd_step()`).
    static constexpr int store_bd_step = 3;

    // Row strides of the A, B, and C tiles in memory, in bytes. They are set
    // once for the whole kernel.
    ir::vreg_t a_stride_, b_stride_, c_stride_;

    // `ptr_buf` and the row stride of a tile there, in bytes. `none` when
    // post-ops are not applicable. `get_wsp_buffer_size()` reserves at least
    // a tile there.
    ir::vreg_t buf_ptr_ = ir::vreg_t::none;
    ir::vreg_t buf_stride_ = ir::vreg_t::none;

    // Creates the accumulators of `blk` and loads them from C at `c_ptr`, or
    // zeroes them when `c_ptr` is `none`. Those that `init_or_load_pos()`
    // places in the first K step are loaded or zeroed there.
    std::vector<ir::vreg_t> create_accumulators(ir::ir_t &ir, const conf_t &cfg,
            const out_block_t &blk, ir::vreg_t c_ptr,
            const prev_block_t &prev) const {

        const int n_m = m_tiles(cfg, blk);
        const int n_n = blk.ld_block2;
        const int n_acc = n_m * n_n;
        std::vector<ir::vreg_t> acc(n_acc, ir::vreg_t::none);

        for (int m = 0; m < n_m; m++) {
            for (int n = 0; n < n_n; n++) {
                const int i = m * n_n + n;
                acc[i] = ir.new_tile(
                        cfg.compute_path.dt_acc, c_tile_shape(cfg, blk, m, n));
                if (init_or_load_pos(i, n_acc, prev) == 0)
                    init_or_load_accumulator(ir, cfg, blk, acc, i, c_ptr);
            }
        }
        return acc;
    }

    // Loads the accumulator `i` of `blk` from C at `c_ptr`, or zeroes it when
    // `c_ptr` is `none`.
    void init_or_load_accumulator(ir::ir_t &ir, const conf_t &cfg,
            const out_block_t &blk, const std::vector<ir::vreg_t> &acc, int i,
            ir::vreg_t c_ptr) const {

        const int m = i / blk.ld_block2;
        const int n = i % blk.ld_block2;
        if (c_ptr == ir::vreg_t::none) {
            ir.tzero(acc[i]);
        } else {
            ir.tload(acc[i], c_ptr, c_tile_off(cfg, blk, m, n), c_stride_);
        }
    }

    // Stores the accumulator `i` of `blk` to C at `c_ptr`.
    void store_accumulator(ir::ir_t &ir, const conf_t &cfg,
            const out_block_t &blk, const std::vector<ir::vreg_t> &acc, int i,
            ir::vreg_t c_ptr) const {

        const int m = i / blk.ld_block2;
        const int n = i % blk.ld_block2;
        ir.tstore(c_ptr, c_tile_off(cfg, blk, m, n), acc[i], c_stride_);
    }

    // Emits the prefetches of `interleave` for accumulator `i`, which go before
    // the `tmma` into it.
    static void emit_prefetches_before_tmma(
            ir::ir_t &ir, const k_step_interleave_t &interleave, int i) {

        if (interleave.prefetches.empty()) return;
        for (const prefetch_t &p : interleave.prefetches[i]) {
            ir.prefetch(p.ptr, p.off, p.kind);
        }
    }

    // Emits what follows the `n_tmma`-th `tmma` of a K step of `blk`: the
    // stores of `prev`, the loads or zeroes that `init_or_load_pos()` places
    // there, then at most one store of an accumulator of `blk`.
    //
    // With `next_block`, each accumulator is stored `store_delay` `tmma` after
    // its last one, so the last `store_delay` accumulators are stored in the
    // first K step of the next block. With `block`, each one is stored right
    // after its last `tmma`.
    void emit_stores_after_tmma(ir::ir_t &ir, const conf_t &cfg,
            const out_block_t &blk, const std::vector<ir::vreg_t> &acc,
            const k_step_interleave_t &interleave, int n_tmma) const {

        const prev_block_t &prev = interleave.prev;
        const int n_acc = (int)acc.size();

        for (int i = 0; i < (int)prev.acc.size(); i++) {
            if (prev_store_pos(i, n_acc, prev) != n_tmma) continue;
            store_accumulator(
                    ir, cfg, prev.blk, prev.acc, i, interleave.store_ptr);
        }
        for (int i = 0; i < n_acc; i++) {
            if (init_or_load_pos(i, n_acc, prev) != n_tmma) continue;
            init_or_load_accumulator(ir, cfg, blk, acc, i, interleave.c_ptr);
        }

        if (interleave.acc_stores == acc_stores_t::none) return;

        int delay = 0;
        if (interleave.acc_stores == acc_stores_t::next_block)
            delay = store_delay;
        const int stored = n_tmma - 1 - delay;
        if (stored >= 0)
            store_accumulator(ir, cfg, blk, acc, stored, interleave.store_ptr);
    }

    // Returns the number of the `tmma` of the first K step after which the
    // accumulator `i` of `prev` is stored, or 0 when the last K step of `prev`
    // stores it. It is the `store_delay`-th `tmma` after the last one into the
    // accumulator, but at most `n_acc`, the last `tmma` of the K step. For 2x2
    // blocks, accumulators 2 and 3 of `prev` are stored after `tmma` 1 and 2.
    static int prev_store_pos(int i, int n_acc, const prev_block_t &prev) {
        const int n_prev_tmma = (int)prev.acc.size();
        const int pos = i + 1 + store_delay - n_prev_tmma;
        if (pos <= 0) return 0;
        return std::min(pos, n_acc);
    }

    // Returns the number of the `tmma` of the first K step after which the
    // accumulator `i` is loaded or zeroed, or 0 when it is loaded or zeroed
    // with the others before the K steps. `n_acc` is the number of accumulators
    // of the block.
    //
    // An accumulator is loaded or zeroed right after a store of `prev` when its
    // first `tmma` is the `store_delay`-th one after that store. The stored
    // accumulator is no longer live there, so the number of live tiles does not
    // grow.
    static int init_or_load_pos(int i, int n_acc, const prev_block_t &prev) {
        for (int j = 0; j < (int)prev.acc.size(); j++) {
            const int pos = prev_store_pos(j, n_acc, prev);
            if (pos > 0 && pos + store_delay - 1 == i) return pos;
        }
        return 0;
    }

    // Rows of the M tiles at index 0 of `a_tile` and `c_tile`. When M has
    // only its tail tile, these are the tail rows.
    static int m_tile_rows(const conf_t &cfg) {
        return cfg.compute_path.c_tile[0][0].rows;
    }

    // Number of M tiles in `blk`. Only the last one can have fewer rows than
    // `m_tile_rows()`.
    static int m_tiles(const conf_t &cfg, const out_block_t &blk) {
        return utils::div_up(blk.bd_block, m_tile_rows(cfg));
    }

    // True if the M tile `m` of `blk` has fewer rows than `m_tile_rows()`,
    // which makes it the M tail. When M has only its tail tile, that tile
    // takes index 0, which holds the tail shape.
    static bool is_m_tail(const conf_t &cfg, const out_block_t &blk, int m) {
        const int tail_rows = blk.bd_block % m_tile_rows(cfg);
        assert(utils::one_of(tail_rows, 0, cfg.compute_path.c_tile[1][0].rows));
        return tail_rows != 0 && m == m_tiles(cfg, blk) - 1;
    }

    // True if the N tile `n` of `blk` holds the N tail, which is the last tile
    // of a block with `ld_tail_mask`.
    static bool is_n_tail(const out_block_t &blk, int n) {
        return blk.ld_tail_mask != ir::vreg_t::none && n == blk.ld_block2 - 1;
    }

    // Shape of the accumulator tile (`m`, `n`) of `blk`.
    static ir::tile_shape_t c_tile_shape(
            const conf_t &cfg, const out_block_t &blk, int m, int n) {
        const compute_path_conf_t &compute_path = cfg.compute_path;
        return compute_path.c_tile[is_m_tail(cfg, blk, m)][is_n_tail(blk, n)];
    }

    // Byte offset in C of the accumulator tile (`m`, `n`) of `blk`.
    static dim_t c_tile_off(
            const conf_t &cfg, const out_block_t &blk, int m, int n) {

        const dim_t bd = blk.bd_start + m * m_tile_rows(cfg);
        return cfg.c_off(bd, cfg.ld_pos(blk, n));
    }

    // Bytes of a row of a tile in `ptr_buf`, one vector register.
    static dim_t buf_row_size(const conf_t &cfg) {
        return cfg.loop.ld_block
                * (dim_t)types::data_type_size(cfg.compute_path.dt_acc);
    }
};

constexpr int tmul_compute_path_t::store_bd_step;

} // namespace

std::unique_ptr<compute_path_t> make_tmul_compute_path(
        ir::ir_t &ir, const conf_t &cfg) {
    return utils::make_unique<tmul_compute_path_t>(ir, cfg);
}

} // namespace brgemm_ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
