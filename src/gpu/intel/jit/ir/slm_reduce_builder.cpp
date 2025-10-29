/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#include "gpu/intel/jit/ir/slm_reduce_builder.hpp"

#include <algorithm>

#include "gpu/intel/jit/ir/reduce.hpp"
#include "gpu/intel/jit/ir/send_builder.hpp"
#include "gpu/intel/jit/utils/trace.hpp"
#include "gpu/intel/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

slm_reduce_builder_t::slm_reduce_builder_t(ir_context_t &ir_ctx,
        const grid_info_t &tg_grid, const expr_t &reg_buf,
        const layout_t &reg_layout, const tile_coord_t &thr_tile_coord,
        dim_idx_t dim)
    : ir_ctx_(&ir_ctx)
    , tg_grid_(tg_grid)
    , reg_buf_(reg_buf)
    , reg_layout_(reg_layout)
    , thr_tile_coord_(thr_tile_coord)
    , dim_(dim) {
    gpu_assert((dim_ != dim_idx::invalid) && (dim_ <= 2));
    gpu_assert(tg_grid_.dim(dim_) > 1);

    tmp_reg_buf_ = ir_ctx.create_tmp_var(type_t::byte(type::attr_t::ptr));
    slm_buf_ = ir_ctx.create_tmp_var(
            type_t::byte(type::attr_t::ptr), "reduce_slm");
    tg_ndims_ = (dim_ != dim_idx_t(2)) ? dim_ + 1 : tg_grid_.ndims();

    build();
}

void slm_reduce_builder_t::build() {
    auto ndims = reg_layout_.ndims();

    // Create SLM layout to store all intermediate buffers from the thread
    // group.
    layout_t slm_layout(reg_layout_.type(), reg_layout_.blocks(),
            reg_layout_.offset(), ndims + tg_ndims_);
    for (int i = tg_ndims_ - 1; i >= 0; i--) {
        slm_layout = slm_layout.with_block({ndims + i, tg_grid_.dim(i)});
    }

    slm_buf_size_ = into<int>(size_bytes(slm_layout));

    // Write thread tile to SLM.
    tile_t write_tile;
    coord_t write_start;
    tile_t reg_tile = reg_layout_.tile();
    for (size_t i = 0; i < ndims; i++)
        write_tile[i] = reg_tile[i];
    for (int i = tg_ndims_ - 1; i >= 0; i--) {
        write_start[ndims + i] = tg_grid_.idx(i);
    }
    auto write = make_access_builder(*ir_ctx_,
            view_t(slm_layout.sub(write_tile, write_start)), slm_buf_, reg_buf_,
            send_op_t::store, send_address_t::slm);
    store_stmt_ = write.stmt();

    auto &write_layout = write.reg_layout();
    gpu_assert(write_layout.is_equal_normalized(reg_layout_))
            << "Incompatible layouts.";

    // Redistribute the layout to read/reduce all k-axis tiles from every
    // thread.
    grid_info_t full_grid = tg_grid_.sub_grid({dim_});
    grid_info_t split_grid;
    auto local_thr_tile_coord = split(reg_layout_, full_grid, &split_grid);
    reg_layout_ = reg_layout_.sub(local_thr_tile_coord.tile);

    if (split_grid.elems() != full_grid.elems()) {
        for (dim_idx_t i = 0; i < full_grid.ndims(); i++) {
            if (split_grid.dim(i) == full_grid.dim(i)) continue;
            auto cond = full_grid.idx(i) < split_grid.dim(i);
            if (reduce_cond_.is_empty())
                reduce_cond_ = std::move(cond);
            else
                reduce_cond_ &= cond;
        }
    }

    tile_t read_tile;
    coord_t read_start;
    tile_t slm_tile = slm_layout.tile();
    for (size_t i = 0; i < ndims; i++) {
        read_tile[i] = local_thr_tile_coord.tile[i];
        read_start[i] = local_thr_tile_coord.coord[i];
        auto cond = read_start[i] < slm_tile[i];
        if (reduce_cond_.is_empty())
            reduce_cond_ = std::move(cond);
        else
            reduce_cond_ &= cond;
    }
    read_tile[ndims + dim_] = tg_grid_.dim(dim_);
    for (dim_idx_t i = 0; i < tg_ndims_; i++) {
        read_start[ndims + i] = (i == dim_) ? 0 : tg_grid_.idx(i);
    }
    auto read = make_access_builder(*ir_ctx_,
            view_t(slm_layout.sub(read_tile, read_start)), slm_buf_,
            tmp_reg_buf_, send_op_t::load, send_address_t::slm);

    load_stmt_ = load_stmt_.append(
            funcs::zero_out(reg_buf_, size_bytes(reg_layout_)));
    load_stmt_ = load_stmt_.append(read.stmt());

    tmp_reg_buf_size_ = std::max(tmp_reg_buf_size_, read.reg_buf_size());

    auto &read_layout = read.reg_layout();
    load_stmt_ = load_stmt_.append(create_reduce_stmt(read_layout, reg_layout_,
            tmp_reg_buf_, reg_buf_, tile_t(), reduction_mask()));

    allocs_.push_back(
            alloc_t::make(slm_buf_, slm_buf_size_, alloc_kind_t::slm));
    allocs_.push_back(
            alloc_t::make(tmp_reg_buf_, tmp_reg_buf_size_, alloc_kind_t::grf));

    if (reduce_cond_) load_stmt_ = if_t::make(reduce_cond_, load_stmt_);
    thr_tile_coord_ = thr_tile_coord_.sub(local_thr_tile_coord);
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
