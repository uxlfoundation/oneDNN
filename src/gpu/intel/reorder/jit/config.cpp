/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "gpu/intel/reorder/jit/config.hpp"
#include "gpu/intel/jit/ir/reorder.hpp"
#include "gpu/intel/reorder/jit/tiler.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace reorder {
namespace jit {

config_t::config_t(
        const dsl::kernel::options_t &ec, layout_t src, layout_t dst) {
    set_options(ec);

    reorder_t::normalize(src, dst);
    src_layout().set_user(src);
    dst_layout().set_user(dst);

    auto rev_tiles = jit::tiles(ec.hw(), src, dst);
    tiles_.assign(rev_tiles.rbegin(), rev_tiles.rend());

    auto ndims = src.ndims();
    const auto &thr_tile = tiles_.front();

    tile_t iter_tile;
    tile_t loop_tile;
    tile_t tg_tile;

    tile_t dims;
    dim_idx_t grid_idx = 0;

    for (size_t i = 0; i < ndims; ++i) {
        pvar_t d(i);

        dim_t tg_dim = thr_tile[i];
        dim_t outer = utils::div_up(dst.elems(i), tg_dim);
        iter_tile[d] = tg_dim;
        loop_tile[d] = 1;
        dims[d] = std::max(src.elems(i), dst.elems(i));
        grid_[grid_idx][d] = 1;

        if (outer != 1) grid_idx = std::min<dim_idx_t>(grid_idx + 1, 2);
    }

    for (size_t i = 0; i < ndims; ++i) {
        pvar_t d(i);
        dim_t tg_dim = thr_tile[i];
        dim_t outer = utils::div_up(dims[d], tg_dim);

        if (outer % 2 == 0) {
            tg_tile[d] = 2;
            break;
        }
    }

    padded_dims().set(dims);
    iter_dims().set(iter_tile);
    loop_dims().set(loop_tile);
    thread_group_dims().set(tg_tile);

    init_kernel_grid(grid_);
    init_thread_group_grid(grid_);
}

compute::nd_range_t config_t::nd_range() const {
    compute::range_t gws = compute::range_t::empty();
    compute::range_t lws = compute::range_t::empty();
    for (dim_idx_t i = 0; i < compute::range_t::max_ndims; ++i) {
        lws[i] = thread_group_grid().dim(i);
        gws[i] = kernel_grid().dim(i) * lws[i];
    }
    lws[0] *= simd();
    gws[0] *= simd();

    return compute::nd_range_t(gws, lws);
}

} // namespace jit
} // namespace reorder
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
