/*******************************************************************************
* Copyright 2021-2025 Intel Corporation
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

#include <cctype>
#include <sstream>
#include <thread>

#include "gpu/intel/jit/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

std::vector<layout_block_t> normalize_blocks(
        const std::vector<layout_block_t> &blocks, bool remove_size_1_blocks) {
    if (blocks.empty()) return {};
    std::vector<layout_block_t> res;

    for (const layout_block_t &block : blocks) {
        if (remove_size_1_blocks && block.block == 1) continue;

        if (!res.empty() && res.back().can_merge(block)) {
            res.back().block *= block.block;
        } else {
            res.emplace_back(block);
        }
    }

    return res;
}

memory_desc_t to_md(const layout_t &l, const memory_desc_t &md_hint) {
    auto dims_hint = md_hint.dims;
    auto ndims = md_hint.ndims;
    memory_desc_t md = {};
    md.ndims = ndims;
    std::copy(dims_hint, dims_hint + ndims, md.dims);
    md.data_type = jit::to_dnnl(l.type());
    md.offset0 = to_cpp<dim_t>(l.offset());
    md.format_kind = format_kind::blocked;

    auto &blk = md.format_desc.blocking;
    bool seen[DNNL_MAX_NDIMS] = {};

    bool in_inner_block = false;
    dim_t prev_stride = 0;

    for (auto it = l.blocks().rbegin(); it != l.blocks().rend(); ++it) {
        auto &b = *it;
        if (!seen[b.dim]) {
            // Outer block.
            gpu_assert(!in_inner_block);
            MAYBE_UNUSED(in_inner_block);
            blk.strides[b.dim] = b.stride;
            md.padded_dims[b.dim] = b.block;
        } else {
            // Inner block.
            md.padded_dims[b.dim] *= b.block;
            blk.inner_idxs[blk.inner_nblks] = b.dim;
            blk.inner_blks[blk.inner_nblks] = b.block;
            blk.inner_nblks++;
            if (prev_stride > 0) {
                // Inner block must be dense.
                gpu_assert(prev_stride == b.block * dim_t(b.stride));
            }
            prev_stride = b.stride;
            in_inner_block = true;
        }
        seen[b.dim] = true;
    }

    for (int i = 0; i < ndims; i++) {
        if (seen[i]) continue;
        gpu_assert(md.dims[i] == 1);
        md.padded_dims[i] = md.dims[i];
        blk.strides[i] = l.elems();
    }

    return md;
}

layout_t layout_t::sub(const tile_t &tile, const coord_t &start) const {
    auto remaining_tile = tile;
    std::vector<layout_block_t> mapped_blocks;

    for (auto &eb : enumerated_blocks()) {
        layout_block_t &b = eb.second;
        bool b_is_outermost = is_outermost(eb);

        dim_t block = b.block;
        if (!remaining_tile.has(b.dim)) remaining_tile[b.dim] = 1;
        dim_t &rem_dim = remaining_tile[b.dim];
        if (rem_dim == 1) {
            if (b_is_outermost) {
                // This is to have similarity between the current and
                // mapped layouts.
                mapped_blocks.emplace_back(b.dim, 1, b.stride);
            }
            continue;
        }
        if (b_is_outermost) {
            block = rem_dim;
        } else if (rem_dim % block != 0) {
            // Try to split the current block and start mapping from
            // scratch.
            if (block % rem_dim == 0)
                return split_block(eb, rem_dim, block / rem_dim)
                        .sub(tile, start);

            // TODO: Remove exception usage.
            gpu_except_not_implemented("Can't map tensor layout.");
        }
        rem_dim /= block;
        mapped_blocks.emplace_back(b.dim, block, b.stride);
    }

    return layout_t(type(), mapped_blocks,
            start.is_empty() ? 0 : operator()(start), ndims_);
}

layout_t reinterpret(
        const layout_t &layout, const type_t &new_type, bool do_normalize) {
    int old_size = layout.type().size();
    int new_size = new_type.size();
    if (new_size == old_size) return layout;

    expr_t new_offset = 0;
    if (!layout.has_zero_offset()) {
        gpu_assert(is_const(layout.offset())) << "Expected constant offset.";
        int64_t off = to_cpp<int64_t>(layout.offset()) * old_size;
        gpu_assert(off % new_size == 0);
        new_offset = off / new_size;
    }

    if (old_size % new_size != 0 && new_size % old_size != 0) {
        gpu_error_not_expected();
        return layout_t();
    }

    auto new_blocks = layout.blocks();
    if (new_blocks.empty()) {
        gpu_error_not_expected() << "Can't reinterpret.";
        return layout_t();
    }

    auto &b0 = new_blocks.front();
    if (dim_t(b0.stride) != 1) {
        gpu_error_not_expected();
        return layout_t();
    }

    if (new_size < old_size) {
        int factor = (old_size / new_size);
        b0.block *= factor;
        // Recompute strides.
        for (auto &b : new_blocks) {
            if (&b == &b0) continue;
            b.stride *= factor;
        }
    } else {
        int factor = (new_size / old_size);
        if (b0.block % factor != 0) {
            gpu_error_not_expected();
            return layout_t();
        }
        b0.block /= factor;
        // Recompute strides.
        for (auto &b : new_blocks) {
            if (&b == &b0) continue;
            if (b.stride % factor != 0) {
                gpu_error_not_expected();
                return layout_t();
            }
            b.stride /= factor;
        }
    }

    return layout_t(new_type, new_blocks, new_offset, layout.ndims(false),
            do_normalize);
}

// Reinterprets layouts to wider data type (up to 4 bytes).
// Example: 16a16b (s8 type) -> 16a4b (s32 type)
bool try_reinterpret_to_wider_type(layout_t &src, layout_t &dst,
        const tile_t &tile, bool do_update, int *new_size_out) {
    if (src.blocks().empty() || dst.blocks().empty()) return false;
    if (src.type() != dst.type()) return false;

    auto &s0 = src[0];
    auto &d0 = dst[0];
    if (s0.dim != d0.dim) return false;
    if (int(s0.stride) != 1) return false;
    if (int(d0.stride) != 1) return false;

    int old_size = src.type().size();
    int s0_old_size = int(s0.block) * old_size;
    int d0_old_size = int(d0.block) * old_size;

    int new_size = math::gcd(s0_old_size, d0_old_size);
    new_size = math::gcd(new_size, 4); // Try types up to 4 bytes.
    if (new_size <= old_size) return false;

    auto tile_ok = [&](const layout_t &l) {
        if (tile.is_empty()) return true;
        int factor = new_size / old_size;
        if (tile[l[0].dim] % factor != 0) return false;
        return true;
    };

    auto strides_ok = [&](const layout_t &l) {
        for (int i = 1; i < int(l.blocks().size()); i++) {
            auto &b = l[i];
            if (int(b.stride) * old_size % new_size != 0) return false;
        }
        return true;
    };

    while (new_size > old_size) {
        bool ok = true;
        ok &= (tile_ok(src) && tile_ok(dst));
        ok &= (strides_ok(src) && strides_ok(dst));
        if (ok) {
            if (do_update) {
                src = reinterpret(src, type_t::s(new_size * 8));
                dst = reinterpret(dst, type_t::s(new_size * 8));
            }
            if (new_size_out) *new_size_out = new_size;
            return true;
        }
        new_size /= 2;
    }
    return false;
}

layout_t layout_t::split_block(const std::pair<int, layout_block_t> &eb,
        dim_t block0, dim_t block1) const {
    int block_idx = eb.first;
    auto &b = eb.second;
    gpu_assert(b.block == block0 * block1) << "Incompatible block sizes.";
    MAYBE_UNUSED(b);

    auto new_blocks = blocks_;

    layout_block_t &b0 = new_blocks[block_idx];
    layout_block_t b1 = b0;

    b0.block = block0;
    b1.block = block1;
    b1.stride = b0.stride * block0;

    new_blocks.insert(new_blocks.begin() + block_idx + 1, b1);

    return with(new_blocks, false);
}

layout_t layout_t::split_into_multi_blocks(
        const std::vector<dim_t> &multi_blocks) const {
    if (is_empty()) return *this;

    layout_t tmp(*this);
    std::vector<dim_t> rem_elems = multi_blocks;
    std::vector<dim_t> cur_elems(rem_elems.size(), 1);
    for (auto &eb : tmp.enumerated_blocks()) {
        auto &b = eb.second;
        for (int i = 0; i < int(rem_elems.size()); i++) {
            auto &e = rem_elems[i];
            if (e == 1) continue;
            if (b.block > e) {
                // Try to split this block.
                dim_t next_block = utils::max_div(b.block, e);
                if (next_block == 1) return layout_t();
                return tmp.split_block(eb, next_block, b.block / next_block)
                        .split_into_multi_blocks(multi_blocks);
            }
            if (e % b.block != 0) return layout_t();
            e /= b.block;
            cur_elems[i] *= b.block;
            break;
        }
    }
    for (int i = 0; i < int(cur_elems.size()); i++) {
        if (cur_elems[i] != multi_blocks[i]) { return layout_t(); }
    }
    return tmp;
}

tile_t layout_t::split_into_max_tile(
        dim_t max_tile_elems, bool is_dense_tile) const {
    stride_t dense_stride = 1;
    std::vector<dim_t> tile_dims(ndims(), 1);
    dim_t cur_elems = 1;
    for (auto &eb : enumerated_blocks()) {
        auto &b = eb.second;
        if (b.block == 1) continue;
        if (b.block * cur_elems <= max_tile_elems) {
            if (is_dense_tile) {
                if (b.stride.is_unknown()) break;
                if (dense_stride != b.stride) break;
                dense_stride = b.block * b.stride;
            }
            cur_elems *= b.block;
            tile_dims[b.dim] *= b.block;
            continue;
        }
        dim_t max_block = utils::max_div(b.block, max_tile_elems / cur_elems);
        if (max_block == 1) break;
        auto tmp_layout = split_block(eb, max_block, b.block / max_block);
        return tmp_layout.split_into_max_tile(max_tile_elems, is_dense_tile);
    }
    return tile_t(tile_dims);
}

void layout_t::align_layouts(layout_t &a, layout_t &b) {
    for (auto &d : a.tile()) {
        auto a_blocks = a.blocks();
        auto b_blocks = b.blocks();

        int a_max = int(a_blocks.size());
        int b_max = int(b_blocks.size());
        int a_idx = 0;
        int b_idx = 0;

        for (;;) {
            while (a_idx < a_max && a_blocks[a_idx].dim != d)
                a_idx++;
            while (b_idx < b_max && b_blocks[b_idx].dim != d)
                b_idx++;

            if (a_idx >= a_max || b_idx >= b_max) break;

            auto &ab = a_blocks[a_idx];
            auto &bb = b_blocks[b_idx];
            dim_t common_block = math::gcd(ab.block, bb.block);
            if (ab.block == common_block && bb.block == common_block) {
                a_idx++;
                b_idx++;
                continue;
            }

            if (ab.block != common_block) {
                a = a.split_block(
                        {a_idx, ab}, common_block, ab.block / common_block);
            }
            if (bb.block != common_block) {
                b = b.split_block(
                        {b_idx, bb}, common_block, bb.block / common_block);
            }
            break;
        }
    }
}

void layout_t::sanity_check() const {
#if !defined(NDEBUG) || defined(DNNL_DEV_MODE)
    return;
#endif
    if (is_empty()) return;

    for (auto &b : blocks_) {
        gpu_assert(b.block > 0) << "Incorrect block size.";
        MAYBE_UNUSED(b);
    }
    if (with_ndims()) gpu_assert(ndims() <= max_ndims);
}

expr_t grid_splitter_t::pop_block(dim_t size) {
    gpu_assert(size > 1);
    gpu_assert(can_pop_block(size));

    dim_t new_stride = cur_stride_ * size;

    auto idx_expr = grid_.idx(cur_idx_);
    if (cur_stride_ != 1) idx_expr /= cur_stride_;
    if (new_stride != grid_.dim(cur_idx_)) idx_expr %= size;

    cur_stride_ = new_stride;
    if (cur_stride_ == grid_.dim(cur_idx_)) {
        // Move to the next dimension.
        cur_idx_--;
        skip_size_1_dims();
        cur_stride_ = 1;
    }
    return idx_expr;
}

stride_t tdim_t::compute_stride(
        const expr_t &e, dim_idx_t idx, const expr_t &var) {
    // e == var -> fixed stride.
    if (e.is_same(var)) return stride_t(1);

    auto e1 = substitute(e, var, var + 1);
    auto e_stride = simplify(e1 - e);

    if (is_const(e_stride)) return stride_t(to_cpp<dim_t>(e_stride));

    // Stride is not a constant.
    return stride_t::unknown();
}

view_t view_t::create_sub_view(const tile_t &tile, const coord_t &coord) const {
    auto ret = *this;
    for (dim_idx_t i = 0; i < nvdims(); i++) {
        ret.vdims_[i] = tile.get(i);
        if (!coord.has(i) || is_zero(coord[i])) continue;
        auto &i_start = coord[i];
        auto &s = ret.vstart_[i];
        s += i_start;
        s = simplify(s);
    }
    return ret;
}

view_t view_t::substitute(const expr_t &from, const expr_t &to) const {
    view_t ret = *this;
    for (dim_idx_t i = 0; i < nvdims(); i++) {
        ret.vstart_[i] = jit::substitute(ret.vstart_[i], from, to);
        ret.vstart_[i] = simplify(ret.vstart_[i]);
    }
    return ret;
}

std::vector<expr_t> view_t::create_vvars(dim_idx_t nvdims) {
    static const int max_nvdims = 128;
    static thread_local std::vector<expr_t> _vvars([] {
        std::vector<expr_t> ret;
        ret.reserve(max_nvdims);
        for (int i = 0; i < max_nvdims; i++)
            ret.push_back(var_t::make(type_t::s32(), "_" + std::to_string(i)));
        return ret;
    }());

    gpu_assert(nvdims <= max_nvdims) << "Too many dimensions: " << nvdims;
    return std::vector<expr_t>(_vvars.begin(), _vvars.begin() + nvdims);
}

layout_t view_t::create_pseudo_vlayout(
        const layout_t &tlayout, bool init_offset) const {
    gpu_assert(!tlayout.is_empty());

    auto rem_vdims = vdims_;
    std::vector<layout_block_t> blocks;

    for (auto &teb : tlayout.enumerated_blocks()) {
        layout_block_t &tb = teb.second;
        bool tb_is_outermost = tlayout.is_outermost(teb);
        dim_t tblock = tb.block;

        auto &tinfo = tdims_[tb.dim];
        if (tb_is_outermost) {
            // Use innermost dimension with maximum remaining size for first
            // block
            dim_idx_t max_idx = dim_idx::invalid;
            dim_idx_t max_vidx = dim_idx::invalid;
            dim_t max_vdim = 1;
            for (int i = tinfo.nvargs() - 1; i >= 0; i--) {
                dim_idx_t vidx = tinfo.vidx(i);
                if (rem_vdims[vidx] > max_vdim) {
                    max_idx = i;
                    max_vidx = vidx;
                    max_vdim = rem_vdims[vidx];
                }
            }

            if (max_vdim > 1) {
                stride_t stride = tinfo.vstride(max_idx);
                blocks.emplace_back(
                        max_vidx, max_vdim, stride * stride_t(tb.stride));
                rem_vdims[max_vidx] = 1;
            }

            for (int i = tinfo.nvargs() - 1; i >= 0; i--) {
                dim_idx_t vidx = tinfo.vidx(i);
                if (rem_vdims[vidx] == 1) continue;

                stride_t stride = tinfo.vstride(i) * tb.stride;
                blocks.emplace_back(vidx, rem_vdims[vidx], stride);
                rem_vdims[vidx] = 1;
            }
            continue;
        }

        gpu_assert(tinfo.is_identity()) << "Can't create pseudo-layout.";

        int vidx = tinfo.vidx(0);
        dim_t &rem_vdim = rem_vdims[vidx];
        if (rem_vdim == 1) continue;

        if (rem_vdim % tblock == 0) {
            rem_vdim /= tblock;
        } else if (rem_vdim % tblock != 0) {
            // Try to split the current block and start from scratch.
            if (tblock % rem_vdim == 0) {
                auto tmp_layout
                        = tlayout.split_block(teb, rem_vdim, tblock / rem_vdim);
                return create_pseudo_vlayout(tmp_layout, init_offset);
            }

            // TODO: Remove exception usage.
            gpu_except_not_implemented("Can't create pseudo-layout.");
        }
        blocks.emplace_back(tb.dim, tblock, tb.stride);
    }

    for (auto &d : rem_vdims) {
        gpu_assert(rem_vdims[d] == 1) << "Can't create pseudo-layout.";
    }

    layout_t ret(tlayout.type(), blocks, 0, nvdims());
    if (!init_offset) return ret;

    auto targs = cvt_vargs_to_targs();
    auto off = tlayout.offset(targs);
    return layout_t(tlayout.type(), blocks, off, nvdims());
}

layout_t dim_assignment_t::map(const layout_t &layout) const {
    std::vector<layout_block_t> new_blocks;
    for (auto &b : layout.blocks()) {
        int new_idx = assignments_[b.dim];
        if (new_idx == -1) continue; // Drop this block.
        auto new_b = b;
        new_b.dim = new_idx;
        new_blocks.push_back(new_b);
    }
    new_blocks = normalize_blocks(new_blocks,
            /*remove_size_1_blocks=*/false);
    auto ret = layout_t(layout.type(), new_blocks, layout.offset(), new_ndims(),
            /*do_normalize=*/false);
    gpu_assert(layout.elems() == ret.elems())
            << "Assignment doesn't preserve number of elements.";
    return ret;
}

layout_t spatials_to_3d(const layout_t &layout, bool with_groups,
        const std::array<int, 3> &dhw_map) {
    const int old_ndims = layout.ndims();
    const int old_sp_ndims = old_ndims - (with_groups ? 3 : 2);
    const int new_ndims = old_ndims - old_sp_ndims + 3;

    dim_assignment_t to_3d(old_ndims, new_ndims);
    for (int i = 0; i < old_ndims; i++) {
        if (i < old_ndims - old_sp_ndims) {
            // Non-spatial dimensions.
            to_3d.assign(i, i);
        } else {
            // Spatial dimensions.
            int old_sp_idx = 3 - (old_ndims - i);
            int new_sp_idx = dhw_map[old_sp_idx];
            to_3d.assign(i, new_ndims - (3 - new_sp_idx));
        }
    }
    return to_3d.map(layout);
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
