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

#include "gemmstone/dsl/tensor.hpp"
#include "dsl/ir/ir.hpp"
#include "dsl/ir/pass/simplify.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {

namespace layout {
std::vector<block_t> normalize_blocks(
        const std::vector<block_t> &blocks, bool remove_size_1_blocks) {
    if (blocks.empty()) return {};
    std::vector<block_t> res;

    for (const block_t &block : blocks) {
        if (remove_size_1_blocks && block.size == 1) continue;

        auto can_merge = [&](const block_t &a, const block_t &b) {
            return a.idx == b.idx && a.stride * a.size == b.stride;
        };
        if (!res.empty() && can_merge(res.back(), block)) {
            res.back().size *= block.size;
        } else {
            res.emplace_back(block);
        }
    }

    return res;
}

tile_iterator_t &tile_iterator_t::operator++() {
    for (size_t i = 0; i < d_.size(); i++) {
        if (d_[i].i < d_[i].end - 1) {
            coord_[d_[i].idx] += d_[i].stride;
            d_[i].i++;
            return *this;
        }
        coord_[d_[i].idx] -= d_[i].i * d_[i].stride;
        d_[i].i = 0;
    }
    *this = end();
    return *this;
}

tile_iterator_t::tile_iterator_t(const layout_t &layout, const tile_t &tile) {
    tile_t strides;
    for (auto &b : layout.blocks()) {
        auto &stride = strides[b.idx];
        auto tile_dim = tile.get(b.idx);
        if (tile_dim == 0) {
            d_.clear();
            return;
        }
        d_.emplace_back(b.idx, b.size, stride, tile_dim);
        coord_[b.idx] = 0;
        stride *= b.size;
    }
    if (!layout.is_empty() && layout.blocks().empty()) {
        auto idx = tile.is_empty() ? idx_t() : *tile.begin();
        d_.emplace_back(idx, 1, 1, 1);
        coord_[idx] = 0;
    }
}

} // namespace layout

layout_t::layout_t(const type_t &type, const std::vector<int64_t> &dims,
        const expr_t &offset, bool do_normalize)
    : type_(type), ndims_(dims.size()), offset_(offset) {
    if (type.is_undef()) {
        *this = layout_t();
        return;
    }

    int64_t stride = 1;
    for (int64_t i = ndims_ - 1; i >= 0; i--) {
        blocks_.emplace_back(i, dims[i], stride);
        stride *= dims[i];
    }
    if (do_normalize) blocks_ = normalize_blocks(blocks_);
    sanity_check();
}

layout_t::layout_t(const type_t &type, const std::vector<block_t> &blocks,
        const expr_t &offset, size_t ndims, bool do_normalize)
    : type_(type), ndims_(ndims), offset_(offset), blocks_(blocks) {
    if (type.is_undef()) {
        *this = layout_t();
        return;
    }

    stride_t stride(1);
    for (auto &b : blocks_) {
        if (b.stride.is_undefined()) {
            b.stride = stride;
        } else {
            stride = b.size;
        }
        stride *= b.size;
    }
    if (do_normalize) blocks_ = normalize_blocks(blocks_);
    sanity_check();
}

layout_t layout_t::with_block(block_t block) const {
    auto new_blocks = blocks();
    if (block.stride.is_unknown()) {
        new_blocks.emplace_back(block);
    } else if (block.stride.is_undefined()) {
        block.stride = !new_blocks.empty()
                ? new_blocks.back().stride * new_blocks.back().size
                : stride_t(1);
        new_blocks.emplace_back(block);
    } else {
        auto it = new_blocks.begin();
        while (it != new_blocks.end() && it->stride <= block.stride) {
            it++;
        }
        new_blocks.insert(it, block);
    }

    auto ret = with(new_blocks);
    if (ret.has_ndims()) {
        if (block.idx.index() == ret.ndims()) ret.ndims_++;
        dsl_assert(has_ndims());
    }
    return ret;
}

template <typename T>
T layout_t::offset(const coord_t &args, bool ignore_offset) const {
    if (args.is_empty()) return ir::expr_cast<T>(offset_);

    expr_t off = 0;
    auto _args = args;
    for (auto &b : blocks()) {
        if (!_args.has(b.idx)) continue;
        auto &idx = _args[b.idx];
        if (idx.is(0)) continue;

        // Do not use modulus for outermost blocks.
        auto i = is_outermost(b) ? idx : (idx % b.size);
        off = i * int64_t(b.stride) + off;
        idx /= b.size;
    }
    if (ignore_offset) return ir::expr_cast<T>(off);
    return ir::expr_cast<T>(offset_ + off);
}

template expr_t layout_t::offset<expr_t>(
        const coord_t &args, bool ignore_offset) const;
template int layout_t::offset<int>(
        const coord_t &args, bool ignore_offset) const;
template int64_t layout_t::offset<int64_t>(
        const coord_t &args, bool ignore_offset) const;

bool layout_t::is_strictly_equal(const layout_t &other, bool compare_offset,
        bool compare_strides) const {
    if (type_ != other.type_) return false;
    if (compare_offset && !offset_.is_equal(other.offset_)) return false;
    if (blocks_.size() != other.blocks_.size()) return false;
    for (size_t i = 0; i < blocks_.size(); i++) {
        auto &b0 = blocks_[i];
        auto &b1 = other.blocks_[i];
        if (b0.idx != b1.idx) return false;
        if (b0.size != b1.size) return false;
        if (compare_strides && b0.stride != b1.stride) return false;
    }
    return true;
}

bool layout_t::operator<=(const layout_t &other) const {
    if (type_ != other.type_) return false;
    auto other_blocks = other.normalize().blocks();
    auto self_blocks = normalize().blocks();
    if (self_blocks.size() > other_blocks.size()) return false;
    if (self_blocks.empty()) return true;

    int i = 0;
    for (; i < (int)self_blocks.size() - 1; i++) {
        if (self_blocks[i] != other_blocks[i]) return false;
    }
    return (self_blocks[i].idx == other_blocks[i].idx
            && self_blocks[i].stride == other_blocks[i].stride
            && other_blocks[i].size % self_blocks[i].size == 0);
}

layout_t layout_t::sub(const tile_t &tile, const coord_t &start) const {
    auto remaining_tile = tile;
    std::vector<block_t> mapped_blocks;

    for (auto &b : blocks()) {
        bool b_is_outermost = is_outermost(b);

        int64_t size = b.size;
        if (!remaining_tile.has(b.idx)) remaining_tile[b.idx] = 1;
        int64_t &rem_dim = remaining_tile[b.idx];
        if (rem_dim == 1) {
            if (b_is_outermost) {
                // This is to have similarity between the current and
                // mapped layouts.
                mapped_blocks.emplace_back(b.idx, 1, b.stride);
            }
            continue;
        }
        if (b_is_outermost) {
            size = rem_dim;
        } else if (rem_dim % size != 0) {
            // Try to split the current block and start mapping from
            // scratch.
            if (size % rem_dim == 0)
                return split_block(b, rem_dim, size / rem_dim).sub(tile, start);

            // TODO: Remove exception usage.
            stub("Can't map tensor layout.");
        }
        rem_dim /= size;
        mapped_blocks.emplace_back(b.idx, size, b.stride);
    }

    return layout_t(type(), mapped_blocks,
            start.is_empty() ? 0 : operator()(start), ndims_);
}

layout_t layout_t::split_block(
        const block_t &b, int64_t size0, int64_t size1) const {
    size_t block_idx = get_idx(b);
    dsl_assert(b.size == size0 * size1) << "Incompatible block sizes.";
    maybe_unused(b);

    auto new_blocks = blocks_;

    block_t &b0 = new_blocks[block_idx];
    block_t b1 = b0;

    b0.size = size0;
    b1.size = size1;
    b1.stride = b0.stride * size0;

    new_blocks.insert(new_blocks.begin() + block_idx + 1, b1);

    return with(new_blocks, false);
}

tile_t layout_t::max_subtile(
        int64_t max, bool is_dense, bool perfectly_divides) const {
    tile_t subtile;
    int64_t elems = 1;
    for (size_t i = 0; i < nblocks(); i++) {
        auto &b = blocks()[i];
        dsl_assert(!b.stride.is_undefined());
        if (is_dense) {
            if (b.stride.is_unknown()) return subtile;
            if (i > 0) {
                auto &b0 = blocks()[i - 1];
                if (b.stride != b0.size * b0.stride) break;
            }
        }
        if (b.size * elems >= max) {
            if (perfectly_divides)
                subtile[b.idx] *= utils::max_div(b.size, max / elems);
            else
                subtile[b.idx] *= max / elems;
            break;
        }
        subtile[b.idx] *= b.size;
        elems *= b.size;
    }
    return subtile;
}

std::string layout_t::desc_str(bool dnnl_style) const {
    if (is_empty()) return "(nil)";
    if (!dnnl_style && blocks_.empty()) return "(scalar:" + type().str() + ")";

    auto to_str = [](const idx_t &idx, bool is_outer) {
        auto ret = idx.str();
        if (ret.length() == 1) {
            if (is_outer) ret[0] -= 'a' - 'A';
            return ret;
        }
        return "<" + ret + ">";
    };
    std::string ret;
    stride_t dense_stride(1);
    idx_map_t<bool> seen;
    for (auto &b : blocks()) {
        std::string b_str;
        if (dnnl_style && is_outermost(b)) {
            b_str += to_str(b.idx, seen.get(b.idx, false));
        } else {
            b_str = std::to_string(b.size);
            b_str += to_str(b.idx, false);
        }
        if (!dnnl_style) {
            if (b.stride.is_unknown()) {
                b_str.append(1, '?');
            } else if (b.stride != dense_stride) {
                b_str.append(1, '*');
            }
        }
        b_str += ret;
        std::swap(ret, b_str);
        dense_stride = b.stride * b.size;
        seen[b.idx] = true;
    }
    ret += ":" + type().str();
    return ret;
}

void layout_t::sanity_check() const {
#if !defined(NDEBUG) || GEMMSTONE_ASSERTIONS
    // TODO: Enable enforcement of sorting, some implementation currently use
    // layout_t to define an iteration order, and in this circumstance sorting
    // is not desired as sorting blocks results in different orders.

    // for (size_t i = 0; i < blocks_.size(); i++) {
    //     dsl_assert(blocks_[i].size > 0) << "Incorrect block size.";
    //     if (i > 0)
    //         dsl_assert(blocks_[i].stride >= blocks_[i - 1].stride)
    //                 << "Block " << blocks_[i]
    //                 << " is incorrectly sorted when compared with "
    //                 << blocks_[i - 1];
    // }
    dsl_assert(has_ndims() || ndims_ == max_ndims);
#endif
}

expr_t global_tensor_t::offset(const icoord_t &sub_coord) const {
    expr_t ret = base_offset;
    for (auto &c : sub_coord) {
        ret += (coord[c] + sub_coord[c]) * strides[c];
    }
    return simplify(ret * type.size());
}

} // namespace dsl
GEMMSTONE_NAMESPACE_END
