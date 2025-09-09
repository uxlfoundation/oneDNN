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

#ifndef GPU_INTEL_JIT_IR_TENSOR_HPP
#define GPU_INTEL_JIT_IR_TENSOR_HPP

#include <algorithm>
#include <array>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>
#include <unordered_map>

#include "common/memory_desc_wrapper.hpp"
#include "gpu/intel/block_structure.hpp"
#include "gpu/intel/jit/ir/ir.hpp"
#include "gpu/intel/jit/ir/problem.hpp"
#include "gpu/intel/jit/pass/simplify.hpp"
#include "gpu/intel/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

class grid_info_t {
public:
    grid_info_t() = default;
    grid_info_t(dim_idx_t ndims) : dims_(ndims), offs_(ndims), idxs_(ndims) {}
    grid_info_t(const std::vector<dim_t> &dims, const std::vector<expr_t> &idxs)
        : grid_info_t(dims, {}, idxs) {}
    grid_info_t(const std::vector<dim_t> &dims, std::string (*genname)(int))
        : grid_info_t(dims, make_idxs(genname, into<dim_idx_t>(dims.size()))) {}
    grid_info_t(const std::vector<dim_t> &dims, const std::vector<dim_t> &offs,
            const std::vector<expr_t> &idxs)
        : dims_(dims), offs_(offs), idxs_(idxs) {
        if (offs_.empty()) offs_.resize(dims.size());
        gpu_assert(dims_.size() == offs_.size());
        gpu_assert(dims_.size() == idxs_.size());
    }

    bool operator==(const grid_info_t &other) const {
        if (ndims() != other.ndims()) return false;
        for (dim_idx_t i = 0; i < ndims(); i++) {
            if (dim(i) != other.dim(i)) return false;
            if (off(i) != other.off(i)) return false;
            if (!idx(i).is_equal(other.idx(i))) return false;
        }
        return true;
    }

    bool is_empty() const { return dims_.empty(); }

    dim_t &dim(dim_idx_t dim_idx) { return dims_[dim_idx]; }
    dim_t &off(dim_idx_t dim_idx) { return offs_[dim_idx]; }
    expr_t &idx(dim_idx_t dim_idx) { return idxs_[dim_idx]; }
    dim_idx_t dim_idx(const expr_t &idx_var) const {
        for (dim_idx_t i = 0; i < ndims(); i++) {
            if (idx(i).is_same(idx_var)) return i;
        }
        gpu_error_not_expected() << "Index not found: " << idx_var;
        return -1;
    }

    const dim_t &dim(dim_idx_t dim_idx) const { return dims_[dim_idx]; }
    const dim_t &dim(const expr_t &idx_var) const {
        return dims_[dim_idx(idx_var)];
    }
    const dim_t &off(dim_idx_t dim_idx) const { return offs_[dim_idx]; }
    const expr_t &idx(dim_idx_t dim_idx) const { return idxs_[dim_idx]; }

    dim_t &operator[](dim_idx_t dim_idx) { return dim(dim_idx); }
    const dim_t &operator[](dim_idx_t dim_idx) const { return dim(dim_idx); }

    dim_idx_t ndims() const { return into<dim_idx_t>(dims_.size()); }
    dim_t elems() const {
        return utils::array_product(dims_.data(), dims_.size());
    }

    grid_info_t sub_grid(std::initializer_list<dim_idx_t> old_dim_idxs) const {
        grid_info_t ret(into<dim_idx_t>(old_dim_idxs.size()));
        dim_idx_t new_dim_idx = 0;
        for (auto old_dim_idx : old_dim_idxs) {
            ret.dim(new_dim_idx) = dim(old_dim_idx);
            ret.off(new_dim_idx) = off(old_dim_idx);
            ret.idx(new_dim_idx) = idx(old_dim_idx);
            new_dim_idx++;
        }
        return ret;
    }

    grid_info_t resize(const std::vector<dim_t> &new_dims) const {
        grid_info_t ret = *this;
        ret.dims_ = new_dims;
        return ret;
    }

    grid_info_t slice(dim_idx_t dim_idx, dim_t new_off, dim_t new_dim,
            const expr_t &new_idx, expr_t &new_idx_value) const {
        gpu_assert(dim_idx >= 0 && dim_idx < ndims());
        gpu_assert(new_dim > 0 && new_off >= 0);
        gpu_assert(new_off + new_dim <= dims_[dim_idx]);

        grid_info_t ret = *this;
        ret.offs_[dim_idx] += new_off;
        ret.dims_[dim_idx] = new_dim;
        if (new_off > 0) {
            new_idx_value = ret.idxs_[dim_idx] - new_off;
            ret.idxs_[dim_idx] = new_idx;
        } else {
            new_idx_value = expr_t();
        }
        ret.parent_dims_ = (parent_dims_.empty() ? dims_ : parent_dims_);
        return ret;
    }

    grid_info_t halven(const expr_t &new_idx, dim_idx_t &dim_idx,
            expr_t &new_idx_value, bool first = true) const {
        for (int i = ndims() - 1; i >= 0; i--) {
            if (dim(i) == 1 || dim(i) % 2 != 0) continue;
            dim_idx = i;
            if (first) return slice(i, 0, dim(i) / 2, new_idx, new_idx_value);
            return slice(i, dim(i) / 2, dim(i) / 2, new_idx, new_idx_value);
        }
        return grid_info_t();
    }

    expr_t slice_condition() const {
        if (parent_dims_.empty()) return expr_t();
        expr_t ret(true);
        for (dim_idx_t i = 0; i < ndims(); i++) {
            auto &idx = idxs_[i];
            if (offs_[i] > 0) ret &= (idx >= 0);
            if (offs_[i] + dims_[i] < parent_dims_[i]) ret &= (idx < dims_[i]);
        }
        if (ret.is_equal(expr_t(true))) return expr_t();
        return ret;
    }

    std::string str() const {
        ostringstream_t oss;
        oss << ir_utils::make_seq_print_helper(dims_, " x ");
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    static std::vector<expr_t> make_idxs(std::string (*genname)(int), int n) {
        std::vector<expr_t> ret;
        ret.reserve(n);
        for (int i = 0; i < n; i++)
            ret.push_back(var_t::make(type_t::s32(), genname(i)));
        return ret;
    }

    std::vector<dim_t> dims_;
    std::vector<dim_t> offs_;
    std::vector<expr_t> idxs_;

    std::vector<dim_t> parent_dims_;
};

class grid_splitter_t {
public:
    grid_splitter_t(const grid_info_t &grid)
        : grid_(grid), cur_idx_(grid.ndims() - 1), cur_stride_(1) {
        skip_size_1_dims();
        gpu_assert(cur_idx_ != dim_idx::invalid);
    }

    dim_t cur_block() const {
        if (is_empty()) return 1;

        return grid_.dim(cur_idx_) / cur_stride_;
    }

    bool is_empty() const { return cur_idx_ == dim_idx::invalid; }

    bool can_pop_block(dim_t size) const {
        if (is_empty()) return false;
        return cur_block() % size == 0;
    }

    expr_t pop_block(dim_t size);

private:
    void skip_size_1_dims() {
        while (cur_idx_ != dim_idx::invalid && grid_.dim(cur_idx_) == 1)
            cur_idx_--;
    }

    grid_info_t grid_;

    dim_idx_t cur_idx_;
    dim_t cur_stride_;
};

struct layout_block_t {
    layout_block_t() = default;

    layout_block_t(const pvar_t &dim, dim_t block, const stride_t &stride)
        : dim(dim), block(block), stride(stride) {}

    bool can_merge(
            const layout_block_t &other, bool same_dim_only = true) const {
        bool dim_ok = !same_dim_only || (dim == other.dim);
        bool is_dense = (stride * block == other.stride);
        return dim_ok && is_dense;
    }

    bool operator==(const layout_block_t &other) const {
        return (dim == other.dim) && (block == other.block)
                && (stride == other.stride);
    }
    bool operator!=(const layout_block_t &other) const {
        return !(*this == other);
    }

    size_t get_hash() const { return 0; }

    std::string str() const {
        std::ostringstream oss;
        oss << "block_t(dim = " << dim;
        oss << ", block = " << block;
        oss << ", stride = " << stride.str();
        oss << ")";
        return oss.str();
    }

    bool is_empty() const { return dim.is_undef(); }

    pvar_t dim;
    dim_t block = 1; // Block size.
    stride_t stride; // Stride between elements of the block.
};

std::vector<layout_block_t> normalize_blocks(
        const std::vector<layout_block_t> &blocks,
        bool remove_size_1_blocks = true);

class layout_t {
public:
    static const dim_idx_t max_ndims = 16;

    layout_t() : type_(type_t::undef()), ndims_(0), offset_(0) {
        sanity_check();
    }

    layout_t(const type_t &type, const expr_t &offset, dim_idx_t ndims,
            const std::vector<std::pair<pvar_t, dim_t>> &parts,
            const std::vector<dim_t> &dims = {}, bool do_normalize = true);

    layout_t(const type_t &type, const expr_t &offset,
            const std::string &format, const std::vector<dim_t> &dims = {},
            bool do_normalize = true)
        : layout_t(type, offset, into<dim_idx_t>(dims.size()),
                parse_format(format, into<dim_idx_t>(dims.size())), dims,
                do_normalize) {}

    layout_t(const memory_desc_wrapper &mdw, const std::string &format,
            bool do_normalize = true)
        : layout_t(to_ir(mdw.data_type()), mdw.offset0(), format,
                std::vector<dim_t>(mdw.dims(), mdw.dims() + mdw.ndims()),
                do_normalize) {}

    layout_t(const memory_desc_wrapper &mdw, const char *format,
            bool do_normalize = true)
        : layout_t(mdw, std::string(format), do_normalize) {}

    layout_t(const memory_desc_wrapper &mdw, bool do_normalize = true);

    layout_t(const type_t &type, const expr_t &offset,
            const std::vector<dim_t> &dims, bool do_normalize = true)
        : type_(type), ndims_(into<dim_idx_t>(dims.size())), offset_(offset) {
        dim_t stride = 1;
        for (int i = ndims_ - 1; i >= 0; i--) {
            blocks_.emplace_back(i, dims[i], stride);
            stride *= dims[i];
        }
        if (do_normalize) blocks_ = normalize_blocks(blocks_);
        sanity_check();
    }

    layout_t(const type_t &type, dim_idx_t ndims, const expr_t &offset,
            const std::vector<layout_block_t> &blocks, bool do_normalize = true)
        : type_(type), ndims_(ndims), offset_(offset), blocks_(blocks) {
        if (do_normalize) blocks_ = normalize_blocks(blocks_);
        sanity_check();
    }

    layout_t(const type_t &type, const expr_t &offset,
            const std::vector<layout_block_t> &blocks, bool do_normalize = true)
        : layout_t(type, dim_idx::invalid, offset, blocks, do_normalize) {}

    layout_t(const type_t &type, const expr_t &offset, const layout_t &other,
            bool do_normalize)
        : layout_t(type, other.ndims(), offset, other.blocks(), do_normalize) {}

    layout_t(const type_t &type, const expr_t &offset = expr_t(0))
        : layout_t(type, dim_idx::invalid, offset,
                std::vector<layout_block_t>()) {}

    bool is_empty() const { return ndims_ == 0; }
    bool with_ndims() const { return ndims_ != dim_idx::invalid; }
    dim_idx_t ndims() const {
        gpu_assert(with_ndims());
        return ndims_;
    }

    dim_t elems(const pvar_t &dim = {}) const {
        dim_t ret = 1;
        for (auto &b : blocks_) {
            if (!dim.is_undef() && b.dim != dim) continue;
            ret *= b.block;
        }
        return ret;
    }

    // Storage size in bytes.
    dim_t size() const {
        if (is_empty()) return 0;
        dim_t max_off = 0;
        dim_t max_block_size = 0;
        for (auto &b : blocks_) {
            max_off += (b.block - 1) * (dim_t)b.stride;
            max_block_size
                    = std::max(max_block_size, b.block * (dim_t)b.stride);
        }
        dim_t max_elems = std::max(max_off + 1, max_block_size);
        return max_elems * type().size() / type().packing();
    }

    // Offset in bytes following the last accessible element.
    dim_t max_off_bytes(bool ignore_offset = false) const {
        if (is_empty()) return 0;
        dim_t max_off = 0;
        for (auto &b : blocks_) {
            max_off += (b.block - 1) * (dim_t)b.stride;
        }
        dim_t after_last = max_off + 1;
        if (!ignore_offset) after_last += expr_cast<dim_t>(offset_);
        return after_last * type().size() / type().packing();
    }

    template <typename T = expr_t>
    T offset(const coord_t &args = {}, bool ignore_offset = false) const {
        if (args.is_empty()) return expr_cast<T>(offset_);

        expr_t off = 0;
        auto _args = args;
        for (auto &eb : enumerated_blocks()) {
            auto &b = eb.second;
            if (!_args.has(b.dim)) continue;
            auto &idx = _args[b.dim];
            if (is_zero(idx)) continue;

            // Do not use modulus for outermost blocks.
            auto i = is_outermost(eb) ? idx : (idx % b.block);
            off = i * dim_t(b.stride) + off;
            idx /= b.block;
        }
        if (ignore_offset) return expr_cast<T>(off);
        return expr_cast<T>(offset_ + off);
    }

    const type_t &type() const { return type_; }

    std::vector<dim_t> dims() const {
        gpu_assert(with_ndims());
        std::vector<dim_t> dims(ndims(), 1);
        for (auto &b : blocks_) {
            dims[b.dim] *= b.block;
        }
        return dims;
    }

    tile_t tile() const {
        tile_t tile;
        for (auto &b : blocks_)
            tile[b.dim] = tile.get(b.dim, 1) * b.block;
        return tile;
    }

    dim_t dim(const pvar_t &dim) const { return elems(dim); }

    stride_t stride(const pvar_t &dim, int dim_block_idx = 0) const {
        int idx = 0;
        for (auto &b : blocks_) {
            if (b.dim != dim) continue;
            if (idx == dim_block_idx) { return b.stride; }
            idx++;
        }
        return stride_t();
    }

    int nblocks() const { return (int)blocks().size(); }

    const std::vector<layout_block_t> &blocks() const { return blocks_; }

    dim_t inner_block(const pvar_t &dim, bool skip_outer = true,
            bool inner_only = true) const {
        std::vector<dim_t> dim_blocks;
        for (auto &b : blocks_) {
            if (b.dim == dim) dim_blocks.push_back(b.block);
        }
        dim_t ret = 1;
        int nblocks = (int)dim_blocks.size();
        int lo = 0;
        int hi = skip_outer ? nblocks - 1 : nblocks;
        if (inner_only) hi = std::min(hi, 1);
        for (int i = lo; i < hi; i++)
            ret *= dim_blocks[i];
        return ret;
    }

    void set_offset(const expr_t &offset) { offset_ = offset; }

    bool is_strictly_equal(const layout_t &other, bool compare_offset = true,
            bool compare_strides = true) const {
        if (type_ != other.type_) return false;
        if (compare_offset && !offset_.is_equal(other.offset_)) return false;
        if (blocks_.size() != other.blocks_.size()) return false;
        for (size_t i = 0; i < blocks_.size(); i++) {
            auto &b0 = blocks_[i];
            auto &b1 = other.blocks_[i];
            if (b0.dim != b1.dim) return false;
            if (b0.block != b1.block) return false;
            if (compare_strides && b0.stride != b1.stride) return false;
        }
        return true;
    }

    bool operator==(const layout_t &other) const { return is_equal(other); }

    bool operator!=(const layout_t &other) const { return !operator==(other); }
    bool operator<=(const layout_t &other) const {
        if (type_ != other.type_) return false;
        auto other_blocks = other.normalize().blocks();
        auto self_blocks = normalize().blocks();
        if (self_blocks.size() > other_blocks.size()) return false;
        if (self_blocks.empty()) return true;

        int i = 0;
        for (; i < (int)self_blocks.size() - 1; i++) {
            if (self_blocks[i] != other_blocks[i]) return false;
        }
        return (self_blocks[i].dim == other_blocks[i].dim
                && self_blocks[i].stride == other_blocks[i].stride
                && other_blocks[i].block % self_blocks[i].block == 0);
    }
    bool operator>=(const layout_t &other) const { return other <= *this; }

    bool is_equal(const layout_t &other, bool compare_offset = true) const {
        return normalize().is_strictly_equal(other.normalize(), compare_offset);
    }

    size_t get_hash() const {
        return ir_utils::get_hash(type_, ndims_, offset_, blocks_);
    }

    expr_t operator()(const coord_t &coord) const { return offset(coord); }

    template <typename T = expr_t>
    T offset_in_bytes(
            const coord_t &coord = {}, bool ignore_offset = false) const {
        return offset<T>(coord, ignore_offset) * type().size()
                / type().packing();
    }

    std::string desc_str(bool dnnl_style = false) const {
        if (is_empty()) return "(nil)";
        if (!dnnl_style && blocks_.empty())
            return "(scalar:" + type().str() + ")";

        auto to_str = [](const pvar_t &dim, bool is_outer) {
            auto ret = dim.str();
            if (ret.length() == 1) {
                if (is_outer) ret[0] -= 'a' - 'A';
                return ret;
            }
            return "<" + ret + ">";
        };
        std::string ret;
        stride_t dense_stride(1);
        pvar_map_t<bool> seen;
        for (auto &eb : enumerated_blocks()) {
            auto &b = eb.second;
            std::string b_str;
            if (dnnl_style && is_outermost(eb)) {
                b_str += to_str(b.dim, seen.get(b.dim, false));
            } else {
                b_str = std::to_string(b.block);
                b_str += to_str(b.dim, false);
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
            dense_stride = b.stride * b.block;
            seen[b.dim] = true;
        }
        ret += ":" + type().str();
        return ret;
    }

    std::string str() const {
        if (is_empty()) return "(nil)";
        ostringstream_t oss;
        oss << desc_str();
        if (!has_zero_offset()) oss << " offset: " << offset_;
        return oss.str();
    }

    IR_DEFINE_DUMP()

    memory_desc_t to_dnnl(const dim_t *dims_hint) const;

    // Returns a vector of <block index, block> pairs.
    // The innermost block (first) has index 0.
    std::vector<std::pair<int, layout_block_t>> enumerated_blocks() const {
        std::vector<std::pair<int, layout_block_t>> ret;
        ret.reserve(blocks_.size());
        for (int i = 0; i < int(blocks_.size()); i++) {
            ret.emplace_back(i, blocks_[i]);
        }
        return ret;
    }

    std::vector<dim_t> strides(const pvar_t &dim) const {
        std::vector<dim_t> ret;
        for (auto &b : blocks_)
            if (b.dim == dim) ret.push_back(b.stride);
        return ret;
    }

    // eb is <block index, block> pair, see enumerated_blocks().
    bool is_outermost(const std::pair<int, layout_block_t> &eb) const {
        return is_outermost(eb, blocks_);
    }

    bool is_plain() const {
        pvar_map_t<bool> seen;
        for (auto &b : blocks_) {
            if (seen.has(b.dim)) return false;
            seen[b.dim] = true;
        }
        return true;
    }

    bool has_zero_offset() const { return offset_.is_equal(expr_t(0)); }

    bool has_unknown_strides() const {
        for (auto &b : blocks_)
            if (b.stride.is_unknown()) return true;
        return false;
    }

    // Returns a canonical representation of the layout:
    // - Size one blocks are removed
    // - Consecutive dense blocks are merged
    layout_t normalize() const {
        auto blocks = normalize_blocks(blocks_);
        return layout_t(type(), ndims_, offset(), blocks);
    }

    layout_t transpose() const {
        gpu_assert(with_ndims())
                << "transpose() requires zero-based dimensions.";
        if (ndims() != 2) gpu_error_not_expected();

        // Flip: 0 -> 1, 1 -> 0.
        auto blocks = blocks_;
        for (auto &b : blocks)
            b.dim = pvar_t(1 - b.dim.index());

        return layout_t(type(), ndims(), offset(), blocks);
    }

    // Returns a new (sub-)layout that fully contains the passed sub-tensor.
    // Strides are kept unchanged.
    // Assumption: the original layout can be tiled by the passed sub-tensor.
    // For example: XaYb4a2b can be tiled into 2x2 sub-tensors but it's not
    // possible to tile it into 3x2 sub-tensors.
    layout_t sub(const tile_t &tile, const coord_t &start = {}) const;

    layout_t sub(const tile_coord_t &tile_coord) const {
        return sub(tile_coord.tile, tile_coord.coord);
    }

    layout_t reinterpret(
            const type_t &new_type, bool do_normalize = true) const;

    layout_t retype(const type_t &new_type) const {
        auto ret = *this;
        ret.type_ = new_type;
        return ret;
    }

    bool is_dense() const {
        stride_t stride = 1;
        for (auto &b : blocks_) {
            if (b.stride != stride) return false;
            stride *= b.block;
        }
        return true;
    }

    bool is_blocked_by(const pvar_t &dim, int block) const {
        if (block == 1) return true;
        if (nblocks() == 0) return false;
        auto &b0 = blocks()[0];
        if (b0.dim != dim) return false;
        if (b0.block % block != 0) return false;
        return true;
    }

    layout_t innermost_block_layout() const {
        pvar_map_t<int> block_count;
        for (auto &b : blocks_)
            block_count[b.dim]++;

        std::vector<layout_block_t> inner_blocks;

        stride_t stride = 1;
        for (auto &b : blocks_) {
            if (b.stride != stride) break; // Not dense anymore.
            if (block_count[b.dim] == 1) break; // Outer block.
            stride *= b.block;
            gpu_assert(block_count[b.dim] > 0);
            block_count[b.dim]--;
            inner_blocks.push_back(b);
        }
        return layout_t(type(), ndims_, 0, inner_blocks);
    }

    // Returns a packed layout where all blocks are contiguous, without gaps.
    layout_t make_dense() const {
        dim_t stride = 1;
        auto new_blocks = blocks_;
        for (auto &b : new_blocks) {
            b.stride = stride;
            stride *= b.block;
        }
        return layout_t(type(), ndims_, 0, new_blocks);
    }

    layout_t make_strided(int _stride, int block_idx = 0) const {
        auto new_blocks = blocks_;
        int factor = 1;
        for (int i = 0; i < (int)new_blocks.size(); i++) {
            auto &b = new_blocks[i];
            if (i == block_idx) {
                int i_stride = (int)b.stride;
                if (_stride % i_stride == 0) {
                    factor = (_stride / i_stride);
                } else if (i_stride % _stride == 0) {
                    factor = -(i_stride / _stride);
                } else {
                    gpu_error_not_expected();
                }
            }
            if (factor > 0) {
                b.stride *= factor;
            } else {
                b.stride = ir_utils::safe_divide((dim_t)b.stride, -factor);
            }
        }
        return layout_t(type(), ndims_, 0, new_blocks);
    }

    layout_t make_with_block(const layout_t &inner) const {
        gpu_assert(type() == inner.type());
        gpu_assert(ndims() == inner.ndims());
        auto cur_tile = tile();
        tile_t rem_tile;
        for (auto &d : cur_tile)
            rem_tile[d] = ir_utils::safe_divide(cur_tile.at(d), inner.elems(d));
        auto ret = inner;
        for (auto &b : blocks()) {
            auto &d = cur_tile[b.dim];
            auto &r = rem_tile[b.dim];
            d = ir_utils::safe_divide(d, b.block);
            if (r <= d) continue;
            auto blk = ir_utils::safe_divide(r, d);
            ret = ret.add_outer_block(b.dim, blk);
            r = ir_utils::safe_divide(r, blk);
        }
        for (auto &d : rem_tile)
            gpu_assert(rem_tile[d] == 1);
        return ret;
    }

    // Returns an equivalent layout where the specified block is split into two.
    // block0 - inner block size.
    // block1 - outer block size.
    layout_t split_block(const std::pair<int, layout_block_t> &eb, dim_t block0,
            dim_t block1) const;

    // Splits blocks so that they can be used to form `multi_blocks` without
    // crossing the block boundaries. `multi_blocks` are ordered from innermost
    // to outermost. Returns an empty layout if such a split is not possible.
    // Example (all blocks are ordered from innermost to outermost):
    //     Input blocks:  [4, 4, 2]
    //     Multi-blocks:  [8, 2]
    //     Output blocks: [4, 2, 2, 2]
    layout_t split_into_multi_blocks(
            const std::vector<dim_t> &multi_blocks) const;

    layout_t add_outer_block(
            const pvar_t &dim, dim_t block, dim_t stride = -1) const {
        if (stride == -1) {
            if (blocks_.empty()) {
                stride = 1;
            } else {
                auto &last = blocks_.back();
                stride = last.block * last.stride;
            }
        }
        gpu_assert(stride >= elems());
        if (with_ndims()) gpu_assert(dim.index() < ndims());
        auto new_blocks = blocks();
        new_blocks.emplace_back(dim, block, stride);
        return layout_t(type(), ndims_, offset(), new_blocks);
    }

    layout_t add_outer_block_and_pad(
            const pvar_t &dim, dim_t block, int pad_bytes) const {
        int type_size = type().size();
        gpu_assert(pad_bytes % type_size == 0);
        if (blocks_.empty())
            return add_outer_block(dim, block, pad_bytes / type_size);
        auto &last = blocks_.back();
        auto stride = utils::rnd_up((dim_t)last.stride * last.block,
                (dim_t)(pad_bytes / type_size));
        return add_outer_block(dim, block, stride);
    }

    // Returns a tensor corresponding to the biggest innermost sub-layout so that
    // 1) It consists of consecutive blocks only.
    // 2) It contains less or equal than max_tile_elems elements.
    // 3) It is dense if is_dense_tile is true.
    tile_t split_into_max_tile(dim_t max_tile_elems, bool is_dense_tile) const;

    tile_coord_t split(const grid_info_t &grid_info,
            grid_info_t *out_grid = nullptr) const {
        tile_coord_t min_tile_coord = tile_coord_t::invalid();
        std::vector<dim_t> cur_dims(grid_info.ndims(), 1);

        for (int iter = 0; iter < grid_info.elems(); iter++) {
            for (dim_idx_t i = 0; i < grid_info.ndims(); i++) {
                if (++cur_dims[i] <= grid_info.dim(i)) break;
                cur_dims[i] = 1;
            }
            auto sub_grid = grid_info.resize(cur_dims);
            auto tile_coord = split_exact(sub_grid);
            if (tile_coord.is_invalid()) continue;
            if (min_tile_coord.is_invalid()
                    || tile_coord.elems() < min_tile_coord.elems()) {
                min_tile_coord = std::move(tile_coord);
                if (out_grid) { *out_grid = std::move(sub_grid); }
            }
        }
        return min_tile_coord;
    }

    tile_coord_t split_exact(const grid_info_t &grid) const {
        tile_t tile;
        if (elems() % grid.elems() != 0) return tile_coord_t::invalid();

        dim_t cur_elems_per_tile = 1;
        dim_t elems_per_tile = elems() / grid.elems();
        for (auto &b : blocks()) {
            dim_t block
                    = std::min(b.block, elems_per_tile / cur_elems_per_tile);
            tile[b.dim] = tile.get(b.dim, 1) * block;
            cur_elems_per_tile *= block;
        }
        if (cur_elems_per_tile != elems_per_tile)
            return tile_coord_t::invalid();

        return split(tile, grid);
    }

    tile_coord_t split_exact(int factor) const {
        if (factor == 1) return tile_coord_t(tile());
        if (elems() % factor != 0) return tile_coord_t::invalid();
        dim_t cur_elems = 1;
        dim_t split_elems = elems() / factor;
        std::vector<layout_block_t> split_blocks;
        for (auto &b : blocks()) {
            if (cur_elems * b.block > split_elems) {
                if (split_elems % cur_elems != 0)
                    return tile_coord_t::invalid();
                auto bb = b;
                bb.block = split_elems / cur_elems;
                if (b.block % bb.block != 0) return tile_coord_t::invalid();
                split_blocks.push_back(bb);
            } else {
                split_blocks.push_back(b);
            }
            cur_elems *= split_blocks.back().block;
            if (cur_elems == split_elems) break;
        }
        tile_t split_tile;
        for (auto &b : split_blocks)
            split_tile[b.dim] = split_tile.get(b.dim, 1) * b.block;
        return tile_coord_t(split_tile);
    }

    tile_coord_t split(const tile_t &tile, const grid_info_t &grid,
            std::vector<layout_block_t> *outer_blocks = nullptr) const {
        if (outer_blocks) outer_blocks->resize(0);

        if (grid.elems() == 1) return tile_coord_t(tile);

        dim_t total_elems = elems();
        dim_t tile_elems = tile.elems();

        grid_splitter_t grid_splitter(grid);
        gpu_assert(tile_elems * grid.elems() == total_elems)
                << "Tile/grid dimensions do not match.";
        MAYBE_UNUSED(total_elems);
        MAYBE_UNUSED(tile_elems);

        tile_t dims;
        coord_t start;
        auto rem_tile = tile;
        for (auto &eb : enumerated_blocks()) {
            auto &b = eb.second;
            if (b.block == 1) continue;

            dim_t &e = rem_tile[b.dim];
            if (e > 1) {
                if (e % b.block == 0) {
                    e /= b.block;
                } else if (b.block % e == 0) {
                    auto tmp_layout = split_block(eb, e, b.block / e);
                    return tmp_layout.split(tile, grid, outer_blocks);
                } else {
                    return tile_coord_t::invalid();
                }
            } else {
                dim_t next_chunk
                        = math::gcd(b.block, grid_splitter.cur_block());
                if (b.block == next_chunk) {
                    auto idx = grid_splitter.pop_block(next_chunk);
                    start[b.dim] += idx * dims[b.dim];
                    if (outer_blocks) outer_blocks->push_back(b);
                } else if (b.block % next_chunk == 0 && next_chunk != 1) {
                    auto tmp_layout
                            = split_block(eb, next_chunk, b.block / next_chunk);
                    return tmp_layout.split(tile, grid, outer_blocks);
                } else {
                    return tile_coord_t::invalid();
                }
            }
            dims[b.dim] *= b.block;
        }
        return tile_coord_t(tile, start);
    }

    // Iterates through tiles of the layout, calling `f` with relative offsets
    // for each tile. The iteration order is defined by the layout blocks -
    // absolute 1D offsets are increasing between callback calls.
    template <typename F>
    void for_each_tile(const tile_t &tile, const F &f) const {
        for (auto &d : tile) {
            gpu_assert(dim(d) % tile[d] == 0);
        }

        int nblocks = int(blocks().size());
        std::vector<dim_t> sub_blocks(nblocks);
        for (int i = 0; i < nblocks; i++)
            sub_blocks[i] = blocks()[i].block;

        for (auto &d : tile) {
            dim_t dim = tile[d];
            for (auto &eb : enumerated_blocks()) {
                auto &b = eb.second;
                if (b.dim != d) continue;
                int block_idx = eb.first;
                if (b.block >= dim) {
                    gpu_assert(b.block % dim == 0);
                    sub_blocks[block_idx] = b.block / dim;
                    break;
                }
                sub_blocks[block_idx] = 1;
                gpu_assert(dim % b.block == 0);
                dim /= b.block;
            }
        }

        int ntiles = int(elems() / tile.elems());

        std::vector<dim_t> sub_block_idxs(nblocks);
        for (int i = 0; i < ntiles; i++) {
            // Convert sub-block indices to dimension indices.
            tile_t dims;
            icoord_t start;
            for (int j = 0; j < nblocks; j++) {
                auto &b = blocks()[j];
                dim_t k = sub_block_idxs[j]
                        * (blocks()[j].block / sub_blocks[j]);
                start[b.dim] += dims[b.dim] * k;
                dims[b.dim] *= b.block;
            }

            // Pass dimension offsets to the callback.
            f(start);

            // Move to the next vector of indices.
            for (int j = 0; j < nblocks; j++) {
                auto &idx = sub_block_idxs[j];
                if (idx + 1 < sub_blocks[j]) {
                    idx++;
                    break;
                }
                idx = 0;
            }
        }
    }

    bool has_outer_block(dim_t block, const pvar_t &dim = {}) const {
        if (block == 1) return true;
        if (blocks().empty()) return false;
        auto &b = blocks().back();
        if (!dim.is_undef() && b.dim != dim) return false;
        if (b.block % block != 0) return false;
        return true;
    }

    stride_t inner_stride() const {
        if (nblocks() == 0) return stride_t(1);
        return blocks()[0].stride;
    }

    // eb is <block index, block> pair, see enumerated_blocks().
    static bool is_outermost(const std::pair<int, layout_block_t> &eb,
            const std::vector<layout_block_t> &blocks) {
        auto &dim = eb.second.dim;
        for (int i = 0; i < int(blocks.size()); i++) {
            if (blocks[i].dim == dim && i > eb.first) return false;
        }
        return true;
    }

    // Assume that layouts are normalized.
    static void align_layouts(layout_t &a, layout_t &b);

    // Reinterprets layouts to wider data type (up to 4 bytes).
    // Example: 16a16b (s8 type) -> 16a4b (s32 type)
    static bool try_reinterpret_to_wider_type(layout_t &src, layout_t &dst,
            const tile_t &tile = {}, bool do_update = true,
            int *new_size_out = nullptr) {
        if (src.blocks().empty() || dst.blocks().empty()) return false;
        if (src.type() != dst.type()) return false;

        auto &s0 = src.blocks()[0];
        auto &d0 = dst.blocks()[0];
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
            if (tile[l.blocks()[0].dim] % factor != 0) return false;
            return true;
        };

        auto strides_ok = [&](const layout_t &l) {
            for (int i = 1; i < int(l.blocks().size()); i++) {
                auto &b = l.blocks()[i];
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
                    src = src.reinterpret(type_t::s(new_size * 8));
                    dst = dst.reinterpret(type_t::s(new_size * 8));
                }
                if (new_size_out) *new_size_out = new_size;
                return true;
            }
            new_size /= 2;
        }
        return false;
    }

private:
    // Returns vector of <dimension index, block size> pairs.
    static std::vector<std::pair<pvar_t, dim_t>> parse_format(
            const std::string &format, int ndims_hint);

    // Returns vector of <dimension letter, block size> pairs.
    static std::vector<std::pair<char, dim_t>> parse_letter_blocks(
            const std::string &format);

    void sanity_check() const;

    // Data type of the layout.
    type_t type_;

    // Number of dimensions.
    dim_idx_t ndims_ = dim_idx::invalid;

    // Offset to the start of the layout (in elements of type).
    expr_t offset_;

    // Blocks ordered from innermost to outermost.
    std::vector<layout_block_t> blocks_;
};

// Helper class to incrementally increase a sub-layout of the given layout.
// One step - adding the minimal factor of the next remaining block. Used
// to find the minimal tile between two layouts that is innermost for both
// layouts.
class layout_iterator_t {
public:
    layout_iterator_t(const layout_t &l) : l_(l), block_idx_(-1), block_(1) {}

    bool has_next() const {
        dim_t b = block_;
        int b_idx = block_idx_;
        while (b == 1) {
            b_idx++;
            if (b_idx >= int(l_.blocks().size())) return false;
            b = int(l_.blocks()[b_idx].block);
        }
        return true;
    }

    layout_iterator_t &operator++() {
        gpu_assert(has_next());
        while (block_ == 1) {
            block_idx_++;
            block_ = int(l_.blocks()[block_idx_].block);
        }
        // Find smallest factor.
        for (int factor = 2; factor <= int(block_); factor++) {
            if (block_ % factor == 0) {
                block_ /= factor;
                return *this;
            }
        }

        gpu_error_not_expected();
        return *this;
    }

    tile_t tile() const {
        tile_t ret;
        for (int i = 0; i <= block_idx_; i++) {
            auto &b = l_.blocks()[i];
            dim_t b_block = b.block;
            if (i == block_idx_) b_block /= block_;
            ret[b.dim] *= b_block;
        }
        return ret;
    }

    int nblocks() const { return block_idx_ + 1; }

    layout_t outer_layout() const {
        auto &blocks = l_.blocks();
        std::vector<layout_block_t> outer_blocks;
        if (block_ > 1) {
            auto &b = blocks[block_idx_];
            outer_blocks.push_back(b);
            outer_blocks[0].block = block_;
            outer_blocks[0].stride = b.stride * (b.block / block_);
        }
        outer_blocks.insert(outer_blocks.end(),
                blocks.begin() + (block_idx_ + 1), blocks.end());
        return layout_t(l_.type(), l_.ndims(), l_.offset(), outer_blocks);
    }

private:
    const layout_t &l_;

    int block_idx_;
    dim_t block_;
};

class mask_tensor_t {
public:
    mask_tensor_t() = default;

    mask_tensor_t(const layout_t &layout)
        : layout_(layout), masks_(layout.elems(), -1) {
        gpu_assert(layout.is_dense());
    }

    mask_tensor_t(const layout_t &layout, const std::vector<int> &masks,
            const object_eq_map_t<expr_t, int> &mask2ids,
            const std::vector<expr_t> &id2masks)
        : layout_(layout)
        , masks_(masks)
        , mask2ids_(mask2ids)
        , id2masks_(id2masks) {
        gpu_assert(int(masks.size()) == elems()) << "Incompatible size.";
    }

    const type_t &type() const { return layout_.type(); }

    const layout_t &layout() const { return layout_; }

    dim_t elems() const { return layout_.elems(); }

    void set_mask(dim_t off, const expr_t &mask) {
        gpu_assert(0 <= off && off < elems()) << "Incorrect offset.";
        if (mask.is_empty()) return;

        auto ret = mask2ids_.insert({mask, int(mask2ids_.size())});
        int id = ret.first->second;
        masks_[off] = id;

        if (ret.second) id2masks_.push_back(mask);
    }

    const expr_t &mask(dim_t off) const {
        gpu_assert(0 <= off && off < elems());
        return id2masks_[masks_[off]];
    }

    void simplify(const constraint_set_t &cset) {
        for (auto &mask : id2masks_) {
            auto new_mask = jit::simplify(mask, cset);
            // Some complex expressions need more than one simplify() call.
            int max_tries = 5;
            for (int i = 0; i < max_tries; i++) {
                mask = new_mask;
                new_mask = jit::simplify(new_mask, cset);
                if (new_mask.is_equal(mask)) break;
            }
        }
        mask2ids_.clear();
        for (int i = 0; i < int(id2masks_.size()); i++) {
            auto ret = mask2ids_.insert({id2masks_[i], i});
            if (!ret.second) {
                for (auto &m : masks_)
                    if (m == i) m = ret.first->second;
            }
        }
    }

    mask_tensor_t sub(const tile_t &tile, const coord_t &start) const {
        coord_t tile_start(start);
        auto sub_layout = layout_.sub(tile);
        mask_tensor_t sub_mask(sub_layout);
        for_each(tile, [&](const icoord_t &sub_start) {
            dim_t sub_off = sub_layout.offset<dim_t>(sub_start);
            dim_t off = layout_.offset<dim_t>(tile_start)
                    + layout_.offset<dim_t>(sub_start);
            sub_mask.set_mask(sub_off, mask(off));
        });
        return sub_mask;
    }

    mask_tensor_t reinterpret(const type_t &new_type) const {
        gpu_assert(!is_empty()) << "Can't reinterpret.";
        dim_t bytes = elems() * type().size();
        if (bytes % new_type.size() != 0 && bytes > new_type.size())
            return mask_tensor_t();
        int new_mask_size = std::max((int)(bytes / new_type.size()), 1);
        std::vector<int> new_masks(new_mask_size);
        for (dim_t i = 0; i < bytes; i += new_type.size()) {
            int mask_id = std::numeric_limits<int>::max();
            for (int j = 0; j < new_type.size() && j < bytes; j++) {
                int cur_mask_id = masks_[(i + j) / type().size()];
                if (mask_id >= int(masks_.size())) {
                    mask_id = cur_mask_id;
                } else if (mask_id != cur_mask_id) {
                    // Mask is not consistent, can't reinterpret.
                    return mask_tensor_t();
                }
            }
            gpu_assert(0 <= mask_id && mask_id < int(masks_.size()));
            new_masks[i / new_type.size()] = mask_id;
        }
        dim_t new_elems = utils::div_up(bytes, new_type.size());
        layout_t _1d_layout(new_type, 0, std::vector<dim_t> {new_elems});
        return mask_tensor_t(_1d_layout, new_masks, mask2ids_, id2masks_);
    }

    expr_t to_expr(dim_t nmasks) const {
        if (elems() % nmasks != 0) return expr_t();

        std::vector<expr_t> vec(nmasks);
        for (int i = 0; i < elems(); i++) {
            auto &channel_mask = vec[i % nmasks];
            auto &cur_mask = id2masks_[masks_[i]];
            if (channel_mask.is_empty()) {
                channel_mask = cur_mask;
                continue;
            }
            if (!channel_mask.is_equal(cur_mask)) return expr_t();
        }
        auto e = shuffle_t::make(vec);
        e = jit::simplify(e);
        e = jit::simplify_propagate_shuffle(e);
        return e;
    }

    bool is_empty() const { return layout_.is_empty(); }

    std::string str() const {
        ostringstream_t oss;
        for (int i = 0; i < int(elems()); i++) {
            if (i != 0) oss << std::endl;
            oss << "mask #" << i << ": ";
            if (masks_[i] == -1) {
                oss << "(nil)";
            } else {
                oss << id2masks_[masks_[i]];
            }
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    layout_t layout_;
    std::vector<int> masks_;

    object_eq_map_t<expr_t, int> mask2ids_;
    std::vector<expr_t> id2masks_;
};

class tdim_t {
public:
    tdim_t() = default;

    tdim_t(const expr_t &expr, const expr_t &mask) : expr_(expr), mask_(mask) {}

    dim_idx_t nvargs() const { return nvargs_; }

    const expr_t &expr() const { return expr_; }

    const expr_t &mask() const { return mask_; }

    void set_mask(const expr_t &value) { mask_ = value; }

    expr_t mask(const expr_t &tvalue, const std::vector<expr_t> &vvars,
            const coord_t &vvalues) const {
        auto ret = substitute(mask_, placeholder_var(), tvalue);
        for (dim_idx_t i = 0; i < vvars.size(); i++) {
            if (contains_object(ret, vvars[i])) {
                ret = substitute(ret, vvars[i], vvalues[i]);
            }
        }
        return ret;
    }

    dim_idx_t vidx(dim_idx_t arg_idx) const {
        gpu_assert(arg_idx < nvargs());
        return vidxs_[arg_idx];
    }

    stride_t vstride(dim_idx_t arg_idx) const {
        gpu_assert(arg_idx < nvargs());
        return vstrides_[arg_idx];
    }

    bool is_empty() const { return expr_.is_empty(); }

    bool is_identity() const { return is_var(expr_); }

    bool is_fixed_stride(dim_idx_t arg_idx) const {
        gpu_assert(arg_idx < nvargs());
        return vstrides_[arg_idx].is_fixed();
    }

    void add_vvar(dim_idx_t vidx, const expr_t &varg) {
        gpu_assert(nvargs_ + 1 <= max_nvargs);
        vidxs_[nvargs_] = vidx;
        vstrides_[nvargs_] = compute_stride(expr_, nvargs_, varg);
        nvargs_++;
    }

    static const expr_t &placeholder_var() {
        static thread_local expr_t ph_var = var_t::make(type_t::s32(), "_ph");
        return ph_var;
    }

    std::string str() const {
        ostringstream_t oss;
        oss << expr_;
        if (mask_) oss << " mask: " << mask_;
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    static const dim_idx_t max_nvargs = 2;

    static stride_t compute_stride(
            const expr_t &e, dim_idx_t idx, const expr_t &var);

    expr_t expr_;

    dim_idx_t nvargs_ = 0;
    std::array<stride_t, max_nvargs> vstrides_;
    std::array<dim_idx_t, max_nvargs> vidxs_;
    expr_t mask_;
};

class view_t {
public:
    view_t() = default;

    view_t(const std::vector<expr_t> &vvars, dim_idx_t ntdims)
        : vvars_(vvars)
        , vdims_(vvars.size())
        , vstart_(vvars.size())
        , tdims_(ntdims) {}

    // Constructs view from a layout.
    explicit view_t(const layout_t &layout,
            const std::vector<expr_t> &_vvars = {},
            uint32_t bound_check_mask = 0)
        : view_t(layout, _vvars, layout.dims(), bound_check_mask) {}

    view_t(const layout_t &layout, const std::vector<expr_t> &_vvars,
            const std::vector<dim_t> &_vdims, uint32_t bound_check_mask)
        : vvars_(_vvars)
        , vdims_(_vdims)
        , vstart_(layout.ndims())
        , tdims_(layout.ndims())
        , tlayout_(layout) {
        if (vvars_.empty()) vvars_ = create_vvars(layout.ndims());
        for (dim_idx_t i = 0; i < nvdims(); i++) {
            expr_t i_mask;
            if ((bound_check_mask & (1 << i)) != 0)
                i_mask = (placeholder_var() < layout.dim(i));
            set_tdim(i, vvars_[i], i_mask);
        }
    }

    const std::vector<expr_t> &vvars() const { return vvars_; }

    const tile_t &vdims() const { return vdims_; }

    const coord_t &vstart() const { return vstart_; }

    tile_coord_t vtile_coord() const { return tile_coord_t(vdims_, vstart_); }

    const layout_t &tlayout() const { return tlayout_; }

    dim_idx_t nvdims() const { return into<dim_idx_t>(vdims_.size()); }

    dim_idx_t ntdims() const { return into<dim_idx_t>(tdims_.size()); }

    dim_t velems() const {
        dim_t ret = 1;
        for (dim_idx_t i = 0; i < nvdims(); i++)
            ret *= vdims_[i];
        return ret;
    }

    const expr_t &vvar(dim_idx_t idx) const {
        gpu_assert(idx < nvdims());
        return vvars_[idx];
    }

    const expr_t &vvar(const std::string &name) const {
        for (auto &v : vvars_)
            if (v.as<var_t>().name == name) return v;
        gpu_error_not_expected() << name;
        return vvars_[0];
    }

    const tdim_t &tdim(dim_idx_t idx) const {
        gpu_assert(idx < ntdims());
        return tdims_[idx];
    }

    void set_tdim(
            dim_idx_t tidx, const expr_t &_texpr, const expr_t &mask = {}) {
        gpu_assert(tdims_[tidx].is_empty());

        auto texpr = simplify(_texpr);

        tdim_t tdim(texpr, mask);
        for (dim_idx_t i = 0; i < nvdims(); i++) {
            if (contains_object(texpr, vvars_[i])) tdim.add_vvar(i, vvars_[i]);
        }
        if (!is_const(texpr)) {
            gpu_assert(tdim.nvargs() > 0)
                    << "Tensor dimension must have at least one view dimension "
                       "that maps to it.";
        }
        tdims_[tidx] = std::move(tdim);
    }

    void set_vdim(const expr_t &varg, dim_t vdim,
            const expr_t &vstart = expr_t(0), bool overwrite = false) {
        dim_idx_t vidx = vvar_index(varg);
        if (!overwrite) gpu_assert(is_zero(vstart_[vidx]));
        vstart_[vidx] = vstart;
        vdims_[vidx] = vdim;
    }

    void set_tlayout(const layout_t &tlayout) { tlayout_ = tlayout; }

    void set_tmasks(const std::unordered_map<std::string, dim_t> &padded_dims) {
        using namespace ir_utils;
        auto &x = placeholder_var();
        for (dim_idx_t i = 0; i < ntdims(); i++) {
            auto &tdim = tdims_[i];
            if (!tdim.is_identity() || tdim.mask()) continue;
            dim_idx_t vidx = tdim.vidx(0);
            dim_t dim = tlayout_.dim(i);
            auto &dim_name = vvars_[vidx].as<var_t>().name;
            dim_t padded_dim = get_or_default(padded_dims, dim_name, dim_t(1));
            if (dim >= padded_dim) continue;
            dim_t inner_blk = ir_utils::max_pow2_divisor(dim);
            dim_t dim_blk = ir_utils::max_pow2_divisor(tlayout_.inner_block(
                    i, /*skip_outer=*/true, /*inner_only=*/false));
            inner_blk = std::min(inner_blk, dim_blk);
            auto tmask = (inner_blk == 1) ? (x < dim)
                                          : (x / inner_blk < dim / inner_blk);
            tdim.set_mask(tmask);
        }
    }

    void set_tmasks(const std::vector<dim_t> &padded_dims) {
        gpu_assert(padded_dims.size() == ntdims());
        std::unordered_map<std::string, dim_t> pd_map;
        for (dim_idx_t i = 0; i < ntdims(); i++) {
            auto &dim_name = vvars_[tdims_[i].vidx(0)].as<var_t>().name;
            pd_map.emplace(dim_name, padded_dims[i]);
        }
        set_tmasks(pd_map);
    }

    std::string str() const {
        using ir_utils::operator<<;

        if (is_empty()) return "(nil)";
        ostringstream_t oss;
        oss << vdims_.str();
        if (!has_zero_vstart()) oss << " vstart: [" << vstart_ << "]";
        oss << " tlayout: " << tlayout_;
        return oss.str();
    }

    IR_DEFINE_DUMP()

    bool is_empty() const { return vdims_.is_empty(); }

    bool has_zero_vstart() const {
        for (dim_idx_t i = 0; i < nvdims(); i++)
            if (!is_zero(vstart_[i])) return false;
        return true;
    }

    bool has_tmask(dim_idx_t tidx) const {
        gpu_assert(tidx != dim_idx::invalid && tidx < ntdims());
        return bool(tdims_[tidx].mask());
    }

    const type_t &type() const { return tlayout_.type(); }

    expr_t offset(const coord_t &vargs = {}, bool ignore_offset = false) const {
        auto targs = cvt_vargs_to_targs(vargs);
        return tlayout_.offset(targs, ignore_offset);
    }

    expr_t offset_in_bytes(
            const coord_t &vargs = {}, bool ignore_offset = false) const {
        return offset(vargs, ignore_offset) * type().size() / type().packing();
    }

    int get_alignment(const constraint_set_t &cset) const {
        // Alignment must be a power of 2.
        const dim_t base_alignment = 128;
        int64_t f = get_max_const_factor(this->offset_in_bytes(), cset);
        dim_t alignment = f ? ir_utils::max_pow2_divisor(f) : base_alignment;
        return static_cast<int>(std::min(base_alignment, alignment));
    }

    dim_idx_t vvar_index(const expr_t &vvar) const {
        for (dim_idx_t i = 0; i < vvars_.size(); i++)
            if (vvar.is_same(vvars_[i])) return i;
        gpu_error_not_expected() << "Can't find view dimension.";
        return dim_idx::invalid;
    }

    view_t create_sub_view(const tile_t &tile, const coord_t &coord) const;

    view_t create_sub_view(const tile_coord_t &tile_coord) const {
        return create_sub_view(tile_coord.tile, tile_coord.coord);
    }

    view_t retype(const type_t &new_type) const {
        auto ret = *this;
        ret.tlayout_ = tlayout_.retype(new_type);
        return ret;
    }

    view_t make_dense() const {
        auto ret = *this;
        ret.tlayout_ = tlayout_.make_dense();
        return ret;
    }

    bool is_masked_vdim(dim_idx_t vidx) const {
        gpu_assert(vidx != dim_idx::invalid && vidx < nvdims());
        gpu_assert(has_zero_vstart())
                << "Can't be reliably determined if the view is a sub-view.";
        for (dim_idx_t i = 0; i < ntdims(); i++) {
            auto &tdim = tdims_[i];
            if (tdim.expr().is_equal(vvars_[vidx])) {
                if (vdims_[vidx] != tlayout_.dim(i)) return true;
            }
            if (has_tmask(i)) {
                for (dim_idx_t j = 0; j < tdim.nvargs(); j++) {
                    if (tdim.vidx(j) == vidx) return true;
                }
            }
        }
        return false;
    }

    // Returns the mask corresponding to `vargs` view indices. The mask is
    // based on:
    // 1) combined tensor masks for the given indices
    // 2) Bounds-based masks for those view dimensions that are used directly
    //    in the tensor
    //    - Example: 32a layout when 'a' dimension is A < 32. In general it's
    //      fine to load/store elements with indices in the range [A, 31]
    //      assuming the zero padding invariant. However in some cases we need
    //      to generate the exact bound condition based on the logical indices.
    expr_t vmask(const coord_t &vargs) const {
        gpu_assert(vargs.size() == nvdims()) << "Incompatible dimensions.";
        gpu_assert(has_zero_vstart())
                << "Can't be reliably determined if the view is a sub-view.";
        auto targs = cvt_vargs_to_targs(vargs);
        auto mask = bool_imm_t::make(true);
        for (dim_idx_t i = 0; i < ntdims(); i++) {
            for (dim_idx_t j = 0; j < nvdims(); j++) {
                if (!tdims_[i].expr().is_equal(vvars_[j])) continue;
                if (vdims_[j] != tlayout_.dim(i)) {
                    mask &= (vargs[j] < vdims_[j]);
                }
            }
            if (has_tmask(i)) {
                auto &tdim = tdims_[i];
                mask &= tdim.mask(targs[i], vvars_, vargs);
            }
        }
        return mask;
    }

    bool can_convert_to_vlayout() const {
        if (nvdims() != ntdims()) return false;
        for (dim_idx_t i = 0; i < nvdims(); i++) {
            if (!tdims_[i].expr().is_same(vvars_[i])) return false;
            if (!tdims_[i].is_fixed_stride(0)) return false;
        }
        return true;
    }

    // Returns the view-based layout constructed based on the view mapping from
    // the tensor layout to the view dimensions. In general such a layout may
    // include "unknown" strides which means it can't be used for offset
    // calculation.
    // However in many cases it's possible to construct a valid "view" layout
    // fully representative of the view and the underlying tensor layout.
    // Mainly it depends on whether each view dimension is a linear combination
    // of tensor layout dimensions.
    // If 1) init_offset is true and 2) the returned layout doesn't contain
    // "unknown" strides then it can be directly used for offset calculation.
    layout_t create_pseudo_vlayout(bool init_offset = false) const {
        return create_pseudo_vlayout(normalized_tlayout(), init_offset);
    }

    layout_t normalized_tlayout() const {
        auto blocks = move_size_1_blocks_outer();
        blocks = normalize_blocks(blocks, false);
        auto layout = layout_t(
                type(), tlayout_.ndims(), tlayout_.offset(), blocks, false);
        return layout;
    }

    layout_t create_dense_vlayout() const {
        return create_pseudo_vlayout().make_dense();
    }

    layout_t create_vlayout(bool force_zero_offset = false) const {
        gpu_assert(can_convert_to_vlayout()) << "Can't convert view to layout.";
        if (force_zero_offset) return tlayout_.sub(vdims_);
        return tlayout_.sub(vdims_, vstart_);
    }

    dim_t vlayout_size() const { return create_vlayout().size(); }

    bool has_same_vlayout(
            const view_t &other, bool compare_offset = true) const {
        return create_vlayout().is_equal(
                other.create_vlayout(), compare_offset);
    }

    view_t split(const grid_info_t &grid, tile_coord_t &vtile_coord,
            grid_info_t *out_grid = nullptr) const {
        auto vlayout = create_pseudo_vlayout();
        vtile_coord = vlayout.split(grid, out_grid);
        return create_sub_view(vtile_coord.tile, vtile_coord.coord);
    }

    view_t split(
            const grid_info_t &grid, grid_info_t *out_grid = nullptr) const {
        tile_coord_t vtile_coord;
        return split(grid, vtile_coord, out_grid);
    }

    // Returns a tensor corresponding to the biggest innermost sub-layout so that
    // 1) It consists of consecutive blocks only.
    // 2) It contains less or equal than max_tile_elems elements.
    // 3) It is dense if is_dense_tile is true.
    tile_t split_into_max_tile(dim_t max_tile_elems, bool is_dense_tile) const {
        auto vlayout = create_pseudo_vlayout();
        return vlayout.split_into_max_tile(max_tile_elems, is_dense_tile);
    }

    template <typename F>
    void for_each_tile(const tile_t &tile, const F &f) const {
        auto vlayout = create_dense_vlayout();
        vlayout.for_each_tile(tile, f);
    }

    view_t substitute(const expr_t &from, const expr_t &to) const;

    mask_tensor_t create_mask_tensor(
            const constraint_set_t &cset, uint32_t tmask = 0xFFFFFFFF) const {
        auto _vlayout = create_dense_vlayout();
        mask_tensor_t mask_tensor(_vlayout);
        icoord_t vargs(nvdims());
        create_mask_tensor(mask_tensor, _vlayout, 0, vargs, tmask);
        mask_tensor.simplify(cset);
        return mask_tensor;
    }

    void try_create_buffer_view(view_t &buf_view, view_t &inv_view) const {
        buf_view = view_t(create_vvars(ntdims()), ntdims());
        inv_view = view_t(vvars(), ntdims());
        for (dim_idx_t i = 0; i < nvdims(); i++) {
            inv_view.set_vdim(vvars()[i], vdims()[i]);
        }
        for (dim_idx_t i = 0; i < ntdims(); i++) {
            auto &tdim = tdims_[i];
            auto &buf_vvar = buf_view.vvars()[i];
            if (tdim.is_identity()) {
                dim_idx_t vidx = tdim.vidx(0);
                buf_view.set_vdim(buf_vvar, vdims()[vidx], vstart()[vidx]);
                buf_view.set_tdim(i, buf_vvar, tdim.mask());
                inv_view.set_tdim(i, tdim.expr());
                continue;
            }
            int64_t buf_vdim = 0;
            bool ok = true;
            for (dim_idx_t j = 0; j < tdim.nvargs(); j++) {
                dim_idx_t vidx = tdim.vidx(j);
                auto &vvar = vvars()[vidx];
                dim_t vdim = vdims()[vidx];
                if (vdim == 1) continue;
                const auto &A = tdim.expr();
                auto B = jit::substitute(A, vvar, vvar + 1);
                auto C = simplify(B - A);
                if (!is_const(C)) {
                    ok = false;
                    break;
                }
                buf_vdim += to_cpp<int64_t>(C) * (vdim - 1);
            }
            buf_vdim++;

            if (!ok) {
                buf_view = view_t();
                inv_view = view_t();
                return;
            }

            auto buf_vstart = tdim.expr();
            auto inv_vstart = tdim.expr();
            for (dim_idx_t j = 0; j < tdim.nvargs(); j++) {
                dim_idx_t vidx = tdim.vidx(j);
                buf_vstart = jit::substitute(
                        buf_vstart, vvars()[vidx], vstart()[vidx]);
                inv_vstart
                        = jit::substitute(inv_vstart, vvars()[vidx], expr_t(0));
            }
            buf_vstart = simplify(buf_vstart);
            inv_vstart = simplify(inv_vstart);

            if (!is_const(inv_vstart)) {
                buf_view = view_t();
                inv_view = view_t();
                return;
            }

            buf_view.set_vdim(buf_vvar, buf_vdim, buf_vstart);

            // Check that mask doesn't contain vvars - they can't be accessed
            // in the buffered view.
            auto &tmask = tdim.mask();
            for (auto &vvar : vvars()) {
                if (contains_object(tmask, vvar)) {
                    buf_view = view_t();
                    inv_view = view_t();
                    return;
                }
            }

            buf_view.set_tdim(i, buf_vvar, tmask);
            inv_view.set_tdim(i, tdim.expr() - inv_vstart);
        }
        buf_view.set_tlayout(tlayout_);
    }

    static const expr_t &placeholder_var() { return tdim_t::placeholder_var(); }

    static std::vector<expr_t> create_vvars(dim_idx_t nvdims);

    coord_t cvt_vargs_to_targs(
            coord_t vcoord = {}, bool ignore_vstart = false) const {
        if (vcoord.is_empty()) vcoord = coord_t(nvdims());

        if (!ignore_vstart) {
            for (dim_idx_t i = 0; i < nvdims(); i++) {
                if (!is_zero(vstart_[i])) vcoord[i] += vstart_[i];
            }
        }

        coord_t tcoord(ntdims());
        for (dim_idx_t i = 0; i < ntdims(); i++) {
            tcoord[i] = tdims_[i].expr();
            for (dim_idx_t j = 0; j < nvdims(); j++) {
                tcoord[i] = jit::substitute(tcoord[i], vvars_[j], vcoord[j]);
            }
        }
        for (dim_idx_t i = 0; i < ntdims(); i++) {
            tcoord[i] = const_fold(tcoord[i]);
        }
        return tcoord;
    }

private:
    layout_t create_pseudo_vlayout(
            const layout_t &tlayout, bool init_offset) const;

    void create_mask_tensor(mask_tensor_t &mask_tensor,
            const layout_t &_vlayout, dim_idx_t vidx, icoord_t &vargs,
            uint32_t tmask) const {
        if (vidx == _vlayout.ndims()) {
            bool is_init = false;
            coord_t vvalues;
            coord_t targs;
            expr_t mask = bool_imm_t::make(true);
            for (dim_idx_t i = 0; i < ntdims(); i++) {
                auto &tdim = tdims_[i];
                if ((tmask & (1 << i)) == 0) continue;
                if (tdim.mask().is_empty()) continue;
                if (!is_init) {
                    // Lazily initialize values
                    vvalues = vstart_.values();
                    for (dim_idx_t i = 0; i < nvdims(); i++)
                        vvalues[i] += vargs[i];
                    targs = cvt_vargs_to_targs(vargs);
                    is_init = true;
                }
                mask &= tdim.mask(targs[i], vvars_, vvalues);
            }
            mask_tensor.set_mask(_vlayout.offset<dim_t>(vargs), mask);
            return;
        }

        for (dim_idx_t i = 0; i < vdims()[vidx]; i++) {
            vargs[vidx] = i;
            create_mask_tensor(mask_tensor, _vlayout, vidx + 1, vargs, tmask);
        }
    }

    std::vector<layout_block_t> move_size_1_blocks_outer() const {
        std::vector<layout_block_t> new_blocks;
        std::vector<layout_block_t> size_1_blocks;
        for (auto &b : tlayout_.blocks()) {
            if (b.block == 1 && vdims_.get(b.dim) == 1) {
                size_1_blocks.emplace_back(b);
            } else {
                new_blocks.emplace_back(b);
            }
        }
        stride_t stride = new_blocks.empty()
                ? stride_t(1)
                : new_blocks.back().block * new_blocks.back().stride;
        for (auto &b : size_1_blocks) {
            b.stride = stride;
            new_blocks.emplace_back(b);
        }
        return new_blocks;
    }

    std::vector<expr_t> vvars_;
    tile_t vdims_;
    coord_t vstart_;

    std::vector<tdim_t> tdims_;
    layout_t tlayout_;
};

class dim_assignment_t {
public:
    dim_assignment_t() = default;

    dim_assignment_t(dim_idx_t old_ndims, dim_idx_t new_ndims)
        : old_ndims_(old_ndims)
        , new_ndims_(new_ndims)
        , assignments_(old_ndims, -1) {}

    void assign(dim_idx_t old_idx, dim_idx_t new_idx) {
        gpu_assert(old_idx != dim_idx::invalid && old_idx < old_ndims_);
        gpu_assert(new_idx != dim_idx::invalid && new_idx < new_ndims_);
        assignments_[old_idx] = new_idx;
    }

    void assign(const std::vector<dim_idx_t> &old_idxes, dim_idx_t new_idx) {
        for (auto old_idx : old_idxes) {
            assign(old_idx, new_idx);
        }
    }

    dim_idx_t operator[](dim_idx_t old_idx) const {
        gpu_assert(old_idx >= 0 && old_idx < old_ndims());
        return assignments_[old_idx];
    }

    dim_idx_t old_ndims() const { return old_ndims_; }

    dim_idx_t new_ndims() const { return new_ndims_; }

    bool is_empty() const { return old_ndims_ == 0 && new_ndims_ == 0; }

    layout_t map(const layout_t &layout) const;

private:
    dim_idx_t old_ndims_ = 0;
    dim_idx_t new_ndims_ = 0;

    // assignments_[old_idx] = new_idx.
    std::vector<dim_idx_t> assignments_;
};

// Adds size one spatial dimensions according to input parameters. Spatial
// dimensions are assumed to be the last dimensions.
layout_t spatials_to_3d(const layout_t &layout, bool with_groups,
        const std::array<int, 3> &dhw_map);

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
