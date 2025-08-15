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

#include "gpu/intel/jit/codegen/reorder.hpp"

#include "gpu/intel/logging.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

copy_operand_t::copy_operand_t(const reg_buf_data_t &rbd) : CopyOperand(rbd) {
    if (!rbd.is_empty()) {
        const auto &rd = rbd.reg_buf();
        if (!rd.with_permute() && rd.blocks() <= 1) return;
        const auto base = rbd.base();
        block_size = rd.with_permute() ? 1 : rd.block_regs();
        const auto blocks = rd.regs() / block_size;
        block_bases.reserve(blocks);
        // TODO: join contiguous registers into blocks
        for (int i = 0, j = 0; i < blocks; ++i, j += block_size) {
            auto block_base = rd.base(j);
            block_bases.push_back(block_base);
            if (block_base <= base && base < block_base + block_size)
                block_off = i;
        }
        advance(rbd.hw(), 0, 1);
    }
}

copy_operand_t &copy_operand_t::advance(
        ngen::HW hw, dim_t elems, uint8_t stride) {
    const auto nblocks = into<int>(block_bases.size());
    const auto grf_bits = ngen::GRF::bytes(hw) << 3;
    const auto type_bit_size = ngen::getBits(type);
    const auto bit_off = (offset + elems * stride) * type_bit_size;
    const auto grf_shift = bit_off / grf_bits;
    if (temp || block_bases.empty())
        grf += grf_shift;
    else {
        const auto orig_block_base = block_bases[block_off];
        const auto grf_off = grf - orig_block_base + grf_shift;
        const auto block_shift = grf_off / block_size;
        block_off += block_shift;
        if (block_off >= nblocks)
            // If we advance past the end of the buffer, continue linearly from
            // (max_base + block_size).
            grf = [&]() {
                const auto past_end = block_size * (block_off - nblocks + 1);
                int max_base = grf;
                for (const auto &base : block_bases)
                    max_base = std::max(max_base, base);
                return (int16_t)(max_base + past_end + (grf_off % block_size));
            }();
        else
            grf = (int16_t)(block_bases[block_off] + (grf_off % block_size));
    }
    offset = (uint8_t)((bit_off % grf_bits) / type_bit_size);
    return *this;
}

void copy_plan_t::mov(int simd, ngen::InstructionModifier mod,
        const copy_operand_t &dst, const copy_operand_t &src) {
    static constexpr ngen::Opcode mov = ngen::Opcode::mov;
    const auto grf_bits = ngen::GRF::bytes(hw()) << 3;

    auto max_simd = [&](const copy_operand_t &op) {
        const auto &block_size = op.block_size;
        const auto &block_bases = op.block_bases;
        // Count contiguous registers
        int regs = block_size - (op.grf - block_bases[op.block_off]);
        for (size_t j = op.block_off; j < block_bases.size() - 1; ++j) {
            if (block_bases[j] + block_size != block_bases[j + 1]) break;
            regs += block_size;
        }
        int type_bits = ngen::getBits(op.type);
        int rem_bits = regs * grf_bits - type_bits * (1 + op.offset);
        return rem_bits / (type_bits * op.stride) + 1;
    };

    auto block_src = src, block_dst = dst;
    while (simd > 0) {
        auto block_simd = simd;
        for (auto &op : {block_dst, block_src}) {
            if (op.block_bases.empty()) continue;
            block_simd = std::min(block_simd, max_simd(op));
        }

        append(phase, mov, block_simd, mod, block_dst, block_src);
        simd -= block_simd;
        // Protection from advancing past the end of the buffer, which may
        // result in OOB accesses when registers are permuted.
        if (simd <= 0) break;
        block_dst.advance(hw(), block_simd, dst.stride);
        block_src.advance(hw(), block_simd, src.stride);
    }
}

reorder_2d_impl_t::reorder_2d_impl_t(ngen::HW hw, tile_t tile,
        const layout_t &src_layout, const layout_t &dst_layout)
    : hw_(hw), tile_(std::move(tile)) {
    gpu_assert(src_layout.type() == dst_layout.type());

    dim_idx_t a_idx, b_idx;
    dim_t tile_a, tile_b;
    tile_to_2d_dims(tile_, a_idx, b_idx, tile_a, tile_b);

    // Convert src/dst to 2D layouts.
    dim_assignment_t to_ab(src_layout.ndims(), 2);
    to_ab.assign(a_idx, 0);
    to_ab.assign(b_idx, 1);
    auto src_ab = to_ab.map(src_layout);
    auto dst_ab = to_ab.map(dst_layout);

    src_ = src_ab;
    dst_ = dst_ab;
    // Find minimal cost reorder path between layouts.
    path_ = find_min_cost_path(hw_, src_ab, dst_ab, tile_a, tile_b);
}

void reorder_2d_impl_t::emit(
        copy_plan_t &plan, copy_operand_t &src, copy_operand_t &dst) {
    auto &orig_type = src_.type();
    auto orig_bits = orig_type.bitsize();

    // Allocate a temporary GRF buffer if needed.
    copy_operand_t tmp;
    const auto &type = dst_.type();
    auto elems = into<int>(dst_.size() * type.packing() / type.size());
    if (path_.size() > 1) tmp = plan.newTemp(to_ngen(type), elems, 1);

    // Iterate through found reorders.
    auto *prev_layout = &src_;
    auto prev_op = src;
    int path_len = int(path_.size());
    for (int i = 0; i < path_len; i++) {
        auto &step = path_[i];
        auto &tile = step.tile;
        auto &type = step.type;
        auto type_bits = type.bitsize();
        auto *next_layout = &step.layout;

        // x -> y reorder.
        auto x = prev_layout->sub(tile).reinterpret(type);
        auto y = next_layout->sub(tile).reinterpret(type);

        bool use_dst = ((path_len - i) % 2 == 1);
        copy_operand_t next_op = use_dst ? dst : tmp;
        auto &x_blocks = x.blocks();
        auto &y_blocks = y.blocks();
        gpu_assert(x_blocks.size() <= 1);
        gpu_assert(y_blocks.size() <= 1);
        int x_stride = (x_blocks.empty() ? 1 : int(x_blocks[0].stride));
        int y_stride = (y_blocks.empty() ? 1 : int(y_blocks[0].stride));
        int width = int(tile.elems()) * orig_type.size() / type.size();

        const auto &src_layout = *prev_layout, &dst_layout = *next_layout;
        auto swizzle = [&](const icoord_t &start) {
            auto src = prev_op, dst = next_op;
            src.type = dst.type = to_ngen(type);
            src.offset = (uint8_t)(src.offset * orig_bits / type_bits);
            dst.offset = (uint8_t)(dst.offset * orig_bits / type_bits);
            auto src_off
                    = src_layout.offset<int>(start) * orig_bits / type_bits;
            auto dst_off
                    = dst_layout.offset<int>(start) * orig_bits / type_bits;
            src.advance(hw_, src_off);
            dst.advance(hw_, dst_off);
            src.stride = (uint8_t)x_stride;
            dst.stride = (uint8_t)y_stride;

            plan.mov(width, dst, src);
        };

        dst_layout.for_each_tile(tile, swizzle);
        prev_layout = next_layout;
        prev_op = std::move(next_op);
        ++plan.phase;
    }
}

void reorder_2d_impl_t::tile_to_2d_dims(const tile_t &tile, dim_idx_t &a_idx,
        dim_idx_t &b_idx, dim_t &a, dim_t &b) {
    a_idx = dim_idx::invalid;
    b_idx = dim_idx::invalid;
    for (dim_idx_t i = 0; i < tile.size(); i++) {
        if (tile[i] == 1) continue;
        if (a_idx == dim_idx::invalid) {
            a_idx = i;
            continue;
        }
        if (b_idx == dim_idx::invalid) {
            b_idx = i;
            continue;
        }
        gpu_error_not_expected();
    }

    for (dim_idx_t i = 0; i < tile.size(); i++) {
        if (utils::one_of(i, a_idx, b_idx)) continue;
        if (a_idx == dim_idx::invalid) {
            a_idx = i;
            continue;
        }
        if (b_idx == dim_idx::invalid) {
            b_idx = i;
            continue;
        }
    }

    if (a_idx > b_idx) std::swap(a_idx, b_idx);

    a = tile[a_idx];
    b = tile[b_idx];
}

auto reorder_2d_impl_t::find_min_cost_path(ngen::HW hw, const layout_t &src,
        const layout_t &dst, dim_t tile_a, dim_t tile_b)
        -> std::vector<reorder_step_t> {
    // Create all possible edges - 2D reorders.
    std::vector<edge_t> edges;
    for (int a = 1; a <= tile_a; a *= 2) {
        for (int b = 1; b <= tile_b; b *= 2) {
            if (src.dim(0) % a != 0) continue;
            if (src.dim(1) % b != 0) continue;
            int idx = int(edges.size());
            edges.emplace_back(idx, a, b);
        }
    }

    int nedges = int(edges.size());

    // Create all possible layouts for tile_a x tile_b tensor.
    std::vector<vertex_t> vertices;
    std::vector<std::vector<std::pair<int, uint32_t>>> edge_vertices(nedges);
    auto all_layouts = generate_all_layouts(src.type(), tile_a, tile_b);
    for (auto &l : all_layouts) {
        // Skip if too many blocks.
        if (int(l.blocks().size()) > max_tile_blocks) continue;
        int v_idx = int(vertices.size());
        vertices.emplace_back(hw, v_idx, l);
        auto &v = vertices.back();
        // Pass all known reorders, the vertex/layout will filter out
        // incompatible reorders.
        v.set_edges(edges);
        // Store all vertices adjacent to a specific edge.
        for (int i = 0; i < nedges; i++) {
            uint32_t mask = v.adj_edge_type_masks[i];
            if (mask != 0) edge_vertices[i].emplace_back(v_idx, mask);
        }
    }

    // Find neighbors between all vertices.
    int nvertices = int(vertices.size());
    for (int i = 0; i < nvertices; i++) {
        auto &v = vertices[i];
        for (int j = 0; j < nedges; j++) {
            uint32_t mask = v.adj_edge_type_masks[j];
            if (mask != 0) {
                for (auto &idx_mask : edge_vertices[j]) {
                    int v_idx = idx_mask.first;
                    if (v_idx == i) continue;
                    uint32_t common_mask
                            = (mask & vertices[v_idx].adj_edge_type_masks[j]);
                    if (common_mask != 0) v.add_neighbor(&vertices[v_idx]);
                }
            }
        }
    }

    // Identify source and destination vertices.
    int src_idx = -1;
    int dst_idx = -1;
    for (int i = 0; i < nvertices; i++) {
        auto &v = vertices[i];
        if (src_idx == -1
                && v.layout.is_strictly_equal(src, /*compare_offset=*/false))
            src_idx = i;
        if (dst_idx == -1
                && v.layout.is_strictly_equal(dst, /*compare_offset=*/false))
            dst_idx = i;
    }

    gpu_assert(src_idx != -1);
    gpu_assert(dst_idx != -1);

    // Layouts are the same, just copy.
    if (src_idx == dst_idx) {
        auto &v = vertices[src_idx];
        edge_t min_edge;
        type_t min_type;
        v.cost(v, edges, min_edge, min_type);
        return {{v.layout, min_edge.tile(), min_type}};
    }

    // Dijkstra's algorithm, find the minimal cost path between src and
    // dst. Use the number of instructions to estimate the cost.
    int inf_cost = std::numeric_limits<int>::max();
    std::vector<int> cost(nvertices, inf_cost);
    std::vector<int> prev(nvertices);
    std::vector<reorder_step_t> reorder_steps(nvertices);
    std::vector<bool> seen(nvertices, false);
    cost[src_idx] = 0;
    for (int i = 0; i < nvertices; i++) {
        int min_idx = -1;
        int min_cost = inf_cost;
        for (int j = 0; j < nvertices; j++) {
            if (seen[j]) continue;
            if (cost[j] < min_cost) {
                min_idx = j;
                min_cost = cost[j];
            }
        }
        seen[min_idx] = true;
        auto &v_min = vertices[min_idx];
        for (auto *v : v_min.adj_vertices) {
            edge_t min_edge;
            type_t min_type;
            int new_cost
                    = cost[min_idx] + v_min.cost(*v, edges, min_edge, min_type);
            if (new_cost < cost[v->idx]) {
                cost[v->idx] = new_cost;
                prev[v->idx] = min_idx;
                reorder_steps[v->idx]
                        = reorder_step_t(v->layout, min_edge.tile(), min_type);
            }
        }
    }

    // Sanity check, ensure the reorder sequence is not too long.
    int max_cost = 256;
    if (cost[dst_idx] > max_cost)
        gpu_warning() << "High cost reorder generated";

    // Restore the shortest reorder path.
    std::vector<reorder_step_t> ret;
    int idx = dst_idx;
    while (idx != src_idx) {
        ret.push_back(reorder_steps[idx]);
        idx = prev[idx];
    }
    std::reverse(ret.begin(), ret.end());
    return ret;
}

void reorder_2d_impl_t::generate_all_layouts_impl(
        std::vector<layout_t> &layouts, std::vector<layout_block_t> &blocks,
        const type_t &type, dim_t a, dim_t b, dim_t stride) {
    if (a == 1 && b == 1) {
        layouts.emplace_back(type, 2, 0, blocks);
        return;
    }
    bool iterate_a = true;
    bool iterate_b = true;

    // Avoid repeating indices to keep only unique layouts.
    if (!blocks.empty()) {
        auto &last = blocks.back();
        iterate_a &= (last.dim.index() != 0);
        iterate_b &= (last.dim.index() != 1);
    }

    if (iterate_a) {
        for (dim_t a_blk = 2; a_blk <= a; a_blk++) {
            if (a % a_blk != 0) continue;
            blocks.emplace_back(0, a_blk, stride);
            generate_all_layouts_impl(
                    layouts, blocks, type, a / a_blk, b, stride * a_blk);
            blocks.pop_back();
        }
    }
    if (iterate_b) {
        for (dim_t b_blk = 2; b_blk <= b; b_blk++) {
            if (b % b_blk != 0) continue;
            blocks.emplace_back(1, b_blk, stride);
            generate_all_layouts_impl(
                    layouts, blocks, type, a, b / b_blk, stride * b_blk);
            blocks.pop_back();
        }
    }
}

void reorder_2d_impl_t::vertex_t::set_edges(const std::vector<edge_t> &edges) {
    adj_edge_type_masks.resize(edges.size());
    int type_size = layout.type().size();
    for (int i = 0; i < int(edges.size()); i++) {
        auto &e = edges[i];
        auto tile = e.tile();
        int max_type_size;
        bool ok = layout_t::try_reinterpret_to_wider_type(
                layout, layout, tile, false, &max_type_size);
        if (!ok) max_type_size = type_size;
        int from = math::ilog2q(type_size);
        int to = math::ilog2q(max_type_size);
        for (int j = from; j <= to; j++) {
            type_t type = type_t::u(8 << j);
            if (can_reorder(tile, type)) adj_edge_type_masks[i] |= (1 << j);
        }
    }
}

// Check the following limitations:
// - Assume at most one block (maybe with non-dense stride)
// - Horizontal stride must be <= 4 for GRF region
// - GRF region can't span more than 2 registers
bool reorder_2d_impl_t::vertex_t::can_reorder(
        const tile_t &tile, const type_t &type) const {
    auto ab_layout = layout.sub(tile).reinterpret(type);
    int nblocks = int(ab_layout.blocks().size());
    if (nblocks == 0) return true;
    if (nblocks > 1) return false;
    auto &last = ab_layout.blocks().back();
    int max_stride = int(last.stride * last.block);
    if (last.stride > 4) return false;
    if ((int)last.stride == 4 && type.size() != 4) return false;
    if (!math::is_pow2(last.stride)) return false;
    int max_stride_bytes = max_stride * type.size();
    int grf_size = ngen::GRF::bytes(hw);
    if (max_stride_bytes > 2 * grf_size) return false;
    return true;
}

// Finds the minimal cost of reordering from this vertex to vertex v.
int reorder_2d_impl_t::vertex_t::cost(const vertex_t &v,
        const std::vector<edge_t> &edges, edge_t &min_edge,
        type_t &min_type) const {
    int min_cost = std::numeric_limits<int>::max();
    for (int i = 0; i < int(edges.size()); i++) {
        type_t i_min_type;
        int new_cost = cost(edges[i], v, i_min_type);
        if (new_cost < min_cost) {
            min_cost = new_cost;
            min_edge = edges[i];
            min_type = i_min_type;
        }
    }
    return min_cost;
}

// Finds the minimal cost of reordering from this vertex to vertex `v`
// through edge `e`. If the reorder is possible, `type` contains the
// reorder type with the minimal cost.
int reorder_2d_impl_t::vertex_t::cost(
        const edge_t &e, const vertex_t &v, type_t &type) const {
    uint32_t mask = (adj_edge_type_masks[e.idx] & v.adj_edge_type_masks[e.idx]);
    if (mask == 0) return std::numeric_limits<int>::max();
    int cur_size = layout.type().size();
    int cur_cost = into<int>(layout.elems() / (e.a * e.b));
    int min_log_bytes = math::ilog2q(cur_size);
    int max_log_bytes = 3;
    int min_cost = std::numeric_limits<int>::max();
    for (int i = min_log_bytes; i <= max_log_bytes; i++) {
        if ((mask & (1 << i)) == 0) continue;
        if (i > min_log_bytes) {
            gpu_assert(!layout.blocks().empty());
            gpu_assert(!v.layout.blocks().empty());
            int dim_idx0 = layout.blocks()[0].dim;
            int dim_idx1 = v.layout.blocks()[0].dim;
            if (dim_idx0 != dim_idx1) continue;
        }
        min_cost = cur_cost;
        type = type_t::u(8 << i);
        break;
    }
    return min_cost;
}

void reorder_impl_t::emit(copy_plan_t &plan, const reg_buf_data_t &src,
        const reg_buf_data_t &dst) {
    auto from_rd = [](const reg_buf_data_t &rd) -> op_init_t {
        return [&](int elems, ngen::DataType dt) {
            return rd.format(0, elems, 1, dt);
        };
    };

    auto from_op = [](const reorder_operand_t &op) -> op_init_t {
        return [&](int elems, ngen::DataType dt) {
            auto buffer = op.buffer;
            buffer.stride = (uint8_t)(buffer.stride
                    * ngen::getBytes(buffer.type) / ngen::getBytes(dt));
            buffer.type = dt;
            return buffer;
        };
    };

    op_init_t from_temp = [&](int elems, ngen::DataType dt) {
        return plan.newTemp(dt, elems, 1);
    };

    auto src_op = init_operand(src_layout_, from_rd(src));
    auto dst_op = init_operand(dst_layout_, from_rd(dst));
    const bool direct_copy = layouts_compatible(src_op.layout, dst_op.layout);

    const auto &src_dt = src_op.layout.type();
    const auto &dst_dt = dst_op.layout.type();
    type_t tmp_dt = intermediate_data_type(src_dt, dst_dt);

    const bool do_pre_conv = src_dt != tmp_dt;
    const bool do_post_conv = dst_dt != tmp_dt;
    // XeHPC+ uses a word-sized channel intermediate for u8<->s8, so even
    // "in-place" conversion would require a larger buffer than either operand
    const int size_mask = hw_ < ngen::HW::XeHPC ? 7 : 6;
    const bool in_place = dst_dt.size() >= tmp_dt.size()
            && (dst_dt == tmp_dt || dst_dt.size() & size_mask);
    layout_t up_layout = make_compact_layout(src_op.layout, tmp_dt, true);
    layout_t down_layout = in_place
            ? make_retyped_layout(dst_op.layout, tmp_dt)
            : make_compact_layout(dst_op.layout, tmp_dt);

    if (direct_copy || !(do_pre_conv || do_post_conv)) {
        // Pure conversion or pure swizzle
        emit(plan, dst_op, src_op);
    } else if (do_pre_conv && do_post_conv) {
        const bool has_swizzle = up_layout != down_layout;
        auto tmp_op = init_operand(std::move(up_layout), from_temp);
        emit(plan, tmp_op, src_op);
        if (has_swizzle) {
            // Integer swizzle
            auto tmp2_op = in_place
                    ? init_operand(std::move(down_layout), from_op(dst_op))
                    : init_operand(std::move(down_layout), from_temp);
            emit(plan, tmp2_op, tmp_op);
            std::swap(tmp_op, tmp2_op);
        }
        emit(plan, dst_op, tmp_op);
    } else if (do_pre_conv) {
        auto tmp_op = init_operand(std::move(up_layout), from_temp);
        emit(plan, tmp_op, src_op);
        emit(plan, dst_op, tmp_op);
    } else if (do_post_conv) {
        auto tmp_op = in_place
                ? init_operand(std::move(down_layout), from_op(dst_op))
                : init_operand(std::move(down_layout), from_temp);
        emit(plan, tmp_op, src_op);
        emit(plan, dst_op, tmp_op);
    }
}

bool reorder_impl_t::layouts_compatible(
        const layout_t &a, const layout_t &b) const {
    // Test to see if all of the non-size-1 blocks of the two layouts are
    // listed in the same order, ignoring strides.
    using iterator_t = decltype(a.blocks().begin());
    iterator_t a_block_it = a.blocks().begin();
    iterator_t b_block_it = b.blocks().begin();
    const iterator_t a_block_end = a.blocks().end();
    const iterator_t b_block_end = b.blocks().end();

    auto skip_size_1_blocks = [](iterator_t &it, const iterator_t &end) {
        while (it != end && it->block == 1)
            it++;
    };

    while (true) {
        skip_size_1_blocks(a_block_it, a_block_end);
        skip_size_1_blocks(b_block_it, b_block_end);
        if (a_block_it == a_block_end || b_block_it == b_block_end) break;

        if (a_block_it->dim != b_block_it->dim) return false;
        if (a_block_it->block != b_block_it->block) return false;
        a_block_it++;
        b_block_it++;
    }

    return a_block_it == a_block_end && b_block_it == b_block_end;
}

layout_t reorder_impl_t::make_retyped_layout(
        const layout_t &layout, const type_t &type) const {
    if (layout.blocks().empty()) return layout;
    const int stride = into<int>(layout.blocks().front().stride);
    return layout.retype(type).make_strided(
            stride * layout.type().size() / type.size());
}

layout_t reorder_impl_t::make_compact_layout(
        const layout_t &layout, const type_t &type, bool is_source) const {
    const auto grf_size = ngen::GRF::bytes(hw_);
    const auto grf_elems = grf_size * type.packing() / type.size();
    const auto align_offset = is_source && layout.type().is_hf8();

    std::vector<layout_block_t> blocks;
    dim_t dense_input_stride = 1;
    dim_t dense_output_stride = 1;
    if (layout.type().is_x8() && type.is_x8() && layout.type() != type)
        // For byte intermediate (only s8<->u8 reorder), use stride-2
        // to avoid using too many temporaries.
        dense_output_stride = 2;
    for (auto &block : layout.blocks()) {
        dim_t input_stride = block.stride;
        blocks.push_back(block);
        auto &stride = blocks.back().stride;
        if (blocks.size() == 1 || input_stride == dense_input_stride)
            stride = dense_output_stride;
        else if (hw_ <= ngen::HW::XeLP && align_offset) {
            // XeLP-specific path; conversion sequence contains
            //   shl x:uw y:ub n
            // which seems to require offset alignment.
            const auto align = grf_size >> 1;
            auto offset = input_stride % align;
            stride = utils::rnd_up(dense_output_stride - offset, align)
                    + offset;
        } else
            stride = std::min(
                    utils::rnd_up(dense_output_stride, grf_elems >> 1),
                    utils::rnd_up_pow2(dense_output_stride));
        dense_output_stride = blocks.back().stride * block.block;
        dense_input_stride = input_stride * block.block;
    }
    return {type, layout.ndims(), 0, blocks, /*do_normalize=*/false};
}

void reorder_impl_t::emit_1d(copy_plan_t &plan, const reorder_operand_t &dst,
        const reorder_operand_t &src) {
    int src_stride, dst_stride;
    auto tile = find_max_tile_with_fixed_stride(
            src.layout, dst.layout, src_stride, dst_stride);
    int tile_elems = int(tile.elems());

    const auto sat = ngen::InstructionModifier::createSaturate();
    ngen::InstructionModifier mod;
    if (needs_saturate(dst.type(), src.type())) mod |= sat;

    dst.layout.for_each_tile(tile, [&](const icoord_t &start) {
        // Tile operands
        auto tile_src = src.buffer, tile_dst = dst.buffer;
        tile_src.stride = (uint8_t)src_stride;
        tile_dst.stride = (uint8_t)dst_stride;
        tile_src.advance(hw_, src.layout.offset<int>(start));
        tile_dst.advance(hw_, dst.layout.offset<int>(start));
        plan.mov(tile_elems, mod, tile_dst, tile_src);
    });
    ++plan.phase;
}

bool reorder_impl_t::try_emit_2d(copy_plan_t &plan,
        const reorder_operand_t &dst, const reorder_operand_t &src) {
    const int grf_size = ngen::GRF::bytes(hw_);

    if (src.layout.type() != dst.layout.type()) return false;
    if (!src.layout.is_dense()) return false;
    if (!dst.layout.is_dense()) return false;

    auto tiles = find_2d_dense_tiles(src.layout, dst.layout);
    const auto base_phase = plan.phase;
    for (const auto &tile : tiles) {
        if (tile.size() < 2) continue;
        if (tile.elems() < 4) break;
        auto src_tile_layout = src.layout.sub(tile);
        auto dst_tile_layout = dst.layout.sub(tile);
        if (!dst_tile_layout.is_dense()) continue;

        // Set layout offset to 0 since the offset is handled by fixing up
        // the register input to try_emit_2d_impl
        src_tile_layout.set_offset(0);
        dst_tile_layout.set_offset(0);

        // Try to allocate/release a temporary buffer to avoid
        // out_of_registers exception.
        auto tile_grfs
                = into<int>(utils::div_up(dst_tile_layout.size(), grf_size));
        ngen::GRFRange dummy;
        plan.alloc_grf(tile_grfs, dummy);
        if (dummy.isInvalid()) continue;

        // Allocation succeeded, can proceed further.
        plan.alloc_grf(0, dummy);

        reorder_2d_impl_t r(hw_, tile, src_tile_layout, dst_tile_layout);
        bool tile_ok = true;
        for (auto &step : r.path())
            if (step.tile.elems() < 2) {
                tile_ok = false;
                break;
            }
        // Skip any 2d reorder that attempts scalar moves
        if (!tile_ok) continue;

        auto emit_2d_tile = [&](const icoord_t &start) {
            auto tile_src = src.buffer, tile_dst = dst.buffer;
            tile_src.advance(hw_, src.layout.offset<int>(start));
            tile_dst.advance(hw_, dst.layout.offset<int>(start));
            plan.phase = base_phase;
            r.emit(plan, tile_src, tile_dst);
        };

        src.layout.for_each_tile(tile, emit_2d_tile);
        return true;
    }
    return false;
}

std::vector<tile_t> reorder_impl_t::find_2d_dense_tiles(
        const layout_t &a, const layout_t &b) {
    static constexpr int max_tile_blocks = reorder_2d_impl_t::max_tile_blocks;

    auto tiles = [](const layout_t &l, uint32_t &dim_mask) {
        int ndims = 0, count = 0;
        dim_t stride = 1;
        std::vector<dim_t> dims(l.ndims(), 1);
        std::vector<tile_t> tiles;
        for (auto &b : l.blocks()) {
            if (b.block == 1) continue;
            if (count >= max_tile_blocks) break;
            uint32_t dim_bit = 1u << b.dim;
            if ((dim_t)b.stride != stride) break;
            if (!(dim_mask & dim_bit)) {
                if (ndims >= 2) break;
                ndims += 1;
                dim_mask |= dim_bit;
            }
            auto pow2_block = b.block & ~(b.block - 1);
            for (dim_t d = 1; d < pow2_block; d *= 2) {
                dims[b.dim] *= 2;
                tiles.emplace_back(dims);
            }
            if (b.block != pow2_block) break;
            stride *= b.block;
            count++;
        }
        return tiles;
    };

    uint32_t a_dim_mask = 0, b_dim_mask = 0;
    auto a_tiles = tiles(a, a_dim_mask);
    auto b_tiles = tiles(b, b_dim_mask);
    if (a_dim_mask != b_dim_mask) return {};

    std::vector<tile_t> ret;
    for (auto a_it = a_tiles.begin(), b_it = b_tiles.begin();
            a_it != a_tiles.end() && b_it != b_tiles.end(); ++a_it, ++b_it)
        if (a_it->operator==(*b_it)) ret.insert(ret.begin(), *b_it);
    return ret;
}

tile_t reorder_impl_t::find_max_tile_with_fixed_stride(const layout_t &src,
        const layout_t &dst, int &src_stride, int &dst_stride) {
    // 1. Split layouts to have aligned blocks.
    auto a = src;
    auto b = dst;
    layout_t::align_layouts(a, b);

    // 2. Find the max innermost tile.
    auto a_blocks = a.blocks();
    auto b_blocks = b.blocks();

    std::vector<dim_t> tile_dims(a.ndims(), 1);
    src_stride = (a_blocks.empty() ? 1 : int(a_blocks[0].stride));
    dst_stride = (b_blocks.empty() ? 1 : int(b_blocks[0].stride));
    if (src_stride & (src_stride - 1) || src_stride > 4) src_stride = 1;
    if (dst_stride & (dst_stride - 1) || dst_stride > 4) dst_stride = 1;
    int src_cur_stride = src_stride;
    int dst_cur_stride = dst_stride;

    int min_blocks = int(std::min(a_blocks.size(), b_blocks.size()));
    for (int i = 0; i < min_blocks; i++) {
        auto &ab = a_blocks[i];
        auto &bb = b_blocks[i];
        if (ab.dim != bb.dim || ab.block != bb.block) break;

        // Strides are supported for the innermost block only.
        if (src_cur_stride != int(ab.stride)) break;
        if (dst_cur_stride != int(bb.stride)) break;

        src_cur_stride = int(ab.block * ab.stride);
        dst_cur_stride = int(bb.block * bb.stride);
        tile_dims[ab.dim] *= ab.block;
    }
    return tile_t(tile_dims);
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
