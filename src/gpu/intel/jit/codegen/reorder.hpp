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

#ifndef GPU_INTEL_JIT_CODEGEN_REORDER_HPP
#define GPU_INTEL_JIT_CODEGEN_REORDER_HPP

#include <functional>

#include "common/utils.hpp"
#include "gpu/intel/jit/codegen/operand.hpp"
#include "gpu/intel/jit/codegen/register_scope.hpp"
#include "gpu/intel/jit/gemm/generator/pieces/copy_plan.hpp"
#include "gpu/intel/jit/ir/reorder.hpp"
#include "gpu/intel/jit/ir/tensor.hpp"
#include "gpu/intel/jit/utils/iterator.hpp"
#include "gpu/intel/jit/utils/range.hpp"
#include "ngen.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

struct copy_operand_t : gemmstone::CopyOperand {
    copy_operand_t() = default;
    copy_operand_t(const CopyOperand &op) : CopyOperand(op) {}

    copy_operand_t(const reg_buf_data_t &rbd) : CopyOperand(rbd) {
        if (!rbd.is_empty()) {
            const auto base = rbd.base();
            const auto &rd = rbd.reg_buf();
            block_size = rd.block_regs();
            if (rd.blocks() <= 1) return;
            block_bases.reserve(rd.blocks());
            for (int i = 0, j = 0; i < rd.blocks(); ++i, j += block_size) {
                auto block_base = rd.base(j);
                block_bases.push_back(block_base);
                if (block_base <= base && base < block_base + block_size)
                    block_off = i;
            }
        }
    }

    copy_operand_t &advance(ngen::HW hw, int elems, uint8_t stride = 1) {
        const auto grf_bits = ngen::GRF::bytes(hw) << 3;
        const auto type_bit_size = ngen::getBits(type);
        const auto bit_off = (offset + elems * stride) * type_bit_size;
        const auto grf_shift = bit_off / grf_bits;
        if (temp || block_bases.empty() || !block_size)
            grf += grf_shift;
        else {
            const auto orig_block_base = block_bases[block_off];
            const auto grf_off = grf - orig_block_base + grf_shift;
            const auto block_shift = grf_off / block_size;
            block_off += block_shift;
            grf = (int16_t)(block_bases[block_off] + (grf_off % block_size));
        }
        offset = (uint8_t)((bit_off % grf_bits) / type_bit_size);
        return *this;
    }

    std::vector<int> block_bases;
    int block_size = 0;
    int block_off = 0;
};

struct copy_plan_t : gemmstone::CopyPlan {
    using gemmstone::CopyPlan::newTemp;

    void mov(int simd, ngen::InstructionModifier mod, const copy_operand_t &dst,
            const copy_operand_t &src) {
        static constexpr ngen::Opcode mov = ngen::Opcode::mov;
        const auto grf_bits = ngen::GRF::bytes(hw()) << 3;

        auto max_simd = [&](const copy_operand_t &op) {
            const auto &block_size = op.block_size;
            const auto &block_bases = op.block_bases;
            // Count contiguous registers
            int regs = block_size - (op.grf - block_bases[op.block_off]);
            for (size_t j = op.block_off; j < block_bases.size(); ++j) {
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
            block_dst.advance(hw(), block_simd, dst.stride);
            block_src.advance(hw(), block_simd, src.stride);
            simd -= block_simd;
        }
    }

    void mov(int simd, const copy_operand_t &dst, const copy_operand_t &src) {
        return mov(simd, {}, dst, src);
    }

    copy_plan_t(ngen_register_scope_t &scope, bool systolic_support)
        : CopyPlan(scope.hw(), systolic_support), scope_(scope) {}

    ngen::HW hw() const { return CopyPlan::hw; }

    template <typename Generator>
    void execute(Generator &generator) {
        using namespace std::placeholders;
        GRFAllocator grf = std::bind(&copy_plan_t::alloc_grf, this, _1, _2);
        FlagAllocator flag = std::bind(&copy_plan_t::alloc_flag, this, _1, _2);
        CopyPlan::materializeTemps(grf, flag);
        CopyPlan::execute(generator);
    }

    void alloc_grf(int count, ngen::GRFRange &range) {
        if (count > 0)
            range = scope_.try_alloc_range(count);
        else
            scope_.safeRelease(range);
    }

    void alloc_flag(int bytes, ngen::FlagRegister &flag) {
        if (bytes > 0)
            flag = scope_.try_alloc_flag(bytes * 8);
        else
            scope_.safeRelease(flag);
    }

    int phase = 0;

protected:
    using CopyPlan::materializeTemps;

    ngen_register_scope_t &scope_;
};

template <typename GeneratorT>
void emit_reorder_1d_tile(GeneratorT *host, ngen_register_scope_t &scope,
        int width, const reg_buf_data_t &src, int src_stride,
        const reg_buf_data_t &dst, int dst_stride) {
    copy_plan_t plan(scope, host->exec_cfg().hw().systolic_support());
    copy_operand_t dst_op = dst;
    copy_operand_t src_op = src;
    dst_op.stride = (uint8_t)dst_stride;
    src_op.stride = (uint8_t)src_stride;

    if (!math::is_pow2(src_stride) || !math::is_pow2(dst_stride)) {
        for (int i = 0; i < width; ++i) {
            plan.mov(1, dst_op, src_op);
            dst_op.advance(plan.hw(), dst_stride);
            src_op.advance(plan.hw(), src_stride);
        }
    } else
        plan.mov(width, dst_op, src_op);
    plan.transform();
    plan.execute(*host);
}

template <typename GeneratorT>
void align_src_dst_offset(GeneratorT *host, ngen_register_scope_t &scope,
        const ngen::InstructionModifier &mod, const reg_buf_data_t &dst,
        reg_buf_data_t &src) {
    int src_stride = src.hs();
    // src is broadcasted, no need to align, return.
    if (src_stride == 0) return;

    bool is_xf = ngen_is_xf(src.type()) || ngen_is_xf(dst.type());
    bool is_bf_to_f = (src.type() == ngen::DataType::bf)
            && (dst.type() == ngen::DataType::f);
    int src_type_size = ngen::getBytes(src.type());
    int dst_type_size = ngen::getBytes(dst.type());
    int src_off = src.offset();
    int dst_off = dst.offset();
    int src_byte_off = src.byte_offset();
    int dst_byte_off = dst.byte_offset();
    int esize = mod.getExecSize();
    const int grf_size = ngen::GRF::bytes(scope.hw());
    // within the current generator, HS == 0 can mean 2 things:
    //   - <0; 1, 0>, i.e. a scalar value so HS is to be treated as 1
    //   - <1; 1, 0>, which is a more compatible representation of <N; N, 1>
    int grf_src = grf_size / std::max(src.hs(), 1);
    int grf_dst = grf_size / std::max(dst.hs(), 1);

    // If src is aligned with dst, return.
    if ((is_xf || is_bf_to_f) && src_off % grf_src == dst_off % grf_dst) return;
    if (!is_xf && src_byte_off % grf_size == dst_byte_off % grf_size) return;

    int new_src_off = (is_xf ? dst_off * src_type_size / dst_type_size
                             : dst_off * dst_type_size / src_type_size);

    int src_size = std::max(src_type_size * esize * src_stride, src_type_size);
    auto new_src = scope.alloc_reg_buf_data(
            utils::div_up(src_size + new_src_off * src_type_size, grf_size));
    new_src = new_src.format(new_src_off, esize, src_stride, src.type());
    emit_reorder_1d_tile(
            host, scope, esize, src, src_stride, new_src, src_stride);
    src = std::move(new_src);
}

template <typename GeneratorT>
void align_src_dst_offset(GeneratorT *host, ngen_register_scope_t &scope,
        const ngen::InstructionModifier &mod, const reg_buf_data_t &dst,
        reg_buf_data_t &src0, reg_buf_data_t &src1) {
    align_src_dst_offset(host, scope, mod, dst, src0);
    align_src_dst_offset(host, scope, mod, dst, src1);
}

template <typename GeneratorT>
void align_src_dst_offset(GeneratorT *host, ngen_register_scope_t &scope,
        const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
        ngen_operand_t &src) {
    if (!src.is_reg_data()) return;
    auto rd = src.reg_buf_data();

    if (!dst.is_reg_data()) {
        // Float pipe requires src operands to align with dst, even if that's
        // the null register. In the case of the null register, we align to the
        // GRF boundary.
        reg_buf_data_t dummy(reg_buf_t(rd.hw(), ngen::GRFRange(0, 1)));
        // This call returns early if everything is already aligned nicely
        align_src_dst_offset(host, scope, mod, dummy, rd);
    } else {
        align_src_dst_offset(host, scope, mod, dst.reg_buf_data(), rd);
    }
    if (rd == src.reg_buf_data()) return;

    bool is_negated = src.is_negated();
    src = ngen_operand_t(rd, src.mod());
    if (is_negated) src = -src;
}

template <typename GeneratorT>
void align_src_dst_offset(GeneratorT *host, ngen_register_scope_t &scope,
        const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
        ngen_operand_t &src0, ngen_operand_t &src1) {
    align_src_dst_offset(host, scope, mod, dst, src0);
    align_src_dst_offset(host, scope, mod, dst, src1);
}

struct reorder_operand_t {
    layout_t layout;
    copy_operand_t buffer;

    const type_t &type() const { return layout.type(); }
};

// Implementation of GRF reorder between 2D dense layouts.
// Requirements for A -> B reorder:
// - A and B must have the same data type
// - Layouts must be 2D and dense
// Reorder may require several steps, in this case a temporary buffer T is
// allocated. For example: A -> T -> B or A -> B -> T -> B
class reorder_2d_impl_t {
    struct reorder_step_t;

public:
    reorder_2d_impl_t(ngen::HW hw, tile_t tile, const layout_t &src_layout,
            const layout_t &dst_layout)
        : hw_(hw), tile_(std::move(tile)) {
        gpu_assert(src_layout.type() == dst_layout.type());

        dim_idx_t a_idx, b_idx;
        int tile_a, tile_b;
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

    const tile_t &tile() const { return tile_; }
    const std::vector<reorder_step_t> &path() const { return path_; }

    void emit(copy_plan_t &plan, copy_operand_t &src, copy_operand_t &dst) {
        auto &orig_type = src_.type();
        auto orig_bits = orig_type.bitsize();

        // Allocate a temporary GRF buffer if needed.
        copy_operand_t tmp;
        const auto &type = dst_.type();
        auto elems = dst_.size() * type.packing() / type.size();
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
            auto x = prev_layout->map(tile).reinterpret(type);
            auto y = next_layout->map(tile).reinterpret(type);

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
            prev_op = next_op;
            ++plan.phase;
        }
    }

    static const int max_tile_blocks = 4;

private:
    // Represents 2D reorder corresponding to (a x b) tile.
    struct edge_t {
        edge_t() = default;
        edge_t(int idx, int a, int b) : idx(idx), a(a), b(b) {}

        tile_t tile() const { return tile_t(std::vector<dim_t> {a, b}); }

        std::string str() const {
            std::ostringstream oss;
            oss << "edge(idx = " << idx << ", a = " << a << ", b = " << b
                << ")";
            return oss.str();
        }

        int idx; // Identifier of the edge.
        int a = 0, b = 0; // Specify tile (a x b).
    };

    // Represents GRF layout between edges-reorders.
    struct vertex_t {
        vertex_t(ngen::HW hw, int idx, const layout_t &layout)
            : hw(hw), idx(idx), layout(layout) {}

        std::string str() const {
            std::ostringstream oss;
            oss << "vertex(idx = " << idx << ", layout = " << layout << ")";
            return oss.str();
        }

        void set_edges(const std::vector<edge_t> &edges) {
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
                    if (can_reorder(tile, type))
                        adj_edge_type_masks[i] |= (1 << j);
                }
            }
        }

        void add_neighbor(const vertex_t *v) { adj_vertices.push_back(v); }

        bool is_neighbor(const vertex_t &v) const {
            for (auto *n : adj_vertices)
                if (n == &v) return true;
            return false;
        }

        // Check the following limitations:
        // - Assume at most one block (maybe with non-dense stride)
        // - Horizontal stride must be <= 4 for GRF region
        // - GRF region can't span more than 2 registers
        bool can_reorder(const tile_t &tile, const type_t &type) const {
            auto ab_layout = layout.map(tile).reinterpret(type);
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
        int cost(const vertex_t &v, const std::vector<edge_t> &edges,
                edge_t &min_edge, type_t &min_type) const {
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
        int cost(const edge_t &e, const vertex_t &v, type_t &type) const {
            uint32_t mask = (adj_edge_type_masks[e.idx]
                    & v.adj_edge_type_masks[e.idx]);
            if (mask == 0) return std::numeric_limits<int>::max();
            int cur_size = layout.type().size();
            int cur_cost = layout.elems() / (e.a * e.b);
            int min_log_bytes = math::ilog2q(cur_size);
            int max_log_bytes = 3;
            int min_cost = std::numeric_limits<int>::max();
            for (int i = min_log_bytes; i <= max_log_bytes; i++) {
                if ((mask & (1 << i)) == 0) continue;
                if (i > min_log_bytes) {
                    gpu_assert(!layout.blocks().empty());
                    gpu_assert(!v.layout.blocks().empty());
                    int dim_idx0 = layout.blocks()[0].dim_idx;
                    int dim_idx1 = v.layout.blocks()[0].dim_idx;
                    if (dim_idx0 != dim_idx1) continue;
                }
                min_cost = cur_cost;
                type = type_t::u(8 << i);
                break;
            }
            return min_cost;
        }

        ngen::HW hw;
        int idx; // Identifier of the vertex.
        layout_t layout; // Layout of the vertex.
        // Specifies a bitmask for every edge: if adj_edge_type_masks[E_idx]
        // has b-th bit set then this vertex can be reordered through E edge
        // using the data type with size 2^b bytes.
        std::vector<uint32_t> adj_edge_type_masks;
        std::vector<const vertex_t *> adj_vertices; // Adjacent vertices.
    };

    // Represents a reorder step.
    struct reorder_step_t {
        reorder_step_t() = default;
        reorder_step_t(
                const layout_t &layout, const tile_t &tile, const type_t &type)
            : layout(layout), tile(tile), type(type) {}

        layout_t layout; // Destination layout.
        tile_t tile; // Tile corresponding to one instruction.
        type_t type; // Registers should be reinterpreted to `type` for reorder.
    };

    // Extracts dimension sizes and their indices from a multidimensional
    // tensor.
    static void tile_to_2d_dims(const tile_t &tile, dim_idx_t &a_idx,
            dim_idx_t &b_idx, int &a, int &b) {
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

    // Finds the optimal sequence of reorders between src and dst layouts.
    static std::vector<reorder_step_t> find_min_cost_path(ngen::HW hw,
            const layout_t &src, const layout_t &dst, int tile_a, int tile_b) {
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
        std::vector<std::vector<std::pair<int, uint32_t>>> edge_vertices(
                nedges);
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
                        uint32_t common_mask = (mask
                                & vertices[v_idx].adj_edge_type_masks[j]);
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
                    && v.layout.is_strictly_equal(
                            src, /*compare_offset=*/false))
                src_idx = i;
            if (dst_idx == -1
                    && v.layout.is_strictly_equal(
                            dst, /*compare_offset=*/false))
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
                int new_cost = cost[min_idx]
                        + v_min.cost(*v, edges, min_edge, min_type);
                if (new_cost < cost[v->idx]) {
                    cost[v->idx] = new_cost;
                    prev[v->idx] = min_idx;
                    reorder_steps[v->idx] = reorder_step_t(
                            v->layout, min_edge.tile(), min_type);
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

    // Returns all possible layouts for (a x b) tensor.
    static std::vector<layout_t> generate_all_layouts(
            const type_t &type, int a, int b) {
        std::vector<layout_t> ret;
        std::vector<block_t> blocks;
        generate_all_layouts_impl(ret, blocks, type, a, b, 1);
        return ret;
    }

    static void generate_all_layouts_impl(std::vector<layout_t> &layouts,
            std::vector<block_t> &blocks, const type_t &type, int a, int b,
            int stride) {
        if (a == 1 && b == 1) {
            layouts.emplace_back(type, 2, 0, blocks);
            return;
        }
        bool iterate_a = true;
        bool iterate_b = true;

        // Avoid repeating indices to keep only unique layouts.
        if (!blocks.empty()) {
            auto &last = blocks.back();
            iterate_a &= (last.dim_idx != 0);
            iterate_b &= (last.dim_idx != 1);
        }

        if (iterate_a) {
            for (int a_blk = 2; a_blk <= a; a_blk++) {
                if (a % a_blk != 0) continue;
                blocks.emplace_back(0, a_blk, stride);
                generate_all_layouts_impl(
                        layouts, blocks, type, a / a_blk, b, stride * a_blk);
                blocks.pop_back();
            }
        }
        if (iterate_b) {
            for (int b_blk = 2; b_blk <= b; b_blk++) {
                if (b % b_blk != 0) continue;
                blocks.emplace_back(1, b_blk, stride);
                generate_all_layouts_impl(
                        layouts, blocks, type, a, b / b_blk, stride * b_blk);
                blocks.pop_back();
            }
        }
    }

    ngen::HW hw_;
    tile_t tile_;
    layout_t src_;
    layout_t dst_;
    std::vector<reorder_step_t> path_;
};

class reorder_impl_t {
public:
    reorder_impl_t(ngen::HW hw, const reorder_t &reorder)
        : hw_(hw)
        , src_layout_(reorder.src_layout)
        , dst_layout_(reorder.dst_layout) {
        layout_t::try_reinterpret_to_wider_type(src_layout_, dst_layout_);
    }

    template <typename GeneratorT>
    void emit(GeneratorT *host, ngen_register_scope_t &scope,
            const reg_buf_data_t &src, const reg_buf_data_t &dst) {
        copy_plan_t plan(scope, host->exec_cfg().hw().systolic_support());

        auto from_rd = [](const reg_buf_data_t &rd) -> op_init_t {
            return [&](dim_t elems, ngen::DataType dt) {
                return rd.format(0, elems, 1, dt);
            };
        };
        op_init_t from_temp = [&](dim_t elems, ngen::DataType dt) {
            return plan.newTemp(dt, elems, 1);
        };

        const bool direct_copy = layouts_compatible(src_layout_, dst_layout_);
        auto src_op = init_operand(src_layout_, from_rd(src));
        auto dst_op = init_operand(dst_layout_, from_rd(dst));

        const auto &src_dt = src_op.layout.type();
        const auto &dst_dt = dst_op.layout.type();
        type_t tmp_dt = intermediate_data_type(src_dt, dst_dt);
        layout_t up_layout = make_compact_layout(src_op.layout, tmp_dt, true);
        layout_t down_layout = make_compact_layout(dst_op.layout, tmp_dt);

        const bool do_pre_conv = src_dt != tmp_dt;
        const bool do_post_conv = dst_dt != tmp_dt;

        if (direct_copy || !(do_pre_conv || do_post_conv)) {
            // Pure conversion or pure swizzle
            emit(plan, dst_op, src_op);
        } else if (do_pre_conv && do_post_conv) {
            auto tmp_op = init_operand(up_layout, from_temp);
            emit(plan, tmp_op, src_op);
            if (up_layout != down_layout) {
                // Integer swizzle
                auto tmp2_op = init_operand(down_layout, from_temp);
                emit(plan, tmp2_op, tmp_op);
                std::swap(tmp_op, tmp2_op);
            }
            emit(plan, dst_op, tmp_op);
        } else if (do_pre_conv) {
            auto tmp_op = init_operand(up_layout, from_temp);
            emit(plan, tmp_op, src_op);
            emit(plan, dst_op, tmp_op);
        } else if (do_post_conv) {
            auto tmp_op = init_operand(down_layout, from_temp);
            emit(plan, tmp_op, src_op);
            emit(plan, dst_op, tmp_op);
        }

        plan.transform();
        plan.execute(*host);
    }

private:
    using op_init_t = std::function<copy_operand_t(dim_t, ngen::DataType)>;

    void emit(copy_plan_t &plan, const reorder_operand_t &dst,
            const reorder_operand_t &src) {
        if (try_emit_2d(plan, dst, src)) return;
        emit_1d(plan, dst, src);
    }

    bool layouts_compatible(const layout_t &a, const layout_t &b) {
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

            if (a_block_it->dim_idx != b_block_it->dim_idx) return false;
            if (a_block_it->block != b_block_it->block) return false;
            a_block_it++;
            b_block_it++;
        }

        return a_block_it == a_block_end && b_block_it == b_block_end;
    }

    reorder_operand_t init_operand(layout_t layout, const op_init_t &init) {
        if (layout.type().is_tf32()) layout = layout.retype(type_t::f32());
        auto elems = size_in_elems(layout);
        auto dt = to_ngen(layout.type());
        auto buffer = init(elems, dt);
        buffer.stride = (uint8_t)1;
        return {layout, buffer};
    }

    layout_t make_compact_layout(const layout_t &layout, const type_t &type,
            bool is_source = false) const {
        const auto grf_size = ngen::GRF::bytes(hw_);
        const auto grf_elems = grf_size * type.packing() / type.size();
        const auto align_offset = is_source && layout.type().is_hf8();

        std::vector<block_t> blocks;
        dim_t dense_input_stride = 1;
        dim_t dense_output_stride = 1;
        if (layout.type().size() == 1 && needs_saturate(layout.type(), type))
            // For byte intermediate (only s8<->u8 reorder), use stride-2
            // to avoid using too many temporaries
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
                stride = utils::rnd_up(dense_output_stride, grf_elems >> 1);
            dense_output_stride = blocks.back().stride * block.block;
            dense_input_stride = input_stride * block.block;
        }
        return {type, layout.ndims(), 0, blocks, /*do_normalize=*/false};
    }

    dim_t size_in_elems(const layout_t &layout) {
        const auto &type = layout.type();
        return layout.size() * type.packing() / type.size();
    }

    type_t intermediate_data_type(const type_t &s, const type_t &d) {
        // Force up-/down-convert of small types
        if (s.is_fp4() || d.is_fp4()) return type_t::f16();
        if (s.is_u4() || d.is_u4()) return type_t::u16();
        if (s.is_s4() || d.is_s4()) return type_t::s16();

        if (s == d) return d; // Swizzle only
        if (s.is_fp8() && d.is_fp()) return type_t::f16();
        if (s.is_fp() && d.is_fp8()) return type_t::f16();
        if (s.size() > 4) return d;
        if (d.size() > 4) return s;
        return s.bitsize() > d.bitsize() ? s : d;
    }

    bool needs_saturate(const type_t &ddt, const type_t &sdt) const {
        if (!ddt.is_int() || !sdt.is_int()) return false;
        if (ddt.bitsize() >= sdt.bitsize()
                && ddt.is_signed() == sdt.is_signed())
            return false;
        return true;
    }

    void emit_1d(copy_plan_t &plan, const reorder_operand_t &dst,
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

    static std::vector<tile_t> find_2d_dense_tiles(
            const layout_t &a, const layout_t &b) {
        using tile_pair_t = std::array<tile_t, 2>;
        static constexpr int max_tile_blocks
                = reorder_2d_impl_t::max_tile_blocks;
        auto dense_2d_blocks = []() {
            dim_t stride = 1;
            int non_one_dims = 0;
            int count = 0;
            std::unordered_set<int> seen;
            return [=](const block_t &b) mutable {
                if ((dim_t)b.stride != stride) return false;
                if (b.block != 1) {
                    count++;
                    stride *= b.block;
                    auto ret = seen.insert(b.dim_idx);
                    if (ret.second) non_one_dims++;
                }
                return non_one_dims <= 2 && count <= max_tile_blocks;
            };
        };

        auto take_smaller = [](const tile_t &a, const tile_t &b) {
            return a.elems() <= b.elems();
        };

        auto equal_tiles = [](const tile_pair_t &p) { return p[0] == p[1]; };

        auto to_single_tile = [](const tile_pair_t &p) { return p[0]; };

        auto all_dims_pow2 = [](const tile_t &tile) {
            for (auto d : tile.values())
                if (!math::is_pow2(d)) return false;
            return true;
        };

        auto a_tiles = inner_tiles(
                a.blocks() | filter(dense_2d_blocks()), a.ndims());
        auto b_tiles = inner_tiles(
                b.blocks() | filter(dense_2d_blocks()), b.ndims());
        auto tiles = merge(a_tiles, b_tiles, take_smaller) | filter(equal_tiles)
                | transform(to_single_tile) | filter(all_dims_pow2);
        std::vector<tile_t> ret;
        for (const auto &tile : tiles)
            ret.insert(ret.begin(), tile);
        return ret;
    }

    bool try_emit_2d(copy_plan_t &plan, const reorder_operand_t &dst,
            const reorder_operand_t &src) {
        const int grf_size = ngen::GRF::bytes(hw_);

        if (src.layout.type() != dst.layout.type()) return false;
        if (!src.layout.is_dense()) return false;
        if (!dst.layout.is_dense()) return false;

        auto tiles = find_2d_dense_tiles(src.layout, dst.layout);
        const auto base_phase = plan.phase;
        for (const auto &tile : tiles) {
            if (tile.size() < 2) continue;
            if (tile.elems() < 4) break;
            auto src_tile_layout = src.layout.map(tile);
            auto dst_tile_layout = dst.layout.map(tile);
            if (!dst_tile_layout.is_dense()) continue;

            // Set layout offset to 0 since the offset is handled by fixing up
            // the register input to reorder_2d_impl_t
            src_tile_layout.set_offset(0);
            dst_tile_layout.set_offset(0);

            // Try to allocate/release a temporary buffer to avoid
            // out_of_registers exception.
            auto tile_grfs = utils::div_up(dst_tile_layout.size(), grf_size);
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

    static tile_t find_max_tile_with_fixed_stride(const layout_t &src,
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
            if (ab.dim_idx != bb.dim_idx || ab.block != bb.block) break;

            // Strides are supported for the innermost block only.
            if (src_cur_stride != int(ab.stride)) break;
            if (dst_cur_stride != int(bb.stride)) break;

            src_cur_stride = int(ab.block * ab.stride);
            dst_cur_stride = int(bb.block * bb.stride);
            tile_dims[ab.dim_idx] *= ab.block;
        }
        return tile_t(tile_dims);
    }

    ngen::HW hw_;
    layout_t src_layout_;
    layout_t dst_layout_;
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
