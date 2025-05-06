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
#include <stack>

#include "gpu/intel/jit/gemm/include/gemmstone/strategy.hpp"
#include "gpu/intel/jit/gemm/ir/builder.hpp"
#include "gpu/intel/jit/gemm/ir/kernel_desc.hpp"
#include "gpu/intel/jit/ir/blocking.hpp"
#include "gpu/intel/jit/pass/pass.hpp"
#include "gpu/intel/jit/v2/conv/plan.hpp"
#include "gpu/intel/jit/v2/ir/tensor.hpp"
namespace gemmstone {

using dim_map_t = std::array<int, 3>;

enum class op_kind_t { load, prefetch, store };

// TODO: Replace with hardware
const int grf_size = 64;
const int min_align_2d = 16;
const int min_pitch_2d = 64;

// Sample transforms on bf16 data with pack_size 16:
// none:           64a64b -> 64a64b
// block:          64a64b -> 4b64a16b
// vnni:           64a64b -> 4b32a16b2a
// transpose_vnni: 64a64b -> 4a32b16a2b
enum class transform_t { none, block, vnni, transpose_vnni };

const ir::pvar_t &m_var = ir::pvars::m;
const ir::pvar_t &n_var = ir::pvars::n;
const ir::pvar_t &k_var = ir::pvars::k;

struct plan2d_t {
    plan2d_t() = default;
    plan2d_t(transform_t transform, int pack_size, int min_alignment,
            int min_pitch, ngen::CacheSettingsLSC cache_hint,
            std::array<ir::pvar_t, 2> dims)
        : transform(transform)
        , pack_size(pack_size)
        , min_alignment(min_alignment)
        , min_pitch(min_pitch)
        , cache_hint(to_ir(cache_hint))
        , dims(dims) {}

    ir::v2::layout_t get_layout(const ir::pvar_tile_t &sizes, ir::type_t type,
            const ir::v2::layout_desc_t &desc) const {

        auto col_var = dims[0];
        auto col = sizes[dims[0]];
        auto row_var = dims[1];
        auto row = sizes[dims[1]];
        auto t = type.size();

        auto normalized = transform;
        if (normalized == transform_t::transpose_vnni) {
            std::swap(col_var, row_var);
            std::swap(col, row);
            normalized = transform_t::vnni;
        }

        if (normalized == transform_t::vnni && t >= 4)
            normalized = transform_t::block;

        int col_inner = pack_size ? pack_size : grf_size;
        if (normalized == transform_t::block && col <= col_inner)
            normalized = transform_t::none;

        switch (normalized) {
            case transform_t::none:
                return ir::v2::layout_t(desc, type, 0,
                        {{col_var, col, 1}, {row_var, row, col}});

            case transform_t::block: {
                int col_outer = col / col_inner;
                return ir::v2::layout_t(desc, type, 0,
                        {{col_var, col_inner, 1}, {row_var, row, col_inner},
                                {col_var, col_outer, row * col_inner}});
            }

            case transform_t::vnni: {
                int row_inner = 4 / t;
                int row_outer = row / row_inner;
                int col_outer = col / col_inner;
                return ir::v2::layout_t(desc, type, 0,
                        {{row_var, row_inner, 1},
                                {col_var, col_inner, row_inner},
                                {row_var, row_outer, col_inner * row_inner},
                                {col_var, col_outer,
                                        row_outer * col_inner * row_inner}});
            }

            // Impossible to hit due to normalization
            case transform_t::transpose_vnni:
            default: stub(); return {};
        }
    }

    // Tile used for 2d Messages
    ir::pvar_tile_t get_2d_tile(ir::type_t type) const {
        if (transform == transform_t::transpose_vnni) {
            auto width = pack_size ? pack_size
                                   : grf_size / std::max(type.size(), 4);
            auto height = 32;
            return {{dims[1], width}, {dims[0], height}};
        }

        auto width = pack_size ? pack_size : grf_size / type.size();
        auto height = 32;
        return {{dims[0], width}, {dims[1], height}};
    }

    // Tile used for Block Loads
    ir::pvar_tile_t get_block_tile(ir::type_t type) const {
        if (transform == transform_t::none) {
            return {{dims[0], 8 * grf_size / type.size()}, {dims[1], 1}};
        } else if (transform == transform_t::block) {
            return {{dims[0], pack_size ? pack_size : grf_size / type.size()},
                    {dims[1], 1}};
        } else {
            stub();
        }
    }

    static ir::send_cache_hint_t to_ir(ngen::CacheSettingsLSC hint) {
        switch (hint) {
            case ngen::CacheSettingsLSC::L1C_L3C:
                return ir::send_cache_hint_t::load_once;
            case ngen::CacheSettingsLSC::Default:
                return ir::send_cache_hint_t::hw_default;
            default: stub(); return ir::send_cache_hint_t::undef;
        }
    }

    transform_t transform = transform_t::none;
    int pack_size = 0;
    int min_alignment = 0;
    int min_pitch = 0;
    ir::send_cache_hint_t cache_hint = ir::send_cache_hint_t::undef;
    std::array<ir::pvar_t, 2> dims = {};
};

enum class layout_t {
    colMajor,
    rowMajor,
    colPacked,
    rowPacked,
    colPackedVNNI,
    rowPackedVNNI,
};

struct tensor_t {
    std::string str() const {
        std::ostringstream oss;
        oss << "buffer:    " << buffer.str() << "\n";
        oss << "layout: " << layout.str() << "\n";
        return oss.str();
    }

    ir::expr_t buffer;
    ir::v2::layout_t layout;
};

struct global_tensor_t {
    ir::expr_t buffer;
    ir::type_t type;
    ir::expr_t base_offset;
    ir::pvar_map_t<ir::expr_t> idxs;
    ir::pvar_map_t<ir::expr_t> strides;
    ir::pvar_map_t<ir::expr_t> sizes;
    ir::pvar_tile_t tile;

    ir::expr_t offset(const ir::pvar_coord_t<int64_t> &coord) const {
        ir::expr_t ret = base_offset;
        for (auto &c : coord) {
            ret += (idxs[c] + coord[c]) * strides[c];
        }
        return ir::simplify(ret * type.size());
    }
};

struct kloop_iterator_t {

    virtual const global_tensor_t &A_prefetch() const = 0;
    virtual const global_tensor_t &A_load() const = 0;
    virtual const global_tensor_t &B_prefetch() const = 0;
    virtual const global_tensor_t &B_load() const = 0;
    virtual const global_tensor_t &C_store() const = 0;

    virtual void inc_prefetch_A(int k_block) = 0;
    virtual void inc_prefetch_B(int k_block) = 0;
    virtual void inc_load(int k_block) = 0;
    virtual void inc_mma(int k_block) = 0;

    virtual ir::expr_t update_C() const = 0;

    // Returns whether the given blocking is completely in bounds
    virtual ir::expr_t is_inbounds(
            int m_block, int n_block, int k_block) const = 0;
};

struct dsl_ctx_t {
    void declare_kernel(
            const ir::kernel_iface_t &interface, ir::ir_context_t &ctx) {
        gpu_assert(stmts_stack.empty())
                << "Invalid generation of a kernel within a kernel";
        interface_ = interface;
        ctx_ = &ctx;

        begin_scope();

        for (int i = 0; i < interface.nargs(); i++) {
            const auto &var = interface.arg_var(i);
            if (var.type().is_ptr()) {
                if (var.type().is_slm()) {
                    append(ir::alloc_t::make(
                            var, 0, ir::alloc_kind_t::slm, ir::stmt_t {}));
                } else {
                    append(ir::alloc_t::make(
                            var, 0, ir::alloc_kind_t::global, ir::stmt_t {}));
                }
            } else {
                append(ir::let_t::make(var, {}, {}));
            }
        }

        for (int i = 0; i < 3; i++) {
            group_ids_[i]
                    = let(ir::type_t::u32(), ir::ir_builder_t::tg_idx(i), {});
            local_ids_[i]
                    = let(ir::type_t::u16(), ir::ir_builder_t::local_id(i), {});
            local_sizes_[i] = let(
                    ir::type_t::u16(), ir::ir_builder_t::local_size(i), {});
        }
    }

    ir::stmt_t end_kernel() {
        gpu_assert(stmts_stack.size() == 1)
                << "Invalid end of kernel, imbalanced scopes detected";
        ctx_ = nullptr;
        return pop_scope();
    }

    const std::array<ir::expr_t, 3> &group_ids() const { return group_ids_; }
    ir::expr_t group_id(int idx) const { return group_ids_[idx]; }
    const std::array<ir::expr_t, 3> &local_ids() const { return local_ids_; }
    ir::expr_t local_id(int idx) const { return local_ids_[idx]; }
    const std::array<ir::expr_t, 3> &local_sizes() const {
        return local_sizes_;
    }
    ir::expr_t local_size(int idx) const { return local_sizes_[idx]; }

    ir::expr_t arg(const std::string &name) {
        auto a = interface_.find_arg(name);
        ir::expr_t value;
        if (ctx_->cset().is_single_value(a, value)) { return value; }
        return a;
    }

    // TODO: Remove IR restriction which requires force_alloc
    ir::expr_t def(ir::type_t _type, const std::string &name,
            ir::expr_t value = {}, bool force_alloc = false) {
        auto type
                = ir::type_t(_type.kind(), _type.elems(), ir::type_attr_t::mut);
        auto alloc_var = var(type, name);
        if (force_alloc || type.is_ptr()) {
            append(ir::alloc_t::make(alloc_var, {}));

            if (!value.is_empty()) {
                gpu_assert(to_cpp<int>(value) == 0);
                append(ir::funcs::zero_out(alloc_var, type.size()));
            };
        } else {
            append(ir::let_t::make(alloc_var, value, {}));
        }
        return alloc_var;
    }

    ir::expr_t def(const std::string &name, ir::expr_t value) {
        return def(value.type(), name, value);
    }

    tensor_t def(ir::v2::layout_t layout, const std::string &name,
            ir::expr_t value = {}) {
        auto t = ir::type_t(
                layout.type().kind(), layout.type().elems() * layout.elems());
        return {def(t, name, value, true), layout};
    }

    ir::expr_t let(ir::type_t type, const std::string &name, ir::expr_t value) {
        auto alloc_var = var(type, name);
        append(ir::let_t::make(alloc_var, value, {}));
        return alloc_var;
    }
    ir::expr_t let(const std::string &name, ir::expr_t value) {
        return let(value.type(), name, value);
    }

    void load_store(const tensor_t &t, const global_tensor_t &g,
            const plan2d_t &plan, op_kind_t op_kind,
            const ir::pvar_coord_t<int64_t> &base) {
        auto tensor_width = g.sizes[plan.dims[0]];
        auto tensor_height = g.sizes[plan.dims[1]];
        auto tensor_pitch = g.strides[plan.dims[1]];
        bool is_prefetch = t.buffer.is_empty();
        auto w_dim = plan.dims[0];
        auto h_dim = plan.dims[1];
        auto type = g.type;
        gpu_assert(is_prefetch || type == t.layout.type());

        if ((plan.min_alignment >= min_align_2d
                    && plan.min_pitch >= min_pitch_2d)
                && ((plan.transform == transform_t::none
                            && t.layout.int_dim_sizes()[w_dim] * type.size()
                                    <= grf_size)
                        || plan.transform == transform_t::block
                        || plan.transform == transform_t::vnni
                        || plan.transform == transform_t::transpose_vnni)) {
            auto tile = plan.get_2d_tile(type);
            ir::v2::for_each(
                    g.tile, tile, [&](const ir::pvar_coord_t<int64_t> &coord) {
                        auto buf = is_prefetch
                                ? ir::expr_t()
                                : t.buffer[t.layout.offset_in_bytes(
                                        base + coord)];
                        auto width = std::min(
                                tile[w_dim], g.tile[w_dim] - coord[w_dim]);
                        auto height = std::min(
                                tile[h_dim], g.tile[h_dim] - coord[h_dim]);
                        // TODO: Add logic to enable count for load operations
                        auto count = std::max(int64_t(1), tile[w_dim] / width);
                        auto width_idx = g.idxs[w_dim]
                                + static_cast<uint32_t>((base + coord)[w_dim]);
                        auto height_idx = g.idxs[h_dim]
                                + static_cast<uint32_t>((base + coord)[h_dim]);
                        auto send_kind = [&]() {
                            switch (op_kind) {
                                case op_kind_t::prefetch:
                                    return ir::send_op_t::prefetch_2d;
                                case op_kind_t::load:
                                    return ir::send_op_t::load_2d;
                                case op_kind_t::store:
                                    return ir::send_op_t::store_2d;
                            }
                        }();

                        auto send_func = ir::send_t::make_2d({}, send_kind,
                                type, tensor_width, tensor_height, tensor_pitch,
                                width, height, count,
                                plan.transform == transform_t::vnni,
                                plan.transform == transform_t::transpose_vnni,
                                /*zero_out=*/true, plan.cache_hint);

                        append(send_func.as<ir::send_t>()(g.buffer,
                                g.base_offset * type.size(), buf, {}, width_idx,
                                height_idx));
                    });
        } else if (plan.transform == transform_t::none
                || plan.transform == transform_t::block) {
            ir::pvar_tile_t tile = plan.get_block_tile(type);
            ir::v2::for_each(
                    g.tile, tile, [&](const ir::pvar_coord_t<int64_t> &coord) {
                        auto buf = is_prefetch
                                ? ir::expr_t()
                                : t.buffer[t.layout.offset_in_bytes(
                                        base + coord)];
                        auto width = std::min(
                                tile[w_dim], g.tile[w_dim] - coord[w_dim]);

                        auto width_bytes = width * type.size();
                        auto coord_local = coord;
                        while (width_bytes > 0) {
                            auto send_type = [&]() {
                                gpu_assert(width_bytes % 16 == 0);
                                auto load_width
                                        = dnnl::impl::utils::rnd_down_pow2(
                                                std::min(width_bytes,
                                                        (int64_t)512));
                                return ir::type_t::oword(load_width / 16);
                            }();
                            auto send_kind = [&]() {
                                switch (op_kind) {
                                    case op_kind_t::prefetch:
                                        return ir::send_op_t::prefetch;
                                    case op_kind_t::load:
                                        return ir::send_op_t::load;
                                    case op_kind_t::store:
                                        return ir::send_op_t::store;
                                }
                            }();

                            auto send_func = ir::send_t::make({}, send_kind,
                                    ir::send_address_t::a64, send_type, 1, true,
                                    true, plan.cache_hint);
                            append(send_func.as<ir::send_t>()(g.buffer,
                                    g.offset(base + coord_local), buf, {}));
                            width_bytes -= send_type.size();
                            coord_local[w_dim]
                                    += send_type.size() / type.size();
                        }
                    });
        } else {
            stub();
        }
    }

    void mma(const tensor_t &C, const tensor_t &A, const tensor_t &B,
            const ir::pvar_tile_t tile, ir::pvar_coord_t<int64_t> base,
            bool is_systolic) {
        if (is_systolic) {
            int64_t simd = 16;
            int64_t sdepth = 8;
            int64_t max_rcount = 8;

            auto dim_simd = C.layout.blocks()[0].dim;
            auto dim_sdepth
                    = A.layout.blocks()[0].dim == C.layout.blocks()[0].dim
                    ? A.layout.blocks()[1].dim
                    : A.layout.blocks()[0].dim;
            auto dim_rcount = C.layout.blocks()[1].dim;
            auto sdepth_pack = 4 / A.layout.type().size();
            auto rcount_pack = grf_size / (4 * sdepth);

            ir::pvar_tile_t inst_tile {{dim_simd, simd},
                    {dim_sdepth, sdepth * sdepth_pack},
                    {dim_rcount, max_rcount}};

            gpu_assert(tile[dim_simd] % simd == 0);
            gpu_assert(tile[dim_sdepth] % (sdepth_pack * sdepth) == 0);
            gpu_assert(tile[dim_rcount] % (rcount_pack) == 0);
            std::vector<ir::stmt_t> dpas_stmts;
            ir::v2::for_each(tile, inst_tile,
                    [&](const ir::pvar_coord_t<int64_t> &coord) {
                        auto simd = inst_tile[dim_simd];
                        auto sdepth = inst_tile[dim_sdepth] / sdepth_pack;
                        auto rcount
                                = std::min(inst_tile[dim_rcount],
                                          tile[dim_rcount] - coord[dim_rcount])
                                / rcount_pack;
                        auto dpas = ir::dpas_t::make(false, simd, sdepth,
                                rcount, C.layout.type(), B.layout.type(),
                                A.layout.type());
                        auto a_off = A.layout.offset_in_bytes(base + coord);
                        auto b_off = B.layout.offset_in_bytes(base + coord);
                        auto c_off = C.layout.offset_in_bytes(base + coord);
                        auto dst = C.buffer[c_off];
                        auto src1 = A.buffer[a_off];
                        auto src2 = B.buffer[b_off];
                        dpas_stmts.emplace_back(
                                dpas.as<ir::dpas_t>()(dst, dst, src1, src2));
                    });
            append(inject_dpas_atomic(ir::stmt_seq_t::make(dpas_stmts),
                    /*filter_by_label=*/false));
        } else {
            auto max_simd = 32;

            auto dim_simd = C.layout.blocks()[0].dim;
            auto dim_rcount = C.layout.blocks()[1].dim;
            auto dim_k = k_var; // Extract this

            ir::pvar_tile_t inst_tile {
                    {{dim_simd, max_simd}, {dim_rcount, 1}, {dim_k, 1}}};

            int M = inst_tile.get(m_var, 1);
            int N = inst_tile.get(n_var, 1);
            int K = inst_tile.get(k_var, 1);
            bool is_a_bcast = (M * K == 1);
            bool is_b_bcast = (K * N == 1);
            int a_stride = is_a_bcast ? 0 : to_cpp<int>(A.layout.stride(m_var));
            int b_stride = is_b_bcast ? 0 : to_cpp<int>(B.layout.stride(n_var));

            gpu_assert(tile[dim_simd] * C.layout.type().size() % grf_size == 0);
            ir::v2::for_each(tile, inst_tile,
                    [&](const ir::pvar_coord_t<int64_t> &coord) {
                        auto simd = std::min(inst_tile[dim_simd],
                                tile[dim_simd] - coord[dim_simd]);

                        auto mad = ir::mad_t::make(ir::hw_t(), C.layout.type(),
                                simd, A.layout.type(), a_stride,
                                B.layout.type(), b_stride);

                        auto a_off = A.layout.offset_in_bytes(base + coord);
                        auto b_off = B.layout.offset_in_bytes(base + coord);
                        auto c_off = C.layout.offset_in_bytes(base + coord);
                        auto dst = C.buffer[c_off];
                        auto src1 = A.buffer[a_off];
                        auto src2 = B.buffer[b_off];

                        append(mad.as<ir::mad_t>()(dst, dst, src1, src2));
                    });
        }
    }

    void prefetch(const global_tensor_t &g, const plan2d_t &plan,
            const ir::pvar_coord_t<int64_t> &base) {
        load_store({}, g, plan, op_kind_t::prefetch, base);
    }

    void load(const tensor_t &t, const global_tensor_t &g, const plan2d_t &plan,
            const ir::pvar_coord_t<int64_t> &base) {
        load_store(t, g, plan, op_kind_t::load, base);
    }

    void store(const global_tensor_t &g, const tensor_t &t,
            const plan2d_t &plan, const ir::pvar_coord_t<int64_t> &base) {
        load_store(t, g, plan, op_kind_t::store, base);
    }
    void assume(ir::expr_t e) { ctx_->add_constraint(e); }

    template <typename F>
    void if_(ir::expr_t cond, F if_body) {
        begin_scope();
        if_body();
        auto if_body_stmt = pop_scope();
        append(ir::if_t::make(cond, if_body_stmt));
    }

    template <typename F, typename G>
    void if_(ir::expr_t cond, F if_body, G else_body) {
        begin_scope();
        if_body();
        auto if_body_stmt = pop_scope();

        begin_scope();
        else_body();
        auto else_body_stmt = pop_scope();

        append(ir::if_t::make(cond, if_body_stmt, else_body_stmt));
    }

    template <typename F>
    void while_(ir::expr_t cond, F body) {
        begin_scope();
        body();
        auto body_stmt = pop_scope();
        append(ir::while_t::make(cond, body_stmt));
    }

    void assign(ir::expr_t var, ir::expr_t value) {
        append(ir::store_t::make(var, 0, value));
    }

    void begin_scope() { stmts_stack.push({}); }

    void end_scope() {
        auto stmt = pop_scope();
        gpu_assert(!stmts_stack.empty());
        append(stmt);
    }

private:
    ir::expr_t var(ir::type_t type, const std::string &name) {
        return ir::var_t::make(type, name);
    }

    ir::stmt_t pop_scope() {
        auto stmt = to_stmt();
        stmts_stack.pop();
        return stmt;
    }

    ir::stmt_t to_stmt() {
        ir::stmt_t stmt;
        size_t size = stmts().size();
        size_t end = size;
        size_t begin = size - 1;
        while (begin < end) {
            auto &s = stmts()[begin];
            if (s.is<ir::alloc_t>() || s.is<ir::let_t>()) {
                ir::stmt_t body = [&]() {
                    if (begin + 1 >= end) return stmt;
                    auto seq = std::vector<ir::stmt_t>(
                            stmts().begin() + begin + 1, stmts().begin() + end);
                    seq.push_back(stmt);
                    return ir::stmt_seq_t::make(seq);
                }();
                end = begin;

                if (s.is<ir::alloc_t>()
                        && s.as<ir::alloc_t>().body.is_empty()) {
                    auto &a = s.as<ir::alloc_t>();
                    if (a.buf.type().is_ptr())
                        stmt = ir::alloc_t::make(
                                a.buf, a.size, a.kind, a.attrs, body);
                    else
                        stmt = ir::alloc_t::make(a.buf, body);
                } else if (s.is<ir::let_t>()
                        && s.as<ir::let_t>().body.is_empty()) {
                    auto &l = s.as<ir::let_t>();
                    stmt = ir::let_t::make(l.var, l.value, body);
                }
            }
            begin--;
        }

        if (end > 0) {
            std::vector<ir::stmt_t> seq(stmts().begin(), stmts().begin() + end);
            seq.push_back(stmt);
            stmt = ir::stmt_seq_t::make(seq);
        }
        return stmt;
    }

    void append(ir::stmt_t stmt) { stmts().emplace_back(stmt); }

    std::vector<ir::stmt_t> &stmts() { return stmts_stack.top(); }
    std::stack<std::vector<ir::stmt_t>> stmts_stack;
    ir::kernel_iface_t interface_;
    ir::ir_context_t *ctx_ = nullptr;
    std::array<ir::expr_t, 3> group_ids_;
    std::array<ir::expr_t, 3> local_ids_;
    std::array<ir::expr_t, 3> local_sizes_;
};

dsl_ctx_t &dsl_ctx() {
    static thread_local dsl_ctx_t ctx;
    return ctx;
}

void declare_kernel(
        const ir::kernel_iface_t &interface, ir::ir_context_t &ctx) {
    dsl_ctx().declare_kernel(interface, ctx);
}
ir::stmt_t end_kernel() {
    return dsl_ctx().end_kernel();
}
const std::array<ir::expr_t, 3> &group_ids() {
    return dsl_ctx().group_ids();
}
ir::expr_t group_id(int idx) {
    return dsl_ctx().group_id(idx);
}
const std::array<ir::expr_t, 3> &local_ids() {
    return dsl_ctx().local_ids();
}
ir::expr_t local_id(int idx) {
    return dsl_ctx().local_id(idx);
}
const std::array<ir::expr_t, 3> &local_sizes() {
    return dsl_ctx().local_sizes();
}
ir::expr_t local_size(int idx) {
    return dsl_ctx().local_size(idx);
}
ir::expr_t arg(const std::string &name) {
    return dsl_ctx().arg(name);
}
void assume(ir::expr_t e) {
    dsl_ctx().assume(e);
}
ir::expr_t def(ir::type_t type, const std::string &name, ir::expr_t value = {},
        bool force_alloc = false) {
    return dsl_ctx().def(type, name, value, force_alloc);
}
ir::expr_t def(const std::string &name, ir::expr_t value) {
    return def(value.type(), name, value);
}
tensor_t def(ir::v2::layout_t layout, const std::string &name,
        ir::expr_t value = {}) {
    return dsl_ctx().def(layout, name, value);
}
ir::expr_t let(ir::type_t type, const std::string &name, ir::expr_t value) {
    return dsl_ctx().let(type, name, value);
}
ir::expr_t let(const std::string &name, ir::expr_t value) {
    return dsl_ctx().let(name, value);
}
void load_store(const tensor_t &t, const global_tensor_t &g,
        const plan2d_t &plan, op_kind_t op_kind,
        const ir::pvar_coord_t<int64_t> &base) {
    dsl_ctx().load_store(t, g, plan, op_kind, base);
}
void prefetch(const global_tensor_t &g, const plan2d_t &plan,
        const ir::pvar_coord_t<int64_t> &base) {
    dsl_ctx().prefetch(g, plan, base);
}
void load(const tensor_t &t, const global_tensor_t &g, const plan2d_t &plan,
        const ir::pvar_coord_t<int64_t> &base) {
    dsl_ctx().load(t, g, plan, base);
}
void store(const global_tensor_t &g, const tensor_t &t, const plan2d_t &plan,
        const ir::pvar_coord_t<int64_t> &base) {
    dsl_ctx().store(g, t, plan, base);
}
void mma(const tensor_t &C, const tensor_t &A, const tensor_t &B,
        const ir::pvar_tile_t tile, ir::pvar_coord_t<int64_t> base,
        bool is_systolic) {
    dsl_ctx().mma(C, A, B, tile, base, is_systolic);
}

template <typename F>
void if_(ir::expr_t cond, F if_body) {
    dsl_ctx().if_(cond, if_body);
}

template <typename F, typename G>
void if_(ir::expr_t cond, F if_body, G else_body) {
    dsl_ctx().if_(cond, if_body, else_body);
}

template <typename F>
void while_(ir::expr_t cond, F body) {
    dsl_ctx().while_(cond, body);
}

void assign(ir::expr_t var, ir::expr_t value) {
    dsl_ctx().assign(var, value);
}
void begin_scope() {
    dsl_ctx().begin_scope();
}
void end_scope() {
    dsl_ctx().end_scope();
}

const char *to_str(AccessType t) {
    switch (t) {
        case AccessType::Scattered: return "Scattered";
        case AccessType::ChannelScattered: return "ChannelScattered";
        case AccessType::Block: return "Block";
        case AccessType::PseudoBlock: return "PseudoBlock";
        case AccessType::Block2D: return "Block2D";
        case AccessType::Block2DTranspose: return "Block2DTranspose";
        case AccessType::Block2DVNNI: return "Block2DVNNI";
        case AccessType::CacheLine: return "CacheLine";
    }
}

plan2d_t get_plan(const MatrixAddressing &matrix_type,
        const MatrixAddressingStrategy &matrix_strategy,
        std::array<ir::pvar_t, 2> dims, bool is_prefetch = false) {
    switch (matrix_strategy.accessType) {
        case AccessType::Scattered:
            // TODO: Remove workaround unimplemented scattered->vnni support.
            if (is_prefetch)
                return plan2d_t(transform_t::none, 0, matrix_type.alignment, 0,
                        matrix_strategy.cachingR, dims);

            return plan2d_t(transform_t::transpose_vnni, matrix_strategy.tileR,
                    matrix_type.alignment, 0, matrix_strategy.cachingR, dims);

        case AccessType::ChannelScattered: stub(); return {};
        case AccessType::Block2DTranspose:
            return plan2d_t(transform_t::transpose_vnni, matrix_strategy.tileR,
                    matrix_type.alignment, min_pitch_2d,
                    matrix_strategy.cachingR, dims);
        case AccessType::Block:
        case AccessType::PseudoBlock:
            return plan2d_t(transform_t::none, matrix_strategy.tileR,
                    matrix_type.alignment, 0, matrix_strategy.cachingR, dims);
        case AccessType::Block2D: {
            return plan2d_t(transform_t::block, matrix_strategy.tileR,
                    matrix_type.alignment, min_pitch_2d,
                    matrix_strategy.cachingR, dims);
        };
        case AccessType::Block2DVNNI: {
            return plan2d_t(transform_t::vnni, matrix_strategy.tileR,
                    matrix_type.alignment, min_pitch_2d,
                    matrix_strategy.cachingR, dims);
        }
        default: stub(); return {};
    }
};
ir::pvar_map_t<ir::expr_t> get_strides(
        MatrixLayout layout, std::array<ir::pvar_t, 2> pvars, ir::expr_t ld) {
    switch (layout) {
        case MatrixLayout::N: return {{pvars[0], 1}, {pvars[1], ld}};
        case MatrixLayout::T: return {{pvars[0], ld}, {pvars[1], 1}};
        default: stub(); return {};
    };
}

// Basic iterator with no iteration over m and n.
struct basic_iterator_t : kloop_iterator_t {
    basic_iterator_t(ir::expr_t m, ir::expr_t n, ir::expr_t k, int m_blk,
            int n_blk, int k_blk, ir::expr_t A_buffer, ir::expr_t A_offset,
            ir::type_t A_type, ir::pvar_map_t<ir::expr_t> A_strides,
            int A_prefetch_copies, int A_load_copies, ir::expr_t B_buffer,
            ir::expr_t B_offset, ir::type_t B_type,
            ir::pvar_map_t<ir::expr_t> B_strides, int B_prefetch_copies,
            int B_load_copies, ir::expr_t C_buffer, ir::expr_t C_offset,
            ir::type_t C_type, ir::pvar_map_t<ir::expr_t> C_strides,
            const std::array<ir::expr_t, 3> &group_ids,
            const std::array<ir::expr_t, 3> &local_ids,
            const std::array<ir::expr_t, 3> &local_sizes)
        : m_idx_ {let("m_idx",
                (group_ids[0] * local_sizes[0] + local_ids[0]) * m_blk)}
        , m_(m)
        , n_idx_ {let("n_idx",
                  (group_ids[1] * local_sizes[1] + local_ids[1]) * n_blk)}
        , n_(n)
        , k_idx_ {def("k_idx", 0u)}
        , k_load_idx_ {k_idx_}
        , k_prefetch_idx_A_ {k_idx_}
        , k_prefetch_idx_B_ {k_idx_}
        , k_ {k}
        , A_prefetch_ {A_buffer, A_type, A_offset,
                  {{m_var, m_idx_}, {k_var, k_prefetch_idx_A_}}, A_strides,
                  {{m_var, m}, {k_var, k}},
                  {{m_var, m_blk}, {k_var, k_blk / A_prefetch_copies}}}
        , A_load_ {A_buffer, A_type, A_offset,
                  {{m_var, m_idx_}, {k_var, k_load_idx_}}, A_strides,
                  {{m_var, m}, {k_var, k}},
                  {{m_var, m_blk}, {k_var, k_blk / A_load_copies}}}
        , B_prefetch_ {B_buffer, B_type, B_offset,
                  {{k_var, k_prefetch_idx_B_}, {n_var, n_idx_}}, B_strides,
                  {{k_var, k}, {n_var, n}},
                  {{k_var, k_blk / B_prefetch_copies}, {n_var, n_blk}}}
        , B_load_ {B_buffer, B_type, B_offset,
                  {{k_var, k_load_idx_}, {n_var, n_idx_}}, B_strides,
                  {{k_var, k}, {n_var, n}},
                  {{k_var, k_blk / B_load_copies}, {n_var, n_blk}}}

        , C_store_ {C_buffer, C_type, C_offset,
                  {{m_var, m_idx_}, {n_var, n_idx_}}, C_strides,
                  {{m_var, m}, {n_var, n}}, {{m_var, m_blk}, {n_var, n_blk}}}

    {
        assume(m_idx_ % m_blk == 0);
        assume(n_idx_ % n_blk == 0);
    }

    const global_tensor_t &A_prefetch() const override { return A_prefetch_; }
    const global_tensor_t &A_load() const override { return A_load_; }
    const global_tensor_t &B_prefetch() const override { return B_prefetch_; }
    const global_tensor_t &B_load() const override { return B_load_; }
    const global_tensor_t &C_store() const override { return C_store_; }

    void inc_prefetch_A(int k_block) override {
        k_prefetch_idx_A_ = ir::simplify(k_prefetch_idx_A_ + k_block);
        A_prefetch_.idxs[k_var] = k_prefetch_idx_A_;
    }

    void inc_prefetch_B(int k_block) override {
        k_prefetch_idx_B_ = ir::simplify(k_prefetch_idx_B_ + k_block);
        B_prefetch_.idxs[k_var] = k_prefetch_idx_B_;
    }

    void inc_load(int k_block) override {
        k_load_idx_ = ir::simplify(k_load_idx_ + k_block);
        A_load_.idxs[k_var] = k_load_idx_;
        B_load_.idxs[k_var] = k_load_idx_;
    }

    void inc_mma(int k_block) override {
        // Prefetch/load computation is relative to k_idx
        inc_prefetch_A(-k_block);
        inc_prefetch_B(-k_block);
        inc_load(-k_block);

        assign(k_idx_, k_idx_ + k_block);
    }

    ir::expr_t update_C() const override { return false; }

    ir::expr_t is_inbounds(
            int m_block, int n_block, int k_block) const override {
        int max_offset = std::max(
                {to_cpp<int>(ir::simplify(k_prefetch_idx_A_ - k_idx_)),
                        to_cpp<int>(ir::simplify(k_prefetch_idx_B_ - k_idx_)),
                        to_cpp<int>(ir::simplify(k_load_idx_ - k_idx_))});

        return ir::simplify(m_idx_ <= m_ - m_block & n_idx_ <= n_ - n_block
                & k_idx_ <= k_ - k_block - max_offset);
    }

private:
    static ir::expr_t offset(const ir::pvar_map_t<ir::expr_t> &idxs,
            const ir::pvar_map_t<ir::expr_t> &strides,
            const ir::pvar_coord_t<int64_t> &coord) {
        ir::expr_t ret = 0;
        for (auto &c : coord) {
            ret += (idxs[c] + coord[c]) * strides[c];
        }
        return ir::simplify(ret);
    }

    ir::expr_t m_idx_;
    ir::expr_t m_;
    ir::expr_t n_idx_;
    ir::expr_t n_;
    ir::expr_t k_idx_;
    ir::expr_t k_load_idx_;
    ir::expr_t k_prefetch_idx_A_;
    ir::expr_t k_prefetch_idx_B_;
    ir::expr_t k_;

    global_tensor_t A_prefetch_;
    global_tensor_t A_load_;
    global_tensor_t B_prefetch_;
    global_tensor_t B_load_;
    global_tensor_t C_store_;
};

struct gemm_ir {
    gemm_ir(const gemm_ir_desc_t &desc)
        : problem(desc.problem), strategy(desc.strategy) {}

    ir::stmt_t build(ir::kernel_iface_t iface, ir::ir_context_t &ctx) {
        declare_kernel(iface, ctx);
        printf("Here\n");

        const auto m = arg("m");
        const auto n = arg("n");
        const auto k = arg("k");

        auto m_blk = strategy.unroll[LoopM];
        auto n_blk = strategy.unroll[LoopN];
        auto k_blk = strategy.unroll[LoopK];
        std::cout << "(m_blk, n_blk, k_blk): (" << m_blk << " " << n_blk << " "
                  << k_blk << ")\n";

        // Pipeline size
        int p_size = lcm(strategy.A_copies, strategy.B_copies);
        auto k_unroll_blk = k_blk / p_size;

        std::array<ir::pvar_t, 2> A_vars = {m_var, k_var};
        std::array<ir::pvar_t, 2> B_vars = {k_var, n_var};
        std::array<ir::pvar_t, 2> C_vars = {m_var, n_var};

        int A_pf_copies
                = strategy.ka_pfStride ? k_blk / strategy.ka_pfStride : 1;
        int B_pf_copies
                = strategy.kb_pfStride ? k_blk / strategy.kb_pfStride : 1;

        basic_iterator_t kloop_it(m, n, k, m_blk, n_blk, k_blk, arg("A"),
                arg("offset_A"), into_ir(problem.Ta_ext),
                get_strides(problem.A.layout, A_vars, arg("lda")), A_pf_copies,
                strategy.A_copies, arg("B"), arg("offset_B"),
                into_ir(problem.Tb_ext),
                get_strides(problem.B.layout, B_vars, arg("ldb")), B_pf_copies,
                strategy.B_copies, arg("C"), arg("offset_C"),
                into_ir(problem.Tc_ext),
                get_strides(problem.C.layout, C_vars, arg("ldc")), group_ids(),
                local_ids(), local_sizes());

        gpu_assert(problem.Ta == problem.Ta_ext);
        auto A_prefetch_plan
                = get_plan(problem.A, strategy.A_prefetch, A_vars, true);
        auto A_load_plan = get_plan(problem.A, strategy.A, A_vars);

        gpu_assert(problem.Tb == problem.Tb_ext);
        auto B_prefetch_plan
                = get_plan(problem.B, strategy.B_prefetch, B_vars, true);
        auto B_load_plan = get_plan(problem.B, strategy.B, B_vars);

        ir::pvar_tile_t C_dims {{{m_var, m_blk}, {n_var, n_blk}}};
        auto C_store_plan = get_plan(problem.C, strategy.C, C_vars);

        tensor_t C = def(C_store_plan.get_layout(
                                 C_dims, into_ir(problem.Tc), gemm_var_desc),
                "C_blk", 0);
        auto store_C
                = [&]() { store(kloop_it.C_store(), C, C_store_plan, {}); };

        k_loop_config_t k_loop_main {m_blk, n_blk, k_blk, k_unroll_blk,
                k_blk - k_unroll_blk, strategy.ka_prefetch,
                strategy.kb_prefetch, kloop_it, A_prefetch_plan, A_load_plan,
                B_prefetch_plan, B_load_plan, C_store_plan, C};

        k_loop_config_t k_loop_short {m_blk, n_blk, k_unroll_blk, k_unroll_blk,
                0, 0, 0, kloop_it, A_prefetch_plan, A_load_plan,
                B_prefetch_plan, B_load_plan, C_store_plan, C};
        if_(kloop_it.is_inbounds(1, 1, 1), [&]() {
            if_(
                    k >= k_loop_main.warmup_k(),
                    [&]() { build_k_loop(k_loop_main); },
                    [&]() { build_k_loop(k_loop_short); });
            store_C();
        });

        return end_kernel();
    }

    struct k_loop_config_t {
        int m_blk;
        int n_blk;
        int k_blk;
        int k_unroll_blk;
        int k_load; // Offset to loads
        int ka_prefetch; // Offset to A prefetch
        int kb_prefetch; // Offset to B prefetch
        basic_iterator_t kloop_it;
        plan2d_t A_prefetch_plan;
        plan2d_t A_load_plan;
        plan2d_t B_prefetch_plan;
        plan2d_t B_load_plan;
        plan2d_t C_store_plan;
        tensor_t C;

        int warmup_k() const {
            return std::max({k_load, ka_prefetch, kb_prefetch});
        }
        int tail_k() const {
            return (warmup_k() / k_blk) * k_blk + ((-warmup_k()) % k_blk);
        }
    };

    void build_k_loop(const k_loop_config_t &cfg) {
        auto m_blk = cfg.m_blk;
        auto n_blk = cfg.n_blk;
        auto k_blk = cfg.k_blk;
        auto k_unroll_blk = cfg.k_unroll_blk;
        auto kloop_it = cfg.kloop_it;
        auto &C = cfg.C;

        ir::pvar_tile_t A_dims {{{m_var, cfg.m_blk}, {k_var, cfg.k_blk}}};
        ir::pvar_tile_t B_dims {{{k_var, cfg.k_blk}, {n_var, cfg.n_blk}}};
        tensor_t A = def(cfg.A_load_plan.get_layout(A_dims,
                                 into_ir(problem.Ta_ext), gemm_var_desc),
                "A_blk");
        tensor_t B = def(cfg.B_load_plan.get_layout(B_dims,
                                 into_ir(problem.Tb_ext), gemm_var_desc),
                "B_blk");

        std::cout << "A layout " << A.layout.str() << "\n";
        std::cout << "B layout " << B.layout.str() << "\n";
        std::cout << "C layout " << C.layout.str() << "\n";

        auto prefetch_A = [&](int k_unroll_idx) {
            if (k_unroll_idx % kloop_it.A_prefetch().tile[k_var] != 0) return;
            prefetch(kloop_it.A_prefetch(), cfg.A_prefetch_plan,
                    {{k_var, k_unroll_idx}});
        };
        auto load_A = [&](int k_unroll_idx) {
            if (k_unroll_idx % kloop_it.A_load().tile[k_var] != 0) return;
            load(A, kloop_it.A_load(), cfg.A_load_plan,
                    {{k_var, k_unroll_idx}});
        };

        auto prefetch_B = [&](int k_unroll_idx) {
            if (k_unroll_idx % kloop_it.B_prefetch().tile[k_var] != 0) return;
            prefetch(kloop_it.B_prefetch(), cfg.B_prefetch_plan,
                    {{k_var, k_unroll_idx}});
        };
        auto load_B = [&](int k_unroll_idx) {
            if (k_unroll_idx % kloop_it.B_load().tile[k_var] != 0) return;
            load(B, kloop_it.B_load(), cfg.B_load_plan,
                    {{k_var, k_unroll_idx}});
        };

        auto store_C
                = [&]() { store(kloop_it.C_store(), C, cfg.C_store_plan, {}); };

        auto k_body = [&](int k_offset, bool do_prefetch_A, bool do_prefetch_B,
                              bool do_load, bool do_mma, int k_inc) {
            if (do_prefetch_A) {
                prefetch_A(k_offset);
                if (k_inc) kloop_it.inc_prefetch_A(k_inc);
            }

            if (do_prefetch_B) {
                prefetch_B(k_offset);
                if (k_inc) kloop_it.inc_prefetch_B(k_inc);
            }

            if (do_load) {
                load_A(k_offset);
                load_B(k_offset);
                if (k_inc) kloop_it.inc_load(k_inc);
            }

            if (do_mma) {
                ir::pvar_tile_t tile = C.layout.int_dim_sizes();
                tile[k_var] = k_unroll_blk;
                mma(C, A, B, tile, {{k_var, k_offset}}, strategy.systolic);
                if (k_inc) kloop_it.inc_mma(k_inc);
            }
        };

        std::cout << "k_load = " << cfg.k_load << "\n";
        std::cout << "ka_prefetch = " << cfg.ka_prefetch << "\n";
        std::cout << "kb_prefetch = " << cfg.kb_prefetch << "\n";

        auto warmup_k = cfg.warmup_k();
        auto tail_k = cfg.tail_k();

        std::cout << "warmup: " << warmup_k << "\n";
        for (int k_unroll_idx = 0; k_unroll_idx < warmup_k;
                k_unroll_idx += k_unroll_blk) {
            bool last_blk = k_unroll_idx >= warmup_k - k_unroll_blk;
            k_body(k_unroll_idx, k_unroll_idx + cfg.ka_prefetch >= warmup_k,
                    k_unroll_idx + cfg.kb_prefetch >= warmup_k,
                    k_unroll_idx + cfg.k_load >= warmup_k, false,
                    last_blk ? cfg.warmup_k() - cfg.k_load : 0);
        }

        std::cout << "k-loop\n";
        while_(kloop_it.is_inbounds(m_blk, n_blk, k_blk), [&]() {
            for (int k_unroll_idx = 0; k_unroll_idx < k_blk;
                    k_unroll_idx += k_unroll_blk) {
                bool last_blk = k_unroll_idx >= k_blk - k_unroll_blk;
                k_body(k_unroll_idx, cfg.ka_prefetch, cfg.kb_prefetch, true,
                        true, last_blk ? k_blk : 0);
            }
            if (!is_const(kloop_it.update_C())
                    || to_cpp<bool>(kloop_it.update_C()))
                if_(kloop_it.update_C(), [&]() { store_C(); });
        });

        if_(kloop_it.is_inbounds(m_blk, n_blk, 1), [&]() {
            for (int k_unroll_idx = 0; k_unroll_idx < k_blk;
                    k_unroll_idx += k_unroll_blk) {
                bool last_blk = k_unroll_idx >= k_blk - k_unroll_blk;
                k_body(k_unroll_idx, cfg.ka_prefetch, cfg.kb_prefetch, true,
                        true, last_blk ? k_blk : 0);
            }
            if (!is_const(kloop_it.update_C())
                    || to_cpp<bool>(kloop_it.update_C()))
                if_(kloop_it.update_C(), [&]() { store_C(); });
        });

        std::cout << "cool down: " << tail_k << "\n";

        for (int k_unroll_idx = 0; k_unroll_idx < tail_k;
                k_unroll_idx += k_unroll_blk) {
            k_body(k_unroll_idx, k_unroll_idx + cfg.ka_prefetch < tail_k,
                    k_unroll_idx + cfg.kb_prefetch < tail_k,
                    k_unroll_idx + cfg.k_load < tail_k, true, 0);
        }
    }

    const GEMMProblem &problem;
    const GEMMStrategy &strategy;
    const ir::v2::layout_desc_t gemm_var_desc {
            {{m_var, 'm'}, {n_var, 'n'}, {k_var, 'k'}}};
};

ir::stmt_t build_ir(const gemm_ir_desc_t &desc, ir::constraint_set_t cset) {
    ir::ir_context_t ctx(desc.compile_ctx().exec_config(), cset);

    auto stmt = gemm_ir(desc).build(desc.kernel_iface(), ctx);
    stmt = ir::simplify(stmt, ctx);
    stmt = ir::inject_send(stmt, ctx);

    // TODO: This should be unnecessary as it could happen at codegen
    stmt = ir::fixup_if_conditions(stmt, ctx);
    stmt = ir::eliminate_common_subexprs(
            stmt, ctx, desc.strategy.GRFs * grf_size);
    return stmt;
}

} // namespace gemmstone
