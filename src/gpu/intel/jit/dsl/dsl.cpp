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

#include "gpu/intel/jit/dsl/dsl.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace dsl {

struct ctx_t {
    void declare_kernel(const kernel_iface_t &interface, ir_context_t &ctx) {
        gpu_assert(stmts_stack.empty())
                << "Invalid generation of a kernel within a kernel";
        interface_ = interface;
        ctx_ = &ctx;

        begin_scope();

        for (int i = 0; i < interface.nargs(); i++) {
            const auto &var = interface.arg_var(i);
            if (var.type().is_ptr()) {
                if (var.type().is_slm()) {
                    append(alloc_t::make(var, 0, alloc_kind_t::slm, stmt_t {}));
                } else {
                    append(alloc_t::make(
                            var, 0, alloc_kind_t::global, stmt_t {}));
                }
            } else {
                append(let_t::make(var, {}, {}));
            }
        }

        for (int i = 0; i < 3; i++) {
            group_ids_[i] = let(type_t::u32(), ir_builder_t::tg_idx(i), {});
            local_ids_[i] = let(type_t::u16(), ir_builder_t::local_id(i), {});
            local_sizes_[i]
                    = let(type_t::u16(), ir_builder_t::local_size(i), {});
        }
    }

    stmt_t end_kernel() {
        gpu_assert(stmts_stack.size() == 1)
                << "Invalid end of kernel, imbalanced scopes detected";
        ctx_ = nullptr;
        return pop_scope();
    }

    const std::array<expr_t, 3> &group_ids() const { return group_ids_; }
    expr_t group_id(int idx) const { return group_ids_[idx]; }
    const std::array<expr_t, 3> &local_ids() const { return local_ids_; }
    expr_t local_id(int idx) const { return local_ids_[idx]; }
    const std::array<expr_t, 3> &local_sizes() const { return local_sizes_; }
    expr_t local_size(int idx) const { return local_sizes_[idx]; }

    expr_t arg(const std::string &name) {
        auto a = interface_.find_arg(name);
        expr_t value;
        if (ctx_->cset().is_single_value(a, value)) { return value; }
        return a;
    }

    // TODO: Remove IR restriction which requires force_alloc
    expr_t def(type_t _type, const std::string &name, expr_t value = {},
            bool force_alloc = false) {
        auto type = type_t(_type.kind(), _type.elems(), type_attr_t::mut);
        auto alloc_var = var(type, name);
        if (force_alloc || type.is_ptr()) {
            append(alloc_t::make(alloc_var, {}));

            if (!value.is_empty()) {
                gpu_assert(to_cpp<int>(value) == 0);
                append(funcs::zero_out(alloc_var, type.size()));
            };
        } else {
            append(let_t::make(alloc_var, value, {}));
        }
        return alloc_var;
    }

    expr_t def(const std::string &name, expr_t value) {
        return def(value.type(), name, value);
    }

    tensor_t def(
            v2::layout_t layout, const std::string &name, expr_t value = {}) {
        auto t = type_t(
                layout.type().kind(), layout.type().elems() * layout.elems());
        return {def(t, name, value, true), layout};
    }

    expr_t let(type_t type, const std::string &name, expr_t value) {
        auto alloc_var = var(type, name);
        append(let_t::make(alloc_var, value, {}));
        return alloc_var;
    }
    expr_t let(const std::string &name, expr_t value) {
        return let(value.type(), name, value);
    }

    enum class load_store_kind_t { load, prefetch, store };

    void scatter_message(const tensor_t &t, const global_tensor_t &g,
            const transform_t &plan, load_store_kind_t &op_kind,
            const icoord_t &base) {
        gpu_warning() << "Scatter messages are not yet implemented\n";
    };

    void block_message(const tensor_t &t, const global_tensor_t &g,
            const transform_t &plan, load_store_kind_t &op_kind,
            const icoord_t &base) {
        auto tensor_width = g.sizes[plan.dims[0]];
        auto tensor_height = g.sizes[plan.dims[1]];
        auto tensor_pitch = g.strides[plan.dims[1]];
        bool is_prefetch = t.buffer.is_empty();
        auto w_dim = plan.dims[0];
        auto h_dim = plan.dims[1];
        auto type = g.type;

        tile_t tile = plan.get_block_tile(type);
        v2::for_each(g.tile, tile, [&](const icoord_t &coord) {
            auto buf = is_prefetch
                    ? expr_t()
                    : t.buffer[t.layout.offset_in_bytes(base + coord)];
            auto width = std::min(tile[w_dim], g.tile[w_dim] - coord[w_dim]);

            auto width_bytes = width * type.size();
            auto coord_local = coord;
            while (width_bytes > 0) {
                auto send_type = [&]() {
                    gpu_assert(width_bytes % 16 == 0);
                    auto load_width = dnnl::impl::utils::rnd_down_pow2(
                            std::min(width_bytes, (int64_t)512));
                    return type_t::oword(load_width / 16);
                }();
                auto send_kind = [&]() {
                    switch (op_kind) {
                        case load_store_kind_t::prefetch:
                            return send_op_t::prefetch;
                        case load_store_kind_t::load: return send_op_t::load;
                        case load_store_kind_t::store: return send_op_t::store;
                    }
                }();

                auto send_func
                        = send_t::make({}, send_kind, send_address_t::a64,
                                send_type, 1, true, true, plan.cache_hint);
                append(send_func.as<send_t>()(
                        g.buffer, g.offset(base + coord_local), buf, {}));
                width_bytes -= send_type.size();
                coord_local[w_dim] += send_type.size() / type.size();
            }
        });
    }

    void block_2d_message(const tensor_t &t, const global_tensor_t &g,
            const transform_t &plan, load_store_kind_t op_kind,
            const icoord_t &base) {
        auto tensor_width = g.sizes[plan.dims[0]];
        auto tensor_height = g.sizes[plan.dims[1]];
        auto tensor_pitch = g.strides[plan.dims[1]];
        bool is_prefetch = t.buffer.is_empty();
        auto w_dim = plan.dims[0];
        auto h_dim = plan.dims[1];
        auto type = g.type;
        auto tile = plan.get_2d_tile(type);

        v2::for_each(g.tile, tile, [&](const icoord_t &coord) {
            auto buf = is_prefetch
                    ? expr_t()
                    : t.buffer[t.layout.offset_in_bytes(base + coord)];
            auto width = std::min(tile[w_dim], g.tile[w_dim] - coord[w_dim]);
            auto height = std::min(tile[h_dim], g.tile[h_dim] - coord[h_dim]);
            // TODO: Add logic to enable count for load operations
            auto count = std::max(int64_t(1), tile[w_dim] / width);
            auto width_idx = g.idxs[w_dim]
                    + static_cast<uint32_t>((base + coord)[w_dim]);
            auto height_idx = g.idxs[h_dim]
                    + static_cast<uint32_t>((base + coord)[h_dim]);
            auto send_kind = [&]() {
                switch (op_kind) {
                    case load_store_kind_t::prefetch:
                        return send_op_t::prefetch_2d;
                    case load_store_kind_t::load: return send_op_t::load_2d;
                    case load_store_kind_t::store: return send_op_t::store_2d;
                }
            }();

            auto send_func = send_t::make_2d({}, send_kind, type, tensor_width,
                    tensor_height, tensor_pitch, width, height, count,
                    plan.transform == transform_t::kind_t::vnni,
                    plan.transform == transform_t::kind_t::transpose_vnni,
                    /*zero_out=*/true, plan.cache_hint);

            append(send_func.as<send_t>()(g.buffer, g.base_offset * type.size(),
                    buf, {}, width_idx, height_idx));
        });
    }

    void load_store(const tensor_t &t, const global_tensor_t &g,
            const transform_t &plan, load_store_kind_t op_kind,
            const icoord_t &base) {
        auto tensor_width = g.sizes[plan.dims[0]];
        auto tensor_height = g.sizes[plan.dims[1]];
        auto tensor_pitch = g.strides[plan.dims[1]];
        bool is_prefetch = t.buffer.is_empty();
        auto w_dim = plan.dims[0];
        auto h_dim = plan.dims[1];
        auto type = g.type;
        gpu_assert(is_prefetch || type == t.layout.type());
        if ((plan.transform == transform_t::kind_t::none
                    && t.layout.int_dim_sizes()[w_dim] * type.size()
                            <= grf_size)
                || plan.transform == transform_t::kind_t::block
                || plan.transform == transform_t::kind_t::vnni
                || plan.transform == transform_t::kind_t::transpose_vnni) {
            if_(
                    tensor_pitch % (min_align_2d / type.size()) == 0
                            & tensor_pitch >= (min_pitch_2d / type.size()),
                    [&]() { block_2d_message(t, g, plan, op_kind, base); },
                    [&]() {
                        if (plan.transform == transform_t::kind_t::block)
                            block_message(t, g, plan, op_kind, base);
                        else
                            scatter_message(t, g, plan, op_kind, base);
                    });
        } else if (plan.transform == transform_t::kind_t::none
                || plan.transform == transform_t::kind_t::block) {
            block_message(t, g, plan, op_kind, base);
        } else {
            scatter_message(t, g, plan, op_kind, base);
        }
    }

    void mma(const tensor_t &C, const tensor_t &A, const tensor_t &B,
            const tile_t tile, const icoord_t &base, bool is_systolic) {
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

            tile_t inst_tile {{dim_simd, simd},
                    {dim_sdepth, sdepth * sdepth_pack},
                    {dim_rcount, max_rcount}};

            gpu_assert(tile[dim_simd] % simd == 0);
            gpu_assert(tile[dim_sdepth] % (sdepth_pack * sdepth) == 0);
            std::vector<stmt_t> dpas_stmts;

            v2::for_each(tile, inst_tile, [&](const icoord_t &coord) {
                auto simd = inst_tile[dim_simd];
                auto sdepth = inst_tile[dim_sdepth] / sdepth_pack;
                auto rcount = std::min(inst_tile[dim_rcount],
                        tile[dim_rcount] - coord[dim_rcount]);

                auto dpas = dpas_t::make(false, simd, into<uint8_t>(sdepth),
                        into<uint8_t>(rcount), C.layout.type(), B.layout.type(),
                        A.layout.type());
                auto a_off = A.layout.offset_in_bytes(base + coord);
                auto b_off = B.layout.offset_in_bytes(base + coord);
                auto c_off = C.layout.offset_in_bytes(base + coord);
                auto dst = C.buffer[c_off];
                auto src1 = A.buffer[a_off];
                auto src2 = B.buffer[b_off];
                dpas_stmts.emplace_back(
                        dpas.as<dpas_t>()(dst, dst, src1, src2));
            });
            append(inject_dpas_atomic(stmt_seq_t::make(dpas_stmts),
                    /*filter_by_label=*/false));
        } else {
            auto max_simd = 32;

            auto dim_simd = C.layout.blocks()[0].dim;
            auto dim_rcount = C.layout.blocks()[1].dim;
            auto m_dim = dim_simd;
            auto n_dim = dim_rcount;
            auto k_dim = utils::one_of(
                                 A.layout.blocks()[1].dim, dim_simd, dim_rcount)
                    ? A.layout.blocks()[0].dim
                    : A.layout.blocks()[1].dim;

            tile_t inst_tile {
                    {{dim_simd, max_simd}, {dim_rcount, 1}, {k_dim, 1}}};

            int M = inst_tile.get(m_dim, 1);
            int N = inst_tile.get(n_dim, 1);
            int K = inst_tile.get(k_dim, 1);
            bool is_a_bcast = (M * K == 1);
            bool is_b_bcast = (K * N == 1);
            int a_stride = is_a_bcast ? 0 : to_cpp<int>(A.layout.stride(m_dim));
            int b_stride = is_b_bcast ? 0 : to_cpp<int>(B.layout.stride(n_dim));

            gpu_assert(tile[dim_simd] * C.layout.type().size() % grf_size == 0);
            v2::for_each(tile, inst_tile, [&](const icoord_t &coord) {
                auto simd = std::min(
                        inst_tile[dim_simd], tile[dim_simd] - coord[dim_simd]);

                auto mad = mad_t::make(ctx_->hw(), C.layout.type(), simd,
                        A.layout.type(), a_stride, B.layout.type(), b_stride);

                auto a_off = A.layout.offset_in_bytes(base + coord);
                auto b_off = B.layout.offset_in_bytes(base + coord);
                auto c_off = C.layout.offset_in_bytes(base + coord);
                auto dst = C.buffer[c_off];
                auto src1 = A.buffer[a_off];
                auto src2 = B.buffer[b_off];

                append(mad.as<mad_t>()(dst, dst, src1, src2));
            });
        }
    }

    void prefetch(const global_tensor_t &g, const transform_t &plan,
            const icoord_t &base) {
        load_store({}, g, plan, load_store_kind_t::prefetch, base);
    }

    void load(const tensor_t &t, const global_tensor_t &g,
            const transform_t &plan, const icoord_t &base) {
        load_store(t, g, plan, load_store_kind_t::load, base);
    }

    void store(const global_tensor_t &g, const tensor_t &t,
            const transform_t &plan, const icoord_t &base) {
        load_store(t, g, plan, load_store_kind_t::store, base);
    }
    void assume(expr_t e) { ctx_->add_constraint(e); }

    void if_(expr_t cond, stmt_t if_body) {
        // if (is_const(cond))
        //     append(to_cpp<bool>(cond) ? if_body : stmt_t());
        // else
        append(if_t::make(cond, if_body));
    }
    void if_(expr_t cond, stmt_t if_body, stmt_t else_body) {
        // if (is_const(cond))
        //     append(to_cpp<bool>(cond) ? if_body : else_body);
        // else
        append(if_t::make(cond, if_body, else_body));
    }
    template <typename F>
    void if_(expr_t cond, F if_body) {
        begin_scope();
        if_body();
        if_(cond, pop_scope());
    }
    template <typename F, typename G>
    void if_(expr_t cond, F if_body, G else_body) {
        begin_scope();
        if_body();
        auto if_body_stmt = pop_scope();

        begin_scope();
        else_body();
        auto else_body_stmt = pop_scope();

        if_(cond, if_body_stmt, else_body_stmt);
    }

    void while_(expr_t cond, stmt_t body) { append(while_t::make(cond, body)); }

    void assign(expr_t var, expr_t value) {
        append(store_t::make(var, 0, value));
    }

    void begin_scope() { stmts_stack.push({}); }

    void end_scope() {
        auto stmt = pop_scope();
        gpu_assert(!stmts_stack.empty());
        append(stmt);
    }

    stmt_t pop_scope() {
        auto stmt = to_stmt();
        stmts_stack.pop();
        return stmt;
    }

private:
    expr_t var(type_t type, const std::string &name) {
        return var_t::make(type, name);
    }

    stmt_t to_stmt() {
        stmt_t stmt;
        size_t size = stmts().size();
        size_t end = size;
        size_t begin = size - 1;
        while (begin < end) {
            auto &s = stmts()[begin];
            if (s.is<alloc_t>() || s.is<let_t>()) {
                stmt_t body = [&]() {
                    if (begin + 1 >= end) return stmt;
                    auto seq = std::vector<stmt_t>(
                            stmts().begin() + begin + 1, stmts().begin() + end);
                    seq.push_back(stmt);
                    return stmt_seq_t::make(seq);
                }();
                end = begin;

                if (s.is<alloc_t>() && s.as<alloc_t>().body.is_empty()) {
                    auto &a = s.as<alloc_t>();
                    if (a.buf.type().is_ptr())
                        stmt = alloc_t::make(
                                a.buf, a.size, a.kind, a.attrs, body);
                    else
                        stmt = alloc_t::make(a.buf, body);
                } else if (s.is<let_t>() && s.as<let_t>().body.is_empty()) {
                    auto &l = s.as<let_t>();
                    stmt = let_t::make(l.var, l.value, body);
                }
            }
            begin--;
        }

        if (end > 0) {
            std::vector<stmt_t> seq(stmts().begin(), stmts().begin() + end);
            seq.push_back(stmt);
            stmt = stmt_seq_t::make(seq);
        }
        return stmt;
    }

    void append(stmt_t stmt) {
        gpu_assert(!stmts_stack.empty())
                << "Cannot instantiate " << stmt << " outside of a kernel";
        stmts().emplace_back(stmt);
    }

    std::vector<stmt_t> &stmts() { return stmts_stack.top(); }
    std::stack<std::vector<stmt_t>> stmts_stack;
    kernel_iface_t interface_;
    ir_context_t *ctx_ = nullptr;
    std::array<expr_t, 3> group_ids_;
    std::array<expr_t, 3> local_ids_;
    std::array<expr_t, 3> local_sizes_;
};

ctx_t &default_ctx() {
    static thread_local ctx_t ctx;
    return ctx;
}

void declare_kernel(const kernel_iface_t &interface, ir_context_t &ctx) {
    default_ctx().declare_kernel(interface, ctx);
}

stmt_t end_kernel() {
    return default_ctx().end_kernel();
}

void begin_scope() {
    default_ctx().begin_scope();
}

void end_scope() {
    default_ctx().end_scope();
}

stmt_t pop_scope() {
    return default_ctx().pop_scope();
}

void assume(expr_t e) {
    default_ctx().assume(e);
}

const std::array<expr_t, 3> &group_ids() {
    return default_ctx().group_ids();
}

expr_t group_id(int idx) {
    return default_ctx().group_id(idx);
}

const std::array<expr_t, 3> &local_ids() {
    return default_ctx().local_ids();
}

expr_t local_id(int idx) {
    return default_ctx().local_id(idx);
}

const std::array<expr_t, 3> &local_sizes() {
    return default_ctx().local_sizes();
}

expr_t local_size(int idx) {
    return default_ctx().local_size(idx);
}

expr_t arg(const std::string &name) {
    return default_ctx().arg(name);
}

expr_t def(
        type_t type, const std::string &name, expr_t value, bool force_alloc) {
    return default_ctx().def(type, name, value, force_alloc);
}

expr_t def(const std::string &name, expr_t value) {
    return def(value.type(), name, value);
}

tensor_t def(v2::layout_t layout, const std::string &name, expr_t value) {
    return default_ctx().def(layout, name, value);
}

expr_t let(type_t type, const std::string &name, expr_t value) {
    return default_ctx().let(type, name, value);
}

expr_t let(const std::string &name, expr_t value) {
    return default_ctx().let(name, value);
}

void prefetch(const global_tensor_t &g, const transform_t &plan,
        const icoord_t &base) {
    default_ctx().prefetch(g, plan, base);
}

void load(const tensor_t &t, const global_tensor_t &g, const transform_t &plan,
        const icoord_t &base) {
    default_ctx().load(t, g, plan, base);
}

void store(const global_tensor_t &g, const tensor_t &t, const transform_t &plan,
        const icoord_t &base) {
    default_ctx().store(g, t, plan, base);
}

void mma(const tensor_t &C, const tensor_t &A, const tensor_t &B,
        const tile_t tile, const icoord_t &base, bool is_systolic) {
    default_ctx().mma(C, A, B, tile, base, is_systolic);
}

void assign(expr_t var, expr_t value) {
    default_ctx().assign(var, value);
}

template <>
void if_(expr_t cond, stmt_t if_body) {
    default_ctx().if_(cond, if_body);
}

template <>
void if_(expr_t cond, stmt_t if_body, stmt_t else_body) {
    default_ctx().if_(cond, if_body, else_body);
}

template <>
void while_(expr_t cond, stmt_t body) {
    default_ctx().while_(cond, body);
}

} // namespace dsl
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
