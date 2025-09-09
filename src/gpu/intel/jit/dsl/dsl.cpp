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

#include "gpu/intel/jit/dsl/dsl.hpp"
#include "gpu/intel/jit/ir/block_2d_utils.hpp"
#include "gpu/intel/jit/ir/builder.hpp"
#include "gpu/intel/jit/ir/message_patterns.hpp"
#include "gpu/intel/jit/ir/v2/tensor.hpp"
#include "gpu/intel/jit/pass/dpas.hpp"
#include "gpu/intel/logging.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace dsl {

struct ctx_t {
    bool new_ir_api() const { return new_ir_api_; }

    void declare_kernel(const kernel_iface_t &interface, ir_context_t &ctx,
            bool new_ir_api = false) {
        slm_byte_offset_ = 0;
        new_ir_api_ = new_ir_api;
        gpu_assert(stmts_stack_.empty())
                << "Invalid generation of a kernel within a kernel";
        interface_ = interface;
        ctx_ = &ctx;

        begin_scope();

        if (new_ir_api_) {
            for (int i = 0; i < 3; i++) {
                local_sizes_[i] = var_t::make(local_size_type(),
                        std::string("local_size_") + "012"[i]);
                local_ids_[i] = var_t::make(
                        local_id_type(), std::string("local_id_") + "012"[i]);
                group_ids_[i] = var_t::make(
                        group_id_type(), std::string("group_id_") + "012"[i]);
            }
        } else {
            for (int i = 0; i < interface.nargs(); i++) {
                const auto &var = interface.arg_var(i);
                if (var.type().is_ptr()) {
                    if (var.type().is_slm()) {
                        append(alloc_t::make(
                                var, 0, alloc_kind_t::slm, stmt_t {}));
                    } else {
                        append(alloc_t::make(
                                var, 0, alloc_kind_t::global, stmt_t {}));
                    }
                } else {
                    if (!new_ir_api_) append(let_t::make(var, {}, {}));
                }
            }

            for (int i = 0; i < 3; i++) {
                group_ids_[i]
                        = let(group_id_type(), ir_builder_t::tg_idx(i), {});
                local_ids_[i]
                        = let(local_id_type(), ir_builder_t::local_id(i), {});
                local_sizes_[i] = let(
                        local_size_type(), ir_builder_t::local_size(i), {});
            }
        }
    }

    kernel_t end_kernel() {
        gpu_assert(stmts_stack_.size() == 1)
                << "Invalid end of kernel, imbalanced scopes detected";
        kernel_t ret {std::move(interface_), pop_scope(), ctx_->exec_cfg()};
        ctx_ = nullptr;
        interface_ = {"undefined_dsl_kernel"};
        return ret;
    }

    int simd() const { return ctx_->exec_cfg().simd(); }

    const std::array<expr_t, 3> &group_ids() const { return group_ids_; }
    const expr_t &group_id(int idx) const { return group_ids_[idx]; }
    const std::array<expr_t, 3> &local_ids() const { return local_ids_; }
    const expr_t &local_id(int idx) const { return local_ids_[idx]; }
    const std::array<expr_t, 3> &local_sizes() const { return local_sizes_; }
    const expr_t &local_size(int idx) const { return local_sizes_[idx]; }

    expr_t arg(const std::string &name, bool allow_empty = false) {
        auto a = interface_.find_arg(name, allow_empty);
        expr_t value;
        if (a && ctx_->cset().is_single_value(a, value)) { return value; }
        return a;
    }

    // TODO: Remove IR restriction which requires force_alloc
    lval_t def(type_t _type, const std::string &name, const expr_t &value = {},
            bool force_alloc = false) {
        auto type = _type.with_attr(_type.attr() | type::attr_t::mut);
        auto alloc_var = var(type, name);
        if (force_alloc || type.is_ptr()) {
            append(alloc_t::make(alloc_var, {}));

            if (!value.is_empty()) {
                gpu_assert(to_cpp<int>(value) == 0);
                append(funcs::zero_out(alloc_var, type.size()));
            }
        } else {
            if (new_ir_api_) {
                if (!value.is_empty()) append(assign_t::make(alloc_var, value));
            } else {
                append(let_t::make(alloc_var, value, {}));
            }
        }
        return lval_t(alloc_var.as<var_t>());
    }

    lval_t def(const std::string &name, const expr_t &value) {
        return def(value.type(), name, value);
    }

    tensor_t def(const layout_t &layout, const std::string &name,
            const expr_t &value = {}) {
        // Tensors need to be grf-aligned for loading/storing
        // TODO: IR should be modified to enable loading small tensors (such as
        // scalar values) without GRF alignment.
        auto elems = std::max(into<int>(layout.type().elems() * layout.elems()),
                grf_size() / layout.type().scalar().size());
        auto t = layout.type()[elems];
        return {def(t, name, value, true), layout};
    }

    expr_t let(type_t type, const std::string &name, const expr_t &value) {
        auto alloc_var = var(type, name);
        append(let_t::make(alloc_var, value, {}));
        return alloc_var;
    }
    expr_t let(const std::string &name, const expr_t &value) {
        return let(value.type(), name, value);
    }

    int slm_byte_offset() const { return slm_byte_offset_; }

    void reserve_slm(int bytes) { slm_byte_offset_ += bytes; }

    void assume(const expr_t &e) { ctx_->add_constraint(e); }

    void begin_scope() { stmts_stack_.emplace(); }

    void end_scope() {
        auto stmt = pop_scope();
        gpu_assert(!stmts_stack_.empty());
        append(stmt);
    }

    stmt_t pop_scope() {
        auto stmt = to_stmt();
        stmts_stack_.pop();
        return stmt;
    }

    void append(stmt_t stmt) {
        gpu_assert(!stmts_stack_.empty())
                << "Cannot instantiate " << stmt << " outside of a kernel";
        stmts().emplace_back(std::move(stmt));
    }

    const ir_context_t *ir_ctx() const { return ctx_; }

private:
    type_t local_id_type() const { return u16; }
    type_t group_id_type() const { return u32; }
    type_t local_size_type() const { return u16; }

    expr_t var(type_t type, const std::string &name) {
        return var_t::make(type, ctx_->create_tmp_name(name));
    }

    stmt_t to_stmt() {
        stmt_t stmt;
        gpu_assert(!stmts_stack_.empty());
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

    std::vector<stmt_t> &stmts() { return stmts_stack_.top(); }
    std::stack<std::vector<stmt_t>> stmts_stack_;
    kernel_iface_t interface_ = {"undefined_dsl_kernel"};
    ir_context_t *ctx_ = nullptr;
    std::array<expr_t, 3> group_ids_;
    std::array<expr_t, 3> local_ids_;
    std::array<expr_t, 3> local_sizes_;
    bool new_ir_api_ = false;
    int slm_byte_offset_ = 0;
};

ctx_t &default_ctx() {
    static thread_local ctx_t ctx;
    return ctx;
}

int grf_size() {
    return default_ctx().ir_ctx()->hw().grf_size();
}
int min_align_2d() {
    return block_2d_base_alignment(default_ctx().ir_ctx()->hw().to_ngen());
}
int min_pitch_2d() {
    return block_2d_pitch_alignment(default_ctx().ir_ctx()->hw().to_ngen());
}

void declare_kernel(
        const kernel_iface_t &interface, ir_context_t &ctx, bool new_ir_api) {
    default_ctx().declare_kernel(interface, ctx, new_ir_api);
}

kernel_t end_kernel() {
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

void append(stmt_t stmt) {
    default_ctx().append(std::move(stmt));
}

void assume(const expr_t &e) {
    default_ctx().assume(e);
}

const std::array<expr_t, 3> &group_ids() {
    return default_ctx().group_ids();
}

const expr_t &group_id(int idx) {
    return default_ctx().group_id(idx);
}

const std::array<expr_t, 3> &local_ids() {
    return default_ctx().local_ids();
}

const expr_t &local_id(int idx) {
    return default_ctx().local_id(idx);
}

const std::array<expr_t, 3> &local_sizes() {
    return default_ctx().local_sizes();
}

const expr_t &local_size(int idx) {
    return default_ctx().local_size(idx);
}

expr_t subgroup_id(int idx) {
    int simd = default_ctx().ir_ctx()->exec_cfg().simd();
    return extract((local_id(idx) / simd), 0);
}

expr_t arg(const std::string &name, bool allow_empty) {
    return default_ctx().arg(name, allow_empty);
}

lval_t def(type_t type, const std::string &name, const expr_t &value,
        bool force_alloc) {
    return default_ctx().def(type, name, value, force_alloc);
}

lval_t def(const std::string &name, const type_t &type, const expr_t &value) {
    return def(type, name, value);
}

lval_t def(const std::string &name, const expr_t &value) {
    return def(value.type(), name, value);
}

tensor_t def(
        const layout_t &layout, const std::string &name, const expr_t &value) {
    return default_ctx().def(layout, name, value);
}

tensor_t def_slm(layout_t layout, const std::string &name) {
    auto alloc_elems = into<int>(layout.size() / layout.type().size());
    auto buf = def(name, layout.type().with_slm()[alloc_elems]);
    int bytes = (to_cpp<int>(layout.offset()) + alloc_elems)
            * layout.type().size();
    auto off = utils::div_up(
            default_ctx().slm_byte_offset(), layout.type().size());
    layout.set_offset(off);
    default_ctx().reserve_slm(bytes);
    return tensor_t(buf, layout);
}

expr_t iif(
        const expr_t &cond, const expr_t &true_expr, const expr_t &false_expr) {
    return iif_t::make(cond, true_expr, false_expr);
}

expr_t extract(const expr_t &expr, int lane) {
    return shuffle_t::make(expr, {lane});
}

lval_t &lval_t::operator=(const expr_t &obj) {
    assign(this->var, obj);
    return *this;
}

expr_t let(type_t type, const std::string &name, const expr_t &value) {
    return default_ctx().let(type, name, value);
}

expr_t let(const std::string &name, const expr_t &value) {
    return default_ctx().let(name, value);
}

void assign(const expr_t &var, const expr_t &value) {
    if (default_ctx().new_ir_api()) {
        append(assign_t::make(var, value));
    } else {
        append(store_t::make(var, 0, value));
    }
}

enum class send_kind_t { load, prefetch, store };

void scatter_send(const tensor_t &t, const global_tensor_t &g,
        send_kind_t &op_kind, const icoord_t &base, const send_hint_t &hint) {
    gpu_warning() << "Scatter messages are not yet implemented";
};

void block_send(const tensor_t &t, const global_tensor_t &g,
        send_kind_t &op_kind, const icoord_t &base, const send_hint_t &hint) {
    bool is_prefetch = t.buf.is_empty();
    auto &operation_tile = is_prefetch ? g.tile : t.layout.tile();

    idx_t w_idx;
    tile_t tile;
    for (auto &var : operation_tile) {
        if (is_const(g.strides[var]) && to_cpp<dim_t>(g.strides[var]) == 1
                && t.layout.elems() != 1) {
            tile[var] = t.layout.blocks()[0].block;
            gpu_assert(t.layout.blocks()[0].dim == var);
            w_idx = var;
        } else {
            tile[var] = 1;
        }
    }
    auto type = g.type;

    v2::for_each(operation_tile, tile, [&](const icoord_t &coord) {
        auto buffer = is_prefetch ? expr_t()
                                  : t.buf[t.layout.offset_in_bytes(coord)];
        auto width = !w_idx.is_undef()
                ? std::min(tile[w_idx], operation_tile[w_idx] - coord[w_idx])
                : 1;

        int width_bytes = into<int>(width * type.size());
        auto coord_local = coord;
        while (width_bytes > 0) {
            auto send_type = [&]() {
                if (width_bytes <= 16) { return type_t::byte(width_bytes); }
                auto load_width = dnnl::impl::utils::rnd_down_pow2(
                        std::min(width_bytes, 512));
                return type_t::oword(load_width / 16);
            }();
            auto send_kind = [&]() {
                switch (op_kind) {
                    case send_kind_t::prefetch: return send_op_t::prefetch;
                    case send_kind_t::load: return send_op_t::load;
                    case send_kind_t::store: return send_op_t::store;
                    default: gpu_error_not_expected(); return send_op_t::undef;
                }
            }();

            auto send_func = send_t::make({}, send_kind, send_address_t::a64,
                    send_type, 1, true, true, hint.cache);
            append(send_func.as<send_t>()(
                    g.buf, g.offset(base + coord_local), buffer, {}));
            width_bytes -= send_type.size();
            coord_local[w_idx] += send_type.size() / type.size();
        }
    });
}

struct conf_2d_t {
    type_t type;
    idx_t w_idx;
    int pack_size;
    bool is_vnni;
    bool is_transpose_vnni;
    bool is_store;

    int unit_size() const {
        return is_transpose_vnni || is_vnni ? std::max(type.size(), 4)
                                            : type.size();
    }

    // Tile used for 2d Messages
    tile_t get_tile(std::array<idx_t, 2> dims) const {
        auto width = pack_size ? pack_size : grf_size() / unit_size();
        auto height = is_store ? 8 : 32;

        if (is_transpose_vnni) return {{dims[1], width}, {dims[0], height}};
        return {{dims[0], width}, {dims[1], height}};
    }
};

void block_2d_send(const conf_2d_t &conf, const tensor_t &t,
        const global_tensor_t &g, send_kind_t op_kind, const icoord_t &base,
        const send_hint_t &hint) {

    bool is_prefetch = t.buf.is_empty();
    auto &operation_tile = is_prefetch ? g.tile : t.layout.tile();

    idx_t w_idx = conf.w_idx;
    idx_t h_idx;
    for (auto &var : operation_tile) {
        if (var != w_idx) {
            gpu_assert(h_idx.is_undef())
                    << "n-dimensional support unimplemented";
            h_idx = var;
        }
    }

    auto tensor_width = g.sizes[w_idx];
    auto tensor_height = g.sizes[h_idx];
    auto tensor_pitch = g.strides[h_idx];
    auto type = g.type;
    auto tile = conf.get_tile({w_idx, h_idx});

    v2::for_each(operation_tile, tile, [&](const icoord_t &coord) {
        auto buffer = is_prefetch ? expr_t()
                                  : t.buf[t.layout.offset_in_bytes(coord)];
        int width = into<int>(
                std::min(tile[w_idx], operation_tile[w_idx] - coord[w_idx]));
        int height = into<int>(
                std::min(tile[h_idx], operation_tile[h_idx] - coord[h_idx]));
        int count = std::max(1, into<int>(tile[w_idx] / width));
        auto width_idx
                = g.coord[w_idx] + static_cast<uint32_t>((base + coord)[w_idx]);
        auto height_idx
                = g.coord[h_idx] + static_cast<uint32_t>((base + coord)[h_idx]);
        auto send_kind = [&]() {
            switch (op_kind) {
                case send_kind_t::prefetch: return send_op_t::prefetch_2d;
                case send_kind_t::load: return send_op_t::load_2d;
                case send_kind_t::store: return send_op_t::store_2d;
                default: gpu_error_not_expected(); return send_op_t::undef;
            }
        }();

        auto send_func = send_t::make_2d({}, send_kind, type, tensor_width,
                tensor_height, tensor_pitch, width, height, count, conf.is_vnni,
                conf.is_transpose_vnni,
                /*zero_out=*/true, hint.cache);

        append(send_func.as<send_t>()(g.buf, g.base_offset * type.size(),
                buffer, {}, width_idx, height_idx));
    });
}

void send(const tensor_t &t, const global_tensor_t &g, send_kind_t op_kind,
        const icoord_t &base, const send_hint_t &hint) {
    bool is_prefetch = t.buf.is_empty();
    auto &operation_tile = is_prefetch ? g.tile : t.layout.tile();
    idx_t w_idx;
    for (auto &var : operation_tile) {
        if (is_const(g.strides[var]) && to_cpp<dim_t>(g.strides[var]) == 1) {
            gpu_assert(w_idx.is_undef())
                    << "Could not determine inner dimension";
            w_idx = var;
        }
    }

    auto type = g.type;

    gpu_assert(is_prefetch || type == t.layout.type());
    if (operation_tile.size() >= 2 && !w_idx.is_undef()) {
        auto conf = [&]() -> conf_2d_t {
            if (is_prefetch) { return {g.type, w_idx, 0, false, false, false}; }
            auto &l = t.layout;
            int pack_idx = l.blocks()[0].block * l.type().size() == 4;
            int pack_size = into<int>(l.blocks()[pack_idx].block);
            bool is_transpose_vnni = l.blocks()[pack_idx].dim != w_idx;
            bool is_vnni = pack_idx == 1 && !is_transpose_vnni;
            bool is_store = op_kind == send_kind_t::store;
            return {g.type, w_idx, pack_size, is_vnni, is_transpose_vnni,
                    is_store};
        }();

        if (conf.pack_size <= grf_size() / conf.unit_size()) {
            block_2d_send(conf, t, g, op_kind, base, hint);
            return;
        }
    }

    if (is_prefetch || t.layout.elems() == 1
            || t.layout.blocks()[0].dim == w_idx) {
        block_send(t, g, op_kind, base, hint);
    } else {
        scatter_send(t, g, op_kind, base, hint);
    }
}

void prefetch(const global_tensor_t &g, const icoord_t &base,
        const send_hint_t &hint) {
    send({}, g, send_kind_t::prefetch, base, hint);
}

void load(const tensor_t &t, const global_tensor_t &g, const icoord_t &base,
        const send_hint_t &hint) {
    send(t, g, send_kind_t::load, base, hint);
}

void store(const global_tensor_t &g, const tensor_t &t, const icoord_t &base,
        const send_hint_t &hint) {
    send(t, g, send_kind_t::store, base, hint);
}

void mma(const tensor_t &C, const tensor_t &A, const tensor_t &B,
        const tile_t &tile, const icoord_t &base, bool is_systolic) {
    if (is_systolic) {
        int64_t simd = 16;
        int64_t sdepth = 8;
        int64_t max_rcount = 8;

        auto simd_idx = C.layout.blocks()[0].dim;
        auto sdepth_idx = A.layout.blocks()[0].dim == C.layout.blocks()[0].dim
                ? A.layout.blocks()[1].dim
                : A.layout.blocks()[0].dim;
        auto rcount_dim = C.layout.blocks()[1].dim;
        auto sdepth_pack = 4 / A.layout.type().size();

        tile_t inst_tile {{simd_idx, simd}, {sdepth_idx, sdepth * sdepth_pack},
                {rcount_dim, max_rcount}};

        gpu_assert(tile[simd_idx] % simd == 0);
        gpu_assert(tile[sdepth_idx] % (sdepth_pack * sdepth) == 0);
        gpu_assert(C.layout.blocks()[0].block == simd);
        std::vector<stmt_t> dpas_stmts;

        v2::for_each(tile, inst_tile, [&](const icoord_t &coord) {
            int simd = (int)inst_tile[simd_idx];
            auto sdepth = inst_tile[sdepth_idx] / sdepth_pack;
            auto rcount = std::min(inst_tile[rcount_dim],
                    tile[rcount_dim] - coord[rcount_dim]);

            auto dpas = dpas_t::make(false, simd, into<uint8_t>(sdepth),
                    into<uint8_t>(rcount), C.layout.type(), B.layout.type(),
                    A.layout.type());
            // FIXME: This code can access out-of-bounds coordinates, adding
            // modulus to keep the old behavior with v2 layout.
            auto get_offset = [](const layout_t &layout, icoord_t coord) {
                for (auto &d : coord) {
                    coord[d] = coord[d] % layout.tile().get(d, coord[d] + 1);
                }
                return layout.offset_in_bytes(coord);
            };
            auto a_off = get_offset(A.layout, base + coord);
            auto b_off = get_offset(B.layout, base + coord);
            auto c_off = C.layout.offset_in_bytes(base + coord);
            auto dst = C.buf[c_off];
            auto src1 = A.buf[a_off];
            auto src2 = B.buf[b_off];
            dpas_stmts.emplace_back(dpas.as<dpas_t>()(dst, dst, src1, src2));
        });
        append(inject_dpas_atomic(stmt_seq_t::make(dpas_stmts),
                /*filter_by_label=*/false));
    } else {
        auto max_simd = 32;

        const auto &simd_idx = C.layout.blocks()[0].dim;
        const auto &rcount_idx = C.layout.blocks()[1].dim;
        const auto &m_idx = simd_idx;
        const auto &n_idx = rcount_idx;
        const auto &k_idx
                = utils::one_of(A.layout.blocks()[1].dim, simd_idx, rcount_idx)
                ? A.layout.blocks()[0].dim
                : A.layout.blocks()[1].dim;

        tile_t inst_tile {{{simd_idx, max_simd}, {rcount_idx, 1}, {k_idx, 1}}};

        int M = (int)inst_tile.get(m_idx, 1);
        int N = (int)inst_tile.get(n_idx, 1);
        int K = (int)inst_tile.get(k_idx, 1);
        bool is_a_bcast = (M * K == 1);
        bool is_b_bcast = (K * N == 1);
        int a_stride = is_a_bcast ? 0 : into<int>(A.layout.stride(m_idx));
        int b_stride = is_b_bcast ? 0 : into<int>(B.layout.stride(n_idx));

        gpu_assert(tile[simd_idx] * C.layout.type().size() % grf_size() == 0);
        v2::for_each(tile, inst_tile, [&](const icoord_t &coord) {
            int simd = (int)std::min(
                    inst_tile[simd_idx], tile[simd_idx] - coord[simd_idx]);

            auto mad = mad_t::make(default_ctx().ir_ctx()->hw(),
                    C.layout.type(), simd, A.layout.type(), a_stride,
                    B.layout.type(), b_stride);

            auto a_off = A.layout.offset_in_bytes(base + coord);
            auto b_off = B.layout.offset_in_bytes(base + coord);
            auto c_off = C.layout.offset_in_bytes(base + coord);
            auto dst = C.buf[c_off];
            auto src1 = A.buf[a_off];
            auto src2 = B.buf[b_off];

            append(mad.as<mad_t>()(dst, dst, src1, src2));
        });
    }
}

void binary(op_kind_t op, const tensor_t &dst, const tensor_t &src0,
        const tensor_t &src1) {
    tile_t tile = dst.layout.tile();
    tile_t matching_subtile = [&] {
        tile_t ret;
        for (auto &var : tile) {
            ret[var] = 1;
        }

        auto &bd = dst.layout.blocks();
        auto &b0 = src0.layout.blocks();
        auto &b1 = src1.layout.blocks();
        for (size_t i = 0; i < bd.size(); i++) {
            if (b0.size() <= i) break;
            if (b1.size() <= i) break;
            if (bd[i] == b0[i] && bd[i] == b1[i])
                ret[bd[i].dim] *= bd[i].block;
            else
                break;
        }

        return ret;
    }();

    auto subtile_elems = matching_subtile.elems();

    v2::for_each(tile, matching_subtile, [&](const icoord_t &coord) {
        auto offd = dst.layout.offset_in_bytes(coord);
        auto off0 = src0.layout.offset_in_bytes(coord);
        auto off1 = src1.layout.offset_in_bytes(coord);

        dim_t simd = default_ctx().simd();
        for (int idx = 0; idx < subtile_elems; idx += simd) {
            int elems = into<int>(std::min(subtile_elems - idx, simd));
            auto s0 = load_t::make(src0.layout.type().with_elems(elems),
                    src0.buf, off0 + idx * src0.layout.type().size());
            auto s1 = load_t::make(src1.layout.type().with_elems(elems),
                    src1.buf, off1 + idx * src1.layout.type().size());
            assign(dst.buf[offd + dst.layout.type().size() * idx],
                    binary_op_t::make(op, s0, s1));
        }
    });
}

void barrier() {
    append(builtin_t::make("barrier")());
}

} // namespace dsl
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
