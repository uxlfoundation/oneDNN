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
#include "gpu/intel/jit/v2/conv/plan.hpp"
#include "gpu/intel/jit/v2/ir/tensor.hpp"
namespace gemmstone {

using dim_map_t = std::array<int, 3>;

enum class plan_kind_t { load, prefetch, store };

struct send_plan_t {
    static send_plan_t make(plan_kind_t kind, ir::type_t type, bool vnni,
            bool transpose, bool dense, bool aligned,
            ngen::CacheSettingsLSC cache_hint, ir::v2::layout_t layout = {}) {
        send_plan_t plan {};

        plan.op = [&]() {
            switch (kind) {
                case plan_kind_t::load:
                    if (dense && aligned) return ir::send_op_t::load_2d;
                    return ir::send_op_t::load;
                case plan_kind_t::prefetch:
                    if (dense && aligned) return ir::send_op_t::prefetch_2d;
                    return ir::send_op_t::prefetch;
                case plan_kind_t::store:
                    if (dense && aligned) return ir::send_op_t::store_2d;
                    return ir::send_op_t::store;
                default: stub();
            }
        }();

        plan.kind = [&]() {
            if (dense && aligned) return ir::send_kind_t::_2d;
            if (dense) return ir::send_kind_t::block;
            return ir::send_kind_t::scattered;
        }();
        plan.type = type;
        plan.cache_hint = to_ir(cache_hint);
        plan.vnni = vnni;
        plan.transpose = transpose;
        plan.layout = layout;
        plan.dims = [&]() -> std::array<ir::pvar_t, 2> {
            if (transpose) {
                if (layout.type().size() < 4) {
                    return {layout.blocks()[0].dim, layout.blocks()[1].dim};
                } else
                    return {layout.blocks()[1].dim, layout.blocks()[0].dim};
            } else {
                if (vnni) {
                    return {layout.blocks()[1].dim, layout.blocks()[0].dim};
                } else {
                    return {layout.blocks()[0].dim, layout.blocks()[1].dim};
                }
            }
        }();
        return plan;
    }

    ir::pvar_tile_t get_tile() const {
        if (kind == ir::send_kind_t::_2d) {
            auto width = vnni || transpose ? 64 / 4 : 64 / type.size();
            auto height = 32;
            if (transpose) return {{dims[1], width}, {dims[0], height}};
            return {{dims[0], width}, {dims[1], height}};
        } else if (kind == ir::send_kind_t::block) {
            auto width = 32;
            auto height = 1;
            if (transpose) return {{dims[1], width}, {dims[0], height}};
            return {{dims[0], width}, {dims[1], height}};
        }
        return {};
    }

    // func_t make(int width, int height) {
    //     return ir::send_t::make_2d({}, op, type, width, height, 1, vnni,
    //             transpose, false, cache_hint);
    // }

    static ir::send_cache_hint_t to_ir(ngen::CacheSettingsLSC hint) {
        switch (hint) {
            case ngen::CacheSettingsLSC::L1C_L3C:
                return ir::send_cache_hint_t::load_once;
            case ngen::CacheSettingsLSC::Default:
                return ir::send_cache_hint_t::hw_default;
            default: stub(); return ir::send_cache_hint_t::undef;
        }
    }

    ir::send_op_t op = ir::send_op_t::undef;
    ir::send_kind_t kind = ir::send_kind_t::undef;
    ir::type_t type;
    ir::send_cache_hint_t cache_hint;
    int slots = 32;
    bool vnni;
    bool transpose;
    ir::v2::layout_t layout;
    std::array<ir::pvar_t, 2> dims;
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
    ir::pvar_map_t<ir::expr_t> idxs;
    ir::pvar_map_t<ir::expr_t> strides;
    ir::expr_t offset(const ir::pvar_coord_t<int64_t> &coord) const {
        ir::expr_t ret = 0;
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
    virtual ir::expr_t is_valid_kloop(int k_block) const = 0;
};

struct scope_builder_t {
    scope_builder_t() { push(); }
    void alloc(
            ir::expr_t var, ir::expr_t value = {}, bool is_external = false) {
        //gpu_assert(var.is<var_t>());
        auto type = var.type();
        if (is_external) {
            if (type.is_ptr()) {
                append(ir::alloc_t::make(
                        var, 0, ir::alloc_kind_t::global, ir::stmt_t {}));
            } else {
                append(ir::let_t::make(var, {}, {}));
            }
        } else {
            if (value.is_empty()) {
                append(ir::alloc_t::make(var, ir::stmt_t {}));
            } else {
                append(ir::let_t::make(var, value, {}));
            }
        }
    };

    ir::expr_t alloc(ir::type_t type, const std::string &name,
            ir::expr_t value = {}, bool is_external = false) {
        auto alloc_var = var(type, name);
        alloc(alloc_var, value, is_external);
        return alloc_var;
    }

    tensor_t alloc_tensor(ir::v2::layout_t layout, const std::string &name,
            ir::expr_t value = {}) {
        auto t = ir::type_t(
                layout.type().kind(), layout.type().elems() * layout.elems());
        return {alloc(t, name, value), layout};
    }

    void load_store(const tensor_t &t, const global_tensor_t &g,
            const send_plan_t &plan, const ir::pvar_coord_t<int64_t> &base) {
        auto tile = plan.get_tile();
        auto sizes = plan.layout.int_dim_sizes();
        bool is_prefetch = t.buffer.is_empty();
        std::vector<ir::pvar_t> dim_order = {plan.dims[0], plan.dims[1]};
        ir::v2::for_each(
                sizes, tile, [&](const ir::pvar_coord_t<int64_t> &coord) {
                    auto t_off = plan.layout.offset_in_bytes(base + coord);
                    auto buf = is_prefetch ? ir::expr_t() : t.buffer[t_off];
                    auto width = std::min(tile[plan.dims[0]],
                            sizes[plan.dims[0]] - coord[plan.dims[0]]);
                    auto height = std::min(tile[plan.dims[1]],
                            sizes[plan.dims[1]] - coord[plan.dims[1]]);

                    auto send_func = [&]() {
                        if (plan.kind == ir::send_kind_t::_2d) {
                            return ir::send_t::make_2d({}, plan.op, plan.type,
                                    width, height, 1, plan.vnni, plan.transpose,
                                    /*zero_out=*/true, plan.cache_hint);
                        } else {
                            return ir::send_t::make({}, plan.op,
                                    ir::send_address_t::a64, plan.type,
                                    plan.slots, true, plan.cache_hint);
                        }
                    }();

                    append(send_func.as<ir::send_t>()(
                            g.buffer, g.offset(base + coord), buf, {}));
                });
    }

    void prefetch(const global_tensor_t &g, const send_plan_t &plan,
            const ir::pvar_coord_t<int64_t> &base) {
        load_store({}, g, plan, base);
    }

    void load(const tensor_t &t, const global_tensor_t &g,
            const send_plan_t &plan, const ir::pvar_coord_t<int64_t> &base) {
        load_store(t, g, plan, base);
    }

    void store(const global_tensor_t &g, const tensor_t &t,
            const send_plan_t &plan, const ir::pvar_coord_t<int64_t> &base) {
        load_store(t, g, plan, base);
    }

    template <typename F>
    void if_(ir::expr_t cond, F body) {
        push();
        body();
        auto body_stmt = pop();
        append(ir::if_t::make(cond, body_stmt));
    }

    template <typename F>
    void while_(ir::expr_t cond, F body) {
        push();
        body();
        auto body_stmt = pop();
        append(ir::while_t::make(cond, body_stmt));
    }

    void assign(ir::expr_t var, ir::expr_t value) {
        append(ir::store_t::make(var, 0, value));
    }

    ir::expr_t var(ir::type_t type, const std::string &name) {
        return ir::var_t::make(type, name);
    }

    void append(ir::stmt_t stmt) { stmts().emplace_back(stmt); }

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

private:
    void push() { stmts_stack.push({}); }
    ir::stmt_t pop() {
        auto stmt = to_stmt();
        stmts_stack.pop();
        return stmt;
    }

    std::vector<ir::stmt_t> &stmts() { return stmts_stack.top(); }
    std::stack<std::vector<ir::stmt_t>> stmts_stack;
};

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

struct pdim_t {
    ir::pvar_t pvar;
    int size;
};

ir::v2::layout_t get_layout(const MatrixAddressing &matrix_type,
        const MatrixAddressingStrategy &matrix_strategy, ir::type_t t,
        std::array<pdim_t, 2> dims, const ir::v2::layout_desc_t &desc) {
    if (matrix_type.layout == MatrixLayout::Pc
            || matrix_type.layout == MatrixLayout::Pr)
        stub();

    const int grf_size = 64;
    auto access_type = matrix_strategy.accessType;
    auto &row = matrix_type.layout == MatrixLayout::N ? dims[1] : dims[0];
    auto &col = matrix_type.layout == MatrixLayout::N ? dims[0] : dims[1];
    if (isTransposing(access_type)) { std::swap(row, col); }
    bool is_vnni = (isTransposing(access_type) && t.size() < 4)
            || access_type == AccessType::Block2DVNNI;

    if (is_vnni) {
        int row_inner = 4 / t.scalar().size();
        int row_outer = row.size / row_inner;
        int col_inner = grf_size / 4;
        int col_outer = col.size / col_inner;
        return ir::v2::layout_t(desc, t, 0,
                {{row.pvar, row_inner, 1}, {col.pvar, col_inner, row_inner},
                        {row.pvar, row_outer, col_inner * row_inner},
                        {col.pvar, col_outer,
                                row_outer * col_inner * row_inner}});
    } else {
        return ir::v2::layout_t(desc, t, 0,
                {{col.pvar, col.size, 1}, {row.pvar, row.size, col.size}});
    }
}

send_plan_t get_plan(const MatrixAddressing &matrix_type,
        const MatrixAddressingStrategy &matrix_strategy, plan_kind_t kind,
        ir::type_t t, std::array<pdim_t, 2> dims,
        const ir::v2::layout_desc_t &desc) {
    auto layout = get_layout(matrix_type, matrix_strategy, t, dims, desc);

    switch (matrix_strategy.accessType) {
        case AccessType::Scattered:
            return send_plan_t::make(kind, t, false, false, true, false,
                    matrix_strategy.cachingR, layout);
        case AccessType::ChannelScattered: stub(); return {};
        case AccessType::Block2DTranspose:
            return send_plan_t::make(
                    kind, t, false, true, true, true, matrix_strategy.cachingR);
        case AccessType::Block:
            return send_plan_t::make(kind, t, false, false, true, false,
                    matrix_strategy.cachingR, layout);
        case AccessType::PseudoBlock:
            return send_plan_t::make(kind, t, false, false, false, false,
                    matrix_strategy.cachingR, layout);
        case AccessType::Block2D: {
            return send_plan_t::make(kind, t, false, false, true, true,
                    matrix_strategy.cachingR, layout);
        };
        case AccessType::Block2DVNNI: {

            return send_plan_t::make(kind, t, true, false, true, true,
                    matrix_strategy.cachingR, layout);
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
    basic_iterator_t(scope_builder_t &scope, ir::expr_t m, ir::expr_t n,
            ir::expr_t k, ir::expr_t A_buffer, ir::type_t A_type,
            ir::pvar_map_t<ir::expr_t> A_strides, ir::expr_t B_buffer,
            ir::type_t B_type, ir::pvar_map_t<ir::expr_t> B_strides,
            ir::expr_t C_buffer, ir::type_t C_type,
            ir::pvar_map_t<ir::expr_t> C_strides)
        : scope_(scope)
        , m_idx_ {scope.alloc(m.type(), "m_idx", 0)}
        , n_idx_ {scope.alloc(n.type(), "n_idx", 0)}
        , k_idx_ {scope.alloc(k.type(), "k_idx", 0)}
        , k_load_idx_ {k_idx_}
        , k_prefetch_idx_A_ {k_idx_}
        , k_prefetch_idx_B_ {k_idx_}
        , k_ {k}
        , A_prefetch_ {A_buffer, A_type,
                  {{ir::pvars::m, m_idx_}, {ir::pvars::k, k_prefetch_idx_A_}},
                  A_strides}
        , A_load_ {A_buffer, A_type,
                  {{ir::pvars::m, m_idx_}, {ir::pvars::k, k_load_idx_}},
                  A_strides}
        , B_prefetch_ {B_buffer, B_type,
                  {{ir::pvars::k, k_prefetch_idx_B_}, {ir::pvars::n, n_idx_}},
                  B_strides}
        , B_load_ {B_buffer, B_type,
                  {{ir::pvars::k, k_load_idx_}, {ir::pvars::n, n_idx_}},
                  B_strides}
        , C_store_ {C_buffer, C_type,
                  {{ir::pvars::m, m_idx_}, {ir::pvars::n, n_idx_}}, C_strides}

    {}

    const global_tensor_t &A_prefetch() const override { return A_prefetch_; }
    const global_tensor_t &A_load() const override { return A_load_; }
    const global_tensor_t &B_prefetch() const override { return B_prefetch_; }
    const global_tensor_t &B_load() const override { return B_load_; }
    const global_tensor_t &C_store() const override { return C_store_; }

    void inc_prefetch_A(int k_block) override {
        k_prefetch_idx_A_ = ir::simplify(k_prefetch_idx_A_ + k_block);
        A_prefetch_.idxs[ir::pvars::k] = k_prefetch_idx_A_;
    }

    void inc_prefetch_B(int k_block) override {
        k_prefetch_idx_B_ = ir::simplify(k_prefetch_idx_B_ + k_block);
        B_prefetch_.idxs[ir::pvars::k] = k_prefetch_idx_B_;
    }

    void inc_load(int k_block) override {
        k_load_idx_ = ir::simplify(k_load_idx_ + k_block);
        A_load_.idxs[ir::pvars::k] = k_load_idx_;
        B_load_.idxs[ir::pvars::k] = k_load_idx_;
    }

    void inc_mma(int k_block) override {
        // Prefetch/load computation is relative to k_idx
        inc_prefetch_A(-k_block);
        inc_prefetch_B(-k_block);
        inc_load(-k_block);

        scope_.assign(k_idx_, k_idx_ + k_block);
    }

    ir::expr_t update_C() const override { return false; }
    ir::expr_t is_valid_kloop(int k_block) const override {
        int max_offset = std::max(
                {to_cpp<int>(ir::simplify(k_prefetch_idx_A_ - k_idx_)),
                        to_cpp<int>(ir::simplify(k_prefetch_idx_B_ - k_idx_)),
                        to_cpp<int>(ir::simplify(k_load_idx_ - k_idx_))});

        return ir::simplify(k_idx_ <= k_ - k_block - max_offset);
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

    scope_builder_t &scope_;

    ir::expr_t m_idx_;
    ir::expr_t n_idx_;
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

struct gemm_ir : public scope_builder_t {
    gemm_ir(const GEMMProblem &problem, const GEMMStrategy &strategy)
        : problem(problem), strategy(strategy) {}

    ir::stmt_t build() {
        ir::kernel_iface_t interface;
        ir_desc_t desc(problem, strategy);
        desc.init_kernel_iface(interface);
        for (int i = 0; i < interface.nargs(); i++) {
            alloc(interface.arg_var(i), {}, true);
        }

        const auto m = interface.find_arg("m");
        const auto n = interface.find_arg("n");
        const auto k = interface.find_arg("k");

        auto m_blk = strategy.unroll[LoopM];
        auto n_blk = strategy.unroll[LoopN];
        auto k_blk = strategy.unroll[LoopK];
        std::cout << "(m_blk, n_blk, k_blk): (" << m_blk << " " << n_blk << " "
                  << k_blk << ")\n";

        static const ir::v2::layout_desc_t var_names {{{ir::pvars::m, 'm'},
                {ir::pvars::n, 'n'}, {ir::pvars::k, 'k'}}};

        // Pipeline size
        int p_size = lcm(strategy.A_copies, strategy.B_copies);
        auto k_unroll_blk = k_blk / p_size;
        auto k_prefetch_offset
                = std::max(strategy.ka_prefetch, strategy.kb_prefetch);
        auto k_load_offset = k_blk - k_unroll_blk;

        gpu_assert(k_prefetch_offset == 0 || k_prefetch_offset > k_load_offset);

        auto m_idx = alloc(m.type(), "m_idx", 0);
        auto n_idx = alloc(n.type(), "n_idx", 0);
        auto k_idx = alloc(k.type(), "k_idx", 0);

        std::array<ir::pvar_t, 2> A_pvars = {ir::pvars::m, ir::pvars::k};
        std::array<ir::pvar_t, 2> B_pvars = {ir::pvars::k, ir::pvars::n};
        std::array<ir::pvar_t, 2> C_pvars = {ir::pvars::m, ir::pvars::n};

        basic_iterator_t kloop_it(*this, m, n, k, interface.find_arg("A_ptr"),
                into_ir(problem.Ta_ext),
                get_strides(
                        problem.A.layout, A_pvars, interface.find_arg("lda")),
                interface.find_arg("B_ptr"), into_ir(problem.Tb_ext),
                get_strides(
                        problem.B.layout, B_pvars, interface.find_arg("ldb")),
                interface.find_arg("C_ptr"), into_ir(problem.Tc_ext),
                get_strides(
                        problem.C.layout, C_pvars, interface.find_arg("ldc")));

        gpu_assert(problem.Ta == problem.Ta_ext);
        std::array<pdim_t, 2> A_dims {
                {{ir::pvars::m, m_blk}, {ir::pvars::k, k_blk}}};
        auto A_prefetch_plan = get_plan(problem.A, strategy.A_prefetch,
                plan_kind_t::prefetch, into_ir(problem.Ta_ext), A_dims,
                var_names);
        auto A_load_plan = get_plan(problem.A, strategy.A, plan_kind_t::load,
                into_ir(problem.Ta_ext), A_dims, var_names);

        gpu_assert(problem.Tb == problem.Tb_ext);
        std::array<pdim_t, 2> B_dims {
                {{ir::pvars::k, k_blk}, {ir::pvars::n, n_blk}}};
        auto B_prefetch_plan = get_plan(problem.B, strategy.B_prefetch,
                plan_kind_t::prefetch, into_ir(problem.Tb_ext), B_dims,
                var_names);
        auto B_load_plan = get_plan(problem.B, strategy.B, plan_kind_t::load,
                into_ir(problem.Tb_ext), B_dims, var_names);

        std::array<pdim_t, 2> C_dims {
                {{ir::pvars::m, m_blk}, {ir::pvars::n, n_blk}}};
        auto C_store_plan = get_plan(problem.C, strategy.C, plan_kind_t::store,
                into_ir(problem.Tc), C_dims, var_names);

        std::cout << "A Layout: " << A_load_plan.layout.str() << "\n";
        std::cout << "B Layout: " << B_load_plan.layout.str() << "\n";
        std::cout << "C Layout: " << C_store_plan.layout.str() << "\n";

        tensor_t A = alloc_tensor(A_load_plan.layout, "A");
        tensor_t B = alloc_tensor(B_load_plan.layout, "B");
        tensor_t C = alloc_tensor(C_store_plan.layout, "C", 0);

        auto prefetch_A = [&](int k_unroll_idx) {
            if (k_unroll_idx % strategy.ka_pfStride != 0) return;
            prefetch(kloop_it.A_prefetch(), A_prefetch_plan,
                    {{ir::pvars::k, k_unroll_idx}});
        };
        auto load_A = [&](int k_unroll_idx) {
            if (k_unroll_idx % (k_blk / strategy.A_copies) != 0) return;
            load(A, kloop_it.A_load(), A_load_plan,
                    {{ir::pvars::k, k_unroll_idx}});
        };

        auto prefetch_B = [&](int k_unroll_idx) {
            if (k_unroll_idx % strategy.kb_pfStride != 0)
                prefetch(kloop_it.B_prefetch(), B_prefetch_plan,
                        {{ir::pvars::k, k_unroll_idx}});
        };
        auto load_B = [&](int k_unroll_idx) {
            if (k_unroll_idx % (k_blk / strategy.B_copies) != 0) return;
            load(B, kloop_it.B_load(), B_load_plan,
                    {{ir::pvars::k, k_unroll_idx}});
        };
        auto store_C
                = [&]() { store(kloop_it.C_store(), C, C_store_plan, {}); };

        auto mma = [&](int k_unroll_idx) {
            ir::pvar_coord_t<int64_t> base = {{ir::pvars::k, k_unroll_idx}};
            ir::pvar_tile_t mnk_inst_tile = [&]() {
                if (strategy.systolic) {
                    bool simd_n = C.layout.blocks()[0].dim == ir::pvars::n;
                    bool simd_m = C.layout.blocks()[0].dim == ir::pvars::m;
                    int pack_size = 4 / A.layout.type().size();
                    if (simd_n) {
                        return ir::pvar_tile_t {{ir::pvars::m, 8},
                                {ir::pvars::n, 16},
                                {ir::pvars::k, 8 * pack_size}};
                    } else if (simd_m) {
                        return ir::pvar_tile_t {{ir::pvars::m, 16},
                                {ir::pvars::n, 8},
                                {ir::pvars::k, 8 * pack_size}};
                    } else {
                        gpu_error_not_expected();
                        return ir::pvar_tile_t {};
                    }
                } else {
                    bool simd_n = C.layout.blocks()[0].dim == ir::pvars::n;
                    bool simd_m = C.layout.blocks()[0].dim == ir::pvars::m;
                    if (simd_n) {
                        return ir::pvar_tile_t {{ir::pvars::m, 1},
                                {ir::pvars::n, strategy.fmaSIMD},
                                {ir::pvars::k, 1}};
                    } else if (simd_m) {
                        return ir::pvar_tile_t {
                                {ir::pvars::m, strategy.fmaSIMD},
                                {ir::pvars::n, 1}, {ir::pvars::k, 1}};
                    } else {
                        gpu_error_not_expected();
                        return ir::pvar_tile_t {};
                    }
                }
            }();
            auto fma = strategy.systolic ? ir::fma_kind_t::dpas
                                         : ir::fma_kind_t::mad;
            auto simd = strategy.systolic ? 16 : strategy.fmaSIMD;

            ir::pvar_tile_t sizes = C.layout.int_dim_sizes();
            sizes[ir::pvars::k] = k_unroll_blk;

            // TODO: Deduplicate this, it is copied from v2/conv/builder.cpp,
            // MNK order (outer -> inner).
            std::vector<ir::pvar_t> dim_order
                    = {ir::pvars::m, ir::pvars::n, ir::pvars::k};

            int M = mnk_inst_tile.get(ir::pvars::m, 1);
            int N = mnk_inst_tile.get(ir::pvars::n, 1);
            int K = mnk_inst_tile.get(ir::pvars::k, 1);
            bool is_a_bcast = (M * K == 1);
            bool is_b_bcast = (K * N == 1);
            ir::func_t fma_func;
            switch (fma) {
                case ir::fma_kind_t::mad: {
                    int a_stride = is_a_bcast
                            ? 0
                            : to_cpp<int>(A.layout.stride(ir::pvars::m));
                    int b_stride = is_b_bcast
                            ? 0
                            : to_cpp<int>(B.layout.stride(ir::pvars::n));
                    fma_func = ir::mad_t::make(ir::hw_t(), C.layout.type(),
                            simd, A.layout.type(), a_stride, B.layout.type(),
                            b_stride);
                    break;
                }
                case ir::fma_kind_t::dpas: {
                    fma_func = ir::dpas_t::make(/*is_dpasw=*/false, simd, 8, 8,
                            C.layout.type(), B.layout.type(), A.layout.type());
                    break;
                }
                default: stub();
            }
            ir::stmt_t call_stmt;
            ir::v2::for_each(sizes, mnk_inst_tile, dim_order,
                    [&](const ir::pvar_coord_t<int64_t> &coord) {
                        auto a_off = A.layout.offset_in_bytes(base + coord);
                        auto b_off = B.layout.offset_in_bytes(base + coord);
                        auto c_off = C.layout.offset_in_bytes(base + coord);
                        auto dst = C.buffer[c_off];
                        auto src1 = A.buffer[a_off];
                        auto src2 = B.buffer[b_off];
                        if (fma == ir::fma_kind_t::dpas) std::swap(src1, src2);
                        append(fma_func.call(
                                {dst, dst, std::move(src1), std::move(src2)}));
                    });
        };

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
                mma(k_offset);
                if (k_inc) kloop_it.inc_mma(k_inc);
            }
        };

        auto warmup_k = std::max(
                {k_load_offset, strategy.ka_prefetch, strategy.kb_prefetch});
        std::cout << "k_load_offset = " << k_load_offset << "\n";
        std::cout << "ka_prefetch = " << strategy.ka_prefetch << "\n";
        std::cout << "kb_prefetch = " << strategy.kb_prefetch << "\n";
        auto tail_k = (warmup_k / k_blk) * k_blk + ((-warmup_k) % k_blk);

        std::cout << "warmup: " << warmup_k << "\n";
        for (int k_unroll_idx = 0; k_unroll_idx < warmup_k;
                k_unroll_idx += k_unroll_blk) {
            bool last_blk = k_unroll_idx >= warmup_k - k_unroll_blk;
            k_body(k_unroll_idx,
                    k_unroll_idx + strategy.ka_prefetch >= warmup_k,
                    k_unroll_idx + strategy.kb_prefetch >= warmup_k,
                    k_unroll_idx + k_load_offset >= warmup_k, false,
                    last_blk ? k_prefetch_offset - k_load_offset : 0);
        }

        std::cout << "k-loop\n";
        while_(kloop_it.is_valid_kloop(k_blk), [&]() {
            for (int k_unroll_idx = 0; k_unroll_idx < k_blk;
                    k_unroll_idx += k_unroll_blk) {
                bool last_blk = k_unroll_idx >= k_blk - k_unroll_blk;
                k_body(k_unroll_idx, strategy.ka_prefetch, strategy.kb_prefetch,
                        true, true, last_blk ? k_blk : 0);
            }
            if (!is_const(kloop_it.update_C())
                    || to_cpp<bool>(kloop_it.update_C()))
                if_(kloop_it.update_C(), [&]() { store_C(); });
        });

        std::cout << "cool down: " << tail_k << "\n";

        for (int k_unroll_idx = 0; k_unroll_idx < tail_k;
                k_unroll_idx += k_unroll_blk) {
            k_body(k_unroll_idx, k_unroll_idx + strategy.ka_prefetch < tail_k,
                    k_unroll_idx + strategy.kb_prefetch < tail_k,
                    k_unroll_idx + k_load_offset < tail_k, true, 0);
        }

        store_C();

        return to_stmt();
    }

    const GEMMProblem &problem;
    const GEMMStrategy &strategy;
};

ir::stmt_t build_ir(const GEMMProblem &problem, const GEMMStrategy &strategy) {
    ir::stmt_t stmt = gemm_ir(problem, strategy).build();
    // stmt = stmt.simplify();
    // stmt = stmt.eliminate_common_subexprs();
    return stmt;
}

} // namespace gemmstone
