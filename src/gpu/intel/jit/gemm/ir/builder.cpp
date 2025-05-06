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

struct send_plan_t {
    static send_plan_t make_load_2d(ir::type_t type, bool vnni, bool transpose,
            ngen::CacheSettingsLSC cache_hint, std::array<ir::pvar_t, 2> dims) {
        send_plan_t plan {};

        plan.op = ir::send_op_t::load_2d;
        plan.type = type;
        plan.cache_hint = to_ir(cache_hint);
        plan.vnni = vnni;
        plan.transpose = transpose;
        plan.dims = dims;
        return plan;
    }

    ir::pvar_tile_t get_tile() const {
        if (op == ir::send_op_t::load_2d) {
            auto width = vnni || transpose ? 64 / 4 : 64 / type.size();
            auto height = 32;
            if (transpose) return {{dims[1], width}, {dims[0], height}};
            return {{dims[0], width}, {dims[1], height}};
        }
        stub();
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
    ir::type_t type;
    ir::send_cache_hint_t cache_hint;
    std::array<ir::pvar_t, 2> dims;
    union {
        // 2D Operation
        struct {
            bool vnni;
            bool transpose;
        };
    };
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
    send_plan_t plan;
};

struct global_tensor_t {
    ir::expr_t buffer;
    ir::type_t type;
    ir::pvar_map_t<ir::expr_t> dim_idxs;
    ir::pvar_map_t<ir::expr_t> strides;

    ir::expr_t get_header(const ir::pvar_coord_t<int64_t> &coord) const {
        ir::expr_t ret = ir::cast_t::make(ir::type_t::u64(), buffer);
        for (auto &c : coord) {
            ret += (dim_idxs[c] + coord[c]) * strides[c] * type.size();
        }
        return ir::simplify(ret);
    }
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

    tensor_t alloc_tensor(ir::v2::layout_t layout, send_plan_t plan,
            const std::string &name, ir::expr_t value = {}) {
        auto t = ir::type_t(
                layout.type().kind(), layout.type().elems() * layout.elems());
        return {alloc(t, name, value), layout, plan};
    }

    void load(const tensor_t &t, const global_tensor_t &g,
            const ir::pvar_coord_t<int64_t> &base) {
        auto tile = t.plan.get_tile();
        auto sizes = t.layout.int_dim_sizes();
        printf("tile: %s, sizes: %s\n", tile.str().c_str(),
                sizes.str().c_str());
        std::vector<ir::pvar_t> dim_order = {t.plan.dims[0], t.plan.dims[1]};
        ir::v2::for_each(
                sizes, tile, [&](const ir::pvar_coord_t<int64_t> &coord) {
                    auto t_off = t.layout.offset_in_bytes(base + coord);
                    auto buf = t.buffer[t_off];
                    auto width = std::min(tile[t.plan.dims[0]],
                            sizes[t.plan.dims[0]] - coord[t.plan.dims[0]]);
                    auto height = std::min(tile[t.plan.dims[1]],
                            sizes[t.plan.dims[1]] - coord[t.plan.dims[1]]);
                    auto header = alloc(
                            ir::type_t::u64(), "h", g.get_header(base + coord));
                    auto send_func = ir::send_t::make_2d({}, t.plan.op,
                            t.plan.type, width, height, 1, t.plan.vnni,
                            t.plan.transpose,
                            /*zero_out=*/true);
                    append(send_func.as<ir::send_t>()(
                            g.buffer, header, buf, {}));
                });
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

layout_t get_layout(const MatrixAddressing &matrix_type,
        const MatrixAddressingStrategy &matrix_strategy) {
    if (matrix_type.layout == MatrixLayout::Pc
            || matrix_type.layout == MatrixLayout::Pr)
        stub();

    std::cout << to_str(matrix_strategy.accessType) << "\n";
    switch (matrix_strategy.accessType) {
        case AccessType::Scattered:
        case AccessType::ChannelScattered:
            return isColMajor(matrix_type.layout) ? layout_t::rowMajor
                                                  : layout_t::colMajor;
        case AccessType::Block2DTranspose:
            return isColMajor(matrix_type.layout) ? layout_t::rowPackedVNNI
                                                  : layout_t::colPackedVNNI;
        case AccessType::Block:
        case AccessType::PseudoBlock:
            return isColMajor(matrix_type.layout) ? layout_t::colMajor
                                                  : layout_t::rowMajor;
        case AccessType::Block2D:
            return isColMajor(matrix_type.layout) ? layout_t::colPacked
                                                  : layout_t::rowPacked;
        case AccessType::Block2DVNNI:
            return isColMajor(matrix_type.layout) ? layout_t::colPackedVNNI
                                                  : layout_t::rowPackedVNNI;
        default: stub(); return layout_t::colMajor;
    }
};

send_plan_t get_load_plan(const MatrixAddressing &matrix_type,
        const MatrixAddressingStrategy &matrix_strategy, ir::type_t t,
        std::array<ir::pvar_t, 2> dims) {
    if (matrix_type.layout == MatrixLayout::Pc
            || matrix_type.layout == MatrixLayout::Pr)
        stub();

    switch (matrix_strategy.accessType) {
        case AccessType::Scattered: stub(); return {};
        case AccessType::ChannelScattered: stub(); return {};
        case AccessType::Block2DTranspose:
            return send_plan_t::make_load_2d(
                    t, false, true, matrix_strategy.cachingR, dims);
        case AccessType::Block: stub(); return {};
        case AccessType::PseudoBlock: stub(); return {};
        case AccessType::Block2D:
            return send_plan_t::make_load_2d(
                    t, false, false, matrix_strategy.cachingR, dims);
        case AccessType::Block2DVNNI:
            return send_plan_t::make_load_2d(
                    t, true, false, matrix_strategy.cachingR, dims);
        default: stub(); return {};
    }
};

struct dim_t {
    ir::pvar_t pvar;
    int size;
};

global_tensor_t get_global_tensor(ir::expr_t buffer, ir::type_t type,
        std::array<ir::pvar_t, 2> pvars, std::array<ir::expr_t, 2> vars,
        ir::expr_t ld, const MatrixAddressing matrix_type) {
    ir::pvar_map_t<ir::expr_t> strides = [&]() -> ir::pvar_map_t<ir::expr_t> {
        switch (matrix_type.layout) {
            case MatrixLayout::N: return {{pvars[0], 1}, {pvars[1], ld}};
            case MatrixLayout::T: return {{pvars[0], ld}, {pvars[1], 1}};
            default: stub(); return {};
        };
    }();

    return {buffer, type, {{pvars[0], vars[0]}, {pvars[1], vars[1]}}, strides};
}

ir::v2::layout_t get_layout(const MatrixAddressing &matrix_type,
        const MatrixAddressingStrategy matrix_strategy,
        const ir::v2::layout_desc_t &desc, ir::type_t t, dim_t row, dim_t col) {
    const int grf_size = 64;
    switch (get_layout(matrix_type, matrix_strategy)) {
        case layout_t::colMajor:
            return ir::v2::layout_t(desc, t, 0,
                    {{col.pvar, col.size, 1}, {row.pvar, row.size, col.size}});
        case layout_t::rowMajor:
            return ir::v2::layout_t(desc, t, 0,
                    {{row.pvar, row.size, 1}, {col.pvar, col.size, row.size}});
        case layout_t::colPacked: {
            int col_inner = grf_size / t.scalar().size();
            int col_outer = col.size / col_inner;
            if (col_outer == 1) {
                return ir::v2::layout_t(desc, t, 0,
                        {{col.pvar, col.size, 1},
                                {row.pvar, row.size, col.size}});
            }
            return ir::v2::layout_t(desc, t, 0,
                    {{col.pvar, col_inner, 1}, {row.pvar, row.size, col_inner},
                            {col.pvar, col_outer, row.size * col_inner}});
        }
        case layout_t::rowPacked: {
            int row_inner = grf_size / t.scalar().size();
            int row_outer = row.size / row_inner;
            if (row_outer == 1) {
                return ir::v2::layout_t(desc, t, 0,
                        {{row.pvar, row.size, 1},
                                {col.pvar, col.size, row.size}});
            }
            return ir::v2::layout_t(desc, t, 0,
                    {{row.pvar, row_inner, 1}, {col.pvar, col.size, row_inner},
                            {row.pvar, row_outer, col.size * row_inner}});
        }
        case layout_t::colPackedVNNI: {
            gpu_assert(t.scalar().size() < 4);
            int row_inner = 4 / t.scalar().size();
            int row_outer = row.size / row_inner;
            int col_inner = grf_size / 4;
            int col_outer = col.size / col_inner;
            return ir::v2::layout_t(desc, t, 0,
                    {{row.pvar, row_inner, 1}, {col.pvar, col_inner, row_inner},
                            {row.pvar, row_outer, col_inner * row_inner},
                            {col.pvar, col_outer,
                                    row_outer * col_inner * row_inner}});
        }
        case layout_t::rowPackedVNNI: {
            gpu_assert(t.scalar().size() < 4);
            int col_inner = 4 / t.scalar().size();
            int col_outer = col.size / col_inner;
            int row_inner = grf_size / 4;
            int row_outer = row.size / row_inner;
            return ir::v2::layout_t(desc, t, 0,
                    {{col.pvar, col_inner, 1}, {row.pvar, row_inner, col_inner},
                            {col.pvar, col_outer, row_inner * col_inner},
                            {row.pvar, row_outer,
                                    col_outer * row_inner * col_inner}});
        }
    }
}

struct mnk_iterator_t {
    ir::expr_t m_idx;
    ir::expr_t n_idx;
    ir::expr_t k_idx;
    ir::expr_t k;
    ir::expr_t k_last;
    int k_blk;
    static constexpr bool has_multi_C_tile = false;

    scope_builder_t &scope;

    void operator+=(int k_offset) { scope.assign(k_idx, k_idx + k_offset); }

    ir::expr_t is_valid() { return k_idx < k_last; };

    ir::expr_t update_C(int k_unroll_idx) { return k_idx >= k - k_unroll_idx; };
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

        static const ir::v2::layout_desc_t var_names {{{ir::pvars::m, 'm'},
                {ir::pvars::n, 'n'}, {ir::pvars::k, 'k'}}};

        gpu_assert(strategy.ka_prefetch == strategy.kb_prefetch
                || strategy.ka_prefetch == 0 || strategy.kb_prefetch == 0);

        // Pipeline size
        int p_size = lcm(strategy.A_copies, strategy.B_copies);
        auto k_unroll_blk = k_blk / p_size;
        auto k_prefetch_offset
                = std::max(strategy.ka_prefetch, strategy.kb_prefetch);
        auto k_load_offset = k_blk - k_unroll_blk;

        gpu_assert(k_prefetch_offset == 0 || k_prefetch_offset > k_load_offset);

        mnk_iterator_t mnk_prefetch = {m, n, alloc(k.type(), "k_prefetch", 0),
                k, (k - k_prefetch_offset), k_blk, *this};

        mnk_iterator_t mnk_load = {m, n, alloc(k.type(), "k_load", 0), k,
                k_load_offset == 0 ? k : (k - k_unroll_blk), k_blk, *this};

        mnk_iterator_t mnk_mma = {m, n, alloc(k.type(), "k_mma", 0), k,
                k - k_unroll_blk, k_blk, *this};

        gpu_assert(problem.Ta == problem.Ta_ext);
        std::array<ir::pvar_t, 2> A_pvars = {ir::pvars::m, ir::pvars::k};
        std::array<ir::expr_t, 2> A_vars = {mnk_load.m_idx, mnk_load.k_idx};
        auto A_tensor = get_global_tensor(interface.find_arg("A_ptr"),
                into_ir(problem.Ta_ext), A_pvars, A_vars,
                interface.find_arg("lda"), problem.A);
        auto A_layout = get_layout(problem.A, strategy.A, var_names,
                into_ir(problem.Ta_ext),
                {ir::pvars::k, k_blk / strategy.A_copies},
                {ir::pvars::m, m_blk});
        A_layout.add_block(ir::pvars::k, strategy.A_copies);
        auto A_plan = get_load_plan(
                problem.A, strategy.A, into_ir(problem.Ta_ext), A_pvars);

        gpu_assert(problem.Tb == problem.Tb_ext);
        std::array<ir::pvar_t, 2> B_pvars = {ir::pvars::k, ir::pvars::n};
        std::array<ir::expr_t, 2> B_vars = {mnk_load.k_idx, mnk_load.n_idx};
        auto B_tensor = get_global_tensor(interface.find_arg("B_ptr"),
                into_ir(problem.Tb_ext), B_pvars, B_vars,
                interface.find_arg("ldb"), problem.B);
        auto B_layout = get_layout(problem.B, strategy.B, var_names,
                into_ir(problem.Tb), {ir::pvars::n, n_blk},
                {ir::pvars::k, k_blk / strategy.B_copies});
        B_layout.add_block(ir::pvars::k, strategy.B_copies);
        auto B_plan = get_load_plan(
                problem.B, strategy.B, into_ir(problem.Tb_ext), B_pvars);

        auto C_layout = get_layout(problem.C, strategy.C, var_names,
                into_ir(problem.Tc), {ir::pvars::n, n_blk},
                {ir::pvars::m, m_blk});

        std::cout << "A Layout: " << A_layout.str() << "\n";
        std::cout << "B Layout: " << B_layout.str() << "\n";
        std::cout << "C Layout: " << C_layout.str() << "\n";

        tensor_t A = alloc_tensor(A_layout, A_plan, "A");
        tensor_t B = alloc_tensor(B_layout, B_plan, "B");
        tensor_t C = alloc_tensor(C_layout, {}, "C", 0);

        auto prefetch_A = [&](const mnk_iterator_t &it, int k_unroll_idx) {};
        auto load_A = [&](const mnk_iterator_t &it, int k_unroll_idx) {
            if (k_unroll_idx % (k_blk / strategy.A_copies) != 0) return;
            load(A, A_tensor, {{ir::pvars::k, k_unroll_idx}});
        };

        auto prefetch_B = [&](const mnk_iterator_t &it, int k_unroll_idx) {};
        auto load_B = [&](const mnk_iterator_t &it, int k_unroll_idx) {
            if (k_unroll_idx % (k_blk / strategy.B_copies) != 0) return;
            load(B, B_tensor, {{ir::pvars::k, k_unroll_idx}});
        };
        auto store_C = [&](const mnk_iterator_t &it) {};

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

        auto k_body = [&](int k_offset, bool prefetch, bool load, bool do_mma,
                              int k_inc) {
            if (prefetch) {
                prefetch_A(mnk_prefetch, k_offset);
                prefetch_B(mnk_prefetch, k_offset);
                if (k_inc) mnk_prefetch += k_inc;
            }

            if (load) {
                load_A(mnk_load, k_offset);
                load_B(mnk_load, k_offset);
                if (k_inc) mnk_load += k_inc;
            }

            if (do_mma) {
                mma(k_offset);
                if (k_inc) mnk_mma += k_inc;
            }
        };

        auto warmup_k = std::max(k_load_offset, k_prefetch_offset);
        auto tail_k = (warmup_k / k_blk) * k_blk + ((-warmup_k) % k_blk);

        for (int k_unroll_idx = 0;
                k_unroll_idx < k_prefetch_offset - k_load_offset;
                k_unroll_idx += k_unroll_blk) {
            bool last_blk = k_unroll_idx
                    >= k_prefetch_offset - k_load_offset - k_unroll_blk;
            k_body(k_unroll_idx, k_prefetch_offset > 0, false, false,
                    last_blk ? k_prefetch_offset - k_load_offset : 0);
        }

        for (int k_unroll_idx = 0; k_unroll_idx < k_load_offset;
                k_unroll_idx += k_unroll_blk) {
            bool last_blk = k_unroll_idx >= k_load_offset - k_unroll_blk;
            k_body(k_unroll_idx, k_prefetch_offset > 0, true, false,
                    last_blk ? k_load_offset : 0);
        }

        while_(k_prefetch_offset > 0 ? mnk_prefetch.is_valid()
                                     : mnk_load.is_valid(),
                [&]() {
                    for (int k_unroll_idx = 0; k_unroll_idx < k_blk;
                            k_unroll_idx += k_unroll_blk) {
                        bool last_blk = k_unroll_idx >= k_blk - k_unroll_blk;
                        k_body(k_unroll_idx, k_prefetch_offset > 0, true, true,
                                last_blk ? k_blk : 0);
                    }
                    if (mnk_mma.has_multi_C_tile)
                        if_(mnk_mma.update_C(0), [&]() { store_C(mnk_mma); });
                });

        for (int k_unroll_idx = 0; k_unroll_idx < tail_k;
                k_unroll_idx += k_unroll_blk) {
            k_body(k_unroll_idx, false, tail_k - k_unroll_idx >= k_load_offset,
                    true, k_unroll_idx % k_blk == 0 ? k_blk : 0);
        }

        store_C(mnk_mma);

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
