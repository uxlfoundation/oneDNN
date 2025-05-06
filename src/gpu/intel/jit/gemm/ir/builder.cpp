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

#include "gpu/intel/jit/gemm/ir/builder.hpp"
#include "gpu/intel/jit/gemm/ir/kernel_desc.hpp"
#include "gpu/intel/jit/v2/conv/plan.hpp"
#include "gpu/intel/jit/v2/ir/tensor.hpp"
namespace gemmstone {

struct tile_t {
    ir::expr_t buffer;
    ir::v2::layout_t layout;
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

    tile_t alloc_tile(ir::v2::layout_t layout, const std::string &name,
            ir::expr_t value = {}) {
        auto t = ir::type_t(
                layout.type().kind(), layout.type().elems() * layout.elems());
        return {alloc(t, name, value), layout};
    }

    std::vector<tile_t> alloc_tiles(ir::v2::layout_t layout,
            const std::string &name, int ntiles, ir::expr_t value = {}) {
        auto t = ir::type_t(
                layout.type().kind(), layout.type().elems() * layout.elems());
        std::vector<tile_t> ret;
        ret.reserve(ntiles);
        for (int i = 0; i < ntiles; i++) {
            ret.push_back({alloc(t, name + std::to_string(i), value), layout});
        }
        return ret;
    }

    template <typename F>
    void for_each(ir::expr_t var, ir::expr_t init, ir::expr_t bound,
            ir::expr_t step, F body) {
        push();
        body(var);
        auto body_stmt = pop();
        append(ir::for_t::make(var, init, bound, body_stmt, step));
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

        auto type_a = into_ir(problem.Ta, m_blk * k_blk);
        auto type_b = into_ir(problem.Tb, k_blk * n_blk);
        auto type_c = into_ir(problem.Tc, m_blk * n_blk);

        static const ir::pvar_map_t<char> pvars = {
                {ir::pvars::m, 'm'}, {ir::pvars::n, 'n'}, {ir::pvars::k, 'k'}};

        auto A_layout = ir::v2::layout_t(pvars, type_a.kind(), 0,
                {{ir::pvars::k, k_blk / strategy.A_copies, 1},
                        {ir::pvars::m, m_blk, k_blk / strategy.A_copies}});
        auto B_layout = ir::v2::layout_t(pvars, type_b.kind(), 0,
                {{ir::pvars::n, n_blk, 1},
                        {ir::pvars::k, k_blk / strategy.B_copies, n_blk}});
        auto C_layout = ir::v2::layout_t(pvars, type_c.kind(), 0,
                {{ir::pvars::n, n_blk, 1}, {ir::pvars::m, m_blk, n_blk}});

        std::cout << "A Layout: " << A_layout.str() << "\n";
        std::cout << "B Layout: " << B_layout.str() << "\n";
        std::cout << "C Layout: " << C_layout.str() << "\n";

        std::vector<tile_t> A = {alloc_tiles(A_layout, "A", strategy.A_copies)};
        std::vector<tile_t> B = {alloc_tiles(B_layout, "B", strategy.B_copies)};
        tile_t C = alloc_tile(C_layout, "C", 0);

        // Pipeline size
        int p_size = lcm(strategy.A_copies, strategy.B_copies);
        auto load_A = [&](int idx) {
            //int A_stride = 1;
            //if (idx % A_stride == 0 && idx * A_stride < strategy.A_copies)
            //    append(load(A));
        };

        auto load_B = [&](int idx) {
            //int B_stride = p_size / strategy.B_copies;
            //if (idx % B_stride == 0 && idx * B_stride < strategy.B_copies)
            //    append(load(A));
        };

        auto mma = [&](const tile_t &C, const tile_t &A, const tile_t &B) {
            ir::pvar_tile_t mnk_inst_tile = [&]() {
                if (strategy.systolic) {
                    stub();
                } else {
                    bool simd_n = C.layout.inner_block(ir::pvars::n)
                                    % strategy.fmaSIMD
                            == 0;
                    bool simd_m = C.layout.inner_block(ir::pvars::m)
                                    % strategy.fmaSIMD
                            == 0;
                    bool prefer_simd_m = simd_m
                            && to_cpp<int>(C.layout.stride(ir::pvars::m)) == 1;
                    if (simd_n && !prefer_simd_m) {
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
            auto simd = strategy.fmaSIMD;

            // TODO: Deduplicate this, it is copied from v2/conv/builder.cpp,
            // includes addition of c_stride to mad
            ir::pvar_tile_t sizes = A.layout.int_dim_sizes();
            auto b_sizes = B.layout.int_dim_sizes();
            for (auto &d : b_sizes) {
                if (sizes.has(d)) gpu_assert(sizes[d] == b_sizes[d]);
                sizes[d] = b_sizes[d];
            }

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
                    int c_stride = is_a_bcast
                            ? to_cpp<int>(C.layout.stride(ir::pvars::m))
                            : to_cpp<int>(C.layout.stride(ir::pvars::n));
                    fma_func = ir::mad_t::make(ir::hw_t(), C.layout.type(),
                            c_stride, simd, A.layout.type(), a_stride,
                            B.layout.type(), b_stride);
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
                        auto a_off = A.layout.offset_in_bytes(coord);
                        auto b_off = B.layout.offset_in_bytes(coord);
                        auto c_off = C.layout.offset_in_bytes(coord);
                        auto dst = C.buffer[c_off];
                        auto src1 = A.buffer[a_off];
                        auto src2 = B.buffer[b_off];
                        if (fma == ir::fma_kind_t::dpas) std::swap(src1, src2);
                        append(fma_func.call(
                                {dst, dst, std::move(src1), std::move(src2)}));
                    });
        };

        for (int load_idx = 0; load_idx < p_size - 1; load_idx++) {
            load_A(load_idx);
            load_B(load_idx);
        }

        ir::expr_t k_bound = k - k_blk;
        for_each(var(k.type(), "k_idx"), 0, k - k_blk, k_blk,
                [&](ir::expr_t k_idx) {
                    for (int mma_idx = 0; mma_idx < p_size; mma_idx++) {
                        int load_idx = (mma_idx ? p_size : mma_idx) - 1;
                        load_A(load_idx);
                        load_B(load_idx);
                        mma(C, A[mma_idx], B[mma_idx]);
                    }
                });

        for (int mma_idx = 0; mma_idx < p_size; mma_idx++) {
            mma(C, A[mma_idx], B[mma_idx]);
        }

        // append(store(C_ptr, C));

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
