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
#include "gpu/intel/jit/gemm/include/gemmstone/strategy.hpp"
#include "gpu/intel/jit/gemm/ir/kernel_desc.hpp"

namespace gemmstone {

using namespace ir::dsl;

const ir::pvar_t &m_var = ir::pvars::m;
const ir::pvar_t &n_var = ir::pvars::n;
const ir::pvar_t &k_var = ir::pvars::k;

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

transform_t get_plan(const MatrixAddressing &matrix_type,
        const MatrixAddressingStrategy &matrix_strategy,
        std::array<ir::pvar_t, 2> dims, bool is_prefetch = false) {
    switch (matrix_strategy.accessType) {
        case AccessType::Scattered:
            // TODO: Remove workaround unimplemented scattered->vnni support.
            if (is_prefetch)
                return transform_t(transform_t::kind_t::none, 0,
                        matrix_strategy.cachingR, dims);

            return transform_t(transform_t::kind_t::transpose_vnni,
                    matrix_strategy.tileR, matrix_strategy.cachingR, dims);

        case AccessType::ChannelScattered: stub(); return {};
        case AccessType::Block2DTranspose:
            return transform_t(transform_t::kind_t::transpose_vnni,
                    matrix_strategy.tileR, matrix_strategy.cachingR, dims);
        case AccessType::Block:
        case AccessType::PseudoBlock:
            return transform_t(transform_t::kind_t::none, matrix_strategy.tileR,
                    matrix_strategy.cachingR, dims);
        case AccessType::Block2D: {
            return transform_t(transform_t::kind_t::block,
                    matrix_strategy.tileR, matrix_strategy.cachingR, dims);
        };
        case AccessType::Block2DVNNI: {
            return transform_t(transform_t::kind_t::vnni, matrix_strategy.tileR,
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
            ir::pvar_t subgroup_dim, const int subgroup_size,
            const std::array<ir::expr_t, 3> &group_ids,
            const std::array<ir::expr_t, 3> &local_ids,
            const std::array<ir::expr_t, 3> &local_sizes)
        : m_idx_ {let("m_idx",
                (group_ids[0] * local_sizes[0] + local_ids[0])
                        * (m_blk
                                / (subgroup_dim == m_var ? subgroup_size : 1)))}
        , m_(m)
        , n_idx_ {let("n_idx",
                  (group_ids[1] * local_sizes[1] + local_ids[1])
                          * (n_blk
                                  / (subgroup_dim == n_var ? subgroup_size
                                                           : 1)))}
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
            const ir::icoord_t &coord) {
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
        : problem(desc.problem), strategy(desc.strategy) {
        gpu_assert(!strategy.kParallel) << "Unimplemented";
        gpu_assert(!strategy.kParallelLocal) << "Unimplemented";
        gpu_assert(!strategy.persistentLoop()) << "Unimplemented";
        gpu_assert(!strategy.slmA) << "Unimplemented";
        gpu_assert(!strategy.slmB) << "Unimplemented";
    }

    ir::stmt_t build(ir::kernel_iface_t iface, ir::ir_context_t &ctx) {
        declare_kernel(iface, ctx);

        const auto m = arg("m");
        const auto n = arg("n");
        const auto k = arg("k");

        auto m_blk = strategy.unroll[LoopM];
        auto n_blk = strategy.unroll[LoopN];
        auto k_blk = strategy.unroll[LoopK];

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

        gpu_assert(problem.Ta == problem.Ta_ext);
        auto A_prefetch_plan
                = get_plan(problem.A, strategy.A_prefetch, A_vars, true);
        auto A_load_plan = get_plan(problem.A, strategy.A, A_vars);

        gpu_assert(problem.Tb == problem.Tb_ext);
        auto B_prefetch_plan
                = get_plan(problem.B, strategy.B_prefetch, B_vars, true);
        auto B_load_plan = get_plan(problem.B, strategy.B, B_vars);

        ir::tile_t C_dims {{{m_var, m_blk}, {n_var, n_blk}}};
        auto C_store_plan = get_plan(problem.C, strategy.C, C_vars);

        tensor_t C = def(C_store_plan.get_layout(
                                 C_dims, into_ir(problem.Tc), gemm_var_desc),
                "C_blk", 0);

        basic_iterator_t kloop_it(m, n, k, m_blk, n_blk, k_blk, arg("A"),
                arg("offset_A"), into_ir(problem.Ta_ext),
                get_strides(problem.A.layout, A_vars, arg("lda")), A_pf_copies,
                strategy.A_copies, arg("B"), arg("offset_B"),
                into_ir(problem.Tb_ext),
                get_strides(problem.B.layout, B_vars, arg("ldb")), B_pf_copies,
                strategy.B_copies, arg("C"), arg("offset_C"),
                into_ir(problem.Tc_ext),
                get_strides(problem.C.layout, C_vars, arg("ldc")),
                C.layout.blocks()[0].dim, strategy.subgroupSize, group_ids(),
                local_ids(), local_sizes());

        auto store_C
                = [&]() { store(kloop_it.C_store(), C, C_store_plan, {}); };

        k_loop_config_t k_loop_main {m_blk, n_blk, k_blk, k_unroll_blk,
                k_blk - k_unroll_blk, strategy.ka_prefetch,
                strategy.kb_prefetch, kloop_it, A_prefetch_plan, A_load_plan,
                B_prefetch_plan, B_load_plan, C_store_plan, C};

        k_loop_config_t k_loop_short {m_blk, n_blk, k_unroll_blk, k_unroll_blk,
                0, 0, 0, kloop_it, A_prefetch_plan, A_load_plan,
                B_prefetch_plan, B_load_plan, C_store_plan, C};

        if (problem.A.alignment) {
            assume(arg("lda") % (problem.A.alignment / problem.Ta_ext) == 0);
        }
        if (problem.B.alignment) {
            assume(arg("ldb") % (problem.B.alignment / problem.Tb_ext) == 0);
        }
        if (problem.C.alignment) {
            assume(arg("ldc") % (problem.C.alignment / problem.Tc_ext) == 0);
        }

        // TODO: This needs moved inside the following if statements
        assume(arg("lda") >= (64 / problem.Ta_ext));
        assume(arg("ldb") >= (64 / problem.Ta_ext));
        assume(arg("ldc") >= (64 / problem.Ta_ext));
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
        transform_t A_prefetch_plan;
        transform_t A_load_plan;
        transform_t B_prefetch_plan;
        transform_t B_load_plan;
        transform_t C_store_plan;
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

        ir::tile_t A_dims {{{m_var, cfg.m_blk}, {k_var, cfg.k_blk}}};
        ir::tile_t B_dims {{{k_var, cfg.k_blk}, {n_var, cfg.n_blk}}};
        tensor_t A = def(cfg.A_load_plan.get_layout(A_dims,
                                 into_ir(problem.Ta_ext), gemm_var_desc),
                "A_blk");
        tensor_t B = def(cfg.B_load_plan.get_layout(B_dims,
                                 into_ir(problem.Tb_ext), gemm_var_desc),
                "B_blk");

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
                ir::tile_t tile = C.layout.int_dim_sizes();
                tile[k_var] = k_unroll_blk;
                mma(C, A, B, tile, {{k_var, k_offset}}, strategy.systolic);
                if (k_inc) kloop_it.inc_mma(k_inc);
            }
        };

        auto warmup_k = cfg.warmup_k();
        auto tail_k = cfg.tail_k();

        for (int k_unroll_idx = 0; k_unroll_idx < warmup_k;
                k_unroll_idx += k_unroll_blk) {
            bool last_blk = k_unroll_idx >= warmup_k - k_unroll_blk;
            k_body(k_unroll_idx, k_unroll_idx + cfg.ka_prefetch >= warmup_k,
                    k_unroll_idx + cfg.kb_prefetch >= warmup_k,
                    k_unroll_idx + cfg.k_load >= warmup_k, false,
                    last_blk ? cfg.warmup_k() - cfg.k_load : 0);
        }

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
