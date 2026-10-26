/*******************************************************************************
* Copyright 2026 ZTE Corporation
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

#include <cstddef>
#include <memory>
#include <type_traits>

#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/rv64/brgemm/brgemm.hpp"
#include "cpu/rv64/rvjit/rvjit_matmul.hpp"

#include "cpu/rv64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace dnnl::impl::utils;

namespace {

static_assert(std::is_standard_layout<brgemm_kernel_params_t>::value,
        "the kernel reads its arguments at offsetof() their fields");

// Whether rvjit generates the brgemm described by `brg`
bool brgemm_uses_rvjit(const brgemm_desc_t &brg) {
    if (brg.dt_a != brg.dt_b) return false;
    if (brg.dt_a == data_type::s8) return brg.dt_c == data_type::s32;
    if (!utils::one_of(
                brg.dt_a, data_type::f32, data_type::bf16, data_type::f16))
        return false;

    // The accumulators are f32; f16 inputs may narrow them to an f16 C
    return brg.dt_c == data_type::f32
            || (brg.dt_c == data_type::f16 && brg.dt_a == data_type::f16);
}

// The matmul a descriptor stands for.
//
// The beta, the bias and the flags selecting them are read from the
// parameters of each call, so that one kernel serves every use of the
// descriptor. M is bounded by `m_bound` and K by one K block. N is bounded by
// `n_bound` when a call's rows are known, which only fits the kernel to a
// layout already chosen: the layout is chosen with N unbounded, so that it does
// not depend on the activation rows, which a caller may change between calls.
//
// An int8 brgemm multiplies the s32 A that the packing widens by the bytes of
// B, which the load extends to s32 as it reads them, in an s32 accumulation.
rvjit_matmul_desc_t make_desc(
        const brgemm_desc_t &brg, dim_t m_bound, dim_t n_bound = 0) {
    using p_t = brgemm_kernel_params_t;
    const bool int8 = brg.is_int8;
    rvjit_matmul_desc_t d;
    d.a = rvjit_type_t(int8 ? data_type::s32 : brg.dt_a);
    d.b = rvjit_type_t(brg.dt_b, int8 ? data_type::s32 : data_type::undef);
    d.c = rvjit_type_t(brg.dt_c, int8 ? data_type::s32 : data_type::f32);

    d.m = rvjit_value_t::at(offsetof(p_t, M), m_bound);
    d.n = n_bound > 0 ? rvjit_value_t::at(offsetof(p_t, N), n_bound)
                      : rvjit_value_t::at(offsetof(p_t, N));
    d.k = rvjit_value_t::at(offsetof(p_t, K),
            nstl::min<dim_t>(brg.reduce_dim, brg.get_k_block()));

    d.ptr_a = offsetof(p_t, ptr_A);
    d.ptr_b = offsetof(p_t, ptr_B);
    d.ptr_c = offsetof(p_t, ptr_C);

    // Leading dimensions are in bytes
    const int a_bytes = int8 ? brg.typesize_C : brg.typesize_A;
    d.lda = rvjit_value_t::literal(brg.LDA * a_bytes);
    d.ldb = rvjit_value_t::literal(brg.LDB * brg.typesize_B);
    d.ldc = rvjit_value_t::literal(brg.LDC * brg.typesize_C);

    // An integer beta is the literal 1 that the flag word of each call enables,
    // as the parameters hold a float. The int8 brgemm has no bias
    d.features = rvjit_beta | (int8 ? 0u : rvjit_bias);
    if (int8)
        d.beta = 1.0;
    else
        d.beta_arg = offsetof(p_t, beta);
    d.ptr_bias = offsetof(p_t, ptr_bias);
    if (brg.alpha != 1.0f) {
        d.features |= rvjit_alpha;
        d.alpha = static_cast<double>(brg.alpha);
    }
    d.rt_flags = offsetof(p_t, rt_flags);
    return d;
}

// The M tile of one vector group of f32 accumulators at the default LMUL (m4)
int default_tile() {
    return static_cast<int>(get_platform_vlen() / 8);
}

// Decides the blocks of `brg` and records them in it
//
// M is bounded by the tile of the weights when it is given, else by the default
// tile unless the weights may be laid out by a wider one. Returns
// status::unimplemented when rvjit cannot take the problem
status_t brgemm_tune(brgemm_desc_t *brg) {
    if (!brgemm_uses_rvjit(*brg)) return status::unimplemented;
    if (brg->bcast_dim <= 0 || brg->reduce_dim <= 0)
        return status::unimplemented;

    dim_t m_bound = brg->bcast_dim;
    if (brg->wei_tile)
        m_bound = nstl::min<dim_t>(m_bound, brg->wei_tile);
    else if (!brg->wide_tile)
        m_bound = nstl::min<dim_t>(m_bound, default_tile());

    // The blocks of the whole problem, which the kernel is then pinned to
    CHECK(rvjit_matmul_tune(
            make_desc(*brg, m_bound, brg->n_bound), brg->blocks));
    brg->bd_block = brg->blocks.m_block();
    if (brg->wei_tile) {
        // The kernel's tile must divide the weights' tile
        if (brg->wei_tile % brg->bd_block != 0) return status::unimplemented;
        brg->bd_block = brg->wei_tile;
    }
    brg->bdb = static_cast<int>(brg->bcast_dim / brg->bd_block);
    brg->bdb_tail = static_cast<int>(brg->bcast_dim % brg->bd_block);
    brg->stride_a_m_block = brg->bd_block;
    return status::success;
}

// The brgemm kernel generated by rvjit
//
// Its blocks are the ones brgemm_tune recorded in the descriptor, which are
// pinned back into the kernel so that both decisions cannot disagree.
struct brgemm_kernel_rvjit_t : public brgemm_kernel_t {
    brgemm_kernel_rvjit_t(const brgemm_desc_t &brg) : brg_(brg) {}

    status_t create_kernel() override {
        return rvjit_matmul_kernel_create(
                make_desc(brg_, brg_.bd_block), brg_.blocks, kernel_);
    }

    void operator()(brgemm_kernel_params_t *p) const override {
        const bool beta = !brg_.is_int8 || p->beta != 0.0f;
        const bool bias = p->ptr_bias && !brg_.is_int8;
        p->rt_flags = rvjit_runtime_flags(brg_.alpha != 1.0f, beta, bias);
        (*kernel_)(p);
    }

    const brgemm_desc_t &get_brg() const override { return brg_; }

private:
    brgemm_desc_t brg_;
    std::unique_ptr<rvjit_matmul_kernel_t> kernel_;
};

} // namespace

status_t brgemm_desc_init(brgemm_desc_t *brg, cpu_isa_t isa,
        brgemm_batch_kind_t type, data_type_t dt_a, data_type_t dt_b,
        brgemm_layout_t layout, float alpha, float beta, dim_t LDA, dim_t LDB,
        dim_t LDC, dim_t M, dim_t N, dim_t K, const brgemm_strides_t *strides,
        bool wide_tile) {

    if (!brg) return status::invalid_arguments;
    if (M <= 0 || K <= 0) return status::invalid_arguments;

    // Supported:
    //   f32  × f32  → f32  (always)
    //   bf16 × bf16 → f32  (Zvfbfwma widening FMA)
    //   f16  × f16  → f32  (Zvfh widening FMA; f16 dt_c narrows on store)
    //   s8   × s8   → s32  (always)
    const bool is_f32 = everyone_is(data_type::f32, dt_a, dt_b);
    const bool is_bf16
            = everyone_is(data_type::bf16, dt_a, dt_b) && mayiuse(zvfbfwma);
    const bool is_f16
            = everyone_is(data_type::f16, dt_a, dt_b) && mayiuse(zvfh);
    const bool is_int8 = everyone_is(data_type::s8, dt_a, dt_b);
    if (!is_f32 && !is_bf16 && !is_f16 && !is_int8)
        return status::unimplemented;

    *brg = utils::zero<brgemm_desc_t>();

    brg->bcast_dim = M;
    brg->load_dim = N;
    brg->reduce_dim = K;
    brg->LDA = LDA;
    brg->LDB = LDB;
    brg->LDC = LDC;
    brg->alpha = alpha;
    brg->beta = beta;
    brg->type = type;
    brg->layout = layout;
    brg->isa_impl = isa;
    brg->wide_tile = wide_tile;

    brg->dt_a = dt_a;
    brg->dt_b = dt_b;
    brg->dt_c = is_int8 ? data_type::s32 : data_type::f32;
    brg->typesize_A = static_cast<int>(types::data_type_size(dt_a));
    brg->typesize_B = static_cast<int>(types::data_type_size(dt_b));
    brg->typesize_C = static_cast<int>(types::data_type_size(brg->dt_c));
    brg->is_f32 = is_f32;
    brg->is_int8 = is_int8;

    if (strides) {
        brg->stride_a = strides->stride_a;
        brg->stride_b = strides->stride_b;
    }

    // The kernels generated by rvjit choose their own blocks. Rejecting here
    // skips the primitive, instead of failing once its kernel is created.
    return brgemm_tune(brg);
}

status_t brgemm_desc_set_weights_tile(
        brgemm_desc_t *brg, int tile, dim_t n_bound) {
    if (!brg || tile <= 0) return status::invalid_arguments;
    brg->wei_tile = tile;
    brg->n_bound = n_bound;
    return brgemm_tune(brg);
}

status_t brgemm_desc_set_dt_c(brgemm_desc_t *brg, data_type_t dt_c) {
    if (!brg) return status::invalid_arguments;
    brg->dt_c = dt_c;
    brg->typesize_C = static_cast<int>(types::data_type_size(dt_c));
    return brgemm_tune(brg);
}

status_t brgemm_kernel_create(
        brgemm_kernel_t **brg_kernel, const brgemm_desc_t &brg) {
    if (!brg_kernel) return status::invalid_arguments;
    *brg_kernel = nullptr;

    if (!brgemm_uses_rvjit(brg)) return status::unimplemented;
    brgemm_kernel_t *kernel = new brgemm_kernel_rvjit_t(brg);
    status_t st = kernel->create_kernel();
    if (st != status::success) {
        delete kernel;
        return st;
    }
    *brg_kernel = kernel;
    return status::success;
}

void brgemm_kernel_destroy(brgemm_kernel_t *brg_kernel) {
    delete brg_kernel;
}

void brgemm_kernel_execute(const brgemm_kernel_t *brg_kernel, const void *ptr_A,
        const void *ptr_B, void *ptr_C, dim_t N, float beta,
        const void *ptr_bias) {

    const auto &brg = brg_kernel->get_brg();

    // int8 A is pre-widened to s32 during packing (4 bytes/elem); dt_c is also
    // s32, so typesize_C gives the correct packed-A stride.
    const int ts_a = brg.is_int8 ? brg.typesize_C : brg.typesize_A;
    const int ts_b = brg.typesize_B;
    const int ts_c = brg.typesize_C;
    const int ts_bias = static_cast<int>(sizeof(float)); // bias is f32
    const int bd = brg.bd_block;
    const dim_t K = brg.reduce_dim;
    const dim_t LDA_bytes = brg.LDA * ts_a;
    const dim_t A_tile_bytes = brg.stride_a_m_block * ts_a;

    const auto *A_base = reinterpret_cast<const char *>(ptr_A);
    const auto *B_base = reinterpret_cast<const char *>(ptr_B);
    auto *C_base = reinterpret_cast<char *>(ptr_C);
    const auto *bias_base = reinterpret_cast<const char *>(ptr_bias);

    const dim_t BK = brg.get_k_block();

    for (dim_t kb = 0; kb < K; kb += BK) {
        const dim_t K_inner = nstl::min(BK, K - kb);
        const float beta_kb = (kb == 0) ? beta : 1.0f;

        const char *A_kb = A_base + kb * LDA_bytes;
        const char *B_kb = B_base + kb * ts_b;
        const char *bias_kb = (kb == 0) ? bias_base : nullptr;

        brgemm_kernel_params_t p;
        p.ptr_B = B_kb;
        p.N = N;
        p.K = K_inner;
        p.beta = beta_kb;

        // Process full bd_block tiles.
        for (int m = 0; m < brg.bdb; m++) {
            p.ptr_A = A_kb + static_cast<dim_t>(m) * A_tile_bytes;
            p.ptr_C = C_base + static_cast<dim_t>(m) * bd * ts_c;
            p.ptr_bias = bias_kb
                    ? bias_kb + static_cast<dim_t>(m) * bd * ts_bias
                    : nullptr;
            p.M = bd;
            (*brg_kernel)(&p);
        }

        // Process M tail (if any).
        if (brg.bdb_tail > 0) {
            p.ptr_A = A_kb + static_cast<dim_t>(brg.bdb) * A_tile_bytes;
            p.ptr_C = C_base + static_cast<dim_t>(brg.bdb) * bd * ts_c;
            p.ptr_bias = bias_kb
                    ? bias_kb + static_cast<dim_t>(brg.bdb) * bd * ts_bias
                    : nullptr;
            p.M = brg.bdb_tail;
            (*brg_kernel)(&p);
        }
    }
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
