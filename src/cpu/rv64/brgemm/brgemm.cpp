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

#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/rv64/brgemm/brgemm.hpp"
#include "cpu/rv64/brgemm/jit_brgemm_kernel.hpp"

#include "cpu/rv64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace dnnl::impl::utils;

status_t brgemm_desc_init(brgemm_desc_t *brg, cpu_isa_t isa,
        brgemm_batch_kind_t type, data_type_t dt_a, data_type_t dt_b,
        brgemm_layout_t layout, float alpha, float beta, dim_t LDA, dim_t LDB,
        dim_t LDC, dim_t M, dim_t N, dim_t K, const brgemm_strides_t *strides) {

    if (!brg) return status::invalid_arguments;
    if (M <= 0 || K <= 0) return status::invalid_arguments;

    // Supported:
    //   f32  × f32  → f32  (always)
    //   bf16 × bf16 → f32  (Zvfbfwma widening FMA)
    //   f16  × f16  → f32  (Zvfh widening FMA)
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

    // Determine bd_block from VLEN. Using LMUL=m4 so that each logical
    // vector register holds VLEN*4/32 f32 elements.
    const int vlen_f32 = get_platform_vlen() / 32; // elements per m1
    brg->bd_block = vlen_f32 * 4; // LMUL=m4 → 4× elements

    brg->bdb = static_cast<int>(M) / brg->bd_block;
    brg->bdb_tail = static_cast<int>(M) % brg->bd_block;

    brg->n_step = 4; // process 4 output columns per inner iteration
    brg->rd_block = 4; // K unroll factor
    brg->rdb = static_cast<int>(K) / brg->rd_block;
    brg->rdb_tail = static_cast<int>(K) % brg->rd_block;

    return status::success;
}

status_t brgemm_kernel_create(
        brgemm_kernel_t **brg_kernel, const brgemm_desc_t &brg) {
    if (!brg_kernel) return status::invalid_arguments;
    *brg_kernel = nullptr;

    // Pick the per-dtype kernel class.
    brgemm_kernel_t *kernel = nullptr;
    if (brg.is_f32) {
        kernel = new brgemm_kernel_common_t(brg);
    } else if (brg.dt_a == data_type::bf16) {
        kernel = new brgemm_kernel_bf16_t(brg);
    } else if (brg.dt_a == data_type::f16) {
        kernel = new brgemm_kernel_f16_t(brg);
    } else if (brg.is_int8) {
        kernel = new brgemm_kernel_s8_t(brg);
    } else {
        return status::unimplemented;
    }
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

    const auto *A_base = reinterpret_cast<const char *>(ptr_A);
    const auto *B_base = reinterpret_cast<const char *>(ptr_B);
    auto *C_base = reinterpret_cast<char *>(ptr_C);
    const auto *bias_base = reinterpret_cast<const char *>(ptr_bias);

    // K-blocking: split the reduction dimension into chunks of BK to keep
    // the A working-set inside the L1D cache.
    const dim_t BK = BRGEMM_BK;

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
            p.ptr_A = A_kb + static_cast<dim_t>(m) * bd * ts_a;
            p.ptr_C = C_base + static_cast<dim_t>(m) * bd * ts_c;
            p.ptr_bias = bias_kb
                    ? bias_kb + static_cast<dim_t>(m) * bd * ts_bias
                    : nullptr;
            p.M = bd;
            (*brg_kernel)(&p);
        }

        // Process M tail (if any).
        if (brg.bdb_tail > 0) {
            p.ptr_A = A_kb + static_cast<dim_t>(brg.bdb) * bd * ts_a;
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
