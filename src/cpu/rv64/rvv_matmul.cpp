/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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
#include "cpu/rv64/rvv_matmul.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"
#include "cpu/rv64/gemm/rvv_gemm_f32.hpp"
#include "cpu/rv64/rvv_postops.hpp"
#include <riscv_vector.h>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace matmul {

status_t rvv_matmul_t::execute(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
    auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper bias_d(pd()->desc()->bias_desc);

    const post_ops_t &post_ops = pd()->attr()->post_ops_;
    rvv_postops_t postops_handler(post_ops);

    const float *bias = CTX_IN_MEM(const float *, DNNL_ARG_BIAS);
    const int ndims = src_d.ndims();
    const int wei_ndims = weights_d.ndims();
    const dim_t *src_dims = src_d.dims();
    const dim_t *wei_dims = weights_d.dims();

    const dim_t batch = pd()->batch_;
    const dim_t M = pd()->M_;
    const dim_t K = pd()->K_;
    const dim_t N = pd()->N_;

    // row-major [M, K] / [M, N] are reinterpreted as column-major matrices and
    // mapped to GEMM as:
    //   C(N x M) = A(N x K) * B(K x M)
    // where:
    //   - A = W^T, with W logical layout [K, N] row-major. Since row-major
    //     [K, N] is equivalent in memory to column-major [N, K] (leading dim N),
    //     we pass transa = 'N' and lda = N.
    //   - B = src, where src logical layout [M, K] row-major is equivalent to
    //     column-major [K, M] with leading dim K, so transb = 'N', ldb = K.
    //   - C is viewed as column-major [N, M] with leading dim N, which matches
    //     row-major [M, N] in memory.
    char transa = 'N';
    char transb = 'N';
    dim_t M_gemm = N;
    dim_t N_gemm = M;
    dim_t K_gemm = K;
    dim_t lda = N; // weights: row-major [K, N] -> col-major [N, K]
    dim_t ldb = K; // src: row-major [M, K] -> col-major [K, M]
    dim_t ldc = N; // dst: row-major [M, N] -> col-major [N, M]
    float alpha = 1.0f;
    float beta = 0.0f;

    // batch strides for row-major src/dst
    const dim_t src_batch_stride = M * K;
    const dim_t dst_batch_stride = M * N;

    const int src_batch_ndims = ndims > 2 ? ndims - 2 : 0;
    const int wei_batch_ndims = wei_ndims > 2 ? wei_ndims - 2 : 0;
    const int batch_dim_shift = src_batch_ndims - wei_batch_ndims;
    const dim_t K_dim = wei_dims[wei_ndims - 2];
    const dim_t N_dim = wei_dims[wei_ndims - 1];
    const dim_t wei_matrix_stride = K_dim * N_dim;

    // GEMM compute
    if (pd()->weights_are_broadcast_) {
        //   C(N x (batch * M)) = A(N x K) * B(K x (batch * M))
        dim_t M_gemm_all = M_gemm; // N
        dim_t N_gemm_all = batch * N_gemm; // batch * M

        status_t st = rvv_gemm_f32(&transa, &transb, &M_gemm_all, &N_gemm_all,
                &K_gemm, &alpha, weights, &lda, src, &ldb, &beta, dst, &ldc,
                /*bias=*/nullptr);
        assert(st == status::success || st == status::unimplemented);
        MAYBE_UNUSED(st);

    } else {
        // one GEMM per logical batch
        parallel_nd(batch, [&](dim_t b) {
            const float *src_base = src + b * src_batch_stride;
            float *dst_base = dst + b * dst_batch_stride;

            dim_t batch_indices[DNNL_MAX_NDIMS] = {};
            if (src_batch_ndims > 0) {
                utils::l_dims_by_l_offset(
                        batch_indices, b, src_dims, src_batch_ndims);
            }

            dim_t weight_batch_index = 0;
            if (wei_batch_ndims > 0) {
                for (int d = 0; d < wei_batch_ndims; ++d) {
                    const int src_dim_idx = d + batch_dim_shift;
                    dim_t idx = (src_dim_idx >= 0) ? batch_indices[src_dim_idx]
                                                   : dim_t(0);
                    const dim_t wei_dim = wei_dims[d];
                    idx = (wei_dim == 1) ? dim_t(0) : idx;
                    weight_batch_index = weight_batch_index * wei_dim + idx;
                }
            }

            const float *wei_base
                    = weights + weight_batch_index * wei_matrix_stride;

            status_t st = rvv_gemm_f32(&transa, &transb, &M_gemm, &N_gemm,
                    &K_gemm, &alpha, wei_base, &lda, src_base, &ldb, &beta,
                    dst_base, &ldc,
                    /*bias=*/nullptr);
            assert(st == status::success || st == status::unimplemented);
            MAYBE_UNUSED(st);
        });
    }

    if (!bias && post_ops.len() == 0) return status::success;

    const int dst_ndims = dst_d.ndims();
    const int bias_ndims = bias_d.ndims();
    const dim_t *bias_dims = bias_d.dims();

    parallel_nd(batch, [&](dim_t b) {
        float *dst_base = dst + b * dst_batch_stride;

        dim_t dst_idx_prefix[DNNL_MAX_NDIMS] = {};
        size_t bias_strides[DNNL_MAX_NDIMS] = {};

        if (bias && bias_ndims > 1) {
            bias_strides[bias_ndims - 1] = 1;
            for (int d = bias_ndims - 2; d >= 0; --d)
                bias_strides[d]
                        = bias_strides[d + 1] * (size_t)bias_dims[d + 1];
        }

        for (dim_t m = 0; m < M; ++m) {
            if (ndims > 2) {
                utils::l_dims_by_l_offset(
                        dst_idx_prefix, b, src_dims, ndims - 2);
            }
            dst_idx_prefix[ndims - 2] = m;

            float *row_dst = dst_base + m * N;

            for (dim_t n0 = 0; n0 < N;) {
                size_t vl = __riscv_vsetvl_e32m1(N - n0);
                vfloat32m1_t acc = __riscv_vle32_v_f32m1(row_dst + n0, vl);

                if (bias) {
                    if (bias_d.nelems() == 1) {
                        acc = __riscv_vfadd_vf_f32m1(acc, bias[0], vl);
                    } else {
                        size_t base_bias_off = 0;
                        if (bias_ndims > 1) {
                            for (int d = 0; d < bias_ndims - 1; ++d) {
                                int dst_dim_idx = d + (dst_ndims - bias_ndims);
                                dim_t idx = (bias_dims[d] == 1)
                                        ? 0
                                        : dst_idx_prefix[dst_dim_idx];
                                base_bias_off += idx * bias_strides[d];
                            }
                        }

                        if (bias_dims[bias_ndims - 1] == 1) {
                            acc = __riscv_vfadd_vf_f32m1(
                                    acc, bias[base_bias_off], vl);
                        } else {
                            const float *bias_ptr = bias + base_bias_off + n0;
                            vfloat32m1_t bias_vec
                                    = __riscv_vle32_v_f32m1(bias_ptr, vl);
                            acc = __riscv_vfadd_vv_f32m1(acc, bias_vec, vl);
                        }
                    }
                }

                acc = postops_handler.apply(acc, vl);
                __riscv_vse32_v_f32m1(row_dst + n0, acc, vl);
                n0 += vl;
            }
        }
    });

    return status::success;
}

} // namespace matmul
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
