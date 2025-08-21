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
#include "common/dnnl_thread.hpp"
#include "cpu/rv64/rvv_postops.hpp"
#include <riscv_vector.h>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace matmul {

void rvv_matmul_colmajor(const float *src, const float *weights, float *dst,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, const float *bias,
        const memory_desc_wrapper &bias_d,
        const rvv_postops_t &postops_handler) {

    const int ndims = src_d.ndims();
    const dim_t *src_dims = src_d.dims();
    const dim_t *wei_dims = weights_d.dims();
    const int weights_ndims = weights_d.ndims();

    dim_t batch = 1;
    for (int i = 0; i < ndims - 2; ++i)
        batch *= src_dims[i];

    const dim_t M = src_dims[ndims - 2];
    const dim_t K = src_dims[ndims - 1];
    const dim_t N = wei_dims[weights_ndims - 1];

    dim_t weights_batch_size = 1;
    for (int i = 0; i < weights_ndims - 2; ++i)
        weights_batch_size *= wei_dims[i];
    const bool weights_are_broadcasted = (weights_batch_size == 1 && batch > 1);

    parallel_nd(batch, M, [&](dim_t b, dim_t m) {
        std::vector<dim_t> dst_idx_prefix(ndims - 1);
        if (ndims > 2) {
            utils::l_dims_by_l_offset(
                    dst_idx_prefix.data(), b, src_dims, ndims - 2);
        }
        dst_idx_prefix[ndims - 2] = m;

        size_t weights_batch_offset = 0;
        if (!weights_are_broadcasted) {
            for (int i = 0; i < weights_ndims - 2; ++i) {
                if (wei_dims[i] != 1) {
                    dim_t b_idx = dst_idx_prefix[i + (ndims - weights_ndims)];
                    weights_batch_offset
                            += b_idx * weights_d.blocking_desc().strides[i];
                }
            }
        }

        const float *src_base_ptr = src + (size_t)b * M * K + (size_t)m * K;
        float *dst_base_ptr = dst + (size_t)b * M * N + (size_t)m * N;
        const float *weights_base_ptr = weights + weights_batch_offset;

        for (dim_t n0 = 0; n0 < N;) {
            size_t vl = __riscv_vsetvl_e32m1(N - n0);
            std::vector<float> out_vals(vl, 0.0f);

            for (dim_t k0 = 0; k0 < K;) {
                size_t k_vl = __riscv_vsetvl_e32m1(K - k0);

                vfloat32m1_t src_vec
                        = __riscv_vle32_v_f32m1(src_base_ptr + k0, k_vl);

                for (size_t ni = 0; ni < vl; ++ni) {
                    const float *weight_col_ptr
                            = weights_base_ptr + (size_t)(n0 + ni) * (size_t)K;
                    vfloat32m1_t wei_vec
                            = __riscv_vle32_v_f32m1(weight_col_ptr + k0, k_vl);

                    vfloat32m1_t prod
                            = __riscv_vfmul_vv_f32m1(src_vec, wei_vec, k_vl);
                    vfloat32m1_t reduced = __riscv_vfredusum_vs_f32m1_f32m1(
                            prod, __riscv_vfmv_v_f_f32m1(0.0f, k_vl), k_vl);
                    float partial = __riscv_vfmv_f_s_f32m1_f32(reduced);

                    out_vals[ni] += partial;
                }

                k0 += k_vl;
            }

            vfloat32m1_t acc = __riscv_vle32_v_f32m1(out_vals.data(), vl);

            if (bias) {
                if (bias_d.nelems() == 1) {
                    acc = __riscv_vfadd_vf_f32m1(acc, bias[0], vl);
                } else {
                    const int dst_ndims = dst_d.ndims();
                    const int bias_ndims = bias_d.ndims();
                    const dim_t *bias_dims = bias_d.dims();

                    std::vector<size_t> bias_strides(bias_ndims);
                    bias_strides[bias_ndims - 1] = 1;
                    for (int d = bias_ndims - 2; d >= 0; --d)
                        bias_strides[d] = bias_strides[d + 1]
                                * (size_t)bias_dims[d + 1];

                    size_t base_bias_off = 0;
                    for (int d = 0; d < bias_ndims - 1; ++d) {
                        int dst_dim_idx = d + (dst_ndims - bias_ndims);
                        dim_t idx = (bias_dims[d] == 1)
                                ? 0
                                : dst_idx_prefix[dst_dim_idx];
                        base_bias_off += idx * bias_strides[d];
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
            __riscv_vse32_v_f32m1(&dst_base_ptr[n0], acc, vl);
            n0 += vl;
        }
    });
}

void rvv_matmul_rowmajor(const float *src, const float *weights, float *dst,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, const float *bias,
        const memory_desc_wrapper &bias_d,
        const rvv_postops_t &postops_handler) {

    const int ndims = src_d.ndims();
    const dim_t *src_dims = src_d.dims();
    const dim_t *wei_dims = weights_d.dims();
    const int weights_ndims = weights_d.ndims();

    dim_t batch = 1;
    for (int i = 0; i < ndims - 2; ++i)
        batch *= src_dims[i];

    const dim_t M = src_dims[ndims - 2];
    const dim_t K = src_dims[ndims - 1];
    const dim_t N = wei_dims[weights_ndims - 1];

    dim_t weights_batch_size = 1;
    for (int i = 0; i < weights_ndims - 2; ++i)
        weights_batch_size *= wei_dims[i];
    const bool weights_are_broadcasted = (weights_batch_size == 1 && batch > 1);

    parallel_nd(batch, M, [&](dim_t b, dim_t m) {
        std::vector<dim_t> dst_idx_prefix(ndims - 1);
        if (ndims > 2) {
            utils::l_dims_by_l_offset(
                    dst_idx_prefix.data(), b, src_dims, ndims - 2);
        }
        dst_idx_prefix[ndims - 2] = m;

        size_t weights_batch_offset = 0;
        if (!weights_are_broadcasted) {
            for (int i = 0; i < weights_ndims - 2; ++i) {
                if (wei_dims[i] != 1) {
                    dim_t b_idx = dst_idx_prefix[i + (ndims - weights_ndims)];
                    weights_batch_offset
                            += b_idx * weights_d.blocking_desc().strides[i];
                }
            }
        }

        const float *src_base_ptr = src + (size_t)b * M * K + (size_t)m * K;
        float *dst_base_ptr = dst + (size_t)b * M * N + (size_t)m * N;
        const float *weights_base_ptr = weights + weights_batch_offset;

        for (dim_t n0 = 0; n0 < N;) {
            size_t vl = __riscv_vsetvl_e32m1(N - n0);
            vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);

            for (dim_t k = 0; k < K; ++k) {
                vfloat32m1_t a_vec
                        = __riscv_vfmv_v_f_f32m1(src_base_ptr[k], vl);
                const float *b_ptr = weights_base_ptr + (size_t)k * N + n0;
                vfloat32m1_t b_vec = __riscv_vle32_v_f32m1(b_ptr, vl);
                acc = __riscv_vfmacc_vv_f32m1(acc, a_vec, b_vec, vl);
            }

            if (bias) {
                if (bias_d.nelems() == 1) {
                    acc = __riscv_vfadd_vf_f32m1(acc, bias[0], vl);
                } else {
                    const int dst_ndims = dst_d.ndims();
                    const int bias_ndims = bias_d.ndims();
                    const dim_t *bias_dims = bias_d.dims();

                    std::vector<size_t> bias_strides(bias_ndims);
                    bias_strides[bias_ndims - 1] = 1;
                    for (int d = bias_ndims - 2; d >= 0; --d)
                        bias_strides[d] = bias_strides[d + 1]
                                * (size_t)bias_dims[d + 1];

                    size_t base_bias_off = 0;
                    for (int d = 0; d < bias_ndims - 1; ++d) {
                        int dst_dim_idx = d + (dst_ndims - bias_ndims);
                        dim_t idx = (bias_dims[d] == 1)
                                ? 0
                                : dst_idx_prefix[dst_dim_idx];
                        base_bias_off += idx * bias_strides[d];
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
            __riscv_vse32_v_f32m1(&dst_base_ptr[n0], acc, vl);
            n0 += vl;
        }
    });
}

rvv_matmul_t::rvv_matmul_t(const pd_t *apd) : primitive_t(apd) {}

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
    if (pd()->is_col_major(weights_d)) {
        rvv_matmul_colmajor(src, weights, dst, src_d, weights_d, dst_d, bias,
                bias_d, postops_handler);
    } else {
        rvv_matmul_rowmajor(src, weights, dst, src_d, weights_d, dst_d, bias,
                bias_d, postops_handler);
    }

    return status::success;
}

} // namespace matmul
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl