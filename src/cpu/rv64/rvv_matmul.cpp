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
#include "cpu/rv64/rvv_matmul_kernel.hpp"
#include "cpu/rv64/rvv_postops.hpp"
#include <riscv_vector.h>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace matmul {

void rvv_matmul_colmajor(const void *src, const void *weights, void *dst,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, const void *bias,
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

        const uint8_t *src_base_ptr = reinterpret_cast<const uint8_t *>(src)
                + ((size_t)b * M * K + (size_t)m * K)
                        * types::data_type_size(src_d.data_type());
        uint8_t *dst_base_ptr = reinterpret_cast<uint8_t *>(dst)
                + ((size_t)b * M * N + (size_t)m * N)
                        * types::data_type_size(dst_d.data_type());
        const uint8_t *weights_base_ptr
                = reinterpret_cast<const uint8_t *>(weights)
                + weights_batch_offset
                        * types::data_type_size(weights_d.data_type());

        rvv_matmul_colmajor_compute_kernel(src_base_ptr, weights_base_ptr,
                dst_base_ptr, K, N, bias, src_d, weights_d, dst_d, bias_d,
                dst_idx_prefix, const_cast<rvv_postops_t &>(postops_handler));
    });
}

void rvv_matmul_rowmajor(const void *src, const void *weights, void *dst,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, const void *bias,
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

        const uint8_t *src_base_ptr = reinterpret_cast<const uint8_t *>(src)
                + ((size_t)b * M * K + (size_t)m * K)
                        * types::data_type_size(src_d.data_type());
        uint8_t *dst_base_ptr = reinterpret_cast<uint8_t *>(dst)
                + ((size_t)b * M * N + (size_t)m * N)
                        * types::data_type_size(dst_d.data_type());
        const uint8_t *weights_base_ptr
                = reinterpret_cast<const uint8_t *>(weights)
                + weights_batch_offset
                        * types::data_type_size(weights_d.data_type());

        rvv_matmul_rowmajor_compute_kernel(src_base_ptr, weights_base_ptr,
                dst_base_ptr, K, N, bias, src_d, weights_d, dst_d, bias_d,
                dst_idx_prefix, const_cast<rvv_postops_t &>(postops_handler));
    });
}

rvv_matmul_t::rvv_matmul_t(const pd_t *apd) : primitive_t(apd) {}

status_t rvv_matmul_t::execute(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper bias_d(pd()->desc()->bias_desc);

    const post_ops_t &post_ops = pd()->attr()->post_ops_;
    rvv_postops_t postops_handler(post_ops);

    const void *bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
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