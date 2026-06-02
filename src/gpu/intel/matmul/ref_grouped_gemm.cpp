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

#include "gpu/intel/matmul/ref_grouped_gemm.hpp"

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY

#include "common/c_types_map.hpp"
#include "gpu/intel/compute/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

// Two grouped matmul patterns are supported:
// 2D grouped src (variable M) x 3D dense wei -> 2D grouped dst (variable M)
// 2D grouped src (variable K) x 2D grouped wei (variable M) -> dense 3D dst
status_t ref_grouped_t::execute(const exec_ctx_t &ctx) const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper wei_d(pd()->weights_md(0));
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const bool is_2dby2d = pd()->is_2dby2d();

    const auto &src_grouped = src_d.sparse_desc().grouped_desc;
    const dim_t group_count = pd()->group_count_;
    const dim_t N = dst_d.dims()[dst_d.ndims() - 1];
    const dim_t total_tokens = src_d.dims()[0];

    // Note, that below are constants for specific patterns
    const dim_t K_fixed
            = is_2dby2d ? 0 : wei_d.dims()[wei_d.ndims() - 2]; // 2Dx3D
    const dim_t M_fixed = is_2dby2d ? src_d.dims()[0] : 0; // 2Dx2D

    // Strides for accessing elements within inner GEMM
    //
    // 2Dx3D
    // row-major src [total_M, K]: stride_m = K, stride_k = 1
    // dense wei [G, K, N]:        stride_k = N, stride_n = 1
    // dst [total_M, N] row-major: stride_m = N, stride_n = 1
    //
    // 2Dx2D
    // col-major src [M, total_K]: stride_m = 1, stride_k = M
    // row-major wei [total_K, N]: stride_k = N, stride_n = 1
    // dst [G, M, N] row-major:    stride_m = N, stride_n = 1
    const auto src_strides = src_d.strides();
    const auto wei_strides = wei_d.strides();
    const auto dst_strides = dst_d.strides();
    const dim_t src_stride_m = src_strides[0];
    const dim_t src_stride_k = src_strides[1];
    const dim_t wei_stride_k = wei_strides[wei_d.ndims() - 2];
    const dim_t wei_stride_n = wei_strides[wei_d.ndims() - 1];
    const dim_t dst_stride_m = dst_strides[dst_d.ndims() - 2];
    const dim_t dst_stride_n = dst_strides[dst_d.ndims() - 1];

    // Strides to access different groups, i.e. base_g_ptr = g_start * g_stride
    //
    // 2Dx3D
    // row-major src [total_M, K]: stride = K
    // dense wei [G, K, N]:        stride = K * N
    // dst [total_M, N] row-major: stride = N
    //
    // 2Dx2D
    // col-major src [M, total_K]: stride = M
    // row-major wei [total_K, N]: stride = N
    // dst [G, M, N] row-major:    stride = M * N
    const dim_t src_group_stride = src_strides[src_grouped.variable_dim_idx];
    const dim_t wei_group_stride = wei_strides[0];
    const dim_t dst_group_stride = dst_strides[0];

    const auto &src_data = CTX_IN_STORAGE(DNNL_ARG_SRC, 0);
    const auto &src_offsets = CTX_IN_STORAGE(DNNL_ARG_SRC, 1);
    const auto &wei_data = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS, 0);
    auto &dst_data = CTX_OUT_STORAGE(DNNL_ARG_DST, 0);

    const bool with_bias = pd()->with_bias();
    const auto &attr_scales = pd()->attr()->scales_;
    const bool with_src_scales = !attr_scales.has_default_values(DNNL_ARG_SRC);
    const bool with_wei_scales
            = !attr_scales.has_default_values(DNNL_ARG_WEIGHTS);

    compute::kernel_arg_list_t arg_list;
    int arg_idx = 0;
    arg_list.set(arg_idx++, src_data);
    arg_list.set(arg_idx++, src_offsets);
    arg_list.set(arg_idx++, wei_data);
    // Tensor sharing src split: wei (2Dx2D) or dst (2Dx3D)
    if (is_2dby2d) {
        arg_list.set(arg_idx++, CTX_IN_STORAGE(DNNL_ARG_WEIGHTS, 1));
    } else {
        arg_list.set(arg_idx++, CTX_OUT_STORAGE(DNNL_ARG_DST, 1));
    }
    arg_list.set(arg_idx++, dst_data);
    arg_list.set(arg_idx++, (int)group_count);
    arg_list.set(arg_idx++, (int)is_2dby2d);
    arg_list.set(arg_idx++, M_fixed);
    arg_list.set(arg_idx++, K_fixed);
    arg_list.set(arg_idx++, N);
    arg_list.set(arg_idx++, src_stride_m);
    arg_list.set(arg_idx++, src_stride_k);
    arg_list.set(arg_idx++, wei_stride_k);
    arg_list.set(arg_idx++, wei_stride_n);
    arg_list.set(arg_idx++, dst_stride_m);
    arg_list.set(arg_idx++, dst_stride_n);
    arg_list.set(arg_idx++, src_group_stride);
    arg_list.set(arg_idx++, wei_group_stride);
    arg_list.set(arg_idx++, dst_group_stride);
    if (with_bias) arg_list.set(arg_idx++, CTX_IN_STORAGE(DNNL_ARG_BIAS));
    if (with_src_scales)
        arg_list.set(
                arg_idx++, CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC));
    if (with_wei_scales)
        arg_list.set(arg_idx++,
                CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS));
    // Post-ops apply to the 2Dx3D pattern only (grouped dst)
    if (pd()->with_post_op_) {
        const auto &po_chain = pd()->po_chain_;
        const memory_storage_t *grouped_scale
                = &memory_storage_t::empty_storage();
        const memory_storage_t *dense_scale
                = &memory_storage_t::empty_storage();
        const memory_storage_t *nvfp4_scale
                = &memory_storage_t::empty_storage();
        for (int i = 0; i < pd()->attr()->post_ops_.len(); ++i) {
            auto &e = pd()->attr()->post_ops_.entry_[i];
            if (!e.is_binary()) continue;
            const int po_arg
                    = DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1;
            if (po_chain[i] == po_kind_t::binary_grouped_scale) {
                grouped_scale = &CTX_IN_STORAGE(po_arg, 0);
            } else if (po_chain[i] == po_kind_t::binary_dense_scale) {
                dense_scale = &CTX_IN_STORAGE(po_arg, 0);
            } else if (po_chain[i] == po_kind_t::binary_nvfp4_scale) {
                nvfp4_scale = &CTX_IN_STORAGE(po_arg, 0);
            }
        }
        arg_list.set(arg_idx++, *grouped_scale);
        arg_list.set(arg_idx++, *dense_scale);
        arg_list.set(arg_idx++, *nvfp4_scale);
    }

    // Simple 3D dispatch for ref impl clarity
    compute::range_t gws
            = {(size_t)group_count, (size_t)total_tokens, (size_t)N};

    return parallel_for(ctx, compute::nd_range_t(gws), kernel_, arg_list);
}

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY
