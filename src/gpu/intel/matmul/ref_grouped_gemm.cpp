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

status_t ref_grouped_t::execute_ref(const exec_ctx_t &ctx) const {
    // buffer 0: values, buffer 1: offsets
    const auto &src_data = CTX_IN_STORAGE(DNNL_ARG_SRC, 0);
    const auto &src_offsets = CTX_IN_STORAGE(DNNL_ARG_SRC, 1);
    const auto &wei_data = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &dst_data = CTX_OUT_STORAGE(DNNL_ARG_DST, 0);
    const auto &dst_offsets = CTX_OUT_STORAGE(DNNL_ARG_DST, 1);

    const auto *src_md = pd()->src_md();
    const auto *wei_md = pd()->weights_md(0);

    const dim_t group_count = pd()->group_count_;
    const dim_t total_tokens = src_md->dims[0];
    const dim_t N = wei_md->dims[2];

    const auto &attr_scales = pd()->attr()->scales_;
    const bool with_src_scales = !attr_scales.has_default_values(DNNL_ARG_SRC);
    const bool with_wei_scales
            = !attr_scales.has_default_values(DNNL_ARG_WEIGHTS);
    const bool with_bias = pd()->with_bias();

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src_data);
    arg_list.set(1, src_offsets);
    arg_list.set(2, wei_data);
    arg_list.set(3, dst_data);
    arg_list.set(4, dst_offsets);
    arg_list.set(5, (int)group_count);

    int next_arg = 6;
    if (with_bias) {
        const auto &bias_data = CTX_IN_STORAGE(DNNL_ARG_BIAS);
        arg_list.set(next_arg++, bias_data);
    }
    if (with_src_scales) {
        const auto &src_scales
                = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
        arg_list.set(next_arg++, src_scales);
    }
    if (with_wei_scales) {
        const auto &wei_scales
                = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
        arg_list.set(next_arg++, wei_scales);
    }
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
        arg_list.set(next_arg++, *grouped_scale);
        arg_list.set(next_arg++, *dense_scale);
        arg_list.set(next_arg++, *nvfp4_scale);
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
