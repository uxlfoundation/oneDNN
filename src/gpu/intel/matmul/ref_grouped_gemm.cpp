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

#include "common/c_types_map.hpp"
#include "gpu/intel/compute/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

status_t ref_grouped_gemm_t::execute_ref(const exec_ctx_t &ctx) const {
    // buffer 0: values, buffer 1: offsets
    const auto &src_data = CTX_IN_STORAGE(DNNL_ARG_SRC, 0);
    const auto &src_offsets = CTX_IN_STORAGE(DNNL_ARG_SRC, 1);
    const auto &wei_data = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &dst_data = CTX_OUT_STORAGE(DNNL_ARG_DST, 0);
    const auto &dst_offsets = CTX_OUT_STORAGE(DNNL_ARG_DST, 1);

    const auto *src_md = pd()->src_md();
    const auto *wei_md = pd()->weights_md(0);

    const dim_t num_groups = pd()->ngroups_;
    const dim_t total_tokens = src_md->dims[0];
    const dim_t N = wei_md->dims[2];

    const auto &attr_scales = pd()->attr()->scales_;
    const bool with_src_scales = !attr_scales.has_default_values(DNNL_ARG_SRC);
    const bool with_bias = pd()->with_bias();

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src_data);
    arg_list.set(1, src_offsets);
    arg_list.set(2, wei_data);
    arg_list.set(3, dst_data);
    arg_list.set(4, dst_offsets);
    arg_list.set(5, (int)num_groups);

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

    // Use total_tokens as upper bound for M dimension
    compute::range_t gws
            = {(size_t)num_groups, (size_t)total_tokens, (size_t)N};

    return parallel_for(ctx, compute::nd_range_t(gws), kernel_, arg_list);
}

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
