/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "gpu/intel/matmul/sparse_ref.hpp"
#include "common/c_types_map.hpp"
#include "gpu/intel/compute/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

status_t ref_sparse_t::execute_ref(const exec_ctx_t &ctx) const {
    const auto a_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto b_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto c_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());

    bool is_src_sparse = a_d.is_sparse_desc();
    int sparse_arg = is_src_sparse ? DNNL_ARG_SRC : DNNL_ARG_WEIGHTS;
    int dense_arg = is_src_sparse ? DNNL_ARG_WEIGHTS : DNNL_ARG_SRC;

    const auto &sp_values = CTX_IN_STORAGE(sparse_arg, 0);
    const auto &sp_meta0 = CTX_IN_STORAGE(sparse_arg, 1);
    const auto &sp_meta1 = CTX_IN_STORAGE(sparse_arg, 2);
    const auto &dense = CTX_IN_STORAGE(dense_arg);
    auto &c = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const auto &sparse_d = is_src_sparse ? a_d : b_d;
    const dim_t nnz = sparse_d.nnz();

    const dim_t M = c_d.dims()[0];
    const dim_t N = c_d.dims()[1];

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, sp_values);
    arg_list.set(1, sp_meta0);
    arg_list.set(2, sp_meta1);
    arg_list.set(3, dense);
    arg_list.set(4, c);
    arg_list.set(5, nnz);
    compute::range_t gws = {(size_t)M, (size_t)N};

    auto nd_range = compute::nd_range_t(gws);

    status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);
    return status;
}

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
