/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

#include "gpu/intel/matmul/gemm.hpp"
#include "gpu/intel/gemm/config.hpp"
#include "gpu/intel/gemm/primitive.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

status_t gemm_t::execute(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC);
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS);
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST);
    const auto bia_d = ctx.memory_mdw(DNNL_ARG_BIAS);

    memory_storage_t *a0
            = &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);

    memory_storage_t *b0
            = &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS);

    memory_storage_t *c0
            = &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);

    gemm::exec_args_t args;
    args.a = &CTX_IN_STORAGE(DNNL_ARG_SRC);
    args.b = &CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    args.c = &CTX_OUT_STORAGE(DNNL_ARG_DST);
    args.bias = &CTX_IN_STORAGE(DNNL_ARG_BIAS);

    // Note: we have to swap `a` and `b` zero-point arguments because,
    // - gemm primitive is created with row major desc,
    // - parameters to gemm are passed as row major
    // - but gemm implementation assumes column major
    args.a_zero_point = b0;
    args.b_zero_point = a0;
    args.c_zero_point = c0;
    args.a_scales = &CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
    args.b_scales = &CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    bool mx_scales = gemm_->pd()->attr()->scales_.get(DNNL_ARG_DST).is_mx();
    args.c_scales = mx_scales
            ? &CTX_OUT_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST)
            : &CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

    args.a_group_sums = &CTX_IN_STORAGE(
            DNNL_ARG_ATTR_PRECOMPUTED_REDUCTIONS | DNNL_ARG_WEIGHTS);
    args.b_group_sums = &CTX_IN_STORAGE(
            DNNL_ARG_ATTR_PRECOMPUTED_REDUCTIONS | DNNL_ARG_SRC);

    args.sround_seed = &CTX_IN_STORAGE(DNNL_ARG_ATTR_ROUNDING_SEED);
    args.exec_args = ctx.args();
    gemm::desc_t desc;
    CHECK(create_gemm_desc(&desc, src_d.md_, weights_d.md_, dst_d.md_,
            bia_d.md_, pd()->desc()->accum_data_type, ctx.stream()->engine()));

    gemm::exec_ctx_t gemm_ctx(ctx, args, &desc);

    auto *nested_grantor = create_nested_grantor(ctx.get_scratchpad_grantor(),
            key_nested, gemm_->pd()->scratchpad_registry());
    gemm_ctx.set_scratchpad_grantor(nested_grantor);

    status_t gemm_exec_status = gemm::gemm(gemm_)->execute(gemm_ctx);
    if (gemm_exec_status != status::success) return gemm_exec_status;

    return status::success;
}

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
