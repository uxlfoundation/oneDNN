/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "gpu/intel/ip/gemm.hpp"

#include "gpu/intel/gemm/primitive.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ip {

status_t gemm_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

    gemm::exec_args_t args;
    args.a = &CTX_IN_STORAGE(DNNL_ARG_SRC);
    args.b = &CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    args.c = &CTX_OUT_STORAGE(DNNL_ARG_DST);
    args.bias = &CTX_IN_STORAGE(DNNL_ARG_BIAS);
    memory_storage_t *a0
            = &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);

    memory_storage_t *b0
            = &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS);

    memory_storage_t *c0
            = &CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);

    args.a_zero_point = b0;
    args.b_zero_point = a0;
    args.c_zero_point = c0;
    args.a_scales = &CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
    args.b_scales = &CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    args.c_scales = &CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
    args.exec_args = ctx.args();

    gemm::exec_ctx_t gemm_ctx(ctx, args);

    auto *nested_grantor = create_nested_grantor(ctx.get_scratchpad_grantor(),
            key_nested, gemm_->pd()->scratchpad_registry());
    gemm_ctx.set_scratchpad_grantor(nested_grantor);

    status_t gemm_exec_status = gemm::gemm(gemm_)->execute(gemm_ctx);

    if (gemm_exec_status != status::success) return gemm_exec_status;

    return status::success;
}

status_t gemm_bwd_data_t::execute_backward_data(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

    gemm::exec_args_t args;
    args.a = &CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    args.b = &CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    args.c = &CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);

    gemm::exec_ctx_t gemm_ctx(ctx, args);

    auto *nested_grantor = create_nested_grantor(ctx.get_scratchpad_grantor(),
            key_nested, gemm_->pd()->scratchpad_registry());
    gemm_ctx.set_scratchpad_grantor(nested_grantor);

    status_t gemm_exec_status = gemm::gemm(gemm_)->execute(gemm_ctx);
    if (gemm_exec_status != status::success) return gemm_exec_status;

    return status::success;
}

status_t gemm_bwd_weights_t::execute_backward_weights(
        const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

    gemm::exec_args_t gemm_args;
    if (pd()->wei_tr()) {
        gemm_args.a = &CTX_IN_STORAGE(DNNL_ARG_SRC);
        gemm_args.b = &CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    } else {
        gemm_args.a = &CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
        gemm_args.b = &CTX_IN_STORAGE(DNNL_ARG_SRC);
    }
    gemm_args.c = &CTX_OUT_STORAGE(DNNL_ARG_DIFF_WEIGHTS);
    if (!pd()->reduction_pd_)
        gemm_args.sum_ab = &CTX_OUT_STORAGE(DNNL_ARG_DIFF_BIAS);
    gemm::exec_ctx_t gemm_ctx(ctx, gemm_args);

    auto *nested_grantor = create_nested_grantor(ctx.get_scratchpad_grantor(),
            key_nested_multiple, gemm_->pd()->scratchpad_registry());
    gemm_ctx.set_scratchpad_grantor(nested_grantor);

    status_t gemm_exec_status = gemm::gemm(gemm_)->execute(gemm_ctx);
    if (gemm_exec_status != status::success) return gemm_exec_status;

    if (pd()->with_bias() && pd()->reduction_pd_) {
        auto diff_dst = ctx.input(DNNL_ARG_DIFF_DST);
        auto diff_bia = ctx.output(DNNL_ARG_DIFF_BIAS);
        exec_args_t r_args;
        r_args[DNNL_ARG_SRC] = memory_arg_t {diff_dst, true};
        r_args[DNNL_ARG_DST] = memory_arg_t {diff_bia, false};
        exec_ctx_t r_ctx(ctx, std::move(r_args));
        auto *nested_grantor = create_nested_grantor(
                ctx.get_scratchpad_grantor(), key_nested_multiple + 1,
                reduction_->pd()->scratchpad_registry());
        r_ctx.set_scratchpad_grantor(nested_grantor);
        return reduction_->execute(r_ctx);
    }

    return status::success;
}

} // namespace ip
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
