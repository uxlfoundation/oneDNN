/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "gpu/intel/ip/matmul.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ip {

status_t matmul_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

    exec_args_t matmul_args = ctx.args();
    exec_ctx_t matmul_ctx(ctx, std::move(matmul_args));

    auto *nested_grantor = create_nested_grantor(ctx.get_scratchpad_grantor(),
            key_nested, matmul_->pd()->scratchpad_registry());
    matmul_ctx.set_scratchpad_grantor(nested_grantor);

    return matmul_->execute(matmul_ctx);
}

status_t matmul_bwd_data_t::execute_backward_data(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

    exec_args_t matmul_args;
    matmul_args[DNNL_ARG_SRC] = ctx.args().at(DNNL_ARG_DIFF_DST);
    matmul_args[DNNL_ARG_WEIGHTS] = ctx.args().at(DNNL_ARG_WEIGHTS);
    matmul_args[DNNL_ARG_DST] = ctx.args().at(DNNL_ARG_DIFF_SRC);
    exec_ctx_t matmul_ctx(ctx, std::move(matmul_args));

    auto *nested_grantor = create_nested_grantor(ctx.get_scratchpad_grantor(),
            key_nested, matmul_->pd()->scratchpad_registry());
    matmul_ctx.set_scratchpad_grantor(nested_grantor);

    return matmul_->execute(matmul_ctx);
}

status_t matmul_bwd_weights_t::execute_backward_weights(
        const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

    exec_args_t matmul_args;
    matmul_args[DNNL_ARG_SRC] = ctx.args().at(DNNL_ARG_DIFF_DST);
    matmul_args[DNNL_ARG_WEIGHTS] = ctx.args().at(DNNL_ARG_SRC);
    matmul_args[DNNL_ARG_DST] = ctx.args().at(DNNL_ARG_DIFF_WEIGHTS);
    if (pd()->with_bias() && !pd()->reduction_pd_)
        matmul_args[DNNL_ARG_REDUCE] = ctx.args().at(DNNL_ARG_DIFF_BIAS);
    exec_ctx_t matmul_ctx(ctx, std::move(matmul_args));

    auto *nested_grantor = create_nested_grantor(ctx.get_scratchpad_grantor(),
            key_nested_multiple, matmul_->pd()->scratchpad_registry());
    matmul_ctx.set_scratchpad_grantor(nested_grantor);

    CHECK(matmul_->execute(matmul_ctx));

    if (pd()->with_bias() && pd()->reduction_pd_) {
        auto diff_dst = ctx.input(DNNL_ARG_DIFF_DST);
        auto diff_bia = ctx.output(DNNL_ARG_DIFF_BIAS);
        exec_args_t r_args;
        r_args[DNNL_ARG_SRC] = memory_arg_t {diff_dst, true};
        r_args[DNNL_ARG_DST] = memory_arg_t {diff_bia, false};
        exec_ctx_t r_ctx(ctx, std::move(r_args));
        auto *r_grantor = create_nested_grantor(ctx.get_scratchpad_grantor(),
                key_nested_multiple + 1,
                reduction_->pd()->scratchpad_registry());
        r_ctx.set_scratchpad_grantor(r_grantor);
        return reduction_->execute(r_ctx);
    }

    return status::success;
}

} // namespace ip
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
