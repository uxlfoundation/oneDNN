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

#include "gpu/intel/ip/matmul.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ip {

// IP's nested gemm primitive is now a matmul (post-Phase-3 migration). IP's
// own keys (DNNL_ARG_SRC/WEIGHTS/DST/BIAS + attr keys) already match what the
// matmul inner expects, so forward the exec_ctx with only the scratchpad
// grantor rebound to the nested registry.
status_t matmul_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

    auto *nested_grantor = create_nested_grantor(ctx.get_scratchpad_grantor(),
            key_nested, matmul_->pd()->scratchpad_registry());
    impl::exec_ctx_t matmul_ctx(ctx, impl::exec_args_t(ctx.args()));
    matmul_ctx.set_scratchpad_grantor(nested_grantor);
    return matmul_->execute(matmul_ctx);
}

status_t matmul_bwd_data_t::execute_backward_data(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

    // Map IP BWD_D's inputs to the inner matmul's SRC/WEIGHTS/DST keys.
    auto *diff_dst = ctx.input(DNNL_ARG_DIFF_DST);
    auto *wei = ctx.input(DNNL_ARG_WEIGHTS);
    auto *diff_src = ctx.output(DNNL_ARG_DIFF_SRC);

    impl::exec_args_t args;
    args[DNNL_ARG_SRC] = {diff_dst, true};
    args[DNNL_ARG_WEIGHTS] = {wei, true};
    args[DNNL_ARG_DST] = {diff_src, false};

    impl::exec_ctx_t matmul_ctx(ctx, std::move(args));
    auto *nested_grantor = create_nested_grantor(ctx.get_scratchpad_grantor(),
            key_nested, matmul_->pd()->scratchpad_registry());
    matmul_ctx.set_scratchpad_grantor(nested_grantor);
    return matmul_->execute(matmul_ctx);
}

status_t matmul_bwd_weights_t::execute_backward_weights(
        const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

    const int user_a_arg
            = pd()->wei_tr() ? DNNL_ARG_SRC : DNNL_ARG_DIFF_DST;
    const int user_b_arg
            = pd()->wei_tr() ? DNNL_ARG_DIFF_DST : DNNL_ARG_SRC;

    auto *user_a = ctx.input(user_a_arg);
    auto *user_b = ctx.input(user_b_arg);
    auto *diff_wei = ctx.output(DNNL_ARG_DIFF_WEIGHTS);

    impl::exec_args_t inner_args;
    inner_args[DNNL_ARG_SRC] = {user_a, true};
    inner_args[DNNL_ARG_WEIGHTS] = {user_b, true};
    inner_args[DNNL_ARG_DST] = {diff_wei, false};
    // Route DIFF_BIAS to the fused matmul's DNNL_ARG_REDUCE slot only when
    // bias is present AND the matmul fused the reduction (reduction_pd_ is
    // null in that case). Without bias, reduction_pd_ is also null but the
    // matmul has no reduce slot — don't bind a non-existent output.
    if (pd()->with_bias() && !pd()->reduction_pd_)
        inner_args[DNNL_ARG_REDUCE]
                = {ctx.output(DNNL_ARG_DIFF_BIAS), false};

    impl::exec_ctx_t matmul_ctx(ctx, std::move(inner_args));
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
