/*******************************************************************************
* Copyright 2022-2025 Arm Ltd. and affiliates
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

#include "common/float16.hpp"
#include "cpu/aarch64/acl_gemm_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

status_t acl_post_ops_t::execute(
        const exec_ctx_t &ctx, void *src, void *dst) const {

    int post_op_index = post_op_start_index_;

    // By default, dst is expected to be the output buffer. However, in some
    // cases we may want to override that behaviour and use a temporary buffer.
    if (dst == nullptr) { dst = CTX_OUT_MEM(void *, DNNL_ARG_DST); }

    // Sum post-op requires distinct src and dst buffers.
    if (has_sum() && dst == src) { return status::runtime_error; }

    for (auto &post_op : post_op_primitives) {
        if (post_op->kind() == primitive_kind::binary) {
            auto binary_post_op = dynamic_cast<acl_binary_t *>(post_op.get());
            if (binary_post_op == nullptr) return status::runtime_error;

            // Sum post op accumulates to dst and changes future src
            if (post_op_index == sum_index) {
                CHECK(binary_post_op->execute_forward(ctx, src, dst, dst));
                src = dst;
            } else {
                const void *src_binary = CTX_IN_MEM(const void *,
                        (DNNL_ARG_ATTR_MULTIPLE_POST_OP(post_op_index)
                                | DNNL_ARG_SRC_1));
                CHECK(binary_post_op->execute_forward(
                        ctx, src, src_binary, src));
            }
        } else if (post_op->kind() == primitive_kind::eltwise) {
            // The post op at the sum index must be binary
            if (post_op_index == sum_index) return status::runtime_error;
            exec_args_t eltwise_args;
            eltwise_args[DNNL_ARG_DST] = ctx.args().at(DNNL_ARG_DST);
            eltwise_args[DNNL_ARG_SRC] = ctx.args().at(DNNL_ARG_DST);
            exec_ctx_t eltwise_ctx(ctx, std::move(eltwise_args));
            CHECK(post_op->execute(eltwise_ctx));
        } else {
            return status::runtime_error;
        }

        ++post_op_index;
    }

    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
