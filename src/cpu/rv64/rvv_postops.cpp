/*******************************************************************************
* Copyright 2025 ZTE Corporation
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
#include "cpu/rv64/rvv_postops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

status_t rvv_postops_t::execute(
        const exec_ctx_t &ctx, void *src, void *dst) const {
    int post_op_index = post_op_start_index_;

    for (auto &post_op : post_op_primitives_) {
        if (post_op->kind() != primitive_kind::binary)
            return status::runtime_error;

        exec_args_t bin_args;
        bin_args[DNNL_ARG_SRC_0] = ctx.args().at(DNNL_ARG_DST);
        bin_args[DNNL_ARG_DST] = ctx.args().at(DNNL_ARG_DST);
        const int rhs_arg = (DNNL_ARG_ATTR_MULTIPLE_POST_OP(post_op_index)
                | DNNL_ARG_SRC_1);
        bin_args[DNNL_ARG_SRC_1] = ctx.args().at(rhs_arg);

        const auto &po = po_.entry_[post_op_index];
        if (po.is_binary() && po.binary.alg == alg_kind::binary_select) {
            const int rhs2_arg = (DNNL_ARG_ATTR_MULTIPLE_POST_OP(post_op_index)
                    | DNNL_ARG_SRC_2);
            bin_args[DNNL_ARG_SRC_2] = ctx.args().at(rhs2_arg);
        }

        exec_ctx_t bin_ctx(ctx, std::move(bin_args));
        CHECK(post_op->execute(bin_ctx));

        ++post_op_index;
    }

    return status::success;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
