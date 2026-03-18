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

#include "gpu/intel/softmax/simple.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace softmax {

status_t simple_fwd_t::execute_generic(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);
    auto &src_scale = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    auto &dst_scale = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
    const bool with_dropout = !pd()->attr()->dropout_.has_default_values();

    compute::kernel_arg_list_t arg_list;
    int arg_idx = 0;
    arg_list.set(arg_idx++, src);
    arg_list.set(arg_idx++, dst);
    arg_list.set(arg_idx++, src_scale);
    arg_list.set(arg_idx++, dst_scale);
    if (with_dropout) {
        const bool use_host_scalars = pd()->attr()->dropout_.use_host_scalars_;
        const bool use_offset = pd()->attr()->dropout_.use_offset_;

        const auto &dropout_p
                = CTX_IN_STORAGE(DNNL_ARG_ATTR_DROPOUT_PROBABILITY);
        const auto &dropout_seed = CTX_IN_STORAGE(DNNL_ARG_ATTR_DROPOUT_SEED);
        const auto &dropout_offset
                = CTX_IN_STORAGE(DNNL_ARG_ATTR_DROPOUT_OFFSET);

        arg_list.set(arg_idx++, CTX_OUT_STORAGE(DNNL_ARG_ATTR_DROPOUT_MASK));
        if (use_host_scalars) {
            int64_t scalar_seed = 0;
            int64_t scalar_offset = 0;
            float scalar_prob = 0.f;
            const host_scalar_memory_storage_t *seed_storage
                    = utils::downcast<const host_scalar_memory_storage_t *>(
                            &dropout_seed);
            CHECK(seed_storage->get_scalar_value(
                    &scalar_seed, sizeof(scalar_seed)));
            if (use_offset) {
                const host_scalar_memory_storage_t *offset_storage
                        = utils::downcast<const host_scalar_memory_storage_t *>(
                                &dropout_offset);
                CHECK(offset_storage->get_scalar_value(
                        &scalar_offset, sizeof(scalar_offset)));
            }
            const host_scalar_memory_storage_t *prob_storage
                    = utils::downcast<const host_scalar_memory_storage_t *>(
                            &dropout_p);
            CHECK(prob_storage->get_scalar_value(
                    &scalar_prob, sizeof(scalar_prob)));
            arg_list.set(arg_idx++, scalar_seed);
            arg_list.set(arg_idx++, scalar_offset);
            arg_list.set(arg_idx++, scalar_prob);
        } else {
            arg_list.set(arg_idx++, dropout_seed);
            arg_list.set(arg_idx++, dropout_offset);
            arg_list.set(arg_idx++, dropout_p);
        }
    }
    append_post_ops_to_arg_list(
            ctx, arg_list, arg_idx++, pd()->attr()->post_ops_, *pd()->dst_md());
    if (pd()->group_size > 1) {
        auto nd_range = compute::nd_range_t(pd()->gws, pd()->lws);
        return parallel_for(ctx, nd_range, kernel_, arg_list);
    } else {
        auto nd_range = compute::nd_range_t(pd()->gws);
        return parallel_for(ctx, nd_range, kernel_, arg_list);
    }
}

status_t simple_bwd_t::execute_generic(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    auto &dst = CTX_IN_STORAGE(DNNL_ARG_DST);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, dst);
    arg_list.set(1, diff_src);
    arg_list.set(2, diff_dst);

    auto nd_range = compute::nd_range_t(pd()->gws, pd()->lws);
    return parallel_for(ctx, nd_range, kernel_, arg_list);
}

} // namespace softmax
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
