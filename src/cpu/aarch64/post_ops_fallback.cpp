/*******************************************************************************
* Copyright 2022-2024, 2026 Arm Ltd. and affiliates
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

#include "cpu/aarch64/post_ops_fallback.hpp"
#include "common/float16.hpp"
#include "common/memory.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/stream.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

void post_ops_fallback_t::init_scratchpad(
        memory_tracking::registrar_t &scratchpad) const {
    using namespace memory_tracking::names;

    for (size_t i = 0; i < post_op_primitives.size(); ++i) {
        scratchpad.book(key_nested_multiple + (int)i,
                post_op_primitives[i]->pd()->scratchpad_registry());
    }
}

status_t post_ops_fallback_t::execute_binary(const exec_ctx_t &ctx,
        const primitive_t *post_op, const void *src0, const void *src1,
        const void *src2, void *dst, int primitive_index) const {
    if (src0 == nullptr || src1 == nullptr || dst == nullptr)
        return status::runtime_error;

    const bool needs_src2
            = !memory_desc_wrapper(post_op->pd()->arg_md(DNNL_ARG_SRC_2))
                       .is_zero();
    if (needs_src2 && src2 == nullptr) return status::runtime_error;

    auto *engine = ctx.stream()->engine();
    std::unique_ptr<memory_t, memory_deleter_t> src0_mem(
            new memory_t(engine, post_op->pd()->arg_md(DNNL_ARG_SRC_0),
                    use_runtime_ptr, const_cast<void *>(src0)));
    std::unique_ptr<memory_t, memory_deleter_t> src1_mem(
            new memory_t(engine, post_op->pd()->arg_md(DNNL_ARG_SRC_1),
                    use_runtime_ptr, const_cast<void *>(src1)));
    std::unique_ptr<memory_t, memory_deleter_t> src2_mem {nullptr};
    if (needs_src2) {
        src2_mem.reset(
                new memory_t(engine, post_op->pd()->arg_md(DNNL_ARG_SRC_2),
                        use_runtime_ptr, const_cast<void *>(src2)));
    }
    std::unique_ptr<memory_t, memory_deleter_t> dst_mem(new memory_t(
            engine, post_op->pd()->arg_md(DNNL_ARG_DST), use_runtime_ptr, dst));

    exec_args_t binary_args;
    binary_args[DNNL_ARG_SRC_0] = {src0_mem.get(), true};
    binary_args[DNNL_ARG_SRC_1] = {src1_mem.get(), true};
    if (needs_src2) binary_args[DNNL_ARG_SRC_2] = {src2_mem.get(), true};
    binary_args[DNNL_ARG_DST] = {dst_mem.get(), false};
    exec_ctx_t binary_ctx(ctx, std::move(binary_args));

    auto *nested_grantor = memory_tracking::create_nested_grantor(
            ctx.get_scratchpad_grantor(),
            memory_tracking::names::key_nested_multiple + primitive_index,
            post_op->pd()->scratchpad_registry());
    binary_ctx.set_scratchpad_grantor(nested_grantor);

    return post_op->execute(binary_ctx);
}

status_t post_ops_fallback_t::execute_eltwise(const exec_ctx_t &ctx,
        const primitive_t *post_op, void *src, int primitive_index) const {
    if (src == nullptr) return status::runtime_error;

    auto *engine = ctx.stream()->engine();
    std::unique_ptr<memory_t, memory_deleter_t> src_mem(new memory_t(
            engine, post_op->pd()->arg_md(DNNL_ARG_SRC), use_runtime_ptr, src));
    std::unique_ptr<memory_t, memory_deleter_t> dst_mem(new memory_t(
            engine, post_op->pd()->arg_md(DNNL_ARG_DST), use_runtime_ptr, src));

    exec_args_t eltwise_args;
    eltwise_args[DNNL_ARG_SRC] = {src_mem.get(), true};
    eltwise_args[DNNL_ARG_DST] = {dst_mem.get(), false};
    exec_ctx_t eltwise_ctx(ctx, std::move(eltwise_args));

    auto *nested_grantor = memory_tracking::create_nested_grantor(
            ctx.get_scratchpad_grantor(),
            memory_tracking::names::key_nested_multiple + primitive_index,
            post_op->pd()->scratchpad_registry());
    eltwise_ctx.set_scratchpad_grantor(nested_grantor);

    return post_op->execute(eltwise_ctx);
}

status_t post_ops_fallback_t::execute(
        const exec_ctx_t &ctx, void *src, void *dst) const {

    int post_op_index = post_op_start_index_;
    int primitive_index = 0;

    // By default, dst is expected to be the output buffer. However, in some
    // cases we may want to override that behaviour and use a temporary buffer.
    if (dst == nullptr) { dst = CTX_OUT_MEM(void *, DNNL_ARG_DST); }

    // Sum post-op requires distinct src and dst buffers.
    if (has_sum() && dst == src) { return status::runtime_error; }

    for (auto &post_op : post_op_primitives) {
        if (post_op->kind() == primitive_kind::binary) {
            // Sum post op accumulates to dst and changes future src
            if (post_op_index == sum_index) {
                CHECK(execute_binary(ctx, post_op.get(), src, dst, nullptr, dst,
                        primitive_index));
                src = dst;
            } else {
                const void *src_binary = CTX_IN_MEM(const void *,
                        (DNNL_ARG_ATTR_MULTIPLE_POST_OP(post_op_index)
                                | DNNL_ARG_SRC_1));
                const void *src_binary2 = nullptr;
                if (!memory_desc_wrapper(post_op->pd()->arg_md(DNNL_ARG_SRC_2))
                                .is_zero()) {
                    src_binary2 = CTX_IN_MEM(const void *,
                            (DNNL_ARG_ATTR_MULTIPLE_POST_OP(post_op_index)
                                    | DNNL_ARG_SRC_2));
                }
                CHECK(execute_binary(ctx, post_op.get(), src, src_binary,
                        src_binary2, src, primitive_index));
            }
        } else if (post_op->kind() == primitive_kind::eltwise) {
            // The post op at the sum index must be binary
            if (post_op_index == sum_index) return status::runtime_error;
            CHECK(execute_eltwise(ctx, post_op.get(), src, primitive_index));
        } else {
            return status::runtime_error;
        }

        ++post_op_index;
        ++primitive_index;
    }

    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
