/*******************************************************************************
* Copyright 2026 Advanced Micro Devices, Inc.
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

#include "cpu/x64/zen64/common/zen_unpack_fallback.hpp"

#include "common/memory.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/reorder.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace zen {

status_t unpack_fallback_t::init(
        const engine_t *engine, const matmul_pd_t &pd) {
    const memory_desc_t &wei_md = *pd.weights_md(0);
    CHECK(memory_desc_init_by_tag(plain_wei_md_, wei_md.ndims, wei_md.dims,
            wei_md.data_type,
            wei_md.ndims == 2 ? format_tag::ab : format_tag::abc));
    CHECK(reorder_primitive_desc_create(
            unpack_pd_, engine, &wei_md, &plain_wei_md_));

    // First implementation in dispatch order; the op descriptor fixes all
    // layouts and keeps the reduction and the accumulation data type.
    CHECK(impl::create_matmul_pd(matmul_pd_, engine, pd.src_md(0),
            &plain_wei_md_, pd.with_bias() ? pd.weights_md(1) : nullptr,
            pd.dst_md(0), pd.attr(),
            pd.with_reduce() ? pd.reduce_md(0) : nullptr, pd.reduce_kind(),
            pd.desc()->accum_data_type));
    name_ = std::string("zen:unpack+") + matmul_pd_->name();
    return status::success;
}

status_t unpack_fallback_t::init_primitives(engine_t *engine) {
    CHECK(unpack_pd_->create_primitive(unpack_, engine));
    return matmul_pd_->create_primitive(matmul_, engine);
}

void unpack_fallback_t::init_scratchpad(
        memory_tracking::registrar_t &scratchpad) const {
    using namespace memory_tracking::names;
    // The weights may change between executions, so they are unpacked on
    // every one.
    scratchpad.book(key_matmul_wei_trans,
            memory_desc_wrapper(plain_wei_md_).size(), /*data_size=*/1,
            /*alignment=*/64);
    scratchpad.book(key_nested_multiple, unpack_pd_->scratchpad_registry());
    scratchpad.book(key_nested_multiple + 1, matmul_pd_->scratchpad_registry());
}

status_t unpack_fallback_t::execute(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;
    engine_t *engine = ctx.stream()->engine();
    const auto &scratchpad = ctx.get_scratchpad_grantor();

    std::unique_ptr<memory_t, memory_deleter_t> plain_wei;
    CHECK(safe_ptr_assign(plain_wei,
            new memory_t(engine, &plain_wei_md_,
                    scratchpad.get_memory_storage(key_matmul_wei_trans))));

    // The matmul reads no weights when the destination is empty.
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, matmul_->pd()->dst_md());
    if (!dst_d.has_zero_dim()) {
        exec_args_t unpack_args;
        unpack_args[DNNL_ARG_FROM] = ctx.args().at(DNNL_ARG_WEIGHTS);
        unpack_args[DNNL_ARG_TO] = {plain_wei.get(), false};
        exec_ctx_t unpack_ctx(ctx, std::move(unpack_args));
        unpack_ctx.set_scratchpad_grantor(create_nested_grantor(scratchpad,
                key_nested_multiple, unpack_->pd()->scratchpad_registry()));
        CHECK(unpack_->execute(unpack_ctx));
    }

    // All other arguments (bias, post-op, scale, zero-point) pass through.
    exec_args_t matmul_args = ctx.args();
    matmul_args[DNNL_ARG_WEIGHTS] = {plain_wei.get(), true};
    exec_ctx_t matmul_ctx(ctx, std::move(matmul_args));
    matmul_ctx.set_scratchpad_grantor(create_nested_grantor(scratchpad,
            key_nested_multiple + 1, matmul_->pd()->scratchpad_registry()));
    return matmul_->execute(matmul_ctx);
}

} // namespace zen
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
