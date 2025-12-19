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

#include "gpu/intel/eltwise/ref.hpp"
#include "gpu/intel/compute/dispatch_reusable.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace eltwise {

status_t ref_jit_params_t::get_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    core.params.def_kernel_macros(kernel_ctx);
    kernel_ctx.define_int("ELTWISE_ALG", core.alg_kind);

    if (core.is_fwd) {
        kernel_ctx.set_data_type(core.src_dt, false);
        def_data_type(kernel_ctx, core.src_dt, "SRC", false);
        def_data_type(kernel_ctx, core.dst_dt, "DST", false);
        kernel_ctx.define_int("IS_FWD", 1);
        def_post_ops_cfg(kernel_ctx, post_ops, core.ndims);
    } else {
        kernel_ctx.set_data_type(core.diff_dst_dt, false);
        def_data_type(kernel_ctx, core.src_dt, "SRC", false);
        def_data_type(kernel_ctx, core.diff_src_dt, "DIFF_SRC", false);
        def_data_type(kernel_ctx, core.diff_dst_dt, "DIFF_DST", false);
    }

    return status::success;
}

status_t ref_fwd_t::pd_t::init_conf(impl::engine_t *engine) {
    compute::named_buffer_t src_buf(compute::name_id_t::src, *src_md());
    compute::named_buffer_t dst_buf(compute::name_id_t::dst, src_buf);

    auto lws_strategy = compute::default_lws_strategy_t(engine);
    compute::reusable_dispatch_config_t config(
            src_buf.get_dim_ids(), lws_strategy);
    CHECK(config.register_buffer(src_buf));
    CHECK(config.register_buffer(dst_buf));

    CHECK(gpu_post_ops_t::make(conf.post_ops, attr()->post_ops_, dst_md()));

    // TODO: This should be moved to a common location
    uint16_t mask = 0;
    for (auto &e : conf.post_ops)
        if (e.is_binary()) mask |= ~e.as_binary().src1_desc.broadcast_mask;
    for (int i = 0; i < 6; i++) {
        int rmd_idx = post_op::relative_md_t::from_md_idx(i, src_buf.ndims, {})
                              .as_int();
        CHECK(config.define_dim_index(
                compute::name_id_t(int64_t(compute::name_id_t::a) << i), i,
                (mask & (1 << rmd_idx)) ? src_buf.dims[i] : 1));
    }

    compute::reusable_dispatch_t dispatch;
    CHECK(config.generate(dispatch));

    conf.core = {dispatch.get_compile_params(), desc()->alg_kind,
            src_md()->ndims, src_md()->data_type, dst_md()->data_type,
            data_type::undef, data_type::undef, true, {}};

    rt_conf = dispatch.get_runtime_params();
    return status::success;
}

status_t ref_fwd_t::execute_forward_dense(const exec_ctx_t &ctx) const {

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, dst);
    arg_list.set(2, alpha);
    arg_list.set(3, beta);
    set_rt_params(arg_list, 4, pd()->rt_conf);

    append_post_ops_to_arg_list(
            ctx, arg_list, 5, pd()->attr()->post_ops_, *pd()->dst_md());

    return parallel_for(ctx, pd()->rt_conf.nd_range, kernel_, arg_list);
}

status_t ref_bwd_t::pd_t::init_conf(impl::engine_t *engine) {
    compute::named_buffer_t src_buf(
            compute::name_id_t::src, use_dst() ? *dst_md() : *src_md());
    compute::named_buffer_t diff_src_buf(compute::name_id_t::diff_src, src_buf);
    compute::named_buffer_t diff_dst_buf(compute::name_id_t::diff_dst, src_buf);

    auto lws_strategy = compute::default_lws_strategy_t(engine);
    compute::reusable_dispatch_config_t config(
            src_buf.get_dim_ids(), lws_strategy);
    CHECK(config.register_buffer(src_buf));
    CHECK(config.register_buffer(diff_src_buf));
    CHECK(config.register_buffer(diff_dst_buf));

    compute::reusable_dispatch_t dispatch;
    CHECK(config.generate(dispatch));

    conf.core
            = {dispatch.get_compile_params(), desc()->alg_kind, src_md()->ndims,
                    use_dst() ? dst_md()->data_type : src_md()->data_type,
                    data_type::undef, diff_src_md()->data_type,
                    diff_dst_md()->data_type, false, {}};

    rt_conf = dispatch.get_runtime_params();
    return status::success;
}

status_t ref_bwd_t::execute_backward_dense(const exec_ctx_t &ctx) const {
    auto &src = pd()->use_dst() ? CTX_IN_STORAGE(DNNL_ARG_DST)
                                : CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);

    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, diff_src);
    arg_list.set(2, diff_dst);
    arg_list.set(3, alpha);
    arg_list.set(4, beta);
    set_rt_params(arg_list, 5, pd()->rt_conf);

    return parallel_for(ctx, pd()->rt_conf.nd_range, kernel_, arg_list);
}

} // namespace eltwise
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
