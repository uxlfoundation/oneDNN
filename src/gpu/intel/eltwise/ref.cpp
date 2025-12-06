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

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace eltwise {

status_t ref_jit_params_t::get_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {

    kernel_ctx.define_int("USE_CUSTOM_GWS_GET_ID", 1);

    core.params.def_kernel_macros(kernel_ctx);
    kernel_ctx.define_int("ELTWISE_ALG", core.alg_kind);

    if (core.is_fwd) {
        kernel_ctx.set_data_type(core.src_dt, false);
        def_data_type(kernel_ctx, core.src_dt, "SRC", false);
        def_data_type(kernel_ctx, core.dst_dt, "DST", false);
        kernel_ctx.define_int("IS_FWD", 1);
        def_post_ops_cfg(kernel_ctx, post_ops, core.ndims);
    } else {
        kernel_ctx.set_data_type(core.diff_dst_dt);
        def_data_type(kernel_ctx, core.src_dt, "SRC", false);
        def_data_type(kernel_ctx, core.diff_src_dt, "DIFF_SRC", false);
        def_data_type(kernel_ctx, core.diff_dst_dt, "DIFF_DST", false);
    }

    return status::success;
}

status_t ref_fwd_t::pd_t::init_conf(impl::engine_t *engine) {
    auto *intel_engine = utils::downcast<intel::engine_t *>(engine);
    compute::named_buffer_t src_buf("SRC", *src_md());
    compute::named_buffer_t dst_buf("DST", src_buf);
    const auto *gpu_attr
            = utils::downcast<gpu_primitive_attr_t *>(attr()->gpu_attr_.get());

    compute::reusable_dispatch_config_t config(
            intel_engine, src_buf.get_dim_ids());
    CHECK(config.register_buffer(src_buf));
    CHECK(config.register_buffer(dst_buf));

    auto ndims = src_md()->ndims;
    CHECK(config.define_dim_index("A", 0, ndims > 0 ? src_buf.dims[0] : 1));
    CHECK(config.define_dim_index("B", 1, ndims > 1 ? src_buf.dims[1] : 1));
    CHECK(config.define_dim_index("C", 2, ndims > 2 ? src_buf.dims[2] : 1));
    CHECK(config.define_dim_index("D", 3, ndims > 3 ? src_buf.dims[3] : 1));
    CHECK(config.define_dim_index("E", 4, ndims > 4 ? src_buf.dims[4] : 1));
    CHECK(config.define_dim_index("F", 5, ndims > 5 ? src_buf.dims[5] : 1));

    compute::reusable_dispatch_t dispatch;
    CHECK(config.generate(
            dispatch, compute::default_lws_strategy_t(intel_engine, gpu_attr)));

    conf.core = {dispatch.get_compile_params(), desc()->alg_kind,
            src_md()->ndims, src_md()->data_type, dst_md()->data_type,
            data_type::undef, data_type::undef, true, {}};
    CHECK(gpu_post_ops_t::make(conf.post_ops, attr()->post_ops_, dst_md()));

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
    arg_list.set(4, pd()->rt_conf.get());

    append_post_ops_to_arg_list(
            ctx, arg_list, 6, pd()->attr()->post_ops_, *pd()->dst_md());

    return large_parallel_for(
            ctx, pd()->rt_conf.nd_range, kernel_, arg_list, 5);
}

status_t ref_bwd_t::pd_t::init_conf(impl::engine_t *engine) {
    auto *intel_engine = utils::downcast<intel::engine_t *>(engine);
    compute::named_buffer_t src_buf("SRC", use_dst() ? *dst_md() : *src_md());
    compute::named_buffer_t diff_src_buf("DIFF_SRC", src_buf);
    compute::named_buffer_t diff_dst_buf("DIFF_DST", src_buf);

    const auto *gpu_attr
            = utils::downcast<gpu_primitive_attr_t *>(attr()->gpu_attr_.get());

    compute::reusable_dispatch_config_t config(
            intel_engine, src_buf.get_dim_ids());
    CHECK(config.register_buffer(src_buf));
    CHECK(config.register_buffer(diff_src_buf));
    CHECK(config.register_buffer(diff_dst_buf));

    compute::reusable_dispatch_t dispatch;
    CHECK(config.generate(
            dispatch, compute::default_lws_strategy_t(intel_engine, gpu_attr)));

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
    arg_list.set(5, pd()->rt_conf.get());

    return large_parallel_for(
            ctx, pd()->rt_conf.nd_range, kernel_, arg_list, 6);
}

} // namespace eltwise
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
