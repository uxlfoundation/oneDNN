/*******************************************************************************
 * Copyright 2025 Intel Corporation
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

#include "graph/backend/dnnl/executables/sdpa.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

sdpa_executable_t::sdpa_executable_t(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout)
    : with_scale_(op->get_attr<bool>(op_attr::with_scale))
    , mask_type_(static_cast<attn_mask_type_t>(
              op->get_attr<int64_t>(op_attr::mask_type))) {

    auto md_q = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto md_k = make_dnnl_memory_desc(
            op->get_input_value(1)->get_logical_tensor());
    auto md_v = make_dnnl_memory_desc(
            op->get_input_value(2)->get_logical_tensor());
    auto md_dst = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());

    auto md_scale = dnnl::memory::desc();
    size_t idx = 3;
    if (with_scale_)
        md_scale = make_dnnl_memory_desc(
                op->get_input_value(idx++)->get_logical_tensor());

    dnnl::memory::desc md_mask;
    with_explicit_mask_ = mask_type_ == attn_mask_type::buffer;
    if (with_explicit_mask_)
        md_mask = make_dnnl_memory_desc(
                op->get_input_value(idx++)->get_logical_tensor());

    dnnl::primitive_attr attr, qk_attr, vs_attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    attr.set_fpmath_mode(static_cast<dnnl::fpmath_mode>(fpmath.mode_));

    is_invert_scale_ = op->has_attr(op_attr::is_invert_scale)
            ? op->get_attr<bool>(op_attr::is_invert_scale)
            : false;

    if (op->has_attr(op_attr::fusion_info)) {
        const auto &sdpa_fusion_info
                = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        qk_attr = make_dnnl_sdpa_primitive_attr(
                op, sdpa_fusion_info, attr_type_t::QK);
        vs_attr = make_dnnl_sdpa_primitive_attr(
                op, sdpa_fusion_info, attr_type_t::VS);
    }

    // Set accumulation mode: the two attributes are requested for
    // dnnl_sdpa, so we can get them directly without calling has_attr().
    qk_attr.set_accumulation_mode(str2accumulation_mode(
            op->get_attr<std::string>(op_attr::qk_acc_mode)));
    vs_attr.set_accumulation_mode(str2accumulation_mode(
            op->get_attr<std::string>(op_attr::vs_acc_mode)));

    dim_t kv_head_number = op->get_input_value(1)->get_logical_tensor().dims[1];

    const std::string &softmax_mode = op->get_attr<std::string>(op_attr::mode);
    const alg_kind_t softmax_alg = softmax_mode == "inf_as_zero"
            ? alg_kind::softmax_accurate_inf_as_zero
            : alg_kind::softmax_accurate;
    status_t s = create_sdpa_pd(sdpa_pd_, p_engine.get(), md_q.get(),
            md_k.get(), md_v.get(), md_dst.get(), md_mask.get(), md_scale.get(),
            is_invert_scale_, kv_head_number, mask_type_, softmax_alg,
            attr.get(), qk_attr.get(), vs_attr.get());
    if (s != dnnl::impl::status::success) {
        is_initialized_ = false;
    } else {
        status_t s = sdpa_pd_->create_primitive(sdpa_prim_, p_engine.get());
        is_initialized_ = s == status::success ? true : false;
    }
}

void sdpa_executable_t::execute(const stream &stream,
        const std::unordered_map<int, memory> &args) const {
    exec_args_t exec_args;
    memory_arg_t mem_arg_q = {(args.at(DNNL_ARG_QUERIES)).get(), true};
    memory_arg_t mem_arg_k = {(args.at(DNNL_ARG_KEYS)).get(), true};
    memory_arg_t mem_arg_v = {(args.at(DNNL_ARG_VALUES)).get(), true};
    memory_arg_t mem_arg_dst = {(args.at(DNNL_ARG_DST)).get(), false};
    memory_arg_t mem_arg_scale
            = {with_scale_ ? (args.at(DNNL_ARG_SCALE)).get() : nullptr, true};
    memory_arg_t mem_arg_mask = {
            with_explicit_mask_ ? (args.at(DNNL_ARG_ATTN_MASK)).get() : nullptr,
            true};
    memory_arg_t mem_arg_k_scale = {
            args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_KEYS) != args.end()
                    ? (args.at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_KEYS)).get()
                    : nullptr,
            true};

    memory_arg_t mem_arg_v_scale = {
            args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_VALUES) != args.end()
                    ? (args.at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_VALUES)).get()
                    : nullptr,
            true};
    memory_arg_t mem_arg_k_zero_points = {
            args.find(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_KEYS) != args.end()
                    ? (args.at(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_KEYS)).get()
                    : nullptr,
            true};
    memory_arg_t mem_arg_v_zero_points = {
            args.find(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_VALUES) != args.end()
                    ? (args.at(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_VALUES))
                              .get()
                    : nullptr,
            true};

    exec_args[DNNL_ARG_QUERIES] = mem_arg_q;
    exec_args[DNNL_ARG_KEYS] = mem_arg_k;
    exec_args[DNNL_ARG_VALUES] = mem_arg_v;
    exec_args[DNNL_ARG_DST] = mem_arg_dst;
    exec_args[DNNL_ARG_SCALE] = mem_arg_scale;
    exec_args[DNNL_ARG_ATTN_MASK] = mem_arg_mask;
    exec_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_KEYS] = mem_arg_k_scale;
    exec_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_VALUES] = mem_arg_v_scale;
    exec_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_KEYS]
            = mem_arg_k_zero_points;
    exec_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_VALUES]
            = mem_arg_v_zero_points;

    exec_ctx_t ctx(stream.get(), std::move(exec_args));
    sdpa_prim_->execute(ctx);
}

#ifdef DNNL_WITH_SYCL
::sycl::event sdpa_executable_t::execute_sycl(const stream &stream,
        const std::unordered_map<int, memory> &args,
        const std::vector<::sycl::event> &deps) const {

    exec_args_t exec_args;
    memory_arg_t mem_arg_q = {(args.at(DNNL_ARG_QUERIES)).get(), true};
    memory_arg_t mem_arg_k = {(args.at(DNNL_ARG_KEYS)).get(), true};
    memory_arg_t mem_arg_v = {(args.at(DNNL_ARG_VALUES)).get(), true};
    memory_arg_t mem_arg_dst = {(args.at(DNNL_ARG_DST)).get(), false};
    memory_arg_t mem_arg_scale
            = {with_scale_ ? (args.at(DNNL_ARG_SCALE)).get() : nullptr, true};
    memory_arg_t mem_arg_mask = {
            with_explicit_mask_ ? (args.at(DNNL_ARG_ATTN_MASK)).get() : nullptr,
            true};
    memory_arg_t mem_arg_k_scale = {
            args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_KEYS) != args.end()
                    ? (args.at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_KEYS)).get()
                    : nullptr,
            true};

    memory_arg_t mem_arg_v_scale = {
            args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_VALUES) != args.end()
                    ? (args.at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_VALUES)).get()
                    : nullptr,
            true};
    memory_arg_t mem_arg_k_zero_points = {
            args.find(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_KEYS) != args.end()
                    ? (args.at(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_KEYS)).get()
                    : nullptr,
            true};
    memory_arg_t mem_arg_v_zero_points = {
            args.find(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_VALUES) != args.end()
                    ? (args.at(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_VALUES))
                              .get()
                    : nullptr,
            true};

    exec_args[DNNL_ARG_QUERIES] = mem_arg_q;
    exec_args[DNNL_ARG_KEYS] = mem_arg_k;
    exec_args[DNNL_ARG_VALUES] = mem_arg_v;
    exec_args[DNNL_ARG_DST] = mem_arg_dst;
    exec_args[DNNL_ARG_SCALE] = mem_arg_scale;
    exec_args[DNNL_ARG_ATTN_MASK] = mem_arg_mask;
    exec_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_KEYS] = mem_arg_k_scale;
    exec_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_VALUES] = mem_arg_v_scale;
    exec_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_KEYS]
            = mem_arg_k_zero_points;
    exec_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_VALUES]
            = mem_arg_v_zero_points;

    auto strm_t = stream.get();
    exec_ctx_t ctx(strm_t, std::move(exec_args));
    auto *sycl_stream_impl = dnnl::impl::utils::downcast<
            dnnl::impl::xpu::sycl::stream_impl_t *>(strm_t->impl());

    strm_t->before_exec_hook();

    if (!deps.empty()) sycl_stream_impl->sycl_ctx().set_deps(deps);

    sdpa_prim_->execute(ctx);

    ::sycl::event return_event = sycl_stream_impl->get_output_event();
    strm_t->after_exec_hook();
    return return_event;
}
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
cl_event sdpa_executable_t::execute_ocl(const stream &stream,
        const std::unordered_map<int, memory> &args,
        const std::vector<cl_event> &deps) const {
    exec_args_t exec_args;
    memory_arg_t mem_arg_q = {(args.at(DNNL_ARG_QUERIES)).get(), true};
    memory_arg_t mem_arg_k = {(args.at(DNNL_ARG_KEYS)).get(), true};
    memory_arg_t mem_arg_v = {(args.at(DNNL_ARG_VALUES)).get(), true};
    memory_arg_t mem_arg_dst = {(args.at(DNNL_ARG_DST)).get(), false};
    memory_arg_t mem_arg_scale
            = {with_scale_ ? (args.at(DNNL_ARG_SCALE)).get() : nullptr, true};
    memory_arg_t mem_arg_mask = {
            with_explicit_mask_ ? (args.at(DNNL_ARG_ATTN_MASK)).get() : nullptr,
            true};
    memory_arg_t mem_arg_k_scale = {
            args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_KEYS) != args.end()
                    ? (args.at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_KEYS)).get()
                    : nullptr,
            true};

    memory_arg_t mem_arg_v_scale = {
            args.find(DNNL_ARG_ATTR_SCALES | DNNL_ARG_VALUES) != args.end()
                    ? (args.at(DNNL_ARG_ATTR_SCALES | DNNL_ARG_VALUES)).get()
                    : nullptr,
            true};
    memory_arg_t mem_arg_k_zero_points = {
            args.find(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_KEYS) != args.end()
                    ? (args.at(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_KEYS)).get()
                    : nullptr,
            true};
    memory_arg_t mem_arg_v_zero_points = {
            args.find(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_VALUES) != args.end()
                    ? (args.at(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_VALUES))
                              .get()
                    : nullptr,
            true};

    exec_args[DNNL_ARG_QUERIES] = mem_arg_q;
    exec_args[DNNL_ARG_KEYS] = mem_arg_k;
    exec_args[DNNL_ARG_VALUES] = mem_arg_v;
    exec_args[DNNL_ARG_DST] = mem_arg_dst;
    exec_args[DNNL_ARG_SCALE] = mem_arg_scale;
    exec_args[DNNL_ARG_ATTN_MASK] = mem_arg_mask;
    exec_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_KEYS] = mem_arg_k_scale;
    exec_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_VALUES] = mem_arg_v_scale;
    exec_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_KEYS]
            = mem_arg_k_zero_points;
    exec_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_VALUES]
            = mem_arg_v_zero_points;

    exec_ctx_t ctx(stream.get(), std::move(exec_args));

    auto *ocl_stream = dnnl::impl::utils::downcast<gpu::intel::ocl::stream_t *>(
            stream.get());

    ocl_stream->before_exec_hook();

    if (!deps.empty()) {
        std::vector<xpu::ocl::wrapper_t<cl_event>> events(deps.size());
        for (size_t i = 0; i < deps.size(); i++)
            events[i] = xpu::ocl::wrapper_t<cl_event>(deps[i], true);
        ocl_stream->ocl_ctx().set_deps(events);
    }

    sdpa_prim_->execute(ctx);

    cl_event return_event = nullptr;
    if ((ocl_stream->flags() & stream_flags::in_order) == 0) {
        auto last = ocl_stream->get_output_event();
        return_event = last.release();
    }

    ocl_stream->after_exec_hook();
    return return_event;
}
#endif

arg_indices_t sdpa_executable_t::get_arg_indices(const op_t *op) {
    arg_indices_t args;
    // inputs: query, key, value
    size_t idx = 0;
    args.insert({DNNL_ARG_QUERIES, {indices_t::type_t::input, idx++}});
    args.insert({DNNL_ARG_KEYS, {indices_t::type_t::input, idx++}});
    args.insert({DNNL_ARG_VALUES, {indices_t::type_t::input, idx++}});
    // Optional inputs: scale, mask
    if (op->get_attr<bool>(op_attr::with_scale)) {
        args.insert({DNNL_ARG_SCALE, {indices_t::type_t::input, idx++}});
    }
    if (op->get_attr<int64_t>(op_attr::mask_type)
            == static_cast<int64_t>(attn_mask_type::buffer)) {
        args.insert({DNNL_ARG_ATTN_MASK, {indices_t::type_t::input, idx++}});
    }

    const auto &sdpa_fusion_info = op->has_attr(op_attr::fusion_info)
            ? op->get_attr<fusion_info_t>(op_attr::fusion_info)
            : fusion_info_t();
    if (sdpa_fusion_info.with_runtime_scales(true, DNNL_ARG_KEYS)) {
        args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_KEYS,
                {indices_t::type_t::input, idx++}});
    }
    if (sdpa_fusion_info.with_runtime_zero_points(true, DNNL_ARG_KEYS)) {
        args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_KEYS,
                {indices_t::type_t::input, idx++}});
    }
    if (sdpa_fusion_info.with_runtime_scales(true, DNNL_ARG_VALUES)) {
        args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_VALUES,
                {indices_t::type_t::input, idx++}});
    }
    if (sdpa_fusion_info.with_runtime_zero_points(true, DNNL_ARG_VALUES)) {
        args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_VALUES,
                {indices_t::type_t::input, idx++}});
    }

    // outputs
    args.insert({DNNL_ARG_DST, {indices_t::type_t::output, 0}});
    args.insert({DNNL_ARG_SCRATCHPAD, {indices_t::type_t::output, 1}});
    return args;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
