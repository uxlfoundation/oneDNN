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

#include "graph/backend/dnnl/executables/gated_mlp.hpp"

#include "common/gated_mlp_iface.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

gated_mlp_executable_t::gated_mlp_executable_t(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout) {
    auto src_md = make_dnnl_memory_desc(op->get_input_logical_tensor(0));
    auto wei0_md = make_dnnl_memory_desc(op->get_input_logical_tensor(1));
    auto wei1_md = make_dnnl_memory_desc(op->get_input_logical_tensor(2));
    auto wei2_md = make_dnnl_memory_desc(op->get_input_logical_tensor(3));
    auto dst_md = make_dnnl_memory_desc(op->get_output_logical_tensor(0));

    dnnl_primitive_attr_t attr = nullptr;
    auto act_algo = op->has_attr(op_attr::alg_kind)
            ? static_cast<dnnl::algorithm>(
                      op->get_attr<int64_t>(op_attr::alg_kind))
            : dnnl::algorithm::undef;
    dnnl_primitive_desc_t pd = nullptr;
    // create primitive desc.
    auto ret = dnnl_gated_mlp_primitive_desc_create(&pd, p_engine.get(),
            src_md.get(), wei0_md.get(), wei1_md.get(), wei2_md.get(),
            dst_md.get(), static_cast<dnnl_alg_kind_t>(act_algo), attr);
    if (pd && ret == dnnl_success) {
        pd_.reset(pd);
        // create primitive
        dnnl_primitive_t prim = nullptr;
        ret = dnnl_primitive_create(&prim, pd_.get());
        if (prim && ret == dnnl_success) { prim_.reset(prim); }
    }
}

void gated_mlp_executable_t::execute(const stream &stream,
        const std::unordered_map<int, memory> &args) const {
    UNUSED(stream);
    UNUSED(args);
    assert(!"gated_mlp_executable_t::execute() is not implemented on cpu");
}

#ifdef DNNL_WITH_SYCL
::sycl::event gated_mlp_executable_t::execute_sycl(const stream &stream,
        const std::unordered_map<int, memory> &args,
        const std::vector<::sycl::event> &deps) const {
    std::vector<dnnl_exec_arg_t> c_args;
    c_args.reserve(args.size());
    for (const auto &a : args)
        c_args.push_back({a.first, a.second.get()});

    sycl::event return_event;
    auto ret = dnnl_sycl_interop_primitive_execute(prim_.get(), stream.get(),
            c_args.size(), c_args.data(), &deps, &return_event);
    dnnl::error::wrap_c_api(
            ret, "could not execute gated mlp primitive with sycl runtime");

    return return_event;
}
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
cl_event gated_mlp_executable_t::execute_ocl(const stream &stream,
        const std::unordered_map<int, memory> &args,
        const std::vector<cl_event> &deps) const {
    std::vector<dnnl_exec_arg_t> c_args;
    c_args.reserve(args.size());
    for (const auto &a : args)
        c_args.push_back({a.first, a.second.get()});

    const cl_event *c_deps = deps.empty() ? nullptr : deps.data();

    cl_event return_event = nullptr;
    auto ret = dnnl_ocl_interop_primitive_execute(prim_.get(), stream.get(),
            static_cast<int>(c_args.size()), c_args.data(), c_deps,
            static_cast<int>(deps.size()), &return_event);
    dnnl::error::wrap_c_api(
            ret, "could not execute gated mlp primitive with ocl runtime");

    return return_event;
}
#endif

#define DNNL_ARG_WEIGHTS_GATE DNNL_ARG_WEIGHTS_0
#define DNNL_ARG_WEIGHTS_UP DNNL_ARG_WEIGHTS_1
#define DNNL_ARG_WEIGHTS_DOWN DNNL_ARG_WEIGHTS_2

arg_indices_t gated_mlp_executable_t::get_arg_indices(const op_t *op) {
    arg_indices_t args;
    // inputs: src, gate weights, up weights, down weights
    size_t idx = 0;
    args.insert({DNNL_ARG_SRC, {indices_t::type_t::input, idx++}});
    args.insert({DNNL_ARG_WEIGHTS_GATE, {indices_t::type_t::input, idx++}});
    args.insert({DNNL_ARG_WEIGHTS_UP, {indices_t::type_t::input, idx++}});
    args.insert({DNNL_ARG_WEIGHTS_DOWN, {indices_t::type_t::input, idx++}});

    // optional scales/zps for quantization
    const auto &fusion_info = op->has_attr(op_attr::fusion_info)
            ? op->get_attr<fusion_info_t>(op_attr::fusion_info)
            : fusion_info_t();
    if (fusion_info.with_runtime_scales(true, DNNL_ARG_WEIGHTS_GATE)) {
        args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS_GATE,
                {indices_t::type_t::input, idx++}});
    }
    if (fusion_info.with_runtime_zero_points(true, DNNL_ARG_WEIGHTS_GATE)) {
        args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS_GATE,
                {indices_t::type_t::input, idx++}});
    }
    if (fusion_info.with_runtime_scales(true, DNNL_ARG_WEIGHTS_UP)) {
        args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS_UP,
                {indices_t::type_t::input, idx++}});
    }
    if (fusion_info.with_runtime_zero_points(true, DNNL_ARG_WEIGHTS_UP)) {
        args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS_UP,
                {indices_t::type_t::input, idx++}});
    }
    if (fusion_info.with_runtime_scales(true, DNNL_ARG_WEIGHTS_DOWN)) {
        args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS_DOWN,
                {indices_t::type_t::input, idx++}});
    }
    if (fusion_info.with_runtime_zero_points(true, DNNL_ARG_WEIGHTS_DOWN)) {
        args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS_DOWN,
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
