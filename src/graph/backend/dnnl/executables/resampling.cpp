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

#include "graph/backend/dnnl/executables/resampling.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

resampling_executable_t::desc_t resampling_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const fpmath_t &fpmath, bool use_block_layout) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::resampling_forward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info)) {
        const fusion_info_t &fusion_info
                = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        prm_attr = make_dnnl_primitive_attr(op, fusion_info);
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    // resampling src doesn't support any
    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto dst = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    dst = to_format_any(dst);

    std::string mode = op->get_attr<std::string>(op_attr::mode);
    algorithm algo = algorithm::undef;
    if (mode == "nearest") {
        algo = algorithm::resampling_nearest;
    } else if (mode == "linear" || mode == "bilinear" || mode == "trilinear") {
        algo = algorithm::resampling_linear;
    } else {
        assert(!"unsupported resampling mode.");
    }

    dnnl::resampling_forward::primitive_desc pd;
    pd = dnnl::resampling_forward::primitive_desc(
            p_engine, prop_kind::forward_inference, algo, src, dst, prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

resampling_bwd_executable_t::desc_t resampling_bwd_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const fpmath_t &fpmath, bool use_block_layout) {
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::resampling_backward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info)) {
        const fusion_info_t &fusion_info
                = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        prm_attr = make_dnnl_primitive_attr(op, fusion_info);
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto mode = op->get_attr<std::string>(op_attr::mode);
    auto algo = algorithm::undef;
    if (mode == "nearest") {
        algo = algorithm::resampling_nearest;
    } else if (mode == "linear" || mode == "bilinear" || mode == "trilinear") {
        algo = algorithm::resampling_linear;
    } else {
        assert(!"unsupported resampling mode.");
    }

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto diff_dst = make_dnnl_memory_desc(
            op->get_input_value(1)->get_logical_tensor());
    dnnl::resampling_forward::primitive_desc fwd_hints(p_engine,
            prop_kind::forward_training, algo, src, to_format_any(diff_dst),
            prm_attr);

    auto diff_src = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    diff_src = to_format_any(diff_src);
    dnnl::resampling_backward::primitive_desc pd(
            p_engine, algo, diff_src, diff_dst, fwd_hints, prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

arg_indices_t resampling_executable_t::get_arg_indices(const op_t *op) {
    return get_arg_indices_for_siso_op(op);
}

arg_indices_t resampling_bwd_executable_t::get_arg_indices(const op_t *op) {
    arg_indices_t arg_indices;

    arg_indices.insert(
            {DNNL_ARG_DIFF_DST, indices_t {indices_t::type_t::input, 1}});

    arg_indices.insert(
            {DNNL_ARG_DIFF_SRC, indices_t {indices_t::type_t::output, 0}});
    arg_indices.insert(
            {DNNL_ARG_SCRATCHPAD, indices_t {indices_t::type_t::output, 1}});

    return arg_indices;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
