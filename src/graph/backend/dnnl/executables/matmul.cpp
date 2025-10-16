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

#include "graph/backend/dnnl/executables/matmul.hpp"

#include "graph/backend/dnnl/dnnl_constant_tensor_cache.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

arg_indices_t matmul_executable_t::get_arg_indices(const op_t *op) {
    return get_arg_indices_for_conv_and_matmul(op);
}

matmul_executable_t::desc_t matmul_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const fpmath_t &fpmath, bool use_block_layout) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<dnnl::matmul::primitive_desc>(
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
    prm_attr.set_fpmath_mode(
            static_cast<dnnl::fpmath_mode>(fpmath.mode_), fpmath.apply_to_int_);

    if (op->has_attr(op_attr::accumulation_mode)) {
        const auto acc_mode
                = op->get_attr<std::string>(op_attr::accumulation_mode);
        prm_attr.set_accumulation_mode(str2accumulation_mode(acc_mode));
    }

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    // For non-constant activation and weight, create primitive desc with
    // strided layout
    bool const_activation
            = logical_tensor_wrapper_t(
                      op->get_input_value(0)->get_logical_tensor())
                      .is_constant()
            && is_constant_cache_enabled(p_engine);
    if (use_block_layout && const_activation) { src = to_format_any(src); }
    auto wei = make_dnnl_memory_desc(
            op->get_input_value(1)->get_logical_tensor());
    bool const_weight = logical_tensor_wrapper_t(
                                op->get_input_value(1)->get_logical_tensor())
                                .is_constant()
            && is_constant_cache_enabled(p_engine);
    if (use_block_layout && const_weight) { wei = to_format_any(wei); }
    auto dst = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    const bool keep_dst_layout = op->has_attr(op_attr::keep_dst_layout)
            && op->get_attr<bool>(op_attr::keep_dst_layout);
    if (dst.get_format_kind() == dnnl::memory::format_kind::any
            && !keep_dst_layout) {
        // convert to strided for avoiding blocked activation. The format kind
        // of dst is possible to be any when:
        // 1) It is created with internal logical tensor
        // 2) It is the partition output and defined by user
        dst = to_ncx_format(dst);
    }
    dnnl::matmul::primitive_desc pd;
    if (op->has_attr(op_attr::with_bias)
            && op->get_attr<bool>(op_attr::with_bias)) {
        auto bias = make_dnnl_memory_desc(
                op->get_input_value(2)->get_logical_tensor());
        bias = to_format_any(bias);
        pd = dnnl::matmul::primitive_desc(
                p_engine, src, wei, bias, dst, prm_attr);
    } else {
        pd = dnnl::matmul::primitive_desc(p_engine, src, wei, dst, prm_attr);
    }

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
