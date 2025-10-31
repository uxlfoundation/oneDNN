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

#include "graph/backend/dnnl/executables/reduction.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

reduction_executable_t::desc_t reduction_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const fpmath_t &fpmath, bool use_block_layout) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<dnnl::reduction::primitive_desc>(
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

    const algorithm alg = static_cast<dnnl::algorithm>(
            op->get_attr<int64_t>(op_attr::alg_kind));
    if (alg == algorithm::undef) { assert(!"unsupported reduction op."); }
    float p = op->has_attr(op_attr::p) ? op->get_attr<float>(op_attr::p) : 0.f;

    float eps = 0.0f;

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto dst = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    dst = to_format_any(dst);

    dnnl::reduction::primitive_desc pd(
            p_engine, alg, src, dst, p, eps, prm_attr);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

arg_indices_t reduction_executable_t::get_arg_indices(const op_t *op) {
    return get_arg_indices_for_siso_op(op);
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
