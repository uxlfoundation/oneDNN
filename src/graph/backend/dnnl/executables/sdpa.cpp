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

arg_indices_t sdpa_executable_t::get_arg_indices(const op_t *op) {
    arg_indices_t arg_indices;
    // Required input args: query, key, value
    size_t index = 0;
    arg_indices.insert(
            {DNNL_ARG_QUERIES, indices_t {indices_t::type_t::input, index++}});
    arg_indices.insert(
            {DNNL_ARG_KEYS, indices_t {indices_t::type_t::input, index++}});
    arg_indices.insert(
            {DNNL_ARG_VALUES, indices_t {indices_t::type_t::input, index++}});
    // Optional args: scale, mask
    if (op->get_attr<bool>(op_attr::with_scale)) {
        arg_indices.insert({DNNL_ARG_SCALE,
                indices_t {indices_t::type_t::input, index++}});
    }
    if (op->get_attr<int64_t>(op_attr::mask_type)
            == static_cast<int64_t>(attn_mask_type::buffer)) {
        arg_indices.insert({DNNL_ARG_ATTN_MASK,
                indices_t {indices_t::type_t::input, index++}});
    }

    const auto &sdpa_fusion_info = op->has_attr(op_attr::fusion_info)
            ? op->get_attr<fusion_info_t>(op_attr::fusion_info)
            : fusion_info_t();
    if (sdpa_fusion_info.with_runtime_scales(true, DNNL_ARG_KEYS)) {
        arg_indices.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_KEYS,
                indices_t {indices_t::type_t::input, index++}});
    }
    if (sdpa_fusion_info.with_runtime_zero_points(true, DNNL_ARG_KEYS)) {
        arg_indices.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_KEYS,
                indices_t {indices_t::type_t::input, index++}});
    }
    if (sdpa_fusion_info.with_runtime_scales(true, DNNL_ARG_VALUES)) {
        arg_indices.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_VALUES,
                indices_t {indices_t::type_t::input, index++}});
    }
    if (sdpa_fusion_info.with_runtime_zero_points(true, DNNL_ARG_VALUES)) {
        arg_indices.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_VALUES,
                indices_t {indices_t::type_t::input, index++}});
    }

    // add output args
    arg_indices.insert(
            {DNNL_ARG_DST, indices_t {indices_t::type_t::output, 0}});
    arg_indices.insert(
            {DNNL_ARG_SCRATCHPAD, indices_t {indices_t::type_t::output, 1}});
    return arg_indices;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
