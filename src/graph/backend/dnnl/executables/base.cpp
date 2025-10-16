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

#include "graph/backend/dnnl/executables/base.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

void get_arg_indices_for_post_ops(
        const op_t *op, arg_indices_t &indices, size_t &base_index) {
    const fusion_info_t &fusion_info = op->has_attr(op_attr::fusion_info)
            ? op->get_attr<fusion_info_t>(op_attr::fusion_info)
            : fusion_info_t();
    const auto &pops = fusion_info.get_post_ops();
    for (size_t i = 0; i < pops.size(); i++) {
        if (pops[i]->is_post_sum()) {
            indices.insert({DNNL_GRAPH_ARG_POST_SRC,
                    {indices_t::type_t::input, base_index++}});
        } else if (pops[i]->get_op()->get_kind() == op_kind::dnnl_binary) {
            indices.insert(
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP((int)i) | DNNL_ARG_SRC_1,
                            {indices_t::type_t::input, base_index++}});
        } else if (pops[i]->get_op()->get_kind() == op_kind::dnnl_convolution) {
            indices.insert({DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS,
                    {indices_t::type_t::input, base_index++}});
            if (pops[i]->get_op()->num_inputs() > 2) {
                indices.insert({DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS,
                        {indices_t::type_t::input, base_index++}});
            }
        } else {
        }
    }
}

// for single-input-single-output op
arg_indices_t get_arg_indices_for_siso_op(const op_t *op) {
    arg_indices_t arg_indices;

    // add input args
    size_t index = 0;
    arg_indices.insert({DNNL_ARG_FROM, {indices_t::type_t::input, index++}});

    const fusion_info_t &fusion_info = op->has_attr(op_attr::fusion_info)
            ? op->get_attr<fusion_info_t>(op_attr::fusion_info)
            : fusion_info_t();

    get_arg_indices_for_post_ops(op, arg_indices, index);
    if (fusion_info.with_runtime_scales(false, 0)) {
        arg_indices.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST,
                {indices_t::type_t::input, index++}});
    }

    // add output args
    arg_indices.insert({DNNL_ARG_TO, {indices_t::type_t::output, 0}});
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, {indices_t::type_t::output, 1}});

    const bool is_training = op->has_attr(op_attr::is_training)
            ? op->get_attr<bool>(op_attr::is_training)
            : false;
    if (is_training) {
        arg_indices.insert(
                {DNNL_ARG_WORKSPACE, {indices_t::type_t::output, 2}});
    }

    return arg_indices;
}

arg_indices_t get_arg_indices_for_miso_op(const op_t *op) {
    arg_indices_t arg_indices;

    for (size_t i = 0; i < op->num_inputs(); ++i) {
        arg_indices.insert({DNNL_ARG_MULTIPLE_SRC + (int)i,
                {indices_t::type_t::input, static_cast<size_t>(i)}});
    }

    arg_indices.insert({DNNL_ARG_DST, {indices_t::type_t::output, 0}});
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, {indices_t::type_t::output, 1}});
    return arg_indices;
}

// Helper function for conv and deconv forward operations argument indices
arg_indices_t get_arg_indices_for_conv_and_matmul(const op_t *op) {
    arg_indices_t arg_indices;

    // add input args
    size_t index = 0;
    arg_indices.insert({DNNL_ARG_SRC, {indices_t::type_t::input, index++}});
    arg_indices.insert({DNNL_ARG_WEIGHTS, {indices_t::type_t::input, index++}});
    if (op->has_attr(op_attr::with_bias)
            && op->get_attr<bool>(op_attr::with_bias)) {
        arg_indices.insert(
                {DNNL_ARG_BIAS, {indices_t::type_t::input, index++}});
    }

    const fusion_info_t &fusion_info = op->has_attr(op_attr::fusion_info)
            ? op->get_attr<fusion_info_t>(op_attr::fusion_info)
            : fusion_info_t();

    if (fusion_info.with_runtime_scales(true, 0)) {
        arg_indices.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC,
                {indices_t::type_t::input, index++}});
    }

    if (fusion_info.with_runtime_scales(true, 1)) {
        arg_indices.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
                {indices_t::type_t::input, index++}});
    }

    if (fusion_info.with_runtime_zero_points(true, 0)) {
        arg_indices.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC,
                {indices_t::type_t::input, index++}});
    }

    if (fusion_info.with_runtime_zero_points(true, 1)) {
        arg_indices.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
                {indices_t::type_t::input, index++}});
    }

    get_arg_indices_for_post_ops(op, arg_indices, index);

    if (fusion_info.with_runtime_scales(false, 0)) {
        arg_indices.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST,
                {indices_t::type_t::input, index++}});
    }

    if (fusion_info.with_runtime_zero_points(false, 0)) {
        arg_indices.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST,
                {indices_t::type_t::input, index++}});
    }

    // add output args
    arg_indices.insert({DNNL_ARG_DST, {indices_t::type_t::output, 0}});
    arg_indices.insert({DNNL_ARG_SCRATCHPAD, {indices_t::type_t::output, 1}});

    return arg_indices;
}

arg_indices_t get_arg_indices_for_lnorm_and_gnorm(const op_t *op) {
    arg_indices_t arg_indices;

    size_t in_index = 0;
    arg_indices.insert({DNNL_ARG_SRC, {indices_t::type_t::input, in_index++}});
    if (!op->has_attr(op_attr::use_affine)
            || op->get_attr<bool>(op_attr::use_affine)) {
        arg_indices.insert(
                {DNNL_ARG_SCALE, {indices_t::type_t::input, in_index++}});
        arg_indices.insert(
                {DNNL_ARG_SHIFT, {indices_t::type_t::input, in_index++}});
    }

    const fusion_info_t &fusion_info = op->has_attr(op_attr::fusion_info)
            ? op->get_attr<fusion_info_t>(op_attr::fusion_info)
            : fusion_info_t();

    get_arg_indices_for_post_ops(op, arg_indices, in_index);

    if (fusion_info.with_runtime_scales(false, 0)) {
        arg_indices.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST,
                {indices_t::type_t::input, in_index++}});
    }

    size_t out_index = 0;
    arg_indices.insert(
            {DNNL_ARG_DST, {indices_t::type_t::output, out_index++}});
    if (!op->has_attr(op_attr::keep_stats)
            || op->get_attr<bool>(op_attr::keep_stats)) {
        arg_indices.insert(
                {DNNL_ARG_MEAN, {indices_t::type_t::output, out_index++}});
        arg_indices.insert(
                {DNNL_ARG_VARIANCE, {indices_t::type_t::output, out_index++}});
    }

    arg_indices.insert(
            {DNNL_ARG_SCRATCHPAD, {indices_t::type_t::output, out_index++}});

    return arg_indices;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
