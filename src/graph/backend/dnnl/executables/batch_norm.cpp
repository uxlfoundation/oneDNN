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

#include "graph/backend/dnnl/executables/batch_norm.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

batchnorm_executable_t::desc_t batchnorm_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const fpmath_t &fpmath, bool use_block_layout) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::batch_normalization_forward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    float epsilon = op->get_attr<float>(op_attr::epsilon);

    auto flags = dnnl::normalization_flags::none;
    // for inference
    if (!op->get_attr<bool>(op_attr::is_training)) {
        flags |= dnnl::normalization_flags::use_global_stats;
        flags |= dnnl::normalization_flags::use_scale;
        flags |= dnnl::normalization_flags::use_shift;
    } else {
        // for training, inputs: [src, mean, variance, gamma, beta]
        if (op->num_inputs() > 3) {
            flags |= dnnl::normalization_flags::use_scale;
            flags |= dnnl::normalization_flags::use_shift;
        }

        if (op->has_attr(op_attr::fuse_relu)
                && op->get_attr<bool>(op_attr::fuse_relu))
            flags |= dnnl::normalization_flags::fuse_norm_relu;
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info)) {
        const fusion_info_t &fusion_info
                = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        prm_attr = make_dnnl_primitive_attr(op, fusion_info);
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto dst = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());
    dst = to_format_any(dst);

    if (src.get_inner_nblks() == 1 && src.get_inner_idxs()[0] == 1
            && src.get_inner_blks()[0] == 4) {
        // to default format
        src = to_ncx_format(src);
    }

    auto pkind = op->get_attr<bool>(op_attr::is_training)
            ? prop_kind::forward_training
            : prop_kind::forward_inference;

    dnnl::batch_normalization_forward::primitive_desc pd(
            p_engine, pkind, src, dst, epsilon, flags, prm_attr);

    pd_cache.insert({op.get(), pd});
    return {pd, false};
}

batchnorm_bwd_executable_t::desc_t batchnorm_bwd_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const fpmath_t &fpmath, bool use_block_layout) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::batch_normalization_backward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    float epsilon = op->get_attr<float>(op_attr::epsilon);

    auto flags = dnnl::normalization_flags::none;
    // [diff_src, diff_scale, diff_shift, scratchpad]
    if (op->num_outputs() > 2) {
        flags |= dnnl::normalization_flags::use_scale;
        flags |= dnnl::normalization_flags::use_shift;
    } else {
        // [diff_src, scratchpad]
        flags |= dnnl::normalization_flags::use_global_stats;
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info)) {
        const fusion_info_t &fusion_info
                = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        prm_attr = make_dnnl_primitive_attr(op, fusion_info);
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());

    if (src.get_inner_nblks() == 1 && src.get_inner_idxs()[0] == 1
            && src.get_inner_blks()[0] == 4) {
        // to default format
        src = to_ncx_format(src);
    }

    auto forward_hints = dnnl::batch_normalization_forward::primitive_desc(
            p_engine, prop_kind::forward_training, src, src, epsilon, flags);

    dnnl::batch_normalization_backward::primitive_desc pd(p_engine,
            prop_kind::backward, src, forward_hints.dst_desc(), src, epsilon,
            flags, forward_hints);

    pd_cache.insert({op.get(), pd});
    return {pd, false};
}

bn_folding_t::desc_t bn_folding_t::create_desc(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const fpmath_t &fpmath, bool use_block_layout) {
    UNUSED(pd_cache);
    UNUSED(fpmath);
    UNUSED(use_block_layout);

    desc_t desc;

    desc.epsilon_ = op->get_attr<float>(op_attr::epsilon);
    desc.data_format_ = op->get_attr<std::string>(op_attr::data_format);
    desc.filter_format_ = op->get_attr<std::string>(op_attr::weights_format);
    desc.with_bias_ = op->get_attr<bool>(op_attr::with_bias);

    size_t in_idx = 0;
    auto weights = make_dnnl_memory_desc(
            op->get_input_value(in_idx++)->get_logical_tensor());
    auto bias = desc.with_bias_ ? make_dnnl_memory_desc(
                        op->get_input_value(in_idx++)->get_logical_tensor())
                                : memory::desc();
    auto scale = make_dnnl_memory_desc(
            op->get_input_value(in_idx++)->get_logical_tensor());
    auto shift = make_dnnl_memory_desc(
            op->get_input_value(in_idx++)->get_logical_tensor());
    auto mean = make_dnnl_memory_desc(
            op->get_input_value(in_idx++)->get_logical_tensor());
    auto variance = make_dnnl_memory_desc(
            op->get_input_value(in_idx++)->get_logical_tensor());

    // 1. sqrt_variance = sqrt(variance + epsilon)

    // temp = variance + epsilon
    memory::dims epsilon_dims(variance.get_ndims(), 1);
    desc.epsilon_desc_ = memory::desc(
            epsilon_dims, memory::data_type::f32, memory::format_tag::a);

#if DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE \
        && DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
    // binary + sqrt post-op fusion is unsupported on NVIDIA GPU
    if (p_engine.get_kind() == dnnl::engine::kind::gpu) {
        primitive_attr add_attr;
        desc.add_pd_
                = dnnl::binary::primitive_desc(p_engine, algorithm::binary_add,
                        variance, desc.epsilon_desc_, variance, add_attr);

        primitive_attr sqrt_attr;
        desc.sqrt_pd_ = dnnl::eltwise_forward::primitive_desc(p_engine,
                prop_kind::forward, algorithm::eltwise_sqrt, variance, variance,
                0.0f, 0.0f, sqrt_attr);
    } else {
        post_ops add_post_ops;
        // sqrt_variance = sqrt(temp)
        add_post_ops.append_eltwise(algorithm::eltwise_sqrt, 0.0f, 0.0f);

        primitive_attr add_attr;
        add_attr.set_post_ops(add_post_ops);
        desc.add_pd_
                = dnnl::binary::primitive_desc(p_engine, algorithm::binary_add,
                        variance, desc.epsilon_desc_, variance, add_attr);
    }

#else
    post_ops add_post_ops;
    // sqrt_variance = sqrt(temp)
    add_post_ops.append_eltwise(algorithm::eltwise_sqrt, 0.0f, 0.0f);

    primitive_attr add_attr;
    add_attr.set_post_ops(add_post_ops);
    desc.add_pd_ = dnnl::binary::primitive_desc(p_engine, algorithm::binary_add,
            variance, desc.epsilon_desc_, variance, add_attr);

#endif
    // 2. updated_weight = weights * scale / sqrt_variance

    // expand 1D scale and variance to same ndims with weights
    desc.new_scale_desc_ = expand(scale, weights.get_ndims());
    desc.new_variance_desc_ = expand(variance, weights.get_ndims());

    // after expand, the c channel is on the last dimension, which
    // meet the requirement of NXC format. But for NCX format, we
    // need permute c channel to the second dimension
    if (desc.filter_format_ == "NCX") { // matmul case
        auto perm = dnnl_impl::utils::cast_to_int32(get_permutation(
                desc.new_scale_desc_.get_ndims(), "NXC", "NCX"));
        desc.new_scale_desc_ = desc.new_scale_desc_.permute_axes(perm);
        desc.new_variance_desc_ = desc.new_variance_desc_.permute_axes(perm);
    }

    // after expand, the c channel is on the last dimension, which
    // meet the requirement of XIO format. But for OIX format, we
    // need permute c channel to the first dimension
    if (desc.filter_format_ == "OIX") { // conv case
        auto perm = dnnl_impl::utils::cast_to_int32(get_permutation(
                desc.new_scale_desc_.get_ndims(), "XIO", "OIX"));
        desc.new_scale_desc_ = desc.new_scale_desc_.permute_axes(perm);
        desc.new_variance_desc_ = desc.new_variance_desc_.permute_axes(perm);
    }

    // temp = weights * scale
    post_ops mul_post_ops;
    // updated_weight = temp / sqrt_variance
    mul_post_ops.append_binary(algorithm::binary_div, desc.new_variance_desc_);

    primitive_attr mul_attr;
    mul_attr.set_post_ops(mul_post_ops);
    desc.mul_pd_ = dnnl::binary::primitive_desc(p_engine, algorithm::binary_mul,
            weights, desc.new_scale_desc_, weights, mul_attr);

    // 3. updated_bias = (bias - mean) * scale / sqrt_variance + shift

    // temp = bias - mean
    memory::desc valid_bias = bias.is_zero() ? mean : bias;

    post_ops sub_post_ops;
    // temp = temp * scale
    sub_post_ops.append_binary(algorithm::binary_mul, scale);
    // temp = temp / sqrt_variance
    sub_post_ops.append_binary(algorithm::binary_div, variance);
    // temp = temp + shift
    sub_post_ops.append_binary(algorithm::binary_add, shift);

    primitive_attr sub_attr;
    sub_attr.set_post_ops(sub_post_ops);
    desc.sub_pd_ = dnnl::binary::primitive_desc(p_engine, algorithm::binary_sub,
            valid_bias, mean, valid_bias, sub_attr);

    memory::dims scratchpad_dims = variance.get_dims();
    // sqrt_variance, zero_bias and others (like epsilon),
    // or no need to alloc bias
    // binary + sqrt post-op fusion is unsupported on NVIDIA GPU, so we need
    // one more scratchpad for sqrt
#if DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE \
        && DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
    size_t factor = 0;
    if (p_engine.get_kind() == dnnl::engine::kind::gpu) {
        factor = bias.is_zero() ? 4 : 3;
    } else {
        factor = bias.is_zero() ? 3 : 2;
    }
#else
    size_t factor = bias.is_zero() ? 3 : 2;
#endif
    scratchpad_dims[0] *= factor;
    desc.scratchpad_desc_ = memory::desc(
            scratchpad_dims, variance.get_data_type(), memory::format_tag::a);

    return desc;
}

arg_indices_t bn_folding_t::get_arg_indices(const op_t *op) {
    arg_indices_t arg_indices;

    size_t in_idx = 0;
    arg_indices.insert(
            {DNNL_ARG_WEIGHTS, indices_t {indices_t::type_t::input, in_idx++}});
    if (op->get_attr<bool>(op_attr::with_bias)) {
        arg_indices.insert({DNNL_ARG_BIAS,
                indices_t {indices_t::type_t::input, in_idx++}});
    }
    arg_indices.insert({DNNL_ARG_WEIGHTS_1,
            indices_t {indices_t::type_t::input, in_idx++}}); // scale
    arg_indices.insert({DNNL_ARG_WEIGHTS_2,
            indices_t {indices_t::type_t::input, in_idx++}}); // shift
    arg_indices.insert({DNNL_ARG_MEAN,
            indices_t {indices_t::type_t::input, in_idx++}}); // mean
    arg_indices.insert({DNNL_ARG_VARIANCE,
            indices_t {indices_t::type_t::input, in_idx++}}); // variance

    // bind output memory
    size_t out_idx = 0;
    arg_indices.insert({DNNL_ARG_DST_0,
            indices_t {
                    indices_t::type_t::output, out_idx++}}); // updated weight
    arg_indices.insert({DNNL_ARG_DST_1,
            indices_t {indices_t::type_t::output, out_idx++}}); // updated bias
    arg_indices.insert({DNNL_ARG_SCRATCHPAD,
            indices_t {indices_t::type_t::output, out_idx++}}); // scratchpad

    return arg_indices;
}

arg_indices_t batchnorm_executable_t::get_arg_indices(const op_t *op) {
    arg_indices_t arg_indices;

    size_t in_index = 0;
    arg_indices.insert(
            {DNNL_ARG_SRC, indices_t {indices_t::type_t::input, in_index++}});
    if (!op->get_attr<bool>(op_attr::is_training)) { // inference
        arg_indices.insert({DNNL_ARG_SCALE,
                indices_t {indices_t::type_t::input, in_index++}});
        arg_indices.insert({DNNL_ARG_SHIFT,
                indices_t {indices_t::type_t::input, in_index++}});
        arg_indices.insert({DNNL_ARG_MEAN,
                indices_t {indices_t::type_t::input, in_index++}});
        arg_indices.insert({DNNL_ARG_VARIANCE,
                indices_t {indices_t::type_t::input, in_index++}});
    } else { // training
        // running_mean/running_variance of last iteration
        arg_indices.insert({DNNL_ARG_SRC_1,
                indices_t {indices_t::type_t::input, in_index++}});
        arg_indices.insert({DNNL_ARG_SRC_2,
                indices_t {indices_t::type_t::input, in_index++}});

        if (op->num_inputs() > 3) {
            arg_indices.insert({DNNL_ARG_SCALE,
                    indices_t {indices_t::type_t::input, in_index++}});
            arg_indices.insert({DNNL_ARG_SHIFT,
                    indices_t {indices_t::type_t::input, in_index++}});
        }
    }

    size_t out_index = 0;
    arg_indices.insert(
            {DNNL_ARG_DST, indices_t {indices_t::type_t::output, out_index++}});
    if (op->get_attr<bool>(op_attr::is_training)) {
        // running_mean
        arg_indices.insert({DNNL_ARG_DST_1,
                indices_t {indices_t::type_t::output, out_index++}});
        // running_variance
        arg_indices.insert({DNNL_ARG_DST_2,
                indices_t {indices_t::type_t::output, out_index++}});
        // batch_mean
        arg_indices.insert({DNNL_ARG_MEAN,
                indices_t {indices_t::type_t::output, out_index++}});
        // batch_variance
        arg_indices.insert({DNNL_ARG_VARIANCE,
                indices_t {indices_t::type_t::output, out_index++}});
    }

    if (op->num_outputs() > out_index) {
        arg_indices.insert({DNNL_ARG_SCRATCHPAD,
                indices_t {indices_t::type_t::output, out_index++}});
    }

    // workspace (for BatchNormForwardTraining with ReLU)
    if (op->num_outputs() > out_index) {
        arg_indices.insert({DNNL_ARG_WORKSPACE,
                indices_t {indices_t::type_t::output, out_index++}});
    }

    return arg_indices;
}

arg_indices_t batchnorm_bwd_executable_t::get_arg_indices(const op_t *op) {
    arg_indices_t arg_indices;
    size_t index = 0;

    arg_indices.insert(
            {DNNL_ARG_SRC, indices_t {indices_t::type_t::input, index++}});
    arg_indices.insert(
            {DNNL_ARG_DIFF_DST, indices_t {indices_t::type_t::input, index++}});

    arg_indices.insert(
            {DNNL_ARG_MEAN, indices_t {indices_t::type_t::input, index++}});
    arg_indices.insert(
            {DNNL_ARG_VARIANCE, indices_t {indices_t::type_t::input, index++}});

    if (op->num_outputs() > 2) {
        // oneDNN only need the scales now
        arg_indices.insert({DNNL_ARG_SCALE,
                indices_t {indices_t::type_t::input, index++}});
    }

    index = 0;
    arg_indices.insert({DNNL_ARG_DIFF_SRC,
            indices_t {indices_t::type_t::output, index++}});
    // check if has diff_scale and diff_shift outputs
    if (op->num_outputs() > 2) {
        arg_indices.insert({DNNL_ARG_DIFF_SCALE,
                indices_t {indices_t::type_t::output, index++}});
        arg_indices.insert({DNNL_ARG_DIFF_SHIFT,
                indices_t {indices_t::type_t::output, index++}});
    }

    if (op->num_outputs() > index) {
        arg_indices.insert({DNNL_ARG_SCRATCHPAD,
                indices_t {indices_t::type_t::output, index++}});
    }

    return arg_indices;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
