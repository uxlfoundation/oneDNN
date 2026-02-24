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

#include "graph/backend/dnnl/op_executable.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

using namespace dnnl::impl::graph::op_kind;

executable_creator_func op_func_t::get_executable_creator(op_kind_t kind) {
    static const std::unordered_map<op_kind_t, executable_creator_func> _map = {
            {_mul_scales, executable_creator<reorder_executable_t>},
            {_constant_scales, executable_creator<const_scales_filler>},
            {_add_zps, dummy_executable_creator},
            {_sub_zps, dummy_executable_creator},
            {_constant_zps, executable_creator<const_zps_filler>},
            {_permute, executable_creator<memory_reparser_t>},
            {_to_group, executable_creator<memory_reparser_t>},
            {_from_group, executable_creator<memory_reparser_t>},
            {_unsqueeze, executable_creator<memory_reparser_t>},
            {_squeeze, executable_creator<memory_reparser_t>},
            {_reshape, executable_creator<memory_reparser_t>},
            {_transpose, executable_creator<memory_reparser_t>},
            {_convolution, executable_creator<conv_fwd_executable_t>},
            {_convtranspose, executable_creator<deconv_fwd_executable_t>},
            {_pool, executable_creator<pool_executable_t>},
            {_bn_folding, executable_creator<bn_folding_t>},
            {_conv_bwd_data, executable_creator<conv_bwd_data_executable_t>},
            {_batchnorm, executable_creator<batchnorm_executable_t>},
            {_binary, executable_creator<binary_executable_t>},
            {_eltwise, executable_creator<eltwise_executable_t>},
            {_eltwise_bwd, executable_creator<eltwise_bwd_executable_t>},
            {_shuffle, executable_creator<shuffle_executable_t>},
            {_sum, executable_creator<sum_executable_t>},
            {_reduction, executable_creator<reduction_executable_t>},
            {_prelu, executable_creator<prelu_executable_t>},
            {_prelu_bwd, executable_creator<prelu_bwd_executable_t>},
            {_batchnorm_bwd, executable_creator<batchnorm_bwd_executable_t>},
            {_softmax_bwd, executable_creator<softmax_bwd_executable_t>},
            {_logsoftmax_bwd, executable_creator<softmax_bwd_executable_t>},
            {_resampling, executable_creator<resampling_executable_t>},
            {_resampling_bwd, executable_creator<resampling_bwd_executable_t>},
            {_concat, executable_creator<concat_executable_t>},
            {_layernorm_bwd, executable_creator<layernorm_bwd_executable_t>},
            {_conv_bwd_weights,
                    executable_creator<conv_bwd_weights_executable_t>},
            {_pool_bwd, executable_creator<pool_bwd_executable_t>},
            {_matmul, executable_creator<matmul_executable_t>},
            {_softmax, executable_creator<softmax_executable_t>},
            {_logsoftmax, executable_creator<softmax_executable_t>},
            {_layernorm, executable_creator<layernorm_executable_t>},
            {_reorder, executable_creator<reorder_executable_t>},
            {_convtranspose_bwd_data,
                    executable_creator<deconv_bwd_data_executable_t>},
            {_convtranspose_bwd_weights,
                    executable_creator<deconv_bwd_weights_executable_t>},
            {_groupnorm, executable_creator<groupnorm_executable_t>},
            {_gen_index, executable_creator<genindex_executable_t>},
            {_mask, executable_creator<memory_reparser_t>},
            {_sdpa, executable_creator<sdpa_executable_t>},
            {_host_scalar, executable_creator<host_scalar_executable_t>},
            {_identity, executable_creator<memory_reparser_t>},
            {_dropout, dummy_executable_creator},
    };

    if (_map.count(kind) == 0) {
        assert(!"no executable creator for the given op kind");
        return nullptr;
    }

    return _map.at(kind);
}

arg_indices_getter_func op_func_t::get_arg_indices_getter(op_kind_t kind) {
    static const std::unordered_map<op_kind_t, arg_indices_getter_func> _map = {
            {_mul_scales, reorder_executable_t::get_arg_indices},
            {_constant_scales, const_scales_filler::get_arg_indices},
            {_add_zps, dummy_arg_indices_getter},
            {_sub_zps, dummy_arg_indices_getter},
            {_constant_zps, const_zps_filler::get_arg_indices},
            {_permute, memory_reparser_t::get_arg_indices},
            {_to_group, memory_reparser_t::get_arg_indices},
            {_from_group, memory_reparser_t::get_arg_indices},
            {_unsqueeze, memory_reparser_t::get_arg_indices},
            {_squeeze, memory_reparser_t::get_arg_indices},
            {_reshape, memory_reparser_t::get_arg_indices},
            {_transpose, memory_reparser_t::get_arg_indices},
            {_convolution, conv_fwd_executable_t::get_arg_indices},
            {_convtranspose, deconv_fwd_executable_t::get_arg_indices},
            {_pool, pool_executable_t::get_arg_indices},
            {_bn_folding, bn_folding_t::get_arg_indices},
            {_conv_bwd_data, conv_bwd_data_executable_t::get_arg_indices},
            {_batchnorm, batchnorm_executable_t::get_arg_indices},
            {_binary, binary_executable_t::get_arg_indices},
            {_eltwise, eltwise_executable_t::get_arg_indices},
            {_eltwise_bwd, eltwise_bwd_executable_t::get_arg_indices},
            {_shuffle, shuffle_executable_t::get_arg_indices},
            {_sum, sum_executable_t::get_arg_indices},
            {_reduction, reduction_executable_t::get_arg_indices},
            {_prelu, prelu_executable_t::get_arg_indices},
            {_prelu_bwd, prelu_bwd_executable_t::get_arg_indices},
            {_batchnorm_bwd, batchnorm_bwd_executable_t::get_arg_indices},
            {_softmax_bwd, softmax_bwd_executable_t::get_arg_indices},
            {_logsoftmax_bwd, softmax_bwd_executable_t::get_arg_indices},
            {_resampling, resampling_executable_t::get_arg_indices},
            {_resampling_bwd, resampling_bwd_executable_t::get_arg_indices},
            {_concat, concat_executable_t::get_arg_indices},
            {_layernorm_bwd, layernorm_bwd_executable_t::get_arg_indices},
            {_conv_bwd_weights, conv_bwd_weights_executable_t::get_arg_indices},
            {_pool_bwd, pool_bwd_executable_t::get_arg_indices},
            {_matmul, matmul_executable_t::get_arg_indices},
            {_softmax, softmax_executable_t::get_arg_indices},
            {_logsoftmax, softmax_executable_t::get_arg_indices},
            {_layernorm, layernorm_executable_t::get_arg_indices},
            {_reorder, reorder_executable_t::get_arg_indices},
            {_convtranspose_bwd_data,
                    deconv_bwd_data_executable_t::get_arg_indices},
            {_convtranspose_bwd_weights,
                    deconv_bwd_weights_executable_t::get_arg_indices},
            {_groupnorm, groupnorm_executable_t::get_arg_indices},
            {_gen_index, genindex_executable_t::get_arg_indices},
            {_mask, memory_reparser_t::get_arg_indices},
            {_sdpa, sdpa_executable_t::get_arg_indices},
            {_host_scalar, host_scalar_executable_t::get_arg_indices},
            {_identity, memory_reparser_t::get_arg_indices},
            {_dropout, dummy_arg_indices_getter},
    };

    if (_map.count(kind) == 0) {
        assert(!"no arg indices getter for the given op kind");
        return nullptr;
    }

    return _map.at(kind);
}

layout_propagator_func op_func_t::get_layout_propagator(op_kind_t kind) {
    static const std::unordered_map<op_kind_t, layout_propagator_func> _map = {
            {_mul_scales, layout_propagator_for_mul_scales},
            {_constant_scales, layout_propagator_for_constant_filler},
            {_add_zps, layout_propagator_for_add_zps},
            {_sub_zps, layout_propagator_for_sub_zps},
            {_constant_zps, layout_propagator_for_constant_filler},
            {_permute, layout_propagator_for_permute},
            {_to_group, layout_propagator_for_to_group},
            {_from_group, layout_propagator_for_from_group},
            {_unsqueeze, layout_propagator_for_unsqueeze},
            {_squeeze, layout_propagator_for_squeeze},
            {_reshape, layout_propagator_for_reshape},
            {_transpose, layout_propagator_for_transpose},
            {_convolution, layout_propagator_for_conv},
            {_convtranspose, layout_propagator_for_deconv},
            {_pool, layout_propagator_for_pool},
            {_bn_folding, layout_propagator_for_bn_folding},
            {_conv_bwd_data, layout_propagator_for_conv_bwd_data},
            {_batchnorm, layout_propagator_for_batchnorm},
            {_binary, layout_propagator_for_binary},
            {_eltwise, layout_propagator_for_eltwise},
            {_eltwise_bwd, layout_propagator_for_eltwise_bwd},
            {_shuffle, layout_propagator_for_shuffle},
            {_sum, layout_propagator_for_sum},
            {_reduction, layout_propagator_for_reduction},
            {_prelu, layout_propagator_for_prelu},
            {_prelu_bwd, layout_propagator_for_prelu_bwd},
            {_batchnorm_bwd, layout_propagator_for_batchnorm_bwd},
            {_softmax_bwd, layout_propagator_for_softmax_bwd},
            {_logsoftmax_bwd, layout_propagator_for_softmax_bwd},
            {_resampling, layout_propagator_for_resampling},
            {_resampling_bwd, layout_propagator_for_resampling_bwd},
            {_concat, layout_propagator_for_concat},
            {_layernorm_bwd, layout_propagator_for_layernorm_bwd},
            {_conv_bwd_weights, layout_propagator_for_conv_bwd_weights},
            {_pool_bwd, layout_propagator_for_pool_bwd},
            {_matmul, layout_propagator_for_matmul},
            {_softmax, layout_propagator_for_softmax},
            {_logsoftmax, layout_propagator_for_softmax},
            {_layernorm, layout_propagator_for_layernorm},
            {_reorder, layout_propagator_for_reorder},
            {_convtranspose_bwd_data, layout_propagator_for_deconv_bwd_data},
            {_convtranspose_bwd_weights,
                    layout_propagator_for_deconv_bwd_weights},
            {_groupnorm, layout_propagator_for_groupnorm},
            {_gen_index, layout_propagator_for_gen_index},
            {_mask, layout_propagator_for_mask},
            {_sdpa, layout_propagator_for_sdpa},
            {_host_scalar, layout_propagator_for_host_scalar},
            {_identity, layout_propagator_for_identity},
    };

    if (_map.count(kind) == 0) {
        assert(!"no layout propagator for the given op kind");
        return nullptr;
    }

    return _map.at(kind);
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
