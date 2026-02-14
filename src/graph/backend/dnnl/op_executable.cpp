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
            {_dnnl_mul_scales, executable_creator<reorder_executable_t>},
            {_dnnl_constant_scales, executable_creator<const_scales_filler>},
            {_dnnl_add_zps, dummy_executable_creator},
            {_dnnl_sub_zps, dummy_executable_creator},
            {_dnnl_constant_zps, executable_creator<const_zps_filler>},
            {_dnnl_permute, executable_creator<memory_reparser_t>},
            {_dnnl_to_group, executable_creator<memory_reparser_t>},
            {_dnnl_from_group, executable_creator<memory_reparser_t>},
            {_dnnl_unsqueeze, executable_creator<memory_reparser_t>},
            {_dnnl_squeeze, executable_creator<memory_reparser_t>},
            {_dnnl_reshape, executable_creator<memory_reparser_t>},
            {_dnnl_transpose, executable_creator<memory_reparser_t>},
            {_dnnl_convolution, executable_creator<conv_fwd_executable_t>},
            {_dnnl_convtranspose, executable_creator<deconv_fwd_executable_t>},
            {_dnnl_pool, executable_creator<pool_executable_t>},
            {_dnnl_bn_folding, executable_creator<bn_folding_t>},
            {_dnnl_conv_bwd_data,
                    executable_creator<conv_bwd_data_executable_t>},
            {_dnnl_batchnorm, executable_creator<batchnorm_executable_t>},
            {_dnnl_binary, executable_creator<binary_executable_t>},
            {_dnnl_eltwise, executable_creator<eltwise_executable_t>},
            {_dnnl_eltwise_bwd, executable_creator<eltwise_bwd_executable_t>},
            {_dnnl_shuffle, executable_creator<shuffle_executable_t>},
            {_dnnl_sum, executable_creator<sum_executable_t>},
            {_dnnl_reduction, executable_creator<reduction_executable_t>},
            {_dnnl_prelu, executable_creator<prelu_executable_t>},
            {_dnnl_prelu_bwd, executable_creator<prelu_bwd_executable_t>},
            {_dnnl_batchnorm_bwd,
                    executable_creator<batchnorm_bwd_executable_t>},
            {_dnnl_softmax_bwd, executable_creator<softmax_bwd_executable_t>},
            {_dnnl_logsoftmax_bwd,
                    executable_creator<softmax_bwd_executable_t>},
            {_dnnl_resampling, executable_creator<resampling_executable_t>},
            {_dnnl_resampling_bwd,
                    executable_creator<resampling_bwd_executable_t>},
            {_dnnl_concat, executable_creator<concat_executable_t>},
            {_dnnl_layernorm_bwd,
                    executable_creator<layernorm_bwd_executable_t>},
            {_dnnl_conv_bwd_weights,
                    executable_creator<conv_bwd_weights_executable_t>},
            {_dnnl_pool_bwd, executable_creator<pool_bwd_executable_t>},
            {_dnnl_matmul, executable_creator<matmul_executable_t>},
            {_dnnl_softmax, executable_creator<softmax_executable_t>},
            {_dnnl_logsoftmax, executable_creator<softmax_executable_t>},
            {_dnnl_layernorm, executable_creator<layernorm_executable_t>},
            {_dnnl_reorder, executable_creator<reorder_executable_t>},
            {_dnnl_convtranspose_bwd_data,
                    executable_creator<deconv_bwd_data_executable_t>},
            {_dnnl_convtranspose_bwd_weights,
                    executable_creator<deconv_bwd_weights_executable_t>},
            {_dnnl_groupnorm, executable_creator<groupnorm_executable_t>},
            {_dnnl_gen_index, executable_creator<genindex_executable_t>},
            {_dnnl_mask, executable_creator<memory_reparser_t>},
            {_dnnl_sdpa, executable_creator<sdpa_executable_t>},
            {_dnnl_host_scalar, executable_creator<host_scalar_executable_t>},
            {_dnnl_identity, executable_creator<memory_reparser_t>},
            {_dnnl_dropout, dummy_executable_creator},
    };

    if (_map.count(kind) == 0) {
        assert(!"no executable creator for the given op kind");
        return nullptr;
    }

    return _map.at(kind);
}

arg_indices_getter_func op_func_t::get_arg_indices_getter(op_kind_t kind) {
    static const std::unordered_map<op_kind_t, arg_indices_getter_func> _map = {
            {_dnnl_mul_scales, reorder_executable_t::get_arg_indices},
            {_dnnl_constant_scales, const_scales_filler::get_arg_indices},
            {_dnnl_add_zps, dummy_arg_indices_getter},
            {_dnnl_sub_zps, dummy_arg_indices_getter},
            {_dnnl_constant_zps, const_zps_filler::get_arg_indices},
            {_dnnl_permute, memory_reparser_t::get_arg_indices},
            {_dnnl_to_group, memory_reparser_t::get_arg_indices},
            {_dnnl_from_group, memory_reparser_t::get_arg_indices},
            {_dnnl_unsqueeze, memory_reparser_t::get_arg_indices},
            {_dnnl_squeeze, memory_reparser_t::get_arg_indices},
            {_dnnl_reshape, memory_reparser_t::get_arg_indices},
            {_dnnl_transpose, memory_reparser_t::get_arg_indices},
            {_dnnl_convolution, conv_fwd_executable_t::get_arg_indices},
            {_dnnl_convtranspose, deconv_fwd_executable_t::get_arg_indices},
            {_dnnl_pool, pool_executable_t::get_arg_indices},
            {_dnnl_bn_folding, bn_folding_t::get_arg_indices},
            {_dnnl_conv_bwd_data, conv_bwd_data_executable_t::get_arg_indices},
            {_dnnl_batchnorm, batchnorm_executable_t::get_arg_indices},
            {_dnnl_binary, binary_executable_t::get_arg_indices},
            {_dnnl_eltwise, eltwise_executable_t::get_arg_indices},
            {_dnnl_eltwise_bwd, eltwise_bwd_executable_t::get_arg_indices},
            {_dnnl_shuffle, shuffle_executable_t::get_arg_indices},
            {_dnnl_sum, sum_executable_t::get_arg_indices},
            {_dnnl_reduction, reduction_executable_t::get_arg_indices},
            {_dnnl_prelu, prelu_executable_t::get_arg_indices},
            {_dnnl_prelu_bwd, prelu_bwd_executable_t::get_arg_indices},
            {_dnnl_batchnorm_bwd, batchnorm_bwd_executable_t::get_arg_indices},
            {_dnnl_softmax_bwd, softmax_bwd_executable_t::get_arg_indices},
            {_dnnl_logsoftmax_bwd, softmax_bwd_executable_t::get_arg_indices},
            {_dnnl_resampling, resampling_executable_t::get_arg_indices},
            {_dnnl_resampling_bwd,
                    resampling_bwd_executable_t::get_arg_indices},
            {_dnnl_concat, concat_executable_t::get_arg_indices},
            {_dnnl_layernorm_bwd, layernorm_bwd_executable_t::get_arg_indices},
            {_dnnl_conv_bwd_weights,
                    conv_bwd_weights_executable_t::get_arg_indices},
            {_dnnl_pool_bwd, pool_bwd_executable_t::get_arg_indices},
            {_dnnl_matmul, matmul_executable_t::get_arg_indices},
            {_dnnl_softmax, softmax_executable_t::get_arg_indices},
            {_dnnl_logsoftmax, softmax_executable_t::get_arg_indices},
            {_dnnl_layernorm, layernorm_executable_t::get_arg_indices},
            {_dnnl_reorder, reorder_executable_t::get_arg_indices},
            {_dnnl_convtranspose_bwd_data,
                    deconv_bwd_data_executable_t::get_arg_indices},
            {_dnnl_convtranspose_bwd_weights,
                    deconv_bwd_weights_executable_t::get_arg_indices},
            {_dnnl_groupnorm, groupnorm_executable_t::get_arg_indices},
            {_dnnl_gen_index, genindex_executable_t::get_arg_indices},
            {_dnnl_mask, memory_reparser_t::get_arg_indices},
            {_dnnl_sdpa, sdpa_executable_t::get_arg_indices},
            {_dnnl_host_scalar, host_scalar_executable_t::get_arg_indices},
            {_dnnl_identity, memory_reparser_t::get_arg_indices},
            {_dnnl_dropout, dummy_arg_indices_getter},
    };

    if (_map.count(kind) == 0) {
        assert(!"no arg indices getter for the given op kind");
        return nullptr;
    }

    return _map.at(kind);
}

layout_propagator_func op_func_t::get_layout_propagator(op_kind_t kind) {
    static const std::unordered_map<op_kind_t, layout_propagator_func> _map = {
            {_dnnl_mul_scales, layout_propagator_for_mul_scales},
            {_dnnl_constant_scales, layout_propagator_for_constant_filler},
            {_dnnl_add_zps, layout_propagator_for_add_zps},
            {_dnnl_sub_zps, layout_propagator_for_sub_zps},
            {_dnnl_constant_zps, layout_propagator_for_constant_filler},
            {_dnnl_permute, layout_propagator_for_permute},
            {_dnnl_to_group, layout_propagator_for_to_group},
            {_dnnl_from_group, layout_propagator_for_from_group},
            {_dnnl_unsqueeze, layout_propagator_for_unsqueeze},
            {_dnnl_squeeze, layout_propagator_for_squeeze},
            {_dnnl_reshape, layout_propagator_for_reshape},
            {_dnnl_transpose, layout_propagator_for_transpose},
            {_dnnl_convolution, layout_propagator_for_conv},
            {_dnnl_convtranspose, layout_propagator_for_deconv},
            {_dnnl_pool, layout_propagator_for_pool},
            {_dnnl_bn_folding, layout_propagator_for_bn_folding},
            {_dnnl_conv_bwd_data, layout_propagator_for_conv_bwd_data},
            {_dnnl_batchnorm, layout_propagator_for_batchnorm},
            {_dnnl_binary, layout_propagator_for_binary},
            {_dnnl_eltwise, layout_propagator_for_eltwise},
            {_dnnl_eltwise_bwd, layout_propagator_for_eltwise_bwd},
            {_dnnl_shuffle, layout_propagator_for_shuffle},
            {_dnnl_sum, layout_propagator_for_sum},
            {_dnnl_reduction, layout_propagator_for_reduction},
            {_dnnl_prelu, layout_propagator_for_prelu},
            {_dnnl_prelu_bwd, layout_propagator_for_prelu_bwd},
            {_dnnl_batchnorm_bwd, layout_propagator_for_batchnorm_bwd},
            {_dnnl_softmax_bwd, layout_propagator_for_softmax_bwd},
            {_dnnl_logsoftmax_bwd, layout_propagator_for_softmax_bwd},
            {_dnnl_resampling, layout_propagator_for_resampling},
            {_dnnl_resampling_bwd, layout_propagator_for_resampling_bwd},
            {_dnnl_concat, layout_propagator_for_concat},
            {_dnnl_layernorm_bwd, layout_propagator_for_layernorm_bwd},
            {_dnnl_conv_bwd_weights, layout_propagator_for_conv_bwd_weights},
            {_dnnl_pool_bwd, layout_propagator_for_pool_bwd},
            {_dnnl_matmul, layout_propagator_for_matmul},
            {_dnnl_softmax, layout_propagator_for_softmax},
            {_dnnl_logsoftmax, layout_propagator_for_softmax},
            {_dnnl_layernorm, layout_propagator_for_layernorm},
            {_dnnl_reorder, layout_propagator_for_reorder},
            {_dnnl_convtranspose_bwd_data,
                    layout_propagator_for_deconv_bwd_data},
            {_dnnl_convtranspose_bwd_weights,
                    layout_propagator_for_deconv_bwd_weights},
            {_dnnl_groupnorm, layout_propagator_for_groupnorm},
            {_dnnl_gen_index, layout_propagator_for_gen_index},
            {_dnnl_mask, layout_propagator_for_mask},
            {_dnnl_sdpa, layout_propagator_for_sdpa},
            {_dnnl_host_scalar, layout_propagator_for_host_scalar},
            {_dnnl_identity, layout_propagator_for_identity},
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
