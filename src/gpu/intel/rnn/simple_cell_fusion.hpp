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

#ifndef GPU_INTEL_RNN_SIMPLE_CELL_FUSION_HPP
#define GPU_INTEL_RNN_SIMPLE_CELL_FUSION_HPP

#include "gpu/intel/rnn/grid.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace rnn {

using namespace rnn_utils;

template <size_t out_ndims, size_t in_ndims>
strides_t<out_ndims> inner(const strides_t<in_ndims> &s);

status_t compute_cell_fwd(const exec_ctx_t &ctx,
        const compute::kernel_t &kernel, dim_t lay, dim_t dir, dim_t iter,
        const workspace_t &workspace, const user_data_t user_data,
        const sub_buffer_t &weights_layer, const sub_buffer_t &weights_iter,
        const sub_buffer_t &cell_layer, const strides_t<4> &cell_layer_strides,
        const sub_buffer_t &cell_iter, const strides_t<4> &cell_iter_strides,
        const sub_buffer_t &scratch_gates,
        const strides_t<2> &scratch_gates_strides,
        const memory_storage_t &scratch_cell, float alpha,
        const memory_storage_t *tm_scales, const conf_t &conf,
        const ocl_conf_t &ocl_conf, const rnn_offsets_t &offsets);

} // namespace rnn
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
