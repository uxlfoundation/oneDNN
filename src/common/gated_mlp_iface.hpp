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

#ifndef COMMON_GATED_MLP_IFACE_HPP
#define COMMON_GATED_MLP_IFACE_HPP

#include "oneapi/dnnl/dnnl_types.h"

/// Creates a primitive descriptor for a gated mlp primitive.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param src_desc Source memory descriptor.
/// @param weights_gate_desc Gate weights memory descriptor.
/// @param weights_up_desc Up weights memory descriptor.
/// @param weights_down_desc Down weights memory descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param activation Gated MLP activation. Possible values are
///     #dnnl_eltwise_gelu_erf, #dnnl_eltwise_gelu_tanh, #dnnl_eltwise_swish.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_gated_mlp_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        const_dnnl_memory_desc_t src_desc,
        const_dnnl_memory_desc_t weights_gate_desc,
        const_dnnl_memory_desc_t weights_up_desc,
        const_dnnl_memory_desc_t weights_down_desc,
        const_dnnl_memory_desc_t dst_desc, dnnl_alg_kind_t activation,
        const_dnnl_primitive_attr_t attr);

#endif
