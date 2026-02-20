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

#include "common/gated_mlp_iface.hpp"
#include "common/gated_mlp_pd.hpp"
#include "common/opdesc.hpp"
#include "common/primitive_desc_iface.hpp"

using namespace dnnl::impl;

status_t dnnl_gated_mlp_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        const memory_desc_t *src_desc, const memory_desc_t *weights_gate_desc,
        const memory_desc_t *weights_up_desc,
        const memory_desc_t *weights_down_desc, const memory_desc_t *dst_desc,
        alg_kind_t activation, const primitive_attr_t *attr) {
    auto gated_mlp_desc
            = dnnl::impl::create_gated_mlp_desc(src_desc, weights_gate_desc,
                    weights_up_desc, weights_down_desc, dst_desc, activation);
    return dnnl::impl::primitive_desc_create(primitive_desc_iface, engine,
            (const dnnl::impl::op_desc_t *)&gated_mlp_desc, nullptr, attr);
}
