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

#include <iostream>

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "engine.hpp"
#include "grouped_gemm_pd.hpp"
#include "memory_desc.hpp"
#include "primitive_cache.hpp"
#include "primitive_desc_iface.hpp"
#include "primitive_hashing.hpp"
#include "type_helpers.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;

namespace dnnl {
namespace impl {

status_t grouped_gemm_primitive_desc_create(
        std::shared_ptr<primitive_desc_t> &pd, engine_t *engine,
        int group_count, const memory_desc_t *const *src_descs,
        const memory_desc_t *const *weights_descs,
        const memory_desc_t *const *bias_descs,
        const memory_desc_t *const *dst_descs, const primitive_attr_t *attr) {

    auto desc = grouped_gemm_desc_t(primitive_kind::grouped_gemm, group_count,
            src_descs, weights_descs, bias_descs, dst_descs);

    primitive_hashing::key_t key(
            engine, reinterpret_cast<op_desc_t *>(&desc), attr, 0, {}, -1);

    pd = primitive_cache().get_pd(key);
    if (pd) return success;

    grouped_gemm_pd_t *gemm_pd = nullptr;
    for (auto impl = engine->get_grouped_gemm_implementation_list(); *impl;
            ++impl) {
        if ((*impl)(&gemm_pd, engine, attr, group_count, src_descs,
                    weights_descs, bias_descs, dst_descs)
                == success) {
            pd.reset(gemm_pd);
            return success;
        }
    }

    return unimplemented;
}

} // namespace impl
} // namespace dnnl

status_t dnnl_grouped_gemm_primitive_desc_create(
        primitive_desc_iface_t **grouped_gemm_pd_iface, engine_t *engine,
        int group_count, const memory_desc_t *const *src_descs,
        const memory_desc_t *const *weights_descs,
        const memory_desc_t *const *bias_descs,
        const memory_desc_t *const *dst_descs, const primitive_attr_t *attr) {

    std::shared_ptr<primitive_desc_t> pd;
    auto status = dnnl::impl::grouped_gemm_primitive_desc_create(pd, engine,
            group_count, src_descs, weights_descs, bias_descs, dst_descs, attr);
    if (status == success) {
        CHECK(safe_ptr_assign(*grouped_gemm_pd_iface,
                new primitive_desc_iface_t(pd, engine)));
    }
    return status;
}