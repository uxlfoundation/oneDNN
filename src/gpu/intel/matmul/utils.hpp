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

#ifndef GPU_INTEL_MATMUL_UTILS_HPP
#define GPU_INTEL_MATMUL_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/matmul_pd.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {

static inline status_t create_matmul_pd(
        std::shared_ptr<primitive_desc_t> &matmul_pd_, engine_t *engine,
        const memory_desc_t *src_md, const memory_desc_t *wei_md,
        const memory_desc_t *bias_md, const memory_desc_t *dst_md,
        const primitive_attr_t *attr, bool skip_ref = false,
        const memory_desc_t *reduce_md = nullptr,
        matmul_reduce_kind_t reduce_kind = matmul_reduce_kind::undef,
        data_type_t acc_dt = data_type::undef) {
    matmul_desc_t matmul_desc;
    CHECK(matmul_desc_init(&matmul_desc, src_md, wei_md, bias_md, dst_md,
            reduce_md, reduce_kind));
    if (acc_dt != data_type::undef) matmul_desc.accum_data_type = acc_dt;

    primitive_attr_t matmul_attr = *attr;

    primitive_desc_iterator_t it(
            engine, (op_desc_t *)&matmul_desc, &matmul_attr, nullptr);
    if (!it.is_initialized()) return status::out_of_memory;

    for (++it; it != it.end(); ++it) {
        matmul_pd_ = *it;
        if (!matmul_pd_) return status::unimplemented;
        if (skip_ref && strstr(matmul_pd_->name(), "ref") != nullptr) continue;
        return status::success;
    }
    return status::unimplemented;
}

} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_MATMUL_UTILS_HPP
