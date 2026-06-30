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

#include "gpu/intel/primitive_attr.hpp"

#include "common/c_types_map.hpp"
#include "common/utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::gpu::intel;

extern "C" dnnl_status_t DNNL_API dnnl_impl_gpu_intel_set_kernel_override(
        primitive_attr_t *attr, const char *kernel) {
    if (utils::any_null(attr)) return status::invalid_arguments;

    // Preserve any GRF setting already carried by the gpu attribute.
    int grf_per_thread = 0;
    if (attr->gpu_attr_) {
        auto *cur = utils::downcast<gpu_primitive_attr_t *>(
                attr->gpu_attr_.get());
        grf_per_thread = cur->grf_per_thread();
    }
    return attr->set_gpu_attr(
            gpu_primitive_attr_t(grf_per_thread, kernel ? kernel : ""));
}

extern "C" dnnl_status_t DNNL_API dnnl_impl_gpu_intel_get_kernel_override(
        const primitive_attr_t *attr, const char **kernel) {
    if (utils::any_null(attr, kernel)) return status::invalid_arguments;

    *kernel = "";
    if (attr->gpu_attr_) {
        auto *cur = utils::downcast<const gpu_primitive_attr_t *>(
                attr->gpu_attr_.get());
        *kernel = cur->kernel_override().c_str();
    }
    return status::success;
}
