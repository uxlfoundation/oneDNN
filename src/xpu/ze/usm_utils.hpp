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

#ifndef XPU_ZE_USM_UTILS_HPP
#define XPU_ZE_USM_UTILS_HPP

#include "oneapi/dnnl/dnnl.h"

namespace dnnl {
namespace impl {
namespace xpu {
namespace ze {

void DNNL_API *malloc_device(dnnl_engine_t engine, size_t size);
void DNNL_API *malloc_shared(dnnl_engine_t engine, size_t size);
void DNNL_API free(dnnl_engine_t engine, void *ptr);
dnnl_status_t DNNL_API memset(
        dnnl_stream_t stream, void *ptr, int value, size_t size);

} // namespace ze
} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif
