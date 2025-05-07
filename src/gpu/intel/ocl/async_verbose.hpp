/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#ifndef GPU_OCL_ASYNC_VERBOSE_HPP
#define GPU_OCL_ASYNC_VERBOSE_HPP

#include "common/c_types_map.hpp"

#include "gpu/generic/async_verbose.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct async_verbose_tracker_t : public gpu::generic::async_verbose_tracker_t {
    async_verbose_tracker_t(const impl::stream_t *stream)
        : gpu::generic::async_verbose(stream) {}

    status_t refresh_tracking_info() const override;
    status_t refresh_tracking_stats() const override;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
