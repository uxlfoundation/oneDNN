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

#include "common/c_types_map.hpp"

#include "gpu/intel/compute/kernel.hpp"
#include "gpu/intel/compute/kernel_ctx.hpp"
#include "gpu/intel/engine.hpp"
#include "gpu/intel/stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

status_t flush(impl::stream_t *stream, size_t bytes, impl::memory_t *data) {
    static compute::kernel_t kernel;
    static std::once_flag flag;

    auto *intel_stream = utils::downcast<intel::stream_t *>(stream);
    auto *intel_engine = utils::downcast<intel::engine_t *>(stream->engine());
    std::call_once(flag, [&]() {
        compute::kernel_ctx_t ctx;
        std::vector<compute::kernel_t> kernels;
        UNUSED_STATUS(
                intel_engine->create_kernels(&kernels, {"flush_cache"}, ctx));
        kernel = kernels[0];
    });
    compute::range_t gws = {bytes / 64, 1, 1};
    compute::range_t lws = {256, 1, 1};
    compute::nd_range_t nd_range(gws, lws);
    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, *data->memory_storage());

    // Disable profiling when flushing the cache.
    intel_stream->set_profiling(false);
    CHECK(kernel.parallel_for(*stream, nd_range, arg_list,
            intel_stream->ctx().get_deps(), intel_stream->ctx().get_deps()));
    intel_stream->set_profiling(true);

    return status::success;
}

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

extern "C" dnnl::impl::status_t DNNL_API dnnl_impl_gpu_flush_cache(
        dnnl::impl::stream_t *stream, size_t bytes,
        dnnl::impl::memory_t *data) {
    return dnnl::impl::gpu::intel::flush(stream, bytes, data);
}

extern "C" size_t DNNL_API dnnl_impl_gpu_l3_cache_size(
        dnnl::impl::engine_t *engine) {
    auto *intel_engine
            = dnnl::impl::utils::downcast<dnnl::impl::gpu::intel::engine_t *>(
                    engine);
    return intel_engine->device_info()->l3_cache_size();
}
