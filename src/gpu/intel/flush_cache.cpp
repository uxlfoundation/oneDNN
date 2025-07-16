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

#include "common/c_types_map.hpp"
#include "gpu/intel/compute/compute_engine.hpp"
#include "gpu/intel/compute/compute_stream.hpp"
#include "gpu/intel/compute/kernel.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

status_t flush(impl::stream_t *stream, size_t bytes, impl::memory_t *data) {
    static compute::kernel_t kernel;
    static std::once_flag flag;

    auto &compute_engine
            = *utils::downcast<compute::compute_engine_t *>(stream->engine());
    auto &compute_stream
            = *utils::downcast<compute::compute_stream_t *>(stream);
    std::call_once(flag, [&]() {
        compute::kernel_ctx_t ctx;
        std::vector<compute::kernel_t> kernels;
        UNUSED_STATUS(
                compute_engine.create_kernels(&kernels, {"flush_cache"}, ctx));
        kernel = kernels[0];
    });
    compute::range_t gws = {bytes / 64, 1, 1};
    compute::range_t lws = {256, 1, 1};
    compute::nd_range_t nd_range(gws, lws);
    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, *data->memory_storage());
    compute_stream.set_profiling(false);
    CHECK(kernel.parallel_for(compute_stream, nd_range, arg_list,
            compute_stream.ctx().get_deps(), compute_stream.ctx().get_deps()));
    compute_stream.set_profiling(true);
    return status::success;
}

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

using namespace dnnl::impl;

extern "C" status_t DNNL_API dnnl_impl_gpu_flush_cache(
        stream_t *stream, size_t bytes, memory_t *data) {
    return dnnl::impl::gpu::intel::flush(stream, bytes, data);
}
