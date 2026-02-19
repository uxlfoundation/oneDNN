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

status_t fill_random(impl::stream_t *stream, size_t size,
        impl::memory_t *memory, int buffer_index, uint32_t seed) {
    static compute::kernel_t kernel;
    static std::once_flag flag;

    auto *intel_stream = utils::downcast<intel::stream_t *>(stream);
    auto *intel_engine = utils::downcast<intel::engine_t *>(stream->engine());

    const uint32_t block_size = 128u;
    const uint32_t simd_width = 16u;
    const uint32_t simd_iters = block_size / simd_width;

    std::call_once(flag, [&]() {
        compute::kernel_ctx_t ctx;
        ctx.define_int("BLOCK_SIZE", block_size);
        ctx.define_int("SIMD_WIDTH", simd_width);
        ctx.define_int("ITERS", simd_iters);
        ctx.define_int("SEED", seed);
        ctx.add_option("-DDT=uint" + std::to_string(simd_width));
        ctx.add_option("-DSIMD_STORE=vstore" + std::to_string(simd_width));
        std::vector<compute::kernel_t> kernels;
        UNUSED_STATUS(
                intel_engine->create_kernels(&kernels, {"fill_random"}, ctx));
        kernel = kernels[0];
    });

    const uint32_t count = static_cast<uint32_t>(size / sizeof(uint32_t));
    if (count == 0) return status::success;

    const uint32_t num_work_items = count / block_size;
    if (num_work_items == 0) return status::success;

    compute::range_t gws = {num_work_items, 1, 1};
    compute::nd_range_t nd_range(gws);
    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, *memory->memory_storage(buffer_index));

    CHECK(kernel.parallel_for(*stream, nd_range, arg_list,
            intel_stream->ctx().get_deps(), intel_stream->ctx().get_deps()));

    return status::success;
}

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

extern "C" dnnl::impl::status_t DNNL_API dnnl_impl_gpu_fill_random(
        dnnl::impl::stream_t *stream, size_t size, dnnl::impl::memory_t *memory,
        int buffer_index, uint32_t seed) {
    return dnnl::impl::gpu::intel::fill_random(
            stream, size, memory, buffer_index, seed);
}
