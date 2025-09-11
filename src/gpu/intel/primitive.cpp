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

#include "gpu/intel/primitive.hpp"
#include "gpu/intel/jit/generator_base.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

status_t primitive_t::create_kernel(impl::engine_t *engine,
        compute::kernel_t *kernel, jit::generator_base_t *jitter,
        bool register_kernel) {
    auto *intel_engine = utils::downcast<intel::engine_t *>(engine);
    if (cache_blob()) {
        VCHECK_KERNEL(
                intel_engine->create_kernel_from_cache_blob(cache_blob(),
                        *kernel, jitter ? jitter->kernel_name() : nullptr),
                VERBOSE_KERNEL_CREATION_FAIL,
                jitter ? jitter->kernel_name() : "cached");
        kernel->hash_dump("blob");
        CHECK(register_kernels({*kernel}));
        return status::success;
    }
    VCHECK_KERNEL(intel_engine->create_kernel(kernel, jitter),
            VERBOSE_KERNEL_CREATION_FAIL, jitter ? jitter->kernel_name() : "");
    kernel->hash_dump("real");
    if (register_kernel) CHECK(register_kernels({*kernel}));
    return status::success;
}

status_t primitive_t::create_kernels(impl::engine_t *engine,
        std::vector<compute::kernel_t> *kernels,
        const std::vector<const char *> &kernel_names,
        const compute::kernel_ctx_t &kernel_ctx) {
    auto *intel_engine = utils::downcast<intel::engine_t *>(engine);
    if (cache_blob()) {
        CHECK(intel_engine->create_kernels_from_cache_blob(
                cache_blob(), *kernels, kernel_names));
        for (auto &k : *kernels)
            k.hash_dump("blob");
        CHECK(register_kernels(*kernels));
        return status::success;
    }
    CHECK(intel_engine->create_kernels(kernels, kernel_names, kernel_ctx));
    for (auto &k : *kernels)
        k.hash_dump("real");
    CHECK(register_kernels(*kernels));
    return status::success;
}

status_t primitive_t::create_kernel(impl::engine_t *engine,
        compute::kernel_t *kernel, const char *kernel_name,
        const compute::kernel_ctx_t &kernel_ctx) {
    std::vector<compute::kernel_t> kernels(1);
    VCHECK_KERNEL(create_kernels(engine, &kernels, {kernel_name}, kernel_ctx),
            VERBOSE_KERNEL_CREATION_FAIL, kernel_name);
    *kernel = kernels[0];
    return status::success;
}

// Intel GPU hardware has a limitation on the size of work group dimensions to
// be at most uint32_t. This function works around that by passing an offset
// argument. The OpenCL native offset cannot be used due to lack of SYCL
// interop support.
status_t primitive_t::large_parallel_for(const exec_ctx_t &ctx,
        const compute::nd_range_t &nd_range, const compute::kernel_t &kernel,
        compute::kernel_arg_list_t &arg_list, int offset_idx) {

    auto global_range = nd_range.global_range();
    auto local_range = nd_range.local_range();

    // Convert global_range to an equivalent 3D nd_range_t
    constexpr size_t range_ndims = 3;
    assert(global_range.ndims() <= range_ndims);
    auto gws = compute::range_t::one(range_ndims);
    for (size_t i = 0; i < global_range.ndims(); i++) {
        gws[i] = global_range[i];
    }

    compute::range_t off_inc(UINT32_MAX, UINT32_MAX, UINT32_MAX);
    if (local_range) {
        for (size_t i = 0; i < local_range.ndims(); i++) {
            off_inc[i] *= local_range[i];
        }
    }

    compute::int64x3_t offset_arg = {};
    auto &offset = offset_arg.array;
    static_assert(
            range_ndims == 3, "Large parallel for loop doesn't match ndims.");
    for_(offset[2] = 0; static_cast<size_t>(offset[2]) < gws[2];
            offset[2] += off_inc[2])
    for_(offset[1] = 0; static_cast<size_t>(offset[1]) < gws[1];
            offset[1] += off_inc[1])
    for_(offset[0] = 0; static_cast<size_t>(offset[0]) < gws[0];
            offset[0] += off_inc[0])
    {
        arg_list.set(offset_idx, offset_arg);
        auto range = compute::range_t::empty(range_ndims);
        for (size_t i = 0; i < range_ndims; i++)
            range[i] = std::min(off_inc[i], gws[i] - offset[i]);

        CHECK(parallel_for(ctx, compute::nd_range_t(range, local_range), kernel,
                arg_list));
    }
    return status::success;
}

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
