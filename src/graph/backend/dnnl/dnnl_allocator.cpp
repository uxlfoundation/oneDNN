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

#include "graph/backend/dnnl/dnnl_allocator.hpp"

#include "common/engine.hpp"
#include "common/utils.hpp"

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
const size_t DNNL_CPU_MEMALIGNMENT = 64;
#endif

#ifdef DNNL_WITH_SYCL
#include "xpu/sycl/engine_impl.hpp"
const size_t DNNL_SYCL_MEMALIGNMENT = 64;
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "xpu/ocl/engine_impl.hpp"
const size_t DNNL_OCL_MEMALIGNMENT = 0;
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

static graph::allocator_t *get_allocator(const engine_t &eng) {
    return reinterpret_cast<graph::allocator_t *>(eng.get_allocator());
}

void *dnnl_allocator_t::malloc(
        size_t size, const engine_t &eng, allocator_t::mem_type_t type) {
    const auto *alc = get_allocator(eng);
    if (eng.kind() == engine_kind::cpu) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        const auto *sycl_eng
                = utils::downcast<const xpu::sycl::engine_impl_t *>(eng.impl());
        return alc->allocate(size, sycl_eng->device(), sycl_eng->context(),
                {type, DNNL_SYCL_MEMALIGNMENT});
#else
        return alc->allocate(size, {type, DNNL_CPU_MEMALIGNMENT});
#endif
    } else if (eng.kind() == engine_kind::gpu) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        const auto *sycl_eng
                = utils::downcast<const xpu::sycl::engine_impl_t *>(eng.impl());
        return alc->allocate(size, sycl_eng->device(), sycl_eng->context(),
                {type, DNNL_SYCL_MEMALIGNMENT});
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        const auto *ocl_eng
                = utils::downcast<const xpu::ocl::engine_impl_t *>(eng.impl());
        return alc->allocate(size, ocl_eng->device(), ocl_eng->context(),
                {type, DNNL_OCL_MEMALIGNMENT});
#else
        return nullptr;
#endif
    } else {
        return nullptr;
    }
}

void dnnl_allocator_t::free(void *p, const engine_t &eng) {
    if (eng.kind() == engine_kind::cpu) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        assert(!"use event based free");
#else
        const auto *alc = get_allocator(eng);
        return alc->deallocate(p);
#endif
    } else if (eng.kind() == engine_kind::gpu) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL \
        || DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        assert(!"use event based free");
#endif
    }
}

#ifdef DNNL_WITH_SYCL
void dnnl_allocator_t::free(
        void *p, const engine_t &eng, const ::sycl::event &deps) {
    const auto *alc = get_allocator(eng);
    const auto *sycl_eng
            = utils::downcast<const xpu::sycl::engine_impl_t *>(eng.impl());
    if (eng.kind() == engine_kind::cpu) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        alc->deallocate(p, sycl_eng->device(), sycl_eng->context(), deps);
#else
        alc->deallocate(p);
#endif
    } else if (eng.kind() == engine_kind::gpu) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        alc->deallocate(p, sycl_eng->device(), sycl_eng->context(), deps);
#endif
    }
}
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
void dnnl_allocator_t::free(
        void *p, const engine_t &eng, const cl_event &deps) {
    if (eng.kind() != engine_kind::gpu) {
        assert(!"the engine kind should be gpu");
        return;
    }
    const auto *alc = get_allocator(eng);
    const auto *ocl_eng
            = utils::downcast<const xpu::ocl::engine_impl_t *>(eng.impl());
    alc->deallocate(p, ocl_eng->device(), ocl_eng->context(), deps);
}
#endif

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
