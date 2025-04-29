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

#include "graph/backend/dnnl/dnnl_allocator.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

#define DNNL_CPU_MEM_ALIGN 64
#define DNNL_SYCL_MEM_ALIGN 64
#define DNNL_OCL_MEM_ALIGN 0

void *dnnl_allocator_t::malloc(size_t size) const {
    void *ret = nullptr;
    const allocator_t::mem_type_t type = allocator_t::mem_type_t::persistent;

    if (ekind_ == engine_kind::cpu) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        ret = alloc_->allocate(
                size, *sycl_dev_, *sycl_ctx_, {type, DNNL_SYCL_MEM_ALIGN});
#else
        ret = alloc_->allocate(size, {type, DNNL_CPU_MEM_ALIGN});
#endif
    } else if (ekind_ == engine_kind::gpu) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        ret = alloc_->allocate(
                size, *sycl_dev_, *sycl_ctx_, {type, DNNL_SYCL_MEM_ALIGN});
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        ret = alloc_->allocate(
                size, ocl_dev_, ocl_ctx_, {type, DNNL_OCL_MEM_ALIGN});
#else
        assert(!"unsupported gpu runtime");
#endif
    } else {
        assert(!"unsupported engine kind");
    }

    return ret;
}

void dnnl_allocator_t::free(void *p) const {
    if (ekind_ == engine_kind::cpu) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        assert(!"use event based free");
#else
        return alloc_->deallocate(p);
#endif
    } else if (ekind_ == engine_kind::gpu) {
        assert(!"use event based free");
    }
}

#ifdef DNNL_WITH_SYCL
void dnnl_allocator_t::free(void *p, const ::sycl::event &deps) const {
    if (ekind_ == engine_kind::cpu) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        alloc_->deallocate(p, *sycl_dev_, *sycl_ctx_, deps);
#else
        alloc_->deallocate(p);
#endif
    } else if (ekind_ == engine_kind::gpu) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        alloc_->deallocate(p, *sycl_dev_, *sycl_ctx_, deps);
#endif
    }
}
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
void dnnl_allocator_t::free(void *p, const cl_event &deps) const {
    if (ekind_ != engine_kind::gpu) {
        assert(!"the engine kind should be gpu");
        return;
    }
    alloc_->deallocate(p, ocl_dev_, ocl_ctx_, deps);
}
#endif

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
