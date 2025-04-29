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

#ifndef GRAPH_BACKEND_DNNL_DNNL_ALLOCATOR_HPP
#define GRAPH_BACKEND_DNNL_DNNL_ALLOCATOR_HPP

#include "common/engine.hpp"

#include "graph/interface/allocator.hpp"

#ifdef DNNL_WITH_SYCL
#include "xpu/sycl/engine_impl.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "xpu/ocl/engine_impl.hpp"
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

struct dnnl_allocator_t {
    dnnl_allocator_t(const engine_t &eng)
        : ekind_(eng.kind())
        , alloc_(reinterpret_cast<allocator_t *>(eng.get_allocator())) {
#ifdef DNNL_WITH_SYCL
        auto *sycl_engine_impl
                = dnnl::impl::utils::downcast<const xpu::sycl::engine_impl_t *>(
                        eng.impl());
        sycl_dev_ = &(const_cast<::sycl::device &>(sycl_engine_impl->device()));
        sycl_ctx_
                = &(const_cast<::sycl::context &>(sycl_engine_impl->context()));
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        const auto *ocl_engine_impl
                = dnnl::impl::utils::downcast<const xpu::ocl::engine_impl_t *>(
                        eng.impl());
        ocl_dev_ = ocl_engine_impl->device();
        ocl_ctx_ = ocl_engine_impl->context();
#endif
    }

    void *malloc(size_t size) const;
    void free(void *p) const;

#ifdef DNNL_WITH_SYCL
    void free(void *p, const ::sycl::event &deps) const;
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    void free(void *p, const cl_event &deps) const;
#endif

private:
    engine_kind_t ekind_ = engine_kind::cpu;
    allocator_t *alloc_ = nullptr;

#ifdef DNNL_WITH_SYCL
    ::sycl::device *sycl_dev_ = nullptr;
    ::sycl::context *sycl_ctx_ = nullptr;
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_device_id ocl_dev_;
    cl_context ocl_ctx_;
#endif
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
