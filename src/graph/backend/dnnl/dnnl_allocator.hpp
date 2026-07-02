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

#ifndef GRAPH_BACKEND_DNNL_DNNL_ALLOCATOR_HPP
#define GRAPH_BACKEND_DNNL_DNNL_ALLOCATOR_HPP

#include "graph/interface/allocator.hpp"

#ifdef DNNL_WITH_SYCL
#include <sycl/sycl.hpp>
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include <CL/cl.h>
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

// dnnl_allocator_t is a utility class that provides static methods for memory
// allocation and deallocation using the graph's allocator.
struct dnnl_allocator_t {
    static void *malloc(
            size_t size, const engine_t &eng, allocator_t::mem_type_t type);

    static void free(void *p, const engine_t &eng);

#ifdef DNNL_WITH_SYCL
    static void free(void *p, const engine_t &eng, const ::sycl::event &deps);
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    static void free(void *p, const engine_t &eng, const cl_event &deps);
#endif
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
