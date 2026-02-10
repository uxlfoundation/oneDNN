/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <gtest/gtest.h>

#include <mutex>
#include <thread>

#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"
#include "interface/allocator.hpp"
#include "interface/c_types_map.hpp"

TEST(test_interface_allocator, DefaultCpuAllocator) {
    dnnl::impl::graph::allocator_t *alloc
            = new dnnl::impl::graph::allocator_t();

    void *mem_ptr = alloc->allocate(static_cast<size_t>(16));
    if (mem_ptr == nullptr) {
        delete alloc;
        ASSERT_TRUE(false);
    } else {
        alloc->deallocate(mem_ptr);
        delete alloc;
    }
}

TEST(test_interface_allocator, AllocatorEarlyDestroy) {
    using namespace dnnl::impl::graph;

    allocator_t *alloc = new allocator_t();
    engine_t *eng = get_engine();
    eng->set_allocator(alloc);
    delete alloc; // destroy after setting to engine.

    allocator_t *engine_alloc
            = reinterpret_cast<allocator_t *>(eng->get_allocator());
    void *mem_ptr = nullptr;
    const size_t alloc_size = 16;
    if (get_test_engine_kind() == engine_kind::cpu) {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
        mem_ptr = engine_alloc->allocate(alloc_size);
#else
        mem_ptr = engine_alloc->allocate(
                alloc_size, get_device(), get_context());
#endif
    } else {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        mem_ptr = engine_alloc->allocate(
                alloc_size, get_device(), get_context());
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        // In gtests, GPU OCL runtime uses library managed device and context.
        mem_ptr = engine_alloc->allocate(alloc_size);
#else
        ASSERT_TRUE(false);
#endif
    }

    if (mem_ptr == nullptr) {
        ASSERT_TRUE(false);
    } else {
        if (get_test_engine_kind() == engine_kind::cpu) {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
            engine_alloc->deallocate(mem_ptr);
#else
            sycl::event e;
            engine_alloc->deallocate(mem_ptr, get_device(), get_context(), e);
#endif
        } else {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
            sycl::event e;
            engine_alloc->deallocate(mem_ptr, get_device(), get_context(), e);
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
            // In gtests, GPU OCL runtime uses library managed device and context.
            engine_alloc->deallocate(mem_ptr);
#else
            ASSERT_TRUE(false);
#endif
        }
    }
}
