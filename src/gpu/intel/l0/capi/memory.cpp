/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#include "oneapi/dnnl/dnnl_l0.h"

#include "common/utils.hpp"
#include "gpu/intel/l0/memory_storage.hpp"

dnnl_status_t DNNL_API dnnl_l0_interop_memory_create(dnnl_memory_t *memory,
        const_dnnl_memory_desc_t memory_desc, dnnl_engine_t engine,
        void *handle) {
    bool ok = !dnnl::impl::utils::any_null(memory, memory_desc, engine)
            && engine->runtime_kind() == dnnl::impl::runtime_kind::l0;
    if (!ok) return dnnl::impl::status::invalid_arguments;

    if (handle != DNNL_MEMORY_NONE && handle != DNNL_MEMORY_ALLOCATE
            && !engine->mayiuse_system_memory_allocators())
        return dnnl::impl::status::invalid_arguments;

    const auto mdw = dnnl::impl::memory_desc_wrapper(memory_desc);
    if (mdw.format_any() || mdw.has_runtime_dims_or_strides())
        return dnnl::impl::status::invalid_arguments;

    unsigned flags = (handle == DNNL_MEMORY_ALLOCATE)
            ? dnnl::impl::memory_flags_t::alloc
            : dnnl::impl::memory_flags_t::use_runtime_ptr;
    handle = (handle == DNNL_MEMORY_ALLOCATE) ? nullptr : handle;

    std::unique_ptr<dnnl::impl::memory_storage_t> mem_storage;
    mem_storage.reset(new dnnl::impl::gpu::intel::l0::memory_storage_t(
            engine, dnnl::impl::gpu::intel::l0::memory_storage_kind_t::device));
    if (!mem_storage) return dnnl::impl::status::out_of_memory;

    CHECK(mem_storage->init(
            flags, dnnl_memory_desc_get_size(memory_desc), handle));

    return safe_ptr_assign(*memory,
            new dnnl::impl::memory_t(
                    engine, memory_desc, std::move(mem_storage)));
}

dnnl_status_t DNNL_API dnnl_l0_interop_memory_create_v2(dnnl_memory_t *memory,
        const_dnnl_memory_desc_t memory_desc, dnnl_engine_t engine,
        size_t nhandles, void **handles) {
    bool ok = !dnnl::impl::utils::any_null(memory, memory_desc, engine, handles)
            && nhandles > 0
            && engine->runtime_kind() == dnnl::impl::runtime_kind::l0;
    if (!ok) return dnnl::impl::status::invalid_arguments;

    const auto mdw = dnnl::impl::memory_desc_wrapper(memory_desc);
    if (mdw.format_any() || mdw.has_runtime_dims_or_strides())
        return dnnl::impl::status::invalid_arguments;

    std::vector<unsigned> flags_vec(nhandles);
    std::vector<void *> handles_vec(nhandles);
    for (size_t i = 0; i < nhandles; i++) {
        unsigned f = (handles[i] == DNNL_MEMORY_ALLOCATE)
                ? dnnl::impl::memory_flags_t::alloc
                : dnnl::impl::memory_flags_t::use_runtime_ptr;
        void *h = (handles[i] == DNNL_MEMORY_ALLOCATE) ? nullptr : handles[i];
        flags_vec[i] = f;
        handles_vec[i] = h;
    }

    std::vector<std::unique_ptr<dnnl::impl::memory_storage_t>> mem_storages(
            nhandles);
    for (size_t i = 0; i < nhandles; i++) {
        if (handles[i] != DNNL_MEMORY_NONE && handles[i] != DNNL_MEMORY_ALLOCATE
                && !engine->mayiuse_system_memory_allocators()) {
            return dnnl::impl::status::invalid_arguments;
        }
        size_t sz = dnnl_memory_desc_get_size_v2(memory_desc, i);
        mem_storages[i].reset(new dnnl::impl::gpu::intel::l0::memory_storage_t(
                engine,
                dnnl::impl::gpu::intel::l0::memory_storage_kind_t::device));
        if (!mem_storages[i]) return dnnl::impl::status::out_of_memory;
        CHECK(mem_storages[i]->init(flags_vec[i], sz, handles_vec[i]));
    }

    return safe_ptr_assign(*memory,
            new dnnl::impl::memory_t(
                    engine, memory_desc, std::move(mem_storages)));
}
