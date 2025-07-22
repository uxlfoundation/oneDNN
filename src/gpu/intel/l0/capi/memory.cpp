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

using namespace dnnl::impl;

dnnl_status_t DNNL_API dnnl_l0_interop_memory_create(dnnl_memory_t *memory,
        const_dnnl_memory_desc_t memory_desc, dnnl_engine_t engine,
        void *handle) {
    bool ok = !utils::any_null(memory, memory_desc, engine)
            && engine->runtime_kind() == runtime_kind::l0;
    if (!ok) return status::invalid_arguments;

    auto *l0_engine = utils::downcast<const gpu::intel::l0::engine_t *>(engine);
    auto kind = gpu::intel::l0::get_memory_storage_kind(
            gpu::intel::l0::get_pointer_type(l0_engine->context(), handle));
    if (handle != DNNL_MEMORY_NONE && handle != DNNL_MEMORY_ALLOCATE
            && kind == gpu::intel::l0::memory_storage_kind_t::unknown
            && !engine->mayiuse_system_memory_allocators())
        return status::invalid_arguments;

    const auto mdw = memory_desc_wrapper(memory_desc);
    if (mdw.format_any() || mdw.has_runtime_dims_or_strides())
        return status::invalid_arguments;

    unsigned flags = (handle == DNNL_MEMORY_ALLOCATE)
            ? memory_flags_t::alloc
            : memory_flags_t::use_runtime_ptr;
    handle = (handle == DNNL_MEMORY_ALLOCATE) ? nullptr : handle;

    std::unique_ptr<memory_storage_t> mem_storage;
    mem_storage.reset(new gpu::intel::l0::memory_storage_t(
            engine, gpu::intel::l0::memory_storage_kind_t::device));
    if (!mem_storage) return status::out_of_memory;

    CHECK(mem_storage->init(
            flags, dnnl_memory_desc_get_size(memory_desc), handle));

    return safe_ptr_assign(
            *memory, new memory_t(engine, memory_desc, std::move(mem_storage)));
}

dnnl_status_t DNNL_API dnnl_l0_interop_memory_create_v2(dnnl_memory_t *memory,
        const_dnnl_memory_desc_t memory_desc, dnnl_engine_t engine,
        size_t nhandles, void **handles) {
    bool ok = !utils::any_null(memory, memory_desc, engine, handles)
            && nhandles > 0 && engine->runtime_kind() == runtime_kind::l0;
    if (!ok) return status::invalid_arguments;

    const auto mdw = memory_desc_wrapper(memory_desc);
    if (mdw.format_any() || mdw.has_runtime_dims_or_strides())
        return status::invalid_arguments;

    std::vector<unsigned> flags_vec(nhandles);
    std::vector<void *> handles_vec(nhandles);
    for (size_t i = 0; i < nhandles; i++) {
        unsigned f = (handles[i] == DNNL_MEMORY_ALLOCATE)
                ? memory_flags_t::alloc
                : memory_flags_t::use_runtime_ptr;
        void *h = (handles[i] == DNNL_MEMORY_ALLOCATE) ? nullptr : handles[i];
        flags_vec[i] = f;
        handles_vec[i] = h;
    }

    auto *l0_engine = utils::downcast<const gpu::intel::l0::engine_t *>(engine);
    std::vector<std::unique_ptr<memory_storage_t>> mem_storages(nhandles);
    for (size_t i = 0; i < nhandles; i++) {
        auto kind = gpu::intel::l0::get_memory_storage_kind(
                gpu::intel::l0::get_pointer_type(
                        l0_engine->context(), handles[i]));
        if (handles[i] != DNNL_MEMORY_NONE && handles[i] != DNNL_MEMORY_ALLOCATE
                && kind == gpu::intel::l0::memory_storage_kind_t::unknown
                && !engine->mayiuse_system_memory_allocators()) {
            return status::invalid_arguments;
        }
        size_t sz = dnnl_memory_desc_get_size_v2(
                memory_desc, static_cast<int>(i));
        mem_storages[i].reset(new gpu::intel::l0::memory_storage_t(
                engine, gpu::intel::l0::memory_storage_kind_t::device));
        if (!mem_storages[i]) return status::out_of_memory;
        CHECK(mem_storages[i]->init(flags_vec[i], sz, handles_vec[i]));
    }

    return safe_ptr_assign(*memory,
            new memory_t(engine, memory_desc, std::move(mem_storages)));
}

dnnl_status_t DNNL_API dnnl_l0_interop_memory_get_mem_object(
        const memory_t *memory, void **mem_object) {
    if (utils::any_null(mem_object)) return status::invalid_arguments;

    if (!memory) {
        mem_object = nullptr;
        return status::success;
    }
    bool args_ok = (memory->engine()->runtime_kind() == runtime_kind::l0);
    if (!args_ok) return status::invalid_arguments;

    void *handle;
    status_t status = memory->get_data_handle(&handle);
    if (status == status::success) mem_object = &handle;

    return status;
}

dnnl_status_t DNNL_API dnnl_l0_interop_memory_set_mem_object(
        memory_t *memory, void *mem_object) {
    bool args_ok = (memory->engine()->runtime_kind() == runtime_kind::l0);
    if (!args_ok) return status::invalid_arguments;

    return memory->set_data_handle(mem_object);
}
