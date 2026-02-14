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

#include "oneapi/dnnl/dnnl_ze.h"

#include "common/engine.hpp"
#include "common/utils.hpp"

#include "xpu/ze/engine_impl.hpp"
#include "xpu/ze/memory_storage.hpp"

using namespace dnnl::impl;

status_t dnnl_ze_interop_memory_create(memory_t **memory,
        const memory_desc_t *md, engine_t *engine, int nhandles,
        void **handles) {
    bool ok = !utils::any_null(memory, md, engine, handles) && nhandles > 0
            && engine->runtime_kind() == runtime_kind::ze;
    if (!ok) return status::invalid_arguments;

    const auto mdw = memory_desc_wrapper(md);
    if (mdw.format_any() || mdw.has_runtime_dims_or_strides()
            || mdw.is_host_scalar_desc())
        return status::invalid_arguments;

    std::vector<unsigned> flags_vec(nhandles);
    std::vector<void *> handles_vec(nhandles);
    for (int i = 0; i < nhandles; i++) {
        unsigned f = (handles[i] == DNNL_MEMORY_ALLOCATE)
                ? memory_flags_t::alloc
                : memory_flags_t::use_runtime_ptr;
        void *h = (handles[i] == DNNL_MEMORY_ALLOCATE) ? nullptr : handles[i];
        flags_vec[i] = f;
        handles_vec[i] = h;
    }

    auto *ze_engine_impl
            = utils::downcast<const xpu::ze::engine_impl_t *>(engine->impl());
    std::vector<std::unique_ptr<memory_storage_t>> mem_storages(nhandles);
    for (int i = 0; i < nhandles; i++) {
        if (handles[i] != DNNL_MEMORY_NONE && handles[i] != DNNL_MEMORY_ALLOCATE
                && xpu::ze::get_memory_storage_kind(xpu::ze::get_pointer_type(
                           ze_engine_impl->context(), handles[i]))
                        == xpu::ze::memory_storage_kind_t::unknown
                && !engine->mayiuse_system_memory_allocators()) {
            return status::invalid_arguments;
        }
        size_t sz = dnnl_memory_desc_get_size_v2(md, i);
        mem_storages[i].reset(new xpu::ze::memory_storage_t(engine));
        if (!mem_storages[i]) return status::out_of_memory;
        CHECK(mem_storages[i]->init(flags_vec[i], sz, handles_vec[i]));
    }

    return safe_ptr_assign(
            *memory, new memory_t(engine, md, std::move(mem_storages)));
}
