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

#include "memory_tracking.hpp"

#include "engine.hpp"

namespace dnnl {
namespace impl {
namespace memory_tracking {

const void *registry_t::entry_t::compute_ptr(const void *base_ptr) const {
    if (size == 0) return nullptr;
    assert(base_ptr != nullptr);

    char *ptr = (char *)base_ptr + offset;
    char *aligned_ptr = utils::align_ptr<char>(ptr, get_alignment(alignment));

    if (memory_debug::is_mem_debug_overflow() && size % getpagesize() != 0) {
        // Align to end of page
        size_t page_end_offset = utils::rnd_up(size, alignment) % getpagesize();
        aligned_ptr += getpagesize() - page_end_offset;
        if (aligned_ptr - getpagesize() > ptr) aligned_ptr -= getpagesize();
        assert((size_t)aligned_ptr % alignment == 0);
    }

    assert(aligned_ptr + size <= ptr + capacity - buffer_protect_size());
    return (const void *)aligned_ptr;
}

// This call returns a pointer to a grantor allocated on the heap. It's user's
// responsibility to free it.
//
// Note: if grabbed by `exec_ctx_t::set_scratchpad_grantor(...)`, a `exec_ctx_t`
// object context will handle its proper destruction.
grantor_t *registry_t::create_grantor(const memory_storage_t *mem_storage,
        const void *base_mem_storage_host_ptr) const {
    // Empty memory storage implies its mapped ptr is empty as well.
    assert(IMPLICATION(!mem_storage, !base_mem_storage_host_ptr));
    return new grantor_t(*this, mem_storage, base_mem_storage_host_ptr);
}

grantor_t::grantor_t(const registry_t &registry,
        const memory_storage_t *base_mem_storage,
        const void *base_mem_storage_host_ptr)
    : registry_(registry)
    , prefix_(0)
    , base_mem_storage_(base_mem_storage)
    , base_mem_storage_host_ptr_(base_mem_storage_host_ptr) {}

grantor_t::grantor_t(const grantor_t &parent, const key_t &prefix)
    : registry_(parent.registry_)
    , prefix_(make_prefix(parent.prefix_, prefix))
    , base_mem_storage_(parent.base_mem_storage_)
    , base_mem_storage_host_ptr_(parent.base_mem_storage_host_ptr_) {}

char *grantor_t::host_ptr(const memory_storage_t *mem_storage) const {
    if (!mem_storage || mem_storage->is_null()) return nullptr;

    void *handle = mem_storage->root_storage()->data_handle();
    char *base_ptr = nullptr;
    if (base_mem_storage_host_ptr_) {
        base_ptr = reinterpret_cast<char *>(
                           const_cast<void *>(base_mem_storage_host_ptr_))
                + mem_storage->base_offset();
    } else {
        assert(mem_storage->is_host_accessible());
        base_ptr = static_cast<char *>(handle);
    }
    return base_ptr;
}

bool grantor_t::is_cpu_engine(const memory_storage_t *mem_storage) const {
    if (!mem_storage) return false;
    auto engine = mem_storage->engine();
    assert(engine);
    if (engine->kind() == engine_kind::cpu) return true;
    return false;
}

} // namespace memory_tracking
} // namespace impl
} // namespace dnnl
