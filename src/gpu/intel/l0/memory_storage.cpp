/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "gpu/intel/l0/memory_storage.hpp"
#include "common/memory_map_manager.hpp"
#include "gpu/intel/l0/engine.hpp"
#include "gpu/intel/l0/stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace l0 {

status_t memory_storage_t::get_data_handle(void **handle) const {
    *handle = ptr_.get();

    return status::success;
}

status_t memory_storage_t::set_data_handle(void *handle) {
    ptr_ = decltype(ptr_)(handle, [](void *) {});
    kind_ = get_memory_storage_kind(
            get_pointer_type(l0_engine()->context(), handle));

    return status::success;
}

bool memory_storage_t::is_host_accessible() const {
    return utils::one_of(kind_, memory_storage_kind_t::host,
            memory_storage_kind_t::shared, memory_storage_kind_t::unknown);
}

struct map_usm_tag;

status_t memory_storage_t::map_data(
        void **mapped_ptr, impl::stream_t *stream, size_t size) const {
    if (is_host_accessible()) {
        *mapped_ptr = ptr();
        return status::success;
    }

    if (!ptr() || size == 0) {
        *mapped_ptr = nullptr;
        return status::success;
    }

    if (!stream) CHECK(engine()->get_service_stream(stream));

    void *host_ptr = malloc_host(size);
    if (!host_ptr) return status::out_of_memory;

    auto leak_guard = decltype(ptr_)(host_ptr, [this](void *p) { free(p); });
    CHECK(memcpy(stream, host_ptr, ptr(), size));
    CHECK(stream->wait());
    leak_guard.release();

    auto *usm_ptr_for_unmap = ptr();
    auto unmap_callback = [size, usm_ptr_for_unmap, this](
                                  impl::stream_t *stream, void *mapped_ptr) {
        CHECK(memcpy(stream, usm_ptr_for_unmap, mapped_ptr, size));
        CHECK(stream->wait());
        free(mapped_ptr);

        return status::success;
    };

    auto &map_manager = memory_map_manager_t<map_usm_tag>::instance();

    *mapped_ptr = host_ptr;

    return map_manager.map(this, stream, *mapped_ptr, unmap_callback);
}

status_t memory_storage_t::unmap_data(
        void *mapped_ptr, impl::stream_t *stream) const {
    if (!mapped_ptr || is_host_accessible()) return status::success;

    if (!stream) CHECK(engine()->get_service_stream(stream));

    auto &map_manager = memory_map_manager_t<map_usm_tag>::instance();

    return map_manager.unmap(this, stream, mapped_ptr);
}

std::unique_ptr<impl::memory_storage_t> memory_storage_t::get_sub_storage(
        size_t offset, size_t size) const {
    void *sub_ptr
            = ptr_ ? reinterpret_cast<uint8_t *>(ptr_.get()) + offset : nullptr;

    auto storage = utils::make_unique<memory_storage_t>(engine(), kind_);
    if (!storage) return nullptr;

    auto status = storage->init(memory_flags_t::use_runtime_ptr, size, sub_ptr);
    if (status != status::success) return nullptr;

    // XXX: Clang has a bug that prevents implicit conversion.
    return std::unique_ptr<memory_storage_t>(storage.release());
}

std::unique_ptr<impl::memory_storage_t> memory_storage_t::clone() const {
    auto storage = utils::make_unique<memory_storage_t>(engine(), kind_);
    if (!storage) return nullptr;

    auto status = storage->init(memory_flags_t::use_runtime_ptr, 0, nullptr);
    if (status != status::success) return nullptr;

    storage->ptr_ = decltype(ptr_)(ptr_.get(), [](void *) {});
    storage->kind_ = kind_;

    // XXX: Clang has a bug that prevents implicit conversion.
    return std::unique_ptr<memory_storage_t>(storage.release());
}

status_t memory_storage_t::init_allocate(size_t size) {
    if (kind_ == memory_storage_kind_t::unknown)
        kind_ = memory_storage_kind_t::device;

    void *ptr_alloc = nullptr;

    switch (kind_) {
        case memory_storage_kind_t::host: ptr_alloc = malloc_host(size); break;
        case memory_storage_kind_t::device:
            ptr_alloc = malloc_device(size);
            break;
        case memory_storage_kind_t::shared:
            ptr_alloc = malloc_shared(size);
            break;
        default: break;
    }
    if (!ptr_alloc) return status::out_of_memory;

    ptr_ = decltype(ptr_)(ptr_alloc, [&](void *ptr) { free(ptr); });

    return status::success;
}

void *memory_storage_t::malloc_host(size_t size) const {
    void *pptr = nullptr;

    ze_host_mem_alloc_desc_t host_mem_alloc_desc = {};
    host_mem_alloc_desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    host_mem_alloc_desc.pNext = nullptr;
    host_mem_alloc_desc.flags = ZE_MEMORY_ACCESS_CAP_FLAG_RW;

    l0::zeMemAllocHost(
            l0_engine()->context(), &host_mem_alloc_desc, size, 0, &pptr);

    return pptr;
}

void *memory_storage_t::malloc_device(size_t size) const {
    void *pptr = nullptr;

    ze_device_mem_alloc_desc_t device_mem_alloc_desc = {};
    device_mem_alloc_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    device_mem_alloc_desc.pNext = nullptr;
    device_mem_alloc_desc.flags = ZE_MEMORY_ACCESS_CAP_FLAG_RW;
    device_mem_alloc_desc.ordinal = 0;

    l0::zeMemAllocDevice(l0_engine()->context(), &device_mem_alloc_desc, size,
            0, l0_engine()->device(), &pptr);

    return pptr;
}

void *memory_storage_t::malloc_shared(size_t size) const {
    void *pptr = nullptr;

    ze_device_mem_alloc_desc_t device_mem_alloc_desc = {};
    device_mem_alloc_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    device_mem_alloc_desc.pNext = nullptr;
    device_mem_alloc_desc.flags = ZE_MEMORY_ACCESS_CAP_FLAG_RW;
    device_mem_alloc_desc.ordinal = 0;

    ze_host_mem_alloc_desc_t host_mem_alloc_desc = {};
    host_mem_alloc_desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    host_mem_alloc_desc.pNext = nullptr;
    host_mem_alloc_desc.flags = ZE_MEMORY_ACCESS_CAP_FLAG_RW;

    l0::zeMemAllocShared(l0_engine()->context(), &device_mem_alloc_desc,
            &host_mem_alloc_desc, size, 0, l0_engine()->device(), &pptr);

    return pptr;
}

void memory_storage_t::free(void *ptr) const {
    l0::zeMemFree(l0_engine()->context(), ptr);
}

status_t memory_storage_t::memcpy(
        impl::stream_t *stream, void *dst, const void *src, size_t size) const {
    auto *l0_stream = utils::downcast<stream_t *>(stream);
    return l0::zeCommandListAppendMemoryCopy(
            l0_stream->list(), dst, src, size, nullptr, 0, nullptr);
}

} // namespace l0
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
