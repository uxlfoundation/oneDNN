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

#ifndef XPU_ZE_MEMORY_STORAGE_HPP
#define XPU_ZE_MEMORY_STORAGE_HPP

#include "common/memory_storage.hpp"
#include "common/utils.hpp"

#include "xpu/ze/engine_impl.hpp"
#include "xpu/ze/utils.hpp"

#include <functional>

namespace dnnl {
namespace impl {
namespace xpu {
namespace ze {

enum class memory_storage_kind_t {
    unknown,
    host,
    device,
    shared,
};

inline memory_storage_kind_t get_memory_storage_kind(ze_memory_type_t type) {
    switch (type) {
        case ZE_MEMORY_TYPE_HOST: return memory_storage_kind_t::host;
        case ZE_MEMORY_TYPE_DEVICE: return memory_storage_kind_t::device;
        case ZE_MEMORY_TYPE_SHARED: return memory_storage_kind_t::shared;
        default: return memory_storage_kind_t::unknown;
    }
}

class memory_storage_t : public impl::memory_storage_t {
public:
    memory_storage_t(impl::engine_t *engine,
            memory_storage_kind_t kind = memory_storage_kind_t::unknown)
        : impl::memory_storage_t(engine), kind_(kind) {}

    void *ptr() const { return ptr_.get(); }

    status_t get_data_handle(void **handle) const override;
    status_t set_data_handle(void *handle) override;

    bool is_host_accessible() const override;

    status_t map_data(void **mapped_ptr, impl::stream_t *stream,
            size_t size) const override;
    status_t unmap_data(
            void *mapped_ptr, impl::stream_t *stream) const override;

    std::unique_ptr<impl::memory_storage_t> get_sub_storage(
            size_t offset, size_t size) const override;
    std::unique_ptr<impl::memory_storage_t> clone() const override;

    // Note: these are static for reusability with external interfaces.
    static void *malloc_device(impl::engine_t *engine, size_t size);
    static void *malloc_shared(impl::engine_t *engine, size_t size);
    static void free(impl::engine_t *engine, void *ptr);

private:
    status_t init_allocate(size_t size) override;

    void *malloc_host(size_t size) const;
    void free(void *ptr) const;
    status_t memcpy(impl::stream_t *stream, void *dst, const void *src,
            size_t size) const;

    std::unique_ptr<void, std::function<void(void *)>> ptr_;
    memory_storage_kind_t kind_ = memory_storage_kind_t::unknown;

    DNNL_DISALLOW_COPY_AND_ASSIGN(memory_storage_t);
};

} // namespace ze
} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif // XPU_ZE_MEMORY_STORAGE_HPP
