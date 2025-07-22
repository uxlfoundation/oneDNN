/*******************************************************************************
* Copyright 2021-2025 Intel Corporation
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

#ifndef GPU_INTEL_L0_MEMORY_STORAGE_HPP
#define GPU_INTEL_L0_MEMORY_STORAGE_HPP

#include <functional>

#include "common/c_types_map.hpp"
#include "common/memory_storage.hpp"
#include "common/utils.hpp"

#include "gpu/intel/l0/engine.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace l0 {

enum class memory_storage_kind_t { unknown, host, device, shared };
inline memory_storage_kind_t get_memory_storage_kind(ze_memory_type_t type) {
    switch (type) {
        case ZE_MEMORY_TYPE_HOST: return memory_storage_kind_t::host;
        case ZE_MEMORY_TYPE_DEVICE: return memory_storage_kind_t::device;
        case ZE_MEMORY_TYPE_SHARED: return memory_storage_kind_t::shared;
        default: return memory_storage_kind_t::unknown;
    }
};

class memory_storage_t : public impl::memory_storage_t {
public:
    memory_storage_t(impl::engine_t *engine, memory_storage_kind_t kind)
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

private:
    status_t init_allocate(size_t size) override;

    gpu::intel::l0::engine_t *l0_engine() const {
        return utils::downcast<gpu::intel::l0::engine_t *>(engine());
    }

    void *malloc_host(size_t size) const;
    void *malloc_device(size_t size) const;
    void *malloc_shared(size_t size) const;
    void free(void *ptr) const;
    status_t memcpy(impl::stream_t *stream, void *dst, const void *src,
            size_t size) const;

    std::unique_ptr<void, std::function<void(void *)>> ptr_;
    memory_storage_kind_t kind_ = memory_storage_kind_t::unknown;

    DNNL_DISALLOW_COPY_AND_ASSIGN(memory_storage_t);
};

} // namespace l0
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_L0_MEMORY_STORAGE_HPP
