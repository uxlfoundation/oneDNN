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

#ifndef COMMON_HOST_SCALAR_MEMORY_STORAGE_HPP
#define COMMON_HOST_SCALAR_MEMORY_STORAGE_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/memory_storage.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"

/**
 * @class host_scalar_memory_storage_t
 * @brief Memory storage implementation for scalar data that
 * is always accessible on the host.
*/
namespace dnnl {
namespace impl {

class host_scalar_memory_storage_t : public memory_storage_t {
public:
    host_scalar_memory_storage_t()
        : memory_storage_t(nullptr), data_(nullptr, release), size_(0) {}
    ~host_scalar_memory_storage_t() override = default;

    status_t get_scalar_value(void *value, size_t value_size) const {
        if (size_ != value_size || data_ == nullptr)
            return status::invalid_arguments;

        std::memcpy(value, data_.get(), value_size);
        return status::success;
    }

    status_t set_scalar_value(const void *scalar_value, size_t value_size) {
        if (size_ != value_size || data_ == nullptr)
            return status::invalid_arguments;

        std::memcpy(data_.get(), scalar_value, value_size);
        return status::success;
    }

    bool is_host_accessible() const override { return true; }

    // Implementations below are required for internals to interact with
    // host scalar memory storage
    status_t get_data_handle(void **handle) const override {
        *handle = data_.get();
        return status::success;
    }

    status_t map_data(
            void **mapped_ptr, stream_t *stream, size_t size) const override {
        UNUSED(size);
        UNUSED(stream);
        return get_data_handle(mapped_ptr);
    }

    // Functions below are not expected to be used for host scalar storage
    status_t set_data_handle(void *handle) override {
        return status::unimplemented;
    }

    status_t unmap_data(void *mapped_ptr, stream_t *stream) const override {
        UNUSED(mapped_ptr);
        UNUSED(stream);
        return status::unimplemented;
    }

    std::unique_ptr<memory_storage_t> get_sub_storage(
            size_t offset, size_t size) const override {
        UNUSED(offset);
        UNUSED(size);
        return nullptr;
    }

    std::unique_ptr<memory_storage_t> clone() const override { return nullptr; }

protected:
    status_t init_allocate(size_t size) override {
        void *ptr = malloc(size, 64); // todo: choose better alignment?
        if (!ptr) return status::out_of_memory;
        data_ = decltype(data_)(ptr, destroy);
        size_ = size;
        return status::success;
    }

private:
    std::unique_ptr<void, void (*)(void *)> data_;
    size_t size_ = 0;

    DNNL_DISALLOW_COPY_AND_ASSIGN(host_scalar_memory_storage_t);

    static void release(void *ptr) {}
    static void destroy(void *ptr) { free(ptr); }
};

} // namespace impl
} // namespace dnnl

#endif // COMMON_HOST_SCALAR_MEMORY_STORAGE_HPP
