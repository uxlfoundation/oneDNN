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

#ifndef COMMON_SCALAR_MEMORY_STORAGE_HPP
#define COMMON_SCALAR_MEMORY_STORAGE_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/memory_storage.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"

// todo: remove
#include <iostream>

/**
 * @class host_scalar_memory_storage_t
 * @brief Scalar memory storage implementation for data that is always accessible on the host.
*/
namespace dnnl {
namespace impl {

class host_scalar_memory_storage_t : public memory_storage_t {
public:
    host_scalar_memory_storage_t()
        : memory_storage_t(nullptr), data_(nullptr, release) {}
    ~host_scalar_memory_storage_t() override = default;

    status_t get_data_handle(void **handle) const override {
        std::cout << "[DEBUG] Getting scalar memory storage data handle"
                  << std::endl;
        *handle = data_.get();
        return status::success;
    }

    status_t set_data_handle(void *handle) override {
        std::cout << "[DEBUG] Setting scalar memory storage data handle"
                  << std::endl;
        data_ = decltype(data_)(handle, release);
        return status::success;
    }

    status_t map_data(
            void **mapped_ptr, stream_t *stream, size_t size) const override {
        UNUSED(size);
        UNUSED(stream);
        return get_data_handle(mapped_ptr);
    }

    status_t unmap_data(void *mapped_ptr, stream_t *stream) const override {
        UNUSED(mapped_ptr);
        UNUSED(stream);
        return status::success;
    }

    bool is_host_accessible() const override { return true; }

    std::unique_ptr<memory_storage_t> get_sub_storage(
            size_t offset, size_t size) const override {
        UNUSED(offset);
        UNUSED(size);
        return nullptr;
    }

    std::unique_ptr<memory_storage_t> clone() const override {
        std::cout << "[DEBUG] Cloning scalar memory storage" << std::endl;
        auto storage = new host_scalar_memory_storage_t();
        if (storage)
            storage->init(memory_flags_t::use_runtime_ptr,
                    0 /* size is not required for use_runtime_ptr */,
                    data_.get());
        return std::unique_ptr<memory_storage_t>(storage);
    }

protected:
    status_t init_allocate(size_t size) override {
        std::cout << "[DEBUG] Initializing scalar memory storage with size: "
                  << size << std::endl;
        void *ptr = malloc(size, 64); // todo: choose better alignment?
        if (!ptr) return status::out_of_memory;
        data_ = decltype(data_)(ptr, destroy);
        return status::success;
    }

private:
    std::unique_ptr<void, void (*)(void *)> data_;

    DNNL_DISALLOW_COPY_AND_ASSIGN(host_scalar_memory_storage_t);

    static void release(void *ptr) {
        std::cout << "[DEBUG] Releasing scalar memory storage" << std::endl;
    }
    static void destroy(void *ptr) {
        free(ptr);
        std::cout << "[DEBUG] Destroying scalar memory storage" << std::endl;
    }
};

} // namespace impl
} // namespace dnnl

#endif // COMMON_SCALAR_MEMORY_STORAGE_HPP
