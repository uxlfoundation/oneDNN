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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <type_traits>

#include "common/c_types_map.hpp"
#include "common/memory_storage.hpp"
#include "common/type_helpers.hpp"

/**
 * @class host_scalar_memory_storage_t
 * @brief Memory storage implementation for scalar data that
 * is always accessible on the host.
*/
namespace dnnl {
namespace impl {

class host_scalar_memory_storage_t : public memory_storage_t {
public:
    static constexpr size_t max_scalar_size = 16;
    static constexpr size_t max_scalar_align = alignof(std::max_align_t);

    host_scalar_memory_storage_t()
        : memory_storage_t(nullptr)
        , data_type_(dnnl_data_type_undef)
        , is_initialized_(false) {}
    ~host_scalar_memory_storage_t() override = default;

    status_t get_scalar_value(void *value, size_t value_size) const {
        if (!is_initialized_ || value == nullptr
                || value_size != types::data_type_size(data_type_)
                || value_size == 0 || value_size > max_scalar_size)
            return status::invalid_arguments;

        // NOTE: std::copy implementation prevents GCC -Warray-bounds warning with older versions
        const auto *src
                = reinterpret_cast<const std::uint8_t *>(&scalar_storage_);
        auto *dst = static_cast<std::uint8_t *>(value);

        std::copy(src, src + value_size, dst);
        return status::success;
    }

    status_t set_scalar_value(const void *scalar_value, data_type_t data_type) {
        if (scalar_value == nullptr || data_type == dnnl_data_type_undef)
            return status::invalid_arguments;

        const auto data_size = types::data_type_size(data_type);

        if (data_size == 0 || data_size > max_scalar_size)
            return status::invalid_arguments;

        // NOTE: std::copy implementation prevents GCC -Warray-bounds warning with older versions
        const auto *src = static_cast<const std::uint8_t *>(scalar_value);
        auto *dst = reinterpret_cast<std::uint8_t *>(&scalar_storage_);

        std::copy(src, src + data_size, dst);

        data_type_ = data_type;
        is_initialized_ = true;

        return status::success;
    }

    bool is_host_accessible() const override { return true; }

    bool is_host_scalar() const override { return true; }

    data_type_t data_type() const { return data_type_; }

    // Required for compatibility
    // Usage for getting a value is discouraged, use get_scalar_value instead
    status_t get_data_handle(void **handle) const override {
        if (!is_initialized_ || handle == nullptr)
            return status::invalid_arguments;
        *handle = const_cast<void *>(
                static_cast<const void *>(&scalar_storage_));
        return status::success;
    }

    // Required for compatibility
    status_t map_data(
            void **mapped_ptr, stream_t *stream, size_t size) const override {
        UNUSED(size);
        UNUSED(stream);
        return get_data_handle(mapped_ptr);
    }

    // Required for compatibility
    status_t unmap_data(void *mapped_ptr, stream_t *stream) const override {
        UNUSED(mapped_ptr);
        UNUSED(stream);
        return status::success;
    }

    // Not supported for scalar storage
    status_t set_data_handle(void *handle) override {
        return status::unimplemented;
    }

    // Not supported for scalar storage
    std::unique_ptr<memory_storage_t> get_sub_storage(
            size_t offset, size_t size) const override {
        UNUSED(offset);
        UNUSED(size);
        return nullptr;
    }

    // Not supported for scalar storage
    std::unique_ptr<memory_storage_t> clone() const override { return nullptr; }

protected:
    // Not supported for scalar storage
    status_t init_allocate(size_t size) override {
        UNUSED(size);
        return status::unimplemented;
    }

private:
    typename std::aligned_storage<max_scalar_size, max_scalar_align>::type
            scalar_storage_;
    data_type_t data_type_;
    bool is_initialized_;

    DNNL_DISALLOW_COPY_AND_ASSIGN(host_scalar_memory_storage_t);
};

} // namespace impl
} // namespace dnnl

#endif // COMMON_HOST_SCALAR_MEMORY_STORAGE_HPP
