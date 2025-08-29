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

#ifndef UTILS_MEMORY_HPP
#define UTILS_MEMORY_HPP

#include <mutex>
#include <unordered_map>

struct memory_registry_t {
    // Increases the registered physically allocated memory.
    void add(void *ptr, size_t size);

    // Decreases the registered physically allocated memory.
    void remove(void *ptr);

    // Increases the registered mapped memory.
    void add_mapped(void *ptr, size_t size);

    // Decreases the registered mapped memory. Use `size` to track repeated
    // mapped regions.
    void remove_mapped(void *ptr, size_t size);

    // Uses `size` as an upper limit to check if allocations fit the
    // expectation. The check takes into account `expected_trh_` which increases
    // the `size`.
    void set_expected_max(size_t size);

    ~memory_registry_t();

private:
    // `expected_trh_` smoothes out small allocations for attributes memory
    // objects.
    static constexpr float expected_trh_ = 1.1f;
    static constexpr size_t unset_ = 0;
    size_t expected_max_ = unset_;
    size_t total_size_ = 0;
    bool has_warned_ = false;
    // For physically allocated memory.
    std::unordered_map<void *, size_t> allocations_;
    // For mapped memory. Mapped memory can overflow the upper limit as it
    // happens in sporadic places.
    std::unordered_map<void *, size_t> mapped_allocations_;
    std::mutex m_;

    size_t size() const { return total_size_; }

    void warn_size_check();
};

memory_registry_t &zmalloc_registry();

void set_zmalloc_max_expected_size(size_t size);

void *zmalloc(size_t size, size_t align);

void zfree(void *ptr);

#endif
