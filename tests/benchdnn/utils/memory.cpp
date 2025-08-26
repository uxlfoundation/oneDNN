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

#include "utils/memory.hpp"
#include "common.hpp"
#include "utils/bench_mode.hpp"

// BENCHDNN_MEMORY_CHECK macro enables guarding mechanism for memory allocation:
// memory block is allocated on a page boundary and the page after the block is
// protected to catch possible invalid accesses.
//
// Note that the macro affects the correctness mode only.
#ifdef __unix__
#define BENCHDNN_MEMORY_CHECK
#endif

#ifdef BENCHDNN_MEMORY_CHECK
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#endif

#include <cassert>
#include <cerrno>
#include <cstring>

void memory_registry_t::add(void *ptr, size_t size) {
    std::lock_guard<std::mutex> g(m_);
    if (!ptr) return;

    assert(allocations_.find(ptr) == allocations_.end());
    allocations_.emplace(std::pair<void *, size_t>(ptr, size));
    total_size_cpu_ += size;

    BENCHDNN_PRINT(8,
            "[CHECK_MEM]: zmalloc (%p) of size %s, CPU total alloc: %s\n", ptr,
            smart_bytes(size).c_str(), smart_bytes(total_size_cpu_).c_str());
    warn_size_check();
}

void memory_registry_t::remove(void *ptr) {
    std::lock_guard<std::mutex> g(m_);
    if (!ptr) return;

    // Use `at` to catch cases when unallocated pointers are removed.
    const size_t size = allocations_.at(ptr);
    total_size_cpu_ -= size;

    BENCHDNN_PRINT(8,
            "[CHECK_MEM]: zfree (%p) of size %s, CPU total alloc: %s\n", ptr,
            smart_bytes(size).c_str(), smart_bytes(total_size_cpu_).c_str());
    allocations_.erase(ptr);
}

void memory_registry_t::add_mapped(void *ptr, size_t size) {
    std::lock_guard<std::mutex> g(m_);
    if (!ptr) return;

    // When the library memories, which are mapped when created, reordered with
    // data, they may choose the fast path with CPU reorder. In that case, the
    // mapped memory pointer for CPU reorder will be used. Since such service
    // reorders follow the consolidated execution path, they'll unmap those new
    // memory objects, but since they are not mapped, unmap won't happen. But
    // after the execution, the mapping happens and returns the same pointer.
    // Accounting such mapping at total_size value doesn't make sense as no
    // extra memory is allocated, but increasing a local state instead to have
    // an ability to properly decrease.
    if (mapped_allocations_.find(ptr) != mapped_allocations_.end()) {
        BENCHDNN_PRINT(8,
                "[CHECK_MEM]: repeated map (%p), increase local size state\n",
                ptr);
        mapped_allocations_[ptr] += size;
    } else {
        mapped_allocations_.emplace(std::pair<void *, size_t>(ptr, size));
        total_size_cpu_ += size;

        BENCHDNN_PRINT(8,
                "[CHECK_MEM]: map (%p) of size %s, CPU total alloc: %s\n", ptr,
                smart_bytes(size).c_str(),
                smart_bytes(total_size_cpu_).c_str());
    }
    // Do not warn on overflow as it can be a temporary jump due to reorder or
    // other internal memory manipulation.
}

void memory_registry_t::remove_mapped(void *ptr, size_t size) {
    std::lock_guard<std::mutex> g(m_);
    if (!ptr) return;

    const size_t stored_size = mapped_allocations_.at(ptr);

    // See `add_mapped` comment.
    // Since double mapping may happen, double unmapping may happen as well.
    // Correspondently decrease the local state and return.
    if (stored_size > size) {
        BENCHDNN_PRINT(8,
                "[CHECK_MEM]: repeated unmap (%p), decrease local size state\n",
                ptr);
        mapped_allocations_[ptr] -= size;
    } else {
        mapped_allocations_.erase(ptr);
        total_size_cpu_ -= size;
        BENCHDNN_PRINT(8,
                "[CHECK_MEM]: unmap (%p) of size %s, CPU total alloc: %s\n",
                ptr, smart_bytes(size).c_str(),
                smart_bytes(total_size_cpu_).c_str());
    }
}

void memory_registry_t::add_device(void *ptr, size_t size) {
    std::lock_guard<std::mutex> g(m_);
    if (!ptr) return;

    assert(allocations_device_.find(ptr) == allocations_device_.end());
    allocations_device_.emplace(std::pair<void *, size_t>(ptr, size));
    total_size_gpu_ += size;

    BENCHDNN_PRINT(8,
            "[CHECK_MEM]: device_malloc (%p) of size %s, device total alloc: "
            "%s\n",
            ptr, smart_bytes(size).c_str(),
            smart_bytes(total_size_gpu_).c_str());
}

void memory_registry_t::remove_device(void *ptr) {
    std::lock_guard<std::mutex> g(m_);
    if (!ptr) return;

    if (mapped_allocations_.find(ptr) != mapped_allocations_.end()) {
        BENCHDNN_PRINT(8,
                "[CHECK_MEM]: device_free (%p) attempts to free a mapped "
                "pointer. Skip this action.\n",
                ptr);
        return;
    }

    // Use `at` to catch cases when unallocated pointers are removed.
    const size_t size = allocations_device_.at(ptr);
    total_size_gpu_ -= size;

    BENCHDNN_PRINT(8,
            "[CHECK_MEM]: device_free (%p) of size %s, device total alloc: "
            "%s\n",
            ptr, smart_bytes(size).c_str(),
            smart_bytes(total_size_gpu_).c_str());
    allocations_device_.erase(ptr);
}

void memory_registry_t::set_expected_max(size_t size) {
    expected_max_ = static_cast<size_t>(expected_trh_ * size);
    has_warned_ = false;
    warn_size_check();
}

void memory_registry_t::warn_size_check() {
    const bool is_max_set = expected_max_ != unset_;
    // Verify the total amount of allocated memory when it starts exceeding
    // 1 GB threshold. Small amount of memory is highly unlikely cause OOM.
    // There's an idea to add a portion of RAM into account as well, keep
    // only 1 GB so far to check if it proves working well.
    const bool is_total_size_cpu_big = total_size_cpu_ >= 1024 * 1024 * 1024;
    const bool is_total_size_cpu_unexpected = total_size_cpu_ > expected_max_;
    // Perf mode might have cold-cache enabled which potentially allocates
    // unaccounted memory. To avoid a dependency on a cold-cache in this
    // file, just rely on perf mode.
    if (!has_bench_mode_bit(mode_bit_t::perf) && !has_warned_ && is_max_set
            && is_total_size_cpu_big && is_total_size_cpu_unexpected) {
        BENCHDNN_PRINT(0,
                "[CHECK_MEM][ERROR]: Memory use is underestimated. Current "
                "allocation size: %s; expected size: %s.\n",
                smart_bytes(total_size_cpu_).c_str(),
                smart_bytes(expected_max_).c_str());
        // Prevent spamming logs with subsequent overflowing allocations;
        has_warned_ = true;
    }
}

memory_registry_t::~memory_registry_t() {
    if (!allocations_.empty()) {
        BENCHDNN_PRINT(
                0, "%s\n", "[CHECK_MEM][ERROR]: Allocations were not cleared");
    }
    if (!mapped_allocations_.empty()) {
        BENCHDNN_PRINT(0, "%s\n",
                "[CHECK_MEM][ERROR]: Mapped allocations were not cleared");
    }
    if (!allocations_device_.empty()) {
        BENCHDNN_PRINT(0, "%s\n",
                "[CHECK_MEM][ERROR]: Device allocations were not cleared");
    }
    if (total_size_cpu_ > 0) {
        BENCHDNN_PRINT(0, "%s\n",
                "[CHECK_MEM][ERROR]: Total size wasn't reduced to 0");
    }
}

void set_zmalloc_max_expected_size(size_t size) {
    memory_registry_t::get_instance().set_expected_max(size);
}

namespace {

#ifdef BENCHDNN_MEMORY_CHECK
void *zmalloc_protect(size_t size) {
    const size_t page_sz = getpagesize();

    const size_t block_sz = size + 3 * sizeof(void *);
    const size_t total_sz = rnd_up(block_sz, page_sz) + page_sz;

    void *mem_ptr;
    int rc = ::posix_memalign(&mem_ptr, page_sz, total_sz);
    if (rc != 0) return nullptr;

    uint8_t *ptr_start = (uint8_t *)mem_ptr;
    uint8_t *ptr = ptr_start + total_sz - page_sz - size;

    // Aligned on a page boundary
    void *ptr_protect = ptr + size;

    // Layout of the allocated region:
    // ptr_start   <- start of the allocated region
    // ptr[-16]    <- stores start address: ptr_start
    // ptr[-8]     <- stores protected address: ptr_protect
    // ptr         <- pointer to be returned from the function
    // ptr_protect <- pointer to the block to protect

    // Protect one page right after the block of size bytes
    int err = mprotect(ptr_protect, page_sz, PROT_NONE);
    if (err != 0) {
        printf("Error: mprotect returned \'%s\'.\n", strerror(errno));
        ::free(ptr_start);
        return nullptr;
    }

    // Align down `ptr` on 8 bytes before storing addresses to make behavior
    // defined.
    ptrdiff_t to_align = reinterpret_cast<ptrdiff_t>(ptr) % sizeof(void *);
    void *ptr_aligned_8 = ptr - to_align;
    // Save pointers for zfree_protect
    ((void **)ptr_aligned_8)[-2] = ptr_start;
    ((void **)ptr_aligned_8)[-1] = ptr_protect;

    return ptr;
}

void zfree_protect(void *ptr) {
    // Get aligned ptr before obtaining addresses
    ptrdiff_t to_align = reinterpret_cast<ptrdiff_t>(ptr) % sizeof(void *);
    void *ptr_aligned_8 = reinterpret_cast<uint8_t *>(ptr) - to_align;

    // Restore read-write access for the protected region
    void *ptr_protect = ((void **)ptr_aligned_8)[-1];
    const size_t page_sz = getpagesize();
    mprotect(ptr_protect, page_sz, PROT_READ | PROT_WRITE);

    // Deallocate the whole region
    void *ptr_start = ((void **)ptr_aligned_8)[-2];
    ::free(ptr_start);
}
#endif

} // namespace

void *zmalloc(size_t size, size_t align) {
#ifdef BENCHDNN_MEMORY_CHECK
    if (has_bench_mode_bit(mode_bit_t::exec)
            && !has_bench_mode_bit(mode_bit_t::perf)) {
        void *ptr = zmalloc_protect(size);
        memory_registry_t::get_instance().add(ptr, size);
        return ptr;
    }
#endif

    void *ptr;
#ifdef _WIN32
    ptr = _aligned_malloc(size, align);
    int rc = ((ptr) ? 0 : errno);
#else
    // posix_memalign requires alignment to be
    // a power of 2 and a multiple of sizeof(void *)
    if (align < sizeof(void *)) align = sizeof(void *);
    assert(((align & (align - 1)) == 0) && "align must be a power of 2");

    // TODO. Heuristics: Increasing the size to alignment increases
    // the stability of performance results.
    if (has_bench_mode_bit(mode_bit_t::perf) && (size < align)) size = align;
    int rc = ::posix_memalign(&ptr, align, size);
#endif /* _WIN32 */
    memory_registry_t::get_instance().add(ptr, size);
    return rc == 0 ? ptr : nullptr;
}

// zfree behavior is aligned with UNIX free().
void zfree(void *ptr) {
    if (!ptr) return;
    memory_registry_t::get_instance().remove(ptr);
#ifdef BENCHDNN_MEMORY_CHECK
    if (has_bench_mode_bit(mode_bit_t::exec)
            && !has_bench_mode_bit(mode_bit_t::perf)) {
        zfree_protect(ptr);
        return;
    }
#endif

#ifdef _WIN32
    _aligned_free(ptr);
#else
    return ::free(ptr);
#endif /* _WIN32 */
}
