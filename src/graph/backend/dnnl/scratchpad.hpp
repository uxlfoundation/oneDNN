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
#ifndef GRAPH_BACKEND_DNNL_SCRATCHPAD_HPP
#define GRAPH_BACKEND_DNNL_SCRATCHPAD_HPP

#include <functional>
#include <memory>
#include <unordered_map>

#include "graph/interface/allocator.hpp"

#include "graph/backend/dnnl/common.hpp"

#include "oneapi/dnnl/dnnl.hpp"

#ifdef DNNL_WITH_SYCL
#include "oneapi/dnnl/dnnl_sycl.hpp"
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

// Unified scratchpad class that handles both library-managed (temporary) and
// user-provided buffer modes. When user_buf is non-null, the buffer lifecycle
// is managed by the user; otherwise the library allocates and frees the buffer.
class scratchpad_t {
public:
    scratchpad_t(const tensor_t *user_buf, size_t size, const engine_t &eng);

    ~scratchpad_t();

    // Disable assignment and copy
    scratchpad_t(const scratchpad_t &) = delete;
    scratchpad_t &operator=(const scratchpad_t &) = delete;

    char *get_buffer() const { return buffer_; }
    size_t size() const { return size_; }
    bool is_user_managed() const { return user_managed_; }

#ifdef DNNL_WITH_SYCL
    void set_deps(::sycl::event event) {
        if (!user_managed_) e_ = std::move(event);
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    void set_deps(cl_event event) {
        if (!user_managed_) ocl_e_ = event;
    }
#endif

private:
    char *buffer_;
    size_t size_;
    bool user_managed_;
    const engine_t *eng_;
#ifdef DNNL_WITH_SYCL
    ::sycl::event e_;
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event ocl_e_;
#endif
};

// This function artificially extends the `scratchpad_t` object's lifetime to
// keep alive the handle this scratchpad object manages. An alternative approach
// is to manage its handles through a system of caches the same way it's done
// for memory objects before they are passed into a primitive execute call.
// No-op if the scratchpad is user-managed.
void prolong_scratchpad_lifetime(
        stream_t *stream, const std::shared_ptr<scratchpad_t> &scratchpad);

class registrar_t;
class grantor_t;

// This class is responsible for computing the correct offset for a given piece
// of memory according to the size and alignment.
class registry_t {
public:
    using key_t = size_t;
    using offset_t = size_t;

    // book a piece of memory according to the size and alignment
    void book(const key_t &key, size_t size, size_t alignment) {
        // If the piece is booked, skip it
        if (offset_map_.count(key)) return;

        if (size_ % alignment != 0) {
            size_ = ((size_ / alignment) + 1) * alignment;
        }

        offset_map_.insert({key, size_});
        size_ += size;

        // Since the user given base pointer may not be aligned, so we need to
        // compute a base_offset to make sure that base_offset + offset_map_[i]
        // can match the alignment requirements.

        // Rational:
        // because:
        //     lcm_alignment_ % alignment == 0 is true
        // and we can set:
        //     aligned_ptr = (base_ptr/lcm_alignment_+1)*lcm_alignment_
        // so:
        //     aligned_ptr % alignment == 0 is true
        // then, because:
        //     aligned_ptr% alignment == 0 is true
        //     offset_map[i] % alignment == 0 is true
        // so:
        //     (aligned_ptr + offset_map[i]) % alignment == 0 is true
        lcm_alignment_ = graph::utils::lcm(lcm_alignment_, alignment);
    }

    // get the offset of a booked piece of memory
    offset_t get(const key_t &key) const {
        if (size_ == 0 || offset_map_.count(key) != 1) return 0;
        return offset_map_.at(key);
    }

    // get the total size, which is the sum of the total booked size and the
    // reserved extra space size for alignments
    size_t size() const { return size_ == 0 ? size_ : size_ + lcm_alignment_; }

    // get the the least common multiple of all registered memories' alignments
    size_t lcm_alignment() const { return lcm_alignment_; }

    // create a registrar from this registry
    registrar_t registrar();

    // create a grantor from this registry
    grantor_t grantor(char *base_ptr) const;

    void clear() {
        offset_map_.clear();
        size_ = 0;
        lcm_alignment_ = 1;
    }

private:
    std::unordered_map<key_t, offset_t> offset_map_;
    size_t size_ {0}; // registered buffers' total size
    size_t lcm_alignment_ {1};
};

// This class is a wrapper of registry_t for usability
class registrar_t {
public:
    registrar_t(registry_t &registry) : registry_(registry) {}

    void book(
            const registry_t::key_t &key, size_t size, size_t alignment = 64) {
        registry_.book(key, size, alignment);
    }

private:
    registry_t &registry_;
};

// This class is used to compute the actual address of each piece of memory
// according to a given base pointer and the information in registry
class grantor_t {
public:
    grantor_t(const registry_t &registry, char *base_ptr)
        : registry_(registry) {
        UNUSED(base_ptr);
        size_t lcm_alignment = registry.lcm_alignment();
        aligned_base_ptr_ = reinterpret_cast<char *>(
                (reinterpret_cast<size_t>(base_ptr) + lcm_alignment - 1)
                / lcm_alignment * lcm_alignment);
    }

    char *get(const registry_t::key_t &key) const {
        return aligned_base_ptr_ ? (aligned_base_ptr_ + registry_.get(key))
                                 : nullptr;
    }

private:
    const registry_t &registry_;
    char *aligned_base_ptr_;
};

inline registrar_t registry_t::registrar() {
    return registrar_t(*this);
}

inline grantor_t registry_t::grantor(char *base_ptr) const {
    return grantor_t(*this, base_ptr);
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
