/*******************************************************************************
* Copyright 2018-2025 Intel Corporation
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

#ifndef COMMON_PRIMITIVE_EXEC_TYPES_HPP
#define COMMON_PRIMITIVE_EXEC_TYPES_HPP

#include <unordered_map>

#include "oneapi/dnnl/dnnl_types.h"

#include "c_types_map.hpp"
#include "memory.hpp"
#include "memory_storage.hpp"

// __VA_ARGS__here is an index of the buffer. It is empty unless the memory
// argument is sparse.
#define CTX_IN_STORAGE(arg, ...) CTX_IN_STORAGe##__VA_ARGS__(arg)

#define CTX_IN_STORAGe(arg) \
    (ctx.input(arg) ? *(ctx.input(arg)->memory_storage()) \
                    : dnnl::impl::memory_storage_t::empty_storage())
#define CTX_IN_STORAGe0(arg) \
    (ctx.input(arg) ? *ctx.input(arg)->memory_storage(0) \
                    : dnnl::impl::memory_storage_t::empty_storage())
#define CTX_IN_STORAGe1(arg) \
    (ctx.input(arg) ? *ctx.input(arg)->memory_storage(1) \
                    : dnnl::impl::memory_storage_t::empty_storage())
#define CTX_IN_STORAGe2(arg) \
    (ctx.input(arg) ? *ctx.input(arg)->memory_storage(2) \
                    : dnnl::impl::memory_storage_t::empty_storage())

// Returns destination memory which may not have been zero pad initialized.
#define CTX_OUT_STORAGE(arg) \
    (ctx.output(arg) ? *(ctx.output(arg)->memory_storage()) \
                     : dnnl::impl::memory_storage_t::empty_storage())

// Returns destination memory which has been zero pad initialized. This macro
// may result in a failure returned via the `status` input since zero pad
// may fail.
#define CTX_OUT_CLEAN_STORAGE(arg, status) \
    (ctx.output(arg) ? *(ctx.output(arg)->memory_storage_clean(ctx, status)) \
                     : dnnl::impl::memory_storage_t::empty_storage())

namespace dnnl {
namespace impl {

namespace memory_tracking {
struct grantor_t;
} // namespace memory_tracking

struct memory_arg_t {
    memory_t *mem;
    bool is_const;
};

struct primitive_desc_t;

using exec_args_t = std::unordered_map<int, memory_arg_t>;

status_t cvt_primitive_args(const primitive_desc_t *pd, int nargs,
        const dnnl_exec_arg_t *c_args, exec_args_t &args);

struct exec_ctx_impl_t;
struct resource_mapper_t;

// Primitive execution context, it helps to pass a stream, memory objects, and
// events.
//
// Despite the fact that `exec_ctx_t` is mutable with setters to objects that
// are not available at creation spot, `execute` signature passes `exec_ctx_t`
// by `const &` preventing from calling non-const methods, thus, changing the
// state is not possible.
struct exec_ctx_t {
    // Doesn't work without a stream and args.
    exec_ctx_t() = delete;

    // An artificial version with stream only for gemm::exec_ctx_t to provide an
    // ability to create a nested scratchpad.
    exec_ctx_t(stream_t *stream)
        : impl_(std::make_shared<exec_ctx_impl_t>(stream)) {}

    // A main version when only a stream and args is available. An impl_ object
    // will create necessary object itself.
    exec_ctx_t(stream_t *stream, exec_args_t &&args)
        : impl_(std::make_shared<exec_ctx_impl_t>(stream, std::move(args))) {}

    // A full version with relevant objects passed to the impl_ creation.
    // Supposed to be used only when nested_scratchpad_t is created.
    exec_ctx_t(const exec_ctx_t &other, exec_args_t &&args)
        : impl_(std::make_shared<exec_ctx_impl_t>(other.stream(),
                std::move(args), other.get_memory_mapping(),
                other.get_resource_mapper())) {}

    // See `exec_ctx_impl_t` setters comment.
    void set_memory_mapping(void *handle, void *host_ptr);
    void set_resource_mapper(const resource_mapper_t *resource_mapper);
    void set_scratchpad_grantor(
            const memory_tracking::grantor_t *scratchpad_grantor);

    stream_t *stream() const;
    const exec_args_t &args() const;

    const std::unordered_map<void *, void *> &get_memory_mapping() const;
    const resource_mapper_t *get_resource_mapper() const;
    // To obtain a pointer to `grantor`, one needs to take an address of
    // returned reference to the object.
    const memory_tracking::grantor_t &get_scratchpad_grantor() const;

    memory_t *input(int arg) const;
    memory_t *output(int arg) const;
    memory_t *memory(int arg) const;

    status_t zero_pad_output(int arg) const;

    void *host_ptr(int arg, bool do_zeropad = false, status_t *status = nullptr,
            int index = 0) const;
    // Exclusively for a scratchpad memory since there's a library scratchpad.
    void *host_ptr(const memory_storage_t *mem_storage) const;

    void *map_memory_storage(const memory_storage_t *storage, stream_t *stream,
            size_t size) const;
    void unmap_memory_storage(const memory_storage_t *storage, void *mapped_ptr,
            stream_t *stream) const;

    memory_desc_wrapper memory_mdw(int arg,
            const memory_desc_t *md_from_primitive_desc = nullptr) const;

private:
    std::shared_ptr<exec_ctx_impl_t> impl_;
};

struct exec_ctx_impl_t {
    exec_ctx_impl_t() = delete;

    exec_ctx_impl_t(stream_t *stream) : stream_(stream) {}

    exec_ctx_impl_t(stream_t *stream, exec_args_t &&args)
        : stream_(stream), args_(std::move(args)) {}

    exec_ctx_impl_t(stream_t *stream, exec_args_t &&args,
            const std::unordered_map<void *, void *> &memory_mapping,
            const resource_mapper_t *resource_mapper)
        : stream_(stream)
        , args_(std::move(args))
        , memory_mapping_(memory_mapping)
        , resource_mapper_(resource_mapper) {}

    // Copying `args` is restricted, must be moved instead.
    exec_ctx_impl_t(const exec_ctx_impl_t &) = delete;
    exec_ctx_impl_t &operator=(const exec_ctx_impl_t &) = delete;

    // There's a bunch of setters due to the fact that not all objects are
    // available at the construction time...
    //
    // ... memory mapping is required by CPU SYCL and can be provided only from
    // a host_task.
    void set_memory_mapping(void *handle, void *host_ptr) {
        assert(memory_mapping_.count(handle) == 0);
        memory_mapping_.insert({handle, host_ptr});
    }
    // ... resource mapper is XXX.
    void set_resource_mapper(const resource_mapper_t *resource_mapper) {
        resource_mapper_ = resource_mapper;
    }
    // ... grantor has a back dependency on `exec_ctx_t`, thus, `exec_ctx` must
    // be created without it first.
    void set_scratchpad_grantor(
            const memory_tracking::grantor_t *scratchpad_grantor) {
        // if `scratchpad_grantor_` is unique, do reset to capture the pointer.
        scratchpad_grantor_ = scratchpad_grantor;
    }

    stream_t *stream() const { return stream_; }
    const exec_args_t &args() const { return args_; }

    const std::unordered_map<void *, void *> &get_memory_mapping() const {
        return memory_mapping_;
    }
    const resource_mapper_t *get_resource_mapper() const {
        return resource_mapper_;
    }
    const memory_tracking::grantor_t &get_scratchpad_grantor() const {
        assert(scratchpad_grantor_);
        return *scratchpad_grantor_;
    }

    memory_t *input(int arg) const;
    memory_t *output(int arg) const;
    memory_t *memory(int arg) const;

    void *host_ptr(const memory_storage_t *mem_storage) const;

    void *map_memory_storage(const memory_storage_t *storage, stream_t *stream,
            size_t size) const;
    void unmap_memory_storage(const memory_storage_t *storage, void *mapped_ptr,
            stream_t *stream) const;

    // Returns memory descriptor wrapper for the corresponding memory argument.
    //
    // To support sub-memory flow (when primitive descriptor was created with
    // a sub-memory, but the primitive is executed on the original memory),
    // it is recommended to pass the memory descriptor from the primitive
    // descriptor. If this memory descriptor is fully defined (i.e. no reason
    // to use memory descriptor from the input memory), exactly it will be
    // returned.
    //
    // Note: fully defined memory descriptor mentioned above is a synonym to
    //       `mdw::has_runtime_dims_or_strides() == false`.
    //
    // XXX: revisit this behavior in oneDNN v2.0. It would be more consistent to
    //      take memory description from the incoming argument. This will
    //      require a sub-memory object, though...
    memory_desc_wrapper memory_mdw(int arg,
            const memory_desc_t *md_from_primitive_desc = nullptr) const;

private:
    stream_t *stream_;
    exec_args_t args_;

    std::unordered_map<void *, void *> memory_mapping_;
    const resource_mapper_t *resource_mapper_ = nullptr;
    const memory_tracking::grantor_t *scratchpad_grantor_ = nullptr;
    // Convert into unique_ptr to emphasize owning and lifetime.
};

} // namespace impl
} // namespace dnnl

#endif
