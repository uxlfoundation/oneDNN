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

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/memory_storage.hpp"

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
// Despite the fact `exec_ctx_t` is mutable via setters to objects that are not
// available at construction place, `execute` call uses `const exec_ctx_t &`
// which prevents from calling non-const methods, thus, changing the state isn't
// possible.
struct exec_ctx_t {
    // Doesn't work without a stream and args.
    exec_ctx_t() = delete;

    exec_ctx_t(stream_t *stream, exec_args_t &&args = {});

    exec_ctx_t(const exec_ctx_t &other, exec_args_t &&args);

    // There's a number of setters due to not all objects the context relies on
    // are available at the construction time. For example, ...
    //
    // ... the memory mapping is required by CPU SYCL runtime and can be filled
    // only from a host_task call.
    //
    // `memory_mapping_` is a bridge between memory storages and their mapped
    // counterparts.
    // Key is a memory storage handle, value is a mapped pointer.
    // Mapping is kept in the context for two reasons:
    // * Provide a mapped pointer to a grantor object. The latter will be
    //   assigned to the current context.
    // * Provide mapped pointers to arguments that require zero-padding at the
    //   end of execution.
    // Methods associated with mapping serve the same purpose.
    void set_memory_mapping(void *handle, void *host_ptr);

    // ... the resource mapper is declared private and can be used only inside
    // primitive_t methods, such as `primitive_t::execute`.
    //
    // `set_resource_mapper` doesn't acquire the ownership of the mapper, as
    // the latter one lives inside a `primitive_t`.
    void set_resource_mapper(const resource_mapper_t *resource_mapper);

    // ... a grantor has a dependency on a host pointer which is available only
    // after mapping is fully set at `primitive_t::execute` call.
    //
    // `set_scratchpad_grantor` acquires the ownership of a grantor by taking a
    // pointer to it allocated by `create_grantor`. Ownership aligns the
    // lifetime of the grantor with an `exec_ctx_t` lifetime. The requirement of
    // identical lifetime comes from asynchronous threadpool runtime.
    void set_scratchpad_grantor(
            const memory_tracking::grantor_t *scratchpad_grantor);

    stream_t *stream() const;
    const exec_args_t &args() const;

    const std::unordered_map<void *, void *> &get_memory_mapping() const;
    const resource_mapper_t *get_resource_mapper() const;
    // Tip: when a pointer to `grantor` is needed, take an address of the
    // returned reference.
    const memory_tracking::grantor_t &get_scratchpad_grantor() const;

    memory_t *input(int arg) const;
    memory_t *output(int arg) const;
    memory_t *memory(int arg) const;

    status_t zero_pad_output(int arg) const;

    void *host_ptr(int arg, bool do_zeropad = false, status_t *status = nullptr,
            int index = 0) const;
    // Returns a mapped pointer for a provided `mem_storage`.
    // If `memory_mapping_` presents, returns a host_ptr, otherwise, returns
    // a storage handle.
    //
    // `require_host_ptr` forces to return a host_ptr, and if it's not
    // available, the function returns `nullptr`.
    //
    // Exclusively for a scratchpad memory in the library scratchpad mode.
    void *host_ptr(const memory_storage_t *mem_storage,
            bool require_host_ptr = false) const;

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
    // `shared_ptr` to the implementation allows the asynchronous threadpool
    // runtime to keep underlying `impl_` members alive when a lambda function
    // from a `parallel` call captures `ctx` or calls `ctx`'s methods which
    // dereferences the `shared_ptr` and increases its ref_count.
    std::shared_ptr<exec_ctx_impl_t> impl_;
};

struct exec_ctx_impl_t {
    // Doesn't work without a stream and args.
    exec_ctx_impl_t() = delete;

    ~exec_ctx_impl_t();

    exec_ctx_impl_t(const exec_ctx_impl_t &) = delete;
    exec_ctx_impl_t &operator=(const exec_ctx_impl_t &) = delete;

    void set_memory_mapping(void *handle, void *host_ptr);

    void set_resource_mapper(const resource_mapper_t *resource_mapper) {
        resource_mapper_ = resource_mapper;
    }
    void set_scratchpad_grantor(
            const memory_tracking::grantor_t *scratchpad_grantor);

    stream_t *stream() const { return stream_; }
    const exec_args_t &args() const { return args_; }

    const std::unordered_map<void *, void *> &get_memory_mapping() const {
        return memory_mapping_;
    }
    const resource_mapper_t *get_resource_mapper() const {
        return resource_mapper_;
    }
    const memory_tracking::grantor_t &get_scratchpad_grantor() const;

    memory_t *input(int arg) const;
    memory_t *output(int arg) const;
    memory_t *memory(int arg) const;

    void *host_ptr(const memory_storage_t *mem_storage,
            bool require_host_ptr = false) const;

    void *map_memory_storage(const memory_storage_t *storage, stream_t *stream,
            size_t size) const;
    void unmap_memory_storage(const memory_storage_t *storage, void *mapped_ptr,
            stream_t *stream) const;

    memory_desc_wrapper memory_mdw(int arg,
            const memory_desc_t *md_from_primitive_desc = nullptr) const;

private:
    stream_t *stream_;
    exec_args_t args_;

    // See `exec_ctx_t::set_memory_mapping` comment.
    std::unordered_map<void *, void *> memory_mapping_;

    // See `exec_ctx_t::set_resource_mapper` comment.
    const resource_mapper_t *resource_mapper_ = nullptr;

    // See `exec_ctx_t::set_scratchpad_grantor` comment.
    std::unique_ptr<const memory_tracking::grantor_t> scratchpad_grantor_;

    // Keep constructors and destructors private to avoid misuse of this object,
    // but let the context call them.
    friend struct exec_ctx_t;

    exec_ctx_impl_t(stream_t *stream, exec_args_t &&args);

    exec_ctx_impl_t(stream_t *stream, exec_args_t &&args,
            const std::unordered_map<void *, void *> &memory_mapping,
            const resource_mapper_t *resource_mapper);
};

} // namespace impl
} // namespace dnnl

#endif
