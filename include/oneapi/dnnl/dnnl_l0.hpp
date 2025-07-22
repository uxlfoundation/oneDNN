/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

#ifndef ONEAPI_DNNL_DNNL_L0_HPP
#define ONEAPI_DNNL_DNNL_L0_HPP

#include "oneapi/dnnl/dnnl.hpp"

/// @cond DO_NOT_DOCUMENT_THIS
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl_l0.h"
/// @endcond

/// @addtogroup dnnl_api
/// @{

namespace dnnl {

/// @addtogroup dnnl_api_interop
/// @{

/// @addtogroup dnnl_api_l0_interop Level Zero interoperability API
/// API extensions to interact with the underlying Level Zero run-time.
///
/// @sa @ref dev_guide_dpcpp_interoperability in developer guide
/// @{

/// Level Zero interoperability namespace
namespace l0_interop {

/// Constructs an engine from Level Zero device and context objects.
///
/// @param adriver Level Zero driver.
/// @param adevice Level Zero device.
/// @param acontext Level Zero context.
///
/// @returns Created engine.
inline engine make_engine(const ze_driver_handle_t adriver,
        const ze_device_handle_t adevice, const ze_context_handle_t acontext) {
    dnnl_engine_t aengine;
    error::wrap_c_api(
            dnnl_l0_interop_engine_create(&aengine, adriver, adevice, acontext),
            "could not create an engine");
    return engine(aengine);
}

/// Returns the Level Zero context associated with an engine.
///
/// @param aengine Engine to query.
///
/// @returns The underlying Level Zero device of the engine.
inline ze_context_handle_t get_context(const engine &aengine) {
    ze_context_handle_t ctx = nullptr;
    error::wrap_c_api(dnnl_l0_interop_engine_get_context(aengine.get(), ctx),
            "could not get a context handle");
    return ctx;
}

/// Returns the Level Zero device associated with an engine.
///
/// @param aengine Engine to query.
///
/// @returns The underlying Level Zero context of the engine.
inline ze_device_handle_t get_device(const engine &aengine) {
    ze_device_handle_t dev = nullptr;
    error::wrap_c_api(dnnl_l0_interop_engine_get_device(aengine.get(), dev),
            "could not get a device handle");
    return dev;
}

/// Returns the Level Zero driver associated with an engine.
///
/// @param aengine Engine to query.
///
/// @returns The underlying Level Zero driver of the engine.
inline ze_driver_handle_t get_driver(const engine &aengine) {
    ze_driver_handle_t dri = nullptr;
    error::wrap_c_api(dnnl_l0_interop_engine_get_driver(aengine.get(), dri),
            "could not get a driver handle");
    return dri;
}

/// Creates an execution stream for a given engine associated with a Level Zero
/// queue.
///
/// @param aengine Engine object to use for the stream.
/// @param alist Level Zero immediate command list to use for the stream.
///
/// @returns An execution stream.
inline stream make_stream(
        const engine &aengine, ze_command_list_handle_t alist) {
    dnnl_stream_t astream;
    error::wrap_c_api(
            dnnl_l0_interop_stream_create(&astream, aengine.get(), alist),
            "could not create a stream");
    return stream(astream);
}

/// Returns the Level Zero immediate command list associated with an execution stream.
///
/// @param astream Execution stream to query.
///
/// @returns Level Zero immediate command list object.
inline ze_command_list_handle_t get_list(const stream &astream) {
    ze_command_list_handle_t list = nullptr;
    error::wrap_c_api(dnnl_l0_interop_stream_get_list(astream.get(), list),
            "could not get a stream handle");
    return list;
}

/// Creates a memory object with multiple handles.
///
/// @param memory_desc Memory descriptor.
/// @param aengine Engine to use.
/// @param handles Handles of the memory buffers to use as underlying storages.
///     For each element of the @p handles array the following applies:
///     - A USM pointer to the user-allocated buffer. In this case the library
///       doesn't own the buffer. Requires @p memory_kind to be equal to
///       dnnl::l0_interop::memory_kind::usm.
///     - The DNNL_MEMORY_ALLOCATE special value. Instructs the library to
///       allocate the buffer for the memory object. In this case the library
///       owns the buffer.
///     - The DNNL_MEMORY_NONE specific value. Instructs the library to
///       create memory object without an underlying buffer.
///
///  If the @p handles vector is not provided the library will allocate all
///  buffers as if all handles have the special value DNNL_MEMORY_ALLOCATE.
///
/// @returns Created memory object.
inline memory make_memory(const memory::desc &memory_desc,
        const engine &aengine, std::vector<void *> handles = {}) {
    if (handles.empty()) {
        const int nhandles = memory_desc.get_num_handles();
        handles.resize(nhandles, DNNL_MEMORY_ALLOCATE);
    }

    dnnl_memory_t c_memory;
    error::wrap_c_api(
            dnnl_l0_interop_memory_create_v2(&c_memory, memory_desc.get(),
                    aengine.get(), handles.size(), handles.data()),
            "could not create a memory");
    return memory(c_memory);
}

/// Creates a memory object.
///
/// Unless @p handle is equal to DNNL_MEMORY_NONE or DNNL_MEMORY_ALLOCATE, the
/// constructed memory object will have the underlying buffer set. In this
/// case, the buffer will be initialized as if:
/// - dnnl::memory::set_data_handle() had been called, if @p memory_kind is
///   equal to dnnl::l0_interop::memory_kind::usm, or
/// - dnnl::l0_interop::set_buffer() has been called, if @p memory_kind is
///   equal to dnnl::l0_interop::memory_kind::buffer.
///
/// @param memory_desc Memory descriptor.
/// @param aengine Engine to use.
/// @param handle Handle of the memory buffer to use as an underlying storage.
///     - A USM pointer to the user-allocated buffer. In this case the library
///       doesn't own the buffer. Requires @p memory_kind to be equal to
///       dnnl::l0_interop::memory_kind::usm.
///     - The DNNL_MEMORY_ALLOCATE special value. Instructs the library to
///       allocate the buffer for the memory object. In this case the library
///       owns the buffer.
///     - The DNNL_MEMORY_NONE specific value. Instructs the library to
///       create memory object without an underlying buffer.
///
/// @returns Created memory object.
inline memory make_memory(
        const memory::desc &memory_desc, const engine &aengine, void *handle) {
    return make_memory(memory_desc, aengine, std::vector<void *> {handle});
}

/// Returns the Level Zero memory object associated with the memory object.
///
/// @param amemory A memory object.
/// @returns Underlying Level Zero memory object.
inline void *get_mem_object(const memory &amemory) {
    void *mem_object;
    error::wrap_c_api(
            dnnl_l0_interop_memory_get_mem_object(amemory.get(), &mem_object),
            "could not get Level Zero buffer object from a memory object");
    return mem_object;
}

/// Sets the Level Zero memory object associated with the memory object.
///
/// For behavioral details see memory::set_data_handle().
///
/// @param amemory A memory object.
/// @param mem_object Level Zero cl_mem object to use as the underlying
///     storage. It must have at least get_desc().get_size() bytes
///     allocated.
inline void set_mem_object(memory &amemory, void *mem_object) {
    error::wrap_c_api(
            dnnl_l0_interop_memory_set_mem_object(amemory.get(), mem_object),
            "could not set Level Zero buffer object from a memory object");
}

/// Executes computations specified by the primitive in a specified stream and
/// returns a Level Zero event.
///
/// Arguments are passed via an arguments map containing
/// <index, memory object> pairs. The index must be one of the `DNNL_ARG_*`
/// values such as `DNNL_ARG_SRC`, and the memory must have a memory descriptor
/// matching the one returned by
/// #dnnl::primitive_desc::query_md(#query::exec_arg_md, index) unless using
/// dynamic shapes (see #DNNL_RUNTIME_DIM_VAL).
///
/// @param aprimitive Primitive to execute.
/// @param astream Stream object. The stream must belong to the same engine
///     as the primitive.
/// @param args Arguments map.
/// @param deps Optional vector with `ze_event_handle_t` dependencies.
///
/// @returns Output event.
inline ze_event_handle_t execute(const dnnl::primitive &aprimitive,
        const stream &astream, const std::unordered_map<int, memory> &args,
        const std::vector<ze_event_handle_t> &deps = {}) {
    std::vector<dnnl_exec_arg_t> c_args;
    c_args.reserve(args.size());
    for (const auto &a : args)
        c_args.push_back({a.first, a.second.get()});

    const ze_event_handle_t *c_deps = deps.empty() ? nullptr : deps.data();

    ze_event_handle_t return_event;
    error::wrap_c_api(dnnl_l0_interop_primitive_execute(aprimitive.get(),
                              astream.get(), c_args.size(), c_args.data(),
                              deps.size(), c_deps, &return_event),
            "could not execute a primitive");
    return return_event;
}

} // namespace l0_interop

/// @} dnnl_api_l0_interop

/// @} dnnl_api_interop

} // namespace dnnl

/// @} dnnl_api

#endif // ONEAPI_DNNL_DNNL_L0_HPP
