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

#ifndef ONEAPI_DNNL_DNNL_L0_H
#define ONEAPI_DNNL_DNNL_L0_H

#include "oneapi/dnnl/dnnl.h"

/// @cond DO_NOT_DOCUMENT_THIS
#include "level_zero/ze_api.h"
/// @endcond

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/// @addtogroup dnnl_api
/// @{

/// @addtogroup dnnl_api_interop
/// @{

/// @addtogroup dnnl_api_l0_interop
/// @{

/// Creates an engine associated with a Level Zero device and a Level Zero context.
///
/// @param engine Output engine.
/// @param driver Pointer to the Level Zero driver to use for the engine.
/// @param device Pointer to the Level Zero device to use for the engine.
/// @param context Pointer to the Level Zero context to use for the engine.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_l0_interop_engine_create(dnnl_engine_t *engine,
        const ze_driver_handle_t adriver, const ze_device_handle_t adevice,
        const ze_context_handle_t acontext);

/// Returns the Level Zero context associated with an engine.
///
/// @param engine Engine to query.
/// @param context Pointer to the underlying Level Zero context of the engine.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_l0_interop_engine_get_context(
        dnnl_engine_t engine, ze_context_handle_t context);

/// Returns the Level Zero device associated with an engine.
///
/// @param engine Engine to query.
/// @param device Pointer to the underlying Level Zero device of the engine.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_l0_interop_engine_get_device(
        dnnl_engine_t engine, ze_device_handle_t device);

/// Returns the Level Zero driver associated with an engine.
///
/// @param engine Engine to query.
/// @param device Pointer to the underlying Level Zero driver of the engine.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_l0_interop_engine_get_driver(
        dnnl_engine_t engine, ze_driver_handle_t driver);

/// Creates an execution stream for a given engine associated with a Level Zero
/// queue.
///
/// @param stream Output execution stream.
/// @param engine Engine to create the execution stream on.
/// @param queue Level Zero command queue to use.
/// @param list Level Zero command list to use.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_l0_interop_stream_create(dnnl_stream_t *stream,
        dnnl_engine_t engine, ze_command_list_handle_t list);

/// Returns the Level Zero command list associated with an execution stream.
///
/// @param stream Execution stream to query.
/// @param list Output Level Zero command list.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_l0_interop_stream_get_list(
        dnnl_stream_t stream, ze_command_list_handle_t list);

/// Creates a memory object.
///
/// Unless @p handle is equal to DNNL_MEMORY_NONE or DNNL_MEMORY_ALLOCATE, the
/// constructed memory object will have the underlying buffer set. In this
/// case, the buffer will be initialized as if:
/// - dnnl_memory_set_data_handle() had been called, if @p memory_kind is equal
///   to dnnl_l0_interop_usm, or
/// - dnnl_l0_interop_memory_set_buffer() has been called, if @p memory_kind
///   is equal to dnnl_l0_interop_buffer.
///
/// @param memory Output memory object.
/// @param memory_desc Memory descriptor.
/// @param engine Engine to use.
/// @param handle Handle of the memory buffer to use as an underlying storage.
///     - A USM pointer to the user-allocated buffer. In this case the library
///       doesn't own the buffer. Requires @p memory_kind to be equal to
///       dnnl::l0_interop::memory_kind::usm.
///     - The DNNL_MEMORY_ALLOCATE special value. Instructs the library to
///       allocate the buffer for the memory object. In this case the library
///       owns the buffer.
///     - The DNNL_MEMORY_NONE specific value. Instructs the library to
///       create memory object without an underlying buffer.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_l0_interop_memory_create(dnnl_memory_t *memory,
        const_dnnl_memory_desc_t memory_desc, dnnl_engine_t engine,
        void *handle);

/// Creates a memory object with multiple handles.
///
/// @param memory Output memory object.
/// @param memory_desc Memory descriptor.
/// @param engine Engine to use.
/// @param memory_kind Memory allocation kind to specify the type of handles.
/// @param nhandles Number of handles.
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
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_l0_interop_memory_create_v2(dnnl_memory_t *memory,
        const_dnnl_memory_desc_t memory_desc, dnnl_engine_t engine,
        size_t nhandles, void **handles);

/// Returns an Level Zero memory object associated with a memory object.
///
/// @param memory Memory object.
/// @param mem_object Output Level Zero memory object.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_l0_interop_memory_get_mem_object(
        const_dnnl_memory_t memory, void **mem_object);

/// Sets Level Zero memory object associated with a memory object.
///
/// For behavioral details, see dnnl_memory_set_data_handle().
///
/// @param memory Memory object.
/// @param mem_object Level Zero memory object.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_l0_interop_memory_set_mem_object(
        dnnl_memory_t memory, void *mem_object);

/// Executes computations specified by the primitive in a specified stream and
/// returns a Level Zero event.
///
/// @param primitive Primitive to execute.
/// @param stream Stream to use.
/// @param nargs Number of arguments.
/// @param args Array of arguments. Each argument is an
///     <index, #dnnl_memory_t> pair. The index is one of the `DNNL_ARG_*`
///     values such as `DNNL_ARG_SRC`. Unless runtime shapes are used (see
///     #DNNL_RUNTIME_DIM_VAL), the memory object must have the same memory
///     descriptor as that returned by
///     #dnnl_primitive_desc_query_md(#dnnl_query_exec_arg_md, index).
/// @param ndeps Number of dependencies.
/// @param deps A pointer to a vector of size @p ndeps that contains
///     dependencies.
/// @param return_event Output event.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_l0_interop_primitive_execute(
        const_dnnl_primitive_t primitive, dnnl_stream_t stream, size_t nargs,
        const dnnl_exec_arg_t *args, size_t ndeps,
        const ze_event_handle_t *deps, ze_event_handle_t *return_event);

/// @} dnnl_api_l0_interop

/// @} dnnl_api_interop

/// @} dnnl_api

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // ONEAPI_DNNL_DNNL_L0_H
