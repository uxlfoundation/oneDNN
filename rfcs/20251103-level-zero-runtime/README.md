# Support for Level Zero GPU runtime (RFC)

## Introduction

This RFC discusses the option of adding the Intel® oneAPI Level Zero (Level Zero) as fully supported oneDNN GPU runtime.

Level Zero is a fully featured API that provides direct-to-metal interfaces to offload Intel® accelerator devices (similarly to OpenCL™). As such, it is the preferred API for providing explicit device controls for runtime APIs and libraries using Intel® GPUs. For example, Level Zero is the default backend of SYCL runtime.

For oneDNN, Level Zero would provide quicker access to the newest features available for Intel® accelerators, including: reusable and mutable command lists, support for multi-device contexts, shared memory between accelerators, as well as lowered execution latency.

Using a modified [oneDNN example](https://github.com/uxlfoundation/oneDNN/blob/main/examples/cnn_inference_f32.cpp) and [POC implementation](https://github.com/uxlfoundation/oneDNN/tree/spalicki/l0_backend), the Level Zero runtime shows better performance than OpenCL™ and SYCL:

|           | Level Zero | OpenCL™ | SYCL with Level Zero | SYCL with OpenCL™ |
| --------- | ---------- | ------- | -------------------- | ----------------- |
| Min (ms): |        557 |     640 |                  599 |               658 |
| Avg (ms): |        579 |     668 |                  621 |               689 |

## Proposal

### Build option

Adding Level Zero runtime does not require new build options and only requires a new runtime `L0` in the existing CMake build option `ONEDNN_GPU_RUNTIME`, and it and assumes that: `ONEDNN_GPU_VENDOR=INTEL`.

### API/ABI change

Adding Level Zero as a new oneDNN runtime is fairly invisible to the end user and requires no API/ABI modification.

### Integration with SYCL runtime

Currently, the Level Zero loader is used when oneDNN is compiled with SYCL runtime and is using the Level Zero backend (similarly to OpenCL™ runtime when using the OpenCL™ backend), since not all required features are available directly in the SYCL API. The common code of SYCL with Level Zero and native Level Zero would be extracted as part of the new Level Zero runtime (to avoid redundancies) in the same way common OpenCL™ code is part of the OpenCL™ runtime.

### Integration with OpenCL™ runtime
To use OpenCL™ C kernels, there are several common functions that need to be extracted from oneDNN OpenCL™ runtime code and moved to `src/gpu/intel/compute/utils.hpp`:
```cpp
// Defined in src/gpu/intel/ocl/engine.hpp implemented in src/gpu/intel/ocl/engine.cpp
status_t preprocess_headers(stringstream_t &pp_code, const char *code, const compute::kernel_ctx_t &kernel_ctx);

// Defined and implemented in src/gpu/intel/ocl/utils.cpp
void debugdump_processed_source(const std::string &source, const std::string &options, const std::string &cl_options);
```

### Interop API

#### Engine creation
```cpp
/// Creates an engine associated with a Level Zero device and a Level Zero context.
///
/// @param engine Output engine.
/// @param driver Pointer to the Level Zero driver to use for the engine.
/// @param device Pointer to the Level Zero device to use for the engine.
/// @param context Pointer to the Level Zero context to use for the engine.
/// @returns #dnnl_success on success and a status describing the error otherwise.
dnnl_status_t DNNL_API dnnl_l0_interop_engine_create(dnnl_engine_t *engine, const ze_driver_handle_t adriver, const ze_device_handle_t adevice, const ze_context_handle_t acontext);
```
#### Getters for context, device and driver used by the engine
```cpp
/// Returns the Level Zero context associated with an engine.
///
/// @param engine Engine to query.
/// @param context Pointer to the underlying Level Zero context of the engine.
/// @returns #dnnl_success on success and a status describing the error otherwise.
dnnl_status_t DNNL_API dnnl_l0_interop_engine_get_context(dnnl_engine_t engine, ze_context_handle_t context);
```
```cpp
/// Returns the Level Zero device associated with an engine.
///
/// @param engine Engine to query.
/// @param device Pointer to the underlying Level Zero device of the engine.
/// @returns #dnnl_success on success and a status describing the error otherwise.
dnnl_status_t DNNL_API dnnl_l0_interop_engine_get_device(dnnl_engine_t engine, ze_device_handle_t device);
```
```cpp
/// Returns the Level Zero driver associated with an engine.
///
/// @param engine Engine to query.
/// @param device Pointer to the underlying Level Zero driver of the engine.
/// @returns #dnnl_success on success and a status describing the error otherwise.
dnnl_status_t DNNL_API dnnl_l0_interop_engine_get_driver(dnnl_engine_t engine, ze_driver_handle_t driver);
```
#### Stream creation
```cpp
/// Creates an execution stream for a given engine associated with a Level Zero queue.
///
/// @param stream Output execution stream.
/// @param engine Engine to create the execution stream on.
/// @param queue Level Zero command queue to use.
/// @param list Level Zero command list to use.
/// @returns #dnnl_success on success and a status describing the error otherwise.
dnnl_status_t DNNL_API dnnl_l0_interop_stream_create(dnnl_stream_t *stream, dnnl_engine_t engine, ze_command_list_handle_t list);
```

#### Getter for command list used by the stream
```cpp
/// Returns the Level Zero command list associated with an execution stream.
///
/// @param stream Execution stream to query.
/// @param list Output Level Zero command list.
/// @returns #dnnl_success on success and a status describing the error otherwise.
dnnl_status_t DNNL_API dnnl_l0_interop_stream_get_list(dnnl_stream_t stream, ze_command_list_handle_t list);
```

#### Memory object creation
```cpp
/// Creates a memory object.
///
/// Unless @p handle is equal to DNNL_MEMORY_NONE or DNNL_MEMORY_ALLOCATE, the constructed memory object will have the underlying buffer set. In this case, the buffer will be initialized as if dnnl_memory_set_data_handle() had been called.
///
/// @param memory Output memory object.
/// @param memory_desc Memory descriptor.
/// @param engine Engine to use.
/// @param handle Handle of the memory buffer to use as an underlying storage.
///     - A USM pointer to the user-allocated buffer. In this case the library doesn't own the buffer. Requires @p memory_kind to be equal to dnnl::l0_interop::memory_kind::usm.
///     - The DNNL_MEMORY_ALLOCATE special value. Instructs the library to allocate the buffer for the memory object. In this case the library owns the buffer.
///     - The DNNL_MEMORY_NONE specific value. Instructs the library to create memory object without an underlying buffer.
/// @returns #dnnl_success on success and a status describing the error otherwise.
dnnl_status_t DNNL_API dnnl_l0_interop_memory_create(dnnl_memory_t *memory, const_dnnl_memory_desc_t memory_desc, dnnl_engine_t engine, void *handle);

/// Creates a memory object with multiple handles.
///
/// @param memory Output memory object.
/// @param memory_desc Memory descriptor.
/// @param engine Engine to use.
/// @param memory_kind Memory allocation kind to specify the type of handles.
/// @param nhandles Number of handles.
/// @param handles Handles of the memory buffers to use as underlying storages.
///     For each element of the @p handles array the following applies:
///     - A pointer to the user-allocated buffer. In this case the library doesn't own the buffer. Requires @p memory_kind to be equal to dnnl::l0_interop::memory_kind::usm.
///     - The DNNL_MEMORY_ALLOCATE special value. Instructs the library to allocate the buffer for the memory object. In this case the library owns the buffer.
///     - The DNNL_MEMORY_NONE specific value. Instructs the library to create memory object without an underlying buffer.
/// @returns #dnnl_success on success and a status describing the error otherwise.
dnnl_status_t DNNL_API dnnl_l0_interop_memory_create_v2(dnnl_memory_t *memory, const_dnnl_memory_desc_t memory_desc, dnnl_engine_t engine, size_t nhandles, void **handles);
```

#### Getter and Setter for Level Zero memory object
```cpp
/// Returns an Level Zero memory object associated with a memory object.
///
/// @param memory Memory object.
/// @param mem_object Output Level Zero memory object.
/// @returns #dnnl_success on success and a status describing the error otherwise.
dnnl_status_t DNNL_API dnnl_l0_interop_memory_get_mem_object(const_dnnl_memory_t memory, void **mem_object);

/// Sets Level Zero memory object associated with a memory object.
///
/// For behavioral details, see dnnl_memory_set_data_handle().
///
/// @param memory Memory object.
/// @param mem_object Level Zero memory object.
/// @returns #dnnl_success on success and a status describing the error otherwise.
dnnl_status_t DNNL_API dnnl_l0_interop_memory_set_mem_object(dnnl_memory_t memory, void *mem_object);
```

#### Primitive exection
```cpp
/// Executes computations specified by the primitive in a specified stream and returns a Level Zero event.
///
/// @param primitive Primitive to execute.
/// @param stream Stream to use.
/// @param nargs Number of arguments.
/// @param args Array of arguments. Each argument is an <index, #dnnl_memory_t> pair. The index is one of the `DNNL_ARG_*` values such as `DNNL_ARG_SRC`. Unless runtime shapes are used (see #DNNL_RUNTIME_DIM_VAL), the memory object must have the same memory descriptor as that returned by #dnnl_primitive_desc_query_md(#dnnl_query_exec_arg_md, index).
/// @param ndeps Number of dependencies.
/// @param deps A pointer to a vector of size @p ndeps that contains dependencies.
/// @param return_event Output event.
/// @returns #dnnl_success on success and a status describing the error otherwise.
dnnl_status_t DNNL_API dnnl_l0_interop_primitive_execute(const_dnnl_primitive_t primitive, dnnl_stream_t stream, size_t nargs, const dnnl_exec_arg_t *args, size_t ndeps, const ze_event_handle_t *deps, ze_event_handle_t *return_event);
```

## Open Questions

### OpenCL™ C code compilation

Currently, Level Zero only supports native or SPIR-V code compilation, with support for OpenCL™ C code compilation to be added in a future version. To support oneDNN OpenCL™ kernels we would have to rely on one of:

1. Native Level Zero OpenCL™ C compiler - use `zeModuleCreate` to directly compile OpenCL™ C code.

This approach is preferable, since it does not add any external dependencies and no runtime overhead. The only downside is that it would limit our support to the newest Level Zero runtime releases with added OpenCL™ C compilation support

2. OpenCL™ runtime compiler - use `clBuildProgram` to create a native binary.

This approach would require oneDNN to load the OpenCL™ runtime, map Level Zero devices to OpenCL™, and use the OpenCL™ compiler to create the binary. This is the same path that oneDNN is currently using for SYCL with the Level Zero backend and what we are trying to remove to avoid having OpenCL™ runtime dependency when using SYCL with Level Zero.

3. OpenCL™ Offline Compiler (OCLOC) - invoke OCLOC API to compile OpenCL™ C code.

This approach would require oneDNN to load OCLOC and use it to create the binary. It is considerably simpler to use than loading the entire OpenCL™ runtime and mapping it to Level Zero as in approach 2, but it requires an extra ~200MB dependency. This is the approach that the Level Zero POC and SYCL offline compiler use and what approach 1 would use (with driver-packaged OCLOC).