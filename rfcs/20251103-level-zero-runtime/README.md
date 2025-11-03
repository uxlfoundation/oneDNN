# Support for Level Zero GPU runtime (RFC)

## Introduction

This RFC discusses the option of adding the Intel® oneAPI Level Zero
(Level Zero) runtime as a fully supported oneDNN GPU runtime.

Level Zero is a fully featured API that provides direct-to-metal interfaces to
offload work onto Intel® accelerator devices (similarly to OpenCL™). As such, it
is the preferred API for providing explicit device controls to runtime APIs and
libraries that use Intel® GPUs. For example, Level Zero is the default backend
of the SYCL runtime.

For oneDNN, Level Zero provides quicker access to the newest features available
for Intel® accelerators, including reusable command lists, support for
multi-device contexts, shared memory between accelerators, and lower execution
latency.

Using a modified [oneDNN example](https://github.com/uxlfoundation/oneDNN/blob/main/examples/cnn_inference_f32.cpp)
and [POC implementation](https://github.com/uxlfoundation/oneDNN/tree/spalicki/l0_backend),
the Level Zero runtime shows better performance than OpenCL™ and SYCL (as of
November 2025):

|           | Level Zero | OpenCL™ | SYCL with Level Zero | SYCL with OpenCL™ |
| --------- | ---------- | ------- | -------------------- | ----------------- |
| Min (ms): |        557 |     640 |                  599 |               658 |
| Avg (ms): |        579 |     668 |                  621 |               689 |

## Proposal

### Build option

Adding the Level Zero runtime does not require any new build options. It only
requires a new runtime value `ZE` in the existing CMake build option
`ONEDNN_GPU_RUNTIME`, which is compatible only with `ONEDNN_GPU_VENDOR=INTEL`
(the default).

### External API/ABI change

Using Level Zero as a new oneDNN runtime through the API that manages runtime
resources on behalf of the user is the same as for other runtimes and does not
require any modifications. Using the Level Zero interoperability API requires
application-side modifications to manage the new runtime resources.

### Interop API

#### Engine creation

```cpp
/// Creates an engine associated with a Level Zero device and a Level Zero
/// context.
///
/// @param engine Output engine.
/// @param driver Pointer to the Level Zero driver to use for the engine.
/// @param device Pointer to the Level Zero device to use for the engine.
/// @param context Pointer to the Level Zero context to use for the engine.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ze_interop_engine_create(dnnl_engine_t *engine,
        ze_driver_handle_t driver, ze_device_handle_t device,
        ze_context_handle_t context);
```

#### Engine creation from a cache blob

To reduce engine creation time, oneDNN can persist its device-specific state
(such as compiled kernels) in a cache blob that the user stores and later
replays when recreating the engine.

```cpp
/// Retrieves a cache blob ID for the Level Zero device.
///
/// @warning
///     This API is intended to be used with
///     #dnnl_ze_interop_engine_get_cache_blob() and
///     #dnnl_ze_interop_engine_create_from_cache_blob(). The returned cache
///     blob ID can only be used as an ID of the cache blob returned by
///     #dnnl_ze_interop_engine_get_cache_blob().
///
/// @note The cache blob ID can be empty (@p size will be 0 and
///     @p cache_blob_id will be nullptr) if oneDNN doesn't have anything to
///     put in the cache blob. (#dnnl_ze_interop_engine_get_cache_blob will
///     return an empty cache blob).
///
/// @param driver A Level Zero driver.
/// @param device A Level Zero device.
/// @param size Size of the cache blob ID in bytes.
/// @param cache_blob_id Cache blob id of size @p size. If the @p cache_blob_id
///     is nullptr then the size of the cache blob ID is returned in @p size.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ze_interop_engine_get_cache_blob_id(
        ze_driver_handle_t driver, ze_device_handle_t device, size_t *size,
        uint8_t *cache_blob_id);

/// Retrieves a cache blob associated with the given engine.
///
/// @note The cache blob can be empty (@p size will be 0 and @p cache_blob
///     will be nullptr) if oneDNN doesn't have anything to put in the cache
///     blob. It's the user's responsibility to check whether it's empty prior
///     to passing it to #dnnl_ze_interop_engine_create_from_cache_blob().
///
/// @param engine Engine to query for the cache blob.
/// @param size Size of the cache blob in bytes.
/// @param cache_blob Cache blob of size @p size. If the @p cache_blob is
///     nullptr then the size of the cache blob is returned in @p size.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ze_interop_engine_get_cache_blob(
        dnnl_engine_t engine, size_t *size, uint8_t *cache_blob);

/// Creates an engine from the given cache blob.
///
/// @param engine Output engine.
/// @param driver The Level Zero driver that this engine will encapsulate.
/// @param device The Level Zero device that this engine will encapsulate.
/// @param context The Level Zero context (containing the device) that this
///     engine will use for all operations.
/// @param size Size of the cache blob in bytes.
/// @param cache_blob Cache blob of size @p size.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ze_interop_engine_create_from_cache_blob(
        dnnl_engine_t *engine, ze_driver_handle_t driver,
        ze_device_handle_t device, ze_context_handle_t context, size_t size,
        const uint8_t *cache_blob);
```

#### Getters for context, device and driver used by the engine

```cpp
/// Returns the Level Zero context associated with an engine.
///
/// @param engine Engine to query.
/// @param context Pointer to the underlying Level Zero context of the engine.
/// @returns #dnnl_success on success and a status describing the error otherwise.
dnnl_status_t DNNL_API dnnl_ze_interop_engine_get_context(dnnl_engine_t engine,
        ze_context_handle_t *context);
```

```cpp
/// Returns the Level Zero device associated with an engine.
///
/// @param engine Engine to query.
/// @param device Pointer to the underlying Level Zero device of the engine.
/// @returns #dnnl_success on success and a status describing the error otherwise.
dnnl_status_t DNNL_API dnnl_ze_interop_engine_get_device(dnnl_engine_t engine,
        ze_device_handle_t *device);
```

```cpp
/// Returns the Level Zero driver associated with an engine.
///
/// @param engine Engine to query.
/// @param driver Pointer to the underlying Level Zero driver of the engine.
/// @returns #dnnl_success on success and a status describing the error otherwise.
dnnl_status_t DNNL_API dnnl_ze_interop_engine_get_driver(dnnl_engine_t engine,
        ze_driver_handle_t *driver);
```

#### Stream creation

```cpp
/// Creates an execution stream for a given engine associated with a Level Zero
/// command list.
///
/// @param stream Output execution stream.
/// @param engine Engine to create the execution stream on.
/// @param list Level Zero immediate command list to use.
/// @param profiling Flag enabling GPU kernels profiling.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ze_interop_stream_create(dnnl_stream_t *stream,
        dnnl_engine_t engine, ze_command_list_handle_t list, int profiling);
```

#### Getter for command list used by the stream

```cpp
/// Returns the Level Zero command list associated with an execution stream.
///
/// @param stream Execution stream to query.
/// @param list Output Level Zero command list.
/// @returns #dnnl_success on success and a status describing the error otherwise.
dnnl_status_t DNNL_API dnnl_ze_interop_stream_get_list(dnnl_stream_t stream,
        ze_command_list_handle_t *list);
```

#### Memory object creation

```cpp
/// Creates a memory object.
///
/// Unless @p handle is equal to DNNL_MEMORY_NONE or DNNL_MEMORY_ALLOCATE, the
///     constructed memory object will have the underlying buffer set.
///     In this case, the buffer will be initialized as if
///     dnnl_memory_set_data_handle() had been called.
///
/// @param memory Output memory object.
/// @param memory_desc Memory descriptor.
/// @param engine Engine to use.
/// @param nhandles Number of handles.
/// @param handles Handles of the memory buffers to use as underlying storages.
///     - A USM pointer to the user-allocated buffer. In this case the library
///           doesn't own the buffer.
///     - The DNNL_MEMORY_ALLOCATE special value. Instructs the library to
///           allocate the buffer for the memory object. In this case the
///           library owns the buffer.
///     - The DNNL_MEMORY_NONE specific value. Instructs the library to create
///           memory object without an underlying buffer.
/// @returns #dnnl_success on success and a status describing the error otherwise.
dnnl_status_t DNNL_API dnnl_ze_interop_memory_create(dnnl_memory_t *memory,
        const_dnnl_memory_desc_t memory_desc, dnnl_engine_t engine,
        int nhandles, void **handles);
```

#### Primitive execution

```cpp
/// Executes computations specified by the primitive in a specified stream and
///     returns a Level Zero event.
///
/// @param primitive Primitive to execute.
/// @param stream Stream to use.
/// @param nargs Number of arguments.
/// @param args Array of arguments. Each argument is an <index, #dnnl_memory_t>
///     pair. The index is one of the `DNNL_ARG_*` values such as
///     `DNNL_ARG_SRC`. Unless runtime shapes are used
///     (see #DNNL_RUNTIME_DIM_VAL), the memory object must have the same memory
///     descriptor as that returned by
///     #dnnl_primitive_desc_query_md(#dnnl_query_exec_arg_md, index).
/// @param ndeps Number of dependencies.
/// @param deps A pointer to a vector of size @p ndeps that contains
///     dependencies.
/// @param return_event Output event.
/// @returns #dnnl_success on success and a status describing the error otherwise.
dnnl_status_t DNNL_API dnnl_ze_interop_primitive_execute(
        const_dnnl_primitive_t primitive, dnnl_stream_t stream, int nargs,
        const dnnl_exec_arg_t *args, int ndeps, const ze_event_handle_t *deps,
        ze_event_handle_t *return_event);
```

### Internal changes

#### Integration with SYCL runtime

The Level Zero loader is already used when oneDNN is compiled with the SYCL
runtime and the Level Zero backend (similarly to how the OpenCL™ runtime is used
with the OpenCL™ backend), since not all required features are available
directly through the SYCL API. The code common to SYCL-with-Level-Zero and
native Level Zero is extracted into the new Level Zero runtime (to avoid
redundancy), in the same way the common OpenCL™ code lives in the OpenCL™
runtime.

#### Integration with OpenCL™ runtime

To use OpenCL™ C kernels, several common functions are extracted from the oneDNN
OpenCL™ runtime code and moved to `src/gpu/intel/compute/utils.hpp`:
```cpp
// Defined in src/gpu/intel/ocl/engine.hpp,
// implemented in src/gpu/intel/ocl/engine.cpp
status_t preprocess_headers(stringstream_t &pp_code, const char *code,
        const compute::kernel_ctx_t &kernel_ctx);

// Defined and implemented in src/gpu/intel/ocl/utils.cpp
void debugdump_processed_source(const std::string &source,
        const std::string &options, const std::string &cl_options);
```

## Resolved Questions

### OpenCL™ C code compilation

At the time of writing, Level Zero only supported native or SPIR-V code
compilation, with OpenCL™ C code compilation planned for a future release. To
support oneDNN OpenCL™ kernels, the following options were considered:

1. Native Level Zero OpenCL™ C compiler - use `zeModuleCreate` to directly
   compile OpenCL™ C code.

   This approach is preferable, since it does not add any external dependencies
   and incurs no runtime overhead. The downside is that only a Level Zero
   runtime release with OpenCL™ C compilation support (a future release) can be
   used; earlier versions cannot.

2. OpenCL™ runtime compiler - use `clBuildProgram` to create a native binary.

   This approach requires oneDNN to load the OpenCL™ runtime, map Level Zero
   devices to OpenCL™, and use the OpenCL™ compiler to create the binary. This
   is the path oneDNN currently uses for SYCL with the Level Zero backend. It is
   undesirable due to the unnecessary OpenCL™ runtime dependency when using SYCL
   with Level Zero and the added complexity of device mapping.

3. OpenCL™ Offline Compiler (OCLOC) - invoke the OCLOC API to compile OpenCL™ C
   code.

   This approach requires oneDNN to load OCLOC and use it to create the binary.
   It is considerably simpler than loading the entire OpenCL™ runtime and
   mapping it to Level Zero as in approach 2, but it requires an extra ~200MB
   dependency. This is the approach used by the Level Zero POC and the SYCL
   offline compiler, and the approach that option 1 relies on internally (with
   the driver-packaged OCLOC).

**Resolution:** Approach 1 was selected and productized. oneDNN compiles
OpenCL™ C kernels directly through `zeModuleCreate` using the `ZE_MODULE_FORMAT_OCLC`
module format, avoiding any additional external dependencies or runtime
overhead.
