Persistent Cache{#dev_guide_persistent_cache}
===========================================================

Creating oneDNN abstractions can be costly for various reasons.
Usually, oneDNN mitigates that overhead by caching such objects but
cache has no effect when the objects are created for the first time.
For some applications, reducing that overhead is critical.

oneDNN provides an API that can be used to create a persistent cache for
oneDNN abstractions. Through the API, users can obtain a cache blob ID
and a cache blob to use as a key and value respectively.

@note
Content and size of the cache blob ID and cache blob objects are not specified.

@note
The oneDNN version and git commit hash (@ref dnnl_version_t::hash) affect the
equality of the cache blob IDs. That is, the queried cache blob ID will differ
for different oneDNN versions and git commit hashes.

@warning
The git commit hash may not be available if the git package is not found during
a CMake call. In this case, the cache blob ID will be the same for different
hashes. This may result in fetching a wrong cache blob from persistent cache.

## Primitive

* The cache blob ID can be obtained via @ref dnnl::engine dnnl::primitive_desc_base::get_cache_blob_id
* The cache blob can be obtained via @ref dnnl::primitive::get_cache_blob
* Each primitive class provides a constructor that takes the cache blob along
with the primitive descriptor.

@note
Primitive creation from a cache blob fails with #dnnl_invalid_arguments when the
cache blob was created by a different oneDNN version or git commit hash, or for
a different device or driver version than the engine of the primitive
descriptor.

### Relation to Primitive Cache

When a primitive is created from a cache blob and the identical
primitive is present in the primitive cache, the one from primitive cache will
be returned to the user, and the given cache blob will not be used. Otherwise,
the cache blob will be used to speed up the primitive creation. The information
about how the primitive was created (`cache_miss`, `cache_hit` or
`persistent_cache_hit`) is part of the verbose output for `profile_create`
verbose level (@ref dev_guide_verbose).

### API Usage Example

The following pseudo-code demonstrates a simple example of persistent cache
implementation for primitives using the oneDNN API:

~~~cpp
using namespace dnnl;

{
    convolution_forward::primitive_desc conv_pd(desc, attr, engine);
    convolution_forward conv(conv_pd);

    std::vector<uint8_t> key = conv_pd.get_cache_blob_id();
    std::vector<uint8_t> value = conv.get_cache_blob();
    store_cache_blob_on_disk(key, value);
}

{
    convolution_forward::primitive_desc conv_pd(desc, attr, engine);
    std::vector<uint8_t> key = conv_pd.get_cache_blob_id();
    std::vector<uint8_t> value = load_cache_blob_from_disk(key);
    convolution_forward conv_from_cache_blob(conv_pd, value);
}
~~~

## Engine

@note
Engine creation from a cache blob fails with #dnnl_invalid_arguments when the
cache blob was created by a different oneDNN version or git commit hash, by an
engine of a different runtime, or under a different driver version than the one
the device currently reports.

### OpenCL GPU Runtime

* The cache blob ID can be obtained via @ref dnnl::ocl_interop::get_engine_cache_blob_id
* The cache blob can obtained via @ref dnnl::ocl_interop::get_engine_cache_blob
* The engine can be created with the cache blob via @ref dnnl::ocl_interop::make_engine(cl_device_id, cl_context, const std::vector<uint8_t> &)

### Level Zero GPU Runtime

* The cache blob ID can be obtained via @ref dnnl::ze_interop::get_engine_cache_blob_id
* The cache blob can obtained via @ref dnnl::ze_interop::get_engine_cache_blob
* The engine can be created with the cache blob via @ref dnnl::ze_interop::make_engine(ze_driver_handle_t, ze_device_handle_t, ze_context_handle_t, const std::vector<uint8_t> &)

### API Usage Example (OpenCL GPU Runtime)

The following pseudo-code demonstrates a simple example of persistent cache
implementation for OpenCL engines using the oneDNN API:

~~~cpp
using namespace dnnl;

{
    cl_device_id device = ...;
    cl_context context = ...;

    engine ocl_engine = ocl_interop::make_engine(device, context);
    std::vector<uint8_t> key = get_engine_cache_blob_id(ocl_interop::get_device(ocl_engine));
    std::vector<uint8_t> value = get_engine_cache_blob(ocl_engine);

    store_cache_blob_on_disk(key, value);
}

{
    cl_device_id device = ...;
    cl_context context = ...;

    std::vector<uint8_t> key = get_engine_cache_blob_id(device);
    std::vector<uint8_t> value = load_cache_blob_from_disk(key);
    engine ocl_engine = ocl_interop::make_engine(device, context, value);
}
~~~

## Device-less Engine

An application that has no access to the GPU driver in the process that
compiles a model can still populate a persistent cache with a device-less
engine. Such an engine is created from an engine cache blob and generates
primitive cache blobs without making any runtime calls:

1. On the target machine, export the engine cache blob from a real engine.
2. In the compilation process, create a device-less engine from that blob via
@ref dnnl::ocl_interop::make_deviceless_engine or
@ref dnnl::ze_interop::make_deviceless_engine, create primitive descriptors and
primitives on it, and store their cache blob IDs and cache blobs.
3. On the same machine, create a real engine of the same runtime from the
device, the context, and the engine cache blob via
@ref dnnl::ocl_interop::make_engine(cl_device_id, cl_context, const std::vector<uint8_t> &)
or
@ref dnnl::ze_interop::make_engine(ze_driver_handle_t, ze_device_handle_t, ze_context_handle_t, const std::vector<uint8_t> &),
and create primitives on it from the stored cache blobs.

The cache blob ID of a primitive descriptor is the same on the device-less and
the real engines, so the cache blobs are found by the same key.

@note
A device-less engine is created only from an engine cache blob exported by an
engine of the same runtime; otherwise, the creation fails with
#dnnl_invalid_arguments. A device-less engine requires the offline compiler
shipped with the GPU driver; when it is missing, the creation fails with
#dnnl_runtime_error.

A device-less engine supports primitive descriptor and primitive creation and
primitive cache blob retrieval only. Primitives can't be executed, and memory
objects and streams can't be created on it. Interop getters and interop memory
creation return #dnnl_invalid_arguments for it. Primitive creation from a cache
blob on a device-less engine is not supported.

### API Usage Example (OpenCL GPU Runtime)

~~~cpp
using namespace dnnl;

// On the target machine.
{
    engine ocl_engine = ocl_interop::make_engine(device, context);
    store_engine_blob_on_disk(ocl_interop::get_engine_cache_blob(ocl_engine));
}

// In a process without driver access.
{
    engine dl_engine = ocl_interop::make_deviceless_engine(
            load_engine_blob_from_disk());
    convolution_forward::primitive_desc conv_pd(dl_engine, ...);
    convolution_forward conv(conv_pd);
    store_cache_blob_on_disk(conv_pd.get_cache_blob_id(), conv.get_cache_blob());
}

// On the target machine.
{
    engine ocl_engine = ocl_interop::make_engine(
            device, context, load_engine_blob_from_disk());
    convolution_forward::primitive_desc conv_pd(ocl_engine, ...);
    std::vector<uint8_t> value
            = load_cache_blob_from_disk(conv_pd.get_cache_blob_id());
    convolution_forward conv(conv_pd, value);
}
~~~

## Memory descriptor

When serializing primitives, a binary blob can be obtained from a
memory descriptor using @ref dnnl::memory::desc::get_blob. Any binary
blob obtained from @ref dnnl::memory::desc::get_blob can be used to
create a memory descriptor @ref dnnl::memory::desc.

@note
When deserializing a constant tensor, the user must verify that the deserialized
memory descriptor matches the memory descriptor expected by the primitive that
will use that memory. The only circumstance where both are guaranteed to match
is when serialization/deserialization happens on the same system and in the same
environment.

## Limitations

* The primitive and engine APIs are implemented for OpenCL and Level Zero
runtimes only. For CPU engine and other runtimes, the library will return
#dnnl_unimplemented (in the case of the C API) or throw a corresponding
@ref dnnl::error exception (in the case of the C++ API).
* Currently, the library cannot differentiate cache blobs created for devices
that have different stepping; therefore, the cache blob can be safely used only
on the system where it is created.
