# Device-less GPU Engine for Offline Kernel Compilation

## Introduction

A oneDNN GPU engine is created from a runtime device and context. Creating
primitive descriptors and primitives on it requires a working driver: the
engine queries the device properties, and kernels are built and loaded by the
runtime.

This RFC proposes a device-less GPU engine. It is created from an engine
cache blob instead of a device and a context, and makes no runtime calls. It
can be used to create primitive descriptors and primitives and to retrieve
primitive cache blobs. Those blobs are later used to create primitives on a
real engine without compiling any kernels.

The feature builds on the existing
[persistent cache API](https://uxlfoundation.github.io/oneDNN/dev_guide_persistent_cache.html):
engine cache blobs, primitive cache blob IDs, and primitive cache blobs.

The proposal is implemented in
[PR #6485](https://github.com/uxlfoundation/oneDNN/pull/6485).

## Motivation

The request comes from the OpenVINO GPU plugin on behalf of WebNN in Chromium.
For security reasons, WebNN splits model compilation into separate stages that
run in different processes with different privileges:

1. **Device enumeration.** A privileged process with driver access collects
   information about the target GPU.
2. **Model compilation.** A sandboxed process without driver access compiles
   the model. For oneDNN, this means creating primitive descriptors, to query
   memory layouts, and primitives, to generate kernel binaries. Calls such as
   `clBuildProgram()` or `clCreateKernel()` are not available.
3. **Model import and execution.** A process with driver access loads the
   compiled model and executes it. Kernel compilation is not allowed at this
   stage. For models with static shapes, all kernel binaries must already be
   in the compiled model.

All three stages run on the same machine, using the same driver installation.

oneDNN supports neither stage 2 nor stage 3 without a device today. The
OpenVINO team built a proof of concept on top of a local oneDNN patch and
validated the flow end to end on Intel Arc A770, Intel Arc B70, and Intel Core
Ultra (Panther Lake H) GPUs with 26 WebNN models. Results were bit-exact with
online compilation, on both OpenCL and Level Zero runtimes.

The feature targets the OpenVINO 2027.0 release, which picks up oneDNN v3.16.

A similar capability exists in cuDNN as
[device-less ahead-of-time compilation](https://docs.nvidia.com/deeplearning/cudnn/v1.28.0/utilities/deviceless-ahead-of-time-compilation.html).
There, a serialized device property descriptor is created with the device,
an execution plan is built and serialized without it, and the plan is
deserialized and executed with the device.

## Proposal

### Flow

The flow maps directly onto the WebNN stages:

1. On the target machine, a real engine is created and its engine cache blob is
   exported with `{ocl,ze}_interop::get_engine_cache_blob()`.
2. In the process without driver access, a device-less engine is created from
   the engine cache blob. Primitive descriptors and primitives are created on
   it, and primitive cache blob IDs and primitive cache blobs are stored.
3. On the same machine, a real engine is created from the device, the context,
   and the engine cache blob. Primitives are created from the stored primitive
   cache blobs found by the cache blob IDs of their primitive descriptors.

The cache blob ID of a primitive descriptor is identical on the device-less and
the real engines, so no extra mapping is needed between steps 2 and 3.

### API

The C API adds one function per runtime:

```c
// dnnl_ocl.h
dnnl_status_t DNNL_API dnnl_ocl_interop_deviceless_engine_create(
        dnnl_engine_t *engine, size_t size, const uint8_t *cache_blob);

// dnnl_ze.h
dnnl_status_t DNNL_API dnnl_ze_interop_deviceless_engine_create(
        dnnl_engine_t *engine, size_t size, const uint8_t *cache_blob);
```

The C++ API adds the corresponding wrappers:

```cpp
// dnnl_ocl.hpp
namespace dnnl::ocl_interop {
engine make_deviceless_engine(const std::vector<uint8_t> &cache_blob);
}

// dnnl_ze.hpp
namespace dnnl::ze_interop {
engine make_deviceless_engine(const std::vector<uint8_t> &cache_blob);
}
```

The API is an extension only. Existing API and ABI are not changed.

### Semantics

A device-less engine:

* Is a GPU engine with the runtime kind of the API used to create it. The
  engine cache blob must be exported by an engine of the same runtime;
  otherwise, the creation fails with `dnnl_invalid_arguments`.
* Takes all device properties from the engine cache blob. The blob carries the
  oneDNN version, which is verified before anything else is read, so a blob
  from a different oneDNN build is rejected with `dnnl_invalid_arguments`.
* Requires the offline compiler shipped with the GPU driver. When it is
  missing, the creation fails with `dnnl_runtime_error`.
* Supports primitive descriptor creation, primitive creation, and primitive
  cache blob retrieval.
* Doesn't support primitive execution, stream creation, or memory creation.
  Interop getters, such as `ocl_interop::get_device()`, and interop memory
  creation return `dnnl_invalid_arguments`.
* Doesn't support primitive creation from a primitive cache blob, since the
  flow never needs it on the device-less side.
* Is reported as `gpu:deviceless` in the verbose output, so its primitives are
  not mistaken for ones created on a device.

### Usage Example

```cpp
using namespace dnnl;

// Stage 1, on the target machine.
{
    engine eng = ocl_interop::make_engine(device, context);
    store_on_disk(ocl_interop::get_engine_cache_blob(eng));
}

// Stage 2, in a process without driver access.
{
    engine eng = ocl_interop::make_deviceless_engine(load_engine_blob());
    matmul::primitive_desc pd(eng, src_md, wei_md, dst_md);
    store_on_disk(pd.get_cache_blob_id(), matmul(pd).get_cache_blob());
}

// Stage 3, on the target machine.
{
    engine eng = ocl_interop::make_engine(device, context, load_engine_blob());
    matmul::primitive_desc pd(eng, src_md, wei_md, dst_md);
    matmul prim(pd, load_primitive_blob(pd.get_cache_blob_id()));
}
```

### Design Choices

#### Engine Cache Blob as the Device Description

The device-less engine is created from an engine cache blob, which the
persistent cache API already provides. The blob carries the complete internal
device description, so primitives on a device-less engine dispatch the same
implementations and generate the same kernels as on the real engine. The
missing blob checks needed for this use (a header with the oneDNN version and
the runtime kind, and strict parsing) are added as part of the work.

Alternatives considered:

* **A public device properties object.** Expose the internal device
  description as an opaque object with its own serialization, similar to the
  cuDNN device property descriptor.
  * Pros: an explicit object that may be extended with queries.
  * Cons: a new public object type with its own lifetime, versioning, and
    validation, which duplicates what the engine cache blob already provides.
* **A user-provided device description.** Let the user describe the device
  with a set of properties, such as the platform and the number of EUs.
  * Pros: no real device is needed at any stage.
  * Cons: any property that the user omits or gets wrong leads to a different
    dispatch or to incorrect kernels on the target device. Some properties are
    critical for correctness. The set of properties oneDNN relies on is an
    implementation detail that changes over time.

#### Placement in the Interop Namespaces

`make_deviceless_engine()` doesn't take any runtime object, so it could be
placed in the common API. It is placed in `ocl_interop` and `ze_interop`
instead, for the following reasons:

* **Pairing with the blob producer.** The engine cache blob is produced only by
  `{ocl,ze}_interop::get_engine_cache_blob()`. Placing the consumer next to the
  producer keeps the flow within a single namespace, the same as for
  `{ocl,ze}_interop::make_engine()` overloads that take a cache blob.
* **The runtime is part of the result.** Primitive cache blob IDs depend on the
  runtime kind and on the runtime-specific device serialization, and primitive
  cache blobs contain runtime-specific kernel binaries. A device-less engine
  must therefore have the same runtime as the real engine it stands in for.
  Choosing the namespace states the runtime explicitly, and a blob from a
  different runtime is rejected instead of silently producing blobs that would
  never be found on the target engine.
* **No new header dependency for users.** Users of this flow already include
  the interop header to export the engine cache blob in stage 1 and to create
  the real engine in stage 3.
* **Scope.** The persistent cache API exists only for the OpenCL and Level Zero
  runtimes. A common API would need to report `dnnl_unimplemented` for the CPU
  engine and for other runtimes, including SYCL.

The common API alternative is
`dnnl_engine_create_deviceless(dnnl_engine_t *, dnnl_engine_kind_t, size_t,
const uint8_t *)` with the runtime taken from the blob.

* Pros: no interop header is needed in the process without driver access.
* Cons: it breaks the symmetry with the blob producer, hides the runtime
  choice, and adds a function to the common API that only works for two GPU
  runtimes.

### Implementation Notes

* The device-less engine is a separate GPU engine implementation that doesn't
  call any OpenCL or Level Zero function. Reusing the runtime engines with
  stubbed runtime calls would make it easy to miss a call, which would only be
  discovered in the target environment.
* nGEN kernels are generated as usual, since nGEN produces binaries without the
  runtime.
* OpenCL C kernels are compiled with the offline compiler library (`ocloc`),
  which is a part of the GPU runtime package. The target is the device IP
  version recorded in the engine cache blob. Since all stages run on the same
  machine with the same driver installation, the compiler version matches the
  driver version recorded in the blob.
* Kernels generated by the DSL IR are not supported yet.

### Validation

#### benchdnn

A new `D` value of the `--mode-modifier` option creates each tested primitive
on a device-less engine as well. The device-less engine is made from the cache
blob of the test engine. The case fails with a "Device-less Engine Error" when:

* The primitive can't be created on the device-less engine, or the
  implementation name differs. This verifies that the device-less engine
  dispatches the same implementation.
* The primitive cache blob ID differs. This verifies that a blob exported from
  the device-less engine is found on the real engine.
* The primitive cache blob differs. This verifies that the same implementation
  generates the same kernels. Equal kernels guarantee that execution on the
  real engine is bit-exact with online compilation, so no separate correctness
  run is needed.

The check is opt-in because it adds one more compilation of every OpenCL C
kernel, which significantly increases the testing time. A new `ADD_DEVICELESS`
value of the `ONEDNN_TEST_SET` build option enables it for the GPU benchdnn
testing targets. It is intended to target weekly testing with ability to trigger
it in pre-commit testing. The graph driver doesn't support the modifier.

#### gtests

`test_persistent_cache_api` covers:

* The whole flow for eltwise, matmul, and a grouped matmul implementation that
  uses microkernels: implementation names and primitive cache blobs are equal
  on the device-less and real engines, and primitives imported from the blobs
  on the real engine compute the same results.
* Rejection of malformed blobs and blobs from another runtime.
* Rejection of stream creation, memory creation, interop getters, and interop
  memory creation.

The tests are skipped when the offline compiler is not available.

## Limitations

* **Kernel generation on primitive descriptor creation.** Some
  implementations, such as GEMM-based ones, generate kernels during primitive
  descriptor creation. Recreating such a primitive descriptor in stage 3
  generates kernels again, even though the primitive is then created from a
  cache blob. This violates the WebNN requirement of no kernel compilation at
  import and blocks a fully compliant flow. Moving kernel generation from
  primitive descriptor creation to primitive creation is tracked separately
  and targets oneDNN v3.16.
* **DSL IR kernels.** Implementations with kernels generated by the DSL IR
  can't be created on a device-less engine. There is no motivation to support
  them until such kernels are used in OpenVINO production, which would likely
  come with future hardware.
* **SYCL runtime.** The persistent cache API isn't available for the SYCL
  runtime, so the device-less engine isn't either. This aligns with the
  request: OpenVINO uses the OpenCL and Level Zero runtimes for this flow.
