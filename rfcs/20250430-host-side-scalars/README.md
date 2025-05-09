
Currently, primitives assume that all inputs passed at execute time
are on the same device as the one they execute on (with the exception
of reorder).

Unfortunately, this is no more true as some users want to pass scalar
values that reside on host device to gpu kernels at execution time.  A
good example is the scale for SDPA operator in Pytorch [^1], which is
a python variable on the host and not a torch.tensor object.

There are two challenges here:
- How to specify to primitive descritor creation that a given input is
  a host scalar? Currently, we don't have any engine specified to
  input memory descriptors.
- How to pass host-side scalars to the execute function? Currently, we
  only take dnnl::memory objects.

## How to specify an input is a host-side scalar value to primitive desc creation?

### Option 1: do nothing and take host-side scalars as primitive descriptor parameters

Here user would pass the host side scalar as a parameter to primitive
creation, as is done today.  The main caveats here are:
- it requires user to recreate primitive descriptors and primitives
  everytime the host-side scalar changes value
- if the implementation jits a kernel that depends on that parameter,
  we can get large overheads due to re-jitting.

Pros:
- oneDNN API can already handle this for parameters that do not
frequently change.

Cons:
- forces user to recreate primitive descriptor and primitive objects
  every-time that host-side scalar value changes.
- potential extra jitting overhead if internal implementation considers that parameter constant.

This option is not recommended if the host-side scalar value is
expected to frequently change.


### Option 2 (recommended): expose a new memory descriptor kind

The idea here is to:
- expose host-side scalars through a new  memory descriptor create function
- have user pass a memory with cpu engine to execute function.

The new kind of memory descriptor would be created through a dedicated function
It would be very lightweights and be limited:
- only scalar value, so no `ndims`, no `dims`, ...
- only scalar datatype can be configurable by user.

```C
dnnl_status_t DNNL_API dnnl_memory_desc_create_host_scalar(
        dnnl_memory_desc_t *memory_desc, dnnl_data_type_t data_type);
```

Internal considerations: we would use a new `format_kind` with no
associated `format_desc`. This would avoid aliasing host-side scalars
with blocked descriptors, and hence host-side scalars will need to be
explicitely supported by implementations.

For users passing primitive objects around and querying them, it makes
sense to expose this format kind, so users can query if a given input
is a host-side scalar, and pass it accordingly to primitive execution.


Pros:
- avoids user from recreating primitives when scalars are changed
- simple enabling internally

Cons:
- ties the memory descriptor to a specific device (host) which we
  avoided so far.


### Option 3: explicitely take engine kind in memory desc.

Here, we would allow users to pass any memory on host or device.
This would officially tie a given memory descritor to a device kind.

For compatibility reasons, we could either use `undef` to specify that
a memory desc will be ties to the same engine kind as the
implementation, or introduce a new `engine_kind` for that purpose.

A few points need to be made:
- mixing host and device memory is already possible through usm shared
  memory for both sycl and opencl runtime.
- there is no clear benefit to mix host and device memory, except for
  the case where the value to pass is a scalar, in which case it can
  be passed to device kernel as a parameter (instead of passing a
  pointer).
- even if two memories share the same engine kind, it does not
  guarentee that they will both be accessible from the same context
  (they might be tied to different engine objects).

As a result, we don't recommend to specify engine kind for memory
descriptors as it would likely provide little benefit over host-side
scalar specific memory descriptor, but being more general, it would
also come with more complications.

Pros:
- clear semantic: all memory descritors are tied to an engine kind

Cons:
- full buffers sharing across devices can already be achieved through runtime mecanisms.
- no clear benefit for performance other than specific scalar case.

## How should host-side scalar be passed to execute call?

Currently, the oneDNN primitive API only accepts `dnnl::memory` objects
(through an `unordered_map` in C++ interface, or an array of
`arg_kind`/`memory` for C interface).

We have two paths:
- encapsulate host-side scalars in memory objects: this would create
  overhead for passing a scalar but would not require execution API
  changes. There is a question of object lifetime to clarify here: for
  host-side scalar, we have to guarentee that they are not accessed
  asynchronously, otherwise users will incur extra overhead of
  allocation and copy for a single scalar.
- expose new execute function to allow passing scalars. For both C and
  C++ APIs, it would require to take extra unordered_map for scalar
  arguments.


Even though the first option has unnecessary overhead, we recommend to
use it to align scalar and non-scalar passing.

## Recap

| option                                       | Pros                                                                              | Cons                                                                                                                 |
|----------------------------------------------|-----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| 1. primitive_desc construction time constant | - oneDNN API can already handle this for parameters that do not frequently change | - forces user to recreate primitive descriptor and primitive objects every-time that host-side scalar value changes. |
|                                              |                                                                                   | - potential extra jitting overhead if internal implementation considers that parameter constant.                     |
|----------------------------------------------|-----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| 2. new memory descriptor kind                | - avoids user from recreating primitives when scalars are changed                 | - ties the memory descriptor to a specific device (host) which we  avoided so far.                                   |
|                                              | - simple enabling internally                                                      |                                                                                                                      |
|----------------------------------------------|-----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| 3. add `engine_kind` to memory desc          | clear semantic: all memory descritors are tied to an engine kind                  | - full buffers sharing across devices can already be achieved through runtime mecanisms.                             |
|                                              |                                                                                   | - no clear benefit for performance other than specific scalar case.                                                  |
|                                              |                                                                                   | - ties the memory descriptor to a specific device (host) which we  avoided so far.                                   |
|----------------------------------------------|-----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|

The recommendation is to go with option 2: add APIs to support
host-side scalars through a new memory descriptor creation function,
and user would encapsulate those in a memory object to pass them at
execution time.

A POC branch is available [^2].

## References

[^1]: [SDPA operator in PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
[^2]: [POC branch for host-side scalar memory descriptors](https://github.com/uxlfoundation/oneDNN/tree/mgouicem/main/host_scalar)
