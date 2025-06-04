# Host-side scalars Support in oneDNN Primitives

## Motivation

Currently, primitives assume that all inputs passed at execute time
are on the same device as the one they execute on (with the exception
of reorder).

A new request is to pass scalar values that reside on host device
to GPU kernels at execution time.
A good example is the scale for SDPA operator in Pytorch [^1], which is
a python variable on the host and not a torch.tensor object.

There are two challenges here:
- How to specify to primitive descriptor creation that a given input is
  a host scalar?
    - Currently, we don't have any engine specified to input memory descriptors.
- How to pass host-side scalars to the execute function?
  - Currently, we only take dnnl::memory objects.

## How to specify an input as a host-side scalar value to primitive desc creation?

### Option 1: do nothing and take host-side scalars as primitive descriptor parameters

Here user would pass the host side scalar as a parameter to primitive
creation, as is done today. The main caveats here are:
- It requires user to recreate primitive descriptors and primitives
  every time the host-side scalar changes value.
- If the implementation jits a kernel that depends on that parameter,
  we can get large overheads due to re-jitting.

Pros:
- oneDNN API can already handle this for parameters that do not
frequently change.

Cons:
- Forces user to recreate primitive descriptor and primitive objects
  every-time that host-side scalar value changes.
- Potential extra jitting overhead if internal implementation considers that parameter constant.

This option is not recommended if the host-side scalar value is
expected to frequently change.

### Option 2 (recommended): expose a new memory descriptor kind

The idea here is to:
- expose host-side scalars through a new memory descriptor create function
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
explicitly supported by implementations.

For users passing primitive objects around and querying them, it makes
sense to expose this format kind, so users can query if a given input
is a host-side scalar, and pass it accordingly to primitive execution.

Pros:
- Avoids user from recreating primitives when scalars are changed.
- Simple enabling internally.
- Allows passing scalar as a kernel argument, that should not introduce
a performance penalty.

Cons:
- Ties the memory descriptor to a specific device (host) which we
  avoided so far.

> Note: Options 2 and 3 formally require creating memory objects with a `CPU` engine.
However, since this is a very specific use case that does not otherwise
require full CPU support in oneDNN, it is suggested to introduce a new `host` or "null" engine.
This would allow users to avoid building oneDNN with complete CPU support just for this scenario.

### Option 3: explicitly take engine kind in memory desc

Here, we would allow users to pass any memory on host or device.
This would officially tie a given memory descriptor to a device kind.

For compatibility reasons, we could either use `undef` to specify that
a memory desc will be ties to the same engine kind as the
implementation, or introduce a new `engine_kind` for that purpose.

A few points need to be made:
- Mixing host and device memory is already possible through usm shared
  memory for both sycl and opencl runtime.
- There is no clear benefit to mix host and device memory, except for
  the case where the value to pass is a scalar, in which case it can
  be passed to device kernel as a parameter (instead of passing a
  pointer).
- Even if two memories share the same engine kind, it does not
  guarantee that they will both be accessible from the same context
  (they might be tied to different engine objects).

Pros:
- Clear semantic: all memory descriptors are tied to an engine kind.

Cons:
- Full buffers sharing across devices can already be achieved through runtime mechanisms.
- No clear benefit for performance other than specific scalar case.

As a result, we don't recommend to specify engine kind for memory
descriptors as it would likely provide little benefit over host-side
scalar specific memory descriptor, but being more general, it would
also come with more complications.

> Note: Options 2 and 3 formally require creating memory objects with a `CPU` engine.
However, since this is a very specific use case that does not otherwise
require full CPU support in oneDNN, it is suggested to introduce a new `host` or "null" engine.
This would allow users to avoid building oneDNN with complete CPU support just for this scenario.

### Option 4 (2nd recommended): do nothing and suggest user to use USM `malloc_host`

With this option, the user allocates a USM memory buffer on the host using `malloc_host`,
stores the scalar value in it, and creates a `dnnl::memory` object from that buffer,
which can then be passed to the execute function.

Pros:
- Allows to pass host-side scalars without changing the API or internals of oneDNN.

Cons:
- Requires user to keep the USM memory alive until kernel computations finish.
- There is a latency overhead when the GPU fetches host USM data.
While the data may be cached in GPU memory after the initial read,
that first access can introduce a delay.

## How should host-side scalar be passed to execute call?

Currently, the oneDNN primitive API only accepts `dnnl::memory` objects
(through an `unordered_map` in C++ interface, or an array of
`arg_kind`/`memory` for C interface).

We have two paths:
1. (recommended) Encapsulate host-side scalars in memory objects: this would create
overhead for passing a scalar but would not require execution API
changes. There is a question of object lifetime to clarify here: for
host-side scalar, we have to guarantee that they are not accessed
asynchronously, otherwise users will incur extra overhead of
allocation and copy for a single scalar.
2. Expose new execute function to allow passing scalars. For both C and C++ APIs,
it would require to take extra unordered_map for scalar arguments.

Even though the first option has unnecessary overhead, we recommend to
use it to align scalar and non-scalar passing.

## Recap

| Option                                         | Pros                                                                                 | Cons                                                                                                                        |
|------------------------------------------------|--------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| **1. Primitive desc construction time constant** | - Already supported by oneDNN API for infrequently changing parameters               | - Forces user to recreate primitive descriptor and primitive objects every time the scalar changes                          |
|                                                |                                                                                      | - Potential extra JIT overhead if implementation treats parameter as constant                                               |
| **2. New memory descriptor kind**               | - Avoids recreating primitives when scalars change                                   | - Ties memory descriptor to a specific device (host), which was previously avoided                                         |
|                                                | - Simple to enable internally                                                        |                                                                                                                             |
| **3. Add `engine_kind` to memory desc**         | - Clear semantics: all memory descriptors are tied to an engine kind                 | - Full buffer sharing across devices already possible via runtime mechanisms                                                |
|                                                |                                                                                      | - No clear performance benefit except for scalar case                                                                      |
|                                                |                                                                                      | - Ties memory descriptor to a specific device (host), which was previously avoided                                         |
| **4. User relies on USM `malloc_host`**         | - Allows passing host-side scalars without API or internal changes to oneDNN         | - Requires user to keep USM memory alive until computations finish                                                          |
|                                                |                                                                                      | - Latency overhead when GPU fetches host USM data (initial access may introduce delay)                                      |

Options 4 is currently under evaluation.
If it presents additional challenges,
the recommendation is to go with option 2: introduce APIs to support host-side scalars
via a new memory descriptor creation function,
allowing users to encapsulate scalars in a memory object for execution.

A POC branch for option 2 is available [^2].

## References

[^1]: [SDPA operator in PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
[^2]: [POC branch for host-side scalar memory descriptors](https://github.com/uxlfoundation/oneDNN/tree/mgouicem/main/host_scalar)
