# Host-side scalars Support in oneDNN Primitives

## Motivation

Currently, primitives assume that all inputs passed at execute time are on the
same device as the one they execute on (with the exception of reorder).

A new request is to pass scalar values that reside on host device to GPU kernels
at execution time. An example is the scale for SDPA operator in Pytorch [^1],
which is a python variable on the host and not a `torch.tensor` object.

There are two challenges here:
- How to specify to primitive descriptor creation that a given input is a host
    scalar?
    - Currently, we don't have any engine specified to input memory descriptors.
- How to pass host-side scalars to the execute function?
    - Currently, we only take `dnnl::memory` objects.

## Options for introducing host-side scalars support

### **Recommended option**: Expose a new memory descriptor kind

The proposal introduces host-side scalars using a dedicated memory descriptor
and memory creation function.

This **new memory descriptor** is lightweight: user only specify the scalar data
type. A special format kind `dnnl_format_kind_host_scalar` marks it as a
host-side scalar, so that implementations can detect this and pass the scalar
as a kernel parameter (see SDPA POC [^2])
(this property also accessible by the user via `dnnl_memory_desc_query` API).

When creating a memory object for a host-side scalar via the **new memory creation function**,
the value is stored internally, so user does not need to manage its lifetime.
No engine is required in the API, so that there is no need for a CPU engine
or building the library with CPU support.
Note that since the memory object is host-based, GPU implementations must be updated to
detect and handle host-side scalars by passing the value as a kernel
argument, not by accessing it via memory. By default, all the GPU implementations
should return an error for this memory unless explicit support is provided.

```C
// C API

// Scalar value
float value = 42.0f;

// Create a memory descriptor for a host-side scalar
dnnl_memory_desc_t scalar_md;
dnnl_memory_desc_create_host_scalar(
        &scalar_md, dnnl_f32);

// Create a memory object for the scalar
dnnl_memory_t scalar_mem;
dnnl_status_t status = dnnl_memory_create_host_scalar(&scalar_mem, scalar_md, &value);

// Use as regular memory object in execute function...
```

```C
// C++ API

// Scalar value
float value = 42.0f;

// Create a memory object for the scalar (no engine needed)
memory scalar_mem(memory::desc::host_scalar(memory::data_type::f32), value);

// Use as regular memory object in execute function...
```

Additional considerations for the new memory object:
- The `get_engine()` method will return `nullptr` for host-side scalar memory objects.
- The `{set, get}_data_handle()` APIs will either return `nullptr` or an error code, since the scalar value is stored internally as a copy (and not user-provided memory) that changes the previous meaning/behavior of those functions.
- To retrieve or update the scalar value stored in a memory object, a new API `{set, get}_host_scalar_value()` will be introduced.

POC for API changes is available in the draft PR [^3],
you could check it out for simple example, API tests, and some implementation
details (although the implementation is not complete yet).

Pros:
- Users do not need to recreate primitives when scalar values change.
- Allows passing scalars as kernel arguments, minimizing performance overhead.

Cons:
- Requires explicit support in each non-host implementation to handle host-side scalars
    correctly.

#### Additional considerations for primitive descriptor creation

For primitives such as matmul, and when using attributes like scales or zero
points, we should also consider whether information about scales and zero
points being host-side scalars should be available during primitive descriptor
creation.

With the option proposed above, to use a scaling factor as a host-side scalar,
the user would do the following:
```C
// alpha is a host-side scalar
memory alpha_m(memory::desc::host_scalar(memory::data_type::f32), 2.0f);

// scaling factor is a single value to be applied to src
primitive_attr attr;
attr.set_scales_mask(DNNL_ARG_SRC, /* mask */ 0);

// No info about scaling factor being host-side scalar is
// passed to primitive descriptor creation as it is an attribute
matmul::primitive_desc matmul_pd(
    eng, a_mem.get_desc(), b_mem.get_desc(), c_mem.get_desc(), attr);
matmul matmul_prim(matmul_pd);

// Info about scaling factor being host-side scalar is only available at execute
// since we pass it as a memory object
std::unordered_map<int, memory> args = {
    {DNNL_ARG_SRC, a_mem},
    {DNNL_ARG_WEIGHTS, b_mem},
    {DNNL_ARG_DST, c_mem},
    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, alpha_m}
};
matmul_prim.execute(s, args);
```

**Option 1:** Add new APIs for host-side scales/zero points:
```C
void dnnl::primitive_attr::set_host_scale(int arg,
            memory::data_type data_type = memory::data_type::f32);

void dnnl::primitive_attr::set_host_zero_point(int arg
            memory::data_type data_type = memory::data_type::s32);
```

**Option 2 (b is recommended):** Extend existing API with a special mask or boolean:
```C
// a. Define new mask, but use existing API.
#define DNNL_MASK_HOST_SCALAR (INT_MIN + (INT_MIN >> 1))

// b. Extend existing C++ API with a boolean parameter to indicate memory placement.
// For C API, add a `*_v2` function with an integer `is_on_host` parameter.
void dnnl::primitive_attr::set_scales(int arg, int mask, const memory::dims &groups,
            memory::data_type data_type = memory::data_type::f32,
            bool is_on_host = false)
```
- Both options make host-side scalars explicit at primitive descriptor creation,
  allowing kernels to prepare for host scalars and providing early error messages
  if unsupported.
- Option 2a is limited to attributes using masks (scales, zero-points); other
  cases would need a different approach. It also changes the mask semantics,
  since host scalar would be represented differently than the device scalar.
- A disadvantage of option 2b is that it requires the user to specify groups,
  which are irrelevant for host-side scalars, and also allows specifying a
  non-zero mask that would never be supported for host scalars. The advantage of
  option 2b is that it provides a single API for all scales or zero-points
  settings.
- Option 1 "disconnects" host scale from the mask `0` for the user, while
  option 2b explicitly sets the mask value as `0`, same as for a "device" scalar.

**Option 3:** No new APIs; detect at runtime.

Kernels support both pointer and value, checking which to use at runtime.

- Errors for unsupported host scalars are only returned at execution,
  making debugging harder but keeping user code simple.

## Other options considered

### Option 1: User to rely on USM `malloc_host` (no changes to oneDNN)

With this option, the user allocates a USM memory buffer on the host using
`malloc_host`, stores the scalar value in it, and creates a `dnnl::memory`
object from that buffer, which can then be passed to the execute function.

Pros:
- Allows to pass host-side scalars without changing the API or internals of
    oneDNN.

Cons:
- Requires user to keep the USM memory alive until kernel computations finish.
- There is a latency overhead when the GPU fetches host USM data. While the data
    may be cached in GPU memory after the initial read, that first access can
    introduce a delay.
- **Update:** After testing this approach, it was found that the overhead is still
    noticeable and managing the lifetime of USM memory can be complex for applications.

### Option 2: Take host-side scalars as primitive descriptor parameters

Here user would pass the host side scalar as a parameter to primitive creation,
as is done today. The main caveats here are:
- It requires user to recreate primitive descriptors and primitives every time
    the host-side scalar changes value.
- If the implementation jits a kernel that depends on that parameter, we can get
    large overheads due to re-jitting.

Pros:
- oneDNN API can already handle this for parameters that do not frequently
    change.

Cons:
- Forces user to recreate primitive descriptor and primitive objects every-time
    that host-side scalar value changes.
- Potential extra jitting overhead if internal implementation considers that
    parameter constant.

This option is not recommended if the host-side scalar value is expected to
frequently change.

### Option 3: Take engine kind in memory descriptor

Here, we would allow users to pass any memory on host or device. This would
officially tie a given memory descriptor to a device kind.

For compatibility reasons, we could either use `undef` to specify that a memory
desc will be ties to the same engine kind as the implementation, or introduce a
new `engine_kind` for that purpose.

A few points need to be made:
- Mixing host and device memory is already possible through usm shared memory
    for both sycl and opencl runtime.
- There is no clear benefit to mix host and device memory, except for the case
    where the value to pass is a scalar, in which case it can be passed to device
    kernel as a parameter (instead of passing a pointer).
- Even if two memories share the same engine kind, it does not guarantee that
    they will both be accessible from the same context (they might be tied to
    different engine objects).

Pros:
- Clear semantic: all memory descriptors are tied to an engine kind.

Cons:
- Full buffers sharing across devices can already be achieved through runtime
    mechanisms.
- No clear benefit for performance other than specific scalar case.

As a result, we don't recommend to specify engine kind for memory descriptors as
it would likely provide little benefit over host-side scalar specific memory
descriptor, but being more general, it would also come with more complications.

> Note: Options 3 formally require creating memory objects with a `CPU` engine.
However, since this is a very specific use case that does not otherwise require
full CPU support in oneDNN, it is suggested to introduce a new `host` or "null"
engine. This would allow users to avoid building oneDNN with complete CPU
support just for this scenario.

## Option 4: Expose new execute function to allow passing scalars.

For both C and C++ APIs, this option would require introducing a new `execute`
function overload that accepts an additional container (such as `unordered_map`
or similar) specifically for scalar arguments. This would allow users to pass
host-side scalars directly as arguments, separate from the usual memory objects.

Pros:
- Provides a clear and explicit way to pass scalars without overloading the
    semantics of memory objects.
- Avoids tying memory descriptors to a specific device or engine.
- Can be implemented in a backward-compatible manner by adding new overloads.
- Remove overhead of creating a memory object and memory descriptor for the scalar.

Cons:
- Requires API changes and additional maintenance for new execute function
    variants.
- May introduce complexity in argument handling and validation.
- Scalar arguments would need to be handled separately from memory objects,
    which could complicate user code and internal implementation.

## Recap

| Option                                    | Pros                                                                                   | Cons                                                                                                                        |
|:-------------------------------------------|:-------------------------------------------------------------------------------------- |:--------------------------------------------------------------------------------------------------------------------------- |
| **New memory descriptor kind**          | - Avoids recreating primitives when scalars change                                     | - Requires explicit support in each implementation                                          |
|                                         | - Allows passing scalar as a kernel argument with minimal performance overhead         |                         |
|                                         | - User does not have to worry about the lifetime of the scalar value                       |                                                                          |
| **User relies on USM `malloc_host`**    | - Allows passing host-side scalars without API or internal changes to oneDNN           | - Requires user to keep USM memory alive until computations finish                                                          |
|                                            |                                                                                        | - Latency overhead when GPU fetches host USM data (initial access may introduce delay)                                      |
| **Primitive desc construction time constant** | - Already supported by oneDNN API for infrequently changing parameters                 | - Forces user to recreate primitive descriptor and primitive objects every time the scalar changes                          |
|                                            |                                                                                        | - Potential extra JIT overhead if implementation treats parameter as constant                                               |
| **Add `engine_kind` to memory desc**    | - Clear semantics: all memory descriptors are tied to an engine kind                   | - Full buffer sharing across devices already possible via runtime mechanisms                                                |
|                                            |                                                                                        | - No clear performance benefit except for scalar case                                                                      |
|                                            |                                                                                        | - Ties memory descriptor to a specific device (host), which was previously avoided                                         |
| **New execute function for scalars**    | - Explicit and flexible way to pass scalars                                            | - Requires new API and additional maintenance                                                                               |
|                                            | - Avoids device/engine coupling in memory descriptors                                  | - Scalar arguments handled separately from memory objects, increasing complexity                                            |

## References

[^1]: [SDPA operator in PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
[^2]: [POC for SDPA with host-side scalars](https://github.com/uxlfoundation/oneDNN/pull/3412)
[^3]: [POC for host-side scalars support](https://github.com/uxlfoundation/oneDNN/pull/3506)

