# s64 (int64) Data Type Support in oneDNN

## Introduction

The main driver for adding `int64` to oneDNN is interoperability with ONNX.
ONNX Runtime represents many tensors as `int64`, including token ids,
indices, and sequence positions in transformer models like GPT-2. The ONNX
type system also requires `int64` for several standard operators (`Gather`
indices, `Shape` outputs, and so on). The OpenVINO ONNX Execution Provider
has flagged conformance issues tied to this gap.

The `dnnl_s64` token already exists in the public enum, but only as a type
definition with no primitive accepting it (details in the next section). In
practice there is no way to do anything with an `s64` tensor, not even
convert it to a friendlier type with `reorder`.

Issue [#4299](https://github.com/uxlfoundation/oneDNN/issues/4299) asks for
this. The scope proposed here is deliberately small: make `s64` usable for
memory descriptors and the `reorder` primitive, and leave everything else
alone. Compute primitives like convolution and matmul are out of scope,
since the use cases below do not need them.

### Use cases

- **ONNX Runtime interop.** Models run through ONNX Runtime regularly produce
  `int64` outputs. Wrapping one of those in a oneDNN memory object means
  branching around `int64` today, because there is no `dnnl_memory_desc` that
  can describe it.

- **ONNX conformance in the OpenVINO Execution Provider.** The OpenVINO ONNX EP
  has to match the ONNX type system, which uses `int64` for the index- and
  shape-related operators. Missing `int64` support underneath shows up as
  conformance gaps.

- **One code path for language bindings.** Generic binding code that takes an
  arbitrary oneDNN tensor and produces a tensor of the same type works for
  every common type but `int64`, which needs its own branch. Supporting `s64`
  lets that branch go away for the Java/JNI, C# and Clojure bindings.

- **Pulling out a subtensor.** A common need is to copy some region of a larger
  `int64` tensor before processing it. `reorder` already does strided copies
  through memory descriptors, so with `s64` in place users can reuse the
  subtensor descriptors they already build instead of writing their own
  stride-aware copy loop in C.

## Current state in oneDNN

The `s64` token and its internal plumbing were introduced in commit
4cab67c4f0 ("api: add s64 data type", Sep 2025), which added the type to the
enum and wired up the type traits but did not enable it in any primitive. On
`main` today the following are in place:

- `dnnl_s64` in the C enum `dnnl_data_type_t`, and `memory::data_type::s64` in
  the C++ header.
- The `"s64"` string in the verbose/debug output.
- The type traits (`prec_traits_t`, `data_traits_t`, and
  `nstl::numeric_limits`) that map `s64` to `int64_t`.
- `data_type_size()` and the min/max/lowest/digits helpers in
  `type_helpers.hpp`, plus `is_integral_dt()` returning true for `s64`.
- `memory_desc_sanity_check()` allowing `s64`, which means an `s64` memory
  descriptor can already be created.
- A `load_int64_value()` helper in `src/cpu/ref_io_helper.hpp`.
- benchdnn already knows how to allocate `s64` memory.

What is not there is a primitive that accepts it. The `reorder` dispatch tables
under `src/cpu/reorder/` have no `s64` entry, so creating a `reorder` with an
`s64` descriptor fails at primitive descriptor creation with `unimplemented`.
In practice `s64` is still unusable, and this proposal is about closing that
last gap.

## Proposal

Make `s64` work for describing memory and moving data, and leave the compute
side untouched.

### Scope

**Memory descriptor.** No API change needed: `memory_desc_sanity_check()`
already accepts `s64` (see Current state), so an `s64` memory descriptor can
be created today. The proposal simply treats it as supported and lists it in
the documented set of memory descriptor data types.

**`reorder` primitive.** Add `s64` to the CPU `reorder`. That covers the plain
copy (`s64` to `s64`, including non-trivial strides for subtensor extraction)
and conversion between `s64` and the float and integer types the use cases care
about:

| From  | To                  |
|:------|:--------------------|
| `s64` | `s64`, `f32`, `s32` |
| `f32` | `s64`               |
| `s32` | `s64`               |

Rounding and saturation follow what the existing integer reorders already do:
integer destinations saturate on out-of-range values (the same way `s32` to
`s8` behaves today). The `s64` to `f32` case is a precision-loss conversion
rather than a saturating one; values beyond 2^24 lose low-order bits.

The narrower integer destinations (`s8`, `u8`) are left out for now to keep the
first change small. The exact set of conversions is something to settle with
the maintainers during review; see the Open Questions below.

**Out of scope.** Convolution, matmul, pooling, normalization, and everything
else on the compute side. `s64` is not proposed as a compute or accumulation
type. GPU reorder is out too: it has its own dispatch
(`src/gpu/gpu_reorder_list.cpp` plus the per-vendor code under `src/gpu/`),
separate from the CPU tables in `src/cpu/reorder/`, and only the CPU side is
touched here. Any of this can come back in a later RFC if a real use case shows
up.

### API changes

There is no new API. The type already exists:

```c
// include/oneapi/dnnl/dnnl_common_types.h (already present)
typedef enum {
    ...
    dnnl_f4_e3m0 = 15,
    /// 64-bit signed integer
    dnnl_s64 = 16,
    dnnl_data_type_max = 0x7fff,
} dnnl_data_type_t;
```

```cpp
// include/oneapi/dnnl/dnnl.hpp (already present)
enum class data_type {
    ...
    f64 = dnnl_f64,
    s64 = dnnl_s64,
    s32 = dnnl_s32,
    ...
};
```

So the actual work is implementation-only: new entries in the CPU `reorder`
dispatch tables under `src/cpu/reorder/` for the type pairs above. No ABI
break, and nothing changes for any other data type.

What it looks like to use:

```cpp
// Wrap an int64 buffer coming from ONNX Runtime or a language binding and
// convert it to f32 for downstream processing, with no custom C loop.
auto src_md = memory::desc(dims, memory::data_type::s64, tag::abcd);
auto dst_md = memory::desc(dims, memory::data_type::f32, tag::abcd);

memory src_mem(src_md, eng, int64_ptr);
memory dst_mem(dst_md, eng);

reorder(src_mem, dst_mem).execute(strm, src_mem, dst_mem);
```

### Testing

- gtests: extend the reorder correctness tests with the new `s64` pairs,
  including a strided `s64` to `s64` subtensor copy and a saturating `s64` to
  `s32` conversion.
- benchdnn: add `s64` reorder cases to the driver inputs. benchdnn can already
  allocate and fill `s64`, so this is mostly a matter of enabling the new
  source/destination combinations in the input files and the reference path.

### Implementation

I plan to open the implementation PR (CPU reorder entries plus the tests
above) as a follow-up once the RFC is accepted.

## Alternatives considered

**Just convert in user code.** Turning `int64` into `int32` or `f32` in C, or
in the binding's own language, is easy enough for contiguous data. The problem
is that it pushes a per-type special case onto every consumer, breaks the "one
API for all types" property that bindings rely on, and gets fiddly once strides
are involved. That is exactly the complaint in the issue, so it is not a good
long-term answer.

**Support `s64` everywhere.** Adding `int64` convolution, matmul and the rest
would be a large amount of work for demand that does not exist. Keeping the
proposal to reorder keeps it small and reviewable.

## Open Questions

1. Which conversions to support. The table above sticks to `f32` and `s32`,
   plus the `s64` to `s64` copy. Worth deciding with the maintainers whether
   `s8`/`u8` destinations should be in from the start or added later on demand.

2. Unsigned `u64`. The issue is about signed `int64`. We could add `u64` at the
   same time for symmetry with `s8`/`u8`, or wait until something actually
   needs it. The proposal leans towards waiting.

3. Performance. The reporter is fine with slow ("even with sub-optimal
   performance"). Is a plain reference CPU implementation good enough to start
   with, leaving optimized paths for later if profiling shows they are needed?

4. Conversion edge cases. `s64` to `f32` loses precision past 2^24, and `s64`
   to `s32` saturates outside the int32 range. Is it enough to document "same
   rounding and saturation as the other integer reorders", or does anyone need
   an explicit precision-loss signal?

5. GPU. This only covers CPU reorder. Is there real demand for an `s64` reorder
   on GPU, or is host-side conversion fine given these tensors usually live in
   main memory anyway?
