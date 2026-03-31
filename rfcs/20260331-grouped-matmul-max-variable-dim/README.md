# Add DNNL_ARG_HINT_MAX_GROUP_M for Grouped GEMM

## Introduction

> [!NOTE]
> Grouped GEMM support is currently marked as experimental
> (available under `ONEDNN_EXPERIMENTAL_GROUPED_MEMORY`), so API/ABI stability is not guaranteed yet,
> and the API may evolve based on feedback.

The grouped GEMM feature (see [previous RFC](https://github.com/uxlfoundation/oneDNN/tree/rfcs/rfcs/20251203-grouped-gemm-support))
enables Mixture-of-Experts (MoE) support by representing variable-length groups of tokens
as a single memory descriptor, where tokens are concatenated contiguously and
per-group boundaries are communicated via a **device-side** offsets buffer.

For a grouped matmul with `E` experts and `total_M` tokens (the sum of all per-expert M values),
the GPU kernel grid is currently: `gws ~ N * total_M * E`.
This generates a much higher number of workers than necessary, since each expert need to process
only `M_i` tokens to compute its output. Work groups that exceed the actual per-expert token range,
as well as work groups for experts with no tokens assigned, exit early.

The issue is that since per-expert size information is available only on the device,
`total_M` is the only safe upper bound that can be used.

This RFC proposes an optional hint argument that lets users communicate a tighter
upper bound, which can be used to reduce the overhead and dispatched grid size significantly.

## Possible Bounds on M

### Natural Bounds for MoE Models

MoE layers use TOP-K routing, where each of `T` input tokens is assigned to `top_k` experts,
so that with `E` experts and `T` tokens we have `total_M = T * top_k`.
This `top_k` (e.g., 2, 4 or 8) strategy is known at model initialization.

In addition, sometimes an expert capacity is limited by a `capacity_factor` parameter
(e.g., `1.25`), that specifies how many tokens a single expert can receive.

Both `top_k` and `capacity_factor` are known at model initialization and could help
provide an upper bound on the per-expert token count.

### Tighter Upper Bound

Additionally, after routing completes, offsets are available.
If an application has host access to offsets or can compute the maximum per-expert token count
from routing metadata, it can provide a tight upper bound as
`max_m = max(offsets[g] - offsets[g-1])` for `g in [0, num_experts)`.

## Proposal

### Optional execute-time hint argument

Add `DNNL_ARG_HINT_MAX_GROUP_M = 384` as a new optional argument in the `execute()`
arguments list for matmul.
This argument is a host scalar memory object containing an `s32` value
representing the upper bound on the per-group M dimension for grid dispatch purposes.

```cpp
int32_t max_m = compute_max_m(...);

auto hint_md = dnnl::memory::desc::host_scalar(dnnl::memory::data_type::s32);
dnnl::memory hint_mem(hint_md, max_m);

std::unordered_map<int, dnnl::memory> args = {
    {DNNL_ARG_SRC, src_mem},
    {DNNL_ARG_WEIGHTS, wei_mem},
    {DNNL_ARG_DST, dst_mem},
    {DNNL_ARG_HINT_MAX_GROUP_M, hint_mem},  // NEW optional argument
};
matmul_prim.execute(stream, args);
```

- The value is a hint: currently it is planned that the implementation uses it
  only to reduce the dispatch grid.
  If not provided, `M_dispatch` falls back to `total_M` as before.
  Results are correct regardless of the value provided, as long as `max_m >= actual max per-group M`.
  If the hint is smaller than the actual maximum, some token rows will be missed and results will be incorrect.
- Parameter is host-accessible via a host scalar memory object,
  as only a single value is needed at execution time.
- Type is `s32`, matching the offsets buffer data type.
- The hint could be applied to different grouped GEMM variants
  (e.g., grouped src by 3D wei, grouped src by grouped wei),
  since it is derived from model configuration and/or routing metadata
  and not tied to a specific tensor.
- No changes to grouped memory descriptor or memory object APIs.

### Note on hint arg and primitive descriptor

A hint does not change which kernel would be selected or how the primitive descriptor is configured.
Idea is that user may pass it or omit it on `execute()` call without affecting correctness.
An implementation that does not support the hint simply ignores it.

`DNNL_ARG_HINT_MAX_GROUP_M` could be placed to a new `DNNL_ARG_HINT_*` category (value `384`),
with an intentional gap after last argument `DNNL_ARG_DIFF_SHIFT = 256` and before the
`DNNL_ARG_ATTR_*`, to make it distinct from regular arguments.

## Alternatives Considered

### B: Grouped Memory Descriptor Field

Add `max_variable_dim` to grouped memory descriptor.

Cons:
- Semantics would be broken, since memory descriptor encodes the structure of a tensor,
  (e.g., shape, data type, group count and which dimension is variable).
  However, `max_variable_dim` is a dynamic hint that can change based on tokens distribution,
  and passing `max_variable_dim` this early at memory descriptor creation time is not ideal,
  as the best max per-group `M` is routing-dependent and will not be known at that time.
- Additionally, `max_variable_dim` would need to appear in both the src and dst
  grouped descriptors.

### C: Third Buffer in Grouped Encoding

Add as a buffer at index 2 in the grouped memory object, alongside
values (index 0) and offsets (index 1).

Cons:
- Main concern is engine mismatch. For GPU execution, the existing two buffers
  (values and offsets) are GPU data.
  All buffers in a memory object share the same engine, so we would need mixed-engine
  buffers in a single memory object.
