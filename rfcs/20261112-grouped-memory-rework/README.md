# Grouped Memory Rework: Align Grouped Matmul Support with PyTorch

## Introduction

PyTorch has several grouped matmul ops for Mixture-of-Experts (MoE) models: `_grouped_mm`, `_scaled_grouped_mm`
(and `_scaled_grouped_mm_v2` that includes FP4 recipes). The public API wraps `_grouped_mm` and `_scaled_grouped_mm_v2`.
Below, `_scaled_grouped_mm` stands for both versions.

Current grouped memory descriptor ([grouped GEMM RFC](https://github.com/uxlfoundation/oneDNN/tree/rfcs/rfcs/20251203-grouped-gemm-support))
covers only part of this functionality, so this RFC proposes a rework of the descriptor.

With this proposal, existing users keep every feature they have today, but will need to update the
descriptors and, if they use them, the scale and zero-point masks and the shapes of the bias and
binary post-op tensors of grouped matmul.
This is allowed because grouped memory is an experimental feature.

In addition to the proposed API changes, the appendix maps the `_scaled_grouped_mm` recipes onto the updated API.

## Current Grouped Encoding and Support

```c
dnnl_status_t dnnl_memory_desc_create_with_grouped_encoding(dnnl_memory_desc_t *memory_desc,
        int ndims, const dnnl_dims_t dims, dnnl_data_type_t data_type, int variable_dim_idx,
        dnnl_dim_t group_count, dnnl_data_type_t offsets_dt);
```

- The descriptor is 2D (`G = group_count` groups concatenated along the variable dim), e.g.
  `[total_M, K]`. Buffer 0 holds the values, buffer 1 the offsets.
- Strides are derived internally: the **variable dim is always outermost** (so
  `variable_dim_idx=0` is row-major and `variable_dim_idx=1` is column-major).
- Grouped matmul variants are supported in the following way:
  - 2D x 3D is src grouped `[total_M, K]` with `variable_dim_idx=0` (var-M) by weights dense
    `[G, K, N]` to dst grouped `[total_M, N]`.
  - 2D x 2D is src grouped `[M, total_K]` with `variable_dim_idx=1` (var-K) by weights grouped
    `[total_K, N]` with `variable_dim_idx=0` (K varies) to dst dense `[G, M, N]`.

## Required Configurations

What PyTorch offers across its backends:

| | `_grouped_mm` | `_scaled_grouped_mm` |
|---|---|---|
| Variants | 2D x 3D (var-M), 2D x 2D (var-K) | same |
| Layouts (*) | various combinations of row- and column-major for grouped data | `mat_a` row-major, `mat_b` column-major |
| Offsets | int32 `[G]`, cumulative end offsets on the device | same |
| Data types for `mat_{a,b}` | bf16, f16, f32 | f8_e4m3, f4_e2m1 |
| Output data type | same type as the inputs | bf16 |
| Scales | not supported | FP8 row-wise, MXFP8, MXFP4, NVFP4 (**) |
| Bias | not supported | same |

(*) In PyTorch, each grouped operand is one concatenated tensor over all groups, e.g. for var-K
`[M, total_K]`:

```
[M, total_K], K_g = offsets[g] - offsets[g-1], offsets[-1] = 0

          |<- K_0 ->|<- K_1 ->| ... |<- K_G-1 ->|
  row 0   | group 0 | group 1 | ... | group G-1 |
  ...     |   ...   |   ...   | ... |    ...    |
  row M-1 | group 0 | group 1 | ... | group G-1 |

row-major in memory:     full row 0 | row 1 | ... | row M-1
column-major in memory:  group 0 | group 1 | ... | group G-1
```

In row-major, each row runs through all groups, so rows are `total_K` apart.
In column-major, each group is one block of `M * K_g` values, and columns are `M` apart.
Note that neither stride depends on the group sizes, so host strides can describe both.

(**) See the appendix for the exact recipes.

What oneDNN supports today:

| PyTorch call | Used for | Notes |
|---|---|---|
| `_grouped_mm` | MoE layers in bf16, including the weight gradient from PyTorch autograd | Supported only in oneDNN layouts |
| `_scaled_grouped_mm` 2D x 3D | MoE forward and data gradient (torchao) | Supported, PyTorch layouts match oneDNN layouts |
| `_scaled_grouped_mm` 2D x 2D | MoE weight gradient (torchao) | **Requires a different layout, some recipes (e.g. FP8 row-wise) also need a group dim** |

Gaps for `_scaled_grouped_mm` 2D x 2D:

- Today's fixed layout: the descriptor takes no strides and always makes the variable dim
  outermost. 2D x 2D `_scaled_grouped_mm` requires K to be contiguous in both operands. So to be
  compatible, a backend that receives these tensors would have to copy both operands into oneDNN's
  layout.
- No group dim: the descriptor is 2D, so a scale mask cannot say "one value per group". For
  example, FP8 row-wise needs one scale per (group, row) of `mat_a` and per (group, column) of
  `mat_b`, that can't be expressed.

Note that even though `_grouped_mm` also supports layouts that are not available in oneDNN today,
there is currently no request to support them, so `_scaled_grouped_mm` 2D x 2D is considered
to be the main driver for the change.

## Proposal

### Memory Descriptor

At a high level, the idea is to change the grouped memory definition to the following:

```
ndims      = 3
dims       = {G, *, K}  // e.g. var-M src; var-M dst {G, *, N}; var-K: src {G, M, *}, weights {G, *, N}
data_type
total
strides
offsets_dt = s32
```

- `*` is a new placeholder constant, `DNNL_VARIABLE_DIM_VAL` (e.g., equal to `INT64_MIN + 1`;
  see [Open Questions](#open-questions) for the name). It only marks which dim varies and is never
  replaced by a number. It replaces `variable_dim_idx`.
- `G = dims[0]` is added explicitly to `dims` and replaces the separate `group_count`.
- `strides` (in elements) are added explicitly and follow these rules:
  - Strides are host values, available at creation.
  - `strides[0]` (`G` dim) is `0` (see also [Open Questions](#open-questions)). The distance between
    groups changes from group to group, so no single number can describe it. The device offsets
    buffer gives the kernel this information.
  - One matrix dim has stride 1, so the tensor is row-major or column-major. The other stride is
    at least the size of the dim with stride 1 (`total` if that is the variable dim). A larger
    value is padded leading dim as usual.
    - For example, the var-K src `{G, M, *}`, the `[M, total_K]` example in
      [Required Configurations](#required-configurations), has strides `{0, total_K, 1}` in
      row-major and `{0, 1, M}` in column-major.
- `total` is the size of the variable dim summed over all groups, i.e. `offsets[G-1]` (see also
  [Open Questions](#open-questions)). It cannot be derived from the strides, so the descriptor
  keeps it (it appears in a stride only when the variable dim has stride 1). For example, the var-M
  row-major src `{G, *, K}` has strides `{0, K, 1}`, and the var-K column-major src `{G, M, *}`
  has `{0, 1, M}`.
- Queries: `get_dims()` returns `DNNL_VARIABLE_DIM_VAL` at the variable dim, and `get_strides()`
  returns the strides. There is no query for `total`, but it equals `get_nnz()` divided by the
  fixed dim.
- Buffers and execution arguments do not change: buffer 0 holds the values and buffer 1 the
  offsets.

Suggested API (signature change):

```c
dnnl_status_t dnnl_memory_desc_create_with_grouped_encoding(dnnl_memory_desc_t *memory_desc,
        int ndims, const dnnl_dims_t dims, dnnl_data_type_t data_type, dnnl_dim_t total,
        const dnnl_dims_t strides, // NULL means variable dim is outermost
        dnnl_data_type_t offsets_dt);
```

```cpp
static desc grouped(const dims &adims, data_type adata_type, dim total, const dims &astrides = {},
        data_type offsets_dt = data_type::s32, bool allow_empty = false);
```

### Masks Update for Scales and Zero Points

With the new descriptor, src and weights have 3D view, with `G` as dim 0, so masks should follow these 3D shapes
(e.g., src is `{G, M_g, K}` in var-M case and wei is `{G, K_g, N}` in var-K case).

For today's users, only the var-M src masks change, since the src goes from 2D `[total_M, K]` to
3D `{G, M_g, K}`. The var-M weights were already 3D, and var-K takes no scales or zero points today.

The table below gives the stored shapes. The data is stored as var-M src `[total_M, K]` (concatenated over groups) and
weights `[G, K, N]`, and var-K src `[M, total_K]` and weights `[total_K, N]`.

| Scales or zero points | Blocks | var-M mask today | Mask proposed | var-M (2D x 3D) scales or zero points shape | var-K (2D x 2D) scales or zero points shape |
|---|---|---|---|---|---|
| src or weights, per tensor | - | 0 (u3 weights zero points only) | 0 | one value | one value |
| src or weights, per group | - | not supported | 1 | `[G]` | `[G]` |
| src, per row (and per group) | - | 1 | 3 | `[total_M]` | `[G, M]` |
| src, blocks along K | `1 x b` | 3 | 7 | `[total_M, K/b]` | `[M, total_K/b]` |
| src, blocks along M and K | `m x b` | not supported | 7 | `[total_M/m, K/b]` | `[M/m, total_K/b]` |
| weights, per column | - | 5 | 5 | `[G, N]` | `[G, N]` |
| weights, blocks along K | `b x 1` | 7 | 7 | `[G, K/b, N]` | `[total_K/b, N]` |
| weights, blocks along K and N | `b x n` | 7 | 7 | `[G, K/b, N/n]` | `[total_K/b, N/n]` |

> [!WARNING]
> **Group-size rule.** Blocks along the variable dim need every group size to be a multiple of the
> block: `K_g % b == 0` in var-K, and `M_g % m == 0` for `m x b` src blocks in var-M. oneDNN
> cannot check this at creation, so the user must ensure it; otherwise the results are undefined.
> PyTorch has the same rule for the var-K MXFP8, MXFP4 and NVFP4 recipes.

### Binary Post-ops and Bias Support Changes

As with the scale and zero-point masks, the binary post-op and bias descriptors change, since `dst`
becomes 3D (includes `G` dim).

| Use (var-M) | Today | Proposed |
|---|---|---|
| Per group, e.g. NVFP4 global scale | dense `[G, 1]` | dense `[G, 1, 1]` |
| Per row, e.g. routing weights | dense or grouped `[total_M, 1]` | grouped `{G, *, 1}` |
| Per element, e.g. gating | dense or grouped `[total_M, N]` | grouped `{G, *, N}` |
| Per tensor, e.g. one NVFP4 global scale for all groups | not supported | dense `[1, 1, 1]` |
| Bias | dense `[G, N]` | dense `[G, 1, N]` |

### Examples

var-M with FP8 row-wise scales, as in `_scaled_grouped_mm` 2D x 3D

```cpp
const memory::dim total_M = num_tokens * top_k; // known before routing

auto src_md = memory::desc::grouped({G, DNNL_VARIABLE_DIM_VAL, K}, dt::f8_e4m3, total_M);
// mat_b [G, K, N], column-major: K contiguous
auto wei_md = memory::desc({G, K, N}, dt::f8_e4m3, memory::format_tag::acb);
auto dst_md = memory::desc::grouped({G, DNNL_VARIABLE_DIM_VAL, N}, dt::bf16, total_M);

primitive_attr attr;
attr.set_scales_mask(DNNL_ARG_SRC, (1 << 0) | (1 << 1));     // [total_M]
attr.set_scales_mask(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 2)); // [G, N]
auto pd = matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr);

memory src_mem(src_md, eng, {src_ptr, offsets_ptr});
memory wei_mem(wei_md, eng, wei_ptr);
memory dst_mem(dst_md, eng, {dst_ptr, offsets_ptr});

matmul(pd).execute(strm, {
        {DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, wei_mem}, {DNNL_ARG_DST, dst_mem},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scales_mem},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scales_mem}});
```

var-K with FP8 row-wise scales and layout as in `_scaled_grouped_mm` 2D x 2D

```cpp
// mat_a [M, total_K], row-major: K contiguous, rows total_K apart
auto src_md = memory::desc::grouped(
        {G, M, DNNL_VARIABLE_DIM_VAL}, dt::f8_e4m3, total_K, {0, total_K, 1});
// mat_b [total_K, N], column-major: K contiguous, columns total_K apart
auto wei_md = memory::desc::grouped(
        {G, DNNL_VARIABLE_DIM_VAL, N}, dt::f8_e4m3, total_K, {0, 1, total_K});
auto dst_md = memory::desc({G, M, N}, dt::bf16, memory::format_tag::abc);

primitive_attr attr;
attr.set_scales_mask(DNNL_ARG_SRC, (1 << 0) | (1 << 1));     // [G, M]
attr.set_scales_mask(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 2)); // [G, N]
auto pd = matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr);

// Memory objects and execution as above
```

## Open Questions and Alternatives Considered

- Naming for `DNNL_VARIABLE_DIM_VAL` constant.
- How should `strides[0]` (the `G` dim) be expressed since no single number fits it?
  - `0` as proposed
  - `DNNL_VARIABLE_DIM_VAL`, as the marker already introduced for `dims`
  - No `G` stride at all, so we pass strides only for the two matrix dims
- Should `total` stay the exact size of the variable dim (`offsets[G-1] == total`) or become a
  capacity, i.e. the size of a preallocated buffer (`offsets[G-1] <= total`)? PyTorch uses a
  capacity and it only checks `offsets[G-1] <= total` on the device and ignores the elements past it
  - Pros:
    - The md depends only on the buffer size, so the same md and primitive work when the real total
      changes.
  - Cons:
    - The GPU microkernel sizes its grid by `total`, so a large unused tail wastes resources. The
      max group size hint (`DNNL_ARG_HINT_MAX_GROUP_SIZE`, see its
      [RFC](https://github.com/uxlfoundation/oneDNN/tree/rfcs/rfcs/20260331-grouped-matmul-max-variable-dim))
      could partially help with that in some cases.
    - In var-M, dst rows past `offsets[G-1]` need a defined result (PyTorch leaves them unwritten).
- Should oneDNN require groups to start at aligned addresses when the variable dim is contiguous?
  - When the variable dim is contiguous, group starts depend on the offsets. For example, in the
    row-major var-K src `[M, total_K]`, group `g` starts at element `offsets[g-1]` of each row,
    which may not be a multiple of 16 bytes. oneDNN reads the device offsets only inside
    the kernel, so the alignment is not known in advance.
    Today the variable dim is always outermost, so a group starts after `offsets[g-1]` full
    rows (or columns), and its alignment depends only on the leading dim.
  - PyTorch has such a rule on the CUDA backend: its CUTLASS and cuBLASLt paths check on the device
    that every group but the last spans a multiple of 16 bytes along the variable dim, so every
    group starts 16-byte aligned.
- Should we express a layout via a format tag instead of strides?
  For example, for the var-K src `{G, M, *}`, `acb` would be the default layout, `bac` row-major
  (strides `{0, total_K, 1}`), and `abc` one row-major `[M, K_g]` block per group.
  - Pros:
    - Can describe one block per group as well.
  - Cons:
    - No padding.
    - Could be somewhat confusing for the grouped case compared to dense memory:
      `abc` on the var-K src is one block per group, but usually reads as row-major.

## Appendix: `_scaled_grouped_mm` Recipes with the New API

- Mask bits stand for the dims of src `{G, M, K}` and weights `{G, K, N}`, in both var-M and var-K
- Blocks are the quantization groups of `set_scales`, in oneDNN order (src `rows x K`, weights
  `K x columns`)

| Recipe | src, weights -> dst | src scales: mask, blocks, type | weights scales: mask, blocks, type | post-op |
|---|---|---|---|---|
| Tensor-wise FP8, one value per tensor (*) | f8 -> bf16, f16, f32 | 0, -, f32 | 0, -, f32 | - |
| Tensor-wise FP8, one value per group (*) | as above | 1, -, f32 | 1, -, f32 | - |
| FP8 row-wise | f8 -> bf16 | 3, -, f32 | 5, -, f32 | - |
| Blockwise FP8: `1x128` src, `128x128` weights (*) | f8 -> bf16, f16, f32 | 7, `1x128`, f32 | 7, `128x128`, f32 | - |
| Blockwise FP8: `1x128` on both (*) | as above | 7, `1x128`, f32 | 7, `128x1`, f32 | - |
| Blockwise FP8: `128x128` src, `1x128` weights (*) | as above | 7, `128x128`, f32 | 7, `128x1`, f32 | - |
| MXFP8 | f8 -> bf16 | 7, `1x32`, e8m0 | 7, `32x1`, e8m0 | - |
| MXFP4 | f4 -> bf16 | 7, `1x32`, e8m0 | 7, `32x1`, e8m0 | - |
| NVFP4, block scales only (*) | f4 -> bf16, f16, f32 | 7, `1x16`, f8_e4m3 | 7, `16x1`, f8_e4m3 | - |
| NVFP4 (**), global scale per group | f4 -> bf16 | 7, `1x16`, f8_e4m3 | 7, `16x1`, f8_e4m3 | `binary_mul`, `[G, 1, 1]` f32 |
| NVFP4 (**), one global scale (*) | f4 -> bf16, f16, f32 | 7, `1x16`, f8_e4m3 | 7, `16x1`, f8_e4m3 | `binary_mul`, `[1, 1, 1]` f32 |

(*) Not in PyTorch `_scaled_grouped_mm` today.

(**) The NVFP4 post-op holds `global_a * global_b`, one value per group or one for all groups. The
caller multiplies the two global scales and passes the product, as PyTorch does.

