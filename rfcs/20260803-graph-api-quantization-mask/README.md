# RFC: Support Quantization Scale Masks for Graph API

## Overview

This RFC identifies a Graph API expressiveness gap exposed while enabling FP8
scaled dot product attention (SDPA): quantization operations cannot describe
scales that vary along more than one data dimension. It proposes several
options and compares their trade-offs.

### Current Design

Graph quantization operations currently use `qtype` to select a granularity
(`per_tensor`, `per_channel`, or `per_group`) and use `axis` to identify the
single channel dimension. Static operations take scales as an attribute;
dynamic operations take scales as an input tensor.

| Granularity               | Example source shape  | Scale shape   |
| ---                       | ---                   | ---           |
| Per-tensor                | `[B, H, S, D]`        | scalar        |
| Per-channel, `axis = 1`   | `[B, H, S, D]`        | `[H]`         |
| Per-group                 | `[B, H, S, D]`        | Defined by `group_shape` |

The model has no representation for a scale that varies over multiple logical
dimensions without grouping each element of the remaining dimensions.

### Motivating Use Case: FP8 SDPA

FP8 SDPA commonly quantizes query, key, and value tensors with scales per
batch and attention head. For a Q/K/V tensor with shape `[B, H, S, D]`, the
required logical scale shape is `[B, H]`:

```
Q_f32[B, H, S, D] --> Quantize(scales[B, H]) --> Q_f8[B, H, S, D]
```

This is conventionally called `per-head` quantization in PyTorch. A related
`per-token` recipe requires scales of logical shape `[B, H, S]`. Neither recipe
is representable by the current `per_channel` definition because it accepts one
`axis`, and neither is naturally a `per_group` recipe.

This limitation prevents a framework from faithfully constructing an FP8 SDPA
graph with oneDNN Graph API and from using oneDNN-optimized kernels.

## Requirements

The extended/revised representation should:

* Express a quantization scale that varies along any subset of source or
  destination dimensions.
* Describe the `per-head` and `per-token` FP8 SDPA recipes used by common APIs.
* Retain the existing `qtype`, `axis`, and `group_shape` graphs unchanged for
  backward compatibility.
* Support group quantization available for block and compressed-weight recipes.

## Options

### Option 1: Add Named `qtype` Values

Add values such as `per_head` and `per_token` to `qtype` values.

Pros:

* The change is additive and minimal.
* Follow the existing style. `qtype` is already defined and accepts
  `per_tensor`, `per_channel` values.

Cons:

* The names encode Transformer's tensor interpretation. Dimension 1 is not
  universally a head dimension, and dimension 2 is not universally a token
  dimension.
* The set is open-ended: per-batch-head, per-head-token, per-expert-token, and
  other layouts would each require another `qtype`.

This option is simple for the immediate case but does not scale as a general
quantization API.

### Option 2: Add a `mask` Attribute

Add an optional integer `mask` attribute. Bit `i` denotes that scales vary over
logical tensor dimension `i`. A scale tensor is packed and contains only the
selected dimensions. For source shape `[B, H, S, D]`:

| Recipe                | Mask  | Scale tensor shape    |
| ---                   | ---   | ---                   |
| Per-tensor            | `0`   | scalar                |
| Per-channel on `H`    | `2`   | `[H]`                 |
| Per-head              | `3`   | `[B, H]`              |
| Per-token             | `7`   | `[B, H, S]`           |

Pros:

* Any subset of dimensions is expressible without adding workload-specific
  terminology.
* It aligns Graph API semantics with the mask argument used by oneDNN primitive
  attributes.
* It covers the current single-axis `per_channel` case as a one-bit mask.

Cons:

* Masks are less self-describing than named recipes.

### Option 3: Extend `per_channel` to Multiple Axes

Change `axis` from a scalar to a list, so `qtype = per_channel` with
`axis = [0, 1]` expresses per-head quantization.

Pros:

* It preserves familiar attribute names.

Cons:

* The term `per_channel` becomes inaccurate for dimensions such as batch and
  sequence.
* A variable-length `axis` attribute complicates the schema and validation.

This option has a smaller API change, but it overloads an existing term and does
not provide a clean general model.

### Comparison

| Criterion                                 | Option 1: Named `qtype` values    | Option 2: `mask` attribute    | Option 3: Multi-axis `per_channel`    |
| ---                                       | ---                               | ---                           | ---                                   |
| Per-head and per-token support            | Yes                               | Yes                           | Yes                                   |
| Arbitrary dimension subset                | No                                | Yes                           | Yes                                   |
| Workload-neutral                          | No                                | Yes                           | Partly                                |
| Compatible with primitive scale masks     | No                                | Yes                           | No                                    |
| Clear without API knowledge               | Yes                               | No                            | Partly                                |
| Future API expansion                      | Repeated new values               | No new API values             | Moderate                              |
| Backward-compatible introduction          | Yes                               | Yes                           | Yes                                   |

The recommendation is to do the option 2.

## `mask` Design Considerations

### API

The implementation will add `op_attr::mask`, expose it through the C and C++
Graph APIs, and extend the schemas of `DynamicQuantize` and `DynamicDequantize`
operations.

| Attribute | Description | Type | Valid values | Required |
| --- | --- | --- | --- | --- |
| `qtype` | Selects the legacy quantization granularity. | string | `per_tensor` (default), `per_channel`, or `per_group` | Optional; must be the default when `mask` is present. |
| `axis` | Selects the dimension for legacy `per_channel` quantization. | s64 | `[-r, r-1]`, where `r` is the rank of `src`; default: `1`. | Optional; ignored when `mask` is present. |
| `group_shape` | Defines block sizes for grouped quantization. | s64[] | Two values `{G0, G1}` for the last two source dimensions. The corresponding mask bits must be set; use `1` for no sub-blocking. | Optional |
| `mask` | Selects the source dimensions over which scale values vary. | s64 | Non-negative. `0` is per-tensor, `1 << axis` is legacy per-channel, `3` is per-head for `[B, H, S, D]`, and `7` is per-token. | Optional; selected for multi-dimensional granularity. |

When needed, the backend can check the `mask` value for pattern matching and
kernel dispatching. The mask value maps directly to the corresponding primitive
attribute though the scale tensor may need to be expanded with singleton
dimensions when creating the primitive scale memory.

### Compatibility

When `mask` is absent, current behavior is unchanged:

* `per_tensor` maps to mask `0`.
* `per_channel` maps to `1 << axis`.
* `per_group` continues to use `group_shape` and retains its existing
  semantics.

When `mask` is present, it defines the non-group scale granularity. The initial
proposal is to reject a graph that specifies both `mask` and a non-default
`qtype` or `axis`, rather than silently resolving conflicting descriptions. A
later API revision may deprecate `qtype` and `axis` after the adoption of `mask`
is sufficient.

`mask` and `group_shape` compose, matching primitive attributes. When
`group_shape = {G0, G1}` is present, it subdivides the last two source
dimensions and both corresponding mask bits must be set. For example, source
shape `[B, H, S, D]`, `mask = 15`, and `group_shape = {G0, G1}` produce scale
shape `[B, H, S / G0, D / G1]`. Group sizes must divide their corresponding
source dimensions.

### Validation

Existing serialized graphs remain valid because the new attribute is optional.

Benchdnn's `--op-attr=` knob can specify the `mask` attribute for testing.
