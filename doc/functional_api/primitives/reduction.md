Reduction {#dev_guide_reduction}
============================
>
> [API Reference](@ref dnnl_api_reduction)
>

## General

The reduction primitive performs reduction operation on arbitrary data. Each
element in the destination is the result of reduction operation with specified
algorithm along one or multiple source tensor dimensions:

\f[
    \dst(f) = \mathop{reduce\_op}\limits_{r}\src(r),
\f]

where \f$reduce\_op\f$ can be max, min, sum, mul, mean, Lp-norm and
Lp-norm-power-p, \f$f\f$ is an index in an idle dimension and \f$r\f$ is an
index in a reduction dimension.

Mean:

\f[
    \dst(f) = \frac{\sum\limits_{r}\src(r)} {R},
\f]

where \f$R\f$ is the size of a reduction dimension.

Lp-norm:

\f[
    \dst(f) = \root p \of {\mathop{eps\_op}(\sum\limits_{r}|src(r)|^p, eps)},
\f]

where \f$eps\_op\f$ can be max and sum.

Lp-norm-power-p:

\f[
    \dst(f) = \mathop{eps\_op}(\sum\limits_{r}|src(r)|^p, eps),
\f]

where \f$eps\_op\f$ can be max and sum.

### Dynamic quantization

The `reduction_dynamic_quantize` algorithm performs symmetric dynamic
quantization in one primitive dispatch. Unlike the other reduction algorithms,
its destination has the same shape as the source (or is empty in compute-only
mode). The reduction statistics are internal; the visible auxiliary output is
the `f32` quantization scale. The reduction constructor's `p` and `eps`
parameters must both be zero.

The scale recipe is specified with a destination quantization attribute:

```cpp
primitive_attr attr;
attr.set_scales(DNNL_ARG_DST, mask, groups, memory::data_type::f32,
        false, quantization_mode::dynamic_fp);
```

The scale is returned through
`DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST`.

For each group \f$g\f$, quantization computes `scale = amax / 127`:

\f[
    scale_g = \max_{i \in g}|src_i| / 127, \qquad
    dst_i = clamp(round(src_i / scale_g), -127, 127).
\f]

The CPU implementation supports 2D sources. `mask` and `groups` encode
per-tensor, per-row (per-token), per-column, grouped-row, or grouped-column
quantization. An empty destination memory descriptor selects compute-only mode,
which writes the scale without producing a quantized tensor.

For source `[M, N]`, the supported recipes are:
- per-tensor: `mask = 0`, `groups = {}`;
- per-row: `mask = 1 << 0`, `groups = {}`;
- per-column: `mask = 1 << 1`, `groups = {}`;
- grouped columns: `mask = 3`, `groups = {1, group_size_n}`;
- grouped rows: `mask = 3`, `groups = {group_size_m, 1}`.

### Notes

 * The reduction primitive requires the source and destination tensors to have
   the same number of dimensions.
 * Reduction dimensions are of size 1 in a destination tensor, except for
   `reduction_dynamic_quantize`, whose reduction statistics are auxiliary
   outputs and whose quantized destination preserves the source shape.
 * The reduction primitive does not have a notion of forward or backward
   propagations.
 * For Lp-norm algorithms, the parameter \f$p\f$ must be a finite value
   greater than or equal to 1.0.

## Execution Arguments

When executed, the inputs and outputs should be mapped to an execution
argument index as specified by the following table.

| Argument                    | Index                                                                     | Type         |
|-----------------------------|---------------------------------------------------------------------------|--------------|
| \src                        | DNNL_ARG_SRC                                                              | Input        |
| \dst                        | DNNL_ARG_DST                                                              | Output       |
| Dynamic scale               | DNNL_ARG_ATTR_SCALES \| DNNL_ARG_DST                                      | Output       |
| \f$\text{binary post-op}\f$ | DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_post_op_position) \| DNNL_ARG_SRC_1 | Input        |
| \                           | DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_post_op_position) \| DNNL_ARG_SRC_2 | Input        |
| [scratchpad]                | DNNL_ARG_SCRATCHPAD                                                       | Output       |

[scratchpad]: @ref dev_guide_attributes_scratchpad

## Implementation Details

### General Notes
 * The \dst memory format can be either specified explicitly or by
   #dnnl::memory::format_tag::any (recommended), in which case the primitive
   will derive the most appropriate memory format based on the format of the
   source tensor.

### Post-Ops and Attributes

The following attributes are supported:

| Type    | Operation                                      | Description                                                                    | Restrictions                        |
|:--------|:-----------------------------------------------|:-------------------------------------------------------------------------------|:------------------------------------|
| Post-op | [Sum](@ref dnnl::post_ops::append_sum)         | Adds the operation result to the destination tensor instead of overwriting it. |                                     |
| Post-op | [Eltwise](@ref dnnl::post_ops::append_eltwise) | Applies an @ref dnnl_api_eltwise operation to the result.                      |                                     |
| Post-op | [Binary](@ref dnnl::post_ops::append_binary)   | Applies a @ref dnnl_api_binary operation to the result                         | General binary post-op restrictions |
| Attribute | [Scales](@ref dnnl::primitive_attr::set_scales) | Defines dynamic quantization and returns scales | `reduction_dynamic_quantize` only |

### Data Types Support

The source and destination tensors may have `f32`, `bf16`, `f16`, `s8`, or `u8` data
types.

For `reduction_dynamic_quantize`, the source may be `f32`, `bf16`, or `f16`;
the full-mode destination is `s8`, and the dynamic scale output is `f32`.
See @ref dev_guide_data_types page for more details.

### Data Representation

#### Sources, Destination

The reduction primitive works with arbitrary data tensors. There is no special
meaning associated with any of the dimensions of a tensor.

## Implementation Limitations

1. Refer to @ref dev_guide_data_types for limitations related to data types
   support.

2. **GPU**
   - Only tensors of 6 or fewer dimensions are supported.
   - `reduction_dynamic_quantize` is not implemented.

3. **Dynamic quantization**
   - The portable CPU implementation supports all documented granularities and
     compute-only mode.
   - The optimized x64 implementation requires AVX-512 and handles dense 2D
     per-row and grouped-column full-mode cases. Other configurations fall
     through to the portable implementation.
   - The optimized x64 implementation is enabled for non-MSVC GCC- and
     Clang-compatible compiler frontends. Other compilers use the portable
     implementation.

## Performance Tips

1. Whenever possible, avoid specifying different memory formats for source
   and destination tensors.

## Examples

See @ref dev_guide_examples page for a complete list. Reduction examples are listed in the
[Tensor Operations](@ref examples_tensor_operations) section.

