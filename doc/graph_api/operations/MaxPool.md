MaxPool {#dev_guide_op_maxpool}
===============================

## General

MaxPool operation performs the computation following the below formulas.
Variable names follow the standard @ref dev_guide_conventions.

\f[
    \dst(n, c, oh, ow) =
        \max\limits_{kh, kw}
        \left(
            \src(n, c, oh \cdot SH + kh \cdot (DH + 1) - PH_L, ow \cdot SW + kw \cdot (DW + 1) - PW_L)
        \right)
\f]

## Operation attributes

| Attribute Name                                             | Description                                                                       | Value Type | Supported Values                                                                  | Required or Optional |
|:-----------------------------------------------------------|:----------------------------------------------------------------------------------|:-----------|:----------------------------------------------------------------------------------|:---------------------|
| [strides](@ref dnnl::graph::op::attr::strides)             | Controls the strides the window is moved.                                         | s64        | A s64 list containing positive values                                             | Required             |
| [pads_begin](@ref dnnl::graph::op::attr::pads_begin)       | Controls number of zeros to be add to the front/top/left of spatial dimensions    | s64        | A s64 list containing non-negative values                                         | Required             |
| [pads_end](@ref dnnl::graph::op::attr::pads_end)           | Controls number of zeros to be add to the back/bottom/right of spatial dimensions | s64        | A s64 list containing non-negative values                                         | Required             |
| [kernel](@ref dnnl::graph::op::attr::kernel)               | Size of pooling window.                                                           | s64        | A s64 list containing positive values                                             | Required             |
| [rounding_type](@ref dnnl::graph::op::attr::rounding_type) | Controls how to do rounding.                                                      | string     | `floor` (default), `ceil`                                                        | Optional             |
| [auto_pad](@ref dnnl::graph::op::attr::auto_pad)           | Controls how the paddings are calculated.                                         | string     | `None` (default), `SAME_UPPER`, `SAME_LOWER`, `VALID`                             | Optional             |
| [dilations](@ref dnnl::graph::op::attr::dilations)         | Denotes the distance in width and height between elements in the window.          | s64        | A s64 list containing positive values, a list of `1`s (default) means no dilation | Optional             |
| [data_format](@ref dnnl::graph::op::attr::data_format)     | Controls how to interpret the shape of `src` and `dst`.                           | string     | `NCX`, `NXC` (default)                                                            | Optional             |

@note If `auto_pad` is set to `VALID`, `SAME_UPPER`, or `SAME_LOWER`, the
attributes `pads_begin` and `pads_end` are ignored. The paddings are calculated
as described in [Auto Padding](@ref dev_guide_op_convolution_auto_pad) of
Convolution, where the kernel size is given by the `kernel` attribute.

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `src`         | Required             |

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `dst`         | Required             |

## Supported data types

MaxPool operation supports the following data type combinations.

| Src  | Dst  |
|:-----|:-----|
| f32  | f32  |
| bf16 | bf16 |
| f16  | f16  |
