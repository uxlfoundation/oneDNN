ConvolutionBackwardData {#dev_guide_op_convolutionbackwarddata}
===============================================================

## General

ConvolutionBackwardData operation accepts \f$\diffdst\f$, \weights and optional
dst shape as inputs, and compute the \f$\diffsrc\f$.

## Operation attributes

| Attribute Name                                               | Description                                                                       | Value Type | Supported Values                                                     | Required or Optional |
|:-------------------------------------------------------------|:----------------------------------------------------------------------------------|:-----------|:---------------------------------------------------------------------|:---------------------|
| [strides](@ref dnnl::graph::op::attr::strides)               | Controls the strides the weights tensor is moved when computing convolution       | s64        | A s64 list containing positive values                                | Required             |
| [pads_begin](@ref dnnl::graph::op::attr::pads_begin)         | Controls number of zeros to be add to the front/top/left of spatial dimensions    | s64        | A s64 list containing non-negative values                            | Required             |
| [pads_end](@ref dnnl::graph::op::attr::pads_end)             | Controls number of zeros to be add to the back/bottom/right of spatial dimensions | s64        | A s64 list containing non-negative values                            | Required             |
| [dilations](@ref dnnl::graph::op::attr::dilations)           | Controls the amount of stretching the kernel before convolution                   | s64        | A s64 list containing positive values (>1 means dilated convolution) | Required             |
| [auto_pad](@ref dnnl::graph::op::attr::auto_pad)             | Controls how the padding is calculated                                            | string     | `None` (default), `SAME_UPPER`, `SAME_LOWER`, `VALID`                | Optional             |
| [output_padding](@ref dnnl::graph::op::attr::output_padding) | Adds additional amount of padding per each spatial axis in `dst`.                 | s64        | A s64 list containing non-negative values, all zeros by default      | Optional             |
| [groups](@ref dnnl::graph::op::attr::groups)                 | Controls how input channels and output channels are divided into                  | s64        | A positive s64 value, `1` by default                                 | Optional             |
| [data_format](@ref dnnl::graph::op::attr::data_format)       | Controls how to interpret the shape of `src` and `dst`.                           | string     | `NCX`, `NXC` (default)                                               | Optional             |
| [weights_format](@ref dnnl::graph::op::attr::weights_format) | Controls how to interpret the shape of `weights`.                                 | string     | `OIX`, `XIO` (default)                                               | Optional             |
| [dst_shape](@ref dnnl::graph::op::attr::dst_shape)           | Denotes the shape of the `dst` tensor.                                            | s64        | A s64 list containing positive values                                | Optional             |

@note If `auto_pad` is set to `VALID`, `SAME_UPPER`, or `SAME_LOWER`, the
attributes `pads_begin` and `pads_end` are ignored. The paddings are calculated
as described in [Auto Padding](@ref dev_guide_op_convolution_auto_pad) of
Convolution, with the input spatial sizes taken from `dst_shape`. Hence they are
the same as the paddings of the forward Convolution whose `src` has the shape
`dst_shape`.

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0   | `diff_dst`      | Required             |
| 1   | `weights`       | Required             |
| 2   | `dst_shape`     | Optional             |

@note
The shape of \weights is
\f$(out\_channels, in\_channels / groups, spatial\_shape)\f$ for `OIX` format or
\f$(spatial\_shape, in\_channels / groups, out\_channels)\f$ for `XIO` format.
Both \f$in\_channels\f$ and \f$out\_channels\f$ must be divisible by *groups*
attribute.

@note Either `dst_shape` input or `dst_shape` attribute should be provided. If
both provided, `dst_shape` input will precede over `dst_shape` attribute.

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `diff_src`    | Required             |

## Supported data types

ConvolutionBackwardData operation supports the following data type combinations.

| Diff_dst | Weights | Diff_src | Dst_shape |
|:---------|:--------|:---------|:----------|
| f32      | f32     | f32      | s32       |
| bf16     | bf16    | bf16     | s32       |
| f16      | f16     | f16      | s32       |
