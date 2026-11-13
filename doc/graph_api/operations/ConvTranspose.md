ConvTranspose {#dev_guide_op_convtranspose}
===========================================

## General

ConvTranspose operation performs the same computation as calculating the
gradient with regard to \src of Convolution operation. To see the difference
visually, you can go to
[visualization page](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md#transposed-convolution-animations).

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
| [weights_format](@ref dnnl::graph::op::attr::weights_format) | Controls how to interpret the shape of `weights`                                  | string     | `IOX`, `XOI` (default)                                               | Optional             |

@note If `auto_pad` is set to `VALID`, `SAME_UPPER`, or `SAME_LOWER`, the
attributes `pads_begin` and `pads_end` are ignored. `VALID` means no padding.
For `SAME_UPPER` and `SAME_LOWER`, the total padding along each spatial axis
is the dilated kernel size minus the stride, clamped to zero. Taking the
height dimension as an example, without padding the height of `dst` is
\f$OH = SH \cdot (IH - 1) + DKH + output\_padding_h\f$, where
\f$IH\f$ is the height of `src`, \f$SH\f$ is the stride, and
\f$DKH = 1 + (KH - 1) \cdot DH\f$ is the dilated kernel height.
With \f$PH_{total} = \max(0, DKH - SH)\f$, the output height is
\f$OH = IH \cdot SH + output\_padding_h + \min(0, DKH - SH)\f$.
When \f$DKH \ge SH\f$, this reduces to \f$OH = IH \cdot SH + output\_padding_h\f$.
The total padding is split between the beginning and the end as described in
[Auto Padding](@ref dev_guide_op_convolution_auto_pad) of Convolution.

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `src`         | Required             |
| 1     | `weights`     | Required             |
| 2     | `bias`        | Optional             |

@note
The shape of \weights is
\f$(in\_channels / groups, out\_channels, spatial\_shape)\f$ for `IOX` format or
\f$(spatial\_shape, out\_channels, in\_channels / groups)\f$ for `XOI` format.
Both \f$in\_channels\f$ and \f$out\_channels\f$ must be divisible by *groups*
attribute.

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `dst`         | Required             |

## Supported data types

ConvTranspose operation supports the following data type combinations.

| Src  | Weights | Bias | Dst  |
|:-----|:--------|:-----|:-----|
| f32  | f32     | f32  | f32  |
| bf16 | bf16    | bf16 | bf16 |
| f16  | f16     | f16  | f16  |
