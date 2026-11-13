Convolution {#dev_guide_op_convolution}
=======================================

## General

Convolution operation performs the convolution between src tensor and weight
tensor, which is defined as by the following formulas. Variable names follow the
standard @ref dev_guide_conventions.

Let \src, \weights and \dst tensors have shape \f$N \times IC \times IH \times
IW\f$, \f$OC \times IC \times KH \times KW\f$, and \f$N \times OC \times OH
\times OW\f$ respectively.

Furthermore, let the remaining convolution parameters be:

| Parameter                        | Depth      | Height     | Width      | Comment                                                                                |
|:---------------------------------|:-----------|:-----------|:-----------|:---------------------------------------------------------------------------------------|
| Paddings: Front, top, and left   | \f$PD_L\f$ | \f$PH_L\f$ | \f$PW_L\f$ | In the attributes we use `pads_begin` to indicate the corresponding vector of paddings |
| Padding: Back, bottom, and right | \f$PD_R\f$ | \f$PH_R\f$ | \f$PW_R\f$ | In the attributes we use `pads_end` to indicate the corresponding vector of paddings   |
| Stride                           | \f$SD\f$   | \f$SH\f$   | \f$SW\f$   | In the attributes we use `strides` to indicate the corresponding vector of strides     |
| Dilation                         | \f$DD\f$   | \f$DH\f$   | \f$DW\f$   | In the attributes we use `dilations` to indicate the corresponding vector of dilations |

To further simplify the formulas, we assume that the attribute `data_format` and
`weights_format` are set to `NCX` and `OIX` respectively. `NCX` means the fist
axis represents batch dimension, the second axis represents channel dimension
and the rest represents spatial dimensions. `OIX` means the first axis
represents output channel dimension, the second axis represents input channel
dimension and the rest represents weights spatial dimensions.

### Regular Convolution

This is the same as the formula in
[Convolution primitive](@ref dev_guide_convolution).

\f[\dst(n, oc, oh, ow) =  \bias(oc) \\
    + \sum_{ic=0}^{IC-1}\sum_{kh=0}^{KH-1}\sum_{kw=0}^{KW-1}
        \src(n, ic, oh \cdot SH + kh - PH_L, ow \cdot SW + kw - PW_L)
        \cdot
        \weights(oc, ic, kh, kw).\f]

Here:

- \f$OH = \left\lfloor{\frac{IH - KH + PH_L + PH_R}{SH}} \right\rfloor + 1,\f$

- \f$OW = \left\lfloor{\frac{IW - KW + PW_L + PW_R}{SW}} \right\rfloor + 1.\f$

### Convolution with Groups

The attribute `groups` is set to \f$>1\f$.

\f[
    \dst(n, g \cdot OC_G + oc_g, oh, ow) =
        \bias(g \cdot OC_G + oc_g) \\
        +
        \sum_{ic_g=0}^{IC_G-1}\sum_{kh=0}^{KH-1}\sum_{kw=0}^{KW-1}
            \src(n, g \cdot IC_G + ic_g, oh \cdot SH + kh - PH_L,
                    ow \cdot SW + kw - PW_L)
            \cdot
            \weights(g, oc_g, ic_g, kh, kw),
\f]

where

- \f$IC_G = \frac{IC}{G}\f$,

- \f$OC_G = \frac{OC}{G}\f$, and

- \f$oc_g \in [0, OC_G).\f$

### Convolution with Dilation

The attribute `dilation` contains the element which is \f$>1\f$. Refer to the
[visualization page](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md#dilated-convolution-animations)
for dilated convolution.

\f[
    \dst(n, oc, oh, ow) =
        \bias(oc) \\
        +
        \sum_{ic=0}^{IC-1}\sum_{kh=0}^{KH-1}\sum_{kw=0}^{KW-1}
            \src(n, ic, oh \cdot SH + kh \cdot DH - PH_L,
                    ow \cdot SW + kw \cdot DW - PW_L)
            \cdot
            \weights(oc, ic, kh, kw).
\f]

Here:

- \f$OH = \left\lfloor{\frac{IH - DKH + PH_L + PH_R}{SH}}
        \right\rfloor + 1,\f$ where \f$DKH = 1 + (KH - 1) \cdot DH\f$, and

- \f$OW = \left\lfloor{\frac{IW - DKW + PW_L + PW_R}{SW}}
        \right\rfloor + 1,\f$ where \f$DKW = 1 + (KW - 1) \cdot DW\f$.

@anchor dev_guide_op_convolution_auto_pad
### Auto Padding

If the attribute `auto_pad` is set to `VALID`, `SAME_UPPER`, or `SAME_LOWER`,
the attributes `pads_begin` and `pads_end` are ignored and the paddings are
calculated automatically. Taking the height dimension as an example:

- `VALID`: no padding is applied, i.e. \f$PH_L = PH_R = 0\f$.

- `SAME_UPPER` and `SAME_LOWER`: the input is padded so that
  \f$OH = \left\lceil IH / SH \right\rceil\f$.

For `SAME_UPPER` and `SAME_LOWER`, the total padding is

\f[
    PH_{total} = \max(0, (\left\lceil IH / SH \right\rceil - 1) \cdot SH + DKH - IH),
\f]

where \f$DKH = 1 + (KH - 1) \cdot DH\f$. The total padding is split between
the two sides as equally as possible. When it is odd, the extra padding is
added at the end for `SAME_UPPER` and at the beginning for `SAME_LOWER`:

\f[
    PH_L = \begin{cases}
        \left\lfloor PH_{total} / 2 \right\rfloor & \text{for SAME\_UPPER} \\
        \left\lceil PH_{total} / 2 \right\rceil & \text{for SAME\_LOWER}
    \end{cases}, \quad
    PH_R = PH_{total} - PH_L.
\f]

The depth and width paddings are calculated in the same way.

## Operation attributes

| Attribute Name                                               | Description                                                                                        | Value Type | Supported Values                                                     | Required or Optional |
|:-------------------------------------------------------------|:---------------------------------------------------------------------------------------------------|:-----------|:---------------------------------------------------------------------|:---------------------|
| [strides](@ref dnnl::graph::op::attr::strides)               | Controls the strides the weights tensor is moved when computing convolution                        | s64        | A s64 list containing positive values                                | Required             |
| [pads_begin](@ref dnnl::graph::op::attr::pads_begin)         | Controls number of zeros to be add to the front/top/left of spatial dimensions                     | s64        | A s64 list containing non-negative values                            | Required             |
| [pads_end](@ref dnnl::graph::op::attr::pads_end)             | Controls number of zeros to be add to the back/bottom/right of spatial dimensions                  | s64        | A s64 list containing non-negative values                            | Required             |
| [dilations](@ref dnnl::graph::op::attr::dilations)           | Controls the amount of stretching the kernel before convolution                                    | s64        | A s64 list containing positive values (>1 means dilated convolution) | Required             |
| [auto_pad](@ref dnnl::graph::op::attr::auto_pad)             | Controls how the padding is calculated. See [Auto Padding](@ref dev_guide_op_convolution_auto_pad) | string     | `None` (default), `SAME_UPPER`, `SAME_LOWER`, `VALID`                | Optional             |
| [groups](@ref dnnl::graph::op::attr::groups)                 | Controls how input channels and output channels are divided into                                   | s64        | A positive s64 value, `1` by default                                 | Optional             |
| [data_format](@ref dnnl::graph::op::attr::data_format)       | Controls how to interpret the shape of `src` and `dst`.                                            | string     | `NCX`, `NXC` (default)                                               | Optional             |
| [weights_format](@ref dnnl::graph::op::attr::weights_format) | Controls how to interpret the shape of `weights`                                                   | string     | `OIX`, `XIO` (default)                                               | Optional             |

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
\f$(out\_channels, in\_channels / groups, spatial\_shape)\f$ for `OIX` format or
\f$(spatial\_shape, in\_channels / groups, out\_channels)\f$ for `XIO` format.
Both \f$in\_channels\f$ and \f$out\_channels\f$ must be divisible by *groups*
attribute.

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `dst`         | Required             |

## Supported data types

Convolution operation supports the following data type combinations.

| Src  | Weights | Bias | Dst  |
|:-----|:--------|:-----|:-----|
| f32  | f32     | f32  | f32  |
| bf16 | bf16    | bf16 | bf16 |
| f16  | f16     | f16  | f16  |
