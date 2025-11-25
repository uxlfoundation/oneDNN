GELU {#dev_guide_op_gelu}
=========================

## General

GELU operation applies following formula on every element of \src tensor (the
variable names follow the standard @ref dev_guide_conventions):

\f[ dst = 0.5 * src * (1.0 + erf(src) / \sqrt2) \f]

When the attribute `mode` is specified as `gelu_tanh`, an approximation
implementation is used:

\f[ dst = 0.5 * src * (1.0 + \tanh[\sqrt{\frac{2}{\pi}} (src + 0.044715 * s^3)]) \f]

## Operation attributes

| Attribute Name                             | Description                             | Value Type | Supported Values                  | Required or Optional |
|:-------------------------------------------|:----------------------------------------|:-----------|:----------------------------------|:---------------------|
| [mode](@ref dnnl::graph::op::attr::mode)   | Specifies the computation mode of GELU. | string     | `gelu_erf` (default), `gelu_tanh` | Optional             |

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

GELU operation supports the following data type combinations.

| Src  | Dst  |
|:-----|:-----|
| f32  | f32  |
| f16  | f16  |
| bf16 | bf16 |

## Implementation Notes

GELU supports in-place operations, meaning that `src` can be used as both input
and output (`dst`). In case of in-place operation, the original `src` data will be
overwritten. Use in-place operations whenever possible for performance.
