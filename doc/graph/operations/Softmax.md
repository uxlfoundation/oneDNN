SoftMax {#dev_guide_op_softmax}
===============================

## General

SoftMax operation applies the following formula on every element of \src tensor
(the variable names follow the standard @ref dev_guide_conventions):

\f[ dst_i = \frac{exp(src_i)}{\sum_{j=1}^{C} exp(src_j)} \f]
where \f$ C \f$ is a size of tensor along axis dimension.

## Operation attributes

| Attribute Name                           | Description                                               | Value Type | Supported Values                     | Required or Optional |
|:-----------------------------------------|:----------------------------------------------------------|:-----------|:-------------------------------------|:---------------------|
| [axis](@ref dnnl::graph::op::attr::axis) | Represents the axis from which the SoftMax is calculated. | s64        | Arbitrary s64 value (`1` in default) | Optional             |
| [mode](@ref dnnl::graph::op::attr::mode) | Specifies the computation mode of SoftMax                 | string     | `none` (default), `inf_as_zero`      | Optional             |

When the operation attribute `mode` is not set or set to `none`, the operation
performs the normal SoftMax calculation. In this case, the operation will
generate `NaN` if all the input elements are `-infinity` along the `axis`
dimension. To prevent this, you can set the attribute to `inf_as_zero` so that
the operation generates zeros for `-infinity` inputs. 

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

SoftMax operation supports the following data type combinations.

| Src  | Dst             |
|:-----|:----------------|
| f32  | f32, bf16, f16  |
| bf16 | bf16            |
| f16  | f16             |
