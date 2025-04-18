Quantized MatMul Fusion Patterns {#dev_guide_graph_quantized_matmul_fusion_patterns}
===========================================================

## Overview

oneDNN supports both floating-point and quantized MatMul fusion patterns to
optimize performance and reduce memory bandwidth requirements. This document
describes the supported quantized fusion patterns for MatMul. For floating-point
MatMul fusion patterns, refer to [MatMul Fusion Patterns](@ref
dev_guide_graph_matmul_fusion_patterns) for more details.

## Pattern Structure

oneDNN defines quantized MatMul fusion patterns as follows.
The blue parts are required when defining a MatMul fusion pattern while the brown
parts are optional.

![quantized MatMul pattern](images/quantized_matmul_pattern.png)

1. **Q2F Conversion Subgraph**: Converts `src` and `weights` tensors
   from quantized to floating-point. It can be one of the following
   subgraphs, while the last two subgraphs apply only to `weights`.
   See [Dequantize](@ref dev_guide_op_dequantize), [TypeCast](@ref
   dev_guide_op_typecast) and [Quantize](@ref dev_guide_op_quantize)
   operations in Graph API.

   ![q2f_conversion_subgraph_1](images/q2f_conversion_1.png)
   ![q2f_conversion_subgraph_2](images/q2f_conversion_2.png)
   ![q2f_conversion_subgraph_3](images/q2f_conversion_3.png) ![q2f_conversion_subgraph_4](images/q2f_conversion_4.png)

2. **F2F Conversion Subgraph**: Converts `bias` tensor from floating-point to
   another floating-point. It is constructed by a [TypeCast](@ref
   dev_guide_op_typecast) operation.

   ![f2f_conversion_subgraph](images/f2f_conversion.png)

3. **MatMul Operation**: Performs matrix multiplication between the `src` and
   `weights` tensors. The `bias` tensor is optional. See the [MatMul](@ref
   dev_guide_op_matmul) operation in the Graph API for more details.
4. **Post-Op Subgraph**: Optional and can include the following operations:
   - [BiasAdd](@ref dev_guide_op_biasadd) operation.
   - **Binary Operations**: [Add](@ref dev_guide_op_add),
     [Subtract](@ref dev_guide_op_subtract), [Maximum](@ref dev_guide_op_maximum),
     [Minimum](@ref dev_guide_op_minimum), [Multiply](@ref dev_guide_op_multiply),
     [Divide](@ref dev_guide_op_divide).
   - **Unary Operations**: [Abs](@ref dev_guide_op_abs),
     [Clamp](@ref dev_guide_op_clamp), [Elu](@ref dev_guide_op_elu),
     [Exp](@ref dev_guide_op_exp), [GELU](@ref dev_guide_op_gelu),
     [HardSigmoid](@ref dev_guide_op_hardsigmoid), [HardSwish](@ref dev_guide_op_hardswish),
     [LeakyReLU](@ref dev_guide_op_leakyrelu), [Log](@ref dev_guide_op_log),
     [Mish](@ref dev_guide_op_mish), [Sigmoid](@ref dev_guide_op_sigmoid),
     [SoftPlus](@ref dev_guide_op_softplus), [ReLU](@ref dev_guide_op_relu),
     [Round](@ref dev_guide_op_round), [Sqrt](@ref dev_guide_op_sqrt),
     [Square](@ref dev_guide_op_square), [Tanh](@ref dev_guide_op_tanh).
   - [Select](@ref dev_guide_op_select) operation.
   - **Data Manipulation Operations**: [StaticTranspose](@ref dev_guide_op_statictranspose),
     [StaticReshape](@ref dev_guide_op_staticreshape), [Reorder](@ref dev_guide_op_reorder).

   Combination rules:

   - 1 to 4 binary/unary operations are supported in the post-op subgraph.
   - **BiasAdd**: If present, must be the first post-op and can only appear once.
   - **Select**: Must precede binary/unary operations and can only appear once.
   - **Reshape**: If present, must precede or follow the Transpose operation.
   - **Reorder**: If present, must follow the Transpose operation.

5. **F2F/F2Q Conversion Subgraph**: Converts the output
   tensor from floating-point to floating-point or quantized data type. It can be one of the
   following subgraphs, the last two subgraphs are implementations for SmoothQuant
   [1]. See [TypeCast](@ref
   dev_guide_op_typecast), [Quantize](@ref dev_guide_op_quantize) and [Multiply](@ref dev_guide_op_multiply) operations
   in Graph API.

   ![f2f_conversion_subgraph](images/f2f_conversion.png)
   ![f2q_conversion_subgraph_1](images/f2q_conversion_1.png) ![f2q_conversion_subgraph_2](images/f2q_conversion_2.png) ![f2q_conversion_subgraph_3](images/f2q_conversion_3.png)
   ![f2q_conversion_subgraph_4](images/f2q_conversion_4.png)

## Data Types

oneDNN supports the following combinations of data types for src, weights, bias
and output:

| src   | weights | bias         | output             |
| :---- | :------ | :----------- | :----------------- |
| u8,s8 | s8,f32  | f32,bf16,f16 | u8,s8,bf16,f16,f32 |

The definition of the data types and support status on different CPU and GPU
platforms follow the general description in the [Data Types Guide](@ref
dev_guide_data_types).

## Limitations

- F2F Conversion Subgraph used for `bias` tensor only supports f32 to bf16 data
  type conversion.

## Reference

[1] SmoothQuant, https://arxiv.org/abs/2211.10438