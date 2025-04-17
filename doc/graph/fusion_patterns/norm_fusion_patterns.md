Norm Fusion Patterns {#dev_guide_graph_norm_fusion_patterns}
===========================================================

## Overview

The Norm category for inference includes operations such as:
GroupNorm, LayerNorm and BatchNormInference.

oneDNN supports various Norm fusion patterns to optimize performance and
reduce memory bandwidth requirements. This document describes the supported
fusion patterns for Norm.

## Pattern Structure

oneDNN defines floating-point Norm fusion patterns as follows.
The blue parts are required when defining a Norm fusion pattern while the
brown parts are optional.

![Norm pattern](images/norm_pattern.png)

1. **Norm Operation**: Performs the corresponding norm operation for the `src`
   tensor. See the [GroupNorm](@ref dev_guide_op_groupnorm),
   [LayerNorm](@ref dev_guide_op_layernorm), [BatchNormInference](@ref dev_guide_batch_normalization)
   operations in the Graph API for more details.
2. **F2F Conversion Subgraph**: Converts the output tensor from floating-point to
   another floating-point. It is constructed by a [TypeCast](@ref
   dev_guide_op_typecast) operation.

   ![f2f_conversion_subgraph](images/f2f_conversion.png)

3. **Post-Op Subgraph**: Optional and can include the following operations:
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

   Combination Rules:

   - 1 to 4 binary/unary operations are supported in the post-op subgraph.

4. **F2Q Conversion Subgraph**: Converts the output
   tensor from floating-point to quantized data type. It can
   be one of the following subgraphs. It is constructed by a
   [Quantize](@ref dev_guide_op_quantize) operation.

   ![f2q_conversion_subgraph_1](images/f2q_conversion_1.png)

## Data Types

oneDNN supports the following combinations of data types for src and output:

| src           | output             |
| :------------ | :----------------- |
| bf16,f16,f32  | u8,s8,bf16,f16,f32 |

The definition of data types and their support status on different CPU and GPU
platforms follow the general description in the [Data Types Guide](@ref
dev_guide_data_types).

## Implementation Limitations

1. BatchNormInference:
   1. The Post-Op Subgraph only supports ReLU, and if present, can only appear once.
   2. F2F and F2Q Conversion Subgraphs are not supported.
