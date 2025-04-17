Convolution Fusion Patterns {#dev_guide_graph_convolution_fusion_patterns}
===========================================================

## Overview

oneDNN supports both floating-point and quantized Convolution fusion patterns to
optimize performance and reduce memory bandwidth requirements. This document
describes the supported floating-point fusion patterns for Convolution. For quantized
Convolution fusion patterns, refer to [Quantized Convolution Fusion Patterns](@ref
dev_guide_graph_quantized_convolution_fusion_patterns) for more details.

## Pattern Structure
### Pattern Structure for Inference

oneDNN defines floating-point Convolution fusion patterns for inference as follows.
The blue parts are required when defining a Convolution fusion pattern while the
brown parts are optional.

![Convolution pattern](images/conv_pattern.png)

1. **Convolution Operation**: Performs convolution between the `src` and
   `weights` tensors. The `bias` tensor is optional. See the [Convolution](@ref
   dev_guide_op_convolution) operation in the Graph API for more details.
2. **Post-Op Subgraph**: Optional and can include the following operations:
   - [BiasAdd](@ref dev_guide_op_biasadd) operation.
   - [BatchNormInference](@ref dev_guide_op_batchnorminference) operation.
   - [Convolution](@ref dev_guide_op_convolution) operation.
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

   Combination Rules:

   - 1 to 4 binary/unary operations are supported in the post-op subgraph.
   - **BiasAdd**: If present, must be the first post-op and can only appear once.
   - **BatchNormInference**: If present, must precede binary/unary operations
     (if present) and can only appear once.
   - **Convolution**: If present, is a Depthwise Convolution which can only be
     fused with 1x1 Convolution and can only appear once.

3. **F2F Conversion Subgraph**: Converts `output` tensor from floating-point to
   another floating-point. It is constructed by a [TypeCast](@ref
   dev_guide_op_typecast) operation.

   ![f2f_conversion_subgraph](images/f2f_conversion.png)

### Pattern Structure for Training

oneDNN defines floating-point Convolution fusion patterns for training as follows.
The blue parts are required when defining a Convolution fusion pattern while the
brown parts are optional.

![ConvolutionBackwardWeights pattern](images/conv_bwd_pattern.png)

1. **ConvolutionBackwardWeights operation**: Accepts `src`, `diff_dst` and
   optional `weights_shape` as inputs, and computes the gradients for weights.
   See the [ConvolutionBackwardWeights](@ref
   dev_guide_op_convolutionbackwardweights) operation in the Graph API for more
   details.
2. **BiasAddBackward Operation**: Computes the gradients for bias based on
   `diff_dst`. See the [BiasAddBackward](@ref dev_guide_op_biasaddbackward)
   operation in the Graph API for more details.
3. The two operations share the same input of `diff_dst`.

## Data Types

oneDNN supports floating-point Convolution patterns with data types `f32`, `bf16`,
and `f16`. You can specify the data type via the input and output logical
tensors' data type fields for each operation. oneDNN also supports limited
mixed-precision in floating-point Convolution patterns.

The definition of data types and their support status on different CPU and GPU
platforms follow the general description in the [Data Types Guide](@ref
dev_guide_data_types).

## Implementation Limitations

1. Convolution as a post op (Depthwise Convolution) is not supported on GPU.
2. F2F Conversion Subgraph used for `output` tensor in inference only supports
   bf16 to f32 data type conversion.

## Example

oneDNN provides a [CPU Convolution
example](https://github.com/oneapi-src/oneDNN/tree/main/examples/graph/cpu_getting_started.cpp)
and a [GPU Convolution example](https://github.com/oneapi-src/oneDNN/tree/main/examples/graph/sycl_getting_started.cpp)
demonstrating how to construct a typical floating-point Convolution pattern for
inference with oneDNN Graph API on CPU and GPU.
