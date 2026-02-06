Quantized Convolution Fusion Patterns {#dev_guide_graph_quantized_convolution_fusion_patterns}
==============================================================================================

## Overview

oneDNN supports both floating-point and quantized Convolution fusion patterns to
optimize performance and reduce memory bandwidth requirements. This document
describes the supported quantized fusion patterns for Convolution. For floating-point
Convolution fusion patterns, refer to
[Convolution Fusion Patterns](@ref dev_guide_graph_convolution_fusion_patterns)
for more details.

## Pattern Structure

oneDNN defines quantized Convolution fusion patterns as follows.
The blue nodes are required when defining a quantized Convolution fusion pattern
while the brown nodes are optional.

![quantized Convolution pattern](images/quantized_conv_pattern.png)

1. **Q2F Conversion Subgraph**: Converts `src` and `weights` tensors
   from quantized to floating-point. It can be one of the following
   subgraphs, while the last two subgraphs apply only to `weights`.
   See [Dequantize](@ref dev_guide_op_dequantize), [TypeCast](@ref dev_guide_op_typecast)
   and [Quantize](@ref dev_guide_op_quantize)
   operations in Graph API.

   ![q2f_conversion_subgraph](images/q2f_conversion_quantized_conv_matmul.png)

2. **F2F Conversion Subgraph**: Converts `bias` tensor from floating-point to
   another floating-point. It is constructed by a [TypeCast](@ref dev_guide_op_typecast)
   operation.

   ![f2f_conversion_subgraph](images/f2f_conversion.png)

3. **Convolution Operation**: Performs convolution between the `src` and
   `weights` tensors. The `bias` tensor is optional. See the [Convolution](@ref dev_guide_op_convolution)
   operation in the Graph API for more details.
4. **Epilogue Subgraph**: Optional and can include the following operations:
   - [BiasAdd](@ref dev_guide_op_biasadd) operation.
   - Binary and Unary operations: refer to the Note in
     [Fusion Patterns](graph_fusion_patterns.html).

   Combination Rules:

   ![epilogue subgraph](images/epilogue_subgraph_general_2.png)

   - **BiasAdd**: If present, must be the first op in the epilogue subgraph and
     can only appear once.
   - N=20, 0 to 20 Binary or Unary operations are supported in the epilogue
     subgraph.

5. **F2F/F2Q Conversion Subgraph**: Converts the output
   tensor from floating-point to floating-point or quantized data type. It can
   be one of the following subgraphs. See [TypeCast](@ref dev_guide_op_typecast)
   and [Quantize](@ref dev_guide_op_quantize) operations in Graph API.

   ![f2q_conversion_subgraph](images/f2q_conversion_quantized_conv.png)


## Data Types

oneDNN supports the following combinations of data types for src, weights, bias
and dst:

| src   | weights | bias         | dst                |
| :---- | :------ | :----------- | :----------------- |
| u8,s8 | s8,f32  | f32,bf16,f16 | u8,s8,bf16,f16,f32 |

The definition of the data types and support status on different CPU and GPU
platforms follow the general description in the [Data Types Guide](@ref dev_guide_data_types).

## Implementation Limitations

1. F2Q Conversion Subgraph used for `dst` tensor only supports
   bf16 to f32 data type conversion.

## Implementation Notes

Post-binary Add operations in the epilogue subgraph support in-place operations when
the post-binary Add is the last operation in the epilogue subgraph and the `dst` output
shape is identical and data type size is the same as the binary Add input.
In case of an in-place operation, the original input data will be overwritten.
Use in-place operations whenever possible for performance.

## Example

oneDNN provides a [quantized Convolution
example](https://github.com/uxlfoundation/oneDNN/tree/main/examples/graph/cpu_inference_int8.cpp)
demonstrating how to construct a typical quantized Convolution pattern with oneDNN
Graph API on CPU.
