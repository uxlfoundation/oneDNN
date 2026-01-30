Interpolate Fusion Patterns {#dev_guide_graph_interpolate_fusion_patterns}
==========================================================================

## Overview

oneDNN supports various Interpolate fusion patterns to optimize performance and
reduce memory bandwidth requirements. This document describes the supported
fusion patterns for Interpolate.

## Pattern Structure

oneDNN defines floating-point Interpolate fusion patterns as follows.
The blue nodes are required when defining an Interpolate fusion pattern while the
brown nodes are optional.

![Interpolate pattern](images/interpolate_pattern.png)

1. **Interpolate Operation**: Performs interpolation for the `src` tensor at spatial
   dimensions. See the [Interpolate](@ref dev_guide_op_interpolate)
   operation in the Graph API for more details.
2. **Epilogue Subgraph**: Optional and can include the following operations:
   - Binary and Unary operations: refer to the Note in
     [Fusion Patterns](graph_fusion_patterns.html).

   Combination Rules:

   ![epilogue subgraph](images/epilogue_subgraph_general_1.png)

   - N=20, 0 to 20 Binary or Unary operations are supported in the epilogue
     subgraph.

## Data Types

oneDNN supports the following combinations of data types for src and dst:

| src          | dst          |
| :----------- | :----------- |
| f32,bf16,f16 | f32,bf16,f16 |

The definition of the data types and support status on different CPU and GPU
platforms follow the general description in the [Data Types Guide](@ref dev_guide_data_types).

## Implementation Limitations

1. The Interpolate operation only supports half_pixel coordinate_transformation_mode.

## Implementation Notes

Post-binary Add operations in the epilogue subgraph support in-place operations when
the post-binary Add is the last operation in the epilogue subgraph and the `dst` output
shape is identical and data type size is the same as the binary Add input.
In case of an in-place operation, the original input data will be overwritten.
Use in-place operations whenever possible for performance.
