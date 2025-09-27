Unary Fusion Patterns {#dev_guide_graph_unary_fusion_patterns}
==============================================================

## Overview

oneDNN supports various unary fusion patterns to optimize performance and
reduce memory bandwidth requirements. This document describes the supported
fusion patterns for Unary operations.

## Pattern Structure

oneDNN defines floating-point Unary fusion patterns as follows.
The blue nodes are required when defining a Unary fusion pattern while the
brown nodes are optional.

![Unary pattern](images/unary_pattern.png)

1. **Unary Operation**: Performs the corresponding unary operation for the
   `src` tensor. Refer to the Note in
   [Fusion Patterns](graph_fusion_patterns.html).
2. **Epilogue Subgraph**: Optional and can include the following operations:
   - Unary operations.
   - Binary operations: refer to the Note in
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
