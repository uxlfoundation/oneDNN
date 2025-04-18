Other Fusion Patterns {#dev_guide_graph_other_fusion_patterns}
===========================================================

## Overview

The Other category currently includes operations such as:
[Reorder](@ref dev_guide_op_reorder), [TypeCast](@ref dev_guide_op_typecast),
[Quantize](@ref dev_guide_op_quantize),
[StaticReshape](@ref dev_guide_op_staticreshape) and
[StaticTranspose](@ref dev_guide_op_statictranspose).

oneDNN supports specialized fusion patterns for Other operations to
optimize performance and reduce memory bandwidth requirements.

## Pattern Structure

![Other pattern](images/other_pattern.png)
