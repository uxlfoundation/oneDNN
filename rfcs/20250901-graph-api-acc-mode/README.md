# Graph API: Add Accumulation Mode Support

## Overview

By default, oneDNN uses `f32` for floating point computation or `s32` for
integer computation as the accumulation data types. However, on some platforms,
using smaller accumulation data types (e.g., `f16`) can result in additional
speed improvement.

Introducing `f16` as the accumulation data type can achieve up to a 2x speedup
for `f16` inference while maintaining a high level of accuracy. It is important
to note that `f16` accumulation is not suitable for training, which requires
full `f32` precision.

For more information about oneDNN accumulation data type, please refer to
[documentation](https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_accumulation_mode.html)
and the
[RFC](https://github.com/uxlfoundation/oneDNN/blob/rfcs/rfcs/20230118-f16-accumulation/README.md)
for the motivation and design details of primitive API.

This document discusses and proposes the corresponding accumulation mode support
for oneDNN Graph API.

## Control granularity

In general, users can control the computation behavior of graph API at 4 levels:

- Global setting
- Graph-level setting
- Partition-level setting
- Operation-level setting

Settings at different level have different scope of effect and lead to different
computation behaviors. As for accumulation mode, users may require different
operations to work with different accumulation modes, even when they are fused
into a single partition. For example, in Scaled-dot product attention (SDPA),
the dot-product of QK can work with `f16` accumulation while the dot-product of
VS works with `f32` accumulation for better computation speed and reasonable
model quality (compared with both dot-products in `f16` accumulation).

As global setting, graph-level setting, and partition-level setting cannot
achieve the above purpose, related API options are not discussed here. To
support operation-level setting of accumulation mode, especially for SDPA, we
provide the below proposal.

## Proposals

### Add a new `accumulation_mode` attribute to MatMul operation

To control the accumulation mode of the MatMul operation, we propose adding
`dnnl_graph_op_attr_accumulation_mode` to the C API enum type
`dnnl_graph_op_attr_t` and `op::attr::accumulation_mode` to the corresponding
C++ API.

The attribute name `accumulation_mode` here is consistent with the name of
`accumulation_mode` of primitive attribute in primitive API.

This attribute can be set on MatMul operation during its creation. The attribute
value is a string and can be one of the following:

- strict
- relaxed
- any
- f32
- s32
- f16

The definitions of each value can be found in oneDNN
[documentation](https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_accumulation_mode.html).

This attribute is optional. When it's not set, the default accumulation mode is
`strict`.

For primitive-based implementation in the backend, the accumulation mode on
MatMul operation will map directly to the accumulation mode of the primitive
attribute when creating matmul primitive.

Example: creating a `f16` MatMul with `f16` accumulation:

```c++
auto src_0 = logical_tensor(ID0, data_type::f16, ...);
auto src_1 = logical_tensor(ID1, data_type::f16, ...);
auto dst   = logical_tensor(ID2, data_type::f16, ...);
auto mm    = op(ID3, op::kind::MatMul, "matmul");
mm.set_attr<bool>(op::attr::transpose_a, false);
mm.set_attr<bool>(op::attr::transpose_b, false);
mm.set_attr<std::string>(op::attr::accumulation_mode, "f16");
mm.add_inputs({src_0, src_1});
mm.add_outputs({dst});
```

When creating SDPA pattern, you can create the MatMul of QK with `f16`
accumulation while leaving the MatMul of VS with `strict` accumulation (or not
set) to achieve the computation behavior described above. This does not affect
the behavior of existing SDPA integration code where accumulation mode is not
set for both MatMul operations and relies on the default `strict` mode.

Similar to other operation attributes, MatMul's `accumulation_mode` and its
value will be serialized into the JSON file during graph dump. Benchdnn will
load the JSON file and check the attribute for proper testing. 

Please note that the proposal focuses on adding `accumulation_mode` attribute to
MatMul, as required by SDPA pattern definition. In the future, the same
attribute and value options also can be supported by other operations (eg.
Convolution, Softmax, etc.) in the same way. Before that, setting
`accumulation_mode` to other operations will lead to failures during the
graph/operation creation.

Please also note that, `f16` (and other) accumulation modes may not be supported
on all platforms supported by oneDNN. If the accumulation mode is not
supported/implemented on the target platform, partition compilation may fail.

END
