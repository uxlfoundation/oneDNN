# Graph API: Add Deterministic Mode Support

## Overview

Multiple applications require deterministic output to achieve reproducible
results and meet certification requirements (e.g., medical, avionics
applications). Popular deep learning frameworks like PyTorch and TensorFlow are
exposing deterministic modes to enable such use cases. Furthermore, these
frameworks rely on deterministic mode for their own validation processes (e.g.,
PyTorch torch.compile validation)[^1].

Currently, oneDNN primitive API already supports deterministic mode through the
`dnnl_primitive_attr_set_deterministic()` and
`dnnl_primitive_attr_get_deterministic()` APIs introduced in the primitive
deterministic mode RFC [^2]. However, the oneDNN Graph API lacks this capability,
creating inconsistency between the two APIs and limiting the use of Graph API in
applications that require deterministic execution.

This RFC proposes extending the oneDNN Graph API to support deterministic mode
at the graph level, ensuring that all operations within a graph execute
deterministically when this mode is enabled. Deterministic execution refers to
returning the bitwise same result when a graph is compiled and executed multiple
times on the same system (fixed hardware configuration) with the same
environment (fixed software environment) and identical inputs.

## Proposal

This RFC proposes adding deterministic mode support to the oneDNN Graph API
through the following methods:

### C API Extensions

The following C API functions will be added to `dnnl_graph.h`:

```c
/// Sets the deterministic mode for a graph.
///
/// @param graph The target graph.
/// @param deterministic Boolean value to enable (non-zero) or disable (zero) 
///     deterministic mode.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_graph_set_deterministic(
        dnnl_graph_graph_t graph, int deterministic);

/// Gets the deterministic mode for a graph.
///
/// @param graph The target graph.
/// @param deterministic Output deterministic mode value. Non-zero indicates
///     deterministic mode is enabled, zero indicates it is disabled.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_graph_get_deterministic(
        const_dnnl_graph_graph_t graph, int *deterministic);
```

### C++ API Extensions

The following methods will be added to the `dnnl::graph::graph` class in
`dnnl_graph.hpp`:

```cpp
/// Sets deterministic mode for the graph.
///
/// @param value Boolean value to enable or disable deterministic mode.
void set_deterministic(bool value) {
    error::wrap_c_api(dnnl_graph_graph_set_deterministic(get(), value),
            "could not set deterministic mode for graph");
}

/// Returns the deterministic mode value for the graph.
///
/// @returns True if deterministic mode is enabled, false otherwise.
bool get_deterministic() const {
    int result;
    error::wrap_c_api(dnnl_graph_graph_get_deterministic(get(), &result),
            "could not get deterministic mode for graph");
    return static_cast<bool>(result);
}
```

### Implementation Details

#### Graph-level Deterministic Mode

Similar to how `fpmath_mode` is managed, the deterministic mode will be set and
stored as a boolean attribute that can be modified before a graph object is
finalized. Once set, this mode affects all the operations in the graph:

When deterministic mode is enabled on a graph:

1. All partitions created from the graph will inherit the deterministic mode
   setting.
2. During partition compilation, the deterministic flag will be propagated to
   all underlying primitive attributes, ensuring that only deterministic
   implementations are selected during primitive creations.
3. Non-deterministic kernel implementations will be avoided in favor of
   deterministic alternatives, even if they may have slightly lower performance.

#### Default Behavior

The default value for deterministic mode will be `false` (disabled). This
maintains backward compatibility and ensures that existing performance
characteristics are preserved for applications that don't explicitly require
deterministic execution. This also aligns with the default behavior of primitive
API.

Users who need deterministic behavior must explicitly enable it by calling the
proposed setter methods.

#### Impact on Graph Serialization

The graph serialization functionality will be extended to include the
deterministic mode setting, similar to how `fpmath_mode` is currently
serialized. This ensures that deterministic mode is preserved when graphs are
serialized to JSON for latter deserialization and debugging.

### Examples

```cpp
#include "oneapi/dnnl/dnnl_graph.hpp"

using namespace dnnl::graph;

// Create a graph with deterministic mode
graph g(engine::kind::gpu);
g.set_deterministic(true);

// Add operations to the graph
op conv_op(0, op::kind::Convolution, "conv");
op relu_op(1, op::kind::ReLU, "relu");
// ... configure operations ...

g.add_op(conv_op);
g.add_op(relu_op);
g.finalize();

// Verify deterministic mode is enabled
assert(g.get_deterministic() == true);

// Get partitions - they will inherit deterministic mode
auto partitions = g.get_partitions();

// Compilation - create primitives with deterministic mode enabled in primitive attribute.
auto cp = partitions[0].compile(eng, inputs, outputs);
```

## Alternative Approaches

### Alternative 1: Operation-Level Deterministic Attributes

Instead of graph-level deterministic mode, we could add deterministic attributes
to individual operations within the graph.

```cpp
// for each operation in the graph, need to define the attribute.
op conv_op(0, op::kind::Convolution, "conv");
conv_op.set_attr<bool>(op::attr::deterministic, true);

op softmax_op(1, op::kind::SoftMax, "softmax");
softmax_op.set_attr<bool>(op::attr::deterministic, true);

g.add_op(conv_op);
g.add_op(softmax_op);
```

**Pros:**
- Fine-grained control over which operations require determinism
- Potentially better performance by only constraining operations that need
  determinism

**Cons:**
- More complex API and user experience
- Difficult to manage propagation through fusions
- Increased implementation complexity

### Alternative 2: Partition-Level Deterministic Mode at Compilation

Instead of setting deterministic mode at the graph level, we could pass the
deterministic flag as a parameter during partition compilation.

```cpp
// C++ API
compiled_partition partition::compile(const engine& eng,
                                    const std::vector<logical_tensor>& inputs,
                                    const std::vector<logical_tensor>& outputs,
                                    bool deterministic = false) const;

// C API  
dnnl_status_t DNNL_API dnnl_graph_partition_compile_with_deterministic(
        dnnl_graph_compiled_partition_t *compiled_partition,
        const_dnnl_graph_partition_t partition,
        const_dnnl_engine_t engine,
        size_t num_inputs,
        const_dnnl_graph_logical_tensor_t *inputs,
        size_t num_outputs, 
        const_dnnl_graph_logical_tensor_t *outputs,
        int deterministic);
```

**Pros:**
- Different partitions from the same graph can have different deterministic
  requirements
- The same partition can be reused abd compiled into different compiled
  partition with and without deterministic mode
- Deterministic mode can be decided at compilation time based on runtime
  conditions

**Cons:**
- Requires overloaded or extended compilation APIs
- More complex for users who want the same deterministic behavior for entire
  graphs
- No such request to support different partitions from the same graph to have
  different deterministic modes

## Decision

The graph-level deterministic mode approach is selected as:

- Less API complexity
- It aligns with how other properties like `fpmath_mode` are managed in Graph
  API
- Users set deterministic mode once rather than managing it per-operation or
  per-partition
- This matches how frameworks handle deterministic modes at the model level
- Current graph API usages show no demand for fine-grained deterministic control
  within the same graph

## Testing and Validation

### Gtests

New gtests cases will be added to verify:

- Correct setting and getting of deterministic mode
- Inheritance of deterministic mode by partitions
- Error handling for invalid inputs

### Benchdnn

The benchdnn graph driver will be extended to support testing deterministic
execution by running the same graph multiple times and comparing results.

Details TBD.

## References

[^1]: [Use deterministic algorithms in PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html)
[^2]: [RFC for deterministic mode in primitives](https://github.com/uxlfoundation/oneDNN/blob/rfcs/rfcs/20231213-determinism/README.md)
