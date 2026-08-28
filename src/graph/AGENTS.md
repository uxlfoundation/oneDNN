# AGENTS.md — src/graph

See root `/AGENTS.md` for build/test/style/commit conventions. This file
covers the Graph API only — a fusion/partitioning layer on top of the
primitive API in `src/common`, `src/cpu`, `src/gpu`.

Graph API concepts (ops, logical tensors, partitions, fusion patterns) are
documented for users in `doc/graph_api/` (`graph_programming_model.rst`,
`graph_fusion_patterns.rst`, `graph_supported_operations.rst`,
`graph_extension.rst`, `graph_dump.md`) — read those for the conceptual
model before changing this code.

## Layout

```
interface/   public-facing graph/op/partition/tensor types, op schema
             registration (op_schema.*, opset.hpp)
backend/dnnl/  the real backend: lowers partitions to primitive-API calls
               (kernels/, executables/)
backend/fake/  no-op backend used when no real backend matches a partition
utils/         pattern-matching (utils/pm), allocator, JSON dump, verbose
```

## Testing

- `tests/gtests/graph/` — gtests for the public C/C++ API and backend/interface
  units.
- `tests/benchdnn`'s `graph` driver (`tests/benchdnn/doc/driver_graph.md`) —
  correctness on JSON-described graphs, not the primitive parameter strings
  other drivers use.
