# RFC: Public Embedding Bag Primitive in oneDNN

## Authors
- AMD ZenDNN team

## Introduction

This RFC adds an Embedding Bag  as a public oneDNN primitive `dnnl::embedding_bag`,
backed first by a portable CPU reference implementation. A thin `dnnl::embedding`
wrapper covers the lookup-only (no-reduction) mode.

This primitive closely resembles PyTorch Embedding Bag semantics.

oneDNN currently has no primitive equivalent to PyTorch Embedding Bag. The closest
dense neighbours (`reduction`, graph `gather`) do not cover
*indexed lookup + variable-length bag reduction* semantics. Framework users that
integrate with oneDNN either need to depend on its framework implementation,
or assemble multiple ops by hand.

The first PR promotes the primitive plumbing to a public C++ API, adds the CPU
reference implementation, and validates the behavior needed by recommendation
and NLP workloads: sum / mean / max reduction over variable-length bags,
optional per-sample weights, `padding_idx` skipping, and the
`include_last_offset` flag matching PyTorch semantics. Future optimized
implementations (AVX-512 intrinsics) reuse the same API without any framework
changes.

## 1. Motivation

Embedding lookups and bag reductions are the dominant CPU operation in
recommendation models (DLRM, DCNv2, MMoE) and token-id-consuming NLP models.
PyTorch `nn.EmbeddingBag`, TensorFlow `embedding_lookup_sparse`, and ONNX
Runtime `EmbedLayerNormalization` each implement this on CPU today — either
through per-framework hand-rolled kernels or FBGEMM — because oneDNN offers no
equivalent primitive.

The main reason to own it in oneDNN is performance and consolidation. A
high-quality embedding kernel requires
- careful work splitting (table-threaded vs bag-threaded),
- int32/int64 index handling,
- optional per-sample weights without branchy inner loops, and
- ISA-specific load/scatter strategies (AVX-512 BF16, AVX-512-FP16).

Different frameworks repeatedly re-implement this. Centralizing it in
oneDNN keeps the optimizations in one place and gives AMD and Intel a single
tuning target.

Embedding bag is a memory bound operator. **It is shown
([Parallelization Strategies for DLRM
Embedding Bag Operator on AMD CPUs] (https://dl.acm.org/doi/10.1109/MM.2024.3423785) )
that embedding bag bandwidth can be improved by 9x over state of the art
implementations using different threading strategies.**

The kernels for the optimized phase come from AMD's ZenDNN, which ships highly
optimized AVX-512 / AVX-512-FP16 implementations through its internal
`lowoha::embag` API. This RFC upstreams those intrinsic kernels into oneDNN as a
first-class primitive lifted into `src/cpu/x64/`, with ZenDNN removed from the
runtime path after integration.

## 2. Non-Goals

- Do not add backward / training support.
- Do not add low precision (`bf16` / `fp16`) and quantized (`s8` / `s4` / `u4`)
  embedding tables in the first PR.
- Do not add high precision (`s64` / `u64`) indexing in the first PR.
- Do not add optimized (AVX-512 / AVX512-FP16) implementations in the first PR.
- Do not add a GPU implementation in the first PR.
- Do not replace any existing oneDNN primitive or Graph API path.

## 3. Proposal — public embedding_bag primitive

The design follows the same skeleton used by other oneDNN primitives. It adds
a public header API, a CPU implementation-list entry, and the CPU reference
kernel; consumers that do not call `dnnl::embedding_bag` are unaffected.

### 3.1 Architecture overview

```
framework (PyTorch nn.EmbeddingBag / zentorch / app)
        │  Primitive API
        ▼
dnnl::embedding_bag::primitive_desc(eng, alg, src, indices, offsets,
                                    weights, dst[, padding_idx, flags])
        │  → dnnl_embedding_bag_primitive_desc_create(...)
        ▼  CASE(embedding_bag) → get_embedding_bag_impl_list → walk impl_list
   ┌──────────────────────────────────────────────────────┐
   │ ref_embedding_bag_t::pd_t::init()  (CPU)             │
   │   2D table · f32 · s32/s64 indices · sum/mean/max/   │
   │   lookup · optional weights / padding_idx            │
   └──────────────────────────────────────────────────────┘
        │ success                         │ unimplemented
        ▼                                 ▼
   dnnl::embedding_bag(pd)           (next impl / fall-through)
        │  execute(): Y[b,:] = REDUCE over bag_b of T[I[k],:] * w[k]
        ▼
   dst written
```

### 3.2 Operation semantics

Let T be a 2D embedding table `[V, D]`, I a 1D indices vector of length N, O a
1D offsets vector defining B bags, and `algo` one of {sum, mean, max, lookup}.

**Lookup mode** (no offsets, no reduction):

```
Y[n, :] = T[I[n], :]          for n in [0, N),  shape Y = [N, D]
```

**Bag modes** (sum / mean / max):

```
bag_b   = { I[k] : O[b] <= k < O[b+1] }
Y[b, :] = REDUCE_algo over bag_b of  T[I[k], :] * w[k]
shape Y = [B, D]
```

- `w[k]` = `per_sample_weights[k]` if provided, else `1.0`
- Indices equal to `padding_idx` are skipped
- `max` mode does not combine with per-sample weights (matches PyTorch)
- When `include_last_offset == true`, `O` has length `B+1` and `O[B]` is read
  from data; otherwise the implicit terminator `O[B] = N` is used

### 3.3 Public C++ API

Add `dnnl::embedding_bag` and its `primitive_desc` to
`include/oneapi/dnnl/dnnl.hpp`. The constructor forwards to the new C entry
`dnnl_embedding_bag_primitive_desc_create`.

```cpp
struct embedding_bag : public primitive {
    struct primitive_desc : public dnnl::primitive_desc {
        primitive_desc() = default;

        // No weights
        primitive_desc(const engine &aengine,
                algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &indices_desc,
                const memory::desc &offsets_desc,
                const memory::desc &dst_desc,
                int64_t  padding_idx = -1,
                bool  include_last_offset = false,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false);

        // With weights
        primitive_desc(const engine &aengine,
                algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &indices_desc,
                const memory::desc &offsets_desc,
                const memory::desc &weights_desc,
                const memory::desc &dst_desc,
                int64_t  padding_idx = -1,
                bool  include_last_offset = false,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false);
    };

    embedding_bag() = default;
    explicit embedding_bag(const primitive_desc &pd) : primitive(pd.get()) {}
};

```

New algorithm values in `dnnl_alg_kind_t` (`0x30000` band to avoid collisions):
`dnnl_embedding_bag_sum`, `dnnl_embedding_bag_mean`, `dnnl_embedding_bag_max`,
`dnnl_embedding_lookup`.

### 3.4 Argument map

| Argument | Memory | When required |
|---|---|---|
| `DNNL_ARG_EMBEDDING_BAG_TABLE` | embedding table `[V, D]` | always |
| `DNNL_ARG_SRC_INDICES` | indices `[N]`, `s32` | always |
| `DNNL_ARG_SRC_OFFSETS` | offsets `[B]` or `[B+1]`, same dtype as indices | bag modes only |
| `DNNL_ARG_WEIGHTS` | per-sample weights `[N]`, `f32` | optional; ingnored if `max` |
| `DNNL_ARG_DST` | output `[B, D]` (bag) or `[N, D]` (lookup) | always |

### 3.5 CPU registration and reference kernel

`cpu_engine.hpp` routes `primitive_kind::embedding_bag` to a new implementation
list (`src/cpu/cpu_embedding_bag_list.cpp`):

```cpp
constexpr impl_list_item_t impl_list[] = REG_EMBEDDING_BAG_P({
        CPU_INSTANCE_X64(embedding_bag_t)   // AVX-512 intrinsic impl (Phase 2)
        CPU_INSTANCE(ref_embedding_bag_t)   // portable C++ reference
        nullptr,
});
```

`ref_embedding_bag_t` supports forward inference for all four algorithms with
`f32` table / output, `s32` indices, and optional per-sample weights.Unsupported
cases return `unimplemented` so that later optimized implementations can be registered above
the reference kernel in the impl-list.

### 3.6 Validation rules

Enforced in `embedding_bag_desc_init` and `embedding_bag_pd_t::init_*`:

1. `src_desc` is 2D, blocked, dtype in the supported set (Phase 1: `f32`).
2. `indices_desc` is 1D; dtype in {`s32`}.
3. Bag modes: `offsets_desc` is 1D, same dtype as indices, length B (or B+1
   with `include_last_offset`); `dst_desc` is 2D `[B, D]`.
4. Lookup mode: `offsets_desc` and `weights_desc` must be empty (`zero_md`);
   `dst_desc` is `[N, D]`.
5. If `weights_desc` is non-empty: `f32`, length N, and
   `alg_kind != embedding_bag_max`.
6. `padding_idx` in `[-1, V)`; `-1` means "no padding".
7. All MDs must be fully described.
8. `dst_desc.dims[1] == src_desc.dims[1]` (embedding dim matches).

## 4. PoC — embedding_bag primitive end-to-end with benchdnn

The public primitive was validated end-to-end through the benchdnn
`--embedding-bag` driver, running against the CPU reference implementation.

### 4.1 Verbose evidence (f32 sum, with weights and padding_idx)

```
onednn_verbose,v1,primitive,exec,cpu,embedding_bag,ref:any,forward_inference,
  src:f32::blocked:ab:100000x128 idx:s32::blocked:a:1000
  off:s32::blocked:a:64 wts:f32::blocked:a:1000
  dst:f32::blocked:ab:64x128,
  ,alg:embedding_bag_sum pad:0 flags:, 1x64x128:1x1000:1x64:1x1000:1x64x128, 3.14
```

### 4.2 Accuracy

- **f32 sum / mean / max:** benchdnn correctness driver (`--mode=C`) passes for
  basic shapes and DLRM-representative shapes.
- **f32 lookup:** passes for index sets with and without `padding_idx`.
- **Edge cases:** all-padding bag returns zero (matches PyTorch); single-element
  bag; `include_last_offset` with explicit `O[B]` entry.

### 4.3 What is validated

- Public `dnnl::embedding_bag` builds, dispatches to `ref_embedding_bag_t` on
  CPU, executes.
- Sum / mean / max / lookup algorithms, with and without per-sample weights.
- `padding_idx` skipping and zero output for fully-padded bags.
- `include_last_offset` flag with explicit and implicit last-offset semantics.
- `s32` index dtypes.

## 5. Framework-Side Changes

Framework backends can map their existing embedding operator to
`dnnl::embedding_bag`. They pass table, indices, offsets, and optional weights
memory descriptors plus `padding_idx` and `flags` to the primitive descriptor.

Applications need no source changes: they keep calling the framework API (for
example PyTorch `nn.EmbeddingBag`), and only the backend mapping changes. Later
oneDNN embedding optimizations reuse the same mapping.

## 6. First PR — Scope and Acceptance Criteria

The first PR is complete when the following are in place and passing on CPU:

- Public `dnnl::embedding_bag` construction and execution work through public
  oneDNN headers only.
- `ref_embedding_bag_t` is registered in the CPU implementation list and appears
  in verbose output as `cpu,embedding_bag,ref:any`.
- Forward inference works for `f32` table/output with `s32` indices
  for all four algorithms.
- Per-sample weights (sum / mean), `padding_idx`, and `include_last_offset` work
  correctly and match PyTorch reference outputs.
- benchdnn `--embedding-bag` covers the CI input set with an independent `f32`
  reference.
- gtests cover descriptor creation, accessors, validation errors, and execute
  smoke cases.

Follow-up work includes the AVX-512 intrinsic kernel (Phase 2), BF16/FP16
paths, quantized tables (Phase 3), and GPU support (future).

## 8. Alternatives Considered

- Keep using framework-native kernels (FBGEMM, hand-rolled). Avoids a new
  primitive API, but each framework carries its own embedding kernel with no
  shared ISA tuning.
- Implement via Graph API gather + reduction fusion. Does not cover
  variable-length bag semantics or per-sample weights without a bespoke graph
  pattern.
- Add a public primitive with a CPU reference implementation. This is the
  proposed option: it gives frameworks a stable entry point immediately and
  leaves optimized implementations as follow-up work behind the same API.

## 9. Open Questions

1. **Intrinsics vs JIT for the optimized impl.** Almost every optimized x64
   oneDNN primitive uses Xbyak JIT. The embedding kernel logic (gather a row,
   multiply by a scalar weight, reduce element-wise) is simple enough that there
   is no shape-dependent code-gen opportunity that benefits from runtime
   emission. ZenDNN's intrinsic implementation is well-validated today. The
   impl-list mechanism leaves room for a `jit_uni_embedding_bag_t` above the
   intrinsic impl in a follow-up if profiling justifies it — users see no API
   change.

2. **Threading strategy for very long bags.** Phase 1 parallelizes across bags
   (`parallel_nd(B, ...)`). For workloads where a single bag is very long
   relative to thread count, an optional inner split via `balance211` could
   improve scaling. Deferred to Phase 2.

3. **`max` + per-sample weights.** PyTorch forbids this; we mirror that. Some
   users may expect `max(T[i] * w[i])`; the RFC defaults to the PyTorch
   behavior.

4. **All-padding bag output.** PyTorch returns zeros for sum / mean and 0 for
   max when all indices in a bag equal `padding_idx`. We match this; document
   and test explicitly.

5. **Performance gate.** Every new oneDNN primitive must show material
   workload-level impact. A documented benchdnn perf comparison vs framework
   baselines on representative DLRM shapes is required before the Phase 2 PR
   can land.
