# RFC: Public SDPA Primitive Support in oneDNN

## Authors
- AMD ZenDNN team

## Introduction

This RFC proposes promoting **Scaled Dot-Product Attention (SDPA)** to a
first-class, **public primitive** in oneDNN: a public `dnnl::sdpa` C++ API (with
its nested `primitive_desc`), backed by a CPU reference implementation
(`ref_sdpa_fwd_t`), with **attention-mask support** (explicit additive masks and
causal masks) and **fully stride-driven layout handling** so the real,
non-contiguous tensors that frameworks produce (fused-QKV, transposed views) are
computed correctly.

Today **oneDNN has no public SDPA *primitive***. Attention is reachable in two
ways, but **neither gives a consumer a `dnnl::sdpa` primitive to call**:

1. **Graph API** - *public and supported*, but as a **fused graph pattern**, not
   a primitive. SDPA is assembled as an MHA partition (MatMul to Softmax to
   MatMul, with optional scale/mask) and validated through the Graph
   complex-fusion tests. The consumer builds and manages a graph, rather than
   issuing one primitive call like `dnnl::matmul`.
2. **An internal SDPA primitive** - a *real primitive*, but **not public**.
   oneDNN already contains the SDPA primitive descriptor and GPU kernels
   (`src/common/sdpa_*`, `sdpa_primitive_desc_create`), yet the only way to
   reach it is a **test-only** wrapper in
   `tests/gtests/internals/sdpa_internal.hpp`.
   The primitive plumbing exists but was parked behind the test/internal
   interface rather than promoted to a supported `dnnl::sdpa` API, and its CPU
   path is effectively absent.

In short: the supported path (Graph) is not a primitive, and the primitive that
exists is not public.

This RFC closes that gap by taking the existing internal primitive plumbing and
adding the missing public and CPU pieces around it. The first PR is
intentionally foundational: it exposes a public `dnnl::sdpa` C++ primitive,
adds a portable CPU reference implementation, validates masking and
stride-aware layout handling, and adds benchdnn coverage. Once this primitive
entry point exists, oneDNN can register optimized SDPA implementations behind
the same API without requiring framework-side branching or new integration
paths.

---

## First PR - immediate ask

The first PR is not the final optimized SDPA kernel. It establishes the public
primitive API, CPU correctness path, and validation infrastructure needed for
optimized implementations to plug in later.

1. **Public SDPA primitive API** - add `dnnl::sdpa` and its
   `primitive_desc` to `include/oneapi/dnnl/dnnl.hpp`, forwarding to the
   existing internal C entry `sdpa_primitive_desc_create`. Forward inference,
   4D BHSD tensors.
2. **CPU reference kernel** - `src/cpu/ref_sdpa.{hpp,cpp}` registered in a new
   `src/cpu/cpu_sdpa_list.cpp`, routed from `cpu_engine.hpp`
   (`primitive_kind::sdpa`). f32 and bf16, default scale `1/sqrt(head_dim)`,
   `softmax_accurate`.
3. **Attention-mask support** - explicit additive masks (f32/bf16, 2D
   `(Sq,Skv)` or 4D `(N,H,Sq,Skv)` with broadcast) and top-left causal masks;
   both may apply together.
4. **Stride-aware layout handling** - every tensor axis addressed through its
   memory-descriptor stride, so dense, transposed (BHSD/BHDS views) and packed
   (fused-QKV BSHD) inputs are all read correctly.
5. **benchdnn SDPA driver** - `tests/benchdnn/sdpa/` with a self-contained f32
   reference (`ref_sdpa.cpp`) and input sets for correctness validation.

Detailed acceptance criteria are in
[Section 7](#7-first-pr--scope-and-acceptance-criteria).

---

## 1. Motivation

Attention is the dominant compute pattern in transformer inference. Frameworks
(PyTorch `scaled_dot_product_attention`, vLLM, ONNX Runtime) want a single,
fused, well-optimised attention call. oneDNN is the CPU primitive library those
frameworks depend on, yet it does **not** expose a public attention primitive.

**Problem.** A consumer that wants oneDNN to run attention today must either:
- build the **Graph** (assemble the MHA fusion pattern), which is heavier to
  integrate and is graph-API-centric; or
- reach the **internal** SDPA primitive, whose API lives in test/internal
  headers (`sdpa_internal.hpp` / `sdpa_test_iface.hpp`) and is not part of the
  public contract and which has no CPU kernel.

So a framework that already calls oneDNN primitives (`dnnl::matmul`,
`dnnl::softmax`, etc.) has no `dnnl::sdpa` to call. AMD `zentorch` plugin
currently routes attention through ZenDNN's existing attention implementation
for exactly this reason.

**Proposal.** Provide a real, public **SDPA primitive** - `dnnl::sdpa` - with a
CPU reference implementation, masks, and stride-aware layout handling. This
creates a stable primitive API that can run on both Intel and AMD platforms and
lets oneDNN dispatch future optimized implementations without framework
integration changes. The first PR scope is limited to the public API and CPU
reference implementation; optimized Flash Attention-style and
BMM-Softmax-BMM-based implementations can be added later. These optimized paths
have different tradeoffs by sequence length, head size, mask type, data type,
and hardware: BMM-Softmax-BMM paths can use oneDNN BRGEMM microkernels on Intel
CPUs and Zen-tuned BMM kernels on AMD CPUs, while Flash Attention-style paths
can use similar tiling and parallelism with the platform-appropriate
matrix-multiply microkernel.

- **Performance.** The CPU reference kernel in the first PR is a correctness
  baseline, not the final performance path. The performance motivation comes
  from existing ZenDNN SDPA integration: `zentorch` rewrites
  `aten::scaled_dot_product_attention` to ZenDNN's SDPA path and shows up to
  15% geomean improvement on Torch Inductor dashboard models on AMD (only SDPA
  rewrite in `zentorch` is enabled; MatMul overrides are disabled). A public
  oneDNN SDPA primitive provides the standard API hook for bringing this kind
  of optimized implementation into oneDNN.
- **Generality.** Standard primitive API. Any consumer that calls oneDNN
  benefits with no graph-building work.

## 2. Goals and Non-Goals

### Goals
- **Public `dnnl::sdpa` primitive** with the usual `primitive_desc` to primitive
  to `execute()` lifecycle, mirroring every other oneDNN primitive.
- **Portable CPU reference kernel** - correct, dependency-free, runs on any CPU
  (all arithmetic in f32; bf16 widened on read, narrowed on write).
- **Masking** - explicit additive mask (f32/bf16, 2D/4D, broadcast) and
  top-left causal; both composable.
- **Stride-aware** - correct on dense, transposed, and packed/fused-QKV layouts.
- **Validation** - benchdnn correctness against a self-contained reference, plus
  model-level evidence from Torch Inductor dashboard models, including BERT
  through `zentorch`.

### Non-Goals
- **Replacing the Graph API path.** The MHA graph fusion stays; this primitive
  is complementary.
- **Quantized / GQA / custom-scale / backward** in the first PR - rejected at
  dispatch (`unimplemented`) and left to follow-ups.
- **GPU kernels** - the existing internal GPU SDPA is untouched; this RFC adds
  the public API + a CPU reference.

## 3. Background - How SDPA exists in oneDNN today

As described above, oneDNN currently supports attention either through the Graph
API's MHA fusion path or through an internal SDPA primitive used for test and
GPU validation. Neither path gives frameworks a public CPU primitive API
equivalent to `dnnl::matmul`. This RFC promotes that primitive path by adding
the missing public C++ wrapper, CPU reference kernel, mask support, and
stride-aware layout handling.

## 4. Proposal - public SDPA primitive

### 4.1 Architecture overview

```
framework (PyTorch SDPA / zentorch / app)
        │  Primitive API
        ▼
dnnl::sdpa::primitive_desc(eng, q, k, v, o[, mask, attn_mask_type])
        │  → ::sdpa_primitive_desc_create(...)   (existing internal C entry)
        ▼  CASE(sdpa) → get_sdpa_impl_list → walk impl_list
   ┌──────────────────────────────────────────────┐
   │ ref_sdpa_fwd_t::pd_t::init()  (CPU)          │
   │   4D · f32/bf16 · mask f32/bf16 2D/4D ·       │
   │   default scale · softmax_accurate            │
   └──────────────────────────────────────────────┘
        │ success                         │ unimplemented
        ▼                                 ▼
   dnnl::sdpa(pd) → ref_sdpa_fwd_t   (next impl / fall-through)
        │  execute(): scores=scale·QKᵀ → mask → softmax → ·V
        ▼
   dst written
```

### 4.2 Operation semantics

SDPA computes, as a single fused op:

```
dst = matmul( softmax( matmul(Q, Kᵀ) * scale + mask ), V )
```

Shapes (4D BHSD): Q,O = `(N,H,Sq,D)`, K,V = `(N,H,Skv,D)`; K is presented to the
primitive as the logical `(N,H,D,Skv)` transpose so scores = Q·K is a plain
matmul.

### 4.3 Public C++ API

Add `dnnl::sdpa` and its `primitive_desc` to
`include/oneapi/dnnl/dnnl.hpp`. Both constructors forward to the existing C
entry `sdpa_primitive_desc_create`. Following oneDNN's own convention for
optional tensors (e.g. `matmul`'s optional bias), the mask is exposed through
**two overloads** rather than a defaulted descriptor: a clean no-mask form and
an explicit mask form:

```cpp
struct sdpa : public primitive {
    struct primitive_desc : public dnnl::primitive_desc {
        primitive_desc() = default;

        // No explicit mask (common case).
        primitive_desc(const engine &aengine,
                const memory::desc &query_desc, const memory::desc &key_desc,
                const memory::desc &value_desc, const memory::desc &dst_desc,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false);

        // Explicit additive buffer mask and/or causal mask.
        primitive_desc(const engine &aengine,
                const memory::desc &query_desc, const memory::desc &key_desc,
                const memory::desc &value_desc, const memory::desc &dst_desc,
                const memory::desc &mask_desc, int attn_mask_type,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false);
        // query/key/value/dst_desc() accessors
    };
    sdpa() = default;
    sdpa(const primitive_desc &pd) : primitive(pd) {}
    sdpa(const primitive_desc &pd, const std::vector<uint8_t> &cache_blob);
};
```

- no mask: `sdpa::primitive_desc(eng, q, k, v, o)`
- explicit additive mask: `... (eng, q, k, v, o, mask_md, /*buffer*/1)`
- causal: `... (eng, q, k, v, o, {}, /*top_left*/2)`

`attn_mask_type` mirrors `dnnl_attn_mask_type_t`: `0` undef/none, `1` explicit
buffer, `2` causal top-left, `3` causal bottom-right.

### 4.4 Dispatch registration (CPU)

`cpu_engine.hpp` routes `primitive_kind::sdpa` to a new impl list
(`src/cpu/cpu_sdpa_list.cpp`) that registers the reference kernel:

```cpp
constexpr impl_list_item_t impl_list[] = REG_SDPA_P({
        CPU_INSTANCE(ref_sdpa_fwd_t)
        nullptr,
});
```

Before this work the CPU engine returned `empty_list` for `sdpa` (no CPU impl).
A richer optimized impl (Flash Attention or BMM-Softmax-BMM) can register ahead
of `ref_sdpa_fwd_t` later, with the reference as a guaranteed fallback.

### 4.5 CPU reference kernel (`ref_sdpa_fwd_t`)

A single, readable reference that mirrors ZenDNN's own `sdpa_encoder_ref`
philosophy: **all arithmetic in f32**, typed I/O at the boundary:

- **dispatch gate (`pd_t::init`)** - 4D Q/K/V/dst; uniform f32 or bf16; mask (if
  present) f32/bf16 and 2D/4D with contiguous inner axis; default scale;
  `softmax_accurate`; reject quantization / bottom-right / custom scale and
  return `unimplemented` (caller falls through).
- **`execute()`** - per `(n,h,sq)`: compute `scores = scale * (q * K)`, apply
  mask, run stable softmax, then compute `out = probs * V`. bf16 elements are
  converted to/from f32 on read/write.

### 4.6 Masking

Two orthogonal channels, both honored, composable:
- **buffer mask** - additive, read via its descriptor (f32/bf16; 2D `(Sq,Skv)`
  or 4D `(N,H,Sq,Skv)`; size-1 axes broadcast via 0 stride); applied as
  `scores += mask`.
- **causal (top-left)** - keys after the query position set to `-inf`.

A buffer mask and a causal flag may be active simultaneously (the descriptor
carries the mask md, the enum carries causal); the kernel applies both.

### 4.7 Stride-aware layout handling (the correctness core)

Frameworks produce attention tensors that are **logically BHSD but physically
non-contiguous**, such as `transpose(1,2)` views and, with fused-QKV
projections, **packed** buffers where K and V sit between Q rows (so Q's
sequence stride is `3·H·D`, not `H·D`). The reference addresses **every axis of
every tensor through its memory-descriptor stride**, so dense, transposed
(BHSD/BHDS), and packed (BSHD/fused-QKV) inputs are all read correctly. A
dense-layout assumption (the original draft) silently read wrong memory on
fused-QKV and collapsed model accuracy; stride-driven addressing fixes it.

### 4.8 No disruption to existing paths

The Graph MHA fusion and the internal GPU primitive are untouched. This RFC adds
a public header surface, a CPU impl-list entry, and the CPU kernel. Consumers
not using `dnnl::sdpa` are unaffected.

## 5. PoC - SDPA primitive end-to-end on Torch Inductor dashboard models

The public primitive runs end-to-end through ZenDNN's `zentorch` plugin by
routing `sdpa_direct` to `dnnl::sdpa`.

E2E model validation follows this path:

```
Torch Inductor dashboard model
        │
        ▼
PyTorch SDPA call
        │
        ▼
zentorch - zenSDPA
(after graph rewrite of aten::sdpa)
        │
        ▼
ZenDNN sdpa_direct
        │
        ▼
oneDNN `dnnl::sdpa` primitive API
        │
        ▼
ref_sdpa_fwd_t on CPU
        │
        ▼
model output / accuracy compared with baseline
```

### 5.1 Verbose evidence (masked SDPA, packed-QKV layout)

```
onednn_verbose,v1,primitive,exec,cpu,sdpa,ref:any,forward_inference,
  query:f32::blocked:acbd:48000x64x384x1 key:f32::blocked:adbc:48000x64x1x384
  val:f32::blocked:acbd:48000x64x384x1 msk:f32::blocked:abcd
  dst:f32::blocked:acbd,
  ,alg:softmax_accurate msk:2d, 1x2x125x64:1x2x64x125:1x2x125x64, 2.62
```

The explicit Q/K/V strides (`...x384x1`) are the fused-QKV packing; `msk:` shows
the additive mask flowing through the primitive. Both are handled by the
stride-aware kernel.

### 5.2 Accuracy (Torch Inductor dashboard models, including BERT-QA)

- **f32:** Torch Inductor dashboard model validation passes.
- **bf16:** Torch Inductor dashboard model validation passes within the expected
  tolerance.

### 5.3 What is validated

- Public `dnnl::sdpa` builds, dispatches to `ref_sdpa_fwd_t` on CPU, executes.
- Masked / causal / no-mask, f32 / bf16, self- and cross-attention.
- Packed/transposed (fused-QKV) layouts produce correct results.

## 6. Framework-Side Changes

Frameworks that want to use this primitive need to route their SDPA operator to
`dnnl::sdpa` instead of decomposing attention into separate MatMul / Softmax /
MatMul calls or relying on a framework-specific integration path. The framework
passes Q/K/V/dst memory descriptors, an optional additive mask descriptor, and
the attention-mask type to the new primitive descriptor.

Example PyTorch framework-side routing:

```
Torch Inductor dashboard model
        |
        v
PyTorch SDPA call
        |
        v
aten::scaled_dot_product_attention
        |
        v
framework oneDNN backend mapping
        |
        v
oneDNN dnnl::sdpa primitive API
        |
        v
ref_sdpa_fwd_t on CPU
        |
        v
model output / accuracy compared with baseline
```

Once this mapping is added, applications do not need source changes. They keep
calling the framework's existing SDPA API (for example PyTorch
`aten::scaled_dot_product_attention`), while the framework backend can dispatch
that call through oneDNN's public primitive API. Future SDPA optimizations can
then be handled inside oneDNN, with no additional framework-side changes.

## 7. First PR - Scope and Acceptance Criteria

The first PR is accepted when the public primitive API, CPU reference path,
masking behavior, stride-aware layouts, and benchdnn validation below are all in
place and passing on CPU.

### 7.1 Public API
- `dnnl::sdpa` + `primitive_desc` in `include/oneapi/dnnl/dnnl.hpp`, forwarding
  to `sdpa_primitive_desc_create`; mask optional via defaults.
- **Acceptance criteria:** no-mask and masked `primitive_desc` construct,
  build a `dnnl::sdpa`, and execute using public oneDNN headers only.

### 7.2 CPU reference kernel
- `src/cpu/ref_sdpa.{hpp,cpp}` + `src/cpu/cpu_sdpa_list.cpp`; `cpu_engine.hpp`
  routes `primitive_kind::sdpa`.
- f32 + bf16; default scale; `softmax_accurate`; forward inference.
- **Acceptance criteria:** `dnnl::sdpa` shows `cpu,sdpa,ref:any` in verbose.

### 7.3 Masking
- Explicit additive (f32/bf16, 2D/4D, broadcast), top-left causal, both
  together.
- **Acceptance criteria:** benchdnn mask cases match the reference within
  tolerance.

### 7.4 Stride-aware layouts
- Per-axis stride addressing for Q/K/V/O.
- **Acceptance criteria:** packed/transposed (fused-QKV) cases produce correct
  results (regression guard for the dense-assumption bug).

### 7.5 benchdnn SDPA driver
- `tests/benchdnn/sdpa/` with self-contained f32 `ref_sdpa.cpp`, problem parser,
  and input sets (`test_sdpa_smoke` / `_ci` / `_all`).
- **Acceptance criteria:** benchdnn `--sdpa` passes the CI input set on CPU.

### 7.6 Additional first PR requirements
- **gtests** - add primitive API tests for descriptor creation, queries, invalid
  arguments, and execute smoke coverage.
- **Public API cleanup** - move SDPA API types and arguments to public headers
  and remove dependency on the test/internal interface.
- **Docs and upstream prep** - document usage, examples, limitations, and split
  the patches for internal review and upstream discussion.

### 7.7 Follow-up work
- AMD-tuned SDPA implementation registered ahead of the reference.
- bf16 numerics parity options; bottom-right causal; cross-impl tolerances.

## 8. Alternatives Considered

### 8.1 Keep using only the Graph API
- **Pros:** already supported; no new public surface.
- **Cons:** graph-building is heavier to integrate than one primitive call;
  consumers wanting a simple fused attention call still have none. A primitive
  and the graph path coexist.

### 8.2 Keep the primitive internal
- **Pros:** no public-API commitment.
- **Cons:** leaves the primitive unusable by real consumers and untested on CPU
  - exactly today's state. Promoting it to public with a CPU kernel is the point
  of this RFC.

### 8.3 Public primitive + CPU reference + masks + strides
- Smallest change that makes SDPA a usable, validated CPU primitive and opens
  the path for optimized platform-specific implementations.
- Many PyTorch Inductor dashboard models already expose SDPA as a fused op/node,
  so frameworks can map that op directly to `dnnl::sdpa` instead of first
  constructing a oneDNN Graph pattern. This is simpler for frameworks that
  already use oneDNN's primitive API.

## 9. Open Questions

- **Future optimized dispatch policy.** For a BMM-Softmax-BMM SDPA path, AMD
  platforms can use ZenDNN's Zen-tuned BMM kernels while Intel platforms can use
  oneDNN's optimized BRGEMM path. Similarly, for a Flash Attention-style path,
  AMD hardware can use ZenDNN kernels while Intel hardware can use oneDNN BRGEMM
  kernels. The open question is whether this dispatch policy remains optimal on
  Intel.
- **First PR implementation scope.** Should the first PR include a
  BMM-Softmax-BMM SDPA path that calls oneDNN BRGEMM, or is the CPU reference
  kernel sufficient for the initial public primitive API and validation?

