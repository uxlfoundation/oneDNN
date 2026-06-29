# RFC: ZenDNN Integration in oneDNN — `zen64` Module

## Authors
- AMD ZenDNN team

## Summary

This RFC proposes registering ZenDNN as a CPU backend in oneDNN through a new
opt-in module under `src/cpu/x64/zen64/`. As an `src/cpu/x64/` module it is a
**generic x86_64 backend that builds and runs on any x86 CPU**; its dispatch
gate is tuned to engage on AMD CPUs with AVX-512 support (the Zen4
microarchitecture and later), where it delivers optimal performance.
Build-time gating: `DNNL_X64_USE_ZEN=ON` (default OFF). Runtime dispatch is
decided automatically by CPU detection inside each `pd_t::init()`; there is no
runtime env var. ZenDNN registers as additional entries in oneDNN's existing
per-primitive impl lists (one per supported primitive); on non-AMD x86 systems,
or for any unsupported
configuration, the existing oneDNN impls run through the standard
`status::unimplemented` fall-through, so behaviour on other x86 CPUs is
unchanged. **No public API changes.** A working proof-of-concept of MatMul
fp32 / bf16 with Reorder is available on the
[`zendnn-onednn-integration`](https://github.com/amd/oneDNN-Zen/tree/zendnn-onednn-integration)
branch of the oneDNN-Zen repository.

The motivation is grounded in ZenDNN's existing, production-measured
performance: **ZenDNN already delivers leading CPU performance on AMD CPUs
today**, with up to **~30% higher throughput** than the stock oneDNN path on
Zen-class CPUs across representative MatMul / LLM-inference workloads, as
shipped in production through
[zentorch](https://github.com/amd/ZenDNN-pytorch-plugin). The goal of this work
is to bring that same AMD-CPU advantage to every oneDNN consumer (PyTorch,
TensorFlow, ONNX Runtime, vLLM, ...) without each having to integrate ZenDNN
separately; the integration's own throughput is validated at the model level
(vLLM + benchdnn) before each production PR (§4.5).

---

> ## First PR — the immediate ask
>
> Once this RFC is accepted, the **first** PR to upstream
> `uxlfoundation/oneDNN` carries exactly the following scope. Everything else
> is follow-up.
>
> 1. **ZenDNN library integration in build infra** — new
>    `DNNL_X64_USE_ZEN` CMake option (default OFF), `cmake/ZenDNN.cmake`
>    consuming ZenDNN's `zendnnl-config.cmake` package config (binary located
>    via `ZENDNNROOT`). No effect on the default build.
> 2. **Add ZenDNN MatMul (+ fused) and Reorder (for format conversion) in the
>    `zen64` module** — new
>    `src/cpu/x64/zen64/{zen_matmul,zen_reorder}.{hpp,cpp}` registered ahead of
>    the existing entries.
>     - **AOCL-DLP MatMul kernels only** — the first PR drives ZenDNN's
>       [AOCL-DLP](https://github.com/amd/aocl-dlp) MatMul path exclusively.
>       ZenDNN's other backends (native ZenDNN kernels, libxsmm, FBGEMM,
>       oneDNN-as-inner-backend) and its multi-backend Auto Tuner are **not**
>       part of the first PR; they are general ZenDNN capabilities introduced
>       in follow-ups.
>     - **BF16 and FP32 datatypes** — the two dtypes covering the broadest LLM
>       inference workloads.
>     - **Fallback to the native oneDNN kernel for non-supported features** —
>       staged `VDISPATCH_*` checks return `status::unimplemented` cleanly so
>       the dispatcher falls through to `brgemm_matmul_t` / existing reorder
>       impls.
>     - **Validation with AMD and Intel CPUs** — benchdnn correctness + perf
>       evidence on Zen, no-regression on Intel as a hard merge gate.
>
> Detailed acceptance criteria are in
> [§4](#4-first-pr--scope-and-acceptance-criteria).

---

## 1. Motivation

oneDNN is the de-facto CPU primitive library that frameworks (PyTorch,
TensorFlow, ONNX Runtime, vLLM) depend on. On Intel CPUs it is well-tuned; on
AMD CPUs the same primitives leave performance on the table when kernel
selection, memory layout, or threading don't match Zen-microarchitecture
preferences. AMD's [ZenDNN](https://github.com/amd/ZenDNN) library closes that
gap and ships in production today through
[zentorch](https://github.com/amd/ZenDNN-pytorch-plugin).

**Problem.** Every framework that wants competitive AMD-CPU performance must
consume ZenDNN directly, bypassing oneDNN. That fragments the integration story:
improvements reach each framework on its own cadence, and frameworks that
consume only stock oneDNN never see them.

**Proposal.** Registering ZenDNN as a CPU backend in oneDNN through a new
opt-in module (the `zen64` module) for primitives oneDNN provides or is adding
(MatMul and Reorder first; BMM, SDPA, GroupMatMul as follow-ups). This first PR
covers MatMul and its weight Reorder only. This gives every oneDNN consumer the
same AMD-tuned path under existing primitive APIs, with **no framework
source-code change required**. The only framework-side step is a minor build
change: build (or link against) a oneDNN compiled with `DNNL_X64_USE_ZEN=ON` to
enable the `zen64` path. The primitive APIs the framework calls stay
byte-for-byte the same.

The contribution maps onto oneDNN's [Library Functionality
Guidelines](../../../CONTRIBUTING.md#library-functionality-guidelines):

| Criterion | How |
|---|---|
| **Performance** | The PoC already brings ZenDNN's optimised dispatch under oneDNN's MatMul primitive. Performance is validated at model level via vLLM and benchdnn before each production PR. |
| **Generality** | Works through oneDNN's standard primitive API. PyTorch, TensorFlow, ONNX Runtime, vLLM: every consumer that already calls oneDNN benefits with no source-code change — only a build-time opt-in (`DNNL_X64_USE_ZEN=ON` plus the ZenDNN dependency). |

## 2. Goals and Non-Goals

### Goals
- **No public API change.** Invisible to users of `dnnl::matmul`,
  `dnnl::reorder`, etc.
- **No mandatory build-time dependency.** Gated by `DNNL_X64_USE_ZEN` (default
  OFF). A standard `cmake` build with no extra flags produces a oneDNN that is
  byte-for-byte unchanged from today.
- **Graceful fallback.** Any time the `zen64` impl rejects a configuration, the
  existing `impl_list` walks to the next candidate and stock oneDNN runs.
- **Validation evidence in every PR.** benchdnn correctness, benchdnn perf, and
  a representative model-level run (vLLM for LLM workloads).

### Non-Goals
- **GPU paths.** `zen64` is CPU-only.
- **Replacing oneDNN's existing CPU kernels.** The `zen64` impl only *registers
  ahead* of the existing kernels in the impl list on AMD systems; it does not
  delete or modify any existing kernel.

## 3. Proposed Solution: `zen64` Module Design

### 3.1 Architecture overview

`zen64` is a new, opt-in CPU sub-target inside oneDNN's existing `src/cpu/x64/`
tree. The diagram below shows where it sits in the library hierarchy and how it
relates to the (optional, externally linked) ZenDNN library:

```
   user code  (PyTorch · TensorFlow · ONNX Runtime · vLLM · llama.cpp · …)
                                  │
                                  ▼
   ┌──────────────────────── oneDNN Library ───────────────────────────┐
   │                                                                    │
   │   Primitive APIs    dnnl::matmul · dnnl::reorder · dnnl::sdpa · …  │
   │                                  │                                 │
   │   Engines                CPU · GPU · XPU · Graph                   │
   │                                  │                                 │
   │   Architectures      x64 · aarch64 · riscv64 · ppc64 · s390x       │
   │                                  │                                 │
   │   x64 impl space     src/cpu/x64/   (the x86_64 family target)     │
   │                ┌──────────────────────────────────────────────┐    │
   │                │   sibling impls under src/cpu/x64/, peers:     │    │
   │                │                                                │    │
   │                │  ┌────────┐ ┌──────┐ ┌──────────┐ ┌─────┐      │    │
   │                │  │ BRGEMM │ │ GEMM │ │ jit_uni_*│ │ ref │ …    │    │
   │                │  └────────┘ └──────┘ └──────────┘ └─────┘      │    │
   │                │                                                │    │
   │                │  ╔══════════════════════════════════════════╗  │    │
   │                │  ║   zen64    (NEW, opt-in)  ── sibling      ║  │    │
   │                │  ║                                          ║  │    │
   │                │  ║     build:    DNNL_X64_USE_ZEN=ON      ║  │    │
   │                │  ║     runtime:  AMD vendor + AVX-512       ║  │    │
   │                │  ║                (Zen4 and later)          ║  │    │
   │                │  ║                                          ║  │    │
   │                │  ║   • registers ahead in cpu_*_list        ║  │    │
   │                │  ║   • PD::init() validation gate           ║  │    │
   │                │  ║   • execute() → ZenDNN                   ║  │    │
   │                │  ╚════════════════╤═════════════════════════╝  │    │
   │                └───────────────────┼────────────────────────────┘    │
   └────────────────────────────────────┼──────────────────────────────────┘
                                        │  zendnnl::lowoha::*_direct(…)
                                        ▼
                            ┌──────────────────────────┐
                            │      ZenDNN library      │
                            │   (linked when build     │
                            │    flag is ON; default   │
                            │    OFF)                  │
                            └──────────────────────────┘
```

`zen64` lives at `src/cpu/x64/zen64/` as a **sibling** of the other x86_64
impls (BRGEMM, GEMM, `jit_uni_*`, ref), not nested under any of them. Like
every other entry in `src/cpu/x64/`, it is a generic x86_64 backend that
**builds and runs on any x86 CPU**; its `pd_t::init()` gate is simply tuned to
engage on AMD CPUs with AVX-512 (Zen4 and later) so it delivers optimal AMD
performance while falling through cleanly to the existing impls everywhere
else. It contributes one impl class
per primitive (e.g. `zen_matmul_t`, `zen_reorder_t`) registered in the relevant
per-primitive impl list(s) ahead of the existing entries. When the
build flag is OFF (default), the entire `zen64` source set is excluded from
compilation and `libdnnl.so` is byte-identical to today's oneDNN.

#### Placement rationale: why `src/cpu/x64/` and not `src/cpu/`

An alternative placement would move `zen64` one level up into `src/cpu/` (as a
peer of the architecture targets `x64`, `aarch64`, ...). We chose
`src/cpu/x64/` instead because `src/cpu/` is organised so that **each entry
corresponds to a distinct CPU architecture family**, and `x64` is the dedicated
home for the entire x86_64 family. ZenDNN is an x86_64 backend, so it belongs
alongside the other x86_64 impls under `src/cpu/x64/` rather than introducing a
new top-level CPU target. This keeps the directory taxonomy consistent and
makes `zen64` a natural sibling of BRGEMM / GEMM / `jit_uni_*` / ref.

### 3.2 Build-time gating

```cmake
option(DNNL_X64_USE_ZEN
       "Enable ZenDNN integration (link against zendnnl when available)"
       OFF)
```

When `OFF` (default): zero ZenDNN headers referenced, zero ZenDNN library
linked, behaviour byte-identical to today's oneDNN.

When `ON`: oneDNN consumes ZenDNN through its CMake package config, compiles
`src/cpu/x64/zen64/*.cpp`, links `libzendnnl`, and conditionally registers impls
via the `CPU_INSTANCE_X64_ZEN(...)` wrapper (gated by `DNNL_X64_ZEN`; see §3.4).

> **Minimum CMake version.** oneDNN's default build requires CMake **3.13**
> (`cmake_minimum_required` in the top-level `CMakeLists.txt`), whereas both
> ZenDNN and AOCL-DLP require **CMake 3.26**. When `DNNL_X64_USE_ZEN=ON`, the
> effective minimum CMake therefore rises to **>= 3.26**; this is gated behind
> the option so the default `OFF` build keeps oneDNN's existing 3.13 floor
> unchanged.

#### Providing the ZenDNN binary

For the initial integration, the **user supplies a prebuilt ZenDNN binary**: the
oneDNN build links against an existing ZenDNN install rather than building it
from source. The user points the build at that install through `ZENDNNROOT`,
either by exporting it or by passing it on the CMake command line:

```bash
# either export it ...
export ZENDNNROOT=/path/to/ZenDNN/build/install/zendnnl
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DDNNL_X64_USE_ZEN=ON

# ... or pass it directly on the build command
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DDNNL_X64_USE_ZEN=ON \
      -DZENDNNROOT=/path/to/ZenDNN/build/install/zendnnl
```

ZenDNN ships its own CMake package config (`zendnnl-config.cmake`), so oneDNN
consumes it directly through a thin `cmake/ZenDNN.cmake` wrapper. If
`DNNL_X64_USE_ZEN=ON` but `ZENDNNROOT` is unset / no ZenDNN install can be
found, configuration fails with an explanatory error so the dependency is never
silently missing.

Building the ZenDNN binary itself is documented in the
[ZenDNN repository](https://github.com/amd/ZenDNN) (the key being
`ZENDNNL_DEPENDS_ONEDNN=OFF` so ZenDNN does not pull in its own oneDNN); this
RFC links to those instructions rather than duplicating them.

> **In progress.** We are adding a convenience path where, instead of requiring
> a prebuilt binary, the oneDNN build can **fetch and build ZenDNN on the fly**
> (e.g. via CMake `FetchContent` / `ExternalProject` cloning the pinned ZenDNN
> revision and building it as part of the oneDNN configure/build step). When
> that lands, `ZENDNNROOT` remains the override for users who prefer to supply
> their own binary, and the auto-fetch path becomes the default fallback when
> `ZENDNNROOT` is not set.

Exact CMake plumbing (the `ZenDNN.cmake` wrapper plus the other CMake files
touched, link-mode handling, and error messages) is finalised at PR time; this
section captures the user-facing contract rather than every build-file detail.

### 3.3 Runtime gating

There is **no runtime env var**. Once built with `DNNL_X64_USE_ZEN=ON`,
dispatch into `zen64` is decided entirely by the CPU-detection checks (AMD
vendor + AVX-512, i.e. Zen4 and later) inside each `pd_t::init()` (§3.5). On
any CPU or configuration those checks reject,
the impl returns `status::unimplemented` and the next entry in the impl list
runs, so behaviour on non-AMD x86 CPUs is unchanged with no env var to set.

### 3.4 Per-primitive registration

The MatMul impl list (`src/cpu/matmul/cpu_matmul_list.cpp`) gains one new entry,
placed ahead of the stock x64 brgemm entries:

```cpp
constexpr impl_list_item_t impl_list[] = REG_MATMUL_P({
        // ...existing AArch64 entries...
        CPU_INSTANCE_X64_ZEN(zen_matmul_t)   // ahead of stock x64 brgemm
        CPU_INSTANCE_AMX(brgemm_matmul_t<avx512_core_amx>)
        // ...existing entries...
        CPU_INSTANCE(ref_matmul_t)
        nullptr,
});
```

Registration uses the dedicated `CPU_INSTANCE_X64_ZEN(...)` wrapper defined in
`src/cpu/cpu_engine.hpp`:

```cpp
#define CPU_INSTANCE_X64_ZEN(...) DNNL_X64_ZEN(CPU_INSTANCE(__VA_ARGS__))
```

which expands the entry only when both `DNNL_X64` and `DNNL_X64_USE_ZEN` are
set, via `DNNL_X64_ZEN` defined alongside the existing `DNNL_*_ONLY` /
`DNNL_AARCH64_ACL_ONLY` family in `src/cpu/platform.hpp`:

```cpp
#if DNNL_X64 && DNNL_X64_USE_ZEN
#define DNNL_X64_ZEN(...) __VA_ARGS__
#else
#define DNNL_X64_ZEN(...)
#endif
```

### 3.5 PD `init()` validation flow

`zen_matmul_t::pd_t::init()` performs the checks inline (no staged
`*_supported()` helpers). When built without Zen it short-circuits to
`status::unimplemented`; otherwise each `VDISPATCH_MATMUL` failure returns
`unimplemented` and the impl-list iterator advances:

```cpp
status_t zen_matmul_t::pd_t::init(engine_t *engine) {
#if !DNNL_X64_USE_ZEN
    return status::unimplemented;
#else
    // Environment gates
    VDISPATCH_MATMUL(DNNL_CPU_THREADING_RUNTIME != DNNL_RUNTIME_THREADPOOL, ...);
    VDISPATCH_MATMUL(engine->kind() == engine_kind::cpu, ...);
    VDISPATCH_MATMUL(is_dense_format_kind(), ...);
    // AMD-only vendor gate via xbyak (portable across GCC/Clang/MSVC).
    VDISPATCH_MATMUL(::dnnl::impl::cpu::x64::cpu().has(Xbyak::util::Cpu::tAMD), ...);
    // AVX-512 ISA gate; on AMD this means the Zen4 microarchitecture and later.
    VDISPATCH_MATMUL(mayiuse(avx512_core), ...);
    // Shapes / dtypes
    VDISPATCH_MATMUL(ndims() == 2, ...);                 // 2D only
    VDISPATCH_MATMUL(!has_zero_dim_memory(), ...);
    VDISPATCH_MATMUL(all_f32 || all_bf16 || bf16_mixed, ...);
    VDISPATCH_MATMUL(desc()->accum_data_type == f32, ...);
    // Bias / attributes / post-ops / scales / zero-points
    VDISPATCH_MATMUL(check_bias(), ...);
    VDISPATCH_MATMUL(attr()->has_default_values(post_ops | sum_dt, dst_dt), ...);
    VDISPATCH_MATMUL(check_postops(), ...);
    VDISPATCH_MATMUL(attr()->scales_.has_default_values(), ...);
    VDISPATCH_MATMUL(attr()->zero_points_.has_default_values(), ...);
    // Layouts / packing / INT_MAX bounds
    VDISPATCH_MATMUL(!has_runtime_dims_or_strides(), ...);
    VDISPATCH_MATMUL(set_default_formats(), ...);
    // advertise opaque zen_packed weights for format_any (see §3.7)
    VDISPATCH_MATMUL(wei_zen_packed || gemm_based::check_gemm_compatible_formats(*this), ...);
    VDISPATCH_MATMUL(dst matches format_tag::ab, ...);
    VDISPATCH_MATMUL(fits_zen_int_api, ...);             // M/N/K/lda/ldb/ldc <= INT_MAX
    return status::success;
#endif
}
```

The AMD-vendor gate reuses oneDNN's existing cached `Xbyak` CPU descriptor
(`::dnnl::impl::cpu::x64::cpu().has(Xbyak::util::Cpu::tAMD)`), so no new
detection code is introduced. The AVX-512 requirement (`mayiuse(avx512_core)`)
restricts dispatch to AMD parts with AVX-512, which in practice means the
**Zen4 microarchitecture and later** (Zen4 is AMD's first AVX-512 CPU); earlier
Zen generations and other non-AVX-512 x86 CPUs fall through to the existing
oneDNN impls. Supported dtype configs are restricted directly (uniform f32,
uniform bf16, and bf16 src/wei to f32 dst); finer per-dtype ISA gates (e.g.
BF16 / VNNI) may be added as dtype coverage grows. There is no
`book_scratchpad()` call.

### 3.6 `execute()` flow

The primitive implements `execute_body()`. It derives shapes / leading-dims via
`matmul_helper_t`, extracts buffer pointers from the exec context, and
dispatches through a translation-unit-local `zen_matmul_direct(...)` wrapper
that calls ZenDNN's `matmul_direct(...)` and maps the status back via
`to_dnnl_status()`:

```cpp
status_t zen_matmul_t::execute_body(const exec_ctx_t &ctx) const {
    matmul_helper_t helper(src_d, weights_d, dst_d);   // M, N, K, transA/B, lda/b/c
    // packed weights => mem_format_b='r', transB='N', ldb=N; else helper-derived
    const void *A = CTX_IN_MEM (const void *, DNNL_ARG_SRC);
    const void *B = CTX_IN_MEM (const void *, DNNL_ARG_WEIGHTS);
    void       *C = CTX_OUT_MEM(void *,       DNNL_ARG_DST);
    const void *bias = pd()->with_bias() ? CTX_IN_MEM(const void*, DNNL_ARG_BIAS) : nullptr;
    // Post-ops are pre-built once in zen_matmul_t::init(engine_t*); only the
    // binary post-op src1 buffer pointers are patched here from the ctx.
    return zen_matmul_direct(src_dt, wei_dt, dst_dt, bia_dt, A, B, C, bias,
            M, N, K, lda, ldb, ldc, transA, transB, mem_format_b,
            postop_template_, postop_po_indices_, beta_, ctx);
}
```

From the dispatcher's view this is one regular primitive call. The post-op
chain is built from oneDNN attributes at primitive `init()` (owned by the
primitive, not `pd_t`, to keep `pd_t` cheaply copyable for the primitive
cache); `sum` maps to ZenDNN `beta`. ZenDNN's full multi-backend Auto Tuner is
not exercised by this first PR.

### 3.7 Memory descriptor and reorder strategy

The implemented default is an **oneDNN-side reorder / prepack** for `format_any`
weights. When the framework leaves the weights layout open (`format_any`) for
bf16 / f32, the PD advertises the dedicated opaque `format_kind::zen_packed`
weights format (`init_zen_packed_md`) — an **internal** format kind, not a new
public `dnnl_format_kind_t` value. The bytes are produced by `zen_reorder_t`
(the Zen backend packer) via an explicit oneDNN reorder ahead of inference, and
consumed directly by the backend with `mem_format_b='r'`; no oneDNN blocked
layout is involved. A descriptor already in `zen_packed` form is accepted as-is
(no re-pack). At execute, packed weights use `transB='N'`, `ldb=N`.

For plain `ab` / `ba` weights, the helper-derived `transB` / `ldb` are used and
the weights are handed to ZenDNN with `mem_format_b='n'`. ZenDNN's internal
packing-and-caching on first use is currently disabled
(`is_weights_const=false`), so the backend packs per call rather than caching a
prepacked tensor; the oneDNN-side `zen_packed` reorder above is therefore the
intended path for repeated-weight inference.

### 3.8 No public API changes

This module adds: one CMake option (default OFF), new impl classes under
`src/cpu/x64/zen64/`, one extra entry in the relevant per-primitive impl
list(s). **No new
headers in `include/oneapi/dnnl/`, no new C entry points, no new public C++
classes, no new public enum values.** (The `zen64` impl classes such as
`zen_matmul_t` / `zen_reorder_t` are internal, and the internal
`format_kind::zen_packed` of §3.7 is not
part of the public API.) A user program built against today's `libdnnl.so`
continues
to link and run unchanged when `libdnnl.so` is rebuilt with
`DNNL_X64_USE_ZEN=ON`.

### 3.9 Binary size impact

The integration adds **zero size cost to the default build** and a bounded,
opt-in cost only when `DNNL_X64_USE_ZEN=ON`. The numbers below are from the
latest build (`libdnnl` shared in all cases) and report the `libdnnl.so`
footprint; in the `ON` cases this already includes the ZenDNN code (statically
folded in for the archive build, and the `zen64` integration layer in both).

| Build config | `libdnnl.so` | Increase vs baseline |
|---|---|---|
| `DNNL_X64_USE_ZEN=OFF` (default), no ZenDNN | 64.70 MB | baseline (byte-identical to today) |
| `ON`, ZenDNN **shared** (`.so`) | 73.95 MB (~74 MB) | **+14%** |
| `ON`, ZenDNN **archive** (`.a`, static) | 75.95 MB (~76 MB) | **+17%** |

Observations:

- **Default `OFF` build, no change.** `libdnnl.so` stays at the baseline
  64.70 MB and is byte-identical to today's oneDNN; consumers who do not opt in
  see no footprint change whatsoever.
- **ZenDNN shared (`.so`).** `libdnnl.so` grows to ~74 MB, a **~14% increase**
  over the baseline.
- **ZenDNN archive (`.a`, static).** ZenDNN is statically folded into
  `libdnnl.so`, taking it to ~76 MB (a **~17% increase**) with no separate
  runtime artifact, the preferred mode for single-binary / embedded
  deployments.

In all cases the cost is incurred only by builds that explicitly enable ZenDNN;
it is never imposed on the stock oneDNN distribution.

### 3.10 ZenDNN version and compatibility

- **API guarantees.** `zen64` couples to ZenDNN only through its stable LOWOHA
  API. ZenDNN is semantically versioned via `project(ZENDNNL VERSION ...)`
  (currently **6.0.0**), so compatibility is guaranteed within a major version;
  we make no cross-major ABI assumption.
- **Extensibility.** LOWOHA entry points take a per-op metadata struct (e.g.
  `matmul_params`) plus trailing optional parameters with safe defaults. New
  metadata is added as defaulted struct fields and new capabilities as trailing
  defaulted arguments; legacy ABI-preserving overloads are retained so existing
  callers (`zen64` included) keep compiling and linking as the API grows.
- **Version-support policy.** Pin a minimum of **ZenDNN 6.0.0** and support the
  **6.x** series; the floor is bumped deliberately when moving to a new major.
- **How it is checked.** A compile-time guard reads ZenDNN's installed
  `zendnnl_version.hpp` and asserts the major:
  `static_assert(ZENDNNL_VERSION_MAJOR == 6, ...)`. The configure-time floor
  check `find_package(zendnnl 6.0.0 CONFIG REQUIRED)` is enabled as part of
  this work: ZenDNN's generated version file is renamed from
  `zendnnl-version.cmake` to the standard `zendnnl-config-version.cmake` (its
  `write_basic_package_version_file` name and the matching `install(FILES ...)`
  entry) so CMake config mode can read the version. `cmake/ZenDNN.cmake` then
  enforces the floor and additionally guards the major
  (`zendnnl_VERSION_MAJOR EQUAL 6`).

### 3.11 Limitations (OS, compiler, tool versions)

These apply only to the opt-in `DNNL_X64_USE_ZEN=ON` build; the default `OFF`
build is unaffected. The build / tool requirements below are enforced at
configure time. Unmet *runtime* conditions (e.g. a non-AMD CPU or no AVX-512)
are not errors: `zen64` returns `status::unimplemented` and dispatch falls back
to the native oneDNN implementation.

- **OS / architecture.** Linux on x86-64 only. Supported distributions follow
  ZenDNN's tested platforms — Ubuntu 22.04 / 24.04 and RHEL 9.2 / 9.5, plus
  ManyLinux Docker for packaged builds.
- **Compiler.** GCC >= 11.2 or Clang >= 14.
- **CMake.** >= 3.26 when `ON` (to satisfy ZenDNN / AOCL-DLP); the default `OFF`
  build keeps oneDNN's existing 3.13 floor (§3.2).
- **CPU.** AMD with AVX-512, i.e. Zen4 and later (§3.5); other CPUs fall
  through to the existing oneDNN impls.

### 3.12 Feature scope (first PR)

A consolidated view of exactly what the `zen64` path covers in the first PR.
Anything outside this set returns `status::unimplemented` and the native oneDNN
impl runs (§3.5, §4.4).

| Area | In scope (first PR) |
|---|---|
| Operators | MatMul, plus a Reorder for weight prepack |
| Data types | FP32; BF16; BF16 inputs → FP32 output |
| Layouts | 2D, stride-aware (plus opaque `zen_packed` weights, §3.7) |
| Bias | Supported (`check_bias()` in `pd_t::init()`) |
| Fusions / post-ops | MatMul post-ops: `sum`, `eltwise` (relu, gelu_tanh, gelu_erf, tanh, logistic/sigmoid, swish), and `binary` (add, mul) |
| Target CPU | AMD with AVX-512 (Zen4 and later) |
| ZenDNN backend | AOCL-DLP MatMul only |

Anything not listed above is handled by native oneDNN: feature gaps
(convolution, BMM/SDPA, FP16/INT8/INT4, batched shapes, other ZenDNN backends)
are planned for follow-up PRs, while non-AMD / non-AVX-512 CPUs are a permanent
runtime fallback and never a `zen64` target.

**No x86-64 branching.** `zen64` is one additional entry in oneDNN's normal
per-primitive dispatch list (§3.4), not a parallel integration path. Frameworks
keep calling the same `dnnl::matmul`; there is no separate x86-64 code path for
them to maintain, and on any unsupported configuration the existing impl runs
unchanged.

### 3.13 Framework integration and compatibility

**Default-on in frameworks (e.g. PyTorch)?** Not as part of this work. ZenDNN is
a build-time opt-in (`DNNL_X64_USE_ZEN`, default OFF), so a default framework
build is unchanged. Whether a framework enables it — now or in the future — is a
separate framework-side decision outside the scope of this RFC.

**Build-time enablement vs. AMD/Intel selection.** The build never distinguishes
AMD from Intel — that choice is made at runtime, not build time. The build only
decides whether ZenDNN is *wanted/available*: `DNNL_X64_USE_ZEN=ON` is set by
intent or availability (an AMD-targeted build, or auto-on when the `zendnnl`
package is found via `ZENDNNROOT`); otherwise it stays OFF and byte-identical to
today. A single x86-64 build then works everywhere: `zen64` engages only on
AMD + AVX-512 (Zen4+) and returns `status::unimplemented` on Intel (and on
non-AVX-512 AMD), so the native oneDNN kernel runs there. Per-CPU activation is
automatic at runtime (§3.3, §3.5).

**Fused operations (e.g. conv+relu).** ZenDNN MatMul supports `sum`, `eltwise`
(relu, gelu_tanh, gelu_erf, tanh, logistic/sigmoid, swish), and `binary` (add,
mul) post-ops. Convolution — and therefore conv+relu — is not in scope;
anything `zen64` does not implement falls back to the native oneDNN impl. That
fallback is the **permanent design**, not a temporary shim: coverage expands in
follow-up PRs without changing the fallback contract.

**Framework-side performance penalty.** None on the execute path. The only added
cost is a single cheap dispatch check at primitive-descriptor creation, which
bails out fast on non-AMD / unsupported configurations and is amortized by the
primitive cache.

**Primitive cache / iDeep compatibility.** `zen64` uses the standard
`primitive_t` / `pd_t` contract — no separate cache, no change to the
primitive-descriptor key, no public-API change. It is simply another impl behind
the same descriptor, so existing caching — including the Intel PyTorch team's
iDeep primitive caching for x86-64 — keeps working and its cache keys stay
valid.

## 4. First PR — Scope and Acceptance Criteria

### 4.1 ZenDNN library integration in build infra

- New CMake option `DNNL_X64_USE_ZEN` (default OFF).
- New `cmake/ZenDNN.cmake` wrapper consuming ZenDNN's own `zendnnl-config.cmake`
  package config; locates a **user-provided prebuilt ZenDNN binary** via
  `ZENDNNROOT` (export or `-DZENDNNROOT=...`). Configuration fails with an
  explanatory error if `ON` and no install is found. (No `Find*` module needed
  since ZenDNN ships CMake metadata.)
- Raise the minimum CMake version **only when `DNNL_X64_USE_ZEN=ON`** to satisfy
  ZenDNN / AOCL-DLP (>= 3.26); the default `OFF` build keeps oneDNN's existing
  3.13 floor.
- Plus the other CMake files touched to wire the option in (enumerated at PR
  time).
- Linux x86_64 only when ON; other platforms fail at configure time with an
  explanatory error.
- **Done-when:** OFF build's `libdnnl.so` byte-identical to today.

### 4.2 Add ZenDNN MatMul (+ fused) and Reorder in the `zen64` module

- `src/cpu/x64/zen64/` directory with common helpers, status mapping, MD
  translation, per-primitive classes.
- **AOCL-DLP MatMul backend only.** This PR drives ZenDNN's AOCL-DLP MatMul path
  exclusively; no other ZenDNN backend (native, libxsmm, FBGEMM, oneDNN-inner)
  or the multi-backend Auto Tuner is enabled here.
- `zen_matmul_t` registered in `cpu_matmul_list.cpp` ahead of `brgemm_matmul_t`.
- `zen_reorder_t` registered in oneDNN's datatype-based reorder lists
  (`cpu_reorder_regular_f32_f32.cpp`, `cpu_reorder_regular_f32_bf16.cpp`,
  `cpu_reorder_regular_bf16.cpp`) covering ZenDNN-preferred weight layouts
  (enables the ahead-of-time weight reorder / prepack).
- Fused MatMul post-ops: `sum`, `eltwise` (relu, gelu_tanh, gelu_erf, tanh,
  logistic/sigmoid, swish), and `binary` (add, mul). Any other post-op is
  rejected in `pd_t::init()` so the native impl handles it.
- Bias supported (validated via `check_bias()` in `pd_t::init()`).
- **Done-when:** `dnnl::matmul` shows `zendnn:matmul:f32|bf16:amd` in verbose;
  reorder shows `zendnn:reorder:...:amd`.

### 4.3 Supported datatypes (BF16 and FP32)

- BF16 path: src / wei / dst all BF16.
- FP32 path: all FP32.
- Mixed BF16 path: src / wei BF16 with FP32 dst (f32 accumulation).
- 2D MatMul only; batched (>2D) shapes are out of scope for the first PR and are
  rejected in `pd_t::init()` so the existing impl handles them.
- benchdnn diff vs. `--skip-impl=zendnn` within tolerance on all supported dtype
  configs.
- Other dtypes (FP16, INT8, INT4) explicitly out of scope; `pd_t::init()`
  rejects them so the existing impl handles those cases unchanged.

### 4.4 Fallback to native oneDNN kernel

- Staged `VDISPATCH_*` checks in `pd_t::init()` (§3.5).
- Targeted dispatch-test set forces unsupported configurations and asserts the
  next impl runs.
- **Done-when:** dispatch-test set passes; on a rejected configuration the impl
  falls through to `brgemm_matmul_t` with logs identical to a
  `DNNL_X64_USE_ZEN=OFF` build.

### 4.5 Validation with AMD and Intel CPUs

- benchdnn correctness + perf on AMD Zen4+ (AVX-512).
- benchdnn full sweep on Intel with flag ON: identical impl selection and
  results to OFF build (**hard merge gate**).
- vLLM model-level numbers, BF16 / FP32, on a representative LLM set.
- `--skip-impl=zendnn` rerun on the same hosts confirms no regression.

## 5. Alternatives Considered

### Native upstream of every ZenDNN kernel

Upstream every ZenDNN kernel as native oneDNN code, even for primitives oneDNN
already has (MatMul, BMM, SDPA).

- **Pros:** Cleanest end-state; oneDNN owns every kernel.
- **Cons:** Years of porting work for kernels that already exist and ship in
  ZenDNN today. The `zen64` mechanism delivers the same end-user benefit on a
  much faster timeline, with the door open to gradually replacing wrapped
  kernels with native-upstreamed ones over time.

### Direct AOCL-DLP integration (considered)

Call [AOCL-DLP](https://github.com/amd/aocl-dlp) directly from oneDNN, skipping
ZenDNN. AOCL-DLP provides GEMM / MatMul kernels only; ZenDNN adds the dispatch,
weight packing/caching, parallelization, and low-overhead call path around
them, and is the production-validated layer used by zentorch. A direct AOCL-DLP
integration would require reimplementing that orchestration inside oneDNN and
would not extend to the broader AMD operator roadmap (BMM, GroupMatMul, SDPA,
and others). It remains an option to evaluate per-kernel in the future.

