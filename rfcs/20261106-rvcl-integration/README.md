# RFC: rv64: rvcl adapter — fast integration of vendor-custom operator libraries

## Introduction

This RFC proposes an opt-in adapter that integrates the rvcl library
into oneDNN as **a fast-integration channel for one or more
vendor-custom operator libraries, delivering higher performance on
RV64 hardware than the in-tree JIT currently provides**. The concrete
commitment: **ZTE's custom-extension operators are planned to complete
integration into oneDNN and a performance demonstration within the
next 3 months (by 2027-02)**. Internal adaptation is already in
progress, and the integration mechanism itself is demonstrated today.

## Proposal

### Positioning: a transitional gap-closer

rvcl is proposed to sit exactly there, and nowhere broader: a
fast-integration channel that closes the immediate performance gap
for vendor-custom libraries on RV64 hardware the in-tree JIT does not
yet serve. Concretely, this RFC proposes an instance of Route 2 — one
opt-in external dependency, default OFF — not a new route and not a
permanent parallel layer:

- **Transitional by design.** Once an extension is ratified, vanilla
  toolchains support it, and the in-tree JIT can serve it, kernels
  are expected to move in tree. rvcl's JIT layer is built on a
  vendored xbyak_riscv at the same tag as oneDNN's
  `third_party/xbyak_riscv`, so validated kernel designs and encoding
  experience remain conveyable to Route 1. rvcl's contribution to
  that long-term picture is intentionally small.
- **Complementary to #6406.** The adapter is a Route 2 instance
  (FetchContent, default OFF, ordinary per-PR review scope); Route 1
  remains the stable delivery vehicle for standard RVV and the
  destination for everything else.

### What rvcl is

rvcl (RISC-V Compute Library,
https://github.com/zte-riscv/rvcl) is an open-source project initiated
and maintained long-term by ZTE: Apache-2.0.For a host, rvcl exposes one 
narrow adapter-facing surface with two internal paths:

1. **A runtime kernel registry.** Kernels are indexed by operation x
   dtype tuples and enter through public registration APIs
   (`rvcl_register_kernel` / `rvcl_unregister_kernel`). Vendor kernels
   — including out-of-tree closed-source `.so` — enter as registry
   entries without any change to the rvcl mainline. **ZTE's
   custom-extension operators will be integrated through this path**;
   any further vendor library would join the same way, if that future
   materializes (see Future direction).
2. **A small set of kernels rvcl itself maintains** as transitional 
   gap-closers, expected to graduate into the tree once validated.

The abstraction sits at the kernel level, not the framework level:

```
Capability (what the hardware can do)
    -> Registry (which kernels exist)
    -> Dispatch (which one should run)
    -> Kernel (execute)
Cross-cutting contracts: Scheduler (injectable orchestrator)
                        + Memory (zero-copy import, workspace reporting)
```

- **Capability:** compile-time macros plus runtime hwprobe detection. 
  rvcl compiles against a single rv64gc baseline and selects paths at
  runtime, so one binary covers different VLENs and microarchitectures.
- **Dispatch:** filters and scores candidates by operation, dtype,
  capability, and shape; returns UNSUPPORTED when no candidate exists,
  so the host gets a clean fallback.
- **Kernel lifecycle:** every kernel implements a validate/configure/run 
  three-phase lifecycle; `run()` must not allocate dynamically, must not
  create threads, takes all memory from pack slots, and correctly executes
  any legal work-range slice. The contract is written into CONTRIBUTING and 
  enforced for all contributions.

The integration surface for oneDNN therefore converges to:

1. **A narrow interface.** The public API is a set of C ABI headers
   (`include/rvcl/*.h`) over an internal C++17 implementation; the
   adapter does nothing but memory-descriptor translation and a
   dispatch call — no language boundary, no framework-level callbacks.
2. **Parallelism belongs to the host.** Kernels never create threads;
   parallel needs are expressed as work ranges. The first PR uses
   oneDNN `parallel()` splitting the M dimension as the only source of
   parallelism.
3. **Aligned memory conventions.** Tensors are imported zero-copy;
   kernels report workspace requirements and the host allocates —
   mapping directly onto oneDNN's scratchpad mechanism.
4. **Clean fallback.** Dispatch returns UNSUPPORTED when it cannot
   serve a request; the adapter lets the driver list fall through,
   with no risk of a wrong selection.

### The adapter: first PR scope

An adapter under `src/cpu/rv64/rvcl/` plus one registration in the
matmul driver list, compiled out entirely when the option is off.

| Dimension | Scope | Rationale |
| --- | --- | --- |
| Operator | matmul | demonstrated capability |
| dtypes | f16 | conservative starting point |
| Feature surface | plain layouts, no post-ops, no bias in the first PR (bias-shape alignment to follow) | conservative predicates; misses fall through, no wrong-selection risk |
| Build | `RV64_USE_RVCL` default OFF, CMake FetchContent pinned to a reviewed commit, `DNNL_RV64_USE_RVCL` defined when enabled; OFF build byte-identical | #6406 Route 2; ZenDNN precedent |
| Registration slot | ahead of `rvv_matmul_t` | rvcl judges whether it can serve the request first; fall-through to oneDNN entries intact |
| Threading | oneDNN `parallel()` splitting the M dimension as the only source of parallelism; kernels single-threaded | eliminates a second parallel framework |
| Integration surface | adapter + one registration macro, compiled out when not enabled | same shape as the ZenDNN integration |
| Validation | per Testing aspects | #6406 validation clauses; ZenDNN acceptance gates |

Timing: the first implementation PR is expected within 2027-02, with
the validation data attached to show the performance improvements.

### Alternatives considered: why not in-tree contributions

For open kernels of ratified standard extensions this is the preferable
route, and rvcl does not compete there.The motivation of rvcl is where 
vendor-custom extensions sit in their lifecycle. RISC-V silicon regularly
ships compute capabilities beyond ratified RVV ahead of the standards 
that would make them portable —
[#6320](https://github.com/uxlfoundation/oneDNN/issues/6320) is a
concrete instance of hardware whose capabilities generic RVV kernels
cannot exploit. Before a custom extension is ratified, upstream
vanilla toolchains and OSes do not support it, so a vendor develops —
in parallel with the silicon — its own toolchain support, kernel
patches, and an operator library just to validate internal prototypes.
That operator library always grows up internally first: internal
prototype validation, landing in key products, closing the commercial
loop. Only afterwards, when users of the vendor's products need the
library beyond those products, does building an upstream ecosystem —
in step with toolchain and OS — become the question. SpacemiT's
planned open-source kernel library follows exactly this pattern, and
ZTE's custom-extension operator library is now at that point.

In that window the library exists and the hardware ships, but oneDNN
users on that hardware have no path to the performance: the in-tree
JIT serves standard RVV, and kernels for an unratified vendor
extension cannot enter the tree at all — the tree cannot build or
carry code that vanilla toolchains do not compile, the spec
underneath is not frozen, and in some cases the sources are not
public yet. During the window, however, the tree cannot host these
libraries at all: vanilla toolchains do not compile kernels for 
unratified vendor extensions, the specs underneath are not frozen,
and in some cases the sources are not public. Runtime registration,
by contrast, accepts a finished vendor library as-is.

### API and ABI backwards compatibility

No changes to the oneDNN public API or ABI. The adapter is
self-contained under `src/cpu/rv64/rvcl/` and adds one registration
to the matmul driver list; it is compiled out entirely when the build
option is off, and an OFF build is byte-identical to a build without
this change.

### Performance implications

The target is higher performance than the in-tree `jit:rvv`
implementation on the served RV64 hardware. The first implementation
PR will attach a benchdnn performance comparison against `jit:rvv`;
the integration and performance demonstration of ZTE's
custom-extension operators are planned by 2027-02. Unsupported cases
fall through to the in-tree implementations, so non-target
configurations are not affected.

### Build system changes

A new CMake option `RV64_USE_RVCL`, default OFF. When ON, rvcl is
pulled in via CMake FetchContent pinned to a reviewed commit, and
`DNNL_RV64_USE_RVCL` is defined for the adapter sources. When OFF, no
new files are compiled and the build is byte-identical. Frameworks
that build the library directly from sources are unaffected unless
they opt in.

### Dependencies and support matrix

Adds one optional dependency: rvcl (Apache-2.0), which itself vendors
xbyak_riscv 1.31 (BSD-3) — the same origin and version as oneDNN's
`third_party/xbyak_riscv`. No new required dependencies; default
builds and all non-RV64 configurations are unaffected. rvcl compiles
against a single rv64gc baseline, so no special toolchain flags enter
the oneDNN build.

### Execution plan

1. First implementation PR (expected 2027-02): adapter + f16 plain
   matmul, with a benchdnn performance comparison against `jit:rvv`
   attached.
2. ZTE's custom-extension operators: integration and performance
   demonstration by 2027-02.
3. Follow-on operator order (conv/brgemm/grouped) per community
   feedback.

### Future direction — not committed by this RFC

- **Multi-vendor registration.** The mechanism is ready (runtime
  registration APIs landed, MVENDORID probed); contact has been
  established with SiFive regarding SKL, and SpacemiT's planned
  open-source library would fit the same channel. Further vendor
  participation is a possibility this RFC does not claim.
- **Incubation of kernels for draft-stage extensions** inside rvcl,
  with hand-over to Route 1 as specs freeze.
- Broader coverage (int8/bf16/post-ops, conv/brgemm), scheduler
  injection, and longer-horizon plans are tracked in rvcl's own
  roadmap in the rvcl repository — rvcl project planning, not oneDNN
  commitments.

## Open Questions

1. **Registration order.** The first PR registers ahead of
   `rvv_matmul_t` (oneDNN's own JIT implementation) and ahead of the
   generic gemm: rvcl first judges whether a vendor custom kernel can
   serve the request, then dispatch falls through to oneDNN's entries
   in order. Is that acceptable?
2. **Dependency form.** We plan to introduce it via CMake FetchContent
   pinned to a reviewed commit — does upstream prefer another form
   (for example a submodule)?
3. **Threading model.** The first PR uses oneDNN `parallel()` splitting
   the M dimension as the only source of parallelism, with rvcl
   providing single-threaded kernels only. Any objections?
4. **Acceptance gates.** We propose the first PR be accepted when: the
   benchdnn differential with `--skip-impl=gemm:rvcl` is fully green;
   there is zero regression on non-target configurations; the OFF
   build is byte-identical; and size and performance are measured with
   the option ON and OFF. Anything to add or change?
5. **Follow-on operator order.** After matmul, we solicit priorities
   for conv/brgemm/grouped (we will adjust the rvcl roadmap based on
   oneDNN community feedback).
6. **Naming.** The `src/cpu/rv64/rvcl/` directory, the
   `DNNL_RV64_USE_RVCL` macro, and the impl name `gemm:rvcl` — do
   these match upstream conventions?

We also welcome: feedback from hardware vendors on the registration
mechanism and the vendor-probe interface; suggestions on how to reuse
the validation assets (QEMU reproduction, differential corpora); and a
realistic assessment of the near-term pace.
