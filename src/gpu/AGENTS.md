# AGENTS.md — src/gpu

See root `/AGENTS.md` for build/test/style/commit conventions and
`src/gpu/README.md` for design background. This file covers GPU-backend
structure only.

## Layout

```
intel/     Intel GPU backend (JIT/nGEN pipeline + OpenCL)
nvidia/    NVIDIA backend        -> src/gpu/nvidia/README.md
amd/       AMD backend           -> src/gpu/amd/README.md
generic/   vendor-agnostic backend (e.g. generic SYCL)
                                 -> src/gpu/generic/sycl/README.md
```

Shared runtime plumbing (queues, streams, memory) that all vendor backends
build on lives in `src/xpu/`, not here — see `src/xpu/README.md`.

## Intel JIT pipeline

`intel/jit/` kernels are built through an IR layer, then lowered to nGEN
assembly (not handwritten OpenCL — see `intel/ocl/` for primitives that use
OpenCL C directly instead). nGEN targets identical encodings to the Intel
Graphics Assembler (IGA); see `src/gpu/intel/jit/README.md` for design background
and links to the Gen assembly reference manuals.

## Implementation registration

Same rule as CPU (`src/cpu/AGENTS.md`): each primitive has an ordered
implementation list at `src/gpu/gpu_<primitive>_list.cpp`. oneDNN tries
implementations in list order and uses the first whose `init()` succeeds.
Adding the kernel files alone does not register an implementation.

## Vendor split

GPU vendor code is split at compile time by `DNNL_GPU_VENDOR`. Wrap
vendor-specific code in `DNNL_GPU_INTEL_ONLY(...)` (defined in
`gpu_impl_list.hpp`), or the matching macro for NVIDIA/AMD/generic.
