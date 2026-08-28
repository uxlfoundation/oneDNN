# AGENTS.md — src/cpu

See root `/AGENTS.md` for build/test/style/commit conventions and
`src/cpu/README.md` for design background. This file covers CPU-backend
structure only.

## Layout

```
x64/       x86-64 JIT kernels (Xbyak-based)
aarch64/   Arm JIT kernels (Xbyak_aarch64-based)
rv64/      RISC-V vector kernels
ppc64/     Power ISA kernels (experimental, limited testing)
s390x/     IBM Z kernels (experimental, limited testing)
*/         portable reference implementations (ref_*), no ISA dependency
```

## Implementation registration

Each primitive has an ordered list at `src/cpu/cpu_<primitive>_list.cpp`.
oneDNN tries implementations in list order and picks the first whose
`init()` succeeds — list order is priority order (JIT before reference).
Adding a new `.cpp`/`.hpp` pair does nothing until it's added to this list.

## JIT kernels

- Extend `jit_generator` (`x64/jit_generator.hpp`, `aarch64/jit_generator.hpp`)
  — wraps Xbyak (x64) / Xbyak_aarch64 (aarch64).
- ISA dispatch goes through `cpu_isa_traits.hpp` (per-arch, under `x64/`,
  `aarch64/`, `rv64/`).
- Reusable JIT building blocks (post-ops, quantization) live under
  `injectors/` (`x64/injectors/`, `aarch64/injectors/`,
  `binary_injector_utils.*`) — check there before writing a new one.
