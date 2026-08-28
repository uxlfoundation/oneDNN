# AGENTS.md

oneDNN is a cross-platform C++ library of deep learning building blocks
(convolution, matmul, normalization, RNN, etc.) across multiple CPU and GPU
architectures and vendors. Full platform list: README.md `System
Requirements` section.

For the full contribution process, RFC rules, and style rules, see
`CONTRIBUTING.md` and `CODING_STANDARDS.md`. This file covers what you need
to build, test, and submit a change.

If you're working inside a vendored copy of this tree (pulled in by another
project as a submodule/subtree), you're more likely debugging or tracing
behavior than building or patching it — the outer project already built
oneDNN with its own chosen options. Dependencies/Build below may not match
what's actually running; Architecture and Debugging still apply, since they
describe oneDNN's behavior regardless of who built it. For how the outer
project configures, patches, or upstreams changes to this copy, check its
own documentation.

## Dependencies

Minimum to build at all: a C++11 compiler and CMake 3.13+ — the default
config (CPU-only, OpenMP) needs nothing else.

Extra runtime/build options pull in extra dependencies:

| Config | Requires |
|---|---|
| `ONEDNN_CPU_RUNTIME=TBB` | Threading Building Blocks (TBB) 2017+ |
| `ONEDNN_CPU_RUNTIME=SYCL` or `ONEDNN_GPU_RUNTIME=SYCL` | Intel oneAPI DPC++/C++ Compiler, TBB |
| `ONEDNN_GPU_RUNTIME=OCL` | OpenCL SDK (1.2+) |
| `ONEDNN_GPU_RUNTIME=SYCL` (Intel GPU) | above, plus OpenCL SDK (3.0+) and oneAPI Level Zero 1.11+ |
| `ONEDNN_AARCH64_USE_ACL=ON` | Arm Compute Library 53.1.0+, built separately |
| `ONEDNN_GPU_VENDOR=NVIDIA` | DPC++ compiler with CUDA support, CUDA driver, cuBLAS 10.1+, cuDNN 7.6+ |
| `ONEDNN_GPU_VENDOR=AMD` | DPC++ compiler with HIP support, ROCm 5.3+, MIOpen 2.18+, rocBLAS 2.45.0+ |

Full requirements (exact versions, OS-specific runtime libraries): README.md
`System Requirements` section.

Before configuring a non-default build, check whether the toolchain it needs
is actually present, instead of assuming — a failed cmake configure over a
missing dependency wastes a full round-trip:

- `icx --version` / `icpx --version` — Intel oneAPI DPC++/C++ Compiler present?
- presence of `libtbb`/`tbb.dll`, or `pkg-config --exists tbb` — TBB present?
- presence of an OpenCL loader (`libOpenCL`/`OpenCL.dll`) — OpenCL SDK present?

If something's missing, say so and help the user install it (name the exact
package from the table above) — don't silently fall back to a different
build config instead. Only use the CPU-only OMP default when the user hasn't
asked for GPU/SYCL/TBB at all.

## Build

Out-of-source CMake build only:

```sh
mkdir build && cd build
cmake ..
cmake --build . --parallel <jobs>
```

Common options:

| Option | Default | Values |
|---|---|---|
| `ONEDNN_CPU_RUNTIME` | `OMP` | `OMP`, `TBB`, `SEQ`, `THREADPOOL`, `SYCL` |
| `ONEDNN_GPU_RUNTIME` | `NONE` | `NONE`, `OCL`, `SYCL` |
| `ONEDNN_GPU_VENDOR` | `INTEL` | `INTEL`, `NVIDIA`, `AMD`, `GENERIC` |
| `ONEDNN_TEST_SET` | `CI` | `SMOKE`, `CI`, `NIGHTLY` |
| `ONEDNN_AARCH64_USE_ACL` | `OFF` | Use Arm Compute Library |

Full list: `doc/getting_started/build_options.md`.

Intel GPU build:
```sh
cmake .. -DONEDNN_CPU_RUNTIME=SYCL -DONEDNN_GPU_RUNTIME=SYCL
```

Windows/MSVC setup, DPC++ compiler setup, and IDE builds: `doc/getting_started/build.md`.

## Test

```sh
cmake .. -DONEDNN_TEST_SET=NIGHTLY   # SMOKE (fast), CI (default), NIGHTLY (full)
cmake --build .
ctest --output-on-failure
ctest -R test_convolution            # run one test
```

See `tests/AGENTS.md` for gtests and benchdnn usage.

## Style

- Format C/C++ with `clang-format -style=file -i <file>.cpp` before
  committing. CI checks against **clang-format-18** specifically
  (`.github/workflows/pr-linter.yml`); a different version can produce a
  failing diff on correctly-styled code.
- `clang-tidy` runs in CI; enable it locally with
  `-DONEDNN_USE_CLANG_TIDY=CHECK`.
- Python: format with `black` and `isort`, lint with `flake8`, type-check
  with `mypy` and `pyright`. All run in CI (`.github/python-linting.yml`).
- Use `src`/`dst` naming, not `input`/`output`.
- No `using namespace` in headers.

## Architecture

```
src/common/   engine-agnostic core: primitive descriptors, public C API,
              primitive cache, ONEDNN_VERBOSE logging
src/cpu/      x64/, aarch64/, rv64/ JIT backends + portable reference code
              -> src/cpu/AGENTS.md
src/gpu/      intel/, nvidia/, amd/, generic/ per-vendor backends
              -> src/gpu/AGENTS.md
src/graph/    Graph API: fusion and partitioning on top of the primitive API
              -> src/graph/AGENTS.md
src/xpu/      shared heterogeneous-runtime code (OCL/SYCL/Level Zero),
              used by src/common, src/cpu (SYCL), src/gpu, and src/graph
              -> src/xpu/README.md — vendor-agnostic, not GPU-specific
```

Every primitive backend (CPU and GPU) registers implementations through an
ordered list file — adding the `.cpp`/`.hpp` files alone does not register
an implementation. See the nested `AGENTS.md` above for the exact pattern
per backend.

## Debugging

```sh
ONEDNN_VERBOSE=all ./my_app        # errors, profiling, and dispatch info
ONEDNN_VERBOSE=dispatch ./my_app   # why a given implementation was/wasn't chosen
ONEDNN_MAX_CPU_ISA=AVX2 ./my_app   # cap ISA at runtime
```

Verbose output needs `ONEDNN_VERBOSE=ON` at build time (default) — a
downstream project's build may have it off. Full mode list, filtering, and
output format: `doc/performance/verbose.md`.

Convert a verbose log into a benchdnn reproducer:
```sh
python3 scripts/verbose_converter/verbose_converter.py -i input.log -s True
```

A benchdnn reproducer is the expected way to report a functional or perf bug
(`.github/ISSUE_TEMPLATE/bug_report.md` asks for one) — always produce one
before filing or commenting on an issue:

1. Get a verbose log of the failing/slow call: `ONEDNN_VERBOSE=all`.
2. Feed it to `verbose_converter.py` above to get a `--DRIVER ...
   PROBLEM-DESCRIPTION` line.
3. Confirm it reproduces: `./benchdnn --DRIVER ... PROBLEM-DESCRIPTION`
   (see `tests/AGENTS.md`).

Suspected security vulnerability, not a regular bug: follow `SECURITY.md`
instead of filing a public issue.

## Commits and pull requests

- Commit message: `<scope>: <imperative description>`, e.g.
  `cpu: x64: conv: fix AVX512 padding for dilated kernels`. Subject line
  under 72 characters.
- Rebase before opening or updating a PR — no merge commits.
- New primitives or API/architecture changes need an approved RFC first
  (`rfcs` branch, see `CONTRIBUTING.md`).
- Fill out `.github/pull_request_template.md` (GitHub adds it to the PR body
  automatically). It also asks for performance data on perf PRs and repro
  steps on bug fixes.

## Further reading

Don't re-derive these from source — they're maintained docs:

| Topic | Path |
|---|---|
| Per-primitive math/semantics | `doc/functional_api/primitives/` |
| Attributes (quantization, rounding, determinism) | `doc/functional_api/programming_model/` |
| Graph API | `doc/graph_api/` |
| Performance tuning, verbose log, VTune/profilers | `doc/performance/` |
| int8, DPC++/OpenCL/Level Zero interop, threadpool | `doc/advanced_topics/` |
| Memory formats, data types, naming conventions | `doc/common_concepts/` |

## Third-party code

`third_party/` holds vendored dependencies, e.g. `xbyak`/`xbyak_aarch64`/
`xbyak_riscv` (per-ISA JIT emitters), `ngen` (Intel GPU assembler), `gtest`,
`spdlog`, OpenCL/Level Zero headers, profiling libs (`ittnotify`, `mdapi`).
Don't hand-edit them.
