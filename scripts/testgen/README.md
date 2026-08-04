# scripts/testgen — oneDNN GPU Nightly Testset Generator

A coverage-driven tool for generating compact benchdnn testsets from observed
GPU nightly run data.

## Problem

The GPU nightly testing suite historically contained **7,837–33,308** tests per
primitive — far more than necessary for good coverage.  Many cases differ only
in shape while covering the same combination of features (datatype, format, post-op
category, etc.).  Running all of them wastes CI time and makes triaging failures
harder.

## Approach: pairwise combinatorial coverage (2-way covering arrays)

Each benchdnn test case is modelled as a vector of *semantic feature values*
across a set of *coverage dimensions*:

| Dimension | Examples |
|-----------|---------|
| `dt` / `sdt` / `ddt` | `f16`, `bf16`, `s8`, `u8`, `f8_e4m3`, … |
| `dir` | `FWD_D`, `BWD_D`, `BWD_W`, `BWD_WB` |
| `stag` / `wtag` / `dtag` | `ab`, `ba`, `axb`, `aBx16b`, `any`, … |
| `bia_dt` | `undef`, `f32`, `bf16`, … |
| `postops_cat` | none, eltwise-only, binary-injector, sum+eltwise, … |
| `scales_cat` | none, common, per-oc, per-tensor, … |
| `zp_cat` | none, src-only, dst-only, both, … |
| `shape_spatial` | 1d, 2d, 3d |
| `shape_size` | tiny, small, medium, large, xlarge |
| `has_groups` | none, regular, depthwise |
| … | (primitive-specific) |

A **coverage requirement** is either:
- a **1-way** requirement `(dim = val)` — "at least one test with this value"
- a **2-way** requirement `(dim_i = val_i, dim_j = val_j)` — "at least one test
  with both values simultaneously"

The generator collects all requirements observed in the nightly logs (not all
theoretically possible combinations — only those that actually occur and were
PASSED), then uses a **greedy max-heap algorithm** to select the smallest subset
of tests that covers all of them:

```
requirements ← all observed 1-way + 2-way pairs
heap         ← max-heap of (uncovered_count, test_index) for each test
selected     ← []

while requirements not empty and budget not exhausted:
    pop test t with highest uncovered_count (recomputing lazily)
    add t to selected
    mark all requirements covered by t as satisfied
    if 100% coverage achieved: stop early
```

The lazy heap recomputation ensures O(n log n) complexity while guaranteeing
the greedy-optimal selection.  After pairwise saturation, any remaining budget
is filled by prioritizing tests that introduce new *shape* classes not yet seen.

### Why pairwise?

Pairwise (2-way) combinatorial coverage has strong empirical backing:
most real bugs are triggered by interactions between at most two parameters.
Going to 3-way offers diminishing returns at greatly increased cost.  For GPU
driver testing, pairwise is the right tradeoff.

## Results (from BMG GPU nightly, June 2026)

All 8 primitives achieve **100% pairwise coverage** with 87–95% fewer tests:

| Primitive   | Original | Generated | Reduction | Pairwise coverage |
|-------------|----------|-----------|-----------|-------------------|
| binary      | 21,213   | 1,000     | **95%**   | 100%              |
| conv        | 32,836   | 2,000     | **94%**   | 100%              |
| lnorm       | 7,837    | 1,000     | **87%**   | 100%              |
| matmul      | 20,093   | 2,000     | **90%**   | 100%              |
| pool        | 22,109   | 1,000     | **95%**   | 100%              |
| prelu       | 12,248   | 1,000     | **92%**   | 100%              |
| resampling  | 12,502   | 1,000     | **92%**   | 100%              |
| softmax     | 9,811    | 1,000     | **90%**   | 100%              |

The generated testsets were validated on BMG GPU: **10/10 PASSED** per primitive.

## Usage

### 1. Prepare log files

Collect a GPU nightly run and extract the PASSED `__REPRO` lines into plain
text files (one command per line), named `<primitive>_nightly_log.txt`.  Place
them in a directory, e.g. `./logs/`.

Example extraction from an MHTML nightly log:
```python
import quopri

with open("matmul_log.mhtml", "rb") as f:
    text = quopri.decodestring(f.read()).decode("utf-8", errors="replace")

with open("logs/matmul_nightly_log.txt", "w") as out:
    for line in text.splitlines():
        if "PASSED" in line and "__REPRO" in line:
            idx = line.find("__REPRO: ")
            if idx >= 0:
                cmd = line[idx + 9:].strip()
                cmd = " ".join(t for t in cmd.split()
                               if not t.startswith("--mode-modifier"))
                out.write(cmd + "\n")
```

Each line should look like:
```
--matmul --engine=gpu --dt=f16:f16:f16 --stag=ab --wtag=ba 10x30:30x20
```

### 2. Run the generator

```bash
python3 generate_testset.py \
    --log-dir ./logs \
    --output-dir ./generated_testsets \
    --max-tests-large 2000 \
    --max-tests-small 1000 \
    --verbose
```

Output files are written to `./generated_testsets/<primitive>_generated.txt`.
Each line is a ready-to-use benchdnn batch file entry (with `--reset`, without
`--engine`/`--<primitive>` flags):

```
--reset --dt=f16:f16:f16 --stag=ab --wtag=ba 10x30:30x20
```

### 3. Run with benchdnn

```bash
benchdnn --matmul --engine=gpu \
    --batch=generated_testsets/matmul_generated.txt
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--log-dir DIR` | `./logs` | Directory containing `*_nightly_log.txt` files |
| `--output-dir DIR` | `./generated_testsets` | Output directory for batch files |
| `--max-tests-large N` | `2000` | Budget for matmul and conv |
| `--max-tests-small N` | `1000` | Budget for all other primitives |
| `--seed N` | `42` | Random seed for tie-breaking |
| `--verbose` | off | Print per-dimension coverage details |

## Supported primitives

`binary`, `conv`, `lnorm`, `matmul`, `pool`, `prelu`, `resampling`, `softmax`

Adding a new primitive requires:
1. Defining its coverage dimensions in `COVERAGE_DIMS`
2. Adding a feature extractor branch in `extract_features()`

## Black-box augmentation

In addition to the log-driven pairwise testsets, a complementary set of
**500 black-box tests** (250 matmul + 250 conv) covers input-space corners
that are unlikely to appear in production nightly logs:

| Category | Matmul examples | Conv examples |
|----------|----------------|---------------|
| Edge dimensions | GEMV (M=1), GEMM-N=1, K=1, M=N=K=1 | IC=1 (grayscale), IC=3 (RGB), OC=1, IC=OC=1 |
| GPU tail sizes | 127×127, 129×129, 255×255, 257×257, 511×511 | prime spatial sizes (7, 13, 27) |
| Layout corners | `ba:ba`, `bca:cab`, transposed 3D perms | `axb` (NXC), asymmetric strides `sh≠sw` |
| Kernel variety | — | 1D (kw=1..21), 3D (3×3×3, 1×3×3), asymmetric (1×3, 3×1, 1×7, 7×1), even (2×2, 4×4) |
| Dilation | — | dh=1,2,3,7; dilated depthwise; dilated 1D |
| Depthwise | — | g=IC=OC for g=16,32,64,128,256 with asymmetric/dilated kernels |
| Attention | Q·Kᵀ decode (M=1, N=64), 4D `abcd` multiply | — |
| LLM FFN | token×d_model, single-token inference | — |
| int4 group sizes | 16, 32, 64, 128, 256, 512 | — |
| fpmath modes | tf32, bf16, f16 applied to f32 src/dst | — |

Generate the black-box augmentation files:

```bash
python3 gen_blackbox_tests.py
# Produces: matmul_blackbox.txt  conv_blackbox.txt
```

These are referenced from the nightly test drivers:
- `tests/benchdnn/inputs/matmul/test_matmul_gpu`
- `tests/benchdnn/inputs/conv/test_conv_gpu`

### Validation against known bugs

| Bug | Hardware | Testset catches? | Notes |
|-----|----------|-----------------|-------|
| MFDNN-14935 (f4_e2m1 mx-scale) | Xe3LPG | ✅ Yes (exact shape in log-driven set) | |
| MFDNN-14869 (bf16:f8_e4m3 ab/ba layout) | PVC | ✅ Yes (targeted augmentation in `harness_matmul_new_nightly_gen`) | Not in log; added via `TARGETED_AUGMENTATIONS` |
| MFDNN-15146 (f16:s4 fpmath abc) | XeHPG | ✅ Yes (shape variant in log-driven set) | |
| MFDNN-15114 (21M-column extreme shape) | XeHPG | ⚠️ Out of scope | Only ref impl; budget-constrained testset cannot include extreme shapes |
