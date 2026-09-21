# GPU/CPU `gelu_erf` + `dst:mx` (e8m0) Divergence — Minimal Traced Repro

## Root cause

CPU reference math (glibc `erff`, used by benchdnn's reference model, see
`src/common/math_utils.hpp:383`) hard-saturates `gelu_erf(x)` to exactly
`0.0f` or `x` once `|x|` exceeds a threshold (`r_thr = 5.542594480354135f`
in gelu's internal argument space). The GPU JIT eltwise injector
(`src/gpu/intel/jit/eltwise_injector.cpp`) instead uses an independent
Abramowitz–Stegun polynomial approximation of `erf`, which does **not**
hard-saturate — it computes a tiny genuine nonzero residual instead
(bounded well under `1e-7` near the threshold).

This residual is invisible in plain floating-point output (see "Why
f16/bf16/f8 destination types can't show it" below), but becomes an exact,
discrete divergence once passed through **MX (e8m0) group scaling**, whose
rounding boundaries are at exact powers of two with zero tolerance band.

## Minimal testcase

```
--dt=f32:f32:f8_e4m3 --attr-scales=dst:mx:e8m0:1x32 --attr-post-ops=gelu_erf 1x1:1x32
```
with `MM_ERF_DEMO_R=5.6` (a benchdnn `fill_dense_fp_values` override added
for this investigation: pins `SRC=1.0`, `WEI[0]=-5.6`, `WEI[1..31]=-20`).
`K=1` guarantees the GEMM has a single accumulation term (no accumulation-order
ambiguity), making this fully deterministic.

## Stage-by-stage trace

### Stage 1 — GEMM (`K=1`, `accumulator[n] = SRC[0] * WEI[n]`, exact/measured)

| n | value |
|---|---|
| 0 | `-5.5999999` (f32(5.6)) |
| 1..31 | `-20` |

### Stage 2 — `gelu_erf` post-op

| n | CPU (glibc `erff`, measured) | GPU (JIT polynomial approx.) |
|---|---|---|
| 0 | **exactly `-0`** — `\|x\|=5.6 > 5.5426` saturation threshold, hard-saturates | tiny nonzero residual (magnitude ~`1.6e-10`–`3.3e-10`, inferred from Stage 4 — undetectable directly, below f32 compare tolerance) |
| 1..31 | exactly `-0` | ~`0` (deep in saturation tail, both agree) |

### Stage 3 — MX group max (`max(\|post_eltwise\|)` over the 32-element group)

| | value |
|---|---|
| CPU | **exactly `0`** (all 32 elements exactly `0`) |
| GPU | **~`2.3e-10`**, driven entirely by the `n=0` residual (the only non-fully-saturated element) |

### Stage 4 — round to nearest e8m0 (measured)

| | scale |
|---|---|
| CPU | `5.87747e-39` = `2^-127` (e8m0's minimum representable value — `0` has no exact e8m0 encoding) |
| GPU | `2.32831e-10` = `2^-32` (measured `DST_SCALES` output) |

**Divergence:** `2^-32 / 2^-127 = 2^95` — a 95-power-of-two scale mismatch
from a sub-`1e-9` per-element residual. This propagates to the final
output: `exp:0` vs `got:-256` at `[0][DST]`.

## The mechanism, summarized

Identical GEMM inputs → CPU exact-zero saturation vs. GPU tiny residual
(Stage 2) → only visible in the group max because `n=0` is the sole
non-"dead" element (Stage 3) → e8m0's log2-domain rounding has no
tolerance band, so it discretely flips buckets (Stage 4).

Two necessary conditions for this class of error to manifest:
1. **Dominance** — the perturbed element must set/drive the group max.
2. **Boundary proximity** — the group max must land near an e8m0
   power-of-two rounding threshold.

This requires "dead" MX groups: all elements saturate to near-zero except
a single least-saturated (boundary) element, which becomes the group max
purely because everything else is even smaller.

## Why f16/bf16/f8_e4m3/f8_e5m2 destination types can't show this directly

Any fixed-mantissa float format has a *fixed absolute quantization step*
near a given value, and that floor is always far above the actual residual:

| dt | epsilon / min. step near zero | vs. residual (`~8.26e-8` max) |
|---|---|---|
| f32 | `1.19e-7` | ~1.4x (borderline, and masked by compare.cpp leniency) |
| f16 | `9.77e-4` | ~11,800x |
| bf16 | `7.81e-3` | ~94,600x |
| f8_e4m3 | `1.95e-3` (min subnormal) | ~23,600x |
| f8_e5m2 | `1.53e-5` (min subnormal) | ~185x |

benchdnn's `compare.cpp` also has two built-in leniency rules that mask
this residual in raw float compares:
- Near-zero exemption: `fabsf(exp) <= 1e-5 && diff < epsilon_dt(f32)`
  (hardcoded to f32 epsilon regardless of actual output dtype).
- Eltwise-relaxed absolute threshold: `max(epsilon_dt(dt), 2e-5)`.

Only e8m0 MX scaling is a **pure exponent-only, group-level** rounding
step (not a per-element value quantization) with rounding boundaries at
exact powers of two and no absolute-step floor — making it the only
mechanism in this pipeline capable of amplifying a sub-`1e-7` residual
into an exact, discrete, and (via `trh=0`) forcibly-caught divergence.

## Where the saturation threshold comes from

`r_thr = 5.542594480354135f` was reverse-engineered empirically (via
`ctypes` bisection against the actual glibc `erff` binary, not a
mathematical constant), then back-solved through gelu's argument
transform to express it in gelu's `r`-space. It marks the smallest `|x|`
at which `erff(x/sqrt2)` bit-exactly evaluates to `±1.0f` in float32.

## Diagnostic fix (for validation only, not a real fix)

An opt-in environment-gated clamp was added to
`src/gpu/intel/jit/eltwise_injector.cpp`'s `gelu_erf_compute_fwd`:

- `DNNL_GELU_ERF_CPU_SAT_THR` unset/`0`: **bit-identical to original
  code** (verified — no extra instructions even when disabled).
- `DNNL_GELU_ERF_CPU_SAT_THR=1`: clamps GPU output to exactly `0`/`x` past
  `r_thr`, matching CPU/glibc saturation. Confirmed to fully resolve the
  MX-scale divergence in both the deterministic repro above and a
  real-shape RNG-based repro
  (`19x1632:1632x64`, `f4_e2m1` matmul with `src`/`dst`/`wei` scales).

This is a diagnostic artifact tied to one glibc build's saturation
behavior, useful for proving root cause, not a production fix.
