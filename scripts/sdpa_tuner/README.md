# SDPA forward config tuner

Tools for the model-based tile selection in `src/gpu/intel/sdpa/select.cpp`.

The library picks the microkernel tile configuration of the fused forward
SDPA kernel in one of three ways, chosen with the dev-mode environment
variable `SDPA_CONFIG_SELECT`:

| value    | behaviour                                                        |
|----------|------------------------------------------------------------------|
| `legacy` | hand-tuned table in `configs.cpp` (default)                      |
| `table`  | full-key lookup table, legacy fallback                           |
| `model`  | lookup table, then the cost model's best candidate, then legacy  |

Other dev-mode knobs:

- `SDPA_CONFIG_DUMP_CANDIDATES=N` prints the N best candidates with their
  estimates (`-1` for all) as `fwd_candidate,...` debuginfo lines.
- `SDPA_CONFIG_TABLE_FILE=path` adds `key config` lines to the lookup table.
  A key may end in `:eu<N>` to apply only to devices with N EUs; such a
  line is preferred over the arch-wide one (`src:table_dev` in the verbose
  line). The tuner writes tagged lines with `--table-tag-device`.
- `SDPA_MODEL_COEFS=name=value,...` overrides cost-model coefficients.
- `SDPA_CONFIG=8 ints` forces a config (existing knob).

Every selection prints one `fwd_select,...` line at `ONEDNN_VERBOSE=debuginfo=4`
carrying the lookup key, the chosen config and its derived quantities, the
device description (`hw:`) and the problem description (`prb:`). The key
ends in `:tk0` or `:tk1` for the K layout: with K rows contiguous
(benchdnn `--ktag=abcd`, the driver default) or with the head dimension
contiguous (`--ktag=abdc`, what frameworks pass). The kernel differs per
layout and decode runs up to 3x faster with the second, so sweep the
layouts your workload uses; `problems_ci.txt` covers both.

## Files

- `sdpa_model.py`: ctypes binding to the cost model in the library. The model
  exists once, in `select.cpp`; a DNNL_DEV_MODE build exports it through the
  `dnnl_impl_sdpa_fwd_*` functions in `select_api.cpp`. The scripts find
  `build/src/dnnl.dll` or `libdnnl.so` under the repo, or take `--lib` /
  `$DNNL_LIB`.
- `tune_fwd_config.py`: runs benchdnn's `--sdpa` driver over problems and
  configs and appends rows to a CSV.
- `fit_fwd_model.py`: reports regret/rank correlation of the model and fits
  coefficients. `--check-recorded` flags a CSV whose recorded estimates came
  from a different model version than the loaded library.
- `analyze_landscape.py`: top configs and one-factor marginals per problem.
- `bake_fit.py`: writes a fit's coefficients into the architecture's case
  of `seed_coefs()` and merges `--table-out` lines into the built-in table.
- `report_fails.py`: configs a sweep could not build that the tiler still
  accepts, and configs that ran but the tiler rejects; both lists should
  be empty before a fit is baked.
- `problems_to_batch.py` and `compare_perf.py`: a benchdnn batch file from
  a problems file, and a problem-by-problem comparison of two perf logs
  (`table` versus `legacy` selection).
- `problems_compute.txt`: compute-bound shapes that pin down the systolic
  terms; `problems_ci.txt`: coverage shapes from the benchdnn harness.
- `problems_kpi.txt`: the SDPA KPI shapes (SD UNet, SDXL head sizes, LLM
  prefill and decode with explicit masks, Whisper) reduced to one line per
  selector key, plus int8 K/V cases in the `graph=` form, which runs a
  verbatim benchdnn `--graph` command line for what the `--sdpa` driver
  cannot express. Replace `KPI_DIR` with the directory of the json cases.

## Workflow

1. Build with `DNNL_DEV_MODE=ON` and the GPU runtime.
2. Sweep. Either list problems or draw a design:

       python scripts/sdpa_tuner/tune_fwd_config.py \
           --benchdnn build/tests/benchdnn/benchdnn.exe \
           --design 60 --seed 1 --dtype f16 --topk 12 --random 8 \
           --out sweep_a750.csv --table-out table_a750.txt

   The reference config is re-timed every `--ref-every` runs; stop if the
   drift warning fires repeatedly (the part is throttling). Measurements use
   benchdnn perf mode P with `--max-ms` per config (default 300 ms); the
   10 ms fast mode (`--perf-mode F`) is quicker but leaves the GPU clock
   unsettled and measured 1.5x off on the A750.
   Candidates that fail to build show `status=fail`; add any such family to
   `fwd_describe()` and `sdpa_model.describe()` so the tiler stops emitting
   them.
3. Score the model and fit:

       python scripts/sdpa_tuner/fit_fwd_model.py sweep_a750.csv --check-recorded -v
       python scripts/sdpa_tuner/fit_fwd_model.py sweep_a750.csv --fit --folds 4 --restarts 1

   Try the fitted coefficients without rebuilding by exporting the printed
   `SDPA_MODEL_COEFS` value. Before baking, check that the tiler agrees with
   the sweep on what builds, and add a rule to `fwd_describe()` for any
   family it still accepts:

       python scripts/sdpa_tuner/report_fails.py sweep_a750.csv

4. Bake the fit and promote the verified winners to the lookup table. The
   `--table-out` lines can be tried first through `SDPA_CONFIG_TABLE_FILE`:

       python scripts/sdpa_tuner/fit_fwd_model.py sweep_a750.csv --fit --folds 4 --restarts 1 > fit_a750.txt
       python scripts/sdpa_tuner/bake_fit.py --arch xe_hpg --device A750 --fit fit_a750.txt --table table_a750.txt
       clang-format -i src/gpu/intel/sdpa/select.cpp

   Rebuild, run the `sdpa_select` tests, and confirm the `seed:` regret of
   `fit_fwd_model.py` now matches the fit's `fitted/train` line.
5. Second SKU of an architecture (A770 after A750, B570 after B580): rerun
   the same problems in `table` mode against `legacy`. Shapes that regress
   get a device-tagged line from a sweep with `--table-tag-device`; the
   arch-wide lines stay shared. Occupancy-bound shapes with few work-groups
   are the ones that differ between SKUs.

## Metrics

Regret is `time(config the model ranks first) / time(best measured config)`
per problem. Targets before switching the default away from `legacy`:
geomean regret at or below 1.05, worst case at or below 1.15, and no shipped
table shape slower than 0.97x of today.

Fit with `--folds 4 --restarts 1` and quote the cross-validated line: it is
the out-of-sample regret over every problem, each scored by a fit that did
not see it. The in-sample `fitted/train` line is always better and is only
the number to bake.

## Model form and A750 results

The cost model (`fwd_estimate_cost` in `select.cpp`, called from Python
through `sdpa_model.py`) is, per subgroup and per key iteration: a systolic k-block
chain (each k-block issues `(unroll_m/8) x (unroll_n/simd)` dpas and takes at
least the pipe latency), K/V loads as one message per k-block of `unroll_m`
rows (latency or bandwidth share), softmax, the S round trip through SLM and
two barriers whose cost grows with the subgroups taking part. Work-groups
are scheduled in waves with a contention term;
memory time is bytes over bandwidth with the achieved bandwidth bounded by
bytes in flight. Compute and memory combine with a smooth max. The FMA path
keeps a flop-based term that no sweep has covered yet.

A750 (xe_hpg), 116 problems, 2677 timed configs: the generic sweep, the
SDPA KPI shapes (`problems_kpi.txt`), the coverage shapes derived from the
benchdnn harness (`problems_ci.txt`: both K layouts, grouped-query
attention, head sizes 16 to 512), the compute-bound shapes
(`problems_compute.txt`: long sequences and large head sizes at 500 to
8000 flops per byte) and full landscapes of ten problems. The data are
the CSVs the workflow above writes (`--out` of the top-k sweeps and of the
`--all` landscape sweeps); they are not kept in the tree. Rerunning the
sweeps and `fit_fwd_model.py --fit --folds 4 --restarts 1` reproduces the
table:

| selector | regret geomean | within 5% | within 15% | worst |
|---|---|---|---|---|
| legacy table | 1.259 | 25% | 46% | 2.29 |
| model, cross-validated | 1.082 | 52% | 76% | 1.49 |
| model, fitted on all (baked seeds) | 1.079 | 53% | 76% | 1.47 |
| built-in table hit | 1.000 by construction | | | |

Two lessons from widening the set. Dropping the barrier-per-subgroup and
causal-tail terms moves the cross-validated regret from 1.086 to 1.094 on
the 60-problem set, so both stay. The compute-bound shapes only pin down
the systolic terms when the throughput floor `mma_cycles_per_flop` is free:
held at its first-principles value the same data cross-validate at 1.105
with a worst case of 2.04, free they give the 1.082 above, with the floor
settling at half the nominal rate. The fitter therefore leaves it free by
default. Without compute-bound shapes a fit cannot tell the chain terms
from the floor at all, which is what the first PVC fit showed.

The remaining misses are short-key prefill (K of 77 and 197) and long
prefill at D 80 to 128 in the framework K layout, where the model prefers
a 128-column Q tile over the 64-column tile that wins; neighbouring tiles
of the same size are also confused at D=512. More measurements rather
than more terms are the next step there.

The model's top 12 plus 8 random candidates contained the best measured
config on every problem, which is how the 154 built-in table entries were
produced.
Other architectures use first-principles seeds until swept.
