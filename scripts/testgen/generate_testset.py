#!/usr/bin/env python3
################################################################################
# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
"""
oneDNN benchdnn testset coverage-aware generator.

Analyzes benchdnn nightly log files (plain-text, one command per line) and
generates reduced testsets with equivalent or better pairwise combinatorial
coverage.

Log file format
---------------
Each log file must be a plain-text file named ``<primitive>_nightly_log.txt``
containing one benchdnn command per line (PASSED tests only), e.g.::

    --matmul --engine=gpu --dt=f16:f16:f16 --stag=ab --wtag=ba 10x30:30x20

These files are produced by extracting PASSED ``__REPRO`` lines from GPU
nightly run HTML/MHTML output. They are not stored in the repository; users
supply them from their own nightly runs.

Usage:
    python3 generate_testset.py [options]

Options:
    --log-dir DIR        Directory containing *_nightly_log.txt files
                         [default: ./logs]
    --output-dir DIR     Directory for generated batch files
                         [default: ./generated_testsets]
    --max-tests-large N  Max tests for matmul and conv [default: 2000]
    --max-tests-small N  Max tests for all other primitives [default: 1000]
    --seed N             Random seed for reproducibility [default: 42]
    --verbose            Print detailed per-dimension coverage breakdown
"""

import argparse
import heapq
import os
import random
import re
import sys
from collections import Counter, defaultdict
from functools import reduce
from itertools import combinations
from operator import mul


# ─── Primitive metadata ──────────────────────────────────────────────────────

LARGE_PRIMITIVES = {"matmul", "conv"}

# Coverage dimensions per primitive.
# Each entry is (dimension_key, description) used to build pairwise pairs.
COVERAGE_DIMS = {
    "matmul": [
        "dt", "stag", "wtag", "dtag", "bia_dt",
        "postops_cat", "scales_cat", "zp_cat",
        "rdm_cat", "shape_ndims", "shape_size", "batch_cat",
    ],
    "conv": [
        "dt", "dir", "stag", "dtag", "bia_dt",
        "postops_cat", "scales_cat", "zp_cat",
        "shape_spatial", "kernel_cat", "has_groups", "shape_size",
    ],
    "binary": [
        "sdt", "ddt", "stag_cat", "alg",
        "postops_cat", "has_scales", "inplace",
        "shape_ndims", "shape_size", "has_broadcast",
    ],
    "lnorm": [
        "dt", "dir", "flags_cat", "stat_tag", "tag_cat",
        "inplace", "shape_ndims", "shape_size",
    ],
    "pool": [
        "dt", "dir", "alg", "tag_cat",
        "postops_cat", "shape_spatial", "shape_size",
    ],
    "prelu": [
        "sdt", "dir", "stag_cat", "shape_ndims", "shape_size",
    ],
    "resampling": [
        "sdt", "ddt", "alg", "tag_cat",
        "postops_cat", "dir", "shape_spatial", "shape_size",
    ],
    "softmax": [
        "sdt", "ddt", "alg", "axis_cat", "dir",
        "stag_cat", "postops_cat", "inplace",
        "shape_ndims", "shape_size",
    ],
}


# ─── Targeted augmentations ───────────────────────────────────────────────────
#
# When pairwise coverage misses known-important combinations because the source
# nightly log contained an incomplete parameter sweep (e.g., only 3-D layouts
# for a given dtype), these entries inject synthetic commands.
#
# Each entry: (primitive, dtype_regex, [extra_command_strings])
# The extra commands are bare benchdnn args (no --primitive / --engine),
# identical to what load_log returns.  They are prepended to the test pool and
# marked as forced so the greedy selector always picks them.
#
# Reference bugs driving each augmentation are noted in comments.
TARGETED_AUGMENTATIONS: list[tuple[str, str, list[str]]] = [
    # MFDNN-14869 (PVC): bf16:f8_e4m3:bf16 with 2-D transposed layouts (ab/ba)
    # failed because the nightly log only contained 3-D (abc) shapes for this
    # dtype, so the pairwise algorithm never generated a 2-D ab/ba test.
    # The bug triggers in the no-scales path; attr-scales changes the kernel
    # selection and avoids the buggy code path.
    (
        "matmul",
        r"^bf16:f8_e4m3:bf16$",
        [
            "--dt=bf16:f8_e4m3:bf16 --stag=ab --wtag=ba --dtag=ab"
            " --skip-impl=ref 1x25:25x32",
            "--dt=bf16:f8_e4m3:bf16 --stag=ab --wtag=ba --dtag=ab"
            " --skip-impl=ref 256x512:512x1024",
        ],
    ),
]


def get_augmentations(cmds: list[str], prim: str) -> list[str]:
    """Return forced synthetic commands for *prim* based on dtypes found in cmds.

    Scans the dt/sdt values present in *cmds*; for each TARGETED_AUGMENTATIONS
    entry that matches, adds the synthetic commands.  Only adds commands not
    already present in *cmds*.
    """
    import re as _re
    existing = set(cmds)
    dt_values: set[str] = set()
    for cmd in cmds:
        for tok in cmd.split():
            if tok.startswith("--dt="):
                dt_values.add(tok[5:])

    result: list[str] = []
    for aug_prim, pat, extra_cmds in TARGETED_AUGMENTATIONS:
        if aug_prim != prim:
            continue
        rx = _re.compile(pat)
        if any(rx.match(v) for v in dt_values):
            for ec in extra_cmds:
                if ec not in existing:
                    result.append(ec)
    return result


# ─── Log parsing ─────────────────────────────────────────────────────────────

def load_log(path: str) -> list[str]:
    """Read a plain-text log file and return unique benchdnn commands.

    Each line must be a full benchdnn command (as printed in PASSED __REPRO
    output).  Empty lines and comment lines starting with '#' are ignored.
    Duplicate commands are silently dropped.
    """
    seen: set = set()
    cmds: list = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line not in seen:
                seen.add(line)
                cmds.append(line)
    return cmds


def parse_flags(cmd: str) -> tuple[dict, list]:
    """Parse a benchdnn command into (flags_dict, positional_args)."""
    flags: dict = {}
    positional: list = []
    tokens = cmd.split()
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            if "=" in t:
                k, v = t[2:].split("=", 1)
                flags[k] = v
            else:
                flags[t[2:]] = True
        elif not t.startswith("-"):
            positional.append(t)
        i += 1
    return flags, positional


_PRIMITIVE_FLAGS = {
    "--matmul", "--conv", "--binary", "--lnorm",
    "--pool", "--prelu", "--resampling", "--softmax",
}

def clean_cmd(cmd: str) -> str:
    """Strip global flags (primitive selector, engine, mode-modifier) from a
    repro line, leaving only per-test-case options suitable for a batch file.
    """
    parts = []
    for tok in cmd.split():
        if tok in _PRIMITIVE_FLAGS:
            continue
        if tok.startswith("--engine=") or tok == "--engine":
            continue
        if tok.startswith("--mode-modifier"):
            continue
        parts.append(tok)
    return " ".join(parts)


# ─── Feature extraction utilities ────────────────────────────────────────────

def postops_cat(v: str | None) -> str:
    """Classify post-ops string into a structural category."""
    if not v:
        return "none"
    ops = v.split("+")
    cats: set = set()
    for op in ops:
        name = op.split(":")[0].lower()
        if name == "sum":
            cats.add("sum")
        elif name in ("relu", "clip", "swish", "linear", "tanh", "gelu",
                      "abs", "sqrt", "logistic", "exp", "pow", "log",
                      "mish", "hardswish", "hardsigmoid", "round"):
            cats.add("eltwise")
        elif name in ("add", "mul", "sub", "div", "max", "min", "prelu"):
            cats.add("binary_op")
        else:
            cats.add("other")
    return "+".join(sorted(cats)) if cats else "none"


def scales_cat(v: str | None) -> str:
    """Classify attr-scales into broad categories."""
    if not v:
        return "none"
    if "src" in v and "wei" in v and "dst" in v:
        return "src+wei+dst"
    if "src" in v and "wei" in v:
        return "src+wei"
    if "wei" in v and "per_oc" in v:
        return "wei_per_oc"
    if "wei" in v:
        return "wei_only"
    if "src" in v or "dst" in v:
        return "src_or_dst"
    return "other"


def zp_cat(v: str | None) -> str:
    """Classify attr-zero-points into broad categories."""
    if not v:
        return "none"
    if "src" in v and "wei" in v and "dst" in v:
        return "src+wei+dst"
    if "src" in v and "dst" in v:
        return "src+dst"
    if "wei" in v and ("per_oc" in v or "per_ocic" in v):
        return "wei_per_oc"
    if "wei" in v:
        return "wei_only"
    if "src" in v:
        return "src_only"
    return "other"


def is_quantized(dt: str) -> bool:
    return any(q in dt for q in ("u8", "s8", "s32", "u4", "s4", "f8"))


def tag_cat(v: str | None) -> str:
    """Simplify a format tag to its structural family."""
    if not v:
        return "default"
    v = v.split(":")[0]  # take first if tag pair
    if v in ("any", "undef"):
        return v
    if re.search(r"[AB][a-z]+\d+[ab]", v):
        return "blocked"
    if v in ("axb", "abx", "acdeb", "acdb", "acb"):
        return "transposed"
    if v in ("ab", "abc", "abcd", "abcde"):
        return "plain"
    return "other"


def stag_cat(v: str | None) -> str:
    """Simplify src/stag pair tag."""
    if not v:
        return "default"
    parts = v.split(":")
    p0 = tag_cat(parts[0])
    p1 = tag_cat(parts[1]) if len(parts) > 1 else "default"
    return f"{p0}:{p1}"


# ─── Shape analysis ───────────────────────────────────────────────────────────

def _parse_spatial(s: str) -> dict:
    """Extract named spatial fields from a conv/pool/resampling shape string."""
    result: dict = {}
    for field in ("mb", "g", "ic", "oc",
                  "id", "ih", "iw", "od", "oh", "ow",
                  "kd", "kh", "kw", "sd", "sh", "sw",
                  "pd", "ph", "pw", "dd", "dh", "dw"):
        m = re.search(rf"(?<![a-z]){field}(\d+)", s)
        if m:
            result[field] = int(m.group(1))
    return result


def spatial_dims_cat(shape_str: str) -> str:
    sp = _parse_spatial(shape_str)
    if "id" in sp or "od" in sp:
        return "3D"
    if "ih" in sp or "oh" in sp:
        return "2D"
    if "iw" in sp or "ow" in sp:
        return "1D"
    return "unk"


def kernel_cat(shape_str: str) -> str:
    sp = _parse_spatial(shape_str)
    kh = sp.get("kh", sp.get("kw", 0))
    kw = sp.get("kw", kh)
    k = max(kh, kw)
    if k == 1:
        return "1x1"
    if k <= 3:
        return "3x3"
    if k <= 5:
        return "5x5"
    return "large"


def has_groups_cat(flags: dict, shape_str: str) -> str:
    sp = _parse_spatial(shape_str)
    g = sp.get("g", 1)
    if g > 1:
        ic = sp.get("ic", 1)
        oc = sp.get("oc", 1)
        return "depthwise" if g == ic == oc else "grouped"
    return "no"


def shape_ndims_matrix(pos: list) -> str:
    """Return ndims of a matrix shape like '10x30:30x20'."""
    if not pos:
        return "unk"
    s = pos[0].split("_n")[0]  # strip _n"name" suffix
    part = s.split(":")[0]
    return str(part.count("x") + 1)


def shape_size_bucket(pos: list, is_spatial: bool = False) -> str:
    """Return a size bucket for the shape."""
    if not pos:
        return "unk"
    s = pos[0].split("_n")[0]
    try:
        nums = [int(x) for x in re.findall(r"\d+", s)]
        if not nums:
            return "unk"
        # Use product of a few dominant dimensions as proxy for complexity
        size = reduce(mul, sorted(nums, reverse=True)[:4], 1)
    except Exception:
        return "unk"
    if size < 1_000:
        return "xs"
    if size < 10_000:
        return "sm"
    if size < 500_000:
        return "md"
    if size < 10_000_000:
        return "lg"
    return "xl"


def batch_cat(pos: list) -> str:
    """Classify matmul batch dimension."""
    if not pos:
        return "unk"
    s = pos[0].split("_n")[0].split(":")[0]
    dims = [int(x) for x in re.findall(r"\d+", s)]
    ndims = len(dims)
    if ndims <= 2:
        return "no_batch"
    batch = reduce(mul, dims[:-2], 1)
    if batch == 1:
        return "batch_1"
    if batch <= 8:
        return "batch_small"
    return "batch_large"


def has_broadcast(pos: list) -> str:
    """Check if binary shape has a broadcast (dim=1 in one tensor)."""
    if not pos:
        return "no"
    s = pos[0]
    if ":" not in s:
        return "no"
    parts = s.split(":")
    dims0 = re.findall(r"\d+", parts[0])
    dims1 = re.findall(r"\d+", parts[1])
    for d0, d1 in zip(dims0, dims1):
        if d0 != d1:
            return "yes"
    return "no"


# ─── Per-primitive feature extraction ─────────────────────────────────────────

def extract_features(cmd: str, prim: str) -> dict:
    """Return a dict of coverage dimension values for a test command."""
    flags, pos = parse_flags(cmd)
    f: dict = {}

    if prim == "matmul":
        f["dt"] = flags.get("dt", "f32:f32:f32")
        f["stag"] = flags.get("stag", "ab")
        f["wtag"] = flags.get("wtag", "ab")
        f["dtag"] = flags.get("dtag", "ab")
        f["bia_dt"] = flags.get("bia-dt", "none")
        f["postops_cat"] = postops_cat(flags.get("attr-post-ops"))
        f["scales_cat"] = scales_cat(flags.get("attr-scales"))
        f["zp_cat"] = zp_cat(flags.get("attr-zero-points"))
        rdm = flags.get("runtime_dims_masks", "")
        f["rdm_cat"] = "none" if not rdm else (
            "full" if rdm == "15:15" else "partial"
        )
        f["shape_ndims"] = shape_ndims_matrix(pos)
        f["shape_size"] = shape_size_bucket(pos)
        f["batch_cat"] = batch_cat(pos)
        # Derived: quantization combo (important interaction)
        dt = f["dt"]
        f["quant_profile"] = (
            f"quant+scales" if is_quantized(dt) and flags.get("attr-scales")
            else "quant_noscales" if is_quantized(dt)
            else "float"
        )

    elif prim == "conv":
        f["dt"] = flags.get("dt", "f32:f32:f32")
        f["dir"] = flags.get("dir", "FWD_B")
        f["stag"] = flags.get("stag", "any")
        f["dtag"] = flags.get("dtag", "any")
        f["bia_dt"] = flags.get("bia-dt", "none")
        f["postops_cat"] = postops_cat(flags.get("attr-post-ops"))
        f["scales_cat"] = scales_cat(flags.get("attr-scales"))
        f["zp_cat"] = zp_cat(flags.get("attr-zero-points"))
        shape_str = pos[0] if pos else ""
        f["shape_spatial"] = spatial_dims_cat(shape_str)
        f["kernel_cat"] = kernel_cat(shape_str)
        f["has_groups"] = has_groups_cat(flags, shape_str)
        f["shape_size"] = shape_size_bucket(pos, is_spatial=True)
        dt = f["dt"]
        f["quant_profile"] = (
            "quant+scales" if is_quantized(dt) and flags.get("attr-scales")
            else "quant_noscales" if is_quantized(dt)
            else "float"
        )

    elif prim == "binary":
        f["sdt"] = flags.get("sdt", "f32:f32")
        f["ddt"] = flags.get("ddt", "f32")
        f["stag_cat"] = stag_cat(flags.get("stag"))
        f["alg"] = flags.get("alg", "add")
        f["postops_cat"] = postops_cat(flags.get("attr-post-ops"))
        f["has_scales"] = "yes" if flags.get("attr-scales") else "no"
        f["inplace"] = str(flags.get("inplace", "false"))
        f["shape_ndims"] = shape_ndims_matrix(pos)
        f["shape_size"] = shape_size_bucket(pos)
        f["has_broadcast"] = has_broadcast(pos)

    elif prim == "lnorm":
        f["dt"] = flags.get("dt", "f32:f32")
        f["dir"] = flags.get("dir", "FWD_D")
        flags_val = flags.get("flags", "")
        f["flags_cat"] = str(flags_val) if flags_val else "none"
        f["stat_tag"] = flags.get("stat_tag", "default")
        f["tag_cat"] = tag_cat(flags.get("tag"))
        f["inplace"] = str(flags.get("inplace", "false"))
        f["shape_ndims"] = shape_ndims_matrix(pos)
        f["shape_size"] = shape_size_bucket(pos)

    elif prim == "pool":
        f["dt"] = flags.get("dt", "f32:f32")
        f["dir"] = flags.get("dir", "FWD_I")
        f["alg"] = flags.get("alg", "max")
        f["tag_cat"] = tag_cat(flags.get("tag"))
        f["postops_cat"] = postops_cat(flags.get("attr-post-ops"))
        shape_str = pos[0] if pos else ""
        f["shape_spatial"] = spatial_dims_cat(shape_str)
        f["shape_size"] = shape_size_bucket(pos, is_spatial=True)

    elif prim == "prelu":
        f["sdt"] = flags.get("sdt", "f32:f32")
        f["dir"] = flags.get("dir", "FWD_I")
        f["stag_cat"] = stag_cat(flags.get("stag"))
        f["shape_ndims"] = shape_ndims_matrix(pos)
        f["shape_size"] = shape_size_bucket(pos)

    elif prim == "resampling":
        f["sdt"] = flags.get("sdt", "f32")
        f["ddt"] = flags.get("ddt", "f32")
        f["alg"] = flags.get("alg", "nearest")
        f["tag_cat"] = tag_cat(flags.get("tag"))
        f["postops_cat"] = postops_cat(flags.get("attr-post-ops"))
        f["dir"] = flags.get("dir", "FWD_D")
        shape_str = pos[0] if pos else ""
        f["shape_spatial"] = spatial_dims_cat(shape_str)
        f["shape_size"] = shape_size_bucket(pos, is_spatial=True)

    elif prim == "softmax":
        f["sdt"] = flags.get("sdt", "f32")
        f["ddt"] = flags.get("ddt", flags.get("sdt", "f32"))
        f["alg"] = flags.get("alg", "SOFTMAX")
        axis = flags.get("axis", "")
        f["axis_cat"] = "0" if axis in ("", "0") else "last" if axis == "-1" else "mid"
        f["dir"] = flags.get("dir", "FWD_I")
        f["stag_cat"] = tag_cat(flags.get("stag"))
        f["postops_cat"] = postops_cat(flags.get("attr-post-ops"))
        f["inplace"] = str(flags.get("inplace", "false"))
        f["shape_ndims"] = shape_ndims_matrix(pos)
        f["shape_size"] = shape_size_bucket(pos)

    # Shape key for diversity tracking
    f["_shape_key"] = pos[0].split("_n")[0] if pos else ""
    return f


# ─── Greedy pairwise coverage selection ───────────────────────────────────────

def greedy_pairwise_select(
    tests: list[str],
    features: list[dict],
    dims: list[str],
    max_count: int,
    seed: int = 42,
) -> list[int]:
    """
    Select up to max_count tests using greedy combinatorial (1-way + 2-way) coverage.

    Builds a combined requirement set:
      • 1-way: every unique (dim, value) pair must appear at least once
      • 2-way (pairwise): every observed (dim_i=val_i, dim_j=val_j) combination

    Uses a lazy-update max-heap (O((n·C(d,2)) · log n)).  Stored scores are
    upper bounds for actual scores (scores only decrease as requirements are
    fulfilled), so stale-high entries are re-inserted with their true score and
    stale-zero entries are discarded.  The loop halts when:
      (a) the top stored score hits 0  → nothing remains to gain, or
      (b) the heap is exhausted.

    After coverage saturation, fills remaining budget with novel shapes.

    Returns a list of selected test indices (length ≤ max_count).
    """
    n = len(features)
    rng = random.Random(seed)

    if n <= max_count:
        return list(range(n))

    dim_pairs = list(combinations(dims, 2))

    # ── Build unified requirement + per-test coverage sets ──────────────────
    # Tuple encoding:
    #   1-way: ("1", dim, val, "", "")
    #   2-way: ("2", d1, v1, d2, v2)
    all_required: set = set()
    test_coverage: list[frozenset] = []

    for f in features:
        tc: set = set()
        for d in dims:
            r = ("1", d, f.get(d, ""), "", "")
            all_required.add(r)
            tc.add(r)
        for d1, d2 in dim_pairs:
            r = ("2", d1, f.get(d1, ""), d2, f.get(d2, ""))
            all_required.add(r)
            tc.add(r)
        test_coverage.append(frozenset(tc))

    uncovered: set = set(all_required)

    # ── Initial scores ───────────────────────────────────────────────────────
    scores: list[int] = [len(tc & uncovered) for tc in test_coverage]
    heap = [(-scores[i], i) for i in range(n)]
    heapq.heapify(heap)

    selected: list[int] = []
    selected_set: set[int] = set()

    # ── Greedy selection with lazy heap updates ──────────────────────────────
    while len(selected) < max_count and uncovered:
        best_i: int | None = None

        while heap:
            neg_stored, i = heapq.heappop(heap)
            stored = -neg_stored

            # Stored score is an upper bound; if the top is 0, everything is 0
            if stored == 0:
                break

            if i in selected_set:
                continue

            actual = len(test_coverage[i] & uncovered)

            if actual == 0:
                # Stale-zero: useless now — discard without re-pushing
                continue

            if actual == stored:
                # Fresh and positive: this IS the global maximum
                best_i = i
                break

            # Stale but positive: correct and re-insert
            heapq.heappush(heap, (-actual, i))

        if best_i is None:
            break  # Nothing more to cover

        selected.append(best_i)
        selected_set.add(best_i)
        uncovered -= test_coverage[best_i]

    # ── Fill remaining budget with shape diversity ───────────────────────────
    if len(selected) < max_count:
        seen_shapes: set = {features[i]["_shape_key"] for i in selected}
        remaining_novel: list = []
        remaining_known: list = []
        for i in range(n):
            if i in selected_set:
                continue
            sk = features[i]["_shape_key"]
            if sk not in seen_shapes:
                remaining_novel.append(i)
                seen_shapes.add(sk)
            else:
                remaining_known.append(i)
        rng.shuffle(remaining_novel)
        rng.shuffle(remaining_known)
        fill = (remaining_novel + remaining_known)[: max_count - len(selected)]
        selected.extend(fill)

    return selected


# ─── Coverage metrics ─────────────────────────────────────────────────────────

def compute_coverage(
    all_feats: list[dict],
    sel_feats: list[dict],
    dims: list[str],
    prim: str,
) -> dict:
    """Compute multi-metric coverage statistics."""
    def uniq(feats, dim):
        return set(f.get(dim, "") for f in feats)

    metrics: dict = {}

    # Per-dimension value recall
    for dim in dims:
        orig_vals = uniq(all_feats, dim)
        sel_vals = uniq(sel_feats, dim)
        metrics[f"dim_{dim}"] = (len(sel_vals), len(orig_vals))

    # Pairwise pair recall
    dim_pairs = list(combinations(dims, 2))
    all_pairs: set = set()
    for f in all_feats:
        for d1, d2 in dim_pairs:
            all_pairs.add((d1, f.get(d1, ""), d2, f.get(d2, "")))
    sel_pairs: set = set()
    for f in sel_feats:
        for d1, d2 in dim_pairs:
            sel_pairs.add((d1, f.get(d1, ""), d2, f.get(d2, "")))
    metrics["pairwise"] = (len(sel_pairs), len(all_pairs))

    # Shape class recall
    orig_shapes = set(f["_shape_key"] for f in all_feats)
    sel_shapes = set(f["_shape_key"] for f in sel_feats)
    metrics["shapes"] = (len(sel_shapes), len(orig_shapes))

    # Datatype recall (primitive-specific key)
    dt_key = "dt" if "dt" in dims else ("sdt" if "sdt" in dims else None)
    if dt_key:
        orig_dt = uniq(all_feats, dt_key)
        sel_dt = uniq(sel_feats, dt_key)
        metrics["datatypes"] = (len(sel_dt), len(orig_dt))

    # Fingerprint recall (coarse): unique (dt, postops_cat, shape_ndims) triplets
    def fingerprint(f):
        return (
            f.get("dt", f.get("sdt", "")),
            f.get("postops_cat", ""),
            f.get("shape_ndims", f.get("shape_spatial", "")),
        )
    orig_fps = set(fingerprint(f) for f in all_feats)
    sel_fps = set(fingerprint(f) for f in sel_feats)
    metrics["fingerprints"] = (len(sel_fps), len(orig_fps))

    return metrics


# ─── Batch file writing ───────────────────────────────────────────────────────

def write_batch_file(path: str, selected_cmds: list[str],
                     aug_cmds: list[str] | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    aug_set = set(aug_cmds) if aug_cmds else set()
    with open(path, "w") as f:
        for cmd in selected_cmds:
            if cmd in aug_set:
                f.write("# targeted augmentation (see TARGETED_AUGMENTATIONS)\n")
            f.write("--reset " + clean_cmd(cmd) + "\n")


# ─── Reporting ────────────────────────────────────────────────────────────────

def pct(a, b) -> str:
    if b == 0:
        return "N/A"
    return f"{100 * a / b:.1f}%"


def print_coverage_table(results: dict):
    """Print a summary coverage table for all primitives."""
    print("\n" + "=" * 115)
    print(f"{'Primitive':<12} {'Orig':>6} {'Gen':>6} "
          f"{'DTypes':>12} {'Shapes':>14} {'PostOps':>16} "
          f"{'Fingerprints':>14} {'Pairs%':>10}")
    print("-" * 115)
    for prim, r in results.items():
        cov = r["coverage"]
        dt_key = "datatypes"
        dt_str = (f"{cov[dt_key][0]}/{cov[dt_key][1]}"
                  f"={pct(*cov[dt_key])}"
                  if dt_key in cov else "N/A")
        po_key = "dim_postops_cat"
        if po_key in cov:
            po = cov[po_key]
            po_str = f"{po[0]}/{po[1]}={pct(*po)}"
        else:
            po_str = "N/A"
        sh = cov["shapes"]
        fp = cov["fingerprints"]
        pw = cov["pairwise"]
        print(
            f"{prim:<12} {r['orig_count']:>6} {r['gen_count']:>6} "
            f"{dt_str:>12} "
            f"{sh[0]}/{sh[1]}={pct(*sh):>7} "
            f"{po_str:>14} "
            f"{fp[0]}/{fp[1]}={pct(*fp):>7} "
            f"{pw[0]}/{pw[1]}={pct(*pw):>7}"
        )
    print("=" * 115)


def print_verbose_coverage(prim: str, metrics: dict, dims: list[str]):
    """Print per-dimension coverage for a primitive."""
    print(f"\n  {prim} — per-dimension coverage:")
    for dim in dims:
        key = f"dim_{dim}"
        if key in metrics:
            s, o = metrics[key]
            flag = " ✓" if s == o else f" ✗ missing {o - s}"
            print(f"    {dim:<20}: {s}/{o} = {pct(s, o)}{flag}")
    if "pairwise" in metrics:
        s, o = metrics["pairwise"]
        print(f"    {'pairwise pairs':<20}: {s}/{o} = {pct(s, o)}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate coverage-optimized oneDNN benchdnn testsets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--log-dir",
        default="./logs",
        help="Directory containing *_nightly_log.txt files",
    )
    parser.add_argument(
        "--output-dir",
        default="./generated_testsets",
        help="Directory for generated batch files",
    )
    parser.add_argument(
        "--max-tests-large",
        type=int,
        default=2000,
        help="Max tests for matmul and conv [default: 2000]",
    )
    parser.add_argument(
        "--max-tests-small",
        type=int,
        default=1000,
        help="Max tests for other primitives [default: 1000]",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility [default: 42]",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-dimension coverage breakdown",
    )
    args = parser.parse_args()

    log_dir = args.log_dir
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Discover plain-text log files
    log_files = {
        fn.replace("_nightly_log.txt", ""): os.path.join(log_dir, fn)
        for fn in sorted(os.listdir(log_dir))
        if fn.endswith("_nightly_log.txt")
    }

    if not log_files:
        print(f"ERROR: No *_nightly_log.txt files found in {log_dir}", file=sys.stderr)
        print("  Provide plain-text log files named <primitive>_nightly_log.txt,",
              file=sys.stderr)
        print("  each containing one benchdnn command per line (PASSED tests).",
              file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(log_files)} primitives: {', '.join(log_files)}")
    print(f"Output directory: {out_dir}\n")

    results: dict = {}

    for prim, log_path in sorted(log_files.items()):
        if prim not in COVERAGE_DIMS:
            print(f"  Skipping {prim}: no coverage dimensions defined")
            continue

        max_count = (
            args.max_tests_large if prim in LARGE_PRIMITIVES
            else args.max_tests_small
        )
        dims = COVERAGE_DIMS[prim]

        print(f"[{prim}]  Loading ... ", end="", flush=True)
        cmds = load_log(log_path)
        print(f"{len(cmds)} unique tests")

        # Collect forced augmentation commands (synthetic tests for known gaps).
        aug_cmds = get_augmentations(cmds, prim)
        if aug_cmds:
            print(f"         Injecting {len(aug_cmds)} targeted augmentation(s)")

        print(f"         Extracting features ... ", end="", flush=True)
        features = [extract_features(c, prim) for c in cmds]
        print("done")

        # Reserve budget for forced augmentations.
        effective_max = max(0, max_count - len(aug_cmds))
        print(f"         Selecting up to {effective_max} tests (+ {len(aug_cmds)} forced) ... ",
              end="", flush=True)
        selected_idx = greedy_pairwise_select(
            cmds, features, dims, effective_max, seed=args.seed
        )
        print(f"selected {len(selected_idx)}")

        sel_cmds = [cmds[i] for i in selected_idx] + aug_cmds
        sel_feats = ([features[i] for i in selected_idx]
                     + [extract_features(c, prim) for c in aug_cmds])

        coverage = compute_coverage(features, sel_feats, dims, prim)

        # Adaptive: if pairwise coverage already saturated at a smaller count,
        # report the actual saturation point
        pw_s, pw_o = coverage["pairwise"]
        if pw_s == pw_o:
            print(f"         ✓ 100% pairwise coverage achieved")

        batch_path = os.path.join(out_dir, f"{prim}_generated.txt")
        write_batch_file(batch_path, sel_cmds, aug_cmds)
        print(f"         Wrote: {batch_path}")

        results[prim] = {
            "orig_count": len(cmds),
            "gen_count": len(selected_idx),
            "coverage": coverage,
        }

        if args.verbose:
            print_verbose_coverage(prim, coverage, dims)

        print()

    print_coverage_table(results)

    # Print summary
    print("\nGenerated batch files:")
    for prim in sorted(results):
        r = results[prim]
        reduction = 100 * (1 - r["gen_count"] / r["orig_count"])
        pw = r["coverage"]["pairwise"]
        print(
            f"  {prim:<12}: {r['orig_count']:>6} → {r['gen_count']:>6} tests "
            f"({reduction:.0f}% reduction), "
            f"pairwise coverage {pct(*pw)}"
        )


if __name__ == "__main__":
    main()
