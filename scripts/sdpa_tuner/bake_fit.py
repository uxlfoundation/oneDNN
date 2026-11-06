#!/usr/bin/env python3
# *******************************************************************************
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
# *******************************************************************************
"""Write a fit into select.cpp and table lines into select_table.db.

Part of scripts/sdpa_tuner; README.md there shows where it fits in the
workflow. Either option may be used alone, e.g. --table only when tuning a
workload without refitting:

    python scripts/sdpa_tuner/bake_fit.py --arch xe_hpg --device A750 \\
        --fit fit_a750.txt --table table_a750.txt --table fixes_a750.txt
    clang-format -i src/gpu/intel/sdpa/select.cpp

--fit     output of fit_fwd_model.py --fit: its "// select.cpp seed_coefs()"
          block replaces the assignments in the --arch case of
          seed_coefs(), and its problem and config counts, cross-validated
          regret and in-sample regret go into the comment above them
--arch    the seed_coefs() case to write, e.g. xe_hpg; needed with --fit
--device  device name for that comment (default: --arch)
--unfix   a coefficient the fit released from its default-fixed seed, as
          passed to fit_fwd_model.py (repeatable); default-fixed
          coefficients are not written otherwise
--table   "key config" lines from tune_fwd_config.py --table-out or
          resolve_collisions.py --out, merged into select_table.db as
          fwd_table_entry_t initializers; repeatable, later files and
          lines win on equal keys, existing entries included
--select  path of select.cpp (default: the one in this repository)
--db      path of select_table.db (default: the one in this repository)

Run clang-format on select.cpp after --fit; select_table.db is written in
its final form.
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_fwd_model import DEFAULT_FIXED  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SELECT = os.path.normpath(
    os.path.join(HERE, "..", "..", "src", "gpu", "intel", "sdpa", "select.cpp")
)
DEFAULT_DB = os.path.normpath(
    os.path.join(
        HERE, "..", "..", "src", "gpu", "intel", "sdpa", "select_table.db"
    )
)


def fmt(v):
    v = float(v)
    if v == 0:
        return "0.0"
    s = "%.4g" % v
    if "e" in s:
        s = "%.6g" % v
    if "." not in s and "e" not in s:
        s += ".0" if abs(v) < 1e15 else ""
    return s


def bake_coefs(src, arch, device, fit_text, unfix=()):
    fixed = set(DEFAULT_FIXED) - set(unfix)
    block = fit_text.split("// select.cpp seed_coefs()", 1)
    if len(block) < 2:
        sys.exit("no '// select.cpp seed_coefs()' block in the fit output")
    coefs = dict(re.findall(r"^\s*m\.(\w+) = ([-+0-9.e]+);", block[1], re.M))
    rows = re.search(r"^(\d+) rows, (\d+) problems", fit_text, re.M)
    cv = re.search(
        r"^cross-validated over \d+ problems: model regret geomean "
        r"x([\d.]+).*legacy geomean x([\d.]+)",
        fit_text,
        re.M,
    )
    fit = re.search(
        r"^fitted/train: \d+ problems, regret geomean x([\d.]+)"
        r".*legacy geomean x([\d.]+)",
        fit_text,
        re.M,
    )
    if not (rows and fit):
        sys.exit("fit output lacks the row count or the fitted/train line")

    case = "        case compute::gpu_arch_t::%s:" % arch
    start = src.find(case)
    if start < 0:
        sys.exit("no case for %s in seed_coefs()" % arch)
    end = src.index("            break;", start)
    body = src[start + len(case) + 1 : end]

    if cv:
        comment = (
            "            // Fitted on %s: %s problems, %s configs, 4-fold CV "
            "regret\n            // %s (legacy table %s), in-sample %s\n"
            % (
                device,
                rows.group(2),
                rows.group(1),
                cv.group(1),
                cv.group(2),
                fit.group(1),
            )
        )
    else:
        comment = (
            "            // Fitted on %s: %s problems, %s configs, in-sample "
            "regret\n            // %s (legacy table %s)\n"
            % (device, rows.group(2), rows.group(1), fit.group(1), fit.group(2))
        )
    body = re.sub(
        r"            // Fitted on [^\n]*\n            // [^\n]*\n", "", body
    )
    present = set()

    def repl(m):
        name = m.group(1)
        present.add(name)
        if name in coefs and name not in fixed:
            return "            m.%s = %s;" % (name, fmt(coefs[name]))
        return m.group(0)

    body = re.sub(
        r"^            m\.(\w+) = [^;]+;[^\n]*", repl, body, flags=re.M
    )
    extra = [n for n in coefs if n not in present and n not in fixed]
    for n in extra:
        body += "            m.%s = %s;\n" % (n, fmt(coefs[n]))
    body = comment + body.lstrip("\n")
    return (
        src[: start + len(case) + 1] + body + src[end:],
        len(coefs),
        len(extra),
    )


# The lookup-key grammar of fwd_key_t::str() in select.cpp, extended with
# the optional device tag of a table line
KEY_RE = re.compile(
    r"^(?P<arch>\w+)"
    r":d(?P<d_qk>\d+)x(?P<d_v>\d+)"
    r":k(?P<keys>\d+):q(?P<queries>\d+):bh(?P<bh>\d+)"
    r":m(?P<mask>\d+)"
    r":t(?P<qb>\d+)/(?P<kb>\d+)/(?P<vb>\d+)/(?P<db>\d+)"
    r":qz(?P<kq>[01])(?P<vq>[01])"
    r":g(?P<kg>\d+)/(?P<vg>\d+)"
    r":(?P<fma>sys|fma)(?P<f32>16|32)"
    r":acc(?P<kacc>16|32)/(?P<vacc>16|32)"
    r":al(?P<qa>\d+)/(?P<ka>\d+)/(?P<va>\d+)/(?P<da>\d+)"
    r":i(?P<integ>[01]):tr(?P<tr>[01]):tk(?P<tk>[01])"
    r"(?::eu(?P<eu>\d+))?$"
)

# One fwd_table_entry_t initializer in select_table.db: the fwd_key_t
# members in declaration order with named types for the discrete-valued
# fields, the device EU tag (any_eu: every device of the architecture),
# and the fwd_config_t
ENTRY_RE = re.compile(
    r"^\{\{(?P<arch>\w+), /\*d\*/ (?P<d_qk>\d+), (?P<d_v>\d+), "
    r"/\*k\*/ (?P<keys>\d+), /\*q\*/ (?P<queries>\d+), /\*bh\*/ (?P<bh>\d+), "
    r"mask_t::(?P<mask>\w+), "
    r"bits_t \{(?P<qb>\d+), (?P<kb>\d+), (?P<vb>\d+), (?P<db>\d+)\}, "
    r"quant_t::(?P<quant>\w+), groups_t \{(?P<kg>\d+), (?P<vg>\d+)\}, "
    r"kernel_t::(?P<kernel>sys16|sys32|fma16|fma32), "
    r"acc_t::(?P<acc>f32_f32|f16_f32|f32_f16|f16_f16), "
    r"align_t \{(?P<qa>\d+), (?P<ka>\d+), (?P<va>\d+), (?P<da>\d+)\}, "
    r"(?P<integ>discrete|integrated), (?P<tr>inference|training), "
    r"(?P<tk>plain_k|transposed_k)\}, (?P<eu>any_eu|\d+), "
    r"\{(?P<cfg>\d+(?:, \d+){7})\}\},$"
)

MASK_NAMES = ["none", "causal_top_left", "causal_bottom_right", "buffer"]
QUANT_NAMES = {"00": "none", "10": "k", "01": "v", "11": "kv"}
ACC_NAMES = {
    ("32", "32"): "f32_f32",
    ("16", "32"): "f16_f32",
    ("32", "16"): "f32_f16",
    ("16", "16"): "f16_f16",
}


def key_to_entry(key, cfg):
    """A 'key config' table line as a fwd_table_entry_t initializer."""
    m = KEY_RE.match(key)
    if not m:
        sys.exit("bad table key: %r" % key)
    vals = cfg.split(",")
    if len(vals) != 8 or not all(v.isdigit() for v in vals):
        sys.exit("bad table config for %r: %r" % (key, cfg))
    g = m.groupdict()
    mask = int(g["mask"])
    if mask >= len(MASK_NAMES):
        sys.exit("bad mask in table key: %r" % key)
    fields = ", ".join(
        [
            g["arch"],
            "/*d*/ " + g["d_qk"],
            g["d_v"],
            "/*k*/ " + g["keys"],
            "/*q*/ " + g["queries"],
            "/*bh*/ " + g["bh"],
            "mask_t::" + MASK_NAMES[mask],
            "bits_t {%s, %s, %s, %s}" % (g["qb"], g["kb"], g["vb"], g["db"]),
            "quant_t::" + QUANT_NAMES[g["kq"] + g["vq"]],
            "groups_t {%s, %s}" % (g["kg"], g["vg"]),
            "kernel_t::%s%s" % (g["fma"], g["f32"]),
            "acc_t::" + ACC_NAMES[(g["kacc"], g["vacc"])],
            "align_t {%s, %s, %s, %s}" % (g["qa"], g["ka"], g["va"], g["da"]),
            "integrated" if g["integ"] == "1" else "discrete",
            "training" if g["tr"] == "1" else "inference",
            "transposed_k" if g["tk"] == "1" else "plain_k",
        ]
    )
    return "{{%s}, %s, {%s}}," % (
        fields,
        g["eu"] or "any_eu",
        cfg.replace(",", ", "),
    )


def entry_to_key(line):
    """A fwd_table_entry_t initializer back as a ('key', 'config') pair."""
    m = ENTRY_RE.match(line)
    if not m:
        sys.exit("bad select_table.db entry: %r" % line)
    g = m.groupdict()
    qz = next(k for k, v in QUANT_NAMES.items() if v == g["quant"])
    acc = next(k for k, v in ACC_NAMES.items() if v == g["acc"])
    key = (
        "%s:d%sx%s:k%s:q%s:bh%s:m%d:t%s/%s/%s/%s:qz%s:g%s/%s:%s"
        ":acc%s/%s:al%s/%s/%s/%s:i%d:tr%d:tk%d"
        % (
            g["arch"],
            g["d_qk"],
            g["d_v"],
            g["keys"],
            g["queries"],
            g["bh"],
            MASK_NAMES.index(g["mask"]),
            g["qb"],
            g["kb"],
            g["vb"],
            g["db"],
            qz,
            g["kg"],
            g["vg"],
            g["kernel"],
            acc[0],
            acc[1],
            g["qa"],
            g["ka"],
            g["va"],
            g["da"],
            g["integ"] == "integrated",
            g["tr"] == "training",
            g["tk"] == "transposed_k",
        )
    )
    if g["eu"] != "any_eu":
        key += ":eu" + g["eu"]
    return key, g["cfg"].replace(", ", ",")


def merge_table(db_text, files):
    marker = "/*@sdpa_fwd_table@*/"
    tstart = db_text.index(marker) + len(marker)
    table = {}
    old = 0
    for line in db_text[tstart:].splitlines():
        line = line.strip()
        if not line:
            continue
        k, c = entry_to_key(line)
        if key_to_entry(k, c) != line:
            sys.exit("select_table.db entry does not round-trip: %r" % line)
        table[k] = c
        old += 1
    added = 0
    for path in files:
        with open(path) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2 or len(parts[1].split(",")) != 8:
                    sys.exit("bad table line in %s: %r" % (path, line))
                if table.get(parts[0]) != parts[1]:
                    added += 1
                table[parts[0]] = parts[1]
    lines = "".join(key_to_entry(k, table[k]) + "\n" for k in sorted(table))
    db_text = db_text[:tstart] + "\n" + lines
    return db_text, old, len(table), added


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--select", default=DEFAULT_SELECT, help="select.cpp")
    ap.add_argument("--db", default=DEFAULT_DB, help="select_table.db")
    ap.add_argument("--arch", help="architecture case to bake, e.g. xe_hpc")
    ap.add_argument("--device", default="", help="device name for the comment")
    ap.add_argument("--fit", help="fit_fwd_model.py --fit output")
    ap.add_argument(
        "--unfix",
        action="append",
        default=[],
        help="default-fixed coefficient the fit released",
    )
    ap.add_argument(
        "--table",
        action="append",
        default=[],
        help="table file to merge (repeatable)",
    )
    args = ap.parse_args()
    if not args.fit and not args.table:
        ap.error("nothing to do: give --fit and/or --table")
    if args.fit and not args.arch:
        ap.error("--fit needs --arch")

    with open(args.select, encoding="utf-8") as f:
        src = f.read()
    if args.fit:
        with open(args.fit) as f:
            fit_text = f.read()
        src, n, extra = bake_coefs(
            src, args.arch, args.device or args.arch, fit_text, args.unfix
        )
        print(
            "baked %d coefficients into the %s case (%d added)"
            % (n, args.arch, extra)
        )
        with open(args.select, "w", encoding="utf-8", newline="\n") as f:
            f.write(src)
        print("wrote %s; run clang-format on it" % args.select)
    if args.table:
        with open(args.db, encoding="utf-8") as f:
            db_text = f.read()
        db_text, old, new, added = merge_table(db_text, args.table)
        print(
            "table: %d entries before, %d after, %d added or changed"
            % (old, new, added)
        )
        with open(args.db, "w", encoding="utf-8", newline="\n") as f:
            f.write(db_text)
        print("wrote %s" % args.db)


if __name__ == "__main__":
    main()
