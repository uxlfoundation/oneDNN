#!/usr/bin/env python3
"""Write a benchdnn batch file for the --sdpa problems of a problems file.

    python problems_to_batch.py problems_kpi.txt > kpi.batch
    SDPA_CONFIG_SELECT=table benchdnn --sdpa --engine=gpu --mode=P --batch=kpi.batch > table.log
    SDPA_CONFIG_SELECT=legacy benchdnn --sdpa --engine=gpu --mode=P --batch=kpi.batch > legacy.log
    python compare_perf.py table.log legacy.log

graph= lines are skipped; run those through the --graph driver separately.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tune_fwd_config import benchdnn_args, parse_spec  # noqa: E402


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    n = 0
    for path in sys.argv[1:]:
        with open(path) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                spec = parse_spec(line)
                if spec["graph"]:
                    continue
                # drop the driver, engine and mode: the batch inherits them
                args = benchdnn_args(spec, "P")[3:]
                print("--reset " + " ".join(args))
                n += 1
    print("%d problems" % n, file=sys.stderr)


if __name__ == "__main__":
    main()
