#!/usr/bin/python3

# *******************************************************************************
# Copyright 2025 Arm Limited and affiliates.
# SPDX-License-Identifier: Apache-2.0
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

"""
Compare two benchdnn runs.

Usage:
    python benchdnn_comparison.py baseline.txt new.txt --out-file out.md
"""

import argparse
from collections import defaultdict
import git
import json
import math
import os
import pathlib
from scipy.stats import ttest_ind
import statistics
import warnings


F_PATH = pathlib.Path(__file__).parent.resolve()
CI_JSON_PATH = F_PATH / "../aarch64/ci.json"


def print_to_github_out(message):
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            print(message.replace("\n", "%0A"), file=f)


def compare_two_benchdnn(file1, file2, out_file=None):
    """
    Compare two benchdnn output files
    """
    with open(file1) as f:
        r1 = f.readlines()

    with open(file2) as f:
        r2 = f.readlines()

    # Trim non-formatted lines and split the problem from time
    r1 = [x.split(",") for x in r1 if x[0:8] == "--mode=P"]
    r2 = [x.split(",") for x in r2 if x[0:8] == "--mode=P"]

    if (len(r1) == 0) or (len(r2) == 0):
        raise Exception("One or both of the test results have zero lines")
    if len(r1) != len(r2):
        raise Exception("The number of benchdnn runs do not match")

    r1_exec = defaultdict(list)
    r1_ctime = defaultdict(list)
    r2_exec = defaultdict(list)
    r2_ctime = defaultdict(list)

    for key, exec_time, ctime in r1:
        # Older versions of benchdnn outputs underscores
        # instead of hyphens for some ops leading to
        # mismatches in problems with newer versions
        key = key.replace("_", "-")
        r1_exec[key].append(float(exec_time))
        r1_ctime[key].append(float(ctime))

    for key, exec_time, ctime in r2:
        key = key.replace("_", "-")
        r2_exec[key].append(float(exec_time))
        r2_ctime[key].append(float(ctime))

    exec_failures, ctime_failures = [], []
    if out_file is not None:
        with open(CI_JSON_PATH) as f:
            ci_json = json.load(f)

        repo = git.Repo(F_PATH / "../../..", search_parent_directories=True)
        head_sha = repo.git.rev_parse(repo.head.object.hexsha, short=6)
        headers = f"| problem | oneDNN ({ci_json['dependencies']['onednn-base']}) time(ms) | oneDNN ({head_sha}) time(ms) | speedup (>1 is faster) |\n"
        with open(out_file, "w") as f:
            f.write(headers + "| :---: | :---: | :---: | :---:|\n")

    for prb in r1_exec:
        if prb not in r2_exec:
            raise Exception(f"{prb} exists in {file1} but not {file2}")

        exec1 = r1_exec[prb]
        exec2 = r2_exec[prb]
        ctime1 = r1_ctime[prb]
        ctime2 = r2_ctime[prb]
        exec_regressed_ttest = ttest_ind(exec2, exec1, alternative="greater")
        exec_improved_ttest = ttest_ind(exec2, exec1, alternative="less")
        ctime_ttest = ttest_ind(ctime2, ctime1, alternative="greater")
        r1_med_exec = statistics.median(exec1)
        r2_med_exec = statistics.median(exec2)
        r1_mean_exec = statistics.mean(exec1)
        r2_mean_exec = statistics.mean(exec2)
        r1_med_ctime = statistics.median(ctime1)
        r2_med_ctime = statistics.median(ctime2)
        r1_mean_ctime = statistics.mean(ctime1)
        r2_mean_ctime = statistics.mean(ctime2)

        use_ttest = len(exec1) >= 3 and len(exec2) >= 3

        if 0 in [
            r1_med_exec,
            min(exec1),
            min(exec2),
            r1_med_ctime,
            min(ctime1),
            r1_mean_exec,
            r2_mean_exec,
            r1_mean_ctime,
            r2_mean_ctime,
        ]:
            warnings.warn(
                f"Avoiding division by 0 for {prb}. "
                f"Exec median: {r1_med_exec}, min: {min(exec1)}; "
                f"Ctime median: {r1_med_ctime}, min: {min(ctime1)}"
            )
            continue

        r1_sem_exec = statistics.stdev(exec1) / math.sqrt(r1_mean_exec)
        r2_sem_exec = statistics.stdev(exec2) / math.sqrt(r2_mean_exec)
        r1_sem_ctime = statistics.stdev(ctime1) / math.sqrt(r1_mean_ctime)
        r2_sem_ctime = statistics.stdev(ctime2) / math.sqrt(r2_mean_ctime)

        # A test fails if execution time:
        # - shows a statistically significant regression and
        # - shows ≥ 10% slowdown in either median or min times
        exec_regressed = (
            not use_ttest or exec_regressed_ttest.pvalue <= 0.05
        ) and (
            (r2_med_exec - r1_med_exec) / r1_med_exec >= 0.1
            or (min(exec2) - min(exec1)) / min(exec1) >= 0.1
        )
        exec_improved = (
            not use_ttest or exec_improved_ttest.pvalue <= 0.05
        ) and (
            r1_med_exec / r2_med_exec >= 1.1 or min(exec1) / min(exec2) >= 1.1
        )
        ctime_regressed = ctime_ttest.pvalue <= 0.05 and (
            (r2_med_ctime - r1_med_ctime) / r1_med_ctime >= 0.1
            or (min(ctime2) - min(ctime1)) / min(ctime1) >= 0.1
        )

        if exec_regressed:
            exec_failures.append(
                f"{prb} exec: "
                f"{r1_mean_exec:.3g}\u00b1{r1_sem_exec:.3g} "
                "\u2192 "
                f"{r2_mean_exec:.3g}\u00b1{r2_sem_exec:.3g} "
                f"(p={exec_regressed_ttest.pvalue:.3g})"
            )

        if ctime_regressed:
            ctime_failures.append(
                f"{prb} ctime: "
                f"{r1_mean_ctime:.3g}\u00b1{r1_sem_ctime} "
                "\u2192 "
                f"{r2_mean_ctime:.3g}\u00b1{r2_sem_ctime} "
                f"(p={ctime_ttest.pvalue:.3g})"
            )

        if out_file is not None and (exec_regressed or exec_improved):
            prb_params = [x.replace("--", "") for x in prb.split(" ")]
            prb_params = [prb_params[2]] + [
                x for x in prb_params if ("dt=" in x) or ("alg=" in x)
            ]  # filter out the problem and data types
            prb_str = (
                "<details>"
                + f"<summary>{' '.join(prb_params)}</summary>"
                + prb
                + "</details>"
            )
            colour = "green" if exec_improved else "red"
            speedup_str = (
                "$${\\color{"
                + colour
                + "}"
                + f"{(r1_med_exec)/r2_med_exec:.3g}\\times"
                + "}$$"
            )
            with open(out_file, "a") as f:
                f.write(
                    f"|{prb_str}|{r1_med_exec:.3g}|{r2_med_exec:.3g}|{speedup_str}|\n"
                )

    print_to_github_out(f"pass={not exec_failures}")

    message = ""
    if ctime_failures:
        message += (
            "\n----The following ctime regression tests failed:----\n"
            + "\n".join(ctime_failures)
            + "\n"
        )

    if not exec_failures:
        print_to_github_out(f"message={message}")
        print(message)
        print("Execution Time regression tests passed")
    else:
        message += (
            "\n----The following exec time regression tests failed:----\n"
            + "\n".join(exec_failures)
            + "\n"
        )
        print_to_github_out(f"message={message}")
        print(message)
        raise Exception("Some regression tests failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare two benchdnn result files"
    )
    parser.add_argument("file1", help="Path to baseline result file")
    parser.add_argument("file2", help="Path to new result file")
    parser.add_argument(
        "--out-file", help="md file to output performance results to"
    )
    args = parser.parse_args()

    compare_two_benchdnn(args.file1, args.file2, args.out_file)
