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

import sys
import os
from collections import defaultdict
from scipy.stats import ttest_ind
import warnings
import statistics


def compare_two_benchdnn(file1, file2, tolerance=0.05):
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
        warnings.warn("One or both of the test results have zero lines")
    if len(r1) != len(r2):
        warnings.warn("The number of benchdnn runs do not match")

    r1_exec = defaultdict(list)
    r1_ctime = defaultdict(list)
    r2_exec = defaultdict(list)
    r2_ctime = defaultdict(list)

    for key, exec_time, ctime in r1:
        r1_exec[key].append(float(exec_time))
        r1_ctime[key].append(float(ctime))

    for key, exec_time, ctime in r2:
        r2_exec[key].append(float(exec_time))
        r2_ctime[key].append(float(ctime))

    failed_tests = []
    for prb in r1_exec:
        if prb not in r2_exec:
            warnings.warn(f"{prb} exists in {file1} but not {file2}")
            continue
        exec1 = r1_exec[prb]
        exec2 = r2_exec[prb]
        ctime1 = r1_ctime[prb]
        ctime2 = r2_ctime[prb]
        res = ttest_ind(exec2, exec1, alternative="greater")
        ctime_test = ttest_ind(ctime2, ctime1, alternative="greater")
        r1_med_exec = statistics.median(exec1)
        r2_med_exec = statistics.median(exec2)
        r1_med_ctime = statistics.median(ctime1)
        r2_med_ctime = statistics.median(ctime2)

        if 0 in [r1_med_exec, min(exec1), r1_med_ctime, min(ctime1)]:
            warnings.warn(
                f"Avoiding division by 0 for {prb}. "
                f"Exec median: {r1_med_exec}, min: {min(exec1)}; "
                f"Ctime median: {r1_med_ctime}, min: {min(ctime1)}"
            )
            continue

        # A test fails if either execution time or creation time:
        # - shows a statistically significant regression and
        # - shows ≥ 10% slowdown in both median or min times
        exec_regressed = res.pvalue <= 0.05 and (
            (r2_med_exec - r1_med_exec) / r1_med_exec >= 0.1
            or (min(exec2) - min(exec1)) / min(exec1) >= 0.1
        )
        ctime_regressed = ctime_test.pvalue <= 0.05 and (
            (r2_med_ctime - r1_med_ctime) / r1_med_ctime >= 0.1
            or (min(ctime2) - min(ctime1)) / min(ctime1) >= 0.1
        )

        if exec_regressed or ctime_regressed:
            failed_tests.append(
                f"{prb} exec: {r1_med_exec:.3g} → {r2_med_exec:.3g} "
                f"(p={res.pvalue:.3g}), "
                f"ctime: {r1_med_ctime:.3g} → {r2_med_ctime:.3g}"
                f"(p={ctime_test.pvalue:.3g})"
            )

    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            print(f"pass={not failed_tests}", file=f)

    if not failed_tests:
        print("Regression tests passed")
    else:
        message = (
            "\n----The following regression tests failed:----\n"
            + "\n".join(failed_tests)
            + "\n"
        )
        if "GITHUB_OUTPUT" in os.environ:
            out_message = message.replace("\n", "%0A")
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                print(f"message={out_message}", file=f)
        print(message)
        raise Exception("Some regression tests failed")


if __name__ == "__main__":
    compare_two_benchdnn(sys.argv[1], sys.argv[2])
