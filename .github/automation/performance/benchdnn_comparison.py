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
import os
import pathlib
from scipy.stats import ttest_ind
import numpy as np
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
    r1 = parse_benchdnn_file(file1)
    r2 = parse_benchdnn_file(file2)

    validate_results(r1, r2)

    r1_exec, r1_ctime = build_timing_maps(r1)
    r2_exec, r2_ctime = build_timing_maps(r2)

    exec_inconclusive, ctime_inconclusive = [], []
    exec_failures, exec_improvements, ctime_failures = [], [], []
    if out_file is not None:
        ci_json, head_sha = load_ci_metadata()
        initialize_markdown_output(out_file, ci_json, head_sha)

    for prb in r1_exec:
        if prb not in r2_exec:
            raise Exception(f"{prb} exists in {file1} but not {file2}")

        exec1 = r1_exec[prb]
        exec2 = r2_exec[prb]
        ctime1 = r1_ctime[prb]
        ctime2 = r2_ctime[prb]
        if not (all(exec1) or all(exec2)):
            continue
        stats = compute_problem_stats(exec1, exec2, ctime1, ctime2, prb)
        if stats is None:
            continue

        is_uncertain_exec = stats["is_uncertain_exec"]
        is_uncertain_ctime = stats["is_uncertain_ctime"]
        exec_regressed = stats["exec_regressed"]
        exec_improved= stats["exec_improved"]
        ctime_regressed = stats["ctime_regressed"]
        
        if is_uncertain_exec:
            exec_inconclusive.append(
                f"{prb} exec: "
                f"{print_shared_exp(stats['mean_exec_1'], stats['error_exec_1'])} "
                "→ "
                f"{print_shared_exp(stats['mean_exec_2'], stats['error_exec_2'])} "
                f"(results too close together for their error bounds)"
            )

        if is_uncertain_ctime:
            ctime_inconclusive.append(
                f"{prb} ctime: "
                f"{print_shared_exp(stats['mean_ctime_1'], stats['error_ctime_1'])} "
                "→ "
                f"{print_shared_exp(stats['mean_ctime_2'], stats['error_ctime_2'])} "
                f"(results too close together for their error bounds)"
            )
            

        if exec_regressed:
            exec_failures.append(
                f"{prb} exec: "
                f"{print_shared_exp(stats['mean_exec_1'], stats['error_exec_1'])} "
                "→ "
                f"{print_shared_exp(stats['mean_exec_2'], stats['error_exec_2'])} "
                f"(p={stats['exec_regressed_ttest'].pvalue:.2E})"
            )
            
        if exec_improved:
            exec_improvements.append(
                f"{prb} exec: "
                f"{print_shared_exp(stats['mean_exec_1'], stats['error_exec_1'])} "
                "→ "
                f"{print_shared_exp(stats['mean_exec_2'], stats['error_exec_2'])} "
                f"(p={stats['exec_improved_ttest'].pvalue:.2E})"
            )

        if ctime_regressed:
            ctime_failures.append(
                f"{prb} ctime: "
                f"{print_shared_exp(stats['mean_ctime_1'], stats['error_ctime_1'])} "
                "→ "
                f"{print_shared_exp(stats['mean_ctime_2'], stats['error_ctime_2'])} "
                f"(p={stats['ctime_ttest'].pvalue:.2E})"
            )

        if out_file is not None and (is_uncertain_exec or exec_improved or exec_regressed):
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
            
            colour = pick_colour(is_uncertain_exec, exec_regressed, exec_improved)

            speedup_str = (
                "$${\\color{"
                + colour
                + "}"
                + f"{(stats['median_exec_1'])/stats['median_exec_2']:.3g}\\times"
                + "}$$"
            )
            digits_exec_1 = int(np.log10(stats['mean_exec_1'] / stats['error_exec_1']))
            digits_exec_2 = int(np.log10(stats['mean_exec_2'] / stats['error_exec_2']))
            with open(out_file, "a") as f:
                f.write(
                    f"|{prb_str}" \
                    f"|{stats['median_exec_1']:.{digits_exec_1}g}"
                    f"|{stats['p10_exec_1']:.{digits_exec_1}g}"
                    f"|{stats['median_exec_2']:.{digits_exec_2}g}"
                    f"|{stats['p10_exec_2']:.{digits_exec_2}g}"
                    f"|{speedup_str}|\n"
                )

    print_to_github_out(f"pass={not exec_failures}")

    message = ""
    if ctime_inconclusive:
        message += (
            "\n----The following ctime regression tests were inconclusive:----\n"
            + "\n".join(ctime_inconclusive)
            + "\n"
        )
    
    if ctime_failures:
        message += (
            "\n----The following ctime regression tests failed:----\n"
            + "\n".join(ctime_failures)
            + "\n"
        )
    
    if exec_inconclusive:
        message += (
            "\n----The following exec regression tests were inconclusive:----\n"
            + "\n".join(exec_inconclusive)
            + "\n"
        )
        
    if exec_improvements:
        message += (
            "\n----The following exec time tests improved:----\n"
            + "\n".join(exec_improvements)
            + "\n"
        )
    
    if exec_failures:
        message += (
            "\n----The following exec time regression tests failed:----\n"
            + "\n".join(exec_failures)
            + "\n"
        )

    print_to_github_out(f"message={message}")
    print(message)
    
    if not exec_failures:
        print("Execution Time regression tests passed")
    else:
        raise Exception("Some regression tests failed")
    
def pick_colour(is_uncertain_exec, exec_regressed, exec_improved):
    if is_uncertain_exec: return "orange"
    if exec_regressed: return "red"
    if exec_improved: return "green"
    return "black"

def print_shared_exp(val, err):
    if val == 0 and err == 0:
        return "0 ± 0"
    val_e = int(f"{val:.0e}".split("e")[1])
    err_e = int(f"{err:.0e}".split("e")[1])
    digits = max(val_e - err_e, 1)
    scale = 10 ** val_e
    v = val / scale
    de = err / scale
    return f"{v:.{digits}f}E{val_e:+03d} ± {de:.{digits}f}E{val_e:+03d}"

def parse_benchdnn_file(path):
    with open(path) as f:
        lines = f.readlines()

    # Trim non-formatted lines and split the problem from time
    return [line.split(",") for line in lines if line[0:8] == "--mode=P"]


def validate_results(r1, r2):
    if (len(r1) == 0) or (len(r2) == 0):
        raise Exception("One or both of the test results have zero lines")
    if len(r1) != len(r2):
        raise Exception("The number of benchdnn runs do not match")


def build_timing_maps(results):
    exec_times = defaultdict(list)
    ctime_times = defaultdict(list)

    for key, exec_time, ctime in results:
        # Older versions of benchdnn outputs underscores
        # instead of hyphens for some ops leading to
        # mismatches in problems with newer versions
        key = key.replace("_", "-")
        exec_times[key].append(float(exec_time))
        ctime_times[key].append(float(ctime))

    return exec_times, ctime_times


def compute_problem_stats(exec1, exec2, ctime1, ctime2, problem):
    assert(len(exec1) == len(exec2))
    n_samples = len(exec1)
    assert(n_samples >= 5)
    
    error_tol = 0.1
    p_threshold = 0.1
    z_threshold = 2
    
    mean_exec_1   = np.mean(exec1)
    median_exec_1 = np.median(exec1)
    p10_exec_1 = np.percentile(exec1, 10)
    error_exec_1 = np.std(exec1) / (n_samples ** 0.5)
    
    mean_ctime_1   = np.mean(ctime1)
    median_ctime_1 = np.median(ctime1)
    p10_ctime_1 = np.percentile(ctime1, 10)
    error_ctime_1 = np.std(ctime1) / (n_samples ** 0.5)
    
    mean_exec_2   = np.mean(exec2)
    median_exec_2 = np.median(exec2)
    p10_exec_2 = np.percentile(exec2, 10)
    error_exec_2 = np.std(exec2) / (n_samples ** 0.5)
    
    mean_ctime_2   = np.mean(ctime2)
    median_ctime_2 = np.median(ctime2)
    p10_ctime_2 = np.percentile(ctime2, 10)
    error_ctime_2 = np.std(ctime2) / (n_samples ** 0.5)
    
    exec_regressed_ttest = ttest_ind(exec2, exec1, alternative="greater", equal_var=False)
    exec_improved_ttest = ttest_ind(exec2, exec1, alternative="less", equal_var=False)
    ctime_ttest = ttest_ind(ctime2, ctime1, alternative="greater", equal_var=False)

    if 0 in [
        median_exec_1,
        min(exec1),
        min(exec2),
        median_ctime_1,
        min(ctime1),
        mean_exec_1,
        mean_exec_2,
        mean_ctime_1,
        mean_ctime_2,
    ]:
        warnings.warn(
            f"Avoiding division by 0 for {problem}. "
            f"Exec median: {median_exec_1}, min: {min(exec1)}; "
            f"Ctime median: {median_ctime_1}, min: {min(ctime1)}"
        )
        return None
    
    has_large_error_exec = abs(error_exec_1 / mean_exec_1) > error_tol \
                        or abs(error_exec_2 / mean_exec_2) > error_tol
    is_uncertain_exec = has_large_error_exec and (
        mean_exec_1 <= mean_exec_2 <= mean_exec_1 + error_exec_1 + error_exec_2
        if mean_exec_1 <= mean_exec_2
        else mean_exec_2 < mean_exec_1 <= mean_exec_2 + error_exec_1 + error_exec_2
    )
    has_large_error_ctime = abs(error_ctime_1 / mean_ctime_1) > error_tol \
                         or abs(error_ctime_2 / mean_ctime_2) > error_tol
    is_uncertain_ctime = has_large_error_ctime and (
        mean_ctime_1 <= mean_ctime_2 <= mean_ctime_1 + error_ctime_1 + error_ctime_2
        if mean_ctime_1 <= mean_ctime_2
        else mean_ctime_2 < mean_ctime_1 <= mean_ctime_2 + error_ctime_1 + error_ctime_2
    )

    z_error_exec = (error_exec_1**2 + error_exec_2**2) ** 0.5
    z_exec = (mean_exec_2 - mean_exec_1) / z_error_exec
    delta_mean_exec = (mean_exec_2 - mean_exec_1) / mean_exec_1
    delta_median_exec = (median_exec_2 - median_exec_1) / median_exec_1
    delta_p10_exec = (p10_exec_2 - p10_exec_1) / p10_exec_1
    
    exec_regressed = (
        not is_uncertain_exec and exec_regressed_ttest.pvalue <= 0.05
    ) and (
        # ( z_mean_exec > 2 and delta_mean_exec > 0.05 ) \
        # or ( z_median_exec > 2 and delta_median_exec > 0.05 )
        z_exec > z_threshold and (delta_mean_exec > p_threshold or
                                  delta_median_exec > p_threshold or
                                  delta_p10_exec > p_threshold)
    )
    exec_improved = (
        not is_uncertain_exec and exec_improved_ttest.pvalue <= 0.05
    ) and (
        # ( z_mean_exec < -2 and delta_mean_exec < -0.05 ) \
        # or ( z_median_exec < -2 and delta_median_exec < -0.05 )
        z_exec < -z_threshold and (delta_mean_exec < -p_threshold or
                                   delta_median_exec < -p_threshold or
                                   delta_p10_exec < -p_threshold)
    )
    
    z_error_ctime = (error_ctime_1**2 + error_ctime_2**2) ** 0.5
    z_ctime = (mean_ctime_2 - mean_ctime_1) / z_error_ctime
    delta_mean_ctime = (mean_ctime_2 - mean_ctime_1) / mean_ctime_1
    delta_median_ctime = (median_ctime_2 - median_ctime_1) / median_ctime_1
    delta_p10_ctime = (p10_ctime_2 - p10_ctime_1) / p10_ctime_1
    
    ctime_regressed = not is_uncertain_ctime and ctime_ttest.pvalue <= 0.05 and (
        # ( z_mean_ctime > 2 and delta_mean_ctime > 0.05 ) \
        # or ( z_median_ctime > 2 and delta_median_ctime > 0.05 )
        z_ctime > z_threshold and (delta_mean_ctime > p_threshold or
                                   delta_median_ctime > p_threshold or
                                   delta_p10_ctime > p_threshold)
    )

    return {
        "has_large_error_exec": has_large_error_exec,
        "has_large_error_ctime": has_large_error_ctime,
        "is_uncertain_exec": is_uncertain_exec,
        "is_uncertain_ctime": is_uncertain_ctime,
        "exec_regressed_ttest": exec_regressed_ttest,
        "exec_improved_ttest": exec_improved_ttest,
        "ctime_ttest": ctime_ttest,
        "median_exec_1": median_exec_1,
        "median_exec_2": median_exec_2,
        "mean_exec_1": mean_exec_1,
        "mean_exec_2": mean_exec_2,
        "p10_exec_1": p10_exec_1,
        "p10_exec_2": p10_exec_2,
        "median_ctime_1": median_ctime_1,
        "median_ctime_2": median_ctime_2,
        "mean_ctime_1": mean_ctime_1,
        "mean_ctime_2": mean_ctime_2,
        "p10_ctime_1": p10_ctime_1,
        "p10_ctime_2": p10_ctime_2,
        "error_exec_1": error_exec_1,
        "error_exec_2": error_exec_2,
        "error_ctime_1": error_ctime_1,
        "error_ctime_2": error_ctime_2,
        "exec_regressed": exec_regressed,
        "exec_improved": exec_improved,
        "ctime_regressed": ctime_regressed,
    }


def load_ci_metadata():
    with open(CI_JSON_PATH) as f:
        ci_json = json.load(f)

    repo = git.Repo(F_PATH / "../../..", search_parent_directories=True)
    head_sha = repo.git.rev_parse(repo.head.object.hexsha, short=6)
    return ci_json, head_sha


def initialize_markdown_output(out_file, ci_json, head_sha):
    headers = (
        f"| problem | {ci_json['dependencies']['onednn-base']} p<sub>50</sub>  time(ms) "
        f"| {ci_json['dependencies']['onednn-base']} p<sub>10</sub> time (ms)"
        f"| {head_sha} p<sub>50</sub>  time(ms) | {head_sha} p<sub>10</sub> time (ms) | speedup (>1 is faster) |\n"
    )
    with open(out_file, "w") as f:
        f.write(headers + "|:---:|:---:|:---:|:---:|:---:|:---:|\n")


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
