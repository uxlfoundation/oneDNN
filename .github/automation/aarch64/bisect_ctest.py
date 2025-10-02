#!/usr/bin/env python

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
import argparse
import ctest_utils
import os
import pathlib
import subprocess

F_PATH = pathlib.Path(__file__).parent.resolve()


def print_to_github_message(message):
    if "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            print(message, file=f)


def create_github_message(results_dict):
    message = (
        "### :x: Benchdnn Test Failures\n"
        "| Benchdnn Test | Bad Hash |\n"
        "| :-----------: | :------: |\n"
    )

    for case, hash in results_dict.items():
        message += f"|{case}|{hash}|\n"

    return message


def main():
    args_parser = argparse.ArgumentParser(
        description="oneDNN log converter",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    args_parser.add_argument("good", nargs="?", help="good hash")
    args_parser.add_argument("file", nargs="?", help="input file")
    args_parser.add_argument(
        "--unique",
        action="store_true",
        help="whether to return only one test case per unique op",
    )
    args = args_parser.parse_args()
    cases = ctest_utils.failed_benchdnn_tests(args.file, args.unique)

    results_dict = {}
    for case in cases:
        bisect_cmd = str(F_PATH / f"git_bisect.sh")
        build_dir = str(F_PATH.parent.parent.parent / "build")
        result = subprocess.run(
            args=[
                f'bash {bisect_cmd} {args.good} HEAD {build_dir} "{case}"'
            ],
            shell=True,
            capture_output=True,
        )
        print(result.stdout.decode("utf-8"))
        if result.returncode != 0:
            print(f"Unable to determine hash for {case}")
            results_dict[case] = "Unknown"
            continue

        bad_hash = (
            result.stdout.decode("utf-8")
            .split("\n")[-2]
            .split("[")[1]
            .split("]")[0]
        )
        print(f"First bad hash for {case}: {bad_hash}")
        results_dict[case] = bad_hash

    if results_dict:
        print_to_github_message(create_github_message(results_dict))


if __name__ == "__main__":
    main()
