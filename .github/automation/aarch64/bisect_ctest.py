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
import argparse
from collections import defaultdict
import subprocess
import pathlib

F_PATH = pathlib.Path(__file__).parent.resolve()


def parse_ctest(args):
    with open(args.file) as f:
        r = f.readlines()

    failed_cases = defaultdict(list)
    for l in r:
        if ":FAILED" in l:
            l = l.split("__REPRO: ")[1]
            op = l.split(" ")[0]
            failed_cases[op].append(l.replace("\n", ""))

    if args.unique:
        return [x[0] for x in failed_cases.values()]

    return [
        x
        for xs in failed_cases.values()
        for x in xs
    ] # Flatten list


def main():
    args_parser = argparse.ArgumentParser(
        description="oneDNN log converter",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    args_parser.add_argument("file", nargs="?", help="input file")
    args_parser.add_argument(
        "--unique",
        action="store_true",
        help="whether to return only one test case per unique op",
    )
    args = args_parser.parse_args()
    cases = parse_ctest(args)

    for case in cases:
        result = subprocess.run(
            args=[
                str(F_PATH / "git_bisect.sh HEAD~5 HEAD ")
                , str(F_PATH.parent.parent.parent / "build")
                , case
            ],
            shell=True,
            # capture_output=True,
        )
        # if result.returncode != 0:
        #     print(f"Unable to determine hash for {case}")
        #     print(result.stdout)
        #     continue

        # bad_hash = result.stdout.decode('utf-8').split('\n')[-2].split('[')[1].split(']')[0]
        # print(
        #     f"First bad hash for {case}: {bad_hash}"
        # )


if __name__ == "__main__":
    main()
