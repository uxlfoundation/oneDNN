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
import subprocess
import sys


def benchdnn_cmd_builder(args):
    driver = args[0]
    batch = args[1]
    return [
        "./build/tests/benchdnn/benchdnn",
        "--" + driver,
        "--mode=L",
        "--batch=" + batch,
    ]


def ctest_cmd_builder(args):
    pattern = args[0]
    return ["ctest", "--test-dir", "./build", "-R", pattern]


def main():
    script = sys.argv[1].split("\n")[1:]

    cmds = []

    for line in script:
        line_parts = line.split(" ")
        test_type = line_parts[0]
        args = line_parts[1:]

        cmd = []

        if test_type == "batch":
            cmd = benchdnn_cmd_builder(args)
        elif test_type == "ctest":
            cmd = ctest_cmd_builder(args)
        else:
            raise ValueError("Unrecognised test_type!")

        cmds.append(cmd)

        print(f"{cmd=}")

    # for cmd in cmds:
    #    subprocess.run(cmd)


if __name__ == "__main__":
    main()
