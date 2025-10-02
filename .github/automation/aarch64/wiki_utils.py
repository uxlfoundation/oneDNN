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

PAGE_TITLE = "AArch64 Testing Status"


class MdConverter:
    # Recursive function to convert dict to md sections
    def __unpack_md_dict(self, data, out_str="", level=1):
        # base case
        if not isinstance(data, dict):
            return out_str + data + "\n"

        for k, v in data.items():
            out_str += "#" * level + " " + k + "\n"
            out_str = self.__unpack_md_dict(v, out_str, level + 1)
        return out_str

    # Recursive function to convert md to dict
    def __unpack_md(self, data, level=1):
        prev = ""
        inner = []
        out = {}
        for l in data:
            if l[:level] == "#" * level and l[level] != "#":
                if prev:
                    out[prev[level + 1 :]] = self.__unpack_md(inner, level + 1)
                    inner = []
                prev = l.strip()
            else:
                inner.append(l)
        if prev:
            out[prev[level + 1 :]] = self.__unpack_md(inner, level + 1)
        elif inner:
            # base case
            return "".join(inner)
        return out

    def dict2md(self, in_dict, out_file):
        output = self.__unpack_md_dict(in_dict)
        with open(out_file, "w") as f:
            f.write(output)

    def md2dict(self, in_file):
        if not os.path.isfile(in_file):
            return {}

        with open(in_file) as f:
            r = f.readlines()

        out = self.__unpack_md(r)

        return out


def parse(file, title, subtitle, body):
    """
    Add a new section/subsection or an existing section/subsection
    without overwriting existing section/subsections
    """
    converter = MdConverter()
    d = converter.md2dict(file)
    if PAGE_TITLE not in d:
        d[PAGE_TITLE] = {}
    k0 = {}
    if title in d[PAGE_TITLE]:
        k0 = d[PAGE_TITLE][title]
    k0[subtitle] = body
    d[PAGE_TITLE][title] = k0
    converter.dict2md(d, file)


def parse_unit(args):
    failed_tests = ctest_utils.get_failed_tests(args.in_file)

    body = ""
    if failed_tests:
        body = "| :x: | Failed Test |\n" "| :-----------: | :------: |\n"
        for test in failed_tests:
            body += f"| :x: | {test} |\n"
        body = body[:-1] # Strip the last '\n'
    else:
        body = ":white_check_mark: unit tests passed"

    parse(args.out_file, "Unit test results", args.title, body)


def parse_perf(args):
    with open(args.in_file) as f:
        body = f.read()
    parse(args.out_file, "Performance test results", args.title, body)


def main():
    parser = argparse.ArgumentParser(
        description="oneDNN wiki update tools",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers()

    unit_parser = subparsers.add_parser("add-unit", help="add unit test result")
    unit_parser.add_argument(
        "--title", required=True, help="title of unit test run"
    )
    # xml is the only machine-readable output format from ctest
    unit_parser.add_argument(
        "--in-file", required=True, help="xml file storing test results"
    )
    # md format required for github wiki
    unit_parser.add_argument(
        "--out-file", required=True, help="md file to write to"
    )
    unit_parser.set_defaults(func=parse_unit)

    perf_parser = subparsers.add_parser(
        "add-perf", help="add performance test result"
    )
    perf_parser.add_argument(
        "--title", required=True, help="title of perf test run"
    )
    perf_parser.add_argument(
        "--in-file",
        required=True,
        help="md file storing performance test results",
    )
    perf_parser.add_argument(
        "--out-file", required=True, help="md file to write to"
    )
    perf_parser.set_defaults(func=parse_perf)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
