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
import os
from typing import Iterable, List, Optional

import ctest_utils

PAGE_TITLE = "AArch64 Testing Status"


class MdTree:
    def __init__(self, title: Optional[str] = None):
        self.depth, self.title = self._parse_depth_and_title(title)
        self.body = ""
        self.children: List[MdTree] = []

    @staticmethod
    def _parse_depth_and_title(title: Optional[str]):
        if title is None:
            return 0, None
        depth = 0
        while depth < len(title) and title[depth] == "#":
            depth += 1
        return depth, title[depth:].strip()

    def add_child(self, child: "MdTree"):
        if not self.children:
            self.children.append(child)
            return
        last_child = self.children[-1]
        if child.depth > last_child.depth:
            last_child.add_child(child)
            return
        self.children.append(child)

    def __contains__(self, key: str):
        try:
            self.__getitem__(key)
        except KeyError:
            return False
        return True

    def __getitem__(self, key: str):
        for child in self.children:
            if child.title == key:
                return child
        raise KeyError(key)

    def __setitem__(self, key: str, value: str):
        try:
            child = self.__getitem__(key)
        except KeyError:
            child = MdTree("#" * (self.depth + 1) + f" {key}")
            self.children.append(child)
        parsed = MdConverter.parse(value.splitlines())
        child.body = parsed.body
        child.children = parsed.children

    def __str__(self):
        my_str = ""
        if self.title is not None:
            my_str = "#" * self.depth + f" {self.title}\n\n"
        body = self.body.strip()
        if body:
            my_str += body + "\n\n"
        for child in self.children:
            child_str = str(child).strip()
            if child_str:
                my_str += child_str + "\n\n"
        return my_str.strip()


class MdConverter:
    @staticmethod
    def parse(data: Iterable[str]):
        root = MdTree()
        last_block = root
        in_triple_tick = False
        for line in data:
            if line.startswith("#") and not in_triple_tick:
                last_block = MdTree(line)
                root.add_child(last_block)
                continue
            if line.startswith("```"):
                in_triple_tick = not in_triple_tick
            last_block.body += line.rstrip() + "\n"
        return root

    @staticmethod
    def tree2md(in_dict: MdTree, out_file):
        with open(out_file, "w") as f:
            f.write(str(in_dict))

    @staticmethod
    def md2tree(in_file):
        if not os.path.isfile(in_file):
            return MdTree()

        with open(in_file) as f:
            return MdConverter.parse(f)


def parse(file, title, subtitle, body):
    """
    Add a new section/subsection or an existing section/subsection
    without overwriting existing section/subsections
    """
    converter = MdConverter()
    d = converter.md2tree(file)
    if PAGE_TITLE not in d:
        d[PAGE_TITLE] = ""
    parent = d[PAGE_TITLE]
    if title not in parent:
        parent[title] = ""
    k0 = parent[title]
    k0[subtitle] = body
    converter.tree2md(d, file)


def parse_unit(args):
    failed_tests = ctest_utils.get_failed_tests(args.in_file)

    body = ""
    if failed_tests:
        body = "| :x: | Failed Test |\n| :-----------: | :------: |\n"
        for test in failed_tests:
            body += f"| :x: | {test} |\n"
        body = body[:-1]  # Strip the last '\n'
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
    # xml required for machine-readable unit test results
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
