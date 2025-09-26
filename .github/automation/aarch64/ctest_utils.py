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
from collections import defaultdict
import xml.etree.ElementTree as ET


def failed_benchdnn_tests(file, unique):
    with open(file) as f:
        r = f.readlines()

    failed_cases = defaultdict(list)
    for l in r:
        if ":FAILED" in l:
            l = l.split("__REPRO: ")[1]
            op = l.split(" ")[0]
            failed_cases[op].append(l.replace("\n", ""))

    if unique:
        return [x[0] for x in failed_cases.values()]

    return [x for xs in failed_cases.values() for x in xs]  # Flatten list


def get_failed_tests(file):
    tree = ET.parse(file)
    root = tree.getroot()
    failed_tests = [
        child.attrib["name"]
        for child in root
        if child.attrib["status"] == "fail"
    ]
    return failed_tests
