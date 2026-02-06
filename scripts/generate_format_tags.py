#!/usr/bin/env python
################################################################################
# Copyright 2021 Intel Corporation
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
################################################################################

import os
import re
import sys


def is_special_tag(t):
    return "undef" in t or "any" in t or "tag_last" in t


def is_abc_tag(t):
    if is_special_tag(t):
        return False
    t = t.replace("dnnl_", "")
    for c in t:
        c = c.lower()
        if c.isdigit():
            continue
        if c.isalpha():
            if "a" <= c and c <= "l":
                continue
        return False
    return True


class Tag:
    def __init__(self, line):
        m = re.match(r"\s*(\w+)\s*=\s*(\w+)", line)
        if m:
            self.lhs = m.group(1)
            self.rhs = m.group(2)
        else:
            m = re.match(r"\s*(\w+)", line)
            self.lhs = None if m is None else m.group(1)
            self.rhs = None

        self.is_special = is_special_tag(self.lhs)
        self.is_abc = is_abc_tag(self.lhs)
        if self.is_special:
            self.rhs = None
        elif not self.rhs:
            assert self.is_abc, "Expected abc-tag: %s" % line

    def lhs_base_tag(self):
        if self.lhs is None:
            return None
        for s in ["undef", "any", "last"]:
            if s in self.lhs:
                return s
        return self.lhs.replace("dnnl_", "")

    def rhs_base_tag(self):
        if self.rhs is None:
            return None
        return self.rhs.replace("dnnl_", "")

    def __str__(self):
        return str((self.lhs, self.rhs))


def usage():
    print(
        """\
Usage: %s

Updates dnnl.hpp header with missing format tags from dnnl_types.h"""
        % sys.argv[0]
    )
    sys.exit(1)


for arg in sys.argv:
    if "-help" in arg:
        usage()

script_root = os.path.dirname(os.path.realpath(__file__))

dnnl_types_h_path = f"{script_root}/../include/oneapi/dnnl/dnnl_types.h"
dnnl_hpp_path = f"{script_root}/../include/oneapi/dnnl/dnnl.hpp"

c_tags = []
cpp_tags = []

# Parse tags from dnnl_types.h
with open(dnnl_types_h_path) as f:
    s = f.read()
    m = re.search(r".*enum(.*?)dnnl_format_tag_t", s, re.S)
    if m is not None:
        for line in m.group(1).splitlines():
            line = line.strip()
            if not line.startswith("dnnl"):
                continue
            c_tags.append(Tag(line))

# Parse tags from dnnl.hpp
with open(dnnl_hpp_path) as f:
    dnnl_hpp_contents = f.read()
    m = re.search(r"(enum class format_tag.*?)};", dnnl_hpp_contents, re.S)
    if m is None:
        print("Can't find format tag enum")
        sys.exit(1)
    dnnl_hpp_format_tag = m.group(1)
    for line in dnnl_hpp_format_tag.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        cpp_tags.append(Tag(line))

# Validate dnnl.hpp tags
for cpp_tag in cpp_tags:
    if cpp_tag.is_special:
        continue
    if cpp_tag.rhs:
        if cpp_tag.lhs_base_tag() == cpp_tag.rhs_base_tag():
            continue
        tags = [t for t in c_tags if t.lhs_base_tag() == cpp_tag.lhs_base_tag()]
        if tags:
            if cpp_tag.rhs_base_tag() == tags[0].rhs_base_tag():
                continue
        print("Can't validate tag: %s" % cpp_tag)

# Find missing aliases in dnnl.hpp
missing_dnnl_hpp_tag_lines = []
for c_tag in c_tags:
    if c_tag.is_special:
        continue
    for tag in cpp_tags:
        if tag.lhs_base_tag() is None:
            continue
        if tag.lhs_base_tag() == c_tag.lhs_base_tag():
            break
    else:
        base = c_tag.lhs_base_tag()
        line = f"        {base} = dnnl_{base},"
        missing_dnnl_hpp_tag_lines.append(line)

if not missing_dnnl_hpp_tag_lines:
    exit(0)

old_dnnl_hpp_format_tag = dnnl_hpp_format_tag
dnnl_hpp_format_tag = dnnl_hpp_format_tag.rstrip()
dnnl_hpp_format_tag += "\n" + "\n".join(missing_dnnl_hpp_tag_lines) + "\n    "

dnnl_hpp_contents = dnnl_hpp_contents.replace(
    old_dnnl_hpp_format_tag, dnnl_hpp_format_tag
)

# Update dnnl.hpp
with open(dnnl_hpp_path, "w") as f:
    f.write(dnnl_hpp_contents)
