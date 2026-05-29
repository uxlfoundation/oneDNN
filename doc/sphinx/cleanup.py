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

DIR = sys.argv[1]


def fix_rst_content(content):
    """Fix RST issues produced by the Doxygen/Doxyrest pipeline."""
    # Join multi-line :math: roles to single lines.
    # Doxygen XML wraps long formulas across lines, and Doxyrest preserves
    # the line breaks, producing multi-line :math: roles that Sphinx rejects.
    content = re.sub(
        r"(:math:`[^`]*)\n\s*([^`]*`)",
        lambda m: m.group(1) + " " + m.group(2).lstrip(),
        content,
    )
    # Repeat to handle formulas wrapped across more than two lines.
    for _ in range(5):
        new = re.sub(
            r"(:math:`[^`]*)\n\s*([^`]*`)",
            lambda m: m.group(1) + " " + m.group(2).lstrip(),
            content,
        )
        if new == content:
            break
        content = new

    return content


# Regex matching Doxygen member-hash separators (_1g, _1a, etc.)
# followed by at least 8 hex digits.
_MEMBER_SEP_RE = re.compile(r"_1(?:gga|ga|gg|g|a)([0-9a-f]{8,})")


def fix_undefined_labels(rst_dir):
    """Generate stub label definitions for cross-references that Doxyrest
    does not emit.  Doxyrest creates :ref: links to group/struct members
    using Doxygen compound IDs, but only defines the top-level compound
    label, not individual member labels."""

    # Remove stale stubs from previous builds so they don't pollute the
    # scan below.
    stubs_path = os.path.join(rst_dir, "_label_stubs.rst")
    if os.path.exists(stubs_path):
        os.remove(stubs_path)

    ref_re = re.compile(r":ref:`[^`]*<(doxid-[^>]+)>`")
    def_re = re.compile(r"^\.\.\s+_(doxid-[^:]+):", re.MULTILINE)
    target_re = re.compile(r":target:`[^`]*<(doxid-[^>]+)>`")

    referenced = set()
    defined = set()
    # Labels defined via :target: inside ref-code-blocks.  These are
    # document-local and cannot be resolved cross-document, but we must
    # not create .. _label: stubs in the same file (that causes duplicates).
    target_defined = {}  # label -> filename
    rst_files = {}  # basename -> full path

    for root, _, fnames in os.walk(rst_dir):
        for fname in fnames:
            if not fname.endswith(".rst"):
                continue
            path = os.path.join(root, fname)
            with open(path) as f:
                content = f.read()
            referenced.update(ref_re.findall(content))
            defined.update(def_re.findall(content))
            for t in target_re.findall(content):
                target_defined[t] = fname
            rst_files[fname] = path

    missing = sorted(referenced - defined)
    if not missing:
        return

    # Map each missing label to the RST file where it belongs.
    file_labels = {}  # filename -> [labels]
    unmapped = []

    for label in missing:
        bare = label[len("doxid-"):]

        # Extract compound ID before the member hash.
        compound = None
        m = _MEMBER_SEP_RE.search(bare)
        if m:
            compound = bare[: m.start()]
        else:
            # Section anchor: page_1section_name
            idx = bare.find("_1")
            if idx > 0:
                compound = bare[:idx]

        if compound:
            # Convert Doxygen compound ID to RST filename.
            fname = compound.replace("_1_1", "_").replace("__", "_") + ".rst"
            # Doxygen concatenates the kind prefix (struct/class/...)
            # directly with the name; RST filenames insert an underscore.
            for pfx in ("struct", "class", "union", "namespace"):
                if (
                    fname.startswith(pfx)
                    and len(fname) > len(pfx) + 4
                    and fname[len(pfx)] != "_"
                ):
                    fname = pfx + "_" + fname[len(pfx):]
                    break
            # If the file already has a :target: for this label, appending
            # a .. _label: to the same file would create a duplicate.
            # Route to orphan stubs instead.
            if fname in rst_files and target_defined.get(label) != fname:
                file_labels.setdefault(fname, []).append(label)
                continue

        unmapped.append(label)

    # Append labels to the end of mapped files.
    for fname, labels in file_labels.items():
        path = rst_files[fname]
        with open(path, "a") as f:
            f.write("\n")
            for lbl in labels:
                f.write(f".. _{lbl}:\n")

    # Create orphan stub file for unmapped labels.
    if unmapped:
        path = os.path.join(rst_dir, "_label_stubs.rst")
        with open(path, "w") as f:
            f.write(":orphan:\n\n")
            for lbl in unmapped:
                f.write(f".. _{lbl}:\n")

    total = sum(len(v) for v in file_labels.values()) + len(unmapped)
    print(f"generated {total} stub labels ({len(unmapped)} orphan)")


# Pass 1: Rename files.
for root, dirs, files in os.walk(DIR):
    for file in files:
        if not file.endswith(".rst"):
            continue
        # XXX: A hack for WSL. Based on setup WSL might not take into account
        # case sensitivity, and as a result doxygen might generate uppercased
        # files.
        # Temp file should be used in order to make files lowcased, because
        # direct renaming doesn't work due to case insensitive file system.
        full_file = os.path.join(root, file)
        if file.lower() != file:
            tmp_file = "tmp_" + file
            full_tmp_file = os.path.join(root, tmp_file)
            os.rename(full_file, full_tmp_file)
            os.rename(full_tmp_file, os.path.join(root, file.lower()))
        if file.startswith("page_dev_guide"):
            stripped_file = file[5:]
            # if destination file already exists then remove the source file
            if stripped_file not in files:
                os.rename(full_file, os.path.join(root, stripped_file))
            else:
                os.remove(full_file)

# Pass 2: Fix RST content (fresh walk to pick up renamed files).
for root, dirs, files in os.walk(DIR):
    for file in files:
        if not file.endswith(".rst"):
            continue
        full_file = os.path.join(root, file)
        with open(full_file, "r") as f:
            content = f.read()
        fixed = fix_rst_content(content)
        if fixed != content:
            with open(full_file, "w") as f:
                f.write(fixed)
            print("fixing RST in " + file)

# Pass 3: Generate missing label definitions.
fix_undefined_labels(DIR)
