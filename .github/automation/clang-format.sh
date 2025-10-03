#!/usr/bin/env bash

#===============================================================================
# Copyright 2019-2020 Intel Corporation
# Copyright 2025 Arm Ltd. and affiliates
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
#===============================================================================

CLANG_FORMAT=clang-format-11

echo "Checking ${CLANG_FORMAT}"
if ! ${CLANG_FORMAT} --version; then
    echo ${CLANG_FORMAT} is not available or not working correctly.
    exit 1
fi

echo "Starting format check..."

src_regex='.*\.(c|h|cpp|hpp|cxx|hxx|cl)$'

# Treat each argument as an input file. If called with no arguments check the
# whole repo.
if [[ $# -gt 0 ]]; then
    base_sha=$1
    file_list=$(git diff --name-only $base_sha | grep -E "$src_regex")
    echo "Checking: $file_list"
    for filename in $file_list; do ${CLANG_FORMAT} -style=file -i $filename; done
else
    find "$(pwd)" -type f -regextype posix-egrep -regex "$src_regex" -exec "$CLANG_FORMAT" -style=file -i {} \+
fi

RETURN_CODE=0

echo $(git status) | grep "nothing to commit" > /dev/null
if [ $? -eq 1 ]; then
    echo "Clang-format check FAILED! The following files must be formatted with ${CLANG_FORMAT}:"
    echo "$(git diff --name-only)"
    echo
    echo "Changes required to pass this check:"
    echo "$(git diff)"
    echo
    RETURN_CODE=3
else
    echo "Clang-format check PASSED!"
fi

exit ${RETURN_CODE}
