#! /bin/bash

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

# Usage: ./git_bisect.sh GOOD_SHA BAD_SHA BUILD_DIR BENCHDNN_CMD

GOOD=$1
BAD=$2
BUILD_DIR=$3
CMD=${@:4}
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

ONEDNN_TEST_SET=NIGHTLY ONEDNN_BUILD_GRAPH=OFF ${SCRIPT_DIR}/build.sh --configure
git bisect reset
git bisect start
git bisect good ${GOOD}
git bisect bad ${BAD}
git bisect run sh -c "make -j -C build && ${BUILD_DIR}/tests/benchdnn/benchdnn ${CMD}"
git bisect log
