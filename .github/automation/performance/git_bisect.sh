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

GOOD=$1
BAD=$2
BUILD_DIR=$3
CMD=${@:4}

git bisect reset
git bisect start
git bisect good ${GOOD}
git bisect bad ${BAD}
git bisect run sh -c "make -j -C ${BUILD_DIR} && ./${BUILD_DIR}/tests/benchdnn/benchdnn ${CMD}"
git bisect log
