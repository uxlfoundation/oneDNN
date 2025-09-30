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

# Usage: bash bench_nightly_performance.sh {baseline_benchdnn_executable} {benchdnn_executable} {baseline_results_file} {new_results_file}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
INPUTS_DIR="${SCRIPT_DIR}/inputs"

TESTS=(
    "--matmul --batch=$INPUTS_DIR/matmul_nightly"
    "--conv --batch=$INPUTS_DIR/conv_nightly"
    "--eltwise --batch=$INPUTS_DIR/eltwise_nightly"
    "--reorder --batch=$INPUTS_DIR/reorder_nightly"
)

for test in "${TESTS[@]}"
do
    $SCRIPT_DIR/run_benchdnn_compare.sh "$1" "$2" "$3" "$4" $test
done
