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
#
# Runs a user-supplied benchdnn command on two oneDNN binaries.
# Usage
#   .github/automation/performance/run_benchdnn_compare.sh \
#       ${BASE_BIN} ${NEW_BIN} ${BASE_OUT} ${NEW_OUT} ${CMD}

set -euo pipefail

BASE_BINARY=$1
NEW_BINARY=$2
BASE_OUT=$3
NEW_OUT=$4

# Capture all benchdnn driver arguments from $5 onward
# This allows passing things like: --conv --dir=FWD_D --dt=f32 --alg=direct mb1_ic64oc256_ih200oh200kh1sh1dh0ph0_iw267ow267kw1sw1dw0pw0
BENCHDNN_ARGS=("${@:5}")

REPS=5
PERF='--perf-template=%prb%,%-time%,%-ctime%'
MODE='--mode=P'

FINAL_ARGS=("${BENCHDNN_ARGS[0]}" "${PERF}" "${MODE}" "${BENCHDNN_ARGS[@]:1}")

echo ${FINAL_ARGS[@]}

SECONDS=0

for i in $(seq 1 "$REPS"); do
  echo "Testing loop ${i} / ${REPS}..."

  $BASE_BINARY ${FINAL_ARGS[@]} >> $BASE_OUT
  $NEW_BINARY ${FINAL_ARGS[@]} >> $NEW_OUT
done

duration=$SECONDS
echo "Completed in $((duration / 60)):$((duration % 60))"
