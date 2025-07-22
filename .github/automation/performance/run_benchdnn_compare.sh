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
# Runs a user-supplied benchdnn command on two oneDNN binaries
# Usage example inside GitHub Actions:
#   .github/automation/performance/run_benchdnn_compare.sh \
#       "${BASE_BIN}" "${NEW_BIN}" "${CMD_ORIG}" "${THREADS:-16}"

set -euo pipefail

BASE_BINARY=$1
NEW_BINARY=$2
CMD_STR=$3
THREADS=${4:-16}
REPS=5
PERF='--perf-template=%prb%,%-time%,%-ctime%'
MODE='--mode=P'

read -ra TOKENS <<< "$CMD_STR"
FINAL_ARGS=()
FINAL_ARGS=("${TOKENS[0]}" "${PERF}" "${MODE}" "${TOKENS[@]:1}")

echo "Final benchdnn command:"
printf '%s ' "${FINAL_ARGS[@]}"
echo

for i in $(seq 1 "$REPS"); do
  echo "Running base iteration $i..."
  OMP_NUM_THREADS="$THREADS" "$BASE_BINARY" "${FINAL_ARGS[@]}" >> base.txt
  
  echo "Running new iteration $i..."
  OMP_NUM_THREADS="$THREADS" "$NEW_BINARY" "${FINAL_ARGS[@]}" >> new.txt
done
