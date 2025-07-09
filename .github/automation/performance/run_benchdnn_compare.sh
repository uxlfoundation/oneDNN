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
#
set -euo pipefail

BASE_BIN=$1       # path to baseline benchdnn binary
NEW_BIN=$2        # path to new benchdnn binary
CMD_ORIG=$3       # full benchdnn command as a single string
THREADS=${4:-16}  # OMP threads (default: 16)
REPS=${5:-5}      # repetitions (default: 5)

# Prepend this to every run (includes mode=P and perf-template)
PERF="--mode=P --perf-template=%prb%,%-time%,%-ctime%"

# Build final args
read -ra FINAL_ARGS <<< "$PERF $CMD_ORIG"

echo "Final arguments:"
printf '  %s\n' "${FINAL_ARGS[@]}"
echo "Threads: $THREADS | Repetitions: $REPS"
echo

# Run benchdnn on baseline
echo "Running BASE_BIN: $BASE_BIN"
ls -l "$BASE_BIN" || { echo "::error::Baseline binary not found"; exit 1; }
echo "Working directory: $PWD"

: > base.txt
for ((i=1; i<=REPS; i++)); do
  echo "[base] iteration $i / $REPS"
  OMP_NUM_THREADS=$THREADS "$BASE_BIN" "${FINAL_ARGS[@]}" >> base.txt 2>&1
done

# Run benchdnn on new build
: > new.txt
for ((i=1; i<=REPS; i++)); do
  echo "[new ] iteration $i / $REPS"
  OMP_NUM_THREADS=$THREADS "$NEW_BIN" "${FINAL_ARGS[@]}" >> new.txt 2>&1
done

echo "===== base.txt ====="
cat base.txt || echo "(empty)"

echo "===== new.txt ====="
cat new.txt || echo "(empty)"
