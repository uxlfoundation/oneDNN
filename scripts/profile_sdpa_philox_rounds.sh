#!/bin/bash
# Sweeps PHILOX_ROUNDS (src/gpu/intel/include/philox.h) from 0 to 10 and
# prints the forward fused-SDPA-dropout perf time at each round count, to
# locate how much of the dropout overhead is pure Philox RNG mixing cost
# vs. fixed per-element scaffolding (key schedule, counter/offset setup,
# threshold compare, tile_binary apply-pass).
#
# Usage: ./scripts/profile_sdpa_philox_rounds.sh <build_dir> <dropout_json> [no_dropout_json]
#
# Requires: a configured CMake build dir with 'ninja dnnl benchdnn' already
# runnable from it (same compiler/env as the original configure), and a
# graph-API JSON test case exercising ocl:micro:reusable with dropout.
set -euo pipefail

BUILD_DIR=${1:?usage: $0 <build_dir> <dropout_json> [no_dropout_json]}
DROPOUT_JSON=${2:?usage: $0 <build_dir> <dropout_json> [no_dropout_json]}
NO_DROPOUT_JSON=${3:-}

ROUNDS_FILE="$(dirname "$0")/../src/gpu/intel/include/philox.h"
REPS=3

restore() {
  git -C "$(dirname "$0")/.." checkout -- "$ROUNDS_FILE"
}
trap restore EXIT

echo "round,rep,min_ms,avg_ms"

if [ -n "$NO_DROPOUT_JSON" ]; then
  for i in $(seq 1 $REPS); do
    LINE=$(./"$BUILD_DIR"/tests/benchdnn/benchdnn --graph --mode=P --engine=gpu \
        --case="$NO_DROPOUT_JSON" 2>&1 | grep "total perf" || true)
    MIN=$(echo "$LINE" | sed -n 's/.*min(ms):\([0-9.]*\).*/\1/p')
    AVG=$(echo "$LINE" | sed -n 's/.*avg(ms):\([0-9.]*\).*/\1/p')
    echo "floor,$i,$MIN,$AVG"
  done
fi

for R in 0 1 2 3 4 5 6 7 8 9 10; do
  sed -i "s/#define PHILOX_ROUNDS .*/#define PHILOX_ROUNDS $R/" "$ROUNDS_FILE"
  ninja -C "$BUILD_DIR" dnnl benchdnn >&2
  for i in $(seq 1 $REPS); do
    LINE=$(./"$BUILD_DIR"/tests/benchdnn/benchdnn --graph --mode=P --engine=gpu \
        --case="$DROPOUT_JSON" 2>&1 | grep "total perf" || true)
    MIN=$(echo "$LINE" | sed -n 's/.*min(ms):\([0-9.]*\).*/\1/p')
    AVG=$(echo "$LINE" | sed -n 's/.*avg(ms):\([0-9.]*\).*/\1/p')
    echo "$R,$i,$MIN,$AVG"
  done
done
