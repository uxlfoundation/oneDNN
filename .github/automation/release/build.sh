#! /bin/bash

# *******************************************************************************
# Copyright 2026 Intel Corporation
# Copyright 2026 Institute of Software, Chinese Academy of Sciences
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

# Builds, smoke tests, and installs a oneDNN release package on Linux. Run
# from the oneDNN source directory.
#
# Usage: build.sh <install-dir>
#
# Environment:
#   COMPILER             gcc, riscv64-gcc, or icx
#   CPU_RUNTIME          value for ONEDNN_CPU_RUNTIME
#   GPU_RUNTIME          value for ONEDNN_GPU_RUNTIME
#   DEVTOOLSET_ROOTPATH  (optional) GCC toolset root, set in manylinux images

set -o errexit -o pipefail -o nounset

if [ $# -ne 1 ]; then
    echo "Usage: $0 <install-dir>" >&2
    exit 1
fi
INSTALL_DIR="$1"

GCC_ROOT="${DEVTOOLSET_ROOTPATH:+$DEVTOOLSET_ROOTPATH/usr}"
GXX="${GCC_ROOT:+$GCC_ROOT/bin/}g++"
CMAKE_EXTRA_ARGS=()
CTEST_ARGS=(-E "gpu|cross-engine")

case "$COMPILER" in
    gcc)
        export CC="${GCC_ROOT:+$GCC_ROOT/bin/}gcc"
        export CXX="$GXX"
        ;;
    riscv64-gcc)
        export CC=riscv64-linux-gnu-gcc-14
        export CXX=riscv64-linux-gnu-g++-14
        CMAKE_EXTRA_ARGS+=(
            -DCMAKE_TOOLCHAIN_FILE="$PWD/cmake/toolchains/riscv64.cmake"
        )
        # Keep the release smoke test short under QEMU.
        CTEST_ARGS=(--no-tests=error -R "^cpu-primitives-matmul-cpp$")
        ;;
    icx)
        set +o errexit +o pipefail +o nounset
        # shellcheck disable=SC1091
        source "${ONEAPI_ROOT:-/opt/intel/oneapi}/setvars.sh"
        set -o errexit -o pipefail -o nounset
        export CC=icx CXX=icpx
        if [ -n "$GCC_ROOT" ]; then
            # Point the compiler to the GCC toolset's libstdc++.
            export CFLAGS="--gcc-toolchain=$GCC_ROOT"
            export CXXFLAGS="$CFLAGS"
        fi
        # Compile host code with g++. This also makes OpenMP builds use
        # libgomp instead of libiomp5.
        CMAKE_EXTRA_ARGS+=(-DONEDNN_DPCPP_HOST_COMPILER="$GXX")
        ;;
    *)
        echo "Unsupported compiler: $COMPILER" >&2
        exit 1
        ;;
esac
command -v "$CXX"

cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DONEDNN_CPU_RUNTIME="$CPU_RUNTIME" \
    -DONEDNN_GPU_RUNTIME="$GPU_RUNTIME" \
    -DONEDNN_BUILD_EXAMPLES=ON \
    -DONEDNN_BUILD_TESTS=OFF \
    "${CMAKE_EXTRA_ARGS[@]}"
cmake --build build --parallel "$(nproc)"
mkdir -p "$INSTALL_DIR/env"
cp "$(dirname "$0")/env/vars.sh" "$INSTALL_DIR/env/"
# Smoke test the CPU engine only, GPU is not expected to be available.
ctest --test-dir build --output-on-failure "${CTEST_ARGS[@]}"
cmake --install build
