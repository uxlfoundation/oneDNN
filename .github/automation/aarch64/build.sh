#! /bin/bash

# *******************************************************************************
# Copyright 2024-2026 Arm Limited and affiliates.
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

# Build oneDNN for aarch64.

set -o errexit -o pipefail -o noclobber

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Defines MP, CC, CXX and OS.
source ${SCRIPT_DIR}/common.sh

CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-"Release"}
ONEDNN_TEST_SET=${ONEDNN_TEST_SET:-"SMOKE"}
ONEDNN_BUILD_GRAPH=${ONEDNN_BUILD_GRAPH:-"ON"}
ONEDNN_EXPERIMENTAL_UKERNEL=${ONEDNN_EXPERIMENTAL_UKERNEL:-"OFF"}

if [[ "$ONEDNN_ACTION" == "configure" ]]; then
    # The caller supplies ACL_ROOT_DIR only when it requests the ACL backend.
    acl_options=()
    if [[ -n "${ACL_ROOT_DIR:-}" ]]; then
        acl_options=(-DDNNL_AARCH64_USE_ACL=ON)
    fi

    if [[ "$GITHUB_JOB" == "pr-clang-tidy" ]]; then
        CC=clang CXX=clang++ \
        cmake \
            -Bbuild -S. \
            "${acl_options[@]}" \
            -DONEDNN_BUILD_GRAPH=ON \
            -DDNNL_CPU_RUNTIME=OMP \
            -DONEDNN_WERROR=ON \
            -DDNNL_BUILD_FOR_CI=ON \
            -DONEDNN_TEST_SET=NO_CORR \
            -DCMAKE_BUILD_TYPE=Debug \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
            -DDNNL_EXPERIMENTAL_UKERNEL=OFF
        set +x
    else
        set -x
        cmake \
            -Bbuild -S. \
            "${acl_options[@]}" \
            -DONEDNN_BUILD_GRAPH=$ONEDNN_BUILD_GRAPH \
            -DDNNL_CPU_RUNTIME=$ONEDNN_THREADING \
            -DONEDNN_WERROR=ON \
            -DDNNL_BUILD_FOR_CI=ON \
            -DONEDNN_TEST_SET=$ONEDNN_TEST_SET \
            -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
            -DCMAKE_SKIP_BUILD_RPATH=FALSE \
            -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
            -DDNNL_EXPERIMENTAL_UKERNEL=$ONEDNN_EXPERIMENTAL_UKERNEL
        set +x
    fi

    # CMake only warns about unused options. Fail if the requested backend was
    # ignored, for example when building a revision that has removed ACL.
    if [[ -n "${ACL_ROOT_DIR:-}" ]] && ! grep -Eq \
            '^DNNL_AARCH64_USE_ACL:(BOOL|INTERNAL)=ON$' build/CMakeCache.txt; then
        echo "ACL was requested but was not enabled by oneDNN" >&2
        exit 1
    fi
elif [[ "$ONEDNN_ACTION" == "build" ]]; then
    set -x
    cmake --build build -j"$(nproc)"
    set +x
else
    echo "Unknown action: $ONEDNN_ACTION"
    exit 1
fi
