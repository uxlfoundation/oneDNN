#===============================================================================
# Copyright 2025 Intel Corporation
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

# Build oneDNN for PR linter checks.

set -o errexit -o pipefail -o noclobber

export CC=clang
export CXX=clang++

if [[ "$ONEDNN_ACTION" == "configure" ]]; then
    if [[ "$GITHUB_JOB" == "pr-clang-tidy" ]]; then
      set -x
      cmake \
          -Bbuild -S. \
          -DCMAKE_BUILD_TYPE=debug \
          -DONEDNN_BUILD_GRAPH=ON \
          -DDNNL_EXPERIMENTAL=ON \
          -DDNNL_EXPERIMENTAL_PROFILING=ON \
          -DDNNL_EXPERIMENTAL_UKERNEL=ON \
          -DONEDNN_EXPERIMENTAL_LOGGING=ON \
          -DDNNL_CPU_RUNTIME=OMP \
          -DDNNL_GPU_RUNTIME=OCL \
          -DDNNL_WERROR=ON \
          -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
      set +x
    elif [[ "$GITHUB_JOB" == "pr-format-tags" ]]; then
      set -x
      cmake -B../build -S. -DONEDNN_BUILD_GRAPH=OFF
      set +x
    else
      echo "Unknown linter job: $GITHUB_JOB"
      exit 1
    fi
elif [[ "$ONEDNN_ACTION" == "build" ]]; then
    set -x
    cmake --build build -j`nproc`
    set +x
else
    echo "Unknown action: $ONEDNN_ACTION"
    exit 1
fi
