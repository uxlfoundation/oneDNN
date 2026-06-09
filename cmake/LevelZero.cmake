#===============================================================================
# Copyright 2026 Intel Corporation
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

# Manage Level Zero headers location
#===============================================================================

if(LevelZero_cmake_included)
    return()
endif()
set(LevelZero_cmake_included true)

if(NOT (DNNL_GPU_VENDOR STREQUAL "INTEL"
        AND DNNL_GPU_RUNTIME MATCHES "^(SYCL|DPCPP|ZE)$"))
    return()
endif()

if(NOT EXISTS "${DNNL_ZE_INCLUDE_DIR}/level_zero/ze_api.h"
        OR NOT EXISTS "${DNNL_ZE_INCLUDE_DIR}/level_zero/ze_intel_gpu.h")
    message(FATAL_ERROR 
        "Level Zero headers not found at '${DNNL_ZE_INCLUDE_DIR}'. Level Zero
        loader (level_zero/ze_api.h) and Intel GPU driver extensions
        (level_zero/ze_intel_gpu.h) headers are required.")
endif()

file(TO_CMAKE_PATH "${DNNL_ZE_INCLUDE_DIR}" DNNL_ZE_INCLUDE_DIR)
message(STATUS "Found Level Zero headers: ${DNNL_ZE_INCLUDE_DIR}")
include_directories_with_host_compiler(${DNNL_ZE_INCLUDE_DIR})
