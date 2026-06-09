#===============================================================================
# Copyright 2019 Intel Corporation
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

# Manage OpenCL-related compiler flags
#===============================================================================

if(OpenCL_cmake_included)
    return()
endif()
set(OpenCL_cmake_included true)

if((DNNL_GPU_SYCL OR DNNL_GPU_RUNTIME STREQUAL "ZE") AND DNNL_GPU_VENDOR STREQUAL "INTEL")
    add_definitions_with_host_compiler(-DCL_TARGET_OPENCL_VERSION=300)
else()
    add_definitions(-DCL_TARGET_OPENCL_VERSION=120)
endif()

if(NOT EXISTS "${DNNL_OCL_INCLUDE_DIR}/CL/cl.h")
    message(FATAL_ERROR
        "OpenCL headers not found at '${DNNL_OCL_INCLUDE_DIR}'. "
        "Set DNNL_OCL_INCLUDE_DIR to a directory containing 'CL/cl.h'.")
endif()

file(TO_CMAKE_PATH "${DNNL_OCL_INCLUDE_DIR}" DNNL_OCL_INCLUDE_DIR)
message(STATUS "Found OpenCL headers: ${DNNL_OCL_INCLUDE_DIR}")
include_directories_with_host_compiler(${DNNL_OCL_INCLUDE_DIR})

# Tests and examples link dynamically against OpenCL ICD loader
if(DNNL_GPU_RUNTIME STREQUAL "OCL" AND (DNNL_BUILD_TESTS OR DNNL_BUILD_EXAMPLES))
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        message(STATUS "Found OpenCL ICD: ${OpenCL_LIBRARY} (found version \"${OpenCL_VERSION_STRING}\")")
    else()
        message(FATAL_ERROR "OpenCL SDK is not found. You can use OPENCLROOT build option"
                "to specify the path or disable building tests and examples"
                "with ONEDNN_BUILD_EXAMPLES=OFF and ONEDNN_BUILD_TESTS=OFF")
    endif()
endif()
