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

if(DNNL_GPU_SYCL AND DNNL_GPU_VENDOR STREQUAL "INTEL")
    add_definitions(-DCL_TARGET_OPENCL_VERSION=300)
else()
    add_definitions(-DCL_TARGET_OPENCL_VERSION=120)
endif()

if(OpenCL_INCLUDE_DIR)
    message(STATUS "Using user provided OpenCL headers from '${OpenCL_INCLUDE_DIR}'")
    file(TO_CMAKE_PATH ${OpenCL_INCLUDE_DIR} CUSTOM_OCL_HEADERS_PATH)
    include_directories_with_host_compiler(${CUSTOM_OCL_HEADERS_PATH})
else()
    include_directories_with_host_compiler(${PROJECT_SOURCE_DIR}/third_party/opencl)
endif()
