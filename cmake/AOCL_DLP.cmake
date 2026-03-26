# ******************************************************************************
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
# ******************************************************************************

if(aocl_dlp_cmake_included)
    return()
endif()
set(aocl_dlp_cmake_included true)
include("cmake/options.cmake")

if(NOT DNNL_TARGET_ARCH STREQUAL "X64")
    return()
endif()

if(NOT DNNL_X64_USE_AOCL_DLP)
    return()
endif()

find_package(AOCL_DLP REQUIRED)

if(AOCL_DLP_FOUND)
    list(APPEND EXTRA_SHARED_LIBS ${AOCL_DLP_LIBRARIES})

    include_directories(${AOCL_DLP_INCLUDE_DIRS})

    message(STATUS "AOCL-DLP library: ${AOCL_DLP_LIBRARIES}")
    message(STATUS "AOCL-DLP headers: ${AOCL_DLP_INCLUDE_DIRS}")

    add_definitions(-DDNNL_X64_USE_AOCL_DLP)
endif()
