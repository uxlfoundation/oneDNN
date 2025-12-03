#===============================================================================
# Copyright 2018 Intel Corporation
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

# Manage TBB-related compiler flags
#===============================================================================

if(TBB_cmake_included)
    return()
endif()
set(TBB_cmake_included true)
include("cmake/Threading.cmake")

macro(add_tbb_threading)
find_package(TBB REQUIRED COMPONENTS tbb)
if(TBB_FOUND)
    if(WIN32)
        # On Windows we must link to debug version of TBB library to ensure ABI compatibility
        # with MSVC debug runtime.
        set_property(TARGET TBB::tbb PROPERTY "MAP_IMPORTED_CONFIG_RELWITHMDD" "DEBUG")
    else()
        # On Linux TBB::tbb target may link to libtbb_debug.so which is not compatible with libtbb.so. Linking
        # application to both may result in undefined behavior.
        # See https://uxlfoundation.github.io/oneTBB/main/intro/limitations.html#debug-tbb-in-the-sycl-program
        set_property(TARGET TBB::tbb PROPERTY "MAP_IMPORTED_CONFIG_RELWITHMDD" "RELEASE")
        set_property(TARGET TBB::tbb PROPERTY "MAP_IMPORTED_CONFIG_DEBUG" "RELEASE")
    endif()
    get_target_property(TBB_INCLUDE_DIRS TBB::tbb INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(TBB_IMPORTED_LOCATION TBB::tbb IMPORTED_LOCATION_RELEASE)
    if(NOT TBB_IMPORTED_LOCATION)
        get_target_property(TBB_IMPORTED_LOCATION TBB::tbb IMPORTED_LOCATION_DEBUG)
    endif()
    message(STATUS "Found TBB: ${TBB_IMPORTED_LOCATION}")

    include_directories_with_host_compiler(${TBB_INCLUDE_DIRS})
    list(APPEND EXTRA_SHARED_LIBS TBB::tbb)
elseif(DNNL_CPU_RUNTIME STREQUAL "NONE")
    message(FATAL_ERROR "For GPU only SYCL configuration TBB is required for testing.")
else()
    message(FATAL_ERROR "DNNL_CPU_THREADING_RUNTIME is ${DNNL_CPU_THREADING_RUNTIME} but TBB is not found.")
endif()
endmacro()

if(NOT DNNL_CPU_THREADING_RUNTIME STREQUAL "TBB")
    return()
endif()

add_tbb_threading()

# Adds definitions for heterogeneous ISA testing
add_definitions(-DTBB_PREVIEW_TASK_ARENA_CONSTRAINTS_EXTENSION=1)
