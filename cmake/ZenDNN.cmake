#*******************************************************************************
# Copyright 2026 Advanced Micro Devices, Inc.
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
#*******************************************************************************

if(zendnn_cmake_included)
    return()
endif()
set(zendnn_cmake_included true)
include("cmake/options.cmake")

if(NOT DNNL_X64_USE_ZEN)
    add_definitions(-DDNNL_X64_USE_ZEN=0)
    return()
endif()

# ZenDNN does not support Windows builds.
if(WIN32)
    message(FATAL_ERROR
        "DNNL_X64_USE_ZEN=ON is not supported on Windows. "
        "Build on Linux, or configure with -DDNNL_X64_USE_ZEN=OFF.")
endif()

# ZenDNN requires CMake >= 3.26.
if(CMAKE_VERSION VERSION_LESS "3.26")
    message(FATAL_ERROR
        "DNNL_X64_USE_ZEN=ON requires CMake >= 3.26. "
        "Current CMake: ${CMAKE_VERSION}. "
        "Upgrade CMake, or configure with -DDNNL_X64_USE_ZEN=OFF.")
endif()

if(NOT ZENDNNROOT AND DEFINED ENV{ZENDNNROOT})
    set(ZENDNNROOT "$ENV{ZENDNNROOT}")
endif()

if(ZENDNNROOT)
    set(zendnnl_DIR "${ZENDNNROOT}/lib/cmake" CACHE PATH "Path to zendnnl CMake config files")
endif()

# Minimum supported ZenDNN version. A missing ZenDNN disables support
# gracefully; a ZenDNN that is present but older than this is treated as a
# misconfiguration and fails the build.
set(ZENDNN_MIN_VERSION "6.0.0")
find_package(zendnnl CONFIG)

if(NOT zendnnl_FOUND)
    message(WARNING "ZenDNN not found. Building oneDNN without ZenDNN support.")
    set(DNNL_X64_USE_ZEN OFF CACHE BOOL "" FORCE)
    add_definitions(-DDNNL_X64_USE_ZEN=0)
    return()
endif()

# zendnnl_VERSION may be unset by some package configs
if(NOT DEFINED zendnnl_VERSION OR zendnnl_VERSION STREQUAL "")
    message(FATAL_ERROR
        "DNNL_X64_USE_ZEN=ON requires ZenDNN >= ${ZENDNN_MIN_VERSION}, but the "
        "ZenDNN package at ${zendnnl_DIR} did not report a version. Use a "
        "ZenDNN package that provides version information, or configure with "
        "-DDNNL_X64_USE_ZEN=OFF.")
elseif("${zendnnl_VERSION}" VERSION_LESS "${ZENDNN_MIN_VERSION}")
    message(FATAL_ERROR
        "DNNL_X64_USE_ZEN=ON requires ZenDNN >= ${ZENDNN_MIN_VERSION}. "
        "Found ZenDNN ${zendnnl_VERSION} at ${zendnnl_DIR}. "
        "Update ZenDNN, or configure with -DDNNL_X64_USE_ZEN=OFF.")
endif()

# Require GCC >= 11.2 or Clang >= 14; ZenDNN builds only with GCC/Clang.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.2")
        message(FATAL_ERROR
            "DNNL_X64_USE_ZEN=ON requires GCC >= 11.2. "
            "Current C++ compiler: ${CMAKE_CXX_COMPILER_ID} "
            "${CMAKE_CXX_COMPILER_VERSION}. "
            "Upgrade GCC, or configure with -DDNNL_X64_USE_ZEN=OFF.")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14")
        message(FATAL_ERROR
            "DNNL_X64_USE_ZEN=ON requires Clang >= 14. "
            "Current C++ compiler: ${CMAKE_CXX_COMPILER_ID} "
            "${CMAKE_CXX_COMPILER_VERSION}. "
            "Upgrade Clang, or configure with -DDNNL_X64_USE_ZEN=OFF.")
    endif()
else()
    message(FATAL_ERROR
        "DNNL_X64_USE_ZEN=ON requires GCC >= 11.2 or Clang >= 14; ZenDNN does "
        "not support other compilers. Current C++ compiler: "
        "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}. "
        "Build with GCC or Clang, or configure with -DDNNL_X64_USE_ZEN=OFF.")
endif()

# shared ZenDNN build is rejected until shared builds are supported.
if(TARGET zendnnl::zendnnl)
    get_target_property(_zendnnl_type zendnnl::zendnnl TYPE)
    if(_zendnnl_type STREQUAL "SHARED_LIBRARY")
        message(FATAL_ERROR
            "DNNL_X64_USE_ZEN=ON requires a static (archive) ZenDNN; the shared "
            "ZenDNN at ${zendnnl_DIR} is unsupported. Rebuild ZenDNN as an "
            "archive (-DZENDNNL_LIB_BUILD_SHARED=OFF -DZENDNNL_LIB_BUILD_ARCHIVE=ON), "
            "or configure with -DDNNL_X64_USE_ZEN=OFF.")
    endif()
endif()

add_definitions(-DDNNL_X64_USE_ZEN=1)
# C++17 requirement is applied per-target via target_compile_features()
# in src/cpu/{,x64/}CMakeLists.txt, not project-wide.

message(STATUS "Found ZenDNN ${zendnnl_VERSION}: ${zendnnl_DIR}")
