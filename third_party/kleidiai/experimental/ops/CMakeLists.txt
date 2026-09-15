#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# -------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.13)
project(
    KleidiAIOps
    VERSION 0.0.1
    DESCRIPTION "Benchmark GEMM kernels of various flavours for AArch64"
    LANGUAGES C CXX)

if(NOT DEFINED PROJECT_IS_TOP_LEVEL)
    if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
        set(PROJECT_IS_TOP_LEVEL ON)
    else()
        set(PROJECT_IS_TOP_LEVEL OFF)
    endif()
endif()

# Search for additional CMake helpers in the cmake/ subdir
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# -------------------------------------------------------------------------------------------------
# cmake-format: off

# Default option values
set(_KLEIDIAI_OPS_ENABLE_THREADS_DEFAULT OFF)
set(_KLEIDIAI_OPS_ENABLE_BIND_THREADS_DEFAULT OFF)
set(_KLEIDIAI_OPS_SVE_ARCH_DEFAULT armv8.2-a+sve+fp16+dotprod)

# Enable threading support by default on Linux and macOS
if(LINUX OR APPLE)
  set(_KLEIDIAI_OPS_ENABLE_THREADS_DEFAULT ON)
endif()

# Enable thread binding support by default on Linux
if(LINUX)
  set(_KLEIDIAI_OPS_ENABLE_BIND_THREADS_DEFAULT ON)
endif()

# Darwin clang needs SME2 enabled when compiling the combined SVE/SME/SME2 source bucket.
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(_KLEIDIAI_OPS_SVE_ARCH_DEFAULT armv9.2-a+nosve+sme2+fp16+dotprod)
endif()

# Configuration options
option(KLEIDIAI_OPS_ENABLE_CCACHE_SUPPORT "Enable use of 'ccache' for build" OFF)
option(KLEIDIAI_OPS_ENABLE_NATIVE "Enable native build" ON)
option(KLEIDIAI_OPS_ENABLE_AARCH32 "Enable AArch32 mode" OFF)
option(KLEIDIAI_OPS_ENABLE_BARE_METAL "Enable bare-metal mode" OFF)
option(KLEIDIAI_OPS_ENABLE_ANDROID "Enable Android build" OFF)
option(KLEIDIAI_OPS_ENABLE_BUILD_BENCHMARK "Enable build of 'gemm' benchmarking executable" ${PROJECT_IS_TOP_LEVEL})
option(KLEIDIAI_OPS_ENABLE_CYCLE_PROFILING "Enable cycle based profiling on the 'new' kernels; reports stats per-thread if threading support is enabled." OFF)

# Benchmarking-specific configuration flags
option(KLEIDIAI_OPS_ENABLE_THREADS "Enable for threading support (not allowed for bare-metal)" ${_KLEIDIAI_OPS_ENABLE_THREADS_DEFAULT})
option(KLEIDIAI_OPS_ENABLE_BIND_THREADS "Enable for threading binding support (Linux-only API)" ${_KLEIDIAI_OPS_ENABLE_BIND_THREADS_DEFAULT})
option(KLEIDIAI_OPS_ENABLE_CYCLE_TIMING "Enable cycle timing on Linux; not compatible with threading support." OFF)
option(KLEIDIAI_OPS_ENABLE_SILENT "Enable minimal output (used for 'xraytools')" OFF)

# The C++ standard
set(KLEIDIAI_OPS_C_STANDARD 11 CACHE STRING "C Standard to use for the library.")
set(KLEIDIAI_OPS_CXX_STANDARD 11 CACHE STRING "CXX Standard to use for the library.")

# Set architecture flags
set(KLEIDIAI_OPS_ARCH armv8-a CACHE STRING "Architecture for the core library.")
set(KLEIDIAI_OPS_FP16_ARCH armv8.2-a+fp16 CACHE STRING "Architecture for the fp16-specific library components.")
set(KLEIDIAI_OPS_SVE_ARCH ${_KLEIDIAI_OPS_SVE_ARCH_DEFAULT} CACHE STRING "Architecture for the sve-specific library components.")

# cmake-format: on
# -------------------------------------------------------------------------------------------------

include(KleidiAIOpsCheckOptions)
include(KleidiAIOpsSetup)

# -------------------------------------------------------------------------------------------------

add_subdirectory(kai/ops)

# -------------------------------------------------------------------------------------------------

if(KLEIDIAI_OPS_ENABLE_BUILD_BENCHMARK)
    add_subdirectory(benchmark)
endif()

# -------------------------------------------------------------------------------------------------

set(KLEIDIAI_OPS_TARGETS_TO_INSTALL kleidiai_ops)
include(KleidiAIOpsInstallProject)

# -------------------------------------------------------------------------------------------------
