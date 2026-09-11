#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# -------------------------------------------------------------------------------------------------

# If this is the top-level project and the installation directory hasn't been set, set it to be
# <project-root>/install/
if(PROJECT_IS_TOP_LEVEL)
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX
            "${CMAKE_BINARY_DIR}/install"
            CACHE PATH "Install path" FORCE)
        message(
            STATUS "CMAKE_INSTALL_PREFIX was not set, assigning default: ${CMAKE_INSTALL_PREFIX}")
    endif()
endif()

# -------------------------------------------------------------------------------------------------

# Introduce CMAKE_INSTALL_BINDIR, CMAKE_INSTALL_LIBDIR, CMAKE_INSTALL_INCLUDEDIR variables
include(GNUInstallDirs)
set(KLEIDIAI_OPS_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})
set(KLEIDIAI_OPS_CONFIG_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
set(KLEIDIAI_OPS_LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
set(KLEIDIAI_OPS_BIN_INSTALL_DIR ${CMAKE_INSTALL_BINDIR})

# -------------------------------------------------------------------------------------------------

# Preserve MTE support when parent builds pass -fsanitize=memtag.
if((CMAKE_C_FLAGS MATCHES "(^| )-fsanitize=[^ ]*memtag" OR CMAKE_CXX_FLAGS MATCHES
                                                           "(^| )-fsanitize=[^ ]*memtag")
   AND NOT KLEIDIAI_OPS_ARCH MATCHES "\\+memtag")
    string(APPEND KLEIDIAI_OPS_ARCH +memtag)
    string(APPEND KLEIDIAI_OPS_FP16_ARCH +memtag)
    string(APPEND KLEIDIAI_OPS_SVE_ARCH +memtag)
endif()

# Add -march to architecture values
string(PREPEND KLEIDIAI_OPS_ARCH -march=)
string(PREPEND KLEIDIAI_OPS_FP16_ARCH -march=)
string(PREPEND KLEIDIAI_OPS_SVE_ARCH -march=)

# Set the compiler launcher for ccache support
if(KLEIDIAI_OPS_ENABLE_CCACHE_SUPPORT)
    find_program(CCACHE_PROGRAM ccache)
    if(NOT CCACHE_PROGRAM)
        message(FATAL_ERROR "Enabled 'ccache' support but could not find ccache.")
    endif()
    set(CMAKE_C_COMPILER_LAUNCHER
        "${CCACHE_PROGRAM}"
        CACHE STRING "C compiler launcher")
    set(CMAKE_CXX_COMPILER_LAUNCHER
        "${CCACHE_PROGRAM}"
        CACHE STRING "CXX compiler launcher")
endif()

# -------------------------------------------------------------------------------------------------

# For multi-config generators, we only do Debug and Release builds. Otherwise default to Release if
# a build type hasn't been specified
if(CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_CONFIGURATION_TYPES
        "Release;Debug"
        CACHE STRING "Allowed project configurations" FORCE)
elseif(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "CMAKE_BUILD_TYPE not set -- assuming Release")
endif()

# -------------------------------------------------------------------------------------------------

if(MSVC)
    # We don't currently provide support for MSVC, but when we do, we likely want to use /Wall
    set(KLEIDIAI_OPS_WARNING_FLAGS_BASE "/Wall")
else()
    set(KLEIDIAI_OPS_WARNING_FLAGS_BASE
        "-Wall"
        "-Wdisabled-optimization"
        "-Wextra"
        "-Wformat-security"
        "-Wformat=2"
        "-Winit-self"
        "-Wstrict-overflow=2"
        "-Wswitch-default"
        "-Wcast-qual"
    )

    # C only flags not present in C++
    set(KLEIDIAI_OPS_WARNING_FLAGS_C
        "-Wmissing-prototypes"
        "-Wstrict-prototypes"
    )

    set(KLEIDIAI_OPS_WARNING_FLAGS_CXX
        "-Wctor-dtor-privacy"
        "-Woverloaded-virtual"
        "-Wsign-promo"
        "-Wno-missing-declarations"
        "-Wno-unused-parameter"
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND KLEIDIAI_OPS_WARNING_FLAGS_CXX "-Wno-unused-private-field")
    endif()
endif()

# Warning flags
set(KLEIDIAI_OPS_WARNING_FLAGS
    ${KLEIDIAI_OPS_WARNING_FLAGS_BASE} $<$<COMPILE_LANGUAGE:C>:${KLEIDIAI_OPS_WARNING_FLAGS_C}>
    $<$<COMPILE_LANGUAGE:CXX>:${KLEIDIAI_OPS_WARNING_FLAGS_CXX}>)

# C/C++ flags
set(KLEIDIAI_OPS_CCXX_FLAGS
    ${KLEIDIAI_OPS_CCXX_FLAGS_INIT}
    $<IF:$<CONFIG:Debug>,${KLEIDIAI_OPS_CCXX_FLAGS_DEBUG},${KLEIDIAI_OPS_CCXX_FLAGS_RELEASE}>
    ${KLEIDIAI_OPS_WARNING_FLAGS}
    CACHE STRING "")

# Linker flags
set(KLEIDIAI_OPS_LINKER_FLAGS
    ${KLEIDIAI_OPS_LINKER_FLAGS_INIT}
    $<IF:$<CONFIG:Debug>,${KLEIDIAI_OPS_LINKER_FLAGS_DEBUG},${KLEIDIAI_OPS_LINKER_FLAGS_RELEASE}>
    CACHE STRING "")

# Compile definitions
set(KLEIDIAI_OPS_COMPILE_DEFINES
    $<$<CONFIG:Release>:NDEBUG> $<$<BOOL:${KLEIDIAI_OPS_ENABLE_SILENT}>:SILENT> ENABLE_FP16_KERNELS)

# Bare-metal vs Linux threading compile definitions
if(KLEIDIAI_OPS_ENABLE_BARE_METAL)
    list(
        APPEND
        KLEIDIAI_OPS_COMPILE_DEFINES
        BARE_METAL
        CYCLE_TIMING
        NO_MULTI_THREADING)
    if(KLEIDIAI_OPS_ENABLE_CYCLE_PROFILING)
        list(APPEND KLEIDIAI_OPS_COMPILE_DEFINES CYCLE_PROFILING)
    endif()
elseif(KLEIDIAI_OPS_ENABLE_THREADS)
    # Linux version threaded
    list(APPEND KLEIDIAI_OPS_COMPILE_DEFINES THREADS)
    if(KLEIDIAI_OPS_ENABLE_BIND_THREADS)
        list(APPEND KLEIDIAI_OPS_COMPILE_DEFINES BIND_THREADS)
    endif()
    if(KLEIDIAI_OPS_ENABLE_CYCLE_PROFILING)
        list(APPEND KLEIDIAI_OPS_COMPILE_DEFINES CYCLE_PROFILING)
    endif()
else()
    # Linux version non-threaded
    list(APPEND KLEIDIAI_OPS_COMPILE_DEFINES NO_MULTI_THREADING)
    if(KLEIDIAI_OPS_ENABLE_CYCLE_TIMING)
        list(APPEND KLEIDIAI_OPS_COMPILE_DEFINES CYCLE_TIMING)
    elseif(KLEIDIAI_OPS_ENABLE_CYCLE_PROFILING)
        list(APPEND KLEIDIAI_OPS_COMPILE_DEFINES CYCLE_PROFILING)
    endif()
endif()

# -------------------------------------------------------------------------------------------------
