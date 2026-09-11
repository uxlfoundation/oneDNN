#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# -------------------------------------------------------------------------------------------------

include_guard(GLOBAL)

# Defines the variables GIT_HASH and GIT_BRANCH
find_package(Git)
if(DEFINED GIT_EXECUTABLE AND GIT_EXECUTABLE)
    # Git hash
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse --short HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE GIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

    # Try to get the branch
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" branch --show-current
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

    # ...fall back to a useful ref if detached
    if(GIT_BRANCH STREQUAL "" OR GIT_BRANCH STREQUAL "HEAD")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" describe --tags --always --dirty
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            OUTPUT_VARIABLE GIT_BRANCH
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    endif()
else()
    message(WARNING "Unable to locate 'git'")
    set(GIT_HASH "unknown")
    set(GIT_BRANCH "unknown")
endif()

# -------------------------------------------------------------------------------------------------
