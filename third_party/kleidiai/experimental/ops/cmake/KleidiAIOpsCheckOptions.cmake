#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# -------------------------------------------------------------------------------------------------

# TODO:
#
# * Add sanity checks for combinations of flags, e.g. enable Android and bare metal?
# * Add sanity checks for C/C++ standard

# Cycle-timing only works in non-threaded mode and can't be used with cycle profiling
if(KLEIDIAI_OPS_ENABLE_CYCLE_TIMING)
    if(KLEIDIAI_OPS_ENABLE_THREADS)
        message(FATAL_ERROR "Cannot enable cycle timing with threads enabled")
    endif()
    if(KLEIDIAI_OPS_ENABLE_CYCLE_PROFILING)
        message(FATAL_ERROR "Cannot enable cycle profiling with cycle timing enabled")
    endif()
endif()

# Thread binding requires threads
if(KLEIDIAI_OPS_ENABLE_BIND_THREADS AND NOT KLEIDIAI_OPS_ENABLE_THREADS)
    message(FATAL_ERROR "Cannot enable thread binding without enabling threads")
endif()

# Only allow cycle profiling if this is the top-level project
if(KLEIDIAI_OPS_ENABLE_CYCLE_PROFILING)
    if(PROJECT_IS_TOP_LEVEL)
        message(
            STATUS
                "Cycle profiling enabled (this feature should only be enabled by ${PROJECT_NAME} developers)"
        )
    else()
        message(FATAL_ERROR "Can't enable cycle profiling when project is not top-level")
    endif()
endif()

# Helper function to ensure the given flags are only enabled on certain platforms
function(allow_flags_only_on_platforms PLATFORM_REGEX)
    foreach(FLAG IN LISTS ARGN)
        if(${FLAG} AND NOT CMAKE_SYSTEM_NAME MATCHES "${PLATFORM_REGEX}")
            message(FATAL_ERROR "Can only enable ${FLAG} on systems matching '${PLATFORM_REGEX}'. "
                                "You're on ${CMAKE_SYSTEM_NAME}.")
        endif()
    endforeach()
endfunction()

# Helper function to ensure the given flags are NOT enabled on certain platforms
function(disallow_flags_on_platforms PLATFORM_REGEX)
    foreach(FLAG IN LISTS ARGN)
        if(${FLAG} AND CMAKE_SYSTEM_NAME MATCHES "${PLATFORM_REGEX}")
            message(FATAL_ERROR "Cannot enable ${FLAG} on systems matching '${PLATFORM_REGEX}'. "
                                "You're on ${CMAKE_SYSTEM_NAME}.")
        endif()
    endforeach()
endfunction()

# We'll only disallow threading on bare-metal for now
disallow_flags_on_platforms("Generic" KLEIDIAI_OPS_ENABLE_THREADS)

# Thread binding uses a Linux-specific API, so it only works on Linux/Android
allow_flags_only_on_platforms("Linux|Android" KLEIDIAI_OPS_ENABLE_BIND_THREADS)

# Cycle timing/profiling is only supported on Linux
allow_flags_only_on_platforms("Linux" KLEIDIAI_OPS_ENABLE_CYCLE_TIMING)

# -------------------------------------------------------------------------------------------------
