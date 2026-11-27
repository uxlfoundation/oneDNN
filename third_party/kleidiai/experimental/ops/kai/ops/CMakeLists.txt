#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# -------------------------------------------------------------------------------------------------
message(STATUS "Entered kleidiai_ops directory")
list(APPEND CMAKE_MESSAGE_INDENT " ")

add_library(kleidiai_ops STATIC)

# Each subdirectory compiles its sources into object libraries and adds their
# objects to kleidiai_ops. The final library's public interface is defined below.
add_subdirectory(common_internal)
# Can't handle depthwise/winograd yet with MSVC as they still have inline assembly
if(NOT MSVC)
    add_subdirectory(conv)
endif()
add_subdirectory(gemm)

# Public include directories for the kleidiai_ops library. Handle both build and install interfaces
set(KLEIDIAI_OPS_PUBLIC_INCLUDES $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
                                 $<INSTALL_INTERFACE:${KLEIDIAI_OPS_INCLUDE_INSTALL_DIR}>)

# Set include directories, compile definitions, and compile options for the kleidiai_ops library
target_include_directories(kleidiai_ops PUBLIC ${KLEIDIAI_OPS_PUBLIC_INCLUDES})
target_compile_definitions(kleidiai_ops PRIVATE ${KLEIDIAI_OPS_COMPILE_DEFINES})
target_compile_options(
    kleidiai_ops
    PRIVATE "$<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:${KLEIDIAI_OPS_CCXX_FLAGS}>")
target_link_options(kleidiai_ops PUBLIC ${KLEIDIAI_OPS_LINKER_FLAGS})

# Set target properties for the kleidiai_ops library
set_target_properties(
    kleidiai_ops
    PROPERTIES C_STANDARD ${KLEIDIAI_OPS_C_STANDARD}
               C_STANDARD_REQUIRED ON
               C_EXTENSIONS OFF
               CXX_STANDARD ${KLEIDIAI_OPS_CXX_STANDARD}
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF
               POSITION_INDEPENDENT_CODE ON)

add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS kleidiai_ops)

message(STATUS "Leaving kleidiai_ops directory")
# -------------------------------------------------------------------------------------------------
