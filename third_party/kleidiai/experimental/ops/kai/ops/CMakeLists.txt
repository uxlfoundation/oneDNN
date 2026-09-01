#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# -------------------------------------------------------------------------------------------------
message(STATUS "Entered kleidiai_ops directory")
list(APPEND CMAKE_MESSAGE_INDENT " ")

add_library(kleidiai_ops STATIC)

add_subdirectory(conv)
add_subdirectory(gemm)

set(ARM_CONV_PUBLIC_INCLUDES $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
                             $<INSTALL_INTERFACE:${KLEIDIAI_OPS_INCLUDE_INSTALL_DIR}>)

target_include_directories(kleidiai_ops PUBLIC ${ARM_CONV_PUBLIC_INCLUDES})
target_compile_definitions(kleidiai_ops PRIVATE ${KLEIDIAI_OPS_COMPILE_DEFINES})
target_compile_options(kleidiai_ops PRIVATE ${KLEIDIAI_OPS_CCXX_FLAGS})
target_link_options(kleidiai_ops PUBLIC ${KLEIDIAI_OPS_LINKER_FLAGS})

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
