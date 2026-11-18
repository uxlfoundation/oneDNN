#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# -------------------------------------------------------------------------------------------------
message(STATUS "Entered conv directory")
list(APPEND CMAKE_MESSAGE_INDENT " ")

# -------------------------------------------------------------------------------------------------

set(KLEIDIAI_OPS_CONV_WINOGRAD_CORE_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/winograd/input_transforms/a64_fp32_6x6.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/input_transforms/arm_fp32_1x8.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/input_transforms/arm_fp32_4x4.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/input_transforms/arm_fp32_6x6.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/input_transforms_fp32.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/output_transforms/arm_fp32_1x2_1x7.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/output_transforms/arm_fp32_1x4_1x5.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/output_transforms/arm_fp32_1x6_1x3.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/output_transforms/arm_fp32_2x2_3x3.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/output_transforms/arm_fp32_2x2_5x5.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/output_transforms/arm_fp32_4x4_3x3.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/output_transforms_fp32.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/weight_transforms/arm_fp32_2x2_3x3.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/weight_transforms/arm_fp32_2x2_5x5.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/weight_transforms/arm_fp32_4x4_3x3.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/weight_transforms/cpp_fp32_1x2_1x7.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/weight_transforms/cpp_fp32_1x4_1x5.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/weight_transforms/cpp_fp32_1x6_1x3.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/weight_transforms_fp32.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/winograd_fp32.cpp)

set(KLEIDIAI_OPS_CONV_WINOGRAD_SVE_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/winograd/input_transforms/sme_fp32_mla_6x6.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/input_transforms/sve_fp32_6x6.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/output_transforms/sme_fp32_mopa_4x4_3x3.cpp)

set(KLEIDIAI_OPS_CONV_WINOGRAD_FP16_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/winograd/input_transforms/a64_fp16_6x6.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/input_transforms_fp16.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/output_transforms/a64_fp16_4x4_3x3.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/output_transforms_fp16.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/weight_transforms/a64_fp16_4x4_3x3.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/weight_transforms_fp16.cpp
    ${CMAKE_CURRENT_LIST_DIR}/winograd/winograd_fp16.cpp)

# -------------------------------------------------------------------------------------------------

# Add extracted kernel .S files to this shared assembly list. All assembly
# sources use the combined SVE architecture settings; enabling these extensions
# does not insert additional instructions into functions that do not use them.
set(KLEIDIAI_OPS_CONV_SOURCES_ASM)

# -------------------------------------------------------------------------------------------------

set(KLEIDIAI_OPS_CONV_PRIVATE_INCLUDES
    # Internal sources
    ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../
    # Public headers
    ${CMAKE_CURRENT_LIST_DIR}/../../../include)

# -------------------------------------------------------------------------------------------------

# Helper to create an OBJECT lib, apply common compile options/compile definitions/include
# directories, and arch
function(arm_add_obj_lib OBJ_NAME)
    # Parse flags
    set(PREFIX ARG)
    set(OPTIONS)
    set(SINGLE_VALUE_ARGS ARCH)
    set(MULTI_VALUE_ARGS SOURCES)
    cmake_parse_arguments(
        ${PREFIX}
        "${OPTIONS}"
        "${SINGLE_VALUE_ARGS}"
        "${MULTI_VALUE_ARGS}"
        ${ARGN})

    # Sanity check
    if(NOT ${PREFIX}_SOURCES)
        message(FATAL_ERROR "No SOURCES provided for '${OBJ_NAME}'")
    endif()

    # Define the object library
    add_library(${OBJ_NAME} OBJECT ${${PREFIX}_SOURCES})
    target_include_directories(${OBJ_NAME} PRIVATE ${KLEIDIAI_OPS_CONV_PRIVATE_INCLUDES})
    target_compile_definitions(${OBJ_NAME} PRIVATE ${KLEIDIAI_OPS_COMPILE_DEFINES})
    # Keep C/C++ options scoped to these languages. Extracted assembly is compiled
    # separately with the combined SVE architecture settings below.
    target_compile_options(
        ${OBJ_NAME} PRIVATE
        "$<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:${KLEIDIAI_OPS_CCXX_FLAGS}>")

    if(${PREFIX}_ARCH)
        target_compile_options(${OBJ_NAME} PRIVATE ${${PREFIX}_ARCH})
    else()
        message(STATUS "Architecture flag unset for '${OBJ_NAME}'")
    endif()

    set_target_properties(
        ${OBJ_NAME}
        PROPERTIES C_STANDARD ${KLEIDIAI_OPS_C_STANDARD}
                   C_STANDARD_REQUIRED ON
                   C_EXTENSIONS OFF
                   CXX_STANDARD ${KLEIDIAI_OPS_CXX_STANDARD}
                   CXX_STANDARD_REQUIRED ON
                   CXX_EXTENSIONS OFF
                   POSITION_INDEPENDENT_CODE ON)
endfunction()

# -------------------------------------------------------------------------------------------------

# Winograd sources; non-SVE, non-FP16
arm_add_obj_lib(
    conv_winograd_core
    SOURCES ${KLEIDIAI_OPS_CONV_WINOGRAD_CORE_SOURCES}
    ARCH ${KLEIDIAI_OPS_ARCH})

# Winograd sources; AArch64-only buckets
arm_add_obj_lib(
    conv_winograd_sve
    SOURCES ${KLEIDIAI_OPS_CONV_WINOGRAD_SVE_SOURCES}
    ARCH ${KLEIDIAI_OPS_SVE_ARCH})

arm_add_obj_lib(
    conv_winograd_fp16
    SOURCES ${KLEIDIAI_OPS_CONV_WINOGRAD_FP16_SOURCES}
    ARCH ${KLEIDIAI_OPS_FP16_ARCH})

# -------------------------------------------------------------------------------------------------

# Source properties are set beside the target that compiles these files.
# No object library is needed until this directory contains assembly sources.
if(KLEIDIAI_OPS_CONV_SOURCES_ASM)
    if(MSVC)
        set_source_files_properties(${KLEIDIAI_OPS_CONV_SOURCES_ASM} PROPERTIES LANGUAGE ASM_MARMASM)
    endif()

    add_library(conv_asm OBJECT ${KLEIDIAI_OPS_CONV_SOURCES_ASM})
    target_include_directories(conv_asm PRIVATE ${KLEIDIAI_OPS_CONV_PRIVATE_INCLUDES})
    target_compile_definitions(conv_asm PRIVATE ${KLEIDIAI_OPS_COMPILE_DEFINES})
    if(NOT MSVC)
        target_compile_options(conv_asm PRIVATE ${KLEIDIAI_OPS_SVE_ARCH})
    endif()
    set_target_properties(conv_asm PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_sources(kleidiai_ops PRIVATE $<TARGET_OBJECTS:conv_asm>)
endif()

target_sources(
    kleidiai_ops
    PRIVATE $<TARGET_OBJECTS:conv_winograd_core>
            $<$<TARGET_EXISTS:conv_winograd_sve>:$<TARGET_OBJECTS:conv_winograd_sve>>
            $<$<TARGET_EXISTS:conv_winograd_fp16>:$<TARGET_OBJECTS:conv_winograd_fp16>>)

message(STATUS "Leaving conv directory")
# -------------------------------------------------------------------------------------------------
