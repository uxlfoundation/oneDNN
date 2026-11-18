#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# Common assembly queries used by the GEMM and convolution drivers.
# A separate object target keeps source properties in the directory that compiles
# these files, as required by CMake's source-property scoping rules.

# Common ASM sources
set(KLEIDIAI_OPS_COMMON_SOURCES_ASM "${CMAKE_CURRENT_LIST_DIR}/kai_ops_common_asm.S")

if(MSVC)
    set_source_files_properties(${KLEIDIAI_OPS_COMMON_SOURCES_ASM} PROPERTIES LANGUAGE ASM_MARMASM)
endif()

# Compile the queries in their own object library.
add_library(kleidiai_ops_common OBJECT ${KLEIDIAI_OPS_COMMON_SOURCES_ASM})
target_compile_definitions(kleidiai_ops_common PRIVATE ${KLEIDIAI_OPS_COMPILE_DEFINES})
if(NOT MSVC)
    target_compile_options(kleidiai_ops_common PRIVATE ${KLEIDIAI_OPS_SVE_ARCH})
endif()
set_target_properties(kleidiai_ops_common PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Assembler architecture settings do not change the caller's CPU requirements.
# Callers must still check SVE or SME support before executing these queries.
# Add their compiled objects to the final kleidiai_ops library.
target_sources(kleidiai_ops PRIVATE $<TARGET_OBJECTS:kleidiai_ops_common>)
