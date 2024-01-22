#===============================================================================
# Copyright 2019-2024 Intel Corporation
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

set(TARGET_NAME ${LIB_PACKAGE_NAME}_sycl)
file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )

include_directories_with_host_compiler(${CMAKE_CURRENT_SOURCE_DIR})

  
if(DNNL_CPU_RUNTIME STREQUAL "NONE")
    list(REMOVE_ITEM SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/sycl_stream_cpu_thunk.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/sycl_stream_submit_cpu_primitive.cpp"
            )
endif()

if(NOT DNNL_GPU_RUNTIME STREQUAL "NONE" AND DNNL_WITH_XE3P)
add_definitions_with_host_compiler(-DDNNL_WITH_XE3P)
add_definitions_with_host_compiler(-DPRERELEASE_HW)
# XE3 macro is required to compile nGEN with XE3P support.
add_definitions_with_host_compiler(-DXE3)
add_definitions_with_host_compiler(-DXE3P)
endif()

set(OBJ_LIB ${LIB_PACKAGE_NAME}_sycl)
add_library(${OBJ_LIB} OBJECT ${SOURCES})

set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)
  
