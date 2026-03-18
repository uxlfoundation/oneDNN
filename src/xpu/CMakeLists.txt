#===============================================================================
# Copyright 2024 Intel Corporation
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

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )

if(DNNL_WITH_SYCL)
    add_subdirectory(sycl)
    if(DNNL_GPU_VENDOR STREQUAL "INTEL")
        add_subdirectory(ocl)
        add_subdirectory(ze)
    endif()
endif()

if(DNNL_GPU_RUNTIME MATCHES "OCL")
    add_subdirectory(ocl)
endif()

if(DNNL_GPU_RUNTIME MATCHES "ZE")
    add_subdirectory(ze)

    # This is a workaround to avoid moving code from 'ocl/utils.cpp' to
    # 'ocl/utils.hpp' as part of functions are used to query extensions since
    # Level Zero doesn't provide those capabilities fully.
    # TODO: Once it does, remove this and all move functions back to original file.
    list(APPEND SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/ocl/utils_shared.cpp)
endif()

set(OBJ_LIB ${LIB_PACKAGE_NAME}_xpu)
add_library(${OBJ_LIB} OBJECT ${SOURCES})
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)
