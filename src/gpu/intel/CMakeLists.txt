#===============================================================================
# Copyright 2024-2025 Intel Corporation
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

# TODO: Extend to all -Wconversion warnings
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" OR CMAKE_CXX_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
    append(CMAKE_CXX_FLAGS "-Wbitfield-enum-conversion -Wbool-conversion -Wconstant-conversion -Wenum-conversion -Wimplicit-int-conversion -Wliteral-conversion -Wnon-literal-null-conversion -Wnull-conversion")
    add_definitions("-DENABLE_LLVM_WCONVERSION")
endif()

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]pp)
set(DIRS "bnorm;gemm;reduction;rnn")
foreach(dir ${DIRS})
    file(GLOB_RECURSE SOURCES_EXTRA
        ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/*.[ch]pp
        )
    list(APPEND SOURCES "${SOURCES_EXTRA}")
endforeach()

include("${PROJECT_SOURCE_DIR}/cmake/gen_gpu_kernel_list.cmake")

file(GLOB_RECURSE CL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cl)
file(GLOB_RECURSE CL_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

set(kernel_list_templ "${PROJECT_SOURCE_DIR}/src/gpu/intel/ocl_kernel_list.cpp.in")
set(kernel_list_src "${PROJECT_BINARY_DIR}/src/gpu/intel/ocl_kernel_list.cpp")

gen_gpu_kernel_list(
    "${kernel_list_templ}"
    "${kernel_list_src}"
    "${CL_SOURCES}"
    "${CL_HEADERS}")

list(APPEND SOURCES ${kernel_list_src})

add_subdirectory(compute)
add_subdirectory(microkernels)
add_subdirectory(jit)

if(DNNL_GPU_RUNTIME STREQUAL "OCL")
    add_subdirectory(ocl)
elseif(DNNL_WITH_SYCL)
    add_subdirectory(sycl)
    add_subdirectory(ocl)
endif()

set(OBJ_LIB ${LIB_PACKAGE_NAME}_gpu_intel)
add_library(${OBJ_LIB} OBJECT ${SOURCES})
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)
