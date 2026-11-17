#===============================================================================
# Copyright 2021 Intel Corporation
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

# Controls enabling primitive options and dispatcher to internal logic
#===============================================================================

if(enable_primitives_cmake_included)
    return()
endif()
set(enable_primitives_cmake_included true)
include("cmake/options.cmake")

set(BUILD_${DNNL_ENABLE_WORKLOAD} TRUE)
message(STATUS "Enabled workload: ${DNNL_ENABLE_WORKLOAD}")

if(DNNL_ENABLE_PRIMITIVE STREQUAL "ALL")
    set(BUILD_PRIMITIVE_ALL TRUE)
else()
    foreach(impl ${DNNL_ENABLE_PRIMITIVE})
        string(TOUPPER ${impl} uimpl)
        if(NOT "${uimpl}" MATCHES
                "^(BATCH_NORMALIZATION|BINARY|CONCAT|CONVOLUTION|DECONVOLUTION|ELTWISE|GROUP_NORMALIZATION|INNER_PRODUCT|LAYER_NORMALIZATION|LRN|MATMUL|POOLING|PRELU|REDUCTION|REORDER|RESAMPLING|RNN|SDPA|SHUFFLE|SOFTMAX|SUM)$")
            message(FATAL_ERROR "Unsupported primitive: ${uimpl}")
        endif()
        set(BUILD_${uimpl} TRUE)
    endforeach()
endif()
message(STATUS "Enabled primitives: ${DNNL_ENABLE_PRIMITIVE}")

if (DNNL_ENABLE_PRIMITIVE_CPU_ISA STREQUAL "ALL")
    set(BUILD_PRIMITIVE_CPU_ISA_ALL TRUE)
else()
    foreach(isa ${DNNL_ENABLE_PRIMITIVE_CPU_ISA})
        string(TOUPPER ${isa} uisa)
        if("${uisa}" MATCHES "^(SSE41|AVX)$")
            message(WARNING "Optimizations for CPU ISA ${uisa} are not available")
        elseif(NOT "${uisa}" MATCHES "^(AVX2|AVX512|AMX)$")
            message(FATAL_ERROR "Unsupported primitive CPU ISA: ${uisa}")
        else()
            set(BUILD_${uisa} TRUE)
        endif()
    endforeach()
endif()
message(STATUS "Enabled primitive CPU ISA: ${DNNL_ENABLE_PRIMITIVE_CPU_ISA}")

if (DNNL_ENABLE_PRIMITIVE_GPU_ISA STREQUAL "ALL")
    set(BUILD_PRIMITIVE_GPU_ISA_ALL TRUE)
else()
    foreach(isa ${DNNL_ENABLE_PRIMITIVE_GPU_ISA})
        string(TOUPPER ${isa} uisa)
        if("${uisa}" MATCHES "^(XEHPG|XEHPC|XE2|XE3|XE3P)$")
            set(BUILD_${uisa} TRUE)
        elseif("${uisa}" MATCHES "^(XELP|XEHP)$")
            message(WARNING "Optimizations for GPU ISA ${uisa} are not available")
        else()
            message(FATAL_ERROR "Unsupported primitive GPU ISA: ${uisa}")
        endif()

    endforeach()
endif()
message(STATUS "Enabled primitive GPU ISA: ${DNNL_ENABLE_PRIMITIVE_GPU_ISA}")

if (ONEDNN_ENABLE_GEMM_KERNELS_ISA STREQUAL "ALL")
    set(BUILD_GEMM_KERNELS_ALL TRUE)
elseif (ONEDNN_ENABLE_GEMM_KERNELS_ISA STREQUAL "NONE")
    set(BUILD_GEMM_KERNELS_NONE TRUE)
else()
    foreach(isa ${ONEDNN_ENABLE_GEMM_KERNELS_ISA})
        string(TOUPPER ${isa} uisa)
        if("${uisa}" MATCHES "^(SSE41|AVX)$")
            message(WARNING "Optimizations for CPU ISA ${uisa} are not available")
        elseif(NOT "${uisa}" MATCHES "^(AVX2|AVX512)$")
            message(FATAL_ERROR "Unsupported primitive CPU ISA: ${uisa}")
        else()
            set(BUILD_GEMM_${uisa} TRUE)
        endif()
    endforeach()
endif()
message(STATUS "Enabled GeMM kernels ISA: ${ONEDNN_ENABLE_GEMM_KERNELS_ISA}")

# When certain primitives, primitive ISA or workload are switched off, some
# functions may become unused which is expected. Switch off warning for unused
# functions in such cases. Implementations filtered out of the implementation
# lists are still compiled but no longer referenced, let the linker drop them.
if (NOT DNNL_ENABLE_PRIMITIVE STREQUAL "ALL" OR
        NOT DNNL_ENABLE_PRIMITIVE_CPU_ISA STREQUAL "ALL" OR
        NOT DNNL_ENABLE_PRIMITIVE_GPU_ISA STREQUAL "ALL" OR
        NOT DNNL_ENABLE_WORKLOAD STREQUAL "TRAINING")
    # CMAKE_CCXX_FLAGS is already applied by platform.cmake at this point.
    if(NOT MSVC)
        append(CMAKE_C_FLAGS "-Wno-error=unused-function -Wno-unused-function")
        append(CMAKE_CXX_FLAGS
                "-Wno-error=unused-function -Wno-unused-function")
    endif()
    if(NOT WIN32 AND NOT APPLE)
        append(CMAKE_C_FLAGS "-ffunction-sections -fdata-sections")
        append(CMAKE_CXX_FLAGS "-ffunction-sections -fdata-sections")
        append(CMAKE_SHARED_LINKER_FLAGS "-Wl,--gc-sections")
        append(CMAKE_EXE_LINKER_FLAGS "-Wl,--gc-sections")
    endif()
endif()
