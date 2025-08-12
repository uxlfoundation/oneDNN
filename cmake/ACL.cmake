# ******************************************************************************
# Copyright 2020-2025 Arm Limited and affiliates.
# SPDX-License-Identifier: Apache-2.0
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
# ******************************************************************************

if(acl_cmake_included)
    return()
endif()
set(acl_cmake_included true)
include("cmake/options.cmake")

if(NOT DNNL_TARGET_ARCH STREQUAL "AARCH64")
    return()
endif()

if(NOT DNNL_AARCH64_USE_ACL)
    return()
endif()

find_package(ACL REQUIRED)

# Required. The minimum compatible major-version as per Semantic Versioning.
set(ACL_MIN_MAJOR_VERSION "52")
set(ACL_MIN_MINOR_VERSION "2")
set(ACL_MIN_VERSION "${ACL_MIN_MAJOR_VERSION}.${ACL_MIN_MINOR_VERSION}")

# Optional. Maximum known compatible version if any.
# Set to an empty-string if none.
set(ACL_MAX_MAJOR_VERSION "")

if(ACL_FOUND)
    file(GLOB_RECURSE ACL_FOUND_VERSION_FILE ${ACL_INCLUDE_DIR}/*/arm_compute_version.embed)
    if ("${ACL_FOUND_VERSION_FILE}" STREQUAL "")
        message(WARNING
            "Build may fail. Could not determine ACL version.\n"
            "File 'arm_compute_version.embed' not found in ${ACL_INCLUDE_DIR}/**\n"
            "Minimum compatible ACL version is ${ACL_MIN_MAJOR_VERSION}\n"
        )
    else()
        file(READ ${ACL_FOUND_VERSION_FILE} ACL_FOUND_VERSION_STRING)

        if("${ACL_FOUND_VERSION_STRING}" MATCHES "arm_compute_version=v([0-9]+)\\.([0-9]+)\\.?([0-9]*)")
            set(ACL_FOUND_MAJOR_VERSION "${CMAKE_MATCH_1}")
            set(ACL_FOUND_MINOR_VERSION "${CMAKE_MATCH_2}")
            set(ACL_FOUND_VERSION "${ACL_FOUND_MAJOR_VERSION}.${ACL_FOUND_MINOR_VERSION}")

            if ("${ACL_FOUND_VERSION}" VERSION_EQUAL "0.0")
                # Unreleased ACL versions come with version string "v0.0-unreleased", and may not be compatible with oneDNN.
                # It is recommended to use a supported major-version of ACL.
                message(WARNING
                    "Build may fail. Using an unreleased ACL version.\n"
                    "Minimum compatible ACL version is ${ACL_MIN_VERSION}\n"
                )
            elseif("${ACL_FOUND_VERSION}" VERSION_LESS "${ACL_MIN_VERSION}")
                message(FATAL_ERROR
                    "Detected ACL version ${ACL_FOUND_VERSION}, but minimum "
                    "compatible is ${ACL_MIN_VERSION}\n"
                )
            elseif("${ACL_FOUND_MAJOR_VERSION}" GREATER "${ACL_MIN_MAJOR_VERSION}")
                # This is not necessarily an error. Need to check if there is a
                # known incompatible maximum version:
                if("${ACL_MAX_MAJOR_VERSION}" STREQUAL "")
                    message(WARNING
                        "Build may fail. Using a newer ACL major version than officially supported.\n"
                        "Detected ACL major version ${ACL_FOUND_MAJOR_VERSION}, but "
                        "supported major version is ${ACL_MIN_MAJOR_VERSION}\n"
                    )
                else()
                    if("${ACL_FOUND_MAJOR_VERSION}" GREATER "${ACL_MAX_MAJOR_VERSION}")
                    message(FATAL_ERROR
                        "Detected ACL version ${ACL_FOUND_MAJOR_VERSION}, but maximum "
                        "compatible version is ${ACL_MAX_MAJOR_VERSION}\n"
                    )
                    endif()
                endif()
            endif()
        else()
            message(WARNING
                "Build may fail. Could not determine ACL version.\n"
                "Unexpected version string format in ${ACL_FOUND_VERSION_FILE}.\n"
            )
        endif()
    endif()

    list(APPEND EXTRA_SHARED_LIBS ${ACL_LIBRARIES})

    include_directories(${ACL_INCLUDE_DIRS})

    message(STATUS "Arm Compute Library: ${ACL_LIBRARIES}")
    message(STATUS "Arm Compute Library headers: ${ACL_INCLUDE_DIRS}")

    add_definitions(-DDNNL_AARCH64_USE_ACL)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_EXTENSIONS "OFF")
endif()
