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

# Locates Doxyrest and configures documentation translation from .xml to .rst
#===============================================================================

if(Doxyrest_cmake_included)
    return()
endif()
set(Doxyrest_cmake_included true)

find_package(Doxyrest)
if (DOXYREST_FOUND)
    set(DOXYREST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/reference)
    set(DOXYREST_STAMP_FILE ${CMAKE_CURRENT_BINARY_DIR}/doxyrest.stamp)
    set(DOXYREST_FRAME_DIR ${CMAKE_CURRENT_BINARY_DIR}/frame)
    file(GLOB_RECURSE DOXYREST_CONFIGS
        ${CMAKE_CURRENT_SOURCE_DIR}/doc/doxyrest/doxyrest-config.lua
        )
    file(COPY ${DOXYREST_CONFIGS}
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
        )
    file(MAKE_DIRECTORY ${DOXYREST_FRAME_DIR})

    # Adding doc/**/*.rst files to Doxyrest output and checking for name collisions
    file(GLOB_RECURSE DOXYREST_RST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/doc/*.rst)
    set(_seen_rst_names "")
    set(_dup_rst_names "")
    foreach(_rst ${DOXYREST_RST_FILES})
        get_filename_component(_rst_name ${_rst} NAME)
        if(_rst_name IN_LIST _seen_rst_names)
            list(APPEND _dup_rst_names ${_rst_name})
        else()
            list(APPEND _seen_rst_names ${_rst_name})
        endif()
    endforeach()
    if(_dup_rst_names)
        list(REMOVE_DUPLICATES _dup_rst_names)
        message(FATAL_ERROR
            "Duplicate documentation .rst file name(s) under doc/: "
            "${_dup_rst_names}. Grouping .rst files are flattened into a single "
            "Sphinx source directory and must have unique file names.")
    endif()
    add_custom_command(
        OUTPUT ${DOXYREST_STAMP_FILE}
        DEPENDS ${DOXYGEN_STAMP_FILE} ${DOXYREST_CONFIGS} ${DOXYREST_RST_FILES}
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/doc/doxyrest/frame
            ${DOXYREST_FRAME_DIR}
        COMMAND ${DOXYREST_EXECUTABLE}
            --frame-dir=${DOXYREST_FRAME_DIR}/common
            --frame-dir=${DOXYREST_FRAME_DIR}/cfamily
            --config=${CMAKE_CURRENT_BINARY_DIR}/doxyrest-config.lua
        COMMAND ${CMAKE_COMMAND} -E copy ${DOXYREST_RST_FILES} ${DOXYREST_OUTPUT_DIR}/rst
        COMMAND ${CMAKE_COMMAND} -E touch ${DOXYREST_STAMP_FILE}
        WORKING_DIRECTORY ${DOXYREST_OUTPUT_DIR}
        COMMENT "Translating documentation from .xml to .rst with Doxyrest" VERBATIM)
    add_custom_target(doc_doxyrest DEPENDS ${DOXYREST_STAMP_FILE} doc_doxygen)
endif(DOXYREST_FOUND)

