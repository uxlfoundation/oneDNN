#===============================================================================
# Copyright 2020-2025 Intel Corporation
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

# Parses kernel names from OpenCL files and updates KER_LIST_EXTERN and
# KER_LIST_ENTRIES variables with the parsed kernel names
function(parse_kernels ker_name ker_path)
    set(entries "${KER_LIST_ENTRIES}")

    file(READ ${ker_path} contents)
    string(REGEX MATCHALL
        "kernel[ \n]+void[ \n]+([a-z0-9_]+)"
        kernels ${contents})
    set(cur_ker_names)
    foreach(k ${kernels})
        string(REGEX REPLACE ".*void[ \n]+" "" k ${k})
        if(ker_name MATCHES "^ref_" AND NOT ${k} MATCHES "^ref_")
            message(FATAL_ERROR "Incorrect OpenCL kernel name: ${k} in ${ker_path}. "
                "All kernels in ref_*.cl files must be prefixed with \"ref_\".")
        endif()
        list(APPEND cur_ker_names ${k})
        list(FIND unique_ker_names ${k} index)
        if (${index} GREATER -1)
            message(WARNING "Kernel name is not unique: ${k}")
        endif()
        set(entries "${entries}\n        { \"${k}\", ${ker_name}_kernel },")
    endforeach()

    set(KER_LIST_EXTERN
        "${KER_LIST_EXTERN}\nextern const char *${ker_name}_kernel;"
        PARENT_SCOPE)
    set(KER_LIST_ENTRIES "${entries}" PARENT_SCOPE)

    set(unique_ker_names "${unique_ker_names};${cur_ker_names}"
        PARENT_SCOPE)
endfunction()

function(gen_gpu_kernel_list ker_list_templ ker_list_src ker_sources headers)
    set(_sources "${SOURCES}")

    set(MINIFY "ON")
    if(DNNL_DEV_MODE OR CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(MINIFY "OFF")
    endif()

    set(KER_LIST_EXTERN)
    set(KER_LIST_ENTRIES)
    set(KER_HEADERS_EXTERN)
    set(KER_HEADERS)
    set(KER_HEADER_NAMES)

    foreach(header_path ${headers})
        get_filename_component(header_file ${header_path} NAME_WE)
        string(REGEX REPLACE ".*\\/src\\/(.*)" "\\1" header_full_path ${header_path})
        get_filename_component(header_dir ${header_full_path} DIRECTORY)
        if (header_dir STREQUAL "gpu\\/intel")
            set(header_name "${header_file}")
        else()
            string(REGEX REPLACE "gpu\\/intel\\/(.*)" "\\1" header_rel_dir ${header_dir})
            string(REGEX REPLACE "\\/" "_" header_name "${header_rel_dir}_${header_file}")
        endif()

        set(gen_file "${CMAKE_CURRENT_BINARY_DIR}/${header_name}_header.cpp")
        add_custom_command(
            OUTPUT ${gen_file}
            COMMAND ${CMAKE_COMMAND}
                -DCL_FILE="${header_path}"
                -DGEN_FILE="${gen_file}"
                -DMINIFY="${MINIFY}"
                -P ${PROJECT_SOURCE_DIR}/cmake/gen_gpu_kernel.cmake
            DEPENDS ${header_path}
        )
        list(APPEND _sources "${gen_file}")
        set(KER_HEADERS_EXTERN
            "${KER_HEADERS_EXTERN}\nextern const char *${header_name}_header;")
        set(KER_HEADER_LIST_ENTRIES
            "${KER_HEADER_LIST_ENTRIES}\n        {\"${header_full_path}\", ${header_name}_header},")
    endforeach()

    set(unique_ker_names)
    foreach(ker_path ${ker_sources})
        get_filename_component(ker_file ${ker_path} NAME_WE)
        string(REGEX REPLACE ".*\\/src\\/(.*)" "\\1" ker_full_path ${ker_path})
        get_filename_component(ker_dir ${ker_full_path} DIRECTORY)
        if (ker_dir STREQUAL "gpu\\/intel")
            set(ker_name "${ker_file}")
        else()
            string(REGEX REPLACE "gpu\\/intel\\/(.*)" "\\1" ker_rel_dir ${ker_dir})
            string(REGEX REPLACE "\\/" "_" ker_name "${ker_rel_dir}_${ker_file}")
        endif()
        set(gen_file "${CMAKE_CURRENT_BINARY_DIR}/${ker_name}_kernel.cpp")
        add_custom_command(
            OUTPUT ${gen_file}
            COMMAND ${CMAKE_COMMAND}
                -DCL_FILE="${ker_path}"
                -DGEN_FILE="${gen_file}"
                -DMINIFY="${MINIFY}"
                -P ${PROJECT_SOURCE_DIR}/cmake/gen_gpu_kernel.cmake
            DEPENDS ${ker_path}
        )
        list(APPEND _sources "${gen_file}")
        parse_kernels(${ker_name} ${ker_path})
    endforeach()

    configure_file("${ker_list_templ}" "${ker_list_src}")
    set(SOURCES "${_sources}" PARENT_SCOPE)
endfunction()
