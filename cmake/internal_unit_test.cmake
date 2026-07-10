#===============================================================================
# Copyright 2026 Intel Corporation
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

# Infrastructure for "next-to-source" internal unit tests.
#
# Motivation
# ----------
# A file named `*_test.cpp` placed *next to* the sources it exercises is
# compiled into its own gtest executable. New test files are discovered
# automatically at configure time, so adding a test is just a matter of dropping
# a file into a source directory - no per-file bookkeeping in CMake.
#
# Dependency handling
# -------------------
# oneDNN's per-directory OBJECT libraries (dnnl_common, dnnl_cpu, dnnl_cpu_x64,
# ...) do not form a clean dependency tree: e.g. common/bfloat16.cpp calls into
# cpu/bfloat16.cpp, and cpu code calls back into x64 kernels. Tracking a minimal
# per-test dependency set would therefore be fragile and high-maintenance.
#
# Instead, each component OBJECT library is wrapped once into a STATIC archive
# (dnnl_common.a, dnnl_cpu.a, dnnl_cpu_x64.a, ...) and every test is linked
# against the *whole set* inside a single linker `--start-group/--end-group`.
# Because these are archives (not raw object files), the linker pulls in only
# the objects a given test actually references, and the group resolves the
# cyclic references between components regardless of ordering. The net effect:
# a test author never has to declare or maintain any dependency edges.
#
# Registration is two-phase to sidestep CMake target-ordering constraints:
#   * dnnl_add_internal_unit_tests()  -- phase 1, called from a component's
#     CMakeLists.txt, records the test files it hosts.
#   * dnnl_finalize_internal_tests()  -- phase 2, called once from
#     src/CMakeLists.txt after every component has been defined, builds the
#     archives and the test executables.

if(DNNL_INTERNAL_UNIT_TEST_INCLUDED)
    return()
endif()
set(DNNL_INTERNAL_UNIT_TEST_INCLUDED TRUE)

# Remove next-to-source test files (`*_test.cpp`) from a component's source
# list so they are never compiled into the shipped oneDNN library. Call right
# before the component's add_library().
function(dnnl_exclude_internal_tests list_var)
    set(kept "")
    foreach(f ${${list_var}})
        if(NOT f MATCHES "/[^/]*_test\\.cpp$")
            list(APPEND kept "${f}")
        endif()
    endforeach()
    set(${list_var} "${kept}" PARENT_SCOPE)
endfunction()

# Minimal gtest entry point shared by every next-to-source unit test. It has no
# dependency on the oneDNN test harness (`dnnl_test_common.hpp`) or public API,
# so a test links against just the internal object files it needs.
set(DNNL_UNIT_TEST_MAIN "${CMAKE_CURRENT_LIST_DIR}/internal_unit_test_main.cpp")

# Build (once) a private gtest archive for the internal unit tests. Kept
# self-contained inside the `src` tree so this infrastructure does not depend
# on `tests/` being configured first.
function(_dnnl_internal_gtest out_var)
    set(tgt dnnl_internal_gtest)
    if(NOT TARGET ${tgt})
        add_library(${tgt} STATIC
            ${PROJECT_SOURCE_DIR}/third_party/gtest/src/gtest-all.cc)
        target_include_directories(${tgt} SYSTEM PUBLIC
            ${PROJECT_SOURCE_DIR}/third_party/gtest
            ${PROJECT_SOURCE_DIR}/third_party/gtest/include)
        find_package(Threads REQUIRED)
        target_link_libraries(${tgt} PUBLIC Threads::Threads)
    endif()
    set(${out_var} ${tgt} PARENT_SCOPE)
endfunction()

# Wrap an OBJECT library into a STATIC archive so the linker can selectively
# extract the object files a test references. Idempotent: at most one archive
# target is created per object library.
function(_dnnl_object_archive obj_lib out_var)
    set(archive "${obj_lib}_ar")
    if(NOT TARGET ${archive})
        add_library(${archive} STATIC $<TARGET_OBJECTS:${obj_lib}>)
        # A STATIC library made purely of object files needs an explicit
        # linker language.
        set_target_properties(${archive} PROPERTIES LINKER_LANGUAGE CXX)
    endif()
    set(${out_var} ${archive} PARENT_SCOPE)
endfunction()

# Phase 1. Record every `*_test.cpp` in the current source directory so it can
# be turned into a gtest executable later. Called from a component CMakeLists.
#
#   obj_lib -- OBJECT library of the current directory; only used to derive a
#              short, unique executable prefix (dnnl_cpu_x64 -> cpu_x64).
function(dnnl_add_internal_unit_tests obj_lib)
    if(NOT DNNL_BUILD_TESTS)
        return()
    endif()

    file(GLOB test_srcs
        ${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp)
    if(NOT test_srcs)
        return()
    endif()

    string(REGEX REPLACE "^${LIB_PACKAGE_NAME}_" "" dir_key "${obj_lib}")
    foreach(test_src ${test_srcs})
        set_property(GLOBAL APPEND PROPERTY DNNL_INTERNAL_TESTS
            "${dir_key}|${test_src}")
    endforeach()
endfunction()

# Phase 2. Build the component archives and one gtest executable per recorded
# test file. Call once from src/CMakeLists.txt after all components are defined.
function(dnnl_finalize_internal_tests)
    if(NOT DNNL_BUILD_TESTS)
        return()
    endif()

    get_property(tests GLOBAL PROPERTY DNNL_INTERNAL_TESTS)
    if(NOT tests)
        return()
    endif()

    # Wrap every component OBJECT library (the same set that feeds the final
    # oneDNN library) into a STATIC archive: dnnl_common.a, dnnl_cpu.a,
    # dnnl_cpu_x64.a, ...
    get_property(lib_deps GLOBAL PROPERTY DNNL_LIB_DEPS)
    set(archives "")
    foreach(dep ${lib_deps})
        string(REGEX REPLACE "^\\$<TARGET_OBJECTS:(.+)>$" "\\1" obj_lib "${dep}")
        if(TARGET ${obj_lib})
            _dnnl_object_archive(${obj_lib} ar)
            list(APPEND archives ${ar})
        endif()
    endforeach()

    _dnnl_internal_gtest(gtest_lib)

    foreach(entry ${tests})
        string(REPLACE "|" ";" parts "${entry}")
        list(GET parts 0 dir_key)
        list(GET parts 1 test_src)

        get_filename_component(base "${test_src}" NAME_WE) # bfloat16_test
        string(REGEX REPLACE "_test$" "" stem "${base}")   # bfloat16
        set(exe "test_internal_${dir_key}_${stem}")

        add_executable(${exe} ${test_src} ${DNNL_UNIT_TEST_MAIN})
        target_include_directories(${exe} PRIVATE
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_BINARY_DIR}/include)

        # Link the whole archive set in a group so the linker resolves the
        # cyclic references between components and extracts only what is needed.
        if(UNIX AND NOT APPLE)
            target_link_libraries(${exe} PRIVATE
                -Wl,--start-group ${archives} -Wl,--end-group)
        else()
            target_link_libraries(${exe} PRIVATE ${archives})
        endif()
        target_link_libraries(${exe} PRIVATE
            ${gtest_lib} ${EXTRA_SHARED_LIBS} ${EXTRA_STATIC_LIBS})

        add_test(NAME ${exe} COMMAND ${exe})
        maybe_configure_windows_test(${exe} TEST)
    endforeach()
endfunction()
