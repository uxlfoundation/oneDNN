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


file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]pp
    )

set(OBJ_LIB ${LIB_PACKAGE_NAME}_graph_interface)
# Keep next-to-source unit tests (*_test.cpp) out of the shipped library.
dnnl_exclude_internal_tests(SOURCES)
add_library(${OBJ_LIB} OBJECT ${SOURCES})

set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)

# Register any *_test.cpp living in this directory as standalone gtest binaries.
dnnl_add_internal_unit_tests(${OBJ_LIB})
