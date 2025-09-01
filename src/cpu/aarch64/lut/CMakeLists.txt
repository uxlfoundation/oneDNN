#*******************************************************************************
# Copyright 2025 Arm Ltd. and affiliates
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
#*******************************************************************************


add_executable(lut_gen "${CMAKE_CURRENT_SOURCE_DIR}/lut_gen.cpp")
set(GEN_LUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
add_custom_command(
  TARGET lut_gen
  POST_BUILD
  COMMAND $<TARGET_FILE:lut_gen> --out-dir "${GEN_LUT_DIR}"
  DEPENDS lut_gen
  COMMENT "Generating all LUT tables"
  VERBATIM
)
file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/lut_tables.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lut_tables.hpp
    )