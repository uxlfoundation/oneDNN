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

set(TEST_SOURCES
    test_regression_binary_stride.cpp
)

if(UNIX AND NOT DNNL_GPU_RUNTIME STREQUAL "NONE" AND DNNL_GPU_VENDOR STREQUAL "INTEL")
    set(TEST_SOURCES "${TEST_SOURCES};test_regression_pvc_war_wa.cpp")
endif()

foreach(TEST_FILE ${TEST_SOURCES})
    get_filename_component(exe ${TEST_FILE} NAME_WE)
    register_exe("${exe}" "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_FILE};${MAIN_SRC_GTEST}" "test" "dnnl_gtest")
endforeach()

if(TARGET test_regression_pvc_war_wa)
    # Regression test for PVC WAR bug workaround. The bug is triggered by setting
    # round-robin thread arbitration policy so override it here.
    set_tests_properties(test_regression_pvc_war_wa
        PROPERTIES ENVIRONMENT "NEOReadDebugKeys=1;OverrideThreadArbitrationPolicy=1")
endif()
