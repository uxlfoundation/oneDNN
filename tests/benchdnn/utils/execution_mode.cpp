/*******************************************************************************
* Copyright 2026 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "utils/execution_mode.hpp"
#include "common.hpp"
#include "dnnl_common.hpp"

execution_mode_t execution_mode {execution_mode_t::direct};

const char *execution_mode2str(execution_mode_t mode) {
#define EXECUTION_MODE_TO_STR(name, ...) \
    if (execution_mode_t::name == mode) return #name;

    EXECUTION_MODE_TO_STR(direct);
    EXECUTION_MODE_TO_STR(graph);
#undef EXECUTION_MODE_TO_STR

    BENCHDNN_PRINT(0, "%s", "Error: execution mode value is not recognized.\n");
    SAFE_V(FAIL);
    return "";
}

execution_mode_t str2execution_mode(const char *str) {
#define STR_TO_EXECUTION_MODE(name, ...) \
    if (!strcasecmp(#name, str)) return execution_mode_t::name;

    STR_TO_EXECUTION_MODE(direct);
    STR_TO_EXECUTION_MODE(graph);
#undef STR_TO_EXECUTION_MODE

    BENCHDNN_PRINT(0, "%s", "Error: execution mode value is not recognized.\n");
    SAFE_V(FAIL);
    return execution_mode_t::direct;
}

bool use_sycl_graph_exec() {
    return is_sycl_engine() && execution_mode == execution_mode_t::graph;
}
