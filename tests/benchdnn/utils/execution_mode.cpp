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

execution_mode_t default_execution_mode {execution_mode_t::direct};
execution_mode_t execution_mode {default_execution_mode};

std::ostream &operator<<(std::ostream &s, execution_mode_t mode) {
    if (mode == execution_mode_t::direct) s << "direct";
    if (mode == execution_mode_t::graph) s << "graph";

    return s;
}

bool use_sycl_graph_exec() {
    return is_sycl_engine() && execution_mode == execution_mode_t::graph;
}
