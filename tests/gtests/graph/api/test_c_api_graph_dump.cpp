/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include <gtest/gtest.h>

#include "oneapi/dnnl/dnnl_graph.h"

TEST(CAPI, GraphDump) {
    ASSERT_EQ(dnnl_graph_set_dump_mode(dnnl_graph_dump_mode_subgraph),
            dnnl_success);
    ASSERT_EQ(
            dnnl_graph_set_dump_mode(dnnl_graph_dump_mode_graph), dnnl_success);
    ASSERT_EQ(dnnl_graph_set_dump_mode(static_cast<dnnl_graph_dump_mode_t>(
                      dnnl_graph_dump_mode_graph
                      | dnnl_graph_dump_mode_subgraph)),
            dnnl_success);
    ASSERT_EQ(
            dnnl_graph_set_dump_mode(dnnl_graph_dump_mode_none), dnnl_success);
}
