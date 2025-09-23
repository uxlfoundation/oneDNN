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
    ASSERT_EQ(dnnl_graph_set_dump_mode("subgraph"), dnnl_success);
    ASSERT_EQ(dnnl_graph_set_dump_mode("graph"), dnnl_success);
    ASSERT_EQ(dnnl_graph_set_dump_mode(""), dnnl_success);
    ASSERT_EQ(dnnl_graph_set_dump_mode("unknown"), dnnl_invalid_arguments);
}
