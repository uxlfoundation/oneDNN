/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "test_api_common.h"

TEST(CAPI, CreateGraphWithEngine) {
    dnnl_graph_graph_t agraph = nullptr;
    dnnl_engine_kind_t engine = dnnl_cpu;

#define CREATE_GRAPH_WITH_ENGINE_DESTROY \
    do { \
        dnnl_graph_graph_destroy(agraph); \
        agraph = NULL; \
    } while (0);

    ASSERT_EQ_SAFE(dnnl_graph_graph_create(&agraph, engine), dnnl_success,
            CREATE_GRAPH_WITH_ENGINE_DESTROY);
    CREATE_GRAPH_WITH_ENGINE_DESTROY;
#undef CREATE_GRAPH_WITH_ENGINE_DESTROY
}

TEST(CAPI, SetAndGetDeterministic) {
    dnnl_graph_graph_t graph = nullptr;
    ASSERT_EQ(dnnl_graph_graph_create(&graph, dnnl_cpu), dnnl_success);

    int deterministic = -1;
    EXPECT_EQ(dnnl_graph_graph_get_deterministic(graph, &deterministic),
            dnnl_success);
    EXPECT_EQ(deterministic, 0);

    EXPECT_EQ(dnnl_graph_graph_set_deterministic(graph, 1), dnnl_success);
    EXPECT_EQ(dnnl_graph_graph_get_deterministic(graph, &deterministic),
            dnnl_success);
    EXPECT_EQ(deterministic, 1);

    EXPECT_EQ(dnnl_graph_graph_get_deterministic(nullptr, &deterministic),
            dnnl_invalid_arguments);
    EXPECT_EQ(dnnl_graph_graph_get_deterministic(graph, nullptr),
            dnnl_invalid_arguments);
    EXPECT_EQ(dnnl_graph_graph_set_deterministic(nullptr, 1),
            dnnl_invalid_arguments);

    ASSERT_EQ(dnnl_graph_graph_finalize(graph), dnnl_success);
    EXPECT_EQ(dnnl_graph_graph_get_deterministic(graph, &deterministic),
            dnnl_success);
    EXPECT_EQ(dnnl_graph_graph_set_deterministic(graph, 0), dnnl_invalid_graph);

    EXPECT_EQ(dnnl_graph_graph_destroy(graph), dnnl_success);
}
