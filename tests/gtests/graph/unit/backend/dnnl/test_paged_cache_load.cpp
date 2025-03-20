/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#include <random>
#include "gtest/gtest.h"

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"
#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(test_paged_cache_load_execute, PagedCacheLoadInference) {
    graph::engine_t *eng = get_engine();

    // cache shape: [block_num, head_num, block_size, head_size]
    // block_table shape: [seq_num, max_block_num_per_seq]
    // output shape: [seq_num, head_num, seq_len, head_size]
    // attribute seq_len

    const int block_num = 4;
    const int head_num = 2;
    const int block_size = 2;
    const int head_size = 4;
    const int seq_num = 2;
    const int seq_len = 3;
    const int max_block_num_per_seq = 2;
    std::vector<float> cache(
            block_num * head_num * block_size * head_size, 0.0);
    for (size_t i = 0; i < cache.size(); i++) {
        cache[i] = i;
    }

    // block_table shape: [seq_num, max_block_num_per_seq]
    // seq_0: 0, 2
    // seq_1: 1, 3
    std::vector<int32_t> block_table({0, 2, 1, 3});

    // dst shape: (2, 2, 3, 4) ncx
    std::vector<float> dst(seq_num * head_num * seq_len * head_size, 0.0);
    std::vector<float> ref_dst = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 32.0,
            33.0, 34.0, 35.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            40.0, 41.0, 42.0, 43.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0,
            23.0, 48.0, 49.0, 50.0, 51.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
            30.0, 31.0, 56.0, 57.0, 58.0, 59.0};

    graph::logical_tensor_t cache_lt = utils::logical_tensor_init(0,
            {block_num, head_num, block_size, head_size},
            graph::data_type::f32);
    graph::logical_tensor_t block_table_lt = utils::logical_tensor_init(
            1, {seq_num, max_block_num_per_seq}, graph::data_type::s32);
    graph::logical_tensor_t dst_lt = utils::logical_tensor_init(
            2, {seq_num, head_num, seq_len, head_size}, graph::data_type::f32);

    graph::engine_t *engine = get_engine();
    graph::graph_t g(engine->kind());

    graph::op_t paged_cache_load_op(graph::op_kind::PagedCacheLoad);
    paged_cache_load_op.set_attr<int64_t>(graph::op_attr::seq_len, seq_len);
    paged_cache_load_op.add_input(cache_lt);
    paged_cache_load_op.add_input(block_table_lt);
    paged_cache_load_op.add_output(dst_lt);

    ASSERT_EQ(g.add_op(&paged_cache_load_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("paged_cache_load_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &cache_lt, &block_table_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), graph::status::success);

    test_tensor_t cache_ts(cache_lt, eng, cache);
    test_tensor_t block_table_ts(block_table_lt, eng, block_table);
    test_tensor_t dst_ts(dst_lt, eng, dst);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {cache_ts.get(), block_table_ts.get()}, {dst_ts.get()});
    strm->wait();

    dst = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < dst.size(); ++i) {
        std::cout << dst[i] << " ";
        if (i % 8 == 7) { std::cout << std::endl; }
        ASSERT_LE(std::abs(dst[i] - ref_dst[i]), 1e-4);
    }
}
