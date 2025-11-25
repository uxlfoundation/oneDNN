/*******************************************************************************
* Copyright 2024 Intel Corporation
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

TEST(test_rms_norm_execute, RMSNormInference) {
    graph::engine_t *eng = get_engine();

    // src shape: (2, 3, 2)
    std::vector<float> src {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f};
    std::vector<float> scale {1.0f, 2.0f};
    std::vector<float> dst(src.size(), 0.0f);

    // Reference values computed with NumPy:
    // RMS = sqrt(mean(x^2, axis=-1) + epsilon), output = (x / RMS) * scale
    std::vector<float> ref_dst {0.63245427f, 2.52981707f, 0.8485278f,
            2.26274079f, 0.90535731f, 2.17285755f, 0.93126606f, 2.12860815f,
            0.94605894f, 2.10235321f, 0.9556189f, 2.08498669f};

    graph::op_t rmsnorm_op(graph::op_kind::RMSNorm);

    rmsnorm_op.set_attr<float>(graph::op_attr::epsilon, 1e-5f);

    graph::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {2, 3, 2}, graph::data_type::f32);
    graph::logical_tensor_t scale_lt
            = utils::logical_tensor_init(1, {2}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(2, {2, 3, 2}, graph::data_type::f32);

    graph::engine_t *engine = get_engine();
    graph::graph_t g(engine->kind());

    rmsnorm_op.add_input(src_lt);
    rmsnorm_op.add_input(scale_lt);
    rmsnorm_op.add_output(dst_lt);

    ASSERT_EQ(g.add_op(&rmsnorm_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("rmsn_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src_lt, &scale_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), graph::status::success);

    test_tensor_t src_ts(src_lt, eng, src);
    test_tensor_t scale_ts(scale_lt, eng, scale);
    test_tensor_t dst_ts(dst_lt, eng, dst);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get(), scale_ts.get()}, {dst_ts.get()});
    strm->wait();

    dst = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_NEAR(dst[i], ref_dst[i], 1e-3);
    }
}

TEST(test_rms_norm_execute, RMSNormInferenceWithoutScale) {
    graph::engine_t *eng = get_engine();

    // src shape: (2, 2, 2)
    std::vector<float> src {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> dst(src.size(), 0.0f);

    // Reference values without scale (scale = 1.0)
    // Computed with NumPy: RMS = sqrt(mean(x^2, axis=-1) + epsilon), output = x / RMS
    std::vector<float> ref_dst {0.63245427f, 1.26490853f, 0.8485278f,
            1.1313704f, 0.90535731f, 1.08642877f, 0.93126606f, 1.06430407f};

    graph::op_t rmsnorm_op(graph::op_kind::RMSNorm);

    rmsnorm_op.set_attr<float>(graph::op_attr::epsilon, 1e-5f);

    graph::logical_tensor_t src_lt
            = utils::logical_tensor_init(0, {2, 2, 2}, graph::data_type::f32);
    graph::logical_tensor_t dst_lt
            = utils::logical_tensor_init(1, {2, 2, 2}, graph::data_type::f32);

    graph::engine_t *engine = get_engine();
    graph::graph_t g(engine->kind());

    rmsnorm_op.add_input(src_lt);
    rmsnorm_op.add_output(dst_lt);

    ASSERT_EQ(g.add_op(&rmsnorm_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("rmsn_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {&src_lt};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), graph::status::success);

    test_tensor_t src_ts(src_lt, eng, src);
    test_tensor_t dst_ts(dst_lt, eng, dst);

    graph::stream_t *strm = get_stream();
    cp.execute(strm, {src_ts.get()}, {dst_ts.get()});
    strm->wait();

    dst = dst_ts.as_vec_type<float>();
    for (size_t i = 0; i < ref_dst.size(); ++i) {
        ASSERT_NEAR(dst[i], ref_dst[i], 1e-3);
    }
}
