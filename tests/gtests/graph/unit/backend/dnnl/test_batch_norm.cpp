/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

#include <memory>

#include "gtest/gtest.h"

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"
#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

static inline void ref_batchnorm_fwd(graph::dim_t mb, graph::dim_t ic,
        graph::dim_t ih, graph::dim_t iw, test_tensor_t *src,
        test_tensor_t *dst, test_tensor_t *scale, test_tensor_t *shift,
        test_tensor_t *mean = nullptr, test_tensor_t *variance = nullptr,
        test_tensor_t *running_mean = nullptr,
        test_tensor_t *running_variance = nullptr,
        test_tensor_t *batch_mean = nullptr,
        test_tensor_t *batch_variance = nullptr, float epsilon = 0.001f,
        float momentum = 0.1f,
        std::function<float(const float)> activation = nullptr,
        bool channel_last = false, bool is_training = false) {
    if (!activation) activation = [](const float v) { return v; };

    size_t mb_ = static_cast<size_t>(mb);
    size_t ic_ = static_cast<size_t>(ic);
    size_t ih_ = static_cast<size_t>(ih);
    size_t iw_ = static_cast<size_t>(iw);

    auto src_ptr = src->as_vec_type<float>();
    auto dst_ptr = dst->as_vec_type<float>();

    if (is_training) {
        auto mean_ptr = batch_mean->as_vec_type<float>();
        auto variance_ptr = batch_variance->as_vec_type<float>();
        auto scale_ptr = scale->as_vec_type<float>();
        auto shift_ptr = shift->as_vec_type<float>();
        auto est_mean_ptr = mean->as_vec_type<float>();
        auto est_variance_ptr = variance->as_vec_type<float>();
        auto running_mean_ptr = running_mean->as_vec_type<float>();
        auto running_variance_ptr = running_variance->as_vec_type<float>();
        // derives ref mean
        for (size_t channel = 0; channel < ic_; ++channel) {
            mean_ptr[channel] = 0.f;
            for (size_t batch = 0; batch < mb_; ++batch) {
                for (size_t h = 0; h < ih_; ++h) {
                    for (size_t w = 0; w < iw_; ++w) {
                        size_t idx = channel_last
                                ? channel + w * ic_ + h * iw_ * ic_
                                        + batch * ih_ * iw_ * ic_
                                : w + h * iw_ + channel * ih_ * iw_
                                        + batch * ic_ * ih_ * iw_;
                        mean_ptr[channel] += src_ptr[idx];
                    }
                }
            }
            mean_ptr[channel] /= static_cast<float>(mb_ * ih_ * iw_);
        }
        // derives ref variance
        for (size_t channel = 0; channel < ic_; ++channel) {
            variance_ptr[channel] = 0.f;
            for (size_t batch = 0; batch < mb_; ++batch) {
                for (size_t h = 0; h < ih_; ++h) {
                    for (size_t w = 0; w < iw_; ++w) {
                        size_t idx = channel_last
                                ? channel + w * ic_ + h * iw_ * ic_
                                        + batch * ih_ * iw_ * ic_
                                : w + h * iw_ + channel * ih_ * iw_
                                        + batch * ic_ * ih_ * iw_;
                        float variance_tmp = src_ptr[idx] - mean_ptr[channel];
                        variance_ptr[channel] += variance_tmp * variance_tmp;
                    }
                }
            }
            variance_ptr[channel] /= static_cast<float>(mb * ih * iw);
        }
        for (size_t batch = 0; batch < mb_; ++batch) {
            for (size_t channel = 0; channel < ic_; ++channel) {
                for (size_t h = 0; h < ih_; ++h) {
                    for (size_t w = 0; w < iw_; ++w) {
                        size_t idx = channel_last
                                ? channel + w * ic_ + h * iw_ * ic_
                                        + batch * ih_ * iw_ * ic_
                                : w + h * iw_ + channel * ih_ * iw_
                                        + batch * ic_ * ih_ * iw_;
                        dst_ptr[idx] = activation(scale_ptr[channel]
                                        * (src_ptr[idx] - mean_ptr[channel])
                                        / sqrtf(variance_ptr[channel] + epsilon)
                                + shift_ptr[channel]);
                    }
                }
            }
        }
        for (size_t k = 0; k < ic_; k++) {
            running_mean_ptr[k]
                    = momentum * est_mean_ptr[k] + (1 - momentum) * mean_ptr[k];
            running_variance_ptr[k] = momentum * est_variance_ptr[k]
                    + (1 - momentum) * variance_ptr[k];
        }
        // write back;
        batch_mean->fill(mean_ptr);
        batch_variance->fill(variance_ptr);
        running_mean->fill(running_mean_ptr);
        running_variance->fill(running_variance_ptr);
    } else {
        std::vector<float> mean_ptr;
        std::vector<float> variance_ptr;
        auto scale_ptr = scale->as_vec_type<float>();
        auto shift_ptr = shift->as_vec_type<float>();
        if (!mean) {
            mean_ptr = std::vector<float>(static_cast<size_t>(ic), .0f);
        } else {
            mean_ptr = mean->as_vec_type<float>();
        }
        if (!variance) {
            variance_ptr = std::vector<float>(static_cast<size_t>(ic), .0f);
        } else {
            variance_ptr = variance->as_vec_type<float>();
        }
        for (size_t batch = 0; batch < mb_; ++batch) {
            for (size_t channel = 0; channel < ic_; ++channel) {
                for (size_t h = 0; h < ih_; ++h) {
                    for (size_t w = 0; w < iw_; ++w) {
                        size_t idx = channel_last
                                ? channel + w * ic_ + h * iw_ * ic_
                                        + batch * ih_ * iw_ * ic_
                                : w + h * iw_ + channel * ih_ * iw_
                                        + batch * ic_ * ih_ * iw_;
                        dst_ptr[idx] = activation(scale_ptr[channel]
                                        * (src_ptr[idx] - mean_ptr[channel])
                                        / sqrtf(variance_ptr[channel] + epsilon)
                                + shift_ptr[channel]);
                    }
                }
            }
        }
    }
    dst->fill(dst_ptr);
}

struct batchnorm_params_t {
    graph::op_kind_t op_kind;
    bool with_relu;
    graph::dim_t N;
    graph::dim_t IC;
    graph::dim_t IH;
    graph::dim_t IW;
    float epsilon;
    std::string data_format;
    std::string backend_name;
    graph::data_type_t data_type;
};

class batch_norm_4d_t : public ::testing::TestWithParam<batchnorm_params_t> {
public:
    void TestBatchnorm(bool dims_in_order = true) {
        using dims = dnnl::impl::graph::dnnl_impl::dims;

        auto params = ::testing::TestWithParam<batchnorm_params_t>::GetParam();

        graph::op_t batchnorm_op(0, params.op_kind, "batchnorm");
        batchnorm_op.set_attr(graph::op_attr::epsilon, params.epsilon);
        batchnorm_op.set_attr(graph::op_attr::data_format, params.data_format);
        graph::op_t relu_op(1, graph::op_kind::ReLU, "relu");

        bool is_training
                = params.op_kind == graph::op_kind::BatchNormForwardTraining;
        const float momentum = 0.1f;
        if (is_training)
            batchnorm_op.set_attr(graph::op_attr::momentum, momentum);

        graph::engine_t *engine = get_engine();

        // Tensor dimensions.
        const graph::dim_t N = params.N, // batch size
                IC = params.IC, // channels
                IH = params.IH, // tensor height
                IW = params.IW; // tensor width
        // Source (src) and destination (dst) tensors dimensions.
        dims src_dims;

        // |                     | == NCX?   | == NXC?   |
        // |---------------------|-----------|-----------|
        // |in-order (true)      | NCHW      | NHWC      |
        // |out-of-order (false) | NHWC      | HCHW      |
        // if ((params.data_format == "NCX") ^ dims_in_order)
        if (params.data_format == "NXC")
            src_dims = {N, IH, IW, IC};
        else
            src_dims = {N, IC, IH, IW};
        // Scale/shift tensor dimensions.
        dims scale_dims = {IC};
        dims shift_dims = {IC};
        dims mean_dims = {IC};
        dims variance_dims = {IC};

        // prepare logical tensor
        graph::logical_tensor_t src = utils::logical_tensor_init(
                0, src_dims, graph::data_type::f32);
        graph::logical_tensor_t scale = utils::logical_tensor_init(
                1, scale_dims, graph::data_type::f32);
        graph::logical_tensor_t shift = utils::logical_tensor_init(
                2, shift_dims, graph::data_type::f32);
        graph::logical_tensor_t mean = utils::logical_tensor_init(
                3, mean_dims, graph::data_type::f32);
        graph::logical_tensor_t variance = utils::logical_tensor_init(
                4, variance_dims, graph::data_type::f32);
        graph::logical_tensor_t running_mean = utils::logical_tensor_init(
                5, mean_dims, graph::data_type::f32);
        graph::logical_tensor_t running_variance = utils::logical_tensor_init(
                6, variance_dims, graph::data_type::f32);
        graph::logical_tensor_t batch_mean = utils::logical_tensor_init(
                7, mean_dims, graph::data_type::f32);
        graph::logical_tensor_t batch_variance = utils::logical_tensor_init(
                8, variance_dims, graph::data_type::f32);
        graph::logical_tensor_t dst = utils::logical_tensor_init(
                9, src_dims, graph::data_type::f32);
        graph::logical_tensor_t relu_dst = utils::logical_tensor_init(
                10, src_dims, graph::data_type::f32);

        batchnorm_op.add_input(src);
        if (!is_training) {
            batchnorm_op.add_input(scale);
            batchnorm_op.add_input(shift);
        }
        batchnorm_op.add_input(mean);
        batchnorm_op.add_input(variance);
        if (is_training) {
            batchnorm_op.add_input(scale);
            batchnorm_op.add_input(shift);
        }
        batchnorm_op.add_output(dst);
        if (is_training) {
            batchnorm_op.add_output(running_mean);
            batchnorm_op.add_output(running_variance);
            batchnorm_op.add_output(batch_mean);
            batchnorm_op.add_output(batch_variance);
        }

        relu_op.add_input(dst);
        relu_op.add_output(relu_dst);

        graph::graph_t g(engine->kind());
        g.add_op(&batchnorm_op);
        if (params.with_relu) g.add_op(&relu_op);
        g.finalize();

        graph::pass::pass_base_ptr apass = params.with_relu
                ? get_pass("fp_bnorm_relu")
                : (is_training ? get_pass("bn_fw_train_pass")
                               : get_pass("bn_pass"));
        apass->run(g);
        ASSERT_EQ(g.get_num_partitions(), 1U);
        auto part = g.get_partitions()[0];

        // compile
        graph::partition_t p;
        p.init(part);
        graph::compiled_partition_t cp(p);

        std::vector<const graph::logical_tensor_t *> inputs {&src};
        if (!is_training) {
            inputs.emplace_back(&scale);
            inputs.emplace_back(&shift);
        }
        inputs.emplace_back(&mean);
        inputs.emplace_back(&variance);
        if (is_training) {
            inputs.emplace_back(&scale);
            inputs.emplace_back(&shift);
        }
        std::vector<const graph::logical_tensor_t *> outputs {&dst};
        if (params.with_relu) outputs[0] = &relu_dst;
        if (is_training) {
            outputs.emplace_back(&running_mean);
            outputs.emplace_back(&running_variance);
            outputs.emplace_back(&batch_mean);
            outputs.emplace_back(&batch_variance);
        }

        ASSERT_EQ(p.compile(&cp, inputs, outputs, engine),
                graph::status::success);

        // Allocate buffers.
        std::vector<float> src_data(static_cast<size_t>(N * IC * IH * IW));
        std::vector<float> scale_data(static_cast<size_t>(IC));
        std::vector<float> shift_data(static_cast<size_t>(IC));
        std::vector<float> mean_data(static_cast<size_t>(IC));
        std::vector<float> variance_data(static_cast<size_t>(IC));
        std::vector<float> running_mean_data(static_cast<size_t>(IC));
        std::vector<float> running_variance_data(static_cast<size_t>(IC));
        std::vector<float> batch_mean_data(static_cast<size_t>(IC));
        std::vector<float> batch_variance_data(static_cast<size_t>(IC));
        std::vector<float> dst_data(src_data.size(), 0.0);
        std::vector<float> ref_dst_data(src_data.size(), 0.0);
        std::vector<float> ref_running_mean_data(running_mean_data.size(), 0.0);
        std::vector<float> ref_running_variance_data(
                running_variance_data.size(), 0.0);
        std::vector<float> ref_batch_mean_data(batch_mean_data.size(), 0.0);
        std::vector<float> ref_batch_variance_data(
                batch_variance_data.size(), 0.0);
        // Initialize src.
        std::generate(src_data.begin(), src_data.end(), []() {
            static int i = 0;
            return std::cos(static_cast<float>(i++) / 10.f);
        });
        // Initialize scale.
        std::generate(scale_data.begin(), scale_data.end(), []() {
            static int i = 0;
            return std::sin(static_cast<float>(i++) * 2.f);
        });
        // Initialize shift.
        std::generate(shift_data.begin(), shift_data.end(), []() {
            static int i = 0;
            return std::tan(i++);
        });
        if (!is_training) {
            // Initialize mean.
            std::generate(mean_data.begin(), mean_data.end(), []() {
                static int i = 0;
                return std::sin(static_cast<float>(i++) * 2.f);
            });
            // Initialize variance.
            std::generate(variance_data.begin(), variance_data.end(), []() {
                static int i = 0;
                return static_cast<float>(i++);
            });
        } else {
            // Initialize mean.
            std::generate(
                    running_mean_data.begin(), running_mean_data.end(), []() {
                        static int i = 0;
                        return std::sin(static_cast<float>(i++) * 2.f);
                    });
            // Initialize variance.
            std::generate(running_variance_data.begin(),
                    running_variance_data.end(), []() {
                        static int i = 0;
                        return static_cast<float>(i++);
                    });
        }

        test_tensor_t src_ts(src, engine, src_data);
        test_tensor_t scale_ts(scale, engine, scale_data);
        test_tensor_t shift_ts(shift, engine, shift_data);
        test_tensor_t mean_ts(mean, engine, mean_data);
        test_tensor_t variance_ts(variance, engine, variance_data);
        test_tensor_t running_mean_ts(running_mean, engine, running_mean_data);
        test_tensor_t running_variance_ts(
                running_variance, engine, running_variance_data);
        test_tensor_t batch_mean_ts(batch_mean, engine, batch_mean_data);
        test_tensor_t batch_variance_ts(
                batch_variance, engine, batch_variance_data);
        test_tensor_t dst_ts(
                params.with_relu ? relu_dst : dst, engine, dst_data);
        test_tensor_t ref_dst_ts(
                params.with_relu ? relu_dst : dst, engine, ref_dst_data);
        test_tensor_t ref_running_mean_ts(
                running_mean, engine, ref_running_mean_data);
        test_tensor_t ref_running_variance_ts(
                running_variance, engine, ref_running_variance_data);
        test_tensor_t ref_batch_mean_ts(
                batch_mean, engine, ref_batch_mean_data);
        test_tensor_t ref_batch_variance_ts(
                batch_variance, engine, ref_batch_variance_data);

        graph::stream_t *strm = get_stream();

        if (!is_training) {
            cp.execute(strm,
                    {src_ts.get(), scale_ts.get(), shift_ts.get(),
                            mean_ts.get(), variance_ts.get()},
                    {dst_ts.get()});
        } else {
            cp.execute(strm,
                    {src_ts.get(), mean_ts.get(), variance_ts.get(),
                            scale_ts.get(), shift_ts.get()},
                    {dst_ts.get(), running_mean_ts.get(),
                            running_variance_ts.get(), batch_mean_ts.get(),
                            batch_variance_ts.get()});
        }

        strm->wait();

        std::function<float(const float)> activation = nullptr;
        if (params.with_relu) {
            activation = [](const float v) { return v < 0.f ? 0.f : v; };
        }
        if (!is_training) {
            ref_batchnorm_fwd(N, IC, IH, IW, &src_ts, &ref_dst_ts, &scale_ts,
                    &shift_ts, &mean_ts, &variance_ts, nullptr, nullptr,
                    nullptr, nullptr, params.epsilon, momentum, activation,
                    params.data_format == "NXC",
                    params.op_kind == graph::op_kind::BatchNormForwardTraining);
        } else {
            ref_batchnorm_fwd(N, IC, IH, IW, &src_ts, &ref_dst_ts, &scale_ts,
                    &shift_ts, &mean_ts, &variance_ts, &ref_running_mean_ts,
                    &ref_running_variance_ts, &ref_batch_mean_ts,
                    &ref_batch_variance_ts, params.epsilon, momentum,
                    activation, params.data_format == "NXC",
                    params.op_kind == graph::op_kind::BatchNormForwardTraining);
            running_mean_data = running_mean_ts.as_vec_type<float>();
            ref_running_mean_data = ref_running_mean_ts.as_vec_type<float>();
            running_variance_data = running_variance_ts.as_vec_type<float>();
            ref_running_variance_data
                    = ref_running_variance_ts.as_vec_type<float>();
            batch_mean_data = batch_mean_ts.as_vec_type<float>();
            ref_batch_mean_data = ref_batch_mean_ts.as_vec_type<float>();
            batch_variance_data = batch_variance_ts.as_vec_type<float>();
            ref_batch_variance_data
                    = ref_batch_variance_ts.as_vec_type<float>();
            for (int64_t i = 0; i < IC; ++i) {
                ASSERT_NEAR(
                        running_mean_data[i], ref_running_mean_data[i], 1.e-6f);
                ASSERT_NEAR(running_variance_data[i],
                        ref_running_variance_data[i], 1.e-6f);
                ASSERT_NEAR(batch_mean_data[i], ref_batch_mean_data[i], 1.e-6f);
                ASSERT_NEAR(batch_variance_data[i], ref_batch_variance_data[i],
                        1.e-6f);
            }
        }
        dst_data = dst_ts.as_vec_type<float>();
        ref_dst_data = ref_dst_ts.as_vec_type<float>();
        for (size_t i = 0; i < dst_data.size(); ++i) {
            ASSERT_NEAR(dst_data[i], ref_dst_data[i], 1.e-6f);
        }
    }

    void Test() {
        TestBatchnorm(true);
        TestBatchnorm(false);
    }
};

TEST_P(batch_norm_4d_t, TestBatchnorm) {
    SKIP_IF_NV_GPU("not supported on NVIDIA GPU");
    Test();
}

INSTANTIATE_TEST_SUITE_P(test_batch_norm_execute, batch_norm_4d_t,
        ::testing::Values(
                batchnorm_params_t {graph::op_kind::BatchNormInference, false,
                        3, 3, 2, 2, 0.001f, "NCX", "dnnl",
                        graph::data_type::f32},
                batchnorm_params_t {graph::op_kind::BatchNormInference, true, 3,
                        3, 2, 2, 0.001f, "NCX", "dnnl", graph::data_type::f32},
                batchnorm_params_t {graph::op_kind::BatchNormInference, false,
                        3, 3, 2, 2, 0.001f, "NXC", "dnnl",
                        graph::data_type::f32},
                batchnorm_params_t {graph::op_kind::BatchNormForwardTraining,
                        false, 3, 3, 2, 2, 0.001f, "NCX", "dnnl",
                        graph::data_type::f32},
                batchnorm_params_t {graph::op_kind::BatchNormForwardTraining,
                        false, 3, 3, 2, 2, 0.001f, "NXC", "dnnl",
                        graph::data_type::f32},
                batchnorm_params_t {graph::op_kind::BatchNormForwardTraining,
                        true, 3, 3, 2, 2, 0.001f, "NXC", "dnnl",
                        graph::data_type::f32}));

TEST(test_batch_norm_compile, BatchNormBackwardFp32) {
    SKIP_IF_NV_GPU("not supported on NVIDIA GPU");
    using dims = dnnl::impl::graph::dnnl_impl::dims;

    graph::op_t bn_op(graph::op_kind::BatchNormTrainingBackward);
    bn_op.set_attr<float>(graph::op_attr::epsilon, 0.0);
    bn_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");

    graph::engine_t *engine = get_engine();

    // Tensor dimensions.
    const graph::dim_t N = 1, // batch size
            IC = 1, // channels
            IH = 2, // tensor height
            IW = 2; // tensor width
    // Source (src) and destination (dst) tensors dimensions.
    dims src_dims = {N, IC, IH, IW};
    // Scale/shift tensor dimensions.
    dims scale_dims = {IC};
    dims shift_dims = {IC};
    dims mean_dims = {IC};
    dims variance_dims = {IC};

    // prepare logical tensor
    graph::logical_tensor_t src = utils::logical_tensor_init(
            0, src_dims, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t scale = utils::logical_tensor_init(
            1, scale_dims, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t mean = utils::logical_tensor_init(
            2, mean_dims, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t variance = utils::logical_tensor_init(3,
            variance_dims, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t diff_dst = utils::logical_tensor_init(
            4, src_dims, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t diff_src = utils::logical_tensor_init(
            5, src_dims, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t diff_scale = utils::logical_tensor_init(
            6, scale_dims, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t diff_shift = utils::logical_tensor_init(
            7, shift_dims, graph::data_type::f32, graph::layout_type::strided);

    // Allocate buffers.
    std::vector<float> src_data {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> scale_data {2.0f};
    std::vector<float> mean_data {2.5f};
    std::vector<float> varience_data {1.25f};
    std::vector<float> diff_dst_data {0.1f, 0.1f, 0.2f, 0.2f};
    std::vector<float> diff_src_data(src_data.size(), 0.0);
    std::vector<float> diff_scale_data(scale_data.size(), 0.0);
    std::vector<float> diff_shift_data(scale_data.size(), 0.0);

    // expected results
    std::vector<float> ref_diff_src_data {
            0.01788854f, -0.05366563f, 0.05366563f, -0.01788854f};
    std::vector<float> ref_diff_scale_data {0.17888544f};
    std::vector<float> ref_diff_shift_data {0.6f};

    bn_op.add_input(src);
    bn_op.add_input(diff_dst);
    bn_op.add_input(mean);
    bn_op.add_input(variance);
    bn_op.add_input(scale);
    bn_op.add_output(diff_src);
    bn_op.add_output(diff_scale);
    bn_op.add_output(diff_shift);

    graph::graph_t g(engine->kind());
    g.add_op(&bn_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("bn_bw_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src, &diff_dst, &mean, &variance, &scale};
    std::vector<const graph::logical_tensor_t *> outputs {
            &diff_src, &diff_scale, &diff_shift};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), graph::status::success);

    test_tensor_t src_ts(src, engine, src_data);
    test_tensor_t scale_ts(scale, engine, scale_data);
    test_tensor_t mean_ts(mean, engine, mean_data);
    test_tensor_t variance_ts(variance, engine, varience_data);
    test_tensor_t diff_dst_ts(diff_dst, engine, diff_dst_data);
    test_tensor_t diff_src_ts(diff_src, engine, diff_src_data);
    test_tensor_t diff_scale_ts(diff_scale, engine, diff_scale_data);
    test_tensor_t diff_shift_ts(diff_shift, engine, diff_shift_data);

    graph::stream_t *strm = get_stream();
    cp.execute(strm,
            {src_ts.get(), diff_dst_ts.get(), mean_ts.get(), variance_ts.get(),
                    scale_ts.get()},
            {diff_src_ts.get(), diff_scale_ts.get(), diff_shift_ts.get()});
    strm->wait();
    diff_src_data = diff_src_ts.as_vec_type<float>();
    for (size_t i = 0; i < diff_src_data.size(); ++i) {
        ASSERT_NEAR(diff_src_data[i], ref_diff_src_data[i], 1e-3);
    }
    diff_scale_data = diff_scale_ts.as_vec_type<float>();
    for (size_t i = 0; i < diff_scale_data.size(); ++i) {
        ASSERT_NEAR(diff_scale_data[i], ref_diff_scale_data[i], 1e-3);
    }
    diff_shift_data = diff_shift_ts.as_vec_type<float>();
    for (size_t i = 0; i < diff_shift_data.size(); ++i) {
        ASSERT_NEAR(diff_shift_data[i], ref_diff_shift_data[i], 1e-3);
    }
}

TEST(test_batch_norm_compile, BatchNormBackwardFp32WithSingleOutput) {
    using dims = dnnl::impl::graph::dnnl_impl::dims;

    graph::op_t bn_op(graph::op_kind::BatchNormTrainingBackward);
    bn_op.set_attr<float>(graph::op_attr::epsilon, 0.0);
    bn_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");

    graph::engine_t *engine = get_engine();

    // Tensor dimensions.
    const graph::dim_t N = 1, // batch size
            IC = 1, // channels
            IH = 2, // tensor height
            IW = 2; // tensor width
    // Source (src) and destination (dst) tensors dimensions.
    dims src_dims = {N, IC, IH, IW};
    // Scale/shift tensor dimensions.
    dims scale_dims = {IC};
    dims shift_dims = {IC};
    dims mean_dims = {IC};
    dims variance_dims = {IC};

    // prepare logical tensor
    graph::logical_tensor_t src = utils::logical_tensor_init(
            0, src_dims, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t mean = utils::logical_tensor_init(
            1, mean_dims, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t variance = utils::logical_tensor_init(2,
            variance_dims, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t diff_dst = utils::logical_tensor_init(
            3, src_dims, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t diff_src = utils::logical_tensor_init(
            4, src_dims, graph::data_type::f32, graph::layout_type::strided);

    // Allocate buffers.
    std::vector<float> src_data {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> mean_data {2.5f};
    std::vector<float> variance_data {1.25f};
    std::vector<float> diff_dst_data {0.1f, 0.1f, 0.2f, 0.2f};
    std::vector<float> diff_src_data(src_data.size(), 0.0);

    bn_op.add_input(src);
    bn_op.add_input(diff_dst);
    bn_op.add_input(mean);
    bn_op.add_input(variance);
    bn_op.add_output(diff_src);

    graph::graph_t g(engine->kind());
    g.add_op(&bn_op);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("bn_bw_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src, &diff_dst, &mean, &variance};
    std::vector<const graph::logical_tensor_t *> outputs {&diff_src};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), graph::status::success);

    test_tensor_t src_ts(src, engine, src_data);
    test_tensor_t mean_ts(mean, engine, mean_data);
    test_tensor_t variance_ts(variance, engine, variance_data);
    test_tensor_t diff_dst_ts(diff_dst, engine, diff_dst_data);
    test_tensor_t diff_src_ts(diff_src, engine, diff_src_data);

    graph::stream_t *strm = get_stream();
    cp.execute(strm,
            {src_ts.get(), diff_dst_ts.get(), mean_ts.get(), variance_ts.get()},
            {diff_src_ts.get()});
    strm->wait();
}

TEST(test_batch_norm_compile, BatchNormForwardTrainingWith1DSpatialInput) {
    SKIP_IF_NV_GPU("not supported on NVIDIA GPU");

    using dims = graph::dnnl_impl::dims;
    using ltw = graph::logical_tensor_wrapper_t;
    graph::engine_t *engine = get_engine();

    // Tensor dimensions.
    const graph::dim_t N = 1, // batch size
            IC = 1, // channels
            IW = 4; // tensor width
    dims src_dims = {N, IC, IW};
    dims scale_dims = {IC};
    dims shift_dims = {IC};
    dims mean_dims = {IC};
    dims variance_dims = {IC};

    // Allocate buffers.
    std::vector<float> src_data {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> scale_data {1.f};
    std::vector<float> shift_data {2.f};
    std::vector<float> mean_data {2.5f};
    std::vector<float> variance_data {1.25f};

    graph::op_t bn_op(graph::op_kind::BatchNormForwardTraining);
    bn_op.set_attr<float>(graph::op_attr::epsilon, 0.0625);
    bn_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");

    // prepare logical tensor
    graph::logical_tensor_t src = utils::logical_tensor_init(
            0, src_dims, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t scale = utils::logical_tensor_init(
            1, scale_dims, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t shift = utils::logical_tensor_init(
            2, shift_dims, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t mean = utils::logical_tensor_init(
            3, mean_dims, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t variance = utils::logical_tensor_init(4,
            variance_dims, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t dst = utils::logical_tensor_init(
            5, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t running_mean = utils::logical_tensor_init(
            6, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t running_variance = utils::logical_tensor_init(
            7, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t batch_mean = utils::logical_tensor_init(
            8, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t batch_variance = utils::logical_tensor_init(
            9, graph::data_type::f32, graph::layout_type::strided);

    bn_op.add_input(src);
    bn_op.add_input(scale);
    bn_op.add_input(shift);
    bn_op.add_input(mean);
    bn_op.add_input(variance);
    bn_op.add_output(dst);
    bn_op.add_output(running_mean);
    bn_op.add_output(running_variance);
    bn_op.add_output(batch_mean);
    bn_op.add_output(batch_variance);

    graph::graph_t g(engine->kind());
    ASSERT_EQ(g.add_op(&bn_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("bn_fw_train_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src, &scale, &shift, &mean, &variance};
    std::vector<const graph::logical_tensor_t *> outputs {&dst, &running_mean,
            &running_variance, &batch_mean, &batch_variance};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), graph::status::success);

    auto cp_outputs = cp.get_outputs();
    ASSERT_EQ(cp_outputs.size(), outputs.size());

    // Shape Infer Correctness Check
    auto output_dst_dims = ltw(cp_outputs[0]).vdims();
    dims output_dst_dims_ref {1, 1, 4};
    ASSERT_TRUE(std::equal(output_dst_dims.begin(), output_dst_dims.end(),
            output_dst_dims_ref.begin()));
    auto output_r_mean_dims = ltw(cp_outputs[1]).vdims();
    auto output_r_var_dims = ltw(cp_outputs[2]).vdims();
    auto output_b_mean_dims = ltw(cp_outputs[3]).vdims();
    auto output_b_var_dims = ltw(cp_outputs[4]).vdims();
    dims output_mean_var_dims_ref {1};
    ASSERT_TRUE(std::equal(output_r_mean_dims.begin(), output_r_mean_dims.end(),
            output_mean_var_dims_ref.begin()));
    ASSERT_TRUE(std::equal(output_r_var_dims.begin(), output_r_var_dims.end(),
            output_mean_var_dims_ref.begin()));
    ASSERT_TRUE(std::equal(output_b_mean_dims.begin(), output_b_mean_dims.end(),
            output_mean_var_dims_ref.begin()));
    ASSERT_TRUE(std::equal(output_b_var_dims.begin(), output_b_var_dims.end(),
            output_mean_var_dims_ref.begin()));

    auto output_strides = ltw(cp_outputs[0]).vstrides();
    dims output_strides_ref {4, 4, 1};
    ASSERT_TRUE(std::equal(output_strides.begin(), output_strides.end(),
            output_strides_ref.begin()));

    test_tensor_t src_ts(src, engine, src_data);
    test_tensor_t scale_ts(scale, engine, scale_data);
    test_tensor_t shift_ts(shift, engine, shift_data);
    test_tensor_t mean_ts(mean, engine, mean_data);
    test_tensor_t variance_ts(variance, engine, variance_data);
    graph::logical_tensor_t dst_, running_mean_, running_variance_, batch_mean_,
            batch_variance_;
    cp.query_logical_tensor(dst.id, &dst_);
    cp.query_logical_tensor(running_mean.id, &running_mean_);
    cp.query_logical_tensor(running_variance.id, &running_variance_);
    cp.query_logical_tensor(batch_mean.id, &batch_mean_);
    cp.query_logical_tensor(batch_variance.id, &batch_variance_);
    test_tensor_t dst_ts(dst_, engine);
    test_tensor_t running_mean_ts(running_mean_, engine);
    test_tensor_t running_variance_ts(running_variance_, engine);
    test_tensor_t batch_mean_ts(batch_mean_, engine);
    test_tensor_t batch_variance_ts(batch_variance_, engine);

    graph::stream_t *strm = get_stream();
    cp.execute(strm,
            {src_ts.get(), scale_ts.get(), shift_ts.get(), mean_ts.get(),
                    variance_ts.get()},
            {dst_ts.get(), running_mean_ts.get(), running_variance_ts.get(),
                    batch_mean_ts.get(), batch_variance_ts.get()});
    strm->wait();
}

TEST(test_batch_norm_compile, BatchNormForwardTrainingWith0DSpatialInput) {
    SKIP_IF_NV_GPU("not supported on NVIDIA GPU");

    using dims = graph::dnnl_impl::dims;
    using ltw = graph::logical_tensor_wrapper_t;
    graph::engine_t *engine = get_engine();

    // Tensor dimensions.
    const graph::dim_t N = 1, // batch size
            IC = 4; // channels
    dims src_dims = {N, IC};
    dims scale_dims = {IC};
    dims shift_dims = {IC};
    dims mean_dims = {IC};
    dims variance_dims = {IC};

    // Allocate buffers.
    std::vector<float> src_data {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> scale_data {1.f, 1.f, 1.f, 1.f};
    std::vector<float> shift_data {2.f, 2.f, 2.f, 2.f};
    std::vector<float> mean_data {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> variance_data {0.f, 0.f, 0.f, 0.f};

    graph::op_t bn_op(graph::op_kind::BatchNormForwardTraining);
    bn_op.set_attr<float>(graph::op_attr::epsilon, 0.0625);
    bn_op.set_attr<std::string>(graph::op_attr::data_format, "NCX");

    // prepare logical tensor
    graph::logical_tensor_t src = utils::logical_tensor_init(
            0, src_dims, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t scale = utils::logical_tensor_init(
            1, scale_dims, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t shift = utils::logical_tensor_init(
            2, shift_dims, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t mean = utils::logical_tensor_init(
            3, mean_dims, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t variance = utils::logical_tensor_init(4,
            variance_dims, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t dst = utils::logical_tensor_init(
            5, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t running_mean = utils::logical_tensor_init(
            6, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t running_variance = utils::logical_tensor_init(
            7, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t batch_mean = utils::logical_tensor_init(
            8, graph::data_type::f32, graph::layout_type::strided);
    graph::logical_tensor_t batch_variance = utils::logical_tensor_init(
            9, graph::data_type::f32, graph::layout_type::strided);

    bn_op.add_input(src);
    bn_op.add_input(scale);
    bn_op.add_input(shift);
    bn_op.add_input(mean);
    bn_op.add_input(variance);
    bn_op.add_output(dst);
    bn_op.add_output(running_mean);
    bn_op.add_output(running_variance);
    bn_op.add_output(batch_mean);
    bn_op.add_output(batch_variance);

    graph::graph_t g(engine->kind());
    ASSERT_EQ(g.add_op(&bn_op), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("bn_fw_train_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src, &scale, &shift, &mean, &variance};
    std::vector<const graph::logical_tensor_t *> outputs {&dst, &running_mean,
            &running_variance, &batch_mean, &batch_variance};
    ASSERT_EQ(p.compile(&cp, inputs, outputs, engine), graph::status::success);

    auto cp_outputs = cp.get_outputs();
    ASSERT_EQ(cp_outputs.size(), outputs.size());

    // Shape Infer Correctness Check
    auto output_dst_dims = ltw(cp_outputs[0]).vdims();
    dims output_dst_dims_ref {1, 4};
    ASSERT_TRUE(std::equal(output_dst_dims.begin(), output_dst_dims.end(),
            output_dst_dims_ref.begin()));
    auto output_r_mean_dims = ltw(cp_outputs[1]).vdims();
    auto output_r_var_dims = ltw(cp_outputs[2]).vdims();
    auto output_b_mean_dims = ltw(cp_outputs[3]).vdims();
    auto output_b_var_dims = ltw(cp_outputs[4]).vdims();
    dims output_mean_var_dims_ref {4};
    ASSERT_TRUE(std::equal(output_r_mean_dims.begin(), output_r_mean_dims.end(),
            output_mean_var_dims_ref.begin()));
    ASSERT_TRUE(std::equal(output_r_var_dims.begin(), output_r_var_dims.end(),
            output_mean_var_dims_ref.begin()));
    ASSERT_TRUE(std::equal(output_b_mean_dims.begin(), output_b_mean_dims.end(),
            output_mean_var_dims_ref.begin()));
    ASSERT_TRUE(std::equal(output_b_var_dims.begin(), output_b_var_dims.end(),
            output_mean_var_dims_ref.begin()));

    auto output_strides = ltw(cp_outputs[0]).vstrides();
    dims output_strides_ref {4, 1};
    ASSERT_TRUE(std::equal(output_strides.begin(), output_strides.end(),
            output_strides_ref.begin()));

    test_tensor_t src_ts(src, engine, src_data);
    test_tensor_t scale_ts(scale, engine, scale_data);
    test_tensor_t shift_ts(shift, engine, shift_data);
    test_tensor_t mean_ts(mean, engine, mean_data);
    test_tensor_t variance_ts(variance, engine, variance_data);
    graph::logical_tensor_t dst_, running_mean_, running_variance_, batch_mean_,
            batch_variance_;
    cp.query_logical_tensor(dst.id, &dst_);
    cp.query_logical_tensor(running_mean.id, &running_mean_);
    cp.query_logical_tensor(running_variance.id, &running_variance_);
    cp.query_logical_tensor(batch_mean.id, &batch_mean_);
    cp.query_logical_tensor(batch_variance.id, &batch_variance_);
    test_tensor_t dst_ts(dst_, engine);
    test_tensor_t running_mean_ts(running_mean_, engine);
    test_tensor_t running_variance_ts(running_variance_, engine);
    test_tensor_t batch_mean_ts(batch_mean_, engine);
    test_tensor_t batch_variance_ts(batch_variance_, engine);

    graph::stream_t *strm = get_stream();
    cp.execute(strm,
            {src_ts.get(), scale_ts.get(), shift_ts.get(), mean_ts.get(),
                    variance_ts.get()},
            {dst_ts.get(), running_mean_ts.get(), running_variance_ts.get(),
                    batch_mean_ts.get(), batch_variance_ts.get()});
    strm->wait();
}

TEST(test_batch_norm_execute, BatchNormInt8) {
    using dims = graph::dnnl_impl::dims;
    graph::engine_t &engine = *get_engine();
    graph::stream_t &strm = *get_stream();

    // Tensor dimensions.
    const graph::dim_t N = 3, // batch size
            IC = 2, // channels
            IH = 3, // tensor height
            IW = 3; // tensor width
    // Source (src) and destination (dst) tensors dimensions.
    dims src_dims = {N, IH, IW, IC};
    // Scale/shift tensor dimensions.
    dims scale_dims = {IC};

    dims shift_dims = {IC};
    dims mean_dims = {IC};
    dims variance_dims = {IC};

    graph::op_t bn_op(0, graph::op_kind::BatchNormInference, "batchnorm");
    bn_op.set_attr<float>(graph::op_attr::epsilon, 0.000001f);
    bn_op.set_attr<std::string>(graph::op_attr::data_format, "NXC");

    std::vector<int64_t> zps = {0};
    std::vector<float> scales_src = {0.2f};
    std::vector<float> scales_out = {0.3f};

    graph::op_t dequant {1, graph::op_kind::Dequantize, "dequant"};
    dequant.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dequant.set_attr(graph::op_attr::scales, scales_src);
    dequant.set_attr(graph::op_attr::zps, zps);

    graph::op_t quant {2, graph::op_kind::Quantize, "quant"};
    quant.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    quant.set_attr(graph::op_attr::scales, scales_out);
    quant.set_attr(graph::op_attr::zps, zps);

    // prepare logical tensor
    graph::logical_tensor_t src_int8
            = utils::logical_tensor_init(0, src_dims, graph::data_type::s8);
    graph::logical_tensor_t src_f32
            = utils::logical_tensor_init(1, src_dims, graph::data_type::f32);
    graph::logical_tensor_t scale
            = utils::logical_tensor_init(2, scale_dims, graph::data_type::f32);
    graph::logical_tensor_t shift
            = utils::logical_tensor_init(3, shift_dims, graph::data_type::f32);
    graph::logical_tensor_t mean
            = utils::logical_tensor_init(4, mean_dims, graph::data_type::f32);
    graph::logical_tensor_t variance = utils::logical_tensor_init(
            5, variance_dims, graph::data_type::f32);
    graph::logical_tensor_t dst
            = utils::logical_tensor_init(6, src_dims, graph::data_type::f32);
    graph::logical_tensor_t dst_int8_out
            = utils::logical_tensor_init(7, src_dims, graph::data_type::s8);

    scale.property = graph::property_type::constant;
    shift.property = graph::property_type::constant;
    mean.property = graph::property_type::constant;
    variance.property = graph::property_type::constant;

    dequant.add_input(src_int8);
    dequant.add_output(src_f32);
    bn_op.add_input(src_f32);
    bn_op.add_input(scale);
    bn_op.add_input(shift);
    bn_op.add_input(mean);
    bn_op.add_input(variance);
    bn_op.add_output(dst);
    quant.add_input(dst);
    quant.add_output(dst_int8_out);

    graph::graph_t g(engine.kind());
    ASSERT_EQ(g.add_op(&dequant), graph::status::success);
    ASSERT_EQ(g.add_op(&bn_op), graph::status::success);
    ASSERT_EQ(g.add_op(&quant), graph::status::success);
    g.finalize();

    test_tensor_t src_ts(src_int8, &engine);
    src_ts.fill<int8_t>(0, 5);
    test_tensor_t scale_ts(scale, &engine);
    scale_ts.fill<float>();
    test_tensor_t shift_ts(shift, &engine);
    shift_ts.fill<float>();
    test_tensor_t mean_ts(mean, &engine);
    mean_ts.fill<float>();
    test_tensor_t variance_ts(variance, &engine);
    variance_ts.fill<float>();
    test_tensor_t dst_ts(dst_int8_out, &engine);
    test_tensor_t ref_dst_ts(dst_int8_out, &engine);

    ASSERT_EQ(run_graph(g, {src_ts, scale_ts, shift_ts, mean_ts, variance_ts},
                      {ref_dst_ts}, engine, strm),
            graph::status::success);

    graph::pass::pass_base_ptr apass = get_pass("s8f32s8_bnorm");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    ASSERT_EQ((g.get_partitions()[0])->get_kind(),
            graph::partition_kind_t::batch_norm_post_ops);
    ASSERT_EQ(g.get_partitions()[0]->get_inputs().size(), 5U);
    ASSERT_EQ(g.get_partitions()[0]->get_outputs().size(), 1U);

    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src_int8, &scale, &shift, &mean, &variance};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_int8_out};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), graph::status::success);

    cp.execute(&strm,
            {src_ts.get(), scale_ts.get(), shift_ts.get(), mean_ts.get(),
                    variance_ts.get()},
            {dst_ts.get()});
    strm.wait();

    static auto isa = dnnl_get_effective_cpu_isa();

    if (engine.kind() == graph::engine_kind::cpu
            && isa < dnnl_cpu_isa_avx512_core_vnni)
        ASSERT_TRUE(allclose<int8_t>(ref_dst_ts, dst_ts, /*rtol*/ 0.1f,
                /*atol*/ 1.f));
    else
        ASSERT_TRUE(allclose<int8_t>(ref_dst_ts, dst_ts, /*rtol*/ 0.01f,
                /*atol*/ 1.f));
}

TEST(test_batch_norm_execute, BatchNormReluInt8) {
    using dims = graph::dnnl_impl::dims;

    graph::op_t bn_op(0, graph::op_kind::BatchNormInference, "batchnorm");
    bn_op.set_attr<float>(graph::op_attr::epsilon, 0.01f);
    bn_op.set_attr<std::string>(graph::op_attr::data_format, "NXC");

    std::vector<int64_t> zps = {0};
    std::vector<float> scales_src = {2.1f};
    std::vector<float> scales_out = {3.1f};

    graph::op_t dequant {1, graph::op_kind::Dequantize, "dequant"};
    dequant.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    dequant.set_attr(graph::op_attr::scales, scales_src);
    dequant.set_attr(graph::op_attr::zps, zps);

    graph::op_t quant {2, graph::op_kind::Quantize, "quant"};
    quant.set_attr<std::string>(graph::op_attr::qtype, "per_tensor");
    quant.set_attr(graph::op_attr::scales, scales_out);
    quant.set_attr(graph::op_attr::zps, zps);

    graph::op_t relu_op(3, graph::op_kind::ReLU, "relu");

    graph::engine_t &engine = *get_engine();

    // Tensor dimensions.
    const graph::dim_t N = 1, // batch size
            IC = 256, // channels
            IH = 14, // tensor height
            IW = 14; // tensor width
    // Source (src) and destination (dst) tensors dimensions.
    dims src_dims = {N, IH, IW, IC};
    // Scale/shift tensor dimensions.
    dims scale_dims = {IC};
    dims shift_dims = {IC};
    dims mean_dims = {IC};
    dims variance_dims = {IC};

    // prepare logical tensor
    graph::logical_tensor_t src_int8
            = utils::logical_tensor_init(0, src_dims, graph::data_type::s8);
    graph::logical_tensor_t src_f32
            = utils::logical_tensor_init(1, src_dims, graph::data_type::f32);
    graph::logical_tensor_t scale
            = utils::logical_tensor_init(2, scale_dims, graph::data_type::f32);
    graph::logical_tensor_t shift
            = utils::logical_tensor_init(3, shift_dims, graph::data_type::f32);
    graph::logical_tensor_t mean
            = utils::logical_tensor_init(4, mean_dims, graph::data_type::f32);
    graph::logical_tensor_t variance = utils::logical_tensor_init(
            5, variance_dims, graph::data_type::f32);
    graph::logical_tensor_t dst
            = utils::logical_tensor_init(6, src_dims, graph::data_type::f32);
    graph::logical_tensor_t relu_dst
            = utils::logical_tensor_init(7, src_dims, graph::data_type::f32);
    graph::logical_tensor_t dst_int8_out
            = utils::logical_tensor_init(8, src_dims, graph::data_type::s8);

    scale.property = graph::property_type::constant;
    shift.property = graph::property_type::constant;
    mean.property = graph::property_type::constant;
    variance.property = graph::property_type::constant;

    dequant.add_input(src_int8);
    dequant.add_output(src_f32);
    bn_op.add_input(src_f32);
    bn_op.add_input(scale);
    bn_op.add_input(shift);
    bn_op.add_input(mean);
    bn_op.add_input(variance);
    bn_op.add_output(dst);
    relu_op.add_input(dst);
    relu_op.add_output(relu_dst);
    quant.add_input(relu_dst);
    quant.add_output(dst_int8_out);

    graph::graph_t g(engine.kind());
    ASSERT_EQ(g.add_op(&dequant), graph::status::success);
    ASSERT_EQ(g.add_op(&bn_op), graph::status::success);
    ASSERT_EQ(g.add_op(&relu_op), graph::status::success);
    ASSERT_EQ(g.add_op(&quant), graph::status::success);
    g.finalize();

    graph::pass::pass_base_ptr apass = get_pass("s8f32s8_bnorm");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    ASSERT_EQ((g.get_partitions()[0])->get_kind(),
            graph::partition_kind_t::batch_norm_post_ops);
    ASSERT_EQ(g.get_partitions()[0]->get_inputs().size(), 5U);
    ASSERT_EQ(g.get_partitions()[0]->get_outputs().size(), 1U);

    auto part = g.get_partitions()[0];

    // compile
    graph::partition_t p;
    p.init(part);
    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> inputs {
            &src_int8, &scale, &shift, &mean, &variance};
    std::vector<const graph::logical_tensor_t *> outputs {&dst_int8_out};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), graph::status::success);

    test_tensor_t src_ts(src_int8, &engine);
    src_ts.fill<int8_t>(0, 5);
    test_tensor_t scale_ts(scale, &engine);
    scale_ts.fill<float>();
    test_tensor_t shift_ts(shift, &engine);
    shift_ts.fill<float>();
    test_tensor_t mean_ts(mean, &engine);
    mean_ts.fill<float>();
    test_tensor_t variance_ts(variance, &engine);
    variance_ts.fill<float>();
    test_tensor_t dst_ts(dst_int8_out, &engine);
    test_tensor_t ref_dst_ts(dst_int8_out, &engine);

    graph::stream_t &strm = *get_stream();

    ASSERT_EQ(run_graph(g, {src_ts, scale_ts, shift_ts, mean_ts, variance_ts},
                      {ref_dst_ts}, engine, strm),
            graph::status::success);

    cp.execute(&strm,
            {src_ts.get(), scale_ts.get(), shift_ts.get(), mean_ts.get(),
                    variance_ts.get()},
            {dst_ts.get()});
    strm.wait();

    static auto isa = dnnl_get_effective_cpu_isa();
    if (engine.kind() == graph::engine_kind::cpu
            && isa < dnnl_cpu_isa_avx512_core_vnni)
        ASSERT_TRUE(allclose<int8_t>(dst_ts, ref_dst_ts, /*rtol*/ 0.1f,
                /*atol*/ 1.f));
    else
        ASSERT_TRUE(allclose<int8_t>(dst_ts, ref_dst_ts, /*rtol*/ 0.01f,
                /*atol*/ 1.f));
}
