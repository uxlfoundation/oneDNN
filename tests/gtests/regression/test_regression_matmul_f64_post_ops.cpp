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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {

// Test for f64 matmul post-ops accuracy. Before the fix, post-ops used f32
// type even for f64, causing precision loss.
class test_regression_matmul_f64_post_ops_t : public ::testing::Test {
    engine eng;

protected:
    void SetUp() override {
        SKIP_IF_CUDA(true, "Unsupported test for CUDA.");
        SKIP_IF_HIP(true, "Unsupported test for HIP.");
        SKIP_IF_GENERIC(true, "Unsupported test for generic GPU.");
        SKIP_IF(engine::get_count(engine::kind::gpu) == 0,
                "GPU engine not found.");
        eng = engine(engine::kind::gpu, 0);
    }

    void run_matmul(const post_ops &po, double src_val, double wei_val,
            double dst_val, double expected, bool use_ref = false) {
        memory::desc src_md(
                {1, 1}, memory::data_type::f64, memory::format_tag::ab);
        memory::desc wei_md(
                {1, 1}, memory::data_type::f64, memory::format_tag::ab);
        memory::desc dst_md(
                {1, 1}, memory::data_type::f64, memory::format_tag::ab);

        primitive_attr attr;
        attr.set_post_ops(po);

        matmul::primitive_desc pd;
        try {
            pd = matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr);
        } catch (const dnnl::error &e) {
            SKIP_IF(e.status == dnnl_unimplemented,
                    "f64 matmul with post-ops is not supported.");
            throw;
        }

        if (use_ref) {
            while (std::string(pd.impl_info_str()).find("ref")
                    == std::string::npos) {
                SKIP_IF(!pd.next_impl(),
                        "ref implementation not found for f64 matmul.");
            }
        }

        auto prim = matmul(pd);

        auto src_mem = test::make_memory(src_md, eng);
        auto wei_mem = test::make_memory(wei_md, eng);
        auto dst_mem = test::make_memory(dst_md, eng);

        {
            auto src_ptr = map_memory<double>(src_mem);
            auto wei_ptr = map_memory<double>(wei_mem);
            auto dst_ptr = map_memory<double>(dst_mem);
            src_ptr[0] = src_val;
            wei_ptr[0] = wei_val;
            dst_ptr[0] = dst_val;
        }

        stream strm(eng);
        prim.execute(strm,
                {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, wei_mem},
                        {DNNL_ARG_DST, dst_mem}});
        strm.wait();

        auto dst_ptr = map_memory<double>(dst_mem);
        ASSERT_EQ(dst_ptr[0], expected);
    }
};

TEST_F(test_regression_matmul_f64_post_ops_t, TestRelu) {
    post_ops po;
    po.append_eltwise(algorithm::eltwise_relu, 0, 0);

    double expected = 16777217.0;
    run_matmul(po, 16777217.0, 1.0, 0.0, expected);
}

TEST_F(test_regression_matmul_f64_post_ops_t, TestSum) {
    post_ops po;
    po.append_eltwise(algorithm::eltwise_relu, 0, 0);
    po.append_sum();

    double expected = 16777218.0;
    run_matmul(po, 1.0, 1.0, 16777217.0, expected, true);
}

} // namespace dnnl
