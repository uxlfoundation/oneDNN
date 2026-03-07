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
protected:
    void SetUp() override {
        SKIP_IF_CUDA(true, "Unsupported test for CUDA.");
        SKIP_IF_HIP(true, "Unsupported test for HIP.");
        SKIP_IF_GENERIC(true, "Unsupported test for generic GPU.");
        SKIP_IF(engine::get_count(engine::kind::gpu) == 0,
                "GPU engine not found.");
        Test();
    }

    void Test() {
        memory::desc src_md(
                {1, 1}, memory::data_type::f64, memory::format_tag::ab);
        memory::desc wei_md(
                {1, 1}, memory::data_type::f64, memory::format_tag::ab);
        memory::desc dst_md(
                {1, 1}, memory::data_type::f64, memory::format_tag::ab);

        post_ops po;
        po.append_eltwise(algorithm::eltwise_relu, 0, 0);
        primitive_attr attr;
        attr.set_post_ops(po);

        auto eng = engine(engine::kind::gpu, 0);

        matmul::primitive_desc pd;
        try {
            pd = matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr);
        } catch (const dnnl::error &e) {
            SKIP_IF(e.status == dnnl_unimplemented,
                    "f64 matmul with post-ops is not supported.");
            throw;
        }
        auto prim = matmul(pd);

        auto src_mem = test::make_memory(src_md, eng);
        auto wei_mem = test::make_memory(wei_md, eng);
        auto dst_mem = test::make_memory(dst_md, eng);

        // 16777217.0 = 2^24 + 1: exact in double, rounds to 2^24 in float.
        const double src_val = 16777217.0;
        const double wei_val = 1.0;
        {
            auto src_ptr = map_memory<double>(src_mem);
            auto wei_ptr = map_memory<double>(wei_mem);
            auto dst_ptr = map_memory<double>(dst_mem);
            src_ptr[0] = src_val;
            wei_ptr[0] = wei_val;
            dst_ptr[0] = 0.0;
        }

        stream strm(eng);
        prim.execute(strm,
                {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, wei_mem},
                        {DNNL_ARG_DST, dst_mem}});
        strm.wait();

        const double expected = src_val * wei_val;
        double result;
        {
            auto dst_ptr = map_memory<double>(dst_mem);
            result = dst_ptr[0];
        }
        ASSERT_EQ(result, expected);
    }
};

TEST_F(test_regression_matmul_f64_post_ops_t, TestF64PostOpAccuracy) {}

} // namespace dnnl
