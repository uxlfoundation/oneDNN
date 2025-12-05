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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

#include <cstdlib>
#include <iostream>

extern "C" const char *dnnl_impl_gpu_intel_get_isa_name(dnnl_engine_t engine);

namespace dnnl {

struct conv_params_t {
    memory::dim MB, IC, OC, IH, OH, KH;
};

// This is a regression test for PVC write-after-read hardware bug workaround.
// The bug manifests as a page fault caused by WAR-related corruption of the
// send header. The test doesn't include any validation, the expected failure
// is a GPU page fault and test segfault.
class test_regression_conv_pvc_war_t
    : public ::testing::TestWithParam<conv_params_t> {
    conv_params_t params;
    engine eng;

protected:
    void SetUp() override {
        SKIP_IF_CUDA(true, "Unsupported test for CUDA.");
        SKIP_IF_HIP(true, "Unsupported test for HIP.");
        SKIP_IF_GENERIC(true, "Unsupported test for generic GPU.");
        SKIP_IF(engine::get_count(engine::kind::gpu) == 0,
                "GPU engine not found.");

        eng = engine(engine::kind::gpu, 0);
        SKIP_IF(dnnl_impl_gpu_intel_get_isa_name(eng.get())
                        != std::string("xe_hpc"),
                "Test is for PVC only");

        params = ::testing::TestWithParam<decltype(params)>::GetParam();
        Test();
    }

    void Test() {
        memory::dims src_dims = {params.MB, params.IC, params.IH, params.IH};
        memory::dims wei_dims = {params.OC, params.IC, params.KH, params.KH};
        memory::dims dst_dims = {params.MB, params.OC, params.OH, params.OH};
        memory::dims strides = {1, 1};
        memory::dims padding = {1, 1};

        memory::desc src_md(
                src_dims, memory::data_type::f64, memory::format_tag::nchw);
        memory::desc wei_md(
                wei_dims, memory::data_type::f64, memory::format_tag::oihw);
        memory::desc dst_md(
                dst_dims, memory::data_type::f64, memory::format_tag::nchw);

        convolution_forward::primitive_desc hint_fwd_pd(eng,
                prop_kind::forward_training, algorithm::convolution_direct,
                src_md, wei_md, dst_md, strides, padding, padding);
        convolution_backward_data::primitive_desc pd(eng,
                algorithm::convolution_direct, src_md, wei_md, dst_md, strides,
                padding, padding, hint_fwd_pd);
        convolution_backward_data prim(pd);

        auto dst_mem_desc = pd.diff_dst_desc();
        auto wei_mem_desc = pd.weights_desc();
        auto src_mem_desc = pd.diff_src_desc();

        memory dst_mem(dst_mem_desc, eng);
        memory wei_mem(wei_mem_desc, eng);
        memory src_mem(src_mem_desc, eng);

        stream strm(eng);

        const int REPEATS = 10;
        for (int i = 0; i < REPEATS; ++i) {
            prim.execute(strm,
                    {{DNNL_ARG_DIFF_DST, dst_mem}, {DNNL_ARG_WEIGHTS, wei_mem},
                            {DNNL_ARG_DIFF_SRC, src_mem}});
        }
        strm.wait();
    }
};

TEST_P(test_regression_conv_pvc_war_t, Tests) {}

GPU_INSTANTIATE_TEST_SUITE_P(All, test_regression_conv_pvc_war_t,
        ::testing::Values(conv_params_t {8, 32, 64, 128, 128, 3},
                conv_params_t {1, 64, 64, 256, 256, 3}));

} // namespace dnnl
