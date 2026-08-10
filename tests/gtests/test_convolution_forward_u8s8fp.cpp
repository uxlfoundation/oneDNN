/*******************************************************************************
* Copyright 2016 Intel Corporation
* Copyright 2026 Arm Ltd. and affiliates
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

#include <algorithm>

#include "oneapi/dnnl/dnnl.hpp"
#include "test_convolution_forward_common.hpp"
#include "tests/test_isa_common.hpp"
namespace dnnl {

using convolution_test
        = convolution_forward_test_t<uint8_t, int8_t, int32_t, float>;

TEST_P(convolution_test, TestConvolution) {}

#define TEST_PARAM_ATTR
#define U8S8
#define DIRECTION_FORWARD
#include "convolution_common.h"
#undef TEST_PARAM_ATTR

#if DNNL_AARCH64
TEST(AArch64MmlaConvolution, Int8SourceZeroPointUsesCurrentWeights) {
    SKIP_IF(get_test_engine_kind() != engine::kind::cpu,
            "This test targets the CPU convolution implementation.");

    constexpr memory::dim mb = 1;
    constexpr memory::dim ic = 64;
    constexpr memory::dim ih = 16;
    constexpr memory::dim iw = 16;
    constexpr memory::dim oc = 64;
    constexpr memory::dim oh = 16;
    constexpr memory::dim ow = 16;
    constexpr memory::dim kh = 3;
    constexpr memory::dim kw = 3;
    constexpr int32_t src_value = 5;
    constexpr int32_t src_zero_point = 3;

    auto eng = get_test_engine();
    auto strm = stream(eng);
    const auto src_md = memory::desc(
            {mb, ic, ih, iw}, memory::data_type::u8, memory::format_tag::nhwc);
    const auto wei_any_md = memory::desc(
            {oc, ic, kh, kw}, memory::data_type::s8, memory::format_tag::any);
    const auto wei_plain_md = memory::desc(
            {oc, ic, kh, kw}, memory::data_type::s8, memory::format_tag::oihw);
    const auto dst_md = memory::desc(
            {mb, oc, oh, ow}, memory::data_type::f32, memory::format_tag::nhwc);
    const auto zero_point_md
            = memory::desc({1}, memory::data_type::s32, memory::format_tag::x);
    const memory::dims strides = {1, 1};
    const memory::dims padding = {1, 1};

    primitive_attr attr;
    attr.set_zero_points_mask(DNNL_ARG_SRC, 0);
    const auto pd = convolution_forward::primitive_desc(eng,
            prop_kind::forward_inference, algorithm::convolution_direct, src_md,
            wei_any_md, memory::desc(), dst_md, strides, padding, padding, attr,
            true);
    SKIP_IF(pd.get() == nullptr
                    || !aarch64_mmla_test::matches_weights_desc(
                            pd.weights_desc(), 1, 0, 8),
            "This test targets AArch64 SVE-I8MM.");
    const auto conv = convolution_forward(pd);

    auto src = memory(src_md, eng);
    auto *src_ptr = src.map_data<uint8_t>();
    ASSERT_NE(src_ptr, nullptr);
    std::fill(src_ptr, src_ptr + mb * ic * ih * iw, src_value);
    src.unmap_data(src_ptr);

    auto zero_point = memory(zero_point_md, eng);
    auto *zero_point_ptr = zero_point.map_data<int32_t>();
    ASSERT_NE(zero_point_ptr, nullptr);
    zero_point_ptr[0] = src_zero_point;
    zero_point.unmap_data(zero_point_ptr);

    auto wei_plain = memory(wei_plain_md, eng);
    auto weights = memory(pd.weights_desc(), eng);
    auto dst = memory(dst_md, eng);
    for (int8_t weight_value : {int8_t(1), int8_t(-2)}) {
        auto *wei_plain_ptr = wei_plain.map_data<int8_t>();
        ASSERT_NE(wei_plain_ptr, nullptr);
        std::fill(
                wei_plain_ptr, wei_plain_ptr + oc * ic * kh * kw, weight_value);
        wei_plain.unmap_data(wei_plain_ptr);

        reorder(wei_plain, weights).execute(strm, wei_plain, weights);
        conv.execute(strm,
                {{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, weights},
                        {DNNL_ARG_DST, dst},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC,
                                zero_point}});
        strm.wait();

        auto *dst_ptr = dst.map_data<float>();
        ASSERT_NE(dst_ptr, nullptr);
        for (memory::dim output_h = 0; output_h < oh; ++output_h) {
            const int valid_kernel_h
                    = output_h == 0 || output_h == oh - 1 ? 2 : 3;
            for (memory::dim output_w = 0; output_w < ow; ++output_w) {
                const int valid_kernel_w
                        = output_w == 0 || output_w == ow - 1 ? 2 : 3;
                const auto expected = static_cast<float>(
                        (src_value - src_zero_point) * weight_value * ic
                        * valid_kernel_h * valid_kernel_w);
                for (memory::dim output_channel = 0; output_channel < oc;
                        ++output_channel) {
                    const auto offset
                            = (output_h * ow + output_w) * oc + output_channel;
                    EXPECT_EQ(dst_ptr[offset], expected)
                            << "weight=" << static_cast<int>(weight_value)
                            << ", index=" << offset;
                }
            }
        }
        dst.unmap_data(dst_ptr);
    }
}
#endif

} // namespace dnnl
