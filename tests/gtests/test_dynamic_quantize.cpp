/*******************************************************************************
* Copyright 2026 Advanced Micro Devices, Inc.
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

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <unordered_map>

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {

using tag = memory::format_tag;
using dt = memory::data_type;

struct dynamic_quantize_params_t {
    dt src_dt;
    memory::dims src_dims;
    memory::dims scale_dims;
    bool compute_only;
};

struct quant_recipe_t {
    int mask;
    memory::dims groups;
};

static quant_recipe_t make_recipe(
        const memory::dims &src_dims, const memory::dims &scale_dims) {
    const memory::dim M = src_dims[0], N = src_dims[1];
    const memory::dim sm = scale_dims[0], sn = scale_dims[1];
    if (sm == 1 && sn == 1) return {0, {}};
    if (sm == M && sn == 1) return {1 << 0, {}};
    if (sm == 1 && sn == N) return {1 << 1, {}};
    if (sm == M && N % sn == 0) return {(1 << 0) | (1 << 1), {1, N / sn}};
    if (sn == N && M % sm == 0) return {(1 << 0) | (1 << 1), {M / sm, 1}};
    return {-1, {}};
}

static primitive_attr make_attr(
        const memory::dims &src_dims, const memory::dims &scale_dims) {
    const auto recipe = make_recipe(src_dims, scale_dims);
    primitive_attr attr;
    attr.set_scales(DNNL_ARG_DST, recipe.mask, recipe.groups, dt::f32, false,
            quantization_mode::dynamic_fp);
    return attr;
}

class dynamic_quantize_reduction_test_t
    : public ::testing::TestWithParam<dynamic_quantize_params_t> {
protected:
    engine eng = get_test_engine();
    stream strm = make_stream(eng);

    static std::vector<float> to_f32(const memory &mem) {
        const auto md = mem.get_desc();
        const memory::dim nelems = static_cast<memory::dim>(
                md.get_size() / memory::data_type_size(md.get_data_type()));

        std::vector<float> out(nelems);
        switch (md.get_data_type()) {
            case dt::f32: {
                auto mapped = map_memory<float>(mem);
                for (memory::dim i = 0; i < nelems; ++i)
                    out[i] = static_cast<float *>(mapped)[i];
            } break;
            case dt::s8: {
                auto mapped = map_memory<int8_t>(mem);
                for (memory::dim i = 0; i < nelems; ++i)
                    out[i] = static_cast<float>(
                            static_cast<int8_t *>(mapped)[i]);
            } break;
            default: assert(!"unexpected data type");
        }
        return out;
    }

    void run(const dynamic_quantize_params_t &p) {
        SKIP_IF(get_test_engine_kind() == engine::kind::gpu,
                "GPU engine is not supported.");
        SKIP_IF(unsupported_data_type(p.src_dt),
                "Engine does not support this data type.");

        const memory::dim M = p.src_dims[0], N = p.src_dims[1];
        const memory::dim sm = p.scale_dims[0], sn = p.scale_dims[1];

        std::vector<float> src_ref(M * N);
        for (memory::dim m = 0; m < M; ++m)
            for (memory::dim n = 0; n < N; ++n)
                src_ref[m * N + n]
                        = 0.37f * static_cast<float>((m + 1) * (n - N / 2))
                        - 0.11f * static_cast<float>(n);

        const auto src_f32_md = memory::desc({M, N}, dt::f32, tag::ab);
        const auto src_md = memory::desc({M, N}, p.src_dt, tag::ab);
        auto src_f32_mem = memory(src_f32_md, eng);
        {
            auto mapped = map_memory<float>(src_f32_mem);
            for (memory::dim i = 0; i < M * N; ++i)
                static_cast<float *>(mapped)[i] = src_ref[i];
        }

        auto src_mem = memory(src_md, eng);
        reorder(src_f32_mem, src_mem).execute(strm, src_f32_mem, src_mem);
        auto src_back = memory(src_f32_md, eng);
        reorder(src_mem, src_back).execute(strm, src_mem, src_back);
        strm.wait();
        const auto src_seen = to_f32(src_back);

        const auto scale_md = memory::desc(p.scale_dims, dt::f32, tag::ab);
        const auto dst_md = p.compute_only
                ? memory::desc()
                : memory::desc({M, N}, dt::s8, tag::ab);
        const auto attr = make_attr(p.src_dims, p.scale_dims);

        reduction::primitive_desc pd;
        ASSERT_NO_THROW(pd = reduction::primitive_desc(eng,
                                algorithm::reduction_dynamic_quantize, src_md,
                                dst_md, 0.f, 0.f, attr));
        ASSERT_TRUE(pd.query_md(query::exec_arg_md,
                            DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST)
                == scale_md);

        auto scale_mem = memory(scale_md, eng);
        memory dst_mem;
        if (!p.compute_only) dst_mem = memory(dst_md, eng);

        std::unordered_map<int, memory> args {
                {DNNL_ARG_SRC, src_mem},
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, scale_mem},
        };
        if (!p.compute_only) args.insert({DNNL_ARG_DST, dst_mem});

        reduction(pd).execute(strm, args);
        strm.wait();

        const auto scale_out = to_f32(scale_mem);
        const auto dst_out
                = p.compute_only ? std::vector<float>() : to_f32(dst_mem);

        const memory::dim rows_per_group = M / sm;
        const memory::dim cols_per_group = N / sn;
        for (memory::dim gm = 0; gm < sm; ++gm) {
            for (memory::dim gn = 0; gn < sn; ++gn) {
                const auto m0 = gm * rows_per_group;
                const auto m1 = m0 + rows_per_group;
                const auto n0 = gn * cols_per_group;
                const auto n1 = n0 + cols_per_group;
                float absmax = 0.f;
                for (memory::dim m = m0; m < m1; ++m) {
                    for (memory::dim n = n0; n < n1; ++n) {
                        const float value = src_seen[m * N + n];
                        absmax = std::max(absmax, std::abs(value));
                    }
                }

                constexpr float scale_eps = 1.0e-30f;
                const float expected_scale
                        = std::max(absmax / 127.f, scale_eps);
                const auto scale_index = gm * sn + gn;
                ASSERT_NEAR(scale_out[scale_index], expected_scale,
                        1e-4f * std::max(1e-3f, expected_scale));

                if (p.compute_only) continue;
                for (memory::dim m = m0; m < m1; ++m) {
                    for (memory::dim n = n0; n < n1; ++n) {
                        const float value = src_seen[m * N + n];
                        float expected = std::nearbyint(value / expected_scale);
                        expected = std::max(-127.f, std::min(127.f, expected));
                        ASSERT_LE(std::abs(dst_out[m * N + n] - expected), 1.f);
                    }
                }
            }
        }
    }
};

TEST_P(dynamic_quantize_reduction_test_t, Correctness) {
    run(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Optimized, dynamic_quantize_reduction_test_t,
        ::testing::Values(
                dynamic_quantize_params_t {dt::f32, {8, 128}, {8, 1}, false},
                dynamic_quantize_params_t {dt::bf16, {4, 256}, {4, 8}, false},
                dynamic_quantize_params_t {dt::f16, {5, 80}, {5, 1}, false}));

INSTANTIATE_TEST_SUITE_P(Reference, dynamic_quantize_reduction_test_t,
        ::testing::Values(
                dynamic_quantize_params_t {dt::f32, {4, 6}, {1, 1}, false},
                dynamic_quantize_params_t {dt::f32, {4, 6}, {1, 6}, false},
                dynamic_quantize_params_t {dt::f32, {4, 6}, {2, 6}, false},
                dynamic_quantize_params_t {dt::f32, {8, 32}, {8, 1}, true}));

TEST(dynamic_quantize_reduction_test_t, NonFiniteVectorLanes) {
    const auto eng = get_test_engine();
    SKIP_IF(get_test_engine_kind() == engine::kind::gpu,
            "GPU engine is not supported.");
    auto strm = make_stream(eng);
    constexpr memory::dim M = 3;
    constexpr memory::dim N = 35;
    std::vector<float> src_data(M * N);
    for (memory::dim n = 0; n < N; ++n) {
        src_data[n] = static_cast<float>(n - 17);
        src_data[N + n] = 5.f;
        src_data[2 * N + n] = -5.f;
    }
    src_data[0] = std::numeric_limits<float>::quiet_NaN();
    src_data[1] = std::numeric_limits<float>::infinity();

    const auto src_md = memory::desc({M, N}, dt::f32, tag::ab);
    const auto dst_md = memory::desc({M, N}, dt::s8, tag::ab);
    const auto scale_md = memory::desc({M, 1}, dt::f32, tag::ab);
    const auto attr = make_attr({M, N}, {M, 1});
    auto src_mem = memory(src_md, eng, src_data.data());
    auto dst_mem = memory(dst_md, eng);
    auto scale_mem = memory(scale_md, eng);

    const auto pd = reduction::primitive_desc(eng,
            algorithm::reduction_dynamic_quantize, src_md, dst_md, 0.f, 0.f,
            attr);
    reduction(pd).execute(strm,
            {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, scale_mem}});
    strm.wait();

    auto dst = map_memory<int8_t>(dst_mem);
    ASSERT_EQ(static_cast<int8_t *>(dst)[0], 0);
    ASSERT_EQ(static_cast<int8_t *>(dst)[1], 0);
    ASSERT_EQ(static_cast<int8_t *>(dst)[N], 127);
    ASSERT_EQ(static_cast<int8_t *>(dst)[2 * N - 1], 127);
    ASSERT_EQ(static_cast<int8_t *>(dst)[2 * N], -127);
    ASSERT_EQ(static_cast<int8_t *>(dst)[3 * N - 1], -127);

    auto scales = map_memory<float>(scale_mem);
    ASSERT_NEAR(static_cast<float *>(scales)[0], 17.f / 127.f, 1e-6f);
    ASSERT_NEAR(static_cast<float *>(scales)[1], 5.f / 127.f, 1e-6f);
    ASSERT_NEAR(static_cast<float *>(scales)[2], 5.f / 127.f, 1e-6f);
}

TEST(dynamic_quantize_reduction_test_t, ConstantSameSignGroups) {
    const auto eng = get_test_engine();
    SKIP_IF(get_test_engine_kind() == engine::kind::gpu,
            "GPU engine is not supported.");
    auto strm = make_stream(eng);

    constexpr memory::dim M = 1;
    constexpr memory::dim N = 36;
    constexpr memory::dim G = 2;
    std::vector<float> src_data(N, 5.f);
    std::fill(src_data.begin() + N / 2, src_data.end(), -5.f);

    const auto src_md = memory::desc({M, N}, dt::f32, tag::ab);
    const auto dst_md = memory::desc({M, N}, dt::s8, tag::ab);
    const auto scale_md = memory::desc({M, G}, dt::f32, tag::ab);
    const auto attr = make_attr({M, N}, {M, G});
    const auto pd = reduction::primitive_desc(eng,
            algorithm::reduction_dynamic_quantize, src_md, dst_md, 0.f, 0.f,
            attr);

    auto src_mem = memory(src_md, eng, src_data.data());
    auto dst_mem = memory(dst_md, eng);
    auto scale_mem = memory(scale_md, eng);
    reduction(pd).execute(strm,
            {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, scale_mem}});
    strm.wait();

    auto scales = map_memory<float>(scale_mem);
    auto dst = map_memory<int8_t>(dst_mem);
    for (memory::dim g = 0; g < G; ++g)
        ASSERT_NEAR(static_cast<float *>(scales)[g], 5.f / 127.f, 1e-6f);
    for (memory::dim n = 0; n < N / 2; ++n)
        ASSERT_EQ(static_cast<int8_t *>(dst)[n], 127);
    for (memory::dim n = N / 2; n < N; ++n)
        ASSERT_EQ(static_cast<int8_t *>(dst)[n], -127);
}

TEST(dynamic_quantize_reduction_test_t, ExtremeFiniteRange) {
    const auto eng = get_test_engine();
    SKIP_IF(get_test_engine_kind() == engine::kind::gpu,
            "GPU engine is not supported.");
    auto strm = make_stream(eng);

    constexpr memory::dim M = 1;
    constexpr memory::dim N = 32;
    const float min_val = std::numeric_limits<float>::lowest();
    const float max_val = std::numeric_limits<float>::max();
    std::vector<float> src_data(N);
    for (memory::dim n = 0; n < N; ++n)
        src_data[n] = n % 2 == 0 ? min_val : max_val;

    const auto src_md = memory::desc({M, N}, dt::f32, tag::ab);
    const auto dst_md = memory::desc({M, N}, dt::s8, tag::ab);
    const auto scale_md = memory::desc({M, 1}, dt::f32, tag::ab);
    const auto attr = make_attr({M, N}, {M, 1});
    const auto pd = reduction::primitive_desc(eng,
            algorithm::reduction_dynamic_quantize, src_md, dst_md, 0.f, 0.f,
            attr);

    auto src_mem = memory(src_md, eng, src_data.data());
    auto dst_mem = memory(dst_md, eng);
    auto scale_mem = memory(scale_md, eng);
    reduction(pd).execute(strm,
            {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, scale_mem}});
    strm.wait();

    const float expected_scale = max_val / 127.f;
    auto scales = map_memory<float>(scale_mem);
    auto dst = map_memory<int8_t>(dst_mem);
    ASSERT_TRUE(std::isfinite(static_cast<float *>(scales)[0]));
    ASSERT_FLOAT_EQ(static_cast<float *>(scales)[0], expected_scale);
    for (memory::dim n = 0; n < N; ++n)
        ASSERT_EQ(static_cast<int8_t *>(dst)[n], n % 2 == 0 ? -127 : 127);
}

TEST(dynamic_quantize_reduction_test_t, RequiresDynamicScaleOutput) {
    const auto eng = get_test_engine();
    SKIP_IF(get_test_engine_kind() == engine::kind::gpu,
            "GPU engine is not supported.");
    auto strm = make_stream(eng);

    constexpr memory::dim M = 2;
    constexpr memory::dim N = 16;
    const auto src_md = memory::desc({M, N}, dt::f32, tag::ab);
    const auto dst_md = memory::desc({M, N}, dt::s8, tag::ab);
    const auto attr = make_attr({M, N}, {M, 1});
    const auto pd = reduction::primitive_desc(eng,
            algorithm::reduction_dynamic_quantize, src_md, dst_md, 0.f, 0.f,
            attr);
    const auto prim = reduction(pd);

    auto src_mem = memory(src_md, eng);
    auto dst_mem = memory(dst_md, eng);
    EXPECT_ANY_THROW(prim.execute(
            strm, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}}));
}

TEST(dynamic_quantize_reduction_validation_t, Errors) {
    const auto eng = get_test_engine();
    const auto src = memory::desc({8, 16}, dt::f32, tag::ab);
    const auto s8_dst = memory::desc({8, 16}, dt::s8, tag::ab);
    const auto invalid_dst = memory::desc({8, 16}, dt::u8, tag::ab);

    EXPECT_ANY_THROW(reduction::primitive_desc(
            eng, algorithm::reduction_dynamic_quantize, src, s8_dst, 0.f, 0.f));

    primitive_attr static_attr;
    static_attr.set_scales(DNNL_ARG_DST, 1 << 0, {}, dt::f32);
    EXPECT_ANY_THROW(reduction::primitive_desc(eng,
            algorithm::reduction_dynamic_quantize, src, s8_dst, 0.f, 0.f,
            static_attr));

    const auto dynamic_attr = make_attr({8, 16}, {8, 1});
    EXPECT_ANY_THROW(reduction::primitive_desc(eng,
            algorithm::reduction_dynamic_quantize, src, invalid_dst, 0.f, 0.f,
            dynamic_attr));

    EXPECT_ANY_THROW(reduction::primitive_desc(
            eng, algorithm::reduction_sum, src, src, 0.f, 0.f));
}

} // namespace dnnl
