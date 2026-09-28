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

/// @example cpu_reduction_dynamic_quantize.cpp
/// > Annotated version: @ref reduction_dynamic_quantize_example_cpp

/// @page reduction_dynamic_quantize_example_cpp Reduction Dynamic Quantization
/// This C++ API example demonstrates fused symmetric per-row dynamic
/// quantization through the reduction primitive.
/// @include cpu_reduction_dynamic_quantize.cpp

#include <vector>
#include <unordered_map>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

void reduction_dynamic_quantize_example(engine::kind engine_kind) {
    engine eng(engine_kind, 0);
    stream strm(eng);

    constexpr memory::dim M = 2;
    constexpr memory::dim N = 8;
    const memory::dims src_dims {M, N};
    const memory::dims scale_dims {M, 1};

    std::vector<float> src_data {
            -4.f,
            -3.f,
            -2.f,
            -1.f,
            0.f,
            1.f,
            2.f,
            3.f,
            -1.f,
            -.5f,
            0.f,
            .5f,
            1.f,
            1.5f,
            2.f,
            2.5f,
    };
    std::vector<int8_t> dst_data(M * N);
    std::vector<float> scales(M);

    const auto src_md = memory::desc(
            src_dims, memory::data_type::f32, memory::format_tag::ab);
    const auto dst_md = memory::desc(
            src_dims, memory::data_type::s8, memory::format_tag::ab);
    const auto scale_md = memory::desc(
            scale_dims, memory::data_type::f32, memory::format_tag::ab);

    auto src_mem = memory(src_md, eng);
    auto dst_mem = memory(dst_md, eng);
    auto scale_mem = memory(scale_md, eng);
    write_to_dnnl_memory(src_data.data(), src_mem);

    primitive_attr attr;
    attr.set_scales(DNNL_ARG_DST, 1 << 0, {}, memory::data_type::f32, false,
            quantization_mode::dynamic_fp);

    const auto pd = reduction::primitive_desc(eng,
            algorithm::reduction_dynamic_quantize, src_md, dst_md, 0.f, 0.f,
            attr);
    const auto prim = reduction(pd);

    prim.execute(strm,
            {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, scale_mem}});
    strm.wait();

    read_from_dnnl_memory(dst_data.data(), dst_mem);
    read_from_dnnl_memory(scales.data(), scale_mem);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            reduction_dynamic_quantize_example, parse_engine_kind(argc, argv));
}
