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

/// @example binary.cpp
/// > Annotated version: @ref binary_example_cpp
///
/// @page binary_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute a
/// [Binary](@ref dev_guide_binary) primitive.
///
/// Key optimizations included in this example:
/// - In-place primitive execution;
/// - Primitive attributes with fused post-ops.
///
/// @page binary_example_cpp Binary Primitive Example
/// @copydetails binary_example_cpp_short
///
/// @include binary.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

void binary_example(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim N = 1, // batch size
            IC = 12, // channels
            IH = 128, // tensor height
            IW = 128; // tensor width

    // Source (src_0 and src_1) and destination (dst) tensors dimensions.
    memory::dims src_0_dims = {1, 1, 1, 1};
    memory::dims src_1_dims = {N, IC, IH, IW};
    memory::dims src_2_dims = {1, 1, 1, IW};

    // Allocate buffers.
    std::vector<float> src_0_data(product(src_0_dims));
    std::vector<float> src_1_data(product(src_1_dims));
    std::vector<int8_t> src_2_data(product(src_2_dims));

    // Initialize src_0 and src_1 (src).
    std::generate(src_0_data.begin(), src_0_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });
    std::generate(src_1_data.begin(), src_1_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 2.f);
    });
    for (size_t idx = 0; idx < src_2_data.size(); ++idx) {
        src_2_data[idx] = idx % 2;
    }

    // Create src and dst memory descriptors.
    auto src_0_md = memory::desc(
            src_0_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto src_1_md = memory::desc(
            src_1_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto src_2_md = memory::desc(
            src_2_dims, memory::data_type::s8, memory::format_tag::nchw);
    auto dst_md = memory::desc(
            src_1_dims, memory::data_type::f32, memory::format_tag::nchw);

    // Create src memory objects.
    auto src_0_mem = memory(src_0_md, engine);
    auto src_1_mem = memory(src_1_md, engine);
    auto src_2_mem = memory(src_2_md, engine);
    auto dst_mem = memory(dst_md, engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_0_data.data(), src_0_mem);
    write_to_dnnl_memory(src_1_data.data(), src_1_mem);
    write_to_dnnl_memory(src_2_data.data(), src_2_mem);

    // Create primitive descriptor.
    auto binary_pd = binary::primitive_desc(engine, algorithm::binary_select,
            src_0_md, src_1_md, src_2_md, dst_md);

    // Create the primitive.
    auto binary_prim = binary(binary_pd);

    // Primitive arguments. Set up in-place execution by assigning src_0 as DST.
    std::unordered_map<int, memory> binary_args;
    binary_args.insert({DNNL_ARG_SRC_0, src_0_mem});
    binary_args.insert({DNNL_ARG_SRC_1, src_1_mem});
    binary_args.insert({DNNL_ARG_SRC_2, src_2_mem});
    binary_args.insert({DNNL_ARG_DST, dst_mem});

    // Primitive execution: binary with ReLU.
    binary_prim.execute(engine_stream, binary_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    //     read_from_dnnl_memory(src_0_data.data(), dst_mem);
}

int main(int argc, char **argv) {
    return handle_example_errors(binary_example, parse_engine_kind(argc, argv));
}
