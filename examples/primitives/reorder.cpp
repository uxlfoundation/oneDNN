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

/// @example reorder.cpp
/// > Annotated version: @ref reorder_example_cpp
///
/// @page reorder_example_cpp_short
///
/// This C++ API demonstrates how to create and execute a
/// [Reorder](@ref dev_guide_reorder) primitive.
///
/// Key optimizations included in this example:
/// - Primitive attributes for output scaling.
///
/// @page reorder_example_cpp Reorder Primitive Example
/// @copydetails reorder_example_cpp_short
///
/// @include reorder.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

void reorder_example(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim IH = 4, IW = 8;

    // Source (src) and destination (dst) tensors dimensions.
    memory::dims src_dims = {IH, IW};
    memory::dims src_strides = {16, 1};

    // Allocate buffers.
    std::vector<float> src_data(64);
    std::vector<float> dst_data(64);

    // Initialize src tensor.
    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return i++ % 8;
    });

    // Create memory descriptors and memory objects for src and dst.
    auto src_md = memory::desc(src_dims, memory::data_type::f32, src_strides);
    auto dst_md = memory::desc(src_dims, memory::data_type::f32, src_strides);

    auto src_mem = memory(src_md, engine);
    auto dst_mem = memory(dst_md, engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_data.data(), src_mem);

    // Create primitive descriptor.
    auto reorder_pd = reorder::primitive_desc(engine, src_md, engine, dst_md);

    // Create the primitive.
    auto reorder_prim = reorder(reorder_pd);

    // Primitive arguments.
    std::unordered_map<int, memory> reorder_args;
    reorder_args.insert({DNNL_ARG_SRC, src_mem});
    reorder_args.insert({DNNL_ARG_DST, dst_mem});

    // Primitive execution: reorder with scaled sum.
    reorder_prim.execute(engine_stream, reorder_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_data.data(), dst_mem);

    for (size_t idx = 0; idx < src_data.size(); ++idx)
        std::cout << src_data[idx] << ",";
    std::cout << std::endl;
    for (size_t idx = 0; idx < dst_data.size(); ++idx)
        std::cout << dst_data[idx] << ",";
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    return handle_example_errors(
            reorder_example, parse_engine_kind(argc, argv));
}
