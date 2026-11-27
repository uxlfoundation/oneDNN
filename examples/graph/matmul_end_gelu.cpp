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

/// @example matmul_end_gelu.cpp
/// > Annotated version: @ref graph_matmul_end_gelu_cpp

/// @page graph_matmul_end_gelu_cpp_brief
/// @brief This example demonstrates how to fuse MatMul and GELU into one
/// partition while also getting the MatMul output as a partition output.

/// @page graph_matmul_end_gelu_cpp MatMul + End + GELU fusion
/// \copybrief graph_matmul_end_gelu_cpp_brief
///
/// > Example code: @ref matmul_end_gelu.cpp
///
/// The graph built in this example:
///
///         src   wei
///           \   /
///           MatMul
///             |
///             +------ End   (marks MatMul output as a partition output)
///             |
///            GELU
///             |
///            dst
///
/// Without the End op, the MatMul output is an intermediate tensor of the
/// fused partition and is not visible to users. With the End op, the library
/// still returns a single partition, and the MatMul output is materialized as
/// an additional output of the partition.

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

#include "graph_example_utils.hpp"

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;

void matmul_end_gelu_example(dnnl::engine::kind ekind) {
    const dim M = 32, K = 256, N = 128;
    const dims src_dims = {M, K};
    const dims wei_dims = {K, N};
    const dims dst_dims = {M, N};

    size_t id = 0;
    auto src = logical_tensor(
            id++, data_type::f32, src_dims, layout_type::strided);
    auto wei = logical_tensor(
            id++, data_type::f32, wei_dims, layout_type::strided);
    auto mm_out = logical_tensor(
            id++, data_type::f32, dst_dims, layout_type::strided);
    auto dst = logical_tensor(
            id++, data_type::f32, dst_dims, layout_type::strided);

    auto matmul = op(id++, op::kind::MatMul, "matmul");
    matmul.add_inputs({src, wei});
    matmul.add_outputs({mm_out});

    // End consumes the MatMul output to keep it as a partition output.
    auto end = op(id++, op::kind::End, "end");
    end.add_input(mm_out);

    auto gelu = op(id++, op::kind::GELU, "gelu");
    gelu.add_input(mm_out);
    gelu.add_output(dst);

    graph g(ekind);
    g.add_op(matmul);
    g.add_op(end);
    g.add_op(gelu);
    g.finalize();

    auto partitions = g.get_partitions();
    if (partitions.size() != 1 || !partitions[0].is_supported()) {
        throw std::runtime_error("expected a single supported partition, got "
                + std::to_string(partitions.size()));
    }
    const partition &part = partitions[0];

    allocator alloc = create_allocator(ekind);
    dnnl::engine eng = make_engine_with_allocator(ekind, 0, alloc);
    dnnl::stream strm(eng);

    // compile the partition with the input and output logical tensors
    compiled_partition cp = part.compile({src, wei}, {mm_out, dst}, eng);

    auto scratchpad_lt = cp.get_scratchpad_logical_tensor();
    tensor scratchpad_ts;
    if (scratchpad_lt.get_mem_size() > 0) {
        scratchpad_ts = tensor(scratchpad_lt, eng);
    }

    // fill input tensors with real user data
    std::vector<float> src_data(product(src_dims), 1.f);
    std::vector<float> wei_data(product(wei_dims), 0.5f);
    auto ts_src = tensor(src, eng);
    write_to_dnnl_tensor(src_data.data(), ts_src);
    auto ts_wei = tensor(wei, eng);
    write_to_dnnl_tensor(wei_data.data(), ts_wei);

    // prepare output tensors. One can bind user-allocated buffer to them.
    auto ts_mm_out = tensor(mm_out, eng);
    auto ts_dst = tensor(dst, eng);

    // execute the partition with the input and output tensors
    cp.execute(strm, {ts_src, ts_wei}, {ts_mm_out, ts_dst});
    strm.wait();
}

int main(int argc, char **argv) {
    return handle_example_errors(
            matmul_end_gelu_example, parse_engine_kind(argc, argv));
}
