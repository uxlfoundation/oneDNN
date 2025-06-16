/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

/// @example cpu_getting_started.cpp
/// @copybrief graph_cpu_getting_started_cpp
/// > Annotated version: @ref graph_cpu_getting_started_cpp

/// @page graph_cpu_getting_started_cpp Getting started on CPU with Graph API
/// This is an example to demonstrate how to build a simple graph and run it on
/// CPU.
///
/// > Example code: @ref cpu_getting_started.cpp
///
/// Some key take-aways included in this example:
///
/// * how to build a graph and get partitions from it
/// * how to create an engine, allocator and stream
/// * how to compile a partition
/// * how to execute a compiled partition
///
/// Some assumptions in this example:
///
/// * Only workflow is demonstrated without checking correctness
/// * Unsupported partitions should be handled by users themselves
///

/// @page graph_cpu_getting_started_cpp
/// @section graph_cpu_getting_started_cpp_headers Public headers
///
/// To start using oneDNN Graph, we must include the @ref dnnl_graph.hpp header
/// file in the application. All the C++ APIs reside in namespace `dnnl::graph`.
///
/// @page graph_cpu_getting_started_cpp
/// @snippet cpu_getting_started.cpp Headers and namespace
//[Headers and namespace]
#include <iostream>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include <assert.h>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "example_utils.hpp"
#include "graph_example_utils.hpp"

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;
//[Headers and namespace]

/// @page graph_cpu_getting_started_cpp
/// @section graph_cpu_getting_started_cpp_tutorial cpu_getting_started_tutorial() function
///
void cpu_getting_started_tutorial() {

    dim MB1 = 1, MB2 = 16, M = 32, K = 256, N = 32;
    dims src_dims {MB1, MB2, M, K};
    dims wei_dims {MB1, MB2, K, N};
    dims sel_cond_dims {1, 1, M, N};
    dims sel_src1_dims {1};
    dims dst_dims {MB1, MB2, M, N};

    logical_tensor mm_src_lt {
            0, data_type::f32, src_dims, layout_type::strided};
    logical_tensor mm_weight_lt {
            1, data_type::f32, wei_dims, layout_type::strided};
    logical_tensor mm_dst_lt {
            2, data_type::f32, dst_dims, layout_type::strided};

    op matmul(
            0, op::kind::MatMul, {mm_src_lt, mm_weight_lt}, {mm_dst_lt}, "mm");

    logical_tensor select_src1_lt {
            3, data_type::f32, sel_src1_dims, layout_type::strided};
    logical_tensor select_cond_lt {
            4, data_type::boolean, sel_cond_dims, layout_type::strided};
    logical_tensor select_dst_lt {
            5, data_type::f32, dst_dims, layout_type::strided};
    op sel(1, op::kind::Select, {select_cond_lt, mm_dst_lt, select_src1_lt},
            {select_dst_lt}, "select");

    graph g(dnnl::engine::kind::cpu);

    g.add_op(matmul);
    g.add_op(sel);

    g.finalize();

    auto partitions = g.get_partitions();

    assert(partitions.size() == 1);

    //[Create engine]
    allocator alloc {};
    dnnl::engine eng
            = make_engine_with_allocator(dnnl::engine::kind::cpu, 0, alloc);
    //[Create engine]

    dnnl::stream strm {eng};
    //[Create stream]
    compiled_partition cp = partitions[0].compile(
            {mm_src_lt, mm_weight_lt, select_cond_lt, select_src1_lt},
            {select_dst_lt}, eng);

    auto ts_mm_src = tensor(mm_src_lt, eng);
    auto ts_mm_wei = tensor(mm_weight_lt, eng);
    auto ts_sel_cond = tensor(select_cond_lt, eng);
    auto ts_sel_src1 = tensor(select_src1_lt, eng);
    auto ts_sel_dst = tensor(select_dst_lt, eng);

    cp.execute(strm, {ts_mm_src, ts_mm_wei, ts_sel_cond, ts_sel_src1},
            {ts_sel_dst});
    // Wait for all compiled partition's execution finished
    strm.wait();
}

int main(int argc, char **argv) {
    return handle_example_errors(
            {engine::kind::cpu}, cpu_getting_started_tutorial);
}
