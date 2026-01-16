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

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

#include "graph_example_utils.hpp"

using namespace dnnl;

using namespace dnnl::graph;
using layout_type = logical_tensor::layout_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;

void grouped_matmul(engine::kind ekind) {
    allocator alloc = create_allocator(ekind);

    // Create execution dnnl::engine.
    dnnl::engine eng = make_engine_with_allocator(ekind, 0, alloc);
    // Create dnnl::stream.
    dnnl::stream strm(eng);

    const size_t num = 8;
    const std::vector<dim> m = {10, 20, 30, 40, 50, 60, 70, 80};
    const dim k = 256;
    const dim n = 1024;

    // Incremental IDs used to create operations.
    size_t id = 100;
    const std::vector<size_t> src_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    const std::vector<size_t> wei_ids = {8, 9, 10, 11, 12, 13, 14, 15};
    const std::vector<size_t> dst_ids = {16, 17, 18, 19, 20, 21, 22, 23};

    // data type of input and output tensors.
    const logical_tensor::data_type dt = logical_tensor::data_type::f32;

    dnnl::graph::graph gmm(ekind);
    for (size_t i = 0; i < num; ++i) {
        // create matmuls and add them into the graph
        auto src = logical_tensor(
                src_ids[i], dt, {m[i], k}, layout_type::strided);
        auto wei = logical_tensor(wei_ids[i], dt, {k, n}, layout_type::strided);
        auto dst = logical_tensor(
                dst_ids[i], dt, {m[i], n}, layout_type::strided);
        auto mm = op(id++, op::kind::MatMul, "mm_" + std::to_string(i));
        mm.add_inputs({src, wei});
        mm.add_outputs({dst});
        gmm.add_op(mm);
    }

    gmm.finalize();
    // Get partitions from the grouped matmul graph.
    std::vector<partition> partitions = gmm.get_partitions();
    // This is just for oneDNN testing purpose.
    if (partitions.size() != 1) {
        std::cout << "Unsupported. Partitions size: " << partitions.size()
                  << std::endl;
        return;
    }

    std::cout << "partition success ..." << std::endl;

    // Compile the partition with inputs, outputs, and an engine.
    std::vector<logical_tensor> inputs;
    std::vector<logical_tensor> outputs;
    for (size_t i = 0; i < num; ++i) {
        auto src = logical_tensor(
                src_ids[i], dt, {m[i], k}, layout_type::strided);
        auto wei = logical_tensor(wei_ids[i], dt, {k, n}, layout_type::strided);
        auto dst = logical_tensor(
                dst_ids[i], dt, {m[i], n}, layout_type::strided);
        inputs.push_back(src);
        inputs.push_back(wei);
        outputs.push_back(dst);
    }

    compiled_partition cp = partitions[0].compile(inputs, outputs, eng);

    std::cout << "compile success ..." << std::endl;

    // Create input and output tensors. Use library managed memory for
    // simplicity.
    std::vector<tensor> ts_inputs;
    std::vector<tensor> ts_outputs;
    for (size_t i = 0; i < inputs.size(); ++i) {
        ts_inputs.emplace_back(inputs[i], eng);
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        ts_outputs.emplace_back(outputs[i], eng);
    }

    // execute the compiled partition
    cp.execute(strm, ts_inputs, ts_outputs);
    strm.wait();
    std::cout << "execute success ..." << std::endl;
}

int main(int argc, char **argv) {
    return handle_example_errors(grouped_matmul, parse_engine_kind(argc, argv));
}
