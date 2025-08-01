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

#include <assert.h>
#include <iostream>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "example_utils.hpp"
#include "graph_example_utils.hpp"

using namespace dnnl::graph;
using layout_type = logical_tensor::layout_type;

void multi_binary_ops(engine::kind ekind, logical_tensor::data_type dt) {
    allocator alloc = create_allocator(ekind);

    // Create execution dnnl::engine.
    engine eng = make_engine_with_allocator(ekind, 0, alloc);
    // Create dnnl::stream.
    stream strm(eng);

    // Incremental IDs used to create logical tensors and operations.
    size_t id = 0;

    const logical_tensor::dims in_sz = {8, 8, 8, 8};

    // Intermediate data type.
    const logical_tensor::data_type dt_inter = logical_tensor::data_type::f32;

    // binary op kind
    const op::kind opk = op::kind::Multiply;

    // inputs
    auto src_0 = logical_tensor(id++, dt, in_sz, layout_type::strided);
    auto src_1 = logical_tensor(id++, dt, in_sz, layout_type::strided);
    auto src_2 = logical_tensor(id++, dt, in_sz, layout_type::strided);
    auto src_3 = logical_tensor(id++, dt, in_sz, layout_type::strided);
    auto src_4 = logical_tensor(id++, dt, in_sz, layout_type::strided);
    auto src_5 = logical_tensor(id++, dt, in_sz, layout_type::strided);
    auto src_6 = logical_tensor(id++, dt, in_sz, layout_type::strided);
    auto src_7 = logical_tensor(id++, dt, in_sz, layout_type::strided);
    auto src_8 = logical_tensor(id++, dt, in_sz, layout_type::strided);
    auto src_9 = logical_tensor(id++, dt, in_sz, layout_type::strided);
    auto src_10 = logical_tensor(id++, dt, in_sz, layout_type::strided);
    auto src_11 = logical_tensor(id++, dt, in_sz, layout_type::strided);
    auto src_12 = logical_tensor(id++, dt, in_sz, layout_type::strided);
    auto src_13 = logical_tensor(id++, dt, in_sz, layout_type::strided);
    auto src_14 = logical_tensor(id++, dt, in_sz, layout_type::strided);
    auto src_15 = logical_tensor(id++, dt, in_sz, layout_type::strided);
    auto src_16 = logical_tensor(id++, dt, in_sz, layout_type::strided);
    auto src_17 = logical_tensor(id++, dt, in_sz, layout_type::strided);
    auto src_18 = logical_tensor(id++, dt, in_sz, layout_type::strided);
    auto src_19 = logical_tensor(id++, dt, in_sz, layout_type::strided);
    auto src_20 = logical_tensor(id++, dt, in_sz, layout_type::strided);
    auto src_21 = logical_tensor(id++, dt, in_sz, layout_type::strided);

    // binary_0
    auto dst_0 = logical_tensor(id++, dt_inter, in_sz, layout_type::strided);
    auto binary_0 = op(id++, opk, "binary_0");
    binary_0.add_inputs({src_0, src_1});
    binary_0.add_outputs({dst_0});

    // binary_1
    auto dst_1 = logical_tensor(id++, dt_inter, in_sz, layout_type::strided);
    auto binary_1 = op(id++, opk, "binary_1");
    binary_1.add_inputs({dst_0, src_2});
    binary_1.add_outputs({dst_1});

    // binary_2
    auto dst_2 = logical_tensor(id++, dt_inter, in_sz, layout_type::strided);
    auto binary_2 = op(id++, opk, "binary_2");
    binary_2.add_inputs({dst_1, src_3});
    binary_2.add_outputs({dst_2});

    // binary_3
    auto dst_3 = logical_tensor(id++, dt_inter, in_sz, layout_type::strided);
    auto binary_3 = op(id++, opk, "binary_3");
    binary_3.add_inputs({dst_2, src_4});
    binary_3.add_outputs({dst_3});

    // binary_4
    auto dst_4 = logical_tensor(id++, dt_inter, in_sz, layout_type::strided);
    auto binary_4 = op(id++, opk, "binary_4");
    binary_4.add_inputs({dst_3, src_5});
    binary_4.add_outputs({dst_4});

    // binary_5
    auto dst_5 = logical_tensor(id++, dt_inter, in_sz, layout_type::strided);
    auto binary_5 = op(id++, opk, "binary_5");
    binary_5.add_inputs({dst_4, src_6});
    binary_5.add_outputs({dst_5});

    // binary_6
    auto dst_6 = logical_tensor(id++, dt_inter, in_sz, layout_type::strided);
    auto binary_6 = op(id++, opk, "binary_6");
    binary_6.add_inputs({dst_5, src_7});
    binary_6.add_outputs({dst_6});

    // binary_7
    auto dst_7 = logical_tensor(id++, dt_inter, in_sz, layout_type::strided);
    auto binary_7 = op(id++, opk, "binary_7");
    binary_7.add_inputs({dst_6, src_8});
    binary_7.add_outputs({dst_7});

    // binary_8
    auto dst_8 = logical_tensor(id++, dt_inter, in_sz, layout_type::strided);
    auto binary_8 = op(id++, opk, "binary_8");
    binary_8.add_inputs({dst_7, src_9});
    binary_8.add_outputs({dst_8});

    // binary_9
    auto dst_9 = logical_tensor(id++, dt_inter, in_sz, layout_type::strided);
    auto binary_9 = op(id++, opk, "binary_9");
    binary_9.add_inputs({dst_8, src_10});
    binary_9.add_outputs({dst_9});

    // tc: dt_inter -> dt
    auto dst_9_dt = dst_9;
    auto tc_0 = op(id++, op::kind::TypeCast, "tc_0");
    if (dt != dt_inter) {
        dst_9_dt = logical_tensor(id++, dt, in_sz, layout_type::strided);
        tc_0.add_input(dst_9);
        tc_0.add_output(dst_9_dt);
    }

    // matmul
    auto dst_10 = logical_tensor(id++, dt_inter, in_sz, layout_type::strided);
    auto mm = op(id++, op::kind::MatMul, "matmul");
    mm.add_inputs({dst_9_dt, src_11});
    mm.add_outputs({dst_10});

    // binary_10
    auto dst_11 = logical_tensor(id++, dt_inter, in_sz, layout_type::strided);
    auto binary_10 = op(id++, opk, "binary_10");
    binary_10.add_inputs({dst_10, src_12});
    binary_10.add_outputs({dst_11});

    // binary_11
    auto dst_12 = logical_tensor(id++, dt_inter, in_sz, layout_type::strided);
    auto binary_11 = op(id++, opk, "binary_11");
    binary_11.add_inputs({dst_11, src_13});
    binary_11.add_outputs({dst_12});

    // binary_12
    auto dst_13 = logical_tensor(id++, dt_inter, in_sz, layout_type::strided);
    auto binary_12 = op(id++, opk, "binary_12");
    binary_12.add_inputs({dst_12, src_14});
    binary_12.add_outputs({dst_13});

    // binary_13
    auto dst_14 = logical_tensor(id++, dt_inter, in_sz, layout_type::strided);
    auto binary_13 = op(id++, opk, "binary_13");
    binary_13.add_inputs({dst_13, src_15});
    binary_13.add_outputs({dst_14});

    // binary_14
    auto dst_15 = logical_tensor(id++, dt_inter, in_sz, layout_type::strided);
    auto binary_14 = op(id++, opk, "binary_14");
    binary_14.add_inputs({dst_14, src_16});
    binary_14.add_outputs({dst_15});

    // binary_15
    auto dst_16 = logical_tensor(id++, dt_inter, in_sz, layout_type::strided);
    auto binary_15 = op(id++, opk, "binary_15");
    binary_15.add_inputs({dst_15, src_17});
    binary_15.add_outputs({dst_16});

    // binary_16
    auto dst_17 = logical_tensor(id++, dt_inter, in_sz, layout_type::strided);
    auto binary_16 = op(id++, opk, "binary_16");
    binary_16.add_inputs({dst_16, src_18});
    binary_16.add_outputs({dst_17});

    // binary_17
    auto dst_18 = logical_tensor(id++, dt_inter, in_sz, layout_type::strided);
    auto binary_17 = op(id++, opk, "binary_17");
    binary_17.add_inputs({dst_17, src_19});
    binary_17.add_outputs({dst_18});

    // binary_18
    auto dst_19 = logical_tensor(id++, dt_inter, in_sz, layout_type::strided);
    auto binary_18 = op(id++, opk, "binary_18");
    binary_18.add_inputs({dst_18, src_20});
    binary_18.add_outputs({dst_19});

    // binary_19
    auto dst_20 = logical_tensor(id++, dt_inter, in_sz, layout_type::strided);
    auto binary_19 = op(id++, opk, "binary_19");
    binary_19.add_inputs({dst_19, src_21});
    binary_19.add_outputs({dst_20});

    // tc: dt_inter -> dt
    auto dst_20_dt = dst_20;
    auto tc_1 = op(id++, op::kind::TypeCast, "tc_1");
    if (dt != dt_inter) {
        dst_20_dt = logical_tensor(id++, dt, in_sz, layout_type::strided);
        tc_1.add_input(dst_20);
        tc_1.add_output(dst_20_dt);
    }

    // construct a graph object
    dnnl::graph::graph gh(ekind);
    gh.add_op(binary_0);
    gh.add_op(binary_1);
    gh.add_op(binary_2);
    gh.add_op(binary_3);
    gh.add_op(binary_4);
    gh.add_op(binary_5);
    gh.add_op(binary_6);
    gh.add_op(binary_7);
    gh.add_op(binary_8);
    gh.add_op(binary_9);
    if (dt != dt_inter) gh.add_op(tc_0);
    gh.add_op(mm);
    gh.add_op(binary_10);
    gh.add_op(binary_11);
    gh.add_op(binary_12);
    gh.add_op(binary_13);
    gh.add_op(binary_14);
    gh.add_op(binary_15);
    gh.add_op(binary_16);
    gh.add_op(binary_17);
    gh.add_op(binary_18);
    gh.add_op(binary_19);
    if (dt != dt_inter) gh.add_op(tc_1);

    gh.finalize();

    // Get partitions from the graph.
    std::vector<partition> partitions = gh.get_partitions();
    if (partitions.size() != 1) {
        std::cout << "unsupported fusion" << std::endl;
        return;
    }

    // Compile the partition with inputs, outputs, and an engine.
    compiled_partition cp = partitions[0].compile(
            {src_0, src_1, src_2, src_3, src_4, src_5, src_6, src_7, src_8,
                    src_9, src_10, src_11, src_12, src_13, src_14, src_15,
                    src_16, src_17, src_18, src_19, src_20, src_21},
            {dst_20_dt}, eng);

    // Create tensor objects
    auto ts_src_0 = tensor(src_0, eng);
    auto ts_src_1 = tensor(src_1, eng);
    auto ts_src_2 = tensor(src_2, eng);
    auto ts_src_3 = tensor(src_3, eng);
    auto ts_src_4 = tensor(src_4, eng);
    auto ts_src_5 = tensor(src_5, eng);
    auto ts_src_6 = tensor(src_6, eng);
    auto ts_src_7 = tensor(src_7, eng);
    auto ts_src_8 = tensor(src_8, eng);
    auto ts_src_9 = tensor(src_9, eng);
    auto ts_src_10 = tensor(src_10, eng);
    auto ts_src_11 = tensor(src_11, eng);
    auto ts_src_12 = tensor(src_12, eng);
    auto ts_src_13 = tensor(src_13, eng);
    auto ts_src_14 = tensor(src_14, eng);
    auto ts_src_15 = tensor(src_15, eng);
    auto ts_src_16 = tensor(src_16, eng);
    auto ts_src_17 = tensor(src_17, eng);
    auto ts_src_18 = tensor(src_18, eng);
    auto ts_src_19 = tensor(src_19, eng);
    auto ts_src_20 = tensor(src_20, eng);
    auto ts_src_21 = tensor(src_21, eng);

    auto ts_dst_20 = tensor(dst_20_dt, eng);

    cp.execute(strm,
            {ts_src_0, ts_src_1, ts_src_2, ts_src_3, ts_src_4, ts_src_5,
                    ts_src_6, ts_src_7, ts_src_8, ts_src_9, ts_src_10,
                    ts_src_11, ts_src_12, ts_src_13, ts_src_14, ts_src_15,
                    ts_src_16, ts_src_17, ts_src_18, ts_src_19, ts_src_20,
                    ts_src_21},
            {ts_dst_20});

    return;
}

void run_multi_binary_ops(engine::kind ekind) {
    multi_binary_ops(ekind, logical_tensor::data_type::f32);
    multi_binary_ops(ekind, logical_tensor::data_type::f16);
    multi_binary_ops(ekind, logical_tensor::data_type::bf16);

    return;
}

int main(int argc, char **argv) {
    return handle_example_errors(
            run_multi_binary_ops, parse_engine_kind(argc, argv));
}
