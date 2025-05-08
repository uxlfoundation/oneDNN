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

void tf_gemm2_gqa(const dim bs, const dim head_num_kv, const dim group,
        const dim seq_len_q, const dim head_size, const dim seq_len_kv) {
    allocator alloc = create_allocator(dnnl::engine::kind::cpu);
    dnnl::engine eng
            = make_engine_with_allocator(dnnl::engine::kind::cpu, 0, alloc);
    dnnl::graph::graph g {engine::kind::cpu};

    // Create dnnl::stream.
    dnnl::stream strm(eng);

    size_t lt_id = 0;
    size_t op_id = 0;

    const dims dot1_in1_shape {bs, head_num_kv, group, seq_len_q, head_size};
    const dims dot1_in2_shape {bs, head_num_kv, 1, head_size, seq_len_kv};
    const dims dot1_out_shape {bs, head_num_kv, group, seq_len_q, seq_len_kv};
    const dims scale_sz = {1};
    logical_tensor lt_dot1_in1 {lt_id++, logical_tensor::data_type::bf16,
            dot1_in1_shape, logical_tensor::layout_type::strided};
    logical_tensor lt_dot1_in2 {lt_id++, logical_tensor::data_type::bf16,
            dot1_in2_shape, logical_tensor::layout_type::strided};
    logical_tensor lt_dot1_out {lt_id++, logical_tensor::data_type::f32,
            dot1_out_shape, logical_tensor::layout_type::strided};
    op matmul_1_op(op_id++, op::kind::MatMul, "matmul_1");
    matmul_1_op.add_inputs({lt_dot1_in1, lt_dot1_in2});
    matmul_1_op.add_output(lt_dot1_out);

    logical_tensor lt_mul1_in2 {lt_id++, logical_tensor::data_type::f32,
            scale_sz, logical_tensor::layout_type::strided};
    logical_tensor lt_mul1_out {lt_id++, logical_tensor::data_type::f32,
            dot1_out_shape, logical_tensor::layout_type::strided};
    op mul_1_op(op_id++, op::kind::Multiply, "multiply_1");
    mul_1_op.add_inputs({lt_dot1_out, lt_mul1_in2});
    mul_1_op.add_output(lt_mul1_out);

    logical_tensor lt_tanh_out {lt_id++, logical_tensor::data_type::f32,
            dot1_out_shape, logical_tensor::layout_type::strided};
    op tanh_op(op_id++, op::kind::Tanh, "tanh");
    tanh_op.add_inputs({lt_mul1_out});
    tanh_op.add_output(lt_tanh_out);

    logical_tensor lt_mul2_in2 {lt_id++, logical_tensor::data_type::f32,
            scale_sz, logical_tensor::layout_type::strided};
    logical_tensor lt_mul2_out {lt_id++, logical_tensor::data_type::f32,
            dot1_out_shape, logical_tensor::layout_type::strided};
    op mul_2_op(op_id++, op::kind::Multiply, "multiply_2");
    mul_2_op.add_inputs({lt_tanh_out, lt_mul2_in2});
    mul_2_op.add_output(lt_mul2_out);

    logical_tensor lt_select_in1 {lt_id++, logical_tensor::data_type::boolean,
            dot1_out_shape, logical_tensor::layout_type::strided};
    logical_tensor lt_select_in2 {lt_id++, logical_tensor::data_type::f32,
            scale_sz, logical_tensor::layout_type::strided};
    logical_tensor lt_select_out {lt_id++, logical_tensor::data_type::f32,
            logical_tensor::layout_type::strided};
    op select_op(op_id++, op::kind::Select, "select");
    select_op.add_input(lt_select_in1);
    select_op.add_input(lt_select_in2);
    select_op.add_input(lt_mul2_out);
    select_op.add_output(lt_select_out);

    op softmax_op(op_id++, op::kind::SoftMax, "softmax");
    softmax_op.set_attr<int64_t>(
            op::attr::axis, -1); // assume -1 means last axis, to confirm
    logical_tensor lt_softmax_out {lt_id++, logical_tensor::data_type::bf16,
            logical_tensor::layout_type::strided};
    softmax_op.add_input(lt_select_out);
    softmax_op.add_output(lt_softmax_out);

    const dims dot2_in2_shape {bs, head_num_kv, 1, seq_len_kv, head_size};
    const dims dot2_out_shape {bs, head_num_kv, group, seq_len_q, head_size};
    logical_tensor lt_dot2_in2 {lt_id++, logical_tensor::data_type::bf16,
            dot2_in2_shape, logical_tensor::layout_type::strided};
    logical_tensor lt_dot2_out {lt_id++, logical_tensor::data_type::bf16,
            dot2_out_shape, logical_tensor::layout_type::strided};

    op matmul_2_op(op_id++, op::kind::MatMul, "matmul_2");
    matmul_2_op.add_inputs({lt_softmax_out, lt_dot2_in2});
    matmul_2_op.add_output(lt_dot2_out);

    g.add_op(matmul_1_op);
    g.add_op(mul_1_op);
    g.add_op(tanh_op);
    g.add_op(mul_2_op);
    g.add_op(select_op);
    g.add_op(softmax_op);
    g.add_op(matmul_2_op);

    g.finalize();
    auto partitions = g.get_partitions();

    if (partitions.size() != 1) {
        std::cout << "unsupported pattern" << std::endl;
        return;
    }

    // Compile the partition with inputs, outputs, and an engine.
    compiled_partition cp = partitions[0].compile(
            {lt_dot1_in1, lt_dot1_in2, lt_mul1_in2, lt_mul2_in2, lt_select_in1,
                    lt_select_in2, lt_dot2_in2},
            {lt_dot2_out}, eng);

    // Create tensor objects
    auto ts_dot1_in1 = tensor(lt_dot1_in1, eng);
    auto ts_dot1_in2 = tensor(lt_dot1_in2, eng);
    auto ts_mul1_in2 = tensor(lt_mul1_in2, eng);
    auto ts_mul2_in2 = tensor(lt_mul2_in2, eng);
    auto ts_select_in1 = tensor(lt_select_in1, eng);
    auto ts_select_in2 = tensor(lt_select_in2, eng);
    auto ts_dot2_in2 = tensor(lt_dot2_in2, eng);
    auto ts_dot2_out = tensor(lt_dot2_out, eng);

    // Warmup run.
    // Execute the compiled partition of mqa.
    cp.execute(strm,
            {ts_dot1_in1, ts_dot1_in2, ts_mul1_in2, ts_mul2_in2, ts_select_in1,
                    ts_select_in2, ts_dot2_in2},
            {ts_dot2_out});

    // Wait for the computation to finish.
    strm.wait();
}

int main(int argc, char **argv) {
    tf_gemm2_gqa(1, 8, 2, 1, 256, 256);
    tf_gemm2_gqa(1, 8, 2, 256, 256, 256);
    return 0;
}
