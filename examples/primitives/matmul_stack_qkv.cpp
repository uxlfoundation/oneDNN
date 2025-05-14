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

/// @example matmul.cpp
/// > Annotated version: @ref matmul_example_cpp
///
/// @page matmul_example_cpp_short
///
/// This C++ API example demonstrates how to create and execute a
/// [MatMul](@ref dev_guide_matmul) primitive.
///
/// Key optimizations included in this example:
/// - Primitive attributes with fused post-ops.
///
/// @page matmul_example_cpp Matmul Primitive Example
/// @copydetails matmul_example_cpp_short
///
/// @include matmul.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

void matmul_example(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim batch_size = 1, head_num = 12, seq_len = 1024,
                      head_dim = 64;

    // Source (src), weights, bias, and destination (dst) tensors dimensions.
    memory::dims src_dims = {batch_size, head_num, seq_len, head_dim};
    memory::dims weights_dims = {batch_size, head_num, head_dim, seq_len};
    memory::dims bias_dims = {batch_size, head_num, seq_len, seq_len};
    memory::dims dst_dims = {batch_size, head_num, seq_len, seq_len};
    memory::dims mul_src_dims = {1, 1, 1, 1};
    memory::dims add_src_dims = {batch_size, 1, 1, seq_len};

    //     memory::dims qkv_strides = {seq_len * head_num * 3 * head_dim, head_dim,
    //             head_num * 3 * head_dim, 1};
    memory::dims src_strides = {seq_len * head_num * 3 * head_dim, head_dim,
            head_num * 3 * head_dim, 1};
    memory::dims weights_strides = {
            seq_len * head_num * 3 * head_dim,
            head_dim,
            1,
            head_num * 3 * head_dim,
    };

    // Allocate buffers.
    std::vector<float> src_data(product(src_dims));
    std::vector<float> weights_data(product(weights_dims));
    // std::vector<float> bias_data(product(bias_dims), 0.f);
    std::vector<float> dst_data1(product(dst_dims));
    std::vector<float> dst_data2(product(dst_dims));
    std::vector<float> add_src_data(product(add_src_dims));
    std::vector<float> mul_src_data(product(mul_src_dims));

    // Initialize src, weights, bias.
    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });
    std::generate(weights_data.begin(), weights_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 2.f);
    });
    std::generate(add_src_data.begin(), add_src_data.end(), []() {
        static int i = 0;
        return std::tanh(float(i++));
    });
    std::generate(mul_src_data.begin(), mul_src_data.end(), []() {
        static int i = 0;
        return std::tanh(float(i++));
    });

    // Create memory descriptors and memory objects for src, weights, bias, and
    // dst.
    auto src_md = memory::desc(
            src_dims, memory::data_type::bf16, memory::format_tag::any);
    auto weights_md = memory::desc(
            weights_dims, memory::data_type::bf16, memory::format_tag::any);
    auto dst_md = memory::desc(
            dst_dims, memory::data_type::f32, memory::format_tag::abcd);
    auto add_src_md = memory::desc(
            add_src_dims, memory::data_type::bf16, memory::format_tag::abcd);
    auto mul_src_md = memory::desc(
            mul_src_dims, memory::data_type::bf16, memory::format_tag::abcd);

    auto user_src_mem
            = memory({src_dims, memory::data_type::bf16, src_strides}, engine);
    auto user_weights_mem = memory(
            {weights_dims, memory::data_type::bf16, weights_strides}, engine);
    auto dst_mem = memory(
            {dst_dims, memory::data_type::f32, memory::format_tag::abcd},
            engine);
    auto add_src_mem = memory(add_src_md, engine);
    auto mul_src_mem = memory(mul_src_md, engine);
    auto src_mem = user_src_mem;
    auto weights_mem = user_weights_mem;

    // Write data to memory object's handles.
    write_to_dnnl_memory(src_data.data(), src_mem);
    write_to_dnnl_memory(weights_data.data(), weights_mem);
    // write_to_dnnl_memory(mul_src_data.data(), mul_src_mem);
    // write_to_dnnl_memory(add_src_data.data(), add_src_mem);

    // Create primitive post-ops.
    post_ops matmul_ops1;
    matmul_ops1.append_binary(algorithm::binary_mul, mul_src_md);
    matmul_ops1.append_binary(algorithm::binary_add, add_src_md);
    primitive_attr matmul_attr1;
    // matmul_attr1.set_post_ops(matmul_ops1);

    // Create primitive descriptor.
    auto matmul_pd1 = matmul::primitive_desc(
            engine, src_md, weights_md, dst_md, matmul_attr1);

    // if (matmul_pd1.src_desc() != src_mem.get_desc()) {
    //     src_mem = memory(matmul_pd1.src_desc(), engine);
    //     reorder(user_src_mem, src_mem)
    //             .execute(engine_stream, user_src_mem, src_mem);
    // }

    // if (matmul_pd1.weights_desc() != weights_mem.get_desc()) {
    //     weights_mem = memory(matmul_pd1.weights_desc(), engine);
    //     reorder(user_weights_mem, weights_mem)
    //             .execute(engine_stream, user_weights_mem, weights_mem);
    // }

    // Create the primitive.
    auto matmul_prim1 = matmul(matmul_pd1);

    // Primitive arguments.
    std::unordered_map<int, memory> matmul_args1;
    matmul_args1.insert({DNNL_ARG_SRC, src_mem});
    matmul_args1.insert({DNNL_ARG_WEIGHTS, weights_mem});
    matmul_args1.insert({DNNL_ARG_DST, dst_mem});
    // matmul_args1.insert(
    //         {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, mul_src_mem});
    // matmul_args1.insert(
    //         {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1, add_src_mem});

    // Primitive execution: matrix multiplication with ReLU.
    matmul_prim1.execute(engine_stream, matmul_args1);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_data1.data(), dst_mem);
}

int main(int argc, char **argv) {
    return handle_example_errors(matmul_example, parse_engine_kind(argc, argv));
}
