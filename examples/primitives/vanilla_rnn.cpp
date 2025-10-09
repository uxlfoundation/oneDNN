/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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
/// @example vanilla_rnn.cpp
/// > Annotated version: @ref vanilla_rnn_example_cpp

/// @page vanilla_rnn_example_cpp Vanilla RNN Primitive Example
/// This C++ API example demonstrates how to create and execute a
/// [Vanilla RNN](@ref dev_guide_rnn) primitive in forward and backward training propagation
/// mode.
///
/// Key optimizations included in this example:
/// - Creation of optimized memory format from the primitive descriptor.
///
/// @include vanilla_rnn.cpp

#include <cstring>
#include <math.h>
#include <numeric>
#include <utility>

#include "oneapi/dnnl/dnnl.hpp"

#include "example_utils.hpp"

using namespace dnnl;

void vanilla_rnn(engine::kind engine_kind) {

    dnnl::engine engine = dnnl::engine(engine_kind, 0);
    dnnl::stream engine_stream = dnnl::stream(engine);

    // Tensor dimensions.
    const memory::dim N = 2, // batch size
            T = 3, // time steps
            C = 4, // channels
            G = 1, // gates, 1 for vanilla RNN
            L = 1, // layers
            D = 1; // directions

    memory::dims src_dims = {T, N, C};
    memory::dims weights_dims = {L, D, C, G, C};
    memory::dims bias_dims = {L, D, G, C};
    memory::dims dst_layer_dims = {T, N, C};
    memory::dims dst_iter_dims = {L, D, N, C};

    // Allocate buffers.
    std::vector<float> src_layer_data(product(src_dims));
    std::vector<float> weights_layer_data(product(weights_dims));
    std::vector<float> weights_iter_data(product(weights_dims));
    std::vector<float> bias_data(product(bias_dims));
    std::vector<float> dst_layer_data(product(dst_layer_dims));
    std::vector<float> dst_iter_data(product(dst_iter_dims));

    // Initialize src, weights, and bias tensors.
    std::generate(src_layer_data.begin(), src_layer_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });
    std::generate(weights_layer_data.begin(), weights_layer_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 2.f);
    });
    std::generate(weights_iter_data.begin(), weights_iter_data.end(), []() {
        static int i = 0;
        return std::sin(i++ * 2.f);
    });
    std::generate(bias_data.begin(), bias_data.end(), []() {
        static int i = 0;
        return std::tanh(float(i++));
    });

    // Create memory descriptors and memory objects for src, bias, and dst.
    auto src_layer_md = memory::desc(
            src_dims, memory::data_type::f32, memory::format_tag::tnc);
    auto bias_md = memory::desc(
            bias_dims, memory::data_type::f32, memory::format_tag::ldgo);
    auto dst_layer_md = memory::desc(
            dst_layer_dims, memory::data_type::f32, memory::format_tag::tnc);

    auto src_layer_mem = memory(src_layer_md, engine);
    auto bias_mem = memory(bias_md, engine);
    auto dst_layer_mem = memory(dst_layer_md, engine);

    // Create memory objects for weights using user's memory layout. In this
    // example, LDIGO (num_layers, num_directions, input_channels, num_gates,
    // output_channels) is assumed.
    auto user_weights_layer_mem = memory(
            {weights_dims, memory::data_type::f32, memory::format_tag::ldigo},
            engine);
    auto user_weights_iter_mem = memory(
            {weights_dims, memory::data_type::f32, memory::format_tag::ldigo},
            engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_layer_data.data(), src_layer_mem);
    write_to_dnnl_memory(bias_data.data(), bias_mem);
    write_to_dnnl_memory(dst_layer_data.data(), dst_layer_mem);
    write_to_dnnl_memory(weights_layer_data.data(), user_weights_layer_mem);
    write_to_dnnl_memory(weights_iter_data.data(), user_weights_iter_mem);

    // Create memory descriptors for weights with format_tag::any. This enables
    // the Vanilla primitive to choose the optimized memory layout.
    auto weights_layer_md = memory::desc(
            weights_dims, memory::data_type::f32, memory::format_tag::any);
    auto weights_iter_md = memory::desc(
            weights_dims, memory::data_type::f32, memory::format_tag::any);

    // Memory descriptors for recurrent data (src_iter_md, dst_iter_md) is optional

    // Create primitive descriptor.
    auto vanilla_rnn_fwd_pd = vanilla_rnn_forward::primitive_desc(
            engine, prop_kind::forward_training, dnnl::algorithm::eltwise_tanh,
            rnn_direction::unidirectional_left2right, src_layer_md,
            memory::desc(), //src_iter_md
            weights_layer_md, weights_iter_md, bias_md, dst_layer_md,
            memory::desc() //dst_iter_md
    );

    // For now, assume that the weights memory layout generated by the primitive
    // and the ones provided by the user are identical.
    auto weights_layer_mem = user_weights_layer_mem;
    auto weights_iter_mem = user_weights_iter_mem;

    // Reorder the data in case the weights memory layout generated by the
    // primitive and the one provided by the user are different. In this case,
    // we create additional memory objects with internal buffers that will
    // contain the reordered data.
    if (vanilla_rnn_fwd_pd.weights_desc()
            != user_weights_layer_mem.get_desc()) {
        weights_layer_mem = memory(vanilla_rnn_fwd_pd.weights_desc(), engine);
        reorder(user_weights_layer_mem, weights_layer_mem)
                .execute(engine_stream, user_weights_layer_mem,
                        weights_layer_mem);
    }

    if (vanilla_rnn_fwd_pd.weights_iter_desc()
            != user_weights_iter_mem.get_desc()) {
        weights_iter_mem
                = memory(vanilla_rnn_fwd_pd.weights_iter_desc(), engine);
        reorder(user_weights_iter_mem, weights_iter_mem)
                .execute(
                        engine_stream, user_weights_iter_mem, weights_iter_mem);
    }

    auto src_iter_mem = memory(vanilla_rnn_fwd_pd.src_iter_desc(), engine);
    auto dst_iter_mem = memory(vanilla_rnn_fwd_pd.dst_iter_desc(), engine);
    // We also create workspace memory based on the information from
    // the workspace_primitive_desc(). This is needed for internal
    // communication between forward and backward primitives during
    // training.

    auto create_ws = [=](dnnl::vanilla_rnn_forward::primitive_desc &pd) {
        return dnnl::memory(pd.workspace_desc(), engine);
    };

    auto workspace_memory = create_ws(vanilla_rnn_fwd_pd);

    // Create the primitive.
    auto vanilla_rnn_fwd_prim = vanilla_rnn_forward(vanilla_rnn_fwd_pd);

    // Primitive arguments
    std::unordered_map<int, memory> vanilla_rnn_fwd_args;
    vanilla_rnn_fwd_args.insert({DNNL_ARG_SRC_LAYER, src_layer_mem});
    vanilla_rnn_fwd_args.insert({DNNL_ARG_WEIGHTS_LAYER, weights_layer_mem});
    vanilla_rnn_fwd_args.insert({DNNL_ARG_WEIGHTS_ITER, weights_iter_mem});
    vanilla_rnn_fwd_args.insert({DNNL_ARG_BIAS, bias_mem});
    vanilla_rnn_fwd_args.insert({DNNL_ARG_DST_LAYER, dst_layer_mem});
    vanilla_rnn_fwd_args.insert({DNNL_ARG_SRC_ITER, src_iter_mem});
    vanilla_rnn_fwd_args.insert({DNNL_ARG_DST_ITER, dst_iter_mem});
    vanilla_rnn_fwd_args.insert({DNNL_ARG_WORKSPACE, workspace_memory});

    // Primitive execution: vanilla rnn forward

    try {
        vanilla_rnn_fwd_prim.execute(engine_stream, vanilla_rnn_fwd_args);
    } catch (const dnnl::error &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return;
    }

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_layer_data.data(), dst_layer_mem);

    // No backward pass for inference

    // Backward primitives will reuse memory from forward
    // and allocate/describe specifics here. Only relevant for training.

    auto src_layer_bwd_mem = src_layer_mem;
    auto bias_bwd_mem = bias_mem;

    // Create memory descriptors and memory objects for src, bias, and dst.

    auto diff_src_layer_md = memory::desc(
            src_dims, memory::data_type::f32, memory::format_tag::tnc);
    auto diff_bias_md = memory::desc(
            bias_dims, memory::data_type::f32, memory::format_tag::ldgo);
    auto diff_dst_layer_md = memory::desc(
            dst_layer_dims, memory::data_type::f32, memory::format_tag::tnc);

    // Memory descriptors for recurrent data (diff_src_iter_md, diff_dst_iter_md) is optional

    auto diff_src_layer_mem = memory(diff_src_layer_md, engine);
    auto diff_bias_mem = memory(diff_bias_md, engine);
    auto diff_dst_layer_mem = memory(diff_dst_layer_md, engine);

    // Create memory objects for weights using user's memory layout. In this
    // example, LDIGO (num_layers, num_directions, input_channels, num_gates,
    // output_channels) is assumed.

    auto weights_layer_bwd_mem = weights_layer_mem;

    auto diff_user_weights_layer_mem = memory(
            {weights_dims, memory::data_type::f32, memory::format_tag::ldigo},
            engine);
    auto diff_user_weights_iter_mem = memory(
            {weights_dims, memory::data_type::f32, memory::format_tag::ldigo},
            engine);

    auto diff_weights_layer_md = memory::desc(
            weights_dims, memory::data_type::f32, memory::format_tag::any);
    auto diff_weights_iter_md = memory::desc(
            weights_dims, memory::data_type::f32, memory::format_tag::any);

    // Create zero-filled vectors for gradients
    std::vector<float> diff_src_layer_data(product(src_dims), 0.0f);
    std::vector<float> diff_dst_layer_data(product(dst_layer_dims), 0.0f);
    std::vector<float> diff_weights_layer_data(product(weights_dims), 0.0f);
    std::vector<float> diff_weights_iter_data(product(weights_dims), 0.0f);
    std::vector<float> diff_bias_data(product(bias_dims), 0.0f);

    write_to_dnnl_memory(diff_src_layer_data.data(), diff_src_layer_mem);
    write_to_dnnl_memory(diff_dst_layer_data.data(), diff_dst_layer_mem);
    write_to_dnnl_memory(
            diff_weights_layer_data.data(), diff_user_weights_layer_mem);
    write_to_dnnl_memory(
            diff_weights_iter_data.data(), diff_user_weights_iter_mem);
    write_to_dnnl_memory(diff_bias_data.data(), diff_bias_mem);

    // Create backward primitive descriptor.
    auto vanilla_rnn_bwd_pd = vanilla_rnn_backward::primitive_desc(engine,
            prop_kind::backward, algorithm::eltwise_tanh,
            rnn_direction::unidirectional_left2right, src_layer_md,
            memory::desc(), //src_iter_md
            weights_layer_md, weights_iter_md, bias_md, dst_layer_md,
            memory::desc(), //dst_iter_md
            diff_src_layer_md,
            memory::desc(), //diff_src_iter_md
            diff_weights_layer_md, diff_weights_iter_md, diff_bias_md,
            diff_dst_layer_md,
            memory::desc(), //diff_dst_iter_md
            vanilla_rnn_fwd_pd);

    auto diff_weights_layer_mem = diff_user_weights_layer_mem;
    auto diff_weights_iter_mem = diff_user_weights_iter_mem;

    // Reorder the data in case the weights memory layout generated by the
    // primitive and the one provided by the user are different. In this case,
    // we create additional memory objects with internal buffers that will
    // contain the reordered data.
    if (vanilla_rnn_bwd_pd.weights_desc()
            != user_weights_layer_mem.get_desc()) {
        weights_layer_mem = memory(vanilla_rnn_bwd_pd.weights_desc(), engine);
        reorder(user_weights_layer_mem, weights_layer_mem)
                .execute(engine_stream, user_weights_layer_mem,
                        weights_layer_mem);
    }

    if (vanilla_rnn_bwd_pd.diff_weights_desc()
            != diff_user_weights_layer_mem.get_desc()) {
        diff_weights_layer_mem
                = memory(vanilla_rnn_bwd_pd.diff_weights_desc(), engine);
        reorder(diff_user_weights_layer_mem, diff_weights_layer_mem)
                .execute(engine_stream, diff_user_weights_layer_mem,
                        diff_weights_layer_mem);
    }

    if (vanilla_rnn_bwd_pd.weights_iter_desc()
            != user_weights_iter_mem.get_desc()) {
        weights_iter_mem
                = memory(vanilla_rnn_bwd_pd.weights_iter_desc(), engine);
        reorder(user_weights_iter_mem, weights_iter_mem)
                .execute(
                        engine_stream, user_weights_iter_mem, weights_iter_mem);
    }

    if (vanilla_rnn_bwd_pd.diff_weights_iter_desc()
            != diff_user_weights_iter_mem.get_desc()) {
        diff_weights_iter_mem
                = memory(vanilla_rnn_bwd_pd.diff_weights_iter_desc(), engine);
        reorder(diff_user_weights_iter_mem, diff_weights_iter_mem)
                .execute(engine_stream, diff_user_weights_iter_mem,
                        diff_weights_iter_mem);
    }

    auto diff_src_iter_mem
            = memory(vanilla_rnn_fwd_pd.diff_src_iter_desc(), engine);
    auto diff_dst_iter_mem
            = memory(vanilla_rnn_fwd_pd.diff_dst_iter_desc(), engine);
    // Create the primitive.
    auto vanilla_rnn_bwd_prim = vanilla_rnn_backward(vanilla_rnn_bwd_pd);

    // Primitive arguments
    std::unordered_map<int, memory> vanilla_rnn_bwd_args;
    vanilla_rnn_bwd_args.insert({DNNL_ARG_SRC_LAYER, src_layer_bwd_mem});
    vanilla_rnn_bwd_args.insert(
            {DNNL_ARG_WEIGHTS_LAYER, weights_layer_bwd_mem});
    vanilla_rnn_bwd_args.insert({DNNL_ARG_WEIGHTS_ITER, weights_iter_mem});
    vanilla_rnn_bwd_args.insert({DNNL_ARG_BIAS, bias_bwd_mem});
    vanilla_rnn_bwd_args.insert({DNNL_ARG_DST_LAYER, dst_layer_mem});
    vanilla_rnn_bwd_args.insert({DNNL_ARG_SRC_ITER, src_iter_mem});
    vanilla_rnn_bwd_args.insert({DNNL_ARG_DST_ITER, dst_iter_mem});
    vanilla_rnn_bwd_args.insert({DNNL_ARG_DIFF_SRC_LAYER, diff_src_layer_mem});
    vanilla_rnn_bwd_args.insert(
            {DNNL_ARG_DIFF_WEIGHTS_LAYER, diff_weights_layer_mem});
    vanilla_rnn_bwd_args.insert(
            {DNNL_ARG_DIFF_WEIGHTS_ITER, diff_weights_iter_mem});
    vanilla_rnn_bwd_args.insert({DNNL_ARG_DIFF_BIAS, diff_bias_mem});
    vanilla_rnn_bwd_args.insert({DNNL_ARG_DIFF_DST_LAYER, diff_dst_layer_mem});
    vanilla_rnn_bwd_args.insert({DNNL_ARG_DIFF_SRC_ITER, diff_src_iter_mem});
    vanilla_rnn_bwd_args.insert({DNNL_ARG_DIFF_DST_ITER, diff_dst_iter_mem});
    vanilla_rnn_bwd_args.insert({DNNL_ARG_WORKSPACE, workspace_memory});

    // Primitive execution: vanilla rnn backward
    try {
        vanilla_rnn_bwd_prim.execute(engine_stream, vanilla_rnn_bwd_args);
    } catch (const dnnl::error &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return;
    }

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(diff_dst_layer_data.data(), diff_dst_layer_mem);

    //
    // User updates weights and bias using diffs
    //
}

int main(int argc, char **argv) {
    return handle_example_errors(vanilla_rnn, parse_engine_kind(argc, argv));
}
