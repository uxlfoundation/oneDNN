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

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "graph_example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

using namespace dnnl;
using tag = memory::format_tag;

using namespace dnnl::graph;
using layout_type = logical_tensor::layout_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;

struct sdpa_dims_t {
    dim mb;
    dim seq_len;
    dim head_num;
    dim head_size;
    dim query_num;
};

int main(int argc, char **argv) {

    // Create execution dnnl::engine.
    dnnl::engine eng(engine::kind::gpu, 0);
    // Create dnnl::stream.
    dnnl::stream strm(eng);

    sdpa_dims_t p = {1, 64, 1, 4, 32};

    // Prepare input and output shapes to construct the sdpa graph.
    const memory::dims q_sz = {p.mb, p.head_num, p.query_num, p.head_size};
    const memory::dims k_sz = {p.mb, p.head_num, p.head_size, p.seq_len};
    const memory::dims v_sz = {p.mb, p.head_num, p.seq_len, p.head_size};
    const memory::dims score_sz = {p.mb, p.head_num, p.query_num, p.seq_len};

    auto query_md = memory::desc(q_sz, memory::data_type::bf16, tag::abcd);
    auto key_md = memory::desc(k_sz, memory::data_type::bf16, tag::abdc);
    auto score_md = memory::desc(score_sz, memory::data_type::f32, tag::abcd);
    auto softmax_output_md
            = memory::desc(score_sz, memory::data_type::bf16, tag::abcd);

    primitive_attr bmm1_attr;
    bmm1_attr.set_scratchpad_mode(scratchpad_mode::user);
    auto bmm1_pd = matmul::primitive_desc(
            eng, query_md, key_md, score_md, bmm1_attr);
    auto bmm1_prim = matmul(bmm1_pd);

    // attention_probs = softmax(masked_score)
    primitive_attr softmax_attr;
    softmax_attr.set_scratchpad_mode(scratchpad_mode::user);
    auto softmax_pd
            = softmax_forward::primitive_desc(eng, prop_kind::forward_inference,
                    algorithm::softmax_accurate, score_md, softmax_output_md,
                    /* axis = */ score_md.get_ndims() - 1, softmax_attr);
    auto softmax_prim = softmax_forward(softmax_pd);
    // attention_output = attention_probs x value
    auto value_md = memory::desc(v_sz, memory::data_type::bf16, tag::abcd);
    auto output_md = memory::desc(q_sz, memory::data_type::bf16, tag::abcd);
    
    primitive_attr bmm2_attr;
    bmm2_attr.set_scratchpad_mode(scratchpad_mode::user);
    auto bmm2_pd = matmul::primitive_desc(
            eng, softmax_output_md, value_md, output_md, bmm2_attr);
    auto bmm2_prim = matmul(bmm2_pd);

    // Create memory objects
    auto m_query = memory(query_md, eng);
    auto m_key = memory(key_md, eng);
    auto m_value = memory(value_md, eng);
    auto m_output = memory(output_md, eng);

    auto output_f32_md = memory::desc(q_sz, memory::data_type::f32, tag::abcd);
    auto m_f32_output = memory(output_f32_md, eng);

    // Allocate user data.
    std::vector<float> query_data = {1, 4, 4, -3, 0, -1, -2, 2, -4, 1, 1, 3, 1,
            0, 3, 4, 2, 4, -2, -1, 4, 1, 4, 4, -3, 3, 4, -2, 4, -1, -2, 0, -1,
            3, 2, 0, 1, -4, 1, -1, 1, 3, 0, 2, -4, -3, -2, 3, -4, -3, 1, 1, 2,
            -2, -1, -1, 4, 4, 1, -3, 2, -1, 0, -4, 0, -4, 4, 4, -2, 0, 3, -4,
            -2, 3, -1, 3, 1, 0, -2, 4, 2, -3, 3, -2, 2, 0, 0, 1, 0, 2, 2, 4, -3,
            3, 4, -4, 0, 2, -1, 3, -1, 0, -4, 1, -4, 4, 4, -4, -2, 1, 0, 0, 2,
            0, 3, -4, 1, -4, -1, 0, 3, -4, 0, 0, 1, -1, 2, 1};
    std::vector<float> key_data = {6, -3, 4, -5, 1, -8, 0, -8, -6, -5, -4, -3,
            4, 4, 3, 2, 4, 7, -8, -6, -3, -3, -4, -5, -4, 4, -5, 3, 1, -2, -5,
            -8, -7, 3, -4, 6, -8, 4, 0, -5, -8, 7, 4, 2, 2, 7, -6, -1, -8, 6, 3,
            0, -6, -8, 7, 5, 0, -4, -8, 4, -3, -2, 0, 2, 4, -1, -7, 5, 2, -7, 2,
            -7, -1, -4, -6, -8, -2, -2, -2, -2, 0, 4, -8, -3, -6, 4, -3, 6, 5,
            2, -2, -5, 3, -2, -7, 5, -7, 3, -5, 4, -3, 6, -1, 8, 0, -1, -3, -4,
            -7, -7, -6, -6, 1, -2, -5, -8, 2, 1, -1, -2, 1, -7, 2, -6, -4, -8,
            4, 0, -3, -4, -5, -5, 2, 7, -4, 1, 8, 0, -7, 2, -7, -5, -3, -1, 6,
            4, 1, -2, -4, -6, -7, 8, -7, -8, -8, -8, 2, 2, 3, 3, -7, 4, -1, -7,
            -7, -1, 6, -5, -6, -5, -3, -2, 1, -8, 1, -7, 2, -7, 1, -8, -1, -8,
            2, -6, -3, 1, 6, -6, -1, 5, -7, -2, 1, 0, -1, -2, 1, 8, -3, 3, 2,
            -5, 5, -1, 0, 8, -1, 7, 6, 7, 8, -8, -8, -3, 2, 8, 4, 6, 7, -8, 5,
            6, 8, -8, 6, -4, 2, 8, 1, 4, 7, -7, -8, 1, -8, 1, -3, -4, -6, -7,
            -6, 2, -7, 2, 0, -5, 7, 2, -8, 0, 7, -3, -6, -3, 0, 2};

    std::vector<float> value_data = {6, 1, -6, 4, 4, -3, -4, 1, -7, -8, -8, 2,
            -8, -6, 0, -3, 4, 2, -1, -2, 0, -6, 5, 3, -7, -3, 0, -7, 1, 2, 1,
            -4, -3, 2, 8, -7, 6, -4, -7, 2, -7, -7, -6, 1, 2, -1, -3, -1, 1, 1,
            2, 0, 6, -8, 4, 5, 6, 1, -8, -3, -6, 0, -8, -6, -3, -8, -5, 4, 7,
            -3, 4, -2, 3, 4, 7, 7, 6, -8, -4, -2, -1, -7, -4, -2, 4, 4, 2, -2,
            3, 6, -1, -7, -2, 1, -7, -8, -4, 7, 0, -5, 4, -6, -8, 2, 4, -1, -5,
            -8, -7, -8, 1, 5, 0, 8, -5, 8, 7, -3, 6, 6, -4, 4, 1, -4, 2, -5, 0,
            -3, 4, 0, -4, 3, -8, -4, -5, -5, -4, 0, 4, -6, 3, 7, -8, 0, -7, 2,
            -6, -2, -8, -3, -2, -7, -5, -1, -3, -6, -5, -1, 2, 4, -5, -4, -7,
            -3, 1, -7, -8, 3, -1, 6, -3, 1, 1, 2, 6, -7, -1, -3, 5, -1, 8, 2, 7,
            8, 2, 7, -8, -6, -7, 7, 7, 0, -5, -8, -3, 2, -6, -5, 3, -8, 6, -5,
            2, -1, 0, 5, 4, 2, 5, -7, -8, -2, -3, 6, -5, 5, 4, 8, -4, -6, -8,
            -2, -6, 0, -5, 1, 2, -1, -2, 8, -8, 3, -7, -5, -2, -7, -8, -6, -6,
            -2, -2, 3, -1, 7, -8, 8, -8, -8, 8, -7, 1, -7, 2, 2, -3, 2};
    std::vector<float> mm1_output(product(score_sz));
    std::vector<uint16_t> softmax_output(product(score_sz));
    std::vector<uint16_t> output_data(product(q_sz));
    std::vector<float> f32_output_data(product(q_sz));

    // Write data to tensor object's handle.
    write_to_dnnl_memory(query_data.data(), m_query);
    write_to_dnnl_memory(key_data.data(), m_key);
    write_to_dnnl_memory(value_data.data(), m_value);

    size_t max_scratchpad_size = 0;
    auto bmm1_scratchpad = bmm1_pd.scratchpad_desc().get_size();
    auto softmax_scratchpad = softmax_pd.scratchpad_desc().get_size();
    auto bmm2_scratchpad = bmm2_pd.scratchpad_desc().get_size();
    for (auto &sz : {bmm1_scratchpad, softmax_scratchpad, bmm2_scratchpad}) {
        if (max_scratchpad_size < sz) max_scratchpad_size = sz;
    }
    auto scratchpad_md
            = memory::desc({static_cast<memory::dim>(max_scratchpad_size)},
                    memory::data_type::u8, tag::a);

    // allocate intermediate memory
    auto m_score = memory(score_md, eng);
    auto m_softmax_output = memory(softmax_output_md, eng);
    auto m_scratchpad = memory(scratchpad_md, eng);

    const auto loop = [&]() {
        // each primitive will use all threads
        bmm1_prim.execute(strm,
                {{DNNL_ARG_SRC, m_query}, {DNNL_ARG_WEIGHTS, m_key},
                        {DNNL_ARG_DST, m_score},
                        {DNNL_ARG_SCRATCHPAD, m_scratchpad}});

        softmax_prim.execute(strm,
                {{DNNL_ARG_SRC, m_score}, {DNNL_ARG_DST, m_softmax_output},
                        {DNNL_ARG_SCRATCHPAD, m_scratchpad}});

        bmm2_prim.execute(strm,
                {{DNNL_ARG_SRC, m_softmax_output}, {DNNL_ARG_WEIGHTS, m_value},
                        {DNNL_ARG_DST, m_output},
                        {DNNL_ARG_SCRATCHPAD, m_scratchpad}});

        reorder(m_output, m_f32_output).execute(strm, m_output, m_f32_output);
    };

    loop();

    // Wait for the computation to finish.
    strm.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(mm1_output.data(), m_score);
    read_from_dnnl_memory(softmax_output.data(), m_softmax_output);
    read_from_dnnl_memory(output_data.data(), m_output);
//     read_from_dnnl_memory(f32_output_data.data(), m_f32_output);

    std::cout << "MM1 output:\n";
    for (size_t idx = 0; idx < mm1_output.size(); ++idx) {
        float e = mm1_output[idx];
        std::cout << e << " ";
        if (idx % 32 == 31) std::cout << std::endl;
    }

    std::cout << "\nSoftmax output elements num: " << softmax_output.size()
              << " ,Softmax output:\n";
    for (size_t idx = 0; idx < softmax_output.size(); ++idx) {
        uint16_t e = softmax_output[idx];
        std::cout << e << " ";
        if (idx % 32 == 31) std::cout << std::endl;
    }

    std::cout << "\nMM2 output:\n";
    for (size_t idx = 0; idx < output_data.size(); ++idx) {
        uint16_t e = softmax_output[idx];
        std::cout << e << " ";
        if (idx % 32 == 31) std::cout << std::endl;
    }

//     std::cout << "\nMM2 f32 output:\n";
//     for (size_t idx = 0; idx < f32_output_data.size(); ++idx) {
//         float e = f32_output_data[idx];
//         std::cout << e << " ";
//         if (idx % 32 == 31) std::cout << std::endl;
//     }
}