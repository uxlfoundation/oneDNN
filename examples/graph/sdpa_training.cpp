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

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

#include "graph_example_utils.hpp"

using namespace dnnl;

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

// this is changed from the fill_random() function in matmul_perf.cpp.
void fill_random(std::vector<float> &out) {
    static std::vector<float> random_data_f;
    constexpr size_t nrand = 1037;

    if (random_data_f.empty()) {
        std::mt19937 generator;
        std::uniform_real_distribution<float> dist_f(-1.0f, 1.0f);

        random_data_f.resize(nrand);
        for (auto &d : random_data_f)
            d = dist_f(generator);
    }

    for (size_t i = 0; i < out.size(); i += nrand) {
        size_t chunk = std::min(nrand, out.size() - i);
        std::memcpy(&out[i], random_data_f.data(), chunk * sizeof(float));
    }
}

// initialize the mask with first 3/4 elements with 0s and the last 1/4 elements
// with -inf.
void fill_mask(std::vector<float> &mask, size_t seq_len) {
    const size_t pos = seq_len * 3 / 4;
    for (size_t i = 0; i < mask.size(); ++i) {
        if (i % seq_len < pos)
            mask[i] = 0.f;
        else
            mask[i] = -1 * std::numeric_limits<float>::infinity();
    }
}

const char *get_type_string(logical_tensor::data_type dt) {
    const char *type_string = "unknown";

#define TYPE_CASE(T) \
    if (dt == logical_tensor::data_type::T) type_string = #T;
    TYPE_CASE(f16);
    TYPE_CASE(f32);
    TYPE_CASE(bf16);
#undef TYPE_CASE

    return type_string;
}

void print_test_case(logical_tensor::data_type dt, const sdpa_dims_t &p) {
    std::cout << '[' << std::setw(4) << get_type_string(dt);
    std::cout << " mb = " << p.mb << ", seq_len = " << p.seq_len
              << ", head_num = " << p.head_num
              << ", head_size = " << p.head_size
              << ", query_num = " << p.query_num;
    std::cout << "] " << std::flush;
}

void bench_sdpa(engine::kind ekind, logical_tensor::data_type dt,
        const sdpa_dims_t &p, double time_limit = 0.) {
    print_test_case(dt, p);

    allocator alloc = create_allocator(ekind);

    // Create execution dnnl::engine.
    dnnl::engine eng = make_engine_with_allocator(ekind, 0, alloc);
    // Create dnnl::stream.
    dnnl::stream strm(eng);

    // Prepare input and output shapes to construct the sdpa graph.
    const dims qv_sz = {p.mb, p.head_num, p.seq_len, p.head_size};
    const dims k_sz = {p.mb, p.head_num, p.seq_len, p.head_size};
    const dims score_sz = {p.mb, p.head_num, p.seq_len, p.seq_len};
    const dims stats_sz = {p.mb, p.head_num, p.seq_len, 1};
    const dims scale_sz = {1};
    const dims mask_sz = {p.mb, p.head_num, p.seq_len, p.seq_len};

    // Incremental IDs used to create logical tensors and operations.
    size_t id = 0;

    // Intermediate data type
    const logical_tensor::data_type dt_inter = logical_tensor::data_type::f32;

    // -------------------------forward graph--------------------------
    // score = query x key.T
    auto query = logical_tensor(id++, dt, qv_sz, layout_type::strided);
    auto key = logical_tensor(id++, dt, k_sz, layout_type::strided);
    auto score = logical_tensor(id++, dt_inter, score_sz, layout_type::strided);
    auto bmm1 = op(id++, op::kind::MatMul, "bmm1");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);
    bmm1.add_inputs({query, key});
    bmm1.add_outputs({score});

    // scaled_score = score / scale
    auto scale = logical_tensor(id++, dt_inter, scale_sz, layout_type::strided);
    auto scaled_score
            = logical_tensor(id++, dt_inter, score_sz, layout_type::strided);
    auto scale_div = op(id++, op::kind::Divide, "scale_div");
    scale_div.add_inputs({score, scale});
    scale_div.add_outputs({scaled_score});

    // masked_score = scaled_score + mask
    auto mask = logical_tensor(id++, dt_inter, mask_sz, layout_type::strided);
    auto masked_score
            = logical_tensor(id++, dt_inter, score_sz, layout_type::strided);
    auto mask_add = op(id++, op::kind::Add, "mask_add");
    mask_add.add_inputs({scaled_score, mask});
    mask_add.add_outputs({masked_score});

    // attention_probs = softmax(masked_score)
    auto probs = logical_tensor(id++, dt, score_sz, layout_type::strided);
    auto stats = logical_tensor(id++, dt_inter, stats_sz, layout_type::strided);
    auto softmax = op(id++, op::kind::SoftMax, "softmax");
    softmax.set_attr<int64_t>(op::attr::axis, -1);
    softmax.set_attr<std::string>(op::attr::mode, "inf_as_zero");
    softmax.add_inputs({masked_score});
    softmax.add_outputs({probs});
    softmax.add_outputs({stats});

    // attention_output = attention_probs x value
    auto value = logical_tensor(id++, dt, k_sz, layout_type::strided);
    auto output = logical_tensor(id++, dt, qv_sz, layout_type::strided);
    auto bmm2 = op(id++, op::kind::MatMul, "bmm2");
    bmm2.add_inputs({probs, value});
    bmm2.add_outputs({output});

    // Construct a sdpa graph with engine kind and operations.
    dnnl::graph::graph sdpa(ekind);
    sdpa.add_op(bmm1);
    sdpa.add_op(scale_div);
    sdpa.add_op(mask_add);
    sdpa.add_op(softmax);
    sdpa.add_op(bmm2);
    sdpa.finalize();

    // Get partitions from the sdpa graph.
    std::vector<partition> partitions = sdpa.get_partitions();
    // This is just for oneDNN testing purpose.
    if (partitions.size() != 1) {
        std::cout << "unsupported sdpa" << std::endl;
        return;
    }

    // Compile the partition with inputs, outputs, and an engine.
    compiled_partition cp = partitions[0].compile(
            {query, key, scale, mask, value}, {output, stats}, eng);

    // Create tensor objects
    auto ts_query = tensor(query, eng);
    auto ts_key = tensor(key, eng);
    auto ts_scale = tensor(scale, eng);
    auto ts_mask = tensor(mask, eng);
    auto ts_value = tensor(value, eng);
    auto ts_output = tensor(output, eng);
    auto ts_stats = tensor(stats, eng);

    // Allocate user data.
    std::vector<float> query_data(product(qv_sz));
    std::vector<float> key_data(product(k_sz));
    std::vector<float> scale_data(product(scale_sz), std::sqrt(p.head_size));
    std::vector<float> mask_data(product(mask_sz));
    std::vector<float> value_data(product(k_sz));
    std::vector<float> output_data(product(qv_sz));
    std::vector<float> stats_data(product(stats_sz));

    fill_random(query_data);
    fill_random(key_data);
    fill_random(value_data);
    fill_mask(mask_data, static_cast<size_t>(p.seq_len));

    // Write data to tensor object's handle.
    write_to_dnnl_tensor(query_data.data(), ts_query);
    write_to_dnnl_tensor(key_data.data(), ts_key);
    write_to_dnnl_tensor(scale_data.data(), ts_scale);
    write_to_dnnl_tensor(mask_data.data(), ts_mask);
    write_to_dnnl_tensor(value_data.data(), ts_value);

    // Warmup run.
    // Execute the compiled partition of sdpa.
    cp.execute(strm, {ts_query, ts_key, ts_scale, ts_mask, ts_value},
            {ts_output, ts_stats});

    // Wait for the computation to finish.
    strm.wait();

    // -------------------------backward graph--------------------------
    // score = query x key.T
    auto query_b = logical_tensor(id++, dt, qv_sz, layout_type::strided);
    auto key_b = logical_tensor(id++, dt, k_sz, layout_type::strided);
    auto score_b
            = logical_tensor(id++, dt_inter, score_sz, layout_type::strided);
    auto bmm1_b = op(id++, op::kind::MatMul, "bmm1");
    bmm1_b.set_attr<bool>(op::attr::transpose_b, true);
    bmm1_b.add_inputs({query_b, key_b});
    bmm1_b.add_outputs({score_b});

    // scaled_score = score / scale
    auto scale_b
            = logical_tensor(id++, dt_inter, scale_sz, layout_type::strided);
    auto scaled_score_b
            = logical_tensor(id++, dt_inter, score_sz, layout_type::strided);
    auto scale_div_b = op(id++, op::kind::Divide, "scale_div");
    scale_div_b.add_inputs({score_b, scale_b});
    scale_div_b.add_outputs({scaled_score_b});

    // masked_score = scaled_score + mask
    auto mask_b = logical_tensor(id++, dt_inter, mask_sz, layout_type::strided);
    auto masked_score_b
            = logical_tensor(id++, dt_inter, score_sz, layout_type::strided);
    auto mask_add_b = op(id++, op::kind::Add, "mask_add");
    mask_add_b.add_inputs({scaled_score_b, mask_b});
    mask_add_b.add_outputs({masked_score_b});

    // attention_probs = softmax(masked_score) = exp(masked_score - stats)
    auto stats_b
            = logical_tensor(id++, dt_inter, stats_sz, layout_type::strided);
    auto sub_out_b = logical_tensor(id++, dt, score_sz, layout_type::strided);
    auto subtract_b = op(id++, op::kind::Subtract, "subtract");
    subtract_b.add_inputs({masked_score_b, stats_b});
    subtract_b.add_outputs({sub_out_b});

    auto probs_b = logical_tensor(id++, dt, score_sz, layout_type::strided);
    auto exp_b = op(id++, op::kind::Exp, "exp");
    exp_b.add_inputs({sub_out_b});
    exp_b.add_outputs({probs_b});

    // compute dvalue = P^T * doutput
    auto doutput = logical_tensor(id++, dt, qv_sz, layout_type::strided);
    auto dvalue = logical_tensor(id++, dt, k_sz, layout_type::strided);
    auto bmm_p_do = op(id++, op::kind::MatMul, "bmm1");
    bmm_p_do.set_attr<bool>(op::attr::transpose_a, true);
    bmm_p_do.add_inputs({probs_b, doutput});
    bmm_p_do.add_outputs({dvalue});

    // compute dprobs = doutput * value^T
    auto value_b = logical_tensor(id++, dt, k_sz, layout_type::strided);
    auto dprobs = logical_tensor(id++, dt, score_sz, layout_type::strided);
    auto bmm_do_v = op(id++, op::kind::MatMul, "bmm2");
    bmm_do_v.set_attr<bool>(op::attr::transpose_b, true);
    bmm_do_v.add_inputs({doutput, value_b});
    bmm_do_v.add_outputs({dprobs});

    // compute dmasked_score =  dsoftmax(dprobs)
    auto dmasked_score
            = logical_tensor(id++, dt_inter, score_sz, layout_type::strided);
    auto softmax_grad = op(id++, op::kind::SoftMaxBackward, "softmax_bwd");
    softmax_grad.set_attr<int64_t>(op::attr::axis, -1);
    softmax_grad.add_inputs({dprobs, probs_b});
    softmax_grad.add_outputs({dmasked_score});

    // compute dscored_score = dmasked_score / scale
    auto dscaled_score
            = logical_tensor(id++, dt_inter, score_sz, layout_type::strided);
    auto scale_div_b2 = op(id++, op::kind::Divide, "scale_div");
    scale_div_b2.add_inputs({dmasked_score, scale_b});
    scale_div_b2.add_outputs({dscaled_score});

    // compute dquery = dscaled_score * key
    auto dquery = logical_tensor(id++, dt, qv_sz, layout_type::strided);
    auto bmm_dscaled_score_k = op(id++, op::kind::MatMul, "bmm3");
    bmm_dscaled_score_k.add_inputs({dscaled_score, key_b});
    bmm_dscaled_score_k.add_outputs({dquery});

    // compute dkey = dscaled_score^T * query
    auto dkey = logical_tensor(id++, dt, k_sz, layout_type::strided);
    auto bmm_dscaled_score_q = op(id++, op::kind::MatMul, "bmm4");
    bmm_dscaled_score_q.set_attr<bool>(op::attr::transpose_a, true);
    bmm_dscaled_score_q.add_inputs({dscaled_score, query_b});
    bmm_dscaled_score_q.add_outputs({dkey});

    // Construct a sdpa graph with engine kind and operations.
    dnnl::graph::graph sdpa_bwd(ekind);
    sdpa_bwd.add_op(bmm1_b);
    sdpa_bwd.add_op(scale_div_b);
    sdpa_bwd.add_op(mask_add_b);
    sdpa_bwd.add_op(subtract_b);
    sdpa_bwd.add_op(exp_b);
    sdpa_bwd.add_op(bmm_p_do);
    sdpa_bwd.add_op(bmm_do_v);
    sdpa_bwd.add_op(softmax_grad);
    sdpa_bwd.add_op(scale_div_b2);
    sdpa_bwd.add_op(bmm_dscaled_score_k);
    sdpa_bwd.add_op(bmm_dscaled_score_q);
    sdpa_bwd.finalize();

    // Get partitions from the sdpa graph.
    std::vector<partition> partitions_b = sdpa_bwd.get_partitions();
    // This is just for oneDNN testing purpose.
    if (partitions_b.size() != 1) {
        std::cout << "unsupported sdpa" << std::endl;
        return;
    }

    // Compile the partition with inputs, outputs, and an engine.
    compiled_partition cp_b
            = partitions_b[0].compile({query_b, key_b, scale_b, mask_b, value_b,
                                              output, stats_b, doutput},
                    {dquery, dkey, dvalue}, eng);

    // Create tensor objects
    auto ts_doutput = tensor(doutput, eng);
    auto ts_dquery = tensor(dquery, eng);
    auto ts_dkey = tensor(dkey, eng);
    auto ts_dvalue = tensor(dvalue, eng);

    // Allocate user data.
    std::vector<float> doutput_data(product(qv_sz));
    fill_random(doutput_data);

    // Use the data from the forward pass.
    // write_to_dnnl_tensor(query_data.data(), ts_query);
    // write_to_dnnl_tensor(key_data.data(), ts_key);
    // write_to_dnnl_tensor(scale_data.data(), ts_scale);
    // write_to_dnnl_tensor(mask_data.data(), ts_mask);
    // write_to_dnnl_tensor(value_data.data(), ts_value);
    // Write data to tensor object's handle.
    write_to_dnnl_tensor(doutput_data.data(), ts_doutput);

    // Warmup run.
    // Execute the compiled partition of sdpa.
    cp_b.execute(strm,
            {ts_query, ts_key, ts_scale, ts_mask, ts_value, ts_output, ts_stats,
                    ts_doutput},
            {ts_dquery, ts_dkey, ts_dvalue});

    // Wait for the computation to finish.
    strm.wait();
}

void bad_args() {
    std::cerr << "Usage: graph-sdpa-cpp [cpu|gpu]\n"
                 "       graph-sdpa-cpp [cpu|gpu] <mb> <seq_len> "
                 "<head_num> <head_size> [<query_num>]\n\n"
                 "On CPU, it's recommended to test with numactl and memory "
                 "allocation tools like jemalloc or tcmalloc.\n\n";
    throw std::invalid_argument("Incorrect input arguments.");
}

void bench(engine::kind ekind, dnnl_data_type_t dt, const sdpa_dims_t &p,
        double time_limit = 0.) {
    bench_sdpa(
            ekind, static_cast<logical_tensor::data_type>(dt), p, time_limit);
    get_mem_pool().clear();
}

void sdpa_perf(engine::kind ekind, int argc, char **argv) {
    // default testing parameters
    sdpa_dims_t params = {32, 384, 16, 64, 384};

    if (argc > 2) {
        if (argc == 6) {
            params.mb = std::atoi(argv[2]);
            params.seq_len = std::atoi(argv[3]);
            params.query_num = std::atoi(argv[3]);
            params.head_num = std::atoi(argv[4]);
            params.head_size = std::atoi(argv[5]);
        } else if (argc == 7) {
            params.mb = std::atoi(argv[2]);
            params.seq_len = std::atoi(argv[3]);
            params.head_num = std::atoi(argv[4]);
            params.head_size = std::atoi(argv[5]);
            params.query_num = std::atoi(argv[6]);
        } else {
            bad_args();
        }

        if (params.mb <= 0 || params.seq_len <= 0 || params.head_num <= 0
                || params.head_size <= 0) {
            bad_args();
        }
    }

    bench(ekind, dnnl_f32, params, 2000.0 /*ms*/);
    bench(ekind, dnnl_bf16, params, 2000.0 /*ms*/);
    bench(ekind, dnnl_f16, params, 2000.0 /*ms*/);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            sdpa_perf, parse_engine_kind(argc, argv, 5), argc, argv);
}
