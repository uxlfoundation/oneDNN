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
using property_type = logical_tensor::property_type;
using data_type = logical_tensor::data_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;

struct sdpa_dims_t {
    dim mb;
    dim seq_len;
    dim head_num;
    dim head_size;
    dim query_num;
};

static const int min_runs = 4;

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

void print_test_case(memory::data_type dt, const sdpa_dims_t &p) {
    std::cout << '[' << std::setw(4) << dnnl_dt2str(memory::convert_to_c(dt));
    std::cout << " mb = " << p.mb << ", seq_len = " << p.seq_len
              << ", head_num = " << p.head_num
              << ", head_size = " << p.head_size
              << ", query_num = " << p.query_num;
    std::cout << "] " << std::flush;
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
    const bool quick_test = (time_limit == 0.);
    print_test_case(dt, p);

    allocator alloc = create_allocator(ekind);

    // Create execution dnnl::engine.
    dnnl::engine eng = make_engine_with_allocator(ekind, 0, alloc);
    dnnl::engine host_eng(engine::kind::cpu, 0);
    // Create dnnl::stream.
    dnnl::stream strm(eng);

    // Prepare input and output shapes to construct the sdpa graph.
    const dims qv_sz = {p.mb, p.head_num, p.query_num, p.head_size};
    const dims k_sz = {p.mb, p.head_num, p.seq_len, p.head_size};
    const dims score_sz = {p.mb, p.head_num, p.query_num, p.seq_len};
    const dims scale_sz = {1};

    // Incremental IDs used to create logical tensors and operations.
    size_t id = 0;

    // score = query x key.T
    auto query = logical_tensor(id++, dt, qv_sz, layout_type::strided);
    auto key = logical_tensor(id++, dt, k_sz, layout_type::strided);
    auto score = logical_tensor(id++, dt, score_sz, layout_type::strided);
    auto bmm1 = op(id++, op::kind::MatMul, "bmm1");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);
    bmm1.add_inputs({query, key});
    bmm1.add_outputs({score});

    // scale_mul_out_lt = score * scale
    auto scale = logical_tensor(id++, dt, scale_sz, layout_type::strided);
    auto scale_mul_out_lt
            = logical_tensor(id++, dt, score_sz, layout_type::strided);
    auto scale_mul_op = op(id++, op::kind::Multiply, "scale_mul_op");
    scale_mul_op.add_inputs({score, scale});
    scale_mul_op.add_outputs({scale_mul_out_lt});

    /* bottom right implicit causal mask

                 gen_index_row_op - mask_add_op - mask_sub_op                   scale_mul_out_lt
                /                                         \                      |
    scale_mul_out_lt                                         mask_greater_equal - mask_select_op -
                \                                         /                      |
                  --------- gen_index_col_op  ---------                      -inf
    */

    auto gen_index_row_op = op(id++, op::kind::GenIndex, "gen_index_row_op");
    gen_index_row_op.set_attr<int64_t>(op::attr::axis, -2);
    auto gen_index_row_out_lt = logical_tensor(
            id++, data_type::s32, score_sz, layout_type::strided);
    gen_index_row_op.add_inputs({scale_mul_out_lt});
    gen_index_row_op.add_outputs({gen_index_row_out_lt});

    auto mask_add_op = op(id++, op::kind::Add, "mask_add_op");
    auto mask_add_in_lt = logical_tensor(id++, data_type::s32, 0,
            layout_type::strided, property_type::host_scalar); //scalar
    auto mask_add_out_lt = logical_tensor(
            id++, data_type::s32, score_sz, layout_type::strided);
    mask_add_op.add_inputs({gen_index_row_out_lt, mask_add_in_lt});
    mask_add_op.add_outputs({mask_add_out_lt});

    auto mask_sub_op = op(id++, op::kind::Subtract, "mask_sub_op");
    auto mask_sub_op_in_lt = logical_tensor(id++, data_type::s32, 0,
            layout_type::strided, property_type::host_scalar); //scalar
    auto mask_sub_op_out_lt = logical_tensor(
            id++, data_type::s32, score_sz, layout_type::strided);
    mask_sub_op.add_inputs({mask_add_out_lt, mask_sub_op_in_lt});
    mask_sub_op.add_outputs({mask_sub_op_out_lt});

    auto gen_index_col_op = op(id++, op::kind::GenIndex, "gen_index_col_op");
    gen_index_col_op.set_attr<int64_t>(op::attr::axis, -1);
    auto gen_index_col_out_lt = logical_tensor(
            id++, data_type::s32, score_sz, layout_type::strided);
    gen_index_col_op.add_inputs({scale_mul_out_lt});
    gen_index_col_op.add_outputs({gen_index_col_out_lt});

    auto mask_ge_op = op(id++, op::kind::GreaterEqual, "mask_ge_op");
    auto mask_ge_out_lt = logical_tensor(
            id++, data_type::boolean, score_sz, layout_type::strided);
    mask_ge_op.add_inputs({mask_sub_op_out_lt, gen_index_col_out_lt});
    mask_ge_op.add_outputs({mask_ge_out_lt});

    auto mask_select_op = op(id++, op::kind::Select, "mask_select_op");
    auto neg_inf = logical_tensor(id++, dt, scale_sz, layout_type::strided);
    auto masked_score
            = logical_tensor(id++, dt, score_sz, layout_type::strided);
    mask_select_op.add_inputs({mask_ge_out_lt, scale_mul_out_lt, neg_inf});
    mask_select_op.add_outputs({masked_score});

    // attention_probs = softmax(masked_score)
    auto probs = logical_tensor(id++, dt, score_sz, layout_type::strided);
    auto softmax = op(id++, op::kind::SoftMax, "softmax");
    softmax.set_attr<int64_t>(op::attr::axis, -1);
    softmax.add_inputs({masked_score});
    softmax.add_outputs({probs});

    // attention_output = attention_probs x value
    auto value = logical_tensor(id++, dt, k_sz, layout_type::strided);
    auto output = logical_tensor(id++, dt, qv_sz, layout_type::strided);
    auto bmm2 = op(id++, op::kind::MatMul, "bmm2");
    bmm2.add_inputs({probs, value});
    bmm2.add_outputs({output});

    // Construct a sdpa graph with engine kind and operations.
    dnnl::graph::graph sdpa(ekind);
    sdpa.add_op(bmm1);
    sdpa.add_op(scale_mul_op);
    sdpa.add_op(gen_index_row_op);
    sdpa.add_op(mask_add_op);
    sdpa.add_op(mask_sub_op);
    sdpa.add_op(gen_index_col_op);
    sdpa.add_op(mask_ge_op);
    sdpa.add_op(mask_select_op);
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
    compiled_partition cp
            = partitions[0].compile({query, key, scale, mask_add_in_lt,
                                            mask_sub_op_in_lt, neg_inf, value},
                    {output}, eng);

    int32_t seq_len_kv = p.seq_len;
    int32_t seq_len_q = p.seq_len;
    auto ts_mask_add = tensor(mask_add_in_lt, host_eng, &seq_len_kv);
    auto ts_mask_sub = tensor(mask_sub_op_in_lt, host_eng, &seq_len_q);
    // Create tensor objects
    auto ts_query = tensor(query, eng);
    auto ts_key = tensor(key, eng);
    auto ts_scale = tensor(scale, eng);
    auto ts_neg_inf = tensor(neg_inf, eng);
    auto ts_value = tensor(value, eng);
    auto ts_output = tensor(output, eng);

    // Allocate user data.
    std::vector<float> query_data(product(qv_sz));
    std::vector<float> key_data(product(k_sz));
    std::vector<float> scale_data(product(scale_sz), std::sqrt(p.head_size));
    std::vector<float> neg_inf_data(product(scale_sz), -1e30);
    std::vector<float> value_data(product(k_sz));
    std::vector<float> output_data(product(qv_sz));

    fill_random(query_data);
    fill_random(key_data);
    fill_random(value_data);

    // Write data to tensor object's handle.
    write_to_dnnl_tensor(query_data.data(), ts_query);
    write_to_dnnl_tensor(key_data.data(), ts_key);
    write_to_dnnl_tensor(scale_data.data(), ts_scale);
    write_to_dnnl_tensor(neg_inf_data.data(), ts_neg_inf);
    write_to_dnnl_tensor(value_data.data(), ts_value);

    // Warmup run.
    // Execute the compiled partition of sdpa.
    cp.execute(strm,
            {ts_query, ts_key, ts_scale, ts_mask_add, ts_mask_sub, ts_neg_inf,
                    ts_value},
            {ts_output});

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
    try {
        bench_sdpa(ekind, static_cast<data_type>(dt), p, time_limit);
        get_mem_pool().clear();
    } catch (dnnl::error &e) {
        // Catch and report unimplemented cases.
        if (e.status == dnnl_unimplemented) {
            std::cout << "unsupported sdpa" << std::endl;
        } else
            throw;
    }
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
