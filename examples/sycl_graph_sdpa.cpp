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

/// @example sycl_graph_sdpa.cpp
/// @brief Demonstrates SYCL command graph recording and replay with oneDNN
///     primitives implementing Scaled Dot-Product Attention (SDPA).
///

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using namespace dnnl;

namespace sycl_ext = sycl::ext::oneapi::experimental;

/// Fill a host buffer with reproducible random values.
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

/// Fill attention mask: first 3/4 positions are 0 (attend), last 1/4 are
/// -infinity (mask out).
void fill_mask(std::vector<float> &mask, size_t seq_len) {
    const size_t pos = seq_len * 3 / 4;
    for (size_t i = 0; i < mask.size(); ++i) {
        mask[i] = (i % seq_len < pos) ? 0.f
                                      : -std::numeric_limits<float>::infinity();
    }
}

void sycl_graph_sdpa(engine::kind engine_kind) {
    // SDPA dimensions
    const memory::dim batch = 2;
    const memory::dim heads = 4;
    const memory::dim seq_len = 64;
    const memory::dim head_size = 32;
    const auto dt = memory::data_type::f32;

    /// Create a SYCL queue and use it to build the oneDNN engine and stream.
    /// The same queue will be used for SYCL graph recording, ensuring that
    /// primitive executions are captured by the graph.
    engine eng(engine_kind, 0);
    sycl::device dev = sycl_interop::get_device(eng);
    sycl::context ctx = sycl_interop::get_context(eng);
    sycl::queue q(ctx, dev);
    dnnl::stream strm = sycl_interop::make_stream(eng, q);

    // Define memory shapes for the SDPA layer.
    const memory::dims q_sz = {batch, heads, seq_len, head_size};
    const memory::dims k_sz = {batch, heads, head_size, seq_len};
    const memory::dims v_sz = {batch, heads, seq_len, head_size};
    const memory::dims score_sz = {batch, heads, seq_len, seq_len};
    const memory::dims scale_sz = {1, 1, 1, 1};
    const memory::dims mask_sz = {batch, 1, seq_len, seq_len};
    const memory::dims out_sz = q_sz;

    // Create memory descriptors.
    auto query_md = memory::desc(q_sz, dt, memory::format_tag::abcd);
    auto key_md = memory::desc(k_sz, dt, memory::format_tag::abdc);
    auto score_md = memory::desc(score_sz, dt, memory::format_tag::abcd);
    auto scale_md = memory::desc(scale_sz, dt, memory::format_tag::abcd);
    auto mask_md = memory::desc(mask_sz, dt, memory::format_tag::abcd);
    auto probs_md = memory::desc(score_sz, dt, memory::format_tag::abcd);
    auto value_md = memory::desc(v_sz, dt, memory::format_tag::abcd);
    auto output_md = memory::desc(out_sz, dt, memory::format_tag::abcd);

    /// Create SDPA primitives with user-managed scratchpad mode.
    /// Using user scratchpad ensures consistent memory addresses across
    /// SYCL graph record and replay.

    // BMM1: score = Q * K^T / sqrt(d_k) + mask
    primitive_attr bmm1_attr;
    bmm1_attr.set_scratchpad_mode(scratchpad_mode::user);
    post_ops bmm1_po;
    bmm1_po.append_binary(algorithm::binary_div, scale_md);
    bmm1_po.append_binary(algorithm::binary_add, mask_md);
    bmm1_attr.set_post_ops(bmm1_po);

    auto bmm1_pd = matmul::primitive_desc(
            eng, query_md, key_md, score_md, bmm1_attr);

    // Softmax: probs = softmax(score) along the last axis
    primitive_attr softmax_attr;
    softmax_attr.set_scratchpad_mode(scratchpad_mode::user);
    auto softmax_pd = softmax_forward::primitive_desc(eng,
            prop_kind::forward_inference, algorithm::softmax_accurate, score_md,
            probs_md, /* axis = */ score_md.get_ndims() - 1, softmax_attr);

    // BMM2: output = probs * V
    primitive_attr bmm2_attr;
    bmm2_attr.set_scratchpad_mode(scratchpad_mode::user);
    auto bmm2_pd = matmul::primitive_desc(
            eng, probs_md, value_md, output_md, bmm2_attr);

    /// Allocate USM device memory for inputs, intermediates, and output.
    const size_t n_q = product(q_sz);
    const size_t n_k = product(k_sz);
    const size_t n_v = product(v_sz);
    const size_t n_score = product(score_sz);
    const size_t n_mask = product(mask_sz);
    const size_t n_out = product(out_sz);

    auto *q_data = sycl::malloc_device<float>(n_q, dev, ctx);
    auto *k_data = sycl::malloc_device<float>(n_k, dev, ctx);
    auto *v_data = sycl::malloc_device<float>(n_v, dev, ctx);
    auto *scale_data = sycl::malloc_device<float>(1, dev, ctx);
    auto *mask_data = sycl::malloc_device<float>(n_mask, dev, ctx);
    auto *score_data = sycl::malloc_device<float>(n_score, dev, ctx);
    auto *probs_data = sycl::malloc_device<float>(n_score, dev, ctx);
    auto *out_data = sycl::malloc_device<float>(n_out, dev, ctx);

    /// Wrap USM device pointers in oneDNN memory objects.
    auto m_query = sycl_interop::make_memory(
            query_md, eng, sycl_interop::memory_kind::usm, q_data);
    auto m_key = sycl_interop::make_memory(
            key_md, eng, sycl_interop::memory_kind::usm, k_data);
    auto m_scale = sycl_interop::make_memory(
            scale_md, eng, sycl_interop::memory_kind::usm, scale_data);
    auto m_mask = sycl_interop::make_memory(
            mask_md, eng, sycl_interop::memory_kind::usm, mask_data);
    auto m_value = sycl_interop::make_memory(
            value_md, eng, sycl_interop::memory_kind::usm, v_data);
    auto m_score = sycl_interop::make_memory(
            score_md, eng, sycl_interop::memory_kind::usm, score_data);
    auto m_probs = sycl_interop::make_memory(
            probs_md, eng, sycl_interop::memory_kind::usm, probs_data);
    auto m_output = sycl_interop::make_memory(
            output_md, eng, sycl_interop::memory_kind::usm, out_data);

    // Prepare host-side input data and write to device memory objects.
    std::vector<float> query_data(n_q);
    std::vector<float> key_data(n_k);
    std::vector<float> value_data(n_v);
    std::vector<float> scale_host(1, std::sqrt(static_cast<float>(head_size)));
    std::vector<float> mask_host(n_mask);

    fill_random(query_data);
    fill_random(key_data);
    fill_random(value_data);
    fill_mask(mask_host, static_cast<size_t>(seq_len));

    write_to_dnnl_memory(query_data.data(), m_query);
    write_to_dnnl_memory(key_data.data(), m_key);
    write_to_dnnl_memory(value_data.data(), m_value);
    write_to_dnnl_memory(scale_host.data(), m_scale);
    write_to_dnnl_memory(mask_host.data(), m_mask);

    // Allocate separate scratchpad memory for each primitive.
    // Using separate scratchpads ensures no conflicts within the SYCL graph.
    auto *bmm1_sp_data = sycl::malloc_device<uint8_t>(
            bmm1_pd.scratchpad_desc().get_size(), dev, ctx);
    auto *softmax_sp_data = sycl::malloc_device<uint8_t>(
            softmax_pd.scratchpad_desc().get_size(), dev, ctx);
    auto *bmm2_sp_data = sycl::malloc_device<uint8_t>(
            bmm2_pd.scratchpad_desc().get_size(), dev, ctx);

    auto m_bmm1_sp = sycl_interop::make_memory(bmm1_pd.scratchpad_desc(), eng,
            sycl_interop::memory_kind::usm, bmm1_sp_data);
    auto m_softmax_sp = sycl_interop::make_memory(softmax_pd.scratchpad_desc(),
            eng, sycl_interop::memory_kind::usm, softmax_sp_data);
    auto m_bmm2_sp = sycl_interop::make_memory(bmm2_pd.scratchpad_desc(), eng,
            sycl_interop::memory_kind::usm, bmm2_sp_data);

    /// Build argument maps for each primitive.
    std::unordered_map<int, memory> bmm1_args = {{DNNL_ARG_SRC, m_query},
            {DNNL_ARG_WEIGHTS, m_key}, {DNNL_ARG_DST, m_score},
            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, m_scale},
            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1, m_mask},
            {DNNL_ARG_SCRATCHPAD, m_bmm1_sp}};

    std::unordered_map<int, memory> softmax_args = {{DNNL_ARG_SRC, m_score},
            {DNNL_ARG_DST, m_probs}, {DNNL_ARG_SCRATCHPAD, m_softmax_sp}};

    std::unordered_map<int, memory> bmm2_args
            = {{DNNL_ARG_SRC, m_probs}, {DNNL_ARG_WEIGHTS, m_value},
                    {DNNL_ARG_DST, m_output}, {DNNL_ARG_SCRATCHPAD, m_bmm2_sp}};

    // Declare ref_output and exec_graph outside the scope so they survive
    // past primitive destruction.
    std::vector<float> ref_output(n_out);
    sycl_ext::command_graph<sycl_ext::graph_state::executable> exec_graph
            = [&] {
        /// Create primitive objects inside a scope block. They will
        /// be automatically destroyed when the scope exits, before
        /// graph replay begins.
        auto bmm1_prim = matmul(bmm1_pd);
        auto softmax_prim = softmax_forward(softmax_pd);
        auto bmm2_prim = matmul(bmm2_pd);

        /// Step 1: Reference execution using
        /// dnnl::sycl_interop::execute(). Each primitive returns a
        /// SYCL event. Passing the returned event as a dependency
        /// to the next primitive call creates a chain:
        ///   BMM1 -> Softmax -> BMM2
        std::cout << "Step 1: Reference SDPA execution..." << std::endl;

        auto e1 = sycl_interop::execute(bmm1_prim, strm, bmm1_args);
        auto e2 = sycl_interop::execute(softmax_prim, strm, softmax_args, {e1});
        auto e3 = sycl_interop::execute(bmm2_prim, strm, bmm2_args, {e2});
        e3.wait();

        // Read reference output from device memory.
        read_from_dnnl_memory(ref_output.data(), m_output);

        std::cout << "  Reference execution completed." << std::endl;

        /// Step 2: Record the same SDPA execution into a SYCL
        /// command graph. When the queue is in recording mode,
        /// dnnl::sycl_interop::execute() calls are captured as
        /// graph nodes. The SYCL events passed between calls become
        /// edges in the graph, preserving the execution order.
        std::cout << "Step 2: Recording SDPA into SYCL command " << "graph..."
                  << std::endl;

        sycl_ext::command_graph<sycl_ext::graph_state::modifiable> graph(
                ctx, dev);

        graph.begin_recording(q);

        auto ge1 = sycl_interop::execute(bmm1_prim, strm, bmm1_args);
        auto ge2 = sycl_interop::execute(
                softmax_prim, strm, softmax_args, {ge1});
        auto ge3 = sycl_interop::execute(bmm2_prim, strm, bmm2_args, {ge2});

        graph.end_recording();

        std::cout << "  Graph recorded and finalized." << std::endl;

        // Finalize and return the executable graph.
        return graph.finalize();
    }();

    /// Step 3: Replay the SYCL command graph after primitives are destroyed.
    /// The SYCL command graph is self-contained: it captured all the GPU
    /// kernels during recording. The oneDNN primitive objects are no longer
    /// needed for replay, demonstrating that the graph owns the execution
    /// state independently.
    std::cout << "Step 3: Primitive objects destroyed. "
              << "Replaying SYCL command graph..." << std::endl;

    // Clear output to ensure replay actually computes new results.
    q.memset(out_data, 0, n_out * sizeof(float)).wait();

    q.submit([&](sycl::handler &cgh) {
        cgh.ext_oneapi_graph(exec_graph);
    }).wait();

    // Read replay output from device and verify against the reference.
    std::vector<float> replay_output(n_out);
    read_from_dnnl_memory(replay_output.data(), m_output);

    const float threshold = 1e-5f;
    for (size_t i = 0; i < static_cast<size_t>(n_out); i++) {
        float diff = std::abs(replay_output[i] - ref_output[i]);
        if (diff > threshold) {
            throw std::runtime_error("Graph replay mismatch at index "
                    + std::to_string(i)
                    + ": graph=" + std::to_string(replay_output[i])
                    + " ref=" + std::to_string(ref_output[i]));
        }
    }
    std::cout << "  First replay verified successfully." << std::endl;

    /// Step 4: Replay again to demonstrate that the graph can be executed
    /// multiple times, producing consistent results each time.
    std::cout << "Step 4: Replaying SYCL command graph again..." << std::endl;

    q.memset(out_data, 0, n_out * sizeof(float)).wait();

    q.submit([&](sycl::handler &cgh) {
        cgh.ext_oneapi_graph(exec_graph);
    }).wait();

    read_from_dnnl_memory(replay_output.data(), m_output);

    for (size_t i = 0; i < static_cast<size_t>(n_out); i++) {
        float diff = std::abs(replay_output[i] - ref_output[i]);
        if (diff > threshold) {
            throw std::runtime_error(
                    "Second replay mismatch at index " + std::to_string(i));
        }
    }
    std::cout << "  Second replay verified successfully." << std::endl;

    // Cleanup USM allocations.
    sycl::free(q_data, ctx);
    sycl::free(k_data, ctx);
    sycl::free(v_data, ctx);
    sycl::free(scale_data, ctx);
    sycl::free(mask_data, ctx);
    sycl::free(score_data, ctx);
    sycl::free(probs_data, ctx);
    sycl::free(out_data, ctx);
    sycl::free(bmm1_sp_data, ctx);
    sycl::free(softmax_sp_data, ctx);
    sycl::free(bmm2_sp_data, ctx);
}

int main(int argc, char **argv) {
    int exit_code = 0;

    engine::kind engine_kind = parse_engine_kind(argc, argv);
    try {
        sycl_graph_sdpa(engine_kind);
    } catch (dnnl::error &e) {
        std::cout << "oneDNN error caught: " << std::endl
                  << "\tStatus: " << dnnl_status2str(e.status) << std::endl
                  << "\tMessage: " << e.what() << std::endl;
        exit_code = 1;
    } catch (std::exception &e) {
        std::cout << "Error in the example: " << e.what() << std::endl;
        exit_code = 2;
    }

    std::cout << "Example " << (exit_code ? "failed" : "passed") << " on "
              << engine_kind2str_upper(engine_kind) << "." << std::endl;
    finalize();
    return exit_code;
}
