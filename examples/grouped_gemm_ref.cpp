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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

// Configuration constants
constexpr int NUM_EXPERTS = 4;
constexpr int TOP_K = 2; // number of top experts to select per token
constexpr int TOKENS_TO_PROCESS = 8;
constexpr int INPUT_DIM = 16;
constexpr int HIDDEN_DIM = 32;
constexpr int OUTPUT_DIM = 16;

/// Initialize a matrix with predictable values for testing
void init_matrix(float *data, int rows, int cols, float seed = 1.0f) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i * cols + j] = seed * ((i * cols + j) % 10) / 10.0f;
        }
    }
}

/// Print a matrix for debugging
void print_matrix(const char *name, const float *data, int rows, int cols,
        int max_rows = 3, int max_cols = 5) {
    std::cout << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < std::min(rows, max_rows); i++) {
        std::cout << "  [";
        for (int j = 0; j < std::min(cols, max_cols); j++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%7.3f", data[i * cols + j]);
            std::cout << buf;
            if (j < std::min(cols, max_cols) - 1) std::cout << ", ";
        }
        if (cols > max_cols) std::cout << ", ...";
        std::cout << "]\n";
    }
    if (rows > max_rows) std::cout << "  ...\n";
    std::cout << "\n";
}

/// Apply ReLU activation
void apply_relu(float *data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = std::max(0.0f, data[i]);
    }
}

/// Structure to hold routing decisions
struct RoutingDecision {
    std::vector<int> expert_ids; // Expert IDs for each token (TOP_K per token)
    std::vector<float> weights; // Routing weights (TOP_K per token)
    std::vector<int> token_counts; // Number of tokens assigned to each expert
    std::vector<std::vector<int>> expert_token_ids; // Token IDs for each expert
    std::vector<int> token_offsets; // Start offset for each expert's tokens
};

/// Structure to hold expert weight offsets in contiguous memory
///
/// Memory layout example for W1 with 4 experts:
///   [Expert0_W1 | Expert1_W1 | Expert2_W1 | Expert3_W1]
///   ^           ^             ^             ^             ^
///   offset[0]   offset[1]     offset[2]     offset[3]     offset[4]
///
/// This allows efficient access: W1_for_expert_i = W1_data + W1_offsets[i]
struct ExpertWeights {
    const float *W1_data; // Contiguous memory for all W1 weights
    const float *b1_data; // Contiguous memory for all b1 biases
    const float *W2_data; // Contiguous memory for all W2 weights
    const float *b2_data; // Contiguous memory for all b2 biases
    std::vector<int> W1_offsets; // Offset for each expert's W1 (num_experts+1)
    std::vector<int> b1_offsets; // Offset for each expert's b1 (num_experts+1)
    std::vector<int> W2_offsets; // Offset for each expert's W2 (num_experts+1)
    std::vector<int> b2_offsets; // Offset for each expert's b2 (num_experts+1)
};

/// Predictable toy routing function using top-2 strategy
/// This function computes routing logits based on input features and selects
/// the top-2 experts for each token with normalized weights.
void compute_routing(const float *input, int tokens_to_process, int input_dim,
        int num_experts, int top_k, RoutingDecision &routing) {
    // Initialize routing decision structures
    routing.expert_ids.resize(tokens_to_process * top_k);
    routing.weights.resize(tokens_to_process * top_k);
    routing.token_counts.resize(num_experts, 0);
    routing.expert_token_ids.resize(num_experts);
    routing.token_offsets.resize(num_experts + 1, 0);

    // Compute routing logits for each token
    std::vector<float> logits(tokens_to_process * num_experts);
    for (int b = 0; b < tokens_to_process; ++b) {
        // Simple predictable routing: sum of input features modulo num_experts
        float feature_sum = 0.0f;
        for (int d = 0; d < input_dim; ++d) {
            feature_sum += input[b * input_dim + d];
        }

        // Generate predictable logits for each expert
        for (int e = 0; e < num_experts; ++e) {
            // Create different patterns for each expert
            logits[b * num_experts + e]
                    = std::sin((feature_sum + e) * 0.5f) + e * 0.3f;
        }
    }

    // Select top-k experts for each token
    for (int b = 0; b < tokens_to_process; ++b) {
        // Create indices and sort by logits
        std::vector<int> indices(num_experts);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int i, int j) {
            return logits[b * num_experts + i] > logits[b * num_experts + j];
        });

        // Select top-k experts
        float weight_sum = 0.0f;
        std::vector<float> top_k_logits(top_k);
        for (int k = 0; k < top_k; ++k) {
            int expert_id = indices[k];
            float logit = logits[b * num_experts + expert_id];
            top_k_logits[k] = std::exp(logit);
            weight_sum += top_k_logits[k];
        }

        // Normalize weights and store routing decisions
        for (int k = 0; k < top_k; ++k) {
            int expert_id = indices[k];
            float weight = top_k_logits[k] / weight_sum;

            routing.expert_ids[b * top_k + k] = expert_id;
            routing.weights[b * top_k + k] = weight;

            // Track tokens assigned to each expert
            routing.expert_token_ids[expert_id].push_back(b);
            routing.token_counts[expert_id]++;
        }
    }

    // Compute token offsets for each expert (cumulative sum)
    for (int e = 0; e < num_experts; ++e) {
        routing.token_offsets[e + 1]
                = routing.token_offsets[e] + routing.token_counts[e];
    }
}

/// Build a contiguous input buffer with all tokens organized by expert
/// Gathers tokens from input and arranges them by expert in grouped_input
void gather_grouped_input(const float *input, float *grouped_input,
        const RoutingDecision &routing, int input_dim) {
    for (int expert_id = 0; expert_id < NUM_EXPERTS; ++expert_id) {
        int num_tokens = routing.token_counts[expert_id];
        if (num_tokens == 0) continue;

        int offset = routing.token_offsets[expert_id];
        const int *token_ids = routing.expert_token_ids[expert_id].data();
        float *expert_gathered = grouped_input + offset * input_dim;

        // Copy each token's data to the grouped buffer
        for (int i = 0; i < num_tokens; ++i) {
            int token_id = token_ids[i];
            const float *src = input + token_id * input_dim;
            float *dst = expert_gathered + i * input_dim;
            std::copy(src, src + input_dim, dst);
        }
    }
}

/// Scatter expert outputs back to original token positions with routing weights
/// Distributes grouped_output back to output, applying routing weights
void scatter_grouped_output(float *output, const float *grouped_output,
        const RoutingDecision &routing, int output_dim) {
    for (int expert_id = 0; expert_id < NUM_EXPERTS; ++expert_id) {
        int num_expert_tokens = routing.token_counts[expert_id];
        if (num_expert_tokens == 0) continue;

        int token_offset = routing.token_offsets[expert_id];
        const float *expert_output = grouped_output + token_offset * output_dim;
        const int *token_ids = routing.expert_token_ids[expert_id].data();

        // Scatter this expert's outputs back to original token positions
        for (int i = 0; i < num_expert_tokens; ++i) {
            int token_id = token_ids[i];

            // Find the routing weight for this token-expert pair
            float weight = 0.0f;
            for (int k = 0; k < TOP_K; ++k) {
                if (routing.expert_ids[token_id * TOP_K + k] == expert_id) {
                    weight = routing.weights[token_id * TOP_K + k];
                    break;
                }
            }

            // Accumulate weighted expert output to final output
            for (int d = 0; d < output_dim; ++d) {
                output[token_id * output_dim + d]
                        += weight * expert_output[i * output_dim + d];
            }
        }
    }
}

/// Reference Grouped GEMM with optional scales support
/// Supports per-tensor, row-wise (src), and column-wise (wei) scales
/// Computes: output = ((input * scale_src) * (weights * scale_wei) + bias) / scale_dst
///
/// @param scale_src_ptrs Array of pointers to src scales per expert (nullptr = no scaling)
/// @param scale_wei_ptrs Array of pointers to wei scales per expert (nullptr = no scaling)
/// @param scale_dst_ptrs Array of pointers to dst scales per expert (nullptr = no scaling)
/// @param src_scale_size Size of src scales: 1 for per-tensor, M for row-wise (nullptr if no scaling)
/// @param wei_scale_size Size of wei scales: 1 for per-tensor, N for column-wise (nullptr if no scaling)
void ref_grouped_gemm(const float **input_ptrs, float **output_ptrs,
        const float **weight_ptrs, const float **bias_ptrs,
        const int *M_per_expert, int num_experts, int K_dim, int N_dim,
        const float **scale_src_ptrs = nullptr,
        const float **scale_wei_ptrs = nullptr,
        const float **scale_dst_ptrs = nullptr,
        const int *src_scale_size = nullptr,
        const int *wei_scale_size = nullptr) {

    for (int expert_id = 0; expert_id < num_experts; ++expert_id) {
        int num_expert_tokens = M_per_expert[expert_id];

        const float *expert_input = input_ptrs[expert_id];
        float *expert_output = output_ptrs[expert_id];
        const float *W = weight_ptrs[expert_id];
        const float *b = bias_ptrs[expert_id];

        const float *src_scales
                = scale_src_ptrs ? scale_src_ptrs[expert_id] : nullptr;
        const float *wei_scales
                = scale_wei_ptrs ? scale_wei_ptrs[expert_id] : nullptr;
        const float *dst_scales
                = scale_dst_ptrs ? scale_dst_ptrs[expert_id] : nullptr;

        // Using size here to determine scaling strategy (row-wise, column-wise, per-tensor)
        int src_scale_sz = src_scale_size ? src_scale_size[expert_id] : 0;
        int wei_scale_sz = wei_scale_size ? wei_scale_size[expert_id] : 0;

        // GEMM with optional scales: output = ((input * scale_src) * (W * scale_wei) + bias) / scale_dst
        //
        // Input: (num_expert_tokens x K_dim)
        // Weights: (K_dim x N_dim)
        // Output dimensions: (num_expert_tokens x N_dim)
        for (int m = 0; m < num_expert_tokens; ++m) {
            // Get src scale: per-tensor (size == 1) or row-wise (size == M)
            float src_scale = 1.0f;
            if (src_scales) {
                src_scale = (src_scale_sz == 1) ? src_scales[0] : src_scales[m];
            }

            for (int n = 0; n < N_dim; ++n) {
                // Get wei scale: per-tensor (size == 1) or column-wise (size == N)
                float wei_scale = 1.0f;
                if (wei_scales) {
                    wei_scale = (wei_scale_sz == 1) ? wei_scales[0]
                                                    : wei_scales[n];
                }

                float sum = 0.0f;
                for (int k = 0; k < K_dim; ++k) {
                    float scaled_input
                            = expert_input[m * K_dim + k] * src_scale;
                    float scaled_weight = W[k * N_dim + n] * wei_scale;
                    sum += scaled_input * scaled_weight;
                }

                // Add bias before dst scale division (quantization semantics)
                sum += b[n];

                // Get dst scale (always per-tensor) and divide
                float dst_scale = dst_scales ? dst_scales[0] : 1.0f;
                expert_output[m * N_dim + n] = sum / dst_scale;
            }
        }
    }
}

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"
using namespace dnnl;

const engine &eng() {
    static const engine eng(engine::kind::cpu, 0);
    return eng;
}

/// oneDNN-style Grouped GEMM Implementation with optional scales support
/// @param use_scales If true, applies row-wise src scales,
/// column-wise wei scales, and per-tensor dst scales
void onednn_style_grouped_gemm(const float **input_ptrs, float **output_ptrs,
        const float **weight_ptrs, const float **bias_ptrs,
        const int *M_per_expert, int num_experts /* active experts */,
        int K_dim, int N_dim, bool use_scales = false) {
#if 0
    if (use_scales) {
        std::cout << "Grouped GEMM with Bias and Scales\n";
    } else {
        std::cout << "Grouped GEMM with Bias\n";
    }

    // Step 1: Create memory descriptors with runtime dimensions
    //
    // Use DNNL_RUNTIME_DIM_VAL for dimensions that vary per expert
    // - Input(A):   shape = [RUNTIME, K_dim] - size varies per expert
    // - Weights(B): shape = [K_dim, N_dim]   - fixed size for all experts
    // - Output(C):  shape = [RUNTIME, N_dim]
    //
    // TODO: Memory format 'ab' means row-major 2D layout,
    // check that weight should follow 'ab' and not the transpose
    memory::dims a_shape = {DNNL_RUNTIME_DIM_VAL, K_dim};
    memory::dims b_shape = {K_dim, N_dim};
    memory::dims bias_shape = {N_dim};
    memory::dims c_shape = {DNNL_RUNTIME_DIM_VAL, N_dim};

    memory::desc a_md(a_shape, memory::data_type::f32, memory::format_tag::ab);
    memory::desc b_md(b_shape, memory::data_type::f32, memory::format_tag::ab);
    memory::desc bias_md(
            bias_shape, memory::data_type::f32, memory::format_tag::a);
    memory::desc c_md(c_shape, memory::data_type::f32, memory::format_tag::ab);

    // Step 2: Create vector of descriptors to be passed later as
    // grouped gemm argument.
    //
    // NOTE: Since assumption that weight matrices are the same size for all experts,
    // we can reuse the same memory descriptor for weights across all experts.
    // Same logic for inputs and outputs, since we're using wildcard dimensions.
    std::vector<memory::desc> a_mds(num_experts, a_md);
    std::vector<memory::desc> b_mds(num_experts, b_md);
    std::vector<memory::desc> bias_mds(num_experts, bias_md);
    std::vector<memory::desc> c_mds(num_experts, c_md);

    // Step 3: Create primitive attributes with optional scales
    primitive_attr attr;
    if (use_scales) {
        // Assuming all experts have the same scaling strategy, but different scaling factors values
        for (int i = 0; i < num_experts; i++) {
            attr.set_scales_mask(
                    DNNL_ARG_MULTIPLE_SRC + i, 1); // row-wise src mask
            attr.set_scales_mask(
                    DNNL_ARG_MULTIPLE_WEIGHTS + i, 2); // column-wise wei mask
            attr.set_scales_mask(
                    DNNL_ARG_MULTIPLE_DST + i, 0); // per-tensor dst mask
        }
    }

    // Step 4: Create primitive descriptor
    //
    // - The number of experts is determined by vector sizes (a_mds.size())
    // - Actual sizes (M_per_expert) will be specified during execution
    auto grouped_gemm_pd = grouped_gemm::primitive_desc(
            eng(), a_mds, b_mds, bias_mds, c_mds, attr);

    // Step 5: Wrap user data in oneDNN memory objects
    //
    // NOTE: zero-copy approach to translate to oneDNN memory objects.
    // - In Pytorch and OpenVINO, memory is a contiguous buffer.
    // - We create memory objects that point directly to user data for each expert,
    //   once the dimensions M_per_expert are known.
    std::vector<memory> a_mem, b_mem, bias_mem, c_mem;
    for (int i = 0; i < num_experts; i++) {
        // Wrap input data: shape [M_per_expert[i] x K_dim]
        a_mem.push_back(
                memory({{M_per_expert[i], K_dim}, memory::data_type::f32,
                               memory::format_tag::ab},
                        eng(), (void *)input_ptrs[i]));

        // Wrap weight data: shape [K_dim x N_dim]
        b_mem.push_back(memory({{K_dim, N_dim}, memory::data_type::f32,
                                       memory::format_tag::ab},
                eng(), (void *)weight_ptrs[i]));

        // Wrap bias data: shape [N_dim]
        bias_mem.push_back(
                memory({{N_dim}, memory::data_type::f32, memory::format_tag::a},
                        eng(), (void *)bias_ptrs[i]));

        // Wrap output data: shape [M_per_expert[i] x N_dim]
        c_mem.push_back(
                memory({{M_per_expert[i], N_dim}, memory::data_type::f32,
                               memory::format_tag::ab},
                        eng(), (void *)output_ptrs[i]));
    }

    // Step 6: Fill scaling factors and create memory objects (if using scales)
    std::vector<std::vector<float>> scale_src_data;
    std::vector<std::vector<float>> scale_wei_data;
    std::vector<std::vector<float>> scale_dst_data;
    std::vector<memory> scale_src_mem, scale_wei_mem, scale_dst_mem;

    if (use_scales) {
        scale_src_data.resize(num_experts);
        scale_wei_data.resize(num_experts);
        scale_dst_data.resize(num_experts);

        // NOTE: Row of scales for source, column of scales for weights, single scale for dst
        for (int i = 0; i < num_experts; i++) {
            int M = M_per_expert[i];
            scale_src_data[i].resize(M);
            for (int m = 0; m < M; ++m) {
                scale_src_data[i][m] = 0.8f + m * 0.06f;
            }
            scale_wei_data[i].resize(N_dim);
            for (int n = 0; n < N_dim; ++n) {
                scale_wei_data[i][n] = 1.2f + n * 0.04f;
            }
            scale_dst_data[i] = {0.5f};
        }

        for (int i = 0; i < num_experts; i++) {
            // Source scales
            if (!scale_src_data[i].empty()) {
                scale_src_mem.push_back(memory(
                        {{(int)scale_src_data[i].size()},
                                memory::data_type::f32, memory::format_tag::a},
                        eng(), scale_src_data[i].data()));
            }

            // Weight scales
            if (!scale_wei_data[i].empty()) {
                scale_wei_mem.push_back(memory(
                        {{(int)scale_wei_data[i].size()},
                                memory::data_type::f32, memory::format_tag::a},
                        eng(), scale_wei_data[i].data()));
            }

            // Destination scales
            if (!scale_dst_data[i].empty()) {
                scale_dst_mem.push_back(memory(
                        {{(int)scale_dst_data[i].size()},
                                memory::data_type::f32, memory::format_tag::a},
                        eng(), scale_dst_data[i].data()));
            }
        }
    }

    // Step 7: Create primitive
    auto grouped_gemm_prim = grouped_gemm(grouped_gemm_pd);

    // Step 8: Build argument map for execution
    //
    // Map each expert's data to the appropriate primitive argument:
    //      DNNL_ARG_MULTIPLE_SRC + i      -> input for expert i
    //      DNNL_ARG_MULTIPLE_WEIGHTS + i  -> weights for expert i
    //      DNNL_ARG_MULTIPLE_BIAS + i     -> bias for expert i
    //      DNNL_ARG_MULTIPLE_DST + i      -> output for expert i
    std::unordered_map<int, memory> grouped_gemm_args;
    for (int i = 0; i < num_experts; i++) {
        grouped_gemm_args.insert({DNNL_ARG_MULTIPLE_SRC + i, a_mem[i]});
        grouped_gemm_args.insert({DNNL_ARG_MULTIPLE_WEIGHTS + i, b_mem[i]});
        grouped_gemm_args.insert({DNNL_ARG_MULTIPLE_BIAS + i, bias_mem[i]});
        grouped_gemm_args.insert({DNNL_ARG_MULTIPLE_DST + i, c_mem[i]});

        if (use_scales) {
            grouped_gemm_args.insert(
                    {DNNL_ARG_ATTR_SCALES | (DNNL_ARG_MULTIPLE_SRC + i),
                            scale_src_mem[i]});
            grouped_gemm_args.insert(
                    {DNNL_ARG_ATTR_SCALES | (DNNL_ARG_MULTIPLE_WEIGHTS + i),
                            scale_wei_mem[i]});
            grouped_gemm_args.insert(
                    {DNNL_ARG_ATTR_SCALES | (DNNL_ARG_MULTIPLE_DST + i),
                            scale_dst_mem[i]});
        }
    }

    // Step 9: Execute the grouped GEMM
    dnnl::stream engine_stream(eng());
    grouped_gemm_prim.execute(engine_stream, grouped_gemm_args);
    engine_stream.wait();

    // Temporary step to get same output as ref implementation
    for (int i = 0; i < num_experts; i++) {
        read_from_dnnl_memory(output_ptrs[i], c_mem[i]);
    }
#endif
}

/// Process all experts through a 2-layer MLP (expert network)
/// Each expert: input -> GEMM1 + bias -> ReLU -> GEMM2 + bias -> output
/// Uses grouped_gemm for both layers (which handles GEMM + bias)
void process_experts_mlp(const float *grouped_input, float *grouped_output,
        const ExpertWeights &weights, const RoutingDecision &routing,
        int input_dim, int hidden_dim, int output_dim) {

    // Allocate buffer for intermediate hidden layer (all experts combined)
    int total_tokens = routing.token_offsets[NUM_EXPERTS];
    std::vector<float> hidden_all(total_tokens * hidden_dim);

    // Identify active experts (== those with tokens assigned)
    std::vector<int> active_expert_ids;
    for (int i = 0; i < NUM_EXPERTS; ++i) {
        if (routing.token_counts[i] > 0) { active_expert_ids.push_back(i); }
    }
    int num_active_experts = active_expert_ids.size();

    // Build pointer arrays ONLY for active experts (zero-copy - just pointers into contiguous buffers)
    std::vector<const float *> input_ptrs(num_active_experts);
    std::vector<float *> hidden_ptrs(num_active_experts);
    std::vector<const float *> W1_ptrs(num_active_experts);
    std::vector<const float *> b1_ptrs(num_active_experts);
    std::vector<int> M_per_expert(num_active_experts);

    for (int idx = 0; idx < num_active_experts; ++idx) {
        int expert_id = active_expert_ids[idx];
        int token_start = routing.token_offsets[expert_id];
        M_per_expert[idx] = routing.token_offsets[expert_id + 1] - token_start;

        input_ptrs[idx] = grouped_input + token_start * input_dim;
        hidden_ptrs[idx] = hidden_all.data() + token_start * hidden_dim;
        W1_ptrs[idx] = weights.W1_data + weights.W1_offsets[expert_id];
        b1_ptrs[idx] = weights.b1_data + weights.b1_offsets[expert_id];
    }

    // Save hidden layer output for comparison
    std::vector<float> hidden_all_ref(total_tokens * hidden_dim);
    std::vector<float> hidden_all_onednn(total_tokens * hidden_dim);

    std::vector<float *> hidden_ptrs_ref(num_active_experts);
    std::vector<float *> hidden_ptrs_onednn(num_active_experts);
    for (int idx = 0; idx < num_active_experts; ++idx) {
        int expert_id = active_expert_ids[idx];
        int token_start = routing.token_offsets[expert_id];
        hidden_ptrs_ref[idx] = hidden_all_ref.data() + token_start * hidden_dim;
        hidden_ptrs_onednn[idx]
                = hidden_all_onednn.data() + token_start * hidden_dim;
    }

    // First layer: input -> hidden (GEMM + bias via ref_grouped_gemm, no scales)
    ref_grouped_gemm(input_ptrs.data(), hidden_ptrs_ref.data(), W1_ptrs.data(),
            b1_ptrs.data(), M_per_expert.data(), num_active_experts, input_dim,
            hidden_dim);

    // First layer: input -> hidden (GEMM + bias via onednn_style_grouped_gemm, no scales)
    onednn_style_grouped_gemm(input_ptrs.data(), hidden_ptrs_onednn.data(),
            W1_ptrs.data(), b1_ptrs.data(), M_per_expert.data(),
            num_active_experts, input_dim, hidden_dim, false);

    // Compare outputs
    float max_diff = 0.0f;
    float total_diff = 0.0f;
    int num_elements = total_tokens * hidden_dim;
    for (int i = 0; i < num_elements; ++i) {
        float diff = std::abs(hidden_all_ref[i] - hidden_all_onednn[i]);
        max_diff = std::max(max_diff, diff);
        total_diff += diff;
    }
    float avg_diff = total_diff / num_elements;

    std::cout << "=== Layer 1 GEMM Comparison ===\n";
    char buf[64];
    snprintf(buf, sizeof(buf), "  Max difference: %.10f\n", max_diff);
    std::cout << buf;
    snprintf(buf, sizeof(buf), "  Avg difference: %.10f\n", avg_diff);
    std::cout << buf;
    if (max_diff < 1e-6f) {
        std::cout << "  Status: PASS\n";
    } else {
        std::cout << "  Status: FAIL\n";
        // Print first few values for debugging
        std::cout << "  First 10 values comparison:\n";
        for (int i = 0; i < std::min(10, num_elements); ++i) {
            snprintf(buf, sizeof(buf),
                    "    [%d] ref=%.6f onednn=%.6f diff=%.6f\n", i,
                    hidden_all_ref[i], hidden_all_onednn[i],
                    hidden_all_ref[i] - hidden_all_onednn[i]);
            std::cout << buf;
        }
    }
    std::cout << "\n";

    // Use ref output for the rest of the pipeline
    hidden_all = hidden_all_ref;
    for (int idx = 0; idx < num_active_experts; ++idx) {
        int expert_id = active_expert_ids[idx];
        int token_start = routing.token_offsets[expert_id];
        hidden_ptrs[idx] = hidden_all.data() + token_start * hidden_dim;
    }

    // Apply ReLU activation to hidden layer
    apply_relu(hidden_all.data(), total_tokens * hidden_dim);

    // Build pointer arrays for second layer (ONLY for active experts)
    std::vector<float *> output_ptrs(num_active_experts);
    std::vector<const float *> W2_ptrs(num_active_experts);
    std::vector<const float *> b2_ptrs(num_active_experts);

    for (int idx = 0; idx < num_active_experts; ++idx) {
        int expert_id = active_expert_ids[idx];
        int token_start = routing.token_offsets[expert_id];
        output_ptrs[idx] = grouped_output + token_start * output_dim;
        W2_ptrs[idx] = weights.W2_data + weights.W2_offsets[expert_id];
        b2_ptrs[idx] = weights.b2_data + weights.b2_offsets[expert_id];
    }

    // Save output for comparison
    std::vector<float> output_all_ref(total_tokens * output_dim);
    std::vector<float> output_all_scaled(total_tokens * output_dim);

    std::vector<float *> output_ptrs_ref(num_active_experts);
    std::vector<float *> output_ptrs_scaled(num_active_experts);
    for (int idx = 0; idx < num_active_experts; ++idx) {
        int expert_id = active_expert_ids[idx];
        int token_start = routing.token_offsets[expert_id];
        output_ptrs_ref[idx] = output_all_ref.data() + token_start * output_dim;
        output_ptrs_scaled[idx]
                = output_all_scaled.data() + token_start * output_dim;
    }

    // Prepare scales:
    //   src: row-wise (0.8 + m*0.06), wei: col-wise (1.2 + n*0.04), dst: 0.5
    std::vector<std::vector<float>> scale_src_data(num_active_experts);
    std::vector<std::vector<float>> scale_wei_data(num_active_experts);
    std::vector<std::vector<float>> scale_dst_data(num_active_experts);

    std::vector<const float *> scale_src_ptrs(num_active_experts);
    std::vector<const float *> scale_wei_ptrs(num_active_experts);
    std::vector<const float *> scale_dst_ptrs(num_active_experts);
    std::vector<int> src_scale_sizes(num_active_experts);
    std::vector<int> wei_scale_sizes(num_active_experts);

    for (int idx = 0; idx < num_active_experts; ++idx) {
        int M = M_per_expert[idx];

        // Row-wise src scales
        scale_src_data[idx].resize(M);
        for (int m = 0; m < M; ++m) {
            scale_src_data[idx][m] = 0.8f + m * 0.06f;
        }
        scale_src_ptrs[idx] = scale_src_data[idx].data();
        src_scale_sizes[idx] = M;

        // Column-wise wei scales
        scale_wei_data[idx].resize(output_dim);
        for (int n = 0; n < output_dim; ++n) {
            scale_wei_data[idx][n] = 1.2f + n * 0.04f;
        }
        scale_wei_ptrs[idx] = scale_wei_data[idx].data();
        wei_scale_sizes[idx] = output_dim;

        // Per-tensor dst scale
        scale_dst_data[idx] = {0.5f};
        scale_dst_ptrs[idx] = scale_dst_data[idx].data();
    }

    // Second layer: hidden -> output (ref with scales)
    ref_grouped_gemm((const float **)hidden_ptrs.data(), output_ptrs_ref.data(),
            W2_ptrs.data(), b2_ptrs.data(), M_per_expert.data(),
            num_active_experts, hidden_dim, output_dim, scale_src_ptrs.data(),
            scale_wei_ptrs.data(), scale_dst_ptrs.data(),
            src_scale_sizes.data(), wei_scale_sizes.data());

    // Second layer with scales: hidden -> output (oneDNN implementation with scales)
    onednn_style_grouped_gemm((const float **)hidden_ptrs.data(),
            output_ptrs_scaled.data(), W2_ptrs.data(), b2_ptrs.data(),
            M_per_expert.data(), num_active_experts, hidden_dim, output_dim,
            true);

    // Compare outputs
    max_diff = 0.0f;
    total_diff = 0.0f;
    num_elements = total_tokens * output_dim;
    for (int i = 0; i < num_elements; ++i) {
        float diff = std::abs(output_all_ref[i] - output_all_scaled[i]);
        max_diff = std::max(max_diff, diff);
        total_diff += diff;
    }
    avg_diff = total_diff / num_elements;

    std::cout << "=== Layer 2 GEMM with Scales Comparison ===\n";
    snprintf(buf, sizeof(buf), "  Max difference: %.10f\n", max_diff);
    std::cout << buf;
    snprintf(buf, sizeof(buf), "  Avg difference: %.10f\n", avg_diff);
    std::cout << buf;
    if (max_diff < 1e-6f) {
        std::cout << "  Status: PASS\n";
    } else {
        std::cout << "  Status: FAIL\n";
        // Print first few values for debugging
        std::cout << "  First 10 values comparison:\n";
        for (int i = 0; i < std::min(10, num_elements); ++i) {
            snprintf(buf, sizeof(buf),
                    "    [%d] ref=%.6f scaled=%.6f diff=%.6f\n", i,
                    output_all_ref[i], output_all_scaled[i],
                    output_all_ref[i] - output_all_scaled[i]);
            std::cout << buf;
        }
    }
    std::cout << "\n";

    // Use ref output for the rest of the pipeline
    for (int i = 0; i < total_tokens * output_dim; ++i) {
        grouped_output[i] = output_all_ref[i];
    }
}

// ============================================================================
// MoE Pipeline
// ============================================================================
//
// Architecture:
//   toy_moe() - Complete MoE pipeline: Gather → Process Experts → Scatter
//
// Component separation:
//   - gather_grouped_input(): Reorganizes input tokens by expert assignment
//   - process_experts_mlp(): Processes each expert's tokens through 2-layer MLP
//       - Uses grouped_gemm() for each layer (swappable GEMM implementation)
//   - scatter_grouped_output(): Distributes expert outputs back to original positions
//
// Weight Organization:
//   - All expert weights stored in contiguous memory (single allocation)
//   - Access via offsets: W1_data + W1_offsets[expert_id]
//   - Benefits: cache-friendly, efficient memory management
//
// This modular design allows:
//   - Easy replacement of grouped_gemm with oneDNN primitives
//   - Clear performance measurement of data movement vs computation
//   - Independent testing and optimization of each component
// ============================================================================
void toy_moe(const float *input, float *output, const ExpertWeights &weights,
        const RoutingDecision &routing, int tokens_to_process, int input_dim,
        int hidden_dim, int output_dim) {

    std::fill(output, output + tokens_to_process * output_dim, 0.0f);

    // Allocate buffers for grouped input and output
    int total_tokens = routing.token_offsets[NUM_EXPERTS];
    std::vector<float> grouped_input(total_tokens * input_dim);
    std::vector<float> grouped_output(total_tokens * output_dim);

    // Step 1: Gather - organize tokens by expert
    gather_grouped_input(input, grouped_input.data(), routing, input_dim);

    // Step 2: Process all experts through their MLPs
    process_experts_mlp(grouped_input.data(), grouped_output.data(), weights,
            routing, input_dim, hidden_dim, output_dim);

    // Step 3: Scatter - distribute results back with routing weights
    scatter_grouped_output(output, grouped_output.data(), routing, output_dim);
}

int main(int argc, char **argv) {
    std::cout << "Toy MoE/Grouped GEMM Implementation with 4 Experts (Top-2 "
                 "Routing)\n\n";

    std::cout << "    Total Number of experts: " << NUM_EXPERTS << "\n";
    std::cout << "    Number of active experts (Top-K routing): " << TOP_K
              << "\n";
    std::cout << "    Total Tokens to process: " << TOKENS_TO_PROCESS << "\n";
    std::cout << "    Input dimension: " << INPUT_DIM << "\n";
    std::cout << "    Hidden dimension: " << HIDDEN_DIM << "\n";
    std::cout << "    Output dimension: " << OUTPUT_DIM << "\n\n";

    // Allocate input and output
    std::vector<float> input(TOKENS_TO_PROCESS * INPUT_DIM);
    std::vector<float> output(TOKENS_TO_PROCESS * OUTPUT_DIM);

    // Initialize input
    init_matrix(input.data(), TOKENS_TO_PROCESS, INPUT_DIM, 1.0f);
    print_matrix("Input", input.data(), TOKENS_TO_PROCESS, INPUT_DIM);

    // Allocate contiguous memory for all expert weights
    std::vector<float> W1_all(NUM_EXPERTS * INPUT_DIM * HIDDEN_DIM);
    std::vector<float> b1_all(NUM_EXPERTS * HIDDEN_DIM);
    std::vector<float> W2_all(NUM_EXPERTS * HIDDEN_DIM * OUTPUT_DIM);
    std::vector<float> b2_all(NUM_EXPERTS * OUTPUT_DIM);

    // Initialize each expert's weights with different patterns
    for (int e = 0; e < NUM_EXPERTS; ++e) {
        init_matrix(W1_all.data() + e * INPUT_DIM * HIDDEN_DIM, INPUT_DIM,
                HIDDEN_DIM, 1.0f + e * 0.1f);
        init_matrix(b1_all.data() + e * HIDDEN_DIM, 1, HIDDEN_DIM,
                0.1f + e * 0.01f);
        init_matrix(W2_all.data() + e * HIDDEN_DIM * OUTPUT_DIM, HIDDEN_DIM,
                OUTPUT_DIM, 0.5f + e * 0.1f);
        init_matrix(b2_all.data() + e * OUTPUT_DIM, 1, OUTPUT_DIM,
                0.05f + e * 0.01f);
    }

    // Setup expert weights structure with offsets
    ExpertWeights expert_weights;
    expert_weights.W1_data = W1_all.data();
    expert_weights.b1_data = b1_all.data();
    expert_weights.W2_data = W2_all.data();
    expert_weights.b2_data = b2_all.data();

    // Initialize offsets (each expert's weights start at regular intervals)
    expert_weights.W1_offsets.resize(NUM_EXPERTS + 1);
    expert_weights.b1_offsets.resize(NUM_EXPERTS + 1);
    expert_weights.W2_offsets.resize(NUM_EXPERTS + 1);
    expert_weights.b2_offsets.resize(NUM_EXPERTS + 1);

    for (int e = 0; e <= NUM_EXPERTS; ++e) {
        expert_weights.W1_offsets[e] = e * INPUT_DIM * HIDDEN_DIM;
        expert_weights.b1_offsets[e] = e * HIDDEN_DIM;
        expert_weights.W2_offsets[e] = e * HIDDEN_DIM * OUTPUT_DIM;
        expert_weights.b2_offsets[e] = e * OUTPUT_DIM;
    }

    // Compute routing
    RoutingDecision routing;
    compute_routing(input.data(), TOKENS_TO_PROCESS, INPUT_DIM, NUM_EXPERTS,
            TOP_K, routing);

    // Print routing decisions
    std::cout << "Routing Decisions (Top-" << TOP_K << "):\n";
    std::cout << "  Token | Choice#1 | Choice#2\n";
    std::cout << "  ------|----------|---------\n";
    for (int b = 0; b < TOKENS_TO_PROCESS; ++b) {
        char buf[64];
        snprintf(buf, sizeof(buf), "   %2d   |   %d      |   %d\n", b,
                routing.expert_ids[b * TOP_K + 0],
                routing.expert_ids[b * TOP_K + 1]);
        std::cout << buf;
    }
    std::cout << "\n";

    std::cout << "Token distribution per expert:\n";
    for (int e = 0; e < NUM_EXPERTS; ++e) {
        std::cout << "  Expert " << e << ": " << routing.token_counts[e]
                  << " tokens (offset in contiguous memory: "
                  << routing.token_offsets[e] << ")\n";
    }
    std::cout << "\n";

    std::cout << "Number of active experts (with assigned tokens): ";
    int active_expert_count = 0;
    for (int e = 0; e < NUM_EXPERTS; ++e) {
        if (routing.token_counts[e] > 0) { active_expert_count++; }
    }
    std::cout << active_expert_count << " out of " << NUM_EXPERTS << "\n\n";

    // Execute grouped GEMM with MoE
    toy_moe(input.data(), output.data(), expert_weights, routing,
            TOKENS_TO_PROCESS, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM);

    // Print output
    print_matrix("Output", output.data(), TOKENS_TO_PROCESS, OUTPUT_DIM);

    // Verify output sanity
    bool valid = true;
    for (int i = 0; i < TOKENS_TO_PROCESS * OUTPUT_DIM; ++i) {
        if (std::isnan(output[i]) || std::isinf(output[i])) {
            valid = false;
            break;
        }
    }

    if (valid) {
        std::cout << "SUCCESS: Grouped GEMM reference implementation "
                     "completed!\n";
        return 0;
    } else {
        std::cout << "FAILED: Invalid output detected!\n";
        return 1;
    }
}
