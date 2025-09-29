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
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

/// @example matmul_with_host_scalar_scale.cpp
/// > Annotated version: @ref matmul_with_host_scalar_scale_cpp

/// @page matmul_with_host_scalar_scale_cpp MatMul with Host Scalar Scale example
///
/// This C++ API example demonstrates matrix multiplication (C = alpha * A * B)
/// with a scalar scale factor using oneDNN.
/// The workflow includes following steps:
/// - Initialize a oneDNN engine and stream for computation.
/// - Allocate and initialize matrices A and B.
/// - Create oneDNN memory objects for matrices A, B, and C.
/// - Prepare a scalar (alpha) as a host-side float value and wrap it in a
///   oneDNN memory object.
/// - Create a matmul primitive descriptor with the scalar scale attribute.
/// - Create a matmul primitive.
/// - Execute the matmul primitive.
/// - Validate the result.
///
/// @include matmul_with_host_scalar_scale.cpp

// Compare straightforward matrix multiplication (C = alpha * A * B)
// with the result from oneDNN memory.
bool check_result(const std::vector<float> &a_data,
        const std::vector<float> &b_data, int M, int N, int K, float alpha,
        dnnl::memory &c_mem) {
    std::vector<float> c_ref(M * N, 0.0f);
    // a: M x K, w: K x N, c: M x N
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            c_ref[i * N + j] = 0.0f;
            for (int k = 0; k < K; ++k) {
                c_ref[i * N + j] += a_data[i * K + k] * b_data[k * N + j];
            }
            c_ref[i * N + j] *= alpha;
        }
    }

    std::vector<float> c_result(M * N, 0.0f);
    read_from_dnnl_memory(c_result.data(), c_mem);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (std::abs(c_result[i * N + j] - c_ref[i * N + j]) > 1e-5) {
                return false;
            }
        }
    }
    return true;
}

// Simple matrix multiplication with alpha as scalar memory
void simple_matmul_with_host_scalar(engine::kind engine_kind) {
    // Initialize a oneDNN engine and stream for computation
    engine eng(engine_kind, 0);
    stream s(eng);

    // Define the dimensions for matrices A (MxK), B (KxN), and C (MxN)
    const int M = 3, N = 3, K = 3;

    // Allocate and initialize matrix A with float values
    // and create a oneDNN memory object for it
    std::vector<float> a_data(M * K, 0.0f);
    for (int i = 0; i < M * K; ++i) {
        a_data[i] = static_cast<float>(i + 1);
    }
    memory::dims a_dims = {M, K};
    memory a_mem({a_dims, memory::data_type::f32, memory::format_tag::ab}, eng);
    write_to_dnnl_memory(a_data.data(), a_mem);

    // Allocate and initialize matrix B with values based on the sum of their indices
    // and create a oneDNN memory object for it
    std::vector<float> b_data(K * N, 0.0f);
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            b_data[i * N + j] = static_cast<float>(i + j);
        }
    }
    memory::dims b_dims = {K, N};
    memory b_mem({b_dims, memory::data_type::f32, memory::format_tag::ab}, eng);
    write_to_dnnl_memory(b_data.data(), b_mem);

    // Create oneDNN memory object for the output matrix C
    memory::dims c_dims = {M, N};
    memory c_mem({c_dims, memory::data_type::f32, memory::format_tag::ab}, eng);

    // Prepare a scalar (alpha) as a host-side float value and wrap it in a oneDNN memory object
    float alpha = 2.0f;
    memory alpha_m(memory::desc::host_scalar(memory::data_type::f32), alpha);

    // Create a matmul primitive descriptor with scaling for source memory (A)
    // Set scaling mask to 0 and use host scalar for alpha
    primitive_attr attr;
    attr.set_host_scale(DNNL_ARG_SRC, memory::data_type::f32);
    matmul::primitive_desc matmul_pd(
            eng, a_mem.get_desc(), b_mem.get_desc(), c_mem.get_desc(), attr);

    // Create a matmul primitive
    matmul matmul_prim(matmul_pd);

    // Prepare the arguments map for the matmul execution
    std::unordered_map<int, memory> args = {{DNNL_ARG_SRC, a_mem},
            {DNNL_ARG_WEIGHTS, b_mem}, {DNNL_ARG_DST, c_mem},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, alpha_m}};

    // Execute matmul
    matmul_prim.execute(s, args);
    s.wait();

    // Verify results
    if (!check_result(a_data, b_data, M, N, N, alpha, c_mem)) {
        throw std::runtime_error("Result verification failed!");
    }
}

int main(int argc, char **argv) {
    return handle_example_errors(
            simple_matmul_with_host_scalar, parse_engine_kind(argc, argv));
}
