/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

// https://github.com/NVIDIA/cudnn-frontend/blob/5040925e9450c399a66240b485b38564226e1212/samples/legacy_samples/f16_flash_mha_sample.cpp

// As mapping the whole SDPA graph from cudnn FE 0.x to oneDNN graph is complex,
// this example only demonstrates the mapping of QKBMM operation.

#include "compat_0_x_helpers.hpp"

static bool allowAllConfig(
        compat_0_x::onednnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

// Used for MHA
void generateMHAStrides(int64_t b, int64_t h, int64_t s_q, int64_t s_kv,
        int64_t d, int64_t *strideA, MHA_Layout layout, MHA_Matrix matrix) {

    // TODO: support mapping of QKV_INTERLEAVED, KV_INTERLEAVED and
    // SBH_INTERLEAVED layouts in cudnn graph
    constexpr int batch_dim_idx = 0;
    constexpr int head_dim_idx = 1;
    constexpr int seqlen_dim_idx = 2;
    constexpr int hidden_dim_idx = 3;

    constexpr int seqlen_transpose_dim_idx = 3;
    constexpr int hidden_transpose_dim_idx = 2;

    constexpr int seqlen_q_dim_idx = 2;
    constexpr int seqlen_kv_dim_idx = 3;

    switch (matrix) {
        case MHA_Matrix::Q_Matrix:
            strideA[hidden_dim_idx] = 1;
            strideA[seqlen_q_dim_idx] = d;
            strideA[head_dim_idx] = s_q * d;
            strideA[batch_dim_idx] = s_q * d * h;
            break;
        case MHA_Matrix::K_Matrix:
            strideA[hidden_dim_idx] = 1;
            strideA[seqlen_q_dim_idx] = d;
            strideA[head_dim_idx] = s_kv * d;
            strideA[batch_dim_idx] = s_kv * d * h;
            break;
        case MHA_Matrix::K_Matrix_Transpose:
            strideA[seqlen_transpose_dim_idx] = 1;
            strideA[hidden_transpose_dim_idx] = s_kv;
            strideA[head_dim_idx] = s_kv * d;
            strideA[batch_dim_idx] = s_kv * d * h;
            break;
        case MHA_Matrix::O_Matrix:
            strideA[seqlen_kv_dim_idx] = 1;
            strideA[seqlen_q_dim_idx] = s_kv;
            strideA[head_dim_idx] = s_q * s_kv;
            strideA[batch_dim_idx] = h * s_q * s_kv;
            break;
    }
}

static compat_0_x::Tensor tensor_create(compat_0_x::DataType_t type, int64_t id,
        compat_0_x::lt::dims const dim, compat_0_x::lt::dims const stride,
        bool is_virtual, bool is_value) {
    int nbDims = 4;
    auto tensor_created = compat_0_x::TensorBuilder()
                                  .setDim(nbDims, dim)
                                  .setStride(nbDims, stride)
                                  .setId(id)
                                  .setAlignment(16)
                                  .setDataType(type)
                                  .setVirtual(is_virtual)
                                  .setByValue(is_value)
                                  .build();
    return tensor_created;
}

static compat_0_x::Tensor createQKBMM(int64_t b, int64_t h, int64_t s_q,
        int64_t s_kv, int64_t d, MHA_Layout layout,
        compat_0_x::DataType_t tensorType,
        std::vector<compat_0_x::Operation> &ops) {
    // Creates the necessary tensor descriptors
    int64_t q_dim[4] = {b, h, s_q, d};
    int64_t q_stride[4];
    generateMHAStrides(
            b, h, s_q, s_kv, d, q_stride, layout, MHA_Matrix::Q_Matrix);

    int64_t k_dim[4] = {b, h, d, s_kv};
    int64_t k_stride[4];
    generateMHAStrides(b, h, s_q, s_kv, d, k_stride, layout,
            MHA_Matrix::K_Matrix_Transpose);

    int64_t s_dim[4] = {b, h, s_q, s_kv};
    int64_t s_stride[4];
    generateMHAStrides(
            b, h, s_q, s_kv, d, s_stride, layout, MHA_Matrix::O_Matrix);

    auto qTensor = tensor_create(tensorType, Q_ID,
            std::vector<int64_t>(q_dim, q_dim + 4),
            std::vector<int64_t>(q_stride, q_stride + 4), false, false);
    auto kTransposeTensor = tensor_create(tensorType, K_ID,
            std::vector<int64_t>(k_dim, k_dim + 4),
            std::vector<int64_t>(k_stride, k_stride + 4), false, false);
    auto sTensor = tensor_create(tensorType, O_ID,
            std::vector<int64_t>(s_dim, s_dim + 4),
            std::vector<int64_t>(s_stride, s_stride + 4), true, false);

    // Define the matmul 1 desc
    // Unlike cudnn FE 0.x, oneDNN Graph doesn't require setting the computation
    // datatype explicitly.
    // As the transpose_b attr is set as false by default, we don't need to set
    // it explicitly in this case, we just set it here to demonstrate the API
    // mapping.
    auto matmul_1_Desc
            = compat_0_x::MatMulDescBuilder().setTransposeB(false).build();

    // Create a matmul 1 Node
    auto matmul_op1 = compat_0_x::OperationBuilder(compat_0_x::op::kind::MatMul)
                              .setaMatDesc(qTensor)
                              .setbMatDesc(kTransposeTensor)
                              .setcMatDesc(sTensor)
                              .setmatmulDesc(std::move(matmul_1_Desc))
                              .build();

    ops.push_back(std::move(matmul_op1));

    return sTensor;
}

void run_bf16_flash_attention_fprop(int64_t b, int64_t h, int64_t s_q,
        int64_t s_kv, int64_t d, MHA_Layout layout,
        compat_0_x::DataType_t tensorType, compat_0_x::Handle &handle,
        const std::shared_ptr<void> devPtrQ,
        const std::shared_ptr<void> devPtrK,
        const std::shared_ptr<void> devPtrO) {
    std::vector<compat_0_x::Operation *> all_ops;
    std::vector<compat_0_x::Operation> ops;
    std::unordered_map<uint64_t, std::shared_ptr<void>> data_ptrs;

    // Q * K^T
    auto sTensor = createQKBMM(b, h, s_q, s_kv, d, layout, tensorType, ops);

    std::cout << "Total ops created: " << ops.size() << std::endl;

    for (unsigned int i = 0; i < ops.size(); i++) {
        all_ops.push_back(&ops[i]);
    }

    // Create an Operation Graph
    auto opGraph = compat_0_x::OperationGraphBuilder()
                           .setHandle(handle)
                           .setOperationGraph(all_ops.size(), all_ops.data())
                           .build();

    compat_0_x::EngineConfigList filtered_configs;
    auto statuses = compat_0_x::get_heuristics_list({"heuristics_instant"},
            opGraph, ::allowAllConfig, filtered_configs, true);

    auto plan = compat_0_x::ExecutionPlanBuilder()
                        .setHandle(handle)
                        .setEngineConfig(std::move(filtered_configs[0]))
                        .build();

    // add all the data pointers to be used in the variant pack
    data_ptrs.insert(std::pair<uint64_t, std::shared_ptr<void>>(Q_ID, devPtrQ));
    data_ptrs.insert(std::pair<uint64_t, std::shared_ptr<void>>(K_ID, devPtrK));
    data_ptrs.insert(std::pair<uint64_t, std::shared_ptr<void>>(O_ID, devPtrO));

    // Unlike cuDNN API, oneDNN graph API does not require explicit workspace.
    auto variantPack = compat_0_x::VariantPackBuilder()
                               .setDataPointers(data_ptrs)
                               .build();

    compat_0_x::onednnGraphExecute(handle, plan, variantPack);
    handle.synchronize();
}

int main(int argc, char **argv) {

    if (argc < 2) {
        std::cerr << "Engine kind not specified. Please specify 'cpu' or 'gpu'."
                  << std::endl;
        std::cerr << "Usage: " << argv[0] << " [cpu|gpu]" << std::endl;
        return -1;
    }

    std::string engine_arg = argv[1];
    dnnl::engine::kind engine_kind;

    if (engine_arg == "cpu") {
        engine_kind = dnnl::engine::kind::cpu;
    } else if (engine_arg == "gpu") {
        engine_kind = dnnl::engine::kind::gpu;
    } else {
        std::cerr << "Invalid engine kind specified: " << engine_arg
                  << std::endl;
        std::cerr << "Usage: " << argv[0] << " [cpu|gpu]" << std::endl;
        return -1;
    }

    // oneDNN handle. Unlike cuDNN, oneDNN needs to know the engine kind.
    auto handle = compat_0_x::Handle(engine_kind, 0);

    int64_t b = 2; // batch size
    int64_t h = 12; // head dim
    int64_t s_q = 2048; // q tensor is padded to this seq length
    int64_t s_kv = 2048; // k and v tensor is padded to this seq length
    int64_t d = 128; // hidden dim

    MHA_Layout layout = MHA_Layout::NOT_INTERLEAVED;

    std::cout << "====PARAMETERS====" << std::endl;
    std::cout << "batch is " << b << ", head dim is " << h
              << ", q sequence length is " << s_q << ", kv sequence length is "
              << s_kv << ", hidden dim is " << d << std::endl;

    const auto dt = dnnl::graph::logical_tensor::data_type::bf16;
    std::shared_ptr<void> devPtrQ;
    std::shared_ptr<void> devPtrK;
    std::shared_ptr<void> devPtrO;
    run_bf16_flash_attention_fprop(
            b, h, s_q, s_kv, d, layout, dt, handle, devPtrQ, devPtrK, devPtrO);

    std::cout << "\n======================================================="
                 "=================================\n";
}
