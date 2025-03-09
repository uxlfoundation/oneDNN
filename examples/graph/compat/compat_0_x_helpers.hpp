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

#ifndef GRAPH_EXAMPLE_COMPAT_COMPAT_HELPERS_HPP
#define GRAPH_EXAMPLE_COMPAT_COMPAT_HELPERS_HPP

#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include "../graph_example_utils.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#include <unordered_map>

#define Q_ID 0
#define K_ID 1
#define O_ID 2

enum class MHA_Layout {
    NOT_INTERLEAVED = 0,
    QKV_INTERLEAVED = 1,
    KV_INTERLEAVED = 2,
    SBH_INTERLEAVED = 3
};

enum class MHA_Matrix {
    Q_Matrix = 0, // queries
    K_Matrix = 1, // keys
    K_Matrix_Transpose = 2, // keys tranposed
    O_Matrix = 3, // final output
};

namespace compat_0_x {

int64_t op_id = 0;
using lt = dnnl::graph::logical_tensor;
using tensor = dnnl::graph::tensor;
using op = dnnl::graph::op;
using partition = dnnl::graph::partition;
using compiled_partition = dnnl::graph::compiled_partition;
using graph = dnnl::graph::graph;
using DataType_t = lt::data_type;

// mimic cudnnHandle_t
class Handle {
    std::shared_ptr<dnnl::engine> eng_;
    std::shared_ptr<dnnl::stream> str_;

public:
    Handle() = default;
    ~Handle() = default;

    Handle(dnnl::engine::kind ekind, int index) {
        eng_ = std::make_shared<dnnl::engine>(ekind, index);
        if (ekind == dnnl::engine::kind::cpu) {
            str_ = std::make_shared<dnnl::stream>(*eng_);
        } else {
            sycl::queue q = sycl::queue(
                    sycl::gpu_selector_v, sycl::property::queue::in_order {});
            auto gpu_stream = dnnl::sycl_interop::make_stream(*eng_, q);
            str_ = std::make_shared<dnnl::stream>(std::move(gpu_stream));
        }
    }

    dnnl::engine::kind get_engine_kind() const { return eng_->get_kind(); }

    dnnl::engine *get_engine() const { return eng_.get(); }
    dnnl::stream *get_stream() const { return str_.get(); }

    void synchronize() const { str_->wait(); }
};

struct cpu_deletor_t {
    cpu_deletor_t() = default;
    void operator()(void *ptr) {
        if (ptr) free(ptr);
    }
};

#ifdef DNNL_WITH_SYCL
struct sycl_deletor_t {
    sycl_deletor_t() = delete;
    ::sycl::context ctx_;
    sycl_deletor_t(const ::sycl::context &ctx) : ctx_(ctx) {}
    void operator()(void *ptr) {
        if (ptr) ::sycl::free(ptr, ctx_);
    }
};
#endif

typedef void *onednnBackendDescriptor_t;

// The following encapsulation are used to mimic the main classes and their
// corresponding builder classes in cudnn FE 0.x API, for details, pls refer to
// https://github.com/NVIDIA/cudnn-frontend/tree/5040925e9450c399a66240b485b38564226e1212/include
class Tensor_v8 {
public:
    friend class TensorBuilder_v8;

    int64_t getDimCount() const { return nDims; }

    lt::dims const getDim() const { return btensor_dimA; }

    lt::dims const getStride() const { return btensor_strA; }

    lt::data_type getDataType() const {
        return static_cast<DataType_t>(data_type);
    }

    int64_t getId() const { return id; }

    int64_t getAlignment() const { return alignment; }

    bool isVirtualTensor() const { return isVirtual; }

    lt const getInternal_lt() const { return internal_lt; }

    Tensor_v8(Tensor_v8 &&from) = default;
    Tensor_v8 &operator=(Tensor_v8 &&) = default;

    ~Tensor_v8() = default;

private:
    Tensor_v8() = default;
    Tensor_v8(Tensor_v8 const &) = delete;
    Tensor_v8 &operator=(Tensor_v8 const &) = delete;

    DataType_t data_type = DataType_t::undef; //! Datatype of the elements
    lt::dims btensor_dimA;
    lt::dims btensor_strA;
    int64_t id = -1; //! Unique id of the tensor
    int64_t alignment = -1; //! Alignment of the tensor.
    //! Certain engine config expect minimum alignment of 16B
    int64_t nDims = -1; //! Number of Dimensions of the tensor
    bool isVirtual
            = false; //! Whether it is an intermediate tensor of an op graph
    bool isByValue
            = false; //! Whether the tensor is in host memory that needs to be passed to the kernel by value
    lt internal_lt;
};

class TensorBuilder_v8 {
public:
    using lt = dnnl::graph::logical_tensor;
    auto setDataType(lt::data_type data_type) -> TensorBuilder_v8 & {
        m_tensor.data_type = data_type;
        return *this;
    }

    auto setDim(int64_t ndim, const lt::dims &dims) -> TensorBuilder_v8 & {
        m_tensor.nDims = dims.size();
        m_tensor.btensor_dimA = dims;
        return *this;
    }

    auto setStride(int64_t ndim, const lt::dims &strides)
            -> TensorBuilder_v8 & {
        m_tensor.btensor_strA = strides;
        return *this;
    }

    auto setId(int64_t id_) -> TensorBuilder_v8 & {
        m_tensor.id = id_;
        return *this;
    }

    auto setAlignment(int64_t alignment_) -> TensorBuilder_v8 & {
        m_tensor.alignment = alignment_;
        return *this;
    }

    auto setVirtual(bool virtual_ = true) -> TensorBuilder_v8 & {
        m_tensor.isVirtual = virtual_;
        return *this;
    }

    auto setByValue(bool isByValue_ = true) -> TensorBuilder_v8 & {
        m_tensor.isByValue = isByValue_;
        return *this;
    }

    Tensor_v8 &&build() {
        auto lt_tmp = dnnl::graph::logical_tensor(m_tensor.getId(),
                m_tensor.getDataType(), m_tensor.getDim(),
                m_tensor.getStride());
        m_tensor.internal_lt = lt_tmp;
        return std::move(m_tensor);
    }

    explicit TensorBuilder_v8() = default;
    ~TensorBuilder_v8() = default;
    TensorBuilder_v8(TensorBuilder_v8 &&) = delete;
    TensorBuilder_v8(TensorBuilder_v8 const &) = delete;
    TensorBuilder_v8 &operator=(TensorBuilder_v8 const &) = delete;

private:
    Tensor_v8 m_tensor;
};

using Tensor = Tensor_v8;
using TensorBuilder = TensorBuilder_v8;

// For other operations in oneDNN Graph, pls define the corresponding Descriptor
// and DescriptorBuilder classes to hold attrs of the operation.
class MatMulDesc_v8 {
public:
    friend class MatMulDescBuilder_v8;
    friend class OperationBuilder_v8;

    MatMulDesc_v8() = default;
    MatMulDesc_v8(MatMulDesc_v8 &&from) = default;
    MatMulDesc_v8 &operator=(MatMulDesc_v8 &&from) = default;
    const bool &getTransposeB() const { return transpose_b; }

    ~MatMulDesc_v8() = default;

private:
    MatMulDesc_v8(MatMulDesc_v8 const &) = delete;
    MatMulDesc_v8 &operator=(MatMulDesc_v8 const &) = delete;
    // Here we define transpose_b attr only, but actually, oneDNN Graph also
    // supports setting transpose_a attr for the fisrt input tensor of MatMul
    // op, pls add it and corresponding setter and getter functions if needed.
    bool transpose_b = false;
};

class MatMulDescBuilder_v8 {
public:
    auto setTransposeB(bool transpose_b) -> MatMulDescBuilder_v8 & {
        m_matMulDesc.transpose_b = transpose_b;
        return *this;
    }

    MatMulDesc_v8 &&build() { return std::move(m_matMulDesc); }

    explicit MatMulDescBuilder_v8() = default;
    ~MatMulDescBuilder_v8() = default;
    MatMulDescBuilder_v8(MatMulDescBuilder_v8 &&) = delete;
    MatMulDescBuilder_v8(MatMulDescBuilder_v8 const &) = delete;
    MatMulDescBuilder_v8 &operator=(MatMulDescBuilder_v8 const &) = delete;

private:
    MatMulDesc_v8 m_matMulDesc;
};
using MatMulDesc = MatMulDesc_v8;
using MatMulDescBuilder = MatMulDescBuilder_v8;

class Operation_v8 {
public:
    friend class OperationBuilder_v8;
    friend class OperationGraphBuilder_v8;

    Operation_v8(Operation_v8 &&from) = default;
    Operation_v8 &operator=(Operation_v8 &&from) = default;

    ~Operation_v8() = default;

private:
    Operation_v8() = default;

    Operation_v8(Operation_v8 const &) = delete;
    Operation_v8 &operator=(Operation_v8 const &) = delete;

    op::kind op_kind = op::kind::Wildcard;
    // For other operations, pls add its corresponding Descriptor and input
    // parameters here.
    MatMulDesc_v8 m_matMulDesc;
    lt amatdesc;
    lt bmatdesc;
    lt cmatdesc;
    std::shared_ptr<op> internal_op;
};

class OperationBuilder_v8 {
private:
    Operation_v8 m_operation;
    op::kind m_op_kind;

    // For other operations, pls add its corresponding build functions here.
    Operation_v8 &&build_matmul_op() {
        std::string unique_op_name = "matmul_op_" + std::to_string(op_id);
        auto matmul_op = op(op_id++, op::kind::MatMul, unique_op_name);
        matmul_op.set_attr<bool>(op::attr::transpose_b,
                m_operation.m_matMulDesc.getTransposeB());
        matmul_op.add_inputs({m_operation.amatdesc, m_operation.bmatdesc});
        matmul_op.add_outputs({m_operation.cmatdesc});
        m_operation.internal_op = std::make_shared<op>(std::move(matmul_op));
        return std::move(m_operation);
    }

public:
    // For other operations, pls add its corresponding parameter setter
    // functions here.
    auto setaMatDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.amatdesc = tensor.getInternal_lt();
        return *this;
    }
    auto setbMatDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.bmatdesc = tensor.getInternal_lt();
        return *this;
    }
    auto setcMatDesc(Tensor_v8 const &tensor) -> OperationBuilder_v8 & {
        m_operation.cmatdesc = tensor.getInternal_lt();
        return *this;
    }

    // For other operations, pls add its corresponding Descriptor setter
    // functions here.
    auto setmatmulDesc(MatMulDesc_v8 &&matmulDesc) -> OperationBuilder_v8 & {
        m_operation.m_matMulDesc = std::move(matmulDesc);
        return *this;
    }

    OperationBuilder_v8(op::kind const &op_kind)
        : m_operation(), m_op_kind(op_kind) {}

    Operation_v8 &&build() {
        m_operation.op_kind = m_op_kind;
        if (m_operation.op_kind == op::kind::MatMul) {
            return build_matmul_op();
        }
        // For other operations, pls add its corresponding build functions here.
        return std::move(m_operation);
    }
};

using Operation = Operation_v8;
using OperationBuilder = OperationBuilder_v8;

class OperationGraph_v8 {
public:
    friend class OperationGraphBuilder_v8;

    const graph *get_internal_graph() const { return internal_graph.get(); }

    const std::shared_ptr<partition> &get_internal_partition() const {
        return internal_partition;
    }

    OperationGraph_v8(OperationGraph_v8 &&from) = default;
    OperationGraph_v8 &operator=(OperationGraph_v8 &&from) = default;

    ~OperationGraph_v8() = default;

    uint64_t getOpCount() const { return numOps; }

    const std::vector<Operation_v8> &getOps() const { return ops; }

private:
    OperationGraph_v8() = default;
    OperationGraph_v8(OperationGraph_v8 const &) = delete;
    OperationGraph_v8 &operator=(OperationGraph_v8 const &) = delete;
    Handle *handle;
    std::vector<Operation_v8> ops;
    int64_t numOps = -1;
    std::shared_ptr<graph> internal_graph;
    std::shared_ptr<partition> internal_partition;
};

class OperationGraphBuilder_v8 {
public:
    auto setHandle(Handle &handle_) -> OperationGraphBuilder_v8 & {
        m_operationGraph.handle = &handle_;
        return *this;
    }

    auto setOperationGraph(int64_t numOps_, Operation_v8 **ops_)
            -> OperationGraphBuilder_v8 & {
        m_operationGraph.numOps = numOps_;
        m_operationGraph.ops.clear();
        for (int64_t i = 0; i < numOps_; i++) {
            m_operationGraph.ops.push_back(std::move(*(ops_[i])));
        }
        return *this;
    }

    OperationGraph_v8 &&build() {

        dnnl::graph::graph op_graph(m_operationGraph.handle->get_engine_kind());

        for (auto i = 0u; i < m_operationGraph.numOps; i++) {
            auto &op = m_operationGraph.ops[i];
            op_graph.add_op(*(op.internal_op));
        }

        op_graph.finalize();
        m_operationGraph.internal_graph = std::make_shared<graph>(op_graph);

        auto parts = op_graph.get_partitions();
        if (parts.size() != 1)
            throw std::runtime_error("operation graph cannot be fused");
        m_operationGraph.internal_partition
                = std::make_shared<partition>(parts[0]);

        return std::move(m_operationGraph);
    }

    explicit OperationGraphBuilder_v8() = default;
    ~OperationGraphBuilder_v8() = default;
    OperationGraphBuilder_v8(OperationGraphBuilder_v8 &&) = delete;
    OperationGraphBuilder_v8(OperationGraphBuilder_v8 const &) = delete;
    OperationGraphBuilder_v8 &operator=(OperationGraphBuilder_v8 const &)
            = delete;

private:
    OperationGraph_v8 m_operationGraph;
};

using OperationGraph = OperationGraph_v8;
using OperationGraphBuilder = OperationGraphBuilder_v8;

class EngineConfig_v8 {
public:
    friend class EngineConfigBuilder_v8;
    friend class ExecutionPlan_v8;
    friend class ExecutionPlanBuilder_v8;

    void set_internal_partition(const std::shared_ptr<partition> &p) {
        internal_partition = p;
    }

    const partition *get_internal_partition() const {
        return internal_partition.get();
    }

    EngineConfig_v8 &operator=(EngineConfig_v8 &&from) = default;

    EngineConfig_v8(EngineConfig_v8 &&from) = default;

    EngineConfig_v8() = default;

    ~EngineConfig_v8() = default;

private:
    EngineConfig_v8(EngineConfig_v8 const &) = delete;
    EngineConfig_v8 &operator=(EngineConfig_v8 const &) = delete;
    std::shared_ptr<partition> internal_partition;
};

using EngineConfigList = std::vector<EngineConfig_v8>;

static inline std::vector<dnnl_status_t> get_heuristics_list(
        std::vector<std::string> const &modes, OperationGraph_v8 &opGraph,
        std::function<bool(onednnBackendDescriptor_t)> filter_fn,
        EngineConfigList &filtered_configs, bool evaluate_all = false) {
    std::vector<dnnl_status_t> statuses;

    auto part = opGraph.get_internal_partition();
    if (filtered_configs.empty()) filtered_configs.emplace_back();

    filtered_configs[0].set_internal_partition(part);
    return statuses;
}

using EngineConfig = EngineConfig_v8;

class ExecutionPlan_v8 {
public:
    friend class ExecutionPlanBuilder_v8;

    ExecutionPlan_v8(ExecutionPlan_v8 &&from) = default;
    ExecutionPlan_v8 &operator=(ExecutionPlan_v8 &&) = default;

    ~ExecutionPlan_v8() = default;

    EngineConfig_v8 const &get_EngineConfig() const { return m_engine_config; }

    const compiled_partition *get_compiled_partition() const {
        return internal_compiled_partition.get();
    }

    ExecutionPlan_v8(ExecutionPlan_v8 const &) = default;
    ExecutionPlan_v8 &operator=(ExecutionPlan_v8 const &) = default;

private:
    ExecutionPlan_v8() = default;
    EngineConfig_v8 m_engine_config;
    Handle *handle;
    std::shared_ptr<compiled_partition> internal_compiled_partition;
};

class ExecutionPlanBuilder_v8 {
public:
    auto setHandle(Handle &handle_) -> ExecutionPlanBuilder_v8 & {
        m_execution_plan.handle = &handle_;
        return *this;
    }

    auto setEngineConfig(EngineConfig_v8 &&engine_config_)
            -> ExecutionPlanBuilder_v8 & {
        m_execution_plan.m_engine_config = std::move(engine_config_);
        return *this;
    }

    ExecutionPlan_v8 &&build() {

        auto partition
                = m_execution_plan.m_engine_config.get_internal_partition();

        std::vector<lt> inputs = partition->get_input_ports();
        std::vector<lt> outputs = partition->get_output_ports();

        m_execution_plan.internal_compiled_partition
                = std::make_shared<compiled_partition>(
                        partition->compile(inputs, outputs,
                                *(m_execution_plan.handle->get_engine())));
        return std::move(m_execution_plan);
    }

    explicit ExecutionPlanBuilder_v8() = default;
    ~ExecutionPlanBuilder_v8() = default;
    ExecutionPlanBuilder_v8(ExecutionPlanBuilder_v8 &&) = delete;
    ExecutionPlanBuilder_v8(ExecutionPlanBuilder_v8 const &) = delete;
    ExecutionPlanBuilder_v8 &operator=(ExecutionPlanBuilder_v8 const &)
            = delete;

private:
    ExecutionPlan_v8 m_execution_plan;
};

using ExecutionPlan = ExecutionPlan_v8;
using ExecutionPlanBuilder = ExecutionPlanBuilder_v8;

class VariantPack_v8 {
public:
    friend class VariantPackBuilder_v8;

    VariantPack_v8(VariantPack_v8 &&from) = default;
    VariantPack_v8 &operator=(VariantPack_v8 &&from) = default;

    ~VariantPack_v8() = default;

    std::unordered_map<uint64_t, std::shared_ptr<void>> const &
    getDataPointers() const {
        return data_pointers;
    }

private:
    VariantPack_v8() = default;
    VariantPack_v8(VariantPack_v8 const &) = delete;
    VariantPack_v8 &operator=(VariantPack_v8 const &) = delete;

    std::unordered_map<uint64_t, std::shared_ptr<void>> data_pointers;
};
class VariantPackBuilder_v8 {
public:
    auto setDataPointers(
            std::unordered_map<uint64_t, std::shared_ptr<void>> const
                    &data_pointers) -> VariantPackBuilder_v8 & {
        m_variant_pack.data_pointers = std::move(data_pointers);
        return *this;
    }

    VariantPack_v8 &&build() { return std::move(m_variant_pack); }

    explicit VariantPackBuilder_v8() = default;
    ~VariantPackBuilder_v8() = default;
    VariantPackBuilder_v8(VariantPackBuilder_v8 &&) = delete;
    VariantPackBuilder_v8(VariantPackBuilder_v8 const &) = delete;
    VariantPackBuilder_v8 &operator=(VariantPackBuilder_v8 const &) = delete;

private:
    VariantPack_v8 m_variant_pack;
};

using VariantPack = VariantPack_v8;
using VariantPackBuilder = VariantPackBuilder_v8;

static inline void onednnGraphExecute(const Handle &handle,
        ExecutionPlan_v8 const &executionPlan,
        VariantPack_v8 const &variantPack) {

    auto eng = handle.get_engine();
    auto strm = handle.get_stream();

    std::vector<lt> inputs = executionPlan.get_EngineConfig()
                                     .get_internal_partition()
                                     ->get_input_ports();
    std::vector<lt> outputs = executionPlan.get_EngineConfig()
                                      .get_internal_partition()
                                      ->get_output_ports();

    // Allocate memory for input and output
    std::vector<tensor> input_tensors;
    std::vector<tensor> output_tensors;
    std::vector<std::shared_ptr<void>> input_handles;
    std::vector<std::shared_ptr<void>> output_handles;

    input_tensors.reserve(inputs.size());
    input_handles.reserve(inputs.size());
    output_tensors.reserve(outputs.size());
    output_handles.reserve(outputs.size());

    if (eng->get_kind() == dnnl::engine::kind::cpu) {
        for (auto &in_lt : inputs) {
            auto mem_size = in_lt.get_mem_size();
            std::shared_ptr<void> mem_handle
                    = variantPack.getDataPointers().at(in_lt.get_id());
            mem_handle.reset(malloc(mem_size), cpu_deletor_t {});
            input_handles.push_back(mem_handle);
            input_tensors.emplace_back(in_lt, *eng, mem_handle.get());
        }
        for (auto &out_lt : outputs) {
            auto mem_size = out_lt.get_mem_size();
            std::shared_ptr<void> mem_handle
                    = variantPack.getDataPointers().at(out_lt.get_id());
            mem_handle.reset(malloc(mem_size), cpu_deletor_t {});
            output_handles.push_back(mem_handle);
            output_tensors.emplace_back(out_lt, *eng, mem_handle.get());
        }
    } else {
        sycl::queue const &q = dnnl::sycl_interop::get_queue(*strm);
        for (auto &in_lt : inputs) {
            auto mem_size = in_lt.get_mem_size();
            std::shared_ptr<void> mem_handle
                    = variantPack.getDataPointers().at(in_lt.get_id());
            mem_handle.reset(::sycl::malloc_shared(
                                     mem_size, q.get_device(), q.get_context()),
                    sycl_deletor_t {q.get_context()});
            input_handles.push_back(mem_handle);
            input_tensors.emplace_back(in_lt, *eng, mem_handle.get());
        }
        for (auto &out_lt : outputs) {
            auto mem_size = out_lt.get_mem_size();
            std::shared_ptr<void> mem_handle
                    = variantPack.getDataPointers().at(out_lt.get_id());
            mem_handle.reset(::sycl::malloc_shared(
                                     mem_size, q.get_device(), q.get_context()),
                    sycl_deletor_t {q.get_context()});
            output_handles.push_back(mem_handle);
            output_tensors.emplace_back(out_lt, *eng, mem_handle.get());
        }
    }

    executionPlan.get_compiled_partition()->execute(
            *(handle.get_stream()), input_tensors, output_tensors);
}

} // namespace compat_0_x

#endif