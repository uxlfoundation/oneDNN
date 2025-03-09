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
    V_Matrix = 3, // values
    V_Matrix_Transpose = 4, // values transposed
    S_Matrix = 5, // output of GEMM1
    O_Matrix = 6, // final output
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
        str_ = std::make_shared<dnnl::stream>(*eng_);
    }

    dnnl::engine::kind get_engine_kind() const { return eng_->get_kind(); }

    dnnl::engine *get_engine() const { return eng_.get(); }
    dnnl::stream *get_stream() const { return str_.get(); }

    void synchronize() const { str_->wait(); }
};

inline void fill_random(std::vector<float> &out) {
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

// Surface is not part fo cuDNN API, but a utility for the example. Here we use
// it to allocate and initialize user data. We rely on the memory allocation in
// `dnnl::graph::tensor` inside the library, to avoid the explicit code for
// cpu/ocl/sycl memory allocation.
struct Surface {
    using lt = dnnl::graph::logical_tensor;

    int64_t n_elems_ = 0;
    Handle *handle_ = NULL;
    lt::data_type dt_ = lt::data_type::undef;
    dnnl::graph::tensor ts_;

protected:
    explicit Surface() {}

public:
    Surface(lt::data_type dt, int64_t n_elems, Handle *handle)
        : n_elems_(n_elems), handle_(handle), dt_(dt) {
        if (handle_ == nullptr) {
            throw std::invalid_argument("Handle pointer cannot be null.");
        }
        std::vector<float> raw_data(n_elems);
        fill_random(raw_data);
        const lt::dims d = {n_elems_};
        auto desc = dnnl::graph::logical_tensor(
                0, dt_, d, lt::layout_type::strided);
        ts_ = dnnl::graph::tensor(desc, *(handle_->get_engine()));
        // TODO(xxx): seems to have memory issue on gpu.
        write_to_dnnl_tensor(raw_data.data(), ts_);
    }

    Surface(lt::data_type dt, int64_t n_elems, Handle *handle, float val)
        : n_elems_(n_elems), handle_(handle), dt_(dt) {
        if (handle_ == nullptr) {
            throw std::invalid_argument("Handle pointer cannot be null.");
        }
        std::vector<float> raw_data(n_elems, val);
        const lt::dims d = {n_elems_};
        auto desc = dnnl::graph::logical_tensor(
                0, dt_, d, lt::layout_type::strided);
        ts_ = dnnl::graph::tensor(desc, *(handle_->get_engine()));
        // TODO(xxx): seems to have memory issue on gpu.
        write_to_dnnl_tensor(raw_data.data(), ts_);
    }

    void *get_ptr() const { return ts_.get_data_handle(); }

    ~Surface() = default;
};

typedef void *onednnBackendDescriptor_t;

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
    lt::dims btensor_dimA; //! n, g, c, d, h, w
    lt::dims btensor_strA; //! n, g, c, d, h, w
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

    Operation_v8 &&build_matmul_op() {

        auto bmm1 = op(op_id++, op::kind::MatMul, "bmm1");
        bmm1.set_attr<bool>(op::attr::transpose_b,
                m_operation.m_matMulDesc.getTransposeB());
        bmm1.add_inputs({m_operation.amatdesc, m_operation.bmatdesc});
        bmm1.add_outputs({m_operation.cmatdesc});
        m_operation.internal_op = std::make_shared<op>(std::move(bmm1));
        return std::move(m_operation);
    }

public:
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
    //std::shared_ptr<Handle> handle;
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
        for (size_t i = 0; i < numOps_; i++) {
            m_operationGraph.ops.push_back(std::move(*(ops_[i])));
        }
        return *this;
    }

    OperationGraph_v8 &&build() {

        dnnl::graph::graph BMM1(m_operationGraph.handle->get_engine_kind());

        for (auto i = 0u; i < m_operationGraph.numOps; i++) {
            auto &op = m_operationGraph.ops[i];
            BMM1.add_op(*(op.internal_op));
        }

        BMM1.finalize();
        m_operationGraph.internal_graph = std::make_shared<graph>(BMM1);

        auto parts = BMM1.get_partitions();
        if (parts.size() != 1) throw std::runtime_error("partition failed ...");
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

    EngineConfig const &get_EngineConfig() const { return m_engine_config; }

    const compiled_partition *get_compiled_partition() const {
        return internal_compiled_partition.get();
    }

    ExecutionPlan_v8(ExecutionPlan_v8 const &) = default;
    ExecutionPlan_v8 &operator=(ExecutionPlan_v8 const &) = default;

private:
    ExecutionPlan_v8() = default;
    EngineConfig m_engine_config;
    //std::shared_ptr<Handle> handle;
    Handle *handle;
    std::shared_ptr<compiled_partition> internal_compiled_partition;
};

///
/// ExecutionPlanBuilder_v8 Class
/// Helper class used to build ExecutionPlan_v8 class
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

    //! constructs the Engine Config by calling the cudnn API
    //! Throws the appropriate error message
    ExecutionPlan_v8 &&build() {

        auto partition
                = m_execution_plan.m_engine_config.get_internal_partition();

        std::vector<lt> inputs
                = partition
                          ->get_input_ports(); // Get the input ports of the partition
        std::vector<lt> outputs
                = partition
                          ->get_output_ports(); // Get the output ports of the partition

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

static inline void onednnGraphExecute(const Handle &handle,
        ExecutionPlan const &executionPlan,
        std::unordered_map<uint64_t, void *> const &data_ptrs) {

    auto eng = handle.get_engine();

    std::vector<lt> inputs = executionPlan.get_EngineConfig()
                                     .get_internal_partition()
                                     ->get_input_ports();
    std::vector<lt> outputs = executionPlan.get_EngineConfig()
                                      .get_internal_partition()
                                      ->get_output_ports();

    auto inputs_num = inputs.size();
    auto query = inputs[Q_ID];
    auto key = inputs[K_ID];
    auto out = outputs[O_ID - inputs_num];

    auto ts_q = tensor(query, *eng, data_ptrs.at(query.get_id()));
    auto ts_k = tensor(key, *eng, data_ptrs.at(key.get_id()));
    auto ts_o = tensor(out, *eng, data_ptrs.at(out.get_id()));

    executionPlan.get_compiled_partition()->execute(
            *(handle.get_stream()), {ts_q, ts_k}, {ts_o});
}

} // namespace compat_0_x

#endif