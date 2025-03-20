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

#ifndef GRAPH_BACKEND_DNNL_KERNELS_GMLP_PRIMITIVE_CONFIG_HPP
#define GRAPH_BACKEND_DNNL_KERNELS_GMLP_PRIMITIVE_CONFIG_HPP

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/gated_mlp_utils.hpp"
#include "common/primitive.hpp"

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/value.hpp"

#include "graph/backend/dnnl/subgraph.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
using op_ptr = std::shared_ptr<op_t>;
using ltw = logical_tensor_wrapper_t;

struct gmlp_primitive_config_t {
public:
    gmlp_primitive_config_t() = default;

    std::shared_ptr<op_t> mm_up_ = nullptr;
    std::shared_ptr<op_t> mm_gate_ = nullptr;
    std::shared_ptr<op_t> mm_down_ = nullptr;

    std::shared_ptr<value_t> src_ = nullptr;
    std::shared_ptr<value_t> w_gate_ = nullptr;
    std::shared_ptr<value_t> w_up_ = nullptr;
    std::shared_ptr<value_t> w_down_ = nullptr;
    std::shared_ptr<value_t> dst_ = nullptr;

    bool quantized_ = false;

    // gated mlp pd and primitive.
    std::shared_ptr<primitive_desc_t> gmlp_pd_;
    std::shared_ptr<primitive_t> gmlp_prim_;

private:
    op_ptr get_post_op(const op_ptr &op) const;

public:
    status_t locate_io(std::shared_ptr<subgraph_t> &sg,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs);

    // The function is used to check if the configuration of gmlp is supported by
    // current implementation of micro kernel. Refer to the following limitation:
    // 1. only support limited pattern
    // 2. only support fp16 data type
    // 3. only support 2-dims tensor
    status_t initial_check(const std::shared_ptr<subgraph_t> &sg,
            const std::vector<logical_tensor_t> &inputs);

    // Initialize parameters and primitive.
    status_t init(std::shared_ptr<subgraph_t> &sg, const dnnl::engine &p_engine,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs);
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
