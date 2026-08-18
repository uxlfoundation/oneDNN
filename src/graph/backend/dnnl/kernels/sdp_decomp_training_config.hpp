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

#ifndef GRAPH_BACKEND_DNNL_KERNELS_SDP_DECOMP_TRAINING_CONFIG_HPP
#define GRAPH_BACKEND_DNNL_KERNELS_SDP_DECOMP_TRAINING_CONFIG_HPP

#include <memory>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"

#include "graph/interface/c_types_map.hpp"

#include "graph/backend/dnnl/kernels/sdp_decomp_config.hpp"
#include "graph/backend/dnnl/scratchpad.hpp"
#include "graph/backend/dnnl/subgraph.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

struct sdp_decomp_training_config_t {
public:
    sdp_decomp_training_config_t() = default;

    // SDP input dimension
    dim_t ndims, batch_size, num_head_q, seq_len_q, seq_len_kv;
    dim_t head_size_qk, head_size_v;

    // SDP input and output strides
    dims src1_strides, wei1_strides, wei2_strides, dst_strides,
            stats_dst_strides;

    // Thread nums during the workflow
    int nthr = 0;

    // Used to record the exact input offset in subgraph
    // [mm1_src, mm1_wei, mm2_wei, mm1_scale]
    std::vector<int> graph_inport;
    enum input_index_t {
        mm1_src = 0,
        mm1_wei,
        mm2_wei,
        mm1_scale,
    };

    // Primitives for the execution pipeline
    sdp_reorder_t sub_reorder0, sub_reorder1, sub_reorder2, sub_reorder3;
    primitive sub_mm1_prim, sub_mm2_prim;

    // Softmax primitive: scores -> P (f32)
    primitive sub_softmax_prim;
    // f32 softmax_out -> xf16 for mm2
    sdp_reorder_t sub_reorder_softmax;

    // Stats computation primitives (logsumexp = max(src) - log(max(P)))
    primitive sub_reduce_max_P_prim;
    primitive sub_reduce_max_src_prim;
    // dense stats -> user layout
    sdp_reorder_t sub_reorder_stats;

    // Args used in execution of primitives
    std::unordered_map<int, memory> sub_reorder0_args, sub_reorder1_args,
            sub_mm1_args, sub_reorder2_args, sub_mm2_args, sub_reorder3_args;
    std::unordered_map<int, memory> sub_softmax_args;
    std::unordered_map<int, memory> sub_reorder_softmax_args;
    std::unordered_map<int, memory> sub_reduce_max_P_args,
            sub_reduce_max_src_args;
    std::unordered_map<int, memory> sub_reorder_stats_args;

    // A map from memory to registry key
    std::unordered_map<dnnl_memory_t, registry_key> mem_key_map;

    // Internal memory objects
    // reorder0
    memory sub_src1;
    // reorder1
    memory sub_wei1_user;
    // mm1
    memory sub_mm1_src, sub_mm1_wei, sub_mm1_dst;
    // mm1 post-ops memories (scale, mask)
    std::vector<memory> sub_mm1_post_mem;
    // softmax output
    memory sub_softmax_out;
    // stats
    memory sub_log_max_P;
    memory sub_stats;
    memory sub_stats_user;
    // reorder2
    memory sub_wei2_user;
    // mm2
    memory sub_mm2_src;
    memory sub_mm2_wei, sub_mm2_dst;
    // reorder3
    memory sub_dst_user;
    // scratchpad
    memory sub_scratchpad;

    bool has_scale = false;
    bool needs_softmax_reorder = false;
    int attn_out_index = 0;
    int stats_out_index = 1;

private:
    std::vector<op_ptr> sdp_op;

public:
    // Check if configuration is supported
    bool initial_check(const std::shared_ptr<subgraph_t> &sg,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs);

    // Construct all params needed for execution
    impl::status_t construct_params(std::shared_ptr<subgraph_t> &sg,
            registry_t &sdp_registry, const dnnl::engine &p_engine,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs);

private:
    op_ptr get_post_op(const op_ptr &op) const;

    impl::status_t record_input_offset(const std::shared_ptr<subgraph_t> &sg,
            const std::vector<logical_tensor_t> &inputs);

    impl::status_t record_sdp_ops(std::shared_ptr<subgraph_t> &sg);

    void memory_planning(registry_t &sdp_registry);

    dnnl::primitive_attr make_primitive_attr(std::shared_ptr<op_t> &op);
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
