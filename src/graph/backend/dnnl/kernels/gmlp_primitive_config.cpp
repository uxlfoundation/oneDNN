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

#include "graph/backend/dnnl/kernels/gmlp_primitive_config.hpp"
#include "graph/backend/dnnl/fusion_info.hpp"

#include "common/compiler_workarounds.hpp"

#define VCHECK_GMLP_PRIMITIVE(cond, status, msg, ...) \
    VCONDCHECK(graph, create, check, gmlp_primitive_kernel_t, (cond), status, \
            msg, ##__VA_ARGS__);

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

op_ptr gmlp_primitive_config_t::get_post_op(const op_ptr &op) const {
    const auto out_val = op->get_output_value(0);
    const auto &consumers = out_val->get_consumers();
    if (consumers.size() != 1) return nullptr;
    return consumers[0].get_op().shared_from_this();
}

status_t gmlp_primitive_config_t::locate_io(std::shared_ptr<subgraph_t> &sg,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {

    using dnnl::impl::utils::one_of;

    auto follow_back = [](std::shared_ptr<value_t> val) {
        while (val->has_producer() && val->get_producer().num_inputs() == 1)
            val = val->get_producer().get_input_value(0);
        return val;
    };

    auto in_tensor_list = [](const value_t *val,
                                  const std::vector<logical_tensor_t> &list) {
        for (auto &t : list)
            if (val->get_logical_tensor().id == t.id) return true;
        return false;
    };

    // Locate ops of interest: matmuls, scale, mask
    const std::unordered_set<op_kind_t> mm_up_post_op_kind
            = {op_kind::dnnl_binary, op_kind::dnnl_softmax, op_kind::dnnl_mask};
    for (const auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_matmul) continue;
        auto post_op = get_post_op(cur_op);
        // mm_up only has one post op: multiply
        // mm_gate has 1 activation post ops
        // mm_down has no post ops
        if (post_op && post_op->get_kind() == op_kind::dnnl_binary) {
            // Locate mm_up and all post ops(scale and mask) here.
            // 1. locate mm_up
            VCHECK_GMLP_PRIMITIVE(mm_up_ == nullptr, status::unimplemented,
                    "Multiple mm_up found");
            mm_up_ = cur_op;
            const auto &ppost_op = get_post_op(post_op);
            if (ppost_op && ppost_op->get_kind() == op_kind::dnnl_matmul) {
                // 2. locate mm_down
                VCHECK_GMLP_PRIMITIVE(mm_down_ == nullptr,
                        status::unimplemented, "Multiple mm_down found");
                mm_down_ = post_op;
            }
        } else {
            VCHECK_GMLP_PRIMITIVE(mm_gate_ == nullptr, status::unimplemented,
                    "Multiple mm_gate found");
            mm_gate_ = cur_op;
        }
    }

    src_ = mm_gate_->get_input_value(0);
    w_gate_ = mm_gate_->get_input_value(1);
    w_up_ = mm_up_->get_input_value(1);
    w_down_ = mm_down_->get_input_value(1);
    dst_ = mm_down_->get_output_value(0);
    // Locate input/outputs: Q, K, V, dst, scale, mask
    VCHECK_GMLP_PRIMITIVE((mm_up_ && mm_down_ && mm_gate_),
            status::unimplemented, "Not all ops are found");
    return status::success;
}

status_t gmlp_primitive_config_t::initial_check(
        const std::shared_ptr<subgraph_t> &sg,
        const std::vector<logical_tensor_t> &inputs) {
    // // At least 3 inputs: Q, K, V
    // VCHECK_GMLP_PRIMITIVE(inputs.size() >= 3, status::invalid_arguments,
    //         "At least 3 inputs are required");

    // // step1(pattern check): Not support gmlpa variants with select as mask
    // // We already have a pattern matcher to ensure that the gmlpa patterns
    // // dispatch to here are knows ones, and we have quant check in gmlpa base
    // // kernel, so here we only check specific variants based on support matrix.
    // const std::unordered_set<graph::op_kind_t> mm1_post_op_kind
    //         = {graph::op_kind::Divide, graph::op_kind::Multiply,
    //                 graph::op_kind::Add, graph::op_kind::Select,
    //                 graph::op_kind::SoftMax};
    // op_ptr mm1 = nullptr, mm2 = nullptr, scale = nullptr;
    // for (const auto &cur_op : sg->get_ops()) {
    //     const auto &op_kind = cur_op->get_kind();
    //     if (op_kind == graph::op_kind::DynamicDequantize
    //             && cur_op->get_attr<std::string>(op_attr::qtype)
    //                     == "per_group") {
    //         if (!cur_op->has_attr(op_attr::group_shape))
    //             return status::invalid_arguments;
    //         const auto &group_shape = cur_op->get_attr<std::vector<int64_t>>(
    //                 op_attr::group_shape);
    //         const auto &input_lt
    //                 = cur_op->get_input_value(0)->get_logical_tensor();
    //         const auto &input_dims = ltw(input_lt).dims();
    //         if (static_cast<int>(group_shape.size()) != ltw(input_lt).ndims())
    //             return status::invalid_arguments;
    //         // Due to the precision issue of ukernel implementation, we only
    //         // support group_num=1 case for now.
    //         for (size_t idx = 0; idx < group_shape.size(); ++idx) {
    //             if (group_shape[idx] != 1
    //                     && group_shape[idx] != input_dims[idx])
    //                 return status::unimplemented;
    //         }
    //         // TODO(zhitao): execute the reorder for scale and zps mannually if the
    //         // transpose attribute is specified as true.
    //         auto post_op = get_post_op(cur_op);
    //         if (post_op && post_op->get_kind() == graph::op_kind::MatMul
    //                 && post_op->has_attr(op_attr::transpose_b)
    //                 && post_op->get_attr<bool>(op_attr::transpose_b))
    //             return status::unimplemented;
    //     }
    //     if (op_kind != graph::op_kind::MatMul) continue;
    //     auto post_op = get_post_op(cur_op);
    //     if (post_op && mm1_post_op_kind.count(post_op->get_kind())) {
    //         mm1 = cur_op;
    //         // Not support select between mm1 and scale(optional)
    //         // GPT-J:[mm1] --> [select] --> [scale]* --> [mask]* --> ...
    //         VCHECK_GMLP_PRIMITIVE(post_op->get_kind() != graph::op_kind::Select,
    //                 status::unimplemented,
    //                 "Not support select between mm1 and scale(optional)");
    //         // scale
    //         if (post_op->get_kind() == graph::op_kind::Divide
    //                 || post_op->get_kind() == graph::op_kind::Multiply) {
    //             // Scale exists, update post_op and traverse to next op
    //             scale = post_op;
    //             post_op = get_post_op(post_op);
    //         }
    //         // mask
    //         if (post_op) {
    //             if (post_op->get_kind() == graph::op_kind::Add) {
    //                 // Mask exists, update post_op and traverse to next op
    //                 post_op = get_post_op(post_op);
    //             }
    //             // Not support select after scale(optional) and mask(optional)
    //             // Distill-Bert:[mm1] --> [scale]* --> [mask]* --> [select] --> ...
    //             VCHECK_GMLP_PRIMITIVE(post_op
    //                             && post_op->get_kind()
    //                                     != graph::op_kind::Select,
    //                     status::unimplemented,
    //                     "Not support select after scale(optional) and "
    //                     "mask(optional)");
    //         }
    //     } else {
    //         mm2 = cur_op;
    //     }
    // }

    // auto find_graph_inport = [&inputs](const std::shared_ptr<value_t> &val) {
    //     auto tmp_val = val;
    //     while (tmp_val->has_producer()) {
    //         const op_t &prod_op = tmp_val->get_producer();
    //         tmp_val = prod_op.get_input_value(0);
    //     }
    //     for (int i = 0; i < (int)inputs.size(); i++) {
    //         if (tmp_val->get_logical_tensor().id == inputs[i].id) { return i; }
    //     }
    //     // If the corresponding input is not found, return an invalid value
    //     return -1;
    // };

    // VCHECK_GMLP_PRIMITIVE(
    //         mm1 && mm2, status::invalid_graph, "mm1 or mm2 is not found");

    // // step3(dims check): only support 4-dims now.
    // int q_id = find_graph_inport(mm1->get_input_value(0));
    // int k_id = find_graph_inport(mm1->get_input_value(1));
    // int v_id = find_graph_inport(mm2->get_input_value(1));

    // VCHECK_GMLP_PRIMITIVE(q_id != -1 && k_id != -1 && v_id != -1,
    //         status::unimplemented, "Q, K, V are not found");
    // VCHECK_GMLP_PRIMITIVE(ltw(inputs[q_id]).vdims().size() == 4
    //                 && ltw(inputs[k_id]).vdims().size() == 4
    //                 && ltw(inputs[v_id]).vdims().size() == 4,
    //         status::unimplemented, "Q, K, V should be 4-dims");

    // // gmlp_primitive only supports single scale value.
    // if (scale) {
    //     const auto &s = scale->get_input_value(1)->get_logical_tensor();
    //     VCHECK_GMLP_PRIMITIVE(ltw(s).nelems() == 1, status::unimplemented,
    //             "Scale should be single value");
    // }

    return status::success;
}

status_t gmlp_primitive_config_t::init(std::shared_ptr<subgraph_t> &sg,
        const dnnl::engine &p_engine,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {

    CHECK(locate_io(sg, inputs, outputs));

    // Retrieve mds and create pd, primitive
    auto md_src = make_dnnl_memory_desc(src_->get_logical_tensor());
    auto md_w_gate = make_dnnl_memory_desc(w_gate_->get_logical_tensor());
    auto md_w_up = make_dnnl_memory_desc(w_up_->get_logical_tensor());
    auto md_w_down = make_dnnl_memory_desc(w_down_->get_logical_tensor());
    // dst is the output of w_down
    auto md_dst = make_dnnl_memory_desc(dst_->get_logical_tensor());

    // TODO: enable quantization

    std::cout << "hardcode activation for mlp Creating GMLP primitive\n";
    const alg_kind_t activation = alg_kind::eltwise_swish;
    CHECK(create_gated_mlp_pd(gmlp_pd_, p_engine.get(), md_src.get(),
            md_w_gate.get(), md_w_up.get(), md_w_down.get(), md_dst.get(),
            activation, nullptr));

    auto status = gmlp_pd_->create_primitive(gmlp_prim_, p_engine.get());

    VCONDCHECK(graph, create, dispatch, gmlp, status == status::success, status,
            "could not create gmlp primitive, falling back\n");
    return status;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
