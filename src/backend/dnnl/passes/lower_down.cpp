/*******************************************************************************
 * Copyright 2021-2022 Intel Corporation
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
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "interface/c_types_map.hpp"
#include "interface/graph.hpp"
#include "interface/op_schema.hpp"
#include "interface/shape_infer.hpp"
#include "utils/utils.hpp"

#include "backend/dnnl/fusion_info.hpp"
#include "backend/dnnl/internal_attrs.hpp"
#include "backend/dnnl/op_executable.hpp"

#include "backend/dnnl/passes/insert_ops.hpp"
#include "backend/dnnl/passes/lower_down.hpp"
#include "backend/dnnl/passes/utils.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
using op_t = impl::op_t;
using op_ptr = std::shared_ptr<impl::op_t>;
using value_ptr = std::shared_ptr<impl::value_t>;
using ltw = impl::logical_tensor_wrapper_t;

static bool need_attr_adjustment(op_kind_t kind) {
    static const std::set<op_kind_t> ops {impl::op_kind::SquaredDifference};
    return ops.count(kind) != 0;
}

static bool has_optional_bias(op_kind_t kind) {
    std::set<op_kind_t> ops {op_kind::dnnl_convolution, op_kind::dnnl_matmul,
            op_kind::dnnl_convtranspose};
    return ops.count(kind) != 0;
}

// TODO(xxx): extend to support other ops
static bool has_int8_support(op_kind_t kind) {
    std::set<op_kind_t> ops {op_kind::dnnl_convolution, op_kind::dnnl_matmul,
            op_kind::dnnl_convtranspose, op_kind::dnnl_reorder};
    return ops.count(kind) != 0;
}

// TODO(xxx): extend to support other ops
static bool is_output_scales_supported(op_kind_t kind) {
    // ops which don't support output scales
    std::set<op_kind_t> ops {op_kind::dnnl_pool, op_kind::dnnl_eltwise};
    return ops.count(kind) == 0;
}

impl::status_t split_squared_difference(std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (need_attr_adjustment(cur_op->get_kind())) {
            if (cur_op->get_kind() == impl::op_kind::SquaredDifference) {
                op_ptr subtract = std::make_shared<op_t>(op_kind::dnnl_binary);
                subtract->set_attr<int64_t>(op_attr::alg_kind,
                        static_cast<int64_t>(dnnl::algorithm::binary_sub));

                rewriter.replace_op(cur_op, subtract);

                op_ptr square = std::make_shared<op_t>(op_kind::dnnl_eltwise);
                square->set_attr<int64_t>(op_attr::alg_kind,
                        static_cast<int64_t>(dnnl::algorithm::eltwise_square));

                const float default_attr_value = 0.0f;
                square->set_attr<float>(op_attr::alpha, default_attr_value);
                square->set_attr<float>(op_attr::beta, default_attr_value);

                rewriter.insert_op_after(square, subtract, 0);
                insert_empty_scratchpad(subtract);
                insert_empty_scratchpad(square);
            }
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

impl::status_t check_with_bias(std::shared_ptr<subgraph_t> &sg) {
    for (auto &cur_op : sg->get_ops()) {
        if (!has_optional_bias(cur_op->get_kind())) continue;
        if (cur_op->num_inputs() == 3) {
            cur_op->set_attr<bool>(op_attr::with_bias, true);
        } else {
            cur_op->set_attr<bool>(op_attr::with_bias, false);
        }
    }
    return impl::status::success;
}

impl::status_t fuse_bias_add(std::shared_ptr<subgraph_t> &sg) {
    std::vector<op_ptr> bias_add_ops;
    subgraph_rewriter_t rewriter(sg);

    std::set<op_t *> visited;
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_binary
                || visited.count(cur_op.get()) != 0)
            continue;

        if (!cur_op->has_attr(op_attr::is_bias_add)
                || !cur_op->get_attr<bool>(op_attr::is_bias_add))
            continue;

        // bias add may be standalone
        if (!cur_op->get_input_value(0)->has_producer()) continue;

        bias_add_ops.emplace_back(cur_op);
        visited.insert(cur_op.get());
    }

    for (auto &bias_add : bias_add_ops) {
        auto in_val = bias_add->get_input_value(0);
        auto &prv_op = in_val->get_producer();
        if (!has_optional_bias(prv_op.get_kind())) continue;
        rewriter.fuse_op_to_predecessor(bias_add);
        prv_op.set_attr<bool>(op_attr::with_bias, true);
    }

    rewriter.run();
    return impl::status::success;
}

impl::status_t replace_quant_dequant_with_mul_scales(
        std::shared_ptr<subgraph_t> &sg) {
    // replaces Quant/Dequant OPs with mul_scales, zps attribute is ignored.
    std::vector<op_ptr> q_dq_ops;
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() == impl::op_kind::Quantize
                || cur_op->get_kind() == impl::op_kind::Dequantize) {
            q_dq_ops.emplace_back(cur_op);
        }
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &cur_op : q_dq_ops) {
        // directly ignoring zps attribute
        const auto &scales
                = cur_op->get_attr<std::vector<float>>(op_attr::scales);
        const auto &qtype = cur_op->get_attr<std::string>(op_attr::qtype);
        const auto &axis = cur_op->get_attr<int64_t>(op_attr::axis);

        auto in_vals = cur_op->get_input_values();
        auto out_vals = cur_op->get_output_values();
        assertm(in_vals.size() == 1 && out_vals.size() == 1,
                "static quantize/dequantize should only have one input and "
                "output");

        op_ptr mul_scales_op = std::make_shared<op_t>(op_kind::dnnl_mul_scales);
        mul_scales_op->set_attr<std::string>(op_attr::qtype, qtype);
        mul_scales_op->set_attr<int64_t>(op_attr::axis, axis);
        if (impl::op_kind::Dequantize == cur_op->get_kind()) {
            mul_scales_op->set_attr<std::vector<float>>(
                    op_attr::scales, scales);
        } else { // Quantize
            assertm(std::all_of(scales.begin(), scales.end(),
                            [](float i) { return i != 0.f; }),
                    "scales can't be zero");

            std::vector<float> inv_scales = dnnl_impl::utils::fmap(
                    scales, [](float s) { return 1.f / s; });
            mul_scales_op->set_attr<std::vector<float>>(
                    op_attr::scales, inv_scales);
        }

        // reconnect
        in_vals[0]->remove_consumer(*cur_op, 0);
        in_vals[0]->add_consumer(*mul_scales_op, 0);
        mul_scales_op->add_input(in_vals[0]);
        mul_scales_op->add_output(out_vals[0]);

        // add new op and delete quantize or dequantize op
        rewriter.to_insert(mul_scales_op);
        rewriter.to_remove(cur_op);
    }
    rewriter.run();
    return impl::status::success;
}

impl::status_t fuse_output_scales(std::shared_ptr<subgraph_t> &sg) {
    auto &mgr = sg->fusion_info_mgr_;
    subgraph_rewriter_t rewriter(sg);

    std::vector<std::pair<op_t *, op_t *>> fuse_groups;

    std::set<op_t *> visited;
    for (auto &cur_op : sg->get_ops()) {
        if ((!has_int8_support(cur_op->get_kind())
                    && cur_op->get_kind() != op_kind::dnnl_softmax)
                || visited.count(cur_op.get()) != 0)
            continue;

        auto out_val = cur_op->get_output_values()[0];
        auto consumers = out_val->get_consumers();
        if (consumers.size() != 1) continue;

        auto &next_op = consumers[0].get_op();
        if (next_op.get_kind() != op_kind::dnnl_mul_scales) continue;

        fuse_groups.emplace_back(
                std::pair<op_t *, op_t *> {cur_op.get(), &next_op});
        visited.insert(cur_op.get());
        visited.insert(&next_op);
    }

    for (auto &fuse_group : fuse_groups) {
        auto base_op = fuse_group.first;
        auto scale_op = fuse_group.second;

        int64_t key = -1;
        if (base_op->has_attr(op_attr::fusion_info_key)
                && base_op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
            key = base_op->get_attr<int64_t>(op_attr::fusion_info_key);
        } else {
            key = mgr.init_info();
            base_op->set_attr<int64_t>(op_attr::fusion_info_key, key);
        }

        fusion_info_t &fusion_info = mgr.get_mutable_info(key);
        fusion_info.set_output_scales(scale_op->shared_from_this());
        rewriter.fuse_op_to_predecessor(scale_op->shared_from_this());
    }

    rewriter.run();
    return impl::status::success;
}

// replace mul_scales and add_zps with binary_mul and binary_add respectively
impl::status_t replace_quant_data_with_binary_post_op(
        std::shared_ptr<subgraph_t> &sg) {
    const auto get_next_op = [](const op_t *op) -> op_t * {
        const value_ptr out_val = op->get_output_value(0);
        if (!out_val->get_consumers().empty()) {
            size_t offset = out_val->get_consumers()[0].get_offset();
            auto &next_op = out_val->get_consumers()[0].get_op();
            if (next_op.get_kind() == dnnl_impl::op_kind::dnnl_add_zps
                    || next_op.get_kind() == dnnl_impl::op_kind::dnnl_sub_zps
                    || next_op.get_kind()
                            == dnnl_impl::op_kind::dnnl_mul_scales)
                // cur_op's output must be mul_scales and add_zps ops' first
                // input
                return offset == 0 ? &next_op : nullptr;
            else
                return &out_val->get_consumers()[0].get_op();
        } else
            return nullptr;
    };

    const std::set<op_kind_t> accepted_kinds_in_chain
            = {op_kind::dnnl_binary, op_kind::dnnl_mul_scales,
                    op_kind::dnnl_add_zps, op_kind::dnnl_sub_zps};

    subgraph_rewriter_t rewriter(sg);
    std::set<op_t *> visited;
    for (const auto &cur_op : sg->get_ops()) {
        if ((is_output_scales_supported(cur_op->get_kind())
                    && cur_op->get_kind() != op_kind::dnnl_softmax)
                || visited.count(cur_op.get()))
            continue;

        visited.insert(cur_op.get());
        op_t *next_op = get_next_op(cur_op.get());
        while (next_op && accepted_kinds_in_chain.count(next_op->get_kind())) {
            // TODO(xxx): handle the case where other binary input is a
            // 'add_zps'. For patterns like 'int8 pool + binary-mul',
            // 'mul_scales' which feeds 'binary-mul' will get removed.
            // Standalone 'add_zps' is not supported right now, and here we can
            // cover that by treating it as other 'add_zps' OPs in a chain.
            if (next_op->get_kind() == op_kind::dnnl_binary
                    || visited.count(next_op)) {
                next_op = get_next_op(next_op);
                continue;
            }

            // replace quant related op with binary
            op_t *quant_data_op = next_op;
            auto algo = (quant_data_op->get_kind() == op_kind::dnnl_mul_scales)
                    ? dnnl::algorithm::binary_mul
                    : quant_data_op->get_kind() == op_kind::dnnl_add_zps
                            ? dnnl::algorithm::binary_add
                            : dnnl::algorithm::binary_sub;
            op_ptr bin_op = std::make_shared<op_t>(op_kind::dnnl_binary);
            bin_op->set_attr<int64_t>(
                    op_attr::alg_kind, static_cast<int64_t>(algo));
            auto in_val = quant_data_op->get_input_value(0);
            in_val->remove_consumer(*quant_data_op, 0);
            in_val->add_consumer(*bin_op, 0);
            bin_op->add_input(in_val);
            auto out_val = quant_data_op->get_output_value(0);
            bin_op->add_output(out_val);
            insert_empty_scratchpad(bin_op);

            // add quant data as a constant input
            const auto qtype
                    = quant_data_op->get_attr<std::string>(op_attr::qtype);
            const auto axis = quant_data_op->get_attr<int64_t>(op_attr::axis);
            const std::vector<int64_t> out_shape
                    = ltw(out_val->get_logical_tensor()).vdims();
            std::vector<int64_t> new_shape(out_shape.size(), 1);
            if (qtype == "per_tensor") new_shape[axis] = out_shape[axis];
            op_ptr const_data_op;
            if (quant_data_op->get_kind() == op_kind::dnnl_mul_scales) {
                const auto scales = quant_data_op->get_attr<std::vector<float>>(
                        op_attr::scales);
                const_data_op
                        = std::make_shared<op_t>(op_kind::dnnl_constant_scales);
                const_data_op->set_attr(op_attr::scales, scales);
            } else { // add_zps
                const auto zps = quant_data_op->get_attr<std::vector<int64_t>>(
                        op_attr::zps);
                const_data_op
                        = std::make_shared<op_t>(op_kind::dnnl_constant_zps);
                const_data_op->set_attr(op_attr::zps, zps);
            }
            const_data_op->set_attr(op_attr::shape, new_shape);
            impl::logical_tensor_t const_data_dst_lt
                    = impl::empty_logical_tensor_with_default_id();
            auto const_data_dst_value = std::make_shared<value_t>(
                    *const_data_op, 0, const_data_dst_lt, true);
            auto out_dtype = const_data_op->has_attr(op_attr::zps)
                    ? impl::data_type::s32
                    : impl::data_type::f32;
            const_data_dst_value->set_data_type(out_dtype);
            const_data_dst_value->set_layout_type(impl::layout_type::strided);
            const_data_op->add_output(const_data_dst_value);

            // connect binary and constant data
            bin_op->connect_input(1, const_data_dst_value);

            rewriter.to_insert(bin_op);
            rewriter.to_insert(const_data_op);
            rewriter.to_remove(quant_data_op->shared_from_this());

            visited.insert(next_op);
            next_op = get_next_op(next_op);
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

impl::status_t fold_mul_scales(std::shared_ptr<subgraph_t> &sg) {
    // lambda function to fold the consecutive mul_scales ops
    auto fold_mul_scales_func = [&]() {
        std::vector<std::pair<op_t *, op_t *>> folding_groups;
        std::set<op_t *> visited;
        for (const auto &cur_op : sg->get_ops()) {
            if (cur_op->get_kind() != op_kind::dnnl_mul_scales
                    || visited.count(cur_op.get()) != 0)
                continue;

            assertm(cur_op->num_outputs() == 1,
                    "cur_op should have only one output value.");
            auto out_val = cur_op->get_output_values()[0];
            auto consumers = out_val->get_consumers();
            if (consumers.empty()) continue;

            auto &consumer_op = consumers[0].get_op();
            if (consumer_op.get_kind() != op_kind::dnnl_mul_scales) continue;

            folding_groups.emplace_back(
                    std::pair<op_t *, op_t *> {cur_op.get(), &consumer_op});
            visited.insert(cur_op.get());
            visited.insert(&consumer_op);
        }

        if (folding_groups.empty()) return false;

        subgraph_rewriter_t rewriter(sg);
        for (auto &folding_ops : folding_groups) {
            auto base_op = folding_ops.first;
            auto next_op = folding_ops.second;

            // update the scales
            const auto &scales_base
                    = base_op->get_attr<std::vector<float>>(op_attr::scales);
            const auto &scales_next
                    = next_op->get_attr<std::vector<float>>(op_attr::scales);
            std::vector<float> new_scales(
                    std::max(scales_base.size(), scales_next.size()), 1.f);
            // per-channel -> per-tensor
            if (scales_base.size() > scales_next.size()) {
                for (size_t i = 0; i < new_scales.size(); ++i)
                    new_scales[i] = scales_base[i] * scales_next[0];
            } else {
                for (size_t i = 0; i < new_scales.size(); ++i)
                    new_scales[i] = scales_base[0] * scales_next[i];
                // set attrs
                base_op->set_attr<int64_t>(op_attr::axis,
                        next_op->get_attr<int64_t>(op_attr::axis));
                base_op->set_attr<std::string>(op_attr::qtype,
                        next_op->get_attr<std::string>(op_attr::qtype));
            }
            base_op->set_attr<std::vector<float>>(op_attr::scales, new_scales);

            rewriter.fuse_op_to_predecessor(next_op->shared_from_this(), 0);
        }
        rewriter.run();
        return true;
    };

    bool changed = true;
    do {
        changed = fold_mul_scales_func();
    } while (changed);
    return impl::status::success;
}

impl::status_t fuse_to_int8_conv_or_deconv(std::shared_ptr<subgraph_t> &sg) {
    std::vector<std::vector<op_t *>> fusion_groups;
    for (const auto &op : sg->get_ops()) {
        if ((op->get_kind() == op_kind::dnnl_convolution
                    || op->get_kind() == op_kind::dnnl_convtranspose)
                && op->get_input_value(0)->has_producer()
                && op->get_input_value(1)->has_producer()) {
            auto &in0 = op->get_input_value(0)->get_producer();
            auto &in1 = op->get_input_value(1)->get_producer();
            if (in0.get_kind() != op_kind::dnnl_mul_scales
                    || in1.get_kind() != op_kind::dnnl_mul_scales)
                continue;

            fusion_groups.emplace_back(
                    std::vector<op_t *> {op.get(), &in0, &in1});
        }
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &fusion_group : fusion_groups) {
        op_t *conv_op = fusion_group[0];
        op_t *in0 = fusion_group[1];
        op_t *in1 = fusion_group[2];

        op_ptr mul_op = std::make_shared<op_t>(op_kind::dnnl_mul_scales);

        auto dq_src_scales = in0->get_attr<std::vector<float>>(op_attr::scales);
        auto dq_wei_scales = in1->get_attr<std::vector<float>>(op_attr::scales);
        std::vector<float> fused_scale(
                std::max(dq_src_scales.size(), dq_wei_scales.size()), 0);
        if (dq_src_scales.size() >= dq_wei_scales.size()) {
            for (int i = 0; i < dq_src_scales.size(); i++)
                fused_scale[i] = (dq_src_scales[i] * dq_wei_scales[0]);
            mul_op->set_attr<int64_t>(
                    op_attr::axis, in0->get_attr<int64_t>(op_attr::axis));
            mul_op->set_attr<std::string>(
                    op_attr::qtype, in0->get_attr<std::string>(op_attr::qtype));
        } else {
            // Currently for ConvTranspose, the output channel in weight tensor
            // (OC/g, IC, H, W) is not equal to the one in output tensor
            // (N, OC, H, W) if `groups` > 1, so the size of weight's
            // per-channel scale is not the same as the output channel in output
            // tensor, here we will broadcast scales from `OC/g` to `OC`.
            int64_t group = conv_op->get_attr<int64_t>(op_attr::groups);
            if (conv_op->get_kind() == op_kind::dnnl_convtranspose
                    && group > 1) {
                fused_scale.resize(group * dq_wei_scales.size(), 0);
                for (int i = 0; i < fused_scale.size(); ++i)
                    fused_scale[i] = (dq_src_scales[0]
                            * dq_wei_scales[i % dq_wei_scales.size()]);
            } else {
                for (int i = 0; i < fused_scale.size(); ++i)
                    fused_scale[i] = (dq_src_scales[0] * dq_wei_scales[i]);
            }
            // FIXME(qun) the axis of output_scales should be related to data
            // format
            mul_op->set_attr<int64_t>(
                    op_attr::axis, 1); // hardcode to 1 for pytorch
            mul_op->set_attr<std::string>(
                    op_attr::qtype, in1->get_attr<std::string>(op_attr::qtype));
        }
        mul_op->set_attr<std::vector<float>>(op_attr::scales, fused_scale);

        auto in0_ivalue = in0->get_input_value(0);
        auto in1_ivalue = in1->get_input_value(0);
        conv_op->connect_input(0, in0_ivalue);
        conv_op->connect_input(1, in1_ivalue);
        in0_ivalue->remove_consumer(*in0, 0);
        in1_ivalue->remove_consumer(*in1, 0);

        rewriter.to_remove(in0->shared_from_this());
        rewriter.to_remove(in1->shared_from_this());

        if (conv_op->num_inputs() == 3) { //with bias
            op_ptr mul_op1 = std::make_shared<op_t>(op_kind::dnnl_mul_scales);

            assertm(std::all_of(fused_scale.begin(), fused_scale.end(),
                            [](float i) { return i != 0.f; }),
                    "scales can't be zero");

            std::vector<float> inv_scales(fused_scale.size());
            for (int i = 0; i < fused_scale.size(); i++)
                inv_scales[i] = 1.f / fused_scale[i];

            // FIXME(xxx) add other attrs
            mul_op1->set_attr<std::vector<float>>(op_attr::scales, inv_scales);
            mul_op1->set_attr<int64_t>(op_attr::axis, 0);
            mul_op1->set_attr<std::string>(op_attr::qtype,
                    mul_op->get_attr<std::string>(op_attr::qtype));

            rewriter.insert_op_before(mul_op1, conv_op->shared_from_this(), 2);
            // Some of oneDNN's conv primitive implementation can't support bf16
            // bias
            mul_op1->get_output_value(0)->set_data_type(impl::data_type::f32);
        }

        rewriter.insert_op_after(mul_op, conv_op->shared_from_this(), 0);
    }

    rewriter.run();
    return infer_shape(sg);
}

impl::status_t fuse_to_int8_matmul(std::shared_ptr<subgraph_t> &sg) {
    std::vector<std::vector<op_t *>> fusion_groups;
    for (const auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_matmul
                || !cur_op->get_input_value(0)->has_producer()
                || !cur_op->get_input_value(1)->has_producer())
            continue;
        auto &in0 = cur_op->get_input_value(0)->get_producer();
        auto &in1 = cur_op->get_input_value(1)->get_producer();
        if (in0.get_kind() == op_kind::dnnl_mul_scales
                && in1.get_kind() == op_kind::dnnl_mul_scales)
            fusion_groups.emplace_back(
                    std::vector<op_t *> {cur_op.get(), &in0, &in1});
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &fusion_group : fusion_groups) {
        op_t *matmul_op = fusion_group[0];
        op_t *in0 = fusion_group[1];
        op_t *in1 = fusion_group[2];

        op_ptr mul_scales_op = std::make_shared<op_t>(op_kind::dnnl_mul_scales);

        const auto &dq_src_scales
                = in0->get_attr<std::vector<float>>(op_attr::scales);
        const auto &dq_wei_scales
                = in1->get_attr<std::vector<float>>(op_attr::scales);
        std::vector<float> fused_scales(
                std::max(dq_src_scales.size(), dq_wei_scales.size()), 1.f);
        // src: per_tensor, weight: per_channel
        if (dq_src_scales.size() < dq_wei_scales.size()) {
            for (size_t i = 0; i < dq_wei_scales.size(); ++i)
                fused_scales[i] = dq_src_scales[0] * dq_wei_scales[i];
            // FIXME(wuxun): if quantization is per-channel, need to set axis
            // to the last dimension of dst
            // find the output edge op to get the ndims
            op_t begin_op = *matmul_op;
            while (!begin_op.get_output_value(0)->get_consumers().empty()) {
                begin_op = begin_op.get_output_value(0)
                                   ->get_consumers()[0]
                                   .get_op();
            }
            // if one of matmul's inputs is 1D tensor, the real ndims of
            // matmul's output should be (cur_ndims + 1), so we only need set
            // axis to cur_ndims
            int64_t new_axis
                    = begin_op.get_output_value(0)->get_logical_tensor().ndims
                    - 1;
            if (matmul_op->get_input_value(0)->get_logical_tensor().ndims == 1
                    || matmul_op->get_input_value(1)->get_logical_tensor().ndims
                            == 1)
                new_axis += 1;
            mul_scales_op->set_attr<int64_t>(op_attr::axis, new_axis);
            mul_scales_op->set_attr<std::string>(
                    op_attr::qtype, in1->get_attr<std::string>(op_attr::qtype));
        } else {
            for (size_t i = 0; i < dq_src_scales.size(); ++i)
                fused_scales[i] = dq_src_scales[i] * dq_wei_scales[0];
            mul_scales_op->set_attr<int64_t>(
                    op_attr::axis, in0->get_attr<int64_t>(op_attr::axis));
            mul_scales_op->set_attr<std::string>(
                    op_attr::qtype, in0->get_attr<std::string>(op_attr::qtype));
        }
        mul_scales_op->set_attr<std::vector<float>>(
                op_attr::scales, fused_scales);

        // update the connection relationship between matmul and mul_scales ops
        auto in0_value = in0->get_input_value(0);
        auto in1_value = in1->get_input_value(0);
        matmul_op->connect_input(0, in0_value);
        matmul_op->connect_input(1, in1_value);
        in0_value->remove_consumer(*in0, 0);
        in1_value->remove_consumer(*in1, 0);

        rewriter.to_remove(in0->shared_from_this());
        rewriter.to_remove(in1->shared_from_this());

        // with bias
        if (matmul_op->num_inputs() == 3) {
            op_ptr bias_mul_op
                    = std::make_shared<op_t>(op_kind::dnnl_mul_scales);

            assertm(std::all_of(fused_scales.begin(), fused_scales.end(),
                            [](float i) { return i != 0.f; }),
                    "scales can't be zero");

            std::vector<float> inv_scales(fused_scales.size(), 1.f);
            for (size_t i = 0; i < inv_scales.size(); ++i)
                inv_scales[i] = 1.f / fused_scales[i];
            bias_mul_op->set_attr<std::vector<float>>(
                    op_attr::scales, inv_scales);
            bias_mul_op->set_attr<int64_t>(op_attr::axis, 0);
            bias_mul_op->set_attr<std::string>(op_attr::qtype,
                    mul_scales_op->get_attr<std::string>(op_attr::qtype));

            rewriter.insert_op_before(
                    bias_mul_op, matmul_op->shared_from_this(), 2);
            // Some of oneDNN's matmul primitive implementation can't support
            // bf16 bias
            bias_mul_op->get_output_value(0)->set_data_type(
                    impl::data_type::f32);
        }

        rewriter.insert_op_after(
                mul_scales_op, matmul_op->shared_from_this(), 0);
    }

    rewriter.run();
    return infer_shape(sg);
}

impl::status_t fuse_to_int8_reorder(std::shared_ptr<subgraph_t> &sg) {
    std::vector<std::vector<op_t *>> fusion_groups;
    for (const auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_reorder
                || !cur_op->get_input_value(0)->has_producer())
            continue;
        auto &in = cur_op->get_input_value(0)->get_producer();
        if (in.get_kind() == op_kind::dnnl_mul_scales)
            fusion_groups.emplace_back(std::vector<op_t *> {cur_op.get(), &in});
    }

    for (auto &fusion_group : fusion_groups) {
        op_t *reorder_op = fusion_group[0];
        op_t *in_op = fusion_group[1];

        // update the connection relationship between reorder and mul_scales ops
        // basically switch the order of these two ops
        auto in_op_in_value = in_op->get_input_value(0);
        auto reorder_op_in_value = reorder_op->get_input_value(0);
        auto out_value = reorder_op->get_output_value(0);

        reorder_op->connect_input(0, in_op_in_value);
        in_op_in_value->remove_consumer(*in_op, 0);
        reorder_op->connect_output(0, reorder_op_in_value);
        reorder_op_in_value->set_producer(*reorder_op);
        reorder_op_in_value->remove_consumer(*reorder_op, 0);

        in_op->connect_input(0, reorder_op_in_value);
        in_op->connect_output(0, out_value);
        out_value->set_producer(*in_op);
    }

    return infer_shape(sg);
}

impl::status_t fuse_to_int8_concat(std::shared_ptr<subgraph_t> &sg) {
    std::vector<op_t *> fusion_ops;
    for (const auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_concat) continue;

        bool matched = true;
        for (auto &in : cur_op->get_input_values()) {
            if (!in->has_producer()
                    || in->get_producer().get_kind()
                            != impl::op_kind::Dequantize) {
                matched = false;
                break;
            }
        }

        if (!matched) continue;

        fusion_ops.emplace_back(cur_op.get());
    }

    if (fusion_ops.empty()) return impl::status::success;

    subgraph_rewriter_t rewriter(sg);
    for (auto &concat_op : fusion_ops) {
        // We only support int8 Concat with the same scales and zps
        // for all inputs. As Concat is not changing the range of values,
        // we are simply ignoring here De/Quantize OPs by disconnecting
        // them from the Concat.
        for (size_t i = 0; i < concat_op->num_inputs(); ++i) {
            op_t &dq_op = concat_op->get_input_value(i)->get_producer();
            assertm(dq_op.get_kind() == impl::op_kind::Dequantize,
                    "int8 concat input values should be produced by Dequantize "
                    "op.");
            value_ptr in_value = dq_op.get_input_value(0);
            in_value->remove_consumer(dq_op, 0);
            concat_op->connect_input(i, in_value);
            rewriter.to_remove((&dq_op)->shared_from_this());
        }

        assertm(concat_op->get_output_value(0)->get_consumers().size() == 1,
                "concat's successor op should only have one consumer.");
        op_t &q_op
                = concat_op->get_output_value(0)->get_consumers()[0].get_op();
        assertm(q_op.get_kind() == impl::op_kind::Quantize,
                "int8 concat output value should be consumed by Quantize op.");
        value_ptr out_value = q_op.get_output_value(0);
        concat_op->connect_output(0, out_value);
        rewriter.to_remove((&q_op)->shared_from_this());
    }

    rewriter.run();
    return impl::status::success;
}

// Moves quant related OPs after pool op.
// This function has effect only if post-ops are present.
//
//       |                   |
//      zps0               zps0
//       |       |           |         |
//    scales0   zps1       pool       zps1
//       |       |           |         |
//     pool   scales1     scales0   scales1
//       \      /              \      /
//        binary      =>        binary
//           |                     |
//        scales2               scales2
//           |                     |
//          zps2                  zps2
//           |                     |
//
impl::status_t fuse_to_int8_pool(std::shared_ptr<subgraph_t> &sg) {
    std::vector<op_ptr> pool_ops;
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() == op_kind::dnnl_pool
                && cur_op->get_input_value(0)->has_producer()
                && !cur_op->get_output_value(0)->get_consumers().empty()
                && cur_op->get_input_value(0)->get_producer().get_kind()
                        == op_kind::dnnl_mul_scales) {
            pool_ops.emplace_back(cur_op);
        }
    }

    if (pool_ops.empty()) return impl::status::success;

    for (auto &pool_op : pool_ops) {
        value_ptr pool_in_val = pool_op->get_input_value(0);
        value_ptr pool_out_val = pool_op->get_output_value(0);
        op_t &scales_op = pool_in_val->get_producer();

        auto csm = pool_out_val->get_consumers()[0];
        op_t &csm_op = csm.get_op();
        const size_t csm_offset = csm.get_offset();

        value_ptr scales_in_val = scales_op.get_input_value(0);
        if (!scales_in_val->has_producer()) continue;

        // connect pooling with a scales input
        scales_in_val->remove_consumer(scales_op, 0);
        pool_op->connect_input(0, scales_in_val);

        // connect zps with a pooling using a fresh value
        impl::logical_tensor_t pool_to_scales_lt
                = impl::empty_logical_tensor_with_default_id();
        auto pool_to_scales_val = std::make_shared<value_t>(
                *pool_op, 0, pool_to_scales_lt, true);
        pool_to_scales_val->set_data_type(
                scales_in_val->get_logical_tensor().data_type);
        pool_op->connect_output(0, pool_to_scales_val);
        scales_op.connect_input(0, pool_to_scales_val);

        // connect scales with a binary using a fresh value
        impl::logical_tensor_t scales_to_bin_lt
                = impl::empty_logical_tensor_with_default_id();
        auto scales_to_bin_val = std::make_shared<value_t>(
                scales_op, 0, scales_to_bin_lt, true);
        scales_to_bin_val->set_data_type(
                scales_in_val->get_logical_tensor().data_type);
        scales_op.connect_output(0, scales_to_bin_val);
        csm_op.connect_input(csm_offset, scales_to_bin_val);
    }

    return infer_shape(sg);
}

// Moves unfused src zps after pool op, to implement it as a post-binary
//
//       |       |           |         |
//     zps0   zps1         pool       zps1
//       |       |           |         |
//     pool   scales1      zps0     scales1
//       \      /              \      /
//        binary      =>        binary
//           |                     |
//        scales2               scales2
//           |                     |
//          zps2                  zps2
//           |                     |
//
impl::status_t defer_src_zps_for_pool(std::shared_ptr<subgraph_t> &sg) {
    std::vector<op_ptr> pool_ops;
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() == op_kind::dnnl_pool
                && cur_op->get_input_value(0)->has_producer()
                && !cur_op->get_output_value(0)->get_consumers().empty()
                && cur_op->get_input_value(0)->get_producer().get_kind()
                        == op_kind::dnnl_sub_zps) {
            pool_ops.emplace_back(cur_op);
        }
    }

    if (pool_ops.empty()) return impl::status::success;

    for (auto &pool_op : pool_ops) {
        value_ptr pool_in_val = pool_op->get_input_value(0);
        value_ptr pool_out_val = pool_op->get_output_value(0);

        op_t &sub_zps_op = pool_in_val->get_producer();

        auto csm = pool_out_val->get_consumers()[0];
        op_t &csm_op = csm.get_op();
        const size_t csm_offset = csm.get_offset();

        value_ptr zps_in_val = sub_zps_op.get_input_value(0);

        // connect pooling with a zps input
        zps_in_val->remove_consumer(*pool_op, 0);
        pool_op->connect_input(0, zps_in_val);

        // connect zps with a pooling using a fresh value
        impl::logical_tensor_t pool_to_zps_lt
                = impl::empty_logical_tensor_with_default_id();
        auto pool_to_zps_val
                = std::make_shared<value_t>(*pool_op, 0, pool_to_zps_lt, true);
        pool_to_zps_val->set_data_type(
                zps_in_val->get_logical_tensor().data_type);
        pool_op->connect_output(0, pool_to_zps_val);
        sub_zps_op.connect_input(0, pool_to_zps_val);

        // connect zps with a binary using a fresh value
        impl::logical_tensor_t zps_to_bin_lt
                = impl::empty_logical_tensor_with_default_id();
        auto zps_to_bin_val
                = std::make_shared<value_t>(sub_zps_op, 0, zps_to_bin_lt, true);
        zps_to_bin_val->set_data_type(
                zps_in_val->get_logical_tensor().data_type);
        sub_zps_op.connect_output(0, zps_to_bin_val);
        csm_op.connect_input(csm_offset, zps_to_bin_val);
    }

    return infer_shape(sg);
}

impl::status_t fuse_to_shuffle(std::shared_ptr<subgraph_t> &sg) {
    std::vector<std::vector<op_t *>> fusion_groups;
    for (const auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_reshape) continue;

        if (cur_op->get_output_value(0)->get_consumers().size() != 1) continue;
        auto &next0 = cur_op->get_output_value(0)->get_consumers()[0].get_op();
        if (next0.get_kind() != op_kind::dnnl_transpose) continue;

        if (next0.get_output_value(0)->get_consumers().size() != 1) continue;
        auto &next1 = next0.get_output_value(0)->get_consumers()[0].get_op();
        if (next1.get_kind() != op_kind::dnnl_reshape) continue;

        fusion_groups.emplace_back(
                std::vector<op_t *> {cur_op.get(), &next0, &next1});
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &fusion_group : fusion_groups) {
        op_t *reshape0 = fusion_group[0];
        op_t *reshape1 = fusion_group[2];
        op_t *transpose = fusion_group[1];

        const auto res = shuffle_fusible(reshape0, reshape1, transpose);
        const bool fusible = res.first;
        if (!fusible) continue;

        op_ptr shuffle = std::make_shared<op_t>(op_kind::dnnl_shuffle);

        value_ptr in_value = reshape0->get_input_value(0);
        value_ptr out_value = reshape1->get_output_value(0);

        const auto src_shape = ltw(in_value->get_logical_tensor()).vdims();
        const auto attr_shape
                = reshape0->get_attr<std::vector<int64_t>>(op_attr::shape);
        const size_t axis = res.second.first;
        const int64_t group = res.second.second;
        shuffle->set_attr<int64_t>(op_attr::axis, static_cast<int64_t>(axis));
        shuffle->set_attr<int64_t>(op_attr::groups, group);

        shuffle->connect_input(0, in_value);
        in_value->remove_consumer(*reshape0, 0);

        shuffle->add_output(out_value);

        insert_empty_scratchpad(shuffle);

        for (auto &del_op : fusion_group) {
            rewriter.to_remove(del_op->shared_from_this());
        }

        rewriter.to_insert(shuffle);
    }

    rewriter.run();
    return impl::status::success;
}

status_t fold_sum_scales(std::shared_ptr<subgraph_t> &sg) {
    std::set<op_t *> visited;
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (!(cur_op->get_kind() == op_kind::dnnl_binary
                    && static_cast<dnnl::algorithm>(
                               cur_op->get_attr<int64_t>(op_attr::alg_kind))
                            == dnnl::algorithm::binary_add)
                || visited.count(cur_op.get()))
            continue;

        visited.insert(cur_op.get());
        size_t mul_scale_op_offset = 2;
        auto lhs_val = cur_op->get_input_value(0);
        auto rhs_val = cur_op->get_input_value(1);

        if (!lhs_val->has_producer() || !rhs_val->has_producer()) { continue; }
        const auto &l_op = lhs_val->get_producer();
        const auto &r_op = rhs_val->get_producer();

        auto consumers = cur_op->get_output_values()[0]->get_consumers();
        if (consumers.empty()
                || consumers[0].get_op().get_kind()
                        != op_kind::dnnl_mul_scales) {
            continue;
        }

        if (l_op.get_kind() != op_kind::dnnl_mul_scales
                || r_op.get_kind() != op_kind::dnnl_mul_scales) {
            continue;
        }
        if (l_op.num_inputs() > 0 && l_op.get_input_value(0)->has_producer()
                && impl::utils::one_of(
                        l_op.get_input_value(0)->get_producer().get_kind(),
                        op_kind::dnnl_matmul, op_kind::dnnl_convolution,
                        op_kind::dnnl_convtranspose, op_kind::dnnl_reorder)) {
            mul_scale_op_offset = 1;
        } else if (r_op.num_inputs() > 0
                && r_op.get_input_value(0)->has_producer()
                && impl::utils::one_of(
                        r_op.get_input_value(0)->get_producer().get_kind(),
                        op_kind::dnnl_matmul, op_kind::dnnl_convolution,
                        op_kind::dnnl_convtranspose, op_kind::dnnl_reorder)) {
            mul_scale_op_offset = 0;
        }

        if (mul_scale_op_offset != 2
                && ltw(lhs_val->get_logical_tensor()).vdims()
                        == ltw(rhs_val->get_logical_tensor()).vdims()) {
            auto in_val = cur_op->get_input_value(mul_scale_op_offset);
            auto &mul_scale_op = in_val->get_producer();
            auto scales = mul_scale_op.get_attr<std::vector<float>>(
                    op_attr::scales);
            assert(scales.size() == 1); // per tensor

            auto tmp = mul_scale_op.get_input_value(0);
            auto &add_zps_op = tmp->get_producer();
            auto zps = add_zps_op.get_attr<std::vector<int64_t>>(op_attr::zps);
            assert(scales.size() == zps.size());

            auto out_val = cur_op->get_output_values()[0];
            auto consumers = out_val->get_consumers();
            auto &next_op = consumers[0].get_op();
            // set sum post-ops' second input scale
            float tmp_scale
                    = next_op.get_attr<std::vector<float>>(op_attr::scales)[0];
            scales[0] *= tmp_scale;
            mul_scale_op.set_attr<std::vector<float>>(op_attr::scales, scales);

            // update the output scales
            auto other_val = cur_op->get_input_value(1 - mul_scale_op_offset);
            auto &oscales_op = other_val->get_producer();
            auto oscales
                    = oscales_op.get_attr<std::vector<float>>(op_attr::scales);
            for (auto &v : oscales)
                v *= tmp_scale;
            oscales_op.set_attr<std::vector<float>>(op_attr::scales, oscales);
            rewriter.fuse_op_to_predecessor(next_op.shared_from_this());
        }
    }

    rewriter.run();
    return status::success;
}

status_t fuse_post_ops(std::shared_ptr<subgraph_t> &sg) {
    // lambda function to fuse one post op into base primitive
    auto fuse_post_ops_func = [&](bool &changed) -> impl::status_t {
        auto &mgr = sg->fusion_info_mgr_;
        subgraph_rewriter_t rewriter(sg);

        std::vector<std::pair<op_t *, op_t *>> fuse_groups;

        std::set<op_t *> visited;
        impl::status_t ret = impl::topo_order_visit(
                sg->get_output_ops(), [&](impl::op_t *op) {
                    const auto &pops_fusible_map = get_post_ops_fusible_map();

                    auto base_op_kind = op->get_kind();
                    // only fuse two ops each time
                    if (!pops_fusible_map.count(base_op_kind)
                            || visited.count(op) != 0 || !fuse_groups.empty())
                        return impl::status::success;

                    auto out_val = op->get_output_values()[0];
                    auto consumers = out_val->get_consumers();

                    // The base op should have and only have one consumer, it's
                    // the post op to be fused
                    if (consumers.size() != 1) return impl::status::success;
                    auto &post_op = consumers[0].get_op();

                    // check if fusible
                    // TODO(qun) make sure bn only fuse relu
                    auto post_op_kind = post_op.get_kind();
                    bool not_fusible = (!pops_fusible_map.at(base_op_kind)
                                                       .count(post_op_kind))
                            || (post_op_kind == op_kind::dnnl_binary
                                    && !post_binary_fusible(op, &post_op))
                            || (post_op_kind == op_kind::dnnl_convolution
                                    && !post_depthwise_conv_fusible(
                                            op, &post_op));
                    if (not_fusible) { return impl::status::success; }

                    // push fusible pair to fuse group for later fusion
                    fuse_groups.emplace_back(
                            std::pair<op_t *, op_t *> {op, &post_op});
                    visited.insert(op);
                    visited.insert(&post_op);
                    return impl::status::success;
                });

        if (ret != impl::status::success) return ret;

        if (fuse_groups.empty()) {
            changed = false;
            return impl::status::success;
        }

        for (auto &fuse_group : fuse_groups) {
            auto base_op = fuse_group.first;
            auto post_op = fuse_group.second;
            // post op fuse to which predecessor
            size_t fuse_op_predecessor_offset = base_op->get_output_value(0)
                                                        ->get_consumers()[0]
                                                        .get_offset();

            int64_t key = -1;
            if (base_op->has_attr(op_attr::fusion_info_key)
                    && base_op->get_attr<int64_t>(op_attr::fusion_info_key)
                            != -1) {
                key = base_op->get_attr<int64_t>(op_attr::fusion_info_key);
            } else {
                key = mgr.init_info();
                base_op->set_attr<int64_t>(op_attr::fusion_info_key, key);
            }

            fusion_info_t &fusion_info = mgr.get_mutable_info(key);

            bool with_op_replacement = false;

            if (post_op->get_kind() == op_kind::dnnl_eltwise) {
                float scale = 1.f;

                const auto alg = static_cast<dnnl::algorithm>(
                        post_op->get_attr<int64_t>(op_attr::alg_kind));

                // for BatchNormForwardTraining, set dnnl_fuse_norm_relu flag
                // instead of post op
                if ((base_op->get_kind() == op_kind::dnnl_batchnorm
                            && base_op->get_attr<bool>(op_attr::is_training))
                        && alg == dnnl::algorithm::eltwise_relu) {
                    base_op->set_attr<bool>(op_attr::fuse_relu, true);
                    // remove the fused post_ops op
                    rewriter.fuse_op_to_predecessor(post_op->shared_from_this(),
                            fuse_op_predecessor_offset);
                    auto tmp_ptr = base_op->shared_from_this();
                    insert_empty_workspace(tmp_ptr);
                    continue;
                }

                auto out_val = post_op->get_output_values()[0];
                auto consumers = out_val->get_consumers();
                if (!consumers.empty()) {
                    auto &next_op = consumers[0].get_op();
                    // set eltwise post-ops scale
                    if (next_op.get_kind() == op_kind::dnnl_mul_scales) {
                        scale = next_op.get_attr<std::vector<float>>(
                                op_attr::scales)[0];
                        rewriter.fuse_op_to_predecessor(
                                next_op.shared_from_this());
                    }
                }
                fusion_info.append_post_eltwise(
                        post_op->shared_from_this(), scale);
            } else if (post_op->get_kind() == op_kind::dnnl_binary
                    && static_cast<dnnl::algorithm>(
                               post_op->get_attr<int64_t>(op_attr::alg_kind))
                            == dnnl::algorithm::binary_add) {
                // If the other in value of Add has mul_scales producer,
                // then this pattern is a int8 pattern
                size_t mul_scale_op_offset = 2;
                auto other_in_val0 = post_op->get_input_value(
                        1 - fuse_op_predecessor_offset);

                if (other_in_val0->has_producer()
                        && other_in_val0->get_producer().get_kind()
                                == op_kind::dnnl_mul_scales) {
                    mul_scale_op_offset = 1 - fuse_op_predecessor_offset;
                }

                auto other_in_val1
                        = post_op->get_input_value(fuse_op_predecessor_offset);

                if (mul_scale_op_offset != 2
                        && is_output_scales_supported(base_op->get_kind())
                        && ltw(other_in_val0->get_logical_tensor()).vdims()
                                == ltw(other_in_val1->get_logical_tensor())
                                           .vdims()) {
                    // for int8 cases (excluding OPs which don't support
                    // output scales attribute and its inputs don't have
                    // same dims)
                    auto in_val = post_op->get_input_value(mul_scale_op_offset);
                    auto &mul_scale_op = in_val->get_producer();
                    auto scales = mul_scale_op.get_attr<std::vector<float>>(
                            op_attr::scales);
                    assert(scales.size() == 1); // per tensor

                    auto tmp = mul_scale_op.get_input_value(0);
                    int32_t zp = 0;
                    if (tmp->has_producer()
                            && tmp->get_producer().get_kind()
                                    == op_kind::dnnl_sub_zps) {
                        auto &sub_zps_op = tmp->get_producer();
                        auto zps = sub_zps_op.get_attr<std::vector<int64_t>>(
                                op_attr::zps);
                        zp = static_cast<int32_t>(zps[0]);
                        assert(scales.size() == zps.size());

                        rewriter.fuse_op_to_successor(
                                sub_zps_op.shared_from_this());
                    }
                    rewriter.fuse_op_to_successor(
                            mul_scale_op.shared_from_this());

                    fusion_info.append_post_sum(post_op->shared_from_this(),
                            std::vector<size_t> {base_op->num_inputs()},
                            scales[0], zp);
                    assertm(!base_op->has_attr(op_attr::with_sum)
                                    || !base_op->get_attr<bool>(
                                            op_attr::with_sum),
                            "not support multiple post sum ops "
                            "currently.");
                    base_op->set_attr<bool>(op_attr::with_sum, true);
                } else {
                    // - the add operation may need broadcast
                    auto fused_in = post_op->get_input_value(
                            fuse_op_predecessor_offset);
                    auto other_in = post_op->get_input_value(
                            1 - fuse_op_predecessor_offset);
                    auto dst = post_op->get_output_value(0);

                    if (ltw(fused_in->get_logical_tensor()).vdims()
                            == ltw(other_in->get_logical_tensor()).vdims()) {
                        if (base_op->get_kind() == op_kind::dnnl_eltwise
                                || base_op->get_kind() == op_kind::dnnl_pool) {
                            fusion_info.append_post_binary(
                                    post_op->shared_from_this(),
                                    std::vector<size_t> {
                                            base_op->num_inputs()});
                        } else {
                            // use sum post-ops for no-broadcast add
                            // map non-first post-sum to post-binary_add
                            if (base_op->has_attr(op_attr::with_sum)
                                    && base_op->get_attr<bool>(
                                            op_attr::with_sum)) {
                                fusion_info.append_post_binary(
                                        post_op->shared_from_this(),
                                        std::vector<size_t> {
                                                base_op->num_inputs()});
                            } else {
                                fusion_info.append_post_sum(
                                        post_op->shared_from_this(),
                                        std::vector<size_t> {
                                                base_op->num_inputs()},
                                        1.0f, 0);
                                base_op->set_attr<bool>(
                                        op_attr::with_sum, true);
                            }
                        }
                    } else {
                        // use binary post-ops for broadcast add
                        fusion_info.append_post_binary(
                                post_op->shared_from_this(),
                                std::vector<size_t> {base_op->num_inputs()});
                    }
                }
            } else if (post_op->get_kind() == op_kind::dnnl_binary
                    && static_cast<dnnl::algorithm>(
                               post_op->get_attr<int64_t>(op_attr::alg_kind))
                            != dnnl::algorithm::binary_add) {
                fusion_info.append_post_binary(post_op->shared_from_this(),
                        std::vector<size_t> {base_op->num_inputs()});
            } else if (post_op->get_kind() == op_kind::dnnl_convolution) {
                const std::string dw_type
                        = (post_op->get_attr<std::vector<int64_t>>(
                                   op_attr::strides)[0]
                                  == 1)
                        ? "k3s1p1"
                        : "k3s2p1";

                fusion_info.append_post_dw_conv(post_op->shared_from_this(),
                        std::vector<size_t> {base_op->num_inputs()});

                op_ptr conv_dw
                        = std::make_shared<op_t>(op_kind::dnnl_conv_depthwise);
                const auto dw_groups
                        = post_op->get_attr<int64_t>(op_attr::groups);
                const auto dw_filter_format = post_op->get_attr<std::string>(
                        op_attr::filter_format);
                conv_dw->set_attr(op_attr::dw_groups, dw_groups);
                conv_dw->set_attr(op_attr::dw_filter_format, dw_filter_format);
                conv_dw->set_attr(op_attr::dw_type, dw_type);

                auto base_op_shared_ptr = base_op->shared_from_this();
                rewriter.replace_op(base_op_shared_ptr, conv_dw);
                conv_dw->merge_attributes(base_op->get_attributes());

                rewriter.fuse_op_to_predecessor(post_op->shared_from_this(),
                        fuse_op_predecessor_offset);
                // conv_depthwise op may have additional output, which is
                // intermediate base conv output. It is needed in layout
                // propagation step.
                auto base_out_lt
                        = base_op->get_output_value(0)->get_logical_tensor();
                op_t &conv_dw_ref = *conv_dw.get();
                auto intermediate_out = std::make_shared<value_t>(
                        conv_dw_ref, conv_dw->num_outputs(), base_out_lt, true);
                conv_dw->connect_output(
                        conv_dw->num_outputs(), intermediate_out);

                with_op_replacement = true;
            } else {
                // unsupported post ops
                continue;
            }

            if (!with_op_replacement) {
                // remove the fused post_ops op
                rewriter.fuse_op_to_predecessor(post_op->shared_from_this(),
                        fuse_op_predecessor_offset);
            }
        }

        rewriter.run();
        changed = true;
        return impl::status::success;
    };

    int cnt = 0;
    const int max_num_limit = static_cast<int>(sg->num_ops());

    bool changed = true;
    do {
        auto ret = fuse_post_ops_func(changed);
        if (ret != impl::status::success) return ret;
        cnt++;
    } while (changed && cnt <= max_num_limit);

    assertm(cnt <= max_num_limit + 1,
            "Failed to fuse all post ops since there has unsupported ones.");
    if (cnt > max_num_limit + 1) return impl::status::unimplemented;
    return status::success;
}

impl::status_t fuse_zero_points(std::shared_ptr<subgraph_t> &sg) {
    auto &mgr = sg->fusion_info_mgr_;

    std::vector<op_t *> zp_ops;

    std::set<op_t *> visited;
    for (auto &cur_op : sg->get_ops()) {
        if ((cur_op->get_kind() != op_kind::dnnl_add_zps
                    && cur_op->get_kind() != op_kind::dnnl_sub_zps)
                || visited.count(cur_op.get()) != 0)
            continue;

        zp_ops.emplace_back(cur_op.get());
        visited.insert(cur_op.get());
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &zp_op : zp_ops) {
        assertm(zp_op->num_outputs() == 1,
                "zp_op should have only one output value.");
        auto out_val = zp_op->get_output_values()[0];
        auto consumers = out_val->get_consumers();

        if (zp_op->get_kind() == op_kind::dnnl_sub_zps) {
            if (!has_int8_support(consumers[0].get_op().get_kind())) continue;

            auto &next_op = consumers[0].get_op();
            auto offset = consumers[0].get_offset();
            if (offset == 0 || offset == 1) {
                int64_t key = -1;
                if (next_op.has_attr(op_attr::fusion_info_key)) {
                    key = next_op.get_attr<int64_t>(op_attr::fusion_info_key);
                } else {
                    key = mgr.init_info();
                    next_op.set_attr<int64_t>(op_attr::fusion_info_key, key);
                }

                fusion_info_t &fusion_info = mgr.get_mutable_info(key);

                auto zps = zp_op->get_attr<std::vector<int64_t>>(op_attr::zps);
                if (!utils::all_zero(zps)) {
                    assertm(zps.size() == 1,
                            "zp attr only support scalar zp, need to use "
                            "runtime arg to support vector zp");
                    fusion_info.set_zero_points(
                            zp_op->shared_from_this(), true, offset);
                }
            }

            rewriter.fuse_op_to_successor(zp_op->shared_from_this());
        } else {
            auto in_val = zp_op->get_input_values()[0];
            auto &prv_op = in_val->get_producer();

            if (!has_int8_support(prv_op.get_kind())) continue;

            int64_t key = -1;
            if (prv_op.has_attr(op_attr::fusion_info_key)) {
                key = prv_op.get_attr<int64_t>(op_attr::fusion_info_key);
            } else {
                key = mgr.init_info();
                prv_op.set_attr<int64_t>(op_attr::fusion_info_key, key);
            }

            fusion_info_t &fusion_info = mgr.get_mutable_info(key);
            fusion_info.set_zero_points(zp_op->shared_from_this(), false, 0);
            rewriter.fuse_op_to_predecessor(zp_op->shared_from_this());
        }
    }
    return impl::status::success;
}

impl::status_t insert_bn_folding(std::shared_ptr<subgraph_t> &sg) {
    std::vector<op_t *> bn_ops;

    std::set<op_t *> visited;
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_batchnorm
                || visited.count(cur_op.get()) != 0)
            continue;

        // only fold bn inference
        if (cur_op->get_attr<bool>(op_attr::is_training)) continue;

        // only folding conv_bn case
        auto in = cur_op->get_input_value(0);
        if (!in->has_producer()
                || in->get_producer().get_kind() != op_kind::dnnl_convolution)
            continue;

        bn_ops.emplace_back(cur_op.get());
        visited.insert(cur_op.get());
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &bn_op : bn_ops) {
        auto &prv_op = bn_op->get_input_value(0)->get_producer();
        if (prv_op.get_kind() != op_kind::dnnl_convolution) continue;

        op_ptr bn_folding_op = std::make_shared<op_t>(op_kind::dnnl_bn_folding);

        bn_folding_op->set_attr<float>(
                op_attr::epsilon, bn_op->get_attr<float>(op_attr::epsilon));
        bn_folding_op->set_attr<std::string>(op_attr::data_format,
                bn_op->get_attr<std::string>(op_attr::data_format));

        bn_folding_op->set_attr<std::string>(op_attr::filter_format,
                prv_op.get_attr<std::string>(op_attr::filter_format));
        bn_folding_op->set_attr<bool>(
                op_attr::with_bias, prv_op.num_inputs() == 3);

        // add conv's weight and bias (if exist) to bn folder op
        for (size_t i = 1; i < prv_op.num_inputs(); i++) {
            auto tmp = prv_op.get_input_value(i);
            tmp->remove_consumer(prv_op, i);
            tmp->add_consumer(*bn_folding_op, bn_folding_op->num_inputs());
            bn_folding_op->add_input(tmp);
        }

        // add bn's scale, shift, mean and variance to bn folder op
        for (size_t i = 1; i < bn_op->num_inputs(); i++) {
            auto tmp = bn_op->get_input_value(i);
            tmp->remove_consumer(*bn_op, i);
            tmp->add_consumer(*bn_folding_op, bn_folding_op->num_inputs());
            bn_folding_op->add_input(tmp);
        }

        auto updated_conv_wei = std::make_shared<value_t>(*bn_folding_op, 0,
                impl::empty_logical_tensor_with_default_id(), true);
        updated_conv_wei->set_data_type(
                prv_op.get_input_value(1)->get_logical_tensor().data_type);
        bn_folding_op->add_output(updated_conv_wei);
        updated_conv_wei->add_consumer(prv_op, 1);
        prv_op.connect_input(1, updated_conv_wei);

        auto updated_conv_bias = std::make_shared<value_t>(*bn_folding_op, 1,
                impl::empty_logical_tensor_with_default_id(), true);
        updated_conv_bias->set_data_type(
                prv_op.get_input_value(1)->get_logical_tensor().data_type);
        bn_folding_op->add_output(updated_conv_bias);
        updated_conv_bias->add_consumer(prv_op, 2);
        prv_op.connect_input(2, updated_conv_bias);

        // add scratchpad output for bn_folding
        insert_empty_scratchpad(bn_folding_op);

        auto bn_out_val = bn_op->get_output_value(0);
        prv_op.connect_output(0, bn_out_val);

        rewriter.to_remove(bn_op->shared_from_this());
        rewriter.to_insert(bn_folding_op);
    }

    rewriter.run();
    return infer_shape(sg);
}

impl::status_t conv_bwd_data_canonicalization(std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_conv_bwd_data) continue;

        // insert permute
        bool need_permute_0 = cur_op->has_attr(op_attr::data_format)
                ? (cur_op->get_attr<std::string>(op_attr::data_format) == "NXC")
                : false;
        bool need_permute_1 = cur_op->has_attr(op_attr::filter_format)
                ? (cur_op->get_attr<std::string>(op_attr::filter_format)
                        == "XIO")
                : false;

        if (need_permute_0) {
            // input permute
            op_ptr in_perm_op
                    = std::make_shared<impl::op_t>(op_kind::dnnl_permute);
            in_perm_op->set_attr<std::string>(op_attr::permute_kind, "permute");
            in_perm_op->set_attr<std::string>(op_attr::from_format, "NXC");
            in_perm_op->set_attr<std::string>(op_attr::to_format, "NCX");
            rewriter.insert_op_before(in_perm_op, cur_op, 0);

            // output permute
            op_ptr out_perm_op
                    = std::make_shared<impl::op_t>(op_kind::dnnl_permute);
            out_perm_op->set_attr<std::string>(
                    op_attr::permute_kind, "permute");
            out_perm_op->set_attr<std::string>(op_attr::from_format, "NCX");
            out_perm_op->set_attr<std::string>(op_attr::to_format, "NXC");
            rewriter.insert_op_after(out_perm_op, cur_op, 0);

            cur_op->set_attr<std::string>(op_attr::data_format, "NCX");
            if (cur_op->has_attr(op_attr::output_shape)) {
                auto nxc_dst_shape
                        = cur_op->get_attr<impl::dims>(op_attr::output_shape);
                auto ncx_dst_shape = impl::canonicalize(nxc_dst_shape, "NXC");
                cur_op->set_attr<impl::dims>(
                        op_attr::output_shape, ncx_dst_shape);
            }
        }

        if (need_permute_1) {
            op_ptr perm_op
                    = std::make_shared<impl::op_t>(op_kind::dnnl_permute);
            perm_op->set_attr<std::string>(op_attr::permute_kind, "permute");
            perm_op->set_attr<std::string>(op_attr::from_format, "XIO");
            perm_op->set_attr<std::string>(op_attr::to_format, "OIX");
            rewriter.insert_op_before(perm_op, cur_op, 1);
            cur_op->set_attr<std::string>(op_attr::filter_format, "OIX");
        }

        // insert to_group
        auto groups = cur_op->get_attr<int64_t>(op_attr::groups);
        if (groups > 1) {
            op_ptr to_group_op
                    = std::make_shared<impl::op_t>(op_kind::dnnl_to_group);
            to_group_op->set_attr<int64_t>(op_attr::groups, groups);
            rewriter.insert_op_before(to_group_op, cur_op, 1);
            cur_op->set_attr<int64_t>(op_attr::groups, 1);
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

impl::status_t conv_bwd_weights_canonicalization(
        std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_conv_bwd_weights
                && cur_op->get_kind()
                        != op_kind::dnnl_convtranspose_bwd_weights)
            continue;

        const auto filter_shape_attr
                = cur_op->get_attr<std::vector<int64_t>>(op_attr::filter_shape);
        const bool is_filter_shape_default = std::all_of(
                filter_shape_attr.begin(), filter_shape_attr.end(),
                [](int64_t d) { return d == 0; });
        if (is_filter_shape_default) {
            const std::vector<int64_t> filter_shape
                    = ltw(cur_op->get_output_value(0)->get_logical_tensor())
                              .vdims();
            cur_op->set_attr(op_attr::filter_shape, filter_shape);
        }

        // insert permute
        bool need_permute_0 = cur_op->has_attr(op_attr::data_format)
                ? (cur_op->get_attr<std::string>(op_attr::data_format) == "NXC")
                : false;
        bool need_permute_1 = cur_op->has_attr(op_attr::filter_format)
                ? (cur_op->get_attr<std::string>(op_attr::filter_format)
                        == "XIO")
                : false;

        if (need_permute_0) {
            // input permute
            op_ptr in0_perm_op
                    = std::make_shared<impl::op_t>(op_kind::dnnl_permute);
            in0_perm_op->set_attr<std::string>(
                    op_attr::permute_kind, "permute");
            in0_perm_op->set_attr<std::string>(op_attr::from_format, "NXC");
            in0_perm_op->set_attr<std::string>(op_attr::to_format, "NCX");
            rewriter.insert_op_before(in0_perm_op, cur_op, 0);

            op_ptr in1_perm_op
                    = std::make_shared<impl::op_t>(op_kind::dnnl_permute);
            in1_perm_op->set_attr<std::string>(
                    op_attr::permute_kind, "permute");
            in1_perm_op->set_attr<std::string>(op_attr::from_format, "NXC");
            in1_perm_op->set_attr<std::string>(op_attr::to_format, "NCX");
            rewriter.insert_op_before(in1_perm_op, cur_op, 1);

            cur_op->set_attr<std::string>(op_attr::data_format, "NCX");
        }
        // output permute
        if (need_permute_1) {
            op_ptr out_perm_op
                    = std::make_shared<impl::op_t>(op_kind::dnnl_permute);
            out_perm_op->set_attr<std::string>(
                    op_attr::permute_kind, "permute");
            out_perm_op->set_attr<std::string>(op_attr::from_format, "OIX");
            out_perm_op->set_attr<std::string>(op_attr::to_format, "XIO");
            rewriter.insert_op_after(out_perm_op, cur_op, 0);

            const auto filter_shape_attr
                    = cur_op->get_attr<std::vector<int64_t>>(
                            op_attr::filter_shape);
            const auto filter_shape_as_oix
                    = impl::canonicalize(filter_shape_attr, "XIO");
            cur_op->set_attr<impl::dims>(
                    op_attr::filter_shape, filter_shape_as_oix);
            cur_op->set_attr<std::string>(op_attr::filter_format, "OIX");
        }

        // insert from_group
        auto groups = cur_op->get_attr<int64_t>(op_attr::groups);
        if (groups > 1) {
            op_ptr from_group_op
                    = std::make_shared<impl::op_t>(op_kind::dnnl_from_group);
            from_group_op->set_attr<int64_t>(op_attr::groups, groups);
            rewriter.insert_op_after(from_group_op, cur_op, 0);

            if (cur_op->get_kind() == op_kind::dnnl_convtranspose_bwd_weights)
                from_group_op->set_attr<bool>(op_attr::is_convtranspose, true);
        }

        cur_op->set_attr<bool>(op_attr::canonicalized, true);
    }

    rewriter.run();
    return infer_shape(sg);
}

impl::status_t pool_fwd_canonicalization(std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_pool) continue;

        // insert permute
        bool need_permute = cur_op->has_attr(op_attr::data_format)
                ? (cur_op->get_attr<std::string>(op_attr::data_format) == "NXC")
                : false;

        if (need_permute) {
            // src permute
            op_ptr in0_perm_op
                    = std::make_shared<impl::op_t>(op_kind::dnnl_permute);
            in0_perm_op->set_attr<std::string>(
                    op_attr::permute_kind, "permute");
            in0_perm_op->set_attr<std::string>(op_attr::from_format, "NXC");
            in0_perm_op->set_attr<std::string>(op_attr::to_format, "NCX");
            rewriter.insert_op_before(in0_perm_op, cur_op, 0);

            // dst permute
            op_ptr out0_perm_op
                    = std::make_shared<impl::op_t>(op_kind::dnnl_permute);
            out0_perm_op->set_attr<std::string>(
                    op_attr::permute_kind, "permute");
            out0_perm_op->set_attr<std::string>(op_attr::from_format, "NCX");
            out0_perm_op->set_attr<std::string>(op_attr::to_format, "NXC");
            rewriter.insert_op_after(out0_perm_op, cur_op, 0);

            cur_op->set_attr<std::string>(op_attr::data_format, "NCX");
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

impl::status_t pool_bwd_canonicalization(std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_pool_bwd) continue;

        // insert permute
        bool need_permute = cur_op->has_attr(op_attr::data_format)
                ? (cur_op->get_attr<std::string>(op_attr::data_format) == "NXC")
                : false;

        if (need_permute) {
            // diff_dst permute
            op_ptr in1_perm_op
                    = std::make_shared<impl::op_t>(op_kind::dnnl_permute);
            in1_perm_op->set_attr<std::string>(
                    op_attr::permute_kind, "permute");
            in1_perm_op->set_attr<std::string>(op_attr::from_format, "NXC");
            in1_perm_op->set_attr<std::string>(op_attr::to_format, "NCX");
            rewriter.insert_op_before(in1_perm_op, cur_op, 0);

            // src permute
            if (cur_op->get_attr<std::string>(op_attr::kind) == "maxpool") {
                op_ptr src_perm_op
                        = std::make_shared<impl::op_t>(op_kind::dnnl_permute);
                src_perm_op->set_attr<std::string>(
                        op_attr::permute_kind, "permute");
                src_perm_op->set_attr<std::string>(op_attr::from_format, "NXC");
                src_perm_op->set_attr<std::string>(op_attr::to_format, "NCX");
                rewriter.insert_op_before(src_perm_op, cur_op, 2);
            }

            // diff_src permute
            op_ptr out_perm_op
                    = std::make_shared<impl::op_t>(op_kind::dnnl_permute);
            out_perm_op->set_attr<std::string>(
                    op_attr::permute_kind, "permute");
            out_perm_op->set_attr<std::string>(op_attr::from_format, "NCX");
            out_perm_op->set_attr<std::string>(op_attr::to_format, "NXC");
            rewriter.insert_op_after(out_perm_op, cur_op, 0);

            cur_op->set_attr<std::string>(op_attr::data_format, "NCX");

            if (cur_op->has_attr(op_attr::input_shape)) {
                auto nxc_dst_shape
                        = cur_op->get_attr<impl::dims>(op_attr::input_shape);
                auto ncx_dst_shape = impl::canonicalize(nxc_dst_shape, "NXC");
                cur_op->set_attr<impl::dims>(
                        op_attr::input_shape, ncx_dst_shape);
            }
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

impl::status_t fuse_mul_sigmoid_to_swish(std::shared_ptr<subgraph_t> &sg) {
    std::vector<std::vector<op_t *>> swish_patterns;
    std::vector<size_t> mul_other_offsets;

    // find all swish pattern in subgraph
    std::set<op_t *> visited;
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_eltwise
                || visited.count(cur_op.get()) != 0)
            continue;

        if (static_cast<dnnl::algorithm>(
                    cur_op->get_attr<int64_t>(op_attr::alg_kind))
                != dnnl::algorithm::eltwise_logistic)
            continue;

        visited.insert(cur_op.get());

        /* check if the sigmoid op belongs to a swish pattern.
        // A swish pattern is composed by a sigmoid op and a multiply op:
        //        any
        //       /   \
        //  sigmoid   |
        //       \   /
        //      multiply
        //         |
        //        any
        */
        auto sigmoid_out = cur_op->get_output_value(0);
        auto sigmoid_csm = sigmoid_out->get_consumers();
        if (sigmoid_csm.size() != 1) continue;

        auto &csm_op = sigmoid_csm[0].get_op();
        if (csm_op.get_kind() != op_kind::dnnl_binary) continue;

        if (static_cast<dnnl::algorithm>(
                    csm_op.get_attr<int64_t>(op_attr::alg_kind))
                != dnnl::algorithm::binary_mul)
            continue;

        size_t offset = sigmoid_csm[0].get_offset(); // offset should be 0 or 1
        size_t mul_other_offset = 1 - offset;
        auto mul_other_in = csm_op.get_input_value(mul_other_offset);
        auto sigmoid_in = cur_op->get_input_value(0);
        if (mul_other_in.get() != sigmoid_in.get()) continue;

        // all checks passed, found a swish pattern
        std::vector<op_t *> pattern {cur_op.get(), &csm_op};
        swish_patterns.emplace_back(pattern);
        mul_other_offsets.emplace_back(mul_other_offset);
    }

    if (swish_patterns.empty()) return impl::status::success;

    // fuse swish pattern to a swish op
    subgraph_rewriter_t rewriter(sg);
    for (size_t i = 0; i < swish_patterns.size(); i++) {
        op_t *sigmoid_op = swish_patterns[i][0];
        op_t *mul_op = swish_patterns[i][1];
        size_t mul_other_offset = mul_other_offsets[i];

        op_ptr swish_op = std::make_shared<op_t>(op_kind::dnnl_eltwise);
        swish_op->set_attr<int64_t>(op_attr::alg_kind,
                static_cast<int64_t>(dnnl::algorithm::eltwise_swish));
        swish_op->set_attr<float>(op_attr::alpha, (float)1.0);

        auto in_val = sigmoid_op->get_input_value(0);
        in_val->remove_consumer(*sigmoid_op, 0);
        in_val->remove_consumer(*mul_op, mul_other_offset);
        swish_op->connect_input(0, in_val);

        auto out_val = mul_op->get_output_value(0);
        swish_op->add_output(out_val);
        out_val->set_producer(*swish_op);

        insert_empty_scratchpad(swish_op);

        rewriter.to_insert(swish_op);
        rewriter.to_remove(sigmoid_op->shared_from_this());
        rewriter.to_remove(mul_op->shared_from_this());
    }

    rewriter.run();
    return impl::status::success;
}

impl::status_t fuse_typecast_to_matmul_or_conv(
        std::shared_ptr<subgraph_t> &sg) {
    std::vector<std::vector<op_t *>> fusion_groups;
    for (const auto &cur_op : sg->get_ops()) {
        if ((cur_op->get_kind() != op_kind::dnnl_matmul
                    && cur_op->get_kind() != op_kind::dnnl_convolution)
                || !cur_op->get_input_value(0)->has_producer()
                || !cur_op->get_input_value(1)->has_producer())
            continue;
        auto &in0 = cur_op->get_input_value(0)->get_producer();
        auto &in1 = cur_op->get_input_value(1)->get_producer();
        if (is_typecast(&in0) && is_typecast(&in1)
                && in0.get_input_value(0)->has_producer()
                && in1.get_input_value(0)->has_producer()
                && in0.get_input_value(0)->get_producer().get_kind()
                        == dnnl_impl::op_kind::dnnl_mul_scales
                && in1.get_input_value(0)->get_producer().get_kind()
                        == dnnl_impl::op_kind::dnnl_mul_scales)
            fusion_groups.emplace_back(
                    std::vector<op_t *> {cur_op.get(), &in0, &in1});
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &fusion_group : fusion_groups) {
        op_t *base_op = fusion_group[0];
        op_t *in0 = fusion_group[1];
        op_t *in1 = fusion_group[2];

        // update the connection relationship between matmul/convolution and
        // typecast ops
        auto in0_value = in0->get_input_value(0);
        auto in1_value = in1->get_input_value(0);
        base_op->connect_input(0, in0_value);
        base_op->connect_input(1, in1_value);
        in0_value->remove_consumer(*in0, 0);
        in1_value->remove_consumer(*in1, 0);

        rewriter.to_remove(in0->shared_from_this());
        rewriter.to_remove(in1->shared_from_this());
    }

    rewriter.run();
    return impl::status::success;
}

impl::status_t fuse_typecast_to_add(std::shared_ptr<subgraph_t> &sg) {
    std::vector<std::vector<op_t *>> fusion_groups;
    for (const auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_binary
                || static_cast<dnnl::algorithm>(
                           cur_op->get_attr<int64_t>(op_attr::alg_kind))
                        != dnnl::algorithm::binary_add)
            continue;
        if (!(cur_op->get_input_value(0)->has_producer()
                    && cur_op->get_input_value(1)->has_producer()))
            continue;

        auto &in0 = cur_op->get_input_value(0)->get_producer();
        auto &in1 = cur_op->get_input_value(1)->get_producer();
        if (is_typecast(&in0)
                && (in1.get_kind() == op_kind::dnnl_matmul
                        || in1.get_kind() == op_kind::dnnl_convolution)) {
            fusion_groups.emplace_back(
                    std::vector<op_t *> {cur_op.get(), &in0});
        } else if (is_typecast(&in1)
                && (in0.get_kind() == op_kind::dnnl_matmul
                        || in0.get_kind() == op_kind::dnnl_convolution)) {
            fusion_groups.emplace_back(
                    std::vector<op_t *> {cur_op.get(), &in1});
        } else {
        }
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &fusion_group : fusion_groups) {
        op_t *add_op = fusion_group[0];
        op_t *typecast_op = fusion_group[1];

        op_ptr new_add_op = std::make_shared<op_t>(op_kind::dnnl_binary);
        new_add_op->merge_attributes(add_op->get_attributes());

        // update the connection relationship between add and typecast ops
        auto tc_in = typecast_op->get_input_value(0);
        auto in0 = add_op->get_input_value(0);
        auto in1 = add_op->get_input_value(1);
        in0->remove_consumer(*add_op, 0);
        in1->remove_consumer(*add_op, 1);
        if (is_typecast(&in0->get_producer())) {
            new_add_op->connect_input(0, tc_in);
            new_add_op->connect_input(1, in1);
            tc_in->remove_consumer(*typecast_op, 0);
        } else {
            new_add_op->connect_input(1, tc_in);
            new_add_op->connect_input(0, in0);
            tc_in->remove_consumer(*typecast_op, 0);
        }

        auto out_val = add_op->get_output_value(0);
        new_add_op->add_output(out_val);
        out_val->set_producer(*new_add_op);

        auto scratchpad_val = add_op->get_output_value(1);
        new_add_op->connect_output(1, scratchpad_val);

        // delete original matmul/convolution and typecast ops
        for (auto &del_op : fusion_group) {
            rewriter.to_remove(del_op->shared_from_this());
        }

        rewriter.to_insert(new_add_op);
    }

    rewriter.run();
    return impl::status::success;
}

impl::status_t fuse_post_typecast_to_matmul_or_conv(
        std::shared_ptr<subgraph_t> &sg) {
    const std::set<op_kind_t> post_ops_kinds {impl::op_kind::GELU};
    std::vector<std::vector<op_t *>> fusion_groups;
    for (const auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_matmul
                && cur_op->get_kind() != op_kind::dnnl_convolution)
            continue;
        auto out = cur_op->get_output_value(0);
        if (out->get_consumers().size() != 1) continue;
        auto &next_op = out->get_consumers()[0].get_op();
        if (post_ops_kinds.count(next_op.get_kind())) {
            auto post_out = next_op.get_output_value(0);
            if (post_out->get_consumers().size() != 1) continue;
            auto &tc_op = post_out->get_consumers()[0].get_op();
            if (!is_typecast(&tc_op)) continue;
            auto tc_out = tc_op.get_output_value(0);
            if (tc_out->get_consumers().size() != 1) continue;
            auto &q_op = tc_out->get_consumers()[0].get_op();
            if (q_op.get_kind() != dnnl_impl::op_kind::dnnl_mul_scales)
                continue;
            post_out->remove_consumer(tc_op, 0);
            tc_out->remove_consumer(q_op, 0);
            q_op.connect_input(0, post_out);
            out->set_data_type(impl::data_type::f32);
            post_out->set_data_type(impl::data_type::f32);
            fusion_groups.emplace_back(std::vector<op_t *> {&tc_op});
        } else {
            if (!is_typecast(&next_op)) continue;
            auto tc_out = next_op.get_output_value(0);
            if (tc_out->get_consumers().size() > 1) continue;
            if (tc_out->get_consumers().size() == 1) {
                // bf16-int8 mix precision case
                auto &q_op = tc_out->get_consumers()[0].get_op();
                if (q_op.get_kind() != dnnl_impl::op_kind::dnnl_mul_scales)
                    continue;
                out->remove_consumer(next_op, 0);
                tc_out->remove_consumer(q_op, 0);
                q_op.connect_input(0, out);
                out->set_data_type(impl::data_type::f32);
                fusion_groups.emplace_back(std::vector<op_t *> {&next_op});
            } else {
                // tc has no consumer in the subgraph
                // which means the fp32-in-bf16 out case
                cur_op->connect_output(0, tc_out);
                fusion_groups.emplace_back(std::vector<op_t *> {&next_op});
            }
        }
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &fusion_group : fusion_groups)
        // delete original typecast ops
        for (auto &del_op : fusion_group) {
            rewriter.to_remove(del_op->shared_from_this());
        }
    rewriter.run();
    return impl::status::success;
}

impl::status_t fuse_post_typecast_to_softmax(std::shared_ptr<subgraph_t> &sg) {
    std::vector<std::vector<op_t *>> fusion_groups;
    for (const auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_softmax) continue;
        auto out = cur_op->get_output_value(0);
        if (out->get_consumers().size() != 1) continue;
        auto &next_op = out->get_consumers()[0].get_op();
        if (!is_typecast(&next_op)) continue;
        auto tc_out = next_op.get_output_value(0);
        cur_op->connect_output(0, tc_out);
        fusion_groups.emplace_back(std::vector<op_t *> {&next_op});
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &fusion_group : fusion_groups)
        // delete original typecast ops
        for (auto &del_op : fusion_group) {
            rewriter.to_remove(del_op->shared_from_this());
        }
    rewriter.run();
    return impl::status::success;
}

impl::status_t fuse_reciprocal_mul_to_div(std::shared_ptr<subgraph_t> &sg) {
    /* transformation below graphs
    Case 1:
        in0       in1
         \         |               in0    in1
          \     reciprocal          \      /
           \       /        --->      div
              mul                      |
               |                     out0
              out0

    Case 2:
        in0       in1
         \         |               in1    in0
      reciprocal   |                \      /
           \       /        --->      div
              mul                      |
               |                      out0
              out0
    */

    std::vector<std::pair<op_t *, op_t *>> div_patterns;
    std::vector<size_t> mul_other_offsets;
    std::set<op_t *> visited;
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_eltwise
                || visited.count(cur_op.get()) != 0)
            continue;

        auto is_reciprocal = [&cur_op]() -> bool {
            bool ok = static_cast<dnnl::algorithm>(
                              cur_op->get_attr<int64_t>(op_attr::alg_kind))
                    == dnnl::algorithm::eltwise_pow;
            if (!ok) return false;

            // check attribute alpha
            float alpha = 0.f;
            if (cur_op->has_attr(op_attr::alpha))
                alpha = cur_op->get_attr<float>(op_attr::alpha);
            if (!utils::compare_float(alpha, 1.f)) return false;

            // check attribute beta
            float beta = 0.f;
            if (cur_op->has_attr(op_attr::beta))
                beta = cur_op->get_attr<float>(op_attr::beta);
            if (!utils::compare_float(beta, -1.f)) return false;
            return true;
        };

        if (!is_reciprocal()) continue;

        visited.insert(cur_op.get());

        auto reciprocal_out = cur_op->get_output_value(0);
        auto reciprocal_csm = reciprocal_out->get_consumers();
        if (reciprocal_csm.size() != 1) continue;

        auto &csm_op = reciprocal_csm[0].get_op();
        if (csm_op.get_kind() != op_kind::dnnl_binary
                || static_cast<dnnl::algorithm>(
                           csm_op.get_attr<int64_t>(op_attr::alg_kind))
                        != dnnl::algorithm::binary_mul)
            continue;

        // offset should be 0 or 1
        size_t offset = reciprocal_csm[0].get_offset();
        size_t mul_other_offset = 1 - offset;
        mul_other_offsets.emplace_back(mul_other_offset);

        div_patterns.emplace_back(
                std::pair<op_t *, op_t *> {cur_op.get(), &csm_op});
    }

    if (div_patterns.empty()) return impl::status::success;

    subgraph_rewriter_t rewriter(sg);
    for (size_t i = 0; i < div_patterns.size(); ++i) {
        auto reciprocal_op = div_patterns[i].first;
        auto mul_op = div_patterns[i].second;
        auto mul_other_offset = mul_other_offsets[i];

        op_ptr div_op = std::make_shared<op_t>(op_kind::dnnl_binary);
        div_op->set_attr<int64_t>(op_attr::alg_kind,
                static_cast<int64_t>(dnnl::algorithm::binary_div));

        auto mul_other_in_val = mul_op->get_input_value(mul_other_offset);
        mul_other_in_val->remove_consumer(*mul_op, mul_other_offset);
        div_op->connect_input(0, mul_other_in_val);

        auto reciprocal_in_val = reciprocal_op->get_input_value(0);
        reciprocal_in_val->remove_consumer(*reciprocal_op, 0);
        div_op->connect_input(1, reciprocal_in_val);

        auto mul_out_val = mul_op->get_output_value(0);
        div_op->add_output(mul_out_val);
        mul_out_val->set_producer(*div_op);

        insert_empty_scratchpad(div_op);

        rewriter.to_insert(div_op);
        rewriter.to_remove(reciprocal_op->shared_from_this());
        rewriter.to_remove(mul_op->shared_from_this());
    }
    rewriter.run();
    return impl::status::success;
}

impl::status_t batchnorm_bwd_canonicalization(std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_batchnorm_bwd) continue;

        // insert permute
        bool need_permute = cur_op->has_attr(op_attr::data_format)
                ? (cur_op->get_attr<std::string>(op_attr::data_format) == "NXC")
                : false;

        if (need_permute) {
            // input0 permute
            op_ptr in_perm_op_0
                    = std::make_shared<impl::op_t>(op_kind::dnnl_permute);
            in_perm_op_0->set_attr<std::string>(
                    op_attr::permute_kind, "permute");
            in_perm_op_0->set_attr<std::string>(op_attr::from_format, "NXC");
            in_perm_op_0->set_attr<std::string>(op_attr::to_format, "NCX");
            rewriter.insert_op_before(in_perm_op_0, cur_op, 0);

            // input1 permute
            op_ptr in_perm_op_1
                    = std::make_shared<impl::op_t>(op_kind::dnnl_permute);
            in_perm_op_1->set_attr<std::string>(
                    op_attr::permute_kind, "permute");
            in_perm_op_1->set_attr<std::string>(op_attr::from_format, "NXC");
            in_perm_op_1->set_attr<std::string>(op_attr::to_format, "NCX");
            rewriter.insert_op_before(in_perm_op_1, cur_op, 1);

            // output permute
            op_ptr out_perm_op
                    = std::make_shared<impl::op_t>(op_kind::dnnl_permute);
            out_perm_op->set_attr<std::string>(
                    op_attr::permute_kind, "permute");
            out_perm_op->set_attr<std::string>(op_attr::from_format, "NCX");
            out_perm_op->set_attr<std::string>(op_attr::to_format, "NXC");
            rewriter.insert_op_after(out_perm_op, cur_op, 0);

            cur_op->set_attr<std::string>(op_attr::data_format, "NCX");
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

impl::status_t fuse_to_dnnl_sum(std::shared_ptr<subgraph_t> &sg) {
    auto is_non_broadcast_add = [](const op_t *op) {
        return op->get_kind() == dnnl_impl::op_kind::dnnl_binary
                && static_cast<dnnl::algorithm>(
                           op->get_attr<int64_t>(op_attr::alg_kind))
                == dnnl::algorithm::binary_add
                && op->has_attr(op_attr::auto_broadcast)
                && op->get_attr<std::string>(op_attr::auto_broadcast) == "none";
    };

    std::vector<std::vector<op_ptr>> op_lists;
    std::set<op_t *> visited;
    for (auto &op : sg->get_ops()) {
        if (!is_non_broadcast_add(op.get()) || visited.count(op.get()))
            continue;
        std::vector<op_ptr> list;
        list.emplace_back(op);
        visited.insert(op.get());

        op_t *cur = op.get();
        while (true) {
            auto csms = cur->get_output_value(0)->get_consumers();
            bool match = csms.size() == 1
                    && is_non_broadcast_add(&csms[0].get_op());
            if (match) {
                cur = &csms[0].get_op();
                list.emplace_back(cur->shared_from_this());
                visited.insert(cur);
            } else {
                break;
            }
        }

        // no need to fuse single binary add
        if (list.size() > 1) op_lists.emplace_back(list);
    }

    if (op_lists.empty()) return impl::status::success;

    subgraph_rewriter_t rewriter(sg);
    for (auto &list : op_lists) {
        op_ptr sum_op = std::make_shared<op_t>(op_kind::dnnl_sum);

        auto graph_in_vals = impl::graph_t(list).get_input_values();
        auto graph_out_vals = impl::graph_t(list).get_output_values();

        int input_idx = 0;
        for (auto &cur_op : list) {
            // reconnect graph's input val to dnnl_sum
            auto input_values = cur_op->get_input_values();
            for (auto &in_val : input_values) {
                if (std::find(graph_in_vals.begin(), graph_in_vals.end(),
                            in_val.get())
                        == graph_in_vals.end())
                    continue;

                in_val->remove_consumer(*cur_op, 0);
                sum_op->connect_input(input_idx++, in_val);
            }

            // reconnect graph's output val to dnnl_sum
            auto output_values = cur_op->get_output_values();
            for (auto &out_val : output_values) {
                if (std::find(graph_out_vals.begin(), graph_out_vals.end(),
                            out_val.get())
                        == graph_out_vals.end())
                    continue;

                // scratchpad output is binary op's 2nd output, we skip it
                if (out_val->get_offset() != 0) continue;

                sum_op->add_output(out_val);
                out_val->set_producer(*sum_op);
            }
        }
        // insert the scratchpad output to match the schema definition
        insert_empty_scratchpad(sum_op);

        // remove original add/mul ops
        for (const auto &op : list) {
            rewriter.to_remove(op);
        }

        // add dnnl_sum to subgraph
        rewriter.to_insert(sum_op);
    }

    rewriter.run();
    return impl::status::success;
}

impl::status_t binary_canonicalization(std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_binary) continue;

        bool is_bias_add = cur_op->has_attr(op_attr::is_bias_add)
                ? cur_op->get_attr<bool>(op_attr::is_bias_add)
                : false;

        // check doable
        auto src0_lt = cur_op->get_input_value(0)->get_logical_tensor();
        auto src1_lt = cur_op->get_input_value(1)->get_logical_tensor();

        bool shape_check_ok = true;
        if (is_bias_add) {
            // special check for BiasAdd
            const auto &data_format = cur_op->has_attr(op_attr::data_format)
                    ? cur_op->get_attr<std::string>(op_attr::data_format)
                    : "NCX";
            const auto channel_num
                    = impl::logical_tensor_wrapper_t(src0_lt).get_src_c(
                            data_format);
            shape_check_ok = channel_num == src1_lt.dims[0];
        } else {
            shape_check_ok
                    = binary_doable(ltw(src0_lt).vdims(), ltw(src1_lt).vdims());
        }

        if (!shape_check_ok) return impl::status::invalid_shape;

        // insert unsqueeze op
        int32_t src0_ndims = src0_lt.ndims;
        int32_t src1_ndims = src1_lt.ndims;
        int32_t target_ndims = std::max(src0_ndims, src1_ndims);
        std::vector<int32_t> in_ndims {src0_ndims, src1_ndims};
        for (size_t i = 0; i < cur_op->num_inputs(); ++i) {
            if (in_ndims[i] == target_ndims) { continue; }

            std::vector<int64_t> axes(target_ndims - in_ndims[i]);
            std::iota(axes.begin(), axes.end(), 0);

            // Only for NCX format BiasAdd, we need to unsqueeze the 1D bias to
            // [1, C, 1, 1]
            const bool channel_first = is_bias_add
                    && (!cur_op->has_attr(op_attr::data_format)
                            || cur_op->get_attr<std::string>(
                                       op_attr::data_format)
                                    == "NCX");
            if (channel_first && axes.size() >= 2) {
                axes.erase(axes.begin() + 1);
                axes.emplace_back(-1);
            }

            auto unsqueeze_op = std::make_shared<op_t>(op_kind::dnnl_unsqueeze);
            unsqueeze_op->set_attr<std::vector<int64_t>>(op_attr::axes, axes);
            rewriter.insert_op_before(unsqueeze_op, cur_op, i);
        }

        // set attr
        cur_op->set_attr<bool>(op_attr::canonicalized, true);
    }

    rewriter.run();
    return infer_shape(sg);
}

impl::status_t binary_broadcast_swap(std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_binary) continue;
        const auto alg_kind = static_cast<dnnl::algorithm>(
                cur_op->get_attr<int64_t>(op_attr::alg_kind));
        if (alg_kind != dnnl::algorithm::binary_add
                && alg_kind != dnnl::algorithm::binary_mul)
            continue;

        // check doable
        auto src0_lt = cur_op->get_input_value(0)->get_logical_tensor();
        auto src1_lt = cur_op->get_input_value(1)->get_logical_tensor();

        if (impl::logical_tensor_wrapper_t(src0_lt).nelems()
                >= impl::logical_tensor_wrapper_t(src1_lt).nelems())
            continue;

        op_ptr binary_op = std::make_shared<op_t>(op_kind::dnnl_binary);
        binary_op->merge_attributes(cur_op->get_attributes());

        // swap src0 and src1 value
        auto src0_value = cur_op->get_input_value(0);
        auto src1_value = cur_op->get_input_value(1);
        src1_value->remove_consumer(*cur_op, 1);
        src1_value->add_consumer(*binary_op, 0);
        binary_op->add_input(src1_value);
        src0_value->remove_consumer(*cur_op, 0);
        src0_value->add_consumer(*binary_op, 1);
        binary_op->add_input(src0_value);
        // connect out0 and out1 value
        auto out0_value = cur_op->get_output_value(0);
        binary_op->add_output(out0_value);
        auto out1_value = cur_op->get_output_value(1);
        binary_op->add_output(out1_value);

        rewriter.to_insert(binary_op);
        rewriter.to_remove(cur_op);
    }

    rewriter.run();
    return impl::status::success;
}

impl::status_t fuse_adjacent_reorders(std::shared_ptr<subgraph_t> &sg) {
    const static std::set<impl::op_kind_t> reorder_op_set
            = {op_kind::dnnl_reorder};

    auto fuse_two_adjacent_reorders = [&](bool &changed) -> impl::status_t {
        auto &mgr = sg->fusion_info_mgr_;
        auto &p_engine = sg->p_engine_;
        auto &pd_cache = sg->pd_cache_;

        std::vector<std::pair<op_t *, op_t *>> fuse_groups;

        std::set<const op_t *> visited;
        impl::status_t ret;
        ret = impl::topo_order_visit(sg->get_output_ops(), [&](impl::op_t *op) {
            if (!reorder_op_set.count(op->get_kind()) || visited.count(op) != 0)
                return impl::status::success;

            auto out_val = op->get_output_values()[0];
            auto consumers = out_val->get_consumers();
            if (consumers.size() != 1) return impl::status::success;
            auto &next_op = consumers[0].get_op();

            // check if fusible
            if (reorder_op_set.count(next_op.get_kind()) == 0) {
                return impl::status::success;
            }

            // two reorders should have same shape
            auto next_op_out = next_op.get_output_value(0);
            auto lhs = out_val->get_logical_tensor();
            auto rhs = next_op_out->get_logical_tensor();
            if (ltw(lhs).vdims() != ltw(rhs).vdims()) {
                return impl::status::success;
            }

            // if reorder do per channel scaling, their axis should be
            // same
            int64_t cur_axis = op->has_attr(op_attr::axis)
                    ? op->get_attr<int64_t>(op_attr::axis)
                    : -1;
            int64_t next_axis = next_op.has_attr(op_attr::axis)
                    ? next_op.get_attr<int64_t>(op_attr::axis)
                    : -1;
            if (cur_axis != -1 && next_axis != -1 && cur_axis != next_axis) {
                return impl::status::success;
            }

            // Skip fusion if reorder's output has extra info, because
            // oneDNN doesn't have good supports for such fused cases.
            // TODO(qun) improve the skip rule
            auto fused_out_lt
                    = next_op.get_output_value(0)->get_logical_tensor();
            auto fused_out_md = make_dnnl_memory_desc(fused_out_lt);
            if (fused_out_md.data.extra.flags != dnnl_memory_extra_flag_none) {
                return impl::status::success;
            }

            // push fusible pair to fuse group for later fusion
            fuse_groups.emplace_back(std::pair<op_t *, op_t *> {op, &next_op});
            visited.insert(op);
            visited.insert(&next_op);
            return impl::status::success;
        });

        if (ret != impl::status::success) return ret;

        if (fuse_groups.empty()) {
            changed = false;
            return impl::status::success;
        }

        subgraph_rewriter_t rewriter(sg);
        for (auto &fuse_group : fuse_groups) {
            auto op1 = fuse_group.first;
            auto op2 = fuse_group.second;

            // get the src_zps, dst_zps and scales.
            auto get_scales_zps = [](const op_t *op, std::vector<float> &scales,
                                          std::vector<int64_t> &src_zps,
                                          std::vector<int64_t> &dst_zps,
                                          size_t &num) {
                scales = op->has_attr(op_attr::scales)
                        ? op->get_attr<std::vector<float>>(op_attr::scales)
                        : std::vector<float> {1.0};
                src_zps = op->has_attr(op_attr::src_zps)
                        ? op->get_attr<std::vector<int64_t>>(op_attr::src_zps)
                        : std::vector<int64_t> {0};
                dst_zps = op->has_attr(op_attr::dst_zps)
                        ? op->get_attr<std::vector<int64_t>>(op_attr::dst_zps)
                        : std::vector<int64_t> {0};

                num = std::max(std::max(scales.size(), src_zps.size()),
                        dst_zps.size());
            };

            size_t num1, num2;
            std::vector<float> scales1, scales2;
            std::vector<int64_t> src_zps1, dst_zps1, src_zps2, dst_zps2;
            get_scales_zps(op1, scales1, src_zps1, dst_zps1, num1);
            get_scales_zps(op2, scales2, src_zps2, dst_zps2, num2);

            // broadcast to same dimension
            size_t max_num = std::max(num1, num2);
            if (scales1.size() < max_num) {
                scales1.resize(max_num, scales1[0]);
            }
            if (src_zps1.size() < max_num) {
                src_zps1.resize(max_num, src_zps1[0]);
            }
            if (dst_zps1.size() < max_num) {
                dst_zps1.resize(max_num, dst_zps1[0]);
            }

            if (scales2.size() < max_num) {
                scales2.resize(max_num, scales2[0]);
            }
            if (src_zps2.size() < max_num) {
                src_zps2.resize(max_num, src_zps2[0]);
            }
            if (dst_zps2.size() < max_num) {
                dst_zps2.resize(max_num, dst_zps2[0]);
            }

            // fuse the scales and zps according to the the formula of reorder
            // op: dst = scales*(src-src_zps)+dst_zps;
            std::vector<float> fused_scales;
            std::vector<int64_t> fused_src_zps, fused_dst_zps;
            fused_src_zps = src_zps1;
            fused_scales.reserve(max_num);
            fused_dst_zps.reserve(max_num);
            for (size_t i = 0; i < max_num; i++) {
                fused_scales.emplace_back(scales1[i] * scales2[i]);
                fused_dst_zps.emplace_back(
                        scales2[i] * (dst_zps1[i] - src_zps2[i]) + dst_zps2[i]);
            }

            int64_t axis = -1;
            if (op1->has_attr(op_attr::axis)) {
                axis = op1->get_attr<int64_t>(op_attr::axis);
            }
            if (op2->has_attr(op_attr::axis)) {
                axis = op2->get_attr<int64_t>(op_attr::axis);
            }

            bool change_layout
                    = (op1->has_attr(op_attr::change_layout)
                              && op1->get_attr<bool>(op_attr::change_layout))
                    || (op2->has_attr(op_attr::change_layout)
                            && op2->get_attr<bool>(op_attr::change_layout));

            // create fused op
            op_ptr fused_op = std::make_shared<op_t>(op_kind::dnnl_reorder);
            fused_op->set_attr<bool>(op_attr::change_layout, change_layout);
            if (axis != -1) fused_op->set_attr<int64_t>(op_attr::axis, axis);

            if (!std::all_of(fused_scales.begin(), fused_scales.end(),
                        [](const float &s) { return s == 1.f; })) {
                fused_op->set_attr<std::vector<float>>(
                        op_attr::scales, fused_scales);
            }

            if (std::find_if(fused_src_zps.begin(), fused_src_zps.end(),
                        [](const int64_t &zp) { return zp != 0; })
                    != fused_src_zps.end()) {
                fused_op->set_attr<std::vector<int64_t>>(
                        op_attr::src_zps, fused_src_zps);
            }

            if (std::find_if(fused_dst_zps.begin(), fused_dst_zps.end(),
                        [](const int64_t &zp) { return zp != 0; })
                    != fused_dst_zps.end()) {
                fused_op->set_attr<std::vector<int64_t>>(
                        op_attr::dst_zps, fused_dst_zps);
            }

            // connect fused op to subgraph and remove original ones
            auto in_val = op1->get_input_value(0);
            in_val->remove_consumer(*op1, 0);
            fused_op->connect_input(0, in_val);

            auto out_val = op2->get_output_value(0);
            fused_op->add_output(out_val);
            out_val->set_producer(*fused_op);

            auto scratchpad_val = insert_empty_scratchpad(fused_op);
            const auto &pd = reorder_executable_t::create_desc(
                    fused_op, *p_engine, mgr, pd_cache);
            const memory::desc scratchpad_desc = pd.scratchpad_desc();
            auto status = fill_layout_info(scratchpad_val, scratchpad_desc);
            if (status != impl::status::success) return status;

            rewriter.to_insert(fused_op);
            rewriter.to_remove(op1->shared_from_this());
            rewriter.to_remove(op2->shared_from_this());
        }
        rewriter.run();
        return impl::status::success;
    };

    int cnt = 0;
    const int max_num_limit = static_cast<int>(sg->num_ops());

    bool changed = true;
    do {
        auto ret = fuse_two_adjacent_reorders(changed);
        if (ret != impl::status::success) return ret;
        cnt++;
    } while (changed && cnt <= max_num_limit);

    assertm(cnt <= max_num_limit + 1, "reorder fusion failed.");
    if (cnt > max_num_limit + 1) return impl::status::unimplemented;

    return impl::status::success;
}

impl::status_t fuse_typecast_to_mul_scales(std::shared_ptr<subgraph_t> &sg) {
    std::vector<std::vector<op_t *>> fusion_groups;
    for (const auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != dnnl_impl::op_kind::dnnl_mul_scales
                || !cur_op->get_input_value(0)->has_producer())
            continue;

        auto &in = cur_op->get_input_value(0)->get_producer();
        if (is_typecast(&in))
            fusion_groups.emplace_back(std::vector<op_t *> {cur_op.get(), &in});
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &fusion_group : fusion_groups) {
        op_t *in0 = fusion_group[1];
        rewriter.fuse_op_to_successor(in0->shared_from_this());
    }
    rewriter.run();
    return impl::status::success;
}

impl::status_t split_static_quant(std::shared_ptr<subgraph_t> &sg) {
    std::vector<op_ptr> q_ops;
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() == impl::op_kind::Quantize) {
            q_ops.emplace_back(cur_op);
        }
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &op : q_ops) {
        const auto &zps = op->get_attr<std::vector<int64_t>>(op_attr::zps);
        const auto &scales = op->get_attr<std::vector<float>>(op_attr::scales);
        const auto &qtype = op->get_attr<std::string>(op_attr::qtype);
        const auto &axis = op->get_attr<int64_t>(op_attr::axis);

        auto in_vals = op->get_input_values();
        auto out_vals = op->get_output_values();
        assertm(in_vals.size() == 1 && out_vals.size() == 1,
                "static quantize/dequantize should only have one input and "
                "output");

        // int8 = f32 / scales + zps
        op_ptr mul_scales_op = std::make_shared<op_t>(op_kind::dnnl_mul_scales);
        op_ptr add_zps_op = std::make_shared<op_t>(op_kind::dnnl_add_zps);

        assertm(std::all_of(scales.begin(), scales.end(),
                        [](float i) { return i != 0.f; }),
                "scales can't be zero");

        std::vector<float> inv_scales = dnnl_impl::utils::fmap(
                scales, [](float s) { return 1.f / s; });
        mul_scales_op->set_attr<std::vector<float>>(
                op_attr::scales, inv_scales);
        add_zps_op->set_attr<std::vector<int64_t>>(op_attr::zps, zps);

        mul_scales_op->set_attr<int64_t>(op_attr::axis, axis);
        mul_scales_op->set_attr<std::string>(op_attr::qtype, qtype);
        add_zps_op->set_attr<int64_t>(op_attr::axis, axis);
        add_zps_op->set_attr<std::string>(op_attr::qtype, qtype);

        // reconnect
        in_vals[0]->remove_consumer(*op, 0);
        in_vals[0]->add_consumer(*mul_scales_op, 0);
        mul_scales_op->add_input(in_vals[0]);

        impl::logical_tensor_t new_lt
                = impl::empty_logical_tensor_with_default_id();
        auto new_val
                = std::make_shared<value_t>(*mul_scales_op, 0, new_lt, true);
        new_val->set_data_type(in_vals[0]->get_logical_tensor().data_type);

        mul_scales_op->add_output(new_val);

        add_zps_op->add_input(new_val);
        new_val->add_consumer(*add_zps_op, 0);
        add_zps_op->add_output(out_vals[0]);

        // add new ops and delete quantize op
        rewriter.to_insert(mul_scales_op);
        rewriter.to_insert(add_zps_op);
        rewriter.to_remove(op);
    }
    rewriter.run();
    return infer_shape(sg);
}

impl::status_t split_static_dequant(std::shared_ptr<subgraph_t> &sg) {
    std::vector<op_ptr> dq_ops;
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() == impl::op_kind::Dequantize) {
            dq_ops.emplace_back(cur_op);
        }
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &cur_op : dq_ops) {
        const auto &zps = cur_op->get_attr<std::vector<int64_t>>(op_attr::zps);
        const auto &scales
                = cur_op->get_attr<std::vector<float>>(op_attr::scales);
        const auto &qtype = cur_op->get_attr<std::string>(op_attr::qtype);
        const auto &axis = cur_op->get_attr<int64_t>(op_attr::axis);

        auto in_vals = cur_op->get_input_values();
        auto out_vals = cur_op->get_output_values();
        assertm(in_vals.size() == 1 && out_vals.size() == 1,
                "static dequantize should only have one input and output");

        // f32 = scales * (int8 - zps)
        op_ptr sub_zps_op = std::make_shared<op_t>(op_kind::dnnl_sub_zps);
        op_ptr mul_scales_op = std::make_shared<op_t>(op_kind::dnnl_mul_scales);

        sub_zps_op->set_attr<std::vector<int64_t>>(op_attr::zps, zps);
        mul_scales_op->set_attr<std::vector<float>>(op_attr::scales, scales);

        sub_zps_op->set_attr<int64_t>(op_attr::axis, axis);
        sub_zps_op->set_attr<std::string>(op_attr::qtype, qtype);
        mul_scales_op->set_attr<int64_t>(op_attr::axis, axis);
        mul_scales_op->set_attr<std::string>(op_attr::qtype, qtype);

        // reconnect
        in_vals[0]->remove_consumer(*cur_op, 0);
        in_vals[0]->add_consumer(*sub_zps_op, 0);
        sub_zps_op->add_input(in_vals[0]);

        impl::logical_tensor_t new_lt
                = impl::empty_logical_tensor_with_default_id();
        auto new_val = std::make_shared<value_t>(*sub_zps_op, 0, new_lt, true);
        new_val->set_data_type(in_vals[0]->get_logical_tensor().data_type);

        sub_zps_op->add_output(new_val);

        mul_scales_op->add_input(new_val);
        new_val->add_consumer(*mul_scales_op, 0);
        mul_scales_op->add_output(out_vals[0]);

        // add new ops and delete dequantize op
        rewriter.to_insert(sub_zps_op);
        rewriter.to_insert(mul_scales_op);
        rewriter.to_remove(cur_op);
    }
    rewriter.run();
    return infer_shape(sg);
}

impl::status_t split_dynamic_quant(std::shared_ptr<subgraph_t> &sg) {
    std::vector<op_ptr> q_ops;
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() == impl::op_kind::DynamicQuantize) {
            q_ops.emplace_back(cur_op);
        }
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &cur_op : q_ops) {
        const auto &qtype = cur_op->get_attr<std::string>(op_attr::qtype);
        const auto &axis = cur_op->get_attr<int64_t>(op_attr::axis);

        auto &in_vals = cur_op->get_input_values();
        auto &out_vals = cur_op->get_output_values();
        assertm((in_vals.size() == 3 || in_vals.size() == 2)
                        && out_vals.size() == 1,
                "dynamic quantize must have 2 or 3 inputs and 1 output");

        // DynamicQuantize has optional zps
        bool has_zps = in_vals.size() == 3;

        value_ptr src = in_vals[0], scales = in_vals[1], dst = out_vals[0], zps;
        if (has_zps) zps = in_vals[2];

        // int8 = f32 / scales + zps
        op_ptr mul_scales = std::make_shared<op_t>(op_kind::dnnl_mul_scales);

        mul_scales->connect_input(1, scales);
        scales->remove_consumer(*cur_op, 1);
        mul_scales->set_attr<int64_t>(op_attr::axis, axis);
        mul_scales->set_attr<std::string>(op_attr::qtype, qtype);
        mul_scales->set_attr<bool>(op_attr::with_runtime_scales, true);

        // connect mul_scales to subgraph
        mul_scales->connect_input(0, src);
        src->remove_consumer(*cur_op, 0);
        mul_scales->add_output(dst);
        rewriter.to_insert(mul_scales);

        // op used to inverse the scales
        auto inv_scales_op = std::make_shared<op_t>(op_kind::dnnl_eltwise);
        // y = alpha*x^beta
        inv_scales_op->set_attr<int64_t>(op_attr::alg_kind,
                static_cast<int64_t>(dnnl::algorithm::eltwise_pow));
        inv_scales_op->set_attr<float>(op_attr::alpha, 1.0f);
        inv_scales_op->set_attr<float>(op_attr::beta, -1.0f);
        rewriter.insert_op_before(inv_scales_op, mul_scales, 1);
        insert_empty_scratchpad(inv_scales_op);

        if (has_zps) {
            op_ptr add_zps = std::make_shared<op_t>(op_kind::dnnl_add_zps);
            add_zps->connect_input(1, zps);
            zps->remove_consumer(*cur_op, 2);
            add_zps->set_attr<int64_t>(op_attr::axis, axis);
            add_zps->set_attr<std::string>(op_attr::qtype, qtype);
            add_zps->set_attr<bool>(op_attr::with_runtime_zps, true);

            // connect add_zps to subgraph
            rewriter.insert_op_after(add_zps, mul_scales, 0, 0);
        }

        rewriter.to_remove(cur_op);
    }

    rewriter.run();
    return infer_shape(sg);
}

impl::status_t split_dynamic_dequant(std::shared_ptr<subgraph_t> &sg) {
    std::vector<op_ptr> dq_ops;
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() == impl::op_kind::DynamicDequantize) {
            dq_ops.emplace_back(cur_op);
        }
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &cur_op : dq_ops) {
        const auto &qtype = cur_op->get_attr<std::string>(op_attr::qtype);
        const auto &axis = cur_op->get_attr<int64_t>(op_attr::axis);

        auto &in_vals = cur_op->get_input_values();
        auto &out_vals = cur_op->get_output_values();
        assertm((in_vals.size() == 3 || in_vals.size() == 2)
                        && out_vals.size() == 1,
                "dynamic dequantize must have 2 or 3 inputs and 1 output");

        // DynamicDequantize has optional zps
        bool has_zps = in_vals.size() == 3;

        value_ptr src = in_vals[0], scales = in_vals[1], dst = out_vals[0], zps;
        if (has_zps) zps = in_vals[2];

        // f32 = scales * (int8 - zps)
        // connect scales to mul_scales op
        op_ptr mul_scales = std::make_shared<op_t>(op_kind::dnnl_mul_scales);
        mul_scales->connect_input(1, scales);
        scales->remove_consumer(*cur_op, 1);
        mul_scales->set_attr<int64_t>(op_attr::axis, axis);
        mul_scales->set_attr<std::string>(op_attr::qtype, qtype);
        mul_scales->set_attr<bool>(op_attr::with_runtime_scales, true);

        // connect mul_scales op to subgraph
        mul_scales->connect_input(0, src);
        src->remove_consumer(*cur_op, 0);
        mul_scales->add_output(dst);
        rewriter.to_insert(mul_scales);

        if (has_zps) {
            op_ptr sub_zps = std::make_shared<op_t>(op_kind::dnnl_sub_zps);
            sub_zps->connect_input(1, zps);
            zps->remove_consumer(*cur_op, 2);
            sub_zps->set_attr<int64_t>(op_attr::axis, axis);
            sub_zps->set_attr<std::string>(op_attr::qtype, qtype);
            sub_zps->set_attr<bool>(op_attr::with_runtime_zps, true);

            // connect sub_zps op to subgraph
            rewriter.insert_op_before(sub_zps, mul_scales, 0, 0);
        }

        rewriter.to_remove(cur_op);
    }

    rewriter.run();
    return infer_shape(sg);
}

impl::status_t fuse_static_mul_scales_add_zps(std::shared_ptr<subgraph_t> &sg) {
    std::vector<std::pair<op_t *, op_t *>> fuse_groups;
    for (const auto &op : sg->get_ops()) {
        // This pass only handle static quantization
        if (op->get_kind() != op_kind::dnnl_mul_scales
                || (op->has_attr(op_attr::with_runtime_scales)
                        && op->get_attr<bool>(op_attr::with_runtime_scales)))
            continue;

        auto out_val = op->get_output_values()[0];
        auto consumers = out_val->get_consumers();
        if (consumers.empty()) continue;

        auto &consumer_op = consumers[0].get_op();
        if (consumer_op.get_kind() != op_kind::dnnl_add_zps) continue;

        fuse_groups.emplace_back(
                std::pair<op_t *, op_t *> {op.get(), &consumer_op});
    }

    if (fuse_groups.empty()) return impl::status::success;

    subgraph_rewriter_t rewriter(sg);
    for (auto &fuse_ops : fuse_groups) {
        op_t *mul_scales_op = fuse_ops.first;
        op_t *add_zps_op = fuse_ops.second;

        const int64_t axis = mul_scales_op->get_attr<int64_t>(op_attr::axis);
        const std::string &qtype
                = mul_scales_op->get_attr<std::string>(op_attr::qtype);
        const std::vector<float> &scales
                = mul_scales_op->get_attr<std::vector<float>>(op_attr::scales);
        const std::vector<int64_t> &zps
                = add_zps_op->get_attr<std::vector<int64_t>>(op_attr::zps);

        op_ptr fused_op = std::make_shared<op_t>(op_kind::dnnl_reorder);
        fused_op->set_attr<bool>(op_attr::change_layout, false);
        fused_op->set_attr<int64_t>(op_attr::axis, axis);
        fused_op->set_attr<std::string>(op_attr::qtype, qtype);
        fused_op->set_attr<std::vector<float>>(op_attr::scales, scales);
        if (!utils::all_zero(zps)) {
            fused_op->set_attr<std::vector<int64_t>>(op_attr::dst_zps, zps);
        }

        auto in_val = mul_scales_op->get_input_value(0);
        in_val->remove_consumer(*mul_scales_op, 0);
        fused_op->connect_input(0, in_val);

        auto out_val = add_zps_op->get_output_value(0);
        fused_op->add_output(out_val);
        out_val->set_producer(*fused_op);

        rewriter.to_insert(fused_op);

        // add scratchpad output
        insert_empty_scratchpad(fused_op);

        rewriter.to_remove(mul_scales_op->shared_from_this());
        rewriter.to_remove(add_zps_op->shared_from_this());
    }

    rewriter.run();
    return impl::status::success;
}

impl::status_t fuse_static_sub_zps_mul_scales(std::shared_ptr<subgraph_t> &sg) {
    std::vector<std::pair<op_t *, op_t *>> fuse_groups;
    for (const auto &op : sg->get_ops()) {
        // This pass only handle static quantization
        if (op->get_kind() != op_kind::dnnl_sub_zps
                || (op->has_attr(op_attr::with_runtime_zps)
                        && op->get_attr<bool>(op_attr::with_runtime_zps)))
            continue;

        auto out_val = op->get_output_values()[0];
        auto consumers = out_val->get_consumers();
        if (consumers.empty()) continue;

        auto &consumer_op = consumers[0].get_op();
        if (consumer_op.get_kind() != op_kind::dnnl_mul_scales) continue;

        fuse_groups.emplace_back(
                std::pair<op_t *, op_t *> {op.get(), &consumer_op});
    }

    if (fuse_groups.empty()) return impl::status::success;

    subgraph_rewriter_t rewriter(sg);
    for (auto &fuse_ops : fuse_groups) {
        op_t *sub_zps_op = fuse_ops.first;
        op_t *mul_scales_op = fuse_ops.second;

        const int64_t axis = sub_zps_op->get_attr<int64_t>(op_attr::axis);
        const std::string &qtype
                = sub_zps_op->get_attr<std::string>(op_attr::qtype);
        const std::vector<float> &scales
                = mul_scales_op->get_attr<std::vector<float>>(op_attr::scales);
        const std::vector<int64_t> &zps
                = sub_zps_op->get_attr<std::vector<int64_t>>(op_attr::zps);

        op_ptr fused_op = std::make_shared<op_t>(op_kind::dnnl_reorder);
        fused_op->set_attr<bool>(op_attr::change_layout, false);
        fused_op->set_attr<int64_t>(op_attr::axis, axis);
        fused_op->set_attr<std::string>(op_attr::qtype, qtype);
        fused_op->set_attr<std::vector<float>>(op_attr::scales, scales);
        if (!utils::all_zero(zps)) {
            fused_op->set_attr<std::vector<int64_t>>(op_attr::src_zps, zps);
        }
        auto in_val = sub_zps_op->get_input_value(0);
        in_val->remove_consumer(*sub_zps_op, 0);
        fused_op->connect_input(0, in_val);

        auto out_val = mul_scales_op->get_output_value(0);
        fused_op->add_output(out_val);
        out_val->set_producer(*fused_op);

        rewriter.to_insert(fused_op);

        // add scratchpad output
        insert_empty_scratchpad(fused_op);

        rewriter.to_remove(sub_zps_op->shared_from_this());
        rewriter.to_remove(mul_scales_op->shared_from_this());
    }

    rewriter.run();
    return impl::status::success;
}

impl::status_t fuse_dynamic_mul_scales_add_zps(
        std::shared_ptr<subgraph_t> &sg) {
    std::vector<std::pair<op_ptr, op_ptr>> fuse_groups;
    std::set<op_t *> visited;
    for (const auto &cur_op : sg->get_ops()) {
        if ((cur_op->get_kind() != op_kind::dnnl_mul_scales)
                || visited.count(cur_op.get()) != 0)
            continue;

        // This pass only handle dynamic quantization
        if (!cur_op->has_attr(op_attr::with_runtime_scales)
                || !cur_op->get_attr<bool>(op_attr::with_runtime_scales))
            continue;

        auto out_val = cur_op->get_output_values()[0];
        auto consumers = out_val->get_consumers();
        if (consumers.empty()) continue;

        auto &consumer_op = consumers[0].get_op();
        if (consumer_op.get_kind() != op_kind::dnnl_add_zps) continue;

        if (!consumer_op.has_attr(op_attr::with_runtime_zps)
                || !consumer_op.get_attr<bool>(op_attr::with_runtime_zps))
            continue;

        fuse_groups.emplace_back(std::pair<op_ptr, op_ptr> {
                cur_op, (&consumer_op)->shared_from_this()});
        visited.insert(cur_op.get());
        visited.insert(&consumer_op);
    }

    if (fuse_groups.empty()) return impl::status::success;

    subgraph_rewriter_t rewriter(sg);
    for (auto &fuse_ops : fuse_groups) {
        op_ptr &mul_scales = fuse_ops.first; // mul_scales
        op_ptr &add_zps = fuse_ops.second; // add_zps

        const int64_t axis = mul_scales->get_attr<int64_t>(op_attr::axis);
        const std::string &qtype
                = mul_scales->get_attr<std::string>(op_attr::qtype);

        op_ptr fused_op = std::make_shared<op_t>(op_kind::dnnl_reorder);
        fused_op->set_attr<bool>(op_attr::change_layout, false);
        fused_op->set_attr<int64_t>(op_attr::axis, axis);
        fused_op->set_attr<std::string>(op_attr::qtype, qtype);

        // src must be the 0-th input
        auto src = mul_scales->get_input_value(0);
        src->remove_consumer(*mul_scales, 0);
        fused_op->connect_input(0, src);

        // fuse scales as output scales
        auto scales = mul_scales->get_input_value(1);
        scales->remove_consumer(*mul_scales, 1);
        fused_op->connect_input(1, scales);
        fused_op->set_attr<bool>(op_attr::with_runtime_scales, true);

        // fuse dst zps
        auto zps = add_zps->get_input_value(1);
        zps->remove_consumer(*add_zps, 1);
        fused_op->connect_input(2, zps);
        fused_op->set_attr<bool>(op_attr::with_runtime_dst_zps, true);

        auto dst = add_zps->get_output_value(0);
        fused_op->add_output(dst);
        dst->set_producer(*fused_op);

        insert_empty_scratchpad(fused_op);

        rewriter.to_insert(fused_op);
        rewriter.to_remove(mul_scales);
        rewriter.to_remove(add_zps);
    }

    rewriter.run();
    return impl::status::success;
}

impl::status_t fuse_dynamic_sub_zps_mul_scales(
        std::shared_ptr<subgraph_t> &sg) {
    std::vector<std::pair<op_ptr, op_ptr>> fuse_groups;
    std::set<op_t *> visited;
    for (const auto &cur_op : sg->get_ops()) {
        if ((cur_op->get_kind() != op_kind::dnnl_sub_zps)
                || visited.count(cur_op.get()) != 0)
            continue;

        // This pass only handle dynamic quantization
        if (!cur_op->has_attr(op_attr::with_runtime_zps)
                || !cur_op->get_attr<bool>(op_attr::with_runtime_zps))
            continue;

        auto out_val = cur_op->get_output_values()[0];
        auto consumers = out_val->get_consumers();
        if (consumers.empty()) continue;

        auto &consumer_op = consumers[0].get_op();
        if (consumer_op.get_kind() != op_kind::dnnl_mul_scales) continue;
        if (!consumer_op.has_attr(op_attr::with_runtime_scales)
                || !consumer_op.get_attr<bool>(op_attr::with_runtime_scales))
            continue;

        fuse_groups.emplace_back(std::pair<op_ptr, op_ptr> {
                cur_op, (&consumer_op)->shared_from_this()});
        visited.insert(cur_op.get());
        visited.insert(&consumer_op);
    }

    if (fuse_groups.empty()) return impl::status::success;

    subgraph_rewriter_t rewriter(sg);
    for (auto &fuse_ops : fuse_groups) {
        op_ptr &op1 = fuse_ops.first; // sub_zps
        op_ptr &op2 = fuse_ops.second; // mul_scales

        const int64_t axis = op1->get_attr<int64_t>(op_attr::axis);
        const std::string &qtype = op1->get_attr<std::string>(op_attr::qtype);

        op_ptr fused_op = std::make_shared<op_t>(op_kind::dnnl_reorder);
        fused_op->set_attr<bool>(op_attr::change_layout, false);
        fused_op->set_attr<int64_t>(op_attr::axis, axis);
        fused_op->set_attr<std::string>(op_attr::qtype, qtype);

        // src must be the 0-th input
        auto src = op1->get_input_value(0);
        src->remove_consumer(*op1, 0);
        fused_op->connect_input(0, src);

        // fuse src zps
        auto zps = op1->get_input_value(1);
        zps->remove_consumer(*op1, 1);
        fused_op->connect_input(1, zps);
        fused_op->set_attr<bool>(op_attr::with_runtime_src_zps, true);

        // fuse scales as output scales
        auto scales = op2->get_input_value(1);
        scales->remove_consumer(*op2, 1);
        fused_op->connect_input(2, scales);
        fused_op->set_attr<bool>(op_attr::with_runtime_scales, true);

        auto dst = op2->get_output_value(0);
        fused_op->add_output(dst);
        dst->set_producer(*fused_op);

        insert_empty_scratchpad(fused_op);

        rewriter.to_insert(fused_op);
        rewriter.to_remove(op1);
        rewriter.to_remove(op2);
    }

    rewriter.run();
    return impl::status::success;
}

impl::status_t reorder_canonicalization(std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_reorder) continue;

        size_t index = 1; // the start index of optional runtime scales and zps

        // check runtime src_zps and add typecast if necessary
        if (cur_op->has_attr(op_attr::with_runtime_src_zps)
                && cur_op->get_attr<bool>(op_attr::with_runtime_src_zps)) {
            auto src_zps = cur_op->get_input_value(index);
            if (src_zps->get_logical_tensor().data_type
                    != impl::data_type::s32) {
                auto tc_op = std::make_shared<op_t>(
                        dnnl_impl::op_kind::dnnl_reorder);
                tc_op->set_attr<bool>(op_attr::change_layout, false);
                rewriter.insert_op_before(tc_op, cur_op, index);
                insert_empty_scratchpad(tc_op);
                tc_op->get_output_value(0)->set_data_type(impl::data_type::s32);
                index++;
            }
        }
        // optionally skip the runtime scales
        if (cur_op->has_attr(op_attr::with_runtime_scales)
                && cur_op->get_attr<bool>(op_attr::with_runtime_scales)) {
            index++;
        }

        // check runtime dst_zps and add typecast if necessary
        if (cur_op->has_attr(op_attr::with_runtime_dst_zps)
                && cur_op->get_attr<bool>(op_attr::with_runtime_dst_zps)) {
            auto dst_zps = cur_op->get_input_value(index);
            if (dst_zps->get_logical_tensor().data_type
                    != impl::data_type::s32) {
                auto tc_op = std::make_shared<op_t>(
                        dnnl_impl::op_kind::dnnl_reorder);
                tc_op->set_attr<bool>(op_attr::change_layout, false);
                rewriter.insert_op_before(tc_op, cur_op, index);
                tc_op->get_output_value(0)->set_data_type(impl::data_type::s32);
                index++;
            }
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

impl::status_t lower_down(std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        op_ptr new_op;
        bool merge_attr = true;
        bool insert_scratchpad = true;
        if (cur_op->get_kind() == impl::op_kind::Convolution) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_convolution);
        } else if (cur_op->get_kind() == impl::op_kind::ConvTranspose) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_convtranspose);
        } else if (cur_op->get_kind()
                == impl::op_kind::ConvTransposeBackpropData) {
            new_op = std::make_shared<op_t>(
                    op_kind::dnnl_convtranspose_bwd_data);
        } else if (cur_op->get_kind()
                == impl::op_kind::ConvTransposeBackpropFilters) {
            new_op = std::make_shared<op_t>(
                    op_kind::dnnl_convtranspose_bwd_weights);
        } else if (cur_op->get_kind()
                == impl::op_kind::ConvolutionBackpropData) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_conv_bwd_data);
        } else if (cur_op->get_kind()
                == impl::op_kind::ConvolutionBackpropFilters) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_conv_bwd_weights);
        } else if (cur_op->get_kind() == impl::op_kind::MaxPool
                || cur_op->get_kind() == impl::op_kind::AvgPool) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_pool);
            if (cur_op->get_kind() == impl::op_kind::MaxPool) {
                new_op->set_attr<std::string>(op_attr::kind, "maxpool");
            } else {
                new_op->set_attr<std::string>(op_attr::kind, "avgpool");
            }
        } else if (cur_op->get_kind() == impl::op_kind::AvgPoolBackprop) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_pool_bwd);
            new_op->set_attr<std::string>(op_attr::kind, "avgpool");
        } else if (cur_op->get_kind() == impl::op_kind::SoftMaxBackprop) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_softmax_bwd);
        } else if (cur_op->get_kind() == impl::op_kind::LogSoftmaxBackprop) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_logsoftmax_bwd);
        } else if (is_binary_kind(cur_op->get_kind())) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_binary);
            new_op->set_attr<int64_t>(op_attr::alg_kind,
                    static_cast<int64_t>(
                            get_binary_alg_map().at(cur_op->get_kind())));
        } else if (cur_op->get_kind() == impl::op_kind::BiasAdd) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_binary);
            new_op->set_attr<int64_t>(op_attr::alg_kind,
                    static_cast<int64_t>(dnnl::algorithm::binary_add));
            new_op->set_attr<bool>(op_attr::is_bias_add, true);
        } else if (cur_op->get_kind() == impl::op_kind::SoftPlus
                || cur_op->get_kind() == impl::op_kind::SoftPlusBackprop) {
            const auto beta = cur_op->get_attr<int64_t>(op_attr::beta);
            const auto algo = dnnl::algorithm::eltwise_soft_relu_v2;
            if (cur_op->get_kind() == impl::op_kind::SoftPlus) {
                new_op = std::make_shared<op_t>(op_kind::dnnl_eltwise);
            } else { // SoftPlusBackprop
                new_op = std::make_shared<op_t>(op_kind::dnnl_eltwise_bwd);
                new_op->set_attr(
                        op_attr::fwd_alg_kind, static_cast<int64_t>(algo));
                new_op->set_attr(op_attr::use_dst, false);
            }
            new_op->set_attr<int64_t>(
                    op_attr::alg_kind, static_cast<int64_t>(algo));
            new_op->set_attr<float>(op_attr::alpha, static_cast<float>(beta));
            merge_attr = false;
        } else if (is_eltwise_kind(cur_op->get_kind())) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_eltwise);
            merge_common_eltwise_attrs(cur_op, new_op);
            merge_attr = false;
            new_op->set_attr<int64_t>(op_attr::alg_kind,
                    static_cast<int64_t>(
                            get_eltwise_alg_map().at(cur_op->get_kind())));
        } else if (is_eltwise_bwd_kind(cur_op->get_kind())) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_eltwise_bwd);
            merge_common_eltwise_attrs(cur_op, new_op);
            const bool use_dst = cur_op->has_attr(op_attr::use_dst)
                    ? cur_op->get_attr<bool>(op_attr::use_dst)
                    : false;
            new_op->set_attr(op_attr::use_dst, use_dst);
            merge_attr = false;
            auto kind = cur_op->get_kind();
            auto bwd_algo = get_eltwise_bwd_alg(kind, use_dst);
            auto fwd_algo = get_eltwise_bwd_alg(kind, false);
            if (bwd_algo == algorithm::undef) {
                DEBUG_PRINT_ERROR("Unsupported etlwise bwd op.");
                return impl::status::unimplemented;
            }
            new_op->set_attr<int64_t>(
                    op_attr::alg_kind, static_cast<int64_t>(bwd_algo));
            new_op->set_attr<int64_t>(
                    op_attr::fwd_alg_kind, static_cast<int64_t>(fwd_algo));
        } else if (cur_op->get_kind() == impl::op_kind::BatchNormInference
                || cur_op->get_kind()
                        == impl::op_kind::BatchNormForwardTraining) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_batchnorm);

            // decide if this is for training or inference
            if (cur_op->get_kind() == impl::op_kind::BatchNormInference)
                new_op->set_attr<bool>(op_attr::is_training, false);
            else
                new_op->set_attr<bool>(op_attr::is_training, true);
        } else if (cur_op->get_kind()
                == impl::op_kind::BatchNormTrainingBackprop) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_batchnorm_bwd);
        } else if (cur_op->get_kind() == impl::op_kind::PReLU) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_prelu);
        } else if (cur_op->get_kind() == impl::op_kind::PReLUBackprop) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_prelu_bwd);
        } else if (is_reduction_kind(cur_op->get_kind())) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_reduction);
            new_op->set_attr<int64_t>(op_attr::alg_kind,
                    static_cast<int64_t>(cur_op->get_kind()));
            new_op->set_attr<int64_t>(op_attr::alg_kind,
                    static_cast<int64_t>(
                            get_reduction_alg_map().at(cur_op->get_kind())));
            if (cur_op->get_kind() == impl::op_kind::ReduceL1)
                new_op->set_attr<float>(op_attr::p, 1.0f);
            else if (cur_op->get_kind() == impl::op_kind::ReduceL2)
                new_op->set_attr<float>(op_attr::p, 2.0f);
        } else if (cur_op->get_kind() == impl::op_kind::Interpolate) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_resampling);
        } else if (cur_op->get_kind() == impl::op_kind::InterpolateBackprop) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_resampling_bwd);
        } else if (cur_op->get_kind() == impl::op_kind::Concat) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_concat);
        } else if (cur_op->get_kind() == impl::op_kind::LayerNormBackprop) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_layernorm_bwd);
        } else if (cur_op->get_kind() == impl::op_kind::MatMul) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_matmul);
        } else if (cur_op->get_kind() == impl::op_kind::SoftMax) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_softmax);
        } else if (cur_op->get_kind() == impl::op_kind::LogSoftmax) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_logsoftmax);
        } else if (cur_op->get_kind() == impl::op_kind::LayerNorm) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_layernorm);
        } else if (cur_op->get_kind() == impl::op_kind::Reorder) {
            // TODO(qun) Need to lower Quantize and Dequantize to
            // dnnl_reorder? Or it's better to use the other passes like
            // split_static_quant pass to perform the lowering?
            new_op = std::make_shared<op_t>(op_kind::dnnl_reorder);
            new_op->set_attr<bool>(op_attr::change_layout, true);
        } else if (cur_op->get_kind() == impl::op_kind::TypeCast) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_reorder);
            new_op->set_attr<bool>(op_attr::change_layout, false);
        } else if (cur_op->get_kind() == impl::op_kind::Reciprocal) {
            // mapping to dnnl algorithm pow
            new_op = std::make_shared<op_t>(op_kind::dnnl_eltwise);
            merge_attr = false;
            new_op->set_attr<int64_t>(op_attr::alg_kind,
                    static_cast<int64_t>(dnnl::algorithm::eltwise_pow));
            new_op->set_attr<float>(op_attr::alpha, 1.f);
            new_op->set_attr<float>(op_attr::beta, -1.f);
        } else if (cur_op->get_kind() == impl::op_kind::StaticReshape) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_reshape);
            insert_scratchpad = false;
        } else if (cur_op->get_kind() == impl::op_kind::StaticTranspose) {
            new_op = std::make_shared<op_t>(op_kind::dnnl_transpose);
            insert_scratchpad = false;
        } else {
            // TODO(xxx) Lower other ops to internal ops
            continue;
        }

        rewriter.replace_op(cur_op, new_op);

        // Not all ops will merge attr, it depends on the op definition in
        // schema. For example, dnnl_eltwise need to convert original attrs to
        // alpha and beta attr, instead of directly merging them.
        if (merge_attr) new_op->merge_attributes(cur_op->get_attributes());

        if (insert_scratchpad)
            auto scratchpad_val = insert_empty_scratchpad(new_op);
    }

    rewriter.run();
    return infer_shape(sg);
}

impl::status_t common_reorder_elimination(std::shared_ptr<subgraph_t> &sg) {
    // Eliminate two equal reorder with same input
    auto cse_func = [&](bool &changed) {
        std::vector<op_t *> fusion;

        // find two fusible ops
        for (auto &op : sg->get_ops()) {
            if (op->get_kind() != impl::op_kind::Reorder) continue;

            auto ins = op->get_input_values();
            // equal op must share same inputs, so it's enough to only look up
            // ins[0]'s consumers.
            auto csms = ins[0]->get_consumers();
            bool found_equal_op = false;
            for (auto csm : csms) {
                auto &csm_op = csm.get_op();
                // the same op, skip
                if (&csm_op == op.get()) continue;

                // not equal op, skip
                bool equal_op = op->get_kind() == csm_op.get_kind()
                        && op->has_same_attr_values(csm_op);
                if (!equal_op) continue;

                // not same inputs, skip
                auto csm_ins = csm_op.get_input_values();
                if (csm_ins.size() != ins.size()) continue;
                size_t i;
                for (i = 0; i < csm_ins.size(); i++) {
                    if (csm_ins[i].get() != ins[i].get()) break;
                }
                if (i < csm_ins.size()) continue;

                // not equal outputs, skip
                auto &outs = op->get_output_values();
                auto &csm_outs = csm_op.get_output_values();
                if (csm_outs.size() != outs.size()) continue;
                for (i = 0; i < csm_outs.size(); i++) {
                    auto lt1 = csm_outs[i]->get_logical_tensor();
                    auto lt2 = outs[i]->get_logical_tensor();
                    if (make_dnnl_memory_desc(lt1)
                            != make_dnnl_memory_desc(lt2))
                        break;
                }
                if (i < csm_outs.size()) continue;

                // all condition matched
                fusion.emplace_back(op.get());
                fusion.emplace_back(&csm_op);
                found_equal_op = true;
                break;
            }
            if (found_equal_op) break;
        }

        if (fusion.empty()) {
            changed = false;
            return impl::status::success;
        }

        // remove op2 and add it's consumers to op1
        auto op1 = fusion[0], op2 = fusion[1];
        auto op2_ins = op2->get_input_values();
        for (size_t i = 0; i < op2_ins.size(); i++) {
            op2_ins[i]->remove_consumer(*op2, i);
        }

        auto op1_outs = op1->get_output_values();
        auto op2_outs = op2->get_output_values();
        for (size_t i = 0; i < op2_outs.size(); i++) {
            auto &csms = op2_outs[i]->get_consumers();
            for (auto &csm : csms) {
                op1_outs[i]->add_consumer(csm.get_op(), csm.get_offset());
                csm.get_op().connect_input(csm.get_offset(), op1_outs[i]);
            }
        }

        subgraph_rewriter_t rewriter(sg);
        rewriter.to_remove(op2->shared_from_this());
        rewriter.run();

        changed = true;
        return impl::status::success;
    };

    int cnt = 0;
    const int max_iter_num = static_cast<int>(sg->num_ops());

    bool changed = true;
    do {
        auto ret = cse_func(changed);
        if (ret != impl::status::success) return ret;
        cnt++;
    } while (changed && cnt <= max_iter_num);

    assertm(cnt <= max_iter_num + 1,
            "Failed to eliminate common reorders since the pass can't "
            "converge.");
    if (cnt > max_iter_num + 1) return impl::status::unimplemented;

    return impl::status::success;
}

impl::status_t remove_unnecessary_quant_dequant(
        std::shared_ptr<subgraph_t> &sg) {
    std::vector<op_ptr> pool_ops;
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() == op_kind::dnnl_pool) {
            pool_ops.emplace_back(cur_op);
        }
    }

    if (pool_ops.empty()) return impl::status::success;

    subgraph_rewriter_t rewriter(sg);
    for (auto &pool_op : pool_ops) {
        value_ptr pool_in_val = pool_op->get_input_value(0);
        value_ptr pool_out_val = pool_op->get_output_value(0);
        if (!pool_in_val->has_producer()
                || pool_out_val->get_consumers().empty())
            continue;

        op_t &quant_op = pool_out_val->get_consumers()[0].get_op();
        const bool pool_with_post_ops
                = quant_op.get_kind() != impl::op_kind::Quantize;
        if (pool_with_post_ops) continue;

        op_t &dequant_op = pool_in_val->get_producer();
        assertm(dequant_op.get_kind() == impl::op_kind::Dequantize,
                "the predecessor of a pooling op should be dequantize.");

        value_ptr dequant_in_val = dequant_op.get_input_value(0);
        value_ptr quant_out_val = quant_op.get_output_value(0);
        dequant_in_val->remove_consumer(dequant_op, 0);
        pool_op->connect_input(0, dequant_in_val);
        pool_op->connect_output(0, quant_out_val);

        rewriter.to_remove(dequant_op.shared_from_this());
        rewriter.to_remove(quant_op.shared_from_this());
    }

    rewriter.run();
    return impl::status::success;
}

// combine scales around binary post op
//
//         |                     |        |
//      base_op               base_op    zps1
//         |         |           |        |
//       zps0       zps1        zps0    new_scales1 (optionall)
//         |         |            \      /
//      scales0   scales1          binary
//           \      /                 |
//            binary      =>     new_scales2
//               |                    |
//            scales2                zps2
//               |                    |
//              zps2
//               |
//
//    where base_op is one of [pool]
//
//    binary-add case:
//    - new_scales1 = scales1 / scales0
//    - new_scales2 = scales0 * scales2
//    binary-mul case:
//    - new_scales1 will be removed
//    - new_scales2 = scales0 * scales1 * scales2
impl::status_t combine_binary_post_op_scales(std::shared_ptr<subgraph_t> &sg) {
    const auto fuse_scales
            = [](const std::vector<float> &scales0,
                      const std::vector<float> &scales1,
                      const std::function<float(float, float)> &operation)
            -> std::vector<float> {
        std::vector<float> fused_scales(
                std::max(scales0.size(), scales1.size()), 1.f);
        if (scales0.size() >= scales1.size()) {
            for (size_t i = 0; i < scales0.size(); ++i) {
                fused_scales[i] = operation(scales0[i], scales1[0]);
            }
        } else {
            for (size_t i = 0; i < scales1.size(); ++i) {
                fused_scales[i] = operation(scales0[0], scales1[i]);
            }
        }
        return fused_scales;
    };
    const auto fuse_scales_attributes = [](const std::vector<op_t *> &scale_ops)
            -> std::pair<std::string, int64_t> {
        for (size_t i = 0; i < scale_ops.size(); ++i) {
            if (scale_ops[i]->get_attr<std::string>(op_attr::qtype)
                    == "per_channel") {
                // assumption: at least one scales per channel will make
                // combined scales per channel
                return std::make_pair("per_channel",
                        scale_ops[i]->get_attr<int64_t>(op_attr::axis));
            }
        }
        // scales per tensor, defaulting axis
        return std::make_pair("per_tensor", static_cast<int64_t>(1));
    };

    std::vector<op_ptr> bin_ops;
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() == op_kind::dnnl_binary) {
            value_ptr bin_in0_val = cur_op->get_input_value(0);
            value_ptr bin_in1_val = cur_op->get_input_value(1);
            value_ptr bin_out_val = cur_op->get_output_value(0);
            if (!bin_in0_val->has_producer() || !bin_in1_val->has_producer()
                    || bin_out_val->get_consumers().empty())
                continue;

            if (bin_in0_val->get_producer().get_kind()
                            != op_kind::dnnl_mul_scales
                    || bin_in1_val->get_producer().get_kind()
                            != op_kind::dnnl_mul_scales
                    || bin_out_val->get_consumers()[0].get_op().get_kind()
                            != op_kind::dnnl_mul_scales)
                continue;

            bin_ops.emplace_back(cur_op);
        }
    }

    if (bin_ops.empty()) return impl::status::success;

    subgraph_rewriter_t rewriter(sg);
    for (auto &bin_op : bin_ops) {
        value_ptr bin_in0_val = bin_op->get_input_value(0);
        value_ptr bin_in1_val = bin_op->get_input_value(1);
        value_ptr bin_out_val = bin_op->get_output_value(0);
        if (!bin_in0_val->has_producer() || !bin_in1_val->has_producer()
                || bin_out_val->get_consumers().empty())
            continue;

        op_t &scales_in0_op = bin_in0_val->get_producer();
        assertm(scales_in0_op.get_kind() == op_kind::dnnl_mul_scales,
                "the first predecessor of a binary op should be mul_scales.");
        if (scales_in0_op.has_attr(op_attr::with_runtime_scales)
                && scales_in0_op.get_attr<bool>(op_attr::with_runtime_scales))
            continue;

        op_t &scales_in1_op = bin_in1_val->get_producer();
        assertm(scales_in1_op.get_kind() == op_kind::dnnl_mul_scales,
                "the second predecessor of a binary op should be mul_scales.");
        if (scales_in1_op.has_attr(op_attr::with_runtime_scales)
                && scales_in1_op.get_attr<bool>(op_attr::with_runtime_scales))
            continue;

        op_t &scales_out_op = bin_out_val->get_consumers()[0].get_op();
        assertm(scales_out_op.get_kind() == op_kind::dnnl_mul_scales,
                "the successor of a binary op should be mul_scales.");
        if (scales_out_op.has_attr(op_attr::with_runtime_scales)
                && scales_out_op.get_attr<bool>(op_attr::with_runtime_scales))
            continue;

        const size_t base_op_branch_idx = [&scales_in0_op]() {
            op_t &zps_op = scales_in0_op.get_input_value(0)->get_producer();
            if (zps_op.get_input_value(0)->has_producer()) {
                const auto zps_predecessor_kind
                        = zps_op.get_input_value(0)->get_producer().get_kind();
                if (zps_predecessor_kind == op_kind::dnnl_eltwise
                        || zps_predecessor_kind == op_kind::dnnl_pool) {
                    return 0;
                }
            }
            return 1;
        }();
        op_t &base_scales_op
                = (base_op_branch_idx) ? scales_in1_op : scales_in0_op;
        op_t &other_scales_op
                = (base_op_branch_idx) ? scales_in0_op : scales_in1_op;

        const auto in0_scales
                = base_scales_op.get_attr<std::vector<float>>(op_attr::scales);
        const auto in1_scales
                = other_scales_op.get_attr<std::vector<float>>(op_attr::scales);
        const auto inv_out_scales
                = scales_out_op.get_attr<std::vector<float>>(op_attr::scales);
        const auto bin_kind = static_cast<dnnl::algorithm>(
                bin_op->get_attr<int64_t>(op_attr::alg_kind));

        std::vector<float> new_scales_in1, new_scales_in0;
        std::string new_qtype_in1;
        std::string new_qtype_in0;
        int64_t new_axis_in1 = 0;
        int64_t new_axis_in0 = 0;
        bool drop_other_scales = false;
        const auto multiplier = std::multiplies<float>();
        switch (bin_kind) {
            case dnnl::algorithm::binary_add:
                assertm(std::all_of(in0_scales.begin(), in0_scales.end(),
                                [](float v) { return v != 0.f; }),
                        "scales can't be zero");
                new_scales_in0
                        = fuse_scales(in0_scales, inv_out_scales, multiplier);
                new_scales_in1
                        = fuse_scales(in1_scales, inv_out_scales, multiplier);
                std::tie(new_qtype_in1, new_axis_in1) = fuse_scales_attributes(
                        {&scales_in1_op, &scales_out_op});
                std::tie(new_qtype_in0, new_axis_in0) = fuse_scales_attributes(
                        {&scales_in0_op, &scales_out_op});
                break;
            case dnnl::algorithm::binary_mul:
                drop_other_scales = true;
                new_scales_in0
                        = fuse_scales(in0_scales, in1_scales, multiplier);
                new_scales_in0 = fuse_scales(
                        new_scales_in0, inv_out_scales, multiplier);
                std::tie(new_qtype_in0, new_axis_in0) = fuse_scales_attributes(
                        {&scales_in0_op, &scales_in1_op, &scales_out_op});
                break;
            default:
                assertm(false, "unsupported binary post-op was provided.");
                break;
        }

        // drop out scales op
        rewriter.fuse_op_to_predecessor(scales_out_op.shared_from_this());

        // if possible, drop other scales, and connect zps with binary
        // otherwise, update other scales data
        if (drop_other_scales) {
            rewriter.fuse_op_to_successor(other_scales_op.shared_from_this());
        } else {
            other_scales_op.set_attr(op_attr::scales, new_scales_in1)
                    .set_attr(op_attr::qtype, new_qtype_in1)
                    .set_attr(op_attr::axis, new_axis_in1);
        }

        // update output scales data
        base_scales_op.set_attr(op_attr::scales, new_scales_in0)
                .set_attr(op_attr::qtype, new_qtype_in0)
                .set_attr(op_attr::axis, new_axis_in0);
    }

    rewriter.run();
    return infer_shape(sg);
}

impl::status_t remove_quant_data_with_no_effect(
        std::shared_ptr<subgraph_t> &sg) {
    std::vector<op_ptr> quant_data_ops;
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() == op_kind::dnnl_mul_scales
                || cur_op->get_kind() == op_kind::dnnl_add_zps
                || cur_op->get_kind() == op_kind::dnnl_sub_zps) {
            quant_data_ops.emplace_back(cur_op);
        }
    }

    if (quant_data_ops.empty()) return impl::status::success;

    subgraph_rewriter_t rewriter(sg);
    for (auto &quant_data_op : quant_data_ops) {
        bool to_remove = false;
        if (quant_data_op->get_kind() == op_kind::dnnl_mul_scales) {
            const auto scales = quant_data_op->get_attr<std::vector<float>>(
                    op_attr::scales);
            to_remove = std::all_of(scales.begin(), scales.end(), [](float s) {
                float expected = 1.0f;
                float eps = 0.000001f;
                return std::abs(s - expected) <= eps;
            });
        } else {
            const auto zps = quant_data_op->get_attr<std::vector<int64_t>>(
                    op_attr::zps);
            to_remove = std::all_of(
                    zps.begin(), zps.end(), [](int64_t z) { return z == 0; });
        }

        if (to_remove) {
            value_ptr quant_data_out_val = quant_data_op->get_output_value(0);
            if (quant_data_out_val->get_consumers().empty()) {
                value_ptr quant_data_in_val = quant_data_op->get_input_value(0);
                op_t &predecessor_op = quant_data_in_val->get_producer();
                predecessor_op.connect_output(0, quant_data_out_val);
                rewriter.to_remove(quant_data_op);
            } else {
                rewriter.fuse_op_to_successor(quant_data_op);
            }
        }
    }

    rewriter.run();
    return impl::status::success;
}

impl::status_t move_scalar_div_behind_matmul(std::shared_ptr<subgraph_t> &sg) {
    using ltw = impl::logical_tensor_wrapper_t;

    while (true) {
        std::vector<std::pair<impl::op_t *, impl::op_t *>> to_be_swapped;
        for (auto &op : sg->get_ops()) {
            bool ok = op->get_kind() == dnnl_impl::op_kind::dnnl_matmul
                    && op->get_input_value(0)->has_producer();
            if (!ok) continue;

            impl::op_t *producer = op->get_input_op(0);
            ok = producer->get_kind() == dnnl_impl::op_kind::dnnl_binary
                    && dnnl::algorithm::binary_div
                            == static_cast<dnnl::algorithm>(
                                    producer->get_attr<int64_t>(
                                            op_attr::alg_kind));
            if (!ok) continue;

            // only match scalar div
            auto div_src0_shape
                    = ltw(producer->get_input_value(0)->get_logical_tensor())
                              .vdims();
            auto div_src1_shape
                    = ltw(producer->get_input_value(1)->get_logical_tensor())
                              .vdims();
            auto div_dst_shape
                    = ltw(producer->get_output_value(0)->get_logical_tensor())
                              .vdims();
            ok = div_dst_shape == div_src0_shape
                    && std::accumulate(div_src1_shape.begin(),
                               div_src1_shape.end(), 1,
                               std::multiplies<int64_t>())
                            == 1;
            if (!ok) continue;

            to_be_swapped.emplace_back(
                    std::pair<impl::op_t *, impl::op_t *> {producer, op.get()});
        }

        if (to_be_swapped.empty()) break;

        for (auto &pair : to_be_swapped) {
            impl::op_t *div = pair.first;
            impl::op_t *mm = pair.second;

            auto div_src0 = div->get_input_value(0);
            auto div_dst = div->get_output_value(0);

            div_src0->remove_consumer(*div, 0);
            mm->connect_input(0, div_src0);

            auto mm_dst = mm->get_output_value(0);
            div->connect_output(0, mm_dst);

            impl::logical_tensor_t new_lt
                    = impl::empty_logical_tensor_with_default_id();
            auto new_val = std::make_shared<value_t>(*mm, 0, new_lt, true);
            new_val->set_data_type(
                    mm->get_input_value(0)->get_logical_tensor().data_type);
            mm->connect_output(0, new_val);
            div->connect_input(0, new_val);
        }
    }
    return infer_shape(sg);
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
