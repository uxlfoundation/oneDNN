/*******************************************************************************
* Copyright 2023 Intel Corporation
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
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>

#include "dnnl_common.hpp"
#include "input_displacer.hpp"
#include "ref_partition.hpp"
#include "ref_primitive.hpp"

#include "utils/parallel.hpp"

namespace graph {

namespace {

void handle_special_dt_set(
        ::graph::deserialized_op_t &op, const ::std::string &dt) {
    auto driver = op.opkind2driver();
    bool is_f8_quantization = (dt == "f8_e5m2" || dt == "f8_e4m3");

    if (op.in_lts_.size() > 1) {
        // Matmul/Conv/Deconv have limited support for quantized configurations.
        if (op.kind_ == "MatMul" || op.kind_ == "Convolution"
                || op.kind_ == "ConvTranspose") {
            if (dt == "u8") {
                // None of them supports u8u8, replace with u8s8.
                op.in_lts_[1].data_type_ = "s8";
            } else if (dt == "s4" || dt == "u4") {
                // None of them supports x4x4, replace with f32x4f32 or
                // xf16x4xf16.
                op.in_lts_[0].data_type_ = op.out_lts_[0].data_type_;
            }
        }
    }
    if (driver == dnnl_driver_t::pool || driver == dnnl_driver_t::binary
            || is_f8_quantization) {
        // pool does not support x8f32 on cpu, and binary does not support
        // x8x8bf16 on gpu, hence replace output with x8.
        // f8 data types needs setting output data type to f8
        op.out_lts_[0].data_type_ = dt;
    } else if (op.out_lts_[0].data_type_ != "bf16") {
        if (op.in_lts_.size() > 1 && op.in_lts_[1].data_type_ == "s8") {
            // Use u8 as output data type for two-input operations to avoid
            // data overflow due to the specific driver logic.
            op.out_lts_[0].data_type_ = "u8";
        } else {
            // Use f32 as output data type since not all primitives support
            // different data types for input and output.
            op.out_lts_[0].data_type_ = "f32";
        }
    }
}

::std::shared_ptr<ref_primitive_t> init_ref_prim_and_fill_data(
        const ::graph::deserialized_op_t &op, res_t *res) {
    auto ref_prim = ::std::make_shared<ref_primitive_t>(op);
    ref_prim->init_prb(res);
    if (res->state == INVALID_ARGUMENTS) return nullptr;

    ref_prim->init_prim(::get_test_engine(), res, /* force_override = */ true);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return nullptr;

    ref_prim->init_memory_args(::get_test_engine(), res);
    ref_prim->init_ref_memory_args(::get_test_engine(), res);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED
            || res->state == DEFERRED)
        return nullptr;
    return ref_prim;
}

} // namespace

sdpa_peaky_cfg_t sdpa_peaky_cfg_t::from_env() {
    sdpa_peaky_cfg_t c;
    const char *e = std::getenv("BENCHDNN_SDPA_FILL");
    if (!e || !*e) return c;

    const std::string s(e);
    const auto field = [&s](size_t idx) -> std::string {
        size_t p = 0;
        for (size_t i = 0; i < idx; i++) {
            p = s.find(':', p);
            if (p == std::string::npos) return "";
            p++;
        }
        const size_t q = s.find(':', p);
        return s.substr(p, q == std::string::npos ? q : q - p);
    };

    const std::string mode = field(0);
    if (mode == "sink_local")
        c.mode = mode_t::sink_local;
    else if (mode == "sink_group_local")
        c.mode = mode_t::sink_group_local;
    else {
        BENCHDNN_PRINT(0, "[DISPLACE]: BENCHDNN_SDPA_FILL: unknown mode %s, ignored.\n", mode.c_str());
        return c;
    }
    if (!field(1).empty()) c.gap_nats = std::stof(field(1));
    if (!field(2).empty()) c.group = std::stoll(field(2));
    if (c.group < 1) c.group = 1;
    if (const char *sc = std::getenv("BENCHDNN_SDPA_FILL_SCALE"))
        c.scale = (float)std::atof(sc);
    return c;
}

partition_data_displacer_t::partition_data_displacer_t(
        const deserialized_graph_t &dg, const dnnl::graph::partition &par)
    : dg_(&dg) {
    const auto &op_ids = par.get_ops();
    op_ids_set_ = std::unordered_set<size_t>(op_ids.begin(), op_ids.end());

    static const std::unordered_set<std::string> main_op_kind {"Convolution",
            "ConvTranspose", "AvgPool", "MaxPool", "MatMul", "Add", "Divide",
            "Maximum", "Minimum", "Multiply", "Subtract", "Select"};

    static const std::unordered_set<std::string> go_through_op_kind {
            "StaticTranspose", "StaticReshape", "TypeCast", "Quantize",
            "Dequantize"};

    static const std::unordered_set<std::string> f8_main_op_kind {
            "MatMul", "Convolution"};

    // The logic below relies on the assumption that deserialized_graph_t is
    // sorted in the chronological order.
    for (const auto &aop : dg_->ops_) {
        // Skip the check if op is not in the partition.
        if (op_ids_set_.find(aop.id_) == op_ids_set_.end()) continue;

        // Here is how quantize filling work
        //
        // partition input (lt)
        // |
        // [go through op]*
        // |
        // x<- quantize filling on this tensor (dq_lt)
        // |
        // dequantize <- The first dq met
        // |
        // [go through op except dq]*
        // |
        // main op (applied for all inputs the op has)

        if (main_op_kind.find(aop.kind_) == main_op_kind.end()) continue;

        // search along the branch for each input of main op
        for (size_t i = 0; i < aop.in_lts_.size(); i++) {
            // Traversing over a chain of allowed ops from the bottom to the
            // top searching for a first dequantize op in the chain.
            // Note: traversing can't be done on non-const references as
            // they will replace the starting point, but const references
            // can't be done because of assignment. So, pointers only.
            auto *parent_op = &aop;
            for (auto *lt = &aop.in_lts_[i]; true;
                    lt = &parent_op->in_lts_[0]) {
                parent_op = &dg_->get_op_by_out_lt(lt->id_);
                if (parent_op->empty()) {
                    if (aop.kind_ == "Divide") {
                        // Division has values > 1.f to reduce final values.
                        static const std::vector<float> user_set {
                                2.f, 4.f, 8.f};
                        // There's a special case for Divide, when second (user)
                        // input should be displaced with power-of-2 values.
                        displace_args_.emplace(lt->id_,
                                displace_args_t {aop, i, *lt,
                                        filling_type_t::fixed_setting,
                                        {user_set, "Div displacer"}});
                    } else if (aop.kind_ == "Multiply") {
                        // Multiplication has values <= 1.f to reduce final values.
                        static const std::vector<float> user_set {
                                0.25f, 0.5f, 1.f};
                        displace_args_.emplace(lt->id_,
                                displace_args_t {aop, i, *lt,
                                        filling_type_t::fixed_setting,
                                        {user_set, "Mul displacer"}});
                    }
                    break;
                }

                if (parent_op->kind_ == "DynamicDequantize"
                        && dg.get_recognized_pattern()
                                == graph_recognized_pattern_t::sdpa_fwd) {
                    // Add filling type for quantized input of SDPA cases
                    const auto &parent_op_in_lt = parent_op->in_lts_[0];
                    const auto &prev_parent_op
                            = dg_->get_op_by_out_lt(parent_op_in_lt.id_);
                    if (prev_parent_op.empty()
                            || op_ids_set_.find(prev_parent_op.id_)
                                    == op_ids_set_.end()) {

                        displace_args_.emplace(parent_op_in_lt.id_,
                                displace_args_t {aop, i, parent_op_in_lt,
                                        filling_type_t::compressed_sdpa});
                        break;
                    }
                }

                if (parent_op->kind_ == "Dequantize") {
                    // Dequantize is accepted when it doesn't have any
                    // predecessors in the partition (though it may have it in
                    // the graph).
                    const auto &parent_op_in_lt = parent_op->in_lts_[0];
                    const auto &prev_parent_op
                            = dg_->get_op_by_out_lt(parent_op_in_lt.id_);
                    if (prev_parent_op.empty()
                            || op_ids_set_.find(prev_parent_op.id_)
                                    == op_ids_set_.end()) {

                        // Skip input displacement for unusupported f8 ops.
                        const auto &lt_dt = parent_op_in_lt.get_data_type();
                        if ((lt_dt == logical_tensor::data_type::f8_e5m2
                                    || lt_dt
                                            == logical_tensor::data_type::
                                                    f8_e4m3)
                                && (f8_main_op_kind.find(aop.kind_)
                                        == f8_main_op_kind.end()))
                            break;

                        displace_args_.emplace(parent_op_in_lt.id_,
                                displace_args_t {aop, i, parent_op_in_lt,
                                        filling_type_t::quantization});
                        break;
                    }
                } else if (parent_op->kind_ == "StaticReshape") {
                    // StaticReshape is accepted when the pattern is
                    // "StaticReshape + Matmul" and it doesn't have any
                    // predecessors in the partition
                    const auto &parent_op_in_lt = parent_op->in_lts_[0];
                    const auto &prev_parent_op
                            = dg_->get_op_by_out_lt(parent_op_in_lt.id_);
                    if (prev_parent_op.empty()
                            || op_ids_set_.find(prev_parent_op.id_)
                                    == op_ids_set_.end()) {
                        if (aop.kind_ == "MatMul") {
                            displace_args_.emplace(parent_op_in_lt.id_,
                                    displace_args_t {aop, i, parent_op_in_lt,
                                            filling_type_t::quantization});
                        }
                        break;
                    }
                }
                // Continue only on allowed ops.
                if (go_through_op_kind.find(parent_op->kind_)
                        == go_through_op_kind.end()) {
                    break;
                }
            }
        }

        // Alternatively, looking for Add->SoftMax chain, which represents
        // explicit SDPA mask, and should be filled with upper-corner with -inf:
        // 0 -inf -inf -inf
        // 0    0 -inf -inf
        // 0    0    0 -inf
        // 0    0    0    0
        // This is done to avoid taking future tokens into account by
        // influencing SoftMax input values.
        while (aop.kind_ == "Add" || aop.kind_ == "Select") {
            // TODO: consider adding `dg.has_known_patterns()` when the list gets bigger.
            if (dg.get_recognized_pattern()
                            != graph_recognized_pattern_t::sdpa_fwd
                    && dg.get_recognized_pattern()
                            != graph_recognized_pattern_t::sdpa_bwd)
                break;
            auto *aop_out_lt = &aop.out_lts_[0];
            // s32 add is a case for bottom-right causal mask
            if (aop.kind_ == "Add" && aop_out_lt->data_type_ == "s32") break;

            // The following op (Softmax or Subtract) must be a part of same
            // partition as the mask. This is to avoid cases, where mask is the
            // last op in the partition, from being modified.
            auto *child_op = &dg_->get_op_by_in_lt(aop_out_lt->id_);
            if (child_op->kind_ != "SoftMax" && child_op->kind_ != "Subtract")
                break;
            if (op_ids_set_.find(child_op->id_) == op_ids_set_.end()) break;

            // Search for an input lt without a parent, this is the one to
            // modify for both explicit and implicit masks.
            const deserialized_lt_t *causal_mask_lt = nullptr;
            size_t offset = SIZE_MAX;
            size_t qk_data_offset = SIZE_MAX;
            // Select condition having a parent or not is the only reliable
            // difference between explicit and implicit causal mask.
            bool select_cond_has_parent = false;
            // Need to iterate over all inputs to handle padding mask expressed
            // through Select op.
            for (size_t i = 0; i < aop.in_lts_.size(); i++) {
                auto *aop_in_lt = &aop.in_lts_[i];
                auto *parent_op = &dg_->get_op_by_out_lt(aop_in_lt->id_);
                if (!parent_op->empty()) {
                    if (aop_in_lt->get_data_type()
                            != logical_tensor::data_type::boolean) {
                        // This is the qk_data, need to know its offset to
                        // properly fill condition for padding mask.
                        qk_data_offset = i;
                    } else {
                        // This means it's implicit causal mask.
                        select_cond_has_parent = true;
                    }
                    continue;
                }

                // Explicit padding mask expressed through the Select op would
                // have two user inputs: condition, hinting where padding
                // occurred and a special value (-inf) to use. In such scenario,
                // unlike for implicit causal mask, it's required to update the
                // condition to always take qk values instead of a special one.
                //
                // Checking for data type to make sure that in case of two user
                // inputs, the condition one will be updated. For implicit
                // causal mask, the condition would have a parent and a check
                // for `causal_mask_lt` being non-empty will fail.
                if (causal_mask_lt
                        && aop_in_lt->get_data_type()
                                != logical_tensor::data_type::boolean)
                    continue;

                causal_mask_lt = aop_in_lt;
                offset = i;
            }
            // No suitable tensor/subgraph for a mask displacement.
            if (!causal_mask_lt) break;

            filling_type_t filling_type = filling_type_t::undef;
            std::string cfg_name;
            float user_set_value = 0.f;
            if (aop.kind_ == "Add") {
                const auto ndims = causal_mask_lt->shape_.size();
                if (ndims < 2) {
                    BENCHDNN_PRINT(7, "%s\n",
                            "[DISPLACE]: Causal mask ndims is less than 2");
                    break;
                }

                const auto M = causal_mask_lt->shape_[ndims - 2];
                if (M == 1) {
                    // This is a padding mask case, when padded tokens should
                    // be removed from the final computations. In case of
                    // benchdnn, there's no such thing as padding as all tokens
                    // are computed. To avoid numerical instabilities, a zero
                    // mask can be applied without compromising validation
                    // capabilities.
                    filling_type = filling_type_t::fixed_setting;
                    cfg_name = "Explicit_padding_mask";
                } else {
                    // This is a look-ahead (or causal) mask case, when future
                    // tokens (row < col) are set to infinity to remove all
                    // connections of current tokens to unissued ones.
                    filling_type = filling_type_t::causal_mask;
                }
            } else if (aop.kind_ == "Select") {
                if (select_cond_has_parent) {
                    // Implicit causal mask case.
                    filling_type = filling_type_t::fixed_setting;
                    user_set_value = -INFINITY;
                    cfg_name = "Implicit_causal_mask";
                } else {
                    // Padding mask.
                    assert(qk_data_offset == 1 || qk_data_offset == 2);
                    // Fill condition depending on qk values tensor to use only
                    // its values, which is equivalent of not using a mask.
                    filling_type = filling_type_t::fixed_setting;
                    if (qk_data_offset == 1) {
                        user_set_value = 1.f;
                    } else if (qk_data_offset == 2) {
                        user_set_value = 0.f;
                    }
                    cfg_name = "Explicit_padding_mask";
                }
            }

            if (filling_type == filling_type_t::undef) {
                BENCHDNN_PRINT(
                        7, "%s\n", "[DISPLACE]: Filling type was not set");
                break;
            } else if (filling_type == filling_type_t::fixed_setting) {
                displace_args_.emplace(causal_mask_lt->id_,
                        displace_args_t {aop, offset, *causal_mask_lt,
                                filling_type, {{user_set_value}, cfg_name}});
            } else if (filling_type == filling_type_t::causal_mask) {
                // Causal mask filling
                displace_args_.emplace(causal_mask_lt->id_,
                        displace_args_t {
                                aop, offset, *causal_mask_lt, filling_type});
            }
            break;
        }

        // Fill proper data for bottom-right implicit casual mask
        while (aop.kind_ == "Add") {
            if (dg.get_recognized_pattern()
                            != graph_recognized_pattern_t::sdpa_fwd
                    && dg.get_recognized_pattern()
                            != graph_recognized_pattern_t::sdpa_bwd)
                break;
            auto *aop_out_lt = &aop.out_lts_[0];
            // add in a bottom-right causal mask should have s32 output dtype
            if (aop_out_lt->data_type_ != "s32") break;

            auto *child_sub_op = &dg_->get_op_by_in_lt(aop_out_lt->id_);
            if (child_sub_op->kind_ != "Subtract") break;

            auto *child_op_out_lt = &child_sub_op->out_lts_[0];
            auto *next_child_op = &dg_->get_op_by_in_lt(child_op_out_lt->id_);
            if (next_child_op->kind_ != "GreaterEqual") break;

            const std::string cfg_name = "Bottom_right_implicit_padding_mask";
            static constexpr int seq_len_q = 0;
            static constexpr int seq_len_kv = 1;

            // The following subtract and greaterEqual must also be a part of
            // the partition.
            if (op_ids_set_.find(child_sub_op->id_) == op_ids_set_.end()
                    || op_ids_set_.find(next_child_op->id_)
                            == op_ids_set_.end())
                break;

            const auto set_seq_len_displace_args
                    = [&](const deserialized_op_t *op, int which_seq_len) {
                const size_t ndims = op->out_lts_[0].shape_.size();
                const size_t seq_len_idx
                        = (which_seq_len == seq_len_q) ? ndims - 2 : ndims - 1;

                for (size_t i = 0; i < op->in_lts_.size(); i++) {
                    auto *parent_op
                            = &dg_->get_op_by_out_lt(op->in_lts_[i].id_);
                    // For add->sub->ge, we consider the inputs of add
                    // and sub as scalars if they have no parent
                    // tensors.
                    if (parent_op->empty()) {
                        float user_set_value = static_cast<float>(
                                op->in_lts_[1 - i].shape_[seq_len_idx]);
                        displace_args_.emplace(op->in_lts_[i].id_,
                                displace_args_t {*op, i, op->in_lts_[i],
                                        filling_type_t::fixed_setting,
                                        {{user_set_value}, cfg_name}});
                    }
                }
            };

            // The bottom-right implicit causal mask handles future tokens
            // differently compared to the top-left casual mask. To support
            // it, the result of `GenIndex` on rows should subtract `seq_len_q`
            // and add `seq_len_kv` to generate masks such as:
            // # s_q=2, s_kv=5            |    # s_q=5, s_kv=2
            //  0    0    0    0  -inf    |      -inf  -inf
            //  0    0    0    0    0     |      -inf  -inf
            //                            |      -inf  -inf
            //                            |        0   -inf
            //                            |        0    0
            // Add the sequence length of Key and Value.
            set_seq_len_displace_args(&aop, seq_len_kv);
            // Subtract the sequence lenght of Query.
            set_seq_len_displace_args(child_sub_op, seq_len_q);
            break;
        }

        // Fill proper data for softmax stats in sdpa backward graph.
        while (aop.kind_ == "Subtract") {
            if (dg.get_recognized_pattern()
                    != graph_recognized_pattern_t::sdpa_bwd)
                break;
            // for softmax stats, it's used as P = exp(S-stats)
            // stats should be an input of the whole backward graph, so it should
            // have no producer.
            auto *aop_in_lt = &aop.in_lts_[1];
            auto *parent_op = &dg_->get_op_by_out_lt(aop_in_lt->id_);
            if (!parent_op->empty()) break;

            // subtract should be followed by exp to resume a softmax functionality.
            auto *aop_out_lt = &aop.out_lts_[0];
            auto *child_exp_op = &dg_->get_op_by_in_lt(aop_out_lt->id_);
            if (child_exp_op->kind_ != "Exp") break;

            displace_args_.emplace(aop_in_lt->id_,
                    displace_args_t {
                            aop, 1, *aop_in_lt, filling_type_t::softmax_stats});
            break;
        }

        // Fill proper data for gated MLP shared activations.
        while (aop.kind_ == "MatMul") {
            if (dg.get_recognized_pattern() != graph_recognized_pattern_t::gmlp)
                break;

            // Bold assumption that first in_lts is shared.
            // TODO: query the index from dg, maybe?
            auto *aop_in_lt = &aop.in_lts_[0];

            // Don't update Down MatMul.
            auto *parent_op = &dg_->get_op_by_out_lt(aop_in_lt->id_);
            if (!parent_op->empty()) break;

            // Don't submit another displacer for same input.
            if (displace_args_.find(aop_in_lt->id_) != displace_args_.end())
                break;

            // Fill activations very-very sparsely with small pow-2 values:
            // (2^-12 ... 2^-14). The rationale is keep MatMul output values
            // small in absolute value to avoid Multiply output values severely
            // increase since it can be a square value of a MatMul output. This
            // might happen because weights will be filled identically as they
            // have identical shape, activations are shared between Gate and Up,
            // unary activation (GeLU, etc.) may keep positive value as is.
            static const std::vector<float> user_set {
                    1.f / 4096.f, 1.f / 8192.f, 1.f / 16384.f};
            // Use roughly 3 non-zero elements per K dim.
            const auto &shape = aop_in_lt->shape_;
            const auto K = shape[shape.size() - 1];
            const float density = 3.f / K;
            displace_args_.emplace(aop_in_lt->id_,
                    displace_args_t {aop, 0, *aop_in_lt,
                            filling_type_t::fixed_setting,
                            {user_set, density,
                                    "GMLP MatMul Activation displacer"}});
            break;
        }
    }

    // BENCHDNN_SDPA_FILL: peaky Q/K filling, applied once per partition to the
    // two graph inputs of the QK MatMul.
    peaky_ = sdpa_peaky_cfg_t::from_env();
    while (peaky_.mode != sdpa_peaky_cfg_t::mode_t::none) {
        if (dg.get_recognized_pattern()
                != graph_recognized_pattern_t::sdpa_fwd) {
            BENCHDNN_PRINT(0, "%s\n",
                    "[DISPLACE]: BENCHDNN_SDPA_FILL: not an sdpa_fwd pattern, ignored.");
            break;
        }
        // Ops are in chronological order, so the QK MatMul is the first one
        // whose both inputs are graph inputs; the VS MatMul consumes SoftMax.
        const deserialized_op_t *qk = nullptr;
        for (const auto &aop : dg_->ops_) {
            if (aop.kind_ != "MatMul" || aop.in_lts_.size() < 2) continue;
            if (op_ids_set_.find(aop.id_) == op_ids_set_.end()) continue;
            if (!dg_->get_op_by_out_lt(aop.in_lts_[0].id_).empty()) continue;
            if (!dg_->get_op_by_out_lt(aop.in_lts_[1].id_).empty()) continue;
            qk = &aop;
            break;
        }
        if (!qk) {
            BENCHDNN_PRINT(0, "%s\n",
                    "[DISPLACE]: BENCHDNN_SDPA_FILL: QK MatMul with plain Q/K graph inputs not found, ignored.");
            break;
        }
        const auto &q_lt = qk->in_lts_[0];
        const auto &k_lt = qk->in_lts_[1];
        const size_t nd = q_lt.shape_.size();
        if (nd < 3 || nd != k_lt.shape_.size()) {
            BENCHDNN_PRINT(0, "%s\n",
                    "[DISPLACE]: BENCHDNN_SDPA_FILL: unsupported Q/K ranks, ignored.");
            break;
        }
        bool transpose_b = false;
        qk->get_attr_bool(transpose_b, "transpose_b");
        peaky_.q_shape = q_lt.shape_;
        peaky_.k_shape = k_lt.shape_;
        peaky_.k_is_sd = transpose_b;
        peaky_.q_lt_id = q_lt.id_;
        peaky_.k_lt_id = k_lt.id_;
        displace_args_.emplace(q_lt.id_,
                displace_args_t {*qk, 0, q_lt, filling_type_t::sdpa_peaky});
        displace_args_.emplace(k_lt.id_,
                displace_args_t {*qk, 1, k_lt, filling_type_t::sdpa_peaky});

        // The scale is otherwise filled at random from {0.25, 0.5, 1.0}, which
        // rescales every logit gap and makes gap_nats meaningless. Pin it, and
        // use the same value when sizing the peaks. operator[] overrides the
        // generic Mul/Div displacer already registered for this tensor.
        const int64_t d = q_lt.shape_[nd - 1];
        if (peaky_.scale <= 0) peaky_.scale = 1.f / std::sqrt((float)d);
        const auto &sop = dg_->get_op_by_in_lt(qk->out_lts_[0].id_);
        if (!sop.empty() && (sop.kind_ == "Multiply" || sop.kind_ == "Divide")) {
            for (size_t i = 0; i < sop.in_lts_.size(); i++) {
                const auto &slt = sop.in_lts_[i];
                if (slt.id_ == qk->out_lts_[0].id_) continue;
                if (!dg_->get_op_by_out_lt(slt.id_).empty()) continue;
                const float sv = (sop.kind_ == "Divide") ? 1.f / peaky_.scale
                                                         : peaky_.scale;
                displace_args_[slt.id_] = displace_args_t {sop, i, slt,
                        filling_type_t::fixed_setting,
                        {{sv}, "SDPA peaky scale"}};
                peaky_.scale_lt_id = slt.id_;
                peaky_.has_scale_lt = true;
                break;
            }
        }
        break;
    }
}

int partition_data_displacer_t::displace_input_data(size_t lt_id,
        const std::unordered_map<size_t, const dnn_mem_t &> &lt_id_2_mems,
        res_t *res) {
    if (!dg_) {
        res->state = FAILED;
        return FAIL;
    }

    if (displace_args_.find(lt_id) == displace_args_.end()) {
        // no need to displace the data of this tensor
        return OK;
    }
    const displace_args_t &d_args = displace_args_.at(lt_id);
    dnn_mem_t &mem = const_cast<dnn_mem_t &>(lt_id_2_mems.at(lt_id));
    const auto &main_op = d_args.main_op_;
    const auto &main_op_offset = d_args.main_op_offset_;
    const auto &tensor = d_args.tensor_;
    const auto &fill_cfg = d_args.fill_cfg_;
    const auto filling_type = d_args.filling_type_;

    auto opkind = opstr2kind(main_op.kind_);
    int main_op_arg = get_prim_arg_name_from_graph_op_input_offset(
            opkind, main_op_offset);

    const auto &get_name = [&filling_type, &fill_cfg]() {
        std::string s;
        if (filling_type == filling_type_t::fixed_setting) {
            s = fill_cfg.name_;
        } else if (filling_type == filling_type_t::causal_mask) {
            s = "Explicit causal mask";
        } else if (filling_type == filling_type_t::quantization) {
            s = "Quantization";
        } else if (filling_type == filling_type_t::compressed_sdpa) {
            s = "Compressed SDPA";
        } else if (filling_type == filling_type_t::sdpa_peaky) {
            s = "SDPA peaky Q/K";
        }
        return s;
    };
    BENCHDNN_PRINT(3, "[DISPLACE]: Op:%s; Arg:%s; Name:%s;\n",
            main_op.kind_.c_str(),
            data_kind2str(exec_arg2data_kind(main_op_arg)), get_name().c_str());

    dnn_mem_t mem_replace;
    if (filling_type == filling_type_t::quantization) {
        SAFE(gen_quantize_filling(
                     main_op, main_op_arg, mem_replace, tensor.data_type_, res),
                WARN);
    } else if (filling_type == filling_type_t::compressed_sdpa) {
        SAFE(gen_compressed_sdpa_filling(
                     main_op, main_op_arg, mem_replace, tensor.data_type_, res),
                WARN);
    } else if (filling_type == filling_type_t::causal_mask) {
        SAFE(gen_causal_mask_filling(mem_replace, mem.md_, res), WARN);
    } else if (filling_type == filling_type_t::fixed_setting) {
        SAFE(gen_fixed_set_filling(mem_replace, mem.md_, fill_cfg, res), WARN);
    } else if (filling_type == filling_type_t::softmax_stats) {
        const auto *softmax_src_lt = &main_op.in_lts_[0];
        const dnn_mem_t &softmax_src_mem = lt_id_2_mems.at(softmax_src_lt->id_);
        SAFE(gen_softmax_stats_filling(main_op, main_op_arg, softmax_src_mem,
                     mem_replace, mem.md_, res),
                WARN);
    } else if (filling_type == filling_type_t::sdpa_peaky) {
        const size_t peer_id
                = (lt_id == peaky_.q_lt_id) ? peaky_.k_lt_id : peaky_.q_lt_id;
        const auto peer_it = lt_id_2_mems.find(peer_id);
        if (peer_it == lt_id_2_mems.end()) {
            BENCHDNN_PRINT(0, "%s\n",
                    "[DISPLACE]: BENCHDNN_SDPA_FILL: peer Q/K memory missing.");
            res->state = FAILED;
            return FAIL;
        }
        SAFE(gen_sdpa_peaky_filling(
                     lt_id, peer_it->second.md_, mem_replace, mem.md_, res),
                WARN);
    } else {
        assert(!"unexpected filling type");
    }

    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    // do the reverse job
    auto *parent_op = &dg_->get_op_by_out_lt(tensor.id_);
    bool backward_path_launched = false;
    while (filling_type == filling_type_t::quantization && !parent_op->empty()
            && op_ids_set_.find(parent_op->id_) != op_ids_set_.end()) {
        backward_path_launched = true;
        // generate the reverse op based on OP kind
        // make a copy of deserialized_op_t to avoid impact on graph execution
        // Currently, we support the following OPs' reverse execution:
        // All of the execution need to swap the input lt and output lt first

        // 1. StaticTranspose: re-permute the 'order' attr to get an inversed effect
        // 2. TypeCast: Do nothing special because the type is already swapped
        // 3. StaticReshape: Do nothing special because the shape is already swapped
        // 4. Quantize: change opkind to Dequantize and keep scales and zps
        // 5. Dequantize: change opkind to Quantize and keep scales and zps

        auto op = dg_->get_op_by_out_lt(tensor.id_);
        BENCHDNN_PRINT(
                3, "[DISPLACE]: Backward path for Op:%s;\n", op.kind_.c_str());

        ::std::swap(op.in_lts_, op.out_lts_);

        auto opkind = opstr2kind(op.kind_);

        switch (opkind) {
            case ::graph::op::kind::Quantize: op.kind_ = "Dequantize"; break;
            case ::graph::op::kind::Dequantize: op.kind_ = "Quantize"; break;
            case ::graph::op::kind::StaticTranspose: {
                ::std::vector<int64_t> order;
                op.get_attr_s64_vector(order, "order");
                const size_t ndims = order.size();
                op.attrs_["order"].s64_vector_
                        = ::std::vector<int64_t>(ndims, 0);
                for (size_t i = 0; i < ndims; i++) {
                    op.attrs_["order"].s64_vector_[(order[i] + ndims) % ndims]
                            = i;
                }
                break;
            }
            case ::graph::op::kind::TypeCast:
            case ::graph::op::kind::StaticReshape: break;
            default:
                assert(!"not support opkind for reverse execution");
                return FAIL;
        }

        // execute the reverse op
        res_t res {};

        ref_primitive_t ref_prim(op);
        ref_prim.init_prb(&res);
        if (res.state == INVALID_ARGUMENTS) return FAIL;
        SAFE_V(ref_prim.init_prim(
                get_cpu_engine(), &res, /* force_override = */ true));

        ref_prim.init_memory_args(get_cpu_engine(), &res);
        SAFE_V(ref_prim.init_ref_memory_args(get_cpu_engine(), &res));

        const auto &src_mem = ref_prim.get_arg(DNNL_ARG_SRC);
        bool mds_are_equal
                = dnnl_memory_desc_equal(mem_replace.md_, src_mem.md_) == 1;
        SAFE(mds_are_equal ? OK : FAIL, WARN);

        // Always use the md generated by current reversed op. E.g., a matmul op
        // will unsqeeze 1 to fit the dimension so the md generated by matmul
        // prb_t will not be the same as defined in graph.
        dnnl_memory_desc_destroy(mem_replace.md_);
        dnnl_memory_desc_clone(&mem_replace.md_, src_mem.md_);
        ref_prim.replace_arg(DNNL_ARG_SRC, mem_replace);
        SAFE_V(ref_prim.execute_prim(&res));

        mem_replace = ::std::move(
                const_cast<dnn_mem_t &>(ref_prim.get_arg(DNNL_ARG_DST)));
        parent_op = &dg_->get_op_by_out_lt(op.out_lts_[0].id_);
    }

    if (backward_path_launched) {
        BENCHDNN_PRINT(3, "%s\n", "[DISPLACE]: Backward path ended.");
    }

    do {
        const bool mds_are_equal
                = dnnl_memory_desc_equal(mem_replace.md_, mem.md_) == 1;
        if (mds_are_equal) {
            SAFE(mem.reorder(mem_replace, res), WARN);
            break;
        }

        // Below are valid cases when `mem_replace.md_` and `mem.md_` might not
        // be equal yet valid.
        //
        // Case: Int8/Int4 descriptors are interchangeable. Treat filled data
        // as mem.dt() and just reorder one to the other.
        const bool mds_are_int8 = is_integral_dt(mem_replace.dt())
                && is_integral_dt(mem.dt()) && mem_replace.sizeof_dt() == 1
                && mem.sizeof_dt() == 1;

        const bool mds_are_fp8
                = is_fp8_dt(mem_replace.dt()) && mem_replace.dt() == mem.dt();
        if (mds_are_int8 || mds_are_fp8) {
            dnnl_memory_desc_destroy(mem_replace.md_);
            dnnl_memory_desc_clone(&mem_replace.md_, mem.md_);
            SAFE(mem.reorder(mem_replace, res), WARN);
            break;
        }

        // Case w/ grouped convolutions when number of dimensions would be +1.
        if (main_op.kind_ == "Convolution"
                || main_op.kind_ == "ConvTranspose") {
            int64_t groups = 0;
            main_op.get_attr_s64(groups, "groups");
            if (groups > 1) {
                dnnl_memory_desc_destroy(mem_replace.md_);
                dnnl_memory_desc_clone(&mem_replace.md_, mem.md_);
                SAFE(mem.reorder(mem_replace, res), WARN);
                break;
            }
        }

        // Case when there're extra unit dims in replaced memory. Memory buffers
        // are identical but different ndims are restricted in reorder API.
        // `mem_replace.md_` requires manual adjustment before reordering.
        const bool is_reshaped_dims = mem_replace.nelems() == mem.nelems()
                && mem_replace.ndims() != mem.ndims();
        if (is_reshaped_dims) {
            dnnl_memory_desc_t new_replace_md {};
            DNN_SAFE_V(dnnl_memory_desc_create_with_strides(&new_replace_md,
                    mem.ndims(), mem.dims(), mem_replace.dt(), mem.strides()));
            dnnl_memory_desc_destroy(mem_replace.md_);
            dnnl_memory_desc_clone(&mem_replace.md_, new_replace_md);
            dnnl_memory_desc_destroy(new_replace_md);
            SAFE(mem.reorder(mem_replace, res), WARN);
            break;
        }

        // Case when there're non-dense strides in `mem.md_` leading to "holes"
        // in the data. To avoid dealing with all kinds of strides,
        // `mem_replace` was filled as regular dense-strided memory letting a
        // reorder to handle dense to srided data conversion.
        const bool has_non_dense_strides
                = !mem.is_dense() && mem_replace.is_dense();
        if (has_non_dense_strides) {
            dnnl_memory_desc_t new_replace_md {};
            DNN_SAFE_V(dnnl_memory_desc_create_with_strides(&new_replace_md,
                    mem.ndims(), mem.dims(), mem_replace.dt(),
                    mem_replace.strides()));
            dnnl_memory_desc_destroy(mem_replace.md_);
            dnnl_memory_desc_clone(&mem_replace.md_, new_replace_md);
            dnnl_memory_desc_destroy(new_replace_md);
            SAFE(mem.reorder(mem_replace, res), WARN);
            break;
        }

        // Non of valid cases were identified.
        SAFE(FAIL, WARN);
    } while (false);

    return OK;
}

int partition_data_displacer_t::gen_compressed_sdpa_filling(
        const ::graph::deserialized_op_t &main_op, int arg, dnn_mem_t &mem,
        const ::std::string &dt, res_t *res) {
    if (!(arg & DNNL_ARG_WEIGHTS)) return FAIL;
    // clone a deserialized op object and modify to specified data type
    ::graph::deserialized_op_t op = main_op;
    bool s8_mem_for_u8_wei = dt == "u8";
    op.in_lts_[0].data_type_ = dt;
    op.in_lts_[1].data_type_ = dt;

    if (dt == "u8") {
        // None of them supports u8u8, replace with u8s8.
        op.in_lts_[1].data_type_ = "s8";
    } else if (dt == "s4" || dt == "u4") {
        // None of them supports x4x4, replace with f32x4f32 or
        // xf16x4xf16.
        op.in_lts_[0].data_type_ = op.out_lts_[0].data_type_;
    }

    if (op.out_lts_[0].data_type_ != "bf16") {
        if (op.in_lts_[1].data_type_ == "s8") {
            // Use u8 as output data type for two-input operations to avoid
            // data overflow due to the specific driver logic.
            op.out_lts_[0].data_type_ = "u8";
        } else {
            // Use f32 as output data type since not all primitives support
            // different data types for input and output.
            op.out_lts_[0].data_type_ = "f32";
        }
    }

    auto ref_prim_ptr = init_ref_prim_and_fill_data(op, res);
    if (!ref_prim_ptr) {
        if (res->state == SKIPPED || res->state == UNIMPLEMENTED
                || res->state == DEFERRED)
            return OK;
        else
            return FAIL;
    }

    auto &gen_mem = const_cast<dnn_mem_t &>(ref_prim_ptr->get_arg(arg));
    if (s8_mem_for_u8_wei) {
        // If s8 data is directly read using the u8 data type, it may lead to
        // overflow issues. For complex patterns like SDPA, this could result
        // in precision degradation. Using a reorder to convert negative values
        // into zeros.
        dnn_mem_t gen_u8_mem(gen_mem, dnnl_u8, tag::abx, gen_mem.engine());
        mem = ::std::move(gen_u8_mem);
    } else
        mem = ::std::move(gen_mem);

    // Reduce data range to avoid false positive results
    // Note: traversing over the mem data twice, which is bad  for
    // performance but doesn't require dealing with external
    // data filling configuration.
    static constexpr int64_t chunk_size = 64;
    const int64_t n_chunks = div_up(mem.nelems(), chunk_size);
    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, mem.nelems());

        // TODO(Zhitao): Adjust data filling strategy based on problem
        // configuration.
        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            int value = static_cast<int32_t>(mem.get_elem(idx));
            mem.set_elem(idx, value / 2);
        }
    });
    return OK;
}

int partition_data_displacer_t::gen_quantize_filling(
        const ::graph::deserialized_op_t &main_op, int arg, dnn_mem_t &mem,
        const ::std::string &dt, res_t *res) {
    // clone a deserialized op object and modify to specified data type
    ::graph::deserialized_op_t op = main_op;
    op.in_lts_[0].data_type_ = dt;
    if (op.in_lts_.size() > 1) op.in_lts_[1].data_type_ = dt;

    handle_special_dt_set(op, dt);
    auto ref_prim = init_ref_prim_and_fill_data(op, res);

    auto &gen_mem = const_cast<dnn_mem_t &>(ref_prim->get_arg(arg));
    mem = ::std::move(gen_mem);
    return OK;
}

int partition_data_displacer_t::gen_fixed_set_filling(dnn_mem_t &mem,
        const_dnnl_memory_desc_t md, const fill_cfg_t &fill_cfg,
        res_t *res) const {

    dnn_mem_t m(md, get_test_engine(), /* prefill = */ false);
    if (!m.is_dense()) {
        m = dnn_mem_t(query_md_ndims(md), query_md_dims(md), dnnl_f32, tag::abx,
                get_test_engine(), /* prefill = */ false);
    }
    // Perf mode hands over unmapped device memory; restore the incoming state
    // so a later reorder can map it itself.
    const bool mapped_here = !m.is_mapped();
    if (mapped_here) m.map();
    const int64_t nelems = m.nelems();

    BENCHDNN_PRINT(6, "%s\n", fill_cfg.print_verbose().c_str());

    const auto &vals = fill_cfg.predefined_set_;
    const int n_vals = static_cast<int>(vals.size());

    /* Do fixed partitioning to have same filling for any number of threads */
    static constexpr int64_t chunk_size = 64;
    const int64_t n_chunks = div_up(nelems, chunk_size);
    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        // Note: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // it has a complexity in O(idx_start). We also add 1 to avoid
        // seeding with 0.
        std::minstd_rand int_seed(idx_start + 1);
        int_seed.discard(1);

        std::uniform_int_distribution<> gen(0, n_vals - 1);
        std::bernoulli_distribution b_dist(fill_cfg.density_);

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            bool is_one = fill_cfg.density_ == 1.f ? true : b_dist(int_seed);
            if (!is_one) {
                m.set_elem(idx, 0.f);
                continue;
            }
            const float val = vals[gen(int_seed)];
            m.set_elem(idx, val);
        }
    });

    if (mapped_here) m.unmap();
    mem = std::move(m);
    return OK;
}

int partition_data_displacer_t::gen_causal_mask_filling(
        dnn_mem_t &mem, const_dnnl_memory_desc_t md, res_t *res) const {

    dnn_mem_t tmp_mem(md, get_test_engine(), /* prefill = */ false);
    if (!tmp_mem.is_dense()) {
        tmp_mem = dnn_mem_t(query_md_ndims(md), query_md_dims(md), dnnl_f32,
                tag::abx, get_test_engine(), /* prefill = */ false);
    }

    const int ndims = query_md_ndims(md);
    assert(ndims >= 2); // This was checked at displacer initialization.
    const auto &dims = query_md_dims(md);
    const int64_t batch = std::accumulate(dims, dims + ndims - 2, (dnnl_dim_t)1,
            std::multiplies<dnnl_dim_t>());
    const int64_t M = dims[ndims - 2];
    const int64_t N = dims[ndims - 1];

    benchdnn_parallel_nd(batch, M, N, [&](int64_t b, int64_t m, int64_t n) {
        int64_t idx = b * M * N + m * N + n;
        float val = m >= n ? 0.f : -INFINITY;
        // The line below masks out the whole prompt to verify the softmax
        // output returns zeroes, not NaNs, as expected by PyTorch.
        if (m == M - 1) val = -INFINITY;
        tmp_mem.set_elem(idx, val);
    });

    mem = std::move(tmp_mem);
    return OK;
}

int partition_data_displacer_t::gen_sdpa_peaky_filling(size_t lt_id,
        const_dnnl_memory_desc_t peer_md, dnn_mem_t &mem,
        const_dnnl_memory_desc_t md, res_t *res) const {
    const auto &c = peaky_;
    const bool is_q = (lt_id == c.q_lt_id);

    // Keep the md identical to the destination so the caller can reorder.
    dnn_mem_t tmp_mem(md, get_test_engine(), /* prefill = */ false);
    // Perf mode hands over unmapped device memory; corr mode is already mapped.
    // Restore whatever state it came in with, so the reorder can map it itself.
    const bool mapped_here = !tmp_mem.is_mapped();
    if (mapped_here) tmp_mem.map();

    const int ndims = tmp_mem.ndims();
    if (query_md_ndims(peer_md) != ndims) {
        BENCHDNN_PRINT(0, "%s\n",
                "[DISPLACE]: BENCHDNN_SDPA_FILL: Q/K rank mismatch.");
        res->state = FAILED;
        return FAIL;
    }
    const int nb = ndims - 2; // leading batch/head dims
    // Index through strides: these tensors are often stored permuted (B,S,H,D).
    const auto *strides = tmp_mem.strides();
    // Dims must come from the memory, not the graph: --in-shapes rewrites the
    // memory but the deserialized logical tensor keeps the original shape.
    const dnnl_dim_t *self_dims = query_md_dims(md);
    const dnnl_dim_t *peer_dims = query_md_dims(peer_md);
    const dnnl_dim_t *q_dims = is_q ? self_dims : peer_dims;
    const dnnl_dim_t *k_dims = is_q ? peer_dims : self_dims;

    const int64_t D = q_dims[ndims - 1];
    const int64_t Sq = q_dims[ndims - 2];
    // Infer K's layout from the descriptor rather than trusting transpose_b:
    // the MatMul's WEI md may already present K as (..., D, S).
    bool k_sd = c.k_is_sd;
    if (k_dims[ndims - 1] == D && k_dims[ndims - 2] != D)
        k_sd = true;
    else if (k_dims[ndims - 2] == D && k_dims[ndims - 1] != D)
        k_sd = false;
    const int64_t Sk = k_sd ? k_dims[ndims - 2] : k_dims[ndims - 1];

    int64_t q_batch = 1, k_batch = 1;
    for (int i = 0; i < nb; i++) {
        q_batch *= q_dims[i];
        k_batch *= k_dims[i];
    }

    const float scale = c.scale > 0 ? c.scale : 1.f / std::sqrt((float)D);
    const float cmax = 3.f / std::sqrt((float)D);
    const float denom = scale * (1.f - cmax);
    if (!(denom > 0.f)) {
        BENCHDNN_PRINT(0, "%s\n",
                "[DISPLACE]: BENCHDNN_SDPA_FILL: degenerate head size.");
        res->state = UNIMPLEMENTED;
        return OK;
    }
    const float r = std::sqrt(c.gap_nats / denom);

    // Deterministic unit vector for key j of kv-slice kb; Q and K must agree.
    const auto unit = [&](int64_t kb, int64_t j, std::vector<float> &u) {
        std::mt19937 g((uint32_t)((uint64_t)kb * 1000003u
                + (uint64_t)j * 2654435761u + 0x9e3779b9u));
        std::normal_distribution<float> nd(0.f, 1.f);
        u.resize(D);
        double nrm = 0;
        for (int64_t d = 0; d < D; d++) {
            u[d] = nd(g);
            nrm += (double)u[d] * u[d];
        }
        const float inv = nrm > 0 ? 1.f / (float)std::sqrt(nrm) : 1.f;
        for (int64_t d = 0; d < D; d++)
            u[d] *= inv;
    };

    const auto own = [&](int64_t i) {
        int64_t j = i + (Sk - Sq);
        if (c.mode == sdpa_peaky_cfg_t::mode_t::sink_group_local)
            j = (j / c.group) * c.group;
        return std::min(std::max<int64_t>(j, 0), Sk - 1);
    };

    if (is_q) {
        benchdnn_parallel_nd(q_batch, Sq, [&](int64_t qb, int64_t i) {
            // Decompose the flat batch index into per-dim coords, and fold it
            // into the matching kv slice (K has 1 where Q carries the group).
            int64_t off = 0, rem = qb, kb = 0, kmul = 1;
            for (int d0 = nb - 1; d0 >= 0; d0--) {
                const int64_t qc = rem % q_dims[d0];
                rem /= q_dims[d0];
                off += qc * strides[d0];
                kb += (k_dims[d0] == 1 ? 0 : qc) * kmul;
                kmul *= k_dims[d0];
            }
            std::vector<float> u0, ui;
            unit(kb, 0, u0);
            unit(kb, own(i), ui);
            off += i * strides[ndims - 2];
            for (int64_t d = 0; d < D; d++)
                tmp_mem.set_elem(
                        off + d * strides[ndims - 1], r * (u0[d] + ui[d]));
        });
    } else {
        const int s_ax = k_sd ? ndims - 2 : ndims - 1;
        const int d_ax = k_sd ? ndims - 1 : ndims - 2;
        benchdnn_parallel_nd(k_batch, Sk, [&](int64_t kb, int64_t j) {
            int64_t off = 0, rem = kb;
            for (int d0 = nb - 1; d0 >= 0; d0--) {
                off += (rem % k_dims[d0]) * strides[d0];
                rem /= k_dims[d0];
            }
            std::vector<float> u;
            unit(kb, j, u);
            off += j * strides[s_ax];
            for (int64_t d = 0; d < D; d++)
                tmp_mem.set_elem(off + d * strides[d_ax], r * u[d]);
        });
    }

    if (mapped_here) tmp_mem.unmap();
    mem = std::move(tmp_mem);
    return OK;
}

int partition_data_displacer_t::gen_softmax_stats_filling(
        const ::graph::deserialized_op_t &main_op, int arg,
        const dnn_mem_t &src_mem, dnn_mem_t &mem, const_dnnl_memory_desc_t md,
        res_t *res) const {

    dnn_mem_t m(md, get_test_engine(), /* prefill = */ false);
    if (!m.is_dense()) {
        m = dnn_mem_t(query_md_ndims(md), query_md_dims(md), dnnl_f32, tag::abx,
                get_test_engine(), /* prefill = */ false);
    }

    logical_tensor::dims softmax_src_shape = main_op.in_lts_[0].shape_;
    logical_tensor::dims softmax_stats_shape = main_op.in_lts_[1].shape_;
    size_t axis = 0;
    for (; axis < softmax_src_shape.size() && axis < softmax_src_shape.size();
            ++axis) {
        if (softmax_src_shape[axis] != softmax_stats_shape[axis]) break;
    }

    int64_t outer_size {1}, inner_size {1}, axis_size {1};
    for (size_t i = 0; i < axis; i++) {
        outer_size *= softmax_src_shape[i];
    }
    for (size_t i = axis + 1; i < softmax_src_shape.size(); i++)
        inner_size *= softmax_src_shape[i];
    axis_size = softmax_src_shape[axis];

    benchdnn_parallel_nd(outer_size, inner_size, [&](int64_t ou, int64_t in) {
        float space_denom = 0.f;
        float space_max = -FLT_MAX;
        int64_t ou_in_offset = ou * axis_size * inner_size + in;

        for (int64_t as = 0; as < axis_size; ++as) {
            int64_t idx = ou_in_offset + as * inner_size;
            space_max = MAX2(space_max, src_mem.get_f32_elem(idx));
        }

        for (int64_t as = 0; as < axis_size; ++as) {
            int64_t idx = ou_in_offset + as * inner_size;
            float s = src_mem.get_f32_elem(idx);
            space_denom += expf(s - space_max);
        }

        // computes stats w.r.t. the softmax input
        // stats = max(input) + log(sum(exp(input - max(input))))
        // consider inf as a zero value
        int64_t stats_idx = ou * inner_size + in;
        float stats_value = space_denom ? space_max + logf(space_denom) : 0.f;
        m.set_f32_elem(stats_idx, stats_value);
    });

    mem = std::move(m);
    return OK;
}

} // namespace graph
