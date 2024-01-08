/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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

#include <assert.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include "binary_elemwise.hpp"
#include "compiler/ir/attr_keys.hpp"
#include "ops/fusible/unary_elemwise.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/brgemm_fusion.hpp>
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <runtime/dynamic_dispatch/ops/impl_type.hpp>
#include <unordered_map>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

template <class T>
static expr_c constant_maker(T data, const sc_data_type_t &dtype) {
    return make_expr<constant_node>(data, dtype);
};

std::vector<std::pair<int, std::vector<tensor_inplace_info_t>>>
binary_elementwise_op_impl_t::get_inplace_map() {
    std::vector<tensor_inplace_info_t> ret;
    auto &inp = get_inputs();
    auto &out_dim = get_outputs()[0]->details_.get_plain_dims();
    for (size_t i = 0; i < inp.size(); i++) {
        if (inp[i]->details_.get_plain_dims() == out_dim) {
            ret.emplace_back(tensor_inplace_info_t {
                    static_cast<int>(i), inplace_kind::ZERO_OFFSET});
        }
    }
    if (ret.empty()) { return {}; }
    return {{0, std::move(ret)}};
}

infer_status_code infer_binary_slice_ranges(
        fusible_op_t *cur, fslice_map &fsmap) {
    COMPILE_ASSERT(cur->get_inputs().size() == 2, "binary op is expected");
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map = search_known_input_slice(cur, fsmap);
    if (known_ranges_map.empty()) return infer_status_code::RETRY;
    // if unkown slice ranges exist.
    if (known_ranges_map.size() < cur->get_inputs().size()) {
        int unknown_idx
                = known_ranges_map.find(0) != known_ranges_map.end() ? 1 : 0;
        known_ranges_map[unknown_idx] = known_ranges_map[1 - unknown_idx];
        // set the other unknown slice range by achieved known_ranges_list
        set_unknown_input_slice(cur, known_ranges_map, fsmap);
    }
    // set outputs slice range
    auto &outslice = fsmap.get(cur->get_outputs()[0]);
    outslice = known_ranges_map[0];
    return infer_status_code::OK;
}

static slice_range_list infer_broadcast_arg_slice(
        const slice_range_list &known_range_list,
        const std::vector<int> &bc_axis, bool keep_dims) {
    slice_range_list bc_arg_range_list(known_range_list.size());
    for (size_t i = 0; i < bc_arg_range_list.size(); i++) {
        auto &known_range = known_range_list[i];
        for (size_t j = 0; j < known_range.size(); j++) {
            if (bc_axis.end() != std::find(bc_axis.begin(), bc_axis.end(), j)) {
                bc_arg_range_list[i].emplace_back(known_range.at(j));
            } else {
                if (keep_dims) {
                    bc_arg_range_list[i].emplace_back(
                            std::make_pair(expr(0), expr(1)));
                }
            }
        }
        if (bc_arg_range_list[i].empty())
            bc_arg_range_list[i].emplace_back(std::make_pair(0, 1));
    }
    return bc_arg_range_list;
}

static slice_range_list infer_broadcast_slice(
        const slice_range_list &known_range_list,
        const std::vector<int> &bc_axis, const std::vector<expr> &bc_dim) {
    slice_range_list bc_range_list(known_range_list.size());
    for (size_t i = 0; i < bc_range_list.size(); i++) {
        auto &known_range = known_range_list[i];
        COMPILE_ASSERT(known_range.size() == bc_dim.size()
                        || bc_axis == std::vector<int> {-1},
                "Unexpected cases found")
        for (size_t j = 0; j < known_range.size(); j++) {
            if (bc_axis.end() != std::find(bc_axis.begin(), bc_axis.end(), j)) {
                bc_range_list[i].emplace_back(known_range.at(j));
            } else {
                bc_range_list[i].emplace_back(
                        std::make_pair(expr(0), bc_dim[j]));
            }
        }
    }
    return bc_range_list;
}

static sc_dims infer_binary_elementwise_output_shape(const sc_dims &lhs_shape,
        const sc_dims &rhs_shape, const std::vector<int> &input_bc_axis) {
    sc_dims output_shape;
    if (input_bc_axis.empty()) {
        output_shape
                = op_traits::may_broadcast_t::infer_auto_broadcast_output_shape(
                        lhs_shape, rhs_shape);
    } else {
        if (lhs_shape.size() != rhs_shape.size()) {
            output_shape = lhs_shape.size() > rhs_shape.size() ? lhs_shape
                                                               : rhs_shape;
        } else {
            output_shape = get_number_of_squeeze_dims(lhs_shape)
                            <= get_number_of_squeeze_dims(rhs_shape)
                    ? lhs_shape
                    : rhs_shape;
        }
    }
    return output_shape;
}

static sc_data_type_t infer_output_dtype(
        sc_data_type_t a, sc_data_type_t b, bool is_b_scalar) {
    if (is_b_scalar) return a;
    // could_promote_dtypes is a map if {dtype, dtype_precision_ranking}
    // dtype mapped to a higher precision ranking value is more precise
    std::unordered_map<sc_data_type_t, int> could_promote_dtypes {
            {datatypes::s32, 0}, {datatypes::bf16, 1}, {datatypes::f32, 2}};
    if (could_promote_dtypes.find(a) != could_promote_dtypes.end()
            && could_promote_dtypes.find(b) != could_promote_dtypes.end()) {
        return could_promote_dtypes[a] >= could_promote_dtypes[b] ? a : b;
    }
    COMPILE_ASSERT(a == b,
            "Binary elementwise op shall have both inputs with the same "
            "dtype except for allow promotion cases.");
    return a;
}

void binary_elementwise_op_impl_t::set_plain_bc_axis() {
    auto lhs_shape = info_.inputs_[0]->details_.get_plain_dims();
    auto rhs_shape = info_.inputs_[1]->details_.get_plain_dims();
    auto output_shape = info_.outputs_[0]->details_.get_plain_dims();
    // get user specified bc_axis of the shorter input
    auto input_bc_axis = attrs_.get_or_else("bc_axis", std::vector<int> {});
    int ref_idx = get_ref_input_index(false);
    if (ref_idx == may_broadcast_t::NOT_DETERMINED) {
        ref_idx = lhs_shape.size() >= rhs_shape.size() ? 0 : 1;
    }
    // user specified bc_axis of the shorter input
    plain_bc_axis_.clear();
    if (input_bc_axis.empty()) {
        plain_bc_axis_.emplace_back(
                op_traits::may_broadcast_t::get_auto_broadcast_bc_axis(
                        lhs_shape, output_shape));
        plain_bc_axis_.emplace_back(
                op_traits::may_broadcast_t::get_auto_broadcast_bc_axis(
                        rhs_shape, output_shape));
    } else {
        COMPILE_ASSERT(ref_idx == 0 || ref_idx == 1,
                "bc_axis is only applicable to uni-directional broadcast.");
        plain_bc_axis_.resize(2);
        plain_bc_axis_[ref_idx]
                = op_traits::may_broadcast_t::get_auto_broadcast_bc_axis(
                        info_.inputs_[ref_idx]->details_.get_plain_dims(),
                        output_shape);
        plain_bc_axis_[1 - ref_idx] = input_bc_axis;
    }
}

binary_elementwise_op_impl_t::binary_elementwise_op_impl_t(
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    COMPILE_ASSERT(
            ins.size() == 2, "Binary elementwise op shall have 2 inputs.");
    // if auto_broadcast is not numpy, lhs and rhs should strictly match
    // if auto_broadcast is set && "bc_axis" is set, we follow "bc_axis"
    // otherwise we consider auto_broadcast rule
    std::string auto_broadcast
            = attrs.get_or_else("auto_broadcast", std::string("numpy"));
    COMPILE_ASSERT(auto_broadcast == "numpy"
                    || ins[0]->details_.get_plain_dims()
                            == ins[1]->details_.get_plain_dims(),
            "Binary elementwise op's lhs and rhs should have the same shape "
            "when auto_broadcast is none.");
    info_.inputs_ = ins;
    attrs_ = attrs;
    auto lhs_shape = info_.inputs_[0]->details_.get_plain_dims();
    auto rhs_shape = info_.inputs_[1]->details_.get_plain_dims();
    // get user specified bc_axis of the shorter input
    auto input_bc_axis = attrs_.get_or_else("bc_axis", std::vector<int> {});
    sc_dims output_shape = infer_binary_elementwise_output_shape(
            lhs_shape, rhs_shape, input_bc_axis);
    // ref_idx shall be the same side as query format's output format
    int ref_idx = get_ref_input_index(false);
    if (ref_idx == may_broadcast_t::NOT_DETERMINED) {
        ref_idx = lhs_shape.size() >= rhs_shape.size() ? 0 : 1;
    }
    if (outs.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
        info_.outputs_[0]->details_.set_plain_dims(output_shape);
        auto output_format = info_.inputs_[ref_idx]->details_.get_format();
        bool is_b_scalar
                = (info_.inputs_[1 - ref_idx]->details_.get_plain_dims()
                        == sc_dims {1});
        auto output_dtype = infer_output_dtype(
                info_.inputs_[ref_idx]->details_.dtype_,
                info_.inputs_[1 - ref_idx]->details_.dtype_, is_b_scalar);
        info_.outputs_[0]->details_.set_format(output_format);
        info_.outputs_[0]->details_.dtype_ = output_dtype;
    } else {
        info_.outputs_ = outs;
    }

    COMPILE_ASSERT(
            gc::graph::check_shape_equal(
                    info_.outputs_[0]->details_.get_plain_dims(), output_shape),
            "Binary elementwise op's output shape is not set correctly.");

    set_plain_bc_axis();
}

binary_elementwise_op_impl_t::binary_elementwise_op_impl_t(
        graph_tensor_ptr lhs, graph_tensor_ptr rhs, elt_operator elt_op)
    : binary_elementwise_op_impl_t({std::move(lhs), std::move(rhs)}, {}, {}) {
    elt_op_ = elt_op;
    switch (elt_op) {
        case elt_operator::ADD: op_name_ = "add"; break;
        case elt_operator::SUB: op_name_ = "sub"; break;
        case elt_operator::MUL: op_name_ = "mul"; break;
        case elt_operator::DIV: op_name_ = "div"; break;
        case elt_operator::MIN: op_name_ = "min"; break;
        case elt_operator::MAX: op_name_ = "max"; break;
        case elt_operator::SQD_DIFF: op_name_ = "squared_diff"; break;
        case elt_operator::PRELU: op_name_ = "prelu"; break;
        default: break;
    }
}

std::vector<int> binary_elementwise_op_impl_t::get_non_broadcast_input_index(
        bool assert_non_empty) const {
    const sc_dims &lhs_dims = info_.inputs_[0]->details_.get_plain_dims();
    const sc_dims &rhs_dims = info_.inputs_[1]->details_.get_plain_dims();
    const sc_dims &out_dims = infer_binary_elementwise_output_shape(lhs_dims,
            rhs_dims, attrs_.get_or_else("bc_axis", std::vector<int> {}));
    std::vector<int> ret;
    for (size_t i = 0; i < info_.inputs_.size(); ++i) {
        if (may_broadcast_t::broadcastable_shape_equal(
                    info_.inputs_[i]->details_.get_plain_dims(), out_dims)) {
            ret.emplace_back(i);
        }
    }
    if (assert_non_empty) {
        COMPILE_ASSERT(!ret.empty(),
                "Binary op is required to have at least one non-broadcast "
                "input at this stage.");
    }
    return ret;
}

int binary_elementwise_op_impl_t::get_ref_input_index(
        bool assert_determined) const {
    auto non_bc_input_indices
            = get_non_broadcast_input_index(assert_determined);
    if (non_bc_input_indices.empty()) {
        return may_broadcast_t::NOT_DETERMINED;
    }
    int non_bc_input_idx
            = non_bc_input_indices.size() > 1 ? -1 : non_bc_input_indices[0];
    if (non_bc_input_idx == -1) {
        // if the shapes are equal, find which side has blocking format.
        if (is_dynamic()) {
            non_bc_input_idx
                    = info_.inputs_[0]->details_.get_format_candidates().size()
                            >= info_.inputs_[1]
                                       ->details_.get_format_candidates()
                                       .size()
                    ? 0
                    : 1;
        } else {
            // Four situations: `both blocking`, `a blocking b not`, `b blocking
            // a not`, `both not blocking`. Only `b blocking a not` need to set
            // non_bc_input_idx to 1.
            non_bc_input_idx = 0;
            if (!info_.inputs_[0]->details_.get_format().is_blocking()
                    && info_.inputs_[1]->details_.get_format().is_blocking()) {
                non_bc_input_idx = 1;
            }
        }
    }
    return non_bc_input_idx;
}

void binary_elementwise_op_impl_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    const auto &in0_format = info_.inputs_[0]->details_.get_format();
    const auto &in1_format = info_.inputs_[1]->details_.get_format();

    int layout_input_idx = get_ref_input_index(true);
    attrs_[op_attr_key::layout_input_index] = layout_input_idx;

    if (info_.inputs_[0]->details_.get_plain_dims().size()
            != info_.inputs_[1]->details_.get_plain_dims().size()) {
        COMPILE_ASSERT(in0_format == sc_data_format_t(format_kinds::A)
                        || in1_format == sc_data_format_t(format_kinds::A),
                "Unsupported format encountered in binary elementwise query "
                "format.");
        in_formats.push_back({in0_format});
        in_formats.push_back({in1_format});
        out_formats.push_back({layout_input_idx ? in1_format : in0_format});
    } else {
        if (layout_input_idx) {
            // propagate layout from input 0 to 1.
            auto target_format = infer_broadcast_format(
                    info_.inputs_[1]->details_, info_.inputs_[0]->details_);
            in_formats.push_back({target_format});
            in_formats.push_back({in1_format});
            out_formats.push_back({in1_format});
        } else {
            // propagate layout from input 1 to 0.
            auto target_format = infer_broadcast_format(
                    info_.inputs_[0]->details_, info_.inputs_[1]->details_);
            in_formats.push_back({in0_format});
            in_formats.push_back({target_format});
            out_formats.push_back({in0_format});
        }
    }
    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}

infer_status_code binary_elementwise_op_impl_t::infer_slice_ranges(
        const context_ptr &ctx, fslice_map &fsmap) {
    COMPILE_ASSERT(get_inputs().size() == 2, "binary op is expected");
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map = search_known_input_slice(this, fsmap);
    if (known_ranges_map.empty()) return infer_status_code::RETRY;
    // double-check all known case
    if (known_ranges_map.size() == get_inputs().size()) {
        // check whether slice size is matched
        if (known_ranges_map[0].size() != known_ranges_map[1].size()) {
            // try to align with smaller one and erase bigger one
            int erase_input_id
                    = known_ranges_map[0].size() < known_ranges_map[1].size()
                    ? 1
                    : 0;
            known_ranges_map.erase(erase_input_id);
            fsmap.erase(get_inputs()[erase_input_id]);
        }
    }
    // if unkown slice ranges exist.
    if (known_ranges_map.size() < get_inputs().size()) {
        int unknown_idx
                = known_ranges_map.find(0) != known_ranges_map.end() ? 1 : 0;
        // check broadcast
        int bc_input_idx = get_broadcast_input();
        if (bc_input_idx >= 0) {
            bool keep_dims = get_inputs()[bc_input_idx]
                                     ->details_.get_blocking_dims()
                                     .size()
                    == get_inputs()[1 - bc_input_idx]
                               ->details_.get_blocking_dims()
                               .size();
            auto bc_axis = get_bc_axis();
            if (unknown_idx != bc_input_idx) {
                slice_range_list bc_range_list = infer_broadcast_slice(
                        known_ranges_map[1 - unknown_idx], bc_axis,
                        get_inputs()[1 - bc_input_idx]
                                ->details_.get_blocking_dims_expr(
                                        get_owner_graph()));
                known_ranges_map[unknown_idx] = bc_range_list;
            } else {
                slice_range_list bc_arg_range_list = infer_broadcast_arg_slice(
                        known_ranges_map[1 - unknown_idx], bc_axis, keep_dims);
                known_ranges_map[unknown_idx] = std::move(bc_arg_range_list);
            }
        } else {
            known_ranges_map[unknown_idx] = known_ranges_map[1 - unknown_idx];
        }
        // set the other unknown slice range by achieved known_ranges_list
        set_unknown_input_slice(this, known_ranges_map, fsmap);
    }
    // set outputs slice range
    auto &outslice = fsmap.get(get_outputs()[0]);
    int bc_idx = get_broadcast_input();
    outslice = known_ranges_map[bc_idx > -1 ? (1 - bc_idx) : 0];
    return infer_status_code::OK;
}

infer_status_code binary_elementwise_op_impl_t::pre_infer_slice_ranges(
        const context_ptr &ctx, fslice_map &fsmap) {
    auto &outslice = fsmap.get(get_outputs()[0]);
    if (outslice.empty()) { return infer_status_code::RETRY; }
    // check broadcast
    int bc_input_idx = get_broadcast_input();
    for (size_t i = 0; i < get_inputs().size(); i++) {
        auto &input = get_inputs()[i];
        auto &inpslice = fsmap.get(input);
        if (inpslice.empty()) {
            if (bc_input_idx == static_cast<int>(i)) {
                auto bc_axis = get_bc_axis();
                inpslice = infer_broadcast_arg_slice(outslice, bc_axis,
                        get_inputs()[bc_input_idx]
                                        ->details_.get_blocking_dims()
                                        .size()
                                == get_inputs()[1 - bc_input_idx]
                                           ->details_.get_blocking_dims()
                                           .size());
            } else {
                inpslice = outslice;
            }
        }
    }
    return infer_status_code::OK;
}

void binary_elementwise_op_impl_t::infer_binding_axis(
        binding_axis_map &bdax_map) {
    // search known axis from any input of cur fusbile op
    auto known_axis_map = search_known_input_axis(this, bdax_map);
    if (!bdax_map.get(get_outputs()[0]).empty()) return;

    if (known_axis_map.size() < get_inputs().size()) {
        int unknown_idx
                = known_axis_map.find(0) != known_axis_map.end() ? 1 : 0;
        // check broadcast
        int bc_input_idx = get_broadcast_input();
        if (bc_input_idx >= 0) {
            bool keep_dims = get_inputs()[bc_input_idx]
                                     ->details_.get_blocking_dims()
                                     .size()
                    == get_inputs()[1 - bc_input_idx]
                               ->details_.get_blocking_dims()
                               .size();
            if (keep_dims) {
                known_axis_map[unknown_idx] = known_axis_map[1 - unknown_idx];
            } else {
                auto bc_axis = plain_bc_axis_[bc_input_idx];
                binding_axis known_axis = known_axis_map[1 - unknown_idx],
                             unknown_axis(known_axis.size());
                if (unknown_idx != bc_input_idx) {
                    if (bc_axis == std::vector<int> {-1}) {
                        bc_axis[0] = get_inputs()[1 - bc_input_idx]
                                             ->details_.get_plain_dims()
                                             .size()
                                - 1;
                    }
                    std::transform(known_axis.begin(), known_axis.end(),
                            unknown_axis.begin(),
                            [&bc_axis](const std::vector<int> &bd_ax) {
                                std::vector<int> ret(bd_ax.size());
                                std::transform(bd_ax.begin(), bd_ax.end(),
                                        ret.begin(), [&bc_axis](const int &ax) {
                                            COMPILE_ASSERT(
                                                    ax < static_cast<int64_t>(
                                                            bc_axis.size()),
                                                    "Unexpected ax found: "
                                                            << ax)
                                            return bc_axis[ax];
                                        });
                                return ret;
                            });
                } else {
                    for (auto &bd_ax : known_axis) {
                        std::vector<int> ret;
                        for (auto &ax : bd_ax) {
                            auto iter = std::find(
                                    bc_axis.begin(), bc_axis.end(), ax);
                            if (iter != bc_axis.end()) {
                                ret.emplace_back(iter - bc_axis.begin());
                            }
                        }
                        unknown_axis.emplace_back(ret);
                    }
                }
                known_axis_map[unknown_idx] = unknown_axis;
            }
        } else {
            known_axis_map[unknown_idx] = known_axis_map[1 - unknown_idx];
        }
    }
    // set outputs slice range
    int bc_idx = get_broadcast_input();
    bdax_map.get(get_outputs()[0])
            = known_axis_map[bc_idx > -1 ? (1 - bc_idx) : 0];

    // set the other unknown slice range by achieved known_ranges_list
    set_unknown_binding_axis(this, known_axis_map, bdax_map);
}

void binary_elementwise_op_impl_t::pre_infer_binding_axis(
        binding_axis_map &bdax_map) {
    auto &outaxis = bdax_map.get(get_outputs()[0]);
    COMPILE_ASSERT(!outaxis.empty(),
            "Unknown output axis found, could not pre infer binding axis")

    // check broadcast
    int bc_input_idx = get_broadcast_input();
    for (size_t i = 0; i < get_inputs().size(); i++) {
        auto &input = get_inputs()[i];
        auto &inpaxis = bdax_map.get(input);
        if (inpaxis.empty()) {
            if (bc_input_idx == static_cast<int>(i)) {
                bool keep_dims = get_inputs()[bc_input_idx]
                                         ->details_.get_blocking_dims()
                                         .size()
                        == get_inputs()[1 - bc_input_idx]
                                   ->details_.get_blocking_dims()
                                   .size();
                if (keep_dims) {
                    inpaxis = outaxis;
                } else {
                    auto bc_axis = plain_bc_axis_[bc_input_idx];
                    for (auto &bd_ax : outaxis) {
                        std::vector<int> ret;
                        for (auto &ax : bd_ax) {
                            auto iter = std::find(
                                    bc_axis.begin(), bc_axis.end(), ax);
                            if (iter != bc_axis.end()) {
                                ret.emplace_back(iter - bc_axis.begin());
                            }
                        }
                        inpaxis.emplace_back(ret);
                    }
                }
            } else {
                inpaxis = outaxis;
            }
            if (auto bd_op = input->producer_owner_->dyn_cast<
                             op_traits::mixed_partition_acceptable>()) {
                bd_op->pre_infer_binding_axis(bdax_map);
            }
        }
    }
}

std::vector<int> binary_elementwise_op_impl_t::get_bc_axis() const {
    int bc_input_idx = get_broadcast_input();
    if (bc_input_idx == -1) { bc_input_idx = 1; }
    if (plain_bc_axis_[bc_input_idx] == std::vector<int> {-1})
        return plain_bc_axis_[bc_input_idx];
    return transform_axis_plain2blocking(
            info_.inputs_[1 - bc_input_idx], plain_bc_axis_[bc_input_idx]);
}

bool binary_elementwise_op_impl_t::register_brgemm_fusion(
        const context_ptr &ctx, const std::vector<tensor_slice *> &outputs,
        const std::vector<const tensor_slice *> &inputs,
        brgemm_fusion_register &brg_reg) {
    if (!fuse_in_brgemm_) { return false; }
    int bc_input_idx = get_broadcast_input();
    // input 0 broadcast, can not be processed in brgemm
    if (bc_input_idx == 0) { return false; }
    return brg_reg.register_op_infos(shared_from_this(),
            outputs[0]->get_tensor_ptr(), inputs[1]->get_tensor_ptr(),
            inputs[1]->get_shape());
}

shape_rl_vec binary_elementwise_op_impl_t::get_dynamic_shape_relations() const {
    shape_rl_vec ret;
    auto &in0_plain_dims = get_inputs()[0]->details_.get_plain_dims();
    auto &in1_plain_dims = get_inputs()[1]->details_.get_plain_dims();
    auto &out_plain_dims = get_outputs()[0]->details_.get_plain_dims();
    assert(in0_plain_dims.size() == in1_plain_dims.size()
            || in0_plain_dims.size() == 1 || in1_plain_dims.size() == 1);
    if (in0_plain_dims.size() == in1_plain_dims.size()) {
        for (size_t i = 0; i < in0_plain_dims.size(); i++) {
            // maybe broadcast
            if ((is_dynamic_dim(in0_plain_dims[i])
                        || is_dynamic_dim(in1_plain_dims[i]))
                    && in0_plain_dims[i] != 1 && in1_plain_dims[i] != 1) {
                ret.emplace_back(in0_plain_dims[i], in1_plain_dims[i]);
            }
        }
    }
    for (size_t i = 0; i < out_plain_dims.size(); i++) {
        if (is_dynamic_dim(out_plain_dims[i])) {
            if (i < in0_plain_dims.size() && in0_plain_dims[i] != 1) {
                ret.emplace_back(in0_plain_dims[i], out_plain_dims[i]);
            }
            if (i < in1_plain_dims.size() && in1_plain_dims[i] != 1) {
                ret.emplace_back(in1_plain_dims[i], out_plain_dims[i]);
            }
        }
    }
    return ret;
}

static stmt select_algorithm(elt_operator elt_op, const expr &in0,
        const expr &in1, const expr &out, const any_map_t &attrs,
        const sc_op_info_t &info) {
    std::vector<stmt_c> cur_list;
    auto var_maker = [&cur_list, &in0](const std::string &name) {
        auto var = builder::make_var(in0->dtype_, name);
        cur_list.emplace_back(builder::make_var_tensor_def_unattached(var));
        return var;
    };
    auto assign_maker = [&cur_list](const expr &def_var, const expr &def_val) {
        cur_list.emplace_back(
                builder::make_assign_unattached(def_var, def_val));
    };
    switch (elt_op) {
        case elt_operator::ADD: {
            return builder::make_assign_unattached(out, in0 + in1);
        } break;
        case elt_operator::SUB: {
            return builder::make_assign_unattached(out, in0 - in1);
        } break;
        case elt_operator::MUL: {
            return builder::make_assign_unattached(out, in0 * in1);
        } break;
        case elt_operator::DIV: {
            return builder::make_assign_unattached(out, in0 / in1);
        } break;
        case elt_operator::MIN: {
            return builder::make_assign_unattached(
                    out, builder::make_min(in0, in1));
        } break;
        case elt_operator::MAX: {
            return builder::make_assign_unattached(
                    out, builder::make_max(in0, in1));
        } break;
        case elt_operator::SQD_DIFF: {
            return builder::make_assign_unattached(
                    out, (in0 - in1) * (in0 - in1));
        } break;
        case elt_operator::PRELU: {
            return builder::make_assign_unattached(out,
                    builder::make_select(in0 >= make_expr<constant_node>(
                                                 (int64_t)0, in0->dtype_),
                            in0, in0 * in1));
        } break;
        case elt_operator::ABS_BWD: {
            return builder::make_assign_unattached(out,
                    builder::make_select(
                            in0 > make_expr<constant_node>(0.f, in0->dtype_),
                            in1,
                            builder::make_select(in0
                                            != make_expr<constant_node>(
                                                    0.f, in0->dtype_),
                                    builder::make_sub(make_expr<constant_node>(
                                                              0.f, in0->dtype_),
                                            in1),
                                    make_expr<constant_node>(
                                            0.f, in0->dtype_))));
        } break;
        case elt_operator::CLAMP_BWD: {
            return builder::make_assign_unattached(out,
                    builder::make_select(in0 > make_expr<constant_node>(
                                                 (float)attrs.get<float>("min"),
                                                 in0->dtype_),
                            builder::make_select(
                                    in0 < make_expr<constant_node>(
                                            (float)attrs.get<float>("max"),
                                            in0->dtype_),
                                    in1,
                                    make_expr<constant_node>(0.f, in1->dtype_)),
                            make_expr<constant_node>(0.f, in0->dtype_)));
        } break;
        case elt_operator::ELU_BWD: {
            expr used_inp = builder::make_select(
                    in0 > make_expr<constant_node>(0.f, in0->dtype_), in1,
                    in1
                            * make_expr<constant_node>(
                                    (float)attrs.get<float>("alpha"),
                                    in0->dtype_)
                            * builder::make_exp(in0));
            expr used_out = builder::make_select(
                    in0 > make_expr<constant_node>(0.f, in0->dtype_), in1,
                    in1
                            * (in0
                                    + make_expr<constant_node>(
                                            (float)attrs.get<float>("alpha"),
                                            in0->dtype_)));
            expr res = attrs.get_or_else<bool>("use_dst", true) ? used_out
                                                                : used_inp;
            return builder::make_assign_unattached(out, res);
        } break;
        case elt_operator::HARDSWISH_BWD: {
            auto alpha = attrs.get_or_else<float>("alpha", 1.f / 6.f);
            auto beta = attrs.get_or_else<float>("beta", 0.5f);
            expr test_expr = make_expr<constant_node>(alpha, in0->dtype_) * in0
                    + make_expr<constant_node>(beta, in0->dtype_);
            expr cal_expr = in1
                    * (make_expr<constant_node>(2.f, in0->dtype_)
                                    * make_expr<constant_node>(
                                            alpha, in0->dtype_)
                                    * in0
                            + make_expr<constant_node>(beta, in0->dtype_));
            expr res = builder::make_select(
                    (test_expr) <= make_expr<constant_node>(0.f, in0->dtype_),
                    make_expr<constant_node>(0.f, in0->dtype_),
                    builder::make_select(
                            (test_expr) >= make_expr<constant_node>(
                                    1.f, in0->dtype_),
                            in1, cal_expr));
            return builder::make_assign_unattached(out, res);
        } break;
        case elt_operator::HARDSIGMOID_BWD: {
            auto alpha = constant_maker<float>(
                    attrs.get<float>("alpha"), in0->dtype_);
            auto beta = constant_maker<float>(
                    attrs.get<float>("beta"), in0->dtype_);
            auto one_f = constant_maker<float>(1.f, in0->dtype_);
            auto zero_f = constant_maker<float>(0.f, in0->dtype_);
            expr test_expr = in0 * alpha + beta;
            auto f_var0 = var_maker("f_var0");
            assign_maker(f_var0, test_expr);
            expr cal_expr = in1 * alpha;
            auto f_var1 = var_maker("f_var1");
            assign_maker(f_var1, cal_expr);
            expr res = builder::make_select(f_var0 < one_f,
                    builder::make_select(f_var0 > zero_f, f_var1, zero_f),
                    zero_f);
            assign_maker(out, res);
            return builder::make_stmts_unattached(cur_list);
        } break;
        case elt_operator::SQRT_BWD: {
            bool use_dst = attrs.get_or_else<bool>("use_dst", true);
            auto half_f = constant_maker<float>(0.5f, in0->dtype_);
            expr f_var = var_maker("f_var");
            if (use_dst) {
                auto dst_expr = (half_f / in0);
                dst_expr->attr()[attr_keys::fast_math] = false;
                assign_maker(f_var, dst_expr);
            } else {
                auto src_expr = half_f / builder::make_sqrt(in0);
                src_expr->attr()[attr_keys::fast_math] = false;
                assign_maker(f_var, src_expr);
            }
            auto ret = f_var * in1;
            ret->attr()[attr_keys::fast_math] = false;
            assign_maker(out, ret);
            return builder::make_stmts_unattached(cur_list);
        } break;
        case elt_operator::MISH_BWD: {
            auto min_inp = builder::make_min(
                    in0, constant_maker<float>(22.180708f, in0->dtype_));
            auto min_var = var_maker("inp_min_var_" + fusion_create_var_idx());
            assign_maker(min_var, min_inp);
            // e^x
            auto exp_f = builder::make_exp(min_var);
            auto var_exp_f = var_maker("exp_var_" + fusion_create_var_idx());
            assign_maker(var_exp_f, exp_f);
            // e^2x
            auto exp_f2 = var_exp_f * var_exp_f;
            auto exp_f2_var = var_maker("exp_f2_var" + fusion_create_var_idx());
            assign_maker(exp_f2_var, exp_f2);
            // 4 * e^2x
            auto formular_0
                    = exp_f2_var * constant_maker<float>(4.f, in0->dtype_);
            auto f_var0 = var_maker("f_var0" + fusion_create_var_idx());
            assign_maker(f_var0, formular_0);
            // e^3x + 4 * e^2x
            auto formular_1
                    = builder::make_fmadd(var_exp_f, exp_f2_var, f_var0);
            auto f_var1 = var_maker("f_var1" + fusion_create_var_idx());
            assign_maker(f_var1, formular_1);
            // x + 1.f
            auto formular_2 = in0 + constant_maker<float>(1.f, in0->dtype_);
            auto f_var2 = var_maker("f_var2" + fusion_create_var_idx());
            assign_maker(f_var2, formular_2);
            auto formular_3 = f_var2 + constant_maker(0.5f, in0->dtype_);
            auto f_var3 = var_maker("f_var3" + fusion_create_var_idx());
            assign_maker(f_var3, formular_3);
            // 4 * (1 + 1.5f)
            auto formular_4 = constant_maker(4.f, in0->dtype_) * f_var3;
            auto f_var4 = var_maker("f_var4" + fusion_create_var_idx());
            assign_maker(f_var4, formular_4);
            // e^3x + 4*e^2x + 4*(x+1.5)*e^x
            auto formular_5 = builder::make_fmadd(f_var4, var_exp_f, f_var1);
            auto f_var5 = var_maker("f_var5" + fusion_create_var_idx());
            assign_maker(f_var5, formular_5);
            // omega = e^3x + 4*e^2x + 4*e^x*(x+1.5) + 4*(x+1)
            auto formular_6 = builder::make_fmadd(
                    constant_maker(4.f, in0->dtype_), f_var2, f_var5);
            auto f_var6 = var_maker("f_var6" + fusion_create_var_idx());
            assign_maker(f_var6, formular_6);
            // delta = (e^x+1)^2 + 1
            auto formular_7
                    = var_exp_f + constant_maker<float>(1.f, in0->dtype_);
            auto f_var7 = var_maker("f_var7" + fusion_create_var_idx());
            assign_maker(f_var7, formular_7);
            auto formular_8 = f_var7 * f_var7;
            auto f_var8 = var_maker("f_var8" + fusion_create_var_idx());
            assign_maker(f_var8, formular_8);
            auto formular_9 = f_var8 + constant_maker<float>(1.f, in0->dtype_);
            auto f_var9 = var_maker("f_var9" + fusion_create_var_idx());
            assign_maker(f_var9, formular_9);
            auto formular_10 = f_var9 * f_var9;
            auto f_var10 = var_maker("f_var10");
            assign_maker(f_var10, formular_10);
            auto formular_11 = exp_f * f_var6;
            auto f_var11 = var_maker("f_var11");
            assign_maker(f_var11, formular_11);
            auto formular_12 = f_var11 / f_var10;
            formular_12->attr()[attr_keys::fast_math] = false;
            auto f_var12 = var_maker("f_var12");
            assign_maker(f_var12, formular_12);
            auto res = f_var12 * in1;
            res->attr()[attr_keys::fast_math] = false;
            assign_maker(out, res);
            return builder::make_stmts_unattached(cur_list);
        } break;
        case elt_operator::TANH_BWD: {
            bool use_dst = attrs.get_or_else<bool>("use_dst", true);
            tanh_op_t tanh_compute(info.inputs_[0]);
            const auto &tanh_ret = tanh_compute.compute_element(in0);
            expr src = builder::make_mul(in1,
                    builder::make_sub(
                            make_expr<constant_node>(1.f, in0->dtype_),
                            builder::make_mul(tanh_ret, tanh_ret)));
            expr dst = builder::make_mul(in1,
                    builder::make_sub(
                            make_expr<constant_node>(1.f, in0->dtype_),
                            builder::make_mul(in0, in0)));
            expr res = use_dst ? dst : src;
            return builder::make_assign_unattached(out, res);
        } break;
        case elt_operator::SOFTPLUS_BWD: {
            float beta = attrs.get_or_else<float>("beta", 1.f);
            sigmoid_op_t sigmoid_compute(info.inputs_[0]);
            bool is_f32 = in0->dtype_.type_code_ == sc_data_etype::F32;
            auto make_cast_f32 = [](const expr &inp) {
                return builder::make_cast(
                        sc_data_type_t::f32(inp->dtype_.lanes_), inp);
            };
            auto sigmoid_inp = is_f32 ? in0 : make_cast_f32(in0);
            const auto &sigmoid_ret
                    = sigmoid_compute.compute_element(sigmoid_inp
                            * constant_maker<float>(beta,
                                    sc_data_type_t::f32(in0->dtype_.lanes_)));
            auto inp2 = is_f32 ? in1 : make_cast_f32(in1);
            auto f_val = builder::make_mul(sigmoid_ret, inp2);
            auto res_val
                    = is_f32 ? f_val : builder::make_cast(in0->dtype_, f_val);
            auto res = builder::make_assign_unattached(out, res_val);
            return res;
        } break;
        default: {
            COMPILE_ASSERT(false,
                    "Unsupport elementwise op "
                    "found.\n");
            return stmt();
        } break;
    }
    return stmt();
}

void compute_block_broadcast(const context_ptr &ctx, sc_graph_t &graph,
        const std::vector<const tensor_slice *> &src, const tensor_slice &dst,
        sc_op_info_t &info, int bc_input_idx, const std::vector<int> &bc_axis,
        const vectorized_info_t &vx_info, const mask_compute_func_t &compute,
        const graph_tensor_ptr &expand_gt, size_t wkld = 0UL,
        bool use_mask = false) {
    //  enable vectorize code
    bool use_vectorized = false;
    vec_backend_require(ctx, use_vectorized);
    // nested loop vars
    std::vector<expr> iter_vars;
    // the indices for multiple inputs. First dim: the input, Second dim: the
    // dimemsions in the tensor
    std::vector<expr> in_idx, in_bc_idx;
    // the indices for the output tensor
    std::vector<expr> dst_idx;

    COMPILE_ASSERT(bc_input_idx == 0 || bc_input_idx == 1,
            "bc_input_idx is expected to be 0 or 1")
    bool is_blocking_shape = is_op_input_blocking_shape(info);
    const tensor_slice *in_tsl = src[1 - bc_input_idx],
                       *in_bc_tsl = src[bc_input_idx];
    bool keep_dims = in_tsl->get_base_dims().size()
            == in_bc_tsl->get_base_dims().size();
    // add output type check, manual downcast
    sc_data_etype out_etype
            = dst.tptr_->dtype_.get_pointer_element().as_etype();
    // use src_indices.at(0) as default
    for (unsigned i = 0; i < dst.nslice_dims(); i++) {
        // make the loop var for the for-loop
        iter_vars.emplace_back(range_from_outer_loop(dst.get_ranges()[i])
                        ? expr(0)
                        : builder::make_var(datatypes::index,
                                std::string("_fuseiter")
                                        + fusion_create_idx()));
        in_idx.emplace_back(iter_vars.back());
        if (std::find(bc_axis.begin(), bc_axis.end(), i) != bc_axis.end()) {
            in_bc_idx.emplace_back(iter_vars.back());
        } else if (keep_dims) {
            in_bc_idx.emplace_back(0);
        }
        /** push an index for output tensor **/
        dst_idx.emplace_back(iter_vars.back());
    }
    // For empty bc_axis
    if (in_bc_idx.empty()) in_bc_idx = {0};
    // tail part
    std::vector<expr> in_idx_tail = in_idx, in_bc_idx_tail = in_bc_idx,
                      dst_idx_tail = dst_idx;
    auto tail_var = builder::make_var(
            datatypes::index, std::string("_fuseiter") + fusion_create_idx());
    in_idx_tail[vx_info.axis] = tail_var;
    dst_idx_tail[vx_info.axis] = tail_var;

    expr indexed_target, indexed_input;
    auto bld = builder::get_current_builder();
    COMPILE_ASSERT(bld, "No active builder is set");
    auto slice_len = dst.get_shape().at(vx_info.axis);
    int lanes = static_cast<int>(vx_info.lanes);
    auto floor = do_cast_and_fold(slice_len / lanes * lanes);
    auto tail = do_cast_and_fold(slice_len % lanes);
    int floor_int = 0;
    int tail_int = 0;
    int floor_len
            = get_const_as_int(slice_len.static_as<constant>()) / lanes * lanes;
    int tail_len = get_const_as_int(slice_len.static_as<constant>()) % lanes;
    if (floor.isa<constant>()) {
        floor_int = get_expr_as_int(floor);
        tail_int = get_expr_as_int(tail);
        COMPILE_ASSERT((floor_int + tail_int), "Don't support shape len is 0.");
    }
    auto last_axis = expr(floor + tail);
    const int INVALID_AXIS_MASK = -64;
    int last_axis_mask = INVALID_AXIS_MASK;
    std::unordered_map<expr, std::pair<expr, expr>> conditions;
    std::unordered_map<expr, std::pair<expr, expr>> conditions_tail;

    if (use_mask) {
        compute_mask_and_generate_condition(graph, src,
                info.inputs_[0]->details_.get_plain_dims(),
                info.inputs_[0]->details_.get_format(), iter_vars,
                vx_info.lanes, conditions, last_axis_mask);
    }
    if (last_axis_mask != INVALID_AXIS_MASK && floor_int > 0) {
        COMPILE_ASSERT(tail_int == 0,
                "Currently we only support mask in vectorize compute not "
                "tail.");
    }
    std::vector<stmt> tcur;
    stmt cur;
    bool bc_input_cast
            = !in_bc_tsl->tptr_->dtype_.get_pointer_element().is_etype(
                    out_etype);
    // if lastdim satisfied threshold, will use scalar version
    bool tail_threshold = tail.isa<constant>() && tail_int <= 1;
    bool last_dim_eq_1 = tail.isa<constant>() && tail_int == 1;
    bool use_scalar = tail_threshold || !use_vectorized || lanes == 1;
    auto func_op_cast = [](sc_data_etype out_etype, expr &indexed_input,
                                bool use_scalar = false) {
        if (use_scalar) {
            indexed_input = builder::make_cast(
                    sc_data_type_t(out_etype), indexed_input);
        } else {
            indexed_input = builder::make_cast(
                    sc_data_type_t(out_etype, indexed_input->dtype_.lanes_),
                    indexed_input);
        }
    };
    auto func_index_bc_input = [&](std::vector<expr> &in_bc_idx,
                                       expr &indexed_bc_input, expr &iter_var,
                                       bool use_scalar = false,
                                       bool has_tail = false) {
        if (bc_axis.back() == static_cast<int64_t>(vx_info.axis)) {
            indexing_from_diff_cond(use_scalar, has_tail, *in_bc_tsl, in_bc_idx,
                    lanes, indexed_bc_input, slice_len, iter_var, floor);
        }
        // IF last dim is excluded in bc_axis.
        else {
            if (use_scalar) {
                indexed_bc_input
                        = builder::make_indexing(in_bc_tsl->tptr_, in_bc_idx);
            } else {
                indexed_bc_input = builder::make_broadcast(
                        builder::make_indexing(in_bc_tsl->tptr_, in_bc_idx),
                        static_cast<int>(vx_info.lanes));
            }
        }
        if (bc_input_cast) {
            func_op_cast(out_etype, indexed_bc_input, use_scalar);
        }
    };
    // recover schedule loop
    for (int i = static_cast<int>(dst.get_shape().size() - 1); i >= 0; i--) {
        stmt body;
        // move broadcast op to body
        if (static_cast<int>(dst.get_shape().size()) == vx_info.axis + 1
                && i == vx_info.axis) {
            if (!floor.isa<constant>() || floor_int) {
                bld->push_scope();
                // if the shape is less than lanes, we don't use mask to
                // process.
                indexing_from_diff_cond(false, false, dst, dst_idx, lanes,
                        indexed_target, slice_len, iter_vars.at(i), floor);
                indexing_from_diff_cond(false, false, *in_tsl, in_idx, lanes,
                        indexed_input, slice_len, iter_vars.at(i), floor);
                if (!in_tsl->tptr_->dtype_.get_pointer_element().is_etype(
                            out_etype)) {
                    func_op_cast(out_etype, indexed_input);
                }

                expr indexed_bc_input;
                func_index_bc_input(
                        in_bc_idx, indexed_bc_input, iter_vars.at(i));
                std::vector<expr::lvalue_proxy_t> target_vec {
                        expr::lvalue_proxy_t(indexed_target, false)};
                auto cond_it = conditions.find(iter_vars[i]);
                if (cond_it != conditions.end()) {
                    assert(last_axis_mask != INVALID_AXIS_MASK);
                    cur = compute(
                            std::vector<expr> {indexed_input, indexed_bc_input},
                            target_vec, cond_it->second.first,
                            cond_it->second.second, vx_info.lanes);
                } else {
                    cur = compute(
                            std::vector<expr> {indexed_input, indexed_bc_input},
                            target_vec);
                }
                cur->attr()[op_traits::workload_computable_t::workload_number]
                        = wkld;
                bld->emit(cur);
                cur = bld->pop_scope();
                if (iter_vars.at(i).isa<var>()) {
                    cur = make_stmt<for_loop_node_t>(iter_vars.at(i), expr(0),
                            floor, expr(int(vx_info.lanes)), cur, true,
                            for_type::NORMAL);
                    bind_loop_axis(expand_gt, cur, i, true);
                }
                tcur.emplace_back(cur);
            }
            if ((!tail.isa<constant>() && !is_blocking_shape) || tail_int) {
                auto res_it = std::find(
                        bc_axis.begin(), bc_axis.end(), vx_info.axis);
                if (res_it != bc_axis.end()) {
                    in_bc_idx_tail[keep_dims ? vx_info.axis
                                             : (res_it - bc_axis.begin())]
                            = tail_var;
                }
                expr indexed_bc_input_tail;
                func_index_bc_input(in_bc_idx_tail, indexed_bc_input_tail,
                        tail_var, use_scalar, true);
                expr indexed_target_tail;
                expr indexed_input_tail;
                indexing_from_diff_cond(use_scalar, true, dst, dst_idx_tail,
                        lanes, indexed_target_tail, slice_len, tail_var, floor,
                        true);
                indexing_from_diff_cond(use_scalar, true, *in_tsl, in_idx_tail,
                        lanes, indexed_input_tail, slice_len, tail_var, floor,
                        true);
                if (!in_tsl->tptr_->dtype_.get_pointer_element().is_etype(
                            out_etype)) {
                    func_op_cast(out_etype, indexed_input_tail);
                }
                std::vector<expr::lvalue_proxy_t> target_vec_tail {
                        expr::lvalue_proxy_t(indexed_target_tail, false)};
                bld->push_scope();
                cur = compute(std::vector<expr> {indexed_input_tail,
                                      indexed_bc_input_tail},
                        target_vec_tail);
                cur->attr()[op_traits::workload_computable_t::workload_number]
                        = wkld;
                bld->emit(cur);
                cur = make_stmt<for_loop_node_t>(tail_var, expr(floor),
                        do_cast_and_fold(floor + tail),
                        use_scalar ? expr(1) : lanes, bld->pop_scope(), true,
                        for_type::NORMAL);
                bind_loop_axis(expand_gt, cur, i, true);
                tcur.emplace_back(cur);
            }
        } else if (iter_vars.at(i).isa<var>()) {
            // Do not generate those dummy loop
            if (!tcur.empty() && tcur[0].defined()) {
                body = make_stmt<stmts_node_t>(std::move(tcur));
                tcur.clear();
                // address special condition, like temp_buffer is used
                cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                        expr(0), dst.get_shape().at(i), expr(1),
                        std::move(body), true, for_type::NORMAL);
            } else if (cur.defined()) {
                body = make_stmt<stmts_node_t>(
                        std::vector<stmt> {std::move(cur)});
                // address special condition, like temp_buffer is used
                cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                        expr(0), dst.get_shape().at(i), expr(1),
                        std::move(body), true, for_type::NORMAL);
            } else {
                // if cur not defined, means last axis of tensor slice
                // has range 1, e.g. tensor_slice{{i, 100},{0, 1}}
                indexed_target = builder::make_indexing(dst.tptr_, dst_idx);

                indexed_input = builder::make_indexing(in_tsl->tptr_, in_idx);

                expr indexed_bc_input
                        = builder::make_indexing(in_bc_tsl->tptr_, in_bc_idx);
                if (bc_input_cast) {
                    indexed_bc_input = builder::make_cast(
                            sc_data_type_t(
                                    out_etype, indexed_bc_input->dtype_.lanes_),
                            indexed_bc_input);
                }
                std::vector<expr::lvalue_proxy_t> target_vec {
                        expr::lvalue_proxy_t(indexed_target, false)};
                bld->push_scope();
                cur = compute(
                        std::vector<expr> {indexed_input, indexed_bc_input},
                        target_vec);
                cur->attr()[op_traits::workload_computable_t::workload_number]
                        = wkld;
                bld->emit(cur);
                cur = make_stmt<for_loop_node_t>(iter_vars.at(i), expr(0),
                        dst.get_shape().at(i), expr(1), bld->pop_scope(), true,
                        for_type::NORMAL);
            }
            bind_loop_axis(expand_gt, cur, i, true);
        }
    }
    if (!tcur.empty() && tcur[0].defined()) {
        // TODO(xxx): currenly we don't add merge_loop attribute for this
        // special case, need stronger loop analysis.
        for (auto &it : tcur) {
            bld->emit(it);
        }
        // TODO(yifei): analyze whether this is safe enough
        cur->attr()[stmt_attr_key::merge_loop] = true;
    } else {
        cur->attr()[stmt_attr_key::merge_loop] = true;
        bld->emit(cur);
    }
}

void binary_elementwise_op_impl_t::compute_block(context_ptr ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &inputs) {
    size_t wkld = compute_fusible_workload(ctx, dst, inputs);
    // set default vectorized information
    vx_info_.axis = dst[0]->get_shape().size() - 1;

    for (int64_t i = dst[0]->nslice_dims() - 1; i >= 0; --i) {
        auto cur_dim = dst[0]->get_shape()[i];
        if (!cur_dim.isa<constant>()
                || get_const_as_int(cur_dim.checked_as<constant>())) {
            vx_info_.axis = i;
            break;
        }
    }
    vx_info_.lanes
            = vectorize_step(ctx, info_.inputs_[0]->details_.dtype_.type_code_);
    bool use_mask = attrs_.get_or_else(op_attr_key::use_padded_mask, true);
    if (get_owner_graph().is_dynamic()) {
        use_mask &= info_.cur_impl_ != impl_kind_t::no_padding;
    }
    // use broad-cast
    int bc_input_idx = get_broadcast_input();
    bool use_broadcast = bc_input_idx != -1;
    auto func = [&](const std::vector<expr> &in,
                        const std::vector<expr::lvalue_proxy_t> &out) -> stmt {
        auto out_dtype = out[0]->dtype_;
        expr in0, in1;
        if (use_broadcast) {
            in0 = in[1 - bc_input_idx], in1 = in[bc_input_idx];
        } else {
            in0 = in[0], in1 = in[1];
            if (in[0]->dtype_ != out_dtype) {
                in0 = builder::make_cast(out_dtype, in[0]);
            }
            if (in[1]->dtype_ != out_dtype) {
                in1 = builder::make_cast(out_dtype, in[1]);
            }
        }
        return select_algorithm(elt_op_, in0, in1, out[0], attrs_, info_);
    };
    if (use_broadcast) {
        // reuse broadcast op
        compute_block_broadcast(ctx, get_owner_graph(), inputs, *dst[0], info_,
                bc_input_idx, get_bc_axis(), vx_info_,
                mask_compute_func_t(func), get_outputs()[0], wkld, use_mask);
    } else {
        compute_vectorized_op(ctx, get_owner_graph(), inputs, *dst[0], info_,
                vx_info_, mask_compute_func_t(func), mask_compute_func_t(func),
                attrs_, get_outputs()[0], wkld, use_mask);
    }
}

void unary_backward_base_t::compute_block(context_ptr ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &inputs) {
    size_t wkld = compute_fusible_workload(ctx, dst, inputs);
    auto vx_info = get_vx_info();
    auto elt_op = get_elt_operator();
    // set default vectorized information
    vx_info.axis = dst[0]->get_shape().size() - 1;

    for (int64_t i = dst[0]->nslice_dims() - 1; i >= 0; --i) {
        auto cur_dim = dst[0]->get_shape()[i];
        if (!cur_dim.isa<constant>()
                || get_const_as_int(cur_dim.checked_as<constant>())) {
            vx_info.axis = i;
            break;
        }
    }
    vx_info.lanes
            = vectorize_step(ctx, info_.inputs_[0]->details_.dtype_.type_code_);
    bool use_mask = attrs_.get_or_else(op_attr_key::use_padded_mask, true);
    if (get_owner_graph().is_dynamic()) {
        use_mask &= info_.cur_impl_ != impl_kind_t::no_padding;
    }
    auto func = [&](const std::vector<expr> &in,
                        const std::vector<expr::lvalue_proxy_t> &out) -> stmt {
        auto out_dtype = out[0]->dtype_;
        expr in0, in1;
        in0 = in[0], in1 = in[1];
        if (in[0]->dtype_ != out_dtype) {
            in0 = builder::make_cast(out_dtype, in[0]);
        }
        if (in[1]->dtype_ != out_dtype) {
            in1 = builder::make_cast(out_dtype, in[1]);
        }
        return select_algorithm(elt_op, in0, in1, out[0], attrs_, info_);
    };
    compute_vectorized_op(ctx, get_owner_graph(), inputs, *dst[0], info_,
            vx_info, mask_compute_func_t(func), mask_compute_func_t(func),
            attrs_, get_outputs()[0], wkld, use_mask);
}

unary_backward_base_t::unary_backward_base_t(
        graph_tensor_ptr lhs, graph_tensor_ptr rhs, elt_operator elt_op)
    : unary_backward_base_t({std::move(lhs), std::move(rhs)}, {}, {}) {
    set_elt_operator(elt_op);
    switch (elt_op) {
        case elt_operator::ABS_BWD: op_name_ = "abs_bwd"; break;
        case elt_operator::CLAMP_BWD: op_name_ = "clamp_bwd"; break;
        case elt_operator::ELU_BWD: op_name_ = "elu_bwd"; break;
        case elt_operator::HARDSWISH_BWD: op_name_ = "hardswish_bwd"; break;
        case elt_operator::HARDSIGMOID_BWD: op_name_ = "hardsigmoid_bwd"; break;
        case elt_operator::MISH_BWD: op_name_ = "mish_bwd"; break;
        case elt_operator::SQRT_BWD: op_name_ = "sqrt_bwd"; break;
        case elt_operator::TANH_BWD: op_name_ = "tanh_bwd"; break;
        case elt_operator::SOFTPLUS_BWD: op_name_ = "soft_plus_bwd"; break;
        default: break;
    }
}

OP_REGISTER(add_op_t, add)
OP_REGISTER(mul_op_t, mul)
OP_REGISTER(sub_op_t, sub)
OP_REGISTER(div_op_t, div)
OP_REGISTER(min_op_t, min)
OP_REGISTER(max_op_t, max)
OP_REGISTER(squared_diff_op_t, squared_diff)
OP_REGISTER(prelu_op_t, prelu)
OP_REGISTER(abs_bwd_op_t, abs_bwd)
OP_REGISTER(clamp_bwd_op_t, clamp_bwd)
OP_REGISTER(elu_bwd_op_t, elu_bwd)
OP_REGISTER(hardswish_bwd_op_t, hardswish_bwd)
OP_REGISTER(hardsigmoid_bwd_op_t, hardsigmoid_bwd)
OP_REGISTER(sqrt_bwd_op_t, sqrt_bwd)
OP_REGISTER(mish_bwd_op_t, mish_bwd)
OP_REGISTER(tanh_bwd_op_t, tanh_bwd)
OP_REGISTER(softplus_bwd_op_t, soft_plus_bwd)
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
