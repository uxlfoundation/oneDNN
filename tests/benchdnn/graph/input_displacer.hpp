/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#ifndef BENCHDNN_GRAPH_INPUT_DISPLACER_HPP
#define BENCHDNN_GRAPH_INPUT_DISPLACER_HPP

#include "ref_primitive.hpp"

#include "src/common/memory_desc.hpp"
#include "utils/fill.hpp"

namespace graph {

enum class filling_type_t {
    undef = 0,
    quantization,
    causal_mask,
    // Fill pre-defined fixed values for data filling, such as 0, 1, -inf, and
    // specified shape information for scalar input.
    fixed_setting,
    // Fill softmax stats for sdpa backward graph by recomputing the forward graph
    softmax_stats,
};
struct displace_args_t {

public:
    displace_args_t() = default;
    displace_args_t(const deserialized_op_t &op, size_t offset,
            const deserialized_lt_t &lt, filling_type_t type,
            const fill_cfg_t &cfg = {})
        : main_op_(op)
        , main_op_offset_(offset)
        , tensor_(lt)
        , filling_type_(type)
        , fill_cfg_(cfg) {}

    deserialized_op_t main_op_;
    size_t main_op_offset_;
    //the tensor as a displace starting point
    deserialized_lt_t tensor_;
    filling_type_t filling_type_;
    fill_cfg_t fill_cfg_;
};

class partition_data_displacer_t {
public:
    partition_data_displacer_t() = default;
    partition_data_displacer_t(
            const deserialized_graph_t &dg, const dnnl::graph::partition &par);
    // TODO: some arguments of displace_input_data() are redundant
    // and can be removed in the future.
    int displace_input_data(size_t lt_id, dnn_mem_t &mem,
            const std::unordered_map<size_t, const dnn_mem_t &> &lt_id_2_mems,
            res_t *res);
    filling_type_t get_filling_type(size_t lt_id) const {
        auto it = displace_args_.find(lt_id);
        if (it == displace_args_.end()) return filling_type_t::undef;
        return it->second.filling_type_;
    }

private:
    const deserialized_graph_t *dg_ = nullptr;
    // A set of op_id values from a partition came to a displacer. Used to
    // identify at displacement stage if Deq is the starting point or not.
    std::unordered_set<size_t> op_ids_set_;
    ::std::unordered_map<size_t, displace_args_t> displace_args_;

    int gen_quantize_filling(const ::graph::deserialized_op_t &main_op, int arg,
            dnn_mem_t &mem, const ::std::string &dt, res_t *res);
    // Generates values in the target memory based on predefined set of values
    // from `fill_cfg`.
    int gen_fixed_set_filling(dnn_mem_t &mem, const_dnnl_memory_desc_t md,
            const fill_cfg_t &fill_cfg, res_t *res) const;
    // Generates causal mask filling for "Add" operation.
    int gen_causal_mask_filling(
            dnn_mem_t &mem, const_dnnl_memory_desc_t md, res_t *res) const;
    int gen_softmax_stats_filling(const ::graph::deserialized_op_t &main_op,
            int arg, const dnn_mem_t &src_mem, dnn_mem_t &mem,
            const_dnnl_memory_desc_t md, res_t *res) const;
};

} // namespace graph

#endif
