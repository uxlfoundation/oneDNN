/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef BENCHDNN_GRAPH_FLEX_REWRITE_HPP
#define BENCHDNN_GRAPH_FLEX_REWRITE_HPP

#include <map>
#include <string>

#include "deserialize.hpp"
#include "utils.hpp"

namespace graph {

struct flex_rewrite_t {
    flex_rewrite_t(const std::map<size_t, std::string> &in_shapes,
            const std::map<size_t, std::string> &op_attrs,
            const graph_fpmath_mode_t &fpmath_mode, const int64_t mb,
            const dnnl_data_type_t dt,
            const std::map<size_t, dnnl_data_type_t> &dt_map,
            const std::map<size_t, std::string> &op_kind_map)
        : in_shapes_(in_shapes)
        , op_attrs_(op_attrs)
        , fpmath_mode_(fpmath_mode)
        , mb_(mb)
        , dt_(dt)
        , dt_map_(dt_map)
        , op_kind_map_(op_kind_map) {}

    int rewrite(deserialized_graph_t &dgraph);

private:
    // input shape info from CML
    std::map<size_t, std::string> in_shapes_;
    // input attributes from CML
    std::map<size_t, std::string> op_attrs_;
    graph_fpmath_mode_t fpmath_mode_;
    int64_t mb_;
    dnnl_data_type_t dt_; // Updates whole graph with a single dt value.
    std::map<size_t, dnnl_data_type_t>
            dt_map_; // Updates specific LT with selected dt values.
    std::map<size_t, std::string> op_kind_map_;

    int infer_output_shape(deserialized_graph_t &dgraph, bool change_stride);
    int inports_shape_rewrite(
            deserialized_graph_t &dgraph, bool &change_stride);
    bool get_inport_shape_stride(const std::string &in_shape,
            std::string &shape, std::string &stride, std::string &mtag,
            std::string &msg);
    int op_attrs_rewrite(deserialized_graph_t &dgraph);
    int quantized_graph_rewrite(deserialized_graph_t &dgraph);
    int update_output_info(deserialized_op_t &aop, deserialized_graph_t &dgraph,
            bool change_stride);
    int graph_attrs_rewrite(deserialized_graph_t &dgraph);
    int dt_rewrite(deserialized_graph_t &dgraph);
    int dt_map_rewrite(deserialized_graph_t &dgraph);
    int op_kind_rewrite(deserialized_graph_t &dgraph);
    // Rewrite some linked attribute and shapes, such as group-shape and
    // scale/zp shape of dynamic dequantization for per-group quantization, to
    // simplify the cml input of rewriting.
    int linked_shape_and_attr_rewrite(deserialized_graph_t &dgraph);
};

} // namespace graph

#endif
