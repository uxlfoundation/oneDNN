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

#include "graph/backend/dnnl/kernels/large_partition.hpp"

#include "graph/backend/dnnl/patterns/fusions.hpp"
#include "graph/backend/dnnl/patterns/pattern_matcher_pass.hpp"
#include "graph/backend/dnnl/patterns/utils.hpp"

#include "graph/utils/pm/pbuilder.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
namespace pattern {

namespace pm = graph::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph_t = pm::pb_graph_t;
using FCreatePattern = graph::pass::FCreatePattern;

DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(bmb)

// binary -> ... -> binary -> matmul -> binary -> ... -> binary ->
DNNL_BACKEND_REGISTER_PATTERN_MATCHER_PASS(dnnl, float_bmb)
        .set_priority(19.5f)
        .set_kind(partition_kind_t::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    // first binary
                    auto first_bin
                            = pgraph->append_alternation(get_binary_ops());
                    first_bin->append_decision_function(
                            check_output_dtype<impl::data_type::f32>);

                    // repetition(alterative(add, div, mul, sub, ..), N).
                    auto multi_bin = std::make_shared<pb_graph_t>();
                    auto alter_bin
                            = multi_bin->append_alternation(get_binary_ops());
                    alter_bin->allow_internal_inputs();
                    multi_bin->create_input_port(0, alter_bin, 0);
                    multi_bin->create_output_port(0, alter_bin, 0);
                    auto prep = pgraph->append_repetition(multi_bin, {0, 0}, 0,
                            MAX_REPETITION, {in_edge(0, first_bin, 0)});

                    // optional typecast
                    auto tc_0 = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *ptypecast_0
                            = tc_0->append_op(graph::op_kind::TypeCast);
                    tc_0->create_input_port(0, ptypecast_0, 0);
                    tc_0->create_output_port(0, ptypecast_0, 0);
                    auto pre_tc_0 = pgraph->append_optional(
                            tc_0, {in_edge(0, prep, 0)});

                    // matmul
                    pm::pb_op_t *mm = pgraph->append_op(
                            graph::op_kind::MatMul, {in_edge(0, pre_tc_0, 0)});

                    // repetition(alterative(add, div, mul, sub, ..), N).
                    auto multi_bin_1 = std::make_shared<pb_graph_t>();
                    auto alter_bin_1
                            = multi_bin_1->append_alternation(get_binary_ops());
                    alter_bin_1->allow_internal_inputs();
                    multi_bin_1->create_input_port(0, alter_bin_1, 0);
                    multi_bin_1->create_output_port(0, alter_bin_1, 0);
                    auto prep_1 = pgraph->append_repetition(multi_bin_1, {0, 0},
                            0, MAX_REPETITION, {in_edge(0, mm, 0)});

                    // optional typecast
                    auto tc_1 = std::make_shared<pb_graph_t>();
                    pm::pb_op_t *ptypecast_1
                            = tc_1->append_op(graph::op_kind::TypeCast);
                    tc_1->create_input_port(0, ptypecast_1, 0);
                    tc_1->create_output_port(0, ptypecast_1, 0);
                    pgraph->append_optional(tc_1, {in_edge(0, prep_1, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<larger_partition_kernel_t>();
        });

DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
